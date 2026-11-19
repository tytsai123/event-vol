"""Shared earnings event analysis and command-line workflow."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import local
from time import sleep

import matplotlib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from matplotlib import dates as mdates
from matplotlib.ticker import PercentFormatter

import download_data as dd
import helper as h

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parent
MAX_DTE = 31
SESSION_LABELS = ("T-1", "T", "T+1")
COLORS = ("#0056AB", "#B34200", "#82268C")


def configure_plots():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#D1DCE4", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlecolor": "#172B3A",
        "axes.titleweight": "bold", "axes.labelcolor": "#536675",
        "text.color": "#172B3A", "xtick.color": "#536675",
        "ytick.color": "#536675", "grid.color": "#DCE5EB",
        "axes.grid.axis": "y", "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })


def event_sessions(events):
    releases = list(events.values())
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=pd.Timestamp(min(releases)) - pd.Timedelta(days=10),
        end_date=pd.Timestamp(max(releases)) + pd.Timedelta(days=10),
    )
    windows = {}
    for quarter, release in events.items():
        position = schedule.index.get_loc(pd.Timestamp(release))
        window = schedule.iloc[position - 1:position + 2]
        windows[quarter] = (tuple(window.index.strftime("%Y-%m-%d")), window["market_close"])
    return windows


def estimate_missing(requests):
    import databento as db
    from dotenv import load_dotenv
    from requests.exceptions import ConnectionError, Timeout

    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("Set DATABENTO_API_KEY in .env or in the environment.")
    clients = local()

    def quote(request):
        path, params = request
        if not hasattr(clients, "historical"):
            clients.historical = db.Historical(key)
        try:
            for delay in (1, 2):
                try:
                    result = clients.historical.metadata.get_cost(**params)
                    break
                except (ConnectionError, Timeout):
                    sleep(delay)
            else:
                result = clients.historical.metadata.get_cost(**params)
            cost = float(result)
        except Exception as error:
            message = str(error).replace(key, "[REDACTED]").splitlines()
            reason = message[0] if message else type(error).__name__
            raise RuntimeError(f"Estimate failed for {path.name}: {reason}") from None
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"Invalid estimate for {path.name}.")
        return cost

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(quote, requests))


def realized_jump(daily, release, following_session):
    if daily is None or not {release, following_session}.issubset(daily.index):
        return np.nan
    close = daily.at[release, "close"]
    opening = daily.at[following_session, "open"]
    if not (np.isfinite([close, opening]).all() and min(close, opening) > 0):
        raise ValueError(f"Invalid realized prices for {release} and {following_session}.")
    return np.log(opening / close)


def plot_term_structure(quarter, snapshots, destination, *, ticker="NVDA"):
    metrics = (
        ("atm_iv_quadratic", "ATM implied volatility", True),
        ("total_variance_quadratic", "ATM total variance", True),
        ("atm_skew", "ATM skew", False),
        ("atm_curvature", "ATM curvature", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.7), sharex=True)
    for ax, (column, title, percent) in zip(axes.flat, metrics):
        for phase, group, color, marker, line in zip(
            SESSION_LABELS, snapshots, COLORS, ("o", "s", "^"), ("-", "--", "-.")
        ):
            near = group.loc[group["dte"] <= 60].sort_values("expiration")
            ax.plot(near["expiration"], near[column], color=color, marker=marker,
                    linestyle=line, markersize=5, label=f"{phase} | {group['session'].iloc[0]}",
                    markerfacecolor="white" if phase != "T+1" else color)
        ax.set_title(title, loc="left")
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_xlabel("Expiration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{ticker} {quarter} | Near-term volatility around earnings", x=0.055,
                 ha="left", weight="bold")
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.94), ncols=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_event_comparison(quarter, comparison, fit, jump, destination, *, ticker="NVDA"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col", sharey="row")
    metrics = [("event_variance", jump**2, "Event variance"),
               ("event_move", abs(jump), "Implied event move")]
    for row, (column, realized, label) in enumerate(metrics):
        for ax, limit in zip(axes[row], (35, 365)):
            panel = comparison.loc[comparison["dte"] <= limit]
            ax.plot(panel["dte"], panel[column], "o-", color=COLORS[0], label="Pre/Post")
            if np.isfinite(fit[column]):
                ax.axhline(fit[column], color=COLORS[1], linestyle="--", label="OLS")
            if np.isfinite(realized):
                ax.axhline(realized, color=COLORS[2], linestyle=":", label="Realized jump")
            ax.set_xlim(0, limit)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.margins(y=0.15)
            if limit == 35:
                last_labeled_dte = -np.inf
                for point in panel.dropna(subset=[column]).itertuples():
                    if point.dte - last_labeled_dte < 7:
                        continue
                    value = getattr(point, column)
                    ax.annotate(
                        f"{value:.3%}", (point.dte, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color="white",
                        bbox={"boxstyle": "round,pad=0.26",
                              "facecolor": COLORS[0], "edgecolor": "none"},
                    )
                    last_labeled_dte = point.dte
        axes[row, 0].set_ylabel(label)
    axes[0, 0].set_title("Short maturities | Up to 35 DTE", loc="left")
    axes[0, 1].set_title("Full range | Up to 365 DTE", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("Calendar days to expiry")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{ticker} {quarter} | Implied earnings move | Pre/Post vs. OLS", x=0.055,
                 ha="left", weight="bold")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.955), ncols=3)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def quote_diagnostics(merged, clean, snapshot, snapshot_time, phase, date):
    """Measure two-sided coverage among mapped quotes within 2% log spot moneyness."""
    def near(frame):
        return frame.loc[
            frame["ts_recv"].eq(snapshot_time)
            & frame["log_spot_moneyness"].abs().le(0.02)
            & frame["dte"].between(1, MAX_DTE)
        ]

    observed = near(merged)
    valid = near(clean)
    usable = snapshot.loc[snapshot["dte"].between(0, MAX_DTE, inclusive="right")
                          & snapshot["atm_iv_quadratic"].notna()]
    return {
        "phase": phase, "session": date, "snapshot_time": snapshot_time,
        "near_atm_quotes": len(observed), "valid_two_sided_quotes": len(valid),
        "two_sided_coverage": len(valid) / len(observed) if len(observed) else np.nan,
        "median_relative_spread": valid["relative_spread"].median(),
        "valid_strike_expiry_pairs": len(valid[["expiration", "strike_price"]].drop_duplicates()),
        "usable_atm_expiries": len(usable),
    }


def analyze_event(quarter, dates, closes, daily, data_dir, output_dir, *,
                  ticker="NVDA", premarket_time=None):
    snapshots = []
    diagnostics = []
    for phase, date, close in zip(SESSION_LABELS, dates, closes):
        result = h.run_pipeline(date, ticker=ticker, data_dir=data_dir, progress=False)
        atm = result["atm_df"]
        eligible = atm["ts_recv"].le(close) & atm["total_variance_quadratic"].notna()
        if not eligible.any():
            raise ValueError(f"{date}: no valid ATM snapshot at or before close.")
        snapshot_time = atm.loc[eligible, "ts_recv"].max()
        snapshots.append(atm.loc[atm["ts_recv"].eq(snapshot_time)].assign(
            phase=phase, session=date, dte=lambda frame: frame["T_calendar"] * 365
        ))
        if premarket_time is not None:
            diagnostics.append(quote_diagnostics(
                result["merged_df"], result["clean_df"], snapshots[-1], snapshot_time, phase, date
            ))

    if premarket_time is None:
        # Admit close-stamped quotes and exclude expiries ending at that close.
        event_cutoff = closes.iloc[1] + pd.Timedelta(1, unit="ns")
        pre_index = 1
    else:
        event_cutoff = pd.Timestamp(f"{dates[1]} {premarket_time}", tz="America/New_York")
        pre_index = 0
    before, after = snapshots[pre_index:pre_index + 2]
    fit = h.fit_event_variance(before, max_dte=MAX_DTE, event_time=event_cutoff)
    comparison = h.compare_pre_post(
        pd.concat([before, after]), before["ts_recv"].iloc[0],
        after["ts_recv"].iloc[0], fit["base_vol"],
    ).merge(before[["expiration", "dte"]], on="expiration", validate="one_to_one")
    comparison = comparison.sort_values("dte")
    jump = realized_jump(daily, dates[pre_index], dates[pre_index + 1])
    summary = {
        "quarter": quarter, "t_minus_1": dates[0], "release_date": dates[1],
        "t_plus_1": dates[2], "pre_time": before["ts_recv"].iloc[0],
        "post_time": after["ts_recv"].iloc[0], **fit, "realized_jump": jump,
        "realized_event_variance": jump**2, "variance_difference": fit["event_variance"] - jump**2,
        "matched_expiries": len(comparison),
    }
    if premarket_time is not None:
        summary["event_cutoff"] = event_cutoff
        summary["realized_close_session"] = dates[pre_index]
        summary["realized_open_session"] = dates[pre_index + 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker}_{quarter}"
    with TemporaryDirectory(dir=output_dir) as temporary:
        staging = Path(temporary)
        pd.concat(snapshots).to_csv(staging / f"{stem}_atm_snapshots.csv", index=False)
        comparison.to_csv(staging / f"{stem}_pre_post.csv", index=False)
        pd.DataFrame([summary]).to_csv(staging / f"{stem}_summary.csv", index=False)
        plot_term_structure(quarter, snapshots, staging / f"{stem}_near_term_volatility.png",
                            ticker=ticker)
        plot_event_comparison(quarter, comparison, fit, jump, staging / f"{stem}_event_move.png",
                              ticker=ticker)
        if diagnostics:
            pd.DataFrame(diagnostics).to_csv(staging / f"{stem}_quote_quality.csv", index=False)
        for file in staging.iterdir():
            file.replace(output_dir / file.name)
    print(f"{quarter}: saved {4 if diagnostics else 3} CSVs and two charts.")


def run_study(events, *, ticker="NVDA", premarket_times=None, output_folder=None):
    parser = argparse.ArgumentParser(description=f"Run the {ticker} earnings event study.")
    parser.add_argument("--quarter", choices=events, action="append", help="Select one or more quarters.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / (output_folder or "nvda-ead-scaled-up") / "results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="Query missing-file prices only.")
    mode.add_argument("--download", action="store_true", help="Download missing Databento inputs, then analyze.")
    parser.add_argument("--download-only", action="store_true", help="With --download, skip analysis.")
    parser.add_argument("--max-cost", type=float, help="Approved total estimated USD ceiling for --download.")
    args = parser.parse_args()
    if args.download and (args.max_cost is None or not math.isfinite(args.max_cost) or args.max_cost < 0):
        parser.error("--download requires an approved, finite, nonnegative --max-cost.")
    if args.max_cost is not None and not args.download:
        parser.error("--max-cost is only used with --download.")
    if args.download_only and not args.download:
        parser.error("--download-only is only used with --download.")

    quarters = args.quarter or events
    windows = event_sessions({quarter: events[quarter] for quarter in quarters})
    sessions = sorted({date for dates, _ in windows.values() for date in dates})
    requests = {date: dd.build_requests(date, ticker=ticker, data_dir=args.data_dir)
                for date in sessions}
    missing = {date: [(path, params) for path, params in files
                      if not path.is_file() or path.stat().st_size == 0]
               for date, files in requests.items()}
    missing_requests = [request for files in missing.values() for request in files]
    print(f"{len(windows)} quarters, {len(sessions)} sessions, {len(missing_requests)} missing DBN files.",
          flush=True)
    if args.estimate or args.download:
        costs = estimate_missing(missing_requests) if missing_requests else []
        prices = dict(zip((path for path, _ in missing_requests), costs))
        session_costs = {day: sum(prices[path] for path, _ in files)
                         for day, files in missing.items()}
        total = math.fsum(session_costs.values())
        for quarter, (dates, _) in windows.items():
            quarter_cost = math.fsum(session_costs[day] for day in dates)
            print(f"{quarter} | {', '.join(dates)} | USD {quarter_cost:.4f}")
        print(f"Total estimated download cost: USD {total:.4f}", flush=True)
        if args.estimate:
            return
        if total > args.max_cost:
            parser.error(f"Estimate USD {total:.4f} exceeds the approved ceiling USD {args.max_cost:.4f}.")
        def download_session(session):
            try:
                dd.download_data(session, ticker=ticker, data_dir=args.data_dir, confirm=False,
                                 max_cost=session_costs[session])
            except Exception as error:
                print(f"{session}: download stopped ({type(error).__name__}).", flush=True)
                return session
            return None

        pending_sessions = [session for session, files in missing.items() if files]
        with ThreadPoolExecutor(max_workers=4) as pool:
            failed_sessions = [session for session in pool.map(download_session, pending_sessions)
                               if session is not None]
        if failed_sessions:
            parser.exit(1, f"Incomplete downloads: {', '.join(failed_sessions)}.\n")
        if args.download_only:
            return

    daily_path = args.data_dir / f"{ticker}_daily_ohlcv.csv"
    daily = pd.read_csv(daily_path, dtype={"session_date": str}).set_index("session_date") if daily_path.is_file() else None
    if daily is not None and not daily.index.is_unique:
        raise ValueError("Daily OHLCV contains duplicate trading dates.")
    incomplete = []
    for quarter, (dates, closes) in windows.items():
        absent = [path.name for date in dates for path, _ in requests[date]
                  if not path.is_file() or path.stat().st_size == 0]
        if absent:
            print(f"{quarter}: skipped, missing {', '.join(absent)}.")
            incomplete.append(quarter)
            continue
        try:
            analyze_event(quarter, dates, closes, daily, args.data_dir, args.output_dir,
                          ticker=ticker,
                          premarket_time=premarket_times[quarter] if premarket_times else None)
        except ValueError as error:
            print(f"{quarter}: analysis failed: {error}.")
            incomplete.append(quarter)
    if incomplete:
        parser.exit(1, f"Incomplete quarters: {', '.join(incomplete)}. Use --estimate to price missing files.\n")
