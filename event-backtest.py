"""Run a post-close earnings backtest against locally cached minute quotes."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from backtest import Backtest, Strategy, FACTORS, VOL_FACTORS
from earnings_event import event_sessions
import helper as h


class Surfaces:
    """Fit only the timestamp/expiry cross-sections requested by the replay."""

    def __init__(self, quotes, schedule):
        self.groups = quotes.groupby(h.KEYS, sort=False)
        self.schedule = schedule
        self.fit = lru_cache(maxsize=None)(self.fit)

    def fit(self, timestamp, expiry):
        key = (timestamp, expiry)
        if key not in self.groups.indices:
            return None
        data = h.filter_quotes(self.groups.get_group(key))
        if data.empty:
            return None
        data = h.add_variance_time(data, schedule=self.schedule)
        pairs = h.build_call_put_pairs(data)
        forward = h.calibrate_forward(pairs)
        if forward.empty or not np.isfinite(forward.forward.iloc[0]):
            return None
        data = data.merge(forward, on=h.KEYS, validate="many_to_one")
        # The quadratic fit only accepts quotes within this forward-moneyness band.
        data = data.loc[np.log(data.strike_price / data.forward).abs().le(.10)]
        if data.empty:
            return None
        atm = h.extract_atm_features(h.build_surface(h.calculate_iv(data, progress=False)))
        return None if atm.empty else atm.iloc[0]

    def components(self, timestamp, expiry):
        row = self.fit(timestamp, expiry)
        if row is None:
            return None
        values = (row.atm_iv_quadratic, row.atm_skew, row.atm_curvature, row.forward)
        return values if np.isfinite(values).all() else None

    def event_move(self, timestamp, event_time, max_dte):
        expiries = self.expiries_at(timestamp)
        rows = []
        for expiry in expiries:
            close = self.schedule.at[expiry.tz_localize(None).normalize(), "market_close"]
            if event_time < close <= timestamp + pd.Timedelta(days=max_dte):
                row = self.fit(timestamp, expiry)
                if row is not None:
                    rows.append(row)
        if len(rows) < 3:
            return None
        try:
            return h.fit_event_variance(pd.DataFrame(rows), max_dte=max_dte,
                                        event_time=event_time)["event_move"]
        except ValueError:
            return None

    def expiries_at(self, timestamp):
        return self.groups.obj.loc[self.groups.obj.ts_recv.eq(timestamp), "expiration"].dropna().unique()


def run_event(ticker, dates, closes, data_dir, strategies, *, entry_phases, entry_minutes,
              max_dte, delta_hedge, fees, vol_breakdown):
    prior, event, following = dates
    windows = [(event, "close"), (following, "open")]
    if "T-1" in entry_phases:
        windows.insert(0, (prior, "close"))
    frames = [
        h.merge_data(*h.load_data(day, ticker, data_dir, window=window))
        for day, window in windows
    ]
    quotes = pd.concat(frames, ignore_index=True)
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=prior, end_date=max(pd.Timestamp(following).date(), quotes.expiration.max().date()),
    )
    market_open = schedule.at[pd.Timestamp(following), "market_open"]
    exit_start = market_open + pd.Timedelta(minutes=1)
    exit_end = market_open + pd.Timedelta(minutes=30)
    event_time = closes.iloc[1] + pd.Timedelta(1, unit="ns")
    surfaces = Surfaces(quotes, schedule)
    engine = Backtest(quotes, schedule, surface=surfaces.components if vol_breakdown else None)
    for phase, close in zip(("T-1", "T"), closes.iloc[:2], strict=True):
        if phase not in entry_phases:
            continue
        for minutes in entry_minutes:
            entry = close - pd.Timedelta(minutes=minutes)
            move = surfaces.event_move(entry, event_time, max_dte) if any(
                strategy.needs_move for strategy in strategies
            ) else None
            tags = dict(ticker=ticker, event_date=event, entry_phase=phase,
                        entry_minutes_before_close=minutes)
            for strategy in strategies:
                replay = engine.run(strategy, entry=entry, exit_start=exit_start, exit_end=exit_end,
                                    event_time=event_time, implied_move=move, max_dte=max_dte,
                                    delta_hedge=delta_hedge, fees=fees)
                yield replay.trades.assign(**tags), replay.legs.assign(**tags)


def summarize(trades):
    keys = ["ticker", "event_date", "strategy", "expiration", "entry_phase",
            "entry_minutes_before_close"]
    priced = trades.loc[trades.status.eq("priced")]
    columns = ["option_pnl_usd", "unhedged_pnl_usd", "stock_pnl_usd", "hedged_pnl_usd",
               "fees_usd", "option_execution_and_model_basis_usd"]
    columns.extend(f"{prefix}{name}_usd" for prefix in ("", "earnings_gap_")
                   for name in (*FACTORS, *VOL_FACTORS))
    metrics = [column for column in columns if column in priced]
    if priced.empty:
        return pd.DataFrame(columns=keys + ["trades", *metrics])
    return priced.groupby(keys, as_index=False).agg(
        trades=("status", "size"),
        attributed_trades=("attribution_status", lambda status: status.eq("attributed").sum()),
        **{column: (column, "mean") for column in metrics},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--event", action="append", required=True, metavar="YYYY-MM-DD",
                        help="Post-close earnings date. Repeat to run more events.")
    parser.add_argument("--strategy", choices=("straddle", "risk-reversal", "butterfly"),
                        action="append", required=True)
    parser.add_argument("--center", choices=("atm", "up", "down"), action="append",
                        help="Repeat to run multiple centers. Default: ATM.")
    parser.add_argument("--long-straddle", action="store_true")
    parser.add_argument("--bearish-risk-reversal", action="store_true")
    parser.add_argument("--wing-width", type=float,
                        help="Exact butterfly wing width in USD. Default: implied-move width.")
    parser.add_argument("--entry-phase", choices=("T-1", "T"), action="append")
    parser.add_argument("--entry-minutes", type=int, action="append")
    parser.add_argument("--max-dte", type=float, default=31)
    parser.add_argument("--delta-hedge", action="store_true")
    parser.add_argument("--fees", type=float, default=0.0, help="Total round-trip USD per structure.")
    parser.add_argument("--no-vol-breakdown", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.max_dte) or args.max_dte <= 0 or not np.isfinite(args.fees) or args.fees < 0:
        parser.error("Max DTE must be positive and fees must be nonnegative.")
    entry_minutes = tuple(dict.fromkeys(args.entry_minutes or (30, 15)))
    if any(not 1 <= value <= 30 for value in entry_minutes):
        parser.error("Entry minutes must be in the cached 1–30 minute window before close.")
    strategies = []
    for kind in dict.fromkeys(args.strategy):
        if kind == "risk-reversal":
            strategies.append(Strategy("risk_reversal", "bearish" if args.bearish_risk_reversal else "bullish"))
            continue
        for center in dict.fromkeys(args.center or ("atm",)):
            direction = "long" if kind == "butterfly" or args.long_straddle else "short"
            width = args.wing_width if kind == "butterfly" else None
            try:
                strategies.append(Strategy(kind, direction, center, width))
            except ValueError as error:
                parser.error(str(error))
    ticker = args.ticker.upper().strip()
    events = {date: date for date in args.event}
    parts, details = [], []
    for event, (dates, closes) in event_sessions(events).items():
        try:
            for trades, legs in run_event(
                ticker, dates, closes, args.data_dir, strategies,
                entry_phases=args.entry_phase or ("T-1", "T"), entry_minutes=entry_minutes,
                max_dte=args.max_dte, delta_hedge=args.delta_hedge, fees=args.fees,
                vol_breakdown=not args.no_vol_breakdown,
            ):
                parts.append(trades)
                if not legs.empty:
                    details.append(legs)
        except FileNotFoundError as error:
            parts.extend(pd.DataFrame([dict(ticker=ticker, event_date=event, strategy=strategy.label,
                                            status=f"Missing input: {error}")]) for strategy in strategies)
        print(f"{ticker} {event} finished", flush=True)
    trades = pd.concat(parts, ignore_index=True)
    legs = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("trades", trades), ("legs", legs), ("summary", summarize(trades))):
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    print(trades.status.value_counts().to_string())


if __name__ == "__main__":
    main()
