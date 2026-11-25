"""Regression checks for executable multi-leg replays and Vega attribution."""

import numpy as np
import pandas as pd

from helpers.backtest import Backtest, Strategy, VOL_FACTORS, split_vega
from helpers import event_vol_research as h


def test_replays():
    days = pd.to_datetime(["2026-08-26", "2026-08-27", "2026-08-28"])
    schedule = pd.DataFrame({
        "market_open": days.tz_localize("UTC") + pd.Timedelta(hours=13, minutes=30),
        "market_close": days.tz_localize("UTC") + pd.Timedelta(hours=20),
    }, index=days)
    entry = pd.Timestamp("2026-08-26 19:45Z")
    event = pd.Timestamp("2026-08-26 20:00Z")
    first = pd.Timestamp("2026-08-27 13:31Z")
    last = first + pd.Timedelta(minutes=1)
    expiry = days[-1].tz_localize("UTC")
    close = schedule.market_close.iloc[-1]
    rows = []
    for time, spot, vol in ((entry, 100, .60), (first, 104, .42), (last, 104.5, .40)):
        _, _, calendar, tau = h.calculate_variance_time(time, close, schedule)
        discount = np.exp(-.04 * calendar)
        for strike in (95, 100, 105):
            for side in ("C", "P"):
                bid = h.black_price(spot / discount, strike, discount, tau, vol, side)
                rows.append(dict(ts_recv=time, spot_time=time, expiration=expiry,
                                 symbol=f"{side}{strike}", instrument_class=side,
                                 strike_price=strike, spot_bid=spot, spot_ask=spot + .01,
                                 bid_px_00=bid, ask_px_00=bid + .02, bid_sz_00=2,
                                 ask_sz_00=1 if time == first else 2))
    quotes = pd.DataFrame(rows)
    surface = {entry: (.58, -.10, .4, 100), first: (.41, -.14, .6, 104),
               last: (.39, -.12, .7, 104.5)}
    args = dict(entry=entry, exit_start=first, exit_end=last, event_time=event,
                implied_move=.04, delta_hedge=True, fees=1)
    engine = Backtest(quotes, schedule, surface=lambda time, _: surface.get(time))
    cases = [
        (Strategy("straddle", "short"), {"C100": -1, "P100": -1}, first),
        (Strategy("risk_reversal", "bullish"), {"C105": 1, "P95": -1}, first),
        (Strategy("butterfly", "long", wing_width=5), {"C95": 1, "C100": -2, "C105": 1}, last),
    ]
    for strategy, quantities, exit_time in cases:
        replay = engine.run(strategy, **args)
        trade = replay.trades.iloc[0]
        assert trade.status == trade.hedge_status == "priced"
        assert trade.exit_time == exit_time
        assert dict(zip(replay.legs.symbol, replay.legs.quantity)) == quantities
        for row in replay.legs.itertuples():
            assert np.isclose(row.option_pnl_usd, row.quantity * 100 * (row.exit_fill - row.entry_fill))
        assert np.isclose(trade.hedged_pnl_usd, replay.legs.option_pnl_usd.sum() + trade.stock_pnl_usd - 1)
        for prefix in ("", "earnings_gap_"):
            assert np.isclose(sum(trade[f"{prefix}{name}_usd"] for name in VOL_FACTORS),
                              trade[f"{prefix}vega_usd"])

    missing = Backtest(quotes, schedule, surface=lambda time, _: surface.get(time) if time == entry else None)
    trade = missing.run(cases[0][0], **args).trades.iloc[0]
    assert trade.status == "priced" and trade.vol_breakdown_status == "incomplete"
    assert np.isfinite(trade.vega_usd) and np.isnan(trade.atm_vol_usd)
    assert trade.earnings_gap_vol_intervals_unavailable == 2
    assert split_vega(100, "C", 100, (.99, .02), .2, .3,
                      (.7, 0, 0, 100), (.1, 0, 0, 100)) is None

    quotes.loc[quotes.ts_recv.eq(entry) & quotes.symbol.eq("C100"), "bid_sz_00"] = 1
    blocked = Backtest(quotes, schedule).run(cases[-1][0], **args)
    assert blocked.legs.empty
    assert blocked.trades.status.iloc[0] == "No executable equal-width butterfly"


if __name__ == "__main__":
    test_replays()
    print("Multileg fills, hedge reconciliation, and Vega decomposition passed.")
