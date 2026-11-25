"""Minute-quote execution and Black–76 P&L attribution for option structures."""

from dataclasses import dataclass
from functools import lru_cache
from math import exp, isfinite, log, sqrt

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import event_vol_research as h

FACTORS = ("delta", "gamma", "theta", "vega")
VOL_FACTORS = ("atm_vol", "skew", "smile", "vol_residual")


class Unavailable(ValueError):
    """Market inputs are insufficient for one replay or attribution."""


@dataclass(frozen=True)
class Strategy:
    kind: str
    direction: str
    center: str = "atm"
    wing_width: float | None = None

    def __post_init__(self):
        directions = {"straddle": ("long", "short"),
                      "risk_reversal": ("bullish", "bearish"),
                      "butterfly": ("long",)}
        if self.direction not in directions.get(self.kind, ()):
            raise ValueError("Unsupported strategy direction")
        if self.center not in ("atm", "up", "down"):
            raise ValueError("Center must be atm, up, or down")
        if self.kind == "risk_reversal" and self.center != "atm":
            raise ValueError("Risk reversal wings are centered on the entry stock bid")
        if self.wing_width is not None and (self.kind != "butterfly"
                                           or not isfinite(self.wing_width) or self.wing_width <= 0):
            raise ValueError("Wing width must be positive and is only used for butterflies")

    @property
    def needs_move(self):
        return (self.kind == "risk_reversal" or self.center != "atm"
                or (self.kind == "butterfly" and self.wing_width is None))

    @property
    def label(self):
        if self.kind == "risk_reversal":
            return f"{self.direction}_risk_reversal"
        return f"{self.center}_{self.direction}_{self.kind}"


@dataclass(frozen=True)
class Leg:
    symbol: str
    side: str
    strike: float
    quantity: int


@dataclass
class Replay:
    trades: pd.DataFrame
    legs: pd.DataFrame


def _price(spot, strike, clock, iv, side):
    discount, tau = clock
    return h.black_price(spot / discount, strike, discount, tau, iv, side)


def _mark(quote, leg, clock):
    spot, bid = quote.spot_bid, quote.bid_px_00
    discount, tau = clock
    low = _price(spot, leg.strike, clock, 0.0, leg.side)
    high = _price(spot, leg.strike, clock, 5.0, leg.side)
    if bid <= low:
        return 0.0, low, bid < low
    if bid >= high:
        return 5.0, high, bid > high
    iv = h.implied_vol(bid, spot / discount, leg.strike, discount, tau, leg.side)
    if not isfinite(iv):
        raise Unavailable("Could not invert an in-bounds option bid")
    return iv, _price(spot, leg.strike, clock, iv, leg.side), False


def _delta(spot, strike, clock, iv, side):
    discount, tau = clock
    forward = spot / discount
    sign = 1 if side == "C" else -1
    if iv == 0:
        return sign * (1.0 if sign * (forward - strike) > 0 else 0.5 if forward == strike else 0.0)
    std = iv * sqrt(tau)
    return sign * ndtr(sign * (log(forward / strike) / std + std / 2))


def split_vega(strike, side, spot, clock, old_iv, new_iv, old_surface, new_surface):
    """Reprice ATM, skew, curvature, then residual with the new spot and clock.

    Each surface is (ATM IV, skew, quadratic coefficient, fitted forward).
    Returns per-share unsigned price changes, or None if decomposition is unavailable.
    """
    if any(surface is None or not np.isfinite(surface).all() or surface[3] <= 0
           for surface in (old_surface, new_surface)):
        return None

    def terms(surface):
        atm, skew, smile, forward = surface
        k = log(strike / forward)
        return np.array([atm, skew * k, smile * k * k])

    steps = old_iv + np.cumsum(terms(new_surface) - terms(old_surface))
    if not np.isfinite(steps).all() or (steps < 0).any():
        return None
    prices = [_price(spot, strike, clock, vol, side) for vol in [old_iv, *steps, new_iv]]
    return np.diff(prices)


def _executable(row, quantity):
    return (row.ask_sz_00 if quantity > 0 else row.bid_sz_00) >= abs(quantity)


def _fill(row, quantity):
    return row.ask_px_00 if quantity > 0 else row.bid_px_00


def _select(chain, strategy, move):
    spot = chain.spot_bid.iloc[0]
    center = spot
    if strategy.center != "atm":
        center *= exp(move if strategy.center == "up" else -move)
    contracts = {}
    for row in chain.itertuples(index=False):
        key = (row.instrument_class, row.strike_price)
        if key in contracts:
            raise Unavailable("Multiple symbols at the same side and strike")
        contracts[key] = row

    def strikes(side, quantity):
        return {strike for (cp, strike), row in contracts.items()
                if cp == side and _executable(row, quantity)}

    def nearest(candidates, target):
        if not candidates:
            raise Unavailable("No executable entry legs")
        return min(candidates, key=lambda strike: (abs(strike - target), strike))

    if strategy.kind == "straddle":
        quantity = 1 if strategy.direction == "long" else -1
        strike = nearest(strikes("C", quantity) & strikes("P", quantity), center)
        specs = [("C", strike, quantity), ("P", strike, quantity)]
    elif strategy.kind == "risk_reversal":
        quantity = 1 if strategy.direction == "bullish" else -1
        call = nearest(strikes("C", quantity), spot * exp(move))
        put = nearest(strikes("P", -quantity), spot * exp(-move))
        if put >= call:
            raise Unavailable("Risk reversal requires distinct ordered wings")
        specs = [("C", call, quantity), ("P", put, -quantity)]
    else:
        width = strategy.wing_width
        target_width = width if width is not None else center * np.sinh(move)
        # Integer ticks avoid binary roundoff when testing equal wing widths.
        tick = 1_000_000
        wings = {round(strike * tick): strike for strike in strikes("C", 1)}
        combinations = []
        for middle in strikes("C", -2):
            middle_tick = round(middle * tick)
            for lower_tick, lower in wings.items():
                gap = middle_tick - lower_tick
                if gap <= 0 or (width is not None and gap != round(width * tick)):
                    continue
                upper = wings.get(middle_tick + gap)
                if upper is not None:
                    combinations.append((abs(middle - center), abs(gap / tick - target_width),
                                         middle, lower, upper))
        if not combinations:
            raise Unavailable("No executable equal-width butterfly")
        _, _, middle, lower, upper = min(combinations)
        specs = [("C", strike, n) for strike, n in ((lower, 1), (middle, -2), (upper, 1))]
    return [Leg(contracts[side, strike].symbol, side, strike, qty) for side, strike, qty in specs]


class Backtest:
    """Replay synchronized merged quotes with explicit timestamps and a session calendar.

    Quotes use event_vol_research.merge_data's columns. The optional surface callable takes
    (UTC timestamp, UTC expiration date) and returns (ATM IV, skew, quadratic
    coefficient, fitted forward), or None. Fit inputs must be point in time.
    """

    def __init__(self, quotes, schedule, *, surface=None, rate=0.04,
                 off_hour_weight=0.4, multiplier=100):
        if (not isfinite(rate) or not 0 <= off_hour_weight <= 1
                or not isfinite(multiplier) or multiplier <= 0):
            raise ValueError("Invalid rate, off-hour weight, or multiplier")
        self.schedule = schedule
        self.surface = surface
        self.rate = rate
        self.weight = off_hour_weight
        self.multiplier = multiplier
        prices = ["bid_px_00", "ask_px_00", "spot_bid", "spot_ask", "strike_price"]
        valid = (
            np.isfinite(quotes[prices]).all(axis=1)
            & quotes.ts_recv.eq(quotes.spot_time) & quotes.expiration.notna()
            & quotes.instrument_class.isin(["C", "P"]) & quotes.symbol.notna()
            & quotes.bid_px_00.gt(0) & quotes.ask_px_00.ge(quotes.bid_px_00)
            & quotes.spot_bid.gt(0) & quotes.spot_ask.ge(quotes.spot_bid)
            & quotes.strike_price.gt(0)
        )
        self.quotes = quotes.loc[valid].sort_values("ts_recv")
        self.snapshots = self.quotes.groupby("ts_recv", sort=False)
        self.chains = self.quotes.groupby(["expiration", "symbol"], sort=False)
        self._clock = lru_cache(maxsize=None)(self._clock)

    def _expiry_close(self, expiry):
        return self.schedule.at[expiry.tz_convert("UTC").tz_localize(None).normalize(), "market_close"]

    def _clock(self, timestamp, expiry):
        close = self._expiry_close(expiry)
        _, _, calendar, variance = h.calculate_variance_time(timestamp, close, self.schedule, self.weight)
        if not isfinite(variance) or variance <= 0:
            raise Unavailable("Option has no positive variance time")
        return exp(-self.rate * calendar), variance

    def run(self, strategy, *, entry, exit_start, exit_end, event_time=None,
            implied_move=None, max_dte=31, delta_hedge=False, fees=0.0):
        """Run each eligible expiry with one structure per expiry."""
        entry, exit_start, exit_end = map(pd.Timestamp, (entry, exit_start, exit_end))
        if any(time.tzinfo is None for time in (entry, exit_start, exit_end)):
            raise ValueError("Execution timestamps must be timezone-aware")
        if not entry < exit_start <= exit_end or not isfinite(max_dte) or max_dte <= 0:
            raise ValueError("Invalid execution window or max DTE")
        if not isfinite(fees) or fees < 0:
            raise ValueError("Fees must be finite and nonnegative")
        if event_time is not None:
            event_time = pd.Timestamp(event_time)
            if event_time.tzinfo is None or not entry < event_time <= exit_start:
                raise ValueError("Event must be timezone-aware and between entry and exit")
        tags = dict(strategy=strategy.label, entry_time=entry, implied_move=implied_move)
        if strategy.needs_move and (implied_move is None or not isfinite(implied_move) or implied_move < 0):
            return Replay(pd.DataFrame([{**tags, "status": "No usable implied move"}]), pd.DataFrame())
        if entry not in self.snapshots.groups:
            return Replay(pd.DataFrame([{**tags, "status": "No entry quotes"}]), pd.DataFrame())

        trades, legs = [], []
        for expiry, chain in self.snapshots.get_group(entry).groupby("expiration"):
            close = self._expiry_close(expiry)
            if not exit_start < close <= entry + pd.Timedelta(days=max_dte):
                continue
            row = {**tags, "expiration": expiry}
            try:
                selection = _select(chain, strategy, implied_move)
                trade, detail = self._trade(selection, expiry, entry, exit_start,
                                            min(exit_end, close - pd.Timedelta(1, unit="ns")),
                                            event_time, delta_hedge, fees)
            except Unavailable as error:
                row["status"] = str(error)
            else:
                row.update(trade)
                legs.extend({**tags, "expiration": expiry, **leg} for leg in detail)
            trades.append(row)
        if not trades:
            trades.append({**tags, "status": "No eligible expiry"})
        return Replay(pd.DataFrame(trades), pd.DataFrame(legs))

    def _trade(self, legs, expiry, entry, first, last, event_time, hedge, fees):
        histories = []
        for leg in legs:
            frame = self.chains.get_group((expiry, leg.symbol))
            frame = frame.loc[frame.ts_recv.between(entry, last)].set_index("ts_recv")
            if not frame.index.is_unique or entry not in frame.index:
                raise Unavailable("Ambiguous or missing contract history")
            histories.append(frame)

        exit_times = histories[0].index[histories[0].index >= first]
        for leg, frame in zip(legs, histories, strict=True):
            size = frame.ask_sz_00 if leg.quantity < 0 else frame.bid_sz_00
            exit_times = exit_times.intersection(size.index[size.ge(abs(leg.quantity))])
        if exit_times.empty:
            raise Unavailable("No same-minute exit for all original legs")
        exit_time = exit_times.min()
        data = [frame.loc[:exit_time] for frame in histories]
        detail = []
        for leg, history in zip(legs, data, strict=True):
            opening = _fill(history.loc[entry], leg.quantity)
            closing = _fill(history.loc[exit_time], -leg.quantity)
            pnl = leg.quantity * self.multiplier * (closing - opening)
            row = dict(symbol=leg.symbol, side=leg.side, strike=leg.strike, quantity=leg.quantity,
                       multiplier=self.multiplier, entry_fill=opening, exit_fill=closing,
                       exit_time=exit_time, option_pnl_usd=pnl)
            try:
                row.update(self._attribute(leg, history, expiry, event_time, pnl))
            except Unavailable as error:
                row["attribution_status"] = str(error)
            detail.append(row)

        trade = dict(status="priced", exit_time=exit_time, entry_stock_bid=data[0].iloc[0].spot_bid,
                     exit_stock_bid=data[0].iloc[-1].spot_bid, fees_usd=fees,
                     option_pnl_usd=sum(row["option_pnl_usd"] for row in detail))
        trade["unhedged_pnl_usd"] = trade["option_pnl_usd"] - fees
        trade["hedge_status"] = "disabled"
        if hedge:
            try:
                trade.update(self._hedge(legs, data, expiry, first, event_time))
            except Unavailable as error:
                trade["hedge_status"] = str(error)
            else:
                trade["hedge_status"] = "priced"
                trade["hedged_pnl_usd"] = trade["unhedged_pnl_usd"] + trade["stock_pnl_usd"]
        trade["attribution_status"] = "attributed"
        if any(row["attribution_status"] != "attributed" for row in detail):
            trade["attribution_status"] = "Incomplete leg attribution"
        else:
            for name in (*FACTORS, *VOL_FACTORS):
                for key in (f"{name}_usd", f"earnings_gap_{name}_usd"):
                    trade[key] = sum(row[key] for row in detail)
            trade["vol_breakdown_status"] = ("attributed" if all(
                row["vol_breakdown_status"] == "attributed" for row in detail
            ) else "incomplete" if self.surface is not None else "disabled")
            for key in ("option_execution_and_model_basis_usd", "clipped_bid_marks",
                        "vol_intervals_unavailable", "earnings_gap_vol_intervals_unavailable"):
                trade[key] = sum(row[key] for row in detail)
        return trade, detail

    def _hedge(self, legs, histories, expiry, exit_start, event_time):
        times = histories[0].index
        for history in histories[1:]:
            times = times.intersection(history.index)
        cutoff = event_time if event_time is not None else exit_start
        shares, cash, updates = 0, 0.0, 0
        initial = carried_at = None
        for timestamp in times[times < cutoff]:
            rows = [history.loc[timestamp] for history in histories]
            clock = self._clock(timestamp, expiry)
            discount, tau = clock
            deltas = []
            for leg, quote in zip(legs, rows, strict=True):
                iv = h.implied_vol(quote.bid_px_00, quote.spot_bid / discount,
                                   leg.strike, discount, tau, leg.side)
                if not isfinite(iv) or iv <= 0:
                    break
                deltas.append(leg.quantity * _delta(quote.spot_bid, leg.strike, clock, iv, leg.side))
            if len(deltas) != len(legs):
                if updates == 0:
                    raise Unavailable("Cannot infer entry bid-side delta")
                continue
            desired = int(np.rint(-self.multiplier * sum(deltas)))
            if initial is None:
                initial = desired
            change = desired - shares
            cash -= change * (rows[0].spot_ask if change > 0 else rows[0].spot_bid)
            shares, carried_at = desired, timestamp
            updates += 1
        if initial is None:
            raise Unavailable("No pre-event hedge snapshots")
        closing = histories[0].iloc[-1]
        cash += shares * (closing.spot_bid if shares > 0 else closing.spot_ask)
        return dict(stock_pnl_usd=cash, initial_hedge_shares=initial, carried_hedge_shares=shares,
                    last_hedge_time=carried_at, hedge_updates=updates)

    def _attribute(self, leg, history, expiry, event_time, pnl):
        totals, gap = np.zeros(4), np.zeros(4)
        vol_totals, vol_gap = np.zeros(4), np.zeros(4)
        unavailable, gap_unavailable, clipped = 0, 0, 0
        previous = None
        scale = leg.quantity * self.multiplier
        first_price = None
        for quote in history.itertuples():
            timestamp, spot = quote.Index, quote.spot_bid
            clock = self._clock(timestamp, expiry)
            iv, mark, bounded = _mark(quote, leg, clock)
            clipped += int(bounded)
            if previous is None:
                first_price = mark
            else:
                old_time, old_spot, old_clock, old_iv, old_mark = previous
                is_gap = event_time is not None and old_time < event_time <= timestamp
                delta = _delta(old_spot, leg.strike, old_clock, old_iv, leg.side) * (spot - old_spot)
                spot_price = _price(spot, leg.strike, old_clock, old_iv, leg.side)
                time_price = _price(spot, leg.strike, clock, old_iv, leg.side)
                factors = scale * np.array([delta, spot_price - old_mark - delta,
                                            time_price - spot_price, mark - time_price])
                totals += factors
                if is_gap:
                    gap += factors
                if self.surface is not None:
                    parts = split_vega(leg.strike, leg.side, spot, clock, old_iv, iv,
                                       self.surface(old_time, expiry), self.surface(timestamp, expiry))
                    if parts is None:
                        unavailable += 1
                        gap_unavailable += int(is_gap)
                    else:
                        vol_totals += scale * parts
                        if is_gap:
                            vol_gap += scale * parts
            previous = timestamp, spot, clock, iv, mark

        basis = pnl - scale * (previous[-1] - first_price)
        if not np.isclose(totals.sum() + basis, pnl, atol=1e-7, rtol=0):
            raise ArithmeticError("Option P&L attribution did not reconcile")
        vol_available = self.surface is not None and unavailable == 0
        gap_available = self.surface is not None and event_time is not None and gap_unavailable == 0
        for values, target, available in ((vol_totals, totals[-1], vol_available),
                                          (vol_gap, gap[-1], gap_available)):
            if available and not np.isclose(values.sum(), target, atol=1e-7, rtol=0):
                raise ArithmeticError("Vega subcomponents did not reconcile")
        return {
            **{f"{name}_usd": value for name, value in zip(FACTORS, totals)},
            **{f"earnings_gap_{name}_usd": value for name, value in zip(FACTORS, gap)},
            **{f"{name}_usd": value if vol_available else np.nan
               for name, value in zip(VOL_FACTORS, vol_totals)},
            **{f"earnings_gap_{name}_usd": value if gap_available else np.nan
               for name, value in zip(VOL_FACTORS, vol_gap)},
            "option_execution_and_model_basis_usd": basis,
            "attribution_status": "attributed",
            "vol_breakdown_status": ("attributed" if vol_available else
                                     "incomplete" if self.surface is not None else "disabled"),
            "clipped_bid_marks": clipped,
            "vol_intervals_unavailable": unavailable,
            "earnings_gap_vol_intervals_unavailable": gap_unavailable,
        }
