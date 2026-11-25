"""A pairs strategy with stock-futures costs: every pair, hedge ratio and z-score is fitted only on
data before the year it trades (the rule itself was chosen in pt.spread and pt.search on the same
years).

Each test year: take the `n` lowest Engle-Granger p-values among same-industry pairs fitted on the
two years before (pt/03), trade each on a rolling z-score of its spread (pt/02), give every pair an
equal slice of capital, and close everything on the year's last day.

Futures legs: a stock future's return is approximated by the stock's dividend-inclusive return minus
cash interest (the carry built into its price), so the strategy's return is what it earns above the
cash the capital would sit in. Basis risk is left out.

Costs (`FUT`): Zerodha's equity-futures charges as listed in November 2026 (the same figures as
beta-capm's hedging study), applied to every year, plus slippage of half a bid-ask spread per side.
Brokerage is ₹20 an order, entered as a rate on a ₹5 lakh order. Positions open at a month end are
rolled to the next contract at a full round trip on both legs. Expiry is approximated by the last
trading day of each month (NSE stock futures expire on the month's last Tuesday, Thursday before
September 2025).
"""

import numpy as np
import pandas as pd

from pt.pairs import positions
from pt.spread import rolling_z

RF = 0.065  # cash rate, a year (91-day T-bills ran roughly 3-7% over 2009-2026)
FUT = {
    "stt_sell": 0.0005,
    "exchange": 0.0000183,
    "sebi": 0.000001,
    "stamp_buy": 0.00002,
    "brokerage": 20.0 / 500_000,  # ₹20 an order on a ₹5 lakh order
    "gst": 0.18,
}


def side_rates(slippage: float = 0.0005) -> tuple[float, float]:
    """Cost per rupee bought and per rupee sold of futures, slippage included."""
    fees = (FUT["exchange"] + FUT["sebi"] + FUT["brokerage"]) * (1 + FUT["gst"])
    return fees + FUT["stamp_buy"] + slippage, fees + FUT["stt_sell"] + slippage


def pair_returns(
    spread: np.ndarray,
    ra: np.ndarray,
    rb: np.ndarray,
    beta: float,
    first: int,
    dt: np.ndarray,
    month_end: np.ndarray,
    window: int = 60,
    entry: float = 2.0,
    exit: float = 0.0,
    lag: int = 1,
    slippage: float = 0.0005,
    rolls: bool = True,
    stop_z: float | None = None,
    max_days: int | None = None,
    stop_loss: float | None = None,
    retired: np.ndarray | None = None,
    beta_path: np.ndarray | None = None,
    z_input: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Daily returns above cash, before and after costs, of one pair over days `first`..end.

    `spread` covers formation and test days (formation first, so the rolling z can look back);
    `ra`, `rb` are the legs' simple returns into each day; `dt` years since the previous close.
    The spread position decided at close t (from z_t) is traded at close t + `lag` and earns returns
    after that; it starts flat on the first test day and is closed by the last close. Long spread =
    weight 1 on A and -beta on B, scaled to a gross exposure of 1.
    Stops (pt/05), each closing the trade and blocking re-entry until |z| < 0.5: `stop_z` when |z|
    passes it, `max_days` after that many days held, `stop_loss` when the trade has lost that share
    of the pair's capital. `retired` (bool per day) closes the pair for good from the first True.
    Time-varying hedge (pt/06): `beta_path` gives each day's hedge ratio known before that day;
    a trade's legs use the ratio of the day it was decided, kept until it closes. `z_input`
    replaces the rolling z of `spread` as the signal.
    Returns (gross, net, trades) for the test days.
    """
    n = len(spread)
    z = rolling_z(spread, window) if z_input is None else np.array(z_input, float)
    z[:first] = np.nan  # nothing is opened from formation-day signals
    if stop_z is None and max_days is None and stop_loss is None and retired is None:
        decided = positions(z, entry, exit) if exit > 0 else _cross_zero(z, entry)
        held = np.zeros(n)
        held[lag:] = decided[:-lag] if lag else decided
    else:
        bu = beta if beta_path is None else np.nan_to_num(beta_path, nan=beta)
        unit = (ra - RF * dt - bu * (rb - RF * dt)) / (1 + np.abs(bu))
        held = _with_stops(z, unit, entry, lag, stop_z, max_days, stop_loss, retired)
    held[: first + lag] = 0.0
    held[-1] = 0.0  # flat by the last close
    if beta_path is None:
        bt = np.full(n, beta)
    else:
        bt = np.full(n, np.nan)
        for t in range(n):
            if held[t] != 0:
                new = t == 0 or held[t] != held[t - 1]
                bt[t] = beta_path[max(t - lag, 0)] if new else bt[t - 1]
        bt = np.nan_to_num(bt, nan=0.0)
    wa, wb = held / (1 + np.abs(bt)), -held * bt / (1 + np.abs(bt))
    gross = np.zeros(n)
    gross[1:] = wa[:-1] * (ra[1:] - RF * dt[1:]) + wb[:-1] * (rb[1:] - RF * dt[1:])
    buy, sell = side_rates(slippage)
    cost = np.zeros(n)
    for w in (wa, wb):
        d = np.diff(w, prepend=0.0)
        cost += np.where(d > 0, d * buy, -d * sell)
        roll = month_end & (np.abs(w) > 0) & (np.r_[np.abs(w[1:]), 0.0] > 0)
        cost += rolls * roll * np.abs(w) * (buy + sell)
    trades = int(np.sum((held != 0) & (np.r_[0.0, held[:-1]] != held)))
    return gross[first:], (gross - cost)[first:], trades


def _with_stops(z, unit, entry, lag, stop_z, max_days, stop_loss, retired) -> np.ndarray:
    """Held spread position per day for the exit-at-zero rule with stops; decisions at close t
    use z_t and the trade's results up to t, and are held from close t + lag."""
    n = len(z)
    decided, held = np.zeros(n), np.zeros(n)
    p, blocked, pnl, days, out_for_good = 0.0, False, 0.0, 0, False
    for t in range(n):
        held[t] = decided[t - lag] if t >= lag else 0.0
        if t > 0 and held[t - 1] != 0:
            pnl += held[t - 1] * unit[t]
        if held[t] != 0 and (t == 0 or held[t] != held[t - 1]):
            pnl, days = 0.0, 0
        days += held[t] != 0
        v = z[t]
        if retired is not None and retired[t]:
            out_for_good = True
        if np.isnan(v) or out_for_good:
            p, blocked = 0.0, False
        elif blocked:
            blocked = abs(v) >= 0.5
        elif p == 0.0:
            p = -1.0 if v > entry else 1.0 if v < -entry else 0.0
        elif (p < 0 and v <= 0) or (p > 0 and v >= 0):
            p = 0.0
        elif (
            (stop_z is not None and abs(v) > stop_z)
            or (max_days is not None and days >= max_days)
            or (stop_loss is not None and held[t] != 0 and pnl < -stop_loss)
        ):
            p, blocked = 0.0, True
        decided[t] = p
    return held


def health_check(la, lb, first: int, month_end: np.ndarray, threshold: float, lookback=250):
    """Retire a pair from the first month end in the test where Engle-Granger on the last
    `lookback` days (hedge ratio refitted) gives p above `threshold`. Bool per day."""
    from pt.pairs import engle_granger

    out = np.zeros(len(la), bool)
    for t in np.flatnonzero(month_end):
        if t < first:
            continue
        if (
            engle_granger(la[t - lookback + 1 : t + 1], lb[t - lookback + 1 : t + 1])["pvalue"]
            > threshold
        ):
            out[t:] = True
            break
    return out


def _cross_zero(z: np.ndarray, entry: float) -> np.ndarray:
    """Positions for exit at the mean: short above +entry, long below -entry, flat once z has
    crossed zero (changed sign) since entry. NaN z = flat."""
    out = np.zeros(len(z))
    p = 0.0
    for t, v in enumerate(z):
        if np.isnan(v):
            p = 0.0
        elif p == 0.0:
            p = -1.0 if v > entry else 1.0 if v < -entry else 0.0
        elif (p < 0 and v <= 0) or (p > 0 and v >= 0):
            p = 0.0
        out[t] = p
    return out


def run(
    logp: pd.DataFrame,
    picks: pd.DataFrame,
    n: int = 20,
    health: float | None = None,
    hedge: str = "fixed",
    sizing: str = "equal",
    **kw,
) -> pd.DataFrame:
    """Daily portfolio returns above cash, gross and net, over every test year in `picks`.

    `picks`: one row per candidate pair-year with year, a, b, beta, pvalue (pt.search.scan rows,
    same industry). Each year the `n` lowest p-values are traded with 1/n of capital each; empty
    slots sit in cash (0 above cash). `hedge`: "fixed" (formation OLS, pt/04), "rolling" (OLS on
    the previous 120 days), "kalman" (Kalman hedge ratio, rolling 60-day z of its spread) or
    "kalman signal" (Kalman hedge ratio, z = prediction error over its standard deviation).
    `sizing`: "equal" (1/n each) or "inverse vol" (capital in proportion to 1 / the standard
    deviation of the pair's daily hedged return over its formation years, same total).
    """
    from pt.kalman import fit_ratio, kalman_hedge, rolling_hedge

    out = []
    for year, g in picks.groupby("year"):
        chosen = g.nsmallest(n, "pvalue")
        f0 = pd.Timestamp(year - 2, 1, 1)
        t0, t1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
        px = logp.loc[f0:t1]
        test_idx = px.loc[t0:].index
        first = len(px) - len(test_idx)
        dt = np.r_[0.0, np.diff(px.index.values).astype("timedelta64[D]").astype(float) / 365.0]
        month = px.index.to_period("M")
        month_end = np.r_[month[1:] != month[:-1], False]
        gross = np.zeros(len(test_idx))
        net = np.zeros(len(test_idx))
        trades = 0
        weights = np.full(len(chosen), 1.0 / n)
        if sizing == "inverse vol":
            sd = []
            for r in chosen.itertuples():
                fa, fb = px[r.a].ffill().to_numpy()[:first], px[r.b].ffill().to_numpy()[:first]
                u = (np.expm1(np.diff(fa)) - r.beta * np.expm1(np.diff(fb))) / (1 + abs(r.beta))
                sd.append(np.nanstd(u))  # a stock listed during formation starts with gaps
            inv = 1 / np.array(sd)
            weights = inv / inv.sum() * len(chosen) / n
        for w_pair, r in zip(weights, chosen.itertuples(), strict=True):
            la, lb = px[r.a].ffill().to_numpy(), px[r.b].ffill().to_numpy()
            ra, rb = np.r_[0.0, np.expm1(np.diff(la))], np.r_[0.0, np.expm1(np.diff(lb))]
            if health is not None:
                kw["retired"] = health_check(la, lb, first, month_end, health)
            extra = {}
            spread = la - r.beta * lb
            if hedge == "rolling":
                bp = rolling_hedge(la, lb, 120)
                spread = la - np.nan_to_num(bp, nan=r.beta) * lb
                extra["beta_path"] = bp
            elif hedge.startswith("kalman"):
                fa, fb = la[:first], lb[:first]
                a0 = fa.mean() - r.beta * fb.mean()
                res_var = float(np.var(fa - a0 - r.beta * fb))
                ratio = fit_ratio(fa, fb, res_var, (a0, r.beta))
                kf = kalman_hedge(la, lb, ratio, res_var, (a0, r.beta))
                extra["beta_path"] = kf["beta"]
                spread = la - kf["beta"] * lb
                if hedge == "kalman signal":
                    extra["z_input"] = kf["error"] / np.sqrt(kf["var"])
            g_, n_, k = pair_returns(spread, ra, rb, r.beta, first, dt, month_end, **kw, **extra)
            gross += np.nan_to_num(g_) * w_pair
            net += np.nan_to_num(n_) * w_pair
            trades += k
        out.append(
            pd.DataFrame({"gross": gross, "net": net, "year": year}, index=test_idx).assign(
                trades=trades
            )
        )
    return pd.concat(out)


def lots_needed(
    beta: float, value_a: float, value_b: float, tol: float = 0.10, max_lots: int = 100
) -> dict:
    """Smallest whole-lot position (n_a lots of A, n_b of B) whose rupee ratio B : A is within
    `tol` of the hedge ratio `beta`; value_a, value_b are one lot's rupee value."""
    best = None
    for na in range(1, max_lots + 1):
        nb = max(1, round(beta * na * value_a / value_b))
        err = nb * value_b / (beta * na * value_a) - 1
        if best is None or abs(err) < abs(best[2]):
            best = (na, nb, err)
        if abs(err) <= tol:
            return {
                "lots_a": na,
                "lots_b": nb,
                "error": err,
                "notional": na * value_a + nb * value_b,
            }
    na, nb, err = best
    return {"lots_a": na, "lots_b": nb, "error": err, "notional": na * value_a + nb * value_b}


def vol_target(r: pd.Series, target: float = 0.10, lookback: int = 60, cap: float = 3.0):
    """Scale daily returns above cash to a target annual volatility: each day's scale is
    target / the realised volatility of the previous `lookback` days (today excluded), capped at
    `cap` and 1 until there is enough history. Costs scale with it (the trades get bigger); the cost
    of changing the scale itself is left out."""
    vol = r.rolling(lookback).std().shift(1) * np.sqrt(250)
    lev = (target / vol).clip(upper=cap).fillna(1.0)
    return r * lev, lev


def summary(r: pd.Series) -> dict:
    """Annual return (calendar time), volatility, Sharpe, worst drawdown of daily returns."""
    years = (r.index[-1] - r.index[0]).days / 365.25
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(len(r) / years)
    ann = eq.iloc[-1] ** (1 / years) - 1
    return {
        "annual": float(ann),
        "vol": float(vol),
        "sharpe": float(r.mean() * len(r) / years / vol) if vol > 0 else np.nan,
        "max drawdown": float((eq / eq.cummax() - 1).min()),
    }
