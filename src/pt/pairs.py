"""Core pair maths: hedge ratio, spread, z-score, half-life, Engle-Granger test and a rule backtest.

numpy only, so every function can be mirrored line for line in pairs.js (tests compare the two).
Inputs are aligned 1-D arrays without gaps; drop days where either stock has no price first.
"""

import math

import numpy as np

# MacKinnon (1994) response-surface coefficients for the asymptotic p-value of a Dickey-Fuller
# t-statistic, constant case, as in statsmodels.tsa.adfvalues (mackinnonp, regression="c").
# Key: number of I(1) variables N (1 = plain ADF, 2 = Engle-Granger residual test on a pair).
# p = Phi(c0 + c1 t + c2 t^2 [+ c3 t^3]): small-p coefficients when t <= star, else large-p.
MACKINNON_C = {
    1: {
        "max": 2.74,
        "min": -18.83,
        "star": -1.61,
        "small": [2.1659, 1.4412, 0.038269],
        "large": [1.7339, 0.93202, -0.12745, -0.010368],
    },
    2: {
        "max": 0.92,
        "min": -18.86,
        "star": -2.62,
        "small": [2.92, 1.5012, 0.039796],
        "large": [2.1945, 0.64695, -0.29198, -0.042377],
    },
}


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and their plain OLS standard errors (residual variance on n - k)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return coef, se


def hedge_ratio(log_a, log_b) -> tuple[float, float]:
    """(alpha, beta) from OLS of log A on log B with an intercept."""
    a, b = np.asarray(log_a, float), np.asarray(log_b, float)
    mb, ma = b.mean(), a.mean()
    beta = ((b - mb) * (a - ma)).sum() / ((b - mb) ** 2).sum()
    return float(ma - beta * mb), float(beta)


def spread(log_a, log_b, alpha: float, beta: float) -> np.ndarray:
    """Log A minus its fitted value from log B: the residual the pair trades."""
    return np.asarray(log_a, float) - alpha - beta * np.asarray(log_b, float)


def zscore(s, window: int | None = None) -> np.ndarray:
    """Spread in standard deviations from its mean.

    window None: full-sample mean and standard deviation (ddof 1), so it uses future data and is
    for description only. With a window: mean and sd of the last `window` values up to and
    including t (NaN before that), so a trade decided at t sees nothing after t.
    """
    s = np.asarray(s, float)
    if window is None:
        return (s - s.mean()) / s.std(ddof=1)
    out = np.full(len(s), np.nan)
    for t in range(window - 1, len(s)):
        w = s[t - window + 1 : t + 1]
        out[t] = (s[t] - w.mean()) / w.std(ddof=1)
    return out


def half_life(s) -> float:
    """Days for a deviation to halve, from the AR(1) fit ds_t = a + b s_{t-1}.

    Half-life = -ln 2 / ln(1 + b); infinite when b >= 0 (no pull back to the mean).
    """
    s = np.asarray(s, float)
    ds, lag = np.diff(s), s[:-1]
    ml, md = lag.mean(), ds.mean()
    b = ((lag - ml) * (ds - md)).sum() / ((lag - ml) ** 2).sum()
    if b >= 0:
        return math.inf
    if b <= -1:
        return 0.0
    return -math.log(2) / math.log(1 + b)


def mackinnon_p(tstat: float, n_vars: int = 2) -> float:
    """MacKinnon's approximate asymptotic p-value for a Dickey-Fuller t-statistic, constant case."""
    c = MACKINNON_C[n_vars]
    if tstat > c["max"]:
        return 1.0
    if tstat < c["min"]:
        return 0.0
    coef = c["small"] if tstat <= c["star"] else c["large"]
    z = sum(k * tstat**i for i, k in enumerate(coef))
    return 0.5 * math.erfc(-z / math.sqrt(2))  # standard normal cdf


def adf(series, lags: int = 1, constant: bool = True, n_vars: int = 1) -> tuple[float, float]:
    """Augmented Dickey-Fuller test with a fixed number of lagged differences.

    Regresses dx_t on x_{t-1}, dx_{t-1}..dx_{t-lags} (and a constant if `constant`), over the
    len(x) - lags - 1 days where every term exists. Returns the t-statistic on x_{t-1} and
    MacKinnon's p-value for `n_vars` I(1) variables (1 for a plain series, 2 for the residual of a
    pair regression, which is already demeaned and so is tested without a constant).
    """
    x = np.asarray(series, float)
    dx = np.diff(x)
    y = dx[lags:]
    cols = [x[lags:-1]] + [dx[lags - j : len(dx) - j] for j in range(1, lags + 1)]
    if constant:
        cols.append(np.ones(len(y)))
    coef, se = _ols(y, np.column_stack(cols))
    t = float(coef[0] / se[0])
    return t, mackinnon_p(t, n_vars)


def engle_granger(log_a, log_b, lags: int = 1) -> dict:
    """Engle-Granger two-step cointegration test of log A on log B.

    Step 1: OLS hedge ratio with an intercept. Step 2: ADF on the residual (no constant, `lags`
    lagged differences), p-value from MacKinnon's two-variable table. Matches
    statsmodels.tsa.stattools.coint(a, b, trend="c", maxlag=lags, autolag=None).
    """
    alpha, beta = hedge_ratio(log_a, log_b)
    s = spread(log_a, log_b, alpha, beta)
    t, p = adf(s, lags, constant=False, n_vars=2)
    return {"alpha": alpha, "beta": beta, "tstat": t, "pvalue": p, "half_life": half_life(s)}


def positions(z, entry: float = 2.0, exit: float = 0.5, stop: float | None = None) -> np.ndarray:
    """Spread position at each close: -1 short, +1 long, 0 flat, from the z-score at that close.

    Short when z > entry, long when z < -entry, flat once |z| < exit, otherwise hold. With a stop,
    |z| > stop closes the trade and stays flat until |z| falls back below exit. NaN z means flat.
    """
    z = np.asarray(z, float)
    out = np.zeros(len(z))
    p, stopped = 0.0, False
    for t, v in enumerate(z):
        if np.isnan(v):
            p, stopped = 0.0, False
        elif stop is not None and abs(v) > stop:
            p, stopped = 0.0, True
        elif stopped:
            if abs(v) < exit:
                stopped = False
        elif v > entry:
            p = -1.0
        elif v < -entry:
            p = 1.0
        elif abs(v) < exit:
            p = 0.0
        out[t] = p
    return out


COST_PER_SIDE = 0.0005  # 5 bp of each rupee traded: futures charges plus slippage, refined later


def backtest(
    log_a,
    log_b,
    beta: float,
    z,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float | None = None,
    cost_per_side: float = COST_PER_SIDE,
) -> dict:
    """Daily returns of trading the spread on a z-score rule, after costs.

    The position is decided at close t from z_t and earns the legs' simple returns from t to t+1,
    so nothing after t affects it. Long the spread = weight 1 on A and -beta on B, scaled so gross
    exposure is 1 (each leg's weight divided by 1 + |beta|). Each day's cost is cost_per_side times
    the change in gross exposure across both legs, charged to the next day's return.

    Returns returns (length n; returns[0] = 0), positions, trades (new positions opened, flips
    included) and equity (growth of 1).
    """
    a, b = np.asarray(log_a, float), np.asarray(log_b, float)
    pos = positions(z, entry, exit, stop)
    wa = pos / (1 + abs(beta))
    wb = -pos * beta / (1 + abs(beta))
    ra, rb = np.expm1(np.diff(a)), np.expm1(np.diff(b))
    turnover = np.abs(np.diff(wa, prepend=0.0)) + np.abs(np.diff(wb, prepend=0.0))
    r = np.zeros(len(a))
    r[1:] = wa[:-1] * ra + wb[:-1] * rb - cost_per_side * turnover[:-1]
    prev = np.r_[0.0, pos[:-1]]
    trades = int(((pos != 0) & (pos != prev)).sum())
    return {"returns": r, "positions": pos, "trades": trades, "equity": np.cumprod(1 + r)}
