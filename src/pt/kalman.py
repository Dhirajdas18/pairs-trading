"""Time-varying hedge ratios: a Kalman filter and a rolling regression, both using only past data.

Kalman model (the same random-walk idea as beta-capm's Kalman beta, with an intercept):
    log A_t = alpha_t + beta_t * log B_t + e_t,   e_t ~ N(0, r)
    (alpha_t, beta_t) = (alpha_{t-1}, beta_{t-1}) + w_t,   w_t ~ N(0, q * I)
Only q / r shapes the path. r is the formation OLS residual variance; q / r is picked from a grid by
maximum likelihood on the formation window only. The filter runs from the start of the formation
window, so it is warmed up by the test year, and each day's hedge ratio is the prediction made
before that day's prices (beta_{t|t-1}).
"""

import numpy as np

GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def kalman_hedge(la, lb, ratio: float, r: float, init: tuple[float, float]) -> dict:
    """Filter (alpha, beta) through log prices la, lb.

    Returns alpha and beta predicted for each day before seeing it, the one-step prediction error
    e_t = la_t - alpha_pred - beta_pred * lb_t, its variance s_t, and the log-likelihood.
    """
    la, lb = np.asarray(la, float), np.asarray(lb, float)
    n = len(la)
    q = ratio * r
    x = np.array(init, float)
    p = np.eye(2) * 1.0  # loose start: the first weeks of formation settle it
    a_pred, b_pred, err, var = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    ll = 0.0
    for t in range(n):
        p = p + q * np.eye(2)
        h = np.array([1.0, lb[t]])
        a_pred[t], b_pred[t] = x
        e = la[t] - h @ x
        s = h @ p @ h + r
        k = p @ h / s
        x = x + k * e
        p = p - np.outer(k, h) @ p
        err[t], var[t] = e, s
        ll += -0.5 * (np.log(2 * np.pi * s) + e * e / s)
    return {"alpha": a_pred, "beta": b_pred, "error": err, "var": var, "loglik": ll}


def fit_ratio(la, lb, r: float, init: tuple[float, float], grid=GRID) -> float:
    """The q / r in `grid` with the highest likelihood on (la, lb), skipping the first 60 days
    while the filter settles."""
    best, best_ll = grid[0], -np.inf
    for g in grid:
        f = kalman_hedge(la, lb, g, r, init)
        ll = -0.5 * np.sum(np.log(2 * np.pi * f["var"][60:]) + f["error"][60:] ** 2 / f["var"][60:])
        if ll > best_ll:
            best, best_ll = g, ll
    return best


def rolling_hedge(la, lb, window: int = 120) -> np.ndarray:
    """Hedge ratio for day t from OLS of log A on log B over the `window` days before t (t itself
    excluded); NaN until there are `window` past days."""
    la, lb = np.asarray(la, float), np.asarray(lb, float)
    out = np.full(len(la), np.nan)
    for t in range(window, len(la)):
        y, x = la[t - window : t], lb[t - window : t]
        xc = x - x.mean()
        out[t] = np.dot(xc, y - y.mean()) / np.dot(xc, xc)
    return out
