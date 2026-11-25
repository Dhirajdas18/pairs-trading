"""Building the spread: which hedge ratio, which z-score, and whether half-life carries over.

Every same-sector pair is fitted on a formation window and followed through the next year with the
fitted numbers frozen (see pt.coint.windows). Rolling z-scores in the test year may look back into
the formation window, which is past data at that point.
"""

import numpy as np
import pandas as pd

from pt.coint import same_sector_pairs, windows
from pt.pairs import engle_granger, half_life

Z_WINDOWS = (20, 60, 120)


def hedge_ratios(log_a, log_b) -> dict[str, tuple[float, float]]:
    """(alpha, beta) in the form log A = alpha + beta * log B, four ways:
    - OLS A on B: least squares of A on B (the Engle-Granger choice).
    - OLS B on A: least squares of B on A, turned around. Differs from the first whenever the fit
      is not perfect, which is why the regression direction matters.
    - TLS: total least squares (orthogonal regression), which treats both stocks alike.
    - ratio: beta fixed at 1, i.e. trade the plain price ratio with equal rupees on each side.
    """
    a, b = np.asarray(log_a, float), np.asarray(log_b, float)
    ma, mb = a.mean(), b.mean()
    saa, sbb = np.var(a), np.var(b)
    sab = np.mean((a - ma) * (b - mb))
    beta_ab = sab / sbb
    delta = sab / saa  # slope of B on A
    tls = (saa - sbb + np.sqrt((saa - sbb) ** 2 + 4 * sab**2)) / (2 * sab)
    return {
        "OLS A on B": (ma - beta_ab * mb, beta_ab),
        "OLS B on A": (ma - mb / delta, 1 / delta),
        "TLS": (ma - tls * mb, tls),
        "ratio": (ma - mb, 1.0),
    }


def rolling_z(s, window: int) -> np.ndarray:
    """Rolling z-score with the mean and sd (ddof 1) of the last `window` values up to and
    including t; same numbers as pt.pairs.zscore(s, window), vectorised."""
    x = pd.Series(np.asarray(s, float))
    r = x.rolling(window)
    return ((x - r.mean()) / r.std(ddof=1)).to_numpy()


def _closed(z: np.ndarray) -> tuple[bool, bool]:
    """(opened, converged): |z| > 2 on some day, and z changed sign after the first such day."""
    hit = np.flatnonzero(np.abs(z) > 2)
    if hit.size == 0:
        return False, False
    return True, bool(np.any(np.sign(z[hit[0] :]) != np.sign(z[hit[0]])))


def first_trade(s: np.ndarray, z: np.ndarray) -> float:
    """Gain in log points (before costs) of the first trade: enter at the first close with |z| > 2
    betting the spread moves back (short it when z > 2, buy it when z < -2), exit at the first
    later close where z has changed sign, or at the last close. NaN if no entry. Entering and
    exiting at the signalling close is the same-day idealisation; pt/04 trades the next day."""
    hit = np.flatnonzero(np.abs(z) > 2)
    if hit.size == 0:
        return float("nan")
    i = hit[0]
    side = -np.sign(z[i])
    later = np.flatnonzero(np.sign(z[i + 1 :]) == -np.sign(z[i]))
    j = i + 1 + later[0] if later.size else len(s) - 1
    return float(side * (s[j] - s[i]))


def pair_spread_stats(form: pd.DataFrame, test: pd.DataFrame, a: str, b: str) -> dict:
    """Formation fit of a, b and how each hedge ratio and z-score choice held up in the test."""
    fa, fb = form[a].to_numpy(), form[b].to_numpy()
    ta, tb = test[a].to_numpy(), test[b].to_numpy()
    eg = engle_granger(fa, fb)
    row = {"a": a, "b": b, "pvalue": eg["pvalue"], "hl_form": eg["half_life"]}
    for name, (alpha, beta) in hedge_ratios(fa, fb).items():
        s_form, s_test = fa - alpha - beta * fb, ta - alpha - beta * tb
        mu, sd = s_form.mean(), s_form.std(ddof=1)
        row[f"{name}|beta"] = beta
        row[f"{name}|sd_ratio"] = float(np.sqrt(np.mean((s_test - mu) ** 2)) / sd)
        row[f"{name}|opened"], row[f"{name}|closed"] = _closed((s_test - mu) / sd)
    # z-score choices, all on the OLS A-on-B spread
    alpha, beta = eg["alpha"], eg["beta"]
    s_form, s_test = fa - alpha - beta * fb, ta - alpha - beta * tb
    mu, sd = s_form.mean(), s_form.std(ddof=1)
    zs = {"formation": (s_test - mu) / sd}
    both = np.concatenate([s_form, s_test])
    for w in Z_WINDOWS:
        zs[f"rolling {w}"] = rolling_z(both, w)[len(s_form) :]
    for name, z in zs.items():
        row[f"z {name}|beyond2"] = float(np.mean(np.abs(z) > 2))
        row[f"z {name}|opened"], row[f"z {name}|closed"] = _closed(z)
        row[f"z {name}|gain"] = first_trade(s_test, z)
    row["hl_test"] = half_life(s_test)
    return row


def spread_study(
    logp: pd.DataFrame, sectors: dict[str, str], years: range, form_years: int = 2
) -> pd.DataFrame:
    """One row per (test year, same-sector pair) of `pair_spread_stats`."""
    rows = []
    for year, form, test in windows(logp, years, form_years):
        for a, b in same_sector_pairs(list(form.columns), sectors):
            rows.append({"year": year} | pair_spread_stats(form, test, a, b))
    return pd.DataFrame(rows)


def summarise(study: pd.DataFrame, prefix_list: list[str], groups: dict[str, pd.Series]) -> dict:
    """For each prefix (a hedge ratio or z choice) and group mask: median sd ratio or share of
    days beyond 2, share opened, share closed once opened."""
    out = {}
    for g, mask in groups.items():
        d = study[mask]
        for p in prefix_list:
            opened = d[f"{p}|opened"].astype(bool)
            stats = {"closed once opened": float(d.loc[opened, f"{p}|closed"].mean())}
            stats["opened"] = float(opened.mean())
            if f"{p}|sd_ratio" in d:
                stats["sd ratio"] = float(d[f"{p}|sd_ratio"].median())
            if f"{p}|beyond2" in d:
                stats["days beyond 2"] = float(d[f"{p}|beyond2"].mean())
                g_ = d[f"{p}|gain"].dropna()
                stats["first-trade gain"] = float(g_.mean())
                stats["first trade won"] = float((g_ > 0).mean())
            out[(g, p)] = stats
    return out
