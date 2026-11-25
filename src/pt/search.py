"""Searching the F&O list for pairs: how many pass by luck, whether multiple-testing corrections
or extra filters pick better pairs, and whether searching across industries helps.

Each candidate is fitted on the formation window (pt.coint.windows) and scored on the next year by
two outcomes: whether a gap that opened to 2 formation standard deviations closed (pt/01), and the
first trade on a rolling 60-day z-score before costs (pt/02).
"""

from itertools import combinations

import numpy as np
import pandas as pd

from pt.coint import follow_pair, windows
from pt.pairs import engle_granger
from pt.spread import first_trade, rolling_z


def scan_window(form: pd.DataFrame, test: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Every pair of the window's stocks: formation test results, whether they share an industry,
    and the next-year outcomes."""
    rows = []
    for a, b in combinations(sorted(form.columns), 2):
        r = follow_pair(form, test, a, b)
        s_form = form[a].to_numpy() - r["beta"] * form[b].to_numpy()
        s_test = test[a].to_numpy() - r["beta"] * test[b].to_numpy()
        z = rolling_z(np.concatenate([s_form, s_test]), 60)[len(s_form) :]
        r["gain60"] = first_trade(s_test, z)
        r["same_sector"] = sectors.get(a) is not None and sectors.get(a) == sectors.get(b)
        rows.append(r)
    return pd.DataFrame(rows)


def scan(logp: pd.DataFrame, sectors: dict[str, str], years: range) -> pd.DataFrame:
    """`scan_window` for every test year in `years`, with a year column."""
    out = [scan_window(f, t, sectors).assign(year=y) for y, f, t in windows(logp, years)]
    return pd.concat(out, ignore_index=True)


def placebo_pvalues(form: pd.DataFrame, pairs: list[tuple[str, str]], seed: int = 0) -> np.ndarray:
    """Engle-Granger p-values for `pairs` after breaking any real link: B's daily log returns are
    rotated by a random offset (at least 60 days) within the window and summed back into prices,
    so B keeps its own volatility, fat tails and calm/stormy spells but no longer lines up with A.
    With no link left, the share below 0.05 is the test's luck rate."""
    rng = np.random.default_rng(seed)
    out = []
    for a, b in pairs:
        rb = np.diff(form[b].to_numpy())
        k = int(rng.integers(60, len(rb) - 60))
        fake_b = np.concatenate([[0.0], np.cumsum(np.roll(rb, k))])
        out.append(engle_granger(form[a].to_numpy(), fake_b)["pvalue"])
    return np.array(out)


def bh_reject(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: True for p-values kept at false-discovery rate q."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    keep = np.zeros(m, bool)
    if passed.any():
        keep[order[: np.flatnonzero(passed).max() + 1]] = True
    return keep


def outcomes(d: pd.DataFrame, years: int | None = None) -> dict:
    """Next-year outcomes of a set of chosen pair-years; "per year" divides by `years` (default:
    the years that have any)."""
    opened = d["max_z"] > 2
    g = d["gain60"].dropna()
    return {
        "pair-years": len(d),
        "per year": len(d) / (years or max(d["year"].nunique(), 1)),
        "closed once opened": float(d.loc[opened, "converged"].mean()) if opened.any() else np.nan,
        "first trade (60-day z)": float(g.mean()) if len(g) else np.nan,
        "won": float((g > 0).mean()) if len(g) else np.nan,
    }
