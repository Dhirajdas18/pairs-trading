"""Correlation vs cointegration: which one picks pairs whose gap stays put the following year?

Every same-sector pair is measured on a formation window (return correlation, Engle-Granger test,
hedge ratio) and then followed through the next year with the formation hedge ratio held fixed.
Nothing from the test year feeds back into the formation numbers.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from pt.pairs import adf, engle_granger, spread


def same_sector_pairs(symbols: list[str], sectors: dict[str, str]) -> list[tuple[str, str]]:
    """Every unordered pair of `symbols` that share an industry, in symbol order."""
    pairs = []
    for a, b in combinations(sorted(symbols), 2):
        if sectors.get(a) is not None and sectors.get(a) == sectors.get(b):
            pairs.append((a, b))
    return pairs


def follow_pair(form: pd.DataFrame, test: pd.DataFrame, a: str, b: str) -> dict:
    """Formation statistics for log prices a, b and how their spread behaved in the test window.

    `form` and `test` hold log prices with no gaps in a or b. The test spread uses the formation
    intercept and hedge ratio, and is measured in formation standard deviations around the
    formation mean, so a pair "drifted" if it ends the test year more than 2 of them away.
    """
    eg = engle_granger(form[a].to_numpy(), form[b].to_numpy())
    corr = float(np.corrcoef(np.diff(form[a].to_numpy()), np.diff(form[b].to_numpy()))[0, 1])
    s_form = spread(form[a].to_numpy(), form[b].to_numpy(), eg["alpha"], eg["beta"])
    s_test = spread(test[a].to_numpy(), test[b].to_numpy(), eg["alpha"], eg["beta"])
    mu, sd = s_form.mean(), s_form.std(ddof=1)
    end_z = (s_test[-1] - mu) / sd
    max_z = np.abs((s_test - mu) / sd).max()
    test_p = adf(s_test, lags=1, constant=True, n_vars=1)[1]  # fixed beta: plain ADF table
    z = (s_test - mu) / sd
    hit = np.flatnonzero(np.abs(z) > 2)
    opened = hit.size > 0
    # a trade opened at the first 2-sd day "converged" if the spread later crossed its mean
    converged = bool(opened and np.any(np.sign(z[hit[0] :]) != np.sign(z[hit[0]])))
    return {
        "a": a,
        "b": b,
        "corr": corr,
        "beta": eg["beta"],
        "tstat": eg["tstat"],
        "pvalue": eg["pvalue"],
        "half_life": eg["half_life"],
        "end_z": float(end_z),
        "max_z": float(max_z),
        "test_p": float(test_p),
        "sd": float(sd),
        "end_gap": float(abs(s_test[-1] - mu)),
        "opened": bool(opened),
        "converged": converged,
    }


def windows(logp: pd.DataFrame, years: range, form_years: int = 2, min_cover: float = 0.98):
    """Yield (test year, formation log prices, test log prices) for each year: formation = the
    `form_years` calendar years before the test year, test = that calendar year. A stock enters a
    window only if it has a price on at least `min_cover` of both parts' days; its gaps are filled
    from the previous close."""
    for year in years:
        f0, f1 = pd.Timestamp(year - form_years, 1, 1), pd.Timestamp(year - 1, 12, 31)
        t0, t1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
        form, test = logp.loc[f0:f1], logp.loc[t0:t1]
        if len(test) < 120 or len(form) < 250 * form_years * 0.9:
            continue
        ok = [
            s
            for s in logp.columns
            if form[s].notna().mean() >= min_cover and test[s].notna().mean() >= min_cover
        ]
        yield year, form[ok].ffill().bfill(), test[ok].ffill().bfill()


def rolling_study(
    logp: pd.DataFrame,
    sectors: dict[str, str],
    years: range,
    form_years: int = 2,
    min_cover: float = 0.98,
) -> pd.DataFrame:
    """One row per (test year, same-sector pair) of `follow_pair` results over `windows`."""
    rows = []
    for year, form, test in windows(logp, years, form_years, min_cover):
        for a, b in same_sector_pairs(list(form.columns), sectors):
            rows.append({"year": year, "sector": sectors[a]} | follow_pair(form, test, a, b))
    return pd.DataFrame(rows)


def group_outcomes(study: pd.DataFrame, top_corr: float = 0.1, alpha: float = 0.05) -> pd.DataFrame:
    """Next-year behaviour of four groups chosen on formation data only: all pairs, the top
    `top_corr` share by return correlation (within each year), the Engle-Granger p < `alpha`
    pairs, and pairs that are both. Rates are pooled over the years in `study`."""
    rank = study.groupby("year")["corr"].rank(pct=True, ascending=False)
    high = rank <= top_corr
    coint = study["pvalue"] < alpha
    groups = {
        "all same-sector pairs": pd.Series(True, index=study.index),
        f"top {top_corr:.0%} by correlation": high,
        f"cointegrated (p < {alpha})": coint,
        "both": high & coint,
    }
    out = {}
    for name, mask in groups.items():
        g = study[mask]
        out[name] = {
            "pairs": len(g),
            "drifted": float((g["end_z"].abs() > 2).mean()),
            "broke 3 sd": float((g["max_z"] > 3).mean()),
            "still reverting": float((g["test_p"] < alpha).mean()),
            "formation sd": float(g["sd"].median()),
            "end gap": float(g["end_gap"].median()),
            "opened 2 sd": float(g["opened"].mean()),
            "converged once opened": float(g.loc[g["opened"], "converged"].mean()),
        }
    return pd.DataFrame(out).T
