import numpy as np
import pandas as pd

from pt.coint import follow_pair, group_outcomes, rolling_study, same_sector_pairs


def _pair(n, rng, coint=True):
    b = np.cumsum(rng.normal(0, 0.01, n))
    if coint:
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.9 * e[t - 1] + rng.normal(0, 0.01)
        a = 0.5 + 1.2 * b + e
    else:
        a = np.cumsum(rng.normal(0, 0.01, n))
    return a, b


def test_same_sector_pairs_only_within_industry():
    sectors = {"A": "bank", "B": "bank", "C": "it", "D": "it", "E": "bank"}
    assert same_sector_pairs(list(sectors), sectors) == [
        ("A", "B"),
        ("A", "E"),
        ("B", "E"),
        ("C", "D"),
    ]


def test_cointegrated_pair_stays_close_random_walks_do_not_always():
    rng = np.random.default_rng(1)
    a, b = _pair(750, rng)
    df = pd.DataFrame({"A": a, "B": b})
    r = follow_pair(df.iloc[:500], df.iloc[500:], "A", "B")
    assert r["pvalue"] < 0.01 and abs(r["beta"] - 1.2) < 0.1
    assert abs(r["end_z"]) < 3 and r["test_p"] < 0.05


def test_formation_numbers_ignore_the_test_window():
    rng = np.random.default_rng(2)
    a, b = _pair(750, rng)
    df = pd.DataFrame({"A": a, "B": b})
    r1 = follow_pair(df.iloc[:500], df.iloc[500:], "A", "B")
    shocked = df.copy()
    shocked.iloc[500:, 0] += 5.0
    r2 = follow_pair(shocked.iloc[:500], shocked.iloc[500:], "A", "B")
    for k in ("corr", "beta", "tstat", "pvalue"):
        assert r1[k] == r2[k]
    assert r2["end_z"] > r1["end_z"]


def test_rolling_study_uses_calendar_windows_and_groups_pool():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-01", "2014-12-31")
    cols = {}
    for k in range(3):
        a, b = _pair(len(idx), rng, coint=k == 0)
        cols[f"A{k}"], cols[f"B{k}"] = a, b
    logp = pd.DataFrame(cols, index=idx)
    sectors = {c: f"s{c[1]}" for c in logp.columns}
    st = rolling_study(logp, sectors, range(2012, 2015))
    assert sorted(st["year"].unique()) == [2012, 2013, 2014]
    assert len(st) == 9  # three same-sector pairs a year
    g = group_outcomes(st)
    assert g.loc["all same-sector pairs", "pairs"] == 9
