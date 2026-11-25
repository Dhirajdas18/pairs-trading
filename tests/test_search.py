import numpy as np
import pandas as pd

from pt.search import bh_reject, outcomes, placebo_pvalues, scan_window


def test_bh_matches_the_textbook_example():
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    # sorted 0.005, 0.01, 0.03, 0.04, 0.5 against 0.01, 0.02, 0.03, 0.04, 0.05: first four pass
    assert bh_reject(p).tolist() == [True, True, True, True, False]
    assert not bh_reject(np.array([0.2, 0.9])).any()


def test_placebo_breaks_a_real_link():
    rng = np.random.default_rng(0)
    n, k = 500, 40
    cols = {}
    for i in range(k):
        b = np.cumsum(rng.normal(0, 0.01, n))
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.8 * e[t - 1] + rng.normal(0, 0.005)
        cols[f"A{i}"], cols[f"B{i}"] = b + e, b
    form = pd.DataFrame(cols)
    pairs = [(f"A{i}", f"B{i}") for i in range(k)]
    p = placebo_pvalues(form, pairs)
    assert (p < 0.05).mean() < 0.25  # real pairs all pass; rotated ones mostly don't


def test_scan_window_marks_sectors_and_scores_next_year():
    rng = np.random.default_rng(1)
    idx = range(750)
    df = pd.DataFrame({s: np.cumsum(rng.normal(0, 0.01, 750)) for s in "ABC"}, index=idx)
    r = scan_window(df.iloc[:500], df.iloc[500:], {"A": "x", "B": "x", "C": "y"})
    assert len(r) == 3
    assert r.set_index(["a", "b"]).loc[("A", "B"), "same_sector"]
    assert not r.set_index(["a", "b"]).loc[("A", "C"), "same_sector"]
    o = outcomes(r.assign(year=2020))
    assert o["pair-years"] == 3
