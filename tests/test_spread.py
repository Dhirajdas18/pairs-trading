import numpy as np
import pandas as pd

from pt.pairs import hedge_ratio, zscore
from pt.spread import first_trade, hedge_ratios, pair_spread_stats, rolling_z


def test_hedge_ratios_agree_on_a_perfect_line_and_ols_matches_pairs():
    rng = np.random.default_rng(0)
    b = np.cumsum(rng.normal(0, 0.01, 500))
    a = 0.3 + 1.5 * b
    h = hedge_ratios(a, b)
    for name in ("OLS A on B", "OLS B on A", "TLS"):
        assert np.allclose(h[name], (0.3, 1.5))
    assert h["ratio"][1] == 1.0
    noisy = a + rng.normal(0, 0.05, 500)
    assert np.allclose(hedge_ratios(noisy, b)["OLS A on B"], hedge_ratio(noisy, b))


def test_ols_directions_bracket_tls_on_noisy_data():
    rng = np.random.default_rng(1)
    b = np.cumsum(rng.normal(0, 0.01, 800))
    a = 1.2 * b + rng.normal(0, 0.05, 800)
    h = hedge_ratios(a, b)
    lo, hi = sorted([h["OLS A on B"][1], h["OLS B on A"][1]])
    assert lo < h["TLS"][1] < hi


def test_rolling_z_matches_the_loop_version():
    s = np.cumsum(np.random.default_rng(2).normal(size=300))
    assert np.allclose(rolling_z(s, 60), zscore(s, 60), equal_nan=True)


def test_test_window_does_not_change_formation_fit():
    rng = np.random.default_rng(3)
    b = np.cumsum(rng.normal(0, 0.01, 750))
    a = 0.5 + b + rng.normal(0, 0.01, 750)
    df = pd.DataFrame({"A": a, "B": b})
    r1 = pair_spread_stats(df.iloc[:500], df.iloc[500:], "A", "B")
    df.iloc[600:, 0] += 1.0
    r2 = pair_spread_stats(df.iloc[:500], df.iloc[500:], "A", "B")
    assert r1["pvalue"] == r2["pvalue"] and r1["OLS A on B|beta"] == r2["OLS A on B|beta"]
    assert r2["OLS A on B|sd_ratio"] > r1["OLS A on B|sd_ratio"]


def test_first_trade_shorts_a_high_spread_and_exits_on_the_sign_change():
    s = np.array([0.0, 0.1, 0.3, 0.2, -0.05, -0.1])
    z = np.array([0.0, 1.0, 2.5, 1.5, -0.2, -1.0])
    assert np.isclose(first_trade(s, z), 0.3 - (-0.05))
    assert np.isnan(first_trade(s, np.zeros(6)))
    # never closes: held to the last close
    assert np.isclose(first_trade(s, np.array([0, -2.5, -1, -1, -1, -1.0])), -0.1 - 0.1)
