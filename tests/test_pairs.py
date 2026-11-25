import math

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller, coint

from pt.data import DATA_PATH
from pt.pairs import (
    adf,
    backtest,
    engle_granger,
    half_life,
    hedge_ratio,
    mackinnon_p,
    positions,
    spread,
    zscore,
)


def _pair(n=1500, beta=1.3, phi=0.95, seed=0):
    """log B a random walk; log A = 0.2 + beta log B + AR(1) noise: cointegrated by construction."""
    rng = np.random.default_rng(seed)
    b = 4 + np.cumsum(rng.normal(0, 0.01, n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0, 0.01)
    return 0.2 + beta * b + e, b


def test_hedge_ratio_recovers_known_beta():
    a, b = _pair()
    alpha, beta = hedge_ratio(a, b)
    assert beta == pytest.approx(1.3, abs=0.05)
    assert np.allclose(spread(a, b, alpha, beta).mean(), 0, atol=1e-12)


def test_zscore_full_and_rolling_use_only_the_past():
    s = np.random.default_rng(1).normal(size=300)
    z = zscore(s)
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.std(ddof=1) == pytest.approx(1)
    zr = zscore(s, 60)
    assert np.isnan(zr[:59]).all()
    w = s[40:100]
    assert zr[99] == pytest.approx((s[99] - w.mean()) / w.std(ddof=1))
    s2 = s.copy()
    s2[150:] += 10  # the future changes, the past z-scores do not
    assert np.array_equal(zscore(s2, 60)[:150], zr[:150], equal_nan=True)


def test_half_life_of_a_known_ar1():
    rng = np.random.default_rng(2)
    phi, s = 0.9, np.zeros(20000)
    for t in range(1, len(s)):
        s[t] = phi * s[t - 1] + rng.normal()
    assert half_life(s) == pytest.approx(-math.log(2) / math.log(phi), rel=0.05)
    assert half_life(np.cumsum(np.abs(rng.normal(size=500)))) == math.inf  # trending away


def test_mackinnon_p_matches_statsmodels_table():
    from statsmodels.tsa.adfvalues import mackinnonp

    for n in (1, 2):
        for t in np.linspace(-20, 3, 231):
            assert mackinnon_p(t, n) == pytest.approx(mackinnonp(t, "c", N=n), abs=1e-12)


def test_adf_with_constant_matches_statsmodels():
    a, b = _pair(seed=3)
    for x in (a, b):
        t, p = adf(x, 1)
        ref = adfuller(x, maxlag=1, autolag=None, regression="c", result_object=False)
        assert t == pytest.approx(ref[0], abs=1e-9) and p == pytest.approx(ref[1], abs=1e-9)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_engle_granger_matches_statsmodels_coint_on_synthetic_pairs(seed):
    a, b = _pair(seed=seed)
    eg = engle_granger(a, b)
    t, p, _ = coint(a, b, trend="c", maxlag=1, autolag=None)
    assert eg["tstat"] == pytest.approx(t, abs=1e-6) and eg["pvalue"] == pytest.approx(p, abs=1e-6)
    assert eg["pvalue"] < 0.01 and eg["half_life"] == pytest.approx(13.5, rel=0.5)


def test_independent_random_walks_rarely_look_cointegrated():
    rng = np.random.default_rng(7)
    ps = []
    for _ in range(40):
        a, b = np.cumsum(rng.normal(size=(2, 750)), axis=1)
        eg = engle_granger(a, b, lags=1)
        t, p, _ = coint(a, b, trend="c", maxlag=1, autolag=None)
        assert eg["pvalue"] == pytest.approx(p, abs=1e-6)
        ps.append(eg["pvalue"])
    assert np.mean(np.array(ps) < 0.05) <= 0.15  # about 5% by chance


@pytest.mark.skipif(not DATA_PATH.exists(), reason="price data not downloaded")
def test_engle_granger_matches_statsmodels_on_real_prices():
    from pt.data import load

    px = np.log(load("adj_close", ["HDFCBANK", "ICICIBANK"]).iloc[-1250:])
    eg = engle_granger(px["HDFCBANK"], px["ICICIBANK"])
    t, p, _ = coint(px["HDFCBANK"], px["ICICIBANK"], trend="c", maxlag=1, autolag=None)
    assert eg["tstat"] == pytest.approx(t, abs=1e-6) and eg["pvalue"] == pytest.approx(p, abs=1e-6)


def test_positions_follow_the_rule():
    z = [0, 2.5, 1.0, 0.3, -2.1, -1.0, 2.2, np.nan, -3.0, 0.0]
    assert positions(z).tolist() == [0, -1, -1, 0, 1, 1, -1, 0, 1, 0]
    # stop at 3: out at |z| > 3 and not back in until |z| < exit
    z = [2.5, 3.5, 2.5, 1.0, 0.2, 2.4]
    assert positions(z, stop=3.0).tolist() == [-1, 0, 0, 0, 0, -1]


def test_backtest_has_no_look_ahead():
    a, b = _pair(n=400, seed=8)
    z = zscore(spread(a, b, *hedge_ratio(a, b)), 60)
    base = backtest(a, b, 1.3, z)
    a2 = a.copy()
    a2[201:] += 0.5  # a price change at t+1 leaves the position at t, and the return to t, alone
    z2 = z.copy()
    z2[201:] = zscore(spread(a2, b, *hedge_ratio(a, b)), 60)[201:]
    moved = backtest(a2, b, 1.3, z2)
    assert np.array_equal(moved["positions"][:201], base["positions"][:201])
    assert np.array_equal(moved["returns"][:201], base["returns"][:201])


def test_backtest_returns_and_costs_by_hand():
    a = np.log([100.0, 100, 102, 101, 101])
    b = np.log([50.0, 50, 50, 51, 51])
    z = [0.0, -2.5, -1.0, 0.0, 0.0]  # long the spread at close 1, flat at close 3
    beta, c = 2.0, 0.001
    bt = backtest(a, b, beta, z, cost_per_side=c)
    wa, wb = 1 / 3, -2 / 3
    assert bt["positions"].tolist() == [0, 1, 1, 0, 0]
    assert bt["trades"] == 1
    gross = abs(wa) + abs(wb)
    expected = [0, 0, wa * 0.02 - c * gross, wa * (101 / 102 - 1) + wb * 0.02, -c * gross]
    assert bt["returns"] == pytest.approx(expected, abs=1e-15)
    assert bt["equity"][-1] == pytest.approx(np.prod(1 + np.array(expected)))
