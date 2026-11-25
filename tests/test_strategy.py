import numpy as np
import pandas as pd

from pt.strategy import _cross_zero, pair_returns, run, side_rates, summary


def test_side_rates_charge_stt_on_sales_and_stamp_on_buys():
    buy, sell = side_rates(0.0)
    assert np.isclose(sell - buy, 0.0005 - 0.00002)
    assert side_rates(0.001)[0] - buy == 0.001


def test_cross_zero_exits_when_z_changes_sign():
    z = np.array([np.nan, 0.5, 2.5, 1.0, 0.1, -0.2, -2.4, -1.0, 0.3])
    assert _cross_zero(z, 2.0).tolist() == [0, 0, -1, -1, -1, 0, 1, 1, 0]


def _setup(n=300, first=200):
    rng = np.random.default_rng(0)
    lb = np.cumsum(rng.normal(0, 0.01, n))
    la = lb + np.sin(np.arange(n) / 5) * 0.05
    ra, rb = np.r_[0.0, np.expm1(np.diff(la))], np.r_[0.0, np.expm1(np.diff(lb))]
    dt = np.full(n, 1 / 250)
    me = np.zeros(n, bool)
    return la, lb, ra, rb, dt, me, first


def test_next_day_execution_ignores_the_future_and_ends_flat():
    la, lb, ra, rb, dt, me, first = _setup()
    g1, n1, k = pair_returns(la - lb, ra, rb, 1.0, first, dt, me, window=20)
    la2 = la.copy()
    la2[260:] += 0.3  # change the future after day 259
    ra2 = np.r_[0.0, np.expm1(np.diff(la2))]
    g2, _, _ = pair_returns(la2 - lb, ra2, rb, 1.0, first, dt, me, window=20)
    assert np.allclose(g1[: 260 - first], g2[: 260 - first])
    assert k > 0 and (n1 <= g1 + 1e-15).all()


def test_costs_are_zero_without_slippage_and_fees_only_differ_by_rolls():
    la, lb, ra, rb, dt, me, first = _setup()
    _, n0, _ = pair_returns(la - lb, ra, rb, 1.0, first, dt, me, window=20)
    me2 = me.copy()
    me2[first:: 21] = True
    _, n1, _ = pair_returns(la - lb, ra, rb, 1.0, first, dt, me2, window=20)
    assert n1.sum() <= n0.sum()


def test_run_and_summary_shapes():
    idx = pd.bdate_range("2018-01-01", "2020-12-31")
    rng = np.random.default_rng(1)
    b = np.cumsum(rng.normal(0, 0.01, len(idx)))
    logp = pd.DataFrame({"A": b + rng.normal(0, 0.01, len(idx)), "B": b}, index=idx)
    picks = pd.DataFrame({"year": [2020], "a": ["A"], "b": ["B"], "beta": [1.0], "pvalue": [0.01]})
    r = run(logp, picks, n=20)
    assert r.index.min() >= pd.Timestamp("2020-01-01") and len(r) > 200
    s = summary(r["net"])
    assert set(s) == {"annual", "vol", "sharpe", "max drawdown"}


def test_lots_needed_finds_the_smallest_close_match():
    from pt.strategy import lots_needed

    r = lots_needed(1.0, 500_000, 500_000)
    assert (r["lots_a"], r["lots_b"], r["error"]) == (1, 1, 0.0)
    r = lots_needed(0.5, 400_000, 1_000_000)  # need ₹0.5 of B per ₹1 of A
    assert abs(r["error"]) <= 0.10 and r["lots_a"] >= 2


def test_stops_path_with_no_stop_triggering_matches_the_plain_rule():
    from pt.strategy import pair_returns

    la, lb, ra, rb, dt, me, first = _setup()
    g0, n0, k0 = pair_returns(la - lb, ra, rb, 1.0, first, dt, me, window=20)
    g1, n1, k1 = pair_returns(la - lb, ra, rb, 1.0, first, dt, me, window=20, stop_z=1e9)
    assert np.allclose(g0, g1) and k0 == k1


def test_retired_pair_stays_flat_and_time_stop_shortens_trades():
    from pt.strategy import pair_returns

    la, lb, ra, rb, dt, me, first = _setup()
    ret = np.zeros(len(la), bool)
    ret[first:] = True
    g, _, k = pair_returns(la - lb, ra, rb, 1.0, first, dt, me, window=20, retired=ret)
    assert k == 0 and np.all(g == 0)
    _, _, k_short = pair_returns(la - lb, ra, rb, 1.0, first, dt, me, window=20, max_days=2)
    assert k_short >= 1
