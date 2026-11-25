import numpy as np

from pt.kalman import fit_ratio, kalman_hedge, rolling_hedge


def _drifting_pair(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    lb = np.cumsum(rng.normal(0, 0.01, n))
    beta = np.linspace(0.6, 1.4, n)  # the true hedge ratio drifts
    la = 0.2 + beta * lb + rng.normal(0, 0.005, n)
    return la, lb, beta


def test_kalman_tracks_a_drifting_hedge_ratio_better_than_a_fixed_fit():
    la, lb, beta = _drifting_pair()
    r = np.var(la - np.polyval(np.polyfit(lb, la, 1), lb))
    ratio = fit_ratio(la[:500], lb[:500], r, (0.0, 1.0))
    f = kalman_hedge(la, lb, ratio, r, (0.0, 1.0))
    fixed = np.polyfit(lb[:500], la[:500], 1)[0]
    err_k = np.abs(f["beta"][500:] - beta[500:]).mean()
    err_f = np.abs(fixed - beta[500:]).mean()
    assert err_k < err_f


def test_kalman_prediction_uses_only_the_past():
    la, lb, _ = _drifting_pair()
    f1 = kalman_hedge(la, lb, 1e-4, 1e-4, (0.0, 1.0))
    la2 = la.copy()
    la2[700:] += 1.0
    f2 = kalman_hedge(la2, lb, 1e-4, 1e-4, (0.0, 1.0))
    assert np.allclose(f1["beta"][:701], f2["beta"][:701])  # day 700's prediction predates it
    assert not np.allclose(f1["beta"][701:], f2["beta"][701:])


def test_rolling_hedge_excludes_today_and_matches_ols():
    la, lb, _ = _drifting_pair()
    b = rolling_hedge(la, lb, 120)
    assert np.isnan(b[119]) and np.isclose(b[120], np.polyfit(lb[:120], la[:120], 1)[0])
