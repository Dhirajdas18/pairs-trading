import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pt import pairs

JS = Path(__file__).resolve().parents[1] / "src" / "pt" / "pairs.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _node(expr: str):
    # The script goes in on stdin: Linux caps a single command-line argument at 128 KB.
    script = (
        f"const m = require({json.dumps(str(JS))});"
        f"console.log(JSON.stringify({expr}, (k, v) => v === Infinity ? 'inf' : v));"
    )
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _pair(n=1250, beta=0.8, phi=0.97, seed=0):
    rng = np.random.default_rng(seed)
    b = 5 + np.cumsum(rng.normal(0, 0.012, n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0, 0.008)
    return 0.4 + beta * b + e, b


def _arr(x) -> str:
    return json.dumps([float(v) for v in x])


def test_browser_normal_cdf_matches_erfc():
    zs = np.r_[np.linspace(-9, 9, 721), [-0.49, -0.5, 0.5, 3.99, 4.0, -4.0]]
    js = _node(f"{_arr(zs)}.map(m.normCdf)")
    py = [0.5 * math.erfc(-z / math.sqrt(2)) for z in zs]
    assert js == pytest.approx(py, rel=1e-13, abs=1e-300)


def test_browser_mackinnon_p_matches_python():
    ts = np.linspace(-20, 3, 461)
    for n in (1, 2):
        js = _node(f"{_arr(ts)}.map((t) => m.mackinnonP(t, {n}))")
        assert js == pytest.approx([pairs.mackinnon_p(t, n) for t in ts], rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_browser_engle_granger_matches_python(seed):
    a, b = _pair(seed=seed)
    if seed == 2:
        a = 5 + np.cumsum(np.random.default_rng(9).normal(0, 0.01, len(b)))  # not cointegrated
    js = _node(f"m.engleGranger({_arr(a)}, {_arr(b)}, 1)")
    py = pairs.engle_granger(a, b, 1)
    for k in ("alpha", "beta", "tstat", "pvalue", "half_life"):
        v = py[k] if py[k] != math.inf else "inf"
        assert js[k] == (v if v == "inf" else pytest.approx(v, rel=1e-9)), k


def test_browser_adf_with_constant_and_more_lags_matches_python():
    a, _ = _pair(seed=3)
    for lags in (1, 3):
        js = _node(f"m.adf({_arr(a)}, {lags}, true, 1)")
        t, p = pairs.adf(a, lags, constant=True, n_vars=1)
        assert js["tstat"] == pytest.approx(t, rel=1e-9)
        assert js["pvalue"] == pytest.approx(p, rel=1e-9)


def test_browser_hedge_spread_zscore_half_life_match_python():
    a, b = _pair(seed=4)
    alpha, beta = pairs.hedge_ratio(a, b)
    js = _node(
        f"(() => {{ const a = {_arr(a)}, b = {_arr(b)}; const [al, be] = m.hedgeRatio(a, b);"
        "const s = m.spread(a, b, al, be);"
        "return {al, be, s, z: m.zscore(s), zr: m.zscore(s, 60), hl: m.halfLife(s)}; })()"
    )
    s = pairs.spread(a, b, alpha, beta)
    assert js["al"] == pytest.approx(alpha, rel=1e-9) and js["be"] == pytest.approx(beta, rel=1e-9)
    assert js["s"] == pytest.approx(s.tolist(), rel=1e-9, abs=1e-12)
    assert js["z"] == pytest.approx(pairs.zscore(s).tolist(), rel=1e-9, abs=1e-9)
    zr = pairs.zscore(s, 60)
    assert js["zr"][:59] == [None] * 59
    assert js["zr"][59:] == pytest.approx(zr[59:].tolist(), rel=1e-9, abs=1e-9)
    assert js["hl"] == pytest.approx(pairs.half_life(s), rel=1e-9)


@pytest.mark.parametrize("stop", [None, 3.0])
def test_browser_backtest_matches_python(stop):
    a, b = _pair(seed=5)
    alpha, beta = pairs.hedge_ratio(a, b)
    z = pairs.zscore(pairs.spread(a, b, alpha, beta), 60)
    zj = json.dumps([None if np.isnan(v) else float(v) for v in z])
    js = _node(
        f"m.backtest({_arr(a)}, {_arr(b)}, {beta!r}, {zj}, 1.5, 0.25, "
        f"{'null' if stop is None else stop}, 0.0007)"
    )
    py = pairs.backtest(a, b, beta, z, 1.5, 0.25, stop, 0.0007)
    assert js["trades"] == py["trades"] and py["trades"] > 3
    assert js["positions"] == py["positions"].tolist()
    assert js["returns"] == pytest.approx(py["returns"].tolist(), rel=1e-9, abs=1e-15)
    assert js["equity"] == pytest.approx(py["equity"].tolist(), rel=1e-9)


def test_browser_corr_matches_numpy():
    x, y = np.random.default_rng(6).normal(size=(2, 500))
    y = 0.6 * x + y
    js = _node(f"m.corr({_arr(x)}, {_arr(y)})")
    assert js == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-12)
