import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from pt.data import SYMBOLS
from pt.site import DAYS, SCALE, build_html, encode, explorer_data


def _prices():
    idx = pd.bdate_range("2020-01-01", periods=1300)
    rng = np.random.default_rng(0)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (1300, len(SYMBOLS))), axis=0)),
                      index=idx, columns=SYMBOLS)
    px.iloc[:400, 0] = np.nan  # a recent listing
    return px


def test_encoding_round_trips_to_a_hundredth_of_a_percent():
    px = pd.Series([np.nan, np.nan, 101.3, 99.8, 102.45, 2500.0])
    enc = encode(px)
    assert enc[:2] == [None, None] and all(isinstance(v, int) for v in enc[2:])
    dec = np.cumsum(enc[2:]) / SCALE
    assert dec == pytest.approx(np.log(px[2:].to_numpy()), abs=0.5 / SCALE + 1e-12)


def test_explorer_data_is_compact_and_keeps_gaps():
    d = explorer_data(_prices())
    assert len(d["dates"]) == DAYS and d["default"] == ["HDFCBANK", "ICICIBANK"]
    s0 = sorted(SYMBOLS)[0]
    assert set(d["symbols"]) == set(SYMBOLS) and set(d["lots"]) == set(d["symbols"])
    first = SYMBOLS[0]
    assert d["px"][first][0] is None and isinstance(d["px"][first][-1], int)
    assert isinstance(d["px"][s0][-1], int)
    assert json.loads(json.dumps(d)) == d


def test_page_builds_without_placeholders():
    html = build_html(_prices())
    assert "function engleGranger" in html and 'id="pair-explorer"' in html
    assert "$" + "js" not in html and "$" + "data" not in html
    assert "survivorship" in html and "not investment advice" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_scripts_parse():
    html = build_html(_prices())
    for i, src in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S)):
        out = subprocess.run(["node", "--check", "-"], input=src, capture_output=True, text=True)
        assert out.returncode == 0, (i, out.stderr[:500])
