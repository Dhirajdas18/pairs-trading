"""Interactive dashboard: a live pair explorer for NSE F&O stocks, as one static page."""

import json
import re
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from pt import plots

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 11 5 7 9 9 15 3%22 fill=%22none%22 stroke=%22%232a78d6%22"
    " stroke-width=%222%22/%3E"
    "%3Cpath d=%22M1 13 5 10 9 12 15 6%22 fill=%22none%22 stroke=%22%23eb6834%22"
    " stroke-width=%222%22/%3E"
    "%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "pairs.js").read_text(encoding="utf-8")
EXPLORER = (HERE / "explorer.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/pairs-trading"
DAYS = 1250  # five years of trading days embedded in the page
SCALE = 1e4  # log prices stored as integers in units of 0.0001 (0.01%)
DEFAULT = ("HDFCBANK", "ICICIBANK")


def encode(px: pd.Series) -> list[int | None]:
    """Log price x 1e4 rounded, as the first value then day-to-day differences (small numbers,
    so the page stays light). None before the stock's first price; gaps inside carry the last."""
    logs = np.round(np.log(px.to_numpy(float)) * SCALE)
    out, last = [], None
    for v in logs:
        if np.isnan(v):
            out.append(None)
            continue
        out.append(int(v) if last is None else int(v - last))
        last = v
    return out


def explorer_data(prices: pd.DataFrame) -> dict:
    from pt.data import LOTS, SECTORS, SYMBOLS

    recent = prices.iloc[-DAYS:]
    symbols = sorted(s for s in SYMBOLS if s in recent and recent[s].notna().any())
    a, b = DEFAULT if all(s in symbols for s in DEFAULT) else symbols[:2]
    return {
        "symbols": symbols,
        "sectors": {s: SECTORS[s] for s in symbols},
        "lots": {s: LOTS[s] for s in symbols},
        "dates": list(recent.index.strftime("%Y-%m-%d")),
        "last": f"{recent.index[-1]:%d %b %Y}",
        "scale": SCALE,
        "px": {s: encode(recent[s]) for s in symbols},
        "default": [a, b],
    }


def explorer_section(prices: pd.DataFrame) -> str:
    return EXPLORER.replace("$js", JS).replace(
        "$data", json.dumps(explorer_data(prices), separators=(",", ":"))
    )


SECTIONS = [
    (
        "Pair explorer",
        "Pick two of today’s NSE F&O stocks and a window. The page regresses the first stock’s "
        "log price on the second’s: the slope is the hedge ratio, and what is left over is the "
        "spread. The Engle-Granger test asks whether that spread keeps returning to its mean (a "
        "low p-value says it does more reliably than chance would allow); the half-life is how "
        "many trading days a gap takes to close by half. The backtest shorts the spread when its "
        "z-score rises above the entry level, buys it below minus the entry level and closes "
        "inside the exit band. The hedge ratio is fitted on the same window it trades, so the "
        "result is in-sample.",
        explorer_section,
    ),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(prices: pd.DataFrame) -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        body = make(prices)
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    d = plots.DARK
    return PAGE.substitute(
        icon=ICON,
        font=FONT,
        repo=REPO,
        updated=f"{prices.index[-1]:%d %b %Y}",
        surface=plots.SURFACE,
        text=plots.TEXT,
        text2=plots.TEXT_2,
        grid=plots.GRID,
        accent=plots.SERIES[0],
        accent2=plots.SERIES[1],
        muted=plots.MUTED,
        d_surface=d[plots.SURFACE],
        d_text=d[plots.TEXT],
        d_text2=d[plots.TEXT_2],
        d_grid=d[plots.GRID],
        d_accent=d[plots.SERIES[0]],
        d_accent2=d[plots.SERIES[1]],
        d_muted="#6b6a64",
        dark=json.dumps({k.lower(): v for k, v in d.items()}),
        dark_lines=json.dumps({k.lower(): v for k, v in plots.DARK_LINES.items()}),
        nav="\n".join(nav),
        sections="\n".join(parts),
    )
