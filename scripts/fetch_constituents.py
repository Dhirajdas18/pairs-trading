"""Refresh docs/data/fno.csv (symbol, industry, lot): NSE stocks with futures and options.

Joins NSE's F&O market-lot file with the NIFTY 500 list for each stock's industry. Index
contracts and any symbol outside the NIFTY 500 are left out. Pass local copies of the two files
as arguments to skip the download: fetch_constituents.py [fo_mktlots.csv ind_nifty500list.csv].
"""

import csv
import io
import sys
import urllib.request

from pt.data import CONSTITUENTS

LOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
N500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI"}
MIN_STOCKS = 150


def read(source: str) -> list[list[str]]:
    """CSV rows from a URL or a local path, every cell stripped of padding."""
    if source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "pairs-trading/0.1 (research)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8-sig")
    else:
        with open(source, encoding="utf-8-sig") as f:
            text = f.read()
    return [[c.strip() for c in row] for row in csv.reader(io.StringIO(text)) if row]


def universe(lots: list[list[str]], n500: list[list[str]]) -> list[tuple[str, str, int]]:
    """(symbol, industry, lot) for each F&O stock in the NIFTY 500; lot from the nearest month."""
    head = n500[0]
    industry = {r[head.index("Symbol")]: r[head.index("Industry")] for r in n500[1:]}
    out = {}
    for row in lots[1:]:
        sym = row[1]
        if sym in INDICES or sym not in industry:
            continue
        lot = next((c for c in row[2:] if c), None)
        if lot:
            out[sym] = (sym, industry[sym], int(lot))
    return sorted(out.values())


def main() -> None:
    lots_src, n500_src = sys.argv[1:3] if len(sys.argv) > 2 else (LOTS_URL, N500_URL)
    rows = universe(read(lots_src), read(n500_src))
    if len(rows) < MIN_STOCKS:
        raise SystemExit(f"expected at least {MIN_STOCKS} stocks, got {len(rows)}; not overwriting")
    with open(CONSTITUENTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "industry", "lot"])
        w.writerows(rows)
    print(f"{len(rows)} F&O stocks -> {CONSTITUENTS}")


if __name__ == "__main__":
    main()
