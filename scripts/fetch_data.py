"""Build data/prices.parquet from Yahoo Finance (no setup needed)."""

import sys
import time
from pathlib import Path

import pandas as pd

from pt.data import DATA_PATH, ETFS, INDEX, fetch_yfinance, flag_adjusted_drops, missing_symbols

MAX_DROPPED = 5  # a few F&O names may have no Yahoo ticker; more than that means a bad download


def download(attempts: int = 4) -> pd.DataFrame:
    """Download everything, then retry only the symbols that came back empty.

    Yahoo occasionally drops a ticker or errors, so each gets a few tries. Symbols still empty after
    that are dropped if there are at most MAX_DROPPED of them (never NIFTY or the ETF); otherwise
    exit rather than write a gappy table."""
    df, want = None, None
    for attempt in range(attempts):
        if attempt:
            time.sleep(5 * attempt)
        try:
            got = fetch_yfinance(want)
        except Exception as exc:  # network or Yahoo error: retry
            print(f"  download attempt {attempt + 1} failed: {exc}")
            continue
        if df is None:
            df = got
        else:
            found = [s for s in want if s not in missing_symbols(got, want)]
            got = got.loc[:, got.columns.get_level_values("symbol").isin(found)]
            df = pd.concat([df.drop(columns=got.columns, errors="ignore"), got], axis=1)
        if not (want := missing_symbols(df)):
            return df.sort_index(axis=1)
        print(f"  attempt {attempt + 1}: no prices for {want}")
    if df is None or len(want) > MAX_DROPPED or {INDEX, *ETFS} & set(want):
        sys.exit(f"download failed for {want}; not writing an incomplete table")
    print(f"  dropping {want}: no Yahoo prices after {attempts} attempts")
    return df.drop(columns=want, level="symbol", errors="ignore").sort_index(axis=1)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH
    df = download()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    span = f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
    n = df["adj_close"].shape[1]
    print(f"Yahoo Finance: {len(df)} rows, {n} series, {span} -> {out}")
    for sym, date in flag_adjusted_drops(df["adj_close"]):
        print(f"  check {sym} {date:%Y-%m-%d}: adjusted price fell >20% in a day")


if __name__ == "__main__":
    main()
