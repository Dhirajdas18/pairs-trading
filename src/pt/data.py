"""Daily OHLC + adjusted closes from Yahoo Finance for today's NSE F&O stocks and the index.

The universe (docs/data/fno.csv) is a snapshot of NSE's F&O market-lot list joined with the NIFTY
500 for industries; refresh it with scripts/fetch_constituents.py. Only stocks with futures can be
shorted for more than a day in India, so every pair leg comes from this list. Using today's list
over past years is survivorship-biased.

Columns are a (field, symbol) MultiIndex. Fields: open, high, low, close, adj_close.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONSTITUENTS = ROOT / "docs" / "data" / "fno.csv"
_UNIVERSE = pd.read_csv(CONSTITUENTS)
SYMBOLS = _UNIVERSE["symbol"].tolist()
SECTORS = dict(_UNIVERSE.set_index("symbol")["industry"])
LOTS = {s: int(v) for s, v in _UNIVERSE.set_index("symbol")["lot"].items()}
INDEX = "NIFTY"
ETFS = ["NIFTYBEES"]  # NIFTY 50 with dividends
START = "2007-01-01"  # includes 2008
DATA_PATH = ROOT / "data" / "prices.parquet"
FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close"}


def tickers(symbols: list[str] | None = None) -> dict[str, str]:
    """Yahoo ticker -> our symbol, for `symbols` (default: every stock, NIFTY and the ETF)."""
    want = symbols if symbols is not None else SYMBOLS + [INDEX] + ETFS
    return {("^NSEI" if s == INDEX else f"{s}.NS"): s for s in want}


def fetch_yfinance(
    symbols: list[str] | None = None, start: str = START, end: str | None = None
) -> pd.DataFrame:
    """Download `symbols` (default: all). Yahoo's open/high/low/close are split-adjusted, not
    dividend-adjusted; returns use adj_close."""
    import yfinance as yf

    names = tickers(symbols)
    # threads=False: parallel downloads can lock yfinance's sqlite cache and drop a ticker
    raw = yf.download(
        list(names), start=start, end=end, auto_adjust=False, progress=False, threads=False
    )
    raw = raw[list(FIELDS)].rename(columns=FIELDS, level=0).rename(columns=names, level=1)
    raw.columns.names = ["field", "symbol"]
    raw.index = pd.DatetimeIndex(raw.index.tz_localize(None), name="date")
    return raw.sort_index(axis=1)


def missing_symbols(table: pd.DataFrame, expected: list[str] | None = None) -> list[str]:
    """Symbols with no adjusted close at all: an all-empty column or one Yahoo left out.

    Short history (a recent listing) is not missing. `expected` defaults to every symbol we
    download (the stocks, NIFTY and the ETF).
    """
    adj = table["adj_close"] if "adj_close" in table.columns.get_level_values(0) else pd.DataFrame()
    want = expected if expected is not None else SYMBOLS + [INDEX] + ETFS
    return [s for s in want if s not in adj.columns or adj[s].isna().all()]


def load_table(symbols: list[str] | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    """The whole (field, symbol) table, limited to `symbols` (default: the stocks + NIFTY)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first."
        )
    keep = symbols or [INDEX] + SYMBOLS
    table = pd.read_parquet(path)
    table = table.loc[:, table.columns.get_level_values("symbol").isin(keep)]
    return table.reindex(columns=keep, level="symbol").sort_index()


def load(field: str, symbols: list[str] | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    """One field as a wide date x symbol frame.

    Keeps only rows where every requested symbol (all, if None) has data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first."
        )
    table = pd.read_parquet(path)
    if field not in table.columns.get_level_values("field"):
        raise KeyError(f"field not in price table: {field}")
    df = table[field]
    if symbols is not None:
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            raise KeyError(f"symbols not in price table: {missing}")
        df = df[symbols]
    df.columns.name = None
    return df.sort_index().dropna(how="any")


def flag_adjusted_drops(
    adj: pd.DataFrame, threshold: float = 0.2
) -> list[tuple[str, pd.Timestamp]]:
    """(symbol, date) pairs where an adjusted price falls more than `threshold` in a day.

    Large caps rarely do that for real; each hit is worth checking for an unadjusted split or bonus.
    """
    moves = adj.pct_change(fill_method=None).stack()
    hits = moves[moves < -threshold]
    return [(sym, date) for date, sym in hits.index]


FIXES = ROOT / "docs" / "data" / "yahoo_fixes.csv"


def apply_fixes(prices: pd.DataFrame, fixes: Path = FIXES) -> pd.DataFrame:
    """Adjusted closes with `start` and `blank` rows of the fixes file applied, gaps inside a
    stock's own history carried forward (`zero` rows act on returns; see daily_returns)."""
    px = prices.copy()
    table = pd.read_csv(fixes, parse_dates=["date"])
    for row in table[table["action"] == "start"].itertuples():
        if row.symbol in px.columns:
            px.loc[px.index < row.date, row.symbol] = float("nan")
    for row in table[table["action"] == "blank"].itertuples():
        if row.symbol in px.columns and row.date in px.index:
            px.loc[row.date, row.symbol] = float("nan")
    started = px.notna().cummax()  # forward-fill only inside a stock's own history
    return px.ffill().where(started)


def daily_returns(prices: pd.DataFrame, fixes: Path = FIXES) -> pd.DataFrame:
    """Simple daily returns of adjusted closes, with known Yahoo errors repaired from
    docs/data/yahoo_fixes.csv (each with its source):
    - zero: a corporate action Yahoo did not adjust for; that day's return is set to 0.
    - blank: a bad print; the price is dropped and the previous price carried forward.
    - start: the series is not this company before the date; earlier prices are dropped.
    """
    r = apply_fixes(prices, fixes).pct_change(fill_method=None)
    table = pd.read_csv(fixes, parse_dates=["date"])
    for row in table[table["action"] == "zero"].itertuples():
        if row.symbol in r.columns and row.date in r.index:
            r.loc[row.date, row.symbol] = 0.0
    return r


def clean_prices(prices: pd.DataFrame, fixes: Path = FIXES) -> pd.DataFrame:
    """Adjusted closes rebuilt from the repaired returns, so `zero` fixes carry into price levels
    (pairs work on log prices). Each series starts at its first real close."""
    r = daily_returns(prices, fixes)
    px = apply_fixes(prices, fixes)
    first = px.apply(lambda s: s.first_valid_index())
    out = (1 + r.fillna(0)).cumprod().where(px.notna())
    for s, d in first.items():
        if d is not None:
            out[s] = out[s] / out.at[d, s] * px.at[d, s]
    return out
