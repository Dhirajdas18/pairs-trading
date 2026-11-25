# Pairs trading

[![CI](https://github.com/Dhirajdas18/pairs-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhirajdas18/pairs-trading/actions/workflows/ci.yml)
[![Dashboard](https://github.com/Dhirajdas18/pairs-trading/actions/workflows/pages.yml/badge.svg)](https://dhirajdas18.github.io/pairs-trading/)

Pairs trading on NSE stocks: which pairs move together closely enough to trade one against the other,
how quickly their gaps close, and what is left after costs. Every leg is an F&O stock, because in India
a short position held overnight needs a stock future. The universe is today's NSE F&O list (stocks
in the NIFTY 500, about 217), with daily adjusted closes from Yahoo Finance since 2007.

**Live pair explorer: [dhirajdas18.github.io/pairs-trading](https://dhirajdas18.github.io/pairs-trading/)**.
Pick two stocks and a window: the page fits the hedge ratio in your browser and shows the Engle-Granger
cointegration test, the spread's half-life, the correlation of daily returns, the spread and its
z-score, and a z-score rule backtest before and after costs. Rebuilt weekly from fresh prices.

## Run it

```sh
uv sync
uv run python scripts/fetch_constituents.py   # refresh docs/data/fno.csv from NSE (optional)
uv run python scripts/fetch_data.py           # download prices to data/prices.parquet
uv run pt report <topic>                      # charts to out/ and the topic's numbers
uv run python scripts/build_site.py           # build site/index.html
uv run pytest -q
```

The pair maths lives in `src/pt/pairs.py` (numpy only) with a line-for-line browser copy in
`src/pt/pairs.js`; the tests check the Engle-Granger statistic and p-value against statsmodels and the
browser copy against Python through Node.

Known Yahoo errors (unadjusted corporate actions, bad prints) are repaired from
`docs/data/yahoo_fixes.csv`, each with its source.

## Findings

Findings are added as each study is completed.

Research only, not investment advice. Results computed on today's F&O list are survivorship-biased:
stocks that were dropped from the list or delisted are not in the data.
