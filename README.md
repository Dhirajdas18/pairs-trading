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

## Features

- [x] Correlation against cointegration: Engle-Granger test matched to statsmodels; which one picks pairs whose gap closes the next year, out of sample 2010-2026; live pair explorer tested against Python through Node
- [ ] Spread, hedge ratio, z-score and half-life
- [ ] Finding pairs across the F&O list, and how many pass by luck
- [ ] A z-score rule out of sample after stock-futures costs and lot sizes
- [ ] When pairs break: stability checks and stop rules
- [ ] Kalman hedge ratio against a fixed one, out of sample
- [ ] A portfolio of pairs against NIFTY after costs

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

### 1. Correlation said almost nothing about cointegration; next year, gaps of cointegrated pairs closed 48% of the time once they opened, against 35% for the most correlated pairs and 30% for all pairs

<a href="docs/img/corr_vs_coint.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/corr_vs_coint.png"><img src="docs/img/corr_vs_coint.gif" alt="Two made-up price pairs: one correlated day to day but drifting apart, one weakly correlated but held to the same level" width="720"></picture></a>

Two different questions. **Correlation** of daily returns asks whether the two stocks move the same way
on the same day. **Cointegration** asks whether the gap between their (log) prices keeps returning to a
fixed level. A pair trader bets on the gap, so cointegration is the property that matters. The made-up
pairs above show the two can disagree completely: correlation 0.84 with a gap that wanders off for good
(Engle-Granger p 0.53), and correlation 0.38 with a gap pulled back every week (p < 0.001).

The Engle-Granger test fits log A = α + β · log B by least squares (β is the hedge ratio) and runs an
augmented Dickey-Fuller test on the residual, the spread, with MacKinnon's two-variable p-values. The
code matches `statsmodels.tsa.stattools.coint` to 1e-6.

<table><tr>
<td><a href="docs/img/corr_vs_pvalue.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/corr_vs_pvalue.png"><img src="docs/img/corr_vs_pvalue.gif" alt="Scatter of return correlation against Engle-Granger p-value for 2,378 same-sector pairs"></picture></a></td>
<td><a href="docs/img/coint_next_year.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/coint_next_year.png"><img src="docs/img/coint_next_year.gif" alt="Share of gaps that closed next year after opening to 2 standard deviations, by how pairs were picked and by formation p-value"></picture></a></td>
</tr></table>

Every pair of today's F&O stocks in the same NSE industry was measured on two calendar years
(formation) and then followed through the next year with the formation hedge ratio, mean and standard
deviation held fixed (test). Test years 2010 to 2026 (2026 to 25 November), 22,139 pair-years. A stock
enters a window only with prices on at least 98% of its days.

- **The two measures are unrelated (left):** across the 2,378 pairs formed on 2024-2025, the rank
  correlation between return correlation and the Engle-Granger p-value was 0.01. High correlation does
  not find cointegrated pairs.
- **Cointegration picks gaps that close (right):** once a test-year spread first moved 2 formation
  standard deviations from its formation mean, it crossed back to that mean before the year ended for
  30% of all pairs, 35% of the top 10% by correlation, 48% of pairs with p < 0.05 and 51% of pairs that
  were both. The rate falls steadily with the p-value: 51% for p < 0.01, 21% for p > 0.5. It held in
  each era: 42% vs 28% (2010-15), 59% vs 38% (2016-20), 45% vs 27% (2021-26).
- **But half still don't close:** cointegrated pairs had tighter formation spreads (median standard
  deviation 0.079 in log price against 0.125), so next year 95% of them moved 2 of those standard
  deviations (87% for all pairs) and 62% ended the year more than 2 away (56%). A spread measured on
  the years it was fitted to looks tighter than it will be.
- **More pass than luck gives, but not by much:** 10.7% of pair-years had p < 0.05, against 5% if no
  pair were cointegrated. Many of the passes are luck, which matters when picking pairs from thousands.

Survivorship: these are today's F&O stocks; pairs where one stock failed or left the list are missing,
which flatters every group.

Research only, not investment advice. Results computed on today's F&O list are survivorship-biased:
stocks that were dropped from the list or delisted are not in the data.
