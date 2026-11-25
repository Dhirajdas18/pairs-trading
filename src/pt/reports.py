"""README findings, one function per topic: `pt report <topic>` writes charts to out/."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pt import plots


def prices() -> pd.DataFrame:
    """Adjusted closes of the universe, NIFTY and the ETF with known Yahoo errors repaired."""
    from pt.data import DATA_PATH, clean_prices

    return clean_prices(pd.read_parquet(DATA_PATH)["adj_close"])


def returns() -> pd.DataFrame:
    from pt.data import DATA_PATH, daily_returns

    return daily_returns(pd.read_parquet(DATA_PATH)["adj_close"])


def last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    return df.loc[df.index[-1] - pd.DateOffset(years=years) :]


def _synthetic_pairs(n: int = 750, seed: int = 7) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Two made-up price pairs: one moves together day to day but drifts apart, the other
    wobbles independently day to day but is tied to the same level."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0004, 0.012, n)
    a1 = np.cumsum(common + rng.normal(0.0006, 0.005, n))  # correlated, own drift
    b1 = np.cumsum(common + rng.normal(-0.0006, 0.005, n))
    b2 = np.cumsum(rng.normal(0.0004, 0.012, n))
    gap = np.zeros(n)
    for t in range(1, n):  # gap pulled back toward zero: half-life about 7 days
        gap[t] = 0.9 * gap[t - 1] + rng.normal(0, 0.03)
    a2 = b2 + gap
    return {"correlated": (a1, b1), "cointegrated": (a2, b2)}


def report_cointegration(out_dir: Path) -> dict:
    """Correlation vs cointegration: a made-up example of each, how the two measures line up on
    every same-sector F&O pair, and which one picks pairs whose gap closes the following year."""
    from scipy.stats import spearmanr

    from pt.coint import group_outcomes, rolling_study
    from pt.data import SECTORS, SYMBOLS
    from pt.pairs import engle_granger

    res = {}
    # 1. made-up pairs
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    titles = {
        "correlated": "Correlated, not cointegrated",
        "cointegrated": "Cointegrated, weakly correlated",
    }
    for ax, (key, (a, b)) in zip(axes, _synthetic_pairs().items(), strict=True):
        corr = np.corrcoef(np.diff(a), np.diff(b))[0, 1]
        eg = engle_granger(a, b)
        res[f"synthetic {key}"] = f"return corr {corr:.2f}, EG p {eg['pvalue']:.3f}"
        x = np.arange(len(a))
        ax.plot(x, 100 * np.exp(a), color=plots.SERIES[0], lw=1.6)
        ax.plot(x, 100 * np.exp(b), color=plots.SERIES[1], lw=1.6)
        plots.label_line_end(ax, x[-1], 100 * np.exp(a[-1]), "A", plots.SERIES[0])
        plots.label_line_end(ax, x[-1], 100 * np.exp(b[-1]), "B", plots.SERIES[1])
        ax.set_title(
            f"{titles[key]}\nreturn correlation {corr:.2f} · cointegration p {eg['pvalue']:.3f}",
            fontsize=10,
            fontweight="normal",
        )
        ax.set_xlabel("trading day")
    axes[0].set_ylabel("price (start = 100)")
    fig.suptitle(
        "Correlation is about daily moves; cointegration is about the gap", x=0.02, ha="left"
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "corr_vs_coint")

    # 2. every same-sector pair, rolling formation (2 calendar years) -> test (next year)
    study = rolling_study(np.log(prices()[SYMBOLS]), SECTORS, range(2010, 2027))
    last = study[study["year"] == study["year"].max()]
    rho = spearmanr(last["corr"], last["pvalue"]).statistic
    res["pairs tested"] = f"{len(study)} pair-years, {study['year'].min()}-{study['year'].max()}"
    res["latest formation"] = (
        f"{len(last)} pairs ({last['year'].iloc[0] - 2}-{last['year'].iloc[0] - 1}); "
        f"Spearman corr vs p {rho:.2f}"
    )
    res["share cointegrated at 5%"] = f"{(study['pvalue'] < 0.05).mean():.1%} (5% expected by luck)"

    y0, y1 = last["year"].iloc[0] - 2, last["year"].iloc[0] - 1
    fig, ax = plt.subplots(figsize=(8, 4.8))
    coint = last["pvalue"] < 0.05
    ax.scatter(
        last.loc[~coint, "corr"],
        last.loc[~coint, "pvalue"],
        s=8,
        color=plots.MUTED,
        alpha=0.5,
        label="not cointegrated (p ≥ 0.05)",
        linewidths=0,
    )
    ax.scatter(
        last.loc[coint, "corr"],
        last.loc[coint, "pvalue"],
        s=10,
        color=plots.SERIES[0],
        label="cointegrated (p < 0.05)",
        linewidths=0,
    )
    ax.axhline(0.05, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("correlation of daily returns")
    ax.set_ylabel("Engle-Granger p-value (log scale)")
    ax.set_title(
        f"{len(last)} same-sector F&O pairs, {y0}-{y1}: rank correlation between the two {rho:.2f}",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    plots.save(fig, out_dir, "corr_vs_pvalue")

    # 3. which filter picks gaps that close next year?
    g = group_outcomes(study)
    for name, row in g.iterrows():
        res[f"next year: {name}"] = (
            f"{int(row['pairs'])} pairs, opened 2 sd {row['opened 2 sd']:.0%}, "
            f"closed once opened {row['converged once opened']:.0%}, "
            f"ended >2 sd away {row['drifted']:.0%}, formation sd {row['formation sd']:.3f}"
        )
    buckets = pd.cut(study["pvalue"], [0, 0.01, 0.05, 0.2, 0.5, 1.0])
    opened = study[study["opened"]]
    by_p = opened.groupby(buckets[study["opened"]], observed=True)["converged"].mean()
    hl = study.groupby(buckets, observed=True)["half_life"].median()
    for k in by_p.index:
        res[f"p {k}"] = f"closed once opened {by_p[k]:.0%}, median half-life {hl[k]:.0f} days"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    names = list(g.index)[::-1]
    vals = g.loc[names, "converged once opened"] * 100
    colors = [plots.SERIES[0] if "cointegrated" in n or n == "both" else plots.MUTED for n in names]
    axes[0].barh(names, vals, color=colors, height=0.55)
    for i, v in enumerate(vals):
        axes[0].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=plots.TEXT)
    axes[0].set_xlim(0, 70)
    axes[0].set_xlabel("gaps that closed after opening to 2 sd, % (next year)")
    axes[0].set_title("Picked by", fontsize=10, fontweight="normal", loc="left")
    labels = ["< 0.01", "0.01-0.05", "0.05-0.2", "0.2-0.5", "> 0.5"]
    axes[1].bar(labels, by_p.to_numpy() * 100, color=plots.SERIES[0], width=0.6)
    for i, v in enumerate(by_p.to_numpy() * 100):
        axes[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color=plots.TEXT)
    axes[1].set_ylim(0, 70)
    axes[1].set_xlabel("formation Engle-Granger p-value")
    axes[1].set_ylabel("gaps that closed after opening, %")
    axes[1].set_title("By cointegration strength", fontsize=10, fontweight="normal", loc="left")
    fig.suptitle(
        f"Out of sample, {study['year'].min()}-{study['year'].max()}: cointegration picks gaps "
        "that close; correlation barely helps",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "coint_next_year")
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {"cointegration": report_cointegration}
