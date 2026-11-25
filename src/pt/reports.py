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


def report_spread(out_dir: Path) -> dict:
    """Building the spread: one pair's spread and z-scores through a year it was not fitted on,
    which hedge ratio keeps gaps closing, which z-score stays honest, and whether half-life
    carries over."""
    from scipy.stats import spearmanr

    from pt.data import SECTORS, SYMBOLS
    from pt.pairs import engle_granger
    from pt.spread import Z_WINDOWS, rolling_z, spread_study, summarise

    res = {}
    logp = np.log(prices()[SYMBOLS])

    # 1. example: HDFC Bank on ICICI Bank, fitted on 2024-2025, followed through 2026
    a, b = "HDFCBANK", "ICICIBANK"
    pair = logp[[a, b]].loc["2024-01-01":].ffill().dropna()
    form, test = pair.loc[:"2025-12-31"], pair.loc["2026-01-01":]
    eg = engle_granger(form[a].to_numpy(), form[b].to_numpy())
    s = pair[a] - eg["alpha"] - eg["beta"] * pair[b]
    mu, sd = s.loc[form.index].mean(), s.loc[form.index].std(ddof=1)
    z60 = pd.Series(rolling_z(s.to_numpy(), 60), index=s.index)
    zf = (s - mu) / sd
    res["example"] = (
        f"{a} on {b}, fitted 2024-25: beta {eg['beta']:.2f}, p {eg['pvalue']:.3f}, half-life "
        f"{eg['half_life']:.0f} days; 2026 days beyond 2 formation sd "
        f"{(zf.loc[test.index].abs() > 2).mean():.0%}, beyond 2 on the 60-day z "
        f"{(z60.loc[test.index].abs() > 2).mean():.0%}"
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[1.2, 1])
    ax = axes[0]
    ax.axvspan(test.index[0], test.index[-1], color=plots.GRID, alpha=0.5, lw=0)
    ax.plot(s.index, 100 * (s - mu), color=plots.SERIES[0], lw=1.4)
    for k in (-2, 2):
        ax.axhline(100 * k * sd, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.text(
        test.index[5],
        100 * 2 * sd,
        " ±2 sd of 2024-25",
        va="bottom",
        fontsize=8,
        color=plots.TEXT_2,
    )
    ax.set_ylabel("spread, % from 2024-25 mean")
    ax.set_title(
        f"{a} against {b}: hedge ratio {eg['beta']:.2f} and band fitted on 2024-25 "
        f"(p {eg['pvalue']:.3f}); 2026 shaded",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    ax = axes[1]
    ax.axvspan(test.index[0], test.index[-1], color=plots.GRID, alpha=0.5, lw=0)
    ax.plot(zf.index, zf, color=plots.SERIES[1], lw=1.2)
    ax.plot(z60.index, z60, color=plots.SERIES[0], lw=1.2)
    plots.label_line_end(ax, zf.index[-1], zf.iloc[-1], "z from 2024-25", plots.SERIES[1])
    plots.label_line_end(ax, z60.index[-1], z60.iloc[-1], "60-day z", plots.SERIES[0])
    for k in (-2, 2):
        ax.axhline(k, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_ylabel("z-score")
    fig.tight_layout()
    plots.save(fig, out_dir, "spread_example")

    # 2-4. every same-sector pair, formation 2 years -> next year
    st = spread_study(logp, SECTORS, range(2010, 2027))
    coint, none = st["pvalue"] < 0.05, st["pvalue"] > 0.5
    groups = {"cointegrated": coint, "no leash": none, "all": st["pvalue"] >= 0}
    hedges = ["OLS A on B", "OLS B on A", "TLS", "ratio"]
    zs = ["z formation"] + [f"z rolling {w}" for w in Z_WINDOWS]
    summ = summarise(st, hedges + zs, groups)
    res["pair-years"] = f"{len(st)}, cointegrated (p < 0.05) {int(coint.sum())}"
    for h in hedges:
        c = summ[("cointegrated", h)]
        res[f"hedge {h}"] = (
            f"cointegrated: closed once opened {c['closed once opened']:.0%}, next-year spread "
            f"{c['sd ratio']:.2f}x formation sd; all pairs closed "
            f"{summ[('all', h)]['closed once opened']:.0%}"
        )
    for zname in zs:
        c, n = summ[("cointegrated", zname)], summ[("no leash", zname)]
        res[zname] = (
            f"days beyond 2: {c['days beyond 2']:.1%}; first trade cointegrated "
            f"{100 * c['first-trade gain']:+.2f}% (won {c['first trade won']:.0%}), no leash "
            f"{100 * n['first-trade gain']:+.2f}% (won {n['first trade won']:.0%})"
        )
    ok = np.isfinite(st["hl_form"]) & np.isfinite(st["hl_test"]) & coint
    rho = spearmanr(st.loc[ok, "hl_form"], st.loc[ok, "hl_test"]).statistic
    res["half-life, cointegrated"] = (
        f"formation median {st.loc[ok, 'hl_form'].median():.0f} days, next year "
        f"{st.loc[ok, 'hl_test'].median():.0f}; rank corr {rho:.2f}; no reversion next year "
        f"{(~np.isfinite(st.loc[coint, 'hl_test'])).mean():.0%}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = {
        "OLS A on B": "least squares A on B",
        "OLS B on A": "least squares B on A",
        "TLS": "total least squares",
        "ratio": "price ratio (β = 1)",
    }
    vals = [100 * summ[("cointegrated", h)]["closed once opened"] for h in hedges][::-1]
    names = [labels[h] for h in hedges][::-1]
    axes[0].barh(
        names,
        vals,
        color=[plots.MUTED if "ratio" in n else plots.SERIES[0] for n in names],
        height=0.55,
    )
    for i, v in enumerate(vals):
        axes[0].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=plots.TEXT)
    axes[0].set_xlim(0, 65)
    axes[0].set_xlabel("gaps that closed after opening to 2 sd, %")
    axes[0].set_title(
        "Hedge ratio (cointegrated pairs)", fontsize=10, fontweight="normal", loc="left"
    )
    xs = np.arange(len(zs))
    w = 0.38
    zl = ["2-year\nformation"] + [f"rolling\n{k} days" for k in Z_WINDOWS]
    cg = [100 * summ[("cointegrated", z)]["first-trade gain"] for z in zs]
    ng = [100 * summ[("no leash", z)]["first-trade gain"] for z in zs]
    axes[1].bar(xs - w / 2, cg, w, color=plots.SERIES[0], label="cointegrated (p < 0.05)")
    axes[1].bar(xs + w / 2, ng, w, color=plots.MUTED, label="no leash (p > 0.5)")
    axes[1].axhline(0, color=plots.TEXT_2, lw=0.8)
    axes[1].set_xticks(xs, zl)
    axes[1].set_ylabel("average first trade, % before costs")
    axes[1].set_title(
        "Z-score for entry at 2 and exit at 0", fontsize=10, fontweight="normal", loc="left"
    )
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Next year: the hedge ratio matters little; a rolling z-score turns losses into gains",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "spread_choices")

    fig, ax = plt.subplots(figsize=(7, 5))
    d = st[ok]
    ax.scatter(d["hl_form"], d["hl_test"], s=6, color=plots.SERIES[0], alpha=0.35, linewidths=0)
    ax.plot([1, 400], [1, 400], color=plots.TEXT_2, lw=0.8, ls="--")
    ax.text(250, 250, "same", fontsize=8, color=plots.TEXT_2, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 400)
    ax.set_ylim(1, 400)
    ax.set_xlabel("half-life in the 2 formation years, days")
    ax.set_ylabel("half-life next year, days")
    ax.set_title(
        f"{ok.sum():,} cointegrated pair-years: median {d['hl_form'].median():.0f} days fitted, "
        f"{d['hl_test'].median():.0f} next year; rank corr {rho:.2f}",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "half_life")
    return res


def report_search(out_dir: Path) -> dict:
    """Searching the whole F&O list: the luck rate of the cointegration test, multiple-testing
    corrections, ranking, industry and extra filters, all scored on the next year."""
    from pt.coint import windows
    from pt.data import SECTORS, SYMBOLS
    from pt.search import bh_reject, outcomes, placebo_pvalues, scan

    res = {}
    logp = np.log(prices()[SYMBOLS])
    sc = scan(logp, SECTORS, range(2010, 2027))
    same, cross = sc[sc["same_sector"]], sc[~sc["same_sector"]]
    placebo = np.concatenate(
        [
            placebo_pvalues(f, list(zip(d["a"], d["b"], strict=True)), seed=y)
            for y, f, _ in windows(logp, range(2010, 2027))
            for d in [same[same["year"] == y]]
        ]
    )
    luck = (placebo < 0.05).mean()
    res["pair-years"] = f"{len(sc)} ({len(same)} same industry), 2010-2026"
    res["share p < 0.05"] = (
        f"placebo {luck:.1%}, same industry {(same['pvalue'] < 0.05).mean():.1%}, "
        f"across industries {(cross['pvalue'] < 0.05).mean():.1%}"
    )
    res["luck share of passes"] = (
        f"same industry {luck / (same['pvalue'] < 0.05).mean():.0%}, across "
        f"{luck / (cross['pvalue'] < 0.05).mean():.0%}"
    )

    def bh(d):
        keep = pd.Series(False, index=d.index)
        for _, g in d.groupby("year"):
            keep[g.index] = bh_reject(g["pvalue"].to_numpy())
        return d[keep]

    def top(d, n):
        return d.sort_values("pvalue").groupby("year").head(n)

    m = same.groupby("year")["pvalue"].transform("size")
    picks = {
        "all same-industry pairs": same,
        "same industry, p < 0.05": same[same["pvalue"] < 0.05],
        "Benjamini-Hochberg 5%": bh(same),
        "Bonferroni 5%": same[same["pvalue"] < 0.05 / m],
        "20 lowest p, same industry": top(same, 20),
        "20 lowest p, any two stocks": top(sc, 20),
        "20 lowest p, across industries": top(cross, 20),
    }
    table = {k: outcomes(v) for k, v in picks.items()}
    for k, o in table.items():
        res[k] = (
            f"{o['per year']:.0f} a year, closed once opened {o['closed once opened']:.0%}, first "
            f"trade {100 * o['first trade (60-day z)']:+.2f}% (won {o['won']:.0%})"
        )
    bh_years = bh(same).groupby("year").size()
    res["BH survivors by year"] = ", ".join(f"{y}: {n}" for y, n in bh_years.items())
    t20 = top(same, 20).groupby("year")["gain60"].mean()
    res["20 lowest p, same industry, by year"] = (
        f"positive in {(t20 > 0).sum()} of {len(t20)}; worst {t20.idxmin()} {100 * t20.min():+.1f}%"
    )
    for n in (5, 10, 50, 100):
        o = outcomes(top(same, n))
        res[f"{n} lowest p, same industry"] = (
            f"closed {o['closed once opened']:.0%}, first trade "
            f"{100 * o['first trade (60-day z)']:+.2f}%"
        )
    c = same[same["pvalue"] < 0.05]
    for name, mask in {
        "hedge ratio 0.5-2": c["beta"].between(0.5, 2),
        "half-life 5-60 days": c["half_life"].between(5, 60),
        "return correlation > 0.5": c["corr"] > 0.5,
    }.items():
        a, b = outcomes(c[mask]), outcomes(c[~mask])
        res[f"p < 0.05 and {name}"] = (
            f"{a['pair-years']} in: {100 * a['first trade (60-day z)']:+.2f}%, "
            f"{b['pair-years']} out: {100 * b['first trade (60-day z)']:+.2f}%"
        )

    # chart 1: p-values with no link vs real pairs
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(0, 1, 21)
    hp, _ = np.histogram(placebo, bins)
    hs, _ = np.histogram(same["pvalue"], bins)
    x = bins[:-1] + 0.025
    ax.bar(
        x - 0.011,
        100 * hs / hs.sum(),
        0.02,
        color=plots.SERIES[0],
        label="real same-industry pairs",
    )
    ax.bar(
        x + 0.011,
        100 * hp / hp.sum(),
        0.02,
        color=plots.MUTED,
        label="same pairs with the link broken (luck only)",
    )
    ax.axhline(5, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_xlabel("Engle-Granger p-value")
    ax.set_ylabel("share of pair-years, %")
    ax.set_title(
        f"Below 0.05: {(same['pvalue'] < 0.05).mean():.1%} of real pairs, {luck:.1%} by luck alone "
        f"({len(same):,} pair-years)",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    plots.save(fig, out_dir, "search_luck")

    # chart 2: how pairs were picked -> next-year first trade and closing rate
    names = list(table)[::-1]
    gains = [100 * table[k]["first trade (60-day z)"] for k in names]
    closes = [100 * table[k]["closed once opened"] for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    col = [plots.SERIES[0] if k == "20 lowest p, same industry" else plots.MUTED for k in names]
    axes[0].barh(names, gains, color=col, height=0.55)
    axes[0].axvline(0, color=plots.TEXT_2, lw=0.8)
    for i, v in enumerate(gains):
        axes[0].text(
            v + (0.1 if v >= 0 else -0.1),
            i,
            f"{v:+.1f}%",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
            color=plots.TEXT,
        )
    axes[0].set_xlim(-7.5, 3.5)
    axes[0].set_xlabel("average first trade, % before costs")
    axes[1].barh(names, closes, color=col, height=0.55)
    for i, v in enumerate(closes):
        axes[1].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=plots.TEXT)
    axes[1].set_xlim(0, 75)
    axes[1].set_xlabel("gaps that closed after opening to 2 sd, %")
    fig.suptitle(
        "Next year: a short list of each industry's strongest pairs beat formal significance tests",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "search_picks")
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "cointegration": report_cointegration,
    "spread": report_spread,
    "search": report_search,
}
