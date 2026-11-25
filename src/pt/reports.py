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


def report_strategy(out_dir: Path) -> dict:
    """The pairs strategy out of sample, 2010-2026, after stock-futures costs: the base rule chosen
    in the spread and search findings, what costs take, how settings change it, and the capital
    whole lots need today."""
    from pt.data import CONSTITUENTS, SECTORS, SYMBOLS
    from pt.search import scan
    from pt.strategy import lots_needed, run, summary

    res = {}
    px = prices()
    logp = np.log(px[SYMBOLS])
    sc = scan(logp, SECTORS, range(2010, 2027))
    picks = sc[sc["same_sector"]]
    base = run(logp, picks)
    g, n = summary(base["gross"]), summary(base["net"])
    trades = base.groupby("year")["trades"].first()
    yearly = base.groupby("year")["net"].apply(lambda d: (1 + d).prod() - 1)
    res["base"] = (
        "20 lowest p same-industry pairs a year, 60-day z, enter |z| > 2, exit at z = 0, next-day "
        "execution, 5 bp slippage a side, monthly rolls"
    )
    res["gross, above cash"] = f"{g['annual']:+.2%} a year"
    res["net, above cash"] = (
        f"{n['annual']:+.2%} a year, volatility {n['vol']:.1%}, Sharpe {n['sharpe']:.2f}, worst "
        f"drawdown {n['max drawdown']:.1%}, {trades.mean():.0f} trades a year"
    )
    res["net by year"] = ", ".join(f"{y} {100 * v:+.1f}%" for y, v in yearly.items())
    yrs = (base.index[-1] - base.index[0]).days / 365.25
    res["years positive"] = (
        f"{(yearly > 0).sum()} of {len(yearly)}; Sharpe t-stat about "
        f"{n['sharpe'] * np.sqrt(yrs):.1f} over {yrs:.1f} years"
    )

    def net(**kw):
        return summary(run(logp, picks, **kw)["net"])

    nothing = net(slippage=0.0, rolls=False)["annual"]
    no_slip = net(slippage=0.0)["annual"] - n["annual"]
    res["cost a year"] = (
        f"{g['annual'] - n['annual']:.2%}: without slippage +{no_slip:.2%}"
        f", without rolls +{net(rolls=False)['annual'] - n['annual']:.2%}, fees and STT on entries "
        f"and exits alone {g['annual'] - nothing:.2%}"
    )
    variants = {
        "base": {},
        "same-day execution": {"lag": 0},
        "slippage 2 bp": {"slippage": 0.0002},
        "slippage 10 bp": {"slippage": 0.001},
        "slippage 20 bp": {"slippage": 0.002},
        "10 pairs": {"n": 10},
        "50 pairs": {"n": 50},
        "enter at 1.5": {"entry": 1.5},
        "enter at 2.5": {"entry": 2.5},
        "20-day z": {"window": 20},
        "120-day z": {"window": 120},
        "exit at |z| < 0.5": {"exit": 0.5},
    }
    vals = {}
    for k, kw in variants.items():
        s_ = net(**kw) if kw else n
        vals[k] = s_
        res[f"variant: {k}"] = (
            f"net {s_['annual']:+.2%}, Sharpe {s_['sharpe']:.2f}, drawdown {s_['max drawdown']:.1%}"
        )

    lots = pd.read_csv(CONSTITUENTS).set_index("symbol")["lot"]
    last = px[SYMBOLS].ffill().iloc[-1]
    today = picks[picks["year"] == picks["year"].max()].nsmallest(20, "pvalue")
    need, one = [], []
    for r in today.itertuples():
        va, vb = lots[r.a] * last[r.a], lots[r.b] * last[r.b]
        need.append(lots_needed(r.beta, va, vb))
        one.append(abs(vb / (r.beta * va) - 1))
    notional = np.array([x["notional"] for x in need])
    res["whole lots today"] = (
        f"{len(need)} pairs picked for {today['year'].iloc[0]}, prices of {last.name:%d %b %Y}: "
        f"one lot each misses the hedge ratio by a median {np.median(one):.0%} "
        f"(up to {max(one):.0%}); "
        f"within 10% needs a median ₹{np.median(notional) / 1e5:.0f} lakh of futures per pair, "
        f"₹{notional.sum() / 1e7:.1f} crore for all"
    )

    fig, ax = plt.subplots(figsize=(10, 4.4))
    eq_g, eq_n = (1 + base["gross"]).cumprod(), (1 + base["net"]).cumprod()
    ax.plot(eq_g.index, 100 * eq_g, color=plots.MUTED, lw=1.4)
    ax.plot(eq_n.index, 100 * eq_n, color=plots.SERIES[0], lw=1.6)
    plots.label_line_end(
        ax, eq_g.index[-1], 100 * eq_g.iloc[-1], f"before costs {g['annual']:+.1%}/yr", plots.MUTED
    )
    plots.label_line_end(
        ax,
        eq_n.index[-1],
        100 * eq_n.iloc[-1],
        f"after costs {n['annual']:+.1%}/yr",
        plots.SERIES[0],
    )
    ax.axhline(100, color=plots.TEXT_2, lw=0.8)
    ax.set_ylabel("growth of 100 above cash")
    ax.set_title(
        f"20 pairs a year, traded the next day, 2010-2026: Sharpe {n['sharpe']:.2f} after costs, "
        f"worst drawdown {n['max drawdown']:.0%}",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    ax.margins(x=0.12)
    fig.tight_layout()
    plots.save(fig, out_dir, "strategy_equity")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    names = list(vals)[::-1]
    v = [100 * vals[k]["annual"] for k in names]
    ax.barh(
        names, v, color=[plots.SERIES[0] if k == "base" else plots.MUTED for k in names], height=0.6
    )
    ax.axvline(0, color=plots.TEXT_2, lw=0.8)
    for i, x in enumerate(v):
        ax.text(
            x + (0.08 if x >= 0 else -0.08),
            i,
            f"{x:+.1f}%",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=9,
            color=plots.TEXT,
        )
    ax.set_xlim(-2.6, 3.6)
    ax.set_xlabel("return above cash after costs, % a year")
    ax.set_title(
        "One setting changed at a time; picking the best one now would be hindsight",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "strategy_variants")
    return res


def report_breaks(out_dir: Path) -> dict:
    """When pairs break: a merger that forced a pair together and then removed it, stop rules and a
    monthly health check on the out-of-sample strategy, and whether formation data warns of
    breaks."""
    from pt.data import SECTORS, SYMBOLS, lake_closes
    from pt.pairs import hedge_ratio
    from pt.search import scan
    from pt.strategy import run, summary

    res = {}
    # 1. HDFC Ltd into HDFC Bank: 42 HDFC Bank shares for 25 HDFC (announced 4 Apr 2022)
    try:
        cl = lake_closes(["HDFC", "HDFCBANK"]).loc["2020-01-01":].dropna()
    except FileNotFoundError as e:
        res["merger"] = f"skipped: {e}"
    else:
        gap = 100 * (cl["HDFC"] / (42 / 25 * cl["HDFCBANK"]) - 1)
        before, after = gap.loc[:"2022-04-01"], gap.loc["2022-04-04":]
        res["merger"] = (
            f"HDFC price vs 42/25 HDFC Bank: before the announcement {before.min():+.1f}% to "
            f"{before.max():+.1f}% (sd {before.std():.1f}); after, {after.min():+.1f}% to "
            f"{after.max():+.1f}% (sd {after.std():.1f}); last day {gap.index[-1]:%d %b %Y}"
        )
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(gap.index, gap, color=plots.SERIES[0], lw=1.3)
        ax.axhline(0, color=plots.TEXT_2, lw=0.8)
        ax.axvline(pd.Timestamp("2022-04-04"), color=plots.TEXT_2, lw=0.8, ls="--")
        ax.text(
            pd.Timestamp("2022-04-11"),
            ax.get_ylim()[1] * 0.85,
            "merger announced\n4 Apr 2022",
            fontsize=8,
            color=plots.TEXT_2,
            va="top",
        )
        ax.set_ylabel("HDFC price vs 42/25 HDFC Bank, %")
        ax.set_title(
            "HDFC Ltd against HDFC Bank: a fixed swap ratio pinned the gap, then HDFC Ltd was "
            "delisted (absent from Yahoo)",
            fontsize=10,
            fontweight="normal",
            loc="left",
        )
        fig.tight_layout()
        plots.save(fig, out_dir, "breaks_merger")

    # 2. stops and a health check on the strategy
    logp = np.log(prices()[SYMBOLS])
    sc = scan(logp, SECTORS, range(2010, 2027))
    picks = sc[sc["same_sector"]]
    rules = {
        "no stop (finding 4)": {},
        "z-score stop at 3": {"stop_z": 3},
        "z-score stop at 4": {"stop_z": 4},
        "time stop 20 days": {"max_days": 20},
        "time stop 60 days": {"max_days": 60},
        "loss stop 5%": {"stop_loss": 0.05},
        "loss stop 10%": {"stop_loss": 0.10},
        "monthly check, drop if p > 0.5": {"health": 0.5},
        "monthly check, drop if p > 0.2": {"health": 0.2},
    }
    out = {}
    for k, kw in rules.items():
        r = run(logp, picks, **kw)
        s_ = summary(r["net"])
        y = r.groupby("year")["net"].apply(lambda d: (1 + d).prod() - 1)
        out[k] = s_ | {"worst year": float(y.min()), "years up": int((y > 0).sum())}
        res[k] = (
            f"net {s_['annual']:+.2%}, Sharpe {s_['sharpe']:.2f}, "
            f"drawdown {s_['max drawdown']:.1%}, "
            f"worst year {y.min():+.1%}, {int((y > 0).sum())} of {len(y)} years up"
        )

    names = list(rules)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, key, lab, lim in (
        (axes[0], "annual", "return above cash after costs, % a year", (-0.8, 2.2)),
        (axes[1], "max drawdown", "worst drawdown, %", (-22, 0)),
    ):
        v = [100 * out[k][key] for k in names]
        ax.barh(
            names,
            v,
            height=0.55,
            color=[plots.SERIES[0] if k.startswith("no stop") else plots.MUTED for k in names],
        )
        ax.axvline(0, color=plots.TEXT_2, lw=0.8)
        for i, x in enumerate(v):
            if key == "annual":
                ax.text(
                    x + (0.05 if x >= 0 else -0.05),
                    i,
                    f"{x:+.1f}%",
                    va="center",
                    ha="left" if x >= 0 else "right",
                    fontsize=9,
                    color=plots.TEXT,
                )
            else:  # bars run left from 0: label just past the bar's far end
                ax.text(
                    x - 0.3, i, f"{x:.0f}%", va="center", ha="right", fontsize=9, color=plots.TEXT
                )
        ax.set_xlim(*lim)
        ax.set_xlabel(lab)
    fig.suptitle(
        "Stops cut the worst falls and cost more return than they saved", x=0.02, ha="left"
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "breaks_stops")

    # 3. does formation data warn of a break? cointegrated same-industry pairs, by quartile
    c = picks[picks["pvalue"] < 0.05].copy()
    from pt.coint import windows

    w = {y: f for y, f, _ in windows(logp, range(2010, 2027))}
    db, drift = [], []
    for r in c.itertuples():
        f = w[r.year]
        y1, y2 = f.loc[: f"{r.year - 2}-12-31"], f.loc[f"{r.year - 1}-01-01" :]
        b1 = hedge_ratio(y1[r.a].to_numpy(), y1[r.b].to_numpy())[1]
        b2 = hedge_ratio(y2[r.a].to_numpy(), y2[r.b].to_numpy())[1]
        db.append(abs(b1 - b2) / max(abs(r.beta), 1e-6))
        s_ = f[r.a].to_numpy() - r.beta * f[r.b].to_numpy()
        h = len(s_) // 2
        drift.append(abs(s_[h:].mean() - s_[:h].mean()) / s_.std())
    c["hedge-ratio change between formation years"] = db
    c["spread drift within formation"] = drift
    for col in (
        "hedge-ratio change between formation years",
        "spread drift within formation",
        "half_life",
    ):
        q = pd.qcut(c[col], 4, labels=False)
        g = c.groupby(q)["gain60"].mean() * 100
        res[f"by quartile of {col}"] = "first trade " + ", ".join(f"{v:+.1f}%" for v in g)
    return res


def report_kalman(out_dir: Path) -> dict:
    """Time-varying hedge ratios out of sample: a Kalman filter and a rolling regression against
    the fixed formation ratio, as hedges and inside the strategy."""
    from pt.data import SECTORS, SYMBOLS
    from pt.kalman import fit_ratio, kalman_hedge, rolling_hedge
    from pt.search import scan
    from pt.strategy import run, summary

    res = {}
    logp = np.log(prices()[SYMBOLS])
    sc = scan(logp, SECTORS, range(2010, 2027))
    picks = sc[sc["same_sector"]]

    # 1. hedge quality on the traded pairs: daily moves the hedge leaves over, test year
    rows, ratios, example = [], [], None
    for y, g in picks.groupby("year"):
        px = logp.loc[f"{y - 2}-01-01" : f"{y}-12-31"].ffill()
        first = len(px.loc[: f"{y - 1}-12-31"])
        for r in g.nsmallest(20, "pvalue").itertuples():
            la, lb = px[r.a].to_numpy(), px[r.b].to_numpy()
            fa, fb = la[:first], lb[:first]
            a0 = fa.mean() - r.beta * fb.mean()
            rv = float(np.var(fa - a0 - r.beta * fb))
            q = fit_ratio(fa, fb, rv, (a0, r.beta))
            ratios.append(q)
            kb = kalman_hedge(la, lb, q, rv, (a0, r.beta))["beta"]
            rb = rolling_hedge(la, lb, 120)
            da, db = np.diff(la)[first - 1 :], np.diff(lb)[first - 1 :]
            rows.append(
                {
                    "fixed": np.std(da - r.beta * db),
                    "Kalman": np.std(da - kb[first:] * db),
                    "rolling": np.std(da - rb[first:] * db),
                    "Kalman beta sd": np.std(kb[first:]),
                    "rolling beta sd": np.std(rb[first:]),
                }
            )
            if y == 2026 and (r.a, r.b) == ("HDFCBANK", "ICICIBANK"):
                example = (px.index, r.beta, kb, rb, first)
    d = pd.DataFrame(rows)
    res["hedge left over, median daily sd"] = (
        f"fixed {100 * d['fixed'].median():.2f}%, Kalman {100 * d['Kalman'].median():.2f}%, "
        f"rolling 120-day {100 * d['rolling'].median():.2f}%; Kalman below fixed for "
        f"{(d['Kalman'] < d['fixed']).mean():.0%} of {len(d)} pair-years, rolling for "
        f"{(d['rolling'] < d['fixed']).mean():.0%}"
    )
    res["hedge ratio movement in the test year, median sd"] = (
        f"Kalman {d['Kalman beta sd'].median():.3f}, rolling {d['rolling beta sd'].median():.2f}"
    )
    res["Kalman q/r chosen on formation data"] = ", ".join(
        f"{k:g}: {v}" for k, v in pd.Series(ratios).value_counts().items()
    )

    # 2. inside the strategy
    variants = {
        "fixed (finding 4)": {"hedge": "fixed"},
        "rolling 120-day ratio": {"hedge": "rolling"},
        "Kalman ratio, 60-day z": {"hedge": "kalman"},
        "Kalman signal, enter at 2": {"hedge": "kalman signal"},
        "Kalman signal, enter at 1": {"hedge": "kalman signal", "entry": 1.0},
    }
    out = {}
    for k, kw in variants.items():
        r = run(logp, picks, **kw)
        s_ = summary(r["net"])
        y = r.groupby("year")["net"].apply(lambda x: (1 + x).prod() - 1)
        out[k] = s_
        res[k] = (
            f"net {s_['annual']:+.2%}, Sharpe {s_['sharpe']:.2f}, drawdown "
            f"{s_['max drawdown']:.1%}, {r.groupby('year')['trades'].first().mean():.0f} trades "
            f"a year, {(y > 0).sum()} of {len(y)} years up"
        )
    grid = (0.75, 1.0, 1.25, 1.5, 2.0)
    sharpe = {h: [] for h in ("fixed", "kalman signal")}
    for h in sharpe:
        for e in grid:
            r = run(logp, picks, hedge=h, entry=e)
            s_ = summary(r["net"])
            sharpe[h].append(s_["sharpe"])
            first_half = (1 + r["net"][r["year"] <= 2017]).prod() ** (1 / 8) - 1
            second_half = (1 + r["net"][r["year"] > 2017]).prod() ** (1 / 8.9) - 1
            res[f"{h}, enter at {e}"] = (
                f"net {s_['annual']:+.2%}, Sharpe {s_['sharpe']:.2f}; 2010-17 "
                f"{first_half:+.2%}, 2018-26 {second_half:+.2%} a year"
            )

    if example is not None:
        idx, b0, kb, rb, first = example
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axvspan(idx[first], idx[-1], color=plots.GRID, alpha=0.5, lw=0)
        ax.plot(idx, kb, color=plots.SERIES[0], lw=1.4)
        ax.plot(idx, rb, color=plots.SERIES[1], lw=1.2)
        ax.axhline(b0, color=plots.TEXT_2, lw=1, ls="--")
        plots.label_line_end(ax, idx[-1], kb[-1], "Kalman", plots.SERIES[0])
        plots.label_line_end(ax, idx[-1], rb[-1], "rolling 120-day", plots.SERIES[1])
        ax.text(idx[5], b0, f" fixed {b0:.2f}", va="bottom", fontsize=8, color=plots.TEXT_2)
        ax.set_ylabel("hedge ratio (log HDFC Bank on log ICICI Bank)")
        ax.set_title(
            "HDFC Bank against ICICI Bank: three hedge ratios, each known before the day "
            "(2026 shaded)",
            fontsize=10,
            fontweight="normal",
            loc="left",
        )
        ax.margins(x=0.1)
        fig.tight_layout()
        plots.save(fig, out_dir, "kalman_example")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    names = list(variants)[::-1]
    v = [100 * out[k]["annual"] for k in names]
    axes[0].barh(
        names,
        v,
        height=0.55,
        color=[plots.SERIES[0] if "fixed" in k else plots.MUTED for k in names],
    )
    axes[0].axvline(0, color=plots.TEXT_2, lw=0.8)
    for i, x in enumerate(v):
        axes[0].text(
            x + (0.06 if x >= 0 else -0.06),
            i,
            f"{x:+.1f}%",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=9,
            color=plots.TEXT,
        )
    axes[0].set_xlim(-2.2, 3)
    axes[0].set_xlabel("return above cash after costs, % a year")
    for h, c, lab in (
        ("fixed", plots.SERIES[0], "fixed ratio, 60-day z"),
        ("kalman signal", plots.SERIES[1], "Kalman signal"),
    ):
        axes[1].plot(grid, sharpe[h], "o-", color=c, lw=2, ms=6)
        plots.label_line_end(axes[1], grid[-1], sharpe[h][-1], lab, c)
    axes[1].axhline(0, color=plots.TEXT_2, lw=0.8)
    axes[1].set_xlabel("entry level, standard deviations")
    axes[1].set_ylabel("Sharpe after costs")
    axes[1].set_xlim(0.6, 2.9)
    fig.suptitle(
        "A moving hedge ratio alone hurt; the Kalman signal entered early did best",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "kalman_strategy")
    return res


def report_portfolio(out_dir: Path) -> dict:
    """The pairs book as a whole against NIFTY, 2010-2026: total return with cash, market exposure,
    crash months, sizing by risk, volatility targeting, and NIFTY plus a pairs sleeve."""
    from pt.data import SECTORS, SYMBOLS
    from pt.search import scan
    from pt.strategy import RF, run, summary, vol_target

    res = {}
    px = prices()
    logp = np.log(px[SYMBOLS])
    sc = scan(logp, SECTORS, range(2010, 2027))
    picks = sc[sc["same_sector"]]
    books = {
        "pairs, equal (finding 4)": run(logp, picks)["net"],
        "pairs, sized by risk": run(logp, picks, sizing="inverse vol")["net"],
        "Kalman signal at 1 (finding 6)": run(logp, picks, hedge="kalman signal", entry=1.0)["net"],
    }
    idx = books["pairs, equal (finding 4)"].index
    nifty = px["NIFTYBEES"].pct_change().reindex(idx).fillna(0.0)
    dt = np.r_[0.0, np.diff(idx.values).astype("timedelta64[D]").astype(float) / 365.0]
    cash = pd.Series(RF * dt, index=idx)
    ex = {"NIFTY ETF": nifty - cash} | books
    lev = {}
    for k in list(books):
        ex[k + ", 10% vol target"], lev[k] = vol_target(books[k])
    ex["70% NIFTY + 30% pairs"] = 0.7 * ex["NIFTY ETF"] + 0.3 * books["pairs, equal (finding 4)"]

    yrs = (idx[-1] - idx[0]).days / 365.25
    stats = {}
    for k, e in ex.items():
        tot = (1 + e + cash).cumprod()
        s_ = summary(e)
        stats[k] = s_ | {
            "total": float(tot.iloc[-1] ** (1 / yrs) - 1),
            "dd": float((tot / tot.cummax() - 1).min()),
            "corr": float(e.corr(ex["NIFTY ETF"])),
            "beta": float(np.cov(e, ex["NIFTY ETF"])[0, 1] / ex["NIFTY ETF"].var()),
        }
        res[k] = (
            f"total {stats[k]['total']:+.2%} a year (cash {RF:.1%}), "
            f"above cash {s_['annual']:+.2%}, "
            f"vol {s_['vol']:.1%}, Sharpe {s_['sharpe']:.2f}, worst drawdown {stats[k]['dd']:.1%}, "
            f"corr with NIFTY {stats[k]['corr']:+.2f}, beta {stats[k]['beta']:+.3f}"
        )
    for k, v in lev.items():
        res[f"{k}: leverage for 10% vol"] = (
            f"median {v.median():.1f}x, at the 3x cap {(v >= 3).mean():.0%}"
        )
    m = pd.DataFrame(
        {
            k: ex[k]
            for k in ("NIFTY ETF", "pairs, equal (finding 4)", "Kalman signal at 1 (finding 6)")
        }
    )
    monthly = (1 + m).resample("ME").prod() - 1
    worst = monthly.nsmallest(5, "NIFTY ETF")
    for d, row in worst.iterrows():
        res[f"NIFTY's bad month {d:%b %Y}"] = (
            f"NIFTY {100 * row.iloc[0]:+.1f}%, pairs {100 * row.iloc[1]:+.1f}%, Kalman signal "
            f"{100 * row.iloc[2]:+.1f}% (above cash)"
        )

    fig, ax = plt.subplots(figsize=(10, 4.6))
    lines = {
        "NIFTY ETF": plots.SERIES[1],
        "70% NIFTY + 30% pairs": plots.SERIES[2],
        "pairs, equal (finding 4)": plots.SERIES[0],
    }
    for k, c in lines.items():
        tot = 100 * (1 + ex[k] + cash).cumprod()
        ax.plot(tot.index, tot, color=c, lw=1.6)
        plots.label_line_end(ax, tot.index[-1], tot.iloc[-1], f"{k} {stats[k]['total']:+.1%}", c)
    c_ = 100 * (1 + cash).cumprod()
    ax.plot(c_.index, c_, color=plots.MUTED, lw=1.2, ls="--")
    plots.label_line_end(ax, c_.index[-1], c_.iloc[-1], f"cash {RF:.1%}", plots.MUTED)
    ax.set_yscale("log")
    ax.set_yticks([100, 150, 200, 300, 400, 600], ["100", "150", "200", "300", "400", "600"])
    ax.minorticks_off()
    ax.set_ylabel("growth of 100, log scale")
    ax.set_title(
        "Total return a year, 2010-2026, after costs (cash earns 6.5%)",
        fontsize=10,
        fontweight="normal",
        loc="left",
    )
    ax.margins(x=0.2)
    fig.tight_layout()
    plots.save(fig, out_dir, "portfolio_growth")

    names = [
        "NIFTY ETF",
        "pairs, equal (finding 4)",
        "pairs, sized by risk",
        "pairs, equal (finding 4), 10% vol target",
        "70% NIFTY + 30% pairs",
        "Kalman signal at 1 (finding 6)",
        "Kalman signal at 1 (finding 6), 10% vol target",
    ][::-1]
    short = {
        "pairs, equal (finding 4), 10% vol target": "pairs, equal, 10% vol target",
        "Kalman signal at 1 (finding 6), 10% vol target": "Kalman signal, 10% vol target",
    }
    labels = [short.get(k, k) for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    col = [
        plots.SERIES[1]
        if k == "NIFTY ETF"
        else plots.MUTED
        if k.startswith("Kalman")
        else plots.SERIES[0]
        for k in names
    ]
    for ax, key, lab in (
        (axes[0], "sharpe", "Sharpe (return above cash ÷ volatility)"),
        (axes[1], "dd", "worst drawdown of total return, %"),
    ):
        v = [stats[k][key] * (100 if key == "dd" else 1) for k in names]
        ax.barh(labels, v, color=col, height=0.55)
        ax.axvline(0, color=plots.TEXT_2, lw=0.8)
        for i, x in enumerate(v):
            if key == "dd":
                ax.text(
                    x - 0.5, i, f"{x:.0f}%", va="center", ha="right", fontsize=9, color=plots.TEXT
                )
            else:
                ax.text(x + 0.01, i, f"{x:.2f}", va="center", fontsize=9, color=plots.TEXT)
        ax.set_xlabel(lab)
    axes[0].set_xlim(0, 0.75)
    axes[1].set_xlim(-45, 0)
    fig.suptitle(
        "Pairs fell far less than NIFTY but earned less per unit of risk; grey: found "
        "with hindsight",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    plots.save(fig, out_dir, "portfolio_risk")
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "cointegration": report_cointegration,
    "spread": report_spread,
    "search": report_search,
    "strategy": report_strategy,
    "breaks": report_breaks,
    "kalman": report_kalman,
    "portfolio": report_portfolio,
}
