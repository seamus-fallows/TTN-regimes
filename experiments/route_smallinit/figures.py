"""Draft figures from results/*.csv (one message per figure, stated in its title).

usage: uv run python experiments/route_smallinit/figures.py [--res DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttn.plotting import MUTED, palette, savefig, set_style, subplots  # noqa: E402

HERE = Path(__file__).resolve().parent
SMALL = -3.0
ORDER = ["D4", "DB4", "B4", "S5A", "S5B", "S7", "H4", "B8", "TIE", "PS"]


def big_fonts():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})


def fig_rules(ev, out):
    """Agreement of the three rules with the pattern actually learned, on transitions where the
    rules disagree, per target (sigma <= 1e-3)."""
    e = ev[(ev.variant == "dense") & (ev.event > 0) & (ev.log10_sigma <= SMALL + 1e-9) & ~ev.floor_flag]
    dis = e[(e.pred_greedy != e.pred_pairwise) | (e.pred_greedy != e.pred_weight)]
    names = [t for t in ORDER if t in set(dis.target)]
    fig, ax = subplots(1, 1, width=7.5, panel_aspect=0.45)
    cols = palette(3)
    w = 0.26
    for k, (rule, lab) in enumerate([("agree_greedy", "greedy rule (route depth, then weight)"),
                                     ("agree_pairwise", "pairwise rule (grow from one learned pattern)"),
                                     ("agree_weight", "largest weight next")]):
        vals = [dis[dis.target == t][rule].mean() for t in names]
        ax.bar(np.arange(len(names)) + (k - 1) * w, vals, w, color=cols[k], label=lab)
        for i, v in enumerate(vals):
            ax.text(i + (k - 1) * w, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(range(len(names)), [f"{t}\nn={int((dis.target == t).sum())}" for t in names])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("fraction of transitions where\nthe rule named the learned pattern")
    ax.set_xlabel("target (only transitions where the three rules disagree; sigma <= 1e-3)")
    ax.set_title("The greedy route rule names the next pattern far more often\nthan the pairwise rule or the weight order")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=9.5)
    savefig(fig, out / "fig_rules")
    plt.close(fig)


def fig_rules_sigma(ev, out):
    """Greedy agreement vs sigma, ties and non-ties separately (all targets pooled)."""
    e = ev[(ev.variant == "dense") & (ev.event > 0) & ~ev.floor_flag]
    fig, ax = subplots(1, 1, width=6.5, panel_aspect=0.55)
    cols = palette(3)
    for k, (mask, lab) in enumerate([(~e.tie, "no tie in route depth"),
                                     (e.tie, "tie in route depth (broken by weight in theory)")]):
        g = e[mask].groupby("sigma").agree_greedy.agg(["mean", "size"])
        ax.plot(g.index, g["mean"], "o-", color=cols[k], label=lab)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("initialization scale sigma")
    ax.set_ylabel("fraction of transitions\nmatching the greedy rule")
    ax.set_title("Away from ties the greedy rule is right at every sigma;\nties in route depth are decided by the init, not by weight")
    ax.legend(loc="lower left")
    savefig(fig, out / "fig_rules_vs_sigma")
    plt.close(fig)


def fig_durations(ev, ref, out):
    """Median plateau duration vs sigma for one representative cell per route depth, with the
    predicted power law sigma^-(c-2) (log law for c = 2)."""
    e = ev[(ev.variant == "dense") & (ev.event > 0) & ~ev.floor_flag & (ev.run_status == "done")]
    cells = e.groupby(["target", "transition", "depth"]).seed.nunique().reset_index()
    picks = []
    for c in (2, 3, 4, 5):
        x = cells[cells.depth == c].sort_values("seed", ascending=False)
        if len(x):
            picks.append(x.iloc[0])
    fig, axes = subplots(1, len(picks), width=11, panel_aspect=0.95)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, picks):
        x = e[(e.target == p.target) & (e.transition == p.transition)]
        g = x.groupby("sigma").duration.quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(g.index, g[0.25], g[0.75], color=palette(1)[0], alpha=0.25, lw=0)
        ax.plot(g.index, g[0.5], "o-", color=palette(1)[0], label="training, median and quartiles")
        r = ref[(ref.target == p.target) & (ref.transition == p.transition)].sort_values("sigma")
        if len(r):
            ax.plot(r.sigma, r.duration, "s", mfc="none", color=palette(2)[1], label="aligned-manifold reference")
        c = int(p.depth)
        s0 = g.index.min()
        y0 = g[0.5].loc[s0]
        sig = np.array(sorted(g.index))
        if c > 2:
            ax.plot(sig, y0 * (sig / s0) ** (-(c - 2)), "--", color=MUTED, label=f"prediction sigma^-{c - 2}")
        else:
            a = x.gamma.iloc[0] * x.pattern.map(lambda q: 1.0).iloc[0]
            ax.plot(sig, y0 + np.log(s0 / sig) / (a * _weight(x)), "--", color=MUTED,
                    label="prediction log(1/sigma)/(s A)")
        ax.set_xscale("log")
        ax.set_yscale("log" if c > 2 else "linear")
        ax.set_xlabel("initialization scale sigma")
        ax.set_title(f"{p.target} {p.transition}: depth c = {c}")
        if ax is axes[0]:
            ax.set_ylabel("plateau duration (gradient-flow time)")
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Plateau durations follow sigma^-(c-2), c = route depth (log law for c = 2)", fontsize=12.5)
    savefig(fig, out / "fig_durations")
    plt.close(fig)


def _weight(x):
    return float(x.pattern_weight.iloc[0]) if "pattern_weight" in x else 1.0


def fig_exponents(ex, out):
    """Per-seed exponents against the route-depth prediction c - 2 (c >= 3 cells)."""
    e = ex[(ex.variant == "dense") & (ex.depth >= 3)]
    fig, ax = subplots(1, 1, width=6.5, panel_aspect=0.7)
    rng = np.random.default_rng(0)
    cols = palette(4)
    for k, c in enumerate(sorted(e.depth.unique())):
        x = e[e.depth == c]
        ax.scatter(c - 2 + rng.uniform(-0.15, 0.15, len(x)), x.exponent, s=10, alpha=0.5, color=cols[k % 4],
                   label=f"route depth c = {c} ({len(x)} seed fits)")
    lo, hi = 0.5, e.depth.max() - 1.5 if len(e) else 3.5
    ax.plot([lo, hi], [lo, hi], "--", color=MUTED, label="prediction: exponent = c - 2")
    ax.set_xlabel("predicted exponent c - 2")
    ax.set_ylabel("fitted exponent per seed\n(duration ~ sigma^-exponent, sigma in [1e-5, 1e-3])")
    ax.set_title("Per-seed duration exponents sit on the route-depth prediction c - 2")
    ax.legend(fontsize=9)
    savefig(fig, out / "fig_exponents")
    plt.close(fig)


def fig_effdepth(ev, out):
    e = ev[(ev.variant == "dense") & (ev.event > 0) & (ev.log10_sigma <= SMALL + 1e-9) & ev.eff_depth.notna()]
    fig, ax = subplots(1, 1, width=6.0, panel_aspect=0.7)
    rng = np.random.default_rng(1)
    ax.scatter(e.depth + rng.uniform(-0.18, 0.18, len(e)), e.eff_depth.clip(upper=8), s=6, alpha=0.3,
               color=palette(1)[0], label="one learning event (sigma <= 1e-3)")
    for c, x in e.groupby("depth"):
        ax.plot([c - 0.25, c + 0.25], [x.eff_depth.median()] * 2, color="k", lw=2)
    ax.plot([1.5, 6.5], [1.5, 6.5], "--", color=MUTED, label="prediction: effective depth = c")
    ax.set_xlabel("route depth c of the learned pattern (number of dormant cores)")
    ax.set_ylabel("effective depth from the growth law")
    ax.set_title("Each learning event grows with the effective depth\nthat the route depth predicts")
    ax.legend(fontsize=9, loc="upper left")
    savefig(fig, out / "fig_effective_depth")
    plt.close(fig)


def fig_onestep(ev, out):
    e = ev[(ev.variant == "dense") & (ev.event > 0) & (ev.log10_sigma <= SMALL + 1e-9)]
    fig, ax = subplots(1, 1, width=7.0, panel_aspect=0.6)
    bins = np.logspace(-2, 6, 65)
    for k, (col, lab) in enumerate([("onestep_ratio", "from the amplitude P_r at plateau entry (brief)"),
                                    ("onestep_ratio_B", "from the size of the dormant route entries")]):
        x = e[col].replace([np.inf, -np.inf], np.nan).dropna()
        x = x[x > 0]
        ax.hist(x.clip(1e-2, 1e6), bins=bins, histtype="step", lw=2, color=palette(2)[k],
                label=f"{lab}: median {x.median():.2g}")
    ax.axvline(1, ls="--", color=MUTED, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_xlabel("observed / predicted plateau time (plateau entry to next event; last bin: >= 1e6 or never)")
    ax.set_ylabel("number of events")
    ax.set_title("The scalar route equation predicts the plateau length from the size\nof the dormant route entries, not from the amplitude P_r")
    ax.legend(fontsize=9)
    savefig(fig, out / "fig_onestep")
    plt.close(fig)


def fig_offtarget(ev, out):
    e = ev[(ev.variant == "dense") & (ev.event > 0)]
    fig, ax = subplots(1, 1, width=6.5, panel_aspect=0.6)
    g = e.groupby("sigma").peak_offtarget.quantile([0.5, 0.9, 1.0]).unstack()
    for k, (q, lab) in enumerate([(0.5, "median"), (0.9, "90th percentile"), (1.0, "maximum")]):
        ax.plot(g.index, g[q].clip(lower=1e-16), "o-", color=palette(3)[k], label=lab)
    ax.axhline(1e-2, ls="--", color=MUTED, label="1% of ||W||^2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("initialization scale sigma")
    ax.set_ylabel("peak off-target fraction of ||W||^2\nduring a transition")
    ax.set_title("During transitions training stays close to the aligned manifold\n(the off-target part of W stays tiny)")
    ax.legend(fontsize=9)
    savefig(fig, out / "fig_offtarget")
    plt.close(fig)


def fig_census(runs, out):
    r = runs[runs.variant == "dense"]
    cats = ["clean", "two patterns at once", "forgetting", "non-canonical plateau", "trapped", "other"]

    def cat(b):
        if b == "clean":
            return "clean"
        if "trapped" in b:
            return "trapped"
        if "unfinished" in b or "failed" in b or "error" in b:
            return "other"
        if "two_at_once" in b:
            return "two patterns at once"
        if "forgetting" in b:
            return "forgetting"
        if "noncanonical" in b:
            return "non-canonical plateau"
        return "other"

    r = r.assign(cat=r.behaviour.map(cat))
    names = [t for t in ORDER if t in set(r.target)]
    fig, ax = subplots(1, 1, width=7.5, panel_aspect=0.45)
    bottom = np.zeros(len(names))
    cols = palette(6)
    for k, c in enumerate(cats):
        vals = np.array([(r[r.target == t].cat == c).mean() for t in names])
        ax.bar(range(len(names)), vals, bottom=bottom, color=cols[k], label=c)
        bottom += vals
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("fraction of runs (all sigma, 40 seeds)")
    ax.set_title("Most runs follow a clean saddle-to-saddle path;\nthe deviations are specific to some targets")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9.5)
    savefig(fig, out / "fig_census")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(HERE / "results"))
    args = ap.parse_args()
    res = Path(args.res)
    set_style("sans")
    big_fonts()
    ev = pd.read_csv(res / "events.csv")
    runs = pd.read_csv(res / "runs.csv")
    try:
        ex = pd.read_csv(res / "exponents.csv")
    except pd.errors.EmptyDataError:
        ex = pd.DataFrame()
    targets = pd.read_csv(res / "targets.csv")
    w = {t: [float(v) for v in s.split()] for t, s in zip(targets.target, targets.weights)}
    ev["pattern_weight"] = [w[t][int(p)] for t, p in zip(ev.target, ev.pattern)]
    ref = ev[(ev.variant == "ref") & (ev.event > 0)]
    fig_rules(ev, res)
    fig_rules_sigma(ev, res)
    fig_durations(ev, ref, res)
    if len(ex):
        fig_exponents(ex, res)
    fig_effdepth(ev, res)
    fig_onestep(ev, res)
    fig_offtarget(ev, res)
    fig_census(runs, res)


if __name__ == "__main__":
    main()
