"""Figures for step 1 (GD vs GF, stability limit, edge of stability, minimum selection).

Reads results/*.csv written by analyze_gd.py, predict_shift.py and map1d.py.
Run: uv run python experiments/optim_samples/plot_gd.py -> results/fig_*.png
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps

from common import RESULTS as RES  # noqa: E402
CLASSES = [("gf_like", "GF-like (monotone, GF's final loss)"),
           ("catapult", "catapult (loss spike, then GF's loss)"),
           ("oscillating", "bounded oscillation"), ("reentry", "thrown back to origin"),
           ("other", "other final loss"), ("diverged", "diverges")]


def fig_gd_vs_gf(df):
    """Relative error of the first event vs eta (persistent events); per-run accuracy of the O(eta)
    prediction is in fig_shift_accuracy.png (analyze_shift.py)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    small = df[df.root_scale.isna() & (df.cls == "gf_like")]
    ax = axes[0]
    b4 = small[(small.tree == "binary") & (small.n == 4) & (small.seed <= 4)]
    for i, sig in enumerate(sorted(b4.sigma.unique())):
        g = b4[b4.sigma == sig].groupby("ratio")
        m, se = g.rel_err1.mean(), g.rel_err1.sem()
        col = ps.BLUES[i + 1]
        ax.errorbar(m.index, m.values, yerr=se.values, fmt="o", color=col, ms=3.5, lw=1,
                    label=f"σ = {sig:g}")
        pred = (b4[b4.sigma == sig].assign(p=lambda x: x.eta * x.pred_shift_over_eta1 / x.t1_gf)
                .groupby("ratio").p.mean())
        ax.plot(pred.index, pred.values, "-", color=col, lw=1)
    ax.set(xscale="log", yscale="log", xlabel="η / η_max", ylabel="(t₁^GD − t₁^GF) / t₁^GF",
           title="(a) binary N=4, first event (mean ± s.e., 5 seeds)")
    ax.legend(loc="upper left")
    ax = axes[1]
    for i, (tree, n) in enumerate([("binary", 8), ("mps", 8)]):
        g0 = small[(small.tree == tree) & (small.n == n)]
        ax.scatter(g0.ratio * (1.04 if i else 0.96), g0.rel_err1, s=12, color=ps.CAT[i + 1],
                   label=f"{tree} N={n} (σ=0.9), each seed")
        pred = g0.assign(p=lambda x: x.eta * x.pred_shift_over_eta1 / x.t1_gf)
        for _, gs in pred.groupby("seed"):
            gs = gs.sort_values("ratio")
            ax.plot(gs.ratio, gs.p, "-", color=ps.CAT[i + 1], lw=0.6, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set(xlabel="η / η_max", ylabel="relative error of t₁ (first event)",
           title="(b) N=8 trees: every seed (lines: prediction)")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(RES / "fig_gd_vs_gf.png")
    plt.close(fig)


def fig_stability(df):
    att = pd.read_csv(RES / "map1d_points.csv")
    trees = [("binary", 4, "binary N=4 (K=3), σ ≤ 0.1", 3), ("binary", 8, "binary N=8 (K=7), σ=0.9", 7),
             ("mps", 8, "MPS N=8 (K=8), σ=0.9", 8)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.1]))
    plain = df[df.root_scale.isna()]
    for j, (tree, n, title, k) in enumerate(trees):
        ax = axes[0, j]
        a = att[(att.K == k) & (att.ratio <= 3.0)]
        ax.plot(a.ratio, a.x, ",", color=ps.INK2, alpha=0.6, rasterized=True)
        ax.axvline(1.0, color=ps.CAT[7], lw=0.8)
        ax.set(ylim=(-0.3, 1.4), title=f"1D balanced map, K={k}", ylabel="attractor x = a / s^{1/K}")
        g = plain[(plain.tree == tree) & (plain.n == n)]
        if tree == "binary" and n == 4:
            g = g[g.sigma <= 0.1]
        frac = (g.groupby("ratio").cls.value_counts(normalize=True).unstack(fill_value=0))
        ax = axes[1, j]
        # stacked step plot on a linear ratio axis
        bottom = np.zeros(len(frac))
        for i, (cls, lab) in enumerate(CLASSES):
            v = frac[cls].values if cls in frac else np.zeros(len(frac))
            ax.fill_between(frac.index, bottom, bottom + v, step="mid", color=ps.CAT[i], lw=0,
                            label=lab)
            bottom += v
        ax.axvline(1.0, color=ps.CAT[7], lw=0.8)
        ax.set(xlim=(0, 3.05), ylim=(0, 1), xlabel="η / η_max", ylabel="fraction of runs",
               title=title)
        ax.grid(False)
    for j in range(3):
        ratios = sorted(df.ratio.unique())
        axes[1, j].plot(ratios, np.full(len(ratios), -0.03), "|", color=ps.INK2, ms=5,
                        clip_on=False)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RES / "fig_stability.png")
    plt.close(fig)


def fig_eos_example():
    tr = pd.read_csv(RES / "eos_example_traj.csv")
    pr = pd.read_csv(RES / "eos_example_probe.csv")
    runs = [("binary4_sig0.1_seed0", 0.9), ("binary4_sig0.1_seed0", 1.2),
            ("binary4_sig0.1_seed0", 2.0)]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 6.2), sharex="col")
    for j, (run, ratio) in enumerate(runs):
        t = tr[(tr.run == run) & (tr.ratio == ratio)]
        p = pr[(pr.run == run) & (pr.ratio == ratio)]
        ax = axes[0, j]
        ax.plot(t.t, t.loss, color=ps.INK2, lw=0.8)
        ax.set(yscale="log", title=f"binary N=4, σ=0.1, η = {ratio} η_max", ylabel="loss L")
        ax = axes[1, j]
        for r in range(3):
            ax.plot(t.t, t[f"a{r + 1}"], color=ps.CAT[r], lw=1, label=f"a_{r + 1} (s={3 - r})")
        ax.set(ylabel="overlap a_r", ylim=(-0.2, 1.6))
        if j == 0:
            ax.legend(loc="center right")
        ax = axes[2, j]
        ax.plot(p.t, p.stab, "o-", color=ps.CAT[0], ms=3, lw=1)
        ax.axhline(1.0, color=ps.CAT[7], lw=0.8)
        ax.set(ylabel="sharpness · η / 2", xlabel="t = η · steps", yscale="log")
    fig.tight_layout()
    fig.savefig(RES / "fig_eos_example.png")
    plt.close(fig)


def fig_min_selection(df):
    """Runs that converged to a global minimum (GF-like or catapult, final loss < 1e-6 L(0));
    binary N=4, sigma=0.1."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    ok = df[df.cls.isin(["gf_like", "catapult"]) & (df.loss_end < 1e-6 * df.loss0)
            & (df.tree == "binary") & (df.n == 4) & (df.sigma == 0.1)]
    series = [("balanced (fan-in) init", ok.root_scale.isna()), ("root core ×20 (scale 2)",
              ok.root_scale == 2.0), ("root ×40 (scale 4)", ok.root_scale == 4.0),
              ("root ×60 (scale 6)", ok.root_scale == 6.0)]
    ax = axes[0]
    for i, (lab, mask) in enumerate(series):
        sub = ok[mask].assign(pred=lambda x: np.minimum(x.sharp_gf_rel, 1 / x.ratio))
        g = sub.groupby("ratio")
        ax.errorbar(g.sharp_end_rel.mean().index, g.sharp_end_rel.mean(),
                    yerr=g.sharp_end_rel.sem(), fmt="o", ms=3.5, color=ps.CAT[i], lw=1, label=lab)
        ax.plot(g.pred.mean().index, g.pred.mean(), "-", color=ps.CAT[i], lw=0.9, alpha=0.8)
    r = np.linspace(0.38, 1.0, 100)
    ax.plot(r, 1 / r, ":", color=ps.MUTED, lw=0.9, label="2/η (stability edge)")
    ax.axhline(1, color=ps.INK2, lw=0.6)
    ax.set(xlabel="η / η_max", ylabel="final sharpness / λ_bal", ylim=(0.9, 2.8),
           title="(a) sharpness of the minimum GD selects")
    ax.legend(loc="upper right", fontsize=6.5)
    ax = axes[1]
    for i, (lab, mask) in enumerate(series):
        sub = ok[mask].assign(rel=lambda x: x.trq_end / x.trq_gf)
        g = sub.groupby("ratio").rel
        ax.errorbar(g.mean().index, g.mean(), yerr=g.sem(), fmt="o-", color=ps.CAT[i], ms=3.5,
                    lw=1, label=lab)
    ax.set(xlabel="η / η_max", ylabel="mean |tr Q_e|, GD / GF", yscale="log",
           title="(b) imbalance at the selected minimum")
    fig.tight_layout()
    fig.savefig(RES / "fig_min_selection.png")
    plt.close(fig)


def main():
    ps.apply()
    df = pd.read_csv(RES / "gd_sweep.csv")
    # step-1 sweep: plain model, chi = 3, steps relative to eta_max, init seeds 0-4
    df = df[~df.centred & (df.chi == 3) & (df.ratio_ref == "teacher") & (df.seed <= 4)]
    fig_gd_vs_gf(df)
    fig_stability(df)
    fig_eos_example()
    fig_min_selection(df)


if __name__ == "__main__":
    main()
