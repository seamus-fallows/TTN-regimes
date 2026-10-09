"""Figures (c) t_1 vs sigma with the per-seed prediction, (d) gaps vs sigma, (f) Hessian at plateau
midpoints vs sigma.

    uv run python experiments/s2s_core/plot_scaling.py [--results results] [--out results]
Reads results/main_runs.csv, main_gap_fits.csv (analyze_main.py) and first_runs.csv (analyze_first.py,
optional).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttn import plotting as tp  # noqa: E402

HERE = Path(__file__).parent
TREES = ["binary4", "mps4", "binary8", "ht4", "mps8"]
KS = {"binary4": 3, "mps4": 4, "binary8": 7, "ht4": 7, "mps8": 8}


def fig_gaps(runs: pd.DataFrame, fits: pd.DataFrame, out: Path, chi: int, teacher: str):
    """(d): t_1, t_2 - t_1, t_3 - t_2 vs sigma, one thin line per seed (complete runs at the seed's
    modal order), runs that did not complete marked at the bottom."""
    trees = [t for t in TREES if t in set(runs.tree)]
    fig, axes = plt.subplots(1, len(trees), figsize=(tp.TEXT_WIDTH, 2.3), layout="constrained", squeeze=False)
    colors = tp.palette(3)
    names = [("t1", "$t_1$"), ("gap1", "$t_2-t_1$"), ("gap2", "$t_3-t_2$")]
    for ax, tree in zip(axes[0], trees):
        K = KS[tree]
        d = runs[(runs.tree == tree) & (runs.chi == chi) & (runs.teacher == teacher)].copy()
        d["gap1"], d["gap2"] = d.t2 - d.t1, d.t3 - d.t2
        comp = d[d["class"] == "complete"]
        for seed, g in comp.groupby("seed"):
            modal = g.order.mode().iloc[0]
            g = g[g.order == modal].sort_values("sigma")
            for (col, _), c in zip(names, colors):
                ax.plot(g.sigma, g[col], color=c, lw=0.7, alpha=0.7, marker="o", ms=1.8)
        f = fits[(fits.tree == tree) & (fits.chi == chi) & (fits.teacher == teacher)]
        lab = []
        for (col, nm), c in zip(names, colors):
            p = f[f"p_{col}"].dropna()
            if len(p):
                lab.append(f"{nm}: {p.mean():.2f}$\\pm${p.std(ddof=1) / np.sqrt(len(p)) if len(p) > 1 else 0:.2f} ({len(p)})")
        # Guide with slope -(K-2) through the median t_1 at the largest sigma.
        if len(comp):
            s0 = comp.sigma.max()
            y0 = comp[comp.sigma == s0].t1.median()
            xs = np.array([comp.sigma.min(), s0])
            ax.plot(xs, y0 * (xs / s0) ** (-(K - 2)), color="k", lw=0.8, ls="--")
        n_inc = (d["class"] != "complete").groupby(d.sigma).sum()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_title(f"{tree} (K={K})\n" + "\n".join(lab), fontsize=6.5)
        txt = " ".join(f"{int(v)}" for v in n_inc.values)
        ax.text(0.02, 0.02, f"incomplete/10: {txt}", transform=ax.transAxes, fontsize=5.5, color=tp.MUTED)
    axes[0][0].set_ylabel("time")
    handles = [plt.Line2D([], [], color=c, lw=1.2, label=nm) for (_, nm), c in zip(names, colors)]
    handles.append(plt.Line2D([], [], color="k", lw=0.8, ls="--", label=r"$\sigma^{-(K-2)}$"))
    fig.legend(handles=handles, loc="outside lower center", ncols=4, fontsize=7)
    tp.savefig(fig, out / f"fig_d_gaps_chi{chi}_{teacher}")


def fig_hessian(runs: pd.DataFrame, out: Path):
    """(f): lambda_min / lambda_max of the gauge-projected Hessian at the first and second plateau
    midpoints vs sigma (complete and incomplete runs)."""
    has = runs[np.isfinite(runs.plateau1_lam_min) & (runs.plateau1_lam_min < 0)]
    trees = [t for t in TREES if t in set(has.tree)]
    fig, axes = plt.subplots(1, len(trees), figsize=(tp.TEXT_WIDTH, 2.2), layout="constrained", squeeze=False,
                             sharey=True)
    colors = tp.palette(4)
    for ax, tree in zip(axes[0], trees):
        K = KS[tree]
        d = runs[runs.tree == tree]
        for j, (chi, teacher) in enumerate([(3, "separated"), (6, "separated"), (3, "close"), (6, "close")]):
            g = d[(d.chi == chi) & (d.teacher == teacher)]
            for col, mk in (("plateau1_lam_min", "o"), ("plateau2_lam_min", "s")):
                v = -g[col] / g.plateau1_lam_max
                ok = v > 0
                ax.scatter(g.sigma[ok], v[ok], s=5, color=colors[j], marker=mk, alpha=0.6,
                           label=f"$\\chi$={chi}, {teacher}" if col == "plateau1_lam_min" else None)
        xs = np.array([d.sigma.min(), d.sigma.max()])
        ref = (-(d.plateau1_lam_min) / d.plateau1_lam_max).clip(lower=1e-300).median()
        ax.plot(xs, ref * (xs / np.median(d.sigma)) ** (K - 2), color="k", lw=0.8, ls="--")
        ax.axhline(1e-3, color=tp.MUTED, lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_title(f"{tree} (K={K})", fontsize=8)
    axes[0][0].set_ylabel(r"$-\lambda_{\min}/\lambda_{\max}$ (gauge-projected)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="k", lw=0.8, ls="--"))
    labels.append(r"$\propto\sigma^{K-2}$")
    handles.append(plt.Line2D([], [], color=tp.MUTED, lw=0.7, ls=":"))
    labels.append("strict threshold $10^{-3}$")
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=6.5)
    tp.savefig(fig, out / "fig_f_hessian")


def fig_first(first: pd.DataFrame, out: Path, chi: int, init: str = "fanin"):
    """(c): t_1 vs sigma per tree, one marker colour per seed, with each seed's zero-parameter
    prediction sigma^-(K-2) T(theta_hat) + c_K (lines); bottom row: ratio t_1 / prediction."""
    trees = [t for t in TREES if t in set(first.tree)]
    fig, axes = plt.subplots(2, len(trees), figsize=(tp.TEXT_WIDTH, 3.6), layout="constrained", squeeze=False,
                             height_ratios=[2, 1])
    for j, tree in enumerate(trees):
        K = KS[tree]
        d = first[(first.tree == tree) & (first.chi == chi) & (first.init == init) & first.resolved]
        ax, axr = axes[0][j], axes[1][j]
        cols = tp.ordinal_colors(7)
        for i, (seed, g) in enumerate(d[d.teacher == "separated"].groupby("seed")):
            g = g.sort_values("sigma")
            c = cols[i % 7]
            ax.scatter(g.sigma, g.t1, s=6, color=c, zorder=3)
            xs = np.geomspace(g.sigma.min(), g.sigma.max(), 50)
            ax.plot(xs, xs ** (-(K - 2)) * g["T"].iloc[0], color=c, lw=0.6)
            axr.plot(g.sigma, g.ratio_with_offset, color=c, lw=0.6, marker="o", ms=1.8)
            axr.plot(g.sigma, g.ratio_leading, color=c, lw=0.6, ls=":", marker="x", ms=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        axr.set_xscale("log")
        axr.axhline(1, color=tp.MUTED, lw=0.7)
        ax.set_title(f"{tree} (K={K})", fontsize=8)
        axr.set_xlabel(r"$\sigma$")
    axes[0][0].set_ylabel("$t_1$")
    axes[1][0].set_ylabel("$t_1$ / prediction")
    handles = [plt.Line2D([], [], color="k", lw=0, marker="o", ms=3, label="GF $t_1$ (one colour per seed)"),
               plt.Line2D([], [], color="k", lw=0.6, label=r"$\sigma^{-(K-2)}T(\hat\theta)$ per seed"),
               plt.Line2D([], [], color="k", lw=0.6, marker="o", ms=2, label=r"ratio to $\sigma^{-(K-2)}T+c_K$"),
               plt.Line2D([], [], color="k", lw=0.6, ls=":", marker="x", ms=2, label=r"ratio to $\sigma^{-(K-2)}T$")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, fontsize=6.5)
    tp.savefig(fig, out / f"fig_c_first_escape_chi{chi}_{init}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    tp.set_style()
    runs = pd.read_csv(res / "main_runs.csv")
    fits = pd.read_csv(res / "main_gap_fits.csv")
    for chi in (3, 6):
        for teacher in ("separated", "close"):
            fig_gaps(runs, fits, out, chi, teacher)
    fig_hessian(runs, out)
    if (res / "first_runs.csv").exists():
        first = pd.read_csv(res / "first_runs.csv")
        for chi in (3, 6):
            for init in ("fanin", "balanced"):
                if len(first[(first.chi == chi) & (first.init == init)]):
                    fig_first(first, out, chi, init)


if __name__ == "__main__":
    main()
