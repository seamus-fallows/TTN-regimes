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


def log_axes(ax):
    """Log-log axes with labelled major ticks only (minor labels collide on short ranges)."""
    from matplotlib.ticker import LogLocator, NullFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
TREES = ["binary4", "mps4", "binary8", "ht4", "mps8"]
KS = {"binary4": 3, "mps4": 4, "binary8": 7, "ht4": 7, "mps8": 8}


def pick_sweep(runs: pd.DataFrame, tree: str, chi: int, teacher: str, init: str = "fanin") -> str | None:
    """The sweep that carries a cell: K = 7, 8 cells are taken from the dedicated sweeps (k78 at
    chi = 3, large at chi = 6; no amplitude-based roundoff stop, longer budgets) when they have it."""
    prefs = (["k78"] if chi == 3 else ["large"]) + ["main"] if KS[tree] >= 7 else ["main"]
    for sw in prefs:
        if len(runs[(runs.sweep == sw) & (runs.tree == tree) & (runs.chi == chi) & (runs.teacher == teacher)
                    & (runs.init == init)]):
            return sw
    return None


def fig_gaps(runs: pd.DataFrame, fits: pd.DataFrame, out: Path, chi: int, teacher: str):
    """(d): t_1, t_2 - t_1, t_3 - t_2 vs sigma, one thin line per seed (complete runs at the seed's
    modal order), runs that did not complete marked at the bottom."""
    trees = [t for t in TREES if t in set(runs.tree)]
    fig, axes = plt.subplots(1, len(trees), figsize=(tp.TEXT_WIDTH, 2.3), layout="constrained", squeeze=False)
    colors = tp.palette(3)
    names = [("t1", "$t_1$"), ("gap1", "$t_2-t_1$"), ("gap2", "$t_3-t_2$")]
    for ax, tree in zip(axes[0], trees):
        K = KS[tree]
        sw = pick_sweep(runs, tree, chi, teacher)
        d = runs[(runs.sweep == sw) & (runs.tree == tree) & (runs.chi == chi) & (runs.teacher == teacher)
                 & (runs.init == "fanin")].copy()
        d["gap1"], d["gap2"] = d.t2 - d.t1, d.t3 - d.t2
        comp = d[d["class"] == "complete"]
        for seed, g in comp.groupby("seed"):
            modal = g.order.mode().iloc[0]
            g = g[g.order == modal].sort_values("sigma")
            for (col, _), c in zip(names, colors):
                ax.plot(g.sigma, g[col], color=c, lw=0.7, alpha=0.7, marker="o", ms=1.8)
        f = fits[(fits.sweep == sw) & (fits.tree == tree) & (fits.chi == chi) & (fits.teacher == teacher)
                 & (fits.init == "fanin")]
        lab = [f"[{sw}]"]
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
        log_axes(ax)
        ax.set_xlabel(r"$\sigma$")
        ax.set_title(f"{tree} (K={K}) " + "\n".join(lab), fontsize=6.5)
        n_tot = d.groupby("sigma").size()
        ax.set_xlabel(r"$\sigma$" + f"  (complete {int(n_tot.sum() - n_inc.sum())}/{int(n_tot.sum())})")
    axes[0][0].set_ylabel("time")
    handles = [plt.Line2D([], [], color=c, lw=1.2, label=nm) for (_, nm), c in zip(names, colors)]
    handles.append(plt.Line2D([], [], color="k", lw=0.8, ls="--", label=r"$\sigma^{-(K-2)}$"))
    fig.legend(handles=handles, loc="outside lower center", ncols=4, fontsize=7)
    tp.savefig(fig, out / f"fig_d_gaps_chi{chi}_{teacher}")


def fig_hessian(runs: pd.DataFrame, fits: pd.DataFrame, out: Path):
    """(f): most negative gauge-projected Hessian eigenvalue at the first and second plateau midpoints
    vs sigma, one line per seed (complete runs at the seed's modal order), against the route
    prediction -(K-2) slope; the per-seed slope (mean +- SE) is the saddle-type rule of
    notes/diagnostics.md (0: strict, k-2: higher order)."""
    comp = runs[(runs["class"] == "complete") & (runs.init == "fanin")]
    comp = comp[[pick_sweep(runs, r.tree, r.chi, r.teacher) == r.sweep for r in comp.itertuples()]]
    trees = [t for t in TREES if t in set(comp.tree) and comp[comp.tree == t].sigma.nunique() >= 4]
    fig, axes = plt.subplots(1, len(trees), figsize=(tp.TEXT_WIDTH, 2.3), layout="constrained", squeeze=False)
    colors = tp.palette(2)
    for ax, tree in zip(axes[0], trees):
        K = KS[tree]
        d = comp[(comp.tree == tree) & (comp.teacher == "separated")]
        for seed, g in d.groupby(["chi", "seed"]):
            modal = g.order.mode().iloc[0]
            g = g[g.order == modal].sort_values("sigma")
            for i, c in ((1, colors[0]), (2, colors[1])):
                v = -g[f"plateau{i}_lam_min"]
                ok = v > 0
                ax.plot(g.sigma[ok], v[ok], color=c, lw=0.6, alpha=0.7, marker="o", ms=1.6)
        f = fits[(fits.tree == tree) & (fits.teacher == "separated") & (fits.init == "fanin")
                 & fits.sweep.isin(set(d.sweep))]
        lab = []
        for i in (1, 2):
            sl = f[f"lam{i}_slope"].dropna()
            if len(sl) > 1:
                lab.append(f"plateau {i}: slope {sl.mean():.2f}$\\pm${sl.std(ddof=1) / np.sqrt(len(sl)):.2f}")
        log_axes(ax)
        ax.set_xlabel(r"$\sigma$")
        ax.set_title(f"{tree} (K={K}), predicted slope {K - 2}\n" + "\n".join(lab), fontsize=6.5)
    axes[0][0].set_ylabel(r"$-\lambda_{\min}$ (gauge-projected Hessian)")
    handles = [plt.Line2D([], [], color=colors[0], lw=1, label="first plateau"),
               plt.Line2D([], [], color=colors[1], lw=1, label="second plateau")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, fontsize=7)
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
    ap.add_argument("--out", required=True, help="output directory (e.g. experiments/s2s_core/results)")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    tp.set_style()
    runs = pd.read_csv(res / "main_runs.csv")
    fits = pd.read_csv(res / "main_gap_fits.csv")
    for chi in (3, 6):
        for teacher in ("separated", "close"):
            fig_gaps(runs, fits, out, chi, teacher)
    fig_hessian(runs, fits, out)
    if (res / "first_runs.csv").exists():
        first = pd.read_csv(res / "first_runs.csv")
        for chi in (3, 6):
            for init in ("fanin", "balanced"):
                if len(first[(first.chi == chi) & (first.init == init)]):
                    fig_first(first, out, chi, init)


if __name__ == "__main__":
    main()
