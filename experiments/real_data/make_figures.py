"""Figures for the real-data experiment from summarize.py output and the raw trajectories.

Usage: uv run python experiments/real_data/make_figures.py <summary.json> <runs_dir> <out_dir> [features]

Figures (features = raw by default):
  loss_curves      train loss vs tau per tree, one curve per sigma (seed 0); linear-time inset view
  bond_spectra     s_k/s_1 on every edge vs tau (seed 0, sigma = 0.1), colour = edge level
  growth_order     time of each edge's first rank increase vs the edge's level, all seeds (sigma = 0.1)
  scaling          t1 vs sigma per seed (slope -(K-2)); gap between the first two drops vs sigma
  locality         mean +- SE train / test loss vs tau for quadtree, scrambled (3 orders), MPS
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analysis import load  # noqa: E402

from ttn.plotting import MUTED, TEXT_WIDTH, ordinal_colors, palette, savefig, set_style  # noqa: E402

TREES = [("binary", "quadtree", "binary, quadtree"), ("binary", "scrambled", "binary, scrambled"),
         ("mps", "snake", "MPS, snake")]
TREE_COLOR = dict(zip([t[2] for t in TREES], palette(3)))
K_CORES = {"binary": 15, "mps": 16}


def tree_key(row) -> str | None:
    for tree, order, label in TREES:
        if row["tree"] == tree and row["ordering"].startswith(order):
            return label
    return None


def run_path(runs_dir, row) -> Path:
    return Path(runs_dir) / f"{row['name']}.npz"


def common_grid(runs, t_max, n=400):
    grid = np.concatenate([[0.0], np.geomspace(1e-1, t_max, n)])
    out = []
    for r in runs:
        lg = r["p2_log"]
        out.append(np.interp(grid, lg["t"], lg["loss"], right=np.nan))
    return grid, np.array(out)


def fig_loss_curves(rows, runs_dir, out, feats, seed=0):
    sigmas = sorted({r["sigma"] for r in rows})
    colors = dict(zip(sigmas, ordinal_colors(len(sigmas))))
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 0.62 * TEXT_WIDTH), sharey="row")
    for j, (_, _, label) in enumerate(TREES):
        for r in rows:
            if tree_key(r) != label or r["seed"] != seed or r["features"] != feats:
                continue
            if label == "binary, scrambled" and r["ordering"] != "scrambled0":
                continue
            run = load(run_path(runs_dir, r))
            lg = run["p2"]
            axes[0, j].semilogx(lg["t"], lg["loss"], color=colors[r["sigma"]], lw=1.0,
                                label=f"{r['sigma']:g}")
            axes[1, j].plot(lg["t"], lg["loss"], color=colors[r["sigma"]], lw=1.0)
        axes[0, j].set_title(label, fontsize=9)
        axes[0, j].set_xlim(1, None)
        axes[1, j].set_xlim(0, 600)
        axes[1, j].set_xlabel(r"$\tau = t - t_a$")
    axes[0, 0].set_ylabel("train loss")
    axes[1, 0].set_ylabel("train loss (linear time)")
    axes[0, 2].legend(title=r"$\sigma$", fontsize=7, title_fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    savefig(fig, Path(out) / f"loss_curves_{feats}")
    plt.close(fig)


def fig_bond_spectra(rows, runs_dir, out, feats, sigma=0.1, seed=0):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH), sharey=True)
    styles = ["-", "--", ":"]
    for j, (_, _, label) in enumerate(TREES):
        cand = [r for r in rows if tree_key(r) == label and r["seed"] == seed and r["features"] == feats
                and abs(r["sigma"] - sigma) < 1e-9 and r["ordering"] in ("quadtree", "snake", "scrambled0")]
        if not cand:
            continue
        r = cand[0]
        run = load(run_path(runs_dir, r))
        p2 = run["p2"]
        fs = p2["fspectra"]
        lv = np.array(r["edge_levels"])
        cols = ordinal_colors(int(lv.max()))
        for e in range(fs.shape[1]):
            for k in range(1, fs.shape[2]):
                axes[j].loglog(p2["t"], fs[:, e, k] / fs[:, e, 0], color=cols[lv[e] - 1],
                               ls=styles[k - 1], lw=0.7)
        axes[j].set_title(label, fontsize=9)
        axes[j].set_xlim(1, p2["t"][-1])
        axes[j].set_ylim(1e-5, 2)
        axes[j].axhline(0.05, color=MUTED, lw=0.6)
        axes[j].set_xlabel(r"$\tau$")
    axes[0].set_ylabel(r"$s_k^e/s_1^e$ (data-weighted)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="k", ls=s, lw=0.8, label=f"k = {k + 2}") for k, s in enumerate(styles)]
    axes[2].legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    fig.text(0.5, -0.02, "colour: edge level (light = bottom, dark = root)", ha="center", fontsize=7)
    fig.tight_layout()
    savefig(fig, Path(out) / f"bond_spectra_{feats}_s{sigma:g}")
    plt.close(fig)


def fig_growth_order(rows, out, feats, sigma=0.1):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH), sharey=True)
    rng = np.random.default_rng(0)
    for j, (_, _, label) in enumerate(TREES):
        for r in rows:
            if tree_key(r) != label or r["features"] != feats or abs(r["sigma"] - sigma) > 1e-9:
                continue
            ev = np.array(r["rank_events"], dtype=float)[:, 0]
            lv = np.array(r["edge_levels"])
            ok = np.isfinite(ev)
            axes[j].scatter(lv[ok] + rng.uniform(-0.2, 0.2, ok.sum()), np.maximum(ev[ok], 0.3),
                            s=6, color=TREE_COLOR[label], alpha=0.6, lw=0)
            axes[j].scatter(lv[~ok] + rng.uniform(-0.2, 0.2, (~ok).sum()),
                            np.full((~ok).sum(), 3e4), s=6, marker="x", color=MUTED, lw=0.6)
        axes[j].set_yscale("log")
        axes[j].set_title(label, fontsize=9)
        axes[j].set_xlabel("edge level (1 = bottom)")
    axes[0].set_ylabel(r"$\tau$ of first $s_2/s_1 \geq 0.05$")
    fig.text(0.5, -0.02, r"x at top: never within $\tau \leq 10^4$", ha="center", fontsize=7)
    fig.tight_layout()
    savefig(fig, Path(out) / f"growth_order_{feats}_s{sigma:g}")
    plt.close(fig)


def per_seed_fit(xs, ys):
    """Slope of log y vs log x by least squares."""
    xs, ys = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    return np.polyfit(xs, ys, 1)[0]


def fig_scaling(rows, out, feats):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.4 * TEXT_WIDTH))
    stats = {}
    for label in [t[2] for t in TREES]:
        by_seed = defaultdict(list)
        for r in rows:
            if tree_key(r) == label and r["features"] == feats and np.isfinite(r.get("t1", np.nan)):
                by_seed[(r["ordering"], r["seed"])].append((r["sigma"], r["t1"], r.get("gaps", [np.nan])))
        slopes = []
        for key, pts in by_seed.items():
            pts.sort()
            s, t1 = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            axes[0].loglog(s, t1, color=TREE_COLOR[label], lw=0.5, alpha=0.6)
            if len(s) >= 3:
                slopes.append(per_seed_fit(s, t1))
        stats[label] = (np.mean(slopes), np.std(slopes) / np.sqrt(max(len(slopes), 1)), len(slopes))
        # gap to the second drop
        sig = sorted({r["sigma"] for r in rows if tree_key(r) == label and r["features"] == feats})
        means, ses = [], []
        for s in sig:
            g = [r["gaps"][0] for r in rows if tree_key(r) == label and r["features"] == feats
                 and abs(r["sigma"] - s) < 1e-9 and r.get("gaps")]
            means.append(np.mean(g) if g else np.nan)
            ses.append(np.std(g) / np.sqrt(len(g)) if len(g) > 1 else np.nan)
        axes[1].errorbar(sig, means, yerr=ses, color=TREE_COLOR[label], marker="o", ms=3, lw=1,
                         capsize=2, label=label)
    for label, (m, se, n) in stats.items():
        axes[0].plot([], [], color=TREE_COLOR[label], label=f"{label}: {m:.3f} $\\pm$ {se:.3f} ({n})")
    axes[0].set_xlabel(r"$\sigma$")
    axes[0].set_ylabel(r"$t_1$ (first escape)")
    axes[0].legend(fontsize=6, frameon=False, title="per-seed slope", title_fontsize=6)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\sigma$")
    axes[1].set_ylabel("gap first -> second drop")
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    savefig(fig, Path(out) / f"scaling_{feats}")
    plt.close(fig)
    return stats


def fig_locality(rows, runs_dir, out, feats, sigma=0.1, t_max=1e4):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.4 * TEXT_WIDTH), sharey=True)
    for label in [t[2] for t in TREES]:
        sel = [r for r in rows if tree_key(r) == label and r["features"] == feats
               and abs(r["sigma"] - sigma) < 1e-9]
        if not sel:
            continue
        runs = [load(run_path(runs_dir, r)) for r in sel]
        for ax, key, idx in ((axes[0], "loss", None), (axes[1], "test", 0)):
            grid = np.concatenate([[0.0], np.geomspace(1e-1, t_max, 400)])
            curves = []
            for run in runs:
                lg = run["p2_log"]
                y = lg[key] if idx is None else lg[key][:, idx]
                curves.append(np.interp(grid, lg["t"], y, right=np.nan))
            curves = np.array(curves)
            m = np.nanmean(curves, 0)
            se = np.nanstd(curves, 0) / np.sqrt(np.sum(np.isfinite(curves), 0))
            ax.semilogx(grid, m, color=TREE_COLOR[label], lw=1.2, label=f"{label} ({len(runs)})")
            ax.fill_between(grid, m - se, m + se, color=TREE_COLOR[label], alpha=0.25, lw=0)
    axes[0].set_ylabel("loss")
    axes[0].set_title("train", fontsize=9)
    axes[1].set_title("test", fontsize=9)
    for ax in axes:
        ax.set_xlabel(r"$\tau$")
        ax.set_xlim(1, t_max)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    savefig(fig, Path(out) / f"locality_{feats}_s{sigma:g}")
    plt.close(fig)


def main(summary_json, runs_dir, out_dir, feats="raw"):
    set_style()
    rows = [r for r in json.load(open(summary_json)) if "error" not in r and not r.get("centred")]
    fig_loss_curves(rows, runs_dir, out_dir, feats)
    for s in sorted({r["sigma"] for r in rows}):
        if abs(s - 0.1) < 1e-9 or len({r["sigma"] for r in rows}) < 4:
            fig_bond_spectra(rows, runs_dir, out_dir, feats, sigma=s)
            fig_growth_order(rows, out_dir, feats, sigma=s)
            fig_locality(rows, runs_dir, out_dir, feats, sigma=s)
    stats = fig_scaling(rows, out_dir, feats)
    print(json.dumps(stats))


if __name__ == "__main__":
    main(*sys.argv[1:])
