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
            lt = run["p2_log"]
            axes[1, j].plot(lt["t"], lt["test"][:, 0], color=colors[r["sigma"]], lw=0.8, ls="--")
        axes[0, j].set_title(label, fontsize=9)
        axes[0, j].set_xlim(1, None)
        axes[1, j].set_xlim(0, 600)
        axes[1, j].set_xlabel(r"$\tau = t - t_a$")
    axes[0, 0].set_ylabel("train loss")
    axes[1, 0].set_ylabel("train (solid) / test (dashed)")
    h, lab = axes[0, 2].get_legend_handles_labels()
    order = np.argsort([float(x) for x in lab])
    axes[0, 2].legend([h[i] for i in order], [lab[i] for i in order], title=r"$\sigma$", fontsize=7,
                      title_fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    savefig(fig, Path(out) / f"loss_curves_{feats}")
    plt.close(fig)


def fig_bond_spectra(rows, runs_dir, out, feats, sigma=0.1, seed=0):
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    from ttn.plotting import sequential_cmap

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH), sharey=True)
    styles = ["-", "--", ":"]
    cmap = sequential_cmap()
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
        norm = Normalize(0.5, lv.max() + 0.5)
        for e in range(fs.shape[1]):
            for k in range(1, fs.shape[2]):
                axes[j].loglog(p2["t"], fs[:, e, k] / fs[:, e, 0], color=cmap(0.25 + 0.75 * norm(lv[e])),
                               ls=styles[k - 1], lw=0.7)
        axes[j].set_title(label, fontsize=9)
        axes[j].set_xlim(1, p2["t"][-1])
        axes[j].set_ylim(1e-5, 2)
        axes[j].axhline(0.05, color=MUTED, lw=0.6)
        axes[j].set_xlabel(r"$\tau$")
        axes[j].text(0.03, 0.03, f"levels 1-{lv.max()}", transform=axes[j].transAxes, fontsize=6,
                     color=MUTED)
    axes[0].set_ylabel(r"$s_k^e/s_1^e$ (data-weighted)")
    handles = [Line2D([], [], color="k", ls=s_, lw=0.8, label=f"$k = {k + 2}$") for k, s_ in enumerate(styles)]
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.03, r"colour: edge level, light = bottom edges, dark = next to the root; grey line: 0.05",
             ha="center", fontsize=7)
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


def _sigma_ticks(ax):
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())


def classified_first_route(rows, tree_key_):
    """{(ordering, seed): [(sigma, gap)]} with gap = (loss drop carrying the seed's first route) - (first
    drop), only at sigma values where the first route equals the one at the smallest sigma."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("drops") and tree_key(r) == tree_key_:
            groups[(r["ordering"], r["seed"])].append(r)
    out = {}
    for key, rs in groups.items():
        rs.sort(key=lambda r: r["sigma"])
        ref = rs[0].get("first_rank2_edges")
        pts = []
        for r in rs:
            later = [d for d in r["drops"][1:] if d["t_hi"] + d["width"] >= r["t_first_rank2"]]
            if r.get("first_rank2_edges") == ref and later:
                pts.append((r["sigma"], later[0]["t"] - r["drops"][0]["t"]))
        out[key] = pts
    return out


def fig_scaling(rows, out, feats):
    rows = [r for r in rows if r["features"] == feats]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH))
    stats = {}
    for label in [t[2] for t in TREES]:
        by_seed = defaultdict(list)
        for r in rows:
            if tree_key(r) == label and np.isfinite(r.get("t1", np.nan)):
                by_seed[(r["ordering"], r["seed"])].append((r["sigma"], r["t1"]))
        slopes = []
        for pts in by_seed.values():
            pts.sort()
            s_, t1 = np.array([p_[0] for p_ in pts]), np.array([p_[1] for p_ in pts])
            axes[0].loglog(s_, t1, color=TREE_COLOR[label], lw=0.5, alpha=0.6)
            if len(s_) >= 3:
                slopes.append(per_seed_fit(s_, t1))
        stats[label] = (float(np.mean(slopes)), float(np.std(slopes) / np.sqrt(max(len(slopes), 1))), len(slopes))
        axes[0].plot([], [], color=TREE_COLOR[label], label=f"{label}: {-stats[label][0]:.3f} ({stats[label][2]})")
    axes[0].set_xlabel(r"$\sigma$")
    axes[0].set_ylabel(r"$t_1$ (first escape)")
    axes[0].legend(fontsize=5.5, frameon=False, title=r"exponent of $t_1 \propto \sigma^{-p}$", title_fontsize=6)
    # (b) classified first-route gap on binary trees
    for label in ("binary, quadtree", "binary, scrambled"):
        for pts in classified_first_route(rows, label).values():
            if len(pts) >= 4:
                axes[1].loglog([p_[0] for p_ in pts], [p_[1] for p_ in pts], color=TREE_COLOR[label],
                               lw=0.7, marker="o", ms=2)
        axes[1].plot([], [], color=TREE_COLOR[label], marker="o", ms=2, label=label)
    xs = np.array([0.05, 0.5])
    axes[1].plot(xs, 15 / xs, color=MUTED, lw=0.8, ls="--", label=r"$\propto\sigma^{-1}$ ($k=3$)")
    axes[1].set_xlabel(r"$\sigma$")
    axes[1].set_ylabel("gap to the first-route drop")
    axes[1].legend(fontsize=6, frameon=False)
    # (c) MPS: time of the first rank increase on edges 4 and 3 vs ln(1/sigma)
    for e, ls in ((4, "-"), (3, "--")):
        by_seed = defaultdict(list)
        for r in rows:
            if tree_key(r) == "MPS, snake" and r.get("drops"):
                t_e = np.array(r["rank_events"], float)[e, 0] - r["drops"][0]["t"]
                by_seed[r["seed"]].append((r["sigma"], t_e))
        for pts in by_seed.values():
            pts.sort()
            axes[2].plot([np.log(1 / p_[0]) for p_ in pts], [p_[1] for p_ in pts],
                         color=TREE_COLOR["MPS, snake"], lw=0.7, ls=ls, marker="o", ms=2)
        axes[2].plot([], [], color=TREE_COLOR["MPS, snake"], ls=ls, label=f"MPS edge {e}")
    axes[2].set_xlabel(r"$\ln(1/\sigma)$")
    axes[2].set_ylabel("time of first rank increase")
    axes[2].legend(fontsize=6, frameon=False)
    for ax in axes[:2]:
        _sigma_ticks(ax)
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


def fig_routes(routes_json, out):
    """Route amplitudes at the rank-1 plateau: lambda_uw against the route degree k, quadtree vs
    scrambled; filled marker = the sibling pair that actually gained rank first."""
    res = [r for r in json.load(open(routes_json)) if "pairs" in r and r["features"] == "raw"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH), sharey=True)
    rng = np.random.default_rng(1)
    for ax, key, label in ((axes[0], "quadtree", "binary, quadtree"), (axes[1], "scrambled", "binary, scrambled")):
        col = TREE_COLOR[label]
        n = 0
        for r in res:
            if not r["ordering"].startswith(key):
                continue
            n += 1
            edges = r["edges"]
            obs = sorted(r.get("first_rank2_edges") or [])
            for p_ in r["pairs"]:
                first = sorted([edges.index(p_["u"]), edges.index(p_["w"])]) == obs
                x = p_["k"] + rng.uniform(-0.35, 0.35)
                ax.scatter(x, p_["lam"], s=10 if first else 4, color=col if first else "none",
                           edgecolors=col, lw=0.5, alpha=1.0 if first else 0.5, zorder=3 if first else 2)
        ax.set_yscale("log")
        ax.set_xticks([3, 5, 7])
        ax.set_xlabel(r"route degree $k$ (cores between the two bottom cores)")
        ax.set_title(f"{label} ({n} runs)", fontsize=9)
    axes[0].set_ylabel(r"route amplitude $\lambda_{uw}$")
    fig.text(0.5, -0.03, "filled: the pair whose edges gained rank first", ha="center", fontsize=7)
    fig.tight_layout()
    savefig(fig, Path(out) / "route_amplitudes")
    plt.close(fig)


def fig_hessians(hess_json, out, levels=(("mps", 0.225, 0.235), ("binary", 0.265, 0.275))):
    """|lambda_min| of the Hessian at the first plateau's midpoint vs sigma; filled = gauge-projected,
    open = raw; only runs at the common plateau level (same saddle)."""
    res = [r for r in json.load(open(hess_json)) if "raw_min" in r]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH / 2, 0.75 * TEXT_WIDTH / 2))
    labels = {"mps": "MPS, snake", "binary": "binary, quadtree"}
    for tree, lo, hi in levels:
        sel = [r for r in res if r["tree"] == tree and lo <= r["loss_mid"] <= hi]
        col = TREE_COLOR[labels[tree]]
        sig = np.array([r["sigma"] for r in sel])
        ax.scatter(sig, [-r["proj_min"] for r in sel], s=10, color=col, lw=0,
                   label=f"{labels[tree]} (L = {np.mean([r['loss_mid'] for r in sel]):.2f})")
        ax.scatter(sig * 1.04, [-r["raw_min"] for r in sel], s=10, facecolors="none", edgecolors=col, lw=0.6)
    xs = np.array([0.05, 0.3])
    ax.plot(xs, 0.3 * xs, color=MUTED, lw=0.8, ls="--")
    ax.text(0.31, 0.09, r"$\propto\sigma$", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$|\lambda_{\min}|$ at the first plateau")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    savefig(fig, Path(out) / "plateau_hessians")
    plt.close(fig)


def fig_regime(regime_json, out):
    """Regime sweep: delta_K(T), train loss vs the linearized-model loss at the same time, test accuracy."""
    from ttn.plotting import regime_colors

    rows = [r for r in json.load(open(regime_json)) if r.get("delta_K_end") is not None]
    cats = ["S2S", "rich", "lazy"]
    cols = regime_colors(cats)
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.33 * TEXT_WIDTH))
    rng = np.random.default_rng(2)
    for r in rows:
        x = r["sigma"] * np.exp(rng.uniform(-0.04, 0.04))
        mk = "o" if r.get("model") == "centred" else "s"
        c = cols.get(r["label"], MUTED)
        axes[0].scatter(x, r["delta_K_end"], s=10, color=c, marker=mk, lw=0)
        axes[1].scatter(x, r["loss_end"], s=10, color=c, marker=mk, lw=0)
        axes[1].scatter(x, r["L_lin_T"], s=10, facecolors="none", edgecolors=MUTED, marker=mk, lw=0.6)
        axes[2].scatter(x, r["test_acc_end"], s=10, color=c, marker=mk, lw=0)
    axes[0].axhline(0.1, color=MUTED, lw=0.6, ls="--")
    axes[0].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
        ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.3, 1, 3]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel(r"$\delta_K(T)$")
    axes[1].set_ylabel("train loss at T (open: linearized)")
    axes[2].set_ylabel("test accuracy at T")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma$")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=cols[k], marker="o", ls="", ms=4, label=k) for k in cats]
    handles += [Line2D([], [], color=MUTED, marker="s", ls="", ms=4, label="plain model"),
                Line2D([], [], color=MUTED, marker="o", ls="", ms=4, label="centred model")]
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    savefig(fig, Path(out) / "regime_sweep")
    plt.close(fig)


def main(summary_json, runs_dir, out_dir, feats="raw", routes_json=None, hess_json=None,
         regime_json=None):
    set_style()
    rows = [r for r in json.load(open(summary_json)) if "error" not in r and not r.get("centred")]
    fig_loss_curves(rows, runs_dir, out_dir, feats)
    for s in sorted({r["sigma"] for r in rows}):
        if any(abs(s - x) < 1e-9 for x in (0.05, 0.1, 0.5)) or len({r["sigma"] for r in rows}) < 4:
            fig_bond_spectra(rows, runs_dir, out_dir, feats, sigma=s)
            fig_growth_order(rows, out_dir, feats, sigma=s)
            fig_locality(rows, runs_dir, out_dir, feats, sigma=s)
    stats = fig_scaling(rows, out_dir, feats)
    print(json.dumps(stats))
    if routes_json:
        fig_routes(routes_json, out_dir)
    if hess_json:
        fig_hessians(hess_json, out_dir)
    if regime_json:
        fig_regime(regime_json, out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
