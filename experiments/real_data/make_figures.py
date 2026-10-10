"""Figures for the real-data experiment from summarize.py output and the raw trajectories.

Usage: uv run python experiments/real_data/make_figures.py <summary.json> <runs_dir> <out_dir> [features]
           [routes.json hessians.json regime_analysis.json]
       uv run python experiments/real_data/make_figures.py --json <results_dir> <out_dir>
       (only the figures made from committed JSON files: route_amplitudes, plateau_hessians,
        regime_sweep; from fix1_analysis.json first_route_gaps, locality_orders, scaling_fix1; from
        order_test.json the pre-registered order test, order_test)

Figures (features = raw by default):
  loss_curves      train loss vs tau per tree, one curve per sigma (seed 0); linear-time inset view
  bond_spectra     s_k/s_1 on every edge vs tau (seed 0, sigma = 0.1), colour = edge level
  growth_order     time of each edge's first rank increase vs the edge's level, all seeds (sigma = 0.1)
  scaling          t1 vs sigma per seed (slope -(K-2)); classified first-route gap vs sigma (fix round 1:
                   attribution by the drop window that contains the route's rank crossing); MPS edges 4, 3
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

from ttn.plotting import (  # noqa: E402
    INK_SECONDARY,
    MUTED,
    TEXT_WIDTH,
    ordinal_colors,
    palette,
    savefig,
    set_style,
)

TREES = [
    ("binary", "quadtree", "binary, quadtree"),
    ("binary", "scrambled", "binary, scrambled"),
    ("mps", "snake", "MPS, snake"),
]
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
            axes[0, j].semilogx(
                lg["t"], lg["loss"], color=colors[r["sigma"]], lw=1.0, label=f"{r['sigma']:g}"
            )
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
    axes[0, 2].legend(
        [h[i] for i in order],
        [lab[i] for i in order],
        title=r"$\sigma$",
        fontsize=7,
        title_fontsize=7,
        frameon=False,
        ncol=2,
    )
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
        cand = [
            r
            for r in rows
            if tree_key(r) == label
            and r["seed"] == seed
            and r["features"] == feats
            and abs(r["sigma"] - sigma) < 1e-9
            and r["ordering"] in ("quadtree", "snake", "scrambled0")
        ]
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
                axes[j].loglog(
                    p2["t"],
                    fs[:, e, k] / fs[:, e, 0],
                    color=cmap(0.25 + 0.75 * norm(lv[e])),
                    ls=styles[k - 1],
                    lw=0.7,
                )
        axes[j].set_title(label, fontsize=9)
        axes[j].set_xlim(1, p2["t"][-1])
        axes[j].set_ylim(1e-5, 2)
        axes[j].axhline(0.05, color=MUTED, lw=0.6)
        axes[j].set_xlabel(r"$\tau$")
        axes[j].text(
            0.03, 0.03, f"levels 1-{lv.max()}", transform=axes[j].transAxes, fontsize=6, color=MUTED
        )
    axes[0].set_ylabel(r"$s_k^e/s_1^e$ (data-weighted)")
    handles = [
        Line2D([], [], color="k", ls=s_, lw=0.8, label=f"$k = {k + 2}$")
        for k, s_ in enumerate(styles)
    ]
    fig.legend(
        handles=handles,
        fontsize=7,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.text(
        0.5,
        -0.03,
        r"colour: edge level, light = bottom edges, dark = next to the root; grey line: 0.05",
        ha="center",
        fontsize=7,
    )
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
            axes[j].scatter(
                lv[ok] + rng.uniform(-0.2, 0.2, ok.sum()),
                np.maximum(ev[ok], 0.3),
                s=6,
                color=TREE_COLOR[label],
                alpha=0.6,
                lw=0,
            )
            axes[j].scatter(
                lv[~ok] + rng.uniform(-0.2, 0.2, (~ok).sum()),
                np.full((~ok).sum(), 3e4),
                s=6,
                marker="x",
                color=MUTED,
                lw=0.6,
            )
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
    """{(ordering, seed): [(sigma, gap)]}: gap = (loss drop whose window contains the seed's first-route rank
    crossing) - (first drop), at sigma values where the first route equals the one at the smallest sigma;
    crossings merged with drop 0 or between drops are left out (`scaling.first_route_points`, fix round 1)."""
    from scaling import first_route_points

    sel = [r for r in rows if tree_key(r) == tree_key_]
    return {
        (tree.split("_", 1)[1].split("/")[0], seed): [(s, g) for s, g, _, _ in d["kept"]]
        for (tree, seed), d in first_route_points(sel).items()
    }


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
        stats[label] = (
            float(np.mean(slopes)),
            float(np.std(slopes) / np.sqrt(max(len(slopes), 1))),
            len(slopes),
        )
        axes[0].plot(
            [],
            [],
            color=TREE_COLOR[label],
            label=f"{label}: {-stats[label][0]:.3f} ({stats[label][2]})",
        )
    axes[0].set_xlabel(r"$\sigma$")
    axes[0].set_ylabel(r"$t_1$ (first escape)")
    axes[0].legend(
        fontsize=5.5,
        frameon=False,
        title=r"exponent of $t_1 \propto \sigma^{-p}$",
        title_fontsize=6,
    )
    # (b) classified first-route gap on binary trees
    for label in ("binary, quadtree", "binary, scrambled"):
        for pts in classified_first_route(rows, label).values():
            if len(pts) >= 4:
                axes[1].loglog(
                    [p_[0] for p_ in pts],
                    [p_[1] for p_ in pts],
                    color=TREE_COLOR[label],
                    lw=0.7,
                    marker="o",
                    ms=2,
                )
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
            axes[2].plot(
                [np.log(1 / p_[0]) for p_ in pts],
                [p_[1] for p_ in pts],
                color=TREE_COLOR["MPS, snake"],
                lw=0.7,
                ls=ls,
                marker="o",
                ms=2,
            )
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
        sel = [
            r
            for r in rows
            if tree_key(r) == label and r["features"] == feats and abs(r["sigma"] - sigma) < 1e-9
        ]
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


def fig_routes(routes_json, out, err_max=0.05):
    """Route amplitudes at the rank-1 plateau: lambda_uw against the route degree k, quadtree vs
    scrambled; filled marker = the sibling pair that actually gained rank first. Only states within
    err_max of a rank-1 product (fix round 1: the amplitude formula assumes a rank-1 state)."""
    states = [r for r in json.load(open(routes_json)) if "pairs" in r and r["features"] == "raw"]
    res = [r for r in states if r["rank1_rel_err"] <= err_max]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH), sharey=True)
    rng = np.random.default_rng(1)
    for ax, key, label in (
        (axes[0], "quadtree", "binary, quadtree"),
        (axes[1], "scrambled", "binary, scrambled"),
    ):
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
                ax.scatter(
                    x,
                    p_["lam"],
                    s=10 if first else 4,
                    color=col if first else "none",
                    edgecolors=col,
                    lw=0.5,
                    alpha=1.0 if first else 0.5,
                    zorder=3 if first else 2,
                )
        ax.set_yscale("log")
        ax.set_xticks([3, 5, 7])
        ax.set_xlabel(r"route degree $k$ (cores between the two bottom cores)")
        ax.set_title(f"{label} ({n} rank-1 states)", fontsize=9)
    axes[0].set_ylabel(r"route amplitude $\lambda_{uw}$")
    fig.text(
        0.5,
        -0.03,
        f"filled: the pair whose edges gained rank first; {len(states) - len(res)} states farther "
        f"than {err_max:g} from rank 1 left out",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout()
    savefig(fig, Path(out) / "route_amplitudes")
    plt.close(fig)


def fig_hessians(hess_json, out, levels=(("mps", 0.225, 0.235), ("binary", 0.265, 0.275))):
    """|lambda_min| of the Hessian at the first plateau's midpoint vs sigma, one line per seed; filled =
    gauge-projected, open = raw; only runs at the common plateau level (same saddle)."""
    res = [r for r in json.load(open(hess_json)) if "raw_min" in r]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH / 2, 0.75 * TEXT_WIDTH / 2))
    labels = {"mps": "MPS, snake", "binary": "binary, quadtree"}
    for tree, lo, hi in levels:
        sel = [r for r in res if r["tree"] == tree and lo <= r["loss_mid"] <= hi]
        col = TREE_COLOR[labels[tree]]
        for i, seed in enumerate(sorted({r["seed"] for r in sel})):
            rs = sorted((r for r in sel if r["seed"] == seed), key=lambda r: r["sigma"])
            sig = np.array([r["sigma"] for r in rs])
            ax.plot(
                sig,
                [-r["proj_min"] for r in rs],
                color=col,
                lw=0.6,
                marker="o",
                ms=2.5,
                label=f"{labels[tree]} (L = {np.mean([r['loss_mid'] for r in sel]):.2f})"
                if i == 0
                else None,
            )
            ax.scatter(
                sig * 1.04,
                [-r["raw_min"] for r in rs],
                s=8,
                facecolors="none",
                edgecolors=col,
                lw=0.5,
            )
    xs = np.array([0.05, 0.3])
    ax.plot(xs, 0.3 * xs, color=MUTED, lw=0.8, ls="--")
    ax.text(0.31, 0.09, r"$\propto\sigma$", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$")
    _sigma_ticks(ax)
    ax.set_ylabel(r"$|\lambda_{\min}|$ at the first plateau")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    savefig(fig, Path(out) / "plateau_hessians")
    plt.close(fig)


def fig_regime(regime_json, out):
    """Regime sweep: delta_K(T), train loss vs the linearized-model loss at the same time (band: the lazy
    floors L_lin(inf) of the five seeds), test accuracy."""
    from ttn.plotting import regime_colors

    rows = [r for r in json.load(open(regime_json)) if r.get("delta_K_end") is not None]
    cats = ["S2S", "rich", "lazy"]
    cols = regime_colors(cats)
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.33 * TEXT_WIDTH))
    rng = np.random.default_rng(2)
    floors = [r["L_lin_floor"] for r in rows]
    axes[1].axhspan(min(floors), max(floors), color=MUTED, alpha=0.15, lw=0)
    for r in rows:
        x = r["sigma"] * np.exp(rng.uniform(-0.04, 0.04))
        mk = "o" if r.get("model") == "centred" else "s"
        c = cols.get(r["label"], MUTED)
        axes[0].scatter(x, r["delta_K_end"], s=10, color=c, marker=mk, lw=0)
        axes[1].scatter(x, r["loss_end"], s=10, color=c, marker=mk, lw=0)
        axes[1].scatter(
            x, r["L_lin_T"], s=10, facecolors="none", edgecolors=MUTED, marker=mk, lw=0.6
        )
        axes[2].scatter(x, r["test_acc_end"], s=10, color=c, marker=mk, lw=0)
    axes[0].axhline(0.1, color=MUTED, lw=0.6, ls="--")
    axes[0].set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.3, 1, 3]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r"$\sigma$")
    axes[0].set_ylabel(r"$\delta_K(T)$")
    axes[1].set_ylabel("train loss at T (open: linearized)")
    axes[1].text(
        0.06,
        max(floors),
        r"band: lazy floors $L_{\rm lin}(\infty)$",
        fontsize=5.5,
        color=MUTED,
        va="bottom",
    )
    axes[2].set_ylabel("test accuracy at T")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=cols[k], marker="o", ls="", ms=4, label=k) for k in cats]
    handles += [
        Line2D([], [], color=MUTED, marker="s", ls="", ms=4, label="plain model"),
        Line2D([], [], color=MUTED, marker="o", ls="", ms=4, label="centred model"),
    ]
    fig.legend(
        handles=handles,
        fontsize=7,
        frameon=False,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout()
    savefig(fig, Path(out) / "regime_sweep")
    plt.close(fig)


ORDER_COLOR = dict(zip(["quadtree", "scrambled0", "scrambled1", "scrambled2"], palette(4)))


def fig_first_route_clean(fix1_json, out):
    """Clean k = 3 first-route series (fix1_analysis.json): gap vs sigma with the A / sigma + B fits (left),
    and sigma x gap vs sigma (right), where A / sigma + B is a straight line of slope B."""
    series = json.load(open(fix1_json))["first_route"]["series"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH))
    markers = dict(zip(ORDER_COLOR, "osD^"))
    seen = set()
    for s in series:
        order = s["tree"].split(":")[1]
        col, mk = ORDER_COLOR[order], markers[order]
        sig, gap = np.array(s["sigmas"]), np.array(s["gaps"])
        xs = np.geomspace(sig.min(), sig.max(), 50)
        lab = order if order not in seen else None
        seen.add(order)
        axes[0].loglog(sig, gap, ls="", marker=mk, ms=3, color=col, label=lab)
        axes[0].loglog(xs, s["A"] / xs + s["B"], color=col, lw=0.8)
        axes[1].plot(sig, sig * gap, ls="", marker=mk, ms=3, color=col)
        axes[1].plot(xs, s["A"] + s["B"] * xs, color=col, lw=0.8)
    xs = np.array([0.05, 0.5])
    axes[0].plot(xs, 10 / xs, color=MUTED, lw=0.8, ls="--", label=r"$\propto\sigma^{-1}$")
    axes[0].set_ylabel(r"gap between loss drops 0 and 1 ($\tau$)")
    axes[1].set_ylabel(r"$\sigma\times$ gap")
    axes[0].set_title(r"lines: fits $A/\sigma + B$", fontsize=8)
    axes[1].set_title(r"$A/\sigma+B$ is a line of slope $B<0$", fontsize=8)
    for ax in axes:
        ax.set_xlabel(r"$\sigma$")
    _sigma_ticks(axes[0])
    # legend below the panels: inside, it would cover the series
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    savefig(fig, Path(out) / "first_route_gaps")
    plt.close(fig)


def fig_locality_orders(fix1_json, out):
    """Per (order, seed) at sigma = 0.1 (fix1_analysis.json 'locality'): final train loss vs the share of
    sum lambda^2 on k <= 5 routes at the first plateau (left), and the final losses grouped by order (right)."""
    lo = json.load(open(fix1_json))["locality"]
    groups = {
        "quadtree": ("quadtree", palette(4)[0]),
        "rowmajor": ("local, other", palette(4)[2]),
        "quadtreeT": ("local, other", palette(4)[2]),
    }
    random_col = palette(4)[1]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.4 * TEXT_WIDTH), sharey=True)
    seen = set()
    for r in lo["rows"]:
        name, col = groups.get(r["ordering"], ("random", random_col))
        mk = "o" if r["source"] == "fix1" else "s"
        key = (name, r["source"])
        lab = (
            f"{name} ({'main-sweep case' if r['source'] == 'fix1' else 'review run'})"
            if key not in seen
            else None
        )
        seen.add(key)
        axes[0].scatter(
            r["share_k5"], r["loss_end"], s=14, color=col, marker=mk, lw=0, label=lab, zorder=3
        )
    sp = lo["spearman_all"]
    axes[0].set_title(f"Spearman {sp['rho']:.2f} (p = {sp['p']:.1g}, n = {sp['n']})", fontsize=8)
    axes[0].set_xlabel(r"share of $\sum\lambda^2$ on $k\leq5$ routes (first plateau)")
    axes[0].set_ylabel(r"train loss at $\tau = 10^4$")
    axes[0].legend(fontsize=5.5, frameon=False, loc="upper right")
    names = list(lo["per_order"])
    rng = np.random.default_rng(3)
    for i, o in enumerate(names):
        v = lo["per_order"][o]["loss_values"]
        _, col = groups.get(o, ("random", random_col))
        axes[1].scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color=col, lw=0, zorder=3)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=60, fontsize=6)
    axes[1].set_title(r"$\sigma = 0.1$, one point per seed", fontsize=8)
    fig.tight_layout()
    savefig(fig, Path(out) / "locality_orders")
    plt.close(fig)


def fig_scaling_fix1(fix1_json, regime_json, out):
    """Fix round 1 replacement of `scaling_raw` from committed JSON: (a) t_a sigma^(K-2) per seed (constant:
    the first-escape identity) for the quadtree (regime_analysis.json, plain runs) and the snake MPS (fix1 MPS
    reruns); (b) MPS edges 4 and 3: rise time after the first drop vs ln(1/sigma) per seed with the
    relative-residual log-law fit; (c) MPS first rise (edge 5) relative to the first drop's rate peak, x =
    already above threshold at tau = 0."""
    fx = json.load(open(fix1_json))
    mr = fx.get("mps_reruns")
    reg = [r for r in json.load(open(regime_json)) if r["model"] == "plain"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH))
    q_col, m_col = TREE_COLOR["binary, quadtree"], TREE_COLOR["MPS, snake"]
    for seed in sorted({r["seed"] for r in reg}):
        rs = sorted((r for r in reg if r["seed"] == seed), key=lambda r: r["sigma"])
        s = np.array([r["sigma"] for r in rs])
        axes[0].loglog(
            s,
            [r["t_a"] * r["sigma"] ** 13 for r in rs],
            color=q_col,
            lw=0.7,
            marker="o",
            ms=2,
            label="binary quadtree ($K-2=13$)" if seed == 0 else None,
        )
    if mr:
        for seed in sorted({r["seed"] for r in mr["runs"]}):
            rs = [r for r in mr["runs"] if r["seed"] == seed]
            s = np.array([r["sigma"] for r in rs])
            axes[0].loglog(
                s,
                [r["t_a"] * r["sigma"] ** 14 for r in rs],
                color=m_col,
                lw=0.7,
                marker="s",
                ms=2,
                label="MPS snake ($K-2=14$)" if seed == 0 else None,
            )
        fits = {e["edge"]: {f["seed"]: f for f in e["per_seed"]} for e in mr["edge_fits"]}
        for e, ls in ((4, "-"), (3, "--")):
            for seed in sorted({r["seed"] for r in mr["runs"]}):
                rs = [
                    r
                    for r in mr["runs"]
                    if r["seed"] == seed and r["rise"][e] is not None and r["rise"][e] > r["drop0"]
                ]
                x = np.log(1 / np.array([r["sigma"] for r in rs]))
                y = np.array([r["rise"][e] - r["drop0"] for r in rs])
                axes[1].plot(x, y, ls="", marker="o" if e == 4 else "^", ms=2.5, color=m_col)
                f = fits.get(e, {}).get(seed)
                if f:
                    xs = np.linspace(x.min(), x.max(), 2)
                    # intercept of the relative-residual fit given its slope (zero a-gradient at the joint optimum)
                    a = np.sum(1 / y - f["log_slope"] * x / y**2) / np.sum(1 / y**2)
                    axes[1].plot(xs, a + f["log_slope"] * xs, ls=ls, lw=0.6, color=m_col)
            axes[1].plot(
                [], [], ls=ls, marker="o" if e == 4 else "^", ms=2.5, color=m_col, label=f"edge {e}"
            )
        rng = np.random.default_rng(4)
        for r in mr["runs"]:
            x = r["sigma"] * np.exp(rng.uniform(-0.05, 0.05))
            if r["first_at_start"]:
                axes[2].scatter(x, -r["drop0"], marker="x", s=10, color=MUTED, lw=0.6)
            else:
                axes[2].scatter(x, r["rise"][r["first"][0]] - r["drop0"], s=8, color=m_col, lw=0)
        axes[2].axhline(0, color=MUTED, lw=0.6, ls="--")
        axes[2].set_xscale("log")
        _sigma_ticks(axes[2])
    axes[0].set_ylabel(r"$t_a\,\sigma^{K-2}$ (per seed)")
    axes[0].set_xlabel(r"$\sigma$")
    axes[0].legend(fontsize=5.5, frameon=False)
    _sigma_ticks(axes[0])
    axes[1].set_xlabel(r"$\ln(1/\sigma)$")
    axes[1].set_ylabel(r"rise after the first drop ($\tau$)")
    axes[1].legend(fontsize=6, frameon=False)
    axes[1].set_title("MPS later routes", fontsize=8)
    axes[2].set_xlabel(r"$\sigma$")
    axes[2].set_ylabel("first rise - first drop peak")
    axes[2].set_title(r"MPS first rise (x: at $\tau = 0$)", fontsize=8)
    fig.tight_layout()
    savefig(fig, Path(out) / "scaling_fix1")
    plt.close(fig)


def fig_order_test(order_json, out):
    """Pre-registered order test (order_test.json): per order, the final loss (left) and log10 of the time to
    L <= 0.25 (right, censored at the horizon) against the data-only predictor S; points = mean over seeds 0-2,
    bars = range over seeds; structured orders labelled; primary Spearman results in the titles."""
    res = json.load(open(order_json))
    t, groups = res["table"], res["orders"]
    cols = {"new_random": palette(4)[1], "structured": palette(4)[0], "old_random": MUTED}
    names = {
        "new_random": "new random orders",
        "structured": "structured orders",
        "old_random": "main-sweep scrambled (secondary)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.4 * TEXT_WIDTH), sharex=True)
    for ax, key in ((axes[0], "O1"), (axes[1], "O2")):
        for g, os_ in groups.items():
            for i, o in enumerate(os_):
                v = t[o]
                if key == "O1":
                    ys = [x["loss_end"] for x in v["per_seed"]]
                else:
                    ys = [
                        np.log10(x["tau_025"])
                        if x["tau_025"] is not None and np.isfinite(x["tau_025"])
                        else 4.0
                        for x in v["per_seed"]
                    ]
                ax.errorbar(
                    v["S"],
                    v[key],
                    yerr=[[v[key] - min(ys)], [max(ys) - v[key]]],
                    fmt="o",
                    ms=3.5,
                    color=cols[g],
                    lw=0.6,
                    capsize=0,
                    label=names[g] if i == 0 else None,
                    zorder=3,
                )
                if g == "structured":
                    # labels placed by hand in empty parts of the panel (orders with the same
                    # pixel pairs share S and would overlap), with a leader line to the point
                    lx, ly = ORDER_LABEL_POS[key].get(o, (v["S"] + 0.03, v[key]))
                    ax.annotate(
                        o,
                        (v["S"], v[key]),
                        xytext=(lx, ly),
                        textcoords="data",
                        fontsize=8,
                        color=INK_SECONDARY,
                        va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2),
                    )
        r = res["primary"]["all"][key]
        lo, hi = r["ci95"]
        ax.set_title(
            rf"all 27 orders: Spearman $r_S={r['rho']:.2f}$ $[{lo:.2f}, {hi:.2f}]$",
            fontsize=8,
        )
        ax.set_xlabel(r"predictor $S$ (share on $k\leq5$ routes, best rank-1 fit)")
        ax.set_xlim(-0.05, 1.3)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel(r"train loss at $\tau=10^4$ (mean of 3 seeds)")
    axes[1].set_ylabel(r"$\log_{10}\tau(L\leq0.25)$ (mean; censored at 4)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    savefig(fig, Path(out) / "order_test")
    plt.close(fig)


# Label positions (data coordinates) of the structured orders in fig_order_test, per outcome.
ORDER_LABEL_POS = {
    "O1": {
        "quadtree": (1.03, 0.104),
        "snake": (1.03, 0.119),
        "rowmajor": (1.03, 0.134),
        "colmajor": (0.40, 0.262),
        "quadtreeT": (0.47, 0.105),
        "hilbert": (0.065, 0.147),
        "mortonrev": (0.53, 0.236),
    },
    "O2": {
        "quadtree": (1.03, 2.25),
        "rowmajor": (1.03, 1.88),
        "snake": (1.03, 1.55),
        "colmajor": (0.40, 2.55),
        "quadtreeT": (0.40, 2.05),
        "hilbert": (0.05, 1.30),
        "mortonrev": (0.56, 2.95),
    },
}


def json_figures(results_dir, out_dir):
    """Figures that need only committed JSON files (no raw trajectories)."""
    from analysis import ERR_MAX

    R = Path(results_dir)
    set_style()
    fig_routes(R / "routes_raw.json", out_dir, err_max=ERR_MAX)
    fig_hessians(R / "hessians_raw.json", out_dir)
    fig_regime(R / "regime_analysis.json", out_dir)
    if (R / "fix1_analysis.json").exists():
        fig_first_route_clean(R / "fix1_analysis.json", out_dir)
        fig_locality_orders(R / "fix1_analysis.json", out_dir)
        fig_scaling_fix1(R / "fix1_analysis.json", R / "regime_analysis.json", out_dir)
    if (R / "order_test.json").exists():
        fig_order_test(R / "order_test.json", out_dir)


def main(
    summary_json, runs_dir, out_dir, feats="raw", routes_json=None, hess_json=None, regime_json=None
):
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
    if sys.argv[1] == "--json":
        json_figures(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
