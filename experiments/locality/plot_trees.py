"""Step 3 figure: one teacher, learned by MPS, binary and HT students through different saddle sequences.

Columns: trees. Rows: (1) loss of one run vs time since the first event (symlog), with the predicted
saddle levels L_j (route rule at rank-constrained optima) as dashed lines labelled by k*_j; (2) effective
bond rank of every edge for that run; (3) time spent at each predicted level vs 1/sigma (median over
seeds, band = range) with the predicted law (k* = 2: log; k* >= 3: slope k* - 2).

The figure is drawn at its printed size (0.8 of the report's text width) in the report style.
Trajectory files are looked up under $LOCALITY_OUT/traj when the path stored in a run record (the
remote machine's) does not exist.

usage: uv run python experiments/locality/plot_trees.py <prefix> [...] --teacher B4 --n 4 --sigma 0.03
       --levels experiments/locality/results/levels_runs.csv --out experiments/locality/results/trees_B4.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from analysis import load_records  # noqa: E402
from bond_events import spectra_trajectory, teacher_spectra  # noqa: E402
from levels import SEQ_FILE, loss_trajectory  # noqa: E402
from rules import get_tree  # noqa: E402

from ttn import plotting as tp  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

TREES = ["mps", "binary", "ht"]
TREE_NAME = {"mps": "MPS", "binary": "binary", "ht": "HT"}
LEVEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]
FIG_WIDTH = 0.8 * tp.TEXT_WIDTH  # the report includes this figure at 0.8 of the text width


def law(k):
    return {1: "no plateau", 2: "log"}.get(k, rf"$\sigma^{{-{k - 2}}}$")


def local_traj(rec):
    """Point rec["traj"] at $LOCALITY_OUT/traj/<file> if the recorded (remote) path does not exist."""
    p = Path(rec["traj"])
    alt = Path(os.environ.get("LOCALITY_OUT", "/root/outputs/locality")) / "traj" / p.name
    if not p.exists() and alt.exists():
        rec["traj"] = str(alt)
    return rec


def symlog_ticks(ax, xmax):
    """At most five labelled ticks on a symlog time axis: -10, 0 and decades up to xmax."""
    emax = int(np.floor(np.log10(max(xmax, 1.0))))
    step = max(1, int(np.ceil((emax + 1) / 3)))
    exps = list(range(0, emax + 1, step))
    ax.set_xticks(
        [-10.0, 0.0] + [10.0**e for e in exps], ["$-10$", "$0$"] + [f"$10^{e}$" for e in exps]
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--trees", nargs="+", default=TREES)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="small")
    ap.add_argument("--levels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [
        local_traj(r)
        for r in load_records(args.inputs)
        if r.get("name") == args.teacher
        and r.get("status") != "timeout"
        and r.get("traj")
        and r["start"] == args.start
    ]
    seqs = json.loads(SEQ_FILE.read_text())
    lev = pd.read_csv(args.levels)
    lev = lev[(lev["case"] == args.teacher) & (lev["start"] == args.start)]
    ncol = len(args.trees)
    tp.set_style()
    fig, axes = plt.subplots(3, ncol, figsize=(FIG_WIDTH, 5.8), squeeze=False, layout="constrained")
    for icol, kind in enumerate(args.trees):
        tree = get_tree(kind, args.n)
        cands = [r for r in recs if r["kind"] == kind and r["seed"] == args.seed]
        if not cands:
            continue
        rec = min(cands, key=lambda r: abs(np.log(r["sigma"] / args.sigma)))
        seq = seqs[f"{args.teacher}_{tree.name}"]
        t, L = loss_trajectory(rec)
        t0 = rec["events"][0][0] if rec["events"] else 0.0
        x = t - t0
        ax = axes[0, icol]
        ax.plot(x, L, color="k", lw=1.2)
        labels = []
        for j, step in enumerate(seq):
            if step["k_star"] == 0:
                continue
            c = LEVEL_COLORS[j % len(LEVEL_COLORS)]
            ax.axhline(step["loss"], color=c, ls="--", lw=0.8)
            labels.append((f"$L_{j}$: $k^*={step['k_star']}$, {law(step['k_star'])}", c))
        # level labels in the empty lower left of the panel (the loss stays above the levels there)
        for i, (text, c) in enumerate(reversed(labels)):
            ax.text(0.04, 0.05 + 0.1 * i, text, color=c, fontsize=8, transform=ax.transAxes)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("log")
        ax.set_ylim(max(L.min(), 1e-4), L.max() * 1.5)
        ax.set_xlim(left=-10)
        symlog_ticks(ax, ax.get_xlim()[1])
        ax.set_title(
            f"{TREE_NAME[kind]}{args.n} ($K={tree.n_nodes}$), $\\sigma={rec['sigma']:.3g}$",
            fontsize=8,
        )
        ax.set_xlabel("$t-t_1$")
        if icol == 0:
            ax.set_ylabel("loss $L(t)$")
        ax = axes[1, icol]
        spec, ts, spectra = spectra_trajectory(rec)
        teacher = pattern_teacher(
            rec["weights"], rec["patterns"], d=rec["d"], seed=rec["teacher_seed"]
        )
        tsp = teacher_spectra(spec, teacher)
        cmap = plt.get_cmap("viridis")
        for i, (v, s) in enumerate(spectra.items()):
            nz = tsp[v][tsp[v] > 1e-9 * tsp[v][0]]
            r_eff = (s > 0.1 * nz.min().item()).sum(1) + 0.05 * (i - len(spectra) / 2)
            ax.plot(ts - t0, r_eff, color=cmap(i / max(len(spectra) - 1, 1) * 0.85), lw=1.0)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(axes[0, icol].get_xlim())
        symlog_ticks(ax, ax.get_xlim()[1])
        ax.set_yticks([0, 1, 2])
        ax.set_xlabel("$t-t_1$")
        if icol == 0:
            ax.set_ylabel("effective bond rank")
        ax = axes[2, icol]
        g = lev[lev["tree"] == tree.name]
        for j, gj in g.groupby("level_j"):
            k = int(gj["k_star"].iloc[0])
            c = LEVEL_COLORS[int(j) % len(LEVEL_COLORS)]
            # detected plateau durations (ttn.diagnostics.find_plateaus, matched to the level); levels
            # too short to be detected (log-time k* = 2) fall back to the time within 2% of the level
            dcol = (
                "detected_duration"
                if gj["detected_duration"].notna().sum() >= 3 * gj["sigma"].nunique() // 2
                else "duration"
            )
            med = gj[gj[dcol] > 0].groupby("sigma")[dcol].agg(["median", "min", "max"])
            if med.empty:
                continue
            xs = 1 / med.index.values
            ax.plot(
                xs,
                med["median"],
                "o-" if dcol == "detected_duration" else "o:",
                ms=2.5,
                lw=1.0,
                color=c,
                label=f"$L_{j}$" + ("" if dcol == "detected_duration" else " (2% band)"),
            )
            ax.fill_between(xs, med["min"], med["max"], color=c, alpha=0.12, lw=0)
            if k >= 3:
                ref = med["median"].iloc[-1]
                ax.plot(xs, ref * (xs / xs[-1]) ** (k - 2), "--", color=c, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$1/\sigma$")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if icol == 0:
            ax.set_ylabel("time at the level")
    # one legend below the panels (in the panels it covered the bands)
    nlev = int(lev["level_j"].max()) + 1 if len(lev) else 0
    handles = [
        plt.Line2D(
            [],
            [],
            color=LEVEL_COLORS[j % len(LEVEL_COLORS)],
            marker="o",
            ms=2.5,
            lw=1.0,
            label=f"level $L_{j}$",
        )
        for j in range(nlev)
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            color=tp.INK_SECONDARY,
            marker="o",
            ms=2.5,
            lw=1.0,
            ls=":",
            label="time within 2% of a level",
        ),
        plt.Line2D([], [], color=tp.INK_SECONDARY, lw=0.8, ls="--", label=r"slope $k^*-2$"),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.savefig(args.out, dpi=300)


if __name__ == "__main__":
    main()
