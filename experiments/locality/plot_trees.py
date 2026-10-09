"""Step 3 figure: one teacher, learned by MPS, binary and HT students through different saddle sequences.

Columns: trees. Rows: (1) loss of one run vs time since the first event (symlog), with the predicted
saddle levels L_j (route rule at rank-constrained optima) as dashed lines labelled by k*_j; (2) effective
bond rank of every edge for that run; (3) time spent at each predicted level vs 1/sigma (median over
seeds, band = range) with the predicted law (k* = 2: log; k* >= 3: slope k* - 2).

usage: uv run python experiments/locality/plot_trees.py <prefix> [...] --teacher B4 --n 4 --sigma 0.03
       --levels experiments/locality/results/levels_runs.csv --out experiments/locality/results/trees_B4.png
"""

from __future__ import annotations

import argparse
import json
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
from ttn.teachers import pattern_teacher  # noqa: E402

TREES = ["mps", "binary", "ht"]
LEVEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]


def law(k):
    return {1: "no plateau", 2: "log"}.get(k, f"σ^-{k - 2}")


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
    recs = [r for r in load_records(args.inputs) if r.get("name") == args.teacher
            and r.get("status") != "timeout" and r.get("traj") and r["start"] == args.start]
    seqs = json.loads(SEQ_FILE.read_text())
    lev = pd.read_csv(args.levels)
    lev = lev[(lev["case"] == args.teacher) & (lev["start"] == args.start)]
    ncol = len(args.trees)
    fig, axes = plt.subplots(3, ncol, figsize=(4.3 * ncol, 10), squeeze=False)
    for col, kind in enumerate(args.trees):
        tree = get_tree(kind, args.n)
        cands = [r for r in recs if r["kind"] == kind and r["seed"] == args.seed]
        if not cands:
            continue
        rec = min(cands, key=lambda r: abs(np.log(r["sigma"] / args.sigma)))
        seq = seqs[f"{args.teacher}_{tree.name}"]
        t, L = loss_trajectory(rec)
        t0 = rec["events"][0][0] if rec["events"] else 0.0
        x = t - t0
        ax = axes[0, col]
        ax.plot(x, L, color="k", lw=1.4)
        for j, step in enumerate(seq):
            if step["k_star"] == 0:
                continue
            c = LEVEL_COLORS[j % len(LEVEL_COLORS)]
            ax.axhline(step["loss"], color=c, ls="--", lw=0.9)
            ax.text(0.98, step["loss"], f"L{j}: k* = {step['k_star']} ({law(step['k_star'])})", color=c,
                    fontsize=7, ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("log")
        ax.set_ylim(max(L.min(), 1e-4), L.max() * 1.5)
        ax.set_xlim(left=-10)
        ax.set_title(f"{tree.name} (K = {tree.n_nodes}), σ = {rec['sigma']:.3g}, seed {rec['seed']}", fontsize=9)
        ax.set_xlabel("t − t(first event)")
        ax.grid(alpha=0.25, lw=0.5)
        if col == 0:
            ax.set_ylabel("population loss")
        ax = axes[1, col]
        spec, ts, spectra = spectra_trajectory(rec)
        teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec["teacher_seed"])
        tsp = teacher_spectra(spec, teacher)
        cmap = plt.get_cmap("viridis")
        for i, (v, s) in enumerate(spectra.items()):
            nz = tsp[v][tsp[v] > 1e-9 * tsp[v][0]]
            r_eff = (s > 0.1 * nz.min().item()).sum(1) + 0.05 * (i - len(spectra) / 2)
            ax.plot(ts - t0, r_eff, color=cmap(i / max(len(spectra) - 1, 1) * 0.9), lw=1.2,
                    label=f"edge {v} | {''.join(map(str, sorted(tree.subtree_sites[v])))} (teacher {len(nz)})")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(left=-10)
        ax.set_xlabel("t − t(first event)")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        ax.grid(alpha=0.25, lw=0.5)
        if col == 0:
            ax.set_ylabel("effective bond rank")
        ax = axes[2, col]
        g = lev[lev["tree"] == tree.name]
        for j, gj in g.groupby("level_j"):
            k = int(gj["k_star"].iloc[0])
            c = LEVEL_COLORS[int(j) % len(LEVEL_COLORS)]
            med = gj[gj["duration"] > 0].groupby("sigma")["duration"].agg(["median", "min", "max"])
            if med.empty:
                continue
            xs = 1 / med.index.values
            ax.plot(xs, med["median"], "o-", ms=3.5, lw=1.2, color=c, label=f"L{j}: k* = {k}")
            ax.fill_between(xs, med["min"], med["max"], color=c, alpha=0.12, lw=0)
            if k >= 3:
                ref = med["median"].iloc[-1]
                ax.plot(xs, ref * (xs / xs[-1]) ** (k - 2), "--", color=c, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("1/σ")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, frameon=False)
        if col == 0:
            ax.set_ylabel("time at level L_j (dashed: slope k* − 2)")
    fig.suptitle(f"Teacher {args.teacher}: weights {rec['weights']}, patterns {rec['patterns']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
