"""Step 3 figure: the same nested teacher learned by MPS, binary and HT students.

Columns: trees. Rows: (1) loss and component overlaps a_r(t) for one run, time measured from the first
learning event (symlog), with the predicted sequence in the title; (2) effective bond rank of every edge
for the same run; (3) plateau length vs 1/sigma for every plateau (median over seeds, band = range),
with the predicted law (k = 2: log, k >= 3: slope k - 2).

usage: uv run python experiments/locality/plot_trees.py <runs.jsonl> --teacher TA --sigma 0.01 --seed 0
       --out experiments/locality/results/trees_TA.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analysis import plateau_table  # noqa: E402
from bond_events import RAW, spectra_trajectory, teacher_spectra  # noqa: E402
from rules import describe_sequence, get_tree, predicted_sequence  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

COMP = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
TREES = ["mps", "binary", "ht"]


def pick_run(recs, kind, sigma, seed):
    cands = [r for r in recs if r["kind"] == kind and r["seed"] == seed and r.get("status") != "timeout"]
    return min(cands, key=lambda r: abs(np.log(r["sigma"] / sigma)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.01])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [json.loads(line) for f in args.files for line in open(f)]
    recs = [r for r in recs if r.get("name") == args.teacher and r.get("n") == args.n]
    df = plateau_table(args.files)
    df = df[df["case"] == args.teacher]
    sigmas = dict(zip(TREES, args.sigma * 3 if len(args.sigma) == 1 else args.sigma))
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5))
    for col, kind in enumerate(TREES):
        rec = pick_run(recs, kind, sigmas[kind], args.seed)
        tree = get_tree(kind, args.n)
        seq = describe_sequence(predicted_sequence(tree, rec["patterns"], rec["weights"]))
        spec, t, spectra = spectra_trajectory(rec, RAW / "pilot3")
        teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec.get("teacher_seed", 1000))
        data = np.load(Path(RAW / "pilot3") / _npz_name(rec))
        ov = np.array([teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y))).numpy()
                       for y in data["y"].T])
        t0 = rec["events"][0][0]
        x = t - t0
        ax = axes[0, col]
        t2 = teacher.norm2().item()
        from ttn.losses import population_loss
        loss = np.array([population_loss(spec, spec.unflatten(torch.from_numpy(y)), teacher,
                                         teacher_norm2=teacher.norm2()).item() for y in data["y"].T])
        ax.plot(x, loss / (0.5 * t2), color="k", lw=1.5, label="L / L(0)")
        for r in range(ov.shape[1]):
            ax.plot(x, ov[:, r], color=COMP[r], lw=1.2, label=f"a{r + 1}")
        for te, r in rec["events"]:
            ax.axvline(te - t0, color=COMP[r], lw=0.6, ls=":")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(left=-30)
        ax.set_title(f"{tree.name}, K = {tree.n_nodes}, σ = {rec['sigma']:.3g}\n{seq}", fontsize=8)
        ax.set_xlabel("t − t(first event)")
        ax.grid(alpha=0.25, lw=0.5)
        if col == 0:
            ax.legend(fontsize=7, frameon=False)
        ax = axes[1, col]
        tsp = teacher_spectra(spec, teacher)
        cmap = plt.get_cmap("viridis")
        for i, (v, s) in enumerate(spectra.items()):
            nz = tsp[v][tsp[v] > 1e-9 * tsp[v][0]]
            theta = 0.1 * nz.min().item()
            r_eff = (s > theta).sum(1) + 0.06 * (i - len(spectra) / 2)
            below = "".join(map(str, sorted(tree.subtree_sites[v])))
            ax.plot(x, r_eff, color=cmap(i / max(len(spectra) - 1, 1)), lw=1.2,
                    label=f"edge {v} (below {below}; teacher rank {len(nz)})")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(left=-30)
        ax.set_ylabel("effective bond rank")
        ax.set_xlabel("t − t(first event)")
        ax.legend(fontsize=6, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
        ax = axes[2, col]
        g = df[(df["kind"] == kind)]
        for (order, j), gj in g.groupby(["order", "j"]):
            if len(gj) < 8:
                continue
            med = gj.groupby("sigma")["gap"].agg(["median", "min", "max"]).reset_index()
            k = int(gj["k_rank"].iloc[0])
            rb = int(gj["before"].iloc[0])
            c = COMP[rb]
            ax.plot(1 / med["sigma"], med["median"], "o-", ms=4, color=c, lw=1.2,
                    label=f"plateau before U{rb + 1} (order {order}): k = {k}")
            ax.fill_between(1 / med["sigma"], med["min"], med["max"], color=c, alpha=0.12, lw=0)
            xs = 1 / med["sigma"].values
            if k >= 3:
                ax.plot(xs, med["median"].iloc[0] * (xs / xs[0]) ** (k - 2), "--", color=c, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("1/σ")
        ax.set_ylabel("plateau length")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(fontsize=6, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)


def _npz_name(rec):
    tree = get_tree(rec["kind"], rec.get("n"))
    tag = rec.get("name") or ("S" + "".join(map(str, rec["S"])))
    tag += "_saddle" if rec.get("start") == "saddle" else ""
    return (f"{tag}_{rec['kind']}{tree.n_sites}_chi{rec['chi']}_d{rec['d']}"
            f"_s{rec['sigma']:.4g}_seed{rec['seed']}.npz")


if __name__ == "__main__":
    main()
