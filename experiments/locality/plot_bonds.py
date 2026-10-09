"""Step 4 figure: bond spectra at the second event for two-component teachers.

For one run per (tree, S): the second singular value s_2^e(t) of every edge (relative to the teacher's
s2 = 0.5), against time since the first event. Predicted rising edges (the edges of the Steiner subtree
T_S) are drawn in colour, all others in grey; the dotted line marks the second event.

usage: uv run python experiments/locality/plot_bonds.py <prefix> --cases mps8:25 binary8:34 ht8:12 --sigma 0.05
       --out experiments/locality/results/bond_spectra.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analysis import load_records  # noqa: E402
from bond_events import spectra_trajectory  # noqa: E402
from rules import get_tree, steiner_k  # noqa: E402


def steiner_edges(tree, S):
    nodes = tree.steiner_nodes(sorted(S))
    return [v for v in tree.edges if v in nodes and tree.parent[v] in nodes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="+")
    ap.add_argument("--cases", nargs="+", required=True, help="tree:S, e.g. mps8:25 (saddle start: ht8:12s)")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [r for r in load_records(args.prefix) if r.get("status") != "timeout" and r.get("traj")]
    fig, axes = plt.subplots(1, len(args.cases), figsize=(4.2 * len(args.cases), 3.6), squeeze=False)
    for ax, spec_str in zip(axes[0], args.cases):
        tname, S = spec_str.split(":")
        start = "saddle" if S.endswith("s") else "small"
        S = [int(c) for c in S.rstrip("s")]
        cands = [r for r in recs if get_tree(r["kind"], r.get("n")).name == tname and r.get("S") == S
                 and r["start"] == start and len(r.get("events", [])) == 2]
        seeds = sorted({r["seed"] for r in cands}, key=lambda x: (x < args.seed, x))
        cands = [r for r in cands if seeds and r["seed"] == seeds[0]]  # first seed >= --seed that escaped
        if not cands:
            ax.set_title(f"{tname} S={S}: no run")
            continue
        rec = min(cands, key=lambda r: abs(np.log(r["sigma"] / args.sigma)))
        tree = get_tree(rec["kind"], rec.get("n"))
        spec, t, spectra = spectra_trajectory(rec)
        (t1, _), (t2, _) = rec["events"]
        rising = steiner_edges(tree, S)
        cmap = plt.get_cmap("viridis")
        for i, v in enumerate(rising):
            ax.plot(t - t1, spectra[v][:, 1] / rec["weights"][1], color=cmap(i / max(len(rising) - 1, 1) * 0.85),
                    lw=1.6, label=f"edge {v} (below {''.join(map(str, sorted(tree.subtree_sites[v])))})")
        for v in tree.edges:
            if v not in rising:
                ax.plot(t - t1, spectra[v][:, 1] / rec["weights"][1], color="0.6", lw=0.9, ls="--")
        ax.plot([], [], color="0.6", ls="--", label="other edges")
        ax.axvline(t2 - t1, color="k", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_ylim(1e-14, 3)
        ax.set_xlim(left=-1)
        k = steiner_k(tree, S)
        ax.set_title(f"{tname}, S = {{{','.join(map(str, S))}}}, k = {k}, σ = {rec['sigma']:.3g} ({start}),"
                     f" seed {rec['seed']}",
                     fontsize=8)
        ax.set_xlabel("t − t1")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    axes[0][0].set_ylabel("second bond singular value s₂ᵉ / s2")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
