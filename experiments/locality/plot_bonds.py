"""Step 4 figure: bond spectra at the second event for two-component teachers.

For one run per (tree, S): the second singular value s_2^e(t) of every edge (relative to the teacher's
s2 = 0.5), against time since the first event. Predicted rising edges (the edges of the Steiner subtree
T_S) are drawn in colour, all others in grey; the dotted line marks the second event.

The figure is drawn at its printed size (the report's text width) in the report style. Trajectory
files are looked up under $LOCALITY_OUT/traj when the path stored in a run record (the remote
machine's) does not exist.

usage: uv run python experiments/locality/plot_bonds.py <prefix> --cases mps8:25 binary8:34 ht8:12 --sigma 0.05
       --out experiments/locality/results/bond_spectra.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analysis import load_records  # noqa: E402
from bond_events import spectra_trajectory  # noqa: E402
from rules import get_tree, steiner_k  # noqa: E402

from ttn import plotting as tp  # noqa: E402

TREE_NAME = {"mps": "MPS", "binary": "binary", "ht": "HT"}


def steiner_edges(tree, S):
    nodes = tree.steiner_nodes(sorted(S))
    return [v for v in tree.edges if v in nodes and tree.parent[v] in nodes]


def local_traj(rec):
    """Point rec["traj"] at $LOCALITY_OUT/traj/<file> if the recorded (remote) path does not exist."""
    p = Path(rec["traj"])
    alt = Path(os.environ.get("LOCALITY_OUT", "/root/outputs/locality")) / "traj" / p.name
    if not p.exists() and alt.exists():
        rec["traj"] = str(alt)
    return rec


def site_set(sites) -> str:
    return "$\\{" + ",".join(map(str, sorted(sites))) + "\\}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="+")
    ap.add_argument(
        "--cases", nargs="+", required=True, help="tree:S, e.g. mps8:25 (saddle start: ht8:12s)"
    )
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [
        local_traj(r)
        for r in load_records(args.prefix)
        if r.get("status") != "timeout" and r.get("traj")
    ]
    tp.set_style()
    fig, axes = plt.subplots(
        1, len(args.cases), figsize=(tp.TEXT_WIDTH, 3.0), squeeze=False, layout="constrained"
    )
    for ax, spec_str in zip(axes[0], args.cases):
        tname, S = spec_str.split(":")
        start = "saddle" if S.endswith("s") else "small"
        S = [int(c) for c in S.rstrip("s")]
        cands = [
            r
            for r in recs
            if get_tree(r["kind"], r.get("n")).name == tname
            and r.get("S") == S
            and r["start"] == start
            and len(r.get("events", [])) == 2
        ]
        seeds = sorted({r["seed"] for r in cands}, key=lambda x: (x < args.seed, x))
        # first seed >= --seed that escaped
        cands = [r for r in cands if seeds and r["seed"] == seeds[0]]
        if not cands:
            ax.set_title(f"{tname} S={S}: no run")
            continue
        rec = min(cands, key=lambda r: abs(np.log(r["sigma"] / args.sigma)))
        tree = get_tree(rec["kind"], rec.get("n"))
        spec, t, spectra = spectra_trajectory(rec)
        (t1, _), (t2, _) = rec["events"]
        rising = steiner_edges(tree, S)
        cmap = plt.get_cmap("viridis")
        # other edges dashed and on top: where one coincides with a Steiner edge, both stay visible
        for v in tree.edges:
            if v not in rising:
                y = spectra[v][:, 1] / rec["weights"][1]
                ax.plot(t - t1, y, color="0.6", lw=0.8, ls="--", zorder=4)
        for i, v in enumerate(rising):
            ax.plot(
                t - t1,
                spectra[v][:, 1] / rec["weights"][1],
                color=cmap(i / max(len(rising) - 1, 1) * 0.85),
                lw=1.4,
                zorder=3,
                label=site_set(tree.subtree_sites[v]),
            )
        ax.plot([], [], color="0.6", ls="--", lw=0.8, label="other edges")
        ax.axvline(t2 - t1, color="k", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_ylim(1e-14, 3)
        ax.set_xlim(left=-1)
        ax.set_yticks([1e-12, 1e-8, 1e-4, 1.0])
        emax = int(np.floor(np.log10(ax.get_xlim()[1])))
        exps = list(range(0, emax + 1, max(1, int(np.ceil((emax + 1) / 3)))))
        ax.set_xticks([0.0] + [10.0**e for e in exps], ["$0$"] + [f"$10^{e}$" for e in exps])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        k = steiner_k(tree, S)
        name = f"{TREE_NAME[rec['kind']]}{tree.n_sites}"
        how = "saddle start" if start == "saddle" else "small init."
        ax.set_title(
            f"{name}, $D=${site_set(S)}, $k={k}$\n"
            f"$\\sigma={rec['sigma']:.3g}$, {how}, seed {rec['seed']}",
            fontsize=8,
        )
        ax.set_xlabel("$t-t_1$")
        ax.legend(
            title="Steiner edges (sites below)",
            title_fontsize=8,
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            ncol=2,
            handlelength=1.5,
            columnspacing=0.8,
        )
    axes[0][0].set_ylabel(r"$s^e_2/s_2$")
    fig.savefig(args.out, dpi=300)


if __name__ == "__main__":
    main()
