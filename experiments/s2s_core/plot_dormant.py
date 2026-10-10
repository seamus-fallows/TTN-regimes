"""Figure (g): dormant-component suppression. Left: fraction of runs per class (complete / trapped
/ stalled / unresolved) per tree, chi and init, pooled over sigma (with the per-sigma counts in
results/<tag>_summary.csv). Right: plateau decay rate of the dead route ends divided by sigma^2, per
seed, vs sigma (mps4, chi = 3): constant per seed if the rate is ~ |Q| ~ sigma^2.

    uv run python experiments/s2s_core/plot_dormant.py [--results results] [--out results]
Reads results/main_runs.csv (analyze_main.py with all sweeps) and results/dormant_decay.csv
(analyze_dormant.py).
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
CLASSES = ["complete", "trapped", "stalled", "unresolved"]
TREES = ["binary4", "mps4", "binary8", "ht4", "mps8"]
NAMES = {"binary4": "binary4", "mps4": "MPS4", "binary8": "binary8", "ht4": "HT4", "mps8": "MPS8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument(
        "--out", required=True, help="output directory (e.g. experiments/s2s_core/results)"
    )
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    tp.set_style()
    runs = pd.read_csv(res / "main_runs.csv")
    runs = runs[runs.teacher == "separated"]
    # One sweep per (tree, chi, init): the dedicated K = 7, 8 sweeps when present.
    cells = []
    for (tree, chi, init), g in runs.groupby(["tree", "chi", "init"]):
        sweeps = set(g.sweep)
        pref = [
            s
            for s in (
                ["k78", "large", "balanced"]
                if tree not in ("binary4", "mps4")
                else ["main", "balanced"]
            )
            if s in sweeps
        ]
        if init == "fanin":
            pref = [s for s in pref if s != "balanced"] or ["main"]
        sw = pref[0] if pref else g.sweep.iloc[0]
        g = g[g.sweep == sw]
        frac = g["class"].value_counts(normalize=True)
        cells.append(
            {
                "label": f"{tree} $\\chi$={chi} {'bal.' if init == 'balanced' else 'fan-in'}",
                "tree": tree,
                "chi": chi,
                "init": init,
                "n": len(g),
                "sweep": sw,
                **{c: frac.get(c, 0.0) for c in CLASSES},
            }
        )
    cells = pd.DataFrame(cells)
    cells["order"] = (
        cells.tree.map({t: i for i, t in enumerate(TREES)}) * 100
        + cells.chi * 2
        + (cells.init == "balanced")
    )
    cells = cells.sort_values("order")
    cells.to_csv(out / "dormant_classes.csv", index=False)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(tp.TEXT_WIDTH, 3.2), layout="constrained", width_ratios=[2.0, 1]
    )
    colors = dict(zip(CLASSES, [tp.REGIME_PALETTE[i] for i in (0, 4, 1, 3)]))
    y = np.arange(len(cells))
    left = np.zeros(len(cells))
    for c in CLASSES:
        ax.barh(
            y,
            cells[c],
            left=left,
            color=colors[c],
            label=c,
            height=0.75,
            edgecolor="white",
            linewidth=0.8,
        )
        left += cells[c].values
    ticks = [
        f"{NAMES[r.tree]}, $\\chi={r.chi}$, {'bal.' if r.init == 'balanced' else 'fan-in'} ({r.n} runs)"
        for r in cells.itertuples()
    ]
    ax.set_yticks(y, ticks, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of runs (separated teacher, all $\\sigma$)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=4, fontsize=8, frameon=False)

    dec = pd.read_csv(res / "dormant_decay.csv")
    dec = dec[(dec.tree == "mps4") & (dec.chi == 3) & dec.core.isin([0, 3])]
    teacher_color = {"separated": tp.PALETTE[0], "close": tp.PALETTE[1]}
    for (seed, teacher), g in dec.groupby(["seed", "teacher"]):
        g = g.groupby("sigma").rate_over_sigma2.mean().sort_index()
        ax2.plot(g.index, g.values, marker="o", ms=2.5, lw=0.9, color=teacher_color[teacher])
    for teacher, c in teacher_color.items():
        ax2.plot([], [], marker="o", ms=2.5, lw=0.9, color=c, label=f"{teacher} teacher")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\sigma$")
    ax2.set_ylabel(r"end-slice decay rate / $\sigma^2$")
    ax2.set_title("MPS4, $\\chi=3$, trapped runs\n(one line per seed and teacher)", fontsize=8)
    ax2.legend(loc="center left", fontsize=8, handlelength=1.6)
    tp.savefig(fig, out / "fig_g_dormant")


if __name__ == "__main__":
    main()
