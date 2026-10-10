"""Figure (a): aligned balanced init, TTN gradient flow (solid) vs the exact mode solutions (dashed).

    uv run python experiments/s2s_core/plot_aligned.py
Reads /root/outputs/s2s_core/aligned/<tree>.npz (dopri5 runs of aligned.py) and results/aligned.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttn import plotting as tp  # noqa: E402

HERE = Path(__file__).parent
RAW = Path("/root/outputs/s2s_core/aligned")
TREES = ["binary4", "mps4", "binary8", "ht4", "mps8"]
NAMES = {"binary4": "binary4", "mps4": "MPS4", "binary8": "binary8", "ht4": "HT4", "mps8": "MPS8"}


def sci(x: float) -> str:
    """x as a mathtext power of ten with one significant digit, e.g. 2\\cdot10^{-10}."""
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\cdot10^{{{int(e)}}}$"


def main():
    import argparse

    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument(
        "--out", required=True, help="output directory (e.g. experiments/s2s_core/results)"
    )
    ap.add_argument("--raw", default=str(RAW), help="directory with <tree>_dopri5.npz (aligned.py)")
    args = ap.parse_args()
    tp.set_style()
    table = pd.read_csv(Path(args.results) / "aligned.csv")
    fig, axes = plt.subplots(
        1, len(TREES), figsize=(tp.TEXT_WIDTH, 2.25), sharey=True, layout="constrained"
    )
    colors = tp.palette(3)
    for ax, tree in zip(axes, TREES):
        d = np.load(Path(args.raw) / f"{tree}_dopri5.npz")
        t = d["t"]
        for r in range(3):
            ax.plot(t, d["overlaps"][:, r], color=colors[r], lw=1.4)
            ax.plot(t, d["pred_overlaps"][:, r], color="k", lw=0.7, ls=(0, (3, 3)))
        row = table[(table.tree == tree) & (table.method == "dopri5")].iloc[0]
        ax.set_title(
            f"{NAMES[tree]}, $K={row.K}$\nmax dev {sci(row.max_abs_dev_overlap)}", fontsize=8
        )
        ax.set_xlabel("$t$")
        ax.set_xlim(0, t[-1])
    axes[0].set_ylabel("$a_r(t)$")
    handles = [
        plt.Line2D([], [], color=colors[r], lw=1.4, label=f"$a_{r + 1}$ (gradient flow)")
        for r in range(3)
    ]
    handles.append(plt.Line2D([], [], color="k", lw=0.7, ls=(0, (3, 3)), label="exact"))
    fig.legend(handles=handles, loc="outside lower center", ncols=4, fontsize=8)
    tp.savefig(fig, Path(args.out) / "fig_a_aligned")


if __name__ == "__main__":
    main()
