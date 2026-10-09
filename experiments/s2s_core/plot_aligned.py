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


def main():
    tp.set_style()
    table = pd.read_csv(HERE / "results" / "aligned.csv")
    fig, axes = tp.subplots(1, len(TREES), width=tp.TEXT_WIDTH, panel_aspect=1.45, sharey=True)
    colors = tp.palette(3)
    for ax, tree in zip(axes, TREES):
        d = np.load(RAW / f"{tree}.npz")
        t = d["t"]
        for r in range(3):
            ax.plot(t, d["overlaps"][:, r], color=colors[r], lw=1.4, label=f"$a_{r + 1}$ (GF)")
            ax.plot(t, d["pred_overlaps"][:, r], color="k", lw=0.7, ls=(0, (3, 3)),
                    label="exact" if r == 0 else None)
        row = table[(table.tree == tree) & (table.method == "dopri5")].iloc[0]
        ax.set_title(f"{tree}, $K={row.K}$\nmax dev {row.max_abs_dev_overlap:.0e}", fontsize=8)
        ax.set_xlabel("$t$")
        ax.set_xlim(0, t[-1])
    axes[0].set_ylabel("$a_r(t)$")
    axes[0].legend(fontsize=6.5, loc="lower right")
    tp.savefig(fig, HERE / "results" / "fig_a_aligned")


if __name__ == "__main__":
    main()
