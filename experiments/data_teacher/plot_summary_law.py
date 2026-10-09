"""Summary figure: duration of the second plateau vs init scale on an MPS-4 student, baseline vs
complications (all dense runs on the library integrator; per-sigma medians over seeds).

Baseline (odeco, whitened): e4a_sigma c = 0 and e2_sigma theta = 0 (aligned covariance).
Complications: non-orthogonal components (e4a_sigma c = 0.2, 0.45), misaligned covariance
(e2_sigma theta = 20 deg), random MPS teachers (e4b_ttn, 5 teachers).
Usage: uv run python experiments/data_teacher/plot_summary_law.py (after analyze_sigma_law.py and
analyze_e4b_order.py). Writes results/summary_plateau_law.png.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import RESULTS  # noqa: E402

HERE = Path(__file__).parent


def main():
    a = pd.read_csv(RESULTS / "e4a_sigma_law.csv")
    b = pd.read_csv(RESULTS / "e2_sigma_law.csv")
    c = pd.read_csv(RESULTS / "e4b_order.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    series = [
        (a[a.overlap == 0.0], "odeco, whitened (baseline)", "k", "o", "-"),
        (b[b.theta_deg == 0], "odeco, aligned covariance", "0.5", "s", "-"),
        (a[a.overlap == 0.2], "non-orthogonal, c = 0.2", "C1", "^", "--"),
        (a[a.overlap == 0.45], "non-orthogonal, c = 0.45", "C3", "v", "--"),
        (b[b.theta_deg == 20], "misaligned covariance, 20 deg", "C0", "D", "--"),
    ]
    for g, lab, col, mk, ls in series:
        m = g[g.dt2 > 0].groupby("sigma")["dt2"].median()
        for ax in axes:
            x = m.index.values if ax is axes[0] else -np.log(m.index.values)
            ax.plot(x, m.values, mk + ls, color=col, label=lab, ms=5)
    for i, (tag, g) in enumerate(c.groupby("teacher")):
        m = g.groupby("sigma")["dt2"].median()
        for ax in axes:
            x = m.index.values if ax is axes[0] else -np.log(m.index.values)
            ax.plot(x, m.values, ".:", color=f"C{4 + i}",
                    label="random MPS teachers" if i == 0 else None)
    xs = np.array([0.03, 0.3])
    axes[0].plot(xs, 30 * (xs / 0.3) ** -2.0, color="k", lw=0.6, alpha=0.5, label="sigma^-2")
    axes[0].set(xscale="log", yscale="log", xlabel="sigma",
                ylabel="second plateau Dt_2 (median over seeds)",
                title="MPS-4: polynomial (baseline) vs logarithmic (complications)")
    axes[1].set(xlabel="log(1/sigma)", ylabel="Dt_2", yscale="log",
                title="same data vs log(1/sigma)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "summary_plateau_law.png", dpi=110)


if __name__ == "__main__":
    main()
