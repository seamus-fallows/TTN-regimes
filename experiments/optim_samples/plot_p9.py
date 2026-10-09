"""Figure and fit for the P9 test (results/p9_sgd.csv from p9_sgd.py).

For every escaped run, invert T_SGD/T_GF = -ln(1-x)/x for x and report c_gamma = x / ((eta/B) T_GF);
trapped runs (no escape by 4 T_GF or t = 1600) give the bound c_gamma >= 1/((eta/B) T_GF) if the law holds.
Run: uv run python experiments/optim_samples/plot_p9.py -> results/fig_p9.png, p9_fit.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import plot_style as ps

RES = Path(__file__).parent / "results"


def law(x):
    return -np.log1p(-x) / x


def invert(ratio):
    if not np.isfinite(ratio) or ratio <= 1:
        return np.nan
    return brentq(lambda x: law(x) - ratio, 1e-9, 1 - 1e-12)


def main():
    ps.apply()
    df = pd.read_csv(RES / "p9_sgd.csv")
    df["z"] = df.eta_over_B * df.t2_gf_run  # (eta/B) T_GF
    df["x_fit"] = [invert(r) for r in df.ratio]
    df["c_gamma"] = df.x_fit / df.z
    df.to_csv(RES / "p9_fit.csv", index=False)
    g = df.groupby(["b0", "B", "eta_over_B"]).agg(z=("z", "first"), ratio=("ratio", "mean"),
                                                   ratio_se=("ratio", "sem"),
                                                   escaped=("escaped", "mean"),
                                                   c_gamma=("c_gamma", "mean"),
                                                   min_a1=("min_a1", "min")).reset_index()
    print(g.to_string(float_format=lambda v: f"{v:.4g}"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    zz = np.geomspace(1e-3, 0.06, 200)
    for c, ls in ((26, ":"), (35, "-"), (45, "--")):
        x = c * zz
        ok = x < 1
        ax.plot(zz[ok], law(x[ok]), ls, color=ps.MUTED, lw=1, label=f"−ln(1−x)/x, c_γ = {c}")
    for i, b0 in enumerate(sorted(g.b0.unique(), reverse=True)):
        s = g[(g.b0 == b0)]
        esc = s[s.escaped == 1]
        ax.errorbar(esc.z, esc.ratio, yerr=esc.ratio_se, fmt="o", ms=4, color=ps.CAT[i], lw=1,
                    label=f"b₀ = {b0:g} (T_GF = {df[df.b0 == b0].t2_gf_run.iloc[0]:.0f})")
        trap = s[s.escaped < 1]
        ax.plot(trap.z, np.full(len(trap), 4.6), "v", ms=6, color=ps.CAT[i], mfc="white")
    ax.text(0.03, 4.95, "▽ trapped: no escape by 4 T_GF", fontsize=6.5, color=ps.INK2, ha="center")
    ax.set(xscale="log", ylim=(0.95, 5.2), xlabel="(η/B) · T_GF",
           ylabel="T_SGD / T_GF (second plateau)", title="(a) P9: lengthening and trapping")
    ax.legend(fontsize=6.3, loc="center left")
    ax = axes[1]
    for i, b0 in enumerate(sorted(df.b0.unique(), reverse=True)):
        s = df[df.b0 == b0]
        ax.scatter(s.eta_over_B * (1 + 0.04 * (s.B == 16)), s.min_a1, s=14, color=ps.CAT[i],
                   label=f"b₀ = {b0:g} (run length up to {s.t_max.iloc[0]:.0f})")
    ax.axhline(0.5, color=ps.CAT[7], lw=0.8)
    ax.set(xscale="log", xlabel="η / B", ylabel="min over the run of a₁",
           title="(b) learned component 1 under noise (P10)")
    ax.legend(fontsize=6.3, loc="lower left")
    fig.tight_layout()
    fig.savefig(RES / "fig_p9.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
