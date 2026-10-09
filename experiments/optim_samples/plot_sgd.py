"""Figures for step 2 (online SGD): plateau times vs batch size and step size, theory overlays.

Reads results/sgd_summary.csv (analyze_sgd.py) and results/sgd_stability.csv (sgd_stability.py).
Run: uv run python experiments/optim_samples/plot_sgd.py -> results/fig_sgd.png
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps

from common import RESULTS as RES  # noqa: E402


def pooled(g, col):
    """Mean over init seeds of a per-init quantity, with its standard error."""
    m = g.groupby("x")[col]
    return m.mean(), m.sem()


def main():
    ps.apply()
    s = pd.read_csv(RES / "sgd_summary.csv")
    s["x"] = s.eta / s.B
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    # (a) relative shift of the mean event time vs B (fan-in init, eta = 5e-4)
    ax = axes[0, 0]
    b = s[(s.block == "B_sweep")]
    for i, sig in enumerate(sorted(b.sigma.unique())):
        g = b[b.sigma == sig].assign(rel1=lambda d: d.shift1 / d.t1_gf,
                                     rel2=lambda d: d.shift2 / d.t2_gf)
        for r, mk in ((1, "o"), (2, "s")):
            m = g.groupby("B")[f"rel{r}"]
            ax.errorbar(m.mean().index, m.mean(), yerr=m.sem(), fmt=mk + "-", ms=3.5, lw=1,
                        color=ps.BLUES[2 * i + 1], mfc="white" if r == 2 else None,
                        label=f"σ={sig:g}, event {r}")
    ax.axhline(0, color=ps.INK2, lw=0.6)
    ax.set(xscale="log", xlabel="batch size B (η = 5·10⁻⁴)",
           ylabel="(E t^SGD − t^GF) / t^GF  (mean ± s.e. over 5 inits)",
           title="(a) noise does not shorten plateaus")
    ax.legend(fontsize=6.5, ncol=2)
    # (b) spread over noise vs eta / B, all fan-in blocks, normalized by sqrt(t_gf)
    ax = axes[0, 1]
    f = s[s.init == "fanin"].copy()
    for i, (blk, lab) in enumerate([("B_sweep", "B sweep, η = 5·10⁻⁴"),
                                   ("eta_sweep_B4", "η sweep, B = 4"),
                                   ("eta_sweep_B64", "η sweep, B = 64")]):
        g = f[(f.block == blk) & (f.sigma == 0.1)].assign(y=lambda d: d.t1_std / np.sqrt(d.t1_gf))
        m, se = pooled(g, "y")
        ax.errorbar(m.index, m, yerr=se, fmt="o", ms=3.5, color=ps.CAT[i], lw=1, label=lab)
    g = f[f.sigma == 0.1].assign(p=lambda d: d.pred_std1 / np.sqrt(d.t1_gf))
    pm = g.groupby("x").p.mean()
    ax.plot(pm.index, pm, "-", color=ps.MUTED, lw=1, label="1D prediction ∝ (η/B)^½")
    ax.set(xscale="log", yscale="log", xlabel="η / B", ylabel="std(t₁) / √t₁^GF  (σ = 0.1)",
           title="(b) spread of the first escape time")
    ax.legend(fontsize=6.5)
    # (c) aligned init: exact 1D theory for mean shift and std
    ax = axes[1, 0]
    a = s[s.block == "aligned"].sort_values("B")
    if not a.empty:
        for r, col in ((1, ps.CAT[0]), (2, ps.CAT[1])):
            ax.errorbar(a.B, a[f"shift{r}"], yerr=a[f"shift{r}_se"], fmt="o", ms=3.5, color=col,
                        lw=1, label=f"mean shift, event {r}")
            ax.plot(a.B, a[f"pred_shift{r}"], "-", color=col, lw=1)
            ax.plot(a.B, a[f"t{r}_std"], "s", ms=3.5, color=col, mfc="white",
                    label=f"std, event {r}")
            ax.plot(a.B, a[f"pred_std{r}"], "--", color=col, lw=1)
        ax.set(xscale="log", yscale="symlog", xlabel="batch size B (η = 5·10⁻⁴)",
               ylabel="time units", title="(c) aligned init: data vs exact 1D theory (lines)")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=6.5)
    # (d) imbalance drift and the stability limit
    ax = axes[1, 1]
    for i, (blk, lab) in enumerate([("B_sweep", "B sweep"), ("eta_sweep_B4", "η sweep, B = 4"),
                                   ("eta_sweep_B64", "η sweep, B = 64")]):
        g = f[(f.block == blk) & (f.sigma == 0.1)]
        m, se = pooled(g, "trq_change")
        ax.errorbar(m.index, m, yerr=se, fmt="o", ms=3.5, color=ps.CAT[i], lw=1, label=lab)
    xs = np.geomspace(f.x.min(), f.x.max(), 10)
    m_all = f[f.sigma == 0.1].groupby("x").trq_change.mean()
    c = np.exp(np.mean(np.log(m_all.values / m_all.index.values)))
    ax.plot(xs, c * xs, "-", color=ps.MUTED, lw=1, label="slope 1")
    ax.set(xscale="log", yscale="log", xlabel="η / B",
           ylabel="mean |Δ tr Q_e| at the end (σ = 0.1)", title="(d) noise drives the gauge balance")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / "fig_sgd.png")
    plt.close(fig)

    if not (RES / "sgd_stability.csv").exists():
        return
    st = pd.read_csv(RES / "sgd_stability.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, (tree, g) in enumerate(st[st.B < 1e8].groupby("tree")):
        ax.plot(g.B, g.eta_ms, "o-", ms=3.5, color=ps.CAT[i], label=f"{tree} N=4: exact mean-square")
        ax.plot(g.B, g.eta_1d, "--", color=ps.CAT[i], lw=1, label=f"{tree}: top-mode estimate")
        ax.axhline(g.eta_gd.iloc[0], color=ps.CAT[i], lw=0.7, ls=":")
    ax.set(xscale="log", yscale="log", xlabel="batch size B", ylabel="stability limit η",
           title="SGD stability limit at the balanced minimum")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(RES / "fig_sgd_stability.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
