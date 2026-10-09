"""The learning-rate axis of the regime map (binary N=4, K=3, odeco [3, 2, 1]), fix round 1.

Every GD run in the (sigma, eta) plane, coloured by outcome class (analyze_gd.py): the step-1 sweep
(plain model, chi = 3, sigma 0.003-0.3) and the chi = 9 runs of configs/gd_centred_chi9.yaml (plain
small init sigma = 0.1, 0.3; centred model sigma = 1-8, where GF converges and the large-sigma side is
genuinely lazy). Lines: 2/lambda_GF(sigma) from the converged chi = 9 GF references (mean over seeds;
sigma = 1-16), the balanced limit eta_max = 2/(K s_1^{2-2/K}), and the SGD mean-square limits.
Run: uv run python experiments/optim_samples/plot_regime.py -> fig_lr_regime_map.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps
from common import RESULTS as RES

CLASSES = [("gf_like", "GF-like (monotone, GF's final loss)"),
           ("catapult", "catapult (loss spike, then GF's loss)"),
           ("oscillating", "bounded oscillation"), ("reentry", "thrown back to origin"),
           ("other", "other final loss"), ("diverged", "diverges")]


def main():
    ps.apply()
    df = pd.read_csv(RES / "gd_sweep.csv")
    step1 = df[(df.tree == "binary") & (df.n == 4) & df.root_scale.isna() & ~df.centred
               & (df.chi == 3) & (df.ratio_ref == "teacher") & (df.seed <= 4)]
    chi9 = df[(df.chi == 9) & (df.ratio_ref == "gf")]
    lam_bal = 3 * 3.0 ** (4 / 3)
    colors = {c: ps.CAT[i] for i, (c, _) in enumerate(CLASSES)}
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    rng = np.random.default_rng(0)
    for sub, mk in ((step1, "o"), (chi9[~chi9.centred], "D"), (chi9[chi9.centred], "s")):
        for c, _ in CLASSES:
            g = sub[sub.cls == c]
            ax.scatter(g.sigma * np.exp(rng.uniform(-0.06, 0.06, len(g))), g.eta, marker=mk, s=12,
                       color=colors[c], alpha=0.7, lw=0)
    gf = pd.read_csv(RES / "centred_chi9_gf.csv")
    m = gf[gf.centred].groupby("sigma").lam_gf.mean()
    ax.plot(m.index, 2 / m.values, "-", color=ps.INK, lw=1.2, label="2/λ_GF(σ), converged GF (χ = 9)")
    ax.plot(m.index, 3 * 2 / m.values, "--", color=ps.INK, lw=0.8,
            label="3 × 2/λ_GF: end of the catapult window (observed)")
    sig = np.geomspace(0.002, 20, 10)
    ax.plot(sig, np.full_like(sig, 2 / lam_bal), ":", color=ps.CAT[7], lw=1,
            label="η_max = 2/(K s₁^{2−2/K}) (balanced minimum)")
    st = RES / "sgd_stability.csv"
    if st.exists():
        s = pd.read_csv(st)
        s = s[(s.tree == "binary") & s.B.isin([1, 16, 256])]
        for _, r in s.iterrows():
            ax.axhline(r.eta_ms, color=ps.MUTED, lw=0.6, ls="--")
            ax.text(14, r.eta_ms * 1.08, f"SGD limit, B={int(r.B)}", fontsize=6.3, color=ps.INK2,
                    ha="right", va="bottom")
    for c, lab in CLASSES:
        ax.scatter([], [], color=colors[c], s=14, label=lab)
    for mk, lab in (("o", "plain, χ = 3 (step 1)"), ("D", "plain, χ = 9"), ("s", "centred, χ = 9")):
        ax.scatter([], [], marker=mk, color=ps.INK2, s=14, label=lab)
    ax.set(xscale="log", yscale="log", xlim=(0.002, 20), xlabel="init scale σ (per core)",
           ylabel="GD step η", title="Learning-rate axis of the regime map (binary N=4, 5 seeds)")
    ax.legend(fontsize=5.8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(RES / "fig_lr_regime_map.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
