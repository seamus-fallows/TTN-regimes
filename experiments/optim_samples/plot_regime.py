"""The learning-rate axis of the regime map (binary N=4, K=3, odeco [3, 2, 1]).

For every init scale sigma (plain model for sigma <= 0.3, centred model for sigma >= 0.3) and
absolute GD step eta: the majority outcome over 5 seeds (results/gd_sweep.csv), with the predicted
stability limits 2/lambda_bal (balanced minimum), 2/lambda_GF(sigma) (sharpness of the minimum the
gradient flow reaches, measured) and the SGD mean-square limits for several batch sizes.

Run: uv run python experiments/optim_samples/plot_regime.py -> results/fig_lr_regime_map.png
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps

from common import RESULTS as RES  # noqa: E402
CLASSES = [("gf_like", "GF-like (monotone, GF's final loss)"),
           ("catapult", "catapult (loss spike, then GF's loss)"),
           ("oscillating", "bounded oscillation"), ("reentry", "thrown back to origin"),
           ("other", "other final loss"), ("diverged", "diverges")]


def main():
    ps.apply()
    df = pd.read_csv(RES / "gd_sweep.csv")
    df = df[(df.tree == "binary") & (df.n == 4) & df.root_scale.isna() & (df.seed <= 4)]
    lam_bal = 3 * 3.0 ** (4 / 3)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {c: ps.CAT[i] for i, (c, _) in enumerate(CLASSES)}
    for centred, mk in ((False, "o"), (True, "s")):
        sub = df[df.centred == centred]
        maj = sub.groupby(["sigma", "eta"]).cls.agg(lambda c: c.value_counts().index[0])
        frac = sub.groupby(["sigma", "eta"]).cls.agg(lambda c: c.value_counts(normalize=True).iloc[0])
        for (sig, eta), c in maj.items():
            ax.scatter(sig * (1.04 if centred else 0.96), eta, marker=mk, s=22,
                       color=colors[c], alpha=0.35 + 0.65 * frac[(sig, eta)], lw=0)
    ref = df.groupby("sigma").agg(gf=("sharp_gf_rel", "mean"))
    ax.plot(ref.index, 2 / (ref.gf * lam_bal), "-", color=ps.INK, lw=1,
            label="2/λ_GF(σ): sharpness of the GF state (measured)")
    lab_df = df[(df.centred & (df.sigma >= 0.3)) | (~df.centred & (df.sigma < 0.3))]
    ranks = lab_df.groupby("sigma").eff_ranks_gf.agg(
        lambda r: sorted({int(str(v).strip("[]").split(",")[0]) for v in r}))
    for sig, rk in ranks.items():
        txt = str(rk[0]) if len(rk) == 1 else f"{rk[0]}-{rk[-1]}"
        ax.text(sig, 0.6, txt, fontsize=6.5, color=ps.INK2, ha="center",
                transform=ax.get_xaxis_transform())
    ax.text(df.sigma.min() * 0.75, 0.6, "GF bond rank:", fontsize=6.5, color=ps.INK2, ha="right",
            transform=ax.get_xaxis_transform())
    sig_all = np.geomspace(df.sigma.min() * 0.8, df.sigma.max() * 1.25, 10)
    ax.plot(sig_all, np.full_like(sig_all, 2 / lam_bal), ":", color=ps.CAT[7], lw=1,
            label="η_max = 2/(K s₁^{2−2/K})")
    st = RES / "sgd_stability.csv"
    if st.exists():
        s = pd.read_csv(st)
        s = s[(s.tree == "binary") & s.B.isin([1, 16, 256])]
        for _, r in s.iterrows():
            ax.axhline(r.eta_ms, color=ps.MUTED, lw=0.7, ls="--")
            ax.text(sig_all[-1], r.eta_ms, f" SGD, B={int(r.B)}", fontsize=6.5, color=ps.INK2,
                    va="center", ha="left")
    for c, lab in CLASSES:
        ax.scatter([], [], color=colors[c], s=22, label=lab)
    ax.scatter([], [], marker="o", color=ps.INK2, s=22, label="plain model")
    ax.scatter([], [], marker="s", color=ps.INK2, s=22, label="centred model")
    ax.set_xlim(df.sigma.min() * 0.35, df.sigma.max() * 3)
    ax.set(xscale="log", yscale="log", xlabel="init scale σ (per core; plain ≤ 0.3, centred ≥ 0.3)",
           ylabel="GD step η",
           title="Learning-rate axis of the regime map (binary N=4, 5 seeds)")
    ax.legend(fontsize=6.3, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(RES / "fig_lr_regime_map.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
