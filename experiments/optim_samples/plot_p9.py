"""P9 test (fix round 1): one c_gamma for all x, the law's shape against alternatives, and P10.

Reads p9_sgd.csv (p9_sgd.py; independent noise seeds). Per condition (b0, B, eta/B): mean of the
escaped runs' T_SGD/T_GF with its s.e. over noise seeds, and the fraction trapped (no persistent event
by 4 T_GF). With z = (eta/B) T_GF, three laws are fitted by weighted least squares on the escaped
condition means, in the same space (R = T_SGD/T_GF) with the same weights (1/s.e.^2):
  P9      R = -ln(1 - c z)/(c z), trapped for c z >= 1   (db/dt = s b^2 - gamma b)
  pole    R = 1/(1 - c z),        trapped for c z >= 1
  linear  R = 1 + c z / 2         (the small-x limit of P9; never traps)
and each fitted law's trapping prediction is checked against the trapped conditions.
Run: uv run python experiments/optim_samples/plot_p9.py -> fig_p9.png, p9_fit.csv, p9_conditions.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import plot_style as ps
from common import RESULTS as RES

LAWS = {
    "P9": lambda x: np.where(x < 1, -np.log1p(-np.minimum(x, 1 - 1e-12)) / np.maximum(x, 1e-12),
                             np.inf),
    "pole": lambda x: np.where(x < 1, 1 / (1 - np.minimum(x, 1 - 1e-12)), np.inf),
    "linear": lambda x: 1 + x / 2,
}


def main():
    ps.apply()
    df = pd.read_csv(RES / "p9_sgd.csv")
    df["z"] = df.eta_over_B * df.t2_gf_run
    cond = df.groupby(["b0", "B", "eta_over_B"]).agg(
        z=("z", "first"), n=("seed", "size"), escaped=("escaped", "mean"),
        ratio=("ratio", "mean"), ratio_se=("ratio", "sem"), min_a1=("min_a1", "min"),
        eta_over_eta_ms=("eta_over_eta_ms", "first"),
        frac_below_half=("frac_time_a1_below_half", "mean")).reset_index()
    cond.to_csv(RES / "p9_conditions.csv", index=False)
    esc = cond[(cond.escaped == 1) & cond.ratio_se.notna()]
    w = 1 / np.maximum(esc.ratio_se.values, 1e-3) ** 2
    fits = []
    for name, law in LAWS.items():
        def chi2(c, law=law):
            pred = law(c * esc.z.values)
            return np.sum(w * (esc.ratio.values - np.where(np.isfinite(pred), pred, 1e6)) ** 2)

        res = minimize_scalar(chi2, bounds=(1, 200), method="bounded")
        c = res.x
        trapped = cond[cond.escaped < 1]
        pred_trap = (c * trapped.z.values >= 1) if name != "linear" else np.zeros(len(trapped), bool)
        pred_esc_ok = (c * esc.z.values < 1) if name != "linear" else np.ones(len(esc), bool)
        fits.append(dict(law=name, c_gamma=c, chi2=res.fun, dof=len(esc) - 1,
                         trapped_conditions=len(trapped),
                         trapped_predicted=int(pred_trap.sum()),
                         escaped_predicted_escaping=int(pred_esc_ok.sum()), escaped=len(esc)))
    fits = pd.DataFrame(fits)
    fits.to_csv(RES / "p9_fit.csv", index=False)
    with pd.option_context("display.width", 200):
        print(cond.to_string(float_format=lambda v: f"{v:.4g}"))
        print(fits.to_string(float_format=lambda v: f"{v:.4g}"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    zz = np.geomspace(5e-4, 0.08, 300)
    for (name, law), ls in zip(LAWS.items(), ("-", "--", ":")):
        c = fits.set_index("law").c_gamma[name]
        y = law(c * zz)
        ok = np.isfinite(y) & (y < 6)
        ax.plot(zz[ok], y[ok], ls, color=ps.MUTED, lw=1, label=f"{name}, c_γ = {c:.0f}")
    for i, b0 in enumerate(sorted(cond.b0.unique(), reverse=True)):
        for B, mk in ((64, "o"), (16, "s")):
            s = cond[(cond.b0 == b0) & (cond.B == B)]
            e = s[s.escaped == 1]
            ax.errorbar(e.z, e.ratio, yerr=e.ratio_se, fmt=mk, ms=4, color=ps.CAT[i], lw=1,
                        mfc="white" if B == 16 else ps.CAT[i],
                        label=f"b₀ = {b0:g}, B = {B}" if len(s) else None)
            part = s[(s.escaped > 0) & (s.escaped < 1)]
            ax.plot(part.z, part.ratio, mk, ms=6, color=ps.CAT[i], mfc="none", mew=1.5)
            trap = s[s.escaped == 0]
            ax.plot(trap.z, np.full(len(trap), 5.6), "v", ms=6, color=ps.CAT[i], mfc="white")
    ax.text(0.97, 0.98, "▽ all runs trapped", transform=ax.transAxes, fontsize=6.5, color=ps.INK2,
            ha="right", va="top")
    ax.set(xscale="log", ylim=(0.95, 6.0), xlabel="z = (η/B) · T_GF",
           ylabel="T_SGD / T_GF (escaped runs, mean ± s.e.)",
           title="(a) P9: one c_γ for all x, three candidate laws")
    ax.legend(fontsize=5.6, loc="upper left", ncol=1)
    ax = axes[1]
    for i, b0 in enumerate(sorted(df.b0.unique(), reverse=True)):
        s = df[df.b0 == b0]
        ax.scatter(s.eta_over_eta_ms * np.exp(np.random.default_rng(i).uniform(-0.05, 0.05, len(s))),
                   s.min_a1, s=12, color=ps.CAT[i],
                   label=f"b₀ = {b0:g} (runs up to {s.t_max.iloc[0]:.0f} time units)")
    ax.axhline(0.5, color=ps.CAT[7], lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.03, 0.06, 0.12, 0.25], ["0.03", "0.06", "0.12", "0.25"])
    ax.minorticks_off()
    ax.set(xlabel="η / η*_ms(B)", ylabel="min over the run of a₁",
           title="(b) learned component under noise (P10)")
    ax.legend(fontsize=6.0, loc="lower left")
    fig.tight_layout()
    fig.savefig(RES / "fig_p9.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
