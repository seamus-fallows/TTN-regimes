"""Escape-time scaling under aligned covariance (configs e1_scaling_sigma.yaml, e1_scaling_lambda.yaml).

- t_1 (first learning event, a_r = 1/2) vs sigma for (Sigma, s), (I, s), (I, Lambda s): fitted exponent
  (prediction -(K-2), unchanged by the covariance) and prefactor ratio Sigma / white (prediction:
  equal to the (I, Lambda s) one, i.e. the covariance acts through Lambda s).
- t_1 vs Lambda s for one-component teachers: fitted exponent (prediction -1).
Fits: least squares of log t on log x over all runs, standard error from the per-x means
(mean +- s.e. over seeds). Writes results/e1_scaling_*.csv and e1_scaling.png.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS, OUT_ROOT  # noqa: E402

HERE = Path(__file__).parent


def fit_loglog(x, y):
    """Slope and standard error of log y vs log x, fitted to per-x means of log y."""
    df = pd.DataFrame({"lx": np.log(x), "ly": np.log(y)}).groupby("lx")["ly"].agg(["mean", "sem"])
    lx, ly = df.index.values, df["mean"].values
    a = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(a, ly, rcond=None)
    dof = max(len(lx) - 2, 1)
    s2 = np.sum((ly - a @ coef) ** 2) / dof
    cov = s2 * np.linalg.inv(a.T @ a)
    return coef[0], np.sqrt(cov[0, 0]), coef[1]


def main():
    sig = pd.DataFrame(pd.read_pickle(OUT_ROOT / "e1_scaling_sigma" / "summary_rows.pkl"))
    sig["t1"] = sig[["t_half_0", "t_half_1"]].min(axis=1)
    lam = pd.DataFrame(pd.read_pickle(OUT_ROOT / "e1_scaling_lambda" / "summary_rows.pkl"))
    lam["t1"] = lam["t_half_0"]
    lam["Ls"] = lam["Lambda_0"] * lam["s_0"]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for tree, k, mk in [("binary", 3, "o"), ("mps", 4, "s")]:
        for variant, c in [("sigma", "C3"), ("white", "C7"), ("white_Ls", "C0")]:
            g = sig[(sig.tree == tree) & (sig.variant == variant)]
            slope, se, icpt = fit_loglog(g["sigma"], g["t1"])
            rows.append(dict(sweep="sigma", tree=tree, variant=variant, slope=slope, se=se,
                             predicted=-(k - 2), n_runs=len(g)))
            m = g.groupby("sigma")["t1"].agg(["mean", "sem"])
            axes[0].errorbar(m.index, m["mean"], m["sem"], fmt=mk, color=c, ms=4,
                             label=f"{tree}, {variant}: slope {slope:.3f}+-{se:.3f}")
            xs = np.array(sorted(g["sigma"].unique()))
            axes[0].plot(xs, np.exp(icpt) * xs**slope, "-", color=c, lw=0.7)
        # paired ratio t1(Sigma) / t1(I, Lambda s) per seed and sigma
        a = sig[(sig.tree == tree) & (sig.variant == "sigma")].set_index(["sigma", "seed"])["t1"]
        b = sig[(sig.tree == tree) & (sig.variant == "white_Ls")].set_index(["sigma", "seed"])["t1"]
        w = sig[(sig.tree == tree) & (sig.variant == "white")].set_index(["sigma", "seed"])["t1"]
        r1 = (a / b).groupby("sigma").agg(["mean", "sem"])
        r2 = (a / w).groupby("sigma").agg(["mean", "sem"])
        axes[1].errorbar(r1.index, r1["mean"], r1["sem"], fmt=mk + "-", color="C0",
                         label=f"{tree}: t1(Sigma,s) / t1(I, Lambda s)")
        axes[1].errorbar(r2.index, r2["mean"], r2["sem"], fmt=mk + "--", color="C7",
                         label=f"{tree}: t1(Sigma,s) / t1(I, s)")
        for _, rr in r1.reset_index().iterrows():
            rows.append(dict(sweep="paired_ratio_Ls", tree=tree, variant="sigma/white_Ls",
                             sigma=rr["sigma"], slope=rr["mean"], se=rr["sem"], predicted=1.0))
        for variant, c in [("sigma", "C3"), ("white_Ls", "C0")]:
            g = lam[(lam.tree == tree) & (lam.variant == variant)]
            slope, se, icpt = fit_loglog(g["Ls"], g["t1"])
            rows.append(dict(sweep="Lambda", tree=tree, variant=variant, slope=slope, se=se,
                             predicted=-1.0, n_runs=len(g)))
            m = g.groupby("Ls")["t1"].agg(["mean", "sem"])
            axes[2].errorbar(m.index, m["mean"], m["sem"], fmt=mk, color=c, ms=4,
                             label=f"{tree}, {variant}: slope {slope:.3f}+-{se:.3f}")
    axes[0].set(xscale="log", yscale="log", xlabel="sigma", ylabel="t_1 (mean +- s.e., 10 seeds)",
                title="first escape vs init scale (teacher T1)")
    axes[1].set(xscale="log", xlabel="sigma", ylabel="paired ratio",
                title="covariance acts through Lambda s (ratio -> 1)")
    axes[1].axhline(1, color="k", lw=0.6)
    axes[2].set(xscale="log", yscale="log", xlabel="Lambda s", ylabel="t_1",
                title="one component: t_1 vs Lambda s (prediction slope -1)")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / "e1_scaling.png", dpi=110)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "e1_scaling_fits.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
