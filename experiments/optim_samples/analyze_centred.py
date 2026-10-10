"""Learning-rate axis across the lazy transition (fix round 1): centred model, chi = 9, binary N=4.

Reads the gd_sweep.py runs of configs/gd_centred_chi9.yaml (via gd_sweep.csv from analyze_gd.py and
the GF reference jsons). Reports
- lambda_GF(sigma), the sharpness of the converged GF minimum (GF stopped at loss 1e-8), with a
  power-law fit per init seed over sigma >= 1.41 (1.05 decades), mean +- s.e. over seeds, and the local
  slopes; predicted lazy exponent 2(K-1) = 4;
- for sigma = 1, 2, 4, 8 (centred) and 0.1, 0.3 (plain small init, chi = 9): GD outcome classes vs
  eta / (2/lambda_GF) (GF-like = monotone to GF's loss; catapult = loss spike, then GF's loss;
  oscillating; diverged; other), the largest loss rise, and the final sharpness x eta/2.
Run: uv run python experiments/optim_samples/analyze_centred.py -> centred_chi9_*.csv, fig_centred_chi9.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps
from common import RAW
from common import RESULTS as RES

LAM_BAL = 3 * 3.0 ** (4 / 3)
CLASSES = [("gf_like", "GF-like (monotone)"), ("catapult", "catapult (spike, then GF's loss)"),
           ("oscillating", "bounded oscillation"), ("reentry", "thrown back"),
           ("other", "other final loss"), ("diverged", "diverges")]


def gf_table():
    rows = []
    for p in (RAW / "gd").glob("binary4_*_chi9_gf.json"):
        d = json.loads(p.read_text())
        tk = d["task"]
        rows.append(dict(sigma=tk["sigma"], seed=tk["seed"], centred=bool(tk.get("centred")),
                         lam_gf=d["sharp_end"], loss_end=d["loss_end"], t_end=d.get("t_end"),
                         eff_ranks=str(d["eff_ranks_end"])))
    return pd.DataFrame(rows)


def fit_slope(x, y):
    a = np.vstack([np.ones_like(x), np.log(x)]).T
    coef, *_ = np.linalg.lstsq(a, np.log(y), rcond=None)
    return coef[1]


def main():
    gf = gf_table()
    gf.to_csv(RES / "centred_chi9_gf.csv", index=False)
    cen = gf[gf.centred]
    print("GF converged (loss <= 1e-6):", int((cen.loss_end <= 1e-6).sum()), "of", len(cen))
    lazy = cen[cen.sigma >= 1.41]
    exps = [fit_slope(g.sigma.values, g.lam_gf.values) for _, g in lazy.groupby("seed")
            if g.sigma.nunique() >= 5]
    m = cen.groupby("sigma").lam_gf.agg(["mean", "sem", "count"])
    local = np.diff(np.log(m["mean"].values)) / np.diff(np.log(m.index.values))
    fit = dict(exponent_mean=float(np.mean(exps)), exponent_se=float(np.std(exps, ddof=1) / np.sqrt(len(exps))),
               n_seeds=len(exps), sigma_min=1.41, sigma_max=float(lazy.sigma.max()), predicted=4.0)
    pd.DataFrame([fit]).to_csv(RES / "centred_chi9_lambda_fit.csv", index=False)
    print(m.assign(ratio_to_bal=m["mean"] / LAM_BAL).to_string(float_format=lambda v: f"{v:.4g}"))
    print("local slopes d ln lambda / d ln sigma:", np.round(local, 3))
    print("per-seed exponent over sigma >= 1.41:", fit)
    df = pd.read_csv(RES / "gd_sweep.csv")
    gd = df[(df.chi == 9) & (df.ratio_ref == "gf")].copy()
    gd["stab_end"] = gd.sharp_end * gd.eta / 2
    gd["model"] = np.where(gd.centred, "centred", "plain")
    classes = gd.groupby(["model", "sigma", "ratio"]).cls.value_counts().unstack(fill_value=0)
    classes.to_csv(RES / "centred_chi9_classes.csv")
    sel = gd[gd.cls.isin(["gf_like", "catapult"])].groupby(["model", "sigma", "ratio"]).agg(
        n=("seed", "size"), stab_end=("stab_end", "mean"), sharp_rel=("sharp_end_rel", "mean"),
        max_rise=("max_rise", "max"))
    sel.to_csv(RES / "centred_chi9_selection.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(classes.to_string())
        print(sel.to_string(float_format=lambda v: f"{v:.3g}"))
    ps.apply()
    sigmas = [("plain", 0.1), ("plain", 0.3), ("centred", 1.0), ("centred", 2.0), ("centred", 4.0),
              ("centred", 8.0)]
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_subplot(2, 4, 1)
    ax.errorbar(m.index, m["mean"], yerr=m["sem"], fmt="o", ms=4, color=ps.CAT[0], lw=1,
                label="centred, χ = 9 (5 seeds)")
    xs = np.geomspace(1.41, m.index.max(), 20)
    y0 = m["mean"].loc[m.index >= 1.41].iloc[0]
    ax.plot(xs, y0 * (xs / xs[0]) ** fit["exponent_mean"], "-", color=ps.MUTED, lw=1,
            label=f"fit σ ≥ 1.41: σ^{fit['exponent_mean']:.2f}")
    ax.axhline(LAM_BAL, color=ps.CAT[7], lw=0.8, ls=":", label="λ_bal = K s₁^{2−2/K}")
    ax.set(xscale="log", yscale="log", xlabel="init scale σ", ylabel="λ_GF (converged GF minimum)",
           title="(a) sharpness of the GF minimum")
    ax.legend(fontsize=5.8)
    colors = {c: ps.CAT[i] for i, (c, _) in enumerate(CLASSES)}
    for j, (model, sig) in enumerate(sigmas):
        ax = fig.add_subplot(2, 4, j + 2 if j < 3 else j + 3)
        if (model, sig) not in classes.index.droplevel(2):
            continue
        frac = classes.loc[(model, sig)]
        frac = frac.div(frac.sum(axis=1), axis=0)
        bottom = np.zeros(len(frac))
        for c, lab in CLASSES:
            v = frac[c].values if c in frac else np.zeros(len(frac))
            ax.bar(np.arange(len(frac)), v, bottom=bottom, color=colors[c], width=0.85,
                   edgecolor="white", linewidth=0.6, label=lab)
            bottom += v
        ax.set_xticks(np.arange(len(frac)), [f"{r:g}" for r in frac.index], fontsize=6.5)
        ax.set(xlabel="η / (2/λ_GF)", ylim=(0, 1),
               title=f"{model}, σ = {sig:g}" + (" (lazy side)" if sig >= 2 else ""))
        ax.grid(False)
    handles, labels = fig.axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.05), fontsize=6.5)
    fig.tight_layout()
    fig.savefig(RES / "fig_centred_chi9.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
