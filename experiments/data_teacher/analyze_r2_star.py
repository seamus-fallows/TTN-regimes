"""Round 2, item 2: star teachers. Threshold of forgetting and forgetting depth.

(1) Saddle-start runs (results/r2_star_saddle.csv): from theta_q + noise sigma, the first thing that
    happens is either "forget q" (a_q < 1/2, the strict piggyback escape) or "learn r" (the polynomial
    route). Fraction forgetting vs rho(S)/s_q per tree and sigma; the threshold w_c(sigma) where the
    fraction crosses 1/2 (linear interpolation in w); prediction w* = s_q/sqrt(2) = 1.4142 as sigma -> 0.
(2) Random-init runs (results/r2_star_random.csv): super-critical (w = 1.6) vs sub-critical (1.2): peak
    of a_q, its minimum afterwards (forgetting depth), time spent at the "q learned" level L = w^2 and at
    the "q forgotten" level L = 2, vs sigma; power-law fit of the depth.
Usage: uv run python experiments/data_teacher/analyze_r2_star.py
Writes results/r2_star_threshold.csv, results/r2_star_depth.csv, results/r2_star.png.
"""

from pathlib import Path

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import OUT_ROOT, RESULTS  # noqa: E402

HERE = Path(__file__).parent
W_STAR = 2.0 / np.sqrt(2.0)


def threshold(ws, frac):
    """w where the forgetting fraction first crosses 1/2 (linear interpolation)."""
    ws, frac = np.asarray(ws), np.asarray(frac)
    for i in range(len(ws) - 1):
        if frac[i] < 0.5 <= frac[i + 1]:
            return ws[i] + (0.5 - frac[i]) * (ws[i + 1] - ws[i]) / (frac[i + 1] - frac[i])
    return np.nan


def main():
    # reclassify every saddle-start run from its saved trajectory: "rotate" = a_q < 0.9 before any
    # a_r > 0.5 (the strict escape from theta_q; on ht4 it ends at a blended rank-1 point with
    # a_q ~ 0.67, left only through a polynomial route, so a_q < 0.5 would time the second stage)
    rows = []
    for f in sorted((OUT_ROOT / "r2_star_saddle").glob("*.npz")):
        d = np.load(f)
        r = json.loads(str(d["summary"]))
        t, ov = d["t"], d["overlaps"]
        k_r = np.argmax((ov[:, 1:] > 0.5).any(axis=1)) if (ov[:, 1:] > 0.5).any() else len(t)
        k_rot = np.argmax(ov[:, 0] < 0.9) if (ov[:, 0] < 0.9).any() else len(t)
        r["rotate"] = bool(k_rot < k_r)
        r["t_rotate"] = float(t[k_rot]) if k_rot < len(t) else np.nan
        r["aq_min"] = float(ov[:, 0].min())
        rows.append(r)
    sad = pd.DataFrame(rows)
    sad.to_csv(RESULTS / "r2_star_saddle.csv", index=False)
    sad["forget"] = sad["rotate"]
    thr = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, tree in enumerate(sorted(sad.tree.unique())):
        ax = axes[0, j]
        for i, (sigma, g) in enumerate(sad[sad.tree == tree].groupby("sigma")):
            f = g.groupby("w")["forget"].mean()
            wc = threshold(f.index.values, f.values)
            thr.append(dict(tree=tree, sigma=sigma, w_c=wc, rho_c_over_sq=np.sqrt(2) * wc / 2.0,
                            w_star=W_STAR, n=len(g)))
            ax.plot(np.sqrt(2) * f.index.values / 2.0, f.values, "o-", color=f"C{i}",
                    label=f"sigma={sigma:g}: w_c={wc:.3f}")
        ax.axvline(1.0, color="k", lw=0.8, ls="--")
        ax.set(xlabel="rho(S) / s_q", ylabel="fraction leaving q (a_q < 0.9) before learning r",
               title=f"{tree}: start at theta_q + noise", ylim=(-0.05, 1.05))
        ax.legend(fontsize=7)
    t = pd.DataFrame(thr)
    t.to_csv(RESULTS / "r2_star_threshold.csv", index=False)
    print(t.to_string())
    # escape time from theta_q vs eps for super-critical stars: log law with slope 1/abs(lambda_min)?
    hes = pd.read_csv(RESULTS / "r2_star_hessian.csv")
    from lawfit import per_seed
    rot = sad[sad.rotate].assign(gap=lambda x: x["t_rotate"])
    ps, sm = per_seed(rot, ["tree", "w"], min_points=3)
    if len(sm):
        sm = sm.merge(hes.rename(columns={"lambda_min": "lam"})[["tree", "w", "lam"]], on=["tree", "w"], how="left")
        sm["pred_A"] = 1 / sm["lam"].abs()
        sm.to_csv(RESULTS / "r2_star_rotation_law.csv", index=False)
        print(sm[["tree", "w", "n_seeds", "A_mean", "A_sem", "pred_A", "p_mean", "log_wins"]].round(3).to_string())
    if not (RESULTS / "r2_star_random.csv").exists():
        fig.tight_layout()
        fig.savefig(RESULTS / "r2_star.png", dpi=100)
        return
    rnd = pd.read_csv(RESULTS / "r2_star_random.csv")
    dep = rnd.groupby(["tree", "w", "sigma"]).agg(
        n=("seed", "size"), frac_forgot=("forgotten", "mean"),
        aq_peak=("aq_peak", "median"), aq_min_median=("aq_min_after", "median"),
        aq_min_lo=("aq_min_after", "min"), aq_min_hi=("aq_min_after", "max"),
        t_at_q_level=("time_at_q_level", "median"),
        t_at_forgot_level=("time_at_forgot_level", "median")).reset_index()
    dep.to_csv(RESULTS / "r2_star_depth.csv", index=False)
    # forgetting depth per seed: slope of log(min a_q) vs log(sigma) over the points with
    # 1e-10 < min a_q < 0.5 (q forgotten, not yet at the float floor)
    ex = []
    for (tree, w, seed), g in rnd[rnd.w == 1.6].groupby(["tree", "w", "seed"]):
        g = g[(g.aq_min_after > 1e-10) & (g.aq_min_after < 0.5)].sort_values("sigma")
        if len(g) >= 3:
            slope = np.polyfit(np.log(g.sigma), np.log(g.aq_min_after), 1)[0]
            ex.append(dict(tree=tree, w=w, seed=seed, n_points=len(g), depth_exponent=slope))
    ex = pd.DataFrame(ex)
    ex.to_csv(RESULTS / "r2_star_depth_exponents.csv", index=False)
    print(ex.groupby("tree").depth_exponent.agg(["mean", "sem", "size"]).round(2).to_string())
    print(dep.round(4).to_string())
    for j, tree in enumerate(sorted(rnd.tree.unique())):
        ax = axes[1, j]
        for w, mk in ((1.6, "o"), (1.2, "s")):
            g = rnd[(rnd.tree == tree) & (rnd.w == w)]
            if not len(g):
                continue
            ax.plot(g["sigma"], g["aq_min_after"], mk, ms=3, alpha=0.4, color="C3" if w == 1.6 else "C0")
            m = g.groupby("sigma")["aq_min_after"].median()
            fit = np.polyfit(np.log(m.index.values), np.log(m.values), 1)
            ax.plot(m.index, m.values, mk + "-", color="C3" if w == 1.6 else "C0",
                    label=f"w={w} (rho/s_q={np.sqrt(2) * w / 2:.2f}): median min a_q ~ sigma^{fit[0]:.2f}")
        ax.set(xscale="log", yscale="log", xlabel="sigma (random init)",
               ylabel="min a_q after its first peak", title=f"{tree}: forgetting depth")
        ax.legend(fontsize=7)
    axes[1, 2].set_visible(len(rnd.tree.unique()) > 2)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_star.png", dpi=100)


if __name__ == "__main__":
    main()
