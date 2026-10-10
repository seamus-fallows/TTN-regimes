"""Round 2, item 2: star teachers. Threshold of forgetting and forgetting depth.

(1) Saddle-start runs (results/r2_star_saddle.csv): from theta_q + noise sigma, the first thing that
    happens is either "forget q" (a_q < 1/2, the strict piggyback escape) or "learn r" (the polynomial
    route). Fraction forgetting vs rho(S)/s_q per tree and sigma; the threshold w_c(sigma) where the
    fraction crosses 1/2 (linear interpolation in w); prediction w* = s_q/sqrt(2) = 1.4142 as sigma -> 0.
(2) Random-init runs (results/r2_star_random.csv): super-critical (w = 1.6) vs sub-critical (1.2): peak
    of a_q, its minimum afterwards, time spent at the "q learned" level L = w^2 and at the "q forgotten"
    level L = 2, vs sigma. Fix round 2: super-critical runs are classified per seed by what a_q does
    (keeps q / near q / blended); the decay of a_q is steep but not a power law (local exponents drift),
    so only per-seed local exponents are reported.
Usage: uv run python experiments/data_teacher/analyze_r2_star.py
Writes results/r2_star_saddle.csv, r2_star_threshold.csv, r2_star_rotation_law.csv, r2_star_depth.csv,
r2_star_random_classes.csv, r2_star_depth_exponents.csv, r2_star.png.
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
    # prediction at every w: lambda_min = -a^(K-2) (rho - s_q), a = s_q^(1/K) (theory_saddles P8; the
    # exact Hessians of r2_star_hessian.csv match it to 1e-15 at w = 1.2 ... 1.6)
    from lawfit import per_seed, per_seed_local
    n_cores = {"mps4": 4, "binary4": 3, "ht4": 7}
    rot = sad[sad.rotate].assign(gap=lambda x: x["t_rotate"])
    ps, sm = per_seed(rot, ["tree", "w"], min_points=3)
    if len(sm):
        _, sl = per_seed_local(rot, ["tree", "w"], n_small=3)
        sm = sm.merge(sl, on=["tree", "w"], how="left")
        k = sm["tree"].map(n_cores)
        sm["lam"] = -(2.0 ** ((k - 2) / k)) * (np.sqrt(2) * sm["w"] - 2.0)
        sm["pred_A"] = 1 / sm["lam"].abs()
        sm["dev"] = sm["A_mean"] / sm["pred_A"] - 1
        sm["dev_loc"] = sm["A_loc_mean"] / sm["pred_A"] - 1
        sm.to_csv(RESULTS / "r2_star_rotation_law.csv", index=False)
        print(sm[["tree", "w", "n_seeds", "A_mean", "A_sem", "A_loc_mean", "pred_A", "dev", "dev_loc", "p_mean",
                  "log_wins"]].round(4).to_string())
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
    # Review of round 2 (MEDIUM 4): random-init super-critical runs never sit at "q learned" (time at
    # that loss level <= 0.13, one seed <= 1.7, vs 24-538 at "q forgotten"), so this is not the P8
    # forgetting of a learned component. The init direction (seed) fixes what happens, at every sigma:
    # "keeps q" (a_q -> 1 and stays; r, r' learned right after), "near q" (a_q peaks at 0.75-0.99, then
    # decays), "blended" (a_q peaks at 0.55-0.68, near the blended rank-1 point, then decays).
    sup = rnd[rnd.w == 1.6].copy()
    cls = sup.groupby(["tree", "seed"]).agg(peak=("aq_peak", "median"), kept=("forgotten", lambda x: not x.any()))
    cls["cls"] = np.where(cls.kept, "keeps q", np.where(cls.peak < 0.72, "blended", "near q"))
    sup = sup.merge(cls["cls"].reset_index(), on=["tree", "seed"])
    sup["forgot_time"] = sup["time_at_forgot_level"]
    sup.to_csv(RESULTS / "r2_star_random_classes.csv", index=False)
    # depth: min a_q (signed; it crosses zero for some seeds at small sigma) and the time on the
    # "q forgotten" plateau, per seed vs sigma; local exponents between neighbouring sigma show the drift
    ex = []
    for (tree, seed), g in sup[sup.cls != "keeps q"].groupby(["tree", "seed"]):
        g = g.sort_values("sigma")
        pos = g[g.aq_min_after > 0]
        loc = np.diff(np.log(pos.aq_min_after.values)) / np.diff(np.log(pos.sigma.values))
        ok = g[g.forgot_time > 0]
        ex.append(dict(tree=tree, seed=seed, cls=g.cls.iloc[0], n_sigma=len(g),
                       n_min_nonpositive=int((g.aq_min_after <= 0).sum()),
                       depth_exponent_all=np.polyfit(np.log(pos.sigma), np.log(pos.aq_min_after), 1)[0],
                       depth_local_min=loc.min(), depth_local_max=loc.max(),
                       depth_local_small_sigma=loc[0], depth_local_large_sigma=loc[-1],
                       min_aq_smallest_sigma=g.aq_min_after.iloc[0],
                       forgot_time_exponent=-np.polyfit(np.log(ok.sigma), np.log(ok.forgot_time), 1)[0]))
    ex = pd.DataFrame(ex)
    ex.to_csv(RESULTS / "r2_star_depth_exponents.csv", index=False)
    print(cls.groupby(["tree", "cls"]).size().to_string())
    print(ex.round(3).to_string())
    print(ex.groupby(["tree", "cls"])[["depth_exponent_all", "forgot_time_exponent"]].agg(
        ["mean", "sem", "size"]).round(2).to_string())
    print(dep.round(4).to_string())
    for j, tree in enumerate(sorted(rnd.tree.unique())):
        ax = axes[1, j]
        for c, col in (("blended", "C3"), ("near q", "C1"), ("keeps q", "C2")):
            g = sup[(sup.tree == tree) & (sup.cls == c)]
            for k, (seed, gg) in enumerate(g.groupby("seed")):
                gg = gg.sort_values("sigma")
                ax.plot(gg.sigma, gg.aq_min_after.abs(), "o-", ms=3, lw=0.8, color=col,
                        label=f"w=1.6, {c} ({g.seed.nunique()} seeds)" if k == 0 else None)
                neg = gg[gg.aq_min_after <= 0]
                ax.plot(neg.sigma, neg.aq_min_after.abs().clip(lower=1e-12), "x", color="k", ms=6)
        g = rnd[(rnd.tree == tree) & (rnd.w == 1.2)]
        m = g.groupby("sigma")["aq_min_after"].median()
        ax.plot(m.index, m.values, "s-", color="C0", label="w=1.2 (sub-critical): median")
        ax.set(xscale="log", yscale="log", xlabel="sigma (random init)", ylabel="|min a_q| after its first peak",
               title=f"{tree}: a_q decay by init class (x: a_q crosses zero)")
        ax.legend(fontsize=7)
    axes[1, 2].set_visible(len(rnd.tree.unique()) > 2)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_star.png", dpi=100)


if __name__ == "__main__":
    main()
