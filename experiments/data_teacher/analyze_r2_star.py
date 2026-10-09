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

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    sad = pd.read_csv(HERE / "results/r2_star_saddle.csv")
    sad["forget"] = sad["outcome"] == "forget_q"
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
        ax.set(xlabel="rho(S) / s_q", ylabel="fraction forgetting q first (5 seeds)",
               title=f"{tree}: start at theta_q + noise", ylim=(-0.05, 1.05))
        ax.legend(fontsize=7)
    t = pd.DataFrame(thr)
    t.to_csv(HERE / "results" / "r2_star_threshold.csv", index=False)
    print(t.to_string())
    rnd = pd.read_csv(HERE / "results/r2_star_random.csv")
    dep = rnd.groupby(["tree", "w", "sigma"]).agg(
        n=("seed", "size"), frac_forgot=("forgotten", "mean"),
        aq_peak=("aq_peak", "median"), aq_min_median=("aq_min_after", "median"),
        aq_min_lo=("aq_min_after", "min"), aq_min_hi=("aq_min_after", "max"),
        t_at_q_level=("time_at_q_level", "median"),
        t_at_forgot_level=("time_at_forgot_level", "median")).reset_index()
    dep.to_csv(HERE / "results" / "r2_star_depth.csv", index=False)
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
    fig.savefig(HERE / "results" / "r2_star.png", dpi=100)


if __name__ == "__main__":
    main()
