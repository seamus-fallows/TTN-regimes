"""Round 2, item 1: escape law from the first saddle per tree, same teacher (saddle-start runs).

Reads every saddle-start run (r2_saddle.yaml, the balanced N = 8 misaligned supplement
r2_saddle_b.yaml and the eps extension r2_saddle_ext.yaml; all in <OUT_ROOT>/r2_saddle/).
For every (teacher, tree), per seed (the same seed = the same noise direction at every eps):
fits of the log law t = A ln(1/eps) + B and of the power law t = C eps^-p, both by least squares on
log t and scored by the rms of the log residuals (lawfit.py); mean +- s.e.m. over seeds. Runs that
did not escape before t_cap are counted and excluded.
Predictions: MPS (strict saddle) A = 1/|lambda_min| of the exact-saddle Hessian
(results/r2_saddle_hessian.csv, r2_saddle_b_hessian.csv); binary / HT (non-strict; 3-core route)
p = 1.
Usage: uv run python experiments/data_teacher/analyze_r2_saddle.py
Writes results/r2_saddle_perseed.csv, results/r2_saddle_fits.csv, results/r2_saddle_law.png.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_ROOT, RESULTS  # noqa: E402
from lawfit import per_seed  # noqa: E402

HERE = Path(__file__).parent
TREES = ("mps", "binary", "ht")
COLORS = {"mps": "C3", "binary": "C0", "ht": "C2"}


def load_runs():
    rows = [json.loads(str(np.load(f)["summary"])) for f in sorted((OUT_ROOT / "r2_saddle").glob("*.npz"))]
    return pd.DataFrame(rows)


def main():
    runs = load_runs()
    runs.to_csv(RESULTS / "r2_saddle_runs_all.csv", index=False)
    hes = pd.concat([pd.read_csv(RESULTS / f) for f in ("r2_saddle_hessian.csv", "r2_saddle_b_hessian.csv")
                     if (RESULTS / f).exists()]).drop_duplicates(["teacher", "tree"])
    runs["escaped"] = np.isfinite(runs["t_esc"])
    ok = runs[runs.escaped].rename(columns={"t_esc": "gap"})
    ps, summ = per_seed(ok, ["teacher", "n", "tree"], min_points=4)
    summ = summ.merge(hes[["teacher", "tree", "lambda_min", "l_saddle"]], on=["teacher", "tree"], how="left")
    summ["pred_A"] = np.where(summ["lambda_min"] < -1e-8, 1 / summ["lambda_min"].abs(), np.nan)
    summ["pred_p"] = np.where(summ["tree"] == "mps", np.nan, 1.0)
    cnt = runs.groupby(["teacher", "tree"]).agg(n_runs=("seed", "size"), n_escaped=("escaped", "sum"),
                                                eps_min=("sigma", "min"), eps_max=("sigma", "max")).reset_index()
    summ = summ.merge(cnt, on=["teacher", "tree"], how="left")
    ps.to_csv(RESULTS / "r2_saddle_perseed.csv", index=False)
    summ.to_csv(RESULTS / "r2_saddle_fits.csv", index=False)
    cols = ["teacher", "tree", "n_seeds", "n_escaped", "n_runs", "eps_min", "lambda_min", "A_mean", "A_sem",
            "pred_A", "p_mean", "p_sem", "rms_log_median", "rms_pow_median", "log_wins"]
    print(summ[cols].round(4).to_string())
    teachers = list(dict.fromkeys(runs.sort_values("n")["teacher"]))
    ncol = 7
    nrow = int(np.ceil(len(teachers) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow), squeeze=False)
    for i, tname in enumerate(teachers):
        ax = axes[i // ncol, i % ncol]
        for tree in TREES:
            g = ok[(ok.teacher == tname) & (ok.tree == tree)]
            if not len(g):
                continue
            m = g.groupby("sigma")["gap"].agg(["mean", "sem"])
            s = summ[(summ.teacher == tname) & (summ.tree == tree)]
            if len(s):
                s = s.iloc[0]
                lab = (f"{tree}: A={s.A_mean:.1f}+-{s.A_sem:.1f} (pred {s.pred_A:.1f})" if tree == "mps"
                       else f"{tree}: p={s.p_mean:.2f}+-{s.p_sem:.2f}")
            else:
                lab = tree
            ax.errorbar(m.index, m["mean"], m["sem"], fmt="o-", ms=3, lw=0.8, color=COLORS[tree], label=lab)
            if tree == "mps" and len(s) and np.isfinite(s.pred_A):
                xs = np.array(sorted(m.index))
                x0, y0 = np.mean(np.log(1 / xs)), m["mean"].mean()
                ax.plot(xs, y0 + s.pred_A * (np.log(1 / xs) - x0), "k--", lw=0.8)
        ax.set(xscale="log", yscale="log", title=tname, xlabel="eps (noise at the saddle)",
               ylabel="escape time")
        ax.legend(fontsize=5.5, loc="upper right")
    for j in range(len(teachers), nrow * ncol):
        axes[j // ncol, j % ncol].set_visible(False)
    fig.suptitle("Escape from the exact balanced rank-1 saddle + noise eps (library LSODA; mean +- s.e.m. over "
                 "5 seeds). Dashed: predicted MPS slope 1/|lambda_min|. Fits per seed in log space.", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_saddle_law.png", dpi=100)


if __name__ == "__main__":
    main()
