"""Round 2, item 1: escape law from the first saddle per tree, same teacher (saddle-start runs).

For every (teacher, tree): fits over sigma (per-sigma means over seeds, 6 sigma over 1.7 decades)
  log law:   t_esc = A ln(1/sigma) + B      (relative rms of the fit)
  power law: t_esc = C sigma^-p             (rms of log t)
and the predictions: MPS (strict saddle) A = 1/|lambda_min| from the exact-saddle Hessian
(results/r2_saddle_hessian.csv); binary / HT (non-strict, 3-core route) p = 1.
The verdict "log" / "power" is the fit with the smaller relative residual.
Usage: uv run python experiments/data_teacher/analyze_r2_saddle.py
Writes results/r2_saddle_fits.csv and results/r2_saddle_law.png.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent


def lsq(x, y):
    a = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    res = y - a @ coef
    dof = max(len(x) - 2, 1)
    se = np.sqrt(res @ res / dof * np.linalg.inv(a.T @ a)[0, 0])
    return coef, se, res


def main():
    runs = pd.read_csv(HERE / "results/r2_saddle_runs.csv")
    hes = pd.read_csv(HERE / "results/r2_saddle_hessian.csv")
    rows = []
    teachers = list(dict.fromkeys(runs["teacher"]))
    ncol = 6
    fig, axes = plt.subplots(2, ncol, figsize=(4 * ncol, 8), squeeze=False)
    colors = {"mps": "C3", "binary": "C0", "ht": "C2"}
    for i, tname in enumerate(teachers):
        ax = axes[i // ncol, i % ncol]
        for tree in ("mps", "binary", "ht"):
            g = runs[(runs.teacher == tname) & (runs.tree == tree)].dropna(subset=["t_esc"])
            if not len(g):
                continue
            m = g.groupby("sigma")["t_esc"].agg(["mean", "sem", "size"])
            x = np.log(1 / m.index.values)
            (A, B), A_se, res_log = lsq(x, m["mean"].values)
            (q, lc), p_se, res_pow = lsq(np.log(m.index.values), np.log(m["mean"].values))
            rel_log = np.sqrt(np.mean((res_log / m["mean"].values) ** 2))
            rel_pow = np.sqrt(np.mean(res_pow**2))
            h = hes[(hes.teacher == tname) & (hes.tree == tree)].iloc[0]
            lam = float(h["lambda_min"])
            pred_A = 1 / abs(lam) if lam < -1e-8 else np.nan
            rows.append(dict(teacher=tname, n=int(g["n"].iloc[0]), tree=tree, n_runs=len(g),
                             lambda_min=lam, strict=lam < -1e-8,
                             log_A=A, log_A_se=A_se, pred_A=pred_A, log_rel_rms=rel_log,
                             power_p=-q, power_p_se=p_se, power_rms_log=rel_pow,
                             verdict="log" if rel_log < rel_pow else "power",
                             t_range=f"{m['mean'].min():.3g}-{m['mean'].max():.3g}"))
            n = int(g["n"].iloc[0])
            ax.errorbar(m.index, m["mean"], m["sem"], fmt="o", color=colors[tree], ms=4,
                        label=f"{tree}{n}: " + (f"A={A:.2f} (pred {pred_A:.2f})" if lam < -1e-8
                                                else f"p={-q:.2f}+-{p_se:.2f} (pred 1)"))
            xs = np.array(sorted(m.index))
            if lam < -1e-8:
                ax.plot(xs, A * np.log(1 / xs) + B, "-", color=colors[tree], lw=0.8)
            else:
                ax.plot(xs, np.exp(lc) * xs**q, "-", color=colors[tree], lw=0.8)
        ax.set(xscale="log", yscale="log", xlabel="sigma (noise at the saddle)",
               ylabel="escape time", title=tname)
        ax.legend(fontsize=6)
    fig.suptitle("Escape from the exact rank-1 saddle + noise sigma (library LSODA; mean +- s.e., 5 seeds). "
                 "MPS: strict, log law (line: fit A ln(1/sigma)+B); binary/HT: non-strict, power law",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "results" / "r2_saddle_law.png", dpi=100)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "r2_saddle_fits.csv", index=False)
    print(out.drop(columns=["t_range"]).round(4).to_string())


if __name__ == "__main__":
    main()
