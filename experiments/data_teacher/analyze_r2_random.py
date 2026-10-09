"""Round 2, item 1, random fan-in init on mps4 and binary4 (configs/r2_random.yaml; dense library runs).

Event protocol of theory_teachers.md v2 section 9: t_1 = first time L is halfway between L_0 and the
rank-one level L_1 (the exact rank-1 critical value of that teacher on that tree,
results/r2_saddle_hessian.csv); events t_e = first time s_2^e >= f sigma_2^e(T) (f = 0.1 and 0.03);
gap = min_e t_e - t_1. Fits over sigma (6 values, 1 decade, 5 seeds; per-sigma means): log law
gap = A ln(1/sigma) + B and power law gap = C sigma^-p.
Predictions: mps4 (strict rank-1 saddle): A = A_e* = (1 - kappa_e)/gamma_e (theory_teachers R6,
gamma_e = lambda^(1/2) s2(M_e), kappa_e = s2(M_e)/lambda, e* = argmax s2(M_e)), whitened teachers only;
binary4 (depth-3 linear network, non-strict): p = 1.
Usage: uv run python experiments/data_teacher/analyze_r2_random.py
Writes results/r2_random_gaps.csv, results/r2_random_fits.csv, results/r2_random_law.png.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_e4b_order import rank1_theory  # noqa: E402
from analyze_r2_saddle import lsq  # noqa: E402
from common import OUT_ROOT  # noqa: E402
from run_generic import expand  # noqa: E402
from setups import build  # noqa: E402

from ttn.dynamics import Trajectory  # noqa: E402

HERE = Path(__file__).parent
NAME = "r2_random"


def gap(tr, l1, tsp, f):
    t, loss, sp = tr.data["t"], tr.data["loss"], np.nan_to_num(tr.data["spectra"])
    l0 = loss[0]
    k1 = np.argmax(loss <= 0.5 * (l0 + l1))
    t1 = t[k1]
    ts = []
    for j in range(sp.shape[1]):
        if tsp[j, 1] < 1e-3:
            continue
        hit = np.nonzero((np.arange(len(t)) >= k1) & (sp[:, j, 1] >= f * tsp[j, 1]))[0]
        if len(hit):
            ts.append(t[hit[0]])
    return t1, (min(ts) - t1) if ts else np.nan


def main():
    cfg = yaml.safe_load(open(HERE / "configs" / f"{NAME}.yaml"))
    hes = pd.read_csv(HERE / "results/r2_saddle_hessian.csv")
    rows, preds = [], {}
    for case in expand(cfg):
        for tree in cfg["trees"]:
            spec, teacher, metrics = build(dict(case, tree=tree), 0)[:3]
            l1 = float(hes[(hes.teacher == case["tag"]) & (hes.tree == tree) & (hes.n == 4)]["l_saddle"].iloc[0])
            if tree == "mps" and metrics is None:
                th = rank1_theory(teacher.dense(), spec)
                preds[case["tag"]] = th[0]["A_star"]
            for sigma in cfg["sigmas"][tree]:
                for seed in range(*cfg["seeds"]):
                    p = OUT_ROOT / f"{NAME}_dense" / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
                    if not p.exists():
                        continue
                    tr = Trajectory.load(p)
                    tsp = tr.data["teacher_spectra"]
                    for f in (0.1, 0.03):
                        t1, g = gap(tr, l1, tsp, f)
                        rows.append(dict(teacher=case["tag"], tree=tree, sigma=sigma, seed=seed, f=f,
                                         t1=t1, gap=g))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / f"{NAME}_gaps.csv", index=False)
    fits = []
    teachers = list(dict.fromkeys(df["teacher"]))
    fig, axes = plt.subplots(1, len(teachers), figsize=(4 * len(teachers), 4), squeeze=False)
    for i, tname in enumerate(teachers):
        ax = axes[0, i]
        for tree, col in (("mps", "C3"), ("binary", "C0")):
            for f, mk in ((0.1, "o"), (0.03, "^")):
                g = df[(df.teacher == tname) & (df.tree == tree) & (df.f == f) & (df.gap > 0)]
                if len(g) < 6:
                    continue
                m = g.groupby("sigma")["gap"].agg(["mean", "sem"])
                (A, B), A_se, rl = lsq(np.log(1 / m.index.values), m["mean"].values)
                (q, lc), p_se, rp = lsq(np.log(m.index.values), np.log(m["mean"].values))
                fits.append(dict(teacher=tname, tree=tree, f=f, n_runs=len(g), log_A=A, log_A_se=A_se,
                                 pred_A=preds.get(tname, np.nan) if tree == "mps" else np.nan,
                                 log_rel_rms=float(np.sqrt(np.mean((rl / m["mean"].values) ** 2))),
                                 power_p=-q, power_p_se=p_se, power_rms_log=float(np.sqrt(np.mean(rp**2)))))
                lab = (f"{tree}4 f={f}: A={A:.2f}+-{A_se:.2f}" + (f" (pred {preds[tname]:.2f})" if tree == "mps" and tname in preds else "")
                       if tree == "mps" else f"{tree}4 f={f}: p={-q:.2f}+-{p_se:.2f} (pred 1)")
                ax.errorbar(m.index, m["mean"], m["sem"], fmt=mk, color=col, ms=4, mfc="none" if f == 0.03 else col,
                            label=lab)
        ax.set(xscale="log", yscale="log", xlabel="sigma (init scale)", ylabel="gap t_e - t_1", title=tname)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"{NAME}_law.png", dpi=100)
    out = pd.DataFrame(fits)
    out.to_csv(HERE / "results" / f"{NAME}_fits.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
