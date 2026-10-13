"""Exploratory (added after the registration, not used by its decisions): the Hessian at the plateau before
an event. From the stored state at the first escape (theta_a, tau = 0) the flow is re-integrated to the
middle of the plateau before the event; there the exact Hessian (`Problem.hessian`) is diagonalized. A
negative eigenvalue means a strict saddle (log(1/sigma) plateau, depth 2); none means a higher-order saddle.
For the most negative direction, the cores carrying >= 10 % of its weight are listed.

usage: uv run python experiments/route_generic/plateau_hessian.py <results_dir> <synthetic_dir> <out.csv> [workers] [sigma ...]

The sigma-scaling of lambda_min (not its sign at one sigma) separates strict saddles (sigma-independent)
from higher-order ones (lambda_min -> 0 with sigma): run at two sigma and compare.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.integrate import LSODA

sys.path.insert(0, str(Path(__file__).parent))
from events import load  # noqa: E402
from run import Model, case_name  # noqa: E402

torch.set_num_threads(1)
SIGMA = 0.001
CELLS = [("pattern", "mps6"), ("cp03", "mps6"), ("cp06", "mps6"), ("ttn", "mps6"),
         ("cp06", "binary8"), ("ttn", "binary8")]


def one(args):
    row, path = args
    run = load(path)
    case = run["meta"]["case"]
    model = Model(case)
    y = run["theta_a"].copy()
    t_target = row["t_prev"] + 0.5 * (row["t_event"] - row["t_prev"])
    if t_target > 0:
        s = LSODA(lambda t, x: -model.loss_grad(x)[1], 0.0, y, t_target, rtol=1e-10, atol=1e-15,
                  jac=lambda t, x: -model.hessian(x))
        while s.status == "running":
            s.step()
        y = s.y
    H = model.hessian(y)
    lam, vec = np.linalg.eigh(H)
    scale = np.abs(lam).max()
    neg = lam < -1e-6 * scale
    out = dict(row)
    out.update(n_negative=int(neg.sum()), lambda_min=float(lam[0]), lambda_min_rel=float(lam[0] / scale),
               loss=float(model.loss_grad(y)[0]))
    w = np.array([float((vec[sl, 0] ** 2).sum()) for sl in model.spec.core_slices()])
    out["cores_min_dir"] = " ".join(str(v) for v in np.nonzero(w >= 0.1)[0])
    out["n_cores_min_dir"] = int((w >= 0.1).sum())
    return out


def main():
    res, synth, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    sigmas = [float(x) for x in sys.argv[5:]] or [SIGMA]
    ev = pd.read_csv(res / "events.csv")
    jobs = []
    for (target, tree), sig in [(c, s) for c in CELLS for s in sigmas]:
        e = ev[(ev["target"] == target) & (ev["tree"] == tree) & np.isclose(ev["sigma"], sig)]
        e = e.sort_values(["teacher_seed", "seed", "event"])
        for (ts, seed), g in e.groupby(["teacher_seed", "seed"]):
            if seed > 1:
                continue
            g = g.reset_index(drop=True)
            for j in range(1, len(g)):
                row = dict(target=target, tree=tree, teacher_seed=int(ts), seed=int(seed), sigma=sig,
                           event=int(g.loc[j, "event"]), rises=g.loc[j, "rises"], prior=g.loc[j, "prior"],
                           bonds=g.loc[j, "bonds"], k_region=int(g.loc[j, "k_region"]),
                           eff_depth=float(g.loc[j, "eff_depth"]),
                           t_prev=float(g.loc[j - 1, "t_event"]) if j > 1 else 0.0,
                           t_event=float(g.loc[j, "t_event"]))
                case = dict(target=target, tree=tree, d=int(g.loc[j, "d"]), teacher_seed=int(ts), sigma=sig,
                            seed=int(seed))
                jobs.append((row, str(synth / f"{case_name(case)}.npz")))
    with Pool(workers) as pool:
        rows = pool.map(one, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    # sigma-scaling: the same event (signature) at the smallest and largest sigma of the call
    if len(sigmas) > 1:
        keys = ["target", "tree", "teacher_seed", "seed", "prior", "rises"]
        lo, hi = df[np.isclose(df["sigma"], min(sigmas))], df[np.isclose(df["sigma"], max(sigmas))]
        m = lo.merge(hi, on=keys, suffixes=("_lo", "_hi"))
        m["lambda_slope"] = np.log(m["lambda_min_hi"].abs() / m["lambda_min_lo"].abs()) / np.log(max(sigmas) / min(sigmas))
        m.to_csv(out_csv.replace(".csv", "_scaling.csv"), index=False, float_format="%.6g")
        print(m.groupby(["target", "tree", "k_region_lo"]).agg(
            n=("lambda_slope", "size"), slope_med=("lambda_slope", "median"),
            slope_q25=("lambda_slope", lambda s: s.quantile(0.25)), slope_q75=("lambda_slope", lambda s: s.quantile(0.75)),
            lam_lo=("lambda_min_lo", "median"), lam_hi=("lambda_min_hi", "median")).to_string())
    print(df.groupby(["target", "tree", "k_region"]).agg(
        n=("n_negative", "size"), strict=("n_negative", lambda s: (s > 0).mean()),
        n_neg_med=("n_negative", "median"), lam_min_med=("lambda_min", "median"),
        cores_dir_med=("n_cores_min_dir", "median"), d_eff=("eff_depth", "median")).to_string())


if __name__ == "__main__":
    main()
