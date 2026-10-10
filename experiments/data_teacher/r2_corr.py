"""Round 2, item 3 (and the review's step-3 follow-up): sigma law under cross-site correlated inputs.

Exact population gradient flow with the dense moment matrix (correlated.py; the library's Problem has
no non-product metric), integrated with scipy LSODA and the exact Hessian as Jacobian
(pilots/pilot_gf.py), in two stages as run_dense.py: (A) until L < 0.98 L(0), (B) on a grid geometric
in t - t_A. Setting of theory_covariance.md v2 section 7: N = 4, d = 3, site-aligned odeco teacher
s = (1, 0.6), shared-latent-factor inputs with correlation rho, mps4 and binary4 students.
Events (theory_covariance.md 10.10): singular values across the middle cut (mps4: 01|23 is edge 1;
binary4: 01|23) relative to their final values, at 1/2, with final-rise widths (10% -> 90%).
Gap = t(s_2 event) - t(s_1 event).
Usage: uv run python experiments/data_teacher/r2_corr.py [--procs 28]
Raw: <OUT_ROOT>/r2_corr/*.npz; summary: results/r2_corr_runs.csv.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from common import OUT_ROOT, RESULTS
from correlated import dense_metric_loss_fn, moment_matrix, site_cross_cov

sys.path.insert(0, str(Path(__file__).parent / "pilots"))
from pilot_gf import run_gf  # noqa: E402

from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

N, D = 4, 3
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]
SIGMAS = [0.0016, 0.0031, 0.0063, 0.0125, 0.025, 0.05]


def final_rise(t, x, final):
    """Event at x = final/2 (last upward crossing) and the final-rise width (last 10% -> 90%)."""
    r = x / final

    def last_up(level):
        above = r >= level
        idx = np.nonzero(above[1:] & ~above[:-1])[0]
        if len(idx) == 0:
            return np.nan
        k = idx[-1]
        return t[k] + (level - r[k]) * (t[k + 1] - t[k]) / (r[k + 1] - r[k])

    return last_up(0.5), last_up(0.9) - last_up(0.1)


def job(args):
    tree, rho, sigma, seed = args
    torch.set_num_threads(1)
    path = OUT_ROOT / "r2_corr" / f"{tree}_rho{rho}_s{sigma}_seed{seed}.npz"
    if path.exists():
        return json.loads(str(np.load(path)["summary"]))
    spec = TTNSpec.uniform(make_tree(tree, N), D, 4)
    teacher = pattern_teacher([1.0, 0.6], [[0] * N, [1] * N], D, basis="standard")
    M = moment_matrix(N, D, site_cross_cov(N, rho))
    loss_fn = dense_metric_loss_fn(spec, teacher.dense(), M)
    cores0 = spec.init(sigma, seed=seed)
    kw = dict(method="LSODA", rtol=1e-9, atol=1e-14, loss_fn=loss_fn, w_star=teacher.dense(),
              keep_y=False)
    a = run_gf(spec, cores0, teacher, None, t_max=1e12, n_rec=300, stop_rel_loss=0.98, **kw)
    t_a = float(a["t"][-1])
    cores_a = spec.unflatten(torch.from_numpy(a["y_final"].copy()))
    grid = np.concatenate([[0.0], np.geomspace(1e-2, 1e7, 3000)])
    b = run_gf(spec, cores_a, teacher, None, t_max=1e7, stop_rel_loss=1e-5, t_eval=grid, **kw)
    t = np.concatenate([a["t"], b["t"][1:] + t_a])
    loss = np.concatenate([a["loss"], b["loss"][1:]])
    mid = 1 if tree == "mps" else 0  # index of the 01|23 edge in spec.tree.edges
    v = spec.tree.edges[mid]
    sp = np.array([s[v][:2] for s in a["spectra"]] + [s[v][:2] for s in b["spectra"][1:]])
    # final values: the teacher's singular values across 01|23 (Frobenius)
    w = teacher.dense().reshape(D * D, D * D)
    fin = torch.linalg.svdvals(w)[:2].numpy()
    e1, w1 = final_rise(t, sp[:, 0], fin[0])
    e2, w2 = final_rise(t, sp[:, 1], fin[1])
    summary = dict(tree=tree, rho=rho, sigma=sigma, seed=seed, loss0=float(loss[0]),
                   t_a=t_a, t_event1=e1, width1=w1, t_event2=e2, width2=w2, gap=e2 - e1,
                   sep_ratio=(e2 - e1) / max(w1, w2), loss_end=float(loss[-1]),
                   q_drift=max(a["q_drift"], b["q_drift"]), t_end=float(t[-1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=t, loss=loss, spectra_mid=sp, summary=json.dumps(summary))
    print(tree, rho, sigma, seed, f"gap={summary['gap']:.4g} rho_sep={summary['sep_ratio']:.3g}",
          flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=28)
    args = ap.parse_args()
    jobs = [(tree, rho, s, seed) for tree in ("mps", "binary") for rho in RHOS for s in SIGMAS
            for seed in range(5)]
    jobs.sort(key=lambda j: j[2])
    with Pool(args.procs) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    import pandas as pd
    pd.DataFrame(rows).to_csv(RESULTS / "r2_corr_runs.csv", index=False)
