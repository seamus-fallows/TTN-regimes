"""Hessian at the plateau between the first two loss drops (saddle type on real data, H3).

Usage: uv run python experiments/real_data/plateau_hessians.py <summary.json> <runs_dir> <out.json> <workers> [filter]

For each run (4x4 only: exact Hessian via the dense residual tensor, `DenseHessianProblem`) the state at the
plateau midpoint (minimum |d theta/dt| between the first two loss drops, on the recorded rows) is rebuilt
by re-integrating from the saved end of phase 1 (LSODA, same tolerances). Reported: the raw Hessian
spectrum's lambda_min / lambda_max / number of negative eigenvalues, and the gauge-projected lambda_min
(`ttn.diagnostics.project_out_gauge`, default rtol 1e-3, i.e. the fixed version of 6c42d85).
Prediction (theory_saddles.md R3, R10, P6): strict saddle (k = 2 route, MPS) -> lambda_min tends to a
sigma-independent negative value; k = 3 (binary rank-1 plateau) -> lambda_min ∝ -sigma. Classified across
sigma by `ttn.diagnostics.saddle_scaling`.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from analysis import load  # noqa: E402
from data import DenseHessianProblem, Task, build  # noqa: E402
from run import TASK_KEYS  # noqa: E402

from ttn.diagnostics import gauge_directions, project_out_gauge  # noqa: E402
from ttn.dynamics import Recorder, integrate_gf  # noqa: E402


def one(args):
    row, runs_dir = args
    try:
        torch.set_num_threads(1)
        run = load(Path(runs_dir) / f"{row['name']}.npz")
        if len(row.get("drops", [])) < 2:
            return {"name": row["name"], "skipped": "fewer than two drops"}
        c = run["meta"]["case"]
        lg = run["p2_log"]
        d1, d2 = row["drops"][0], row["drops"][1]
        inside = np.flatnonzero((lg["t"] > d1["t_hi"]) & (lg["t"] < d2["t_lo"]))
        if inside.size == 0:
            return {"name": row["name"], "skipped": "no sample between the drops"}
        tau_mid = float(lg["t"][inside[np.argmin(lg["speed"][inside])]])
        task = Task(**{k: (tuple(c[k]) if k in ("pos", "neg") else c[k]) for k in TASK_KEYS})
        rp = build(task, c["tree"], c["ordering"], c["chi"], c["features"])
        prob = DenseHessianProblem(rp.problem.spec, rp.problem.teacher, dataset=rp.problem.dataset)
        theta_a = torch.as_tensor(run["p1_log"]["theta_end"])
        # re-integrate with the cheap Gauss-Newton Newton matrix (rp.problem); exact Hessian only at the end
        tr = integrate_gf(rp.problem, theta_a, tau_mid, log_times=[tau_mid], rtol=c["rtol"], atol=c["atol"],
                          method="lsoda", recorder=Recorder(spectra=False, conservation=False,
                                                            fine=False, overlaps=False))
        theta = torch.as_tensor(tr["theta_end"])
        h = -prob.jacobian(theta)
        h = 0.5 * (h + h.T)
        raw = torch.linalg.eigvalsh(h).numpy()
        _, proj = project_out_gauge(h, gauge_directions(prob.spec, theta))
        proj = proj.numpy()
        return {"name": row["name"], "tree": c["tree"], "ordering": c["ordering"],
                "features": c["features"], "sigma": c["sigma"], "seed": c["seed"], "tau_mid": tau_mid,
                "loss_mid": float(prob.loss(theta)), "first_rank2_edges": row.get("first_rank2_edges"),
                "raw_min": float(raw[0]), "raw_max": float(raw[-1]), "raw_n_neg": int((raw < 0).sum()),
                "raw_lowest": raw[:6].tolist(), "proj_min": float(proj[0]),
                "proj_n_neg": int((proj < -1e-3 * abs(raw[-1])).sum()), "proj_lowest": proj[:6].tolist()}
    except Exception:
        import traceback
        return {"name": row["name"], "error": traceback.format_exc()}


def main(summary_json, runs_dir, out_json, workers, flt=""):
    rows = [r for r in json.load(open(summary_json)) if "error" not in r and "drops" in r
            and any(f in r["name"] for f in flt.split(",")) and not r.get("centred")]
    with Pool(int(workers)) as pool:
        res = pool.map(one, [(r, runs_dir) for r in rows])
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_json, "w"))
    print(len(res), "runs;", sum("error" in r for r in res), "errors")


if __name__ == "__main__":
    main(*sys.argv[1:])
