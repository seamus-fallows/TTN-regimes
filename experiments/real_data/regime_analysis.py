"""Regime sweep (centred model) analysis: one row per run plus the linearized (lazy) reference.

Usage: uv run python experiments/real_data/regime_analysis.py <summary.json> <runs_dir> <out.json> [workers]

For the centred model f_theta - f_theta0 starts at 0 and its linearization at theta_0 follows
df/dt = -(1/n) J J^T (f - y), J = per-sample d f / d theta at theta_0 (n x P). With J = U S V^T,
    L_lin(t) = (1/2n) [ sum_i exp(-2 s_i^2 t / n) (u_i . y)^2 + |(I - U U^T) y|^2 ],
exactly; L_lin(inf) is the lazy floor (`theory_regimes.md` R7: deficient bonds, chi = 4 << 256).
A run is compared with L_lin at the same total time t_a + tau.

Labels (notes/diagnostics.md, with the convergence rule adapted; notes/real_data.md section 3):
  lazy       delta_K(T) < 0.1
  S2S        delta_K >= 0.1, >= 2 loss drops, every rho > 3
  rich       delta_K >= 0.1 otherwise
The notes' "stalled" (L(T) >= 1e-3 L(0)) is not used: no chi = 4 TTN fits the 4000 labels to 1e-3 L(0),
so every run would be "stalled"; the final train / test losses are reported instead, next to L_lin.
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
from data import Task, build  # noqa: E402
from run import TASK_KEYS  # noqa: E402


def linearized_curve(row, runs_dir):
    run = load(Path(runs_dir) / f"{row['name']}.npz")
    c = run["meta"]["case"]
    task = Task(**{k: (tuple(c[k]) if k in ("pos", "neg") else c[k]) for k in TASK_KEYS})
    rp = build(task, c["tree"], c["ordering"], c["chi"], c["features"])
    theta0 = torch.as_tensor(run["theta0"])
    jac = rp.problem.per_sample_jacobian(theta0).numpy()
    y = rp.problem.dataset[1].numpy()
    n = len(y)
    u, s, _ = np.linalg.svd(jac, full_matrices=False)
    keep = s > 1e-10 * s.max()
    u, s = u[:, keep], s[keep]
    proj = u.T @ y
    floor = 0.5 * (y @ y - proj @ proj) / n

    def l_lin(t):
        return 0.5 * np.sum(np.exp(-2 * s**2 * t / n) * proj**2) / n + floor

    t_total = run["meta"]["t_a"] + float(run["p2_log"]["t"][-1])
    return {"name": row["name"], "L_lin_floor": float(floor), "L_lin_T": float(l_lin(t_total)),
            "rank_J": int(keep.sum()), "t_total": t_total}


def label(row, dk_thr=0.1, sep=3.0):
    if row.get("delta_K_end") is None or not np.isfinite(row["delta_K_end"]):
        return "undetermined"
    if row["delta_K_end"] < dk_thr:
        return "lazy"
    rho = np.asarray(row.get("rho", []), float)
    if row.get("n_drops", 0) >= 2 and np.all(rho > sep):
        return "S2S"
    return "rich"


def _safe(args):
    try:
        return linearized_curve(*args)
    except Exception as exc:
        return {"name": args[0]["name"], "error": repr(exc)}


def main(summary_json, runs_dir, out_json, workers=8):
    rows = [r for r in json.load(open(summary_json)) if r.get("centred") and "error" not in r]
    with Pool(int(workers)) as pool:
        lin = {d["name"]: d for d in pool.map(_safe, [(r, runs_dir) for r in rows])}
    out = []
    for r in rows:
        d = {k: r.get(k) for k in ("name", "sigma", "seed", "delta_K_end", "kernel_scale_0",
                                    "kernel_scale_end", "loss_end", "test_loss_end", "test_acc_end",
                                    "train_acc_end", "n_drops", "rho", "t2_unit", "stop", "t_a", "t2")}
        d.update(lin.get(r["name"], {}))
        d["labels"] = {f"dK{th}_rho{sp}": label(r, th, sp) for th in (0.05, 0.1, 0.2)
                       for sp in (2.0, 3.0, 5.0)}
        d["label"] = label(r)
        out.append(d)
    json.dump(out, open(out_json, "w"), default=float)
    print(len(out), "runs")


if __name__ == "__main__":
    main(*sys.argv[1:])
