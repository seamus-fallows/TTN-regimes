"""Sweeps that need time resolution of short later plateaus, on the library integrator.

Gradient flow with `ttn.dynamics.integrate_gf(method="lsoda", rtol=1e-9, atol=1e-14)` in two stages:
(A) from the init on a log grid until the loss first drops below 0.98 L(0) (the first escape has
started), (B) from there on a grid geometric in t - t_A (from 1e-2 to t_max, n_dense points), so
that plateaus lasting O(1) time units right after a long first plateau are resolved. Stage B stops
at L < stop_rel_loss L(0). Parameters are saved at every log time.

Configs as for run_generic.py (same `setups.build`), plus `n_dense`. Product covariances only (the
library's Problem has no non-product metric).
Usage: uv run python experiments/data_teacher/run_dense.py configs/<name>.yaml [--procs 28]
Writes /root/outputs/data_teacher/<name>_dense/<tag>_<tree>_s<sigma>_seed<seed>.npz (library
Trajectory format: data + meta) and results/<name>_dense_summary.csv.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from common import RESULTS, OUT_ROOT, map_cores
from run_generic import dense_bond_spectra, expand
from setups import build

from ttn.dynamics import Problem, Recorder, Trajectory, integrate_gf, log_time_grid

HERE = Path(__file__).parent


def two_stage(problem, theta0, t_max, n_dense, stop_rel, rtol=1e-9, atol=1e-14):
    l0 = float(problem.loss(theta0))
    rec = dict(spectra=True, overlaps=hasattr(problem, "teacher") and hasattr(
        getattr(problem, "teacher", None), "factors"), save_params=True)
    a = integrate_gf(problem, theta0, t_max=t_max, n_log=300, rtol=rtol, atol=atol,
                     loss_floor=0.98 * l0, method="lsoda", recorder=Recorder(**rec))
    t_a = float(a.data["t"][-1])
    theta_a = torch.as_tensor(a.data["theta_end"])
    grid = np.unique(np.r_[np.geomspace(1e-2, max(t_max - t_a, 1.0), n_dense)])
    b = integrate_gf(problem, theta_a, t_max=max(t_max - t_a, 1.0), log_times=grid[:-1],
                     rtol=rtol, atol=atol, loss_floor=stop_rel * l0, method="lsoda",
                     recorder=Recorder(**rec))
    data = {}
    for k in a.data:
        if k == "theta_end" or k not in b.data:
            continue
        xa, xb = np.asarray(a.data[k]), np.asarray(b.data[k])
        if k in ("t", "fine_t"):  # stage-B times are relative to t_a (fine rows too)
            xb = xb + t_a
        if xa.ndim == 0:
            continue
        data[k] = np.concatenate([xa, xb[1:]], axis=0)
    data["theta_end"] = np.asarray(b.data["theta_end"])
    meta = dict(stage_a=a.meta, stage_b=b.meta, t_a=t_a, loss0=l0)
    return Trajectory(data, meta)


def job(args):
    name, case, tree, sigma, seed, cfg = args
    torch.set_num_threads(1)
    path = OUT_ROOT / f"{name}_dense" / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
    if path.exists():
        return Trajectory.load(path).meta["summary"]
    built = build(dict(case, tree=tree), seed)
    if built is None:
        return None
    spec, teacher, metrics, loss_fn, init_map = built
    assert loss_fn is None, "non-product metrics are not supported by the library Problem"
    cores0 = spec.init(sigma, seed=seed)
    if init_map is not None:
        cores0 = map_cores(spec, cores0, init_map)
    problem = Problem(spec, teacher, site_metrics=metrics)
    traj = two_stage(problem, spec.flatten(cores0), float(cfg["t_max"][tree]), int(cfg["n_dense"]),
                     float(cfg["stop_rel_loss"]))
    tsp = dense_bond_spectra(teacher.dense(), spec.tree, metrics)
    traj.data["teacher_spectra"] = np.array([np.pad(tsp[v][: spec.bond_dims[v]],
                                                    (0, max(spec.bond_dims) - min(spec.bond_dims[v], len(tsp[v]))))[: max(spec.bond_dims)]
                                             for v in spec.tree.edges])
    summary = dict({k: v for k, v in case.items() if not isinstance(v, (list, dict))}, tree=tree,
                   sigma=sigma, seed=seed, t_a=traj.meta["t_a"], loss0=traj.meta["loss0"],
                   t_end=float(traj.data["t"][-1]), loss_end=float(traj.data["loss"][-1]),
                   stop=traj.meta["stage_b"].get("stop_reason"),
                   max_drift=float(np.nanmax(traj.data["conservation_drift"])))
    traj.meta["summary"] = summary
    path.parent.mkdir(parents=True, exist_ok=True)
    traj.save(path)
    print(case["tag"], tree, sigma, seed, f"t_a={summary['t_a']:.1f}", summary["stop"], flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=28)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    jobs = [(cfg["name"], case, tree, sigma, seed, cfg)
            for case in expand(cfg) for tree in cfg["trees"] for sigma in cfg["sigmas"][tree]
            for seed in range(*cfg["seeds"])]
    jobs.sort(key=lambda j: j[3])
    with Pool(a.procs) as pool:
        rows = [r for r in pool.map(job, jobs, chunksize=1) if r is not None]
    pd.DataFrame(rows).to_csv(RESULTS / f"{cfg['name']}_dense_summary.csv", index=False)
    print("wrote", len(rows), "rows")
