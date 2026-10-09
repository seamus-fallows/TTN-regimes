"""Gradient flow on the empirical loss of a real-data problem; one trajectory .npz per case.

Usage: uv run python experiments/real_data/run.py <config.yaml> <out_dir> <workers>

The config lists `cases` (each a dict; missing keys come from `defaults`), expanded over the lists
`sigmas` and `seeds`. Keys: dataset, size, pos, neg, n_train, n_test, data_seed (the task); tree
(binary | mps), ordering (quadtree | snake | scrambled<k> | rowmajor), chi, features (raw | white),
centred (bool), sigma, seed (init), t_max, n_per_decade (log-grid density), rtol, atol, method
(dopri5 | lsoda | two_phase: dopri5 until the speed has dropped back below 1e-3 x its peak, then
lsoda), wall_limit (s). Finished cases are skipped (resumable). Every process uses one thread.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import itertools
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data import Task, build  # noqa: E402

from ttn.dynamics import Recorder, integrate_gf, log_time_grid  # noqa: E402
from ttn.network import forward  # noqa: E402

TASK_KEYS = ("dataset", "size", "pos", "neg", "n_train", "n_test", "data_seed")
DEFAULTS = dict(dataset="mnist", size=4, pos=[3], neg=[5], n_train=4000, n_test=1600, data_seed=0,
                tree="binary", ordering="quadtree", chi=4, features="raw", centred=False,
                t_max=1e12, n_per_decade=60, rtol=1e-8, atol=1e-14, method="dopri5",
                wall_limit=3000)


def case_name(c: dict) -> str:
    pos = "".join(map(str, c["pos"]))
    neg = "".join(map(str, c["neg"]))
    return (f"{c['dataset']}{c['size']}_{pos}v{neg}_n{c['n_train']}_{c['features']}_{c['tree']}_"
            f"{c['ordering']}_chi{c['chi']}{'_centred' if c['centred'] else ''}_"
            f"s{c['sigma']:.4g}_seed{c['seed']}_{c['method']}")


def expand(config: dict) -> list[dict]:
    base = {**DEFAULTS, **config.get("defaults", {})}
    out = []
    for case in config["cases"]:
        c = {**base, **case}
        sigmas = c.pop("sigmas", [c.get("sigma")])
        seeds = c.pop("seeds", [c.get("seed", 0)])
        for sigma, seed in itertools.product(sigmas, seeds):
            out.append({**c, "sigma": float(sigma), "seed": int(seed)})
    return out


def extra_diagnostics(rp, cores0_problem):
    """Test loss, train/test accuracy (sign of f), and the fitted function's norm on the train set."""
    problem = rp.problem

    def test_metrics(prob, theta, t):
        spec, cores = prob.effective(theta)
        with torch.no_grad():
            f_te = forward(spec, cores, rp.x_test)
            f_tr = forward(spec, cores, prob.dataset[0])
        y_tr = prob.dataset[1]
        return np.array([
            0.5 * (f_te - rp.y_test).pow(2).mean().item(),
            (torch.sign(f_te) == rp.y_test).double().mean().item(),
            (torch.sign(f_tr) == y_tr).double().mean().item(),
            f_tr.pow(2).mean().sqrt().item(),
        ])

    return [("test", test_metrics)]


def run_case(c: dict, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    name = case_name(c)
    path = out_dir / f"{name}.npz"
    if path.exists():
        return {"name": name, "status": "exists"}
    task = Task(**{k: (tuple(c[k]) if k in ("pos", "neg") else c[k]) for k in TASK_KEYS})
    rp0 = build(task, c["tree"], c["ordering"], c["chi"], c["features"])
    spec = rp0.problem.spec
    cores0 = spec.init(c["sigma"], seed=c["seed"])
    rp = build(task, c["tree"], c["ordering"], c["chi"], c["features"],
               centred_cores0=cores0 if c["centred"] else None)
    problem = rp.problem
    theta0 = spec.flatten(cores0)
    rec = Recorder(overlaps=False, extra=extra_diagnostics(rp, None))
    # log grid from 1e-3 tau_0 (tau_0 = |theta_0| / |v(theta_0)|) to t_max
    speed0 = problem.velocity(theta0).norm().item()
    tau0 = theta0.norm().item() / speed0
    t_min = 1e-3 * min(tau0, c["t_max"])
    n_log = int(np.ceil(c["n_per_decade"] * np.log10(c["t_max"] / t_min)))
    grid = log_time_grid(t_min, c["t_max"], n_log)
    wall0 = time.time()
    method = "dopri5" if c["method"] == "two_phase" else c["method"]
    traj = integrate_gf(problem, theta0, c["t_max"], log_times=grid, rtol=c["rtol"],
                        atol=c["atol"], recorder=rec, method=method,
                        max_steps=c.get("max_steps", 10**7))
    traj.meta.update(case=c, name=name, tau0=tau0, order=rp.order.tolist(),
                     wall_total=time.time() - wall0)
    traj.save(path)
    return {"name": name, "status": traj.meta.get("stop_reason"), "wall": time.time() - wall0}


def _worker(args):
    c, out_dir = args
    try:
        return run_case(c, out_dir)
    except Exception:
        return {"name": case_name(c), "status": "error", "trace": traceback.format_exc()}


def main(config_path: str, out_dir: str, workers: int) -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = expand(config)
    print(f"{len(cases)} cases -> {out}", flush=True)
    with Pool(workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_worker, [(c, out) for c in cases]):
            print(json.dumps(res), flush=True)
            with open(out / "runs.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
