"""Hessian check of the k = 2 route used by collapse_test.py (MPS N=4, pattern teacher [3, 0.3]).

Gradient flow (lsoda) from fan-in init, sigma in {0.1, 0.07, 0.05, 0.03, 0.02, 0.01}, seeds 0-4; at the
midpoint of the second plateau (minimum speed between the two events) the gauge-projected Hessian
spectrum is computed with `ttn.diagnostics`. Saddle type by the sweep-level rule of
notes/diagnostics.md: a strict saddle has lambda_min -> a negative sigma-independent constant as
sigma -> 0 (prediction for a k = 2 route: -s2 * s1^{(K-2)/K} = -0.5196); raw and projected values
are both reported; a higher-order saddle has
lambda_min -> 0 like sigma^(k-2).

Run: uv run python experiments/optim_samples/check_k2_saddle.py [--workers 15] [--out-dir DIR]
Writes k2_saddle_hessian.json (sigma = 0.1, seed 0, full detail) and k2_saddle_sweep.csv.
"""

import argparse
import os
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from common import RESULTS, TEACHER_SEED
from ttn.diagnostics import (
    crossing_times,
    gauge_directions,
    hessian,
    plateau_midpoints,
    project_out_gauge,
    saddle_type,
)
from ttn.dynamics import Problem, Recorder, integrate_gf
from ttn.network import TTNSpec
from ttn.teachers import pattern_teacher
from ttn.tree import make_tree


def plateau_lambda_min(prob, spec, sigma, seed):
    t_max = 20.0 / sigma**2  # first escape ~ sigma^{-(K-2)} = sigma^-2 for K = 4
    traj = integrate_gf(prob, spec.init(sigma, seed=seed), t_max, method="lsoda",
                        log_times=np.unique(np.concatenate([np.geomspace(1e-2, t_max, 1500),
                                                            np.linspace(t_max / 3000, t_max, 3000)])),
                        recorder=Recorder(spectra=False, save_params=True), loss_floor=1e-6)
    events = crossing_times(traj["t"], traj["overlaps"], 0.5)
    if not np.all(np.isfinite(events)):
        return dict(sigma=sigma, seed=seed, events=events.tolist(), lambda_min=np.nan)
    # Second pass with a dense grid around the two events, so the plateau midpoint is resolved
    # (the second plateau is only ~5-15 time units long while the first lasts ~sigma^-2).
    t1, t2 = np.sort(events)
    lo, hi = max(t1 - 0.5 * (t2 - t1) - 1.0, 0.0), t2 + 0.5 * (t2 - t1) + 1.0
    traj = integrate_gf(prob, spec.init(sigma, seed=seed), hi, method="lsoda",
                        log_times=np.unique(np.concatenate([np.geomspace(1e-2, lo, 300),
                                                            np.linspace(lo, hi, 2000)])),
                        recorder=Recorder(spectra=False, save_params=True))
    events = crossing_times(traj["t"], traj["overlaps"], 0.5)
    mid = plateau_midpoints(traj["t"], traj["speed"], np.sort(events))[0]
    i = int(np.argmin(np.abs(traj["t"] - mid)))
    theta = torch.from_numpy(traj["params"][i])
    h = hessian(prob.loss, theta)
    raw = torch.linalg.eigvalsh(h)
    # Projection with the library default (rtol = 1e-3, after the fix of commit 6c42d85); the raw
    # spectrum is reported alongside (notes/diagnostics.md).
    h_proj, _ = project_out_gauge(h, gauge_directions(spec, theta))
    eigs = torch.linalg.eigvalsh(h_proj)
    return dict(sigma=sigma, seed=seed, t1=float(np.min(events)), t2=float(np.max(events)),
                midpoint=float(traj["t"][i]), a2_mid=float(traj["overlaps"][i][1]),
                lambda_min=eigs[0].item(), lambda_2=eigs[1].item(), lambda_max=eigs[-1].item(),
                lambda_min_raw=raw[0].item(), lambda_2_raw=raw[1].item(),
                n_negative_fallback=saddle_type(eigs)["n_negative"]), eigs


def sweep_point(task):
    torch.set_num_threads(1)
    spec, prob = make_problem()
    try:
        res = plateau_lambda_min(prob, spec, *task)
    except Exception as err:  # reported, not dropped (analysis standards)
        return dict(sigma=task[0], seed=task[1], lambda_min=np.nan, error=repr(err))
    return res[0] if isinstance(res, tuple) else res


def make_problem():
    spec = TTNSpec.uniform(make_tree("mps", 4), 3, 3)
    teacher = pattern_teacher([3.0, 0.3], [[0, 0, 0, 0], [1, 1, 0, 0]], 3, seed=TEACHER_SEED)
    return spec, Problem(spec, teacher)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(15, os.cpu_count() or 1))
    args = ap.parse_known_args()[0]
    # The sweep runs first: autograd must not be used in the parent before forking.
    tasks = [(sigma, seed) for sigma in (0.1, 0.07, 0.05, 0.03, 0.02, 0.01) for seed in range(5)]
    with Pool(args.workers) as pool:
        rows = pool.map(sweep_point, tasks)
    torch.set_num_threads(1)
    spec = TTNSpec.uniform(make_tree("mps", 4), 3, 3)
    teacher = pattern_teacher([3.0, 0.3], [[0, 0, 0, 0], [1, 1, 0, 0]], 3, seed=TEACHER_SEED)
    prob = Problem(spec, teacher)
    traj = integrate_gf(prob, spec.init(0.1, seed=0), 60.0, method="lsoda",
                        log_times=np.linspace(0.05, 60, 1200),
                        recorder=Recorder(spectra=False, save_params=True))
    events = crossing_times(traj["t"], traj["overlaps"], 0.5)
    mid = plateau_midpoints(traj["t"], traj["speed"], np.sort(events))[0]
    i = int(np.argmin(np.abs(traj["t"] - mid)))
    theta = torch.from_numpy(traj["params"][i])
    h = hessian(prob.loss, theta)
    h_proj, _ = project_out_gauge(h, gauge_directions(spec, theta))
    eigs = torch.linalg.eigvalsh(h_proj)
    out = dict(lowest_raw_eigs=torch.linalg.eigvalsh(h)[:4].tolist(),events=events.tolist(), midpoint=float(traj["t"][i]),
               overlaps_at_midpoint=traj["overlaps"][i].tolist(),
               lowest_eigs=eigs[:4].tolist(), top_eig=eigs[-1].item(),
               saddle=saddle_type(eigs), predicted_lambda_min=-0.3 * 3.0**0.5)
    (RESULTS / "k2_saddle_hessian.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "k2_saddle_sweep.csv", index=False)
    print(df.to_string())
    print(df.groupby("sigma")[["lambda_min", "lambda_min_raw"]].agg(["mean", "sem", "count"]))


if __name__ == "__main__":
    main()
