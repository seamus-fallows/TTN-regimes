"""Empirical SGD stability limit at the balanced global minimum (binary N=4, teacher [3, 2, 1]).

Start 1e-5 (Gaussian perturbation of every parameter) off the balanced aligned minimum and run online
SGD (`ttn.dynamics.run_sgd`) at eta = ratio x eta*_ms(B), the exact mean-square limit of
`sgd_stability.py` (0.00107 at B = 1, 0.0161 at B = 16). Outcome after the run: converged (loss below
its starting value), diverged (non-finite or > 1e3), or neither. Theory note P8: almost-sure
convergence up to eta_as ~ 1.9 eta*_ms (B = 1), 1.9 (B = 16).

Run: uv run python experiments/optim_samples/sgd_limit_test.py [--workers 4] [--out-dir DIR]
Writes sgd_limit_test.csv.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from common import RESULTS, setup
from ttn.dynamics import Problem, Recorder, run_sgd
from ttn.network import aligned_init

ETA_MS = {1: 0.0010702, 16: 0.0160597}
RATIOS = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
T_MAX = 25.0


def run(task):
    torch.set_num_threads(1)
    batch, ratio, seed = task
    spec, teacher = setup("binary", 4)
    prob = Problem(spec, teacher)
    amps = teacher.weights.unsqueeze(0).expand(spec.n_cores, 3) ** (1 / spec.n_cores)
    theta = spec.flatten(aligned_init(spec, teacher.factors, amps.clone()))
    gen = torch.Generator().manual_seed(40_000 + seed)
    theta = theta + 1e-5 * torch.randn(theta.shape, generator=gen)
    loss0 = prob.loss(theta).item()
    eta = ratio * ETA_MS[batch]
    n_steps = int(T_MAX / eta)
    traj = run_sgd(prob, theta, eta, n_steps, batch, gen, diverge_factor=1e3 / loss0,
                   log_steps=np.linspace(0, n_steps, 200),
                   recorder=Recorder(spectra=False, overlaps=False, conservation=False))
    loss_end = traj.meta["loss_end"]
    if traj.meta["stop_reason"] == "diverged" or not np.isfinite(loss_end):
        outcome = "diverged"
    elif loss_end < loss0:
        outcome = "converged"
    else:
        outcome = "neither"
    return dict(B=batch, ratio=ratio, eta=eta, seed=seed, loss0=loss0, loss_end=loss_end,
                max_loss=float(np.nanmax(traj["loss"])), outcome=outcome,
                wall=traj.meta["wall_time"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_known_args()[0]
    tasks = [(b, r, s) for b in ETA_MS for r in RATIOS for s in range(5)]
    with Pool(args.workers) as pool:
        rows = pool.map(run, tasks)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sgd_limit_test.csv", index=False)
    print(df.groupby(["B", "ratio"]).outcome.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
