"""Stochastic collapse test (Chen, Kunin et al. 2023) with ttn.dynamics.run_sgd.

Binary N=4, d=3, chi=3, pattern teacher [3, s2] with patterns [[0,0,0,0],[1,1,0,0]]: the second
component differs from the first only on sites 0, 1, so after component 1 is learned it escapes
through k = 2 cores (bottom-left core gets a new bond channel, root a new entry): a strict saddle
with growth rate lambda ~ q s2, q = s1^{1/3} (learned amplitude of the third core).

Prediction (notes/optim_samples.md section 4): per SGD step the dormant amplitude is multiplied by
1 + eta q xi with E xi = s2 and E xi^2 = s2^2 + (V + zeta^2)/B (V: residual product-feature
variance, zeta: label-noise std), so its Lyapunov exponent eta q s2 - eta^2 q^2 E xi^2 / 2 is negative,
and the component is never learned, iff about s2 < eta q (V + zeta^2) / (2B). The dormant direction's
feature is Z_2 = prod_i <u_2^(i), x_i> itself, so V = s2^2 (E Z^4 - 1) = 80 s2^2: without label noise
collapse needs s2 > 2B / (80 eta q) ~ 35 at eta = 5e-4, i.e. never; with label noise components with
s2 < ~eta q zeta^2 / 2 collapse (0.0032, 0.036, 0.32 for zeta = 3, 10, 30).

Run: uv run python experiments/optim_samples/collapse_test.py [--workers 15]
Raw: /root/outputs/optim_samples/collapse/*.json; summary: results/collapse_test.csv
"""

import argparse
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import RAW, TEACHER_SEED
from ttn.diagnostics import crossing_times
from ttn.dynamics import Problem, Recorder, integrate_gf, run_sgd
from ttn.network import TTNSpec
from ttn.teachers import pattern_teacher
from ttn.tree import make_tree

OUT = RAW / "collapse"
RES = Path(__file__).parent / "results"
SIGMA, ETA, BATCH, T_MAX = 0.1, 5e-4, 1, 120.0


def problem(s2):
    spec = TTNSpec.uniform(make_tree("binary", 4), 3, 3)
    teacher = pattern_teacher([3.0, s2], [[0, 0, 0, 0], [1, 1, 0, 0]], 3, seed=TEACHER_SEED)
    return spec, Problem(spec, teacher)


def run(task):
    torch.set_num_threads(1)
    s2, zeta, noise_seed = task
    path = OUT / f"s2{s2}_zeta{zeta}_noise{noise_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    spec, prob = problem(s2)
    theta0 = spec.init(SIGMA, seed=0)
    if noise_seed < 0:  # gradient-flow reference
        traj = integrate_gf(prob, theta0, T_MAX, method="lsoda", log_times=np.linspace(0.05, T_MAX, 2400),
                            recorder=Recorder(spectra=False))
    else:
        n_steps = int(T_MAX / ETA)
        gen = torch.Generator().manual_seed(20_000 + noise_seed)
        traj = run_sgd(prob, theta0, ETA, n_steps, BATCH, gen, label_noise=zeta,
                       log_steps=np.arange(0, n_steps + 1, 100), recorder=Recorder(spectra=False))
    a = traj["overlaps"]
    out = dict(s2=s2, zeta=zeta, noise_seed=noise_seed, t50=crossing_times(traj["t"], a, 0.5).tolist(),
               a_end=a[-1].tolist(), a2_max=float(np.nanmax(a[:, 1])),
               loss_end=float(traj["pop_loss"][-1]), stop=traj.meta.get("stop_reason"),
               wall=traj.meta["wall_time"])
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=15)
    args = ap.parse_args()
    s2s, zetas, seeds = [1.0, 0.3, 0.1], [0.0, 3.0, 10.0, 30.0], [0, 1, 2, 3, 4]
    tasks = [(s2, 0.0, -1) for s2 in s2s] + list(itertools.product(s2s, zetas, seeds))
    with Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(run, tasks))
    df = pd.DataFrame(rows)
    q = 3.0 ** (1 / 3)
    df["pred_threshold_s2"] = ETA * q * (81 * df.s2**2 + df.zeta**2) / (2 * BATCH)
    df["learned2"] = [t[1] == t[1] for t in df.t50]  # not NaN
    RES.mkdir(exist_ok=True)
    df.sort_values(["s2", "zeta", "noise_seed"]).to_csv(RES / "collapse_test.csv", index=False)
    print(df.groupby(["s2", "zeta"])[["learned2", "pred_threshold_s2", "a2_max"]].mean())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
