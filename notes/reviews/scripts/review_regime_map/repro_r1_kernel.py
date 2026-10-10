"""R1: probe-set vs tensor-space kernel change at the measured lazy boundary (binary N=4, d=3, centred, SP).

Replicates run_regime.run_point's integration (same build, budget, stop floor, probes) without writing
into experiments/. Outputs one JSON line per run to data/review_regime_map/r1_kernel.jsonl.
"""

import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
REPO = "c:/Users/seamu/Iliad-ML/TTN_regimes"
sys.path.insert(0, REPO + "/experiments/regime_map")
sys.path.insert(0, REPO)
import run_regime as rr  # noqa: E402

from ttn import diagnostics as dg  # noqa: E402
from ttn.dynamics import Recorder, integrate_gf  # noqa: E402

OUTF = REPO + "/data/review_regime_map/r1_kernel.jsonl"


def one(chi, seed, sigma):
    cfg = dict(
        exp="r1",
        kind="binary",
        n=4,
        d=3,
        chi=chi,
        cap=False,
        model="centred",
        sigma=sigma,
        seed=seed,
        l_floor=1e-6,
    )
    tic = time.time()
    problem, theta0, spec, info = rr.build(cfg)
    K = spec.n_cores
    s_min = 1.0
    t_esc = (0.3 * sigma) ** (-(K - 2)) / ((K - 2) * s_min)
    t_max = 50 * t_esc + 1e4 / max(info["lam_bar"], 1.0)
    L_ref = 0.5 * float(info["base"].norm2())
    traj = integrate_gf(
        problem,
        theta0,
        t_max,
        n_log=300,
        loss_floor=1e-6 * L_ref,
        max_steps=100_000,
        recorder=Recorder(spectra=False, conservation=False),
        rtol=1e-8,
        atol=1e-12,
        method="dopri5",
    )
    th_end = torch.from_numpy(traj["theta_end"])
    probes = torch.randn(
        rr.N_PROBE,
        4,
        3,
        generator=torch.Generator().manual_seed(rr.PROBE_SEED),
        dtype=torch.float64,
    )
    w = problem.precond
    dK_probe = dg.kernel_change(
        dg.tangent_kernel(spec, th_end, probes, w), dg.tangent_kernel(spec, theta0, probes, w)
    )
    T0 = dg.tensor_tangent_kernel(spec, theta0, w)
    T1 = dg.tensor_tangent_kernel(spec, th_end, w)
    dK_tensor = dg.kernel_change(T1, T0)
    x = math.sqrt(14.0) / sigma**3
    c1, c2 = 341.5 / chi**3, 537.0 / chi**3
    row = dict(
        chi=chi,
        seed=seed,
        sigma=sigma,
        x=x,
        l_end=float(traj["loss"][-1] / L_ref),
        stop=traj.meta["stop_reason"],
        dK_probe=dK_probe,
        dK_tensor=dK_tensor,
        ratio=dK_probe / dK_tensor,
        theory_xi0=math.sqrt((c1 * x) ** 2 + (c2 * x * x) ** 2),
        theory_lo=math.sqrt(max((c1 * x) ** 2 - 2 * c1 * c2 * x**3 + (c2 * x * x) ** 2, 0)),
        theory_hi=math.sqrt((c1 * x) ** 2 + 2 * c1 * c2 * x**3 + (c2 * x * x) ** 2),
        wall=time.time() - tic,
    )
    print(json.dumps(row), flush=True)
    with open(OUTF, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    import pandas as pd

    b = pd.read_csv(REPO + "/experiments/regime_map/results/boundaries_per_seed_chi_b4.csv")
    b = b[(b.model == "centred")]
    jobs = []
    for chi in (32, 128, 64, 16):
        for seed in (0, 1):
            s = float(b[(b.chi == chi) & (b.seed == seed)].sigma_K01.iloc[0])
            jobs.append((chi, seed, s))
    for chi, seed, s in jobs:
        one(chi, seed, s)
