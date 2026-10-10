"""Independent check of the fan-in GF stall on MPS N=8 (claim 6): is the dormant component's death an
integrator-tolerance artifact? Fan-in init (library), my dense loss, scipy LSODA at two tolerances.

    uv run python indep_stall.py SIGMA SEED T_MAX OUTCSV
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.integrate
import torch

sys.path.insert(0, str(Path(__file__).parent))
from indep_check import Prob  # noqa: E402

torch.set_num_threads(1)


def run(p, th0, t_max, rtol, atol, grid):
    def f(t, y):
        _, g = p.loss_grad(torch.from_numpy(y))
        return -g.numpy()

    t0 = time.time()
    sol = scipy.integrate.solve_ivp(
        f, (0, t_max), th0.numpy(), method="LSODA", rtol=rtol, atol=atol, t_eval=grid
    )
    a = np.array([p.overlaps(torch.from_numpy(y)) for y in sol.y.T])
    L = np.array([p.loss(torch.from_numpy(y)).item() for y in sol.y.T])
    return sol.t, a, L, sol.nfev, time.time() - t0


if __name__ == "__main__":
    import pandas as pd

    sigma, seed, t_max, out = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    p = Prob("mps", 8, 3, 4, [3.0, 2.0, 1.0])
    th0 = p.init(sigma, seed)
    grid = np.unique(np.concatenate([[0.0], np.geomspace(1e-2, t_max, 400)]))
    rows = []
    for rtol, atol in [(1e-9, 1e-14), (1e-11, 1e-30)]:
        t, a, L, nfev, wall = run(p, th0, t_max, rtol, atol, grid)
        for i in range(len(t)):
            rows.append(
                dict(rtol=rtol, atol=atol, t=t[i], a0=a[i, 0], a1=a[i, 1], a2=a[i, 2], loss=L[i])
            )
        print(
            f"rtol {rtol} atol {atol}: nfev {nfev} wall {wall:.1f}s  final a = {a[-1]}, loss {L[-1]:.4g}; "
            f"min |a| over t>10: {np.abs(a[t > 10]).min(axis=0)}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out, index=False)
