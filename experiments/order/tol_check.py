"""Integrator accuracy check for the first-event runs: default tolerances (rtol 1e-9, atol 1e-14, as in
the sweeps) vs tighter ones (rtol 1e-11, atol 1e-16) on the same inits; winner agreement, relative
difference of t1, and the conservation-law drift max_e |Q_e(t1) - Q_e(0)| / max_v |G_v(t1)|_F^2.

Usage: uv run python experiments/order/tol_check.py [--procs N]  -> results/tol_check.csv
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).parent))
from common import RES, default_procs, loss_grad_fn, make_spec, make_teacher, overlaps, single_thread, unit_init  # noqa: E402
from ttn.network import conserved_quantities  # noqa: E402

SETTINGS = [("binary", 4, 2, 4), ("mps", 4, 2, 4), ("mps", 6, 2, 4), ("binary", 8, 2, 4),
            ("mps", 8, 2, 4)]
RHO, SIGMA, SEEDS = 0.8, 0.01, 100


def first_event(spec, teacher, y0, rtol, atol):
    _, rhs = loss_grad_fn(spec, teacher)

    def ev(t, y):
        return overlaps(spec, teacher, y).max() - 0.5

    ev.terminal, ev.direction = True, 1
    # clock restart when the step size reaches the float64 resolution of t (as common.gf_first_event)
    offset, y = 0.0, y0
    for _ in range(21):
        sol = solve_ivp(rhs, (0, 1e20 - offset), y, method="RK45", rtol=rtol, atol=atol, events=ev)
        if sol.status != -1 or sol.t[-1] == 0.0:
            break
        offset, y = offset + sol.t[-1], sol.y[:, -1]
    return offset + sol.t[-1], sol.y[:, -1]


def drift(spec, y0, y1):
    q0 = conserved_quantities(spec, spec.unflatten(torch.from_numpy(y0)))
    c1 = spec.unflatten(torch.from_numpy(y1))
    q1 = conserved_quantities(spec, c1)
    scale = max(float((c**2).sum()) for c in c1)
    return max(float((q1[v] - q0[v]).abs().max()) for v in q0) / scale


def job(args):
    kind, n, d, chi, seed = args
    single_thread()
    spec = make_spec(kind, n, d, chi)
    teacher = make_teacher(n, d, [1.0, RHO])
    y0 = SIGMA * unit_init(spec, seed)
    t_a, y_a = first_event(spec, teacher, y0, 1e-9, 1e-14)
    t_b, y_b = first_event(spec, teacher, y0, 1e-11, 1e-16)
    return dict(tree=f"{kind}{n}", d=d, chi=chi, seed=seed, t1=t_a, t1_tight=t_b,
                rel_dt=abs(t_a - t_b) / t_b,
                winner=int(np.argmax(overlaps(spec, teacher, y_a))),
                winner_tight=int(np.argmax(overlaps(spec, teacher, y_b))),
                drift=drift(spec, y0, y_a), drift_tight=drift(spec, y0, y_b))


def main():
    procs = (int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv
             else default_procs())
    jobs = [s + (seed,) for s in SETTINGS for seed in range(SEEDS)]
    with Pool(procs) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=1))
    df.to_csv(RES / "tol_check.csv", index=False, float_format="%.6g")
    print(df.groupby("tree").agg(n=("seed", "size"),
                                 winner_agree=("winner", lambda w: np.mean(w == df.loc[w.index, "winner_tight"])),
                                 rel_dt_max=("rel_dt", "max"), drift_max=("drift", "max"),
                                 drift_tight_max=("drift_tight", "max")).to_string())


if __name__ == "__main__":
    main()
