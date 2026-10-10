"""Gradient flow from theta_S + sigma * xi (xi Gaussian, unit-variance entries): escape time vs sigma.

Theorem 2 / Corollary 3 predict that where the saddle order drops to 2l < k*, the escape time scales
as sigma^-(2l-2) instead of the route law sigma^-(k*-2). Escape = first time L - L_S < -1e-3.
Also records the amplitude of the non-residual pattern 110000 at escape (generalized-piggyback signature).

Usage: uv run python experiments/theory_checks/saddle_order/gf_escape.py CASE [CASE ...]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).parent))
from saddle_order import CASES, OUT, Instance  # noqa: E402
from ttn.network import to_dense  # noqa: E402

SIGMAS = [0.04, 0.02, 0.01, 0.005]


def escape(inst, sigma, seed, t_max=1e8):
    rng = np.random.default_rng(1000 + seed)
    y0 = (inst.theta_S + sigma * torch.from_numpy(rng.standard_normal(inst.theta_S.numel()))).numpy()

    def grad(t, y):
        th = torch.from_numpy(y).requires_grad_(True)
        val = inst.excess(th - inst.theta_S)
        (g,) = torch.autograd.grad(val, th)
        return -g.numpy()

    def event(t, y):
        return inst.excess(torch.from_numpy(y) - inst.theta_S).item() + 1e-3

    event.terminal = True
    sol = solve_ivp(grad, (0, t_max), y0, method="LSODA", rtol=1e-10, atol=1e-14, events=event)
    t_esc = float(sol.t_events[0][0]) if sol.t_events[0].size else None
    W = to_dense(inst.spec, inst.spec.unflatten(torch.from_numpy(sol.y[:, -1])))
    spur = W[(1, 1) + (0,) * (W.ndim - 2)].item()  # pattern 1100..0 (not a residual)
    return t_esc, spur


if __name__ == "__main__":
    for name in sys.argv[1:]:
        inst = Instance(**CASES[name])
        rows = []
        for seed in range(3):
            for sigma in SIGMAS:
                t0 = time.time()
                t_esc, spur = escape(inst, sigma, seed)
                rows.append({"seed": seed, "sigma": sigma, "t_esc": t_esc, "amp_110000": spur})
                print(f"{name} seed={seed} sigma={sigma} t_esc={t_esc} amp(1100..)={spur:+.3f} ({time.time()-t0:.0f}s)", flush=True)
        (OUT / f"gf_{name}.json").write_text(json.dumps(rows, indent=1))
