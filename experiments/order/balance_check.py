"""Check (literature critique item 3): the leading-order escape direction is balanced.

Integrates the leading-order flow d theta/dt = grad g (one component, unit init) until |theta| has
grown by 1e4 and reports, per core, ||G_v||^2 / ||theta||^2 (prediction 1/K for a balanced KKT
point) and the largest per-bond imbalance ||Q_e|| / ||theta||^2.
Usage: uv run python experiments/order/balance_check.py  -> results/balance_check.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).parent))
from common import make_spec, make_teacher, single_component, unit_init  # noqa: E402
from ttn.network import conserved_quantities  # noqa: E402

rows = []
for kind, n, d, chi in [("binary", 4, 2, 4), ("mps", 4, 2, 4), ("mps", 6, 4, 8), ("binary", 8, 2, 4),
                        ("mps", 8, 2, 8)]:
    spec = make_spec(kind, n, d, chi)
    teacher = single_component(make_teacher(n, d, [1.0, 0.8]), 0)
    for seed in range(5):
        u0 = unit_init(spec, seed)

        def rhs(t, y):
            f = torch.from_numpy(y).requires_grad_(True)
            (g,) = torch.autograd.grad(teacher.inner_ttn(spec, spec.unflatten(f)), f)
            return g.numpy()

        def ev(t, y):
            return np.linalg.norm(y) - 1e4 * np.linalg.norm(u0)

        ev.terminal = True
        sol = solve_ivp(rhs, (0, 1e12), u0, method="RK45", rtol=1e-10, atol=1e-14, events=ev)
        y = torch.from_numpy(sol.y[:, -1])
        r2 = float(y @ y)
        cores = spec.unflatten(y)
        frac = [float((c**2).sum()) / r2 for c in cores]
        q = conserved_quantities(spec, cores)
        rows.append(dict(tree=f"{kind}{n}", d=d, chi=chi, seed=seed, K=spec.n_cores,
                         max_dev_from_1_over_K=max(abs(f - 1 / spec.n_cores) for f in frac),
                         max_rel_imbalance=max(float(torch.linalg.matrix_norm(x, 2)) for x in q.values()) / r2))
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / "results" / "balance_check.csv", index=False, float_format="%.3g")
print(df.groupby("tree")[["max_dev_from_1_over_K", "max_rel_imbalance"]].max())
