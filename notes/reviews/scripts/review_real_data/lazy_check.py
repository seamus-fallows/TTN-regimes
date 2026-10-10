"""Independent linearized-model check for a centred (lazy) run: J at theta0 by torch.func (library forward
only), closed-form L_lin(t) at the run's total time, compared with the run's final train loss.
Usage: uv run python -I lazy_check.py <run.npz> <indep.py dir>"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import torch

sys.path.insert(0, sys.argv[2])
from indep import Model  # own data pipeline  # noqa: E402

from ttn.network import forward  # noqa: E402

torch.set_num_threads(1)
f = np.load(sys.argv[1])
meta = json.loads(str(f["__meta__"]))
c = meta["case"]
m = Model(c["tree"], c["ordering"], c["sigma"], c["seed"])
theta0 = torch.as_tensor(f["theta0"])
assert np.allclose(theta0.numpy(), m.theta0), "init mismatch"
spec = m.spec


def fx(th):
    return forward(spec, spec.unflatten(th), m.X)


J = torch.func.jacrev(fx)(theta0).numpy()  # (n, P)
y = m.Y.numpy()
n = len(y)
u, s, _ = np.linalg.svd(J, full_matrices=False)
keep = s > 1e-10 * s.max()
u, s = u[:, keep], s[keep]
proj = u.T @ y
floor = 0.5 * (y @ y - proj @ proj) / n
t_total = meta["t_a"] + float(f["p2_t"][-1])
L_lin = 0.5 * np.sum(np.exp(-2 * s**2 * t_total / n) * proj**2) / n + floor
# the run's own loss: centred model fits f_theta - f_theta0
with torch.no_grad():
    th_end = torch.as_tensor(f["p2_theta_end"])
    g = fx(th_end) - fx(theta0)
    L_run = 0.5 * ((g - m.Y) ** 2).mean().item()
print(json.dumps({"sigma": c["sigma"], "seed": c["seed"], "t_a": meta["t_a"], "t2": meta.get("t2"), "t_total": t_total,
                  "rank_J": int(keep.sum()), "P": int(J.shape[1]), "L_lin_floor": float(floor), "L_lin_T": float(L_lin),
                  "L_run_T_recomputed": L_run, "L_run_T_logged": float(f["p2_loss"][-1]),
                  "delta_K_end": float(f["p2_delta_K"][-1, 0]) if "p2_delta_K" in f.files else None}))
