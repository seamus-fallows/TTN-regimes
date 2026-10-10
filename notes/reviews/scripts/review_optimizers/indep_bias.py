"""Independent finite-sample check (claim 5): binary4, d=3, chi=4, odeco (3,2,1), n=45 Gaussian samples drawn
here (numpy RNG, not the agent's make_dataset), my empirical loss via dense features, GF (scipy LSODA) and my Adam
(lr 1e-3). Stop at empirical loss < 1e-8 L0. Report bond ranks (#s > 0.1), population loss, Hessian ratio.
    uv run python indep_bias.py OUTCSV"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.integrate
import torch

sys.path.insert(0, str(Path(__file__).parent))
from indep_check import Prob  # noqa: E402

torch.set_num_threads(1)
p = Prob("binary", 4, 3, 4, [3.0, 2.0, 1.0])
N, D = 4, 3


def features(x):  # x (n, N, d) -> (n, d^N) with site order 0..N-1
    f = x[:, 0, :]
    for i in range(1, N):
        f = (f[:, :, None] * x[:, i, None, :]).reshape(len(x), -1)
    return f


def emp_loss(theta, phi, y):
    w = p.W(theta).reshape(-1)
    return 0.5 * ((phi @ w - y) ** 2).mean()


def grad(theta, phi, y):
    th = theta.detach().clone().requires_grad_(True)
    L = emp_loss(th, phi, y)
    (g,) = torch.autograd.grad(L, th)
    return L.item(), g


def ranks(theta):
    w = p.W(theta).detach()
    out = []
    for v in p.spec.tree.edges:
        rows = sorted(p.spec.tree.subtree_sites[v])
        cols = [s for s in range(N) if s not in rows]
        s = torch.linalg.svdvals(w.permute(rows + cols).reshape(D ** len(rows), -1))
        out.append(int((s > 0.1).sum()))
    return "".join(map(str, out))


rows = []
for seed in range(3):
    rng = np.random.default_rng(1000 + seed)
    x = torch.from_numpy(rng.standard_normal((45, N, D)))
    phi = features(x)
    y = phi @ p.wstar.reshape(-1)
    for sigma in (0.01, 0.1):
        th0 = p.init(sigma, seed)
        L0 = emp_loss(th0, phi, y).item()
        t0 = time.time()

        def f(t, z):
            return -grad(torch.from_numpy(z), phi, y)[1].numpy()

        def ev(t, z):
            return np.log(emp_loss(torch.from_numpy(z), phi, y).item() / (1e-8 * L0))

        ev.terminal = True
        sol = scipy.integrate.solve_ivp(
            f, (0, 1e6), th0.numpy(), method="LSODA", rtol=1e-9, atol=1e-14, events=ev
        )
        th_gf = torch.from_numpy(sol.y[:, -1])
        # Adam
        th = th0.clone()
        m = torch.zeros_like(th)
        v = torch.zeros_like(th)
        for n in range(1, 300001):
            L, g = grad(th, phi, y)
            if L < 1e-8 * L0:
                break
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            th = th - 1e-3 * (m / (1 - 0.9**n)) / ((v / (1 - 0.999**n)).sqrt() + 1e-8)
        for meth, tt, info in (("gf", th_gf, sol.status), ("adam", th, n)):
            r = dict(
                seed=seed,
                sigma=sigma,
                method=meth,
                train_rel=emp_loss(tt, phi, y).item() / L0,
                pop_loss=p.loss(tt).item(),
                ranks=ranks(tt),
                info=info,
            )
            rows.append(r)
            print(r, f"{time.time() - t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
