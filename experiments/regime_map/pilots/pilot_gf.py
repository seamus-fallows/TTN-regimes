"""Throwaway pilot (before ttn/dynamics.py lands): gradient flow on the centred model with
scipy solve_ivp, to see the sigma x chi landscape for binary N=4 (d=3, odeco [3,2,1])."""
import math
import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ttn.losses import population_loss
from ttn.network import TTNSpec, antisymmetric_double, forward
from ttn.teachers import ShiftedTeacher, odeco_teacher
from ttn.tree import make_tree

torch.set_num_threads(1)


def run(kind="binary", n=4, d=3, chi=3, sigma=0.1, seed=0, weights=(3.0, 2.0, 1.0), cap=True,
        t_max=None, n_eval=400, rtol=1e-9, atol=1e-14, n_probe=128, construction="centred"):
    tree = make_tree(kind, n)
    spec = TTNSpec.uniform(tree, d, chi, cap=cap)
    K = spec.n_cores
    base = odeco_teacher(n, d, list(weights), seed=1000 + seed)
    cores0 = spec.init(sigma, seed=seed)
    mask = None
    if construction == "centred":
        teacher = ShiftedTeacher(base, spec, [c.clone() for c in cores0])
    else:  # two-copy: W(theta+) - W(theta-), masked gradient, W_0 = 0
        spec, cores0, mask = antisymmetric_double(spec, cores0)
        teacher = base
    t2 = teacher.norm2()
    flat0 = spec.flatten(cores0).clone()
    # component directions: overlaps of W - W0 with each rank-one term
    u = [base.factors[:, i, :] for i in range(n)]
    ov0 = forward(spec, cores0, u)

    def loss_grad(flat_np):
        flat = torch.from_numpy(flat_np).requires_grad_(True)
        L = population_loss(spec, spec.unflatten(flat), teacher, teacher_norm2=t2)
        (g,) = torch.autograd.grad(L, flat)
        if mask is not None:
            g = g * mask
        return L.item(), g.numpy()

    def rhs(t, y):
        return -loss_grad(y)[1]

    # time scales: escape ~ sigma^-(K-2), lazy ~ sigma^-2(K-1)
    if t_max is None:
        t_max = 30.0 * (sigma ** (-(K - 2)) if sigma < 1 else 30 * sigma ** (-2 * (K - 1)))
    t0 = 1e-3 * min(1.0, sigma ** (-2 * (K - 1)))
    t_eval = np.concatenate([[0.0], np.geomspace(t0, t_max, n_eval)])
    tic = time.time()
    sol = solve_ivp(rhs, (0, t_max), flat0.numpy(), method="RK45", t_eval=t_eval, rtol=rtol, atol=atol)
    wall = time.time() - tic
    Ls, A = [], []
    for k in range(sol.y.shape[1]):
        cores = spec.unflatten(torch.from_numpy(sol.y[:, k]))
        Ls.append(population_loss(spec, cores, teacher, teacher_norm2=t2).item())
        A.append(((forward(spec, cores, u) - ov0) / base.weights).numpy())
    Ls, A = np.array(Ls), np.array(A)

    # tangent kernel on probes
    g = torch.Generator().manual_seed(12345)
    probes = torch.randn(n_probe, n, d, generator=g, dtype=torch.float64)

    def kernel(flat):
        J = torch.func.jacrev(lambda p: forward(spec, spec.unflatten(p), probes))(flat)
        if mask is not None:
            J = J * mask
        return J @ J.T

    th0 = flat0
    thT = torch.from_numpy(sol.y[:, -1])
    K0, KT = kernel(th0), kernel(thT)
    dK = ((KT - K0).norm() / K0.norm()).item()
    dth = ((thT - th0).norm() / th0.norm()).item()
    return dict(t=sol.t, L=Ls, A=A, dK=dK, dth=dth, K=K, wall=wall, nfev=sol.nfev, status=sol.status,
                n_params=spec.n_params)


def events(t, A, lo=0.1, mid=0.5, hi=0.9):
    out = []
    for r in range(A.shape[1]):
        a = A[:, r]
        def cross(level):
            idx = np.nonzero(a >= level)[0]
            if len(idx) == 0:
                return np.nan
            i = idx[0]
            if i == 0:
                return t[0]
            # linear interpolation in t
            return t[i - 1] + (level - a[i - 1]) / (a[i] - a[i - 1]) * (t[i] - t[i - 1])
        out.append((cross(mid), cross(hi) - cross(lo)))
    return out


if __name__ == "__main__":
    chi = int(sys.argv[1]); sigma = float(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    r = run(chi=chi, sigma=sigma, seed=seed)
    ev = sorted(events(r["t"], r["A"]))
    print(f"chi={chi} sigma={sigma} K={r['K']} P={r['n_params']} wall={r['wall']:.1f}s nfev={r['nfev']} "
          f"Lfin/L0={r['L'][-1]/r['L'][0]:.2e} dK={r['dK']:.3f} dth={r['dth']:.3f}")
    print("  events (t, width):", [(f"{a:.3g}", f"{b:.3g}") for a, b in ev])
    if len(ev) > 1:
        rho = [(ev[i+1][0]-ev[i][0])/max(ev[i][1], ev[i+1][1]) for i in range(len(ev)-1)]
        print("  rho:", [f"{x:.2f}" for x in rho])
