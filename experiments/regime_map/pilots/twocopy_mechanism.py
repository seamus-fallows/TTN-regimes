"""Throwaway: why two-copy runs at small sigma have a sigma-independent delay between events 1
and 2. Records each copy's overlaps a_r^+ = <W(theta+), U_r>/s_r and a_r^- = <W(theta-), U_r>/s_r."""
import sys
import numpy as np
import torch
from scipy.integrate import solve_ivp
from ttn.losses import population_loss
from ttn.network import TTNSpec, _block_index, antisymmetric_double, forward, sum_difference_cores
from ttn.teachers import odeco_teacher
from ttn.tree import make_tree

torch.set_num_threads(1)
chi, sigma, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
spec = TTNSpec.uniform(make_tree("binary", 4), 3, chi, cap=False)
base = odeco_teacher(4, 3, [3.0, 2.0, 1.0], seed=1000 + seed)
c0 = spec.init(sigma, seed=seed)
spec2, cores2, mask = antisymmetric_double(spec, c0)
b2 = base.norm2()
u = [base.factors[:, i, :] for i in range(4)]


def split(flat):
    cores = spec2.unflatten(flat)
    t = spec.tree
    plus, minus = [], []
    for v in range(t.n_nodes):
        kids = [spec.bond_dims[c] for c in t.children[v]]
        own = None if v == t.root else spec.bond_dims[v]
        n_phys = len(t.sites[v])
        plus.append(cores[v][_block_index(n_phys, kids, [0] * len(kids), own, 0)])
        m = cores[v][_block_index(n_phys, kids, [1] * len(kids), own, 1)]
        minus.append(-m if own is None else m)
    return plus, minus


def rhs(t, y):
    flat = torch.from_numpy(y).requires_grad_(True)
    L = population_loss(spec2, sum_difference_cores(spec2, spec2.unflatten(flat)), base, teacher_norm2=b2)
    return -(torch.autograd.grad(L, flat)[0] * mask).numpy()


T = 4.0 / sigma
ts = np.linspace(0, T, 400)
sol = solve_ivp(rhs, (0, T), spec2.flatten(cores2).numpy(), t_eval=ts, rtol=1e-9, atol=1e-14)
for k in range(0, len(sol.t), 8):
    p, m = split(torch.from_numpy(sol.y[:, k]))
    ap = (forward(spec, p, u) / base.weights).numpy()
    am = (forward(spec, m, u) / base.weights).numpy()
    print(f"t={sol.t[k]:8.2f}  a+={np.round(ap, 3)}  a-={np.round(am, 3)}  a={np.round(ap - am, 3)}")
