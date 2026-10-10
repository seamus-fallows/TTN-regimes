"""Independent check (no agent code, no ttn library): Hessian of L = 1/2||W - W*||^2 at the
'wrong' saddle B (smaller component learned, balanced, amplitude q = s2^(1/K) per core, bond index 0)
for binary4 (= A R B^T) and MPS chains with pattern / odeco teachers in the standard basis.
Gauge directions are null at a critical point, so the raw lambda_min is the quantity to compare."""
import math, sys
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def mps_bonds(N, d, chi):
    return [min(chi, d ** (k + 1), d ** (N - k - 1)) for k in range(N - 1)]

def mps_W(cores):
    W = cores[0]                                   # (d, b0)
    for G in cores[1:-1]:                          # (d, b_prev, b_next)
        W = torch.einsum('...a,iab->...ib', W, G)
    return torch.einsum('...a,ia->...i', W, cores[-1])   # last core (d, b_last)

def mps_shapes(N, d, chi):
    b = mps_bonds(N, d, chi)
    return [(d, b[0])] + [(d, b[k - 1], b[k]) for k in range(1, N - 1)] + [(d, b[-1])]

def b4_shapes(d, chi):
    c = min(chi, d * d)
    return [(d, d, c), (d, d, c), (c, c)]

def b4_W(cores):
    A, B, R = cores
    return torch.einsum('ija,klb,ab->ijkl', A, B, R)

def teacher(patterns, weights, N, d):
    W = torch.zeros((d,) * N)
    for p, s in zip(patterns, weights):
        W[tuple(p)] += s
    return W

def saddle(shapes, kind, pattern2, s2):
    K = len(shapes)
    q = s2 ** (1.0 / K)
    cores = [torch.zeros(s) for s in shapes]
    if kind == 'binary4':
        cores[0][pattern2[0], pattern2[1], 0] = q
        cores[1][pattern2[2], pattern2[3], 0] = q
        cores[2][0, 0] = q
    else:
        for v, G in enumerate(cores):
            idx = (pattern2[v],) + (0,) * (G.dim() - 1)
            G[idx] = q
    return cores

def hess_eigs(shapes, Wfun, Wstar, cores):
    sizes = [math.prod(s) for s in shapes]
    flat0 = torch.cat([c.reshape(-1) for c in cores])
    def loss(flat):
        cs, i = [], 0
        for s, n in zip(shapes, sizes):
            cs.append(flat[i:i + n].view(s)); i += n
        return 0.5 * ((Wfun(cs) - Wstar) ** 2).sum()
    g = torch.autograd.grad(loss(flat0.clone().requires_grad_(True)), [])  if False else None
    f = flat0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(loss(f), f)
    H = torch.autograd.functional.hessian(loss, flat0)
    ev = torch.linalg.eigvalsh(0.5 * (H + H.T)).numpy()
    return ev, grad.norm().item()

rows = []
def run(name, kind, N, d, chi, patterns, rho):
    shapes = b4_shapes(d, chi) if kind == 'binary4' else mps_shapes(N, d, chi)
    K = len(shapes)
    Wfun = b4_W if kind == 'binary4' else mps_W
    Wstar = teacher(patterns, [1.0, rho], N, d)
    cores = saddle(shapes, kind, patterns[1], rho)
    ev, gn = hess_eigs(shapes, Wfun, Wstar, cores)
    q = rho ** (1.0 / K)
    pred_rot = -(1 - rho) * q ** (K - 2)
    pred_add = -1.0 * q ** (K - 2)
    neg = ev[ev < -1e-10]
    print(f"{name:28s} K={K} rho={rho}: |grad|={gn:.1e} lam_min={ev[0]:.6f} "
          f"pred -(s1-s2)q^(K-2)={pred_rot:.6f} pred -s1 q^(K-2)={pred_add:.6f} "
          f"negatives={np.round(neg, 6).tolist()}", flush=True)
    rows.append((name, K, rho, ev[0], pred_rot, pred_add, len(neg)))

for rho in [0.5, 0.7, 0.8, 0.9]:
    run('binary4 odeco chi4', 'binary4', 4, 2, 4, [[0]*4, [1]*4], rho)
for rho in [0.5, 0.8]:
    run('mps4 odeco (m=4) chi4', 'mps', 4, 2, 4, [[0]*4, [1]*4], rho)
    run('mps4 sep pair (0,2) chi4', 'mps', 4, 2, 4, [[0]*4, [1, 0, 1, 0]], rho)
    run('mps4 adj pair (1,2) chi4', 'mps', 4, 2, 4, [[0]*4, [0, 1, 1, 0]], rho)
    run('mps6 ends pair (0,5) chi4', 'mps', 6, 2, 4, [[0]*6, [1, 0, 0, 0, 0, 1]], rho)
    run('mps6 adj pair (2,3) chi4', 'mps', 6, 2, 4, [[0]*6, [0, 0, 1, 1, 0, 0]], rho)
    run('mps6 adj pair (2,3) chi1', 'mps', 6, 2, 1, [[0]*6, [0, 0, 1, 1, 0, 0]], rho)
    run('mps6 m=3 (0,1,2) chi4', 'mps', 6, 2, 4, [[0]*6, [1, 1, 1, 0, 0, 0]], rho)
