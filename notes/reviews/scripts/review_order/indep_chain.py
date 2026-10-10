"""Independent check (no code from experiments/order): two-component odeco teacher (standard basis)
on an MPS (any N, d, chi) or on binary4 (= A R B^T). Own dense contraction, closed-form gradients
(numpy), DOP853 (agent: RK45 / torch autograd). Initial cores = library fan-in init (same seeds),
so results compare seed by seed. Also LO blow-up times tau_r of the single-component flow
d theta/dt = grad T_r, T_r = W[r,...,r] (radial extrapolation after |theta| grows 1e3).
Usage: indep_chain.py OUT.jsonl TREE(mps4|mps6|mps8|binary4) D CHI SEED0 SEED1 RHOS SIGMAS RACE(0/1)
  RHOS, SIGMAS comma-separated; every (rho, sigma) pair is run."""
import json, math, sys, time
import numpy as np
import torch
torch.set_num_threads(1)
from scipy.integrate import solve_ivp
from ttn.tree import make_tree
from ttn.network import TTNSpec

out, tree, D, CHI, s0, s1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
RHOS = [] if sys.argv[7] == '-' else [float(x) for x in sys.argv[7].split(',')]
SIGMAS = [] if sys.argv[8] == '-' else [float(x) for x in sys.argv[8].split(',')]
RACE = sys.argv[9] == '1'
kind, N = ('binary', 4) if tree == 'binary4' else ('mps', int(tree[3:]))
import os
CAP = os.environ.get('CAP', '1') == '1'
spec = TTNSpec.uniform(make_tree(kind, N), D, CHI, cap=CAP)
SHAPES = list(spec.shapes)
SIZES = [math.prod(s) for s in SHAPES]

def unpack(y):
    out, i = [], 0
    for s, n in zip(SHAPES, SIZES):
        out.append(y[i:i + n].reshape(s)); i += n
    return out

if kind == 'binary':
    def dense(c):
        return np.einsum('ija,klb,ab->ijkl', c[0], c[1], c[2][:, :, 0])
    def grad_res(c, R):          # d <W(c), R> / d cores
        Rm = c[2][:, :, 0]
        gA = np.einsum('ijkl,klb,ab->ija', R, c[1], Rm)
        gB = np.einsum('ijkl,ija,ab->klb', R, c[0], Rm)
        gR = np.einsum('ijkl,ija,klb->ab', R, c[0], c[1])[:, :, None]
        return np.concatenate([gA.ravel(), gB.ravel(), gR.ravel()])
else:
    def mats(c):  # core k as (d, b_left, b_right) with b_left = 1 for k = 0
        m = [c[0][:, None, :]] + [x for x in c[1:-1]] + [c[-1]]
        return m   # last core already (d, b, 1)
    def dense(c):
        m = mats(c)
        W = m[0][:, 0, :]                       # (d, b0)
        for G in m[1:]:
            W = np.tensordot(W, G, axes=([W.ndim - 1], [1]))   # (..., d, b)
        return W[..., 0]
    def grad_res(c, R):
        m = mats(c)
        n = len(m)
        left = [np.ones((1, 1))]                 # left[k]: (d^k, b_{k-1}) contraction of cores < k
        for k in range(n - 1):
            G = m[k]                             # (d, bl, br)
            L = np.einsum('Pa,iab->Pib', left[-1], G).reshape(-1, G.shape[2])
            left.append(L)
        right = [None] * n
        right[n - 1] = np.ones((1, 1))           # right[k]: (d^(n-1-k), b_k) cores > k
        for k in range(n - 1, 0, -1):
            G = m[k]
            Rk = np.einsum('iab,Qb->iQa', G, right[k]).reshape(-1, G.shape[1])
            right[k - 1] = Rk
        grads = []
        for k in range(n):
            Rk = R.reshape(left[k].shape[0], D, right[k].shape[0])
            g = np.einsum('PiQ,Pa,Qb->iab', Rk, left[k], right[k])
            if k == 0:
                g = g[:, 0, :]
            grads.append(g.ravel())
        return np.concatenate(grads)

def teacher(weights):
    T = np.zeros((D,) * N)
    for r, s in enumerate(weights):
        T[(r,) * N] = s
    return T

def gf_first(y0, weights, rtol=1e-10, atol=1e-15):
    T = teacher(weights); w = np.asarray(weights)
    def rhs(t, y):
        c = unpack(y)
        return grad_res(c, T - dense(c))
    def ev(t, y):
        W = dense(unpack(y))
        return max(W[(r,) * N] / w[r] for r in range(len(w))) - 0.5
    ev.terminal, ev.direction = True, 1
    t_off, y = 0.0, y0
    for _ in range(20):      # clock restart if the step hits the float64 resolution of t
        sol = solve_ivp(rhs, (0, 1e20), y, method='DOP853', rtol=rtol, atol=atol, events=ev)
        if sol.status != -1:
            break
        t_off, y = t_off + sol.t[-1], sol.y[:, -1]
    W = dense(unpack(sol.y[:, -1]))
    a = [W[(r,) * N] / w[r] for r in range(len(w))]
    return float(t_off + sol.t[-1]), a, int(np.argmax(a)), int(sol.status)

def lo_tau(u0, r, grow=1e3, rtol=1e-10, atol=1e-14):
    E = np.zeros((D,) * N); E[(r,) * N] = 1.0
    K = len(SHAPES)
    def rhs(t, y):
        return grad_res(unpack(y), E)
    r0 = np.linalg.norm(u0)
    def ev(t, y):
        return np.linalg.norm(y) - grow * r0
    ev.terminal = True
    sol = solve_ivp(rhs, (0, 1e12), u0, method='DOP853', rtol=rtol, atol=atol, events=ev)
    y = sol.y[:, -1]; Rr = np.linalg.norm(y)
    g = dense(unpack(y / Rr))[(r,) * N]
    return float(sol.t[-1] + Rr ** (-(K - 2)) / (K * (K - 2) * g))

with open(out, 'a') as f:
    for seed in range(s0, s1):
        t0 = time.time()
        u0 = spec.flatten(spec.init(1.0, seed=seed)).numpy()
        rec = dict(tree=tree, d=D, chi=CHI, cap=CAP, seed=seed)
        if RACE:
            rec['tau'] = [lo_tau(u0, r) for r in range(2)]
        runs = []
        for rho in RHOS:
            for sg in SIGMAS:
                t1, a, win, st = gf_first(sg * u0, [1.0, rho])
                runs.append(dict(rho=rho, sigma=sg, t1=t1, a=a, winner=win, status=st))
        rec['runs'] = runs
        rec['wall'] = time.time() - t0
        f.write(json.dumps(rec) + '\n'); f.flush()
