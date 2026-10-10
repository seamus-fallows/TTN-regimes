"""Independent check of the MPS race claim (no code from experiments/order).
Own MPS contraction (dense W, N = 4), own loss/gradient (closed form, numpy), own LO blow-up times,
DOP853 integrator (agent: RK45). Initial cores taken from the library's fan-in init
(ttn TTNSpec.init, same seeds) so that results can be compared seed by seed.
Usage: indep_mps.py OUT.jsonl SEED0 SEED1 [sigma ...]"""
import json, math, sys, time
import numpy as np
import torch
torch.set_num_threads(1)
from scipy.integrate import solve_ivp
from ttn.tree import make_tree
from ttn.network import TTNSpec

N, D, CHI = 4, 2, 4
spec = TTNSpec.uniform(make_tree("mps", N), D, CHI)
SHAPES = [tuple(s for s in sh if True) for sh in spec.shapes]   # (2,2),(2,2,4),(2,4,2),(2,2,1)
SIZES = [math.prod(s) for s in SHAPES]

def unpack(y):
    out, i = [], 0
    for s, n in zip(SHAPES, SIZES):
        out.append(y[i:i + n].reshape(s)); i += n
    return out

def dense(c):
    # c0[i,a] c1[j,a,b] c2[k,b,e] c3[l,e,0]
    return np.einsum('ia,jab,kbe,le->ijkl', c[0], c[1], c[2], c[3][:, :, 0])

def grad_from_residual(c, R):
    """d/dcores of <W(c), R> for a fixed tensor R (W multilinear in the cores)."""
    c3 = c[3][:, :, 0]
    g0 = np.einsum('ijkl,jab,kbe,le->ia', R, c[1], c[2], c3)
    g1 = np.einsum('ijkl,ia,kbe,le->jab', R, c[0], c[2], c3)
    g2 = np.einsum('ijkl,ia,jab,le->kbe', R, c[0], c[1], c3)
    g3 = np.einsum('ijkl,ia,jab,kbe->le', R, c[0], c[1], c[2])[:, :, None]
    return np.concatenate([g0.ravel(), g1.ravel(), g2.ravel(), g3.ravel()])

def teacher(weights):
    T = np.zeros((D,) * N)
    for r, s in enumerate(weights):
        T[(r,) * N] = s
    return T

def gf_first(y0, weights, rtol=1e-10, atol=1e-15):
    T = teacher(weights)
    w = np.asarray(weights)
    def rhs(t, y):
        c = unpack(y)
        return grad_from_residual(c, T - dense(c))      # -dL/dtheta
    def ev(t, y):
        W = dense(unpack(y))
        return max(W[(r,) * N] / w[r] for r in range(len(w))) - 0.5
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, (0, 1e15), y0, method='DOP853', rtol=rtol, atol=atol, events=ev)
    W = dense(unpack(sol.y[:, -1]))
    a = [W[(r,) * N] / w[r] for r in range(len(w))]
    return float(sol.t[-1]), a, int(np.argmax(a)), sol.status

def lo_tau(u0, r, grow=1e3, rtol=1e-10, atol=1e-14):
    """Blow-up time of d theta/dt = grad T_r (T_r = W[r,r,r,r]); stop at |theta| = grow |theta0|,
    then add the exact remaining time of the radial equation dR/dt = K g(n) R^(K-1)."""
    E = np.zeros((D,) * N); E[(r,) * N] = 1.0
    def rhs(t, y):
        return grad_from_residual(unpack(y), E)
    r0 = np.linalg.norm(u0)
    def ev(t, y):
        return np.linalg.norm(y) - grow * r0
    ev.terminal = True
    sol = solve_ivp(rhs, (0, 1e9), u0, method='DOP853', rtol=rtol, atol=atol, events=ev)
    y = sol.y[:, -1]; R = np.linalg.norm(y)
    g = dense(unpack(y / R))[(r,) * N]
    return float(sol.t[-1] + R ** (-(N - 2)) / (N * (N - 2) * g))

if __name__ == '__main__':
    out, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sigmas = [float(x) for x in sys.argv[4:]] or [0.01]
    with open(out, 'a') as f:
        for seed in range(s0, s1):
            t0 = time.time()
            u0 = spec.flatten(spec.init(1.0, seed=seed)).numpy()
            taus = [lo_tau(u0, r) for r in range(2)]
            runs = []
            for rho in [0.5, 0.8]:
                for sg in sigmas:
                    t1, a, win, st = gf_first(sg * u0, [1.0, rho])
                    runs.append(dict(rho=rho, sigma=sg, t1=t1, a=a, winner=win, status=int(st)))
            f.write(json.dumps(dict(seed=seed, tau=taus, runs=runs, wall=time.time() - t0)) + '\n')
            f.flush()
