"""Gradient flow of a TTN from small random init on a pattern target (experiment route_nonstar).

Model: whitened inputs, exact population loss L = 1/2 ||W - W*||^2 with W* = sum_a s_a U_a in the
standard site basis, fan-in Gaussian init `spec.init(sigma, seed)` on all parameters, bond dimension
on edge e = (rank of W* across e) + 2 spare channels (capped at the largest rank the cut allows).
The targets are small (at most 8 sites, d = 2), so W is contracted densely (`ttn.network.to_dense`)
and the gradient and the exact Hessian come from torch.func.

Integrator: scipy's LSODA class driven step by step with the exact Hessian as Jacobian (rtol 1e-10,
atol 1e-14), as in experiments/saddle_dyn/sdyn.py (notes/saddle_dyn.md section 8): every recorded
state is a solver step (LSODA's dense output is not used), the solver clock is re-zeroed when the
step falls below 1e-11 of the local time (deviation 5 there), a restarted solver starts with a
step of 0.01 (deviation 6b), failed steps are retried from the last good step (at most 5). The
plain loss is used (no reference point): the float64 floor is checked a posteriori per plateau.
"""

from __future__ import annotations

import math
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.integrate import LSODA  # noqa: E402

from ttn.network import TTNSpec, to_dense  # noqa: E402
from ttn.routes import RouteTarget  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

torch.set_num_threads(1)
DT = torch.float64
U = 2.0**-53
RESTART_H = 1e-2


def pattern_str(p) -> str:
    return "".join(str(int(x)) for x in p)


def cut_rank(w: np.ndarray, tree, e: int) -> int:
    below = sorted(tree.subtree_sites[e])
    rest = [s for s in range(tree.n_sites) if s not in below]
    m = np.transpose(w, below + rest).reshape(int(np.prod([w.shape[s] for s in below])), -1)
    return int(np.linalg.matrix_rank(m, tol=1e-10))


class Model:
    """Pattern target on a tree, dense contraction, gradient and Hessian."""

    def __init__(self, kind: str, n: int, patterns, weights, d: int = 2, spare: int = 2):
        self.kind, self.n, self.d = kind, n, d
        self.tree = tree = make_tree(kind, n)
        self.target = RouteTarget(tree, patterns, weights)
        self.patterns = self.target.patterns
        self.weights = np.array(self.target.weights)
        self.n_pat = len(self.patterns)
        w = np.zeros((d,) * tree.n_sites)
        for p, s in zip(self.patterns, self.weights):
            w[p] = s
        self.w_star_np = w
        self.W_star = torch.from_numpy(w)
        self.L_star = 0.5 * float((w**2).sum())
        self.ranks = {e: cut_rank(w, tree, e) for e in tree.edges}
        bonds = [1] * tree.n_nodes
        for e in tree.edges:
            below = len(tree.subtree_sites[e])
            cap = min(d**below, d ** (tree.n_sites - below))
            bonds[e] = min(self.ranks[e] + spare, cap)
        self.spec = TTNSpec(tree, (d,) * tree.n_sites, tuple(bonds))
        self.n_params = self.spec.n_params
        self.flat_idx = np.array([np.ravel_multi_index(p, w.shape) for p in self.patterns])
        self._grad = torch.func.grad_and_value(self._loss)
        self._hess = torch.func.hessian(self._loss)

    # ---- tensors ---------------------------------------------------------------------------
    def W(self, theta: torch.Tensor) -> torch.Tensor:
        return to_dense(self.spec, self.spec.unflatten(theta))

    def _loss(self, theta):
        e = self.W(theta) - self.W_star
        return 0.5 * (e * e).sum()

    def grad(self, theta: torch.Tensor):
        g, val = self._grad(theta)
        return g, float(val)

    def hessian(self, theta: torch.Tensor) -> torch.Tensor:
        return self._hess(theta)

    def summary(self, theta: torch.Tensor) -> dict:
        """Amplitudes P_a = <U_a, W>, off-target mass (fraction of ||W||^2 outside the target
        span) and the dense W (flattened)."""
        w = self.W(theta).detach().reshape(-1).numpy()
        amps = w[self.flat_idx]
        tot = float((w**2).sum())
        off = max(tot - float((amps**2).sum()), 0.0) / tot if tot > 0 else 0.0
        return dict(P=amps, off=off, w=w)

    def grad_roundoff(self, theta: torch.Tensor) -> np.ndarray:
        """Upper bound (10 u x the gradient of |E| . |W|(|theta|)) on the roundoff in the gradient
        of the plain loss: every term of <E, dW/dtheta> is bounded by |E| |dW/dtheta| and E itself
        carries an error u (|W| + |W*|)."""
        e = (self.W(theta) - self.W_star).abs().detach() + self.W(theta).abs().detach() + self.W_star
        th = theta.abs().clone().requires_grad_(True)
        (g,) = torch.autograd.grad((e * self.W(th)).sum(), th)
        return 10 * U * g.numpy()

    def init(self, sigma: float, seed: int) -> np.ndarray:
        return self.spec.flatten(self.spec.init(sigma, seed=seed)).numpy().copy()


def integrate(model: Model, y0, t_max=1e17, rtol=1e-10, atol=1e-14, wall=1800.0, conv=1e-10,
              trap_g=1e-14, trap_decades=2.0):
    """Gradient flow from y0. Records every accepted solver step. Stops when L < conv * L_star
    ("converged"), at t_max ("t_max"), at the wall clock ("wall"), after more than 5 failed steps
    ("failed"), or when the largest gradient entry stays below trap_g at every check (log-spaced by
    10^0.1 in t, t > 1e3) for trap_decades decades of time while not converged ("trapped": the
    remaining routes are at roundoff and cannot escape)."""
    t_wall = time.time()
    cache: dict = {}

    def evalg(y):
        key = y.tobytes()
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            g, val = model.grad(torch.from_numpy(y))
            cache[key] = (g.numpy(), val)
        return cache[key]

    def rhs(t, y):
        return -evalg(y)[0]

    def jac(t, y):
        return -model.hessian(torch.from_numpy(y)).numpy()

    y0 = np.array(y0, dtype=np.float64)
    ts, ys, Ls, gs, gn = [0.0], [y0.copy()], [], [], []
    toffs, tlocs = [0.0], [0.0]  # absolute time = offset + local time (exact in long double)
    g0, L0 = evalg(y0)
    Ls.append(L0)
    gs.append(float(np.abs(g0).max()))
    gn.append(float(np.linalg.norm(g0)))
    off, t_loc_old, n_restart, fails = 0.0, 0.0, 0, []
    solver = LSODA(rhs, 0.0, y0, t_max, rtol=rtol, atol=atol, jac=jac)
    nxt, count, status = 1e3, 0, "running"
    n_checks = int(round(10 * trap_decades))
    while True:
        if time.time() - t_wall > wall:
            status = "wall"
            break
        msg = solver.step()
        if solver.status == "failed":
            fails.append(off + t_loc_old)
            if len(fails) > 5:
                status = "failed: " + str(msg)
                break
            off += t_loc_old
            t_loc_old = 0.0
            solver = LSODA(rhs, 0.0, ys[-1], t_max - off, rtol=rtol, atol=atol, jac=jac,
                           first_step=RESTART_H)
            continue
        t_loc, y = solver.t, np.array(solver.y, copy=True)
        t = off + t_loc
        g, L = evalg(y)
        ts.append(t)
        toffs.append(off)
        tlocs.append(t_loc)
        ys.append(y)
        Ls.append(L)
        gmax = float(np.abs(g).max())
        gs.append(gmax)
        gn.append(float(np.linalg.norm(g)))
        t_loc_old = t_loc
        if L < conv * model.L_star:
            status = "converged"
            break
        if solver.status == "finished":
            status = "t_max"
            break
        if t_loc > 0 and solver.step_size < 1e-11 * t_loc:
            off += t_loc
            t_loc_old = 0.0
            n_restart += 1
            solver = LSODA(rhs, 0.0, y, t_max - off, rtol=rtol, atol=atol, jac=jac,
                           first_step=RESTART_H)
        if t >= nxt:
            nxt = t * 10**0.1
            # only after a pattern was learned: on the first plateau from the origin the gradient
            # is ~ sigma^(K-1), tiny but exact (no cancellation), and it does escape
            learned = bool((model.summary(torch.from_numpy(y))["P"] > model.weights / 2).any())
            count = count + 1 if (gmax < trap_g and learned) else 0
            if count >= n_checks:
                status = "trapped"
                break
    return dict(t=np.array(ts), toff=np.array(toffs), tloc=np.array(tlocs), y=np.array(ys), L=np.array(Ls), gmax=np.array(gs), gnorm=np.array(gn),
                status=status,
                n_restart=n_restart, fails=fails, secs=time.time() - t_wall)
