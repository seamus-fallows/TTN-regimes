"""Optimizers compared in this experiment, all on the exact population loss (float64).

Clock of each method (the `time` column of its trajectory):
  gf            gradient-flow time t (scipy DOP853, pilot until ttn.dynamics lands)
  gd, momentum, adam, rmsprop, regauge, precond, riemannian    steps n
  als, dmrg     sweeps (fractional: one micro-step = one core / bond update)
"""

from __future__ import annotations

import time as _time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from experiments.optimizers.common import Problem, Recorder, record_schedule
from experiments.optimizers.gauge import (
    canonicalize,
    env_grams,
    euler_tour,
    hsvd,
    leg_to,
    move_center,
)
from ttn.network import TTNSpec, apply_on_leg, to_dense


class Stop:
    """Convergence / divergence test shared by all methods."""

    def __init__(self, loss0: float, tol: float, blowup: float = 1e6):
        self.loss0, self.tol, self.blowup = loss0, tol, blowup
        self.status = "budget"

    def __call__(self, loss: float) -> bool:
        if not np.isfinite(loss) or loss > self.blowup * self.loss0:
            self.status = "diverged"
            return True
        if loss < self.tol * self.loss0:
            self.status = "converged"
            return True
        return False


# ---- gradient flow (reference) ---------------------------------------------------------------


def run_gf(p: Problem, x0: torch.Tensor, cfg: dict, rec: Recorder) -> dict:
    t_max, rtol, atol = cfg["t_max"], cfg.get("rtol", 1e-9), cfg.get("atol", 1e-14)
    loss0 = p.loss(x0).item()
    stop = Stop(loss0, cfg.get("tol", 1e-6))
    n_eval = [0]

    def rhs(t, y):
        n_eval[0] += 1
        _, g = p.loss_grad(torch.from_numpy(y))
        return -g.numpy()

    def converged(t, y):
        return p.loss(torch.from_numpy(y)).item() - stop.tol * loss0

    converged.terminal = True
    sol = solve_ivp(rhs, (0, t_max), x0.numpy(), method="DOP853", rtol=rtol, atol=atol,
                    dense_output=True, events=converged)
    t_end = sol.t[-1]
    grid = np.unique(np.concatenate([
        [0.0], np.geomspace(min(1e-3, t_end), t_end, 1500), np.linspace(0, t_end, 3001)]))
    for i, t in enumerate(grid):
        rec.record(float(t), torch.from_numpy(sol.sol(t)), force_heavy=(i % 3 == 0))
    stop.status = "converged" if sol.status == 1 else ("budget" if sol.status == 0 else "failed")
    return {"status": stop.status, "n_grad": n_eval[0], "t_end": t_end}


# ---- first-order optimizers (torch.optim on the flat parameter vector) ------------------------


def make_torch_opt(cfg: dict, x: torch.Tensor):
    name, lr = cfg["optimizer"], cfg["lr"]
    if name in ("gd", "regauge"):
        return torch.optim.SGD([x], lr=lr)
    if name == "momentum":
        return torch.optim.SGD([x], lr=lr, momentum=cfg.get("beta", 0.9))
    if name == "adam":
        return torch.optim.Adam([x], lr=lr, betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)),
                                eps=cfg.get("eps", 1e-8))
    if name == "rmsprop":
        return torch.optim.RMSprop([x], lr=lr, alpha=cfg.get("alpha", 0.99), eps=cfg.get("eps", 1e-8))
    raise ValueError(name)


def run_first_order(p: Problem, x0: torch.Tensor, cfg: dict, rec: Recorder) -> dict:
    """GD, heavy-ball momentum, Adam, RMSProp; 'regauge' = GD with the cores brought to the
    root-centred canonical form every `period` steps (W unchanged, gradient changes)."""
    n_max = int(cfg["n_steps"])
    sched = set(record_schedule(n_max).tolist())
    x = x0.clone().requires_grad_(True)
    opt = make_torch_opt(cfg, x)
    period = int(cfg.get("period", 0)) if cfg["optimizer"] == "regauge" else 0
    loss0 = p.loss(x0).item()
    stop = Stop(loss0, cfg.get("tol", 1e-6))
    k = 0
    for n in range(n_max + 1):
        if period and n % period == 0:
            with torch.no_grad():
                x.copy_(p.spec.flatten(canonicalize(p.spec, p.spec.unflatten(x.detach()))))
        opt.zero_grad()
        loss = p.loss(x)
        loss_v = loss.item()
        done = stop(loss_v) or n == n_max
        if n in sched or done:
            rec.record(float(n), x, loss_v, force_heavy=done or (k % 3 == 0))
            k += 1
        if done:
            break
        loss.backward()
        opt.step()
    return {"status": stop.status, "n_grad": n + 1, "n_steps": n}


# ---- per-core Gram preconditioning (block-diagonal Gauss-Newton, simultaneous) ---------------


def precondition(spec: TTNSpec, cores, grads, damping: float):
    """Apply the inverse of each core's exact Hessian block, the Kronecker product of the
    environment Grams (as in ALS normal equations), damped by `damping` x (mean eigenvalue) per
    Kronecker factor."""
    t = spec.tree
    below, above = env_grams(spec, cores)
    out = []
    for v in range(t.n_nodes):
        g = grads[v]
        legs = [(len(t.sites[v]) + k, below[c]) for k, c in enumerate(t.children[v])]
        if v != t.root:
            legs.append((g.ndim - 1, above[v]))
        for leg, m in legs:
            reg = damping * torch.trace(m) / m.shape[0]
            minv = torch.linalg.inv(m + reg * torch.eye(m.shape[0], dtype=m.dtype))
            g = apply_on_leg(g, minv, leg)
        out.append(g)
    return out


def run_precond(p: Problem, x0: torch.Tensor, cfg: dict, rec: Recorder) -> dict:
    n_max, lr, damping = int(cfg["n_steps"]), cfg["lr"], cfg.get("damping", 1e-3)
    sched = set(record_schedule(n_max).tolist())
    x = x0.clone()
    stop = Stop(p.loss(x0).item(), cfg.get("tol", 1e-6))
    k = 0
    for n in range(n_max + 1):
        loss_v, g = p.loss_grad(x)
        done = stop(loss_v) or n == n_max
        if n in sched or done:
            rec.record(float(n), x, loss_v, force_heavy=done or (k % 3 == 0))
            k += 1
        if done:
            break
        cores = p.spec.unflatten(x)
        step = precondition(p.spec, cores, p.spec.unflatten(g), damping)
        x = x - lr * p.spec.flatten(step)
    return {"status": stop.status, "n_grad": n + 1, "n_steps": n}


# ---- Riemannian GD on the fixed-bond-dimension manifold ----------------------------------------


def tangent_project(spec: TTNSpec, cores, z: torch.Tensor, rcond: float = 1e-12) -> torch.Tensor:
    """Orthogonal projection of a dense tensor onto range(dW/dtheta) at `cores` (the tangent
    space of the fixed-rank TTN manifold at full-rank points)."""
    x = spec.flatten(cores)

    def w_of(flat):
        return to_dense(spec, spec.unflatten(flat)).reshape(-1)

    jac = torch.func.jacfwd(w_of)(x)
    coef = torch.linalg.lstsq(jac, z.reshape(-1, 1), rcond=rcond, driver="gelsd").solution
    return (jac @ coef).reshape(z.shape)


def run_riemannian(p: Problem, x0: torch.Tensor, cfg: dict, rec: Recorder) -> dict:
    """W <- HSVD_chi(W - lr P_T(W) (W - W*)): Riemannian gradient step plus HSVD retraction."""
    n_max, lr = int(cfg["n_steps"]), cfg["lr"]
    spec = p.spec
    sched = set(record_schedule(n_max).tolist())
    w_star = p.teacher.dense()
    cores = hsvd(spec, to_dense(spec, spec.unflatten(x0)))
    stop = Stop(p.loss(x0).item(), cfg.get("tol", 1e-6))
    k = 0
    for n in range(n_max + 1):
        w = to_dense(spec, cores)
        loss_v = 0.5 * (w - w_star).pow(2).sum().item()
        done = stop(loss_v) or n == n_max
        if n in sched or done:
            rec.record(float(n), spec.flatten(cores), loss_v, force_heavy=done or (k % 3 == 0))
            k += 1
        if done:
            break
        w_new = w - lr * tangent_project(spec, cores, w - w_star)
        cores = hsvd(spec, w_new)
    return {"status": stop.status, "n_grad": n + 1, "n_steps": n}


# ---- ALS / DMRG sweeps ---------------------------------------------------------------------------


def _ls_core(p: Problem, cores, v: int) -> torch.Tensor:
    """Exact least-squares core v when the cores are in canonical form centred at v: the
    environment is an isometry, so the minimizer is the gradient of <W, W*> in core v."""
    g = cores[v].detach().clone().requires_grad_(True)
    cs = list(cores)
    cs[v] = g
    (grad,) = torch.autograd.grad(p.teacher.inner_ttn(p.spec, cs), g)
    return grad


def _merged_ls(p: Problem, cores, v: int, u: int):
    """Least-squares two-site tensor on the bond (v, u), centre on {v, u}: returned as a matrix
    (legs of v except the bond) x (legs of u except the bond), with the leg shapes."""
    spec = p.spec
    lv, lu = leg_to(spec, v, u), leg_to(spec, u, v)
    gu = torch.movedim(cores[u], lu, 0)  # (bond, other legs of u)
    other_u = gu.shape[1:]
    f = int(np.prod(other_u))
    # Widen the bond to f and let u be the identity map: core v then holds the merged tensor.
    eye = torch.eye(f, dtype=gu.dtype).reshape(f, *other_u)
    bonds = list(spec.bond_dims)
    child = v if spec.tree.parent[v] == u else u
    bonds[child] = f
    wide = TTNSpec(spec.tree, spec.phys_dims, tuple(bonds))
    gv = torch.movedim(cores[v], lv, -1)
    merged = torch.tensordot(gv, gu, dims=([gv.ndim - 1], [0]))  # (other legs of v, other legs of u)
    other_v = gv.shape[:-1]
    x = merged.reshape(*other_v, f).clone().requires_grad_(True)
    cs = list(cores)
    cs[v] = torch.movedim(x, -1, lv)
    cs[u] = torch.movedim(eye, 0, lu)
    (grad,) = torch.autograd.grad(p.teacher.inner_ttn(wide, cs), x)
    return grad.reshape(-1, f), other_v, other_u


def run_sweeps(p: Problem, x0: torch.Tensor, cfg: dict, rec: Recorder) -> dict:
    """'als': one-site ALS (exact LS per core, centre moved by QR along a depth-first tour).
    'dmrg': two-site updates with SVD truncation (relative cutoff, at most the spec's bond dim),
    as in Stoudenmire & Schwab (2016) but with exact least squares instead of a gradient step."""
    spec = p.spec
    two_site = cfg["optimizer"] == "dmrg"
    cutoff = cfg.get("cutoff", 1e-12)
    tour = euler_tour(spec)
    moves = list(zip(tour[:-1], tour[1:]))
    cores = canonicalize(spec, spec.unflatten(x0.clone()), tour[0])
    stop = Stop(p.loss(x0).item(), cfg.get("tol", 1e-6))
    rec.record(0.0, spec.flatten(cores), force_heavy=True)
    n_micro = 0
    for sweep in range(int(cfg["n_sweeps"])):
        for i, (v, u) in enumerate(moves):
            if two_site:
                m, other_v, other_u = _merged_ls(p, cores, v, u)
                uu, s, vh = torch.linalg.svd(m, full_matrices=False)
                chi = spec.bond_dims[v if spec.tree.parent[v] == u else u]
                keep = min(chi, int((s > cutoff * s[0]).sum()))
                s = torch.where(torch.arange(len(s)) < keep, s, torch.zeros_like(s))
                cores[v] = torch.movedim(uu[:, :chi].reshape(*other_v, chi), -1, leg_to(spec, v, u))
                new_u = (s[:chi, None] * vh[:chi]).reshape(chi, *other_u)
                cores[u] = torch.movedim(new_u, 0, leg_to(spec, u, v))
            else:
                cores[v] = _ls_core(p, cores, v)
                cores = move_center(spec, cores, v, u)
                if i == len(moves) - 1:
                    cores[u] = _ls_core(p, cores, u)
            n_micro += 1
            x = spec.flatten(cores)
            loss_v = p.loss(x).item()
            done = stop(loss_v)
            rec.record(sweep + (i + 1) / len(moves), x, loss_v, force_heavy=True)
            if done:
                return {"status": stop.status, "n_micro": n_micro, "n_sweeps": sweep + 1}
    return {"status": stop.status, "n_micro": n_micro, "n_sweeps": int(cfg["n_sweeps"])}


RUNNERS = {
    "gf": run_gf,
    "gd": run_first_order,
    "momentum": run_first_order,
    "adam": run_first_order,
    "rmsprop": run_first_order,
    "regauge": run_first_order,
    "precond": run_precond,
    "riemannian": run_riemannian,
    "als": run_sweeps,
    "dmrg": run_sweeps,
}


def run(p: Problem, x0: torch.Tensor, cfg: dict) -> dict:
    rec = Recorder(p, x0)
    t0 = _time.process_time()
    info = RUNNERS[cfg["optimizer"]](p, x0, cfg, rec)
    info["cpu_seconds"] = _time.process_time() - t0
    out = rec.arrays()
    out["info"] = np.array(repr(info))
    return out, info
