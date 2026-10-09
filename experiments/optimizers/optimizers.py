"""Optimizers compared in this experiment, on the exact population loss (float64).

Every runner returns a `ttn.dynamics.Trajectory`: the library Recorder's diagnostics (plus the
balance index and ||Q_e|| per edge) at scheduled points, and for the discrete methods a cheap fine
trace `fine_t`, `fine_loss`, `fine_overlaps` (every `stride` steps) used for event times.

Clock of each method (`t` in the trajectory):
  gf                                         gradient-flow time (ttn.dynamics.integrate_gf, LSODA)
  gd, momentum, adam, rmsprop, regauge,
  precond, riemannian                         steps n (not lr * n)
  als, dmrg                                   sweeps (one micro-step = one core / bond update)
"""

from __future__ import annotations

import time as _time

import numpy as np
import torch

from experiments.optimizers.common import make_recorder, record_schedule, teacher_spectra
from experiments.optimizers.gauge import (
    canonicalize,
    env_grams,
    euler_tour,
    hsvd,
    leg_to,
    move_center,
)
from ttn.dynamics import Problem, Trajectory, integrate_gf
from ttn.network import TTNSpec, apply_on_leg, to_dense


class Log:
    """Library Recorder at scheduled points plus a cheap fine trace for event timing."""

    def __init__(self, problem: Problem, theta0: torch.Tensor):
        self.p = problem
        self.rec = make_recorder()
        self.rec.start(problem, theta0)
        self.fine: dict[str, list] = {"fine_t": [], "fine_loss": [], "fine_overlaps": []}

    def full(self, theta: torch.Tensor, t: float, **info):
        self.rec.record(self.p, theta.detach(), t, **info)

    @torch.no_grad()
    def cheap(self, theta: torch.Tensor, t: float, loss: float):
        spec, cores = self.p.effective(theta.detach())
        self.fine["fine_t"].append(t)
        self.fine["fine_loss"].append(loss)
        self.fine["fine_overlaps"].append(self.p.target.component_overlaps(spec, cores).numpy())

    def trajectory(self, theta_end: torch.Tensor, meta: dict) -> Trajectory:
        data = self.rec.arrays()
        data.update({k: np.array(v) for k, v in self.fine.items() if v})
        data["theta_end"] = theta_end.detach().numpy()
        data["teacher_spectra"] = teacher_spectra(self.p)
        return Trajectory(data, meta)


class Stop:
    """Convergence (loss < tol L0) / divergence (non-finite or > 1e6 L0) test."""

    def __init__(self, loss0: float, tol: float, blowup: float = 1e6):
        self.loss0, self.tol, self.blowup = loss0, tol, blowup
        self.reason = "budget"

    def __call__(self, loss: float) -> bool:
        if not np.isfinite(loss) or loss > self.blowup * self.loss0:
            self.reason = "diverged"
            return True
        if loss < self.tol * self.loss0:
            self.reason = "converged"
            return True
        return False


def _fine_stride(n_max: int) -> int:
    return max(1, n_max // 50_000)


# ---- gradient flow (reference) ---------------------------------------------------------------


def run_gf(p: Problem, x0: torch.Tensor, cfg: dict) -> Trajectory:
    """Library integrator (LSODA: later plateaus are stiff), on a log + linear time grid."""
    t_max = float(cfg["t_max"])
    loss0 = p.loss(x0).item()
    grid = np.unique(np.concatenate([np.geomspace(1e-3, t_max, 600), np.linspace(0, t_max, 1201)]))
    rec = make_recorder()
    traj = integrate_gf(p, x0, t_max, log_times=grid[grid > 0], rtol=cfg.get("rtol", 1e-9),
                        atol=cfg.get("atol", 1e-14), loss_floor=cfg.get("tol", 1e-6) * loss0,
                        recorder=rec, method=cfg.get("method", "lsoda"))
    traj.data["teacher_spectra"] = teacher_spectra(p)
    return traj


# ---- first-order optimizers (torch.optim on the flat parameter vector) ------------------------


def make_torch_opt(cfg: dict, x: torch.Tensor):
    name, lr = cfg["optimizer"], float(cfg["lr"])
    if name in ("gd", "regauge"):
        return torch.optim.SGD([x], lr=lr)
    if name == "momentum":
        return torch.optim.SGD([x], lr=lr, momentum=cfg.get("beta", 0.9))
    if name == "adam":
        return torch.optim.Adam([x], lr=lr, betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)),
                                eps=float(cfg.get("eps", 1e-8)))
    if name == "rmsprop":
        return torch.optim.RMSprop([x], lr=lr, alpha=cfg.get("alpha", 0.99),
                                   eps=float(cfg.get("eps", 1e-8)))
    raise ValueError(name)


def run_first_order(p: Problem, x0: torch.Tensor, cfg: dict) -> Trajectory:
    """GD, heavy-ball momentum, Adam, RMSProp. 'regauge' = GD with the cores brought to the
    root-centred canonical form every `period` steps (W unchanged, the gradient changes)."""
    n_max = int(cfg["n_steps"])
    sched = set(record_schedule(n_max).tolist())
    stride = _fine_stride(n_max)
    x = x0.clone().requires_grad_(True)
    opt = make_torch_opt(cfg, x)
    period = int(cfg.get("period", 0)) if cfg["optimizer"] == "regauge" else 0
    log = Log(p, x0)
    stop = Stop(p.loss(x0).item(), cfg.get("tol", 1e-6))
    for n in range(n_max + 1):
        if period and n % period == 0:
            with torch.no_grad():
                x.copy_(p.spec.flatten(canonicalize(p.spec, p.spec.unflatten(x.detach()))))
        opt.zero_grad()
        loss = p.loss(x)
        loss_v = loss.item()
        done = stop(loss_v) or n == n_max
        if n % stride == 0 or done:
            log.cheap(x, float(n), loss_v)
        if n in sched or done:
            log.full(x, float(n), step=n)
        if done:
            break
        loss.backward()
        opt.step()
    return log.trajectory(x, {"stop_reason": stop.reason, "n_steps": n, "n_evals": n + 1})


# ---- per-core Gram preconditioning (block-diagonal Gauss-Newton, simultaneous) ---------------


def precondition(spec: TTNSpec, cores, grads, damping: float, damping_abs: float = 0.0):
    """Apply the inverse of each core's exact Hessian block (the Kronecker product of the
    environment Grams, as in the ALS normal equations), each Kronecker factor damped by
    `damping` x its mean eigenvalue (scale-free) plus `damping_abs` (an absolute floor, like
    Adam's eps or Levenberg-Marquardt damping)."""
    t = spec.tree
    below, above = env_grams(spec, cores)
    out = []
    for v in range(t.n_nodes):
        g = grads[v]
        legs = [(len(t.sites[v]) + k, below[c]) for k, c in enumerate(t.children[v])]
        if v != t.root:
            legs.append((g.ndim - 1, above[v]))
        for leg, m in legs:
            reg = damping * torch.trace(m) / m.shape[0] + damping_abs
            minv = torch.linalg.inv(m + reg * torch.eye(m.shape[0], dtype=m.dtype))
            g = apply_on_leg(g, minv, leg)
        out.append(g)
    return out


def run_precond(p: Problem, x0: torch.Tensor, cfg: dict) -> Trajectory:
    n_max, lr, damping = int(cfg["n_steps"]), float(cfg["lr"]), float(cfg.get("damping", 1e-3))
    damping_abs = float(cfg.get("damping_abs", 0.0))
    sched = set(record_schedule(n_max).tolist())
    stride = _fine_stride(n_max)
    x = x0.clone()
    log = Log(p, x0)
    # the simultaneous block update overshoots by many orders of magnitude before it returns
    stop = Stop(p.loss(x0).item(), cfg.get("tol", 1e-6), blowup=float(cfg.get("blowup", 1e6)))
    for n in range(n_max + 1):
        loss, g = p.loss_and_grad(x)
        loss_v = loss.item()
        done = stop(loss_v) or n == n_max
        if n % stride == 0 or done:
            log.cheap(x, float(n), loss_v)
        if n in sched or done:
            log.full(x, float(n), step=n)
        if done:
            break
        step = precondition(p.spec, p.spec.unflatten(x), p.spec.unflatten(g), damping, damping_abs)
        x = x - lr * p.spec.flatten(step)
    return log.trajectory(x, {"stop_reason": stop.reason, "n_steps": n, "n_evals": n + 1})


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


def run_riemannian(p: Problem, x0: torch.Tensor, cfg: dict) -> Trajectory:
    """W <- HSVD_chi(W - lr P_T(W) (W - W*)): Riemannian gradient step, HSVD retraction.
    Population loss only (needs the dense teacher)."""
    n_max, lr = int(cfg["n_steps"]), float(cfg["lr"])
    spec = p.spec
    sched = set(record_schedule(n_max).tolist())
    w_star = p.target.dense()
    cores = hsvd(spec, to_dense(spec, spec.unflatten(x0)))
    log = Log(p, x0)
    stop = Stop(p.loss(x0).item(), cfg.get("tol", 1e-6))
    for n in range(n_max + 1):
        w = to_dense(spec, cores)
        loss_v = 0.5 * (w - w_star).pow(2).sum().item()
        x = spec.flatten(cores)
        done = stop(loss_v) or n == n_max
        log.cheap(x, float(n), loss_v)
        if n in sched or done:
            log.full(x, float(n), step=n)
        if done:
            break
        cores = hsvd(spec, w - lr * tangent_project(spec, cores, w - w_star))
    return log.trajectory(x, {"stop_reason": stop.reason, "n_steps": n, "n_evals": n + 1})


# ---- ALS / DMRG sweeps ---------------------------------------------------------------------------


def _ls_core(p: Problem, cores, v: int) -> torch.Tensor:
    """Exact least-squares core v when the cores are in canonical form centred at v: the
    environment is an isometry, so the minimizer is the gradient of <W, W*> in core v."""
    g = cores[v].detach().clone().requires_grad_(True)
    cs = list(cores)
    cs[v] = g
    (grad,) = torch.autograd.grad(p.target.inner_ttn(p.spec, cs), g)
    return grad


def _merged_ls(p: Problem, cores, v: int, u: int):
    """Least-squares two-site tensor on the bond (v, u) with the centre on {v, u}, as a matrix
    (legs of v except the bond) x (legs of u except the bond), plus the two leg shapes.

    Trick: widen the bond to f = size of u's other legs and make u the identity map, so that core
    v holds the merged tensor and its gradient is the two-site least-squares solution."""
    spec = p.spec
    lv, lu = leg_to(spec, v, u), leg_to(spec, u, v)
    other_u = torch.movedim(cores[u], lu, 0).shape[1:]
    f = int(np.prod(other_u))
    eye = torch.eye(f, dtype=torch.float64).reshape(f, *other_u)
    bonds = list(spec.bond_dims)
    bonds[v if spec.tree.parent[v] == u else u] = f
    wide = TTNSpec(spec.tree, spec.phys_dims, tuple(bonds))
    other_v = torch.movedim(cores[v], lv, -1).shape[:-1]
    x = torch.zeros(*other_v, f, dtype=torch.float64, requires_grad=True)
    cs = list(cores)
    cs[v] = torch.movedim(x, -1, lv)
    cs[u] = torch.movedim(eye, 0, lu)
    (grad,) = torch.autograd.grad(p.target.inner_ttn(wide, cs), x)
    return grad.reshape(-1, f), other_v, other_u


def run_sweeps(p: Problem, x0: torch.Tensor, cfg: dict) -> Trajectory:
    """'als': one-site ALS (exact least squares per core; centre moved by QR along a depth-first
    tour). 'dmrg': two-site updates with SVD truncation (relative cutoff, at most the spec's bond
    dimension), as Stoudenmire & Schwab (2016) but with exact least squares instead of a gradient
    step. Population loss only."""
    spec = p.spec
    two_site = cfg["optimizer"] == "dmrg"
    cutoff = float(cfg.get("cutoff", 1e-12))
    tour = euler_tour(spec)
    moves = list(zip(tour[:-1], tour[1:]))
    cores = canonicalize(spec, spec.unflatten(x0.clone()), tour[0])
    log = Log(p, x0)
    stop = Stop(p.loss(x0).item(), cfg.get("tol", 1e-6))
    x = spec.flatten(cores)
    log.cheap(x, 0.0, p.loss(x).item())
    log.full(x, 0.0, step=0)
    n_micro, done = 0, False
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
            t = sweep + (i + 1) / len(moves)
            log.cheap(x, t, loss_v)
            log.full(x, t, step=n_micro)
            if done:
                break
        if done:
            break
    return log.trajectory(x, {"stop_reason": stop.reason, "n_micro": n_micro,
                              "n_sweeps": sweep + 1, "micro_per_sweep": len(moves)})


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


def run(p: Problem, x0: torch.Tensor, cfg: dict) -> Trajectory:
    t0, c0 = _time.time(), _time.process_time()
    traj = RUNNERS[cfg["optimizer"]](p, x0, cfg)
    traj.meta.update({"cfg": cfg, "wall_seconds": _time.time() - t0,
                      "cpu_seconds": _time.process_time() - c0, "problem": p.describe()})
    return traj
