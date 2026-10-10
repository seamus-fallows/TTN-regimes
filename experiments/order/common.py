"""Shared code for the `order` experiment (Q4): which teacher component is learned first.

- Teachers are odeco / pattern CP teachers in the *standard* site basis. Fan-in Gaussian init is
  invariant in distribution under an orthogonal rotation of any physical leg, so every statistic of
  the order is the same as with Haar-random site bases; the standard basis makes the
  component-specific "slices" of the cores explicit (see notes/order.md section 1).
- `gf_first_event`: full gradient flow on the exact population loss until the first learning event
  (some overlap a_r crosses 1/2, notes/diagnostics.md).
- `lo_blowup`: the leading-order flow d theta/dt = grad g, g = <W, W*>, from the unit-scale init,
  stopped when |theta| has grown by `grow`; returns the blow-up time (radially extrapolated) and the
  component amplitudes at the stop. Stopping at grow = 1/sigma mimics full GF at init scale sigma
  (scale invariance of the leading-order flow).
- `init_proxies`: candidate "init overlap" statistics per component, for fitting selection rules.

Integration uses scipy's Dormand-Prince 5(4) (`solve_ivp`, RK45) on the flat parameter vector with
torch autograd gradients, float64; to be replaced by `ttn.dynamics` once it lands on main.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ttn.losses import population_loss
from ttn.network import TTNSpec, forward
from ttn.teachers import CPTeacher, odeco_teacher, pattern_teacher
from ttn.tree import make_tree

RTOL, ATOL = 1e-9, 1e-14


def _dir_option(flag: str, env: str, default: Path) -> Path:
    """Output directories: `--raw DIR` / `--res DIR` on any script's command line (removed from
    sys.argv here), else the environment variable, else the default."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        value = sys.argv[i + 1]
        del sys.argv[i : i + 2]
        return Path(value)
    return Path(os.environ.get(env, default))


def single_thread():
    """One thread per worker process for torch and BLAS (scipy's implicit solvers call LAPACK, and
    OpenBLAS otherwise starts 64 threads per process: with 60 workers the load reached ~2000)."""
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    threadpool_limits(1)
    torch.set_flush_denormal(True)  # decaying dead routes otherwise crawl through denormals


RAW = _dir_option("--raw", "ORDER_RAW", Path("/root/outputs/order"))  # raw outputs (not in git)
RES = _dir_option("--res", "ORDER_RES", Path(__file__).parent / "results")  # summaries, figures
RES.mkdir(parents=True, exist_ok=True)


def make_spec(kind: str, n: int, d: int, chi: int, cap: bool = True) -> TTNSpec:
    return TTNSpec.uniform(make_tree(kind, n), d, chi, cap=cap)


def make_teacher(n: int, d: int, weights, pattern=None) -> CPTeacher:
    if pattern is None:
        return odeco_teacher(n, d, list(weights), basis="standard")
    return pattern_teacher(list(weights), pattern, d, basis="standard")


def unit_init(spec: TTNSpec, seed: int) -> np.ndarray:
    """Fan-in init at sigma = 1 (the run at scale sigma starts from sigma times this)."""
    return spec.flatten(spec.init(1.0, seed=seed)).numpy()


def overlaps(spec: TTNSpec, teacher: CPTeacher, y: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y))).numpy()


def loss_grad_fn(spec: TTNSpec, teacher: CPTeacher):
    t2 = teacher.norm2()

    def loss(f: torch.Tensor) -> torch.Tensor:
        return population_loss(spec, spec.unflatten(f), teacher, teacher_norm2=t2)

    def rhs(t, y):
        f = torch.from_numpy(y).requires_grad_(True)
        (g,) = torch.autograd.grad(loss(f), f)
        return -g.numpy()

    return loss, rhs


def gf_first_event(spec, teacher, y0, threshold=0.5, rtol=RTOL, atol=ATOL, t_max=1e20,
                   max_restarts=20):
    """Gradient flow until max_r a_r(t) = threshold. Returns (t, overlaps, nfev, status).

    The flow is autonomous, so when the step size hits the float64 resolution of t (at t ~ 1e14 the
    spacing of representable times is ~0.02 while the escape itself takes O(1) time; scipy then
    stops with status -1) the integration is restarted from the last state with the clock reset,
    and the elapsed time is carried as an offset."""
    _, rhs = loss_grad_fn(spec, teacher)

    def event(t, y):
        return overlaps(spec, teacher, y).max() - threshold

    event.terminal = True
    event.direction = 1
    offset, y, nfev = 0.0, y0, 0
    for _ in range(max_restarts + 1):
        sol = solve_ivp(rhs, (0.0, t_max - offset), y, method="RK45", rtol=rtol, atol=atol,
                        events=event)
        nfev += sol.nfev
        if sol.status != -1 or sol.t[-1] == 0.0:
            break
        offset, y = offset + sol.t[-1], sol.y[:, -1]
    return offset + sol.t[-1], overlaps(spec, teacher, sol.y[:, -1]), nfev, sol.status


def lo_blowup(spec, teacher, u0, grow=1e4, rtol=RTOL, atol=ATOL):
    """Leading-order flow of g = <W, W*> from u0 until |theta| = grow |u0|.

    Returns (tau, T) with tau the blow-up time (stop time plus the radial remainder
    R^-(K-2) / (K (K-2) g(n)), exact once the direction has converged) and T_r = <W, U_r> at the stop.
    """
    k = spec.n_cores

    def rhs(t, y):
        f = torch.from_numpy(y).requires_grad_(True)
        (g,) = torch.autograd.grad(teacher.inner_ttn(spec, spec.unflatten(f)), f)
        return g.numpy()

    r0 = np.linalg.norm(u0)

    def event(t, y):
        return np.linalg.norm(y) - grow * r0

    event.terminal = True
    sol = solve_ivp(rhs, (0.0, 1e12), u0, method="RK45", rtol=rtol, atol=atol, events=event)
    y = sol.y[:, -1]
    radius = np.linalg.norm(y)
    with torch.no_grad():
        g_dir = teacher.inner_ttn(spec, spec.unflatten(torch.from_numpy(y / radius))).item()
        amps = forward(spec, spec.unflatten(torch.from_numpy(y)), teacher._metric_vectors(None))
    tau = sol.t[-1] + radius ** (-(k - 2)) / (k * (k - 2) * g_dir) if g_dir > 0 else np.inf
    return tau, amps.numpy()


def single_component(teacher: CPTeacher, r: int) -> CPTeacher:
    w = torch.zeros_like(teacher.weights)
    w[r] = 1.0
    return CPTeacher(w, teacher.factors)


# ---- init statistics ("init overlaps") --------------------------------------------------------


def reduced_cores(spec: TTNSpec, cores, vectors) -> list[torch.Tensor]:
    """Cores with every physical leg contracted with the given site vectors (one per site)."""
    t = spec.tree
    out = []
    for v in range(t.n_nodes):
        g = cores[v]
        for s in t.sites[v]:
            g = torch.tensordot(vectors[s], g, dims=([0], [0]))
        out.append(g)  # legs: child bonds (in order), parent bond
    return out


def _contract_except(g: np.ndarray, vecs: list[np.ndarray], skip: int) -> np.ndarray:
    """Contract every leg of g except `skip` with the corresponding vector."""
    letters = "abcdefghijklmnop"[: g.ndim]
    ops, subs = [g], [letters]
    for j, x in enumerate(vecs):
        if j != skip:
            ops.append(x)
            subs.append(letters[j])
    out = letters[skip] if skip >= 0 else ""
    return np.einsum(",".join(subs) + "->" + out, *ops)


def kkt_projection(spec: TTNSpec, red, restarts: int = 4, sweeps: int = 500, seed: int = 0):
    """max over unit bond vectors b_e of F = sum_v red_v[b_children, b_parent].

    F / sqrt(K) is the projection of the init onto the nearest balanced bond-rank-one configuration
    of one component (the KKT point of g on the unit sphere, up to orthogonal gauge). Alternating
    maximization (F is linear in each b_e) from several starts. Returns (F_max, per-core values).
    """
    t = spec.tree
    gen = np.random.default_rng(seed)
    red = [r.numpy() for r in red]

    def legs(v, b):
        return [b[c] for c in t.children[v]] + [b[v]]

    best_val, best_cores = -np.inf, None
    for _ in range(restarts):
        b = {t.root: np.ones(1)}
        for e in t.edges:
            x = gen.standard_normal(spec.bond_dims[e])
            b[e] = x / np.linalg.norm(x)
        prev = -np.inf
        for _ in range(sweeps):
            for e in t.edges:
                p = t.parent[e]
                grad = _contract_except(red[e], legs(e, b), red[e].ndim - 1)
                grad = grad + _contract_except(red[p], legs(p, b), t.children[p].index(e))
                nrm = np.linalg.norm(grad)
                if nrm > 0:
                    b[e] = grad / nrm
            vals = [float(_contract_except(red[v], legs(v, b), -1)) for v in range(t.n_nodes)]
            val = sum(vals)
            if val - prev <= 1e-12 * abs(val):
                break
            prev = val
        if val > best_val:
            best_val, best_cores = val, vals
    return best_val, best_cores


def init_proxies(spec: TTNSpec, teacher: CPTeacher, u0: np.ndarray, seed: int = 0) -> list[dict]:
    """Per component r: candidate init statistics deciding its escape time."""
    cores = spec.unflatten(torch.from_numpy(u0))
    k = spec.n_cores
    out = []
    for r in range(teacher.n_components):
        vecs = [teacher.factors[r, i] for i in range(teacher.n_sites)]
        red = reduced_cores(spec, cores, vecs)
        f = torch.from_numpy(u0.copy()).requires_grad_(True)
        amp = forward(spec, spec.unflatten(f), [v.unsqueeze(0) for v in vecs])[0]
        (grad,) = torch.autograd.grad(amp, f)
        fmax, per_core = kkt_projection(spec, red, seed=seed)
        out.append(
            dict(
                slice_norm=math.sqrt(sum(float((x**2).sum()) for x in red)),
                amp0=float(amp.detach()),
                grad2=float((grad**2).sum()),
                kkt_proj=fmax / math.sqrt(k),
                core_prod=float(np.prod(np.abs(per_core))),
            )
        )
    return out


# ---- full trajectories and saddle diagnostics ----------------------------------------------------


def gf_trajectory(spec, teacher, y0, t_max, loss_stop=1e-10, rtol=1e-10, atol=1e-15,
                  method="LSODA", dead_floor=1e-25):
    """Full gradient flow with dense output until L < loss_stop * L(0), until some component's
    overlap |a_r| decays below `dead_floor` (its escape route has died: the run is stalled, since
    any later escape would be seeded by roundoff), or t_max.

    Later plateaus sit near learned saddles with O(1) positive curvature, which makes explicit
    integrators step-size limited over plateaus of length sigma^-(k-2); the default is therefore
    LSODA with the exact Hessian as Jacobian (stiff-capable)."""
    loss, rhs = loss_grad_fn(spec, teacher)
    l0 = loss(torch.from_numpy(y0)).item()

    def done(t, y):
        with torch.no_grad():
            return loss(torch.from_numpy(y)).item() - loss_stop * l0

    def dead(t, y):
        return np.abs(overlaps(spec, teacher, y)).min() - dead_floor

    dead.terminal = True
    dead.direction = -1

    def jac(t, y):
        return -torch.autograd.functional.hessian(loss, torch.from_numpy(y)).numpy()

    done.terminal = True
    kw = dict(jac=jac) if method in ("LSODA", "Radau", "BDF") else {}
    return solve_ivp(rhs, (0.0, t_max), y0, method=method, rtol=rtol, atol=atol,
                     events=[done, dead], dense_output=True, **kw)


def crossings(t, x, level=0.5):
    """Times where x crosses `level` (linear interpolation), with direction +1 / -1."""
    out = []
    s = np.sign(x - level)
    for i in np.nonzero(s[1:] != s[:-1])[0]:
        if s[i] == 0:
            continue
        tc = t[i] + (level - x[i]) * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
        out.append((tc, int(np.sign(x[i + 1] - x[i]))))
    return out


def projected_hessian(spec, teacher, y):
    """Gauge-projected Hessian spectrum at y (library `ttn.diagnostics`, default rtol 1e-3 since
    commit 6c42d85), the number of gauge directions removed, and the raw (unprojected) spectrum."""
    from ttn.diagnostics import gauge_complement, gauge_directions, project_out_gauge

    loss, _ = loss_grad_fn(spec, teacher)
    flat = torch.from_numpy(np.asarray(y, dtype=np.float64))
    h = torch.autograd.functional.hessian(loss, flat)
    u = gauge_directions(spec, flat)
    _, eigs = project_out_gauge(h, u)
    raw = torch.linalg.eigvalsh(0.5 * (h + h.T))
    return eigs.numpy(), spec.n_params - gauge_complement(u).shape[1], raw.numpy()


def gf_events(spec, teacher, y0, t_max, rtol=1e-9, atol=1e-14, loss_floor=1e-8, dead_floor=1e-25):
    """Gradient flow (LSODA + exact Hessian) with exact event location of the first up-crossing of
    every a_r = 1/2; stops when L < loss_floor L(0) (converged), when min_r |a_r| < dead_floor (some
    route has died: stalled, any later escape would be seeded by roundoff), or at t_max.

    Returns (times (R,), final overlaps, final loss, min |a_r| before its event (R,), status, t_end,
    number of steps); status 'converged', 'stalled', 't_max' or 'failed'."""
    loss, rhs = loss_grad_fn(spec, teacher)
    l0 = loss(torch.from_numpy(y0)).item()
    n = teacher.n_components

    def up(r):
        def ev(t, y):
            return overlaps(spec, teacher, y)[r] - 0.5
        ev.direction = 1
        return ev

    def done(t, y):
        with torch.no_grad():
            return loss(torch.from_numpy(y)).item() - loss_floor * l0

    def dead(t, y):
        return np.abs(overlaps(spec, teacher, y)).min() - dead_floor

    done.terminal = dead.terminal = True
    dead.direction = -1

    def jac(t, y):
        return -torch.autograd.functional.hessian(loss, torch.from_numpy(y), vectorize=True).numpy()

    sol = solve_ivp(rhs, (0.0, t_max), y0, method="LSODA", rtol=rtol, atol=atol, jac=jac,
                    events=[up(r) for r in range(n)] + [done, dead])
    times = np.array([te[0] if len(te) else np.inf for te in sol.t_events[:n]])
    a_steps = np.array([overlaps(spec, teacher, y) for y in sol.y.T])
    min_dormant = [float(np.abs(a_steps[sol.t < times[r], r]).min()) if np.any(sol.t < times[r])
                   else float("nan") for r in range(n)]
    status = ("failed" if sol.status < 0 else "converged" if len(sol.t_events[n])
              else "stalled" if len(sol.t_events[n + 1]) else "t_max")
    with torch.no_grad():
        l_end = loss(torch.from_numpy(sol.y[:, -1])).item()
    return (times, overlaps(spec, teacher, sol.y[:, -1]), l_end, min_dormant, status,
            float(sol.t[-1]), len(sol.t))
