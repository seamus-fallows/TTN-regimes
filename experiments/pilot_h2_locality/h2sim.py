"""Gradient flow of a TTN on a two-component "pattern" teacher (pilot test of hypothesis H2).

One run = (tree, N, teacher kind, sigma, seed). The flow d theta/dt = -grad L(theta) of the exact
population loss is integrated step by step with an adaptive scipy integrator. Default: LSODA with
the exact Hessian as Jacobian, because once one component is learned the flow is stiff (Hessian
eigenvalues ~K) while the next escape is slow, so explicit Runge-Kutta steps are stability-limited
on long plateaus. Explicit DOP853/RK45 are used to cross-check (check_accuracy.py).
After every accepted step the loss and the component overlaps a_r are evaluated; crossings of the
tracked levels are then located by root-finding on that step's dense-output interpolant, so crossing
times are as accurate as the integration itself, not limited by a sampling grid.
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy import integrate
from scipy.optimize import brentq

from ttn.losses import population_loss
from ttn.network import TTNSpec, aligned_init, bond_spectra, conserved_quantities
from ttn.teachers import pattern_teacher
from ttn.tree import Tree, make_tree

A_LEVELS = (0.1, 0.5, 0.9)  # upward crossings of each overlap a_r
# Downward loss crossings. L0 = 0.625; 0.125 = level with only component 1 learned, 0.5 = only
# component 2 learned; 0.375 and 0.0625 are the midpoints of the two drops in the greedy order.
L_LEVELS = (0.6, 0.5625, 0.5, 0.375, 0.25, 0.125, 0.0625, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

SOLVERS = {
    "RK45": integrate.RK45,
    "DOP853": integrate.DOP853,
    "Radau": integrate.Radau,
    "BDF": integrate.BDF,
    "LSODA": integrate.LSODA,
}
IMPLICIT = ("Radau", "BDF", "LSODA")


def differing_sites(kind: str, n: int) -> list[int]:
    """Sites where the second teacher component differs from the first."""
    if kind == "local":
        return [0, 1]
    if kind == "far":
        return [0, n - 1]
    if kind == "global":
        return list(range(n))
    if kind.startswith("sites"):  # e.g. "sites0-2" for an arbitrary set
        return [int(s) for s in kind[len("sites") :].split("-")]
    raise ValueError(f"unknown teacher kind {kind!r}")


def teacher_patterns(kind: str, n: int) -> list[list[int]]:
    diff = set(differing_sites(kind, n))
    return [[0] * n, [1 if i in diff else 0 for i in range(n)]]


def predicted_k(tree: Tree, kind: str) -> int:
    """Number of cores that must acquire a new direction: the Steiner subtree of the differing
    sites (its edges are exactly the bonds whose teacher rank is 2)."""
    return len(tree.steiner_nodes(differing_sites(kind, tree.n_sites)))


def default_t_max(n_cores: int, sigma: float, factor: float = 1e6) -> float:
    # Generous: second plateaus can be ~1e3 times the first, and runs that converge to the
    # rank-one saddle must be followed long enough to see it.
    return factor * sigma ** (-max(n_cores - 2, 1))


@dataclass
class RunConfig:
    tree: str  # "mps" or "binary"
    n: int
    kind: str
    sigma: float
    seed: int
    d: int = 2
    chi: int = 2
    weights: tuple[float, float] = (1.0, 0.5)
    # Fixed for all runs and unrelated to the init seeds. (Teachers have their own random stream since
    # library commit 57d5262; before it, teacher seed 0 and init seed 0 drew the same Gaussians.)
    teacher_seed: int = 1000
    method: str = "LSODA"
    rtol: float = 1e-10
    atol_factor: float = 1e-3  # atol = rtol * sigma * atol_factor (parameters start at ~sigma/2)
    max_step: float = math.inf
    t_max: float | None = None
    wall_limit: float = 900.0  # seconds
    loss_stop: float = 1e-7
    # "init": fan-in init of scale sigma (the main experiment). "saddle": start at the rank-one saddle
    # where component 1 is exactly learned (balanced, aligned cores) plus a fan-in perturbation of
    # scale sigma, to time the second escape alone.
    start: str = "init"
    t_max_factor: float = 1e6  # t_max = t_max_factor * sigma^-(K-2) unless t_max is given
    # Stop early once W has collapsed onto the rank-one saddle (see `collapsed`).
    collapse_stop: bool = True

    def tag(self) -> str:
        m = "" if (self.method, self.rtol) == ("LSODA", 1e-10) else f"_{self.method}_r{self.rtol:g}"
        if math.isfinite(self.max_step):
            m += f"_h{self.max_step:g}"
        if self.atol_factor != 1e-3:
            m += f"_a{self.atol_factor:g}"
        if self.t_max is not None:
            m += f"_T{self.t_max:g}"
        if self.chi != 2:
            m += f"_chi{self.chi}"
        if self.start != "init":
            m += f"_{self.start}"
        if not self.collapse_stop:
            m += "_nocollapse"
        return f"{self.tree}{self.n}_{self.kind}_sig{self.sigma:.6g}_seed{self.seed}{m}"


class Problem:
    def __init__(self, cfg: RunConfig):
        self.tree = make_tree(cfg.tree, cfg.n)
        self.spec = TTNSpec.uniform(self.tree, cfg.d, cfg.chi)
        self.teacher = pattern_teacher(
            list(cfg.weights), teacher_patterns(cfg.kind, cfg.n), cfg.d, seed=cfg.teacher_seed
        )
        self.tn2 = self.teacher.norm2()

    def loss_t(self, th: torch.Tensor) -> torch.Tensor:
        return population_loss(
            self.spec, self.spec.unflatten(th), self.teacher, teacher_norm2=self.tn2
        )

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        th = torch.from_numpy(y).requires_grad_(True)
        (g,) = torch.autograd.grad(self.loss_t(th), th)
        return -g.numpy()

    def jac(self, t: float, y: np.ndarray) -> np.ndarray:
        h = torch.autograd.functional.hessian(self.loss_t, torch.from_numpy(y), vectorize=True)
        return -h.numpy()

    def diagnostics(self, y: np.ndarray) -> tuple[float, np.ndarray]:
        with torch.no_grad():
            th = torch.from_numpy(y)
            loss = self.loss_t(th).item()
            a = self.teacher.component_overlaps(self.spec, self.spec.unflatten(th)).numpy()
        return loss, a

    def spectra(self, y: np.ndarray, n_sv: int = 2) -> np.ndarray:
        """(n_edges, n_sv) top bond singular values (zero-padded)."""
        with torch.no_grad():
            sp = bond_spectra(self.spec, self.spec.unflatten(torch.from_numpy(y)))
        out = np.zeros((len(self.tree.edges), n_sv))
        for i, v in enumerate(self.tree.edges):
            s = sp[v].numpy()[:n_sv]
            out[i, : len(s)] = s
        return out

    def conservation_drift(self, y0: np.ndarray, y1: np.ndarray) -> tuple[float, float]:
        """max |Q_v(t1) - Q_v(t0)| over edges and entries, and the scale max ||G_v||_F^2 at t1."""
        c0 = self.spec.unflatten(torch.from_numpy(y0))
        c1 = self.spec.unflatten(torch.from_numpy(y1))
        q0, q1 = conserved_quantities(self.spec, c0), conserved_quantities(self.spec, c1)
        drift = max((q1[v] - q0[v]).abs().max().item() for v in q0)
        scale = max(c.pow(2).sum().item() for c in c1)
        return drift, scale


def collapsed(t: list[float], s2max: list[float], t_first: float, s2_ref: float) -> bool:
    """W has collapsed onto the rank-one saddle where only one component is learned.

    s2max = largest second singular value of W over all bonds (at the kept trajectory points).
    Collapsed when, at t >= 30 t_first, s2max is below 1e-9 of its value just after the first
    escape and still falling steeply (1000x within the last factor 3 in time) or at round-off.
    Checked offline on 483 trajectories of an earlier (pre-fix) sweep: it never fired on a run
    that later learned the second component, including runs whose rank-2 content dipped by 1e6
    and recovered; runs it fires on kept decaying to round-off when followed to 1e3-1e4 t_first.
    """
    if t[-1] < 30 * t_first:
        return False
    j = bisect.bisect_left(t, t[-1] / 3)
    now = s2max[-1]
    return now <= 1e-9 * s2_ref and (now <= 1e-3 * s2max[j] or now <= 1e-14)


def crossing_time(f, t0: float, t1: float) -> float:
    """Root of f on [t0, t1] (f changes sign there), to ~1e-15 relative accuracy."""
    f0, f1 = f(t0), f(t1)
    if f0 * f1 > 0:  # rounding at an endpoint that sits exactly on the level
        return t0 if abs(f0) < abs(f1) else t1
    return brentq(f, t0, t1, xtol=1e-12)


def simulate(cfg: RunConfig) -> tuple[dict, dict]:
    """Integrate one run. Returns (summary of scalars, trajectory arrays)."""
    wall0 = time.perf_counter()
    prob = Problem(cfg)
    spec = prob.spec
    n_cores = spec.n_cores
    y0 = spec.flatten(spec.init(cfg.sigma, seed=cfg.seed)).numpy().copy()
    if cfg.start == "saddle":
        amps = torch.zeros(n_cores, 2, dtype=torch.float64)
        amps[:, 0] = cfg.weights[0] ** (1.0 / n_cores)
        y0 += spec.flatten(aligned_init(spec, prob.teacher.factors, amps)).numpy()
    t_max = cfg.t_max if cfg.t_max is not None else default_t_max(n_cores, cfg.sigma, cfg.t_max_factor)
    kwargs = dict(rtol=cfg.rtol, atol=cfg.rtol * cfg.sigma * cfg.atol_factor, max_step=cfg.max_step)
    if cfg.method in IMPLICIT:
        kwargs["jac"] = prob.jac
    solver = SOLVERS[cfg.method](prob.rhs, 0.0, y0, t_max, **kwargs)

    loss, a = prob.diagnostics(y0)
    a_cross = {(r, lev): math.nan for r in range(2) for lev in A_LEVELS}
    l_cross = {lev: math.nan for lev in L_LEVELS}
    a_max_after_half = [math.nan, math.nan]
    a_min_after_half = [math.nan, math.nan]
    traj = {"t": [0.0], "loss": [loss], "a": [a.copy()], "spec": [prob.spectra(y0)]}
    s2max = [float(traj["spec"][0][:, 1].max())]  # largest 2nd bond singular value, per kept point
    s2_ref = None  # largest 2nd bond singular value just after the first escape
    kept = (0.0, loss, a.copy())
    status, n_steps = "running", 0

    def keep(t, y, loss_, a_):
        traj["t"].append(t)
        traj["loss"].append(loss_)
        traj["a"].append(a_.copy())
        traj["spec"].append(prob.spectra(y))
        s2max.append(float(traj["spec"][-1][:, 1].max()))

    while True:
        t_prev, loss_prev, a_prev = solver.t, loss, a
        msg = solver.step()
        n_steps += 1
        if solver.status == "failed":
            status = f"failed: {msg}"
            break
        t, y = solver.t, solver.y
        loss, a = prob.diagnostics(y)
        dense = None
        crossed = False
        for r in range(2):
            for lev in A_LEVELS:
                if math.isnan(a_cross[r, lev]) and a_prev[r] < lev <= a[r]:
                    dense = solver.dense_output() if dense is None else dense
                    a_cross[r, lev] = crossing_time(
                        lambda s: prob.diagnostics(dense(s))[1][r] - lev, t_prev, t
                    )
                    crossed = True
            if not math.isnan(a_cross[r, 0.5]):
                a_max_after_half[r] = np.nanmax([a_max_after_half[r], a[r]])
                a_min_after_half[r] = np.nanmin([a_min_after_half[r], a[r]])
        for lev in L_LEVELS:
            if math.isnan(l_cross[lev]) and loss_prev > lev >= loss:
                dense = solver.dense_output() if dense is None else dense
                l_cross[lev] = crossing_time(
                    lambda s: prob.diagnostics(dense(s))[0] - lev, t_prev, t
                )
                crossed = True
        t_k, loss_k, a_k = kept
        if (
            crossed
            or math.log10(t / max(t_k, 1e-300)) >= 0.01
            or abs(loss - loss_k) >= 2e-3
            or np.abs(a - a_k).max() >= 2e-3
        ):
            keep(t, y, loss, a)
            kept = (t, loss, a.copy())
            learned = [a_cross[r, 0.5] for r in range(2) if not math.isnan(a_cross[r, 0.5])]
            if cfg.collapse_stop and len(learned) == 1:
                if s2_ref is None:
                    s2_ref = s2max[-1]
                elif collapsed(traj["t"], s2max, learned[0], s2_ref):
                    status = "collapsed"
        if status == "collapsed":
            pass
        elif loss < cfg.loss_stop:
            status = "converged"
        elif solver.status == "finished":
            status = "t_max"
        elif time.perf_counter() - wall0 > cfg.wall_limit:
            status = "timeout"
        if status != "running":
            if traj["t"][-1] != t:
                keep(t, y, loss, a)
            break

    drift, scale = prob.conservation_drift(y0, solver.y)
    summary = asdict(cfg)
    summary["weights"] = list(cfg.weights)
    summary.update(
        tag=cfg.tag(),
        n_cores=n_cores,
        k_pred=predicted_k(prob.tree, cfg.kind),
        diff_sites=differing_sites(cfg.kind, cfg.n),
        n_params=spec.n_params,
        t_max_used=t_max,
        status=status,
        t_end=float(solver.t),
        loss_end=float(loss),
        a1_end=float(a[0]),
        a2_end=float(a[1]),
        nfev=int(solver.nfev),
        njev=int(getattr(solver, "njev", 0)),
        n_steps=n_steps,
        wall_s=time.perf_counter() - wall0,
        q_drift=drift,
        q_scale=scale,
        s2_end_max=float(traj["spec"][-1][:, 1].max()),  # largest 2nd bond singular value at end
        s2_max_over_run=float(np.max(np.array(traj["spec"])[:, :, 1])),
        a1_max_after_half=a_max_after_half[0],
        a2_max_after_half=a_max_after_half[1],
        a1_min_after_half=a_min_after_half[0],
        a2_min_after_half=a_min_after_half[1],
    )
    for (r, lev), v in a_cross.items():
        summary[f"t_a{r + 1}_{lev:g}"] = v
    for lev, v in l_cross.items():
        summary[f"t_L_{lev:g}"] = v
    arrays = {
        "t": np.array(traj["t"]),
        "loss": np.array(traj["loss"]),
        "a": np.array(traj["a"]),
        "spectra": np.array(traj["spec"]),
        "edges": np.array(prob.tree.edges),
        "y0": y0,
        "y_end": np.array(solver.y),
    }
    return summary, arrays
