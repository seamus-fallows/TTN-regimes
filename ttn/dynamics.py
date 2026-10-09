"""Training dynamics: gradient flow, gradient descent and SGD on flat parameter vectors.

Conventions
- theta is the flat float64 parameter vector `spec.flatten(cores)`; `spec.unflatten(theta)` gives
  the cores back. Functions taking theta0 also accept a list of cores.
- The training loss L is the exact population loss 1/2 ||W - W*||_M^2 (M_i = site metrics = input
  covariances, identity if None) unless the `Problem` holds a fixed dataset (x, y); then it is the
  empirical loss 1/(2n) sum_j (f(x_j) - y_j)^2.
- Gradient flow: d theta/dt = v(theta) = -(eta * m) grad L(theta), with eta the per-core
  learning-rate multipliers expanded over each core's entries and m the optional 0/1 gradient mask.
  t is gradient-flow time.
- Gradient descent: theta_{k+1} = theta_k + lr v(theta_k), logged at t = lr k, so GD approaches the
  gradient flow at equal t as lr -> 0. SGD uses the same update with a minibatch gradient.
- Two-copy (`antisymmetric_double`) and centred (`ShiftedTeacher`) problems are evaluated without
  cancellation at any init scale (`Problem.effective`).
- Every run returns a `Trajectory`: diagnostics at the log times (see `Recorder`), "fine rows"
  (loss, overlaps, spectra) at the integrator steps where the state changed, which resolve learning
  events in linear time (`merged_rows` joins both), the final parameters `theta_end`, and metadata
  (step counts, wall time, stop reason, problem description).

Gradient flow: `integrate_gf(method='dopri5')` is the reference; past the first escape the flow is
stiff and method='lsoda' is far cheaper (see its docstring). All tensors are tiny, so run with
torch.set_num_threads(1) (one process per core): multithreaded LAPACK calls (QR, Cholesky) are
100-1000x slower on a loaded machine. The scipy solvers (lsoda, radau) factorize P x P matrices
with numpy/scipy's OpenBLAS, which ignores torch.set_num_threads: `integrate_gf` limits it to one
thread itself (threadpoolctl).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.integrate
import torch
from threadpoolctl import threadpool_limits

from ttn.losses import empirical_loss, population_loss, sample_inputs
from ttn.network import (
    Cores,
    TTNSpec,
    antisymmetric_double,
    bond_spectra,
    conserved_quantities,
    sum_difference_cores,
    two_copy_cores,
)
from ttn.teachers import CPTeacher, ShiftedTeacher, TTNTeacher

# ---- problem -----------------------------------------------------------------------------------


@dataclass(eq=False)
class Problem:
    """What is trained, on what, with which per-parameter learning rates.

    site_metrics: input covariances Sigma_i (None = whitened). They define the population loss and
        are the covariances of the fresh inputs drawn by online SGD.
    lr_multipliers: per-core learning-rate multipliers eta_v (None = all 1).
    mask: flat 0/1 gradient mask, e.g. from `antisymmetric_double` (None = no mask).
    dataset: fixed (x, y), x of shape (n, N, d); if given, `loss` is the empirical loss.
    two_copy: theta is an `antisymmetric_double` TTN (W = W(theta+) - W(theta-)). None (default)
        detects it from the mask. Losses then go through `sum_difference_cores`: no cancellation at
        any init scale, and the autograd gradient is the masked one. Q is conserved per copy only.

    With a `ShiftedTeacher` (centred model) the population loss 1/2 ||W - W_0 - W*||^2 is likewise
    evaluated without cancellation (as `centred_population_loss`), and overlaps and spectra refer
    to W - W_0. In all cases `effective(theta)` is the TTN that is fit to `target`.
    """

    spec: TTNSpec
    teacher: CPTeacher | TTNTeacher | ShiftedTeacher
    site_metrics: list[torch.Tensor] | None = None
    lr_multipliers: list[float] | None = None
    mask: torch.Tensor | None = None
    dataset: tuple[torch.Tensor, torch.Tensor] | None = None
    two_copy: bool | None = None

    def __post_init__(self):
        if self.two_copy is None:
            self.two_copy = _is_two_copy_mask(self.spec, self.mask)
        self.shifted = isinstance(self.teacher, ShiftedTeacher)
        self.target = self.teacher.base if self.shifted else self.teacher
        with torch.no_grad():
            self.target_norm2 = self.target.norm2(self.site_metrics).detach()
        precond = None
        if self.lr_multipliers is not None:
            precond = torch.ones(self.spec.n_params, dtype=torch.float64)
            for block, eta in zip(self.spec.core_slices(), self.lr_multipliers):
                precond[block] = eta
        if self.mask is not None:
            mask = self.mask.to(torch.float64)
            precond = mask if precond is None else precond * mask
        self.precond = precond  # eta * m as a flat vector; None means all ones
        self._neg_precond = None if precond is None else -precond

    def effective(self, theta: torch.Tensor) -> tuple[TTNSpec, Cores]:
        """A TTN for the tensor that is fit to `target`, written so that no large terms cancel:
        W(theta); W(theta+) - W(theta-) for the two-copy model; W(theta) - W(theta0) for a
        ShiftedTeacher (on doubled bonds)."""
        cores = self.spec.unflatten(theta)
        if self.shifted:
            spec2, cores2 = two_copy_cores(self.spec, cores, self.teacher.cores0)
            return spec2, sum_difference_cores(spec2, cores2)
        if self.two_copy:
            return self.spec, sum_difference_cores(self.spec, cores)
        return self.spec, cores

    def loss(self, theta: torch.Tensor) -> torch.Tensor:
        """Training loss at flat theta: population loss, or empirical loss if a dataset is set."""
        if self.dataset is None:
            return self.population_loss(theta)
        return self.batch_loss(theta, *self.dataset)

    def batch_loss(self, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """1/(2n) sum_j (f(x_j) - y_j)^2 for the fitted function f (`effective`; f - f_0 for a
        ShiftedTeacher), so labels are values of `target` (plus noise)."""
        return empirical_loss(*self.effective(theta), x, y)

    def population_loss(self, theta: torch.Tensor) -> torch.Tensor:
        """Exact 1/2 ||W(theta) - W*||_M^2 (whatever the training loss is)."""
        spec, cores = self.effective(theta)
        return population_loss(spec, cores, self.target, self.site_metrics, self.target_norm2)

    def loss_and_grad(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = theta.detach().requires_grad_(True)
        loss = self.loss(theta)
        (grad,) = torch.autograd.grad(loss, theta)
        return loss.detach(), grad

    def grad(self, theta: torch.Tensor) -> torch.Tensor:
        return self.loss_and_grad(theta)[1]

    def direction(self, grad: torch.Tensor) -> torch.Tensor:
        """-(eta * m) * grad: the velocity belonging to a (possibly stochastic) gradient."""
        return -grad if self._neg_precond is None else grad * self._neg_precond

    def loss_and_velocity(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loss, grad = self.loss_and_grad(theta)
        return loss, self.direction(grad)

    def velocity(self, theta: torch.Tensor) -> torch.Tensor:
        """The gradient-flow vector field v(theta) = -(eta * m) grad L(theta)."""
        return self.loss_and_velocity(theta)[1]

    def hessian(self, theta: torch.Tensor) -> torch.Tensor:
        """(P, P) Hessian of the training loss (vectorized double backward)."""
        return torch.autograd.functional.hessian(self.loss, theta.detach(), vectorize=True)

    def jacobian(self, theta: torch.Tensor) -> torch.Tensor:
        """dv/dtheta = -diag(eta * m) H, the Jacobian of the gradient-flow vector field."""
        hess = self.hessian(theta)
        return -hess if self.precond is None else self._neg_precond[:, None] * hess

    def describe(self) -> dict:
        """JSON-friendly summary stored in every Trajectory's meta. For a CP target also
        teacher_weights, component_targets (<W*, M U_r>, the denominators of the overlaps) and
        orthogonal_components (Gram matrix <U_r, M U_r'> diagonal: then each overlap is one learning
        event; otherwise one escape can raise several overlaps, see notes/diagnostics.md)."""
        t, teacher = self.spec.tree, self.teacher
        out = {
            "tree": t.name,
            "parent": list(t.parent),
            "sites": [list(s) for s in t.sites],
            "phys_dims": list(self.spec.phys_dims),
            "bond_dims": list(self.spec.bond_dims),
            "n_params": self.spec.n_params,
            "teacher": type(teacher).__name__,
            "target_norm2": float(self.target_norm2),
            "site_metrics": self.site_metrics is not None,
            "lr_multipliers": None if self.lr_multipliers is None else list(self.lr_multipliers),
            "masked": self.mask is not None,
            "two_copy": self.two_copy,
            "n_samples": None if self.dataset is None else int(self.dataset[0].shape[0]),
        }
        if isinstance(self.target, CPTeacher):
            out["teacher_weights"] = self.target.weights.tolist()
            with torch.no_grad():
                gram = self.target.gram(self.site_metrics)
                off = gram - torch.diag(torch.diagonal(gram))
                out["component_targets"] = (gram @ self.target.weights).tolist()
                diagonal = off.abs().max() <= 1e-12 * torch.diagonal(gram).abs().max()
                out["orthogonal_components"] = bool(diagonal)
        return out


def _is_two_copy_mask(spec: TTNSpec, mask: torch.Tensor | None) -> bool:
    """Whether `mask` is the block mask that `antisymmetric_double` produces for `spec`."""
    t = spec.tree
    if mask is None or t.n_nodes == 1 or any(spec.bond_dims[v] % 2 for v in t.edges):
        return False
    half_bonds = tuple(1 if v == t.root else spec.bond_dims[v] // 2 for v in range(t.n_nodes))
    half = TTNSpec(t, spec.phys_dims, half_bonds)
    zeros = [torch.zeros(shape, dtype=torch.float64) for shape in half.shapes]
    return torch.equal(mask.to(torch.float64), antisymmetric_double(half, zeros)[2])


def make_dataset(
    problem: Problem, n: int, generator: torch.Generator, label_noise: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """n inputs x^(i) ~ N(0, site_metrics[i]) with labels y = f*(x) + label_noise * N(0, 1), f* the
    problem's target (the base teacher of a ShiftedTeacher, see `Problem.batch_loss`).

    Use as `dataclasses.replace(problem, dataset=make_dataset(problem, n, g))`. Assumes the same
    physical dimension at every site.
    """
    spec = problem.spec
    x = sample_inputs(n, spec.tree.n_sites, spec.phys_dims[0], generator, problem.site_metrics)
    with torch.no_grad():
        y = problem.target(x)
    if label_noise > 0:
        y = y + label_noise * torch.randn(n, generator=generator, dtype=y.dtype)
    return x, y


# ---- diagnostics -------------------------------------------------------------------------------


class Recorder:
    """Diagnostics evaluated at every log time; `arrays()` stacks them into numpy arrays.

    Default (cheap) set, with n log times, K cores, E edges (in `spec.tree.edges` order):
      t (n,)                    time
      loss (n,)                 training loss (population, or empirical if the problem has a dataset)
      pop_loss (n,)             exact population loss (equal to loss without a dataset)
      speed (n,)                |v(theta)| = |d theta/dt| of the gradient flow of the training loss
                                (plateau midpoints are its minima, see notes/diagnostics.md)
      param_norm (n,)           |theta|
      rel_dist_from_init (n,)   |theta - theta0| / |theta0|
      core_norms (n, K)         Frobenius norm of every core
      spectra (n, E, chi_max)   singular values of the fitted tensor (`Problem.effective`: W, or
                                W+ - W- / W - W_0) across every edge, decreasing, NaN-padded to the
                                largest bond dimension; with site metrics the function-space ones,
                                of (M_1^1/2 (x) ... (x) M_N^1/2) W
      overlaps (n, R)           CP targets: `component_overlaps` of the fitted tensor
                                (a_r = <W, M U_r> / <W*, M U_r>, 1 when component r is learned)
      conservation_drift (n,)   max_v |Q_v(t) - Q_v(0)|_max / (max_v |Q_v(0)|_max + max_u |G_u|_F^2/eta_u)
    plus what the integrator passes to `record` (step, batch_loss, ...).

    Q_v (`conserved_quantities` with the lr multipliers; per copy for the two-copy model) is a
    difference of two Gram matrices, each bounded entrywise by max_u |G_u|_F^2 / eta_u (cores at
    time t), so the drift measures the integration error relative to the size of the terms that
    cancel; |Q(0)| keeps the scale for strongly imbalanced inits. Exact gradient flow keeps it at
    roundoff (~1e-15); at the default tolerances DP5 gives ~1e-10, LSODA 1e-12 to ~5e-7 (runs to
    t = 1e10 with K = 7).

    Options: spectra / overlaps / conservation switch those diagnostics off; save_params stores
    theta at every log time as params (n, P); extra = [(name, fn(problem, theta, t) -> array)].

    Fine rows (fine=True): a log grid cannot resolve learning events (an escape lasts 0.5-2 time
    units at t ~ 10-1e4, where log points are 8% of t apart), so the integrators also call `step`
    after every accepted step (GF), every step (GD, SGD), and a light row is kept whenever, since the
    last kept row, |theta| changed by fine_tol relatively or the training loss by fine_tol in
    log (below 1e-12 L(0) loss changes are ignored, they are roundoff; the loss trigger is not used
    by SGD, whose step loss is a minibatch loss). Kept as
      fine_t (m,), fine_loss (m,)   time and training loss
      fine_overlaps (m, R)          as overlaps (if recorded)
      fine_spectra (m, E, chi_max)  as spectra (if recorded)
    Their cost is one forward pass per step plus the row's diagnostics; DP5 and LSODA take 20-200
    steps per learning event, so events are sampled at least ~20 times. `merged_rows` joins log and
    fine rows into one time series.
    """

    def __init__(
        self,
        spectra: bool = True,
        overlaps: bool = True,
        conservation: bool = True,
        save_params: bool = False,
        extra: list[tuple[str, Callable]] | tuple = (),
        fine: bool = True,
        fine_tol: float = 0.01,
    ):
        self.spectra = spectra
        self.overlaps = overlaps
        self.conservation = conservation
        self.save_params = save_params
        self.extra = list(extra)
        self.fine = fine
        self.fine_tol = fine_tol

    @torch.no_grad()
    def start(self, problem: Problem, theta0: torch.Tensor) -> None:
        """Reset the logs and store the reference quantities of the initial point."""
        spec = problem.spec
        self.rows: dict[str, list] = {}
        self.fine_rows: dict[str, list] = {}
        self.theta0 = theta0.detach().clone()
        self.norm0 = float(self.theta0.norm())
        self.eta = problem.lr_multipliers or [1.0] * spec.n_cores
        self.n_copies = 2 if problem.two_copy else 1
        if self.conservation:
            cores0 = spec.unflatten(self.theta0)
            self.q0 = conserved_quantities(spec, cores0, problem.lr_multipliers, self.n_copies)
            self.q0_scale = max(q.abs().max().item() for q in self.q0.values())
        spec_eff, _ = problem.effective(self.theta0)
        self.max_bond = max(spec_eff.bond_dims[v] for v in spec_eff.tree.edges)
        if self.fine:
            loss0 = problem.loss(self.theta0).item()
            self._loss_clip = 1e-12 * loss0 if loss0 > 0 else np.finfo(float).tiny
            self._fine_last = (self.theta0, self.norm0, self._log_loss(loss0))

    def record(
        self,
        problem: Problem,
        theta: torch.Tensor,
        t: float,
        loss: float | None = None,
        velocity: torch.Tensor | None = None,
        **info,
    ) -> None:
        """Log row at time t. loss / velocity: the training loss and v(theta) if the caller has
        them already (integrators pass theirs, which saves a gradient evaluation per row)."""
        theta = theta.detach()
        row = self._builtin(problem, theta, t, loss)
        if velocity is None:
            velocity = problem.velocity(theta)
        row["speed"] = velocity.norm().item()
        row.update(info)
        for name, fn in self.extra:
            row[name] = np.asarray(fn(problem, theta, t))
        for key, value in row.items():
            self.rows.setdefault(key, []).append(value)

    def step(self, problem: Problem, theta: torch.Tensor, t: float, loss: float | None = None):
        """Integrator step at time t: keep a fine row if the state moved enough (class docstring).
        loss: the training loss at theta, if known (None: only the parameter trigger is used)."""
        if not self.fine:
            return
        theta = theta.detach()
        last_theta, last_norm, last_log_loss = self._fine_last
        moved = (theta - last_theta).norm().item() >= self.fine_tol * last_norm
        log_loss = None if loss is None else self._log_loss(loss)
        if moved or (log_loss is not None and abs(log_loss - last_log_loss) >= self.fine_tol):
            self._fine_row(problem, theta, t, loss)

    def _log_loss(self, loss: float) -> float:
        return math.log(max(loss, self._loss_clip))

    @torch.no_grad()
    def _fine_row(self, problem: Problem, theta: torch.Tensor, t: float, loss: float | None):
        if loss is None:
            loss = problem.loss(theta).item()
        row = {"fine_t": float(t), "fine_loss": loss}
        spec_eff, cores_eff = problem.effective(theta)
        if self.overlaps and isinstance(problem.target, CPTeacher):
            row["fine_overlaps"] = problem.target.component_overlaps(
                spec_eff, cores_eff, problem.site_metrics
            ).numpy()
        if self.spectra:
            row["fine_spectra"] = self._padded_spectra(spec_eff, cores_eff, problem.site_metrics)
        for key, value in row.items():
            self.fine_rows.setdefault(key, []).append(value)
        self._fine_last = (theta.clone(), theta.norm().item(), self._log_loss(loss))

    def _padded_spectra(self, spec_eff: TTNSpec, cores_eff: Cores, metrics) -> np.ndarray:
        spectra = bond_spectra(spec_eff, cores_eff, metrics)
        padded = np.full((len(spec_eff.tree.edges), self.max_bond), np.nan)
        for j, v in enumerate(spec_eff.tree.edges):
            padded[j, : spectra[v].numel()] = spectra[v].numpy()
        return padded

    @torch.no_grad()
    def _builtin(self, problem: Problem, theta: torch.Tensor, t: float, loss) -> dict:
        spec, metrics = problem.spec, problem.site_metrics
        cores = spec.unflatten(theta)
        loss = problem.loss(theta).item() if loss is None else float(loss)
        pop = loss if problem.dataset is None else problem.population_loss(theta).item()
        row = {
            "t": float(t),
            "loss": loss,
            "pop_loss": pop,
            "param_norm": theta.norm().item(),
            "rel_dist_from_init": (theta - self.theta0).norm().item() / self.norm0,
            "core_norms": torch.stack([c.norm() for c in cores]).numpy(),
        }
        spec_eff, cores_eff = problem.effective(theta)
        if self.spectra:
            row["spectra"] = self._padded_spectra(spec_eff, cores_eff, metrics)
        if self.overlaps and isinstance(problem.target, CPTeacher):
            row["overlaps"] = problem.target.component_overlaps(
                spec_eff, cores_eff, metrics
            ).numpy()
        if self.conservation:
            q = conserved_quantities(spec, cores, problem.lr_multipliers, self.n_copies)
            change = max((q[v] - self.q0[v]).abs().max().item() for v in q)
            gram = max(c.pow(2).sum().item() / eta for c, eta in zip(cores, self.eta))
            row["conservation_drift"] = change / (self.q0_scale + gram)
        if self.save_params:
            row["params"] = theta.numpy().copy()
        return row

    def arrays(self) -> dict[str, np.ndarray]:
        rows = self.rows | getattr(self, "fine_rows", {})
        return {key: np.array(values) for key, values in rows.items()}


def merged_rows(data: dict, keys=("loss", "overlaps", "spectra")) -> dict[str, np.ndarray]:
    """Log rows and fine rows of a recorded trajectory (`Recorder`) as one time series sorted by
    time: {"t": ..., key: ...} for each key recorded in both (a fine row at a log time is dropped).
    Without fine rows (fine=False, or GD/SGD logged at every step), just the log rows."""
    keys = [k for k in keys if k in data]
    t = data["t"]
    if "fine_t" not in data:
        return {"t": t} | {k: data[k] for k in keys}
    new = ~np.isin(data["fine_t"], t)
    order = np.argsort(np.concatenate([t, data["fine_t"][new]]), kind="stable")
    out = {"t": np.concatenate([t, data["fine_t"][new]])[order]}
    for k in keys:
        if f"fine_{k}" in data:
            out[k] = np.concatenate([data[k], data[f"fine_{k}"][new]])[order]
    return out


# ---- trajectory --------------------------------------------------------------------------------


@dataclass
class Trajectory:
    """`data`: numpy arrays with the log index first (see `Recorder`) plus `theta_end`, the final
    parameters (P,). `meta`: JSON-serializable run information (stop_reason, n_steps, wall_time,
    problem description, ...)."""

    data: dict[str, np.ndarray]
    meta: dict

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def save(self, path: str | Path) -> Path:
        """Write a single .npz file; meta is stored inside as a JSON string under '__meta__'."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        meta = np.array(json.dumps(self.meta, default=_to_json))
        np.savez_compressed(path, __meta__=meta, **self.data)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["__meta__"]))
            data = {k: f[k] for k in f.files if k != "__meta__"}
        return cls(data, meta)


def _to_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray | np.generic):
        return obj.tolist()
    return str(obj)


# ---- gradient flow -----------------------------------------------------------------------------

# Dormand-Prince 5(4) tableau. Row 6 equals the 5th-order weights, so the 7th stage is evaluated at
# the new point and is reused as the first stage of the next step (FSAL).
_DP_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
# 5th-order minus embedded 4th-order weights; the error estimate is h * sum_i E_i k_i.
_DP_E = (71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)
_SAFETY, _FACTOR_MIN, _FACTOR_MAX = 0.9, 0.2, 5.0


def log_time_grid(t_min: float, t_max: float, n_log: int = 200) -> np.ndarray:
    """t = 0 followed by n_log log-spaced times from t_min to t_max."""
    return np.concatenate([[0.0], np.geomspace(t_min, t_max, n_log)])


def default_log_times(
    problem: Problem, theta0: torch.Tensor | Cores, t_max: float, n_log: int = 200
) -> np.ndarray:
    """The default log grid of `integrate_gf`: log_time_grid(t_min, t_max, n_log) with
    t_min = 1e-3 min(tau_0, tau_L, t_max).

    tau_0 = |theta0| / |v(theta0)| is the time over which theta would move by its own size at the
    initial speed, tau_L = L(theta0) / |dL/dt(theta0)| (dL/dt = grad L . v) the time over which
    the loss would change by its own size. At small init tau_0 is the smaller one (the first log
    point is ~1e-3 of the first plateau); at large init the function moves ~sigma^K times faster
    than the parameters and tau_L is (the lazy fit is over long before tau_0). The grid always spans
    at least three decades.
    """
    theta0 = _as_flat(problem.spec, theta0)
    loss, grad = problem.loss_and_grad(theta0)
    velocity = problem.direction(grad)
    speed, rate = velocity.norm().item(), abs((grad * velocity).sum().item())
    scales = [t_max]
    if speed > 0:
        scales.append(theta0.norm().item() / speed)
    if rate > 0:
        scales.append(loss.item() / rate)
    scales = [s for s in scales if s > 0]
    return log_time_grid(1e-3 * min(scales), t_max, n_log)


def integrate_gf(
    problem: Problem,
    theta0: torch.Tensor | Cores,
    t_max: float,
    log_times=None,
    n_log: int = 200,
    rtol: float = 1e-8,
    atol: float = 1e-12,
    max_steps: int = 10**7,
    loss_floor: float | None = None,
    recorder: Recorder | None = None,
    method: str = "dopri5",
    max_wall_time: float | None = None,
) -> Trajectory:
    """Gradient flow d theta/dt = v(theta), recording diagnostics at the log times.

    method='dopri5' (default, the reference): adaptive Dormand-Prince 5(4) with FSAL and local
    extrapolation. A step of size h is accepted if
        err = rms( h sum_i E_i k_i / (atol + rtol * max(|y|, |y_new|)) ) <= 1,
    and the next step is h * min(5, max(0.2, 0.9 err^-1/5)) (no increase right after a rejection).
    The first step uses Hairer's selection (Hairer, Norsett & Wanner I, II.4). On a plateau near
    the origin the solution varies on the time scale |theta| / |v|, so steps grow to a fixed
    fraction (~rtol^1/5) of it and the plateau costs O(log) steps whatever its length.

    Stiffness: once a component has been learned the Hessian has O(1) eigenvalues lambda (K s^(2-2/K)
    for an odeco component of weight s), while the next escape still takes ~sigma^-(K-2). An
    explicit step is stable only for h lambda < ~3.3, so DP5 needs ~T lambda / 3.3 steps for a later
    plateau of length T (and after convergence, until t_max: use loss_floor). Hairer's test
    (h |f(y_new) - f(y_6)| / |y_new - y_6| > 3.25 on 15 consecutive steps) detects this; the first
    such time is stored in meta['stiff_at']. Without a loss_floor a warning is issued at once if
    reaching t_max at the current step would take > 1e5 steps; with a loss_floor (which usually ends
    the run much earlier) only at the end, if more than 1e5 steps were spent after stiff_at.

    method='lsoda' (recommended past the first escape, e.g. saddle-to-saddle runs at K >= 4):
    ODEPACK's LSODA through scipy, which switches between Adams (nonstiff) and BDF (stiff, with the
    exact Hessian `Problem.jacobian`); meta['stiff_at'] is its first BDF step. Global error ~1e-7
    at rtol = 1e-8 (DP5: ~5e-8). method='radau': scipy's implicit Radau IIA (order 5), ~100x more
    accurate at equal rtol but ~2x more expensive, and it crawls on very flat plateaus: there the
    gradient (|v| ~ 1e-12) carries ~1e-16 absolute roundoff, and scipy's Newton test rejects
    corrections that stop shrinking even when they are far below tolerance (scipy's BDF too, which
    is why it is not offered). Both use scipy's error norms with the same rtol/atol.

    Log times are hit exactly by clamping the step (no interpolation): {0} + log_times + {t_max},
    restricted to [0, t_max]. The scipy solvers are not restarted at log times (their bound, and
    LSODA's TCRIT, are moved), so they keep step size, Jacobian and history. Default grid:
    `default_log_times(problem, theta0, t_max, n_log)`. Between log times the recorder keeps fine
    rows at the accepted steps (`Recorder`), which resolve learning events.

    Stops at t_max ('completed'), when the training loss drops below loss_floor ('converged'), after
    max_steps accepted steps ('max_steps'), after max_wall_time seconds ('max_wall_time'), or on
    failure: for dopri5, step-size underflow ('nonfinite' if caused by non-finite values, which
    otherwise only lead to rejected steps; else 'step_size_underflow'); for the implicit methods,
    scipy's failure ('failed', see meta['message']). An early stop records the final state as the
    last log entry.
    """
    wall0 = time.perf_counter()
    rec = Recorder() if recorder is None else recorder
    y = _as_flat(problem.spec, theta0)
    loss, f = problem.loss_and_velocity(y)
    if log_times is None:
        log_times = default_log_times(problem, y, t_max, n_log)
    times = _merge_times(log_times, t_max)
    deadline = math.inf if max_wall_time is None else wall0 + max_wall_time
    rec.start(problem, y)
    rec.record(problem, y, 0.0, loss=loss.item(), velocity=f, step=0)
    if method == "dopri5":
        y, t, info = _dopri5(
            problem, y, loss, f, times, rtol, atol, max_steps, loss_floor, rec, deadline
        )
    else:
        # LU factorizations of P x P Jacobians go through scipy's OpenBLAS, whose default thread
        # pool ignores torch.set_num_threads (a 688 x 688 LU: 0.8 s with 24 threads, 6 ms with one).
        with threadpool_limits(limits=1, user_api="blas"):
            y, t, info = _implicit(
                problem, y, times, method, rtol, atol, max_steps, loss_floor, rec, deadline
            )
    meta = {
        "integrator": method,
        "rtol": rtol,
        "atol": atol,
        "t_max": t_max,
        "t_end": t,
        **info,
        "wall_time": time.perf_counter() - wall0,
        "problem": problem.describe(),
    }
    return _trajectory(rec, y, meta)


def _dopri5(problem, y, loss, f, times, rtol, atol, max_steps, loss_floor, rec, deadline):
    n = y.numel()
    k = torch.empty(7, n, dtype=y.dtype)
    k[0] = f
    a_rows = [torch.tensor(row, dtype=y.dtype) for row in _DP_A]
    e = torch.tensor(_DP_E, dtype=y.dtype)
    inv_sqrt_n = 1.0 / math.sqrt(n)
    t, t_logged, i_log = 0.0, 0.0, 1
    n_steps, n_rejected, n_evals = 0, 0, 1
    n_stiff, n_nonstiff, stiff_at, stiff_step = 0, 0, None, 0
    loss_val = loss.item()
    after_reject = False
    stop = None
    if not (math.isfinite(loss_val) and bool(torch.isfinite(f).all())):
        stop = "nonfinite"
    elif len(times) == 1:
        stop = "completed"
    else:
        h = _initial_step(problem, y, f, rtol, atol)
        n_evals += 1

    while stop is None:
        t_next = times[i_log]
        hit = h >= t_next - t
        h_step = t_next - t if hit else h
        for i in range(1, 7):
            y_new = torch.addmv(y, k[:i].T, a_rows[i], alpha=h_step)
            loss_new, k[i] = problem.loss_and_velocity(y_new)
            if i == 5:
                y6 = y_new
        n_evals += 6
        scale = torch.maximum(y.abs(), y_new.abs()).mul_(rtol).add_(atol)
        err = torch.mv(k.T, e).mul_(h_step).div_(scale).norm().item() * inv_sqrt_n
        loss_new = loss_new.item()

        if err <= 1.0 and math.isfinite(loss_new):
            h_lambda = (h_step * (k[6] - k[5]).norm() / (y_new - y6).norm()).item()
            if h_lambda > 3.25:
                n_stiff, n_nonstiff = n_stiff + 1, 0
                if n_stiff == 15 and stiff_at is None:
                    stiff_at, stiff_step = t, n_steps
                    remaining = (times[-1] - t) / h
                    if remaining > 1e5 and loss_floor is None:
                        warnings.warn(
                            f"gradient flow became stiff at t = {t:.4g}; at the stability-limited "
                            f"step, reaching t_max takes ~{remaining:.1e} more steps. Use "
                            "method='lsoda' (or a loss_floor if training has converged).",
                            stacklevel=3,
                        )
            else:
                n_nonstiff += 1
                if n_nonstiff == 6:
                    n_stiff = 0
            factor = _FACTOR_MAX if err == 0.0 else min(_FACTOR_MAX, _SAFETY * err**-0.2)
            if after_reject:
                factor = min(1.0, factor)
            h_new = h_step * factor
            if hit and factor == _FACTOR_MAX:
                # The step was shortened to land on a log time; keep the earlier, larger proposal.
                h_new = max(h_new, h)
            h = h_new
            y = y_new
            k[0] = k[6]
            loss_val = loss_new
            n_steps += 1
            after_reject = False
            t = t_next if hit else t + h_step
            rec.step(problem, y, t, loss=loss_val)
            if hit:
                rec.record(problem, y, t, loss=loss_val, velocity=k[0], step=n_steps)
                t_logged = t
                i_log += 1
                if i_log == len(times):
                    stop = "completed"
            if stop is None:
                if loss_floor is not None and loss_val < loss_floor:
                    stop = "converged"
                elif n_steps >= max_steps:
                    stop = "max_steps"
                elif time.perf_counter() > deadline:
                    stop = "max_wall_time"
        else:
            n_rejected += 1
            after_reject = True
            finite = math.isfinite(err) and math.isfinite(loss_new)
            factor = max(_FACTOR_MIN, _SAFETY * err**-0.2) if finite else _FACTOR_MIN
            h = h_step * factor
            if t + h == t:
                stop = "step_size_underflow" if finite else "nonfinite"
        if stop is not None and t != t_logged:
            rec.record(problem, y, t, loss=loss_val, velocity=k[0], step=n_steps)

    if stiff_at is not None and loss_floor is not None and n_steps - stiff_step > 1e5:
        warnings.warn(
            f"gradient flow became stiff at t = {stiff_at:.4g} and DP5 then took "
            f"{n_steps - stiff_step} stability-limited steps; method='lsoda' is much cheaper.",
            stacklevel=3,
        )
    info = {
        "loss_end": loss_val,
        "stop_reason": stop,
        "n_steps": n_steps,
        "n_rejected": n_rejected,
        "n_evals": n_evals,
        "stiff_at": stiff_at,
    }
    return y, t, info


_SCIPY_SOLVERS = {"lsoda": scipy.integrate.LSODA, "radau": scipy.integrate.Radau}


def _implicit(problem, y, times, method, rtol, atol, max_steps, loss_floor, rec, deadline):
    def fun(_, theta):
        return problem.velocity(torch.from_numpy(theta)).numpy()

    def jac(_, theta):
        return problem.jacobian(torch.from_numpy(theta)).numpy()

    t, t_logged, i_log, n_steps, stop, message, stiff_at = 0.0, 0.0, 1, 0, None, None, None
    counts = {"nfev": 0, "njev": 0, "nlu": 0}
    if len(times) == 1:
        stop = "completed"
    else:
        solver = _SCIPY_SOLVERS[method](
            fun, 0.0, y.numpy().copy(), times[1], rtol=rtol, atol=atol, jac=jac
        )
    while stop is None:
        message = solver.step()
        if solver.status == "failed":
            stop = "failed"
            break
        n_steps += 1
        y, t = torch.from_numpy(solver.y.copy()), solver.t
        loss = _loss_value(problem, y)
        rec.step(problem, y, t, loss=loss)
        # LSODA's IWORK(19) is the method of the last step: 1 = Adams (nonstiff), 2 = BDF (stiff).
        if (
            method == "lsoda"
            and stiff_at is None
            and solver._lsoda_solver._integrator.iwork[18] == 2
        ):
            stiff_at = t
        if solver.status == "finished":
            rec.record(problem, y, t, loss=loss, step=n_steps)
            t_logged = t
            i_log += 1
            if i_log == len(times):
                stop = "completed"
            else:
                # Moving the bound (instead of restarting) keeps the step size, Jacobian and history.
                solver.t_bound, solver.status = times[i_log], "running"
                if method == "lsoda":
                    # scipy's wrapper sets TCRIT = RWORK(1) only at construction; ODEPACK reads it on
                    # every call, so steps keep landing exactly on the bound.
                    solver._lsoda_solver._integrator.rwork[0] = times[i_log]
        if stop is None:
            if loss_floor is not None and loss < loss_floor:
                stop = "converged"
            elif n_steps >= max_steps:
                stop = "max_steps"
            elif time.perf_counter() > deadline:
                stop = "max_wall_time"
    if t != t_logged:
        rec.record(problem, y, t, step=n_steps)
    if len(times) > 1:
        counts = {"nfev": int(solver.nfev), "njev": int(solver.njev), "nlu": int(solver.nlu)}

    info = {
        "loss_end": _loss_value(problem, y),
        "stop_reason": stop,
        "message": message,
        "n_steps": n_steps,
        **counts,
        "stiff_at": stiff_at,
    }
    return y, t, info


def _initial_step(problem: Problem, y0, f0, rtol: float, atol: float, order: int = 5) -> float:
    """Hairer's initial step: h0 = 0.01 |y0| / |f0|, then h1 with h1^order * max(|f0|, |f'|) = 0.01
    (norms scaled by atol + rtol |y0|), and h = min(100 h0, h1)."""
    scale = atol + rtol * y0.abs()
    d0, d1 = _rms(y0 / scale), _rms(f0 / scale)
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    f1 = problem.velocity(y0 + h0 * f0)
    d2 = _rms((f1 - f0) / scale) / h0
    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / order)
    return min(100 * h0, h1)


def _rms(x: torch.Tensor) -> float:
    return x.norm().item() / math.sqrt(x.numel())


def _merge_times(log_times, t_max: float) -> list[float]:
    ts = np.asarray(log_times, dtype=np.float64)
    ts = np.unique(ts[(ts > 0) & (ts < t_max)])
    return [0.0] + ts.tolist() + ([float(t_max)] if t_max > 0 else [])


# ---- gradient descent and SGD ------------------------------------------------------------------


def run_gd(
    problem: Problem,
    theta0: torch.Tensor | Cores,
    lr: float,
    n_steps: int,
    log_steps=None,
    n_log: int = 200,
    loss_floor: float | None = None,
    diverge_factor: float = 1e6,
    recorder: Recorder | None = None,
    max_wall_time: float | None = None,
) -> Trajectory:
    """Gradient descent theta_{k+1} = theta_k + lr v(theta_k) on the training loss, logged at
    t = lr * k (fine rows: see `Recorder`).

    Log steps: {0} + log_steps + {n_steps}; by default ~n_log log-spaced integers in [1, n_steps].
    Stops after n_steps ('completed'), when the loss drops below loss_floor ('converged'; the final
    state is recorded), after max_wall_time seconds ('max_wall_time'; recorded), or when the loss
    becomes non-finite or exceeds diverge_factor * L(theta0) ('diverged'; the diverged state is not
    recorded, meta holds its step and loss).
    """
    wall0 = time.perf_counter()
    deadline = math.inf if max_wall_time is None else wall0 + max_wall_time
    rec = Recorder() if recorder is None else recorder
    theta = _as_flat(problem.spec, theta0)
    steps = _merge_steps(log_steps, n_steps, n_log)
    rec.start(problem, theta)
    i_log, stop, loss0 = 0, None, None
    for step in range(n_steps + 1):
        loss, velocity = problem.loss_and_velocity(theta)
        loss = loss.item()
        loss0 = loss if loss0 is None else loss0
        if not math.isfinite(loss) or loss > diverge_factor * loss0:
            stop = "diverged"
            break
        if step > 0:
            rec.step(problem, theta, lr * step, loss=loss)
        logged = steps[i_log] == step
        if logged:
            rec.record(problem, theta, lr * step, loss=loss, velocity=velocity, step=step)
            i_log += 1
        if loss_floor is not None and loss < loss_floor:
            stop = "converged"
        elif step == n_steps:
            stop = "completed"
        elif time.perf_counter() > deadline:
            stop = "max_wall_time"
        if stop is not None:
            if not logged:
                rec.record(problem, theta, lr * step, loss=loss, velocity=velocity, step=step)
            break
        theta = torch.add(theta, velocity, alpha=lr)
    meta = {
        "integrator": "gd",
        "lr": lr,
        "n_steps_max": n_steps,
        "n_steps": step,
        "n_evals": step + 1,
        "t_end": lr * step,
        "loss_end": loss,
        "stop_reason": stop,
        "wall_time": time.perf_counter() - wall0,
        "problem": problem.describe(),
    }
    return _trajectory(rec, theta, meta)


def run_sgd(
    problem: Problem,
    theta0: torch.Tensor | Cores,
    lr: float,
    n_steps: int,
    batch_size: int,
    generator: torch.Generator,
    label_noise: float = 0.0,
    dataset: tuple[torch.Tensor, torch.Tensor] | None = None,
    log_steps=None,
    n_log: int = 200,
    loss_floor: float | None = None,
    diverge_factor: float = 1e6,
    recorder: Recorder | None = None,
    max_wall_time: float | None = None,
) -> Trajectory:
    """Minibatch SGD theta_{k+1} = theta_k - lr (eta * m) grad L_B(theta_k) with
    L_B = 1/(2B) sum_b (f(x_b) - y_b)^2 (`Problem.batch_loss`), logged at t = lr * k (fine rows:
    see `Recorder`; the parameter trigger only).

    Online (no dataset, here or in the problem): fresh inputs x ~ N(0, site_metrics) every step
    (`sample_inputs` draws; Cholesky factors computed once) with labels y = f*(x) of the target.
    With a fixed dataset (argument, else problem.dataset): random reshuffling, i.e. a new
    permutation each epoch cut into batches (batch_size >= n gives full-batch GD on the empirical
    loss). label_noise adds fresh N(0, label_noise^2) noise to the batch labels at every step in
    both modes (label-noise SGD).

    Logs, besides the Recorder set (where `loss` is the training loss and `pop_loss` the exact
    population loss): step, batch_loss (minibatch loss at the logged step, before its update) and
    batch_loss_avg (mean minibatch loss since the previous log step). Stops after n_steps
    ('completed'), when the training loss at a log step is below loss_floor ('converged'), after
    max_wall_time seconds ('max_wall_time', checked at log steps), or when the minibatch loss is
    non-finite or exceeds diverge_factor * L(theta0) ('diverged').
    """
    wall0 = time.perf_counter()
    deadline = math.inf if max_wall_time is None else wall0 + max_wall_time
    if dataset is not None:
        problem = dataclasses.replace(problem, dataset=dataset)
    rec = Recorder() if recorder is None else recorder
    spec, data = problem.spec, problem.dataset
    n_sites, d = spec.tree.n_sites, spec.phys_dims[0]
    theta = _as_flat(spec, theta0)
    steps = _merge_steps(log_steps, n_steps, n_log)
    rec.start(problem, theta)
    loss0 = _loss_value(problem, theta)
    if data is not None:
        n_data = data[0].shape[0]
        batch = min(batch_size, n_data)
        perm, pos = torch.randperm(n_data, generator=generator), 0
    elif problem.site_metrics is not None:
        # sample_inputs(..., site_covs) would refactor the covariances at every step; same draws.
        chol = torch.stack([torch.linalg.cholesky(m) for m in problem.site_metrics])
    i_log, stop, loss_sum, loss_count = 0, None, 0.0, 0
    for step in range(n_steps + 1):
        if data is None:
            x = sample_inputs(batch_size, n_sites, d, generator)
            if problem.site_metrics is not None:
                x = torch.einsum("bnk,ndk->bnd", x, chol)
            with torch.no_grad():
                y = problem.target(x)
        else:
            if pos + batch > n_data:
                perm, pos = torch.randperm(n_data, generator=generator), 0
            idx = perm[pos : pos + batch]
            pos += batch
            x, y = data[0][idx], data[1][idx]
        if label_noise > 0:
            y = y + label_noise * torch.randn(y.shape[0], generator=generator, dtype=y.dtype)
        theta_req = theta.detach().requires_grad_(True)
        batch_loss = problem.batch_loss(theta_req, x, y)
        (grad,) = torch.autograd.grad(batch_loss, theta_req)
        batch_loss = batch_loss.item()
        if not math.isfinite(batch_loss) or batch_loss > diverge_factor * loss0:
            stop = "diverged"
            break
        loss_sum, loss_count = loss_sum + batch_loss, loss_count + 1
        if step > 0:
            rec.step(problem, theta, lr * step)
        if steps[i_log] == step:
            avg = loss_sum / loss_count
            rec.record(
                problem, theta, lr * step, step=step, batch_loss=batch_loss, batch_loss_avg=avg
            )
            i_log, loss_sum, loss_count = i_log + 1, 0.0, 0
            if loss_floor is not None and _loss_value(problem, theta) < loss_floor:
                stop = "converged"
                break
            if time.perf_counter() > deadline:
                stop = "max_wall_time"
                break
        if step == n_steps:
            stop = "completed"
            break
        theta = torch.add(theta, problem.direction(grad), alpha=lr)
    meta = {
        "integrator": "sgd",
        "lr": lr,
        "batch_size": batch_size,
        "label_noise": label_noise,
        "online": data is None,
        "n_steps_max": n_steps,
        "n_steps": step,
        "t_end": lr * step,
        "loss_end": _loss_value(problem, theta),
        "batch_loss_end": batch_loss,
        "stop_reason": stop,
        "wall_time": time.perf_counter() - wall0,
        "problem": problem.describe(),
    }
    return _trajectory(rec, theta, meta)


def _merge_steps(log_steps, n_steps: int, n_log: int) -> list[int]:
    if log_steps is None:
        log_steps = np.geomspace(1, max(n_steps, 1), n_log)
    s = np.rint(np.asarray(log_steps, dtype=np.float64)).astype(np.int64)
    s = np.unique(np.concatenate([[0], s[(s > 0) & (s < n_steps)], [n_steps]]))
    return s.tolist()


# ---- helpers -----------------------------------------------------------------------------------


@torch.no_grad()
def _loss_value(problem: Problem, theta: torch.Tensor) -> float:
    return problem.loss(theta).item()


def _as_flat(spec: TTNSpec, theta: torch.Tensor | Cores) -> torch.Tensor:
    if isinstance(theta, list | tuple):
        theta = spec.flatten(theta)
    return theta.detach().to(torch.float64).clone()


def _trajectory(rec: Recorder, theta: torch.Tensor, meta: dict) -> Trajectory:
    data = rec.arrays()
    data["theta_end"] = theta.detach().numpy().copy()
    return Trajectory(data, meta)
