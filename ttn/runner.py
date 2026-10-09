"""Build, run, summarize and sweep training runs described by a `ttn.config.RunConfig`.

    traj = run(cfg)                                  # one run -> ttn.dynamics.Trajectory
    row = summary(traj)                              # scalar summary (dict)
    configs = expand_grid(cfg, {"init.sigma": [0.1, 0.05]}, seeds=range(5))
    table = run_sweep(configs, "data/00_smoke/grid_sigma", workers=8)

Command line:

    uv run python -m ttn.runner CONFIG.yaml [--grid GRID.yaml] [--seeds 0-9] [--out DIR]
        [--workers N] [--recompute] [--dry-run] [key=value ...]

GRID.yaml maps dotted keys to value lists ({init.sigma: [0.1, 0.05]}) and may hold a `seeds` list
(--seeds overrides it); key=value overrides apply to the base config. --out defaults to
data/<config name>.

Output layout: <out>/<run name>.npz (the Trajectory: recorded arrays and theta_end) and
<out>/<run name>.json ({"summary": ..., "meta": ...}; meta holds the config, git commit, hash of
the library code, seeds and timing). The .json is written last and atomically, so its presence
marks a finished run: sweeps skip those (resumable) if their stored config is the requested one.
<out>/summary.csv holds the summaries of the sweep's finished runs, one row per run, with the
flattened config (dotted keys) as further columns. A failed run leaves <out>/<run name>.error.txt
with its traceback.

Every random draw of a run comes from streams derived from its seed (`ttn.config.stream_seed`),
so a run is reproducible from its config and the code version alone.

Threads: all tensors are tiny, and multithreaded LAPACK calls are 100-1000x slower on a busy
machine (a 688 x 688 LU: 0.8 s with 24 threads, 6 ms with one). Training runs with one
numpy/scipy (OpenBLAS) thread (threadpoolctl, also inside `integrate_gf` for lsoda / radau);
call torch.set_num_threads(1) in your own process (sweep workers do).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import opt_einsum
import pandas as pd
import scipy
import torch
from threadpoolctl import threadpool_info, threadpool_limits
from tqdm import tqdm

from ttn.config import SEED_STREAMS, RunConfig, flatten_dict, load_yaml, stream_seed
from ttn.diagnostics import (
    classify_regime,
    crossing_times,
    find_plateaus,
    gauge_directions,
    hessian,
    kernel_alignment,
    kernel_change,
    merge_events,
    plateau_midpoints,
    project_out_gauge,
    projected_fit_loss,
    ratio_lower_bounds,
    rise_times,
    saddle_type,
    sample_spacing,
    separation_ratios,
    spectral_events,
    tangent_kernel,
    upward_crossings,
)
from ttn.dynamics import (
    Problem,
    Recorder,
    Trajectory,
    default_log_times,
    integrate_gf,
    make_dataset,
    merged_rows,
    run_gd,
    run_sgd,
)
from ttn.losses import power_law_cov, sample_inputs
from ttn.network import TTNSpec, aligned_init, antisymmetric_double, bond_spectra, to_dense
from ttn.teachers import (
    CPTeacher,
    ShiftedTeacher,
    TTNTeacher,
    odeco_teacher,
    pattern_teacher,
    random_cp_teacher,
    random_ttn_teacher,
)
from ttn.tree import make_tree

# ---- building a run ------------------------------------------------------------------------------


def make_spec(cfg: RunConfig) -> TTNSpec:
    """The student's TTN: uniform d; chi on every edge, or per edge in `tree.edges` order."""
    tree = make_tree(cfg.tree.kind, cfg.tree.n_sites)
    if isinstance(cfg.chi, int):
        return TTNSpec.uniform(tree, cfg.d, cfg.chi, cap=cfg.cap_bonds)
    if len(cfg.chi) != len(tree.edges):
        raise ValueError(f"chi needs one entry per edge ({len(tree.edges)}), got {cfg.chi}")
    bonds = [1] * tree.n_nodes
    for v, chi in zip(tree.edges, cfg.chi):
        below = len(tree.subtree_sites[v])
        max_rank = min(cfg.d**below, cfg.d ** (tree.n_sites - below))  # as in TTNSpec.uniform
        bonds[v] = min(chi, max_rank) if cfg.cap_bonds else chi
    return TTNSpec(tree, (cfg.d,) * tree.n_sites, tuple(bonds))


def make_teacher(cfg: RunConfig, seed: int) -> CPTeacher | TTNTeacher:
    t, n, d = cfg.teacher, cfg.tree.n_sites, cfg.d
    if t.kind == "odeco":
        return odeco_teacher(n, d, t.weights, seed=seed, basis=t.basis)
    if t.kind == "pattern":
        return pattern_teacher(t.weights, t.patterns, d, seed=seed, basis=t.basis)
    if t.kind == "random_cp":
        return random_cp_teacher(n, d, t.weights, seed=seed)
    tree = make_tree(t.ttn_tree_kind or cfg.tree.kind, n)
    return random_ttn_teacher(TTNSpec.uniform(tree, d, t.ttn_chi), seed=seed, norm=t.ttn_norm)


def make_site_metrics(cfg: RunConfig) -> list[torch.Tensor] | None:
    """Input covariances Sigma_i (None = whitened)."""
    inputs, n = cfg.inputs, cfg.tree.n_sites
    if inputs.cov == "identity":
        return None
    if inputs.cov == "power_law":
        return [power_law_cov(cfg.d, inputs.exponent)] * n
    m = torch.as_tensor(inputs.matrices, dtype=torch.float64)
    return [m] * n if m.ndim == 2 else list(m)


@dataclass
class Setup:
    """Everything `run` builds from a config before training."""

    problem: Problem
    theta0: torch.Tensor
    spec: TTNSpec  # the student's TTN (before the two-copy doubling)
    teacher: CPTeacher | TTNTeacher  # W*, the target (base of the centred model's teacher)
    seeds: dict[str, int]  # run seed and the derived stream seeds (teacher: the one used)
    generators: dict[str, torch.Generator]  # init, data, probe (data continues after the dataset)


def build(cfg: RunConfig) -> Setup:
    """Tree, spec, teacher, input covariances, init and `Problem` of a run.

    Init schemes (`InitConfig`): fanin; aligned (CP teachers; amplitudes aligned_a0 broadcast to
    (K, R), default sigma * core_scales[v]); antisymmetric (two-copy model with gradient mask,
    W_0 = 0); centered (target W* + W_0). With loss.kind = empirical the problem gets n_train
    samples from the data stream (labels with label_noise).
    """
    seeds = {"run": cfg.seed} | {s: stream_seed(cfg.seed, s) for s in SEED_STREAMS}
    if cfg.teacher.seed is not None:
        seeds["teacher"] = cfg.teacher.seed
    gens = {s: torch.Generator().manual_seed(seeds[s]) for s in ("init", "data", "probe")}
    spec = make_spec(cfg)
    teacher = make_teacher(cfg, seeds["teacher"])
    metrics = make_site_metrics(cfg)
    init, lr_mult = cfg.init, cfg.lr_multipliers
    if init.scheme == "aligned":
        if not isinstance(teacher, CPTeacher):
            raise ValueError("aligned init needs a CP teacher (odeco, pattern or random_cp)")
        cores = aligned_init(spec, teacher.factors, _aligned_amplitudes(cfg, spec.n_cores, teacher))
        problem, theta0 = Problem(spec, teacher, metrics, lr_mult), spec.flatten(cores)
    else:
        cores = spec.init(init.sigma, generator=gens["init"], core_scales=init.core_scales)
        if init.scheme == "fanin":
            problem, theta0 = Problem(spec, teacher, metrics, lr_mult), spec.flatten(cores)
        elif init.scheme == "antisymmetric":
            spec2, cores2, mask = antisymmetric_double(spec, cores)
            problem = Problem(spec2, teacher, metrics, lr_mult, mask=mask)
            theta0 = spec2.flatten(cores2)
        else:
            shifted = ShiftedTeacher(teacher, spec, cores)
            problem, theta0 = Problem(spec, shifted, metrics, lr_mult), spec.flatten(cores)
    if cfg.loss.kind == "empirical":
        data = make_dataset(problem, cfg.loss.n_train, gens["data"], cfg.loss.label_noise)
        problem = dataclasses.replace(problem, dataset=data)
    return Setup(problem, theta0, spec, teacher, seeds, gens)


def _aligned_amplitudes(cfg: RunConfig, n_cores: int, teacher: CPTeacher) -> torch.Tensor:
    shape = (n_cores, teacher.n_components)
    if cfg.init.aligned_a0 is None:
        scales = torch.tensor(cfg.init.core_scales or [1.0] * n_cores, dtype=torch.float64)
        return (cfg.init.sigma * scales)[:, None].expand(shape).clone()
    return torch.broadcast_to(torch.as_tensor(cfg.init.aligned_a0, dtype=torch.float64), shape)


def target_spectra(
    spec: TTNSpec, teacher: CPTeacher | TTNTeacher, site_metrics=None
) -> list[list[float]] | None:
    """Schmidt values of the target W* across every edge of the student's tree (`tree.edges`
    order; function-space values with site metrics), the reference of spectral learning events.

    CP teachers exactly (written as a TTN of bond dimension R with `aligned_init`), TTN teachers on
    the student's tree by `bond_spectra`; None for a TTN teacher on another tree.
    """
    tree = spec.tree
    if isinstance(teacher, CPTeacher):
        n_comp = teacher.n_components
        bonds = tuple(1 if v == tree.root else n_comp for v in range(tree.n_nodes))
        tspec = TTNSpec(tree, spec.phys_dims, bonds)
        amps = torch.ones(tree.n_nodes, n_comp, dtype=torch.float64)
        amps[tree.root] = teacher.weights
        cores = aligned_init(tspec, teacher.factors, amps)
    elif teacher.spec.tree == tree:
        tspec, cores = teacher.spec, teacher.cores
    else:
        return None
    with torch.no_grad():
        values = bond_spectra(tspec, cores, site_metrics)
    return [values[v].tolist() for v in tree.edges]


def lazy_floor(problem: Problem, theta0: torch.Tensor, max_size: int = 10**7) -> float | None:
    """Loss at the end of a lazy run from theta0: `projected_fit_loss` of the target minus the
    fitted tensor (`Problem.effective`) over the trained coordinates, with the site metrics.

    0 when every bond is full, > 0 when a bond is deficient (theory_regimes R7). None for an
    empirical loss, or when the dense Jacobian (prod d_i x n_params) has more than max_size entries.
    """
    spec = problem.spec
    if problem.dataset is not None or math.prod(spec.phys_dims) * spec.n_params > max_size:
        return None
    with torch.no_grad():
        residual = problem.target.dense() - to_dense(*problem.effective(theta0))
    return projected_fit_loss(spec, theta0, residual, problem.precond, problem.site_metrics)


# ---- recording -----------------------------------------------------------------------------------


def gauge_projected_eigs(problem: Problem, theta: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending) of the Hessian of the training loss over the trained coordinates
    (precond != 0: gradient mask and nonzero lr multipliers), and of its restriction to the
    orthogonal complement of the gauge directions (`project_out_gauge`), NaN-padded to the same
    length.

    Only gauge directions that leave the untrained coordinates fixed are directions of the trained
    flow, so the others are dropped: for the two-copy model this keeps exactly the per-copy gauge
    orbits, for a frozen core the gauges of its bonds.
    """
    h = hessian(problem.loss, theta)
    if problem.precond is None:
        trained = torch.ones(h.shape[0], dtype=torch.bool)
    else:
        trained = problem.precond != 0
    h = h[trained][:, trained]
    u = gauge_directions(problem.spec, theta)
    if not trained.all():
        u = u[:, u[~trained].abs().amax(dim=0) <= 1e-12 * u.abs().amax()]
    if u.shape[1]:
        _, vals = project_out_gauge(h, u[trained])
    else:
        vals = torch.linalg.eigvalsh(h)
    gauge = np.full(h.shape[0], np.nan)
    gauge[: len(vals)] = vals.numpy()
    return torch.linalg.eigvalsh(h).numpy(), gauge


class RunRecorder(Recorder):
    """`ttn.dynamics.Recorder` plus the runner's opt-in diagnostics.

    probe_x (n_probe, N, d) given: at every log time
      ntk_change     delta_K(t) = |Theta_t - Theta_0|_F / |Theta_0|_F
      ntk_alignment  y^T Theta_t y / (|Theta_t|_F |y|^2), y = target values on the probe inputs
    with Theta_t = J diag(eta * m) J^T the tangent kernel on the probe inputs
    (`ttn.diagnostics.tangent_kernel`; eta * m = lr multipliers times gradient mask), as defined in
    notes/diagnostics.md. Cost: one batched backward pass per log time.
    hessian_times given: at each log time equal to one of them (relative 1e-12), the Hessian H of
    the training loss over the n trained coordinates is stored (`gauge_projected_eigs`) as
      hessian_t (m,)              the times
      hessian_eigs (m, n)         eigenvalues of H, ascending
      hessian_eigs_gauge (m, n)   eigenvalues of H restricted to the orthogonal complement of the
                                  gauge directions of the trained flow (per copy for the two-copy
                                  model), ascending, NaN-padded
    indexed by Hessian time, not by log time. Cost: one vectorized double backward per Hessian.
    """

    def __init__(self, probe_x=None, hessian_times=(), **kwargs):
        super().__init__(**kwargs)
        self.probe_x = probe_x
        self.hessian_times = [float(t) for t in hessian_times]

    def start(self, problem: Problem, theta0: torch.Tensor) -> None:
        super().start(problem, theta0)
        self.hessian_rows: dict[str, list] = {}
        if self.probe_x is not None:
            self.kernel0 = self._kernel(problem, theta0)
            with torch.no_grad():
                self.probe_y = problem.target(self.probe_x)

    def _kernel(self, problem: Problem, theta: torch.Tensor) -> torch.Tensor:
        return tangent_kernel(problem.spec, theta, self.probe_x, mask=problem.precond)

    def record(self, problem, theta, t, loss=None, velocity=None, **info) -> None:
        if self.probe_x is not None:
            kernel = self._kernel(problem, theta)
            info["ntk_change"] = kernel_change(kernel, self.kernel0)
            info["ntk_alignment"] = kernel_alignment(kernel, self.probe_y)
        super().record(problem, theta, t, loss=loss, velocity=velocity, **info)
        if any(math.isclose(t, th, rel_tol=1e-12, abs_tol=0.0) for th in self.hessian_times):
            eigs, gauge = gauge_projected_eigs(problem, theta.detach())
            rows = self.hessian_rows
            rows.setdefault("hessian_t", []).append(float(t))
            rows.setdefault("hessian_eigs", []).append(eigs)
            rows.setdefault("hessian_eigs_gauge", []).append(gauge)

    def arrays(self) -> dict[str, np.ndarray]:
        out = super().arrays()
        out.update({key: np.array(values) for key, values in self.hessian_rows.items()})
        return out


# ---- one run -------------------------------------------------------------------------------------


def run(cfg: RunConfig, extra=()) -> Trajectory:
    """Build and train one run; the Trajectory's meta gets the config, git commit and code hash,
    seeds, timing, thread counts and versions, the target's spectra (`target_spectra`) and, with a
    tangent-kernel probe, the lazy floor (`lazy_floor`).

    optimizer.kind: gf -> `integrate_gf` (method, t_max, rtol, atol, max_steps), gd -> `run_gd`,
    sgd -> `run_sgd` (online fresh samples for the population loss, with label_noise as fresh
    label noise; random reshuffling of the training set for the empirical loss). Logging: n_log
    log-spaced times (steps for gd / sgd; gf: `default_log_times`) or logging.log_times, plus the
    hessian_at times, plus fine rows (`ttn.dynamics.Recorder`); see `RunRecorder` for the
    tangent-kernel and Hessian diagnostics. extra: further diagnostics
    [(name, fn(problem, theta, t) -> array)] at every log time (picklable for `run_sweep` workers).
    Training uses one OpenBLAS thread; call torch.set_num_threads(1) in your own process.
    """
    wall0, cpu0 = time.perf_counter(), time.process_time()
    started = datetime.now(UTC).isoformat(timespec="seconds")
    setup = build(cfg)
    problem, theta0, opt, log = setup.problem, setup.theta0, cfg.optimizer, cfg.logging
    probe_x = None
    if log.ntk_probe > 0:
        n, d = cfg.tree.n_sites, cfg.d
        probe_x = sample_inputs(
            log.ntk_probe, n, d, setup.generators["probe"], problem.site_metrics
        )
    stepped = opt.kind in ("gd", "sgd")
    hessian_times = list(log.hessian_at)
    if stepped:
        hessian_steps = [round(t / opt.lr) for t in log.hessian_at]
        hessian_times = [opt.lr * s for s in hessian_steps]
    rec = RunRecorder(
        probe_x=probe_x,
        hessian_times=hessian_times,
        spectra=log.spectra,
        save_params=log.save_params,
        extra=extra,
        fine=log.fine,
        fine_tol=log.fine_tol,
    )
    t_build = time.perf_counter() - wall0
    common = {"loss_floor": opt.loss_floor, "recorder": rec, "max_wall_time": opt.max_wall_time}
    with threadpool_limits(limits=1, user_api="blas"):
        blas_threads = [(lib["internal_api"], lib["num_threads"]) for lib in threadpool_info()]
        if opt.kind == "gf":
            log_times = log.log_times
            if log_times is None:
                log_times = default_log_times(problem, theta0, opt.t_max, log.n_log)
            traj = integrate_gf(
                problem,
                theta0,
                opt.t_max,
                log_times=np.union1d(log_times, hessian_times),
                rtol=opt.rtol,
                atol=opt.atol,
                max_steps=opt.max_steps,
                method=opt.method,
                **common,
            )
        else:
            if log.log_times is None:
                steps = np.rint(np.geomspace(1, max(opt.n_steps, 1), log.n_log))  # run_gd's default
            else:
                steps = np.rint(np.asarray(log.log_times, dtype=float) / opt.lr)
            log_steps = np.union1d(steps, hessian_steps)
            if opt.kind == "gd":
                traj = run_gd(problem, theta0, opt.lr, opt.n_steps, log_steps=log_steps, **common)
            else:
                noise = cfg.loss.label_noise if problem.dataset is None else 0.0
                traj = run_sgd(
                    problem,
                    theta0,
                    opt.lr,
                    opt.n_steps,
                    opt.batch_size,
                    setup.generators["data"],
                    label_noise=noise,
                    log_steps=log_steps,
                    **common,
                )
    t_train = time.perf_counter() - wall0
    recorded = traj.data.get("hessian_t", [])
    missing = [
        t for t in hessian_times if not any(math.isclose(t, r, rel_tol=1e-12) for r in recorded)
    ]
    if missing:
        warnings.warn(
            f"{cfg.name}: no Hessian at t = {missing} (the run stopped at t = "
            f"{traj.meta['t_end']:.4g}, {traj.meta['stop_reason']})",
            stacklevel=2,
        )
    traj.meta.update(
        name=cfg.name,
        config=cfg.to_dict(),
        seeds=setup.seeds,
        git=git_info(),
        started=started,
        hessian_missing=missing,
        teacher_spectra=target_spectra(setup.spec, setup.teacher, problem.site_metrics),
        lazy_floor=lazy_floor(problem, theta0) if probe_x is not None else None,
        host=platform.node(),
        threads={"torch": torch.get_num_threads(), "blas": blas_threads}
        | {k: os.environ.get(k) for k in _ONE_THREAD},
        versions={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opt_einsum": opt_einsum.__version__,
            "platform": platform.platform(),
        },
    )
    traj.meta["timing"] = {
        "build": t_build,
        "train": traj.meta["wall_time"],
        "total": time.perf_counter() - wall0,
        "after_train": time.perf_counter() - wall0 - t_train,
        "cpu": time.process_time() - cpu0,
    }
    return traj


@functools.cache
def git_info() -> dict:
    """Commit of the repository holding `ttn`, whether `ttn/` has uncommitted or untracked changes
    (results then depend on code that is not in the commit; None outside a git checkout), and
    ttn_sha256 (`code_hash`), which identifies the library code in either case."""
    repo = Path(__file__).resolve().parents[1]
    out = {"commit": None, "ttn_dirty": None, "ttn_sha256": code_hash()}
    if shutil.which("git") is None:
        return out
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    if head.returncode != 0:
        return out
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "ttn"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return out | {"commit": head.stdout.strip(), "ttn_dirty": bool(status.stdout.strip())}


def code_hash() -> str:
    """sha256 of the library code: the names and contents of ttn/*.py in sorted order, line endings
    normalized to LF (the hash of a commit's ttn/*.py files is computed the same way)."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


# ---- events and summary --------------------------------------------------------------------------

# A dormant component whose escape signal is below this (relative to |W(t)|) is seeded by
# roundoff, not by the init (CLAUDE.md).
FLOAT64_FLOOR = 1e-15
RESOLVED_FACTOR = 3.0  # an event is resolved if its width is >= this many sample intervals


def learning_events(traj: Trajectory) -> dict | None:
    """The learning events of a run (notes/diagnostics.md), as `summary` uses them, on the log rows
    and fine rows together (`merged_rows`). None without overlaps and spectra.

    Overlap events (CP teacher whose rank-one terms are orthogonal under the input metrics,
    meta['problem']['orthogonal_components']): component r's event is the last upward crossing of
    a_r = 1/2 (the first unless a transient crossed and fell back) and its width the final rise
    0.1 -> 0.9. Otherwise one escape can raise several overlaps at once and later escapes may
    raise none, so spectral events are used (`spectral_events` with the target's spectra
    meta['teacher_spectra'] as reference when known, merged when closer than their widths).

    Returns a dict: kind ('overlap' | 'spectral'); per event (components in order for overlap
    events; merged events sorted by time for spectral ones): time, width, start and end of the rise,
    label (component 'a<r>', or the merged '+'-joined 'e<edge>k<index>'), spacing (largest sample
    interval across the rise, `sample_spacing`), resolved (width >= 3 spacing; otherwise the width
    is an overestimate and the separation ratios with it are lower bounds), min_signal (smallest
    escape signal before the event relative to |W(t)|: |a_r| |<W*, M U_r>|, or s_k^e; below 1e-15
    it is roundoff).
    """
    meta, rows = traj.meta, merged_rows(traj.data)
    t, problem = rows["t"], meta["problem"]
    # 1 / |W|: inf where W = 0 (two-copy and centred models at t = 0), ignored by _min_before
    with np.errstate(divide="ignore"):
        inv_norm = 1.0 / _tensor_norm(rows, meta)
    # Trajectories recorded before these keys existed: orthogonal, unit targets.
    if "overlaps" in rows and problem.get("orthogonal_components", True):
        ov = rows["overlaps"]
        times = crossing_times(t, ov, 0.5, last=True)
        start, end = rise_times(t, ov)
        targets = np.abs(problem.get("component_targets", np.ones(ov.shape[1])))
        with np.errstate(invalid="ignore"):
            signal = np.abs(ov) * targets * inv_norm[:, None]
        events = {
            "kind": "overlap",
            "time": times,
            "width": end - start,
            "start": start,
            "end": end,
            "label": [f"a{r}" for r in range(ov.shape[1])],
            "min_signal": _min_before(t, signal, times),
        }
    elif "spectra" in rows:
        ref = meta.get("teacher_spectra")
        spec = spectral_events(t, rows["spectra"], ref, rank_rtol=1e-8 if ref else 1e-3)
        times, widths, groups = merge_events(spec["time"], spec["width"], return_groups=True)
        with np.errstate(invalid="ignore"):
            values = rows["spectra"][:, spec["edge"], spec["k"]] * inv_norm[:, None]
        signal = _min_before(t, values, spec["time"])
        events = {
            "kind": "spectral",
            "time": times,
            "width": widths,
            "start": np.array([np.fmin.reduce(spec["t_lo"][g]) for g in groups]),
            "end": np.array([np.fmax.reduce(spec["t_hi"][g]) for g in groups]),
            "label": ["+".join(f"e{spec['edge'][i]}k{spec['k'][i]}" for i in g) for g in groups],
            "min_signal": np.array([np.fmin.reduce(signal[g]) for g in groups]),
        }
    else:
        return None
    events["spacing"] = sample_spacing(t, events["start"], events["end"])
    events["resolved"] = events["width"] >= RESOLVED_FACTOR * events["spacing"]
    return events


def _tensor_norm(rows: dict, meta: dict) -> np.ndarray:
    """|W(t)| (Frobenius, function-space with metrics) of the fitted tensor at every row, from the
    spectra of the first edge; the target's norm if no spectra were recorded."""
    if "spectra" in rows:
        return np.sqrt(np.nansum(rows["spectra"][:, 0, :] ** 2, axis=-1))
    return np.full(len(rows["t"]), math.sqrt(meta["problem"]["target_norm2"]))


def _min_before(t, values, times) -> np.ndarray:
    """Per column, the smallest finite value at t <= times[j] (all t if times[j] is NaN)."""
    out = np.full(values.shape[1], np.nan)
    for j in range(values.shape[1]):
        before = values[t <= times[j], j] if np.isfinite(times[j]) else values[:, j]
        before = before[np.isfinite(before)]
        if before.size:
            out[j] = before.min()
    return out


def summary(traj: Trajectory) -> dict:
    """Scalars of one run (JSON-friendly; NaN where undefined), computed on the log rows and fine
    rows together (`merged_rows`).

    name, seed, stop_reason, t_end, n_steps, n_evals (gradient evaluations), n_jacobians
    (exact Hessians, by lsoda / radau; cost in experiments/00_smoke/README.md), loss_0, loss_end
    (training loss at the end; for a diverged GD/SGD run the diverging value), pop_loss_end,
    max_rel_change = max_t |theta - theta0| / |theta0|, conservation_drift (max over log times),
    wall_time and cpu_time (whole run), n_fine_rows, n_plateaus / n_drops (`find_plateaus` on the
    training loss).
    CP teachers, per component r: t_cross_r (last upward crossing of a_r = 1/2, the first unless
    a transient crossed and fell back), width_r (final rise 0.1 -> 0.9), n_cross_r (upward
    crossings of 1/2: > 1 flags a transient), transient_r (peak of a_r before the last decrease
    that precedes its final rise: the height of a transient excursion, ~a_r(0) without one);
    t_event_k (k = 1..R: the crossing times sorted, NaN-padded), order (components by crossing
    time, e.g. "0-1-2"), n_learned (a_r >= 0.5 at the end).
    Events (`learning_events`: overlap events, or spectral ones for non-orthogonal CP and TTN
    teachers): event_kind, n_events, n_unresolved (events narrower than 3 sample intervals),
    rho_min (smallest separation ratio, NaN with < 2 events), min_signal and float64_floor (some
    dormant escape signal fell below 1e-15 |W|: the run is seeded by roundoff).
    With a tangent-kernel probe: delta_K_end, lazy_floor (`lazy_floor`), loss_end_over_floor,
    delta_K_at_floor (delta_K when the loss first came within 10% of a positive floor), and with
    events the regime (`classify_regime` with the lower-bound flags and the lazy floor). The regime
    is 'diverged' whenever the run stopped as diverged / nonfinite.
    """
    meta, data = traj.meta, traj.data
    rows = merged_rows(data)
    t, loss = rows["t"], rows["loss"]
    timing = meta.get("timing", {})
    out = {
        "name": meta.get("name"),
        "seed": meta.get("config", {}).get("seed"),
        "stop_reason": meta["stop_reason"],
        "t_end": meta["t_end"],
        "n_steps": meta["n_steps"],
        "n_evals": meta.get("n_evals", meta.get("nfev")),
        "n_jacobians": meta.get("njev", 0),
        "loss_0": data["loss"][0],
        "loss_end": meta.get("loss_end", data["loss"][-1]),
        "pop_loss_end": data["pop_loss"][-1],
        "max_rel_change": np.max(data["rel_dist_from_init"]),
        "conservation_drift": np.max(data.get("conservation_drift", np.nan)),
        "wall_time": timing.get("total", meta["wall_time"]),
        "cpu_time": timing.get("cpu", np.nan),
        "n_fine_rows": len(data.get("fine_t", [])),
    }
    plateaus, drops = find_plateaus(t, loss)
    out["n_plateaus"], out["n_drops"] = len(plateaus), len(drops)
    if "overlaps" in rows:
        ov = rows["overlaps"]
        times = crossing_times(t, ov, 0.5, last=True)
        start, end = rise_times(t, ov)
        n_cross = upward_crossings(ov, 0.5)
        for r in range(ov.shape[1]):
            out[f"t_cross_{r}"], out[f"width_{r}"] = times[r], end[r] - start[r]
            out[f"n_cross_{r}"] = int(n_cross[r])
            # The final rise starts after the last decrease of a_r before t(lo); anything above
            # that point earlier was a transient excursion.
            i_lo = np.searchsorted(t, start[r]) if np.isfinite(start[r]) else len(t)
            a = ov[:i_lo, r]
            falls = np.flatnonzero(a[:-1] > a[1:])
            trough = falls[-1] + 1 if falls.size else 0
            out[f"transient_{r}"] = np.max(a[: trough + 1]) if a.size else np.nan
        for k, t_event in enumerate(np.sort(times), start=1):  # NaN (never learned) sorts last
            out[f"t_event_{k}"] = t_event
        learned = np.flatnonzero(np.isfinite(times))
        out["order"] = "-".join(str(r) for r in learned[np.argsort(times[learned], kind="stable")])
        out["n_learned"] = int(np.sum(ov[-1] >= 0.5))
    events = learning_events(traj)
    rho, lower = np.array([]), None
    if events is not None:
        finite = np.isfinite(events["time"])
        rho = separation_ratios(events["time"], events["width"])
        lower = ratio_lower_bounds(events["time"], ~events["resolved"])
        out["event_kind"] = events["kind"]
        out["n_events"] = int(finite.sum())
        out["n_unresolved"] = int(np.sum(finite & ~events["resolved"]))
        out["rho_min"] = np.min(rho) if rho.size else np.nan
        out["min_signal"] = np.fmin.reduce(events["min_signal"]) if finite.any() else np.nan
        out["float64_floor"] = bool(out["min_signal"] < FLOAT64_FLOOR)
    floor = meta.get("lazy_floor")
    if "ntk_change" in data:
        out["delta_K_end"] = data["ntk_change"][-1]
        if floor is not None:
            out["lazy_floor"] = floor
            out["loss_end_over_floor"] = out["loss_end"] / floor if floor > 0 else np.nan
            near = np.flatnonzero(data["loss"] <= 1.1 * floor) if floor > 0 else []
            out["delta_K_at_floor"] = data["ntk_change"][near[0]] if len(near) else np.nan
    if meta["stop_reason"] in ("diverged", "nonfinite"):
        out["regime"] = "diverged"
    elif "ntk_change" in data and events is not None:
        out["regime"] = classify_regime(
            out["delta_K_end"],
            rho,
            out["loss_end"],
            out["loss_0"],
            lower_bounds=lower,
            lazy_floor=floor,
        )
    return {k: _scalar(v) for k, v in out.items()}


def _scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def plateau_hessians(cfg: RunConfig, traj: Trajectory) -> list[dict]:
    """Gauge-projected Hessian spectra at the plateau midpoints of a run recorded with
    logging.save_params = true (notes/diagnostics.md).

    Between consecutive learning events (`learning_events`) the plateau midpoint is the log time of
    minimum speed (`plateau_midpoints`); the problem is rebuilt from cfg (same seeds, so the same
    problem) and the Hessian taken at the saved parameters there (`gauge_projected_eigs`). Returns
    one dict per plateau: t, eigs_gauge and the entries of `saddle_type`. Strict vs higher order
    is decided across a sigma sweep with `saddle_scaling`, not by one spectrum.
    """
    events = learning_events(traj)
    t, params = traj["t"], traj["params"]
    problem = build(cfg).problem
    out = []
    for t_mid in plateau_midpoints(t, traj["speed"], events["time"]):
        if np.isfinite(t_mid):
            theta = torch.from_numpy(params[np.flatnonzero(t == t_mid)[0]])
            _, eigs = gauge_projected_eigs(problem, theta)
            out.append({"t": float(t_mid), "eigs_gauge": eigs} | saddle_type(eigs))
    return out


# ---- saving and loading --------------------------------------------------------------------------


def save_run(traj: Trajectory, out_dir: str | Path) -> tuple[Path, Path]:
    """Write <out>/<name>.npz, then <out>/<name>.json ({"summary", "meta"}), each atomically
    (through temporary files named with the process id)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name, pid = traj.meta["name"], os.getpid()
    npz, record = out / f"{name}.npz", out / f"{name}.json"
    tmp = traj.save(out / f"{name}.{pid}.tmp.npz")
    os.replace(tmp, npz)
    text = json.dumps({"summary": summary(traj), "meta": traj.meta}, indent=1, default=_json)
    tmp_json = out / f"{name}.{pid}.tmp.json"
    tmp_json.write_text(text, encoding="utf-8")
    os.replace(tmp_json, record)
    return npz, record


def _json(obj):
    if isinstance(obj, torch.Tensor | np.ndarray | np.generic):
        return obj.tolist()
    return str(obj)


def load_run(out_dir: str | Path, name: str) -> Trajectory:
    return Trajectory.load(Path(out_dir) / f"{name}.npz")


def load_summaries(out_dir: str | Path, names=None, recompute: bool = False) -> pd.DataFrame:
    """One row per finished run in out_dir (or per finished run among `names`): its summary plus
    its flattened config (dotted keys). Summaries are those stored when the run was saved;
    recompute=True recomputes them from the .npz files with the current `summary`."""
    out = Path(out_dir)
    if names is None:
        paths = [p for p in sorted(out.glob("*.json")) if not p.name.endswith(".tmp.json")]
    else:
        paths = [out / f"{name}.json" for name in names if (out / f"{name}.json").exists()]
    rows = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        name = path.name.removesuffix(".json")
        row = summary(load_run(out, name)) if recompute else record["summary"]
        rows.append(row | flatten_dict(record["meta"]["config"]))
    return pd.DataFrame(rows)


# ---- grids and sweeps ----------------------------------------------------------------------------


def expand_grid(base: RunConfig, grid: dict | None = None, seeds=None) -> list[RunConfig]:
    """One config per point of the grid (outer product of the value lists) and seed.

    grid: {dotted key: [values]} (every value must be a list). seeds: run seeds (None keeps
    base.seed). Run names are "<base name>__<key>=<value>__...__seed=<s>", in grid-key order, with
    each key shortened to its last component unless that is ambiguous ("sigma" for init.sigma;
    "teacher.seed" next to the run seed) and values written as stored in the config (floats by
    repr, lists joined by "-").
    """
    grid = dict(grid or {})
    for key, values in grid.items():
        if not isinstance(values, list | tuple):
            raise TypeError(f"grid values must be lists, got {key}: {values!r}")
    keys = list(grid) + ([] if seeds is None else ["seed"])
    short = [k.rsplit(".", 1)[-1] for k in keys]
    labels = {k: s if short.count(s) == 1 else k for k, s in zip(keys, short)}
    configs = []
    for values in itertools.product(*grid.values()):
        for seed in [None] if seeds is None else list(seeds):
            overrides = dict(zip(grid, values))
            if seed is not None:
                overrides["seed"] = int(seed)
            cfg = base.with_overrides(overrides)
            parts = [f"{labels[k]}={_format_value(cfg.get(k))}" for k in overrides]
            cfg.name = "__".join([base.name, *parts])
            configs.append(cfg)
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError("expand_grid: run names are not unique (repeated grid values?)")
    return configs


def _format_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "-".join(
            f"({_format_value(v)})" if isinstance(v, list) else _format_value(v) for v in value
        )
    return "none" if value is None else str(value)


def parse_seeds(text: str) -> list[int]:
    """ "0-9" -> 0..9 (inclusive), "0,3,5" -> [0, 3, 5], "0-4,10" -> [0, 1, 2, 3, 4, 10]."""
    seeds = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def run_sweep(
    configs: list[RunConfig],
    out_dir: str | Path,
    workers: int = 1,
    extra=(),
    recompute: bool = False,
) -> pd.DataFrame:
    """Run every config whose <out_dir>/<name>.json does not exist yet; write summary.csv and
    return the summary table (`load_summaries`) of the requested runs that finished.

    Resuming: a finished run is reused only if its stored config equals the requested one; a run of
    the same name from another config raises ValueError before anything runs (use another out_dir,
    or delete the old files). Finished runs made by different library code (`code_hash`) are
    reused, with a note. recompute=True recomputes the summaries from the .npz files (after a change
    to `summary`).
    One sweep per directory at a time: <out_dir>/.sweep.lock (pid, host, start time) is created
    exclusively and removed at the end; delete it by hand if a killed sweep left it behind.
    workers >= 1: a pool of that many processes (spawn start method) with one torch, OpenMP and
    OpenBLAS thread each (`_ONE_THREAD` is put into this process's environment, unless set, for
    the workers to inherit). A failing run does not stop the others: its traceback goes to
    <name>.error.txt as soon as it fails, and after summary.csv is written a RuntimeError names the
    failed runs (rerunning the sweep retries only those).
    workers = 0: one after the other in this process (one torch and one OpenBLAS thread), stopping
    at the first error, for debugging. extra: passed to `run` (module-level functions for workers).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError("run names in a sweep must be unique")
    lock = _lock_sweep(out)
    try:
        pending, changed, other_code = [], [], 0
        for cfg in configs:
            path = out / f"{cfg.name}.json"
            if not path.exists():
                pending.append(cfg)
                continue
            meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
            if not _same_config(meta["config"], cfg):
                changed.append(cfg.name)
            other_code += meta["git"].get("ttn_sha256") != git_info()["ttn_sha256"]
        if changed:
            raise ValueError(
                f"{len(changed)} finished runs in {out} have the same name but another config "
                f"(use another out_dir, or delete them): {changed[:5]}"
            )
        print(
            f"{len(configs)} runs, {len(configs) - len(pending)} done, {len(pending)} to run -> {out}"
        )
        if other_code:
            print(f"note: {other_code} finished runs were made by other library code (ttn_sha256)")
        failed = {}
        if workers == 0:
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            with threadpool_limits(limits=1, user_api="blas"):
                for cfg in tqdm(pending, desc=out.name):
                    _run_and_save(cfg, out, extra)
            torch.set_num_threads(threads)
        elif pending:
            for key, value in _ONE_THREAD.items():
                os.environ.setdefault(key, value)
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(workers, mp_context=context, initializer=_init_worker) as pool:
                futures = {pool.submit(_run_and_save, cfg, out, extra): cfg.name for cfg in pending}
                for future in tqdm(as_completed(futures), total=len(futures), desc=out.name):
                    exc = future.exception()
                    if exc is not None:
                        name = futures[future]
                        failed[name] = exc
                        text = "".join(traceback.format_exception(exc))
                        (out / f"{name}.error.txt").write_text(text, encoding="utf-8")
                        tqdm.write(f"FAILED {name}: {exc!r}")
        table = load_summaries(out, names, recompute=recompute)
        table.to_csv(out / "summary.csv", index=False)
    finally:
        lock.unlink()
    if failed:
        raise RuntimeError(f"{len(failed)} runs failed (see *.error.txt in {out}): {list(failed)}")
    return table


def _same_config(stored: dict, cfg: RunConfig) -> bool:
    """Whether a stored config (meta['config']) is cfg; fields added since then take defaults."""
    try:
        return RunConfig.from_dict(stored) == cfg
    except (KeyError, TypeError, ValueError):  # a field removed or renamed since
        return False


def _lock_sweep(out: Path) -> Path:
    lock = out / ".sweep.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = lock.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            f"another sweep is writing to {out} ({holder}); if it is not running, delete {lock}"
        ) from None
    started = datetime.now(UTC).isoformat(timespec="seconds")
    os.write(fd, f"pid {os.getpid()} on {platform.node()} since {started}".encode())
    os.close(fd)
    return lock


# Read by OpenMP / OpenBLAS / MKL when a worker loads them (torch.set_num_threads covers only torch).
_ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _init_worker() -> None:
    torch.set_num_threads(1)


def _run_and_save(cfg: RunConfig, out: Path, extra=()) -> str:
    traj = run(cfg, extra)
    save_run(traj, out)
    (out / f"{cfg.name}.error.txt").unlink(missing_ok=True)
    return cfg.name


# ---- command line --------------------------------------------------------------------------------

_SHOW = [
    "name",
    "stop_reason",
    "t_end",
    "loss_end",
    "n_plateaus",
    "order",
    "rho_min",
    "regime",
    "wall_time",
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ttn.runner", description="Run a config, or a grid of configs, in parallel."
    )
    parser.add_argument("config", help="base config (YAML)")
    parser.add_argument("overrides", nargs="*", metavar="key=value", help="dotted-key overrides")
    parser.add_argument("--grid", help="YAML mapping dotted keys to value lists (and `seeds`)")
    parser.add_argument("--seeds", help='run seeds, e.g. "0-9" or "0,3,5"')
    parser.add_argument("--out", help="output directory (default data/<config name>)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes (default 1; 0 = in this process, for debugging)",
    )
    parser.add_argument(
        "--recompute", action="store_true", help="recompute the summaries of finished runs"
    )
    parser.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    args = parser.parse_intermixed_args(argv)
    base = RunConfig.load(args.config, args.overrides)
    grid = dict(load_yaml(args.grid) or {}) if args.grid else {}
    grid_seeds = grid.pop("seeds", None)
    seeds = parse_seeds(args.seeds) if args.seeds else grid_seeds
    configs = expand_grid(base, grid, seeds)
    out = Path(args.out) if args.out else Path("data") / base.name
    if args.dry_run:
        print("\n".join(c.name for c in configs))
        print(f"{len(configs)} runs -> {out}")
        return
    table = run_sweep(configs, out, args.workers, recompute=args.recompute)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table[[c for c in _SHOW if c in table.columns]].to_string(index=False))
    print(f"summary: {out / 'summary.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])
