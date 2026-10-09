"""Problem setup and recording shared by all optimizers of this experiment.

A run is described by a flat dict (one row of a sweep, see `run.py`):
  tree, n_sites, d, chi, teacher ('odeco' | 'pattern'), weights, patterns (pattern teacher),
  teacher_seed, sigma, seed, optional n_samples / data_seed (finite-sample variant), optimizer and
  its hyperparameters.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from ttn.dynamics import Problem, Recorder, make_dataset
from ttn.network import TTNSpec, conserved_quantities
from ttn.teachers import odeco_teacher, pattern_teacher
from ttn.tree import make_tree


def make_problem(cfg: dict) -> Problem:
    tree = make_tree(cfg["tree"], cfg["n_sites"])
    spec = TTNSpec.uniform(tree, cfg["d"], cfg["chi"])
    if cfg["teacher"] == "odeco":
        teacher = odeco_teacher(cfg["n_sites"], cfg["d"], cfg["weights"], seed=cfg["teacher_seed"])
    elif cfg["teacher"] == "pattern":
        teacher = pattern_teacher(cfg["weights"], cfg["patterns"], cfg["d"], seed=cfg["teacher_seed"])
    else:
        raise ValueError(cfg["teacher"])
    problem = Problem(spec, teacher)
    if cfg.get("n_samples"):
        gen = torch.Generator().manual_seed(int(cfg.get("data_seed", cfg["seed"])) + 7_000_000)
        problem = dataclasses.replace(problem, dataset=make_dataset(problem, int(cfg["n_samples"]), gen))
    return problem


def initial_point(problem: Problem, cfg: dict) -> torch.Tensor:
    """Fan-in init (default) or, with init: balanced, the same W_0 in the balanced gauge
    (Q_e = 0 on every edge; avoids the conservation-law trap of notes/theory_saddles.md R12)."""
    cores = problem.spec.init(cfg["sigma"], seed=cfg["seed"])
    if cfg.get("init", "fanin") == "balanced":
        from experiments.optimizers.gauge import balance

        cores = balance(problem.spec, cores)
    return problem.spec.flatten(cores)


# ---- extra diagnostics (passed to ttn.dynamics.Recorder) ------------------------------------------


def balance_index(problem: Problem, theta: torch.Tensor, t: float) -> np.ndarray:
    """Per edge: ||Q_e||_F / (||G_v^T G_v||_F + ||G_p,(v) G_p,(v)^T||_F), in [0, 1].

    0 = balanced on that edge (the min-norm gauge, which gradient flow from small init keeps up to
    O(sigma^2)); 1 = all weight on one side (e.g. canonical form: isometries below, norm above)."""
    spec = problem.spec
    t_ = spec.tree
    cores = spec.unflatten(theta)
    q = conserved_quantities(spec, cores)
    out = []
    for v in t_.edges:
        p = t_.parent[v]
        gv = cores[v].reshape(-1, cores[v].shape[-1])
        leg = len(t_.sites[p]) + t_.children[p].index(v)
        gp = torch.movedim(cores[p], leg, 0).reshape(cores[p].shape[leg], -1)
        scale = (gv.T @ gv).norm() + (gp @ gp.T).norm()
        out.append((q[v].norm() / scale).item() if scale > 0 else 0.0)
    return np.array(out)


def q_norms(problem: Problem, theta: torch.Tensor, t: float) -> np.ndarray:
    """||Q_e||_F per edge (to follow how far the conserved quantities of GF move)."""
    q = conserved_quantities(problem.spec, problem.spec.unflatten(theta))
    return np.array([q[v].norm().item() for v in problem.spec.tree.edges])


def make_recorder(speed: bool = True) -> Recorder:
    rec = Recorder(extra=[("balance", balance_index), ("q_norm", q_norms)])
    if not speed:  # the library's record() always adds |v|; the GD-type loops pass it themselves
        pass
    return rec


def teacher_spectra(problem: Problem) -> np.ndarray:
    """Bond spectra of W* on the student's tree, from the dense teacher (small N only)."""
    spec = problem.spec
    w = problem.target.dense()
    edges = spec.tree.edges
    chi_max = max(spec.bond_dims[v] for v in edges)
    out = np.full((len(edges), chi_max), np.nan)
    for i, v in enumerate(edges):
        rows = list(spec.tree.subtree_sites[v])
        cols = [s for s in range(spec.tree.n_sites) if s not in rows]
        m = w.permute(rows + cols).reshape(math.prod(w.shape[s] for s in rows), -1)
        sv = torch.linalg.svdvals(m)[:chi_max].numpy()
        out[i, : len(sv)] = sv
    return out


def record_schedule(n_max: int, n_log: int = 300, n_lin: int = 400) -> np.ndarray:
    """Step indices 0..n_max on a union of a log grid and a linear grid.

    The linear grid lets plateaus be judged in linear time (notes/diagnostics.md)."""
    log = np.unique(np.round(np.logspace(0, math.log10(max(n_max, 1)), n_log)).astype(int))
    lin = np.linspace(0, n_max, n_lin + 1).round().astype(int)
    return np.unique(np.concatenate([[0], log, lin, np.arange(min(n_max, 50) + 1)]))
