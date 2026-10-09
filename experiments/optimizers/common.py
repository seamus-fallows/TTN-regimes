"""Problem setup and trajectory recording shared by all optimizers of this experiment.

A run is described by a flat dict (one row of a sweep, see `run.py`):
  tree, n_sites, d, chi, teacher ('odeco' | 'pattern'), weights, patterns (pattern teacher),
  teacher_seed, sigma, seed, optimizer and its hyperparameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ttn.losses import population_loss
from ttn.network import TTNSpec, bond_spectra, conserved_quantities
from ttn.teachers import CPTeacher, odeco_teacher, pattern_teacher
from ttn.tree import make_tree


@dataclass
class Problem:
    spec: TTNSpec
    teacher: CPTeacher
    teacher_norm2: torch.Tensor

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return population_loss(self.spec, self.spec.unflatten(x), self.teacher, None, self.teacher_norm2)

    def loss_grad(self, x: torch.Tensor) -> tuple[float, torch.Tensor]:
        x = x.detach().requires_grad_(True)
        loss = self.loss(x)
        (grad,) = torch.autograd.grad(loss, x)
        return loss.item(), grad

    def overlaps(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.teacher.component_overlaps(self.spec, self.spec.unflatten(x)).numpy()


def make_problem(cfg: dict) -> Problem:
    tree = make_tree(cfg["tree"], cfg["n_sites"])
    spec = TTNSpec.uniform(tree, cfg["d"], cfg["chi"])
    if cfg["teacher"] == "odeco":
        teacher = odeco_teacher(cfg["n_sites"], cfg["d"], cfg["weights"], seed=cfg["teacher_seed"])
    elif cfg["teacher"] == "pattern":
        teacher = pattern_teacher(cfg["weights"], cfg["patterns"], cfg["d"], seed=cfg["teacher_seed"])
    else:
        raise ValueError(cfg["teacher"])
    return Problem(spec, teacher, teacher.norm2())


def initial_point(problem: Problem, cfg: dict) -> torch.Tensor:
    cores = problem.spec.init(cfg["sigma"], seed=cfg["seed"])
    return problem.spec.flatten(cores)


# ---- recording ---------------------------------------------------------------------------------


def balance_index(spec: TTNSpec, cores) -> np.ndarray:
    """Per edge: ||Q_e||_F / (||G_v^T G_v||_F + ||G_p,(v) G_p,(v)^T||_F), in [0, 1].

    0 = balanced (min-norm gauge on that edge), 1 = all weight on one side of the edge."""
    t = spec.tree
    q = conserved_quantities(spec, cores)
    out = []
    for v in t.edges:
        p = t.parent[v]
        gv = cores[v].reshape(-1, cores[v].shape[-1])
        leg = len(t.sites[p]) + t.children[p].index(v)
        gp = torch.movedim(cores[p], leg, 0).reshape(cores[p].shape[leg], -1)
        scale = (gv.T @ gv).norm() + (gp @ gp.T).norm()
        out.append((q[v].norm() / scale).item() if scale > 0 else 0.0)
    return np.array(out)


class Recorder:
    """Records loss, overlaps, bond spectra, Q_e drift and balance at chosen 'times'.

    `time` is the optimizer's own clock (gradient-flow time, steps, or sweeps); the caller decides.
    Cheap quantities (loss, overlaps) are recorded at every call of `record`; bond spectra,
    Q drift and balance only every `heavy_every`-th record.
    """

    def __init__(self, problem: Problem, x0: torch.Tensor, heavy_every: int = 1):
        self.p = problem
        self.spec = problem.spec
        self.edges = self.spec.tree.edges
        self.chi_max = max(self.spec.bond_dims[v] for v in self.edges)
        cores0 = self.spec.unflatten(x0.detach())
        self.q0 = {v: q.clone() for v, q in conserved_quantities(self.spec, cores0).items()}
        self.x0_norm = x0.norm().item()
        self.x0 = x0.detach().clone()
        self.heavy_every = heavy_every
        self.n = 0
        self.rows: dict[str, list] = {k: [] for k in ("time", "loss", "overlaps", "dtheta", "core_norms")}
        self.heavy: dict[str, list] = {k: [] for k in ("time", "spectra", "q_drift", "balance")}

    def record(self, time: float, x: torch.Tensor, loss: float | None = None, force_heavy=False):
        x = x.detach()
        if loss is None:
            with torch.no_grad():
                loss = self.p.loss(x).item()
        cores = self.spec.unflatten(x)
        self.rows["time"].append(time)
        self.rows["loss"].append(loss)
        self.rows["overlaps"].append(self.p.overlaps(x))
        self.rows["dtheta"].append((x - self.x0).norm().item() / self.x0_norm)
        self.rows["core_norms"].append([c.norm().item() for c in cores])
        if force_heavy or self.n % self.heavy_every == 0:
            with torch.no_grad():
                spectra = bond_spectra(self.spec, cores)
                sp = np.full((len(self.edges), self.chi_max), np.nan)
                for i, v in enumerate(self.edges):
                    sp[i, : len(spectra[v])] = spectra[v].numpy()
                q = conserved_quantities(self.spec, cores)
                drift = [(q[v] - self.q0[v]).norm().item() for v in self.edges]
                self.heavy["time"].append(time)
                self.heavy["spectra"].append(sp)
                self.heavy["q_drift"].append(drift)
                self.heavy["balance"].append(balance_index(self.spec, cores))
        self.n += 1

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v, dtype=float) for k, v in self.rows.items()}
        out.update({"h_" + k: np.asarray(v, dtype=float) for k, v in self.heavy.items()})
        out["teacher_weights"] = self.p.teacher.weights.numpy()
        out["teacher_spectra"] = teacher_spectra(self.p)
        return out


def teacher_spectra(problem: Problem) -> np.ndarray:
    """Bond spectra of W* on the student's tree, from the dense teacher (small N only)."""
    spec = problem.spec
    w = problem.teacher.dense()
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


def record_schedule(n_max: int, n_log: int = 1500, n_lin: int = 3000) -> np.ndarray:
    """Step indices 0..n_max on a union of a log grid and a linear grid.

    The linear grid lets plateaus be judged in linear time (notes/diagnostics.md)."""
    log = np.unique(np.round(np.logspace(0, math.log10(max(n_max, 1)), n_log)).astype(int))
    lin = np.linspace(0, n_max, n_lin + 1).round().astype(int)
    return np.unique(np.concatenate([[0], log, lin, np.arange(min(n_max, 200) + 1)]))
