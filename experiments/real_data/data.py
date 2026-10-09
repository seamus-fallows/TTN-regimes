"""Real-data problems: downsampled MNIST / Fashion-MNIST, cos/sin site features, pixel orderings.

A problem is the project's TTN regression f(x) = <W, phi_1(x) (x) ... (x) phi_N(x)> trained on the
empirical square loss 1/(2n) sum_j (f(x_j) - y_j)^2 with y = +-1 (`ttn.dynamics.Problem` with a fixed
dataset). There is no teacher tensor: `Problem` gets a zero CP teacher, so its `population_loss` is
meaningless here (it is 1/2 ||W||^2); train and test losses are what count.

Pixel orderings (site s of the tree carries pixel order[s], pixels numbered row-major):
- quadtree: Morton (Z) order, so every binary-tree subtree is a square or a 2:1 rectangle of pixels;
- snake: boustrophedon rows (MPS chain with nearest neighbours adjacent in the image);
- scrambled<seed>: a random permutation (locality destroyed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ttn.dynamics import Problem
from ttn.network import TTNSpec
from ttn.teachers import CPTeacher, ShiftedTeacher
from ttn.tree import Tree, binary_tree, mps_tree

DATA_DIR = Path("/root/outputs/real_data/data")
WHITEN_FLOOR = 1e-3  # eigenvalue floor (relative to the largest) of the per-site second moment


@dataclass
class Task:
    """A +-1 regression task: `pos` classes -> +1, `neg` classes -> -1, balanced subsample."""

    dataset: str = "mnist"
    size: int = 4
    pos: tuple[int, ...] = (3,)
    neg: tuple[int, ...] = (5,)
    n_train: int = 4000
    n_test: int = 1600
    data_seed: int = 0


def load_task(task: Task, data_dir: Path = DATA_DIR):
    """Pixel images in [0, 1] of shape (n, size^2) (row-major) and labels +-1, train and test."""
    raw = np.load(data_dir / f"{task.dataset}_{task.size}x{task.size}.npz")
    rng = np.random.default_rng(task.data_seed)
    out = []
    for split, n in (("train", task.n_train), ("test", task.n_test)):
        x, lab = raw[f"x_{split}"], raw[f"y_{split}"]
        idx_pos = np.flatnonzero(np.isin(lab, task.pos))
        idx_neg = np.flatnonzero(np.isin(lab, task.neg))
        n_pos = n // 2
        if n_pos > min(len(idx_pos), len(idx_neg)):
            raise ValueError(f"not enough {split} samples for n = {n}")
        idx = np.concatenate([rng.choice(idx_pos, n_pos, replace=False),
                              rng.choice(idx_neg, n - n_pos, replace=False)])
        idx = rng.permutation(idx)
        y = np.where(np.isin(lab[idx], task.pos), 1.0, -1.0)
        out += [x[idx].reshape(len(idx), -1), y]
    return out  # x_train, y_train, x_test, y_test


def cos_sin(pixels: np.ndarray) -> np.ndarray:
    """(n, N) pixels in [0, 1] -> (n, N, 2) features (cos(pi x / 2), sin(pi x / 2))."""
    a = 0.5 * math.pi * pixels
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def whitening_maps(phi_train: np.ndarray) -> np.ndarray:
    """Per-site maps A_i with A_i E[phi_i phi_i^T] A_i^T = I (uncentred second moment, so the span
    of the site functions is unchanged), eigenvalues floored at WHITEN_FLOOR x the largest."""
    maps = []
    for i in range(phi_train.shape[1]):
        m = phi_train[:, i, :].T @ phi_train[:, i, :] / phi_train.shape[0]
        vals, vecs = np.linalg.eigh(m)
        vals = np.maximum(vals, WHITEN_FLOOR * vals.max())
        maps.append((vecs / np.sqrt(vals)) @ vecs.T)
    return np.stack(maps)


def pixel_order(name: str, size: int) -> np.ndarray:
    """order[s] = row-major pixel index carried by site s."""
    n = size * size
    if name == "quadtree":
        def morton(r, c):
            key = 0
            for b in range(int(math.log2(size))):
                key |= ((c >> b) & 1) << (2 * b) | ((r >> b) & 1) << (2 * b + 1)
            return key
        keys = [morton(p // size, p % size) for p in range(n)]
        return np.argsort(keys)
    if name == "snake":
        rows = [list(range(r * size, (r + 1) * size)) for r in range(size)]
        return np.array([p for r, row in enumerate(rows) for p in (row if r % 2 == 0 else row[::-1])])
    if name.startswith("scrambled"):
        return np.random.default_rng(int(name[len("scrambled"):]) + 12345).permutation(n)
    if name == "rowmajor":
        return np.arange(n)
    raise ValueError(name)


def make_tree(kind: str, n_sites: int) -> Tree:
    return {"binary": binary_tree, "mps": mps_tree}[kind](n_sites)


def zero_teacher(n_sites: int, d: int = 2) -> CPTeacher:
    """Placeholder target for `Problem` (no teacher tensor on real data)."""
    return CPTeacher(torch.zeros(1, dtype=torch.float64),
                     torch.ones(1, n_sites, d, dtype=torch.float64) / math.sqrt(d))


@dataclass
class RealProblem:
    problem: Problem
    x_test: torch.Tensor
    y_test: torch.Tensor
    order: np.ndarray


def build(task: Task, tree_kind: str, ordering: str, chi: int, features: str = "raw",
          centred_cores0=None) -> RealProblem:
    """Problem on the training set, plus the test set. features: 'raw' or 'white'.
    centred_cores0: initial cores for the centred model (train f - f_0), else None."""
    x_tr, y_tr, x_te, y_te = load_task(task)
    order = pixel_order(ordering, task.size)
    phi_tr, phi_te = cos_sin(x_tr[:, order]), cos_sin(x_te[:, order])
    if features == "white":
        maps = whitening_maps(phi_tr)
        phi_tr = np.einsum("ide,nie->nid", maps, phi_tr)
        phi_te = np.einsum("ide,nie->nid", maps, phi_te)
    elif features != "raw":
        raise ValueError(features)
    n_sites = task.size ** 2
    spec = TTNSpec.uniform(make_tree(tree_kind, n_sites), 2, chi)
    teacher = zero_teacher(n_sites)
    if centred_cores0 is not None:
        teacher = ShiftedTeacher(teacher, spec, centred_cores0)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    problem = Problem(spec, teacher, dataset=(t(phi_tr), t(y_tr)))
    return RealProblem(problem, t(phi_te), t(y_te), order)
