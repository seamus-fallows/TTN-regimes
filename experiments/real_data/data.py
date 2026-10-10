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
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ttn.dynamics import Problem
from ttn.network import forward
from ttn.network import TTNSpec
from ttn.teachers import CPTeacher, ShiftedTeacher
from ttn.tree import Tree, binary_tree, mps_tree

# directory written by download.py; override with the environment variable REAL_DATA_DIR
DATA_DIR = Path(os.environ.get("REAL_DATA_DIR", "/root/outputs/real_data/data"))
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


def load_task(task: Task, data_dir: Path | None = None):
    """Pixel images in [0, 1] of shape (n, size^2) (row-major) and labels +-1, train and test."""
    raw = np.load(Path(data_dir or DATA_DIR) / f"{task.dataset}_{task.size}x{task.size}.npz")
    rng = np.random.default_rng(task.data_seed)
    out = []
    for split, n in (("train", task.n_train), ("test", task.n_test)):
        x, lab = raw[f"x_{split}"], raw[f"y_{split}"]
        idx_pos = np.flatnonzero(np.isin(lab, task.pos))
        idx_neg = np.flatnonzero(np.isin(lab, task.neg))
        n_pos = n // 2
        if n_pos > min(len(idx_pos), len(idx_neg)):
            raise ValueError(f"not enough {split} samples for n = {n}")
        idx = np.concatenate(
            [
                rng.choice(idx_pos, n_pos, replace=False),
                rng.choice(idx_neg, n - n_pos, replace=False),
            ]
        )
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
        return np.array(
            [p for r, row in enumerate(rows) for p in (row if r % 2 == 0 else row[::-1])]
        )
    if name.startswith("scrambled"):
        return np.random.default_rng(int(name[len("scrambled") :]) + 12345).permutation(n)
    if name == "rowmajor":
        return np.arange(n)
    # orders added for the pre-registered order test (notes/real_data.md section 9)
    if name == "quadtreeT":  # Morton order of the transposed image (as in the round-1 review runs)
        return np.array([(p % size) * size + p // size for p in pixel_order("quadtree", size)])
    if name == "colmajor":
        return np.array([(s % size) * size + s // size for s in range(n)])
    if name == "hilbert":
        return np.array([y * size + x for x, y in (_hilbert_xy(size, d) for d in range(n))])
    if name == "mortonrev":  # site s carries the pixel whose Morton key is s with its bits reversed
        bits = 2 * int(math.log2(size))
        rev = [int(format(s, f"0{bits}b")[::-1], 2) for s in range(n)]
        return pixel_order("quadtree", size)[rev]
    raise ValueError(name)


def _hilbert_xy(size: int, d: int) -> tuple[int, int]:
    """(x, y) of the d-th cell of the Hilbert curve on a size x size grid (size a power of 2)."""
    x = y = 0
    s, t = 1, d
    while s < size:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x, y = x + s * rx, y + s * ry
        t //= 4
        s *= 2
    return x, y


def make_tree(kind: str, n_sites: int) -> Tree:
    return {"binary": binary_tree, "mps": mps_tree}[kind](n_sites)


def zero_teacher(n_sites: int, d: int = 2) -> CPTeacher:
    """Placeholder target for `Problem` (no teacher tensor on real data)."""
    return CPTeacher(
        torch.zeros(1, dtype=torch.float64),
        torch.ones(1, n_sites, d, dtype=torch.float64) / math.sqrt(d),
    )


@dataclass
class RealProblem:
    problem: Problem
    x_test: torch.Tensor
    y_test: torch.Tensor
    order: np.ndarray


def build(
    task: Task, tree_kind: str, ordering: str, chi: int, features: str = "raw", centred_cores0=None
) -> RealProblem:
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
    n_sites = task.size**2
    spec = TTNSpec.uniform(make_tree(tree_kind, n_sites), 2, chi)
    teacher = zero_teacher(n_sites)
    if centred_cores0 is not None:
        teacher = ShiftedTeacher(teacher, spec, centred_cores0)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    problem = GNProblem(spec, teacher, dataset=(t(phi_tr), t(y_tr)))
    return RealProblem(problem, t(phi_te), t(y_te), order)


class GNProblem(Problem):
    """`Problem` whose `jacobian` (used only by the implicit integrators, as the Newton matrix) is the
    Gauss-Newton part -(1/n) J^T J of -Hessian, J = per-sample d f / d theta.

    The exact Hessian costs ~P gradient evaluations (20-60 s here); J costs about one. LSODA's error
    control does not depend on the Jacobian (it only sets how fast its Newton iterations converge), so
    the trajectory has the same accuracy; the missing residual term sum_j r_j d^2 f_j / n makes LSODA
    take somewhat smaller steps where it matters.
    """

    def per_sample_jacobian(self, theta: torch.Tensor) -> torch.Tensor:
        """J = d f(x_j) / d theta (n x P) in closed form: f is linear in each core, so d f / d G_v is
        the outer product of the core's inputs (site vectors, child messages) and the message from
        above (`messages`). The centred model fits f_theta - f_theta0, whose Jacobian is that of
        f_theta (theta0 is fixed); the two-copy model is not supported here."""
        if self.two_copy:
            raise NotImplementedError("closed-form Jacobian is for plain / centred models")
        spec = self.spec
        t = spec.tree
        x = self.dataset[0]
        cores = spec.unflatten(theta.detach())
        with torch.no_grad():
            up, down = messages(spec, cores, x)
            blocks = []
            for v in range(t.n_nodes):
                ins = [x[:, s, :] for s in t.sites[v]] + [up[c] for c in t.children[v]] + [down[v]]
                out = ins[0]
                for m in ins[1:]:
                    out = (out[:, :, None] * m[:, None, :]).reshape(len(x), -1)
                blocks.append(out)
        return torch.cat(blocks, dim=1)

    def jacobian(self, theta: torch.Tensor) -> torch.Tensor:
        jac = self.per_sample_jacobian(theta)
        return -(jac.T @ jac) / jac.shape[0]


class DenseHessianProblem(GNProblem):
    """Exact Hessian for small N (d^N entries): H = (1/n) J^T J + Hessian of <R, W(theta)>, with the
    residual tensor R = (1/n) sum_j (f_j - y_j) Phi_j formed densely (two half-tensors and one
    matmul). ~2 s at N = 16, chi = 4, against ~20 s for autograd's Hessian of the empirical loss."""

    def residual_tensor(self, theta: torch.Tensor) -> torch.Tensor:
        from ttn.diagnostics import product_features

        x, y = self.dataset
        with torch.no_grad():
            r = forward(*self.effective(theta), x) - y
        half = x.shape[1] // 2
        left = product_features(x[:, :half])
        right = product_features(x[:, half:])
        return ((left * (r / x.shape[0])[:, None]).T @ right).reshape(-1)

    def jacobian(self, theta: torch.Tensor) -> torch.Tensor:
        from ttn.network import to_dense

        theta = theta.detach()
        res = self.residual_tensor(theta)
        hres = torch.func.hessian(
            lambda th: (to_dense(*self.effective(th)).reshape(-1) * res).sum()
        )(theta)
        jac = self.per_sample_jacobian(theta)
        hess = (jac.T @ jac) / jac.shape[0] + hres
        return -hess


_LEG = "abcdefghijklmnopqrstuvw"


def messages(spec: TTNSpec, cores, x: torch.Tensor):
    """Per-sample messages across every edge: up[v] (n, chi_v) = the subtree of v contracted with
    the sample's site vectors, down[v] (n, chi_v) = everything else, so f(x_j) = up[v]_j . down[v]_j
    for every edge v."""
    t = spec.tree
    vecs = [x[:, i, :] for i in range(x.shape[1])]
    up = [None] * t.n_nodes
    for v in t.postorder:
        n_phys, kids = len(t.sites[v]), t.children[v]
        legs = _LEG[: n_phys + len(kids) + 1]
        ops = [vecs[s] for s in t.sites[v]] + [up[c] for c in kids]
        eq = ",".join([legs] + ["z" + legs[k] for k in range(len(ops))]) + "->z" + legs[-1]
        up[v] = torch.einsum(eq, cores[v], *ops)
    down = [None] * t.n_nodes
    down[t.root] = torch.ones(x.shape[0], 1, dtype=x.dtype)
    for v in t.preorder:
        n_phys, kids = len(t.sites[v]), t.children[v]
        legs = _LEG[: n_phys + len(kids) + 1]
        for k, c in enumerate(kids):
            ops, terms = [down[v]], ["z" + legs[-1]]
            for j, s in enumerate(t.sites[v]):
                ops.append(vecs[s])
                terms.append("z" + legs[j])
            for k2, c2 in enumerate(kids):
                if k2 != k:
                    ops.append(up[c2])
                    terms.append("z" + legs[n_phys + k2])
            eq = ",".join([legs] + terms) + "->z" + legs[n_phys + k]
            down[c] = torch.einsum(eq, cores[v], *ops)
    return up, down


def function_spectra(spec: TTNSpec, cores, x: torch.Tensor) -> np.ndarray:
    """Data-weighted bond spectra (E, chi_max), NaN-padded, edges in `spec.tree.edges` order:
    across edge v, f(x) = sum_a up_a(x_below) down_a(x_above), and the Schmidt values with respect to
    the empirical marginals of the two sides are sqrt(eig(G_up^1/2 G_down G_up^1/2)), G = (1/n) M^T M
    of the per-sample messages. Gauge invariant; for a product input law sum_k s_k^2 = E[f^2]. Unlike
    the Frobenius spectra of W, directions of W that the data does not see do not count."""
    with torch.no_grad():
        up, down = messages(spec, cores, x)
        n = x.shape[0]
        edges = spec.tree.edges
        chi = max(spec.bond_dims[v] for v in edges)
        out = np.full((len(edges), chi), np.nan)
        for j, v in enumerate(edges):
            g_up = up[v].T @ up[v] / n
            g_dn = down[v].T @ down[v] / n
            vals, vecs = torch.linalg.eigh(g_up)
            root = (vecs * vals.clamp_min(0).sqrt()) @ vecs.T
            ev = torch.linalg.eigvalsh(root @ g_dn @ root).clamp_min(0).sqrt().flip(0)
            out[j, : ev.numel()] = ev.numpy()
    return out
