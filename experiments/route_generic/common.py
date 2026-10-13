"""Targets, model and per-step measurements of the route_generic experiment.

Target families (all whitened, exact population loss L = 1/2 ||W - W*||^2):
- pattern: a star-condition pattern target (`ttn.routes.RouteTarget`), the control inside the theory;
- cp: R rank-one components with per-site overlap c between every pair, <u_r^(i), u_q^(i)> = c at every
  site i; factors u_r^(i) = O_i C[r] with C the Cholesky factor of the Gram (1 - c) I + c 11^T and
  O_i a Haar-random rotation per site; weights drawn uniformly in [0.3, 1], sorted, the largest set to 1;
- ttn: a random TTN on the student's tree (`ttn.teachers.random_ttn_teacher`, Gaussian cores,
  ||W*|| = 1) with bond dimension `teacher_chi`.
Teacher randomness uses `teacher_generator(teacher_seed)`, independent of the init seed.

The loss is evaluated on the dense tensor (N <= 8 sites, d <= 3: at most 6561 entries); bond spectra
are singular values of the dense matricization across each edge's cut (absolute accuracy ~1e-16 s_1).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ttn.network import TTNSpec, to_dense
from ttn.routes import RouteTarget
from ttn.teachers import (
    CPTeacher,
    random_orthogonal,
    random_ttn_teacher,
    teacher_generator,
)
from ttn.tree import make_tree

TREES = {"mps6": ("mps", 6), "binary8": ("binary", 8)}


def get_tree(name: str):
    return make_tree(*TREES[name])


def cp_overlap_teacher(n_sites: int, d: int, n_comp: int, overlap: float, seed: int) -> CPTeacher:
    """Random CP teacher with per-site pairwise overlap `overlap` and random weights in [0.3, 1]."""
    g = teacher_generator(seed)
    gram = (1 - overlap) * torch.eye(n_comp, dtype=torch.float64) + overlap
    chol = torch.linalg.cholesky(gram)  # rows have inner products `gram`
    vecs = torch.zeros(n_comp, d, dtype=torch.float64)
    vecs[:, :n_comp] = chol
    factors = torch.stack(
        [vecs @ random_orthogonal(d, g).T for _ in range(n_sites)], dim=1
    )  # (R, N, d)
    w = 0.3 + 0.7 * torch.rand(n_comp, generator=g, dtype=torch.float64)
    w = torch.sort(w, descending=True).values
    w = w / w[0]
    return CPTeacher(w, factors)


def build_target(case: dict):
    """Returns (spec, teacher, info): the student spec, a teacher with .dense(), and target info."""
    tree = get_tree(case["tree"])
    fam = case["family"]
    d = case["d"]
    info: dict = {}
    if fam == "pattern":
        rt = RouteTarget(tree, case["patterns"], case["weights"])
        assert rt.is_star, "pattern control must satisfy the star condition"
        teacher = rt.teacher(d)
        chi = max(rt.n_blocks.values()) + 2
        info["route_target"] = rt
    elif fam == "cp":
        teacher = cp_overlap_teacher(tree.n_sites, d, case["n_comp"], case["overlap"], case["teacher_seed"])
        chi = case["n_comp"] + 2
    elif fam == "ttn":
        tspec = TTNSpec.uniform(tree, d, case["teacher_chi"])
        teacher = random_ttn_teacher(tspec, seed=case["teacher_seed"], norm=1.0)
        chi = case["teacher_chi"] + 2
    else:
        raise ValueError(fam)
    spec = TTNSpec.uniform(tree, d, chi)
    return spec, teacher, info


def cuts(tree):
    """For every edge e (named by its lower node): the sites below e, in a fixed order."""
    out = {}
    for e in tree.edges:
        below = list(tree.subtree_sites[e])
        out[e] = sorted(below)
    return out


def spectra_dense(w: torch.Tensor, tree, keep: dict[int, int]) -> np.ndarray:
    """Concatenated top-`keep[e]` singular values of W across every edge (edge order tree.edges)."""
    n = tree.n_sites
    d = w.shape[0]
    out = []
    for e in tree.edges:
        below = sorted(tree.subtree_sites[e])
        rest = [s for s in range(n) if s not in below]
        mat = w.permute(below + rest).reshape(d ** len(below), -1)
        sv = torch.linalg.svdvals(mat)
        out.append(sv[: keep[e]])
    return torch.cat(out).numpy()


def teacher_spectra(w_star: torch.Tensor, tree) -> dict[int, np.ndarray]:
    """Full bond spectra of the target, per edge."""
    n = tree.n_sites
    d = w_star.shape[0]
    out = {}
    for e in tree.edges:
        below = sorted(tree.subtree_sites[e])
        rest = [s for s in range(n) if s not in below]
        mat = w_star.permute(below + rest).reshape(d ** len(below), -1)
        out[e] = torch.linalg.svdvals(mat).numpy()
    return out


def target_ranks(ref: dict[int, np.ndarray], tol: float = 1e-10) -> dict[int, int]:
    return {e: int((s > tol * s[0]).sum()) for e, s in ref.items()}


def steiner_nodes(tree, nodes) -> set[int]:
    nodes = list(nodes)
    if not nodes:
        return set()
    out = {nodes[0]}
    for v in nodes[1:]:
        out.update(tree.path(nodes[0], v))
    return out


def k_region(tree, edges) -> int:
    """Cores in the smallest subtree containing the end nodes of `edges` (edge e joins e and its parent)."""
    ends = set()
    for e in edges:
        ends.update((e, tree.parent[e]))
    return len(steiner_nodes(tree, ends))


def dense_teacher(teacher) -> torch.Tensor:
    if isinstance(teacher, CPTeacher):
        return teacher.dense()
    return to_dense(teacher.spec, teacher.cores)


def sigma_grid(lo_exp: float, hi_exp: float, step: float = 0.25) -> list[float]:
    n = int(round((hi_exp - lo_exp) / step))
    return [float(10 ** (lo_exp + step * i)) for i in range(n + 1)]


def fmt_sigma(s: float) -> str:
    return f"{math.log10(s):+.2f}"
