"""Round 2 helpers: generic teachers with d = 2, exact rank-1 saddles, the saddle-start protocol.

Saddle-start protocol (lead's suggestion): start gradient flow at the exact first saddle (the
balanced bond-rank-1 critical point, zero-padded into the student's bond dimensions) plus fan-in
noise of scale sigma on every core, and measure the escape time. This isolates the law of the
escape from the first saddle from the (expensive, sigma^-(K-2)) first escape from the origin.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from constrained import constrained_minima, metric_loss_of_dense, ranked_spec
from setups import rotation
from ttn.losses import power_law_cov
from ttn.network import TTNSpec, to_dense
from ttn.teachers import CPTeacher, TTNTeacher, random_ttn_teacher
from ttn.tree import Tree, make_tree

D = 2


def teacher_for(kind: str, n: int, **kw):
    """Generic teachers on n sites, d = 2. Returns (teacher, site_metrics or None, description).

    nonorth: u_1 = e1, u_2 = c e1 + sqrt(1-c^2) e2 at every site, weights (1, s2).
    misaligned: odeco on e1, e2 (weights 1, 0.8), Sigma_i = R(theta) diag(power_law_cov(2, alpha)) R^T.
    random_mps / random_binary: random TTN teacher (ttn.teachers.teacher_generator stream) on an mps /
    binary tree of n sites, all bond ranks 2, bond index k weighted by decay^k, ||W*|| = 1.
    """
    if kind == "nonorth":
        c, s2 = kw["c"], kw.get("s2", 0.7)
        u1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        u2 = torch.tensor([c, math.sqrt(1 - c * c)], dtype=torch.float64)
        f = torch.stack([u1.expand(n, D), u2.expand(n, D)]).clone()
        return CPTeacher(torch.tensor([1.0, s2], dtype=torch.float64), f), None
    if kind == "misaligned":
        rot = rotation(D, math.radians(kw["theta_deg"]))
        cov = rot @ power_law_cov(D, kw.get("alpha", 1.0)) @ rot.T
        eye = torch.eye(D, dtype=torch.float64)
        f = torch.stack([eye[0].expand(n, D), eye[1].expand(n, D)]).clone()
        return CPTeacher(torch.tensor([1.0, 0.8], dtype=torch.float64), f), [cov] * n
    if kind in ("random_mps", "random_binary"):
        tree = make_tree(kind.split("_")[1], n)
        bonds = [1] * tree.n_nodes
        for v in tree.edges:
            below = len(tree.subtree_sites[v])
            bonds[v] = min(2, D ** below, D ** (n - below))
        spec_t = TTNSpec(tree, (D,) * n, tuple(bonds))
        base = random_ttn_teacher(spec_t, seed=kw["teacher_seed"], norm=1.0)
        q = kw.get("decay", 0.6)
        cores = [c_.clone() for c_ in base.cores]
        for v in tree.edges:
            cores[v] = cores[v] * torch.tensor([q**k for k in range(bonds[v])], dtype=torch.float64)
        t = TTNTeacher(spec_t, cores)
        scale = t.norm2().sqrt() ** (-1.0 / spec_t.n_cores)
        return TTNTeacher(spec_t, [c_ * scale for c_ in cores]), None
    raise ValueError(kind)


def student(tree_kind: str, n: int, chi: int = 4) -> TTNSpec:
    return TTNSpec.uniform(make_tree(tree_kind, n), D, chi)


def rank1_saddles(spec: TTNSpec, teacher, metrics, n_starts=12, seed=0):
    """Distinct local minima of L on bond-rank-1 TTNs (the candidate first saddles), with cores."""
    spec1 = ranked_spec(spec, {v: 1 for v in spec.tree.edges})
    out = constrained_minima(spec1, metric_loss_of_dense(teacher.dense(), metrics),
                             n_starts=n_starts, seed=seed)
    for m in out:
        m["cores"] = [c.detach().clone() for c in spec1.unflatten(torch.from_numpy(m["y"]))]
        m["spec1"] = spec1
    return out


def balance_rank1(cores):
    """Rank-1 bonds: Q_e = |G_v|^2 - |G_p|^2, so balanced = equal core norms (product preserved)."""
    norms = [float(c.norm()) for c in cores]
    target = math.exp(np.mean(np.log(norms)))
    return [c * (target / nv) for c, nv in zip(cores, norms)]


def embed(spec: TTNSpec, cores1):
    """Zero-pad bond-rank-1 cores into spec's bond dimensions (learned index 0 on every bond)."""
    out = []
    for g1, shape in zip(cores1, spec.shapes):
        g = torch.zeros(shape, dtype=torch.float64)
        idx = tuple(slice(0, s) for s in g1.shape)
        g[idx] = g1
        out.append(g)
    return out


def saddle_start(spec: TTNSpec, saddle_cores, sigma: float, seed: int):
    """theta0 = balanced saddle + fan-in noise of scale sigma (spec.init stream, seed)."""
    noise = spec.init(sigma, seed=seed)
    return spec.flatten([a + b for a, b in zip(saddle_cores, noise)])
