"""Teachers and settings of the committed experiments, rebuilt for the hopper.

Dense teachers in the same basis conventions as the experiments. For pattern and odeco teachers with
whitened inputs the site bases do not matter (the problem is invariant under rotations of each
physical leg), so the standard basis is used; random TTN teachers and covariances are rebuilt with the
experiments' own random streams (`ttn.teachers.teacher_generator`).
"""

from __future__ import annotations

import math

import torch

from hopper import Setting
from ttn.losses import power_law_cov
from ttn.network import TTNSpec, to_dense
from ttn.teachers import TTNTeacher, random_ttn_teacher
from ttn.tree import Tree, make_tree

DT = torch.float64


def custom_tree(name: str) -> Tree:
    """The two custom trees of experiments/locality/rules.py."""
    if name == "star3":
        return Tree((3, 3, 3, -1), ((0, 1), (2, 3), (4, 5), ()), name="star3")
    if name == "midroot6":
        return Tree((1, 2, 3, -1, 3, 4), ((0,), (1,), (2,), (3,), (4,), (5,)), name="midroot6")
    raise ValueError(name)


def get_tree(kind: str, n: int | None = None) -> Tree:
    return make_tree(kind, n) if kind in ("mps", "binary", "ht") else custom_tree(kind)


def rank_one(vectors) -> torch.Tensor:
    X = vectors[0]
    for v in vectors[1:]:
        X = torch.einsum("...,j->...j", X, v)
    return X


def cp_dense(weights, factors):
    """W* = sum_r s_r (x)_i u_r^(i) and the labelled components (s_r, U_r)."""
    comps, W = [], 0
    for s, f in zip(weights, factors):
        U = rank_one([torch.as_tensor(u, dtype=DT) for u in f])
        comps.append((float(s), U))
        W = W + float(s) * U
    return W, comps


def pattern(patterns, weights, d):
    """Pattern teacher in the standard basis: component r is e_{p_r(i)} at site i."""
    eye = torch.eye(d, dtype=DT)
    return cp_dense(weights, [[eye[p] for p in pat] for pat in patterns])


def odeco(n, d, weights):
    return pattern([[r] * n for r in range(len(weights))], weights, d)


def two_component(n, S, weights=(1.0, 0.5)):
    return [[0] * n, [1 if i in S else 0 for i in range(n)]], list(weights)


def nonorth(n, d, c, weights):
    """theory_teachers 4 / data_teacher 4(a): u_1 = e1, u_2 = c e1 + sqrt(1-c^2) e2 at every site."""
    u1 = torch.zeros(d, dtype=DT)
    u1[0] = 1.0
    u2 = torch.zeros(d, dtype=DT)
    u2[0], u2[1] = c, math.sqrt(1 - c * c)
    return cp_dense(weights, [[u1] * n, [u2] * n])


def rotation(d, theta_deg):
    r = torch.eye(d, dtype=DT)
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
    return r


def misaligned(n, d, theta_deg, weights, alpha=1.0):
    """data_teacher step 2: odeco on e1, e2; Sigma_i = R(theta) power_law_cov(d, alpha) R^T."""
    rot = rotation(d, theta_deg)
    cov = rot @ power_law_cov(d, alpha) @ rot.T
    eye = torch.eye(d, dtype=DT)
    W, comps = cp_dense(weights, [[eye[r]] * n for r in range(len(weights))])
    return W, comps, [cov] * n


def ttn_teacher(teacher_tree: Tree, d, ranks, teacher_seed, decay):
    """data_teacher ttn_teacher / r2_common random_* teachers: random TTN on teacher_tree with the given
    bond ranks (dict edge -> rank), bond index k weighted by decay^k, ||W*|| = 1."""
    bonds = [1] * teacher_tree.n_nodes
    for v, r in ranks.items():
        bonds[v] = r
    spec_t = TTNSpec(teacher_tree, (d,) * teacher_tree.n_sites, tuple(bonds))
    base = random_ttn_teacher(spec_t, seed=teacher_seed, norm=1.0)
    cores = [c.clone() for c in base.cores]
    for v in teacher_tree.edges:
        cores[v] = cores[v] * torch.tensor([decay**k for k in range(bonds[v])], dtype=DT)
    t = TTNTeacher(spec_t, cores)
    scale = t.norm2().sqrt() ** (-1.0 / spec_t.n_cores)
    return to_dense(spec_t, [c * scale for c in cores]).detach()


def r2_random_teacher(kind, n, teacher_seed, decay=0.6, d=2):
    """r2_common.teacher_for('random_mps' / 'random_binary'): all bond ranks min(2, cap)."""
    tree = make_tree(kind, n)
    ranks = {}
    for v in tree.edges:
        below = len(tree.subtree_sites[v])
        ranks[v] = min(2, d**below, d ** (n - below))
    return ttn_teacher(tree, d, ranks, teacher_seed, decay)


def setting(tree, d, chi, W, comps=None, metrics=None, init="fanin", name="", start_component=None):
    return Setting(
        tree=tree,
        d=d,
        chi=chi,
        w_star=W,
        metrics=metrics,
        components=comps,
        init=init,
        name=name,
        start_component=start_component,
    )
