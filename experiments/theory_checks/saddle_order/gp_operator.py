"""Generalized piggyback operator S and the predicted saddle order (notes/theory_saddle_order.md).

q = e_0 on every site, so on core v the learned vector is the grouped index 0 and q_v^perp is the
grouped index range 1..D_v-1. Sector B of a tensor = grouped indices >= 1 on B, = 0 elsewhere.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def core_grouped(tree, d, R):
    """Reshape a dense site-order tensor (numpy) to one leg per core (core order 0..K-1)."""
    perm = [s for v in range(tree.n_nodes) for s in tree.sites[v]]
    X = np.transpose(R, perm)
    return X.reshape([d ** len(tree.sites[v]) for v in range(tree.n_nodes)])


def sector(Rc, B):
    idx = tuple(slice(1, None) if v in B else 0 for v in range(Rc.ndim))
    return Rc[idx]  # legs: cores of B in increasing order


def steiner_cores(tree, C):
    """Steiner subtree of a set of cores (works for cores without sites)."""
    C = sorted(C)
    out = {C[0]}
    for v in C[1:]:
        out.update(tree.path(C[0], v))
    return out


def deg(tree, C):
    return 1 if len(C) == 1 else len(steiner_cores(tree, C))


def k_star(tree, Rc, tol=1e-14):
    K = tree.n_nodes
    best = math.inf
    for size in range(1, K + 1):
        for B in itertools.combinations(range(K), size):
            blk = sector(Rc, set(B))
            if blk.size and np.abs(blk).max() > tol:
                best = min(best, len(steiner_cores(tree, B)))
    return best


def S_matrix(tree, Rc, max_deg):
    """Matrix of S on the direct sum of T(C) over nonempty C with deg(C) <= max_deg."""
    K = tree.n_nodes
    sets = [
        frozenset(C)
        for size in range(1, K + 1)
        for C in itertools.combinations(range(K), size)
        if deg(tree, C) <= max_deg
    ]
    dims = [int(np.prod([Rc.shape[v] - 1 for v in sorted(C)])) for C in sets]
    off = np.concatenate([[0], np.cumsum(dims)])
    S = np.zeros((off[-1], off[-1]))
    for i, C1 in enumerate(sets):
        for j, C2 in enumerate(sets):
            if C1 & C2:
                continue
            B = sorted(C1 | C2)
            block = sector(Rc, set(B))  # legs in order B
            pos1 = [B.index(v) for v in sorted(C1)]
            pos2 = [B.index(v) for v in sorted(C2)]
            M = np.transpose(block, pos1 + pos2).reshape(dims[i], dims[j])
            S[off[i] : off[i + 1], off[j] : off[j + 1]] = M
    return S, sets


def predicted_order(tree, Rc, s_q):
    """(k*, order predicted by Theorems 1-2, lambda_max(S|L_l) for l < k*/2)."""
    ks = k_star(tree, Rc)
    lams = {}
    order = ks
    for ell in range(1, math.ceil(ks / 2)):
        if 2 * ell >= ks:
            break
        S, _ = S_matrix(tree, Rc, ell)
        lam = float(np.linalg.eigvalsh(S).max()) if S.size else 0.0
        lams[ell] = lam
        if lam > s_q and 2 * ell < order:
            order = 2 * ell
    return ks, order, lams
