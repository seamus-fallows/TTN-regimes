"""Predicted escape degree k for pattern teachers: the Steiner rule (H2) and its bond-rank
generalization (notes/locality.md, sections 2 and 5).

Pattern teachers use orthonormal basis vectors at every site, so the matricization of any partial
sum sum_{r in L} s_r U_r across an edge is block-structured: its rank equals the rank of the small
matrix M[alpha, beta] = sum of s_r over components with below-pattern alpha and above-pattern beta.
"""

from __future__ import annotations

import numpy as np

from ttn.tree import Tree, make_tree


def custom_tree(name: str) -> Tree:
    if name == "star3":  # root without sites, three children carrying two sites each (N = 6, K = 4)
        return Tree((3, 3, 3, -1), ((0, 1), (2, 3), (4, 5), ()), name="star3")
    if name == "midroot6":  # chain 0-1-2-3-4-5 rooted at node 3 (same graph as mps6)
        return Tree((1, 2, 3, -1, 3, 4), ((0,), (1,), (2,), (3,), (4,), (5,)), name="midroot6")
    raise ValueError(name)


def get_tree(kind: str, n: int | None = None) -> Tree:
    if kind in ("mps", "binary", "ht"):
        return make_tree(kind, n)
    return custom_tree(kind)


def steiner_k(tree: Tree, S) -> int:
    """H2: number of cores in the Steiner subtree of the cores carrying the sites in S."""
    return len(tree.steiner_nodes(sorted(S)))


def edge_rank(tree: Tree, patterns, weights, members, v: int) -> int:
    """Rank of sum_{r in members} s_r U_r matricized across edge v (below = subtree of v)."""
    if not members:
        return 0
    below = sorted(tree.subtree_sites[v])
    above = [i for i in range(tree.n_sites) if i not in below]
    rows, cols, entries = {}, {}, []
    for r in members:
        a = tuple(patterns[r][i] for i in below)
        b = tuple(patterns[r][i] for i in above)
        entries.append((rows.setdefault(a, len(rows)), cols.setdefault(b, len(cols)), weights[r]))
    m = np.zeros((len(rows), len(cols)))
    for i, j, s in entries:
        m[i, j] += s
    return int(np.linalg.matrix_rank(m, tol=1e-9))


def rank_rule(tree: Tree, patterns, weights, learned, r):
    """Generalized rule: the cores that must change to add component r to the learned set are the
    endpoints of the edges whose rank increases; k = their number (1 if no rank increases).

    Returns (k, nodes, rising_edges)."""
    learned = list(learned)
    rising = [v for v in tree.edges
              if edge_rank(tree, patterns, weights, learned + [r], v)
              > edge_rank(tree, patterns, weights, learned, v)]
    nodes = set()
    for v in rising:
        nodes.update((v, tree.parent[v]))
    if not nodes:
        return 1, set(), []
    return len(nodes), nodes, rising


def two_component_patterns(n: int, S) -> list[list[int]]:
    return [[0] * n, [1 if i in S else 0 for i in range(n)]]
