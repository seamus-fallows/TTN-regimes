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


def predicted_sequence(tree: Tree, patterns, weights):
    """Greedy prediction of the saddle sequence under the rank rule.

    Stage 0 learns the largest component plus every component that can join it with k = 1. Each
    later stage learns the unlearned component with the smallest k (ties: larger weight) together
    with any k = 1 followers. Returns a list of stages (members, k, tie, conflict): `tie` marks a
    k tie decided by weight only; `conflict` marks two k = 1 candidates that are not jointly k = 1
    (the escape then learns a mixture and the component picture is ambiguous)."""
    order = sorted(range(len(weights)), key=lambda r: -weights[r])
    learned: list[int] = []
    stages = []
    while len(learned) < len(weights):
        rest = [r for r in order if r not in learned]
        if not learned:
            ks = {r: (0 if r == rest[0] else 99) for r in rest}
        else:
            ks = {r: rank_rule(tree, patterns, weights, learned, r)[0] for r in rest}
        kmin = min(ks.values())
        cands = [r for r in rest if ks[r] == kmin]
        tie = len(cands) > 1 and kmin >= 2
        first = cands[0]
        members = [first]
        learned.append(first)
        conflict = False
        while True:
            rest = [r for r in order if r not in learned]
            k1 = [r for r in rest if rank_rule(tree, patterns, weights, learned, r)[0] == 1]
            if not k1:
                break
            if len(k1) > 1:
                both = all(rank_rule(tree, patterns, weights, learned + [a], b)[0] == 1
                           for a in k1 for b in k1 if a != b)
                conflict = conflict or not both
            members.append(k1[0])
            learned.append(k1[0])
        stages.append((members, kmin, tie, conflict))
    return stages


def describe_sequence(stages) -> str:
    def law(k):
        return {0: "origin", 1: "none", 2: "log"}.get(k, f"s^-{k - 2}")
    parts = []
    for members, k, tie, conflict in stages:
        flag = ("~" if tie else "") + ("!" if conflict else "")
        parts.append(f"[{law(k)}{flag}] U{'+U'.join(str(m + 1) for m in members)}")
    return " -> ".join(parts)


def furman_components(tree: Tree, patterns, weights, learned):
    """Connected components of rank-deficient edges (student rank < teacher rank) at the saddle
    W_L = sum_{r in learned} s_r U_r, with N = number of nodes touching each component
    ([Furman2026] Lemmas D.8-D.9: a descent curve of order N exists, an upper bound on the
    saddle order). Returns a list of (edges, nodes)."""
    everything = list(range(len(weights)))
    deficient = [v for v in tree.edges
                 if edge_rank(tree, patterns, weights, list(learned), v)
                 < edge_rank(tree, patterns, weights, everything, v)]
    comps = []
    left = set(deficient)
    while left:
        stack = [left.pop()]
        comp = set(stack)
        while stack:
            e = stack.pop()
            ends = {e, tree.parent[e]}
            for f in list(left):
                if {f, tree.parent[f]} & ends:
                    left.remove(f)
                    comp.add(f)
                    stack.append(f)
        nodes = set()
        for e in comp:
            nodes.update((e, tree.parent[e]))
        comps.append((sorted(comp), nodes))
    return comps


def furman_N(tree: Tree, patterns, weights, learned, rising) -> int:
    """N of the deficient component that contains the given rising edges (0 if none)."""
    for edges, nodes in furman_components(tree, patterns, weights, learned):
        if set(rising) & set(edges):
            return len(nodes)
    return 0


def naive_k(tree: Tree, patterns, learned, r) -> int:
    """Literal H2 with several learned components: Steiner size of the sites where U_r differs from
    the closest learned component (minimum over learned components)."""
    best = None
    for j in learned:
        S = [i for i in range(tree.n_sites) if patterns[r][i] != patterns[j][i]]
        k = steiner_k(tree, S) if S else 0
        best = k if best is None else min(best, k)
    return best
