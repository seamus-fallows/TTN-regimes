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


# ---- route rule (theory_saddles.md R1): lowest degree of a route with non-zero local residual ----

def connected_subsets(tree: Tree, max_size: int):
    """All connected node sets of size <= max_size (as frozensets)."""
    nbrs = {v: set(tree.children[v]) | ({tree.parent[v]} if tree.parent[v] >= 0 else set())
            for v in range(tree.n_nodes)}
    found = {frozenset([v]) for v in range(tree.n_nodes)}
    frontier = set(found)
    for _ in range(max_size - 1):
        new = set()
        for c in frontier:
            for v in c:
                for u in nbrs[v] - c:
                    new.add(c | {u})
        new -= found
        found |= new
        frontier = new
    return found


def dense_pattern(patterns, weights, members, d: int) -> np.ndarray:
    n = len(patterns[0])
    w = np.zeros((d,) * n)
    for r in members:
        w[tuple(patterns[r])] += weights[r]
    return w


def _project_sites(t: np.ndarray, sites, basis: np.ndarray) -> np.ndarray:
    """Project the axes `sites` of t onto span(columns of basis) (basis over the grouped sites)."""
    n = t.ndim
    rest = [i for i in range(n) if i not in sites]
    perm = list(sites) + rest
    m = np.transpose(t, perm).reshape(int(np.prod([t.shape[i] for i in sites])), -1)
    m = basis @ (basis.T @ m)
    out = m.reshape([t.shape[i] for i in perm])
    return np.transpose(out, np.argsort(perm))


def learned_subspace(w: np.ndarray, sites, tol=1e-10) -> np.ndarray:
    """Orthonormal basis of the span of w's matricization restricted to the given sites (rows)."""
    rest = [i for i in range(w.ndim) if i not in sites]
    m = np.transpose(w, list(sites) + rest).reshape(int(np.prod([w.shape[i] for i in sites])), -1)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return u[:, :0]
    return u[:, s > tol * s[0]]


def route_rule(tree: Tree, w_saddle: np.ndarray, w_star: np.ndarray, max_size: int | None = None,
               tol: float = 1e-9):
    """k* = smallest connected core set C whose local residual R_C is non-zero.

    R_C is the residual R = W* - W projected, on every subtree hanging off C, onto the subspace
    that the (fixed) outside cores already represent (the learned bond subspace of the saddle). A
    route on C (new entries inside C, new bond indices on its internal edges) changes W only inside
    that space, so its leading term is <R_C, .> of degree |C|. k* = 1 means a non-zero gradient.
    Returns (k*, list of minimal C with their ||R_C|| / ||R||)."""
    R = w_star - w_saddle
    nR = np.linalg.norm(R)
    if nR == 0:
        return 0, []
    max_size = max_size or tree.n_nodes
    subsets = sorted(connected_subsets(tree, max_size), key=len)
    best, hits = None, []
    for C in subsets:
        if best is not None and len(C) > best:
            break
        rc = R
        for v in tree.edges:
            p = tree.parent[v]
            if (v in C) == (p in C):
                continue
            far = list(tree.subtree_sites[v]) if v not in C else \
                [i for i in range(tree.n_sites) if i not in tree.subtree_sites[v]]
            far = sorted(far)
            rc = _project_sites(rc, far, learned_subspace(w_saddle, far))
        norm = np.linalg.norm(rc) / nR
        if norm > tol:
            best = len(C)
            hits.append((sorted(C), norm))
    return best, hits


# ---- whole-sequence prediction: route rule at rank-constrained optima (theory_saddles.md 5.1) ----

def constrained_optimum(tree: Tree, profile: dict, w_star: np.ndarray, d: int, restarts: int = 6,
                        seed: int = 0):
    """Best approximation of w_star by a TTN with bond dimensions `profile` (edge -> rank), found by
    L-BFGS from several random starts. Returns (dense W, loss = ||W - W*||^2 / 2)."""
    import torch
    from scipy.optimize import minimize

    from ttn.network import TTNSpec, to_dense

    bonds = tuple(1 if v == tree.root else int(profile[v]) for v in range(tree.n_nodes))
    spec = TTNSpec(tree, (d,) * tree.n_sites, bonds)
    target = torch.from_numpy(w_star)

    def f(x):
        xt = torch.from_numpy(x).requires_grad_(True)
        diff = to_dense(spec, spec.unflatten(xt)) - target
        L = 0.5 * (diff ** 2).sum()
        (g,) = torch.autograd.grad(L, xt)
        return L.item(), g.numpy()

    best = None
    for r in range(restarts):
        x0 = spec.flatten(spec.init(1.0, seed=seed + r)).numpy()
        res = minimize(f, x0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=20000, gtol=1e-12, ftol=1e-15))
        if best is None or res.fun < best.fun:
            best = res
    w = to_dense(spec, spec.unflatten(torch.from_numpy(best.x))).detach().numpy()
    return w, float(best.fun)


def route_sequence(tree: Tree, patterns, weights, d: int = 2, chi_max: int = 3, max_steps: int = 10,
                   tol: float = 1e-5):
    """Predicted saddle sequence: start from bond rank 1 everywhere (after the first escape), take the
    rank-constrained optimum, find the minimal routes (route_rule), raise the rank of the internal
    edges of the minimal route (ties: largest |R_C|), repeat until W* is fitted.
    Returns a list of dicts (profile, loss, k_star, routes)."""
    w_star = dense_pattern(patterns, weights, range(len(weights)), d)
    teacher_rank = {v: edge_rank(tree, patterns, weights, list(range(len(weights))), v)
                    for v in tree.edges}
    profile = {v: 1 for v in tree.edges}
    steps = []
    for _ in range(max_steps):
        w, loss = constrained_optimum(tree, profile, w_star, d)
        if loss < 1e-12:
            steps.append(dict(profile=dict(profile), loss=loss, k_star=0, routes=[]))
            break
        k, hits = route_rule(tree, w, w_star, tol=tol)
        steps.append(dict(profile=dict(profile), loss=loss, k_star=k, routes=hits))
        C = set(max(hits, key=lambda h: h[1])[0])
        for v in tree.edges:
            if v in C and tree.parent[v] in C:
                profile[v] = min(profile[v] + 1, chi_max, teacher_rank[v])
    return steps
