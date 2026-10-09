"""Gauge operations on TTN cores: canonical forms (QR sweeps), environment Gram matrices, HSVD.

All functions take and return lists of cores in the layout of `ttn.network` (physical legs, child
legs, parent leg).
"""

from __future__ import annotations

import math

import torch

from ttn.network import TTNSpec, apply_on_leg


def leg_to(spec: TTNSpec, v: int, u: int) -> int:
    """Index of core v's leg that points to the neighbouring node u."""
    t = spec.tree
    if t.parent[v] == u:
        return len(spec.shapes[v]) - 1
    return len(t.sites[v]) + t.children[v].index(u)


def _qr_on_leg(core: torch.Tensor, leg: int) -> tuple[torch.Tensor, torch.Tensor]:
    """core = Q R with Q an isometry from the other legs onto `leg` (same shape as core)."""
    moved = torch.movedim(core, leg, -1)
    shape = moved.shape
    if math.prod(shape[:-1]) < shape[-1]:
        raise ValueError("bond larger than the rest of the core; cap the bond dimensions")
    q, r = torch.linalg.qr(moved.reshape(-1, shape[-1]))
    return torch.movedim(q.reshape(shape), -1, leg), r


def move_center(spec: TTNSpec, cores: list[torch.Tensor], v: int, u: int) -> list[torch.Tensor]:
    """Make core v an isometry towards its neighbour u and absorb the R factor into u."""
    cores = list(cores)
    q, r = _qr_on_leg(cores[v], leg_to(spec, v, u))
    cores[v] = q
    cores[u] = apply_on_leg(cores[u], r, leg_to(spec, u, v))
    return cores


def canonicalize(spec: TTNSpec, cores: list[torch.Tensor], center: int | None = None):
    """Orthogonal canonical form with orthogonality centre `center` (default: the root).

    Every other core becomes an isometry from its legs pointing away from the centre onto its leg
    pointing towards it; W is unchanged and all of its norm sits in the centre core."""
    t = spec.tree
    center = t.root if center is None else center
    dist = {v: len(t.path(v, center)) - 1 for v in range(t.n_nodes)}
    for v in sorted(range(t.n_nodes), key=lambda w: -dist[w]):
        if v != center:
            cores = move_center(spec, cores, v, t.path(v, center)[1])
    return cores


def euler_tour(spec: TTNSpec) -> list[int]:
    """Nodes visited by a depth-first walk from the root and back (consecutive nodes adjacent)."""
    t = spec.tree
    tour = []

    def visit(v):
        tour.append(v)
        for c in t.children[v]:
            visit(c)
            tour.append(v)

    visit(t.root)
    return tour


def env_grams(spec: TTNSpec, cores: list[torch.Tensor]):
    """below[v] = Phi_v^T Phi_v and above[v] = Psi_v^T Psi_v for every edge v (whitened inputs).

    Phi_v is the subtree tensor under edge v and Psi_v its environment; the Hessian of the loss in
    core v is the Kronecker product of below[c] over v's children, I on v's physical legs and
    above[v] on its parent leg."""
    t = spec.tree
    below: list[torch.Tensor | None] = [None] * t.n_nodes
    for v in t.postorder:
        g = cores[v]
        g2 = g
        for k, c in enumerate(t.children[v]):
            g2 = apply_on_leg(g2, below[c], len(t.sites[v]) + k)
        chi = g.shape[-1]
        below[v] = g.reshape(-1, chi).T @ g2.reshape(-1, chi)
    above: list[torch.Tensor | None] = [None] * t.n_nodes
    above[t.root] = torch.ones(1, 1, dtype=cores[0].dtype)
    for p in t.preorder:
        n_phys = len(t.sites[p])
        h = apply_on_leg(cores[p], above[p], cores[p].ndim - 1)
        for k, v in enumerate(t.children[p]):
            hv = h
            for k2, c in enumerate(t.children[p]):
                if c != v:
                    hv = apply_on_leg(hv, below[c], n_phys + k2)
            leg = n_phys + k
            gp = torch.movedim(cores[p], leg, 0).reshape(cores[p].shape[leg], -1)
            hp = torch.movedim(hv, leg, 0).reshape(cores[p].shape[leg], -1)
            above[v] = gp @ hp.T
    return below, above


def hsvd(spec: TTNSpec, w: torch.Tensor) -> list[torch.Tensor]:
    """Leaves-to-root hierarchical SVD of a dense tensor (sites in order 0..N-1) truncated to the
    spec's bond dimensions. Returns cores in root-centred canonical form (Grasedyck 2010, quasi-
    optimal within (2 + sqrt 2) sqrt(N) of the best approximation)."""
    t = spec.tree
    # current tensor: its axes are labelled by ('s', site) or ('b', node) for compressed subtrees
    cur = w
    labels = [("s", s) for s in range(t.n_sites)]
    cores: list[torch.Tensor | None] = [None] * t.n_nodes
    for v in t.postorder:
        own = [("s", s) for s in t.sites[v]] + [("b", c) for c in t.children[v]]
        idx = [labels.index(lab) for lab in own]
        rest = [i for i in range(len(labels)) if i not in idx]
        perm = cur.permute(idx + rest)
        own_shape = perm.shape[: len(idx)]
        m = perm.reshape(math.prod(own_shape), -1)
        if v == t.root:
            cores[v] = m.reshape(*own_shape, 1)
            break
        chi = spec.bond_dims[v]
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        u = u[:, :chi]
        if u.shape[1] < chi:  # pad with zero columns (bond larger than the matricization rank)
            u = torch.cat([u, u.new_zeros(u.shape[0], chi - u.shape[1])], dim=1)
        cores[v] = u.reshape(*own_shape, chi)
        rest_shape = perm.shape[len(idx) :]
        cur = (u.T @ m).reshape(chi, *rest_shape)
        labels = [("b", v)] + [labels[i] for i in rest]
    return cores
