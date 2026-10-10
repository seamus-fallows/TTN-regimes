"""Independent re-implementation for verifying notes/theory_saddle_order.md.

Uses ttn.tree only for the tree topologies (so that mps/binary/ht mean the same trees as in the
note); the contraction, teachers, saddle point, gauge directions and losses are written here from
scratch and do not import the theory-check code in ../saddle_order/.
Core convention: G_v has legs (own sites..., child bonds..., parent bond); root bond has size 1.
"""
import math

import numpy as np
import torch

from ttn.tree import make_tree

torch.set_num_threads(1)
DT = torch.float64
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Net:
    def __init__(self, kind, n, d=2, chi=2):
        t = make_tree(kind, n)
        self.kind, self.n, self.d, self.tree = kind, n, d, t
        K = t.n_nodes
        self.K = K
        self.parent = list(t.parent)
        self.sites = [tuple(s) for s in t.sites]
        self.children = [[c for c in range(K) if self.parent[c] == v] for v in range(K)]
        self.bond = [1] * K  # bond[v] = dim of edge (v, parent(v))
        for v in range(K):
            if self.parent[v] >= 0:
                below = len(t.subtree_sites[v])
                self.bond[v] = min(chi, d**below, d ** (n - below))
        self.shapes = [
            tuple([d] * len(self.sites[v]) + [self.bond[c] for c in self.children[v]] + [self.bond[v]])
            for v in range(K)
        ]
        self.n_phys = [len(self.sites[v]) for v in range(K)]
        self.sizes = [math.prod(s) for s in self.shapes]
        self.off = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        self.P = int(self.off[-1])
        site_l = LETTERS[:n]
        edge_l = LETTERS[n : n + K]
        ops = [
            "".join(site_l[s] for s in self.sites[v]) + "".join(edge_l[c] for c in self.children[v]) + edge_l[v]
            for v in range(K)
        ]
        self.eq = ",".join(ops) + "->" + site_l + edge_l[t.root]
        self.root = t.root

    def cores(self, theta):
        return [theta[self.off[v] : self.off[v + 1]].view(self.shapes[v]) for v in range(self.K)]

    def dense(self, theta):
        return torch.einsum(self.eq, *self.cores(theta))[..., 0]

    # ---- bookkeeping of entries -------------------------------------------------------
    def entry_info(self):
        """Per flat parameter: (core, is_old, physical multi-index)."""
        core, old, phys = [], [], []
        for v in range(self.K):
            for idx in np.ndindex(*self.shapes[v]):
                core.append(v)
                old.append(all(i == 0 for i in idx[self.n_phys[v] :]))
                phys.append(idx[: self.n_phys[v]])
        return np.array(core), np.array(old), phys

    def site_core(self):
        out = {}
        for v in range(self.K):
            for s in self.sites[v]:
                out[s] = v
        return out

    def steiner(self, cores):
        """Cores of the smallest subtree containing `cores` (union of pairwise paths)."""
        cores = list(cores)
        if len(cores) <= 1:
            return set(cores)

        def path_to_root(v):
            p = [v]
            while self.parent[p[-1]] >= 0:
                p.append(self.parent[p[-1]])
            return p

        out = set()
        for i in range(len(cores)):
            for j in range(i + 1, len(cores)):
                a, b = path_to_root(cores[i]), path_to_root(cores[j])
                common = set(a) & set(b)
                out |= {x for x in a if x not in common} | {x for x in b if x not in common}
                lca = next(x for x in a if x in common)
                out.add(lca)
        return out


class PatternSaddle:
    """Pattern teacher W* = s_q e_q + sum_r s_r e_r (site basis, q = 0...0) and its saddle theta_S."""

    def __init__(self, net, residuals, s_q=1.0):
        self.net, self.s_q = net, s_q
        n, d = net.n, net.d
        self.q = (0,) * n
        self.residuals = [(tuple(r), float(w)) for r, w in residuals]
        ws = torch.zeros((d,) * n, dtype=DT)
        ws[self.q] = s_q
        wstar = ws.clone()
        for r, w in self.residuals:
            wstar[r] += w
        self.W_S, self.W_star, self.R = ws, wstar, wstar - ws
        a = s_q ** (1.0 / net.K)
        theta = torch.zeros(net.P, dtype=DT)
        cores = net.cores(theta)
        for v in range(net.K):
            idx = tuple(self.q[s] for s in net.sites[v]) + (0,) * (len(net.shapes[v]) - net.n_phys[v])
            cores[v][idx] = a
        self.theta_S = theta
        self.a = a
        sc = net.site_core()
        self.terminals = [sorted({sc[s] for s in range(n) if r[s] != self.q[s]}) for r, _ in self.residuals]
        self.k_r = [len(net.steiner(T)) for T in self.terminals]
        self.k_star = min(self.k_r)
        self.L_S = 0.5 * float((self.R**2).sum())

    def dL(self, theta):
        """L(theta) - L_S without cancellation: 1/2 |Delta|^2 - <R, Delta>, Delta = W - W_S."""
        delta = self.net.dense(theta) - self.W_S
        return 0.5 * (delta**2).sum() - (self.R * delta).sum()

    def loss(self, theta):
        return 0.5 * ((self.net.dense(theta) - self.W_star) ** 2).sum()


def gauge_basis(net, theta):
    """Orthonormal basis (columns) of the gauge-orbit tangent space at theta."""
    cores = [c.detach().numpy() for c in net.cores(theta)]
    cols = []
    for v in range(net.K):
        p = net.parent[v]
        if p < 0:
            continue
        chi = net.bond[v]
        leg_p = net.n_phys[p] + net.children[p].index(v)
        for i in range(chi):
            for j in range(chi):
                dc = [np.zeros_like(c) for c in cores]
                dc[v][..., j] = cores[v][..., i]
                src = np.take(cores[p], j, axis=leg_p)
                sl = [slice(None)] * cores[p].ndim
                sl[leg_p] = i
                dc[p][tuple(sl)] -= src
                cols.append(np.concatenate([x.reshape(-1) for x in dc]))
    G = np.array(cols).T
    u, s, _ = np.linalg.svd(G, full_matrices=False)
    rank = int((s > 1e-10 * max(1.0, s[0])).sum())
    return u[:, :rank]
