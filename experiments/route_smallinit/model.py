"""TTN on a pattern target with a dense residual: W(theta) is contracted in full (at most 3^8
entries here), so E = W - W* is formed entry by entry. Entries off the target support carry no
cancellation at all; at a learned pattern the cancellation is W[a] - s_a only (the library's
population_loss forms ||W||^2 - 2<W, W*> + ||W*||^2 instead, whose gradient is a difference of O(1)
vectors).

Bond e has dimension blocks(e) + spare, capped at the largest rank the cut allows (brief, sec. 3).
"""

from __future__ import annotations

import string
from functools import cached_property

import numpy as np
import opt_einsum
import torch

from ttn.network import TTNSpec
from ttn.routes import AlignedManifold, RouteTarget
from ttn.tree import make_tree

torch.set_num_threads(1)
DT = torch.float64
U = 2.0**-53


def build_target(cfg: dict) -> RouteTarget:
    tree = make_tree(cfg["tree"], int(cfg["n_sites"]))
    pats = [[int(c) for c in p] for p in cfg["patterns"]]
    return RouteTarget(tree, pats, [float(w) for w in cfg["weights"]])


class SpareManifold(AlignedManifold):
    """AlignedManifold whose bond e has dimension min(blocks(e) + spare, max rank of the cut).
    (The library class takes one chi for all edges; the brief asks for blocks + 2 per edge.)"""

    def __init__(self, target: RouteTarget, d: int, spare: int = 2):
        tree = target.tree
        bonds = [1] * tree.n_nodes
        for e in tree.edges:
            below = len(tree.subtree_sites[e])
            cap = min(d**below, d ** (tree.n_sites - below))
            bonds[e] = max(target.n_blocks[e], min(target.n_blocks[e] + spare, cap))
        # build the parent with the largest chi, then rebuild the index maps on the per-edge spec
        super().__init__(target, d, chi=None)
        self.spec = TTNSpec(tree, (d,) * tree.n_sites, tuple(bonds))
        slices = self.spec.core_slices()
        self.flat = np.array(
            [
                slices[v].start + int(np.ravel_multi_index(idx, self.spec.shapes[v]))
                for v, idx in self.entries
            ]
        )
        self.__dict__.pop("mask", None)


def dense_expression(spec: TTNSpec):
    """opt_einsum expression: cores -> dense W with axes in site order."""
    tree = spec.tree
    letters = iter(string.ascii_letters)
    site = [next(letters) for _ in range(tree.n_sites)]
    bond = {e: next(letters) for e in tree.edges}
    root_leg = next(letters)
    terms = []
    for v in range(tree.n_nodes):
        legs = [site[s] for s in tree.sites[v]] + [bond[c] for c in tree.children[v]]
        legs.append(bond[v] if v != tree.root else root_leg)
        terms.append("".join(legs))
    eq = ",".join(terms) + "->" + "".join(site)
    return opt_einsum.contract_expression(eq, *spec.shapes, optimize="optimal")


class Model:
    def __init__(self, cfg: dict, spare: int = 2, arith: str = "dense"):
        self.cfg = cfg
        self.arith = arith  # "dense" (E entrywise) or "lib" (library norm-form gradient: floor check)
        self.target = target = build_target(cfg)
        self.d = d = int(cfg["d"])
        self.man = SpareManifold(target, d, spare)
        self.spec = spec = self.man.spec
        self.P = spec.n_params
        self.expr = dense_expression(spec)
        self.idx = [tuple(p) for p in target.patterns]
        self.s = np.array(target.weights)
        Ws = torch.zeros((d,) * target.tree.n_sites, dtype=DT)
        for p, w in zip(self.idx, target.weights):
            Ws[p] = w
        self.W_star = Ws
        self.norm_star = float(Ws.norm())
        if arith == "lib":
            from ttn.dynamics import Problem

            self._problem = Problem(spec, self.man.teacher)
        self.flat_idx = torch.tensor(
            [int(np.ravel_multi_index(p, Ws.shape)) for p in self.idx], dtype=torch.long
        )

    # ---- tensors ----------------------------------------------------------------------------

    def W(self, theta: torch.Tensor) -> torch.Tensor:
        return self.expr(*self.spec.unflatten(theta), backend="torch")

    def evaluate(self, theta: torch.Tensor):
        """(W, E, velocity = -grad L) with E = W - W* entrywise."""
        th = theta.detach().clone().requires_grad_(True)
        W = self.W(th)
        E = W.detach() - self.W_star
        if self.arith == "lib":
            return W.detach(), E, self._problem.velocity(theta.detach())
        (g,) = torch.autograd.grad(W, th, grad_outputs=E)
        return W.detach(), E, -g

    def loss_terms(self, W: torch.Tensor, E: torch.Tensor):
        """L, amplitudes P_a, off-target mass fraction."""
        L = 0.5 * float((E * E).sum())
        flat = W.reshape(-1)
        P = flat[self.flat_idx].numpy().copy()
        tot = float((flat * flat).sum())
        off = max(tot - float((P * P).sum()), 0.0) / tot if tot > 0 else 0.0
        return L, P, off

    def hessian(self, theta: torch.Tensor) -> torch.Tensor:
        def f(th):
            E = self.W(th) - self.W_star
            return 0.5 * (E * E).sum()

        return torch.func.hessian(f)(theta)

    def grad_amplitude(self, theta: torch.Tensor, a: int) -> torch.Tensor:
        th = theta.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(self.W(th).reshape(-1)[self.flat_idx[a]], th)
        return g

    def grad_roundoff_bound(self, theta: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Entrywise upper bound (10 u x sum of |terms|) on the roundoff in the gradient."""
        th = theta.detach().abs().clone().requires_grad_(True)
        W_abs = self.W(th)
        A = W_abs.detach() + self.W_star.abs() + E.abs()
        (g,) = torch.autograd.grad(W_abs, th, grad_outputs=A)
        return 10 * U * g

    # ---- reading off the manifold structure -------------------------------------------------

    @cached_property
    def n_cores(self) -> int:
        return self.target.tree.n_nodes

    def cores(self, theta: torch.Tensor):
        return [c.clone() for c in self.spec.unflatten(theta.detach())]
