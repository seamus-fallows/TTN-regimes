"""Cells, predictions and the cancellation-free full-space model of the route-escape experiment.

A cell is (pattern target, learned set J). The network starts at the balanced partial-target point
theta_J (W = sum over J of s_a U_a) on the block-aligned manifold M* of `ttn.routes`, with every bond
max(number of star blocks) + 2 wide (spare channels).

Cancellation-free loss (as experiments/saddle_dyn/sdyn.py). With reference cores S, W(S) is exactly
sum_a P_a(S) U_a with P_a(S) = 0 for missing patterns (S has only active entries), R' = W* - W(S), and
Delta = W(G) - W(S) by the exact telescoping sum over cores, so E = W - W* = Delta - R' and
L - L_J = 1/2 ||Delta||^2 - <R', Delta> + 1/2 sum over J of (s_a - P_a(S))^2 contain no O(1) terms
that cancel. The gradient is d<E, W(theta)>/d theta with E held fixed.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

from ttn.network import to_dense
from ttn.routes import AlignedManifold, RouteTarget
from ttn.tree import make_tree

torch.set_num_threads(1)
DT = torch.float64
U = 2.0**-53
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("ROUTE_ESCAPE_OUT", "/root/outputs/route-escape"))
CONFIG = yaml.safe_load((HERE / "configs" / "cells.yaml").read_text())


def pattern(s: str) -> tuple[int, ...]:
    return tuple(int(c) for c in s)


class Cell:
    """Target, learned set, aligned manifold, balanced theta_J and the route-theory predictions."""

    def __init__(self, name: str):
        cfg = CONFIG["cells"][name]
        self.name, self.cfg = name, cfg
        kind, n = cfg["tree"]
        self.tree_name = f"{kind}{n}"
        self.tree = make_tree(kind, n)
        self.d = int(cfg["d"])
        self.pattern_strings = list(cfg["patterns"])
        self.patterns = [pattern(p) for p in self.pattern_strings]
        self.weights = np.array(cfg["weights"], dtype=float)
        self.J = list(cfg["J"])
        self.target = RouteTarget(self.tree, self.patterns, self.weights)
        if not self.target.is_star:
            raise ValueError(f"{name} violates the star condition")
        self.chi = max(self.target.n_blocks.values()) + CONFIG["chi_extra"]
        self.man = AlignedManifold(self.target, self.d, self.chi)
        self.spec = self.man.spec
        self.missing = self.target.missing(self.J)
        g = self.man.partial_target(self.J, relax_from=CONFIG["relax_from"])
        self.g_J = g
        self.theta_J = self.man.theta(g)
        self.active = self.man.used_by(self.J)  # per M* entry
        self.dormant = ~self.active
        # flat masks over all parameters
        self.mask = self.man.mask.numpy()
        self.active_flat = np.zeros(self.spec.n_params, dtype=bool)
        self.active_flat[self.man.flat[self.active]] = True
        self.dormant_flat = np.zeros(self.spec.n_params, dtype=bool)  # dormant M* entries
        self.dormant_flat[self.man.flat[self.dormant]] = True
        self.L_J = self.target.partial_loss(self.J)

    # ---- predictions ---------------------------------------------------------------------------

    def depth(self, r: int) -> int:
        return self.target.depth(self.J, r)

    @property
    def c_star(self) -> int:
        return self.target.min_depth(self.J)

    def A(self, r: int) -> float:
        """A_r(J): product of the active entries on the route of r at theta_J (eq. C.3)."""
        idx = [k for k in self.man.route[r] if self.active[k]]
        return float(np.prod(self.g_J[idx]))

    def route_dormant(self, r: int) -> list[int]:
        """M* entry indices of the dormant factors of r (one per node of R_J(r), Prop. 8.2)."""
        return [int(k) for k in self.man.route[r] if self.dormant[k]]

    def pairwise(self, r: int) -> int:
        return min(self.target.pair_cost(a, r) for a in self.J)

    def greedy(self) -> int:
        """Appendix C rule: smallest c_J(r), ties to the larger s_r A_r(J) (at the balanced theta_J)."""
        return min(self.missing, key=lambda r: (self.depth(r), -self.weights[r] * self.A(r)))

    def thm92(self) -> dict | None:
        """Theorem 9.2 formula: min over parents a of min(k*_a, 2 l*_a), residuals assigned by
        `parents` in the config (default: the learned pattern with the smallest pair cost)."""
        parents = self.cfg.get("parents")
        if parents is None:
            parents = {r: min(self.J, key=lambda a: self.target.pair_cost(a, r)) for r in self.missing}
        groups: dict[int, list[int]] = {}
        for r, a in parents.items():
            groups.setdefault(int(a), []).append(int(r))
        per = {}
        for a, rs in groups.items():
            o = self.target.one_component_order(a, residuals=rs)
            per[a] = dict(residuals=rs, k_star=o["k_star"], l_star=o["l_star"], mu=o["mu"],
                          order=o["order"], mechanism=o["mechanism"])
        return dict(per_parent=per, order=min(p["order"] for p in per.values()))

    def describe_missing(self) -> list[dict]:
        rows = []
        for r in self.missing:
            rows.append(dict(
                r=r, pattern=self.pattern_strings[r], weight=float(self.weights[r]),
                depth=self.depth(r), region=sorted(self.target.region(self.J, r)),
                pairwise=self.pairwise(r), A=self.A(r), sA=float(self.weights[r] * self.A(r)),
                n_shared_dormant=int(sum(
                    any(k in self.man.route[q] for q in self.missing if q != r)
                    for k in self.route_dormant(r))),
            ))
        return rows


@lru_cache(maxsize=None)
def get_cell(name: str) -> Cell:
    return Cell(name)


def dyn_cells() -> list[str]:
    return [n for n, c in CONFIG["cells"].items() if c["role"] == "dyn"]


def all_cells() -> list[str]:
    return list(CONFIG["cells"])


# ---- full-space model ------------------------------------------------------------------------


class FullModel:
    """Exact population loss 1/2 ||W - W*||^2 of the cell's TTN, evaluated cancellation-free about a
    reference point (see module docstring)."""

    def __init__(self, cell: Cell):
        self.cell = cell
        self.spec = cell.spec
        self.K = self.spec.n_cores
        self.P = self.spec.n_params
        shape = (cell.d,) * cell.tree.n_sites
        self.W_star = torch.zeros(shape, dtype=DT)
        for p, w in zip(cell.patterns, cell.weights):
            self.W_star[p] = w
        self.target_mask = torch.zeros(shape, dtype=torch.bool)
        for p in cell.patterns:
            self.target_mask[p] = True
        self.idx = [tuple(p) for p in cell.patterns]
        self.active_flat = torch.from_numpy(cell.active_flat)
        self.set_ref(cell.theta_J)

    def cores(self, theta: torch.Tensor):
        return self.spec.unflatten(theta)

    def W(self, theta: torch.Tensor) -> torch.Tensor:
        return to_dense(self.spec, self.cores(theta))

    def active_part(self, theta: torch.Tensor) -> torch.Tensor:
        """theta restricted to the active entries of theta_J (a point of M* with W in span over J)."""
        out = torch.zeros_like(theta)
        out[self.active_flat] = theta[self.active_flat]
        return out

    def set_ref(self, ref: torch.Tensor):
        self.ref = ref.detach().clone()
        self.ref_cores = self.cores(self.ref)
        WS = self.W(self.ref)
        off = WS.masked_fill(self.target_mask, 0.0)
        if float(off.abs().max()) != 0.0:
            raise ValueError("reference is not on the active part of M*")
        self.WS = WS
        self.Rp = self.W_star - WS
        self.P_ref = torch.stack([WS[i] for i in self.idx])
        J = self.cell.J
        self.c0 = 0.5 * float(((torch.as_tensor(self.cell.weights[J]) - self.P_ref[J]) ** 2).sum())

    def delta(self, theta: torch.Tensor) -> torch.Tensor:
        G, S = self.cores(theta), self.ref_cores
        out = torch.zeros_like(self.W_star)
        for j in range(self.K):
            ops = list(S[:j]) + [G[j] - S[j]] + list(G[j + 1 :])
            out = out + to_dense(self.spec, ops)
        return out

    def delta_abs(self, theta: torch.Tensor) -> torch.Tensor:
        G, S = self.cores(theta), self.ref_cores
        out = torch.zeros_like(self.W_star)
        for j in range(self.K):
            ops = [s.abs() for s in S[:j]] + [(G[j] - S[j]).abs()] + [g.abs() for g in G[j + 1 :]]
            out = out + to_dense(self.spec, ops)
        return out

    def residual(self, theta: torch.Tensor):
        """E = W - W*, L - L_J, and Delta."""
        D = self.delta(theta)
        dL = 0.5 * float((D * D).sum()) - float((self.Rp * D).sum()) + self.c0
        return D - self.Rp, dL, D

    def amplitudes_from_delta(self, D: torch.Tensor) -> np.ndarray:
        """P_a = <U_a, W> = P_a(S) + Delta[a] (exact for missing patterns: P_r(S) = 0)."""
        return np.array([float(self.P_ref[k] + D[i]) for k, i in enumerate(self.idx)])

    def off_target_mass(self, theta: torch.Tensor, D: torch.Tensor | None = None) -> float:
        """||W off the target strings||^2 / ||W||^2, from Delta (W(S) is zero off-target)."""
        if D is None:
            D = self.delta(theta)
        off = float((D.masked_fill(self.target_mask, 0.0) ** 2).sum())
        W = self.WS + D
        return off / float((W * W).sum())

    def grad_from_E(self, theta: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        th = theta.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad((E * self.W(th)).sum(), th)
        return g

    def grad(self, theta: torch.Tensor):
        E, dL, D = self.residual(theta)
        return self.grad_from_E(theta, E), dL, D

    def grad_err_bound(self, theta: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Upper bound 10 u x (gradient contraction with absolute values) on the roundoff of the gradient."""
        A = E.abs() + self.delta_abs(theta) + self.Rp.abs()
        th = theta.abs().detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad((A * self.W(th)).sum(), th)
        return 10 * U * g

    def hessian(self, theta: torch.Tensor) -> torch.Tensor:
        """Hessian of the plain loss (used as the LSODA Jacobian; roundoff there only affects Newton
        convergence, not the solution)."""

        def f(th):
            E = self.W(th) - self.W_star
            return 0.5 * (E * E).sum()

        return torch.func.hessian(f)(theta.detach())


# ---- reduced model on M* ---------------------------------------------------------------------


class ReducedModel:
    """Gradient flow on M* in the entries g (Theorem 7.7), with L - L_J free of cancellation."""

    def __init__(self, cell: Cell):
        self.cell = cell
        self.man = cell.man
        self.w = cell.weights
        self.J = np.array(cell.J)
        self.miss = np.array(cell.missing)

    def dL(self, g: np.ndarray) -> float:
        P = self.man.amplitudes(g)
        return float(0.5 * ((P[self.J] - self.w[self.J]) ** 2).sum()
                     + (0.5 * P[self.miss] ** 2 - self.w[self.miss] * P[self.miss]).sum())

    def velocity(self, g):
        return self.man.velocity(g)

    def jacobian(self, g):
        return self.man.jacobian(g)


# ---- isolated-route prediction (eqs. 6.14-6.16 with unbalanced starts) -----------------------


def isolated_route_time(b0: np.ndarray, gamma: float, s: float, thr: float = 1e-3) -> float:
    """Escape time of one isolated route y = gamma prod(b), db_v/dt = gamma (s - y) prod_{u != v} b_u
    (eq. 6.14 with unequal factors; Prop. 6.2: b_v^2 - b_u^2 is conserved), from b(0) = b0 to the
    first time -s y + y^2 / 2 < -thr (the escape criterion L < L_J - thr restricted to this route).

    In the smallest-|b| coordinate x the other factors are +-sqrt(x^2 + c_v); with x~ = sign x (sign =
    product of the other factors' signs) y = gamma x~ prod sqrt(x~^2 + c_v) and
    dx~/dt = gamma (s - y) prod sqrt(x~^2 + c_v), so t = int dx~ / [gamma (s - y) prod sqrt(.)] from
    x~(0) to y = y_thr (an ODE quadrature in x~; no event location needed). inf if the route is dead
    (x~(0) <= 0 and some c_v = 0: two factors of equal size and a negative product)."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    b0 = np.asarray(b0, dtype=float)
    k = int(np.argmin(np.abs(b0)))
    others = np.delete(b0, k)
    c = others**2 - b0[k] ** 2
    sign = float(np.prod(np.sign(others))) if len(others) else 1.0
    x0 = sign * b0[k]
    if x0 <= 0 and len(c) and min(c) <= 0:
        return math.inf
    y_thr = s - math.sqrt(s * s - 2 * thr)  # -s y + y^2/2 = -thr

    def y_of(x):
        return gamma * x * float(np.prod(np.sqrt(x * x + c)))

    hi = max(abs(x0), 1e-300)
    while y_of(hi) < y_thr:
        hi *= 2
    x_thr = brentq(lambda x: y_of(x) - y_thr, -1e-300 if x0 > 0 else x0, hi, xtol=1e-300, rtol=1e-15)
    if x0 >= x_thr:
        return 0.0

    def rhs(x, t):
        return [1.0 / (gamma * (s - y_of(x)) * float(np.prod(np.sqrt(x * x + c))))]

    sol = solve_ivp(rhs, (x0, x_thr), [0.0], method="DOP853", rtol=1e-12, atol=1e-30)
    return float(sol.y[0, -1])


def leading_616(eps: float, c: int, gamma: float, s: float) -> float:
    """Eq. 6.16 from balanced b(0) = eps: (gamma s)^-1 log(1/eps) (c = 2), eps^-(c-2) / ((c-2) gamma s)."""
    if c == 2:
        return math.log(1.0 / eps) / (gamma * s)
    return eps ** (-(c - 2)) / ((c - 2) * gamma * s)
