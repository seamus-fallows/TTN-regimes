"""Numerical saddle order at a learned rank-one saddle theta_S (one learned component q).

m(t) = min_{||delta|| = t} L(theta_S + delta) - L_S is computed by L-BFGS on the sphere (random
restarts + continuation in t). The saddle order is the exponent p in |m(t)| ~ C t^p; the theory
(notes/theory_saddle_order.md) predicts p = k* = |Steiner(N(D_r))| minimized over residuals r,
provided the fixed-rank (old) block is positive definite modulo gauge (rho(S) < s_q).

Usage: uv run python experiments/theory_checks/saddle_order/saddle_order.py CASE [CASE ...]
       (CASE names in CASES below; "all" runs every case).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from ttn.network import TTNSpec, to_dense
from ttn.tree import make_tree

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUT = Path(__file__).parent / "results"
T_GRID = [0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025]


class Instance:
    """Teacher s_q e_q + sum_r s_r e_r (standard site basis), saddle theta_S at q = 0...0."""

    def __init__(self, tree_kind, n_sites, d, chi, s_q, residuals, R_dense=None):
        self.tree = make_tree(tree_kind, n_sites)
        self.spec = TTNSpec.uniform(self.tree, d, chi)
        self.s_q, self.residuals = s_q, residuals  # residuals: list of (pattern tuple, weight)
        K = self.tree.n_nodes
        self.a = s_q ** (1.0 / K)
        self.cores_S = []
        for shape in self.spec.shapes:
            c = torch.zeros(shape)
            c[(0,) * len(shape)] = self.a
            self.cores_S.append(c)
        self.theta_S = self.spec.flatten(self.cores_S)
        self.W_S = to_dense(self.spec, self.cores_S)
        self.R = torch.zeros_like(self.W_S)
        for pat, s in residuals:
            self.R[tuple(pat)] += s
        site_node = self.tree.site_node
        self.k_star = min(
            len(self.tree.steiner_nodes([i for i, x in enumerate(p) if x != 0]))
            for p, _ in residuals
        ) if residuals else math.inf
        self.terminals = [
            sorted({site_node[i] for i, x in enumerate(p) if x != 0}) for p, _ in residuals
        ]
        self.old_mask = self._old_mask()
        if R_dense is not None:  # general residual (must vanish on sectors of size <= 1)
            from gp_operator import core_grouped, k_star

            self.R = torch.as_tensor(np.asarray(R_dense), dtype=torch.float64)
            self.k_star = k_star(self.tree, core_grouped(self.tree, d, np.asarray(R_dense)))
            self.terminals = "general R"

    def _old_mask(self):
        """1 for old entries (all bond indices 0), 0 for new entries."""
        parts = []
        for v, shape in enumerate(self.spec.shapes):
            n_phys = len(self.tree.sites[v])
            m = torch.zeros(shape)
            idx = (slice(None),) * n_phys + (0,) * (len(shape) - n_phys)
            m[idx] = 1.0
            parts.append(m.reshape(-1))
        return torch.cat(parts)

    def excess(self, delta):
        """L(theta_S + delta) - L_S, cancellation-free."""
        cores = self.spec.unflatten(self.theta_S + delta)
        D = to_dense(self.spec, cores) - self.W_S
        return 0.5 * (D * D).sum() - (self.R * D).sum()

    def route_prediction(self, t):
        """Best single-route value -max_r s_r a^(K-k) (t^2/k)^(k/2) over residuals with k = k*."""
        K = self.tree.n_nodes
        best = 0.0
        for (pat, s), term in zip(self.residuals, self.terminals):
            k = len(self.tree.steiner_nodes([i for i, x in enumerate(pat) if x != 0]))
            if k == self.k_star:
                best = max(best, s * self.a ** (K - k) * (t * t / k) ** (k / 2))
        return -best


def sphere_min(inst, t, starts, scale, maxiter=3000):
    """Minimize excess(t u/|u|) from each start; return the best (value, unit vector)."""

    def fun(u_np):
        u = torch.from_numpy(u_np).requires_grad_(True)
        val = inst.excess(t * u / u.norm()) / scale
        (g,) = torch.autograd.grad(val, u)
        return val.item(), g.numpy()

    best = (math.inf, None)
    for u0 in starts:
        res = minimize(fun, u0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": 1e-12, "ftol": 1e-15, "maxcor": 30})
        u = res.x / np.linalg.norm(res.x)
        if res.fun < best[0]:
            best = (res.fun, u)
    return best[0] * scale, best[1]


def random_starts(inst, rng, n):
    """Full random, new-entries-only and old-entries-only Gaussian directions."""
    P = inst.theta_S.numel()
    old = inst.old_mask.numpy()
    out = []
    for i in range(n):
        u = rng.standard_normal(P)
        if i % 3 == 1:
            u = u * (1 - old)
        elif i % 3 == 2:
            u = u * old + 0.1 * u * (1 - old)
        out.append(u)
    return out


def run_case(name, spec_kw, n_rand=12, seed=0, t_grid=T_GRID):
    inst = Instance(**spec_kw)
    rng = np.random.default_rng(seed)
    rows, prev = [], None
    for t in t_grid:
        starts = random_starts(inst, rng, n_rand)
        if prev is not None:
            starts = [prev] + starts
        scale = t ** inst.k_star
        t0 = time.time()
        m, u = sphere_min(inst, t, starts, scale)
        prev = u
        old_frac = float(np.sum((u * inst.old_mask.numpy()) ** 2))
        rows.append({"t": t, "m": m, "m_over_tk": m / scale, "route_pred": inst.route_prediction(t),
                     "old_frac": old_frac, "secs": time.time() - t0})
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        b["slope"] = (math.log(abs(a["m"])) - math.log(abs(b["m"]))) / math.log(a["t"] / b["t"]) \
            if a["m"] < 0 and b["m"] < 0 else None
    out = {"name": name, "spec": {k: v for k, v in spec_kw.items()}, "k_star": inst.k_star,
           "terminals": inst.terminals, "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def show(out):
    print(f"== {out['name']}  k*={out['k_star']}  terminals={out['terminals']}")
    for r in out["rows"]:
        sl = r.get("slope")
        print(f"  t={r['t']:.3f}  m={r['m']:+.4e}  m/t^k={r['m_over_tk']:+.4f}  "
              f"route={r['route_pred']/r['t']**out['k_star']:+.4f}  slope={sl if sl is None else round(sl, 3)}  "
              f"old_frac={r['old_frac']:.2e}  {r['secs']:.0f}s")


def one(kind, n, pats, s_q=1.0, d=2, chi=2):
    """Case spec: residual patterns given as strings ('1001') with weights."""
    return dict(tree_kind=kind, n_sites=n, d=d, chi=chi, s_q=s_q,
                residuals=[(tuple(int(c) for c in p), s) for p, s in pats])


CASES = {
    # single residual, k* = 2..5
    "mps4_k2": one("mps", 4, [("1100", 0.5)]),
    "mps4_k3": one("mps", 4, [("1010", 0.5)]),
    "mps4_k4": one("mps", 4, [("1001", 0.5)]),
    "mps5_k5": one("mps", 5, [("10001", 0.5)]),
    "mps5_k4": one("mps", 5, [("10010", 0.5)]),
    "binary4_k3": one("binary", 4, [("1010", 0.5)]),
    "binary6_k4": one("binary", 6, [("100001", 0.5)]),
    "binary6_k5": one("binary", 6, [("100100", 0.5)]),
    "ht4_k3": one("ht", 4, [("1100", 0.5)]),
    "ht4_k5": one("ht", 4, [("1010", 0.5)]),
    "mps5_m3_k5": one("mps", 5, [("10101", 0.5)]),
    "mps4_k4_chi3": one("mps", 4, [("1001", 0.5)], chi=3),
    "mps4_k4_d3": one("mps", 4, [("1001", 0.5)], d=3),
    # s_r close to / equal to s_q (piggyback block nearly / exactly degenerate)
    "mps4_k4_sr09": one("mps", 4, [("1001", 0.9)]),
    "mps4_k4_sr10": one("mps", 4, [("1001", 1.0)]),
    "mps5_k5_sr10": one("mps", 5, [("10001", 1.0)]),
    # several residuals
    "mps4_star_sub": one("mps", 4, [("1010", 0.6), ("1001", 0.6)]),
    "mps5_two_k4k5": one("mps", 5, [("10010", 0.5), ("10001", 0.4)]),
    "mps5_shared_blocks": one("mps", 5, [("10001", 0.5), ("10011", 0.4), ("11001", 0.4)]),
    # generalized piggyback (pair node {0,1} shared by two m=3 residuals): predicted order 4 < k*=5
    "mps6_pairstar_08": one("mps", 6, [("110001", 0.8), ("110010", 0.8)]),
    "mps6_pairstar_06": one("mps", 6, [("110001", 0.6), ("110010", 0.6)]),  # control: rho<s_q
    # single residual, m=3, wrong order: split {0,1}|{5} is low -> predicted order 4 < k*=6
    "mps6_m3_wrong": one("mps", 6, [("110001", 1.5)]),
    "mps6_m3_right": one("mps", 6, [("110001", 0.5)]),  # control: order 6
    # single residual, m=3, wrong order, no low split -> predicted order k*=5
    "mps5_m3_wrong": one("mps", 5, [("10101", 1.5)]),
    "mps5_m2_wrong": one("mps", 5, [("10001", 1.5)]),  # m=2 wrong order: strict, order 2
    # HT4, all four leaves differ (k*=7); split {L0,L1}|{L2,L3} has degrees 3, 3 < 3.5
    "ht4_1111_wrong": one("ht", 4, [("1111", 1.5)]),  # predicted order 6
    "ht4_1111_right": one("ht", 4, [("1111", 0.5)]),  # predicted order 7
}
T_GRID_HIGH = [0.4, 0.3, 0.2, 0.14, 0.1, 0.07, 0.05]  # k* = 7: t^7 reaches the float64 floor sooner


if __name__ == "__main__":
    names = list(CASES) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for nm in names:
        grid = T_GRID_HIGH if nm.startswith("ht4_1111") else T_GRID
        show(run_case(nm, CASES[nm], t_grid=grid))
