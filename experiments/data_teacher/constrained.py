"""Critical points of the bond-rank-constrained problem (predicted plateau states).

The saddle-to-saddle picture predicts that each plateau sits at a critical point of L restricted to
TTNs whose bond ranks equal the student's current effective ranks. For teachers without a closed
form (misaligned covariance, non-orthogonal CP, random TTN teachers, correlated inputs) we find
these numerically: minimize L over a TTN on the same tree with bond dimensions fixed to the ranks,
from many random starts (L-BFGS on the flat vector), and keep the distinct local minima.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

from ttn.network import TTNSpec, to_dense


def ranked_spec(spec: TTNSpec, ranks: dict[int, int]) -> TTNSpec:
    """Same tree and physical dims, bond dimension of edge v set to ranks[v]."""
    bonds = list(spec.bond_dims)
    for v, r in ranks.items():
        bonds[v] = r
    return TTNSpec(spec.tree, spec.phys_dims, tuple(bonds))


def constrained_minima(spec_r: TTNSpec, loss_of_dense, n_starts=20, seed=0, tol=1e-6,
                       gtol=1e-10, max_iter=5000):
    """Distinct local minima of loss_of_dense(W) over TTNs with spec_r's bond dims.

    loss_of_dense maps the dense tensor W (torch, float64) to the scalar loss. Returns a list of
    dicts (loss, W) sorted by loss; minima whose losses agree to `tol` (relative) are merged.
    """

    def fg(y):
        flat = torch.from_numpy(y).requires_grad_(True)
        loss = loss_of_dense(to_dense(spec_r, spec_r.unflatten(flat)))
        (g,) = torch.autograd.grad(loss, flat)
        return loss.item(), g.numpy().copy()

    found = []
    for k in range(n_starts):
        y0 = spec_r.flatten(spec_r.init(1.0, seed=seed + k)).numpy()
        res = minimize(fg, y0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, gtol=gtol, ftol=1e-15, maxcor=50))
        w = to_dense(spec_r, spec_r.unflatten(torch.from_numpy(res.x))).detach()
        found.append(dict(loss=res.fun, W=w, grad_norm=float(np.linalg.norm(res.jac))))
    found.sort(key=lambda f: f["loss"])
    distinct = []
    for f in found:
        if not distinct or all(abs(f["loss"] - d["loss"]) > tol * max(abs(d["loss"]), 1e-12)
                               for d in distinct):
            f["count"] = 1
            distinct.append(f)
        else:
            next(d for d in distinct
                 if abs(f["loss"] - d["loss"]) <= tol * max(abs(d["loss"]), 1e-12))["count"] += 1
    return distinct


def metric_loss_of_dense(w_star: torch.Tensor, metrics=None):
    """1/2 ||W - W*||_M^2 for a product metric (list of site matrices) or None (Frobenius)."""
    from ttn.network import apply_on_leg

    def loss(w):
        r = w - w_star
        mr = r
        if metrics is not None:
            for i, m in enumerate(metrics):
                mr = apply_on_leg(mr, m, i)
        return 0.5 * (r * mr).sum()

    return loss


def matrix_loss_of_dense(w_star: torch.Tensor, metric: torch.Tensor):
    """1/2 (w - w*)^T M (w - w*) with a full (d^N x d^N) metric (correlated inputs)."""

    def loss(w):
        r = (w - w_star).reshape(-1)
        return 0.5 * r @ (metric @ r)

    return loss
