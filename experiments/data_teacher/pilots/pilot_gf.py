"""Throwaway pilot: gradient flow with scipy solve_ivp on the flat parameter vector.

Used only until ttn/dynamics.py lands (brief: "keep pilot code minimal"). Records loss, dual
component overlaps (a_r = 1 at W = W*, also for non-orthogonal CP teachers) and bond spectra on a
log-spaced time grid.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ttn.losses import population_loss
from ttn.network import apply_on_leg, bond_spectra, conserved_quantities, forward, to_dense


def dual_overlaps(spec, cores, teacher):
    """Coefficients c_r of W projected (Euclidean) on span{U_r}, divided by s_r."""
    b = forward(spec, cores, [teacher.factors[:, i, :] for i in range(teacher.n_sites)])
    gram = torch.ones(teacher.n_components, teacher.n_components, dtype=torch.float64)
    for i in range(teacher.n_sites):
        gram = gram * (teacher.factors[:, i, :] @ teacher.factors[:, i, :].T)
    return torch.linalg.solve(gram, b) / teacher.weights


def dense_loss_fn(spec, w_star, metrics=None):
    """L = 1/2 <W - W*, M (W - W*)> through the dense tensor: ~5x faster than the tree
    contraction for the tiny pilot networks (N <= 6), identical to 1e-16."""

    def loss(flat):
        r = to_dense(spec, spec.unflatten(flat)) - w_star
        mr = r
        if metrics is not None:
            for i, m in enumerate(metrics):
                mr = apply_on_leg(mr, m, i)
        return 0.5 * (r * mr).sum()

    return loss


def run_gf(spec, cores0, teacher, metrics=None, t_max=1e4, n_rec=400, rtol=1e-9, atol=1e-14,
           method="DOP853", record_overlaps=True, w_star=None):
    t2 = teacher.norm2(metrics)
    if w_star is not None:
        loss_of = dense_loss_fn(spec, w_star, metrics)
    else:
        def loss_of(flat):
            return population_loss(spec, spec.unflatten(flat), teacher, metrics, t2)

    def rhs(_t, y):
        flat = torch.from_numpy(y).requires_grad_(True)
        (g,) = torch.autograd.grad(loss_of(flat), flat)
        return -g.numpy()

    y0 = spec.flatten(cores0).detach().numpy().copy()
    t_eval = np.concatenate([[0.0], np.logspace(-2, np.log10(t_max), n_rec)])
    t_eval[-1] = t_max
    tic = time.time()
    sol = solve_ivp(rhs, (0, t_max), y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval)
    wall = time.time() - tic
    rec = {"t": sol.t, "loss": [], "overlaps": [], "spectra": [], "nfev": sol.nfev, "wall": wall}
    q0 = conserved_quantities(spec, cores0)
    drift = 0.0
    for k in range(sol.y.shape[1]):
        flat = torch.from_numpy(sol.y[:, k].copy())
        cores = spec.unflatten(flat)
        rec["loss"].append(loss_of(flat).item())
        if record_overlaps:
            rec["overlaps"].append(dual_overlaps(spec, cores, teacher).numpy())
        sp = bond_spectra(spec, cores, metrics)
        rec["spectra"].append({v: s.numpy() for v, s in sp.items()})
        q = conserved_quantities(spec, cores)
        drift = max(drift, max((q[v] - q0[v]).abs().max().item() for v in q))
    rec["q_drift"] = drift
    rec["loss"] = np.array(rec["loss"])
    if record_overlaps:
        rec["overlaps"] = np.array(rec["overlaps"])
    rec["y_final"] = sol.y[:, -1]
    return rec


def crossing_time(t, a, level=0.5):
    """First time a(t) crosses `level` (linear interpolation in t), nan if never."""
    idx = np.nonzero(a >= level)[0]
    if len(idx) == 0:
        return np.nan
    k = idx[0]
    if k == 0:
        return t[0]
    return t[k - 1] + (level - a[k - 1]) * (t[k] - t[k - 1]) / (a[k] - a[k - 1])
