"""Gauge-balanced initialization: the same tensor W_0 with every bond conserved quantity Q_e = 0.

For edge v (parent p), M = G_v^T G_v and N = G_p,(v) G_p,(v)^T (`conserved_quantities` convention,
Q_v = M - N). The gauge G_v -> G_v A, G_p -> A^-1 (on leg v) leaves W unchanged and maps
M -> A M A, N -> A^-1 N A^-1; with A = S^(1/2), S = M^-1/2 (M^1/2 N M^1/2)^1/2 M^-1/2 (the matrix
geometric mean of M^-1 and N), both become equal, so Q_v = 0. Balancing one edge changes the Grams
of the neighbouring edges, so we sweep over all edges until max_v |Q_v| is at roundoff.
Used to test whether the fan-in init's root imbalance causes the stalls of theory_saddles R12.
"""

from __future__ import annotations

import torch

from ttn.network import TTNSpec, apply_on_leg, conserved_quantities


def _sym_pow(m: torch.Tensor, p: float) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh((m + m.T) / 2)
    return (vecs * vals.clamp_min(1e-300) ** p) @ vecs.T


def gauge_balance(spec: TTNSpec, cores, tol: float = 1e-13, max_sweeps: int = 2000):
    t = spec.tree
    cores = [c.clone() for c in cores]
    for sweep in range(max_sweeps):
        for v in t.postorder:
            if v == t.root:
                continue
            p = t.parent[v]
            gv = cores[v].reshape(-1, cores[v].shape[-1])
            leg = len(t.sites[p]) + t.children[p].index(v)
            gp = torch.movedim(cores[p], leg, 0).reshape(cores[p].shape[leg], -1)
            M, N = gv.T @ gv, gp @ gp.T
            mh, mih = _sym_pow(M, 0.5), _sym_pow(M, -0.5)
            S = mih @ _sym_pow(mh @ N @ mh, 0.5) @ mih
            A, Ainv = _sym_pow(S, 0.5), _sym_pow(S, -0.5)
            cores[v] = apply_on_leg(cores[v], A.T, cores[v].ndim - 1)  # G_v A on the parent leg
            cores[p] = apply_on_leg(cores[p], Ainv, leg)
        q = conserved_quantities(spec, cores)
        scale = max(c.pow(2).sum().item() for c in cores)
        err = max(x.abs().max().item() for x in q.values()) / scale
        if err < tol:
            return cores, err, sweep + 1
    return cores, err, max_sweeps
