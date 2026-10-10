"""Data-only predictor of the pre-registered order test (notes/real_data.md section 9.2).

Usage: uv run python experiments/real_data/order_predictor.py <out.json> <workers> <order> [<order> ...]

For each pixel order of binary16: the best bond-rank-1 fit f = prod_v g_v . Phi_v of the training labels (8
bottom cores, Phi_v = phi_s (x) phi_s' of the core's two sites), found by 32 starts of higher-order power iteration
on T = (1/n) sum_j y_j (x)_v Phi_v(x_j) followed by alternating least squares on the square loss; at that fit, the
route amplitudes lambda_uw = |P_u M_uw P_w|_2 of all 28 pairs of bottom cores (the formula of route_amplitudes.py)
and S = share of sum lambda^2 on pairs whose tree path has k <= 5 cores. No gradient flow, no training run.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import Task, cos_sin, load_task, make_tree, pixel_order  # noqa: E402

N_STARTS, N_POWER, TOL, MAX_SWEEPS = 32, 50, 1e-12, 2000


def bottom_features(order_name: str, size: int = 4):
    """(tree, bottom nodes, {v: Phi_v (n, 4)}, y) on the training set of the default task."""
    x_tr, y_tr, _, _ = load_task(Task(size=size))
    phi = cos_sin(x_tr[:, pixel_order(order_name, size)])
    tree = make_tree("binary", size * size)
    bottom = [v for v in range(tree.n_nodes) if len(tree.sites[v]) == 2]
    feats = {}
    for v in bottom:
        s1, s2 = tree.sites[v]
        feats[v] = np.einsum("na,nb->nab", phi[:, s1], phi[:, s2]).reshape(len(y_tr), 4)
    return tree, bottom, feats, y_tr


def _prod_except(qs, skip):
    out = np.ones_like(qs[0])
    for i, q in enumerate(qs):
        if i not in skip:
            out = out * q
    return out


def rank1_fit(Phis, y, seed):
    """One start: power iteration on T, least-squares amplitude, then ALS on the square loss."""
    rng = np.random.default_rng(1000 + seed)
    m, n = len(Phis), len(y)
    g = [rng.standard_normal(4) for _ in range(m)]
    g = [v / np.linalg.norm(v) for v in g]
    for _ in range(N_POWER):
        for v in range(m):
            qs = [P @ gu for P, gu in zip(Phis, g)]
            gv = Phis[v].T @ (y * _prod_except(qs, {v})) / n
            g[v] = gv / np.linalg.norm(gv)
    f = _prod_except([P @ gu for P, gu in zip(Phis, g)], set())
    g[0] = g[0] * (f @ y) / (f @ f)
    loss_prev, sweeps = np.inf, 0
    qs = [P @ gu for P, gu in zip(Phis, g)]
    for sweeps in range(1, MAX_SWEEPS + 1):
        for v in range(m):
            a = Phis[v] * _prod_except(qs, {v})[:, None]
            g[v] = np.linalg.lstsq(a, y, rcond=None)[0]
            qs[v] = Phis[v] @ g[v]
        loss = 0.5 * np.mean((_prod_except(qs, set()) - y) ** 2)
        if abs(loss_prev - loss) <= TOL * loss:
            break
        loss_prev = loss
    # balance the core norms (same f): the amplitudes' common scale then cancels in the shares
    norms = np.array([np.linalg.norm(gv) for gv in g])
    scale = np.prod(norms) ** (1 / m)
    g = [gv * scale / nv for gv, nv in zip(g, norms)]
    return g, float(loss), sweeps


def amplitudes(tree, bottom, feats, y, g):
    """lambda_uw, k_uw for all pairs of bottom cores at the rank-1 state g (dict v -> 4-vector)."""
    qs = {v: feats[v] @ g[v] for v in bottom}
    f = np.prod(np.stack([qs[v] for v in bottom]), axis=0)
    r = f - y
    out = []
    for i, u in enumerate(bottom):
        for w in bottom[i + 1 :]:
            h = np.prod(np.stack([qs[v] for v in bottom if v not in (u, w)]), axis=0)
            m = (feats[u] * (r * h)[:, None]).T @ feats[w] / len(y)
            pu = np.eye(4) - np.outer(g[u], g[u]) / (g[u] @ g[u])
            pw = np.eye(4) - np.outer(g[w], g[w]) / (g[w] @ g[w])
            out.append(
                {
                    "u": u,
                    "w": w,
                    "k": len(tree.path(u, w)),
                    "lam": float(np.linalg.norm(pu @ m @ pw, 2)),
                }
            )
    return out


def share(pairs, kmax=5):
    lam = np.array([p["lam"] for p in pairs])
    k = np.array([p["k"] for p in pairs])
    return float((lam[k <= kmax] ** 2).sum() / (lam**2).sum())


def predictor(order_name: str) -> dict:
    tree, bottom, feats, y = bottom_features(order_name)
    Phis = [feats[v] for v in bottom]
    fits = []
    for s in range(N_STARTS):
        g, loss, sweeps = rank1_fit(Phis, y, s)
        pairs = amplitudes(tree, bottom, feats, y, dict(zip(bottom, g)))
        fits.append(
            {
                "start": s,
                "loss": loss,
                "sweeps": sweeps,
                "S": share(pairs),
                "S_k3": share(pairs, 3),
                "pairs": pairs,
            }
        )
    best = min(fits, key=lambda f: f["loss"])
    # distinct optima (losses equal to 1e-6 relative are the same point)
    optima = []
    for f in sorted(fits, key=lambda f: f["loss"]):
        if not optima or abs(f["loss"] - optima[-1]["loss"]) > 1e-6 * f["loss"]:
            optima.append({"loss": f["loss"], "S": f["S"], "count": 1})
        else:
            optima[-1]["count"] += 1
    return {
        "order": order_name,
        "pixel_order": pixel_order(order_name, 4).tolist(),
        "S": best["S"],
        "S_k3": best["S_k3"],
        "best_loss": best["loss"],
        "best_sweeps": best["sweeps"],
        "best_pairs": best["pairs"],
        "optima": optima,
        "S_basin_weighted": float(sum(o["S"] * o["count"] for o in optima) / N_STARTS),
    }


def main(out_json, workers, *orders):
    with Pool(int(workers)) as pool:
        res = pool.map(predictor, orders)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "protocol": {
                "starts": N_STARTS,
                "power_steps": N_POWER,
                "tol": TOL,
                "max_sweeps": MAX_SWEEPS,
            },
            "orders": res,
        },
        open(out_json, "w"),
        indent=1,
    )
    for r in res:
        print(
            f"{r['order']:14s} S = {r['S']:.3f}  (k = 3 share {r['S_k3']:.3f})  best rank-1 loss {r['best_loss']:.4f}"
            f"  sweeps {r['best_sweeps']}  optima {len(r['optima'])} (best found {r['optima'][0]['count']}/{N_STARTS})"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
