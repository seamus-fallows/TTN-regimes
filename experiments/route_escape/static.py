"""Static saddle order at theta_J: m(t) = min of L - L_J over the sphere ||theta - theta_J|| = t.

Generalises experiments/theory_checks/saddle_order/saddle_order.py to the reference theta_J with the
cancellation-free excess 1/2 ||D||^2 - <R, D>, D = W(theta) - W(theta_J) (telescoping sum),
R = W* - W(theta_J). Two spaces: "Mstar" (perturbations restricted to the allowed entries of the
block-aligned manifold, `AlignedManifold.mask`) and "full" (all parameters). L-BFGS on the sphere,
12 random starts per t plus continuation from the previous t's minimiser. The order is the slope of
log|m| against log t. Also: the extreme Hessian eigenvalues at theta_J (full space and M*).

Usage: uv run python experiments/route_escape/static.py CELL SPACE   (SPACE in Mstar, full)
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
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, FullModel, get_cell  # noqa: E402

torch.set_num_threads(1)
T_GRID = [0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025, 0.0175, 0.0125]
N_RAND = 12


def sphere_min(model, t, starts, sub, scale, maxiter=4000):
    """min over u of excess(t u / |u|) restricted to the coordinates `sub` (bool mask)."""
    sub_t = torch.from_numpy(sub)
    base = model.cell.theta_J.clone()

    def fun(u_np):
        u = torch.from_numpy(u_np).requires_grad_(True)
        delta = torch.zeros_like(base).masked_scatter(sub_t, t * u / u.norm())
        D = model.delta(base + delta)
        val = (0.5 * (D * D).sum() - (model.Rp * D).sum()) / scale
        (g,) = torch.autograd.grad(val, u)
        return val.item(), g.numpy()

    best = (math.inf, None, None)
    for u0 in starts:
        res = minimize(fun, u0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": 1e-13, "ftol": 1e-16, "maxcor": 30})
        if res.fun < best[0]:
            best = (res.fun, res.x / np.linalg.norm(res.x), res.nit)
    return best[0] * scale, best[1], best[2]


def describe_minimiser(cell, model, t, u, sub):
    """Where the minimiser lives: weight on active / dormant-M* / off-M* entries, off-target mass of
    D, the largest target and non-target entries of D."""
    delta = np.zeros(cell.spec.n_params)
    delta[sub] = t * u
    w = delta**2 / (t * t)
    th = cell.theta_J + torch.from_numpy(delta)
    D = model.delta(th)
    tm = model.target_mask
    off = float((D.masked_fill(tm, 0.0) ** 2).sum())
    on = float((D[tm] ** 2).sum())
    flat = D.masked_fill(tm, 0.0).abs().reshape(-1)
    top = torch.topk(flat, 4)
    shape = D.shape
    top_off = [("".join(map(str, np.unravel_index(int(i), shape))), float(D.reshape(-1)[i]))
               for i in top.indices]
    P = {cell.pattern_strings[k]: float(D[p]) for k, p in enumerate(cell.patterns)}
    return dict(w_active=float(w[cell.active_flat].sum()), w_dormant=float(w[cell.dormant_flat].sum()),
                w_offM=float(w[~cell.mask].sum()), D_offtarget_frac=off / (off + on) if off + on > 0 else 0.0,
                D_target=P, D_top_offtarget=top_off)


def hessian_eigs(cell, model):
    H = model.hessian(cell.theta_J).numpy()
    ev = np.linalg.eigvalsh(H)
    idx = np.where(cell.mask)[0]
    evm = np.linalg.eigvalsh(H[np.ix_(idx, idx)])
    tol = 1e-10
    return dict(lam_min=float(ev[0]), n_neg=int((ev < -tol).sum()), n_zero=int((abs(ev) <= tol).sum()),
                lam_min_pos=float(ev[ev > tol][0]) if (ev > tol).any() else None,
                lam_min_Mstar=float(evm[0]), n_neg_Mstar=int((evm < -tol).sum()))


def run(cell_name: str, space: str, seed: int = 0) -> dict:
    cell = get_cell(cell_name)
    model = FullModel(cell)
    sub = cell.mask.copy() if space == "Mstar" else np.ones(cell.spec.n_params, dtype=bool)
    n = int(sub.sum())
    rng = np.random.default_rng(seed)
    sub_active = cell.active_flat[sub]
    p_ref = cell.c_star
    rows, prev = [], None
    for t in T_GRID:
        starts = []
        for k in range(N_RAND):
            u = rng.standard_normal(n)
            if k % 3 == 1:  # non-active entries only
                u = u * (~sub_active)
            elif k % 3 == 2:  # mostly active entries
                u = u * sub_active + 0.1 * u * (~sub_active)
            starts.append(u)
        if prev is not None:
            starts = [prev] + starts
        t0 = time.time()
        m, u, nit = sphere_min(model, t, starts, sub, t**p_ref)
        prev = u
        row = dict(t=t, m=m, m_over_tc=m / t**p_ref, secs=time.time() - t0, nit=nit)
        row.update(describe_minimiser(cell, model, t, u, sub))
        rows.append(row)
        print(f"{cell_name} {space} t={t:.4f} m={m:+.4e} m/t^c*={m / t**p_ref:+.4f} "
              f"w_dorm={row['w_dormant']:.3f} w_offM={row['w_offM']:.3f} offD={row['D_offtarget_frac']:.3f} "
              f"{row['secs']:.0f}s", flush=True)
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        b["slope"] = (math.log(abs(a["m"])) - math.log(abs(b["m"]))) / math.log(a["t"] / b["t"]) \
            if a["m"] < 0 and b["m"] < 0 else None
    out = dict(cell=cell_name, space=space, c_star=cell.c_star, rows=rows)
    if space == "full":
        out["hessian"] = hessian_eigs(cell, model)
    return out


def path_for(cell, space):
    return OUT / "static" / f"{cell}_{space}.json"


if __name__ == "__main__":
    cell, space = sys.argv[1], sys.argv[2]
    with threadpool_limits(limits=1, user_api="blas"):
        o = run(cell, space)
    p = path_for(cell, space)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1, default=float))
    print(json.dumps({k: v for k, v in o.items() if k != "rows"}, default=float))
