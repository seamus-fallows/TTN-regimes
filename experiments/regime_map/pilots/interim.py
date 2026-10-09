"""Interim sweep driver (scipy RK45 gradient flow) used until ttn/dynamics.py + ttn/runner.py land.

One run = one (tree, d, chi, sigma, seed, model, root_scale) point. Models:
  centred  : target W* + W_0 (cancellation-free `centred_population_loss`)
  plain    : target W*
  twocopy  : W(theta+) - W(theta-) with masked gradient (`antisymmetric_double`), W_0 = 0
Per run we save the trajectory (t, L, overlaps a_r of W - W_0 with the teacher components) to
/root/outputs/regime_map/<exp>/<run_id>.npz and return a summary row (end-of-training diagnostics).

Dynamics: gradient flow, scipy RK45, rtol 1e-9, atol 1e-14, float64. Stopped when l < 1e-12, or
l < 1e-4 and t |grad L|^2 < 1e-7 L(0) (converged and stationary), or at the budget
T = 50 t_esc + tau_max / max(lambda_bar, 1), with t_esc = (0.3 sigma)^-(K-2) / ((K-2) s_min) the aligned
escape-time estimate of the smallest component and lambda_bar = tr Theta(0) / D the kernel scale
(notes/theory_regimes.md P1).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lazy_theory import lazy_predictions  # noqa: E402

from ttn.losses import centred_population_loss, population_loss  # noqa: E402
from ttn.network import (  # noqa: E402
    TTNSpec,
    antisymmetric_double,
    conserved_quantities,
    forward,
    sum_difference_cores,
    to_dense,
)
from ttn.teachers import odeco_teacher  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

torch.set_num_threads(1)
OUT = "/root/outputs/regime_map"
PROBE_SEED = 12345


def make_spec(kind, n, d, chi, cap, bonds=None):
    tree = make_tree(kind, n)
    if bonds is not None:  # explicit per-edge bond dims, e.g. {"by_size": {2: 18, 4: 162}}
        b = [1] * tree.n_nodes
        for v in tree.edges:
            b[v] = bonds[len(tree.subtree_sites[v])]
        return TTNSpec(tree, (d,) * n, tuple(b))
    return TTNSpec.uniform(tree, d, chi, cap=cap)


def run_point(cfg: dict) -> dict:
    """cfg keys: exp, kind, n, d, chi, cap, sigma, seed, model, weights, root_scale (default 1),
    tau_max (default 1e4), bonds (optional), lr (sp|ntk|mup, default sp)."""
    tic = time.time()
    kind, n, d, chi = cfg["kind"], cfg["n"], cfg["d"], cfg["chi"]
    sigma, seed, model = cfg["sigma"], cfg["seed"], cfg["model"]
    weights = cfg.get("weights", [3.0, 2.0, 1.0])
    spec = make_spec(kind, n, d, chi, cfg.get("cap", True), cfg.get("bonds"))
    K = spec.n_cores
    tree = spec.tree
    scales = [1.0] * K
    scales[tree.root] = cfg.get("root_scale", 1.0)
    base = odeco_teacher(n, d, list(weights), seed=1000 + seed)
    cores_init = spec.init(sigma, seed=seed, core_scales=scales)
    eta_core = [1.0] * K  # per-core learning-rate multipliers (sp)
    if cfg.get("lr", "sp") == "ntk":
        eta_core = [1.0 / f for f in spec.fan_in]
    flat_eta1 = torch.cat([torch.full((sz,), e, dtype=torch.float64) for sz, e in zip(spec.sizes, eta_core)])
    b2 = base.norm2()

    if model == "twocopy":
        spec_run, cores0, mask = antisymmetric_double(spec, cores_init)
        eta = mask * torch.cat(
            [torch.full((sz,), e, dtype=torch.float64) for sz, e in zip(spec_run.sizes, eta_core)]
        )
    else:
        spec_run, cores0 = spec, cores_init
        eta = flat_eta1
    flat0 = spec_run.flatten(cores0).clone()

    def loss_of(flat):
        cores = spec_run.unflatten(flat)
        if model == "centred":
            return centred_population_loss(spec_run, cores, cores0, base, base_norm2=b2)
        if model == "plain":
            return population_loss(spec_run, cores, base, teacher_norm2=b2)
        return population_loss(spec_run, sum_difference_cores(spec_run, cores), base, teacher_norm2=b2)

    def loss_grad(y):
        flat = torch.from_numpy(y).requires_grad_(True)
        L = loss_of(flat)
        (g,) = torch.autograd.grad(L, flat)
        return L.item(), g * eta

    L0 = loss_grad(flat0.numpy())[0]

    # overlaps a_r(t) = <W(t) - W_0, U_r> / s_r (twocopy: W_0 = 0)
    u_vecs = [base.factors[:, i, :] for i in range(n)]

    def overlaps(y):
        cores = spec_run.unflatten(torch.from_numpy(y))
        if model == "twocopy":
            cores = sum_difference_cores(spec_run, cores)
        return forward(spec_run, cores, u_vecs)

    ov0 = overlaps(flat0.numpy())
    if model == "twocopy":
        ov0 = torch.zeros_like(ov0)

    # kernel scale and time budget
    probes = torch.randn(128, n, d, generator=torch.Generator().manual_seed(PROBE_SEED), dtype=torch.float64)

    def probe_kernel(flat):
        jac = torch.func.jacrev(lambda p: forward(spec_run, spec_run.unflatten(p), probes))(flat)
        jac = jac * eta.sqrt()
        return jac @ jac.T

    D = d**n
    alpha = math.prod(sigma * s for s in scales)
    lam_bar = alpha**2 / D * sum(e * f / (sigma * s) ** 2 for e, f, s in zip(eta_core, spec.fan_in, scales))
    if model == "twocopy":
        lam_bar *= 2
    s_min = min(weights)
    t_esc = (0.3 * sigma) ** (-(K - 2)) / ((K - 2) * s_min) if K > 2 else math.log(1 / (0.3 * sigma)) / s_min
    tau_max = cfg.get("tau_max", 1e4)
    t_max = cfg.get("t_max") or (50 * t_esc + tau_max / max(lam_bar, 1.0))

    def stop(t, y):
        L, g = loss_grad(y)
        if L / L0 < 1e-12 or (L / L0 < 1e-4 and t * float(g @ (g / eta.clamp_min(1e-300))) < 1e-7 * L0):
            return -1.0
        return 1.0

    stop.terminal = True
    t_first = min(1e-3 / lam_bar, 1e-3 * t_esc, 1e-3)
    t_eval = np.concatenate([[0.0], np.geomspace(t_first, t_max, 600)])
    sol = solve_ivp(lambda t, y: -loss_grad(y)[1].numpy(), (0, t_max), flat0.numpy(), method="RK45",
                    t_eval=t_eval, rtol=cfg.get("rtol", 1e-9), atol=cfg.get("atol", 1e-14), events=stop)
    y_end = sol.y_events[0][0] if sol.status == 1 else sol.y[:, -1]
    t_end = float(sol.t_events[0][0]) if sol.status == 1 else float(sol.t[-1])
    ts = np.append(sol.t, t_end) if sol.status == 1 else sol.t
    ys = np.concatenate([sol.y, y_end[:, None]], axis=1) if sol.status == 1 else sol.y
    Ls = np.array([loss_grad(ys[:, k])[0] for k in range(ys.shape[1])])
    A = np.array([((overlaps(ys[:, k]) - ov0) / base.weights).numpy() for k in range(ys.shape[1])])

    th0, thT = flat0, torch.from_numpy(y_end)
    K0, KT = probe_kernel(th0), probe_kernel(thT)
    dK = ((KT - K0).norm() / K0.norm()).item()
    y_al = base(probes)
    align = lambda k: (y_al @ k @ y_al / (k.norm() * (y_al @ y_al))).item()  # noqa: E731
    dK_dir = abs((y_al @ (KT - K0) @ y_al).item()) / (y_al @ K0 @ y_al).item()
    dth = ((thT - th0).norm() / th0.norm()).item()
    sl = spec_run.core_slices()
    dth_core = [((thT[s] - th0[s]).norm() / th0[s].norm()).item() for s in sl]
    # gauge conservation-law drift (relative to the largest initial Q or Gram scale)
    q_kw = dict(lr_multipliers=eta_core, n_copies=2 if model == "twocopy" else 1)
    q0 = conserved_quantities(spec_run, spec_run.unflatten(th0), **q_kw)
    qT = conserved_quantities(spec_run, spec_run.unflatten(thT), **q_kw)
    gram = max((spec_run.unflatten(thT)[v].reshape(-1, spec_run.unflatten(thT)[v].shape[-1]) ** 2).sum().item()
               for v in tree.edges)
    q_drift = max((qT[v] - q0[v]).norm().item() for v in tree.edges) / gram

    # best attainable centred loss for binary N=4 (one bipartition: exact truncated SVD)
    l_best = None
    if model == "centred" and kind == "binary" and n == 4:
        w = (to_dense(spec, cores_init) + base.dense()).reshape(d * d, d * d)
        sv = torch.linalg.svdvals(w)
        l_best = float((sv[min(spec.bond_dims[v] for v in tree.edges):] ** 2).sum() / (2 * L0))

    row = dict(cfg)
    row.update(
        K=K, P=spec_run.n_params, alpha=alpha, lam_bar=lam_bar, t_max=t_max, t_end=t_end,
        status=int(sol.status), nfev=int(sol.nfev), L0=L0, l_end=float(Ls[-1] / L0), l_best=l_best,
        dK=dK, dK_dir=dK_dir, align0=align(K0), alignT=align(KT), dth=dth, dth_core=dth_core,
        q_drift=q_drift, wall=time.time() - tic,
    )
    path = os.path.join(OUT, cfg["exp"], "traj")
    os.makedirs(path, exist_ok=True)
    np.savez_compressed(os.path.join(path, run_id(cfg) + ".npz"), t=ts, L=Ls / L0, A=A)
    return row


def lazy_row(cfg: dict) -> dict:
    """Lazy-limit predictions at sigma = 1 for the run's init seed (sigma-independent)."""
    kind, n, d, chi, seed = cfg["kind"], cfg["n"], cfg["d"], cfg["chi"], cfg["seed"]
    spec = make_spec(kind, n, d, chi, cfg.get("cap", True), cfg.get("bonds"))
    scales = [1.0] * spec.n_cores
    scales[spec.tree.root] = cfg.get("root_scale", 1.0)
    base = odeco_teacher(n, d, list(cfg.get("weights", [3.0, 2.0, 1.0])), seed=1000 + seed)
    probes = torch.randn(128, n, d, generator=torch.Generator().manual_seed(PROBE_SEED), dtype=torch.float64)
    eta = None
    if cfg.get("lr", "sp") == "ntk":
        eta = torch.cat([torch.full((sz,), 1.0 / f, dtype=torch.float64) for sz, f in zip(spec.sizes, spec.fan_in)])
    p = lazy_predictions(spec, spec.init(1.0, seed=seed, core_scales=scales), base, probes, mask=eta)
    return p


def run_id(cfg):
    keys = ["model", "kind", "n", "d", "chi", "sigma", "seed", "root_scale", "lr"]
    return "_".join(f"{k}{cfg.get(k)}" for k in keys if cfg.get(k) is not None)


def safe_run(cfg):
    try:
        return run_point(cfg)
    except Exception:  # noqa: BLE001
        row = dict(cfg)
        row["error"] = traceback.format_exc()
        return row


def run_sweep(cfgs, exp, n_proc=340):
    """Run all configs in parallel (resumable: rows already in results.jsonl are skipped)."""
    from multiprocessing import Pool

    os.makedirs(os.path.join(OUT, exp), exist_ok=True)
    out = os.path.join(OUT, exp, "results.jsonl")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    done.add(run_id(r))
    todo = [c for c in cfgs if run_id(c) not in done]
    # longest first: small sigma and large chi
    todo.sort(key=lambda c: (-c["n"], c["sigma"]))
    print(f"{exp}: {len(todo)} runs to do ({len(done)} done)", flush=True)
    with Pool(min(n_proc, max(1, len(todo)))) as pool, open(out, "a") as f:
        for i, row in enumerate(pool.imap_unordered(safe_run, todo)):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if "error" in row:
                print("ERROR", run_id(row), row["error"][-300:], flush=True)
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(todo)} done", flush=True)
