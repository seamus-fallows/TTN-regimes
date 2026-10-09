"""One regime-map run with the shared library (ttn.dynamics / ttn.diagnostics), and a parallel,
resumable sweep loop (until ttn.runner lands).

cfg keys: exp, kind, n, d, chi, cap, sigma, seed, model (centred | plain | twocopy); optional
weights (default [3, 2, 1]), root_scale (root init multiplier, imbalance), lr (sp | ntk | mup),
bonds ({subtree size: bond dim}), method (dopri5 | lsoda | auto = lsoda for sigma < 0.7), tau_max
(kernel-time budget, default 1e4), l_floor (stop when L < l_floor L(0), default 1e-6), max_steps
(default 1e5; a run stopped there is labelled from its state at that time), rtol (1e-8), atol (1e-12).

Time budget: T = 50 t_esc + tau_max / max(lambda_bar, 1), t_esc = (0.3 sigma)^-(K-2) / ((K-2) s_min)
(aligned escape-time estimate for the smallest component) and lambda_bar = tr Theta(0) / D
(kernel scale, notes/theory_regimes.md P1, with the per-core learning rates and init scales).
Outputs: Trajectory under /root/outputs/regime_map/<exp>/traj/<run_id>.npz and one summary row
(JSON) per run in /root/outputs/regime_map/<exp>/results.jsonl.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback

import numpy as np
import torch

from ttn import diagnostics as dg
from ttn.dynamics import Problem, Recorder, integrate_gf
from ttn.network import TTNSpec, antisymmetric_double, to_dense
from ttn.teachers import ShiftedTeacher, odeco_teacher
from ttn.tree import make_tree

OUT = "/root/outputs/regime_map"
PROBE_SEED = 12345
N_PROBE = 128


def make_spec(kind, n, d, chi, cap=True, bonds=None) -> TTNSpec:
    tree = make_tree(kind, n)
    if bonds is not None:  # explicit bond dims by subtree size, e.g. {2: 18, 4: 162}
        b = [1] * tree.n_nodes
        for v in tree.edges:
            b[v] = int(bonds[len(tree.subtree_sites[v])])
        return TTNSpec(tree, (d,) * n, tuple(b))
    return TTNSpec.uniform(tree, d, chi, cap=cap)


def parameterization(spec: TTNSpec, cfg: dict):
    """Per-core init multipliers m_v and learning-rate multipliers eta_v (theory_regimes 4.3):
    sp (1, 1); ntk (1, 1/F_v); mup = muP-TTN (root init x (sum of child-leg dims)^-1/2,
    eta_v = chi_v / F_v). `root_scale` multiplies the root's init on top (imbalance)."""
    root = spec.tree.root
    lr = cfg.get("lr", "sp")
    scales = [1.0] * spec.n_cores
    if lr == "sp":
        eta = [1.0] * spec.n_cores
    elif lr == "ntk":
        eta = [1.0 / f for f in spec.fan_in]
    elif lr == "mup":
        eta = [spec.bond_dims[v] / spec.fan_in[v] for v in range(spec.n_cores)]
        scales[root] = sum(spec.shapes[root][:-1]) ** -0.5
    else:
        raise ValueError(lr)
    scales[root] *= cfg.get("root_scale", 1.0)
    return scales, eta


def run_id(cfg: dict) -> str:
    keys = ["model", "kind", "n", "d", "chi", "sigma", "seed", "root_scale", "lr", "alpha"]
    return "_".join(f"{k}{cfg.get(k)}" for k in keys if cfg.get(k) is not None)


def build(cfg: dict):
    """(problem, theta0, spec, info) for a config."""
    spec = make_spec(cfg["kind"], cfg["n"], cfg["d"], cfg["chi"], cfg.get("cap", True), cfg.get("bonds"))
    scales, eta = parameterization(spec, cfg)
    weights = cfg.get("weights", [3.0, 2.0, 1.0])
    base = odeco_teacher(cfg["n"], cfg["d"], list(weights), seed=1000 + cfg["seed"])
    cores0 = spec.init(cfg["sigma"], seed=cfg["seed"], core_scales=scales)
    model = cfg["model"]
    if model == "centred":
        problem = Problem(spec, ShiftedTeacher(base, spec, cores0), lr_multipliers=eta)
        theta0 = spec.flatten(cores0)
    elif model == "plain":
        problem = Problem(spec, base, lr_multipliers=eta)
        theta0 = spec.flatten(cores0)
    elif model == "twocopy":
        spec2, cores2, mask = antisymmetric_double(spec, cores0)
        problem = Problem(spec2, base, lr_multipliers=eta, mask=mask)
        theta0 = spec2.flatten(cores2)
    else:
        raise ValueError(model)
    alpha = math.prod(cfg["sigma"] * s for s in scales)
    D = cfg["d"] ** cfg["n"]
    lam_bar = alpha**2 / D * sum(e * f / (cfg["sigma"] * s) ** 2 for e, f, s in zip(eta, spec.fan_in, scales))
    if model == "twocopy":
        lam_bar *= 2
    return problem, theta0, spec, dict(alpha=alpha, lam_bar=lam_bar, base=base, cores0=cores0, eta=eta)


def kernel_weights(problem: Problem) -> torch.Tensor | None:
    """Per-parameter weights eta * mask so that the probe kernel is the kernel of the flow."""
    return problem.precond


def refine_events(problem: Problem, t, A, params, kw, n_dense=200, max_steps=100_000):
    """Event times and final-rise widths resolved in linear time.

    The log-time grid cannot resolve O(1)-wide events at late times (spacing ~1% of t). For each
    component whose final rise is recorded, re-integrate from the saved parameters at the last log
    point before the rise (last a_r < 0.1 before the first a_r >= 0.9) to the first log point with
    a_r >= 0.9, on a linear grid of n_dense points, and take t(1/2) and the width there. If the first
    crossing of 1/2 lies before the window (a transient bump), the coarse event time is kept."""
    times = dg.crossing_times(t, A, 0.5)
    widths = dg.event_widths(t, A)
    refined = []
    for r in range(A.shape[1]):
        hi = np.flatnonzero(A[:, r] >= 0.9)
        if hi.size == 0:
            refined.append(False)
            continue
        i1 = hi[0]
        lo = np.flatnonzero(A[:i1, r] < 0.1)
        i0 = lo[-1] if lo.size else 0
        if i1 - i0 < 1 or i0 == i1:
            refined.append(False)
            continue
        span = t[i1] - t[i0]
        grid = np.linspace(0.0, span, n_dense + 1)[1:]
        sub = integrate_gf(problem, torch.from_numpy(params[i0]), span, log_times=grid, max_steps=max_steps,
                           recorder=Recorder(spectra=False, conservation=False), **kw)
        ts, a = sub["t"] + t[i0], sub["overlaps"][:, r]
        w = dg.event_widths(ts, a)
        t_half = dg.crossing_times(ts, a, 0.5)
        if np.isfinite(w):
            widths[r] = w
        if np.isfinite(t_half) and not (times[r] < t[i0]):
            times[r] = t_half
        refined.append(True)
    return times, widths, refined


def run_point(cfg: dict) -> dict:
    torch.set_num_threads(1)
    tic = time.time()
    problem, theta0, spec, info = build(cfg)
    K = spec.n_cores
    sigma = cfg["sigma"]
    weights = cfg.get("weights", [3.0, 2.0, 1.0])
    s_min = min(weights)
    t_esc = (0.3 * sigma) ** (-(K - 2)) / ((K - 2) * s_min) if K > 2 else math.log(1 / (0.3 * sigma)) / s_min
    t_max = cfg.get("t_max") or (50 * t_esc + cfg.get("tau_max", 1e4) / max(info["lam_bar"], 1.0))
    L0 = problem.loss(theta0).item()
    method = cfg.get("method", "dopri5")
    if method == "auto":  # LSODA for the stiff later plateaus at small sigma, dopri5 otherwise
        method = "lsoda" if sigma < 0.7 else "dopri5"
    kw = dict(rtol=cfg.get("rtol", 1e-8), atol=cfg.get("atol", 1e-12), method=method)
    floor = cfg.get("l_floor", 1e-6) * L0

    def integrate(kw):
        return integrate_gf(problem, theta0, t_max, n_log=cfg.get("n_log", 300), loss_floor=floor,
                            max_steps=cfg.get("max_steps", 100_000), recorder=Recorder(save_params=True),
                            **kw)

    traj = integrate(kw)
    if traj.meta["stop_reason"] == "failed" and method == "lsoda":  # fall back to the reference
        kw["method"] = "dopri5"
        traj = integrate(kw)
    if traj.meta["stop_reason"] in ("failed", "nonfinite", "step_size_underflow"):
        raise RuntimeError(f"integration failed: {traj.meta['stop_reason']} {traj.meta.get('message')}")
    params = traj.data.pop("params")  # (n_log, P); used for the event refinement only, not saved
    theta_end = torch.from_numpy(traj["theta_end"])
    t, A = traj["t"], traj["overlaps"]
    times, widths, refined = refine_events(problem, t, A, params, kw, max_steps=cfg.get("max_steps", 100_000))
    rhos = dg.separation_ratios(times, widths)

    probes = torch.randn(N_PROBE, cfg["n"], cfg["d"], generator=torch.Generator().manual_seed(PROBE_SEED),
                         dtype=torch.float64)
    w = kernel_weights(problem)
    K0 = dg.tangent_kernel(problem.spec, theta0, probes, w)
    KT = dg.tangent_kernel(problem.spec, theta_end, probes, w)
    dK = dg.kernel_change(KT, K0)
    y = info["base"](probes)
    dK_dir = abs(float(y @ (KT - K0) @ y)) / float(y @ K0 @ y)
    l_end = float(traj["loss"][-1] / L0)
    label = dg.classify_regime(dK, rhos, traj["loss"][-1], L0)

    sl = problem.spec.core_slices()
    dth_core = [float((theta_end[s] - theta0[s]).norm() / theta0[s].norm()) for s in sl]
    # float64 floor (lead's note): smallest |a_r| of a not-yet-learned component at the first event
    first = np.nanmin(times) if np.isfinite(times).any() else np.nan
    dormant = np.nan
    if np.isfinite(first):
        i = np.searchsorted(t, first)
        later = [r for r in range(A.shape[1]) if not (times[r] <= first * 1.0001)]
        if later:
            dormant = float(np.min(np.abs(A[max(i - 1, 0), later])))
    l_best = None
    if cfg["model"] == "centred" and cfg["kind"] == "binary" and cfg["n"] == 4:
        d = cfg["d"]
        wmat = (to_dense(spec, info["cores0"]) + info["base"].dense()).reshape(d * d, d * d)
        sv = torch.linalg.svdvals(wmat)
        chi_min = min(spec.bond_dims[v] for v in spec.tree.edges)
        l_best = float((sv[chi_min:] ** 2).sum() / (2 * L0))

    row = {k: v for k, v in cfg.items()}
    row.update(
        K=K, P=problem.spec.n_params, alpha=info["alpha"], lam_bar=info["lam_bar"], t_max=t_max,
        integrator=kw["method"],
        t_end=float(traj.meta["t_end"]), stop=traj.meta["stop_reason"], stiff_at=traj.meta.get("stiff_at"),
        n_steps=traj.meta["n_steps"], L0=L0, l_end=l_end, l_best=l_best, dK=dK, dK_dir=dK_dir,
        dth=float(traj["rel_dist_from_init"][-1]), dth_core=dth_core,
        q_drift=float(np.nanmax(traj["conservation_drift"])), event_times=times.tolist(),
        event_widths=widths.tolist(), rhos=rhos.tolist(), label=label, dormant_min=dormant,
        refined=refined,
        wall=time.time() - tic,
    )
    path = os.path.join(OUT, cfg["exp"], "traj")
    os.makedirs(path, exist_ok=True)
    traj.save(os.path.join(path, run_id(cfg) + ".npz"))
    return row


def safe_run(cfg):
    try:
        return run_point(cfg)
    except Exception:  # noqa: BLE001
        row = dict(cfg)
        row["error"] = traceback.format_exc()
        return row


def run_sweep(cfgs, exp=None, n_proc=340, order=None):
    """Run configs (possibly from several experiments, each cfg has 'exp') in one process pool;
    rows go to /root/outputs/regime_map/<exp>/results.jsonl, and runs already there (without error)
    are skipped, so sweeps are resumable."""
    from multiprocessing import get_context

    exps = sorted({c["exp"] for c in cfgs})
    done, files = set(), {}
    for e in exps:
        os.makedirs(os.path.join(OUT, e), exist_ok=True)
        path = os.path.join(OUT, e, "results.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if "error" not in r:
                        done.add((e, run_id(r)))
        files[e] = open(path, "a")
    todo = [c for c in cfgs if (c["exp"], run_id(c)) not in done]
    todo.sort(key=order or (lambda c: (-c["n"], c["sigma"])))
    print(f"{exps}: {len(todo)} runs to do ({len(done)} done)", flush=True)
    with get_context("fork").Pool(min(n_proc, max(1, len(todo))), maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.imap_unordered(safe_run, todo)):
            f = files[row["exp"]]
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            if "error" in row:
                print("ERROR", row["exp"], run_id(row), row["error"][-400:], flush=True)
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(todo)} done", flush=True)
    for f in files.values():
        f.close()
