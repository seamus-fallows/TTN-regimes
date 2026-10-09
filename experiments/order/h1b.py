"""H1b: is a wrong learning order corrected through a strict saddle?

Usage: uv run python experiments/order/h1b.py experiments/order/configs/<h1b config>.yaml

Two kinds of runs (config key `mode`):
- `natural`: random fan-in init at scale sigma (seeds listed in the config or read from a sweep's
  runs.jsonl as the seeds whose first event was the smaller component); full gradient flow to
  convergence.
- `saddle`: start at the balanced wrong saddle (component 2 learned, aligned with amplitude
  s_2^(1/K) per core) plus a fan-in random perturbation at scale sigma; full gradient flow.
For every run: event times (first up-crossing of a_r = 1/2, later down/up crossings), loss levels,
second bond singular value, and the gauge-projected Hessian at the midpoint (minimum |grad L|)
of the wrong plateau. Writes /root/outputs/order/<name>/<run>.npz (trajectory) and
results/<name>.csv (one row per run).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    crossings,
    gf_trajectory,
    loss_grad_fn,
    make_spec,
    make_teacher,
    overlaps,
    projected_hessian,
    unit_init,
)
from ttn.network import aligned_init, bond_spectra  # noqa: E402

OUT = Path("/root/outputs/order")
RES = Path(__file__).parent / "results"


def wrong_seeds(sweep: str, kind, n, d, chi, rho, sigma, limit):
    out = []
    for line in open(OUT / sweep / "runs.jsonl"):
        r = json.loads(line)
        if (r["kind"], r["n"], r["d"], r["chi"]) != (kind, n, d, chi):
            continue
        for run in r["runs"]:
            if run["rho"] == rho and run["sigma"] == sigma and run["winner"] == 1:
                out.append(r["seed"])
    return sorted(out)[:limit]


def saddle_init(spec, teacher, rho, sigma, seed):
    k = spec.n_cores
    amps = torch.zeros(k, teacher.n_components, dtype=torch.float64)
    amps[:, 1] = rho ** (1.0 / k)  # component 2 (weight rho) learned, balanced
    base = spec.flatten(aligned_init(spec, teacher.factors, amps)).numpy()
    return base + sigma * unit_init(spec, seed)


def event_crossings(spec, teacher, sol, r):
    """Up/down crossings of a_r = 1/2: bracketed on the solver's own steps (which resolve every
    transition), refined by root finding on the dense output."""
    ts = sol.t
    a = np.array([overlaps(spec, teacher, sol.sol(t))[r] for t in ts])
    out = []
    for t, sgn in crossings(ts, a):
        i = np.searchsorted(ts, t)
        lo, hi = ts[max(i - 1, 0)], ts[min(i, len(ts) - 1)]
        f = lambda x: overlaps(spec, teacher, sol.sol(x))[r] - 0.5  # noqa: E731
        out.append((brentq(f, lo, hi, xtol=1e-12 * max(hi, 1.0)) if f(lo) * f(hi) < 0 else t, sgn))
    return out, a


def analyse(spec, teacher, sol, rho):
    loss, _ = loss_grad_fn(spec, teacher)
    t_end = sol.t[-1]
    a1x, a1_steps = event_crossings(spec, teacher, sol, 0)
    a2x, a2_steps = event_crossings(spec, teacher, sol, 1)
    t1 = next((t for t, s in a1x if s > 0), np.inf)
    # saddle starts begin with a_2 ~ 1 (component 2 learned at t = 0)
    t2 = 0.0 if a2_steps[0] >= 0.5 else next((t for t, s in a2x if s > 0), np.inf)
    # record grid: log-spaced, uniform, and dense around the first two events
    t_lo = 0.7 * min(t1, t2) if np.isfinite(min(t1, t2)) else 0.0
    t_hi = max(t for t in (t1, t2, 0.0) if np.isfinite(t))
    t_hi = min(t_end, t_hi + 2 * (abs(t1 - t2) if np.isfinite(t1 - t2) else 0) + 20)
    grid = np.unique(np.concatenate([np.geomspace(max(t_end * 1e-8, 1e-3), t_end, 600),
                                     np.linspace(0, t_end, 1000), np.linspace(t_lo, t_hi, 2000)]))
    edge = spec.tree.edges[0]
    rows = []
    for t in grid:
        y = sol.sol(t)
        f = torch.from_numpy(y).requires_grad_(True)
        lval = loss(f)
        (g,) = torch.autograd.grad(lval, f)
        with torch.no_grad():
            bs = bond_spectra(spec, spec.unflatten(torch.from_numpy(y)))[edge]
        a = overlaps(spec, teacher, y)
        rows.append([t, lval.item(), *a, g.norm().item(), bs[0].item(),
                     bs[1].item() if len(bs) > 1 else 0.0])
    traj = np.array(rows)  # t, L, a1, a2, |grad|, s1(edge), s2(edge)
    rec = dict(t1=t1, t2=t2, wrong=bool(t2 < t1), final_loss=traj[-1, 1], t_end=t_end,
               a2_down=next((t for t, s in a2x if s < 0 and t > t2), np.nan),
               a2_reup=next((t for t, s in a2x if s > 0 and t > t2 + 1e-9 and t > t1), np.nan))
    if rec["wrong"] and np.isfinite(t1):
        i1 = np.searchsorted(traj[:, 0], t1)
        rec["a2_min_after_t1"] = traj[i1:, 3].min()
        rec["s2_bond_at_t1"] = np.interp(t1, traj[:, 0], traj[:, 6])
        # wrong plateau: between t2 and t1, minimum parameter speed
        mask = (traj[:, 0] > t2) & (traj[:, 0] < t1)
        if mask.sum() >= 3:
            def speed(t):
                f = torch.from_numpy(sol.sol(t)).requires_grad_(True)
                (gg,) = torch.autograd.grad(loss(f), f)
                return gg.norm().item()
            idx = np.nonzero(mask)[0]
            im = idx[np.argmin(traj[idx, 4])]
            lo, hi = traj[max(im - 1, idx[0]), 0], traj[min(im + 1, idx[-1]), 0]
            res = minimize_scalar(speed, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-9 * max(hi, 1.0)})
            tm = res.x
            ev, gauge_rank = projected_hessian(spec, teacher, sol.sol(tm))
            q = rho ** (1.0 / spec.n_cores)
            rec["eigs"] = ev.tolist()
            rec.update(t_mid=tm, loss_mid=loss(torch.from_numpy(sol.sol(tm))).item(),
                       a_mid=overlaps(spec, teacher, sol.sol(tm)).tolist(),
                       grad_mid=res.fun, lam_min=ev[0], lam_2=ev[1], lam_max=ev[-1],
                       n_neg=int((ev < -1e-3 * ev[-1]).sum()), gauge_rank=gauge_rank,
                       lam_pred=-(1 - rho) * q ** (spec.n_cores - 2))
            # loss on the plateau after the rotation (if any): minimum speed between t1 and re-up
            if np.isfinite(rec["a2_reup"]):
                res2 = minimize_scalar(speed, bounds=(t1, rec["a2_reup"]), method="bounded")
                rec["loss_after_rotation"] = loss(torch.from_numpy(sol.sol(res2.x))).item()
    return rec, traj


def job(args):
    cfg, seed, sigma, rho = args
    torch.set_num_threads(1)
    t0 = time.time()
    spec = make_spec(cfg["kind"], cfg["n"], cfg["d"], cfg["chi"])
    teacher = make_teacher(cfg["n"], cfg["d"], [1.0, rho], cfg.get("pattern"))
    if cfg["mode"] == "natural":
        y0 = sigma * unit_init(spec, seed)
    else:
        y0 = saddle_init(spec, teacher, rho, sigma, seed)
    sol = gf_trajectory(spec, teacher, y0, t_max=cfg.get("t_max", 1e9))
    rec, traj = analyse(spec, teacher, sol, rho)
    rec.update(kind=cfg["kind"], n=cfg["n"], d=cfg["d"], chi=cfg["chi"], rho=rho, sigma=sigma,
               seed=seed, mode=cfg["mode"], status=sol.status, wall=time.time() - t0)
    out = OUT / cfg["name"]
    np.savez_compressed(out / f"rho{rho}_s{sigma:g}_seed{seed}.npz", traj=traj,
                        eigs=np.array(rec.pop("eigs", [])))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("PROCS", 90)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    (OUT / cfg["name"]).mkdir(parents=True, exist_ok=True)
    jobs = []
    for rho, sigma in itertools.product(cfg["rho"], cfg["sigma"]):
        if cfg["mode"] == "natural" and "from_sweep" in cfg:
            seeds = wrong_seeds(cfg["from_sweep"], cfg["kind"], cfg["n"], cfg["d"], cfg["chi"],
                                rho, sigma, cfg.get("max_seeds", 10**9))
        else:
            seeds = list(range(cfg["seeds"]))
        jobs += [(cfg, s, sigma, rho) for s in seeds]
    print(f"{cfg['name']}: {len(jobs)} runs", flush=True)
    with Pool(args.procs) as pool:
        recs = pool.map(job, jobs, chunksize=1)
    RES.mkdir(exist_ok=True)
    df = pd.DataFrame(recs)
    df.to_csv(RES / f"{cfg['name']}.csv", index=False, float_format="%.6g")
    print(df.drop(columns=["a_mid"], errors="ignore").describe().T.to_string(), flush=True)


if __name__ == "__main__":
    main()
