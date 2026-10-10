"""Step 4: full learning order with R >= 3 odeco components (weights 1, rho, rho^2, ...).

Usage: uv run python experiments/order/multi.py experiments/order/configs/multi3.yaml [--procs N]
       [--raw DIR]

One job = (tree, d, chi, R, rho, seed). Gradient flow (LSODA with the exact Hessian, as
`ttn.dynamics.integrate_gf(method="lsoda")`, via `common.gf_events`) from sigma * u0 with exact
event location for every first crossing a_r = 1/2, stopped when the loss is below `loss_floor` x
L(0), at t_max, or (unless `stall_stop: false`) when some component's slice of some core has
fallen to 1e-14 x the largest core norm (route dead at float64 roundoff: 'stalled'). Per component
it also stores the leading-order blow-up time tau_r (only component r present, unit-scale init),
whose ranking tau_r / s_r is the race-model prediction of the order, the minimum overlap |a_r| and
the minimum slice-norm ratio before its event (roundoff flag). Results: JSON lines in
RAW/<name>/runs.jsonl (resumable).
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
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    RAW as OUT,
    default_procs,
    gf_events,
    lo_blowup,
    make_spec,
    make_teacher,
    single_component,
    single_thread,
    unit_init,
)


def job(args):
    cfg, tree, d, chi, n_comp, rho, seed = args
    single_thread()
    t0 = time.time()
    kind, n = tree.split(":")
    n = int(n)
    spec = make_spec(kind, n, d, chi, cap=cfg.get("cap", True))
    weights = [rho**r for r in range(n_comp)]
    teacher = make_teacher(n, d, weights)
    u0 = unit_init(spec, seed)
    taus = [lo_blowup(spec, single_component(teacher, r), u0)[0] for r in range(n_comp)]
    sigma = cfg["sigma"][tree] if isinstance(cfg["sigma"], dict) else cfg["sigma"]
    ev = gf_events(spec, teacher, sigma * u0, cfg["t_max"], rtol=cfg.get("rtol", 1e-9),
                   atol=cfg.get("atol", 1e-14), loss_floor=cfg.get("loss_floor", 1e-8),
                   slice_floor=cfg.get("slice_floor", 1e-14),
                   stall_stop=cfg.get("stall_stop", True))
    return dict(tree=tree, d=d, chi=chi, bonds=list(spec.bond_dims), R=n_comp, rho=rho,
                sigma=sigma, seed=seed,
                times=[None if not np.isfinite(x) else float(x) for x in ev["times"]], tau=taus,
                final_a=ev["final_a"].tolist(), final_loss=ev["final_loss"],
                min_dormant=ev["min_dormant"], min_slice=ev["min_slice"], status=ev["status"],
                t_end=ev["t_end"], n_steps=ev["n_steps"], wall=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=default_procs())
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = OUT / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add((r["tree"], r["d"], r["chi"], r["R"], r["rho"], r["seed"]))
    jobs = [(cfg, *key) for key in itertools.product(cfg["trees"], cfg["d"], cfg["chi"], cfg["R"],
                                                     cfg["rho"], range(cfg["seeds"]))
            if key not in done]
    print(f"{cfg['name']}: {len(jobs)} jobs ({len(done)} done)", flush=True)
    t0 = time.time()
    with Pool(args.procs) as pool, open(path, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
