"""Order-statistics sweep: P(larger component learned first) over trees, d, chi, rho, sigma, seeds.

Usage: uv run python experiments/order/sweep.py experiments/order/configs/<name>.yaml [--procs 90]

One job = one (tree, d, chi, seed). It computes, from the same unit-scale init u0:
  - per component r: init statistics (`common.init_proxies`) and, if `lo_race`, the blow-up time
    tau_r of the leading-order flow with only component r present (the exact race for odeco
    teachers on an MPS, an approximation for trees with shared cores);
  - for every (rho, sigma): full gradient flow from sigma * u0 to the first learning event, and,
    if `lo_coupled`, the coupled leading-order flow with both components (grow = `lo_grow`).
Results are appended as JSON lines to /root/outputs/order/<name>/runs.jsonl; finished jobs are
skipped on restart.
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
    gf_first_event,
    init_proxies,
    lo_blowup,
    make_spec,
    make_teacher,
    single_component,
    unit_init,
)

OUT = Path("/root/outputs/order")


def settings(cfg):
    """Distinct (kind, n, d, chi) settings; duplicates (same capped bond dims) are dropped."""
    seen, out = set(), []
    for tree, d, chi in itertools.product(cfg["trees"], cfg["d"], cfg["chi"]):
        kind, n = tree.split(":")
        spec = make_spec(kind, int(n), d, chi, cap=cfg.get("cap", True))
        key = (kind, int(n), d, spec.bond_dims)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(kind=kind, n=int(n), d=d, chi=chi, bonds=list(spec.bond_dims)))
    return out


def weights_for(rho, n_comp):
    return [rho**r for r in range(n_comp)]


def job(args):
    cfg, st, seed = args
    torch.set_num_threads(1)
    t0 = time.time()
    spec = make_spec(st["kind"], st["n"], st["d"], st["chi"], cap=cfg.get("cap", True))
    pattern = cfg.get("pattern")
    n_comp = cfg.get("n_components", 2)
    u0 = unit_init(spec, seed)
    base = make_teacher(st["n"], st["d"], weights_for(1.0, n_comp), pattern)
    rec = dict(st, seed=seed)
    if cfg.get("proxies", True):
        rec["proxies"] = init_proxies(spec, base, u0, seed=seed)
    if cfg.get("lo_race", True):
        rec["tau"] = [lo_blowup(spec, single_component(base, r), u0)[0] for r in range(n_comp)]
    runs = []
    for rho, sigma in itertools.product(cfg["rho"], cfg["sigma"]):
        teacher = make_teacher(st["n"], st["d"], weights_for(rho, n_comp), pattern)
        t1, a, nfev, status = gf_first_event(spec, teacher, sigma * u0)
        run = dict(rho=rho, sigma=sigma, t1=t1, a=a.tolist(), winner=int(np.argmax(a)),
                   nfev=int(nfev), status=int(status))
        if cfg.get("lo_coupled", False):
            grow = cfg.get("lo_grow", 1e4)
            _, amps = lo_blowup(spec, teacher, u0, grow=grow)
            run["lo_winner"] = int(np.argmax(amps / teacher.weights.numpy()))
        runs.append(run)
    rec["runs"] = runs
    rec["wall"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("PROCS", 90)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = OUT / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runs.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add((r["kind"], r["n"], r["d"], r["chi"], r["seed"]))
    seeds = range(cfg.get("seed0", 0), cfg.get("seed0", 0) + cfg["seeds"])
    jobs = [(cfg, st, s) for st in settings(cfg) for s in seeds
            if (st["kind"], st["n"], st["d"], st["chi"], s) not in done]
    print(f"{cfg['name']}: {len(jobs)} jobs to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with Pool(args.procs) as pool, open(path, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
