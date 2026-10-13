"""Parallel, resumable sweep: full-network runs from small init, floor-check duplicates and the
reference runs on the aligned manifold. One JSON (run, events, plateaus) + one small npz trajectory
per run under /root/outputs/route-smallinit/runs/<variant>/<target>/ (existing JSON is skipped).

usage: uv run python experiments/route_smallinit/sweep.py configs/sweep.yaml [--procs N] [--only T1,T2]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(1)
OUT = Path(os.environ.get("ROUTE_SMALLINIT_OUT", "/root/outputs/route-smallinit")) / "runs"


def sigma_grid(spec):
    lo, hi, step = spec["log10_max"], spec["log10_min"], spec["step"]
    n = int(round((lo - hi) / step))
    return [round(lo - k * step, 4) for k in range(n + 1)]


def path_of(variant, target, lsig, seed):
    tag = f"ls{lsig:+.2f}" + ("" if seed is None else f"_seed{seed:02d}")
    return OUT / variant / target / f"{tag}.json"


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def work(task):
    import run as R  # noqa: E402

    variant, cfg, lsig, seed, wall = task
    out = path_of(variant, cfg["name"], lsig, seed)
    if out.exists():
        return task, 0.0, "skip"
    sigma = 10.0**lsig
    t0 = time.time()
    with threadpool_limits(limits=1):
        try:
            if variant == "ref":
                man, traj, info = R.run_reference(cfg, sigma, wall=wall)
                run, ev, pl, small = R.reduce_run(cfg, traj, info, sigma)
            else:
                model, traj, info = R.run_full(cfg, sigma, seed, wall=wall, arith=variant)
                run, ev, pl, small = R.reduce_run(cfg, traj, info, sigma, model=model)
        except Exception as exc:  # noqa: BLE001 (record and continue)
            run, ev, pl, small = dict(status="error", message=traceback.format_exc()[-1500:]), [], [], None
            print("ERROR", variant, cfg["name"], lsig, seed, repr(exc), flush=True)
    meta = dict(variant=variant, target=cfg["name"], tree=f"{cfg['tree']}{cfg['n_sites']}", d=cfg["d"],
                sigma=sigma, log10_sigma=lsig, seed=-1 if seed is None else seed, wall=time.time() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    if small is not None:
        np.savez_compressed(out.with_suffix(".npz"), **small)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(dict(meta=meta, run=run, events=ev, plateaus=pl))))
    tmp.rename(out)
    return task, time.time() - t0, run.get("status")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=300)
    ap.add_argument("--only", default=None)
    ap.add_argument("--variants", default="ref,dense,reorder,lib")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    targets = yaml.safe_load(open(HERE / cfg["targets"]))["targets"]
    if args.only:
        targets = [t for t in targets if t["name"] in args.only.split(",")]
    sigmas = sigma_grid(cfg["sigma"])
    seeds = list(range(cfg["seeds"]))
    fcfg = cfg["floor_check"]
    reorder_sigmas = sorted(sigmas)[: fcfg["reorder_n_smallest_sigmas"]]
    lib_sigmas = sorted(sigmas)[: fcfg["lib_n_smallest_sigmas"]]
    variants = args.variants.split(",")
    tasks = []
    for t in targets:
        for ls in sigmas:
            if "ref" in variants:
                tasks.append(("ref", t, ls, None, cfg["wall"]))
            for s in seeds:
                if "dense" in variants:
                    tasks.append(("dense", t, ls, s, cfg["wall"]))
                if "reorder" in variants and ls in reorder_sigmas:
                    tasks.append(("reorder", t, ls, s, cfg["wall"]))
                if "lib" in variants and ls in lib_sigmas and s < fcfg["lib_seeds"]:
                    tasks.append(("lib", t, ls, s, fcfg["lib_wall"]))
    # main runs first, then the floor checks; within each, expensive first (small sigma, many sites)
    rank = {"ref": 0, "dense": 0, "reorder": 1, "lib": 2}
    tasks.sort(key=lambda x: (rank[x[0]], x[2], -int(x[1]["n_sites"])))
    tasks = [x for x in tasks if not path_of(x[0], x[1]["name"], x[2], x[3]).exists()]
    print(f"{len(tasks)} tasks, {args.procs} processes", flush=True)
    t0, done = time.time(), 0
    with Pool(args.procs, maxtasksperchild=50) as pool:
        for task, secs, status in pool.imap_unordered(work, tasks):
            done += 1
            if done % 50 == 0 or status not in ("done", "skip"):
                print(f"[{time.time() - t0:7.0f}s] {done}/{len(tasks)} {task[0]} {task[1]['name']} "
                      f"ls={task[2]} seed={task[3]} {status} {secs:.0f}s", flush=True)
    print("finished", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
