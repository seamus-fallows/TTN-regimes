"""Run a sweep from a YAML config (parallel, resumable).

    uv run python -m experiments.optimizers.run experiments/optimizers/configs/<name>.yaml [--workers N]

Config layout: `name`, then `blocks`, each with `base` (problem + init settings), `grid` (lists,
expanded as a Cartesian product) and `optimizers` (one dict per optimizer setting; list-valued
entries are expanded too). Each run writes /root/outputs/optimizers/<name>/<run_id>.npz and one
JSON line to index.jsonl in that folder; existing runs are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import yaml

OUT_ROOT = Path(os.environ.get("OPTIMIZERS_OUT", "/root/outputs/optimizers"))


def expand(d: dict) -> list[dict]:
    keys = [k for k, v in d.items() if isinstance(v, list) and k not in ("weights", "patterns")]
    if not keys:
        return [dict(d)]
    out = []
    for combo in itertools.product(*(d[k] for k in keys)):
        row = dict(d)
        row.update(zip(keys, combo))
        out.append(row)
    return out


def run_id(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def jobs_from_config(path: str) -> tuple[str, list[dict]]:
    conf = yaml.safe_load(Path(path).read_text())
    jobs = []
    for block in conf["blocks"]:
        for g in expand(block.get("grid", {})):
            for opt in block["optimizers"]:
                for o in expand(opt):
                    cfg = {**block["base"], **g, **o}
                    cfg = {k: (float(v) if isinstance(v, str) and _is_number(v) else v)
                           for k, v in cfg.items()}
                    jobs.append(cfg)
    return conf["name"], jobs


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def work(args):
    cfg, out_dir = args
    import torch

    torch.set_num_threads(1)
    from experiments.optimizers.common import initial_point, make_problem
    from experiments.optimizers.optimizers import run

    rid = run_id(cfg)
    path = out_dir / f"{rid}.npz"
    if path.exists():
        return None
    p = make_problem(cfg)
    x0 = initial_point(p, cfg)
    t0 = time.time()
    arrays, info = run(p, x0, cfg)
    info["wall_seconds"] = time.time() - t0
    tmp = out_dir / f"{rid}.tmp.npz"
    np.savez_compressed(tmp, cfg=np.array(json.dumps(cfg)), **arrays)
    tmp.rename(path)
    return {"id": rid, **cfg, **info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 88)))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    name, jobs = jobs_from_config(a.config)
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out_dir / f"{run_id(j)}.npz").exists()]
    print(f"{name}: {len(jobs)} runs, {len(todo)} to do", flush=True)
    if a.dry:
        return
    # longest jobs first (rough: small sigma and many steps)
    todo.sort(key=lambda j: -(j.get("n_steps", 0) + j.get("t_max", 0)) / j.get("sigma", 1))
    with mp.get_context("fork").Pool(min(a.workers, max(len(todo), 1)), maxtasksperchild=20) as pool:
        with open(out_dir / "index.jsonl", "a") as f:
            for i, res in enumerate(pool.imap_unordered(work, [(j, out_dir) for j in todo])):
                if res is None:
                    continue
                f.write(json.dumps(res) + "\n")
                f.flush()
                print(f"[{i + 1}/{len(todo)}] {res['optimizer']} sigma={res.get('sigma')} "
                      f"seed={res.get('seed')} {res['status']} {res['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
