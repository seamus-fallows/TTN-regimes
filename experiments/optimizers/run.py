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


def resolve_budget(cfg: dict) -> dict:
    """Optional budget in gradient-flow time units: budget = c * sigma^-power (keys budget_c,
    budget_power). Sets t_max for gf and n_steps = budget / (lr / (1 - momentum)) for GD-type
    methods (Adam-type methods and sweeps need explicit n_steps / n_sweeps)."""
    if "budget_c" not in cfg:
        return cfg
    budget = cfg["budget_c"] * cfg["sigma"] ** (-cfg.get("budget_power", 0.0))
    cfg = dict(cfg)
    if cfg["optimizer"] == "gf":
        cfg["t_max"] = float(budget)
    elif "n_steps" not in cfg:
        lr_eff = cfg["lr"] / (1 - cfg.get("beta", 0.0)) if cfg["optimizer"] == "momentum" else cfg["lr"]
        cfg["n_steps"] = int(min(budget / lr_eff, cfg.get("max_steps", 2_000_000)))
    return cfg


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
                    jobs.append(resolve_budget(cfg))
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
    traj = run(p, x0, cfg)
    tmp = traj.save(out_dir / f"{rid}.tmp.npz")
    tmp.rename(path)
    m = traj.meta
    return {"id": rid, **cfg, "stop_reason": m.get("stop_reason"), "wall_seconds": m["wall_seconds"],
            "cpu_seconds": m["cpu_seconds"]}


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
                      f"seed={res.get('seed')} {res['stop_reason']} {res['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
