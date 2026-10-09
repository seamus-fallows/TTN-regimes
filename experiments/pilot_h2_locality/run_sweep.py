"""Run a sweep of gradient-flow runs defined by a YAML config, in parallel.

    uv run python experiments/pilot_h2_locality/run_sweep.py experiments/pilot_h2_locality/configs/n4.yaml

Each run writes <tag>.json (scalars) and <tag>.npz (trajectory) to the config's `out_dir`
(default data/pilot_h2_locality/runs/). Existing runs are skipped, so a sweep can be extended or
resumed. Afterwards all runs in `out_dir` are collected into one table next to it
(data/pilot_h2_locality/<csv>, gitignored); analyze.py writes the compact summary to results/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
DEFAULT_OUT = REPO / "data" / "pilot_h2_locality" / "runs"
MAX_WORKERS = 8  # the machine is shared


def _init_worker():
    import torch

    torch.set_num_threads(1)


def _run_one(cfg_dict: dict, out_dir: str) -> tuple[str, dict]:
    from h2sim import RunConfig, simulate

    cfg = RunConfig(**cfg_dict)
    summary, arrays = simulate(cfg)
    out = Path(out_dir)
    np.savez_compressed(out / f"{cfg.tag()}.npz", **arrays)
    with open(out / f"{cfg.tag()}.json", "w") as f:
        json.dump(summary, f, indent=1)
    return cfg.tag(), summary


def expand(config: dict) -> list[dict]:
    """One dict of RunConfig fields per run."""
    common = {k: config[k] for k in ("method", "rtol", "wall_limit", "atol_factor", "t_max_factor")
              if k in config}
    runs = []
    for group in config["groups"]:
        seeds = group.get("seeds", config.get("seeds", [0]))
        for kind, sigma, seed in itertools.product(group["kinds"], group["sigmas"], seeds):
            cfg = dict(
                common,
                tree=group["tree"],
                n=group["n"],
                kind=kind,
                sigma=float(sigma),
                seed=int(seed),
            )
            cfg.update(group.get("overrides", {}))
            runs.append(cfg)
    return runs


def expected_cost(cfg: dict) -> float:
    # Longest runs first for load balancing: bigger trees and smaller sigma are slower.
    n_cores = cfg["n"] - 1 if cfg["tree"] == "binary" else cfg["n"]
    return n_cores * 10 + np.log10(1 / cfg["sigma"])


def collect(out_dir: Path, csv_path: Path) -> None:
    import pandas as pd

    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["diff_sites"] = df["diff_sites"].map(lambda s: "-".join(map(str, s)))
    df["weights"] = df["weights"].map(lambda s: "-".join(map(str, s)))
    df.to_csv(csv_path, index=False, float_format="%.10g")
    print(f"collected {len(df)} runs into {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="rerun existing runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = REPO / config["out_dir"] if "out_dir" in config else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers or config.get("workers", MAX_WORKERS), MAX_WORKERS, os.cpu_count())

    from h2sim import RunConfig

    todo = []
    for cfg in expand(config):
        tag = RunConfig(**cfg).tag()
        if args.force or not (out_dir / f"{tag}.json").exists():
            todo.append(cfg)
    todo.sort(key=expected_cost, reverse=True)
    print(f"{len(todo)} runs to do ({workers} workers) -> {out_dir}", flush=True)
    if args.dry_run:
        for cfg in todo:
            print("  ", RunConfig(**cfg).tag())
        return

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_run_one, cfg, str(out_dir)): cfg for cfg in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                tag, s = fut.result()
            except Exception as exc:  # keep the sweep going; report the failure
                print(f"[{i}/{len(todo)}] FAILED {futures[fut]}: {exc!r}", flush=True)
                continue
            print(
                f"[{i}/{len(todo)}] {tag}: {s['status']} t1={s['t_a1_0.5']:.6g} "
                f"t2={s['t_a2_0.5']:.6g} wall={s['wall_s']:.1f}s ({time.time() - t0:.0f}s total)",
                flush=True,
            )
    # full per-run table next to the raw outputs (gitignored); analyze.py writes the compact one
    collect(out_dir, out_dir.parent / config.get("csv", "runs.csv"))


if __name__ == "__main__":
    main()
