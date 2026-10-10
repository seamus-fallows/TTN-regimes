"""Reviewer driver: runs the agent's committed sweep code (experiments.optimizers.run.work) on
Windows, where run.py's mp.get_context("fork") does not exist. Uses a spawn pool instead; the
per-run code path (make_problem, initial_point, optimizers.run, Trajectory.save) is the agent's.

    uv run python driver.py CONFIG [CONFIG ...] --out DIR --workers N
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
sys.path.insert(0, str(REPO))

from experiments.optimizers.run import jobs_from_config, run_id, work  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    todo = []
    for c in a.configs:
        name, jobs = jobs_from_config(c)
        out_dir = Path(a.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for j in jobs:
            if not (out_dir / f"{run_id(j)}.npz").exists():
                todo.append((j, out_dir))
        print(f"{name}: {len(jobs)} runs", flush=True)
    # longest first, as run.py
    todo.sort(
        key=lambda jo: -(jo[0].get("n_steps", 0) + jo[0].get("t_max", 0)) / jo[0].get("sigma", 1)
    )
    print(f"{len(todo)} to do", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(a.workers, max(len(todo), 1)), maxtasksperchild=20) as pool:
        for i, res in enumerate(pool.imap_unordered(work, todo)):
            if res is None:
                continue
            with open(Path(a.out) / "index_all.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
            print(
                f"[{i + 1}/{len(todo)}] {time.time() - t0:7.1f}s {res['optimizer']} lr={res.get('lr')} "
                f"sigma={res.get('sigma')} seed={res.get('seed')} {res['stop_reason']} "
                f"{res['wall_seconds']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
