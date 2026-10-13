"""Resumable process-pool sweep over the escape grid and the static orders.

Every task writes one JSON file under /root/outputs/route-escape/{escape,static}/; existing files are
skipped, so a restart resumes. One single-threaded process per worker (torch threads 1, BLAS pinned
by threadpoolctl inside the tasks, OPENBLAS/OMP/MKL env set before numpy is imported).

Usage: uv run python experiments/route_escape/sweep.py [--workers N] [--only escape|static]
       [--cells A,B] [--kinds A,B] [--wall SECONDS]
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
GRID = yaml.safe_load((HERE / "configs" / "sweep.yaml").read_text())


def escape_tasks(cells, kinds, window_only=False):
    from common import CONFIG

    lo, hi = GRID["window"]
    out = []
    for c in cells:
        if CONFIG["cells"][c]["role"] != "dyn":
            continue
        for k in kinds:
            n_dir = 1 if k == "A_bal" else GRID["directions"]
            for i in range(n_dir):
                for le in GRID["log_eps"]:
                    if window_only and not (lo - 1e-9 <= le <= hi + 1e-9):
                        continue
                    out.append(("escape", c, k, i, float(le)))
    return out


def static_tasks(cells):
    return [("static", c, s) for c in cells for s in ("Mstar", "full")]


def cost(task):
    """Rough cost order (largest first): small eps, deep routes, kind C, static first."""
    from common import get_cell

    if task[0] == "static":
        return 1e9
    _, c, k, i, le = task
    return -le * get_cell(c).c_star + (2 if k == "C" else 0)


def run_task(task):
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1, user_api="blas"):
            if task[0] == "escape":
                import escape

                _, c, k, i, le = task
                p = escape.path_for(c, k, i, le)
                if p.exists():
                    return task, "skip", 0.0
                t0 = time.time()
                o = escape.run(c, k, i, le, wall=GRID["wall"])
                escape.save(o, p)
                return task, o["status"], time.time() - t0
            import static

            _, c, s = task
            p = static.path_for(c, s)
            if p.exists():
                return task, "skip", 0.0
            t0 = time.time()
            o = static.run(c, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(o, indent=1, default=float))
            return task, "done", time.time() - t0
    except Exception:
        return task, "error: " + traceback.format_exc(limit=3), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=118)
    ap.add_argument("--only", default="all")
    ap.add_argument("--cells", default=None)
    ap.add_argument("--kinds", default="A,A_bal,B,C")
    ap.add_argument("--window-only", action="store_true", help="only the fit-window eps (used for Cp)")
    ap.add_argument("--rerun-unfinished", action="store_true",
                    help="delete and rerun every escape run that ended without an outcome (deviation 1)")
    args = ap.parse_args()
    from common import all_cells

    cells = args.cells.split(",") if args.cells else all_cells()
    tasks = []
    if args.only in ("all", "static"):
        tasks += static_tasks(cells)
    if args.only in ("all", "escape"):
        tasks += escape_tasks(cells, args.kinds.split(","), args.window_only)
    import escape
    import static

    if args.rerun_unfinished:
        for t in tasks:
            if t[0] != "escape":
                continue
            p = escape.path_for(*t[1:])
            if p.exists():
                o = json.loads(p.read_text())
                if o["t_esc"] is None and o["status"] in ("wall", "failed", "t_max"):
                    old = p.parent / "unfinished_try1"
                    old.mkdir(exist_ok=True)
                    p.rename(old / p.name)
                    print("rerun", t, o["status"], flush=True)

    todo = []
    for t in tasks:
        p = escape.path_for(*t[1:]) if t[0] == "escape" else static.path_for(*t[1:])
        if not p.exists():
            todo.append(t)
    todo.sort(key=cost, reverse=True)
    print(f"{len(tasks)} tasks, {len(todo)} to do, {args.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=50) as pool:
        for task, status, secs in pool.imap_unordered(run_task, todo, chunksize=1):
            done += 1
            print(f"[{done}/{len(todo)} {time.time() - t0:.0f}s] {task} {status} {secs:.0f}s", flush=True)
    print("finished", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
