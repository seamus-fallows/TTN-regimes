"""Run the registered grid (notes/saddle_dyn.md section 3) in parallel; resumable (skips existing JSON).

Usage (from the repo root):
    uv run python experiments/saddle_dyn/sweep.py ha [--cells PS08,PS06] [--procs 180]
    uv run python experiments/saddle_dyn/sweep.py checks      # tolerance + plain-loss reruns (5.3)
    uv run python experiments/saddle_dyn/sweep.py hb [--seeds-ps08 0-39] [--seeds-b8w 0-29]
Outputs: /root/outputs/saddle_dyn/{HA,HB}/<cell>/s<seed>_x<eps>[_tag].json
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from runs import CELLS, out_path, run_ha, run_hb, save  # noqa: E402

HB_GRID = {"PS08": (-1, -3.5), "PS06": (-1, -3.5), "B8W": (-0.5, -2.5)}


def grid(lo_hi, step=0.25):
    a, b = lo_hi
    n = int(round((a - b) / step)) + 1
    return [round(a - i * step, 4) for i in range(n)]


def ha_values(cell):
    c = CELLS[cell]
    xs = [10**e for e in grid(c["grid"])]
    return xs + list(c.get("extra", []))


def window_exps(cell):
    lo, hi = CELLS[cell]["window"]
    return [e for e in grid(CELLS[cell]["grid"]) if lo - 1e-9 <= e <= hi + 1e-9]


def jobs_ha(cells, seeds=range(12)):
    out = []
    for cell in cells:
        for seed in seeds:
            for x in ha_values(cell):
                out.append(("HA", cell, seed, x, ""))
    return out


def jobs_checks(cells):
    out = []
    for cell in cells:
        w = window_exps(cell)
        tol_e = sorted({w[0], w[len(w) // 2], w[-1]})
        small = sorted(grid(CELLS[cell]["grid"]))[:2]
        for seed in range(3):
            out += [("HA", cell, seed, 10**e, "tol") for e in tol_e]
            out += [("HA", cell, seed, 10**e, "plain") for e in small]
    return out


def jobs_hb(seeds):
    out = []
    for cell, ss in seeds.items():
        for seed in ss:
            for e in grid(HB_GRID[cell]):
                out.append(("HB", cell, seed, 10**e, ""))
    return out


def work(job):
    kind, cell, seed, x, tag = job
    path = out_path(kind, cell, seed, x, tag)
    if path.exists():
        return job, "exists", 0.0
    t0 = time.time()
    try:
        if kind == "HA":
            kw = dict(rtol=1e-12, atol=1e-17) if tag == "tol" else {}
            rec = run_ha(cell, seed, x, plain=(tag == "plain"), follow=(tag == ""), **kw)
            st = rec["status"] + (f" t_esc={rec.get('t_esc_0.001')}" if rec.get("escaped") else "")
        else:
            rec = run_hb(cell, seed, x)
            st = rec["status"] + (f" dt={rec.get('dt')}" if rec.get("escaped") else "")
        save(rec, path)
        return job, st, time.time() - t0
    except Exception:
        return job, "ERROR " + traceback.format_exc(limit=3).replace("\n", " | "), time.time() - t0


def seeds_arg(s):
    a, b = (int(v) for v in s.split("-"))
    return list(range(a, b + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ha", "checks", "hb"])
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--procs", type=int, default=180)
    ap.add_argument("--seeds-ps08", default="0-39")
    ap.add_argument("--seeds-ps06", default="0-11")
    ap.add_argument("--seeds-b8w", default="0-29")
    args = ap.parse_args()
    cells = args.cells.split(",")
    if args.what == "ha":
        jobs = jobs_ha(cells)
    elif args.what == "checks":
        jobs = jobs_checks(cells)
    else:
        jobs = jobs_hb({"PS08": seeds_arg(args.seeds_ps08), "PS06": seeds_arg(args.seeds_ps06),
                        "B8W": seeds_arg(args.seeds_b8w)})
    jobs.sort(key=lambda j: j[3])  # smallest eps / sigma (longest runs) first
    print(f"{len(jobs)} jobs, {args.procs} processes", flush=True)
    t0 = time.time()
    with Pool(args.procs) as pool:
        for i, (job, st, secs) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:7.0f}s] {job[0]} {job[1]} seed={job[2]} x={job[3]:.4g} {job[4]} "
                  f"{st} ({secs:.0f}s)", flush=True)
    print("done", time.time() - t0, flush=True)
