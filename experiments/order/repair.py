"""Re-integrate first-event runs that stopped with an integrator failure (status -1).

Usage: uv run python experiments/order/repair.py <name> [--procs N] [--raw DIR]
Runs of RAW/<name>/runs.jsonl with status != 1 are re-run with `common.gf_first_event` (clock
restarts when the step size reaches the float64 resolution of t; see its docstring). The original
file is kept as runs_orig.jsonl and runs.jsonl is rewritten with the repaired runs (each marked
"repaired": true).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW, gf_first_event, make_spec, make_teacher, unit_init  # noqa: E402


def fix(args):
    rec, cfg_pattern, n_comp = args
    torch.set_num_threads(1)
    spec = make_spec(rec["kind"], rec["n"], rec["d"], rec["chi"])
    u0 = unit_init(spec, rec["seed"])
    for run in rec["runs"]:
        if run["status"] == 1:
            continue
        teacher = make_teacher(rec["n"], rec["d"], [run["rho"] ** r for r in range(n_comp)],
                               cfg_pattern)
        t1, a, nfev, status = gf_first_event(spec, teacher, run["sigma"] * u0)
        run.update(t1=t1, a=a.tolist(), winner=int(np.argmax(a)), nfev=int(nfev),
                   status=int(status), repaired=True)
    return rec


def main():
    name = sys.argv[1]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else os.cpu_count()
    path = RAW / name / "runs.jsonl"
    recs = [json.loads(line) for line in open(path)]
    bad = [r for r in recs if any(run["status"] != 1 for run in r["runs"])]
    print(f"{name}: {len(bad)} jobs with failed runs", flush=True)
    with Pool(procs) as pool:
        fixed = {(r["kind"], r["n"], r["d"], r["chi"], r["seed"]): r
                 for r in pool.map(fix, [(r, None, 2) for r in bad], chunksize=1)}
    shutil.copy(path, path.with_name("runs_orig.jsonl"))
    with open(path, "w") as f:
        for r in recs:
            r = fixed.get((r["kind"], r["n"], r["d"], r["chi"], r["seed"]), r)
            f.write(json.dumps(r) + "\n")
    left = sum(run["status"] != 1 for r in fixed.values() for run in r["runs"])
    print(f"repaired; runs still failing: {left}", flush=True)


if __name__ == "__main__":
    main()
