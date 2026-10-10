"""Reviewer driver: run regime_map configs with the committed run_regime.run_point on Windows
(spawn instead of fork), outputs under a given root. Usage:
uv run python driver.py OUT_ROOT N_PROC JOBS.json
JOBS.json: list of cfg dicts (each with 'exp'). Resumable (skips run_ids already in results.jsonl)."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time

REPO = "c:/Users/seamu/Iliad-ML/TTN_regimes"
sys.path.insert(0, REPO + "/experiments/regime_map")
sys.path.insert(0, REPO)


def _init(out_root):
    import torch

    torch.set_num_threads(1)
    import run_regime

    run_regime.OUT = out_root


def _job(cfg):
    import run_regime

    t0 = time.time()
    row = run_regime.safe_run(cfg)
    row["wall_driver"] = time.time() - t0
    return row


if __name__ == "__main__":
    from multiprocessing import get_context

    import run_regime

    out_root, n_proc, jobs_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(jobs_path) as f:
        cfgs = json.load(f)
    files, done = {}, set()
    for e in sorted({c["exp"] for c in cfgs}):
        os.makedirs(os.path.join(out_root, e), exist_ok=True)
        p = os.path.join(out_root, e, "results.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    if "error" not in r:
                        done.add((e, run_regime.run_id(run_regime.row_cfg(r))))
        files[e] = open(p, "a")
    todo = [c for c in cfgs if (c["exp"], run_regime.run_id(c)) not in done]
    print(f"{len(todo)} to do, {len(done)} done", flush=True)
    with get_context("spawn").Pool(
        min(n_proc, max(1, len(todo))), initializer=_init, initargs=(out_root,), maxtasksperchild=None
    ) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, todo)):
            files[row["exp"]].write(json.dumps(row, default=float) + "\n")
            files[row["exp"]].flush()
            tag = (
                "ERROR " + row["error"][-300:]
                if "error" in row
                else f"{row.get('label')} dK={row.get('dK'):.3g} l_end={row.get('l_end'):.3g}"
            )
            print(
                time.strftime("%H:%M:%S"), f"[{i + 1}/{len(todo)}] {run_regime.run_id(row)} {tag} wall={row['wall_driver']:.1f}s",
                flush=True,
            )
    for f in files.values():
        f.close()
