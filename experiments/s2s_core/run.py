"""Run a sweep from a YAML config (until `ttn.runner` lands): every combination of trees, bond
dimensions, teachers, sigmas and seeds, one process per run, resumable (finished runs are skipped).

    uv run python experiments/s2s_core/run.py experiments/s2s_core/configs/<config>.yaml [--workers 60]

Raw outputs (trajectory .npz + summary .json per run) go to /root/outputs/s2s_core/<config name>/.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import itertools
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402


def expand(config: dict) -> list[dict]:
    """All run configs of a sweep. `sigmas` maps 'treeN' (e.g. 'binary8') to a list."""
    runs = []
    for (tree, n), chi, (tname, weights), init, seed in itertools.product(
        config["trees"], config["chis"], config["teachers"].items(), config["inits"], config["seeds"]
    ):
        for sigma in config["sigmas"][f"{tree}{n}"]:
            cfg = {
                "tree": tree, "N": n, "d": config["d"], "chi": chi, "teacher": tname,
                "weights": [float(w) for w in weights], "teacher_seed": config["teacher_seed"],
                "init": init, "sigma": float(sigma), "seed": seed,
                **config.get("integrator", {}),
            }
            if init == "aligned":
                cfg["a0"] = config["a0"][f"{tree}{n}"]
            cfg["name"] = s2s.name_of(cfg)
            runs.append(cfg)
    return runs


def work(args):
    cfg, out_dir, hessians = args
    try:
        s = s2s.run_one(cfg, out_dir, hessians=hessians)
        return cfg["name"], s["stop_reason"], s["wall_time"], None
    except Exception:
        return cfg["name"], "error", 0.0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--raw", default=str(s2s.OUT), help="parent directory of the raw outputs")
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.raw) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [c for c in expand(config) if not (out_dir / f"{c['name']}.json").exists()]
    # Longest runs first (large K, small sigma) so the pool drains evenly.
    runs.sort(key=lambda c: -(len(c["tree"]) + c["N"]) / c["sigma"])
    print(f"{config['name']}: {len(runs)} runs to do -> {out_dir}", flush=True)
    t0 = time.time()
    jobs = [(c, out_dir, config.get("hessians", True)) for c in runs]
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for i, (name, stop, wall, err) in enumerate(pool.imap_unordered(work, jobs), 1):
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {name}: {stop} ({wall:.0f}s)", flush=True)
            if err:
                print(err, flush=True)
                (out_dir / f"{name}.error").write_text(err)
    print("done", json.dumps({"n": len(jobs), "wall": time.time() - t0}), flush=True)


if __name__ == "__main__":
    main()
