"""Tolerance check of run classes (review s2s_core round 1, issue 4): re-run a few K = 7, 8 runs of a
sweep with tighter tolerances (rtol 1e-8, atol 1e-11 sigma instead of 1e-6 / 1e-9 sigma) and the same
budget, and compare events and class (complete / trapped / stalled / unresolved).

    uv run python experiments/s2s_core/tolcheck.py --raw /root/outputs/s2s_core/k78 --per-class 2 [--workers 12]
Outputs: /root/outputs/s2s_core/<sweep>_tight/ (raw) and results/tolcheck_<sweep>.csv.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402

HERE = Path(__file__).parent
KEYS = ("tree", "N", "d", "chi", "teacher", "weights", "teacher_seed", "init", "sigma", "seed")


def classify(path: Path) -> str:
    s = json.loads(path.read_text())
    if len(s["order"]) == len(s["weights"]):
        return "complete"
    d = np.load(path.with_suffix(".npz"))
    never = [r for r in range(len(s["weights"])) if r not in s["order"]]
    if s["order"]:
        after = d["t"] > 1.2 * s["events"][s["order"][0]]
        if after.any() and np.nanmin(d["slice_norms"][after][:, :, never]) / s["sigma"] < 1e-8:
            return "trapped"
    return "stalled" if s["stop_reason"] == "completed" else "unresolved"


def job(args):
    cfg, out = args
    s = s2s.run_one(cfg, out, hessians=False)
    return cfg["name"], s["stop_reason"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--per-class", type=int, default=2, help="runs per (tree, init, class)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True, help="output directory (e.g. experiments/s2s_core/results)")
    args = ap.parse_args()
    raw = Path(args.raw)
    out_raw = raw.parent / f"{raw.name}_tight"
    out_raw.mkdir(exist_ok=True)
    rows, picks = [], {}
    for f in sorted(raw.glob("*.json")):
        s = json.loads(f.read_text())
        if s["K"] < 7:
            continue
        key = (s["tree"] + str(s["N"]), s["init"], classify(f))
        if key[2] not in ("trapped", "stalled") or len(picks.get(key, [])) >= args.per_class:
            continue
        picks.setdefault(key, []).append(f)
    jobs = []
    for key, files in picks.items():
        for f in files:
            s = json.loads(f.read_text())
            cfg = {k: s[k] for k in KEYS}
            # 1.6x the wall limit: the tighter tolerances cost ~1.5x (checked on binary8 chi = 6).
            cfg.update(method="lsoda", rtol=1e-8, atol_rel=1e-11, wall_limit=1.6 * s.get("wall_limit", 2400),
                       max_steps=s.get("max_steps", 200000), roundoff_stop=False, t_max=s["t_max"],
                       name=f.stem)
            jobs.append((cfg, out_raw))
    with Pool(min(args.workers, len(jobs))) as pool:
        pool.map(job, jobs)
    for key, files in picks.items():
        for f in files:
            a, b = json.loads(f.read_text()), json.loads((out_raw / f.name).read_text())
            rows.append({"run": f.stem, "tree": key[0], "init": key[1], "class_loose": key[2],
                         "class_tight": classify(out_raw / f.name),
                         "order_loose": a["order"], "order_tight": b["order"],
                         "events_loose": a["events"], "events_tight": b["events"],
                         "t_end_loose": a["t_end"], "t_end_tight": b["t_end"]})
    df = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / f"tolcheck_{raw.name}.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
