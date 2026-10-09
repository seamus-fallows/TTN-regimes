"""Step 3: plateau levels and their durations for multi-component teachers.

The route-rule algorithm (`rules.route_sequence`) predicts the saddle sequence as a list of loss
levels L_j (rank-constrained optima) with escape degrees k*_j. For every run we compute L(t) on the saved
trajectory and measure the time spent within a relative band of each predicted level; that occupation
time is the plateau length at saddle j. Per seed we fit its scaling with sigma.

usage: uv run python experiments/locality/levels.py <prefix> [...] --out experiments/locality/results/levels
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from analysis import load_records, mean_se, slope_se  # noqa: E402
from bond_events import RAW, npz_path  # noqa: E402
from rules import get_tree, route_sequence  # noqa: E402
from ttn.losses import population_loss  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

BAND = 0.02


def loss_trajectory(rec, raw_dir=RAW / "pilot3"):
    tree = get_tree(rec["kind"], rec.get("n"))
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"],
                              seed=rec.get("teacher_seed", 1000))
    t2 = teacher.norm2()
    data = np.load(npz_path(rec, raw_dir))
    ys = torch.from_numpy(data["y"])
    L = np.array([population_loss(spec, spec.unflatten(ys[:, i]), teacher, teacher_norm2=t2).item()
                  for i in range(ys.shape[1])])
    return data["t"], L


def occupation(t, L, level, band=BAND):
    inside = np.abs(L / level - 1) < band
    both = inside[:-1] & inside[1:]
    return float(np.sum(np.diff(t)[both])), bool(inside.any())


_SEQ: dict = {}


def predicted(rec):
    key = json.dumps([rec["kind"], rec.get("n"), rec["patterns"], rec["weights"], rec["d"]])
    if key not in _SEQ:
        tree = get_tree(rec["kind"], rec.get("n"))
        _SEQ[key] = route_sequence(tree, rec["patterns"], rec["weights"], d=rec["d"],
                                   chi_max=rec["chi"])
    return _SEQ[key]


def measure(rec):
    try:
        t, L = loss_trajectory(rec)
    except FileNotFoundError:
        return []
    rows = []
    for j, step in enumerate(predicted(rec)):
        if step["k_star"] == 0:
            continue
        dur, visited = occupation(t, L, step["loss"])
        rows.append(dict(case=rec.get("name"), tree=get_tree(rec["kind"], rec.get("n")).name,
                         start=rec.get("start", "small"), sigma=rec["sigma"], seed=rec["seed"],
                         level_j=j, level=step["loss"], k_star=step["k_star"], visited=visited,
                         duration=dur, L_final=float(L[-1])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    recs = [r for r in load_records(args.inputs) if r.get("status") != "timeout" and r.get("name")]
    # warm the prediction cache once per (teacher, tree) before forking
    for r in recs:
        predicted(r)
    with Pool(args.workers) as pool:
        rows = [x for rs in pool.map(measure, recs) for x in rs]
    df = pd.DataFrame(rows)
    df.to_csv(args.out + "_runs.csv", index=False)
    summ = []
    for (case, tree, start, j), g in df.groupby(["case", "tree", "start", "level_j"]):
        k = int(g["k_star"].iloc[0])
        p, lg = [], []
        for _, gs in g.groupby("seed"):
            gs = gs[gs["duration"] > 0]
            if gs["sigma"].nunique() >= 3:
                x = np.log(1 / gs["sigma"].values)
                p.append(slope_se(x, np.log(gs["duration"].values))[0])
                lg.append(slope_se(x, gs["duration"].values)[0])
        pm, pse = mean_se(p)
        lm, lse = mean_se(lg)
        summ.append(dict(case=case, tree=tree, start=start, level_j=j, level=g["level"].iloc[0],
                         k_star=k, visited_frac=g["visited"].mean(), seeds=g["seed"].nunique(),
                         fit_seeds=len(p), p_pred=max(k - 2, 0), p_fit=pm, p_se=pse,
                         logslope=lm, logslope_se=lse,
                         median_duration_smallest_sigma=g[g["sigma"] == g["sigma"].min()]["duration"].median()))
    s = pd.DataFrame(summ)
    s.to_csv(args.out + "_summary.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.precision", 3)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
