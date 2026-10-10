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
from analysis import law_rms, load_records, mean_se, slope_se  # noqa: E402
from bond_events import RAW, npz_path  # noqa: E402
from rules import get_tree, route_sequence  # noqa: E402
from ttn.losses import population_loss  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

BAND = 0.02


def loss_trajectory(rec, raw_dir=RAW / "pilot3", max_points: int = 10**9):
    tree = get_tree(rec["kind"], rec.get("n"))
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"],
                              seed=rec.get("teacher_seed", 1000))
    t2 = teacher.norm2()
    data = np.load(npz_path(rec, raw_dir))
    t, y = data["t"], data["y"]
    if len(t) > max_points:  # accepted steps are dense where L changes fast: uniform-in-step subsample
        idx = np.unique(np.linspace(0, len(t) - 1, max_points).astype(int))
        t, y = t[idx], y[:, idx]
    ys = torch.from_numpy(y)
    L = np.array([population_loss(spec, spec.unflatten(ys[:, i]), teacher, teacher_norm2=t2).item()
                  for i in range(ys.shape[1])])
    return t, L


def occupation(t, L, level, band=BAND):
    """Time spent with |L/level - 1| < band, with the band entry and exit times interpolated linearly in
    log L between stored integrator steps (every accepted LSODA step is stored). Returns (duration,
    visited, n_inside): n_inside = stored steps inside the band; durations resting on fewer than 3 steps
    are not resolved (analysis standard 1) and are reported as unresolved."""
    x = np.log(L / level)
    lo, hi = np.log(1 - band), np.log(1 + band)
    inside = (x > lo) & (x < hi)
    dur = 0.0
    for i in range(len(t) - 1):
        a, b = x[i], x[i + 1]
        if inside[i] and inside[i + 1]:
            dur += t[i + 1] - t[i]
            continue
        # partial interval: the segment from a to b crosses one or both band edges
        if a == b:
            continue
        lo_s, hi_s = sorted((a, b))
        overlap = max(0.0, min(hi_s, hi) - max(lo_s, lo))
        dur += (t[i + 1] - t[i]) * overlap / (hi_s - lo_s)
    n_in = int(inside.sum())
    return float(dur), bool(dur > 0), n_in


_SEQ: dict = {}


SEQ_FILE = Path(__file__).parent / "results" / "route_sequences.json"


def predicted(rec):
    """Predicted sequence: from results/route_sequences.json (predict_sequences.py) if present."""
    tree = get_tree(rec["kind"], rec.get("n"))
    key = f"{rec.get('name')}_{tree.name}"
    if not _SEQ and SEQ_FILE.exists():
        _SEQ.update(json.loads(SEQ_FILE.read_text()))
    if key not in _SEQ:
        _SEQ[key] = route_sequence(tree, rec["patterns"], rec["weights"], d=rec["d"], chi_max=rec["chi"])
    return _SEQ[key]


def measure(rec):
    try:
        t, L = loss_trajectory(rec)
    except FileNotFoundError:
        return []
    from ttn.diagnostics import find_plateaus

    # plateaus detected in L(t) by the library rule (linear time, separation ratio > 3)
    try:
        plats, _ = find_plateaus(t, L)
    except Exception:  # noqa: BLE001
        plats = []
    # plateau-midpoint losses recorded by the runner (minimum speed between component events)
    l_mids = [p["L_mid"] for p in rec.get("plateaus", []) if p.get("L_mid") is not None]
    rows = []
    for j, step in enumerate(predicted(rec)):
        if step["k_star"] == 0:
            continue
        dur, visited, n_in = occupation(t, L, step["loss"])
        # best-matching detected plateau (level within 5%) and nearest recorded midpoint loss
        match = [pl for pl in plats if abs(pl.level / step["loss"] - 1) < 0.05]
        best = min(match, key=lambda pl: abs(pl.level / step["loss"] - 1)) if match else None
        mid_err = min((abs(lm / step["loss"] - 1) for lm in l_mids), default=np.nan)
        rows.append(dict(case=rec.get("name"), tree=get_tree(rec["kind"], rec.get("n")).name,
                         n_inside=n_in, resolved=n_in >= 3,
                         start=rec.get("start", "small"), sigma=rec["sigma"], seed=rec["seed"],
                         order=">".join(str(r) for _, r in rec.get("events", [])),
                         level_j=j, level=step["loss"], k_star=step["k_star"],
                         occupation=dur, duration=dur, L_final=float(L[-1]),
                         detected=best is not None,
                         detected_duration=(best.t_end - best.t_start) if best else np.nan,
                         detected_level_err=abs(best.level / step["loss"] - 1) if best else np.nan,
                         n_detected=len(plats), L_mid_err=mid_err))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    recs = [r for r in load_records(args.inputs) if r.get("status") not in ("timeout", "error", "stalled")
            and r.get("name") and r.get("traj")]
    # warm the prediction cache once per (teacher, tree) before forking
    for r in recs:
        predicted(r)
    with Pool(args.workers) as pool:
        rows = [x for rs in pool.map(measure, recs) for x in rs]
    df = pd.DataFrame(rows)
    df.to_csv(args.out + "_runs.csv", index=False)
    summ = []
    # analysis standard 3: runs that learned the components in a different order visited different
    # saddles; fit each event order separately (orders seen in < 3 seeds are listed, not fitted)
    for (case, tree, start, order, j), g in df.groupby(["case", "tree", "start", "order", "level_j"]):
        k = int(g["k_star"].iloc[0])
        p, lg, rl, rp = [], [], [], []
        for _, gs in g.groupby("seed"):
            gs = gs[(gs["duration"] > 0) & gs["resolved"]].sort_values("sigma", ascending=False)
            if gs["sigma"].nunique() >= 3:
                x = np.log(1 / gs["sigma"].values)
                p.append(slope_se(x, np.log(gs["duration"].values))[0])
                lg.append(slope_se(x, gs["duration"].values)[0])
            if gs["sigma"].nunique() >= 4:  # fair log-vs-power comparison (analysis standard 5)
                x = np.log(1 / gs["sigma"].values)
                rl.append(law_rms(x, gs["duration"].values, "log"))
                rp.append(law_rms(x, gs["duration"].values, "power"))
        pm, pse = mean_se(p)
        lm, lse = mean_se(lg)
        # exponent of the detected plateau's duration (find_plateaus), per seed
        pdet = []
        for _, gs in g.groupby("seed"):
            gs = gs[gs["detected"] & (gs["detected_duration"] > 0)]
            if gs["sigma"].nunique() >= 3:
                pdet.append(slope_se(np.log(1 / gs["sigma"].values), np.log(gs["detected_duration"].values))[0])
        rlm, rpm = (float(np.median(rl)) if rl else np.nan), (float(np.median(rp)) if rp else np.nan)
        summ.append(dict(case=case, tree=tree, start=start, order=order, level_j=j, level=g["level"].iloc[0],
                         k_star=k, detected_frac=g["detected"].mean(), runs=len(g),
                         L_mid_err_median=g["L_mid_err"].median(),
                         p_detected=mean_se(pdet)[0], p_detected_se=mean_se(pdet)[1], detected_fit_seeds=len(pdet),
                         law=("log" if rlm < rpm else "power") if rl else "n/a",
                         unresolved=int((~g["resolved"]).sum()), seeds=g["seed"].nunique(),
                         fit_seeds=len(p), p_pred=max(k - 2, 0), p_fit=pm, p_se=pse,
                         logslope=lm, logslope_se=lse,
                         rms_log_law=rlm, rms_power_law=rpm,
                         median_duration_smallest_sigma=g[g["sigma"] == g["sigma"].min()]["duration"].median()))
    s = pd.DataFrame(summ)
    s.to_csv(args.out + "_summary.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.precision", 3)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
