"""Collect the per-run summaries of a sweep (run.py) into one table and summary tables.

    uv run python experiments/s2s_core/analyze_main.py [--raw /root/outputs/s2s_core/main] [--out results]

Writes <out>/main_runs.csv (one row per run) and <out>/main_summary.csv (per tree, chi, teacher,
sigma: run counts by class, S2S fraction, plateau-level errors, Hessian summary). Classes:
  complete    all components learned, every event resolved
  floor       not complete, and some dormant amplitude |<W,U_r>| fell below 1e-14 after the first
              escape (its later escape would be seeded by float64 roundoff: not physics)
  stalled     not complete, dormant amplitudes stayed above 1e-14 (run hit t_max / step / wall limit)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FLOOR = 1e-14


def load(raw: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob("*.json")):
        s = json.loads(f.read_text())
        R = len(s["weights"])
        ev = np.array([np.nan if x is None else x for x in s["events"]], dtype=float)
        order = s["order"]
        t_sorted = [ev[r] for r in order] + [np.nan] * (R - len(order))
        w = np.array([np.nan if x is None else x for x in s["widths"]], dtype=float)
        complete = len(order) == R and np.all(np.isfinite(ev))
        # Smallest dormant amplitude after the first escape (intervals 1..), over components that are
        # still dormant then.
        after = [v for row in s["dormant_min_between_events"][1:] for v in row.values()]
        min_after = min(after) if after else np.nan
        cls = "complete" if complete else ("floor" if np.isfinite(min_after) and min_after < FLOOR else "stalled")
        plateaus = s["plateaus"]
        rel_level = [abs(p["loss"] - p["partial_fit_level"]) / p["partial_fit_level"] for p in plateaus]
        lam_min = [p["saddle"]["lambda_min"] for p in plateaus if "saddle" in p]
        lam_max = [p["saddle"]["lambda_max"] for p in plateaus if "saddle" in p]
        n_neg = [p["saddle"]["n_negative"] for p in plateaus if "saddle" in p]
        transient = [r for r in range(R) if np.isfinite(ev[r]) and s["first_crossings"][r] is not None
                     and s["first_crossings"][r] < ev[r] * (1 - 1e-6)]
        row = {
            "tree": f"{s['tree']}{s['N']}", "K": s["K"], "chi": s["chi"], "teacher": s["teacher"],
            "sigma": s["sigma"], "seed": s["seed"], "init": s["init"], "class": cls,
            "stop": s["stop_reason"], "n_learned": len(order), "order": "".join(map(str, order)),
            "greedy": order == sorted(order) and len(order) == R,
            "t1": t_sorted[0], "t2": t_sorted[1], "t3": t_sorted[2] if R > 2 else np.nan,
            "w_max": np.nanmax(w) if np.isfinite(w).any() else np.nan,
            "rho_min": min(s["separation_ratios"]) if s["separation_ratios"] else np.nan,
            "s2s": s["s2s"], "relearned": ",".join(map(str, transient)),
            "max_rel_plateau_level_err": max(rel_level) if rel_level else np.nan,
            "plateau1_lam_min": lam_min[0] if lam_min else np.nan,
            "plateau2_lam_min": lam_min[1] if len(lam_min) > 1 else np.nan,
            "plateau1_lam_max": lam_max[0] if lam_max else np.nan,
            "plateau1_n_neg": n_neg[0] if n_neg else np.nan,
            "plateau2_n_neg": n_neg[1] if len(n_neg) > 1 else np.nan,
            "min_dormant_after_first": min_after,
            "drift": s["max_conservation_drift"], "q0_max": s["q0_max"], "wall": s["wall_time"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["tree", "K", "chi", "teacher", "sigma"])
    return g.agg(
        n=("seed", "size"),
        complete=("class", lambda c: int((c == "complete").sum())),
        floor=("class", lambda c: int((c == "floor").sum())),
        stalled=("class", lambda c: int((c == "stalled").sum())),
        s2s=("s2s", "sum"),
        greedy=("greedy", "sum"),
        rho_min_median=("rho_min", "median"),
        level_err_max=("max_rel_plateau_level_err", "max"),
        drift_max=("drift", "max"),
    ).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/outputs/s2s_core/main")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.raw))
    df.to_csv(out / f"{args.tag}_runs.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / f"{args.tag}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(summ.to_string())


if __name__ == "__main__":
    main()
