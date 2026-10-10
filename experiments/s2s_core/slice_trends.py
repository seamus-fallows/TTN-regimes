"""Slice trends of stalled runs (review s2s_core round 1, issue 5): for every run that is not complete
and not trapped (all route ends above 1e-8 sigma), the smallest per-core slice of each never-learned
component at the end of the run (in units of sigma) and its log-slope d log(slice) / d log t over the
last decade of time (negative: still decaying; ~0: pinned; positive: growing).

    uv run python experiments/s2s_core/slice_trends.py --runs results/main_runs.csv --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = Path("/root/outputs/s2s_core")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = pd.read_csv(args.runs)
    runs = runs[runs["class"].isin(["stalled", "unresolved"])]
    rows = []
    for r in runs.rename(columns={"class": "cls"}).itertuples():
        w = "-".join(f"{x:g}" for x in ([3.0, 2.0, 1.0] if r.teacher == "separated" else [1.0, 0.85, 0.7]))
        f = Path(args.raw) / r.sweep / f"{r.tree}_chi{r.chi}_w{w}_{r.init}_s{r.sigma:g}_seed{r.seed}.json"
        s = json.loads(f.read_text())
        d = np.load(f.with_suffix(".npz"))
        t, sn = d["t"], d["slice_norms"]
        never = [q for q in range(len(s["weights"])) if q not in s["order"]]
        mins = np.nanmin(sn[:, :, never], axis=(1, 2))  # smallest dormant slice at each time
        last = t >= t[-1] / 10
        ok = last & (mins > 0)
        slope = np.polyfit(np.log(t[ok]), np.log(mins[ok]), 1)[0] if ok.sum() >= 3 else np.nan
        rows.append({"sweep": r.sweep, "tree": r.tree, "chi": r.chi, "teacher": r.teacher, "init": r.init,
                     "sigma": r.sigma, "seed": r.seed, "class": r.cls,
                     "min_slice_end_over_sigma": mins[-1] / s["sigma"], "log_slope_last_decade": slope})
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "slice_trends.csv", index=False)
    g = df.groupby(["sweep", "tree", "chi", "init"])
    print(g.agg(n=("seed", "size"), slice_median=("min_slice_end_over_sigma", "median"),
                slope_median=("log_slope_last_decade", "median"),
                frac_decaying=("log_slope_last_decade", lambda x: float(np.mean(x < -0.1))),
                frac_growing=("log_slope_last_decade", lambda x: float(np.mean(x > 0.1)))).round(3).to_string())


if __name__ == "__main__":
    main()
