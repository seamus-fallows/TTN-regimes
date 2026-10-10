"""Independent refit of the trap decay rate of dead route ends (mps4 chi = 3, main sweep).
Window: t > 2 x last event, slice between 1e-10 (absolute) and 1e-2 x the slice at the window start,
then (a) least squares of ln(slice) vs t, (b) median of local finite-difference rates.
Also reports the floor level (min slice over the run) and how many points sit below 1e-11."""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
rows = []
for f in sorted(glob.glob(str(REPO / "data/s2s_core/main/mps4_chi3_*.json"))):
    s = json.loads(Path(f).read_text())
    R = len(s["weights"])
    order = s["order"]
    if len(order) == R:
        continue
    d = np.load(f[:-5] + ".npz")
    t, sn = d["t"], d["slice_norms"]
    ev = [e for e in s["events"] if e is not None and np.isfinite(e)]
    never = [r for r in range(R) if r not in order]
    for r in never:
        for core in range(sn.shape[1]):
            y = sn[:, core, r]
            win = t > 2.0 * max(ev)
            tt, yy = t[win], y[win]
            if len(yy) < 5:
                continue
            y0 = yy[0]
            sel = (yy < 1e-2 * y0) & (yy > 1e-10)
            # keep only the first contiguous decaying segment
            idx = np.flatnonzero(sel)
            if len(idx) < 4:
                rows.append(
                    dict(
                        teacher=s["teacher"],
                        sigma=s["sigma"],
                        seed=s["seed"],
                        comp=r,
                        core=core,
                        rate=np.nan,
                        n=len(idx),
                        floor=float(np.nanmin(y)) if np.isfinite(y).any() else np.nan,
                    )
                )
                continue
            stop = idx[0] + np.flatnonzero(np.diff(np.r_[idx, -5]) != 1)[0] + 1
            idx = np.arange(idx[0], stop)
            a, b = np.polyfit(tt[idx], np.log(yy[idx]), 1)
            loc = -np.diff(np.log(yy[idx])) / np.diff(tt[idx])
            rows.append(
                dict(
                    teacher=s["teacher"],
                    sigma=s["sigma"],
                    seed=s["seed"],
                    comp=r,
                    core=core,
                    rate=-a,
                    rate_s2=-a / s["sigma"] ** 2,
                    loc_med_s2=np.median(loc) / s["sigma"] ** 2,
                    loc_iqr=np.subtract(*np.percentile(loc, [75, 25])) / np.median(loc),
                    n=len(idx),
                    floor=float(np.nanmin(y)),
                    n_below_1e11=int((y < 1e-11).sum()),
                    t_end=s["t_end"],
                    stop=s["stop_reason"],
                )
            )
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
df.to_csv(sys.argv[1], index=False)
end = df[df.core.isin([0, 3])]
print(end.sort_values(["teacher", "seed", "core", "sigma"]).round(5).to_string())
summ = end.groupby(["teacher", "seed", "core"]).agg(
    n_sigma=("rate_s2", "count"),
    lo=("rate_s2", "min"),
    hi=("rate_s2", "max"),
    sig_lo=("sigma", "min"),
    sig_hi=("sigma", "max"),
)
summ["spread"] = summ.hi / summ.lo
print(summ.to_string())
print("inner cores:")
print(
    df[df.core.isin([1, 2])]
    .groupby(["teacher", "seed", "core"])
    .agg(n=("rate", "count"), floor_min=("floor", "min"))
    .to_string()
)
