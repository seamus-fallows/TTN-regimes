"""Per-seed local slopes and curvature of later-gap scaling on binary8 (later_b8_summary.csv)."""
import sys
import numpy as np
import pandas as pd

R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optimizers/results/"
s = pd.read_csv(R + "later_b8_summary.csv", dtype={"order": str})
s["m"] = s.optimizer + "_" + s.lr.fillna(0).astype(str)
rows = []
for (m, seed), g in s.groupby(["m", "seed"]):
    if m.startswith("regauge"):
        continue
    g = g[g.converged]
    modal = g.order.value_counts().index[0]
    g = g[g.order == modal].sort_values("sigma")
    for q in ["t_first", "gap_0", "gap_1"]:
        res = "t_first_resolved" if q == "t_first" else q + "_resolved"
        gg = g[g[res].astype(bool) & np.isfinite(g[q])]
        if len(gg) < 4:
            continue
        x = np.log(1 / gg.sigma.values)
        y = np.log(gg[q].values)
        # linear and quadratic fits in ln-ln space
        A1 = np.vstack([np.ones_like(x), x]).T
        c1, r1, *_ = np.linalg.lstsq(A1, y, rcond=None)
        A2 = np.vstack([np.ones_like(x), x, x**2]).T
        c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
        # local slopes at smallest and largest sigma ends from quadratic
        slope_small = c2[1] + 2 * c2[2] * x.max()
        slope_large = c2[1] + 2 * c2[2] * x.min()
        # half-range fits
        xs = np.median(x)
        lo = x >= xs; hi = x <= xs
        p_small = np.polyfit(x[lo], y[lo], 1)[0] if lo.sum() >= 3 else np.nan
        p_large = np.polyfit(x[hi], y[hi], 1)[0] if hi.sum() >= 3 else np.nan
        rows.append(dict(m=m, seed=seed, q=q, order=modal, n=len(gg), p=c1[1], p_small_half=p_small,
                         p_large_half=p_large, slope_at_smallest=slope_small, slope_at_largest=slope_large))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.round(2).to_string())
print()
summ = df.groupby(["m", "q"]).agg(p=("p", "mean"), p_se=("p", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                   p_small=("p_small_half", "mean"), p_small_se=("p_small_half", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                   p_large=("p_large_half", "mean"), p_large_se=("p_large_half", lambda v: v.std(ddof=1) / np.sqrt(len(v))), n=("seed", "count"))
print(summ.round(2).to_string())
out = sys.argv[1] if len(sys.argv) > 1 else None
if out:
    df.to_csv(out + "/b8_local_slopes_per_seed.csv", index=False)
    summ.to_csv(out + "/b8_local_slopes_summary.csv")
