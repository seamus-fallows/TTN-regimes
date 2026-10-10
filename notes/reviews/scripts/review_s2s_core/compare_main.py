"""Compare the reviewer's rerun tables (analyze_main.py on data/review_s2s_core/raw/*) with the committed
experiments/s2s_core/results/main_runs.csv, run by run, and per-seed exponents on the same sigmas."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("C:/Users/seamu/Iliad-ML/TTN_regimes")
ANA = REPO / "data/review_s2s_core/analysis"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

com = pd.read_csv(REPO / "experiments/s2s_core/results/main_runs.csv")
mine = pd.concat(
    [pd.read_csv(f) for f in sorted(ANA.glob("*_runs.csv")) if f.name.split("_")[0] in ("main", "balanced", "large", "k78")],
    ignore_index=True,
)
keys = ["sweep", "tree", "chi", "teacher", "init", "sigma", "seed"]
m = mine.merge(com, on=keys, suffixes=("", "_c"), how="left")
print(f"{len(mine)} rerun rows, {m['class_c'].notna().sum()} matched to committed rows")

cols = []
for c in ("t1", "t2", "t3"):
    m[f"{c}_rel"] = (m[c] - m[f"{c}_c"]) / m[f"{c}_c"]
m["lam1_ratio"] = m["plateau1_lam_min"] / m["plateau1_lam_min_c"]
m["lam2_ratio"] = m["plateau2_lam_min"] / m["plateau2_lam_min_c"]
show = [
    "sweep",
    "tree",
    "chi",
    "init",
    "sigma",
    "seed",
    "class",
    "class_c",
    "order",
    "order_c",
    "t1",
    "t1_rel",
    "t2_rel",
    "t3_rel",
    "max_rel_plateau_level_err",
    "max_rel_plateau_level_err_c",
    "max_rel_plateau_spectrum_err",
    "max_rel_plateau_spectrum_err_c",
    "edge_rank_change_all_one",
    "edge_rank_change_all_one_c",
    "lam1_ratio",
    "lam2_ratio",
    "s2s",
    "s2s_c",
    "min_dormant_slice_over_sigma",
    "min_dormant_slice_over_sigma_c",
    "wall",
    "wall_c",
]
out = m[show].sort_values(["sweep", "tree", "chi", "init", "seed", "sigma"])
out.to_csv(ANA / "compare_runs.csv", index=False)
print(out.to_string(float_format=lambda x: f"{x:.3g}"))

print("\nclass agreement:", (m["class"] == m["class_c"]).sum(), "/", m["class_c"].notna().sum())
print("order agreement:", (m["order"].astype(str) == m["order_c"].astype(str)).sum())
for c in ("t1", "t2", "t3"):
    x = m[f"{c}_rel"].abs().dropna()
    if len(x):
        print(f"{c}: max |rel diff| = {x.max():.2e}, median {x.median():.2e}, n = {len(x)}")


def fit(g, col):
    x = np.log(g.sigma.values)
    y = g[col].values
    ok = np.isfinite(y) & (y > 0)
    return -np.polyfit(x[ok], np.log(y[ok]), 1)[0] if ok.sum() >= 3 else np.nan


rows = []
for key, g in m[m["class"] == "complete"].groupby(["sweep", "tree", "chi", "init", "seed"]):
    g = g.sort_values("sigma")
    gc = g.copy()
    for c in ("t1", "t2", "t3"):
        gc[c] = g[f"{c}_c"]
    r = dict(zip(["sweep", "tree", "chi", "init", "seed"], key))
    r["n_sigma"] = len(g)
    for tag, d in (("rerun", g), ("committed_same_sigmas", gc)):
        d = d.assign(gap1=d.t2 - d.t1, gap2=d.t3 - d.t2)
        r[f"p_t1_{tag}"] = fit(d, "t1")
        r[f"p_gap1_{tag}"] = fit(d, "gap1")
        r[f"p_gap2_{tag}"] = fit(d, "gap2")
    rows.append(r)
ex = pd.DataFrame(rows)
ex.to_csv(ANA / "compare_exponents.csv", index=False)
print(ex.to_string(float_format=lambda x: f"{x:.4f}"))
