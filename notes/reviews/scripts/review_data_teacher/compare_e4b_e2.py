"""Reviewer: compare rerun plateau tables (analyze_ttn.py e4b_ttn; analyze_plateaus.py e2_misaligned)
with the committed ones."""

import sys

import pandas as pd

repo, copy, out = sys.argv[1:4]
pd.set_option("display.width", 250)
L = []
mine = pd.read_csv(f"{copy}/results/e4b_ttn_plateaus.csv")
theirs = pd.read_csv(f"{repo}/results/e4b_ttn_plateaus.csv")
theirs = theirs[
    (theirs.tag == "teacher_seed103")
    & theirs.sigma.isin(mine.sigma.unique())
    & theirs.seed.isin(mine.seed.unique())
]
cols = [
    "sigma",
    "seed",
    "plateau",
    "t",
    "loss",
    "ranks",
    "dist_crit",
    "dist_hsvd",
    "dist_crit_hsvd",
    "loss_crit",
    "loss_hsvd",
]
L.append(
    f"e4b teacher 103: rerun plateaus {len(mine)}, committed plateaus (same sigma/seeds) {len(theirs)}"
)
m = mine[cols].merge(
    theirs[cols], on=["sigma", "seed", "plateau"], how="outer", suffixes=("_rerun", "_comm")
)
L.append(m.round(5).to_string())
for name, d in (("rerun", mine), ("committed", theirs)):
    r1 = d[d.ranks == "[1, 1, 1]"]
    L.append(
        f"{name}: rank-1 plateaus {len(r1)}; closer to crit than to HSVD in "
        f"{(r1.dist_crit < r1.dist_hsvd).sum()}; median dist_crit {r1.dist_crit.median():.4f}, "
        f"dist_hsvd {r1.dist_hsvd.median():.4f}; all plateaus closer to crit or equal: "
        f"{(d.dist_crit <= d.dist_hsvd + 1e-6).sum()}/{len(d)}"
    )

try:
    e2m = pd.read_csv(f"{copy}/results/e2_misaligned_plateaus.csv")
    e2t = pd.read_csv(f"{repo}/results/e2_misaligned_plateaus.csv")
    e2t = e2t[(e2t.tag == e2m.tag.iloc[0]) & (e2t.tree == "mps") & e2t.seed.isin(e2m.seed)]
    c2 = [
        "seed",
        "t1",
        "t2",
        "plateau_duration",
        "plateau_rel",
        "pred_level",
        "level_rel_err",
        "cos_race_early",
        "cos_saddle_early",
        "cos_saddle_plateau",
    ]
    L.append("e2_misaligned alpha 1, theta 20, MPS-4, sigma 0.1: rerun vs committed")
    L.append(e2m[c2].merge(e2t[c2], on="seed", suffixes=("_rerun", "_comm")).round(6).T.to_string())
    p_m = pd.read_csv(f"{copy}/results/rank1_predictions_e2_misaligned.csv")
    p_t = pd.read_csv(f"{repo}/results/rank1_predictions_e2_misaligned_e4a_nonorth.csv")
    p_t = p_t[(p_t.exp == "e2_misaligned") & (p_t.tag == e2m.tag.iloc[0]) & (p_t.tree == "mps")]
    L.append("predicted rank-1 critical points, rerun:\n" + p_m.round(8).to_string())
    L.append("committed:\n" + p_t.round(8).to_string())
except FileNotFoundError as e:
    L.append(f"e2 part skipped: {e}")
txt = "\n".join(L)
print(txt)
open(out, "w").write(txt + "\n")
