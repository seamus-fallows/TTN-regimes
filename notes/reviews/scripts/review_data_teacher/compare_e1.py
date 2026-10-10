"""Reviewer: compare the rerun step-1 paired results (copy's results/) with the committed CSVs."""

import sys

import numpy as np
import pandas as pd

repo, copy, out = sys.argv[1], sys.argv[2], sys.argv[3]
mine = pd.read_csv(f"{copy}/results/e1_paired.csv")
theirs = pd.read_csv(f"{repo}/results/e1_paired.csv")
key = ["tree", "teacher", "sigma", "seed", "other"]
m = mine.merge(theirs, on=key, suffixes=("_mine", "_theirs"))
lines = [f"matched rows: {len(m)} of {len(mine)} rerun rows"]
for col in ("t1_sigma", "t1_other", "t2_sigma", "t2_other"):
    rel = np.abs(m[f"{col}_mine"] - m[f"{col}_theirs"]) / np.abs(m[f"{col}_theirs"])
    lines.append(
        f"{col}: max rel diff rerun vs committed {rel.max():.2e} (median {rel.median():.2e})"
    )
for col in ("same_first", "same_swaps"):
    lines.append(
        f"{col}: identical in {(m[col + '_mine'] == m[col + '_theirs']).mean() * 100:.0f}% of rows"
    )


# aggregate over the rerun seeds, and over the same seeds / all 40 seeds of the committed table
def agg(df):
    return df.groupby(["tree", "teacher", "other", "sigma"]).agg(
        n=("seed", "size"),
        same_first=("same_first", "mean"),
        rel_dt1_mean=("rel_dt1", "mean"),
        rel_dt1_absmax=("rel_dt1", lambda x: x.abs().max()),
    )


a_mine = agg(mine)
seeds = sorted(mine["seed"].unique())
a_sub = agg(theirs[theirs["seed"].isin(seeds) & theirs["sigma"].isin(mine["sigma"].unique())])
a_all = pd.read_csv(f"{repo}/results/e1_paired_agg.csv").set_index(
    ["tree", "teacher", "other", "sigma"]
)
tab = a_mine.join(a_sub, rsuffix="_theirs_sameseeds").join(
    a_all[["n", "same_first", "rel_dt1_mean", "rel_dt1_absmax"]], rsuffix="_theirs_40seeds"
)
pd.set_option("display.width", 250)
lines.append(tab.round(5).to_string())
txt = "\n".join(lines)
print(txt)
open(out, "w").write(txt + "\n")
