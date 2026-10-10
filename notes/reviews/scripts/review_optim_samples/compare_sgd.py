"""Compare the reviewer's SGD first-plateau re-run (new noise seeds) with the committed sgd_summary.csv."""

import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
COMMITTED = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
mine = pd.read_csv(sys.argv[1] + "/sgd_summary.csv")
old = pd.read_csv(COMMITTED + "sgd_summary.csv")
old = old[
    (old.block == "B_sweep")
    & (old.init == "fanin")
    & (old.sigma == 0.1)
    & (old.seed <= 2)
    & old.B.isin([1, 4, 16, 64, 256])
]
key = ["seed", "B"]
m = mine.merge(old, on=key, suffixes=("_new", "_old"))
cols = [
    "seed",
    "B",
    "t1_gf_new",
    "t1_gf_old",
    "shift1_new",
    "shift1_se_new",
    "shift1_old",
    "shift1_se_old",
    "t1_std_new",
    "t1_std_old",
    "pred_std1_new",
    "t1_n_new",
    "t1_n_old",
]
print(m[cols].to_string(float_format=lambda v: f"{v:.4g}"))
# pooled over inits (mean of per-init shifts, s.e. over inits as in the note)
for name, d, suf in (("re-run", m, "_new"), ("committed", m, "_old")):
    g = d.groupby("B")[f"shift1{suf}"].agg(["mean", "sem"])
    print(
        name,
        "mean shift over 3 inits by B:",
        {int(b): f"{r['mean']:+.3f}+-{r['sem']:.3f}" for b, r in g.iterrows()},
    )
# exponent of std(t1) vs eta/B per seed
for name, suf in (("re-run", "_new"), ("committed", "_old")):
    ps = []
    for seed, g in m.groupby("seed"):
        x = np.log(0.0005 / g.B.values)
        y = np.log(g[f"t1_std{suf}"].values)
        ps.append(np.polyfit(x, y, 1)[0])
    print(
        name,
        "per-seed exponent p:",
        np.round(ps, 3),
        f"mean {np.mean(ps):.3f} +- {np.std(ps, ddof=1) / np.sqrt(len(ps)):.3f}",
    )
