"""Side-by-side: committed results/twocomp_table.csv (+ balanced) vs the same analysis on the reviewer's rerun."""

import sys

import pandas as pd

C = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
mine = pd.read_csv(sys.argv[1])
ref = pd.concat(
    [pd.read_csv(C + "twocomp_table.csv"), pd.read_csv(C + "twocomp_balanced_table.csv")]
)
cols = [
    "p_fit",
    "p_se",
    "p_local",
    "logslope",
    "logslope_se",
    "hess_min_small",
    "hess_exp",
    "hess_exp_se",
    "gap_small_med",
    "greedy",
    "other_order",
    "rotation",
    "stalled",
]
m = mine.merge(ref, on=["tree", "case", "start", "d", "chi", "w"], suffixes=("_rerun", "_commit"))
pd.set_option("display.width", 250, "display.max_columns", 60, "display.precision", 4)
for _, r in m.iterrows():
    print(f"{r['tree']:8s} {r['case']:7s} {r['start']:14s} k={r['k_rank_rerun']}")
    for c in cols:
        a, b = r[c + "_rerun"], r[c + "_commit"]
        print(f"    {c:15s} rerun {a:>12.5g}   committed {b:>12.5g}")
