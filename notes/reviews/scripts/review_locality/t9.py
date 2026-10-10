import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
cls = ["greedy","other_order","rotation","stalled","timeout","incomplete","error"]
for f in ["twocomp_table.csv","twocomp_balanced_table.csv","twocomp_chi4_table.csv","robust_table.csv"]:
    t = pd.read_csv(R + f)
    t["tot"] = t[cls].sum(axis=1)
    print(f, "rows", len(t), "total runs (classes)", t.tot.sum(), "stalled", t.stalled.sum(), "rotation", t.rotation.sum(), "timeout", t.timeout.sum(), "incomplete", t.incomplete.sum())
    print("  runs==greedy+rotation:", (t.runs == t.greedy + t.rotation).mean(), " runs==greedy:", (t.runs==t.greedy).mean())
    print("  stalled by k:", t.groupby("k_rank").stalled.sum().to_dict(), " total by k:", t.groupby("k_rank").tot.sum().to_dict())
    if "start" in t:
        print("  stalled by k,start:", t.groupby(["k_rank","start"]).stalled.sum().to_dict())
t = pd.read_csv(R + "twocomp_table.csv"); t["tot"] = t[cls].sum(axis=1)
print("k<=3 runs:", t[t.k_rank<=3].tot.sum(), "k>=4 runs", t[t.k_rank>=4].tot.sum(), "k>=4 stalled", t[t.k_rank>=4].stalled.sum())
b = pd.read_csv(R + "twocomp_balanced_table.csv"); b["tot"] = b[cls].sum(axis=1)
print(b[["tree","case","k_rank","runs","greedy","rotation","stalled","tot"]].to_string())
