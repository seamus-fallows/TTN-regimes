import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
for f in ["twocomp_table.csv","twocomp_balanced_table.csv","twocomp_chi4_table.csv","robust_table.csv"]:
    t = pd.read_csv(R + f)
    t["smin"] = t.sigma_range.str.split("-").str[0].astype(float)
    t["Q_scale_small"] = t.smin ** 2
    t["drift_over_sigma2"] = t.Qdrift_max / t.Q_scale_small
    print(f, "max Qdrift", t.Qdrift_max.max())
    print(t.sort_values("drift_over_sigma2", ascending=False)[["tree","case","start","sigma_range","Qdrift_max","drift_over_sigma2"]].head(6).to_string())
