import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
for f in ["twocomp_table.csv","twocomp_balanced_table.csv","twocomp_chi4_table.csv"]:
    t = pd.read_csv(R + f)
    t = t[t.stalled > 0]
    t["partial_seed_stalls"] = (t.stalled % t.n_sigma) != 0
    print(f); print(t[["tree","case","start","k_rank","n_sigma","seeds","fit_seeds","stalled","partial_seed_stalls","p_fit","p_se"]].to_string())
