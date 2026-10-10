import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
for f in ["robust_table.csv","twocomp_balanced_table.csv","twocomp_chi4_table.csv"]:
    r = pd.read_csv(R + f)
    r["c"] = r["tree"] + " " + r["case"].str.replace("S=","") + " " + r["start"].str[:5]
    print("=====", f)
    cols = ["c","d","chi","w","k_rank","sigma_range","n_sigma","seeds","fit_seeds","p_fit","p_se","p_median","p_local","p_local_se","logslope","logslope_se","logslope_pred","hess_min_small","hess_exp","hess_exp_se","greedy","other_order","rotation","stalled","timeout","incomplete"]
    print(r[[c for c in cols if c in r]].round(3).to_string())
