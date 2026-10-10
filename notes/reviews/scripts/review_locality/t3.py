import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
t = pd.read_csv(R + "twocomp_table.csv")
t["c"] = t["tree"] + " " + t["case"].str.replace("S=","") + " " + t["start"].str[:2]
print(t[t.k_rank==2][["c","dormant_slope","dormant_slope_se","dormant_pred","gap_small_med"]].round(4).to_string())
for f in ["robust_table.csv","twocomp_balanced_table.csv","twocomp_chi4_table.csv"]:
    r = pd.read_csv(R + f)
    r["c"] = r["tree"] + " " + r["case"].str.replace("S=","") + " " + r["start"].str[:5]
    print("=====", f)
    print(r[["c","d","chi","w","k_rank","sigma_range","n_sigma","seeds","fit_seeds","p_fit","p_se","p_median","p_local","p_local_se","logslope","logslope_se","logslope_pred","rms_log_law","rms_power_law","hess_min_small","hess_exp","hess_exp_se","greedy","other_order","rotation","stalled","timeout","incomplete"]].round(3).to_string())
