import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
t = pd.read_csv(R + "twocomp_table.csv")
t["c"] = t["tree"] + " " + t["case"].str.replace("S=","") + " " + t["start"].str[:2]
cols = ["c","k_rank","logslope","logslope_se","logslope_pred","rms_log_law","rms_power_law","log_law_wins","hess_min_small","hess_neg_small","hess_exp","hess_exp_se"]
print(t[t.k_rank<=2][cols].round(4).to_string())
cols = ["c","k_rank","p_fit","hess_min_small","hess_neg_small","hess_exp","hess_exp_se","dormant_slope","dormant_slope_se","dormant_pred","floor_flags"]
print(t[t.k_rank>=3][cols].round(4).to_string())
