import pandas as pd, numpy as np, json
pd.set_option("display.width", 260, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
s = pd.read_csv(R + "levels_summary.csv")
cols = ["case","tree","start","order","level_j","level","k_star","visited_frac","runs","unresolved","seeds","fit_seeds","p_pred","p_fit","p_se","logslope","logslope_se","rms_log_law","rms_power_law","median_duration_smallest_sigma"]
print(s[cols].round(4).to_string())
