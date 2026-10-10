import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
s = pd.read_csv(R + "growth_summary.csv")
print(s.to_string())
g = pd.read_csv(R + "growth_runs.csv")
print(g.columns.tolist(), len(g))
print(g.groupby("k_pred").agg(n=("g","size"), g_med=("g","median"), g_q05=("g", lambda x: x.quantile(0.05)), g_q95=("g", lambda x: x.quantile(0.95)),
      k_med=("k_est","median"), k_q05=("k_est", lambda x: x.quantile(0.05)), k_q95=("k_est", lambda x: x.quantile(0.95)), pts_med=("points","median")))
# fraction of single runs with round(k_est) == k_pred
g["kr"] = np.round(g.k_est)
print(g.groupby("k_pred").apply(lambda x: (x.kr == x.k_pred).mean()))
