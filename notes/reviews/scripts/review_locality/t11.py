import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
s = pd.read_csv(R + "route_amps_summary.csv")
print(s.round(3).to_string())
a = pd.read_csv(R + "route_amps.csv")
print(a.columns.tolist(), len(a))
print(a.head(3).to_string())
