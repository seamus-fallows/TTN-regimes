import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
s = pd.read_csv(R + "route_amps_summary.csv")
print(s[["tree","S","start","k","end_over_sigma_median"]].to_string())
a = pd.read_csv(R + "route_amps.csv")
print(a["where"].value_counts())
# ratio midpoint/start of end amplitudes per run, median per (tree,S,start)
p = a.pivot_table(index=["tree","S","start","sigma","seed","node","end"], columns="where", values="eps").reset_index()
p = p[p.end == True]
cols = [c for c in p.columns if c not in ("tree","S","start","sigma","seed","node","end")]
print(cols)
if "mid" in p and "start" in p:
    p["ratio"] = p["mid"] / p["start"]
    print(p.groupby(["tree","S","start"]).ratio.median().round(2).to_string())
