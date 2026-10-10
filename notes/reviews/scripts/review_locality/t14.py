import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
a = pd.read_csv(R + "route_amps.csv").rename(columns={"start": "protocol"})
p = a.pivot_table(index=["tree","S","protocol","sigma","seed","node","end"], columns="where", values="eps").reset_index()
p["ratio"] = p["mid"] / p["start"]
e = p[p.end == True]
g = e.groupby(["tree","S","protocol"]).agg(med_ratio=("ratio","median"), q10=("ratio", lambda x: x.quantile(0.1)), n=("ratio","size"))
i = p[p.end == False].groupby(["tree","S","protocol"]).ratio.median().rename("interior_med_ratio")
print(g.join(i).round(3).to_string())
