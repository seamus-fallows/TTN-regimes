import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
for f in ["bond_events.csv","bond_events_f0.05.csv","bond_events_f0.03.csv"]:
    b = pd.read_csv(R + f)
    b["two"] = b["case"].str.startswith("S=")
    print(f, len(b), "match", b.match.mean().round(4), "two-comp events", b.two.sum(), "multi", (~b.two).sum())
    print(b.groupby("two").match.agg(["size","mean"]))
    # mismatches by sigma
    mm = b[~b.match]
    print("mismatch by start:", mm.groupby("start").size().to_dict(), "total by start", b.groupby("start").size().to_dict())
    # sigma distribution of mismatches
    b["sbin"] = pd.cut(b.sigma, [0,0.003,0.01,0.03,0.1,0.2,1])
    print(b.groupby("sbin").match.agg(["size","mean"]).T)
    # mismatch type: observed superset or subset
    def typ(r):
        p = set(eval(r.predicted)); o = set(eval(r.observed))
        if o == p: return "eq"
        if o < p: return "missing"
        if o > p: return "extra"
        return "both"
    b["typ"] = b.apply(typ, axis=1)
    print(b.groupby(["two","typ"]).size())
    print(b[~b.match].groupby(["tree","case","start"]).size().sort_values(ascending=False).head(15))
