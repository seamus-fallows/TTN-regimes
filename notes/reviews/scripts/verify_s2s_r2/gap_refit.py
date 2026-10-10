"""Independent per-seed gap-exponent refit from main_runs.csv with three run-set rules:
  A (agent rule)  : modal first comp / modal pair & rho1>3 / modal order & rho1,rho2>3, >=4 sigmas
  B (no rho cut)  : modal first comp / modal pair / modal order (complete), >=4 sigmas
  C (greedy only) : order 1-2-3 prefix, rho cut as A, >=4 sigmas
Prints mean +- SE and range per cell, and which seeds were dropped and why."""

import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)
r = pd.read_csv(sys.argv[1])
r = r[r.n_learned >= 1].copy()
r["order"] = r.order.astype(str)
r["first"] = r.order.str.split("-").str[0]
r["pair"] = r.order.str.split("-").str[:2].str.join("-")


def fit(sig, y):
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 4:
        return np.nan, int(ok.sum())
    return -np.polyfit(np.log(sig[ok]), np.log(y[ok]), 1)[0], int(ok.sum())


rows = []
for key, g in r.groupby(["sweep", "tree", "K", "chi", "teacher", "init", "seed"]):
    mf = g["first"].mode().iloc[0]
    two = g[g.n_learned >= 2]
    comp = g[g["class"] == "complete"]
    res = dict(zip(["sweep", "tree", "K", "chi", "teacher", "init", "seed"], key))
    t1 = g[g["first"] == mf]
    res["A_t1"], res["n_t1"] = fit(t1.sigma.values, t1.t1.values)
    for rule in "ABC":
        if len(two):
            mp = two.pair.mode().iloc[0] if rule != "C" else "1-2"
            s1 = two[two.pair == mp]
            if rule in "AC":
                s1 = s1[s1.rho1 > 3]
            res[f"{rule}_gap1"], res[f"{rule}_n1"] = fit(s1.sigma.values, (s1.t2 - s1.t1).values)
        else:
            res[f"{rule}_gap1"], res[f"{rule}_n1"] = np.nan, 0
        if len(comp):
            mo = comp.order.mode().iloc[0] if rule != "C" else "1-2-3"
            s2 = comp[comp.order == mo]
            if rule in "AC":
                s2 = s2[(s2.rho1 > 3) & (s2.rho2 > 3)]
            res[f"{rule}_gap2"], res[f"{rule}_n2"] = fit(s2.sigma.values, (s2.t3 - s2.t2).values)
        else:
            res[f"{rule}_gap2"], res[f"{rule}_n2"] = np.nan, 0
    rows.append(res)
f = pd.DataFrame(rows)
f.to_csv(sys.argv[2], index=False)


def ms(x):
    x = x.dropna()
    if not len(x):
        return ""
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return f"{x.mean():.3f}+-{se:.3f} [{x.min():.2f},{x.max():.2f}] n={len(x)}"


out = f.groupby(["sweep", "tree", "chi", "teacher", "init"]).agg(
    t1=("A_t1", ms),
    A1=("A_gap1", ms),
    A2=("A_gap2", ms),
    B1=("B_gap1", ms),
    B2=("B_gap2", ms),
    C1=("C_gap1", ms),
    C2=("C_gap2", ms),
)
print(out.to_string())
