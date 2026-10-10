"""Claim 3 on the reproduced runs: sigma-(in)dependence of event times / convergence of the gauge-aware
methods, per seed (max relative spread across sigma), plus GD / heavy ball vs GF per run (claim 4).
    uv run python compare_invariant.py"""

import os
import sys
from pathlib import Path

REPO = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
OUT = REPO / "data/review_optimizers"
os.environ["OPTIMIZERS_OUT"] = str(OUT / "raw")
os.environ["OPTIMIZERS_RESULTS"] = str(OUT / "results")
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.optimizers.analysis import collect  # noqa: E402
from experiments.optimizers.plots import with_keys  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

rows = []
for name, keys in (("scaling_b4", ["seed"]), ("deep", ["tree", "teacher", "seed"])):
    df = with_keys(collect(name))
    df = df[df["optimizer"].isin(["als", "dmrg", "riemannian", "regauge", "precond"])]
    for (m, *k), g in df.groupby(["method"] + keys):
        r = {
            "sweep": name,
            "method": m,
            **dict(zip(keys, k)),
            "n_sigma": g["sigma"].nunique(),
            "stop": "/".join(sorted(g["meta_stop_reason"].astype(str).unique())),
            "orders": "/".join(sorted(g["order"].astype(str).unique())),
        }
        for q in ("t_0", "t_1", "t_2", "t_loss_0.001", "meta_n_steps", "meta_n_micro"):
            if q in g and g[q].notna().any():
                v = g[q].astype(float)
                r[q + "_spread"] = (v.max() - v.min()) / abs(v.median()) if v.median() else np.nan
                r[q + "_med"] = v.median()
        rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(OUT / "results" / "cmp_invariant.csv", index=False)
agg = t.groupby(["sweep", "method"]).agg(
    n=("n_sigma", "count"),
    stop=("stop", lambda x: " | ".join(sorted(set(x)))),
    t0_spread_max=("t_0_spread", "max"),
    t2_spread_max=("t_2_spread", "max"),
    tconv_spread_max=("t_loss_0.001_spread", "max"),
    tconv_med=("t_loss_0.001_med", "median"),
)
print(agg.round(6).to_string())

# claim 4: GD / heavy ball vs GF, same (sigma, seed), GF clock
df = with_keys(collect("scaling_b4"))
gf = df[df.optimizer == "gf"].set_index(["sigma", "seed"])
out = []
for m in ("gd", "momentum"):
    g = df[df.optimizer == m]
    for _, r in g.iterrows():
        ref = gf.loc[(r["sigma"], r["seed"])]
        for q in ("t_0", "t_1", "t_2"):
            out.append(
                dict(
                    method=m,
                    sigma=r["sigma"],
                    seed=r["seed"],
                    q=q,
                    ratio=r[q] * r["clock"] / ref[q],
                    same_order=str(r["order"]) == str(ref["order"]),
                )
            )
o = pd.DataFrame(out)
o.to_csv(OUT / "results" / "cmp_gd_hb_vs_gf.csv", index=False)
print(
    o.groupby(["method", "q"])["ratio"].agg(["count", "min", "median", "max"]).round(4).to_string()
)
print(o.groupby(["method", "sigma"])["ratio"].median().unstack(0).round(4).to_string())
print("same order:", o.groupby("method")["same_order"].mean().to_dict())
