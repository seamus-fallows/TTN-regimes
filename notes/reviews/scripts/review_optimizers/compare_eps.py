"""eps-law test on the reproduced runs (agent code) vs the committed per-seed table, plus the independent
fan-in check (indep/eps.csv).  uv run python compare_eps.py"""

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

pd.set_option("display.width", 250)

eps = collect("eps_mps8", refresh=True)
eps["lr"] = eps["lr"].astype(float).round(8); eps["eps"] = eps["eps"].astype(float)
deep = collect("deep", refresh=True)
ref = deep[
    (deep["optimizer"] == "gf")
    & (deep["tree"] == "mps")
    & (deep["teacher"] == "odeco")
    & (deep["init"] == "balanced")
    & (deep["sigma"] == 0.3)
].set_index("seed")["t_first"]
sign = eps[eps["eps"] == 1e-30].set_index(["lr", "seed"])["t_first"]
com = pd.read_csv(REPO / "experiments/optimizers/results/eps_mps8_prediction_per_seed.csv")
rows = []
for _, r in eps[eps["eps"] == 1e-8].iterrows():
    pred = 1e-8 / r["lr"] * ref[r["seed"]] * (0.3 / r["sigma"]) ** 6 + sign[(r["lr"], r["seed"])]
    c = com[
        (np.isclose(com["lr"], r["lr"]))
        & (np.isclose(com["sigma"], r["sigma"]))
        & (com["seed"] == r["seed"])
    ]
    rows.append(
        dict(
            lr=r["lr"],
            sigma=r["sigma"],
            seed=r["seed"],
            measured=r["t_first"],
            predicted=pred,
            ratio=r["t_first"] / pred,
            committed_measured=c["measured"].iloc[0],
            committed_ratio=c["ratio"].iloc[0],
            T_GF03=ref[r["seed"]],
            n_sign=sign[(r["lr"], r["seed"])],
        )
    )
t = pd.DataFrame(rows).sort_values(["lr", "sigma", "seed"])
print(t.round(4).to_string())
t.to_csv(OUT / "results" / "cmp_eps_per_seed.csv", index=False)

ind = pd.read_csv(OUT / "indep" / "eps.csv")
gf = ind[ind.method == "gf"].set_index(["seed", "sigma"])["t_first"]
ad = ind[ind.method == "adam"]
rows = []
for seed in sorted(ad.seed.unique()):
    nsign = ad[(ad.seed == seed) & (ad.eps == 1e-30)]["t_first"].iloc[0]
    for _, r in ad[(ad.seed == seed) & (ad.eps == 1e-8)].iterrows():
        direct = 1e-5 * gf[(seed, r["sigma"])] + nsign
        extrap = 1e-5 * gf[(seed, 0.3)] * (0.3 / r["sigma"]) ** 6 + nsign
        rows.append(
            dict(
                seed=seed,
                sigma=r["sigma"],
                n1=r["t_first"],
                n_sign=nsign,
                T_GF=gf[(seed, r["sigma"])],
                T_GF_extrap=gf[(seed, 0.3)] * (0.3 / r["sigma"]) ** 6,
                ratio_direct=r["t_first"] / direct,
                ratio_extrap=r["t_first"] / extrap,
            )
        )
print(pd.DataFrame(rows).round(4).to_string())
pd.DataFrame(rows).to_csv(OUT / "results" / "indep_eps_ratios.csv", index=False)
