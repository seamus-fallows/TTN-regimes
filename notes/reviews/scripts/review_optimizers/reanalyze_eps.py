"""Recompute the committed eps-law ratios with event-located GF times (indep_eps_balanced output) instead of
the grid-interpolated deep-sweep T_GF(0.3); also with T_GF measured directly at sigma (no (0.3/sigma)^6).
    uv run python reanalyze_eps.py EXACTCSV"""

import sys

import numpy as np
import pandas as pd

R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optimizers/results/"
pd.set_option("display.width", 250)
ex = pd.read_csv(sys.argv[1]).set_index(["seed", "sigma"])["T_exact"]
com = pd.read_csv(R + "eps_mps8_prediction_per_seed.csv")
summ = pd.read_csv(R + "eps_mps8_summary.csv", dtype={"order": str})
sign = summ[summ.eps == 1e-30].groupby(["lr", "seed"])["t_first"].median()
deep = pd.read_csv(R + "deep_summary.csv", dtype={"order": str}, low_memory=False)
grid = deep[
    (deep.optimizer == "gf")
    & (deep.tree == "mps")
    & (deep.teacher == "odeco")
    & (deep.init == "balanced")
    & (deep.sigma == 0.3)
].set_index("seed")["t_first"]
rows = []
for _, r in com.dropna(subset=["measured"]).iterrows():
    s, sd, lr = r["sigma"], int(r["seed"]), r["lr"]
    ns = sign[(lr, sd)]
    t03 = ex[(sd, 0.3)]
    pred_ex = 1e-8 / lr * t03 * (0.3 / s) ** 6 + ns
    pred_direct = 1e-8 / lr * ex[(sd, s)] + ns if (sd, s) in ex.index else np.nan
    rows.append(
        dict(
            lr=lr,
            sigma=s,
            seed=sd,
            ratio_committed=r["ratio"],
            ratio_exactT03=r["measured"] / pred_ex,
            ratio_directT=r["measured"] / pred_direct,
            T03_grid=grid[sd],
            T03_exact=t03,
            T03_grid_err=grid[sd] / t03 - 1,
        )
    )
t = pd.DataFrame(rows).sort_values(["lr", "sigma", "seed"])
print(t.round(4).to_string())
for c in ("ratio_committed", "ratio_exactT03", "ratio_directT"):
    for smax in (0.03, 0.05):
        v = t[t.sigma <= smax][c].dropna()
        print(f"{c:16s} sigma<={smax}: n={len(v)} min {v.min():.4f} max {v.max():.4f}")
t.to_csv(sys.argv[1].replace(".csv", "_reanalysis.csv"), index=False)
