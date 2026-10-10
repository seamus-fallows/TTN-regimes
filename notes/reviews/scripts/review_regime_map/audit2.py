"""Audit part 2: seed fixed effects, sigma_chi_b4 table 2.1, sigma x K collapse."""

import numpy as np
import pandas as pd

R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/regime_map/results/"

b = pd.read_csv(R + "boundaries_per_seed_chi_b4.csv")
c = b[b.model == "centred"].copy()
print("=== chi_b4: per-seed slopes vs chi (seed fixed effect), mean +- SE over 6 seeds")
for col in ("sigma_c", "sigma_c_pred", "sigma_K01"):
    for cmin in (16, 32):
        sl = []
        for s, g in c[(c.chi >= cmin)].groupby("seed"):
            g = g[np.isfinite(g[col])]
            sl.append(np.polyfit(np.log(g.chi), np.log(g[col]), 1)[0])
        sl = np.array(sl)
        print(
            f"  {col} chi>={cmin}: per-seed slopes {np.round(sl, 3)} -> {sl.mean():.3f} +- {sl.std(ddof=1) / np.sqrt(len(sl)):.3f}"
        )

print(
    "\n=== sigma_chi_b4 (table 2.1): geometric mean x/ SE factor of s2s_3, n seeds, threshold ratios"
)
for model in ("centred", "plain", "twocopy"):
    st = pd.read_csv(R + f"sensitivity_per_seed_sigma_chi_b4_{model}.csv")
    print(f" --- {model}")
    for chi, g in st.groupby("chi"):
        v = g.s2s_3.dropna()
        lv = np.log(v)
        gm, sef = (
            np.exp(lv.mean()),
            np.exp(lv.std(ddof=1) / np.sqrt(len(lv))) if len(lv) > 1 else np.nan,
        )
        r25 = (g.s2s_2 / g.s2s_5).dropna()
        lz = g["lazy_0.1"].dropna()
        rl = (g["lazy_0.05"] / g["lazy_0.2"]).dropna()
        print(
            f"  chi={chi:3d}: S2S {gm:.3f} x/ {sef:.2f} (n={len(v)}/6; spread {v.min():.3f}-{v.max():.3f}) "
            f"rho2/5 {r25.mean():.2f}+-{r25.std(ddof=1) / np.sqrt(len(r25)) if len(r25) > 1 else np.nan:.2f} (n={len(r25)}) "
            f"lazy {np.exp(np.log(lz).mean()) if len(lz) else np.nan:.2f} (n={len(lz)}) dK.05/.2 {rl.mean() if len(rl) else np.nan:.2f}"
        )
