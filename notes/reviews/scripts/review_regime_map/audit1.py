"""Audit of committed regime_map CSVs (no raw data needed)."""

import numpy as np
import pandas as pd

R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/regime_map/results/"


def slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    cov = (r @ r) / max(len(x) - 2, 1) * np.linalg.inv(A.T @ A)
    return coef[0], np.sqrt(cov[0, 0]), len(x)


def slope_seedmeans(df, col, cmin):
    """slope fitted to per-chi geometric means (the per-chi seed-mean, SE from residual)"""
    g = (
        df[(df.chi >= cmin) & np.isfinite(df[col])]
        .groupby("chi")[col]
        .apply(lambda v: np.exp(np.mean(np.log(v))))
    )
    return slope(g.index.values, g.values)


b = pd.read_csv(R + "boundaries_per_seed_chi_b4.csv")
c = b[b.model == "centred"].copy()
print(
    "=== chi_b4 centred: S2S boundary slopes (label-based sigma_c, quantized grid ratio",
    round((4 / 0.004) ** (1 / 21), 3),
    ")",
)
for cmin in (9, 12, 16, 24, 32):
    s = c[c.chi >= cmin]
    print(
        f" chi>={cmin}: label sigma_c slope {slope(s.chi, s.sigma_c)[0]:.3f} +- {slope(s.chi, s.sigma_c)[1]:.3f} (n={len(s)});"
        f" pred slope {slope(s.chi, s.sigma_c_pred)[0]:.3f} +- {slope(s.chi, s.sigma_c_pred)[1]:.3f};"
        f" lazy slope {slope(s.chi[np.isfinite(s.sigma_K01)], s.sigma_K01.dropna())[0]:.3f} +- {slope(s.chi[np.isfinite(s.sigma_K01)], s.sigma_K01.dropna())[1]:.3f}"
    )
print(" per-chi-mean fits (6 seeds averaged first):")
for cmin in (16, 32):
    for col in ("sigma_c", "sigma_c_pred", "sigma_K01"):
        sl, se, n = slope_seedmeans(c, col, cmin)
        print(f"  chi>={cmin} {col}: {sl:.3f} +- {se:.3f} (n_chi={n})")

print("\n=== sigma_c*sqrt(chi): medians of measured (label) vs predicted (aligned)")
for chi, g in c.groupby("chi"):
    m = np.median(g.sigma_c) * np.sqrt(chi)
    gm = np.exp(np.mean(np.log(g.sigma_c))) * np.sqrt(chi)
    p = np.median(g.sigma_c_pred) * np.sqrt(chi)
    lz = np.median(g.sigma_K01.dropna()) * np.sqrt(chi)
    lzv = g.sigma_K01.dropna().values * np.sqrt(chi)
    print(
        f" chi={chi:4d}: measured median {m:.3f} geo {gm:.3f} | pred median {p:.3f} | lazy*sqrt(chi) median {lz:.2f}"
        f" mean {lzv.mean():.2f} +- {lzv.std(ddof=1) / np.sqrt(len(lzv)):.2f} (n={len(lzv)}) ratio to 6.50: {lz / 6.5:.3f}"
    )

print("\n=== chi_b4 per-seed pred/measured ratios")
c["ratio"] = c.sigma_c_pred / c.sigma_c
print(c.groupby("chi").ratio.agg(["min", "median", "max"]).round(2).to_string())
print(" outliers |log ratio| > log 1.5:")
print(
    c[np.abs(np.log(c.ratio)) > np.log(1.5)][
        ["chi", "seed", "sigma_c", "sigma_c_pred", "rho_slope", "n_s2s"]
    ].to_string()
)
print(" rho_slope per chi (mean, sd, min, max):")
print(c.groupby("chi").rho_slope.agg(["mean", "std", "min", "max"]).round(3).to_string())

# competing-law test: slope -1/2 vs -1/3
print(
    "\n=== competing laws, chi>=16 label-based S2S slope and lazy slope: distance from -1/2 and -1/3 in SE"
)
for cmin in (16, 32):
    s = c[c.chi >= cmin]
    for col in ("sigma_c", "sigma_c_pred", "sigma_K01"):
        ss = s[np.isfinite(s[col])]
        sl, se, n = slope(ss.chi, ss[col])
        print(
            f"  chi>={cmin} {col}: {sl:.3f}+-{se:.3f}: (sl+1/2)/se={(sl + 0.5) / se:.1f}, (sl+1/3)/se={(sl + 1 / 3) / se:.1f}"
        )

# lazy boundary vs theory P5 bracket values
print("\n=== lazy boundary medians vs theory_regimes P5 brackets")
for chi, pr in ((32, (1.10, 1.19)), (64, (0.81, 0.83)), (128, (0.57, 0.58))):
    v = c[c.chi == chi].sigma_K01.dropna()
    print(
        f" chi={chi}: median {v.median():.3f} mean {v.mean():.3f} +- {v.std(ddof=1) / np.sqrt(len(v)):.3f}; P5 bracket {pr}; "
        f"implied c2_eff/c2 (asymptotic) = {(v.median() * np.sqrt(chi) / 6.50) ** 6:.2f}"
    )
tc = b[b.model == "twocopy"]
for chi, g in tc.groupby("chi"):
    v = g.sigma_K01
    print(
        f" twocopy chi={chi}: median*sqrt(chi) {v.median() * np.sqrt(chi):.2f} vs 5.16 -> {v.median() * np.sqrt(chi) / 5.16:.3f}"
    )
