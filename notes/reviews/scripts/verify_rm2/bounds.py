"""Per-seed S2S boundaries from my relabel CSVs: chi_b4 (label rule), sigma_chi_b4 (rho = 3 crossing),
collapse fits, and the K-sweep walk. Usage: uv run python bounds.py"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out")
ROOT = "c:/Users/seamu/Iliad-ML/TTN_regimes"


def rd(e):
    return pd.read_csv(os.path.join(OUTD, f"relabel_{e}.csv"))


def label_block(g, col="label_new"):
    g = g.sort_values("sigma")
    lab, sig = g[col].values, g.sigma.values
    s = np.flatnonzero(lab == "s2s")
    if s.size == 0:
        return np.nan, "never", np.nan, np.nan
    st = s[0]
    e = st
    while e + 1 < len(lab) and lab[e + 1] == "s2s":
        e += 1
    if e == len(lab) - 1:
        return np.nan, "all", sig[e], np.nan
    status = "unresolved" if lab[e + 1] == "stalled" else "ok"
    return float(np.sqrt(sig[e] * sig[e + 1])), status, sig[e], sig[e + 1]


def rho_cross(g, thr=3.0, col="min_rho_new"):
    g = g.sort_values("sigma")
    sig, rho = g.sigma.values, np.nan_to_num(g[col].values.astype(float), nan=0.0)
    if not rho[0] > thr:
        return np.nan
    below = np.flatnonzero(rho <= thr)
    if below.size == 0:
        return np.nan
    i = below[0]
    if rho[i] <= 0:
        return float(np.sqrt(sig[i - 1] * sig[i]))
    w = (np.log(thr) - np.log(rho[i - 1])) / (np.log(rho[i]) - np.log(rho[i - 1]))
    return float(np.exp(np.log(sig[i - 1]) + w * (np.log(sig[i]) - np.log(sig[i - 1]))))


def ols(x, y):
    x, y = np.log(x), np.log(y)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    se = np.sqrt(res @ res / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
    return b, se


def gm(v):
    v = np.log(np.asarray(v, float))
    return np.exp(v.mean()), np.exp(v.std(ddof=1) / np.sqrt(len(v))), len(v)


# ---- chi_b4 --------------------------------------------------------------------------------
c = rd("chi_b4")
c = c[c.model == "centred"]
rows = []
for (chi, seed), g in c.groupby(["chi", "seed"]):
    sc, st, lo, hi = label_block(g)
    sco, sto, _, _ = label_block(g, "stored_label")
    rows.append(dict(chi=chi, seed=seed, sc=sc, status=st, lo=lo, hi=hi, sc_old=sco))
cb = pd.DataFrame(rows)
print("chi_b4 seeds:", len(cb), "statuses", cb.status.value_counts().to_dict())
print("changed per-seed boundaries:\n", cb[~np.isclose(cb.sc, cb.sc_old)].to_string())
for cmin in (9, 16, 32):
    s = cb[(cb.chi >= cmin) & np.isfinite(cb.sc)]
    b, se = ols(s.chi.values, s.sc.values)
    per = [ols(h.chi.values, h.sc.values)[0] for _, h in s.groupby("seed")]
    pm, pse = np.mean(per), np.std(per, ddof=1) / np.sqrt(len(per))
    print(
        f"chi>={cmin}: n={len(s)} pooled slope {b:.4f} +- {se:.4f}; (b+1/3)/se = {(b + 1 / 3) / se:.2f}; "
        f"(b+1/2)/se = {(b + 0.5) / se:.2f}; per-seed {pm:.4f} +- {pse:.4f} -> {(pm + 1 / 3) / pse:.2f} SE"
    )
    # old labels
    so = cb[(cb.chi >= cmin) & np.isfinite(cb.sc_old)]
    bo, seo = ols(so.chi.values, so.sc_old.values)
    print(f"   pre-fix labels: slope {bo:.4f} +- {seo:.4f}")
med = cb.groupby("chi").sc.median()
print("measured sigma_c sqrt(chi) medians:", {k: round(v * np.sqrt(k), 3) for k, v in med.items()})

# ---- sigma_chi_b4 (centred) -------------------------------------------------------------------
s = rd("sigma_chi_b4")
s = s[s.model == "centred"]
rows = []
for (chi, seed), g in s.groupby(["chi", "seed"]):
    rows.append(
        dict(
            chi=chi,
            seed=seed,
            b_new=rho_cross(g),
            b_old=rho_cross(g, col="min_rho_old"),
            b2=rho_cross(g, 2.0),
            b5=rho_cross(g, 5.0),
        )
    )
sb = pd.DataFrame(rows)
print(
    "\nsigma_chi_b4 per-seed rho=3 crossings that moved:\n",
    sb[~np.isclose(sb.b_new, sb.b_old, equal_nan=True)].to_string(),
)
for chi, g in sb.groupby("chi"):
    v = g.b_new.dropna()
    vo = g.b_old.dropna()
    m, f, n = gm(v)
    mo, fo, no = gm(vo)
    r25 = (g.b2 / g.b5).dropna()
    print(
        f"chi={chi}: new {m:.4f} x/ {f:.3f} (n={n}); old {mo:.4f} x/ {fo:.3f} (n={no}); rho2/rho5 {r25.mean():.3f}"
    )
sb.to_csv(os.path.join(OUTD, "sigma_chi_b4_seed_bounds.csv"), index=False)
cb.to_csv(os.path.join(OUTD, "chi_b4_seed_bounds.csv"), index=False)
