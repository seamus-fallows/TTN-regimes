"""P5 / P6 / R4 / P16 / P20 checks from the tensor-kernel cache (verified by tensor_check.py) and my relabel rows."""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out")
CACHE = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/regime_map_local/tensor_kernel"
T = np.sqrt(14.0)


def load(e):
    d = pd.read_csv(os.path.join(OUTD, f"relabel_{e}.csv"))
    d = d[d.found]
    c = pd.read_csv(os.path.join(CACHE, f"{e}.csv")).drop_duplicates("run_id", keep="last")
    d["dKt"] = d.rid.map(dict(zip(c.run_id, c.dK_tensor)))
    return d


def kb(g, col, thr=0.1):
    g = g.sort_values("sigma")
    g = g[(g.label_new != "stalled") & np.isfinite(g[col].astype(float))]
    s, k = g.sigma.values, g[col].values.astype(float)
    idx = np.flatnonzero((k[:-1] >= thr) & (k[1:] < thr))
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    w = (np.log(thr) - np.log(k[i])) / (np.log(k[i + 1]) - np.log(k[i]))
    return float(np.exp(np.log(s[i]) + w * (np.log(s[i + 1]) - np.log(s[i]))))


def p5(chi, thr=0.1):
    c1, c2 = 341.5 / chi**3, 537 / chi**3
    xs = [(s * -c1 + np.sqrt(c1**2 + 4 * c2 * thr)) / (2 * c2) for s in (1, -1)]
    return sorted((T / x) ** (1 / 3) for x in xs)


d = load("chi_b4")
rows = []
for (m, chi, seed), g in d.groupby(["model", "chi", "seed"]):
    rows.append(
        dict(
            model=m,
            chi=chi,
            seed=seed,
            t=kb(g, "dKt"),
            p=kb(g, "dK"),
            t05=kb(g, "dKt", 0.05),
            t2=kb(g, "dKt", 0.2),
        )
    )
lb = pd.DataFrame(rows)
print("chi_b4 lazy boundaries (median over seeds; n finite):")
for (m, chi), g in lb.groupby(["model", "chi"]):
    lo, hi = p5(chi)
    mt, mp = g.t.median(), g.p.median()
    extra = ""
    if m == "centred":
        extra = f"P5 [{lo:.4f}, {hi:.4f}] -> {'inside' if lo <= mt <= hi else ('%+.1f%%' % (100 * (mt / hi - 1) if mt > hi else 100 * (mt / lo - 1)))}"
    else:
        cen = lb[(lb.model == "centred") & (lb.chi == chi)].t.median()
        extra = f"P6 {5.16 / np.sqrt(chi):.3f}; two-copy/centred {mt / cen:.3f}"
    print(
        f"  {m:8s} chi={chi:4d} tensor {mt:.4f} (n={g.t.notna().sum()}) x sqrt(chi) {mt * np.sqrt(chi):.3f} | probe {mp:.4f} | "
        f"ratio 0.05/0.2 tensor {np.nanmedian(g.t05 / g.t2):.3f} | {extra}"
    )
sl = lb[(lb.model == "centred")]
for cmin in (16, 32):
    s = sl[(sl.chi >= cmin) & np.isfinite(sl.t)]
    b, a = np.polyfit(np.log(s.chi), np.log(s.t), 1)
    r = np.log(s.t) - (a + b * np.log(s.chi))
    se = np.sqrt(r @ r / (len(s) - 2) / np.sum((np.log(s.chi) - np.log(s.chi).mean()) ** 2))
    print(f"  tensor lazy slope chi>={cmin}: {b:.4f} +- {se:.4f} (n={len(s)})")
# R4 check and probe/tensor ratio near the boundary
cc = d[(d.model == "centred") & (d.chi >= 32) & (d.label_new != "stalled") & np.isfinite(d.dKt)]
nb = cc[(cc.dKt > 0.03) & (cc.dKt < 0.3)]
x = T / nb.sigma**3
c1, c2 = 341.5 / nb.chi**3, 537 / nb.chi**3
r4 = nb.dKt / np.sqrt((c1 * x) ** 2 + (c2 * x**2) ** 2)
pr = nb.dK / nb.dKt
print(
    f"  R4 ratio (0.03<dKt<0.3, chi>=32): n={len(nb)} median {r4.median():.3f} range {r4.min():.2f}-{r4.max():.2f}; "
    f"probe/tensor median {pr.median():.3f} range {pr.min():.2f}-{pr.max():.2f}"
)

# ---- large chi (P16) -------------------------------------------------------------------------
L = load("large_chi_b4")
L["lr"] = L.lr.fillna("sp")
print("\nlarge_chi_b4 tensor dK slopes, chi 16-128:")
for (m, sg, lr), g in L[(L.chi >= 16)].groupby(["model", "sigma", "lr"]):
    g = g[np.isfinite(g.dKt)]
    b, a = np.polyfit(np.log(g.chi), np.log(g.dKt), 1)
    r = np.log(g.dKt) - (a + b * np.log(g.chi))
    se = np.sqrt(r @ r / (len(g) - 2) / np.sum((np.log(g.chi) - np.log(g.chi).mean()) ** 2))
    meds = g.groupby("chi").dKt.median()
    loc = np.log(meds.loc[128] / meds.loc[64]) / np.log(2)
    print(
        f"  {m:8s} sigma={sg} {lr:4s}: slope {b:+.3f} +- {se:.3f} (n={len(g)}); medians "
        + ", ".join(f"{k}:{v:.4g}" for k, v in meds.items())
        + f"; local 64-128 {loc:+.3f}; probe/tensor "
        f"{(g[g.chi >= 32].dK / g[g.chi >= 32].dKt).min():.2f}-{(g[g.chi >= 32].dK / g[g.chi >= 32].dKt).max():.2f}"
    )

# ---- imbalance P20 ---------------------------------------------------------------------------
I = load("imbalance_b4")
I = I[(I.chi == 16) & np.isclose(I.alpha.astype(float), 3.375)]
print(
    "\nP20 point: n runs", len(I), "unique (seed, lam)", I.groupby(["seed", "root_scale"]).ngroups
)
for col in ("dKt", "dK"):
    piv = I.pivot_table(index="seed", columns="root_scale", values=col)
    lam = piv.columns.values
    print(f"  {col} medians:", ", ".join(f"{l:g}:{piv[l].median():.3g}" for l in lam))
    sl = []
    for i in range(len(lam) - 1):
        s = np.log(piv[lam[i + 1]] / piv[lam[i]]) / np.log(lam[i + 1] / lam[i])
        sl.append(
            f"{lam[i]:g}-{lam[i + 1]:g}: {s.mean():.2f} +- {s.std(ddof=1) / np.sqrt(len(s)):.2f}"
        )
    print("   per-seed local slopes:", "; ".join(sl[:5]))
    print(
        "   range lam>=10:",
        piv[[l for l in lam if l >= 10]].min().min().round(3),
        piv[[l for l in lam if l >= 10]].max().max().round(3),
    )
print(
    "  labels:", I.groupby("root_scale").label_new.value_counts().unstack(fill_value=0).to_string()
)
