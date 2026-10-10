"""Audit part 3: sigma x K (centred d=2 chi=4): statuses, u0 vs alpha collapse, fair comparison."""

import numpy as np
import pandas as pd

R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/regime_map/results/"
t = pd.read_csv(R + "summary_sigma_c_per_seed.csv")
t["N"] = t.n
d = 2.0
print("=== status counts per tree (centred d=2 chi=4)")
t["base_status"] = t.s2s_status.str.split("+").str[0]
t["stall_below"] = t.s2s_status.str.contains("stall_below")
print(
    t.groupby(["K", "kind", "n"])
    .agg(
        n=("seed", "size"),
        ok=("base_status", lambda s: (s == "ok").sum()),
        unresolved=("base_status", lambda s: (s == "unresolved").sum()),
        all_s2s=("base_status", lambda s: (s == "all_s2s").sum()),
        stall_below=("stall_below", "sum"),
        median_n_s2s=("n_s2s", "median"),
        median_sc=("sigma_c", "median"),
    )
    .to_string()
)

v = pd.read_csv(R + "sigma_c_vs_K.csv")
print("\n=== sigma_c_vs_K.csv n_unresolved column vs actual (status startswith 'unresolved'):")
for _, r in v[v.exp == "sigma_K_centred_d2"].iterrows():
    g = t[(t.kind + t.n.astype(str)) == r.tree]
    print(
        f"  {r.tree}: csv n_unresolved={r.n_unresolved}, actual={int(g.s2s_status.str.startswith('unresolved').sum())}"
    )

# collapse comparison
med = t.groupby(["K", "N", "kind"]).sigma_c.median().reset_index()
med = med[med.K >= 4]
med["log_alpha"] = med.K * np.log(med.sigma_c)
med["log_u0"] = med.log_alpha - med.N / 2 * np.log(d)
print("\n=== per-K medians: alpha_c and u0c")
print(med.assign(alpha_c=np.exp(med.log_alpha), u0c=np.exp(med.log_u0)).round(4).to_string())


def fit(m, var):
    """best constant in log-variable space, residuals in log sigma"""
    lc = m[var].mean()
    if var == "log_alpha":
        pred = np.exp(lc / m.K)
    else:
        pred = np.exp((lc + m.N / 2 * np.log(d)) / m.K)
    res = np.log(m.sigma_c / pred)
    return np.exp(lc), res


# per-K SE of log median by bootstrap over seeds
rng = np.random.default_rng(0)
se = {}
for K, g in t[t.K >= 4].groupby("K"):
    x = g.sigma_c.dropna().values
    bs = [np.log(np.median(rng.choice(x, len(x)))) for _ in range(4000)]
    se[K] = np.std(bs)
med["se_log"] = med.K.map(se)
for label, sel in (("K=4..12 (all medians)", med.K <= 12), ("K=4..7 (resolved only)", med.K <= 7)):
    m = med[sel]
    print(
        f"\n--- {label}: sd of log var across K: alpha {m.log_alpha.std(ddof=1):.2f}, u0 {m.log_u0.std(ddof=1):.2f}"
    )
    for var in ("log_alpha", "log_u0"):
        c, res = fit(m, var)
        chi2 = ((res / m.se_log) ** 2).sum()
        print(
            f"  {var}: best const {c:.4g}; rms log-sigma residual {np.sqrt((res**2).mean()):.3f}; chi2={chi2:.1f} (dof {len(m) - 1});"
            f" residuals {np.round(res.values, 3)}"
        )
    # fixed u0c = 0.01 (as in the note)
    pred = (0.01 * d ** (m.N / 2)) ** (1 / m.K)
    res = np.log(m.sigma_c / pred)
    print(
        f"  u0c=0.01 fixed: rms {np.sqrt((res**2).mean()):.3f}; chi2={((res / m.se_log) ** 2).sum():.1f}"
    )
print("\nper-K bootstrap SE of log median:", {k: round(v, 3) for k, v in se.items()})

# u0 implied by each 'ok' seed only
print("\n=== using only 'ok' (resolved) seeds")
ok = t[t.base_status == "ok"]
for K, g in ok.groupby("K"):
    N = g.N.iloc[0]
    sc = g.sigma_c.median()
    print(
        f"  K={K} N={N}: n_ok={len(g)} median {sc:.3f} -> alpha {sc**K:.3g}, u0 {sc**K / d ** (N / 2):.3g}"
    )
