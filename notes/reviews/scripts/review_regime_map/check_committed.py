"""Internal-consistency checks on the committed regime_map CSVs (no raw data needed)."""
import numpy as np, pandas as pd
R = "experiments/regime_map/results/"
t = pd.read_csv(R + "boundaries_per_seed_chi_b4.csv")
c = t[t.model == "centred"]
def fit(sel, col):
    x, y = np.log(sel.chi.values), np.log(sel[col].values)
    A = np.stack([x, np.ones_like(x)], 1); coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2 = ((y - A @ coef) ** 2).sum() / (len(x) - 2); se = np.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0])
    return coef[0], se, len(x)
for lo in (9, 16, 32):
    s = c[(c.chi >= lo) & c.sigma_c.notna()]; print("S2S slope chi>=%d: %.3f +- %.3f (n=%d)" % ((lo,) + fit(s, "sigma_c")))
    s = c[(c.chi >= lo) & c.sigma_K01.notna()]; print("lazy slope chi>=%d: %.3f +- %.3f (n=%d)" % ((lo,) + fit(s, "sigma_K01")))
s = c[(c.chi >= 9) & c.sigma_c_pred.notna()]; print("S2S PRED slope chi>=9: %.3f +- %.3f (n=%d)" % fit(s, "sigma_c_pred"))
s = c[(c.chi >= 16) & c.sigma_c_pred.notna()]; print("S2S PRED slope chi>=16: %.3f +- %.3f (n=%d)" % fit(s, "sigma_c_pred"))
g = c.groupby("chi")
print(pd.DataFrame({"s2s_gm*sqrt(chi)": g.sigma_c.apply(lambda v: np.exp(np.log(v).mean())) * np.sqrt(g.sigma_c.mean().index),
                    "s2s_med*sqrt": g.sigma_c.median() * np.sqrt(g.sigma_c.median().index),
                    "lazy_med*sqrt": g.sigma_K01.median() * np.sqrt(g.sigma_K01.median().index),
                    "pred/meas med": g.apply(lambda d: np.median(d.sigma_c_pred / d.sigma_c)),
                    "rho_slope mean": g.rho_slope.mean(), "window(lazy/s2s, medians)": g.sigma_K01.median() / g.sigma_c.median()}).round(3))
tw = t[t.model == "twocopy"].groupby("chi").sigma_K01.median(); print("twocopy lazy*sqrt(chi):", (tw * np.sqrt(tw.index)).round(2).to_dict())
# sigma x chi (sigma_chi_b4) sensitivity: geometric means and dropped seeds
st = pd.read_csv(R + "sensitivity_per_seed_sigma_chi_b4_centred.csv")
for chi, d in st.groupby("chi"):
    v = d.s2s_3.values; n_nan = np.isnan(v).sum()
    gm = np.exp(np.nanmean(np.log(v)))
    print(f"chi={chi}: s2s_3 geo-mean {gm:.3f} (NaN seeds: {n_nan}); rho2/rho5 median {np.nanmedian(d.s2s_2/d.s2s_5):.2f}; "
          f"lazy_0.1 geo-mean {np.exp(np.nanmean(np.log(d['lazy_0.1']))) if d['lazy_0.1'].notna().any() else np.nan:.3f}; "
          f"lazy0.05/0.2 med {np.nanmedian(d['lazy_0.05']/d['lazy_0.2']) if d['lazy_0.2'].notna().any() else np.nan:.2f}")
# sigma x K: statuses
k = pd.read_csv(R + "boundaries_per_seed_sigma_K_centred_d2_chi4.csv")
print(k.groupby("tree").s2s_status.value_counts().unstack(fill_value=0))
sk = pd.read_csv(R + "sigma_c_vs_K.csv"); sk = sk[sk.series.str.startswith("centred")]
D = {"binary4": 16, "mps4": 16, "mps5": 32, "mps6": 64, "binary8": 256, "mps8": 256, "mps10": 1024, "mps12": 4096, "binary16": 65536}
sk = sk.assign(D=sk.tree.map(D)); sk["alpha_c"] = sk.sigma_c_median ** sk.K; sk["u0c"] = sk.alpha_c / np.sqrt(sk.D)
print(sk[["tree", "K", "n_resolved", "n_unresolved", "sigma_c_median", "alpha_c", "u0c"]].sort_values("K").round(4).to_string())
