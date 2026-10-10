"""Fair comparison of scaling laws for Adam's later gaps on binary8 (later_b8_summary.csv):
per seed (modal order, resolved, converged), all fitted by least squares in ln(n) space:
  power:  ln n = c + p ln(1/sigma)                                   (2 params)
  log:    n = a + b ln(1/sigma)                                       (2 params, nonlinear in ln space)
  memory: n = (2/(1-beta2)) ln(1 + k T_GF(sigma, seed))  (1 param k; T_GF = same seed's GF gap, same order)
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optimizers/results/"
s = pd.read_csv(R + "later_b8_summary.csv", dtype={"order": str})
gf = s[s.optimizer == "gf"].set_index(["sigma", "seed"])
rows = []
for lr in (1e-3, 1e-2):
    a = s[(s.optimizer == "adam") & (s.lr == lr)]
    for seed, g in a.groupby("seed"):
        g = g[g.converged]
        modal = g.order.value_counts().index[0]
        g = g[g.order == modal]
        for q in ("gap_0", "gap_1"):
            gg = g[g[q + "_resolved"].astype(bool) & np.isfinite(g[q])].sort_values("sigma")
            if len(gg) < 4:
                continue
            x = np.log(1 / gg.sigma.values)
            y = np.log(gg[q].values)
            n = gg[q].values
            # power
            A = np.vstack([np.ones_like(x), x]).T
            cp, *_ = np.linalg.lstsq(A, y, rcond=None)
            rms_pow = np.sqrt(np.mean((y - A @ cp) ** 2))
            # log law, fitted in ln space
            def r_log(p):
                pred = p[0] + p[1] * x
                return np.log(np.clip(pred, 1e-12, None)) - y
            cl, *_ = np.linalg.lstsq(A, n, rcond=None)
            p0 = cl if np.all(A @ cl > 0) else np.array([n.min(), (n.max() - n.min()) / max(np.ptp(x), 1e-9)])
            fl = least_squares(r_log, p0)
            rms_log = np.sqrt(np.mean(fl.fun ** 2))
            # memory-phase law with the same seed's GF gap (same order only)
            Tgf, keep = [], []
            for sg in gg.sigma.values:
                if (sg, seed) in gf.index and gf.loc[(sg, seed), "order"] == modal and np.isfinite(gf.loc[(sg, seed), q]):
                    Tgf.append(gf.loc[(sg, seed), q]); keep.append(True)
                else:
                    Tgf.append(np.nan); keep.append(False)
            keep = np.array(keep); Tgf = np.array(Tgf)
            rms_mem, kfit, nmem = np.nan, np.nan, int(keep.sum())
            if keep.sum() >= 3:
                def r_mem(p):
                    return np.log(2000.0 * np.log1p(np.exp(p[0]) * Tgf[keep])) - y[keep]
                fm = least_squares(r_mem, [np.log(1e-3)])
                rms_mem, kfit = np.sqrt(np.mean(fm.fun ** 2)), np.exp(fm.x[0])
                # power law restricted to the same points, for a like-for-like score
                Ak = A[keep]
                ck, *_ = np.linalg.lstsq(Ak, y[keep], rcond=None)
                rms_pow_k = np.sqrt(np.mean((y[keep] - Ak @ ck) ** 2))
            else:
                rms_pow_k = np.nan
            rows.append(dict(lr=lr, seed=seed, q=q, order=modal, n=len(gg), p=cp[1], rms_power=rms_pow,
                             rms_log=rms_log, n_mem=nmem, rms_memory_1param=rms_mem, rms_power_same_pts=rms_pow_k,
                             k_mem=kfit))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.round(3).to_string())
print(df.groupby(["lr", "q"])[["rms_power", "rms_log", "rms_memory_1param", "rms_power_same_pts"]].mean().round(3))
if len(sys.argv) > 1:
    df.to_csv(sys.argv[1] + "/b8_law_comparison.csv", index=False)
