"""Fair power vs log law comparison (both fitted by least squares in ln T space) per seed,
binary4 sweep, quantities t_first, gap_0, gap_1, using the agent's selection (converged, resolved, modal order)."""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optimizers/results/"
s = pd.read_csv(R + "scaling_b4_summary.csv", dtype={"order": str})
s["lr"] = s["lr"].fillna(0)
s["beta2"] = s["beta2"].fillna(0.999)
s["clock"] = np.where(s.optimizer == "gd", s.lr, np.where(s.optimizer == "momentum", s.lr / (1 - s.beta.fillna(0.9)), 1.0))
s = s[s.optimizer.isin(["gf", "gd", "momentum", "adam", "rmsprop"])]
rows = []
for (opt, lr, b2), g0 in s.groupby(["optimizer", "lr", "beta2"]):
    for seed, g in g0.groupby("seed"):
        g = g[g.converged]
        modal = g.order.value_counts().index[0]
        g = g[g.order == modal]
        for q in ("t_first", "gap_0", "gap_1"):
            res = "t_first_resolved" if q == "t_first" else q + "_resolved"
            gg = g[g[res].astype(bool) & np.isfinite(g[q])]
            if len(gg) < 4:
                continue
            x = np.log(1 / gg.sigma.values)
            T = (gg[q] * gg.clock).values
            y = np.log(T)
            A = np.vstack([np.ones_like(x), x]).T
            cp, *_ = np.linalg.lstsq(A, y, rcond=None)
            rp = np.sqrt(np.mean((y - A @ cp) ** 2))
            cl, *_ = np.linalg.lstsq(A, T, rcond=None)
            p0 = cl if np.all(A @ cl > 0) else np.array([T.min(), 0.0])
            f = least_squares(lambda p: np.log(np.clip(p[0] + p[1] * x, 1e-12, None)) - y, p0)
            rl = np.sqrt(np.mean(f.fun ** 2))
            # the agent's scoring: log law fitted in linear T space, clipped
            ra = np.sqrt(np.mean((y - np.log(np.clip(A @ cl, 1e-300, None))) ** 2))
            rows.append(dict(opt=opt, lr=lr, b2=b2, seed=seed, q=q, p=cp[1], rms_power=rp, rms_log_fair=rl, rms_log_agent=ra))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.groupby(["opt", "lr", "b2", "q"])[["p", "rms_power", "rms_log_fair", "rms_log_agent"]].mean().round(3).to_string())
if len(sys.argv) > 1:
    df.to_csv(sys.argv[1] + "/b4_law_comparison.csv", index=False)
