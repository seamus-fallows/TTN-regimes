"""Escape degree from single runs (theory_saddles.md R5 / P4): on a plateau
d log(da/dt) / d log a -> 2 - 2/k - a/(1-a), k = number of cores the escape needs.
Fitted for the first-learned component of every e1_order run, per sigma, over the window
a in [max(10 |a(0)|, 1e-5), 1e-2] (review of round 1: the old fixed window [1e-4, 1e-2] overlapped
the init transient at sigma = 0.2, where |a(0)| ~ 1e-4). Prediction: 2 - 2/K = 1.333 (binary-4),
1.5 (MPS-4), the same for every covariance variant. Writes results/e1_kexp.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_ROOT, RESULTS

HERE = Path(__file__).parent


def local_exponent(t, a, lo, hi=1e-2):
    m = (a > lo) & (a < hi)
    if m.sum() < 5:
        return np.nan, int(m.sum())
    tt, aa = t[m], a[m]
    da = np.gradient(aa, tt)
    ok = da > 0
    return np.polyfit(np.log(aa[ok]), np.log(da[ok]), 1)[0], int(m.sum())


rows = []
df = pd.DataFrame(pd.read_pickle(OUT_ROOT / "e1_order" / "summary_rows.pkl"))
for _, r in df.iterrows():
    d = np.load(OUT_ROOT / "e1_order" / f"{r.tree}_s{r.sigma}_{r.teacher}_{r.variant}_seed{r.seed}.npz",
                allow_pickle=True)
    t, a = d["t"], d["overlaps"][:, int(r["first"])]
    k_end = np.argmax(a > 0.5)
    lo = max(10 * abs(a[0]), 1e-5)
    p, npts = local_exponent(t[:k_end], a[:k_end], lo)
    rows.append(dict(tree=r.tree, variant=r.variant, sigma=r.sigma, exponent=p, n_points=npts,
                     window_lo=lo))
out = pd.DataFrame(rows)
agg = out.groupby(["tree", "sigma", "variant"]).agg(n=("exponent", "count"), exponent=("exponent", "mean"),
                                                    exponent_sem=("exponent", "sem"),
                                                    points=("n_points", "median")).reset_index()
agg["pred"] = agg["tree"].map({"binary": 4 / 3, "mps": 1.5})
agg.to_csv(RESULTS / "e1_kexp.csv", index=False)
print(agg.round(4).to_string())
