"""Escape degree from single runs (theory_saddles.md R5): on a plateau
d log(da/dt) / d log a -> 2 - 2/k, k = number of cores the escape needs. Fitted for the first-learned
component over a in [1e-4, 1e-2] in every run of the e1_order sweep. Prediction: k = K (3 for
binary-4, 4 for MPS-4) for every variant, i.e. the covariance does not change the escape degree.
Writes results/e1_kexp.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, OUT_ROOT

HERE = Path(__file__).parent


def local_exponent(t, a, lo=1e-4, hi=1e-2):
    m = (a > lo) & (a < hi)
    if m.sum() < 5:
        return np.nan
    tt, aa = t[m], a[m]
    da = np.gradient(aa, tt)
    ok = da > 0
    return np.polyfit(np.log(aa[ok]), np.log(da[ok]), 1)[0]


rows = []
df = pd.DataFrame(pd.read_pickle(OUT_ROOT / "e1_order" / "summary_rows.pkl"))
for _, r in df.iterrows():
    d = np.load(OUT_ROOT / "e1_order" / f"{r.tree}_s{r.sigma}_{r.teacher}_{r.variant}_seed{r.seed}.npz",
                allow_pickle=True)
    t, a = d["t"], d["overlaps"][:, int(r["first"])]
    k_end = np.argmax(a > 0.5)
    p = local_exponent(t[:k_end], a[:k_end])
    rows.append(dict(tree=r.tree, variant=r.variant, sigma=r.sigma, exponent=p, k_est=2 / (2 - p)))
out = pd.DataFrame(rows)
agg = out.groupby(["tree", "variant"]).agg(n=("k_est", "size"), exponent=("exponent", "mean"),
                                           exponent_sem=("exponent", "sem"),
                                           k_est=("k_est", "median"))
agg.to_csv(RESULTS / "e1_kexp.csv")
print(agg.to_string())
