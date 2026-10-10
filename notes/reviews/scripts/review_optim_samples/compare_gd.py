"""Compare the reviewer's re-run of gd_sweep/predict_shift/analyze_gd with the committed CSVs."""

import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
REPO = "c:/Users/seamu/Iliad-ML/TTN_regimes/"
COMMITTED = REPO + "experiments/optim_samples/results/"
mine_dir = sys.argv[1]

old = pd.read_csv(COMMITTED + "gd_sweep.csv")
new = pd.read_csv(mine_dir + "/gd_sweep.csv")
key = ["tree", "n", "sigma", "seed", "ratio"]
for d in (old, new):
    d.drop(d[d.root_scale.notna() | d.centred].index, inplace=True)
m = new.merge(old, on=key, suffixes=("_new", "_old"), how="left")
print("rows re-run:", len(new), " matched to committed:", int(m.cls_old.notna().sum()))
same = m[m.cls_old.notna()]
print("class agreement:", int((same.cls_new == same.cls_old).sum()), "/", len(same))
for c in ["t1_gf", "t2_gf", "t3_gf", "t1_gd", "t2_gd", "t3_gd", "loss_tail_mean", "sharp_end"]:
    a, b = same[c + "_new"].astype(float), same[c + "_old"].astype(float)
    ok = np.isfinite(a) & np.isfinite(b)
    rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300)
    nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
    print(
        f"{c:15s} n={ok.sum():4d} max rel diff {rel.max() if len(rel) else np.nan:.2e}  nan mismatches {nan_mismatch}"
    )
print("\nclass table (re-run, all seeds incl. new 50-52):")
print(
    new.groupby(["sigma", "ratio"]).cls.agg(lambda c: dict(c.value_counts())).unstack().to_string()
)

# oscillation excess loss vs 1D map (committed map1d_attractor.csv)
mp = pd.read_csv(COMMITTED + "map1d_attractor.csv")
print("\nexcess loss (tail mean - GF end) vs s1^2 x map loss:")
for r in (1.05, 1.1, 1.2):
    g = new[(new.ratio == r) & (new.cls == "oscillating")]
    ex = g.loss_tail_mean - g.loss_gf_end
    ml = 9 * mp[(mp.K == 3) & np.isclose(mp.ratio, r)].loss.values[0]
    print(f"ratio {r}: n={len(g)} excess {ex.mean():.4f} +- {ex.std():.4f}   9*L_map {ml:.4f}")

# event shifts
print("\nshift/eta at ratio 0.1 vs predicted c_e (re-run predict_shift):")
cols = []
for e in (1, 2, 3):
    new[f"dev{e}"] = new[f"shift_over_eta{e}"] / new[f"pred_shift_over_eta{e}"] - 1
    cols += [f"shift_over_eta{e}", f"pred_shift_over_eta{e}", f"dev{e}"]
print(new[new.ratio == 0.1][["sigma", "seed"] + cols].to_string(float_format=lambda v: f"{v:.4g}"))
sp_old = pd.read_csv(COMMITTED + "gd_shift_prediction.csv")
sp_new = pd.read_csv(mine_dir + "/gd_shift_prediction.csv")
sp_new = sp_new[sp_new.root_scale.isna()]
mm = sp_new.merge(sp_old, on=["tree", "n", "sigma", "seed", "comp"], suffixes=("_new", "_old"))
mm["rel"] = np.abs(mm.pred_shift_over_eta_new / mm.pred_shift_over_eta_old - 1)
print("\npredicted c_e, re-run vs committed: n =", len(mm), " max rel diff %.2e" % mm.rel.max())
