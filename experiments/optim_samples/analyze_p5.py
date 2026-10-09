"""P5 of notes/theory_optimizer.md: time-averaged excess loss of GD above eta_max vs the 1D map.

For oscillating runs (gd_sweep.csv, plain model, chi = 3, seeds 0-4) at eta_max < eta <= 1.3 eta_max:
excess = (time-averaged loss over the last 20% of steps, or <= 2000 steps) - L_GF(T), compared with
s_1^2 x (mean loss of the attractor of the balanced 1D map at h/h_c = eta/eta_max, map1d.py) and with
the theory note's linear form 3 K^2 s_1^2 / (4 (K-1)(5K-4)) (eta/eta_max - 1). Runs whose GF reference
never learns component 1 by T (the oscillating component) are excluded and counted.

Run after analyze_gd.py and map1d.py: uv run python experiments/optim_samples/analyze_p5.py
Writes p5_excess_loss.csv.
"""

import numpy as np
import pandas as pd

from common import RESULTS as RES

S1 = 3.0


def main():
    df = pd.read_csv(RES / "gd_sweep.csv")
    df = df[~df.centred & df.root_scale.isna() & (df.chi == 3) & (df.ratio_ref == "teacher")
            & (df.seed <= 4) & (df.ratio > 1) & (df.ratio <= 1.3)]
    att = pd.read_csv(RES / "map1d_attractor.csv")
    rows = []
    for (tree, n, k, ratio), g in df.groupby(["tree", "n", "K", "ratio"]):
        learned1 = g.t1_gf.notna()  # GF learns component 1 (the oscillating one) by T
        osc = g[(g.cls == "oscillating") & learned1]
        ex = osc.loss_tail_mean - osc.loss_gf_end
        a = att[(att.K == k) & np.isclose(att.ratio, ratio)]
        rows.append(dict(tree=tree, n=n, K=k, ratio=ratio, n_runs=len(g),
                         n_excluded_no_component1=int((~learned1).sum()), n_oscillating=len(osc),
                         excess_mean=ex.mean(), excess_se=ex.sem(),
                         map1d=S1**2 * a.loss.values[0] if len(a) else np.nan,
                         linear_P5=3 * k**2 * S1**2 / (4 * (k - 1) * (5 * k - 4)) * (ratio - 1)))
    out = pd.DataFrame(rows)
    out["rel_dev_map"] = out.excess_mean / out.map1d - 1
    out.to_csv(RES / "p5_excess_loss.csv", index=False)
    print(out.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
