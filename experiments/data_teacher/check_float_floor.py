"""Float64-floor check (lead's note, library v0): a dormant component whose signal falls below
~1e-14 during an earlier escape is re-seeded by roundoff, not by the init. For every CP-teacher run,
the minimum |a_r| before component r's first learning event (a_r = 1/2) is recorded, and its
sliding-window envelope; runs whose envelope falls below 1e-12 are flagged. Writes results/float_floor_check.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, OUT_ROOT

HERE = Path(__file__).parent
rows = []
for name in ("e1_order", "e1_scaling_sigma", "e1_scaling_lambda", "e2_misaligned", "e3_correlated",
             "e4a_nonorth", "e4a_sigma", "e2_sigma"):
    for f in sorted((OUT_ROOT / name).glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        ov = d["overlaps"]
        if ov.size == 0:
            continue
        for r in range(ov.shape[1]):
            a = ov[:, r]
            k = np.argmax(a >= 0.5) if (a >= 0.5).any() else len(a)
            seg = np.abs(a[1:max(k, 2)])
            # sliding-window maximum (+-5 output points): a sign change of a_r passes through 0
            # without the signal being lost; a lost signal stays small over the whole window
            env = np.array([seg[max(0, i - 5):i + 6].max() for i in range(len(seg))])
            rows.append(dict(sweep=name, run=f.stem, comp=r, min_abs_before=float(seg.min()),
                             min_envelope=float(env.min())))
df = pd.DataFrame(rows)
df["flag"] = df["min_envelope"] < 1e-12
df.to_csv(RESULTS / "float_floor_check.csv", index=False)
print(df.groupby("sweep").agg(n=("run", "size"), n_flag=("flag", "sum"),
                              min_envelope=("min_envelope", "min"),
                              min_abs=("min_abs_before", "min")).to_string())
