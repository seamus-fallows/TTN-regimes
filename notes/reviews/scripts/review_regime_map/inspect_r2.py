"""Inspect overlaps a_r(t) of the R2 runs (chi=12, seed 4) around the first events."""

import glob
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
from ttn.dynamics import Trajectory  # noqa: E402

for p in sorted(
    glob.glob(
        "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_regime_map/r2_chi_b4_check/traj/*.npz"
    )
):
    tr = Trajectory.load(p)
    t, A = tr["t"], tr["overlaps"]
    name = p.split("sigma")[1].split("_seed")[0]
    print(f"--- sigma {name}: n_log={len(t)}")
    # first times each component exceeds 0.1, 0.5, 0.9; max of a_2 and a_1 before their first 0.9
    for r in range(A.shape[1]):
        a = A[:, r]
        f = lambda lev: t[np.argmax(a >= lev)] if (a >= lev).any() else np.nan  # noqa: E731
        i9 = np.argmax(a >= 0.9) if (a >= 0.9).any() else len(a)
        # transient: does a_r exceed 0.5 and then fall below 0.4 before reaching 0.9?
        above = np.flatnonzero(a[:i9] >= 0.5)
        dip = above.size > 0 and (a[above[0] : i9] < 0.4).any()
        print(
            f"  a_{r + 1}: t(0.1)={f(0.1):.2f} t(0.5)={f(0.5):.2f} t(0.9)={f(0.9):.2f}  transient>0.5 then dip: {dip}"
        )
    # overlaps at a few times around the first event
    tc = t[np.argmax(A.max(axis=1) >= 0.5)]
    for tt in (0.9 * tc, 0.97 * tc, tc, 1.03 * tc, 1.1 * tc, 1.3 * tc):
        i = np.searchsorted(t, tt)
        i = min(i, len(t) - 1)
        print(f"   t={t[i]:8.2f}  a={np.round(A[i], 3)}")
