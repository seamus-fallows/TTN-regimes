"""Relabel the R2 runs with the diagnostics.md rule for transients (last upward crossing of 1/2)."""

import glob
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
from ttn import diagnostics as dg  # noqa: E402
from ttn.dynamics import Trajectory  # noqa: E402

rows = {
    round(r["sigma"], 6): r
    for r in map(
        json.loads, open("c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_regime_map/r2_seed.jsonl")
    )
}
for p in sorted(
    glob.glob(
        "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_regime_map/r2_chi_b4_check/traj/*.npz"
    )
):
    tr = Trajectory.load(p)
    t, A = tr["t"], tr["overlaps"]
    s = float(p.split("sigma")[1].split("_seed")[0])
    r = rows[round(s, 6)]
    first = np.asarray(r["event_times"])
    last = np.asarray(dg.crossing_times(t, A, 0.5, last=True))  # coarse log grid
    w = np.asarray(r["event_widths"])
    ev = sorted(zip(last, w))
    rho_last = [(ev[i + 1][0] - ev[i][0]) / max(ev[i][1], ev[i + 1][1]) for i in range(len(ev) - 1)]
    lab = "s2s" if all(x > 3 for x in rho_last) else "rich"
    print(
        f"sigma={s:.5f} pipeline label={r['label']:16s} rhos(first)={np.round(r['rhos'], 2)} "
        f"t_first={np.round(first, 2)} t_last(coarse)={np.round(last, 2)} rhos(last)={np.round(rho_last, 2)} -> {lab}"
    )
