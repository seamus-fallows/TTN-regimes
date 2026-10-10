"""Scan reviewer reruns for overlap transients above 1/2 (event-time rule of notes/diagnostics.md:
use the last upward crossing when a transient exceeds 1/2; run_regime.refine_events keeps the first)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
from ttn import diagnostics as dg
from ttn.dynamics import Trajectory, merged_rows

root = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_regime_map"
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/regime_map")
from run_regime import row_cfg, run_id

tot, aff, flips = 0, [], []
for exp in ("chi_b4", "sigma_chi_b4", "imbalance_b4", "large_chi_b4", "sigma_K_centred_d2"):
    p = os.path.join(root, exp, "results.jsonl")
    if not os.path.exists(p):
        continue
    for line in open(p):
        r = json.loads(line)
        if "error" in r:
            continue
        f = os.path.join(root, exp, "traj", run_id(row_cfg(r)) + ".npz")
        if not os.path.exists(f):
            continue
        tr = Trajectory.load(f)
        m = merged_rows(tr.data, keys=("overlaps",))
        t, A = m["t"], m["overlaps"]
        n_up = dg.upward_crossings(A, 0.5)
        tot += 1
        if (n_up > 1).any():
            last = dg.crossing_times(t, A, 0.5, last=True)
            times = np.array(r["event_times"], float)
            new = np.where(n_up > 1, last, times)
            rh_new = dg.separation_ratios(new, np.array(r["event_widths"], float))
            lab_new = dg.classify_regime(r["dK"], rh_new, r["l_end"] * r["L_ref"], r["L_ref"])
            aff.append((exp, r.get("chi"), r.get("kind"), r.get("n"), r["seed"], r["sigma"], r.get("root_scale"), r.get("lr"),
                        n_up.tolist(), r["label"], lab_new, np.round(r["rhos"], 2).tolist(), np.round(rh_new, 2).tolist()))
print(f"runs scanned: {tot}; with a transient above 1/2: {len(aff)}")
for a in aff:
    print(a)
