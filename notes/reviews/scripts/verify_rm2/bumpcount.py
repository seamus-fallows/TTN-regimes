"""Count runs with n_up > 1 under two definitions: (a) learned components only (before first 0.9);
(b) also never-learned components (upward crossings over the whole curve), as in run_regime.final_rise."""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relabel import RAW, merged, OUTD
for e in sys.argv[1:]:
    d = pd.read_csv(os.path.join(OUTD, f"relabel_{e}.csv"))
    d = d[d.found]
    na = nb = 0
    for rid in d.rid:
        with np.load(os.path.join(RAW, e, "traj", rid + ".npz"), allow_pickle=False) as z:
            t, A = merged(z)
        fa = fb = False
        for c in range(A.shape[1]):
            a = A[:, c]
            reach = np.flatnonzero(a >= 0.9)
            seg = a[: reach[0] + 1] if reach.size else a
            ab = seg >= 0.5
            n = int(ab[0]) + int(np.sum(ab[1:] & ~ab[:-1]))
            if n > 1:
                fb = True
                if reach.size:
                    fa = True
        na += fa; nb += fb
    print(e, "runs", len(d), "bump(learned only)", na, "bump(incl. never-learned)", nb, flush=True)
