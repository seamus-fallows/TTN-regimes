"""R3: what are the 'stalled' runs that bracket the S2S window at K = 8 (sigma_K_centred_d2, MPS N = 8,
d = 2, chi = 4, teacher [3, 2])? Reruns seed 0 at the smallest sigma (stalled below the window) and at the
first sigma above it (the 'unresolved' upper end), with run_regime.run_point redirected to
data/review_regime_map. For the upper run, compares l_end with bounds on the best bond-rank-chi
approximation of W* + W_0 (max over edges of the Eckart-Young tail <= best^2 <= sum of tails).
"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
REPO = "c:/Users/seamu/Iliad-ML/TTN_regimes"
sys.path.insert(0, REPO + "/experiments/regime_map")
sys.path.insert(0, REPO)
import run_regime as rr  # noqa: E402

from ttn.dynamics import Trajectory  # noqa: E402
from ttn.network import to_dense  # noqa: E402

rr.OUT = REPO + "/data/review_regime_map"
OUTF = REPO + "/data/review_regime_map/r3_stall.jsonl"
SIG = [float(f"{s:.4g}") for s in np.geomspace(0.1, 2, 12)]


def floor_bounds(cfg):
    problem, theta0, spec, info = rr.build(cfg)
    W = (to_dense(spec, info["cores0"]) + info["base"].dense()).numpy()
    n = cfg["n"]
    tails = []
    for v in spec.tree.edges:
        S = sorted(spec.tree.subtree_sites[v])
        rest = [i for i in range(n) if i not in S]
        M = np.transpose(W, S + rest).reshape(int(np.prod([W.shape[i] for i in S])), -1)
        s = np.linalg.svd(M, compute_uv=False)
        tails.append(float((s[spec.bond_dims[v] :] ** 2).sum()))
    L_ref2 = float(info["base"].norm2())  # ||W*||^2; l = ||err||^2 / ||W*||^2
    return max(tails) / L_ref2, sum(tails) / L_ref2


if __name__ == "__main__":
    picks = [int(a) for a in sys.argv[1:]] or [9, 0]
    for i in picks:
        s = SIG[i]
        cfg = dict(
            exp="r3_sigma_K_centred_d2",
            kind="mps",
            n=8,
            d=2,
            chi=4,
            cap=True,
            model="centred",
            weights=[3.0, 2.0],
            method="lsoda",
            l_floor=1e-6,
            max_steps=20000,
            sigma=s,
            seed=0,
        )
        row = rr.run_point(cfg)
        tr = Trajectory.load(os.path.join(rr.OUT, cfg["exp"], "traj", rr.run_id(cfg) + ".npz"))
        key = "spectra_W" if "spectra_W" in tr.data else "spectra"
        spec_end = tr[key][-1]  # (edges, chi_max)
        s2_min = float(np.nanmin(spec_end[:, 1])) if spec_end.shape[1] > 1 else float("nan")
        lo, hi = floor_bounds(cfg)
        keep = {
            k: row[k]
            for k in (
                "sigma",
                "label",
                "l_end",
                "dK",
                "event_times",
                "event_widths",
                "rhos",
                "stop",
                "t_end",
                "t_max",
                "n_steps",
                "integrator",
                "dormant_min",
                "wall",
            )
        }
        keep.update(spectra_key=key, s2_min_end=s2_min, floor_lo=lo, floor_hi=hi)
        print(json.dumps(keep, default=float), flush=True)
        with open(OUTF, "a") as f:
            f.write(json.dumps(keep, default=float) + "\n")
