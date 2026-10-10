"""R2: the chi_b4 outlier (chi = 12, seed 4: label-based S2S boundary 0.0126 vs aligned prediction 0.075)
and a control seed. Uses run_regime.run_point with its output root redirected to data/review_regime_map.
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

rr.OUT = REPO + "/data/review_regime_map"
OUTF = REPO + "/data/review_regime_map/r2_seed.jsonl"
SIG = [float(f"{s:.4g}") for s in np.geomspace(0.004, 4, 22)][:7]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "outlier"
    jobs = [(12, 4, s) for s in SIG] if which == "outlier" else [(12, 0, s) for s in SIG[3:6]]
    for chi, seed, s in jobs:
        cfg = dict(
            exp="r2_chi_b4_check",
            kind="binary",
            n=4,
            d=3,
            chi=chi,
            cap=False,
            model="centred",
            sigma=s,
            seed=seed,
            l_floor=1e-6,
        )
        row = rr.run_point(cfg)
        keep = {
            k: row[k]
            for k in (
                "chi",
                "seed",
                "sigma",
                "label",
                "l_end",
                "dK",
                "event_times",
                "event_widths",
                "rhos",
                "refined",
                "stop",
                "t_end",
                "dormant_min",
                "wall",
            )
        }
        print(json.dumps(keep, default=float), flush=True)
        with open(OUTF, "a") as f:
            f.write(json.dumps(keep, default=float) + "\n")
