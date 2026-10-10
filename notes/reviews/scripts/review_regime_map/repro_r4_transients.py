"""R4: are the other low per-seed S2S boundaries (binary N = 4) also transient-bump artifacts?
Reruns the suspicious seeds just above their reported boundary with run_regime.run_point (output root
redirected to data/review_regime_map) and relabels with the diagnostics.md transient rule (last upward
crossing of 1/2 on the log grid, final-rise widths from the pipeline)."""

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

from ttn import diagnostics as dg  # noqa: E402
from ttn.dynamics import Trajectory  # noqa: E402

rr.OUT = REPO + "/data/review_regime_map"
OUTF = REPO + "/data/review_regime_map/r4_transients.jsonl"
G19 = [float(f"{s:.4g}") for s in np.geomspace(0.01, 40, 19)]  # sigma_chi_b4 grid
G14 = [float(f"{s:.4g}") for s in np.geomspace(0.02, 2, 14)]  # sigma_K_centred_d2 binary4 grid

JOBS = [
    # (exp tag, cfg overrides) -- sigma_chi_b4 centred, d = 3, teacher [3,2,1]
    *[("r4_b4", dict(chi=4, seed=3, sigma=s)) for s in G19[2:6]],
    *[("r4_b4", dict(chi=4, seed=2, sigma=s)) for s in G19[1:5]],
    *[("r4_b4", dict(chi=3, seed=3, sigma=s)) for s in G19[4:8]],
    # sigma_K_centred_d2 binary4 (K = 3), d = 2, teacher [3,2], chi = 4, seed 0 (boundary 0.069)
    *[
        (
            "r4_k3",
            dict(
                d=2,
                chi=4,
                cap=True,
                weights=[3.0, 2.0],
                method="lsoda",
                max_steps=20000,
                seed=0,
                sigma=s,
            ),
        )
        for s in G14[3:7]
    ],
]

if __name__ == "__main__":
    for tag, over in JOBS:
        cfg = dict(exp=tag, kind="binary", n=4, d=3, cap=False, model="centred", l_floor=1e-6)
        cfg.update(over)
        row = rr.run_point(cfg)
        tr = Trajectory.load(os.path.join(rr.OUT, tag, "traj", rr.run_id(cfg) + ".npz"))
        t, A = tr["t"], tr["overlaps"]
        last = np.asarray(dg.crossing_times(t, A, 0.5, last=True))
        w = np.asarray(row["event_widths"])
        ev = sorted(zip(last, w))
        rho_last = [
            (ev[i + 1][0] - ev[i][0]) / max(ev[i][1], ev[i + 1][1]) for i in range(len(ev) - 1)
        ]
        i9 = [
            int(np.argmax(A[:, r] >= 0.9)) if (A[:, r] >= 0.9).any() else len(t)
            for r in range(A.shape[1])
        ]
        bump = [
            bool(
                ((A[: i9[r], r] >= 0.5).any())
                and (A[np.argmax(A[: i9[r], r] >= 0.5) : i9[r], r] < 0.4).any()
            )
            if i9[r] > 0
            else False
            for r in range(A.shape[1])
        ]
        out = dict(
            tag=tag,
            chi=cfg["chi"],
            seed=cfg["seed"],
            sigma=cfg["sigma"],
            label=row["label"],
            rhos_pipeline=row["rhos"],
            rhos_lastcross=rho_last,
            transient_bump=bump,
            label_lastcross=(
                "saddle-to-saddle"
                if row["label"] in ("saddle-to-saddle", "rich") and all(x > 3 for x in rho_last)
                else row["label"]
                if row["label"] != "saddle-to-saddle"
                else "rich"
            ),
            l_end=row["l_end"],
            dK=row["dK"],
        )
        print(json.dumps(out, default=float), flush=True)
        with open(OUTF, "a") as f:
            f.write(json.dumps(out, default=float) + "\n")
