"""GF first-event time at sigma = 0.3 (mps8 odeco balanced) for all seeds, with fine rows (accepted steps),
compared to the agent's deep-sweep t_first (coarse log+linear output grid). Also reruns with the agent's grid
(600 log + 1201 linear points to t_max = 2e6, fine rows on/off) to locate any difference.
usage: python gf_mps8_T03.py OUTDIR"""
import json
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
from experiments.optimizers.common import initial_point, make_problem  # noqa: E402
from ttn.diagnostics import crossing_times  # noqa: E402
from ttn.dynamics import Recorder, integrate_gf  # noqa: E402

agent = {0: 968.042372, 1: 4503.750535, 2: 7168.588245, 3: 5196.272831, 4: 1540.799249, 5: 3388.414025}
rows = []
for seed in range(6):
    cfg = dict(tree="mps", n_sites=8, d=3, chi=4, teacher="odeco", weights=[3.0, 2.0, 1.0], teacher_seed=100,
               sigma=0.3, seed=seed, init="balanced")
    p = make_problem(cfg)
    x0 = initial_point(p, cfg)
    L0 = p.loss(x0).item()
    res = {"seed": seed, "agent_t_first": agent[seed]}
    for label, kw in (("fine", dict(n_log=400)),
                      ("agent_grid_coarse", dict(log_times=None))):
        if label == "agent_grid_coarse":
            t_max = 2.0e6
            grid = np.unique(np.concatenate([np.geomspace(1e-3, t_max, 600), np.linspace(0, t_max, 1201)]))
            rec = Recorder(spectra=False, fine=False)
            traj = integrate_gf(p, x0, 1.5 * agent[seed], log_times=grid[(grid > 0) & (grid < 1.5 * agent[seed])],
                                rtol=1e-9, atol=1e-14, recorder=rec, method="lsoda")
        else:
            rec = Recorder(spectra=False)
            traj = integrate_gf(p, x0, 1.5 * agent[seed], rtol=1e-9, atol=1e-14,
                                recorder=rec, method="lsoda", **kw)
        d = traj.data
        use_fine = "fine_t" in d and label == "fine"
        t = d["fine_t"] if use_fine else d["t"]
        a = d["fine_overlaps"] if use_fine else d["overlaps"]
        ev = crossing_times(t, a, 0.5)
        res[label] = float(np.nanmin(ev))
        res[label + "_comp"] = int(np.nanargmin(ev))
    print(json.dumps(res), flush=True)
    rows.append(res)
with open(sys.argv[1] + "/gf_mps8_T03_resolution.json", "w") as f:
    json.dump(rows, f, indent=1)
