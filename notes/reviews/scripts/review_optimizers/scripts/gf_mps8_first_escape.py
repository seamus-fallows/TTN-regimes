"""Independent check of the GF assumption behind the eps-crossover prediction (notes/optimizers.md 2.7):
T_GF(sigma) = T_GF(0.3) (0.3/sigma)^6 on mps8 odeco (d=3, chi=4, weights 3,2,1, teacher seed 100), balanced init.
Runs GF (library LSODA, rtol 1e-9, atol 1e-14) until the loss falls below 0.45 L0 (just after the first
component, s=3, crosses a=1/2) and reports the first-event time per seed and sigma.
usage: python gf_mps8_first_escape.py OUTDIR seed [seed ...]"""
import json
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
from experiments.optimizers.common import initial_point, make_problem  # noqa: E402
from ttn.diagnostics import crossing_times  # noqa: E402
from ttn.dynamics import Recorder, integrate_gf  # noqa: E402

out = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]]
T03 = {0: 968.042372, 1: 4503.750535, 2: 7168.588245, 4: 1540.799249}  # agent's deep sweep, t_first at 0.3
rows = []
for seed in seeds:
    for sigma in (0.3, 0.15, 0.1):
        cfg = dict(tree="mps", n_sites=8, d=3, chi=4, teacher="odeco", weights=[3.0, 2.0, 1.0], teacher_seed=100,
                   sigma=sigma, seed=seed, init="balanced")
        p = make_problem(cfg)
        x0 = initial_point(p, cfg)
        L0 = p.loss(x0).item()
        pred = T03[seed] * (0.3 / sigma) ** 6
        t0 = time.time()
        rec = Recorder(spectra=False)
        traj = integrate_gf(p, x0, 3.0 * pred, n_log=400, rtol=1e-9, atol=1e-14, loss_floor=0.45 * L0,
                            recorder=rec, method="lsoda")
        d = traj.data
        t = d["fine_t"] if "fine_t" in d else d["t"]
        a = d["fine_overlaps"] if "fine_overlaps" in d else d["overlaps"]
        ev = crossing_times(t, a, 0.5)
        first = float(np.nanmin(ev))
        r = dict(seed=seed, sigma=sigma, t_first=first, comp=int(np.nanargmin(ev)), pred_from_agent_T03=pred,
                 ratio=first / pred, stop=traj.meta.get("stop_reason"), wall=time.time() - t0, L0=L0)
        print(json.dumps(r), flush=True)
        rows.append(r)
with open(f"{out}/gf_mps8_first_escape_seeds{'_'.join(map(str, seeds))}.json", "w") as f:
    json.dump(rows, f, indent=1)
