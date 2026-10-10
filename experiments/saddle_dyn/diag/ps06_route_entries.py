"""Diagnostic (not registered): dormant r2-route entries of PS06 at the q saddle reached from small init.

For seeds with plateau exponent 3 vs 4, integrate from the fan-in init to the visit (L = L_S + 1e-4),
relax 50 time units, re-gauge (bond q-messages -> e_0), and print the new entries along the k* = 5
route of r2 = 110010 (cores 0-4, new bond index 1; edge 4-5 old) and the k = 6 route of r1 = 110001.
Usage: uv run python experiments/saddle_dyn/diag/ps06_route_entries.py SEED SIGMA [SIGMA ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runs  # noqa: E402
import sdyn  # noqa: E402

seed = int(sys.argv[1])
out = []
for sigma in map(float, sys.argv[2:]):
    m = runs.make_model("PS06")
    m.set_ref(None)
    y0 = runs.hb_init(m, "PS06", seed, sigma)
    ev = [sdyn.Event("visit", lambda t, y: m.residual(torch.from_numpy(y))[1] - 1e-4, -1, terminal=True)]
    res, _ = sdyn.integrate(m, y0.numpy(), 1e30, 1e-10, 1e-15, events_extra=ev, trap=False)
    res1b, _ = sdyn.integrate(m, res.y[:, -1], runs.RELAX, 1e-10, 1e-15, trap=False)
    G = [c.numpy() for c in m.net.cores(m.regauge(torch.from_numpy(res1b.y[:, -1])))]
    # MPS cores: core 0 (site, bond01); cores 1-4 (site, child bond, parent bond); core 5 (site, bond45, 1)
    r2 = {"G0[1,1]": G[0][1, 1], "G1[1,1,1]": G[1][1, 1, 1], "G2[0,1,1]": G[2][0, 1, 1],
          "G3[0,1,1]": G[3][0, 1, 1], "G4[1,1,0]": G[4][1, 1, 0], "G5[0,0,0]": G[5][0, 0, 0]}
    r1_extra = {"G4[0,1,1]": G[4][0, 1, 1], "G5[1,1,0]": G[5][1, 1, 0]}
    newn = {f"|new core {v}|": float(np.linalg.norm(c[m.new_mask.numpy()[m.net.off[v]:m.net.off[v + 1]].reshape(c.shape)]))
            for v, c in enumerate(G)}
    rec = dict(seed=seed, sigma=sigma, **{k: float(v) for k, v in {**r2, **r1_extra}.items()}, **newn)
    out.append(rec)
    print(json.dumps(rec), flush=True)
