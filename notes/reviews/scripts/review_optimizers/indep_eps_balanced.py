"""Exact (event-located) GF first-escape times for the agent's BALANCED MPS8 inits (init built with the agent's
common.initial_point, dynamics/event mine), at sigma = 0.3 and directly at sigma = 0.03 / 0.05, to see whether
the 0.97-1.05 spread of the committed eps-law ratios is the GF output-grid interpolation error.
    uv run python indep_eps_balanced.py OUTCSV"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
sys.path.insert(0, str(Path(__file__).parent))
from experiments.optimizers.common import initial_point, make_problem  # noqa: E402  (init only)
from indep_check import Prob, first_escape_gf  # noqa: E402

torch.set_num_threads(1)
p = Prob("mps", 8, 3, 4, [3.0, 2.0, 1.0])
rows = []
for seed in range(6):
    for sigma in [0.3, 0.05, 0.03]:
        cfg = dict(
            tree="mps",
            n_sites=8,
            d=3,
            chi=4,
            teacher="odeco",
            weights=[3.0, 2.0, 1.0],
            teacher_seed=100,
            init="balanced",
            sigma=sigma,
            seed=seed,
        )
        x0 = initial_point(make_problem(cfg), cfg).detach().clone()
        t, c, nf = first_escape_gf(p, x0, 1e13)
        rows.append(dict(seed=seed, sigma=sigma, T_exact=t, comp=c))
        print(seed, sigma, t, c, nf, flush=True)
        pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
