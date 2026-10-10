"""binary8 balanced inits (agent's initial_point used ONLY to build the init), own Adam (lr 1e-3, 1e-2) and exact GF
first-escape event, full later_b8 grid (7 sigma x 6 seeds)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
sys.path.insert(0, str(Path(__file__).parent))
from experiments.optimizers.common import initial_point, make_problem
from indep_check import Prob, first_escape_gf, first_escape_adaptive
torch.set_num_threads(1)
p = Prob("binary", 8, 3, 4, [3.0, 2.0, 1.0])
rows = []
for seed in range(6):
    for sigma in [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8]:
        cfg = dict(tree="binary", n_sites=8, d=3, chi=4, teacher="odeco", weights=[3.0, 2.0, 1.0], teacher_seed=100,
                   init="balanced", sigma=sigma, seed=seed)
        x0 = initial_point(make_problem(cfg), cfg).detach().clone()
        tg, cg, _ = first_escape_gf(p, x0, 1e6)
        rows.append(dict(method="gf", lr=0, sigma=sigma, seed=seed, t_first=tg, comp=cg))
        for lr in (1e-3, 1e-2):
            n1, c1 = first_escape_adaptive(p, x0, "adam", lr, 20000)
            rows.append(dict(method="adam", lr=lr, sigma=sigma, seed=seed, t_first=n1, comp=c1))
        print(seed, sigma, [f"{r['method']} {r['lr']} {r['t_first']:.5g} c{r['comp']}" for r in rows[-3:]], flush=True)
        pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
