"""eps law on binary4 (K = 3) far below sigma_eps = (eps/c_g)^(1/2) = 2e-4: own Adam vs exact GF event."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from indep_check import Prob, first_escape_gf, first_escape_adaptive
torch.set_num_threads(1)
p = Prob("binary", 4, 3, 4, [3.0, 2.0, 1.0])
rows = []
for seed in [0, 1]:
    for sigma in [1e-4, 1e-6, 1e-8, 1e-9]:
        th0 = p.init(sigma, seed)
        t0 = time.time()
        tg, cg, _ = first_escape_gf(p, th0, 100.0 / sigma, atol=1e-15 * sigma)
        n1, c1 = first_escape_adaptive(p, th0, "adam", 1e-3, 400000, eps=1e-8)
        n0, c0 = first_escape_adaptive(p, th0, "adam", 1e-3, 400000, eps=1e-30)
        pred = 1e-8 / 1e-3 * tg + n0
        rows.append(dict(seed=seed, sigma=sigma, T_GF=tg, n1_eps1e8=n1, n_sign=n0, pred=pred, ratio=n1 / pred))
        print(rows[-1], f"{time.time() - t0:.1f}s", flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
