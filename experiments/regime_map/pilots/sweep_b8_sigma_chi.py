"""Interim sigma x chi sweep, binary N = 8 (K = 7), d = 3, odeco [3, 2, 1]; centred model.
chi in {3, 6, 12, 24, 81} with capped bonds (81 = all bonds full), plus 'full-overparameterized'
bonds 18 (bottom, 2 sites) / 162 (top, 4 sites), i.e. twice the cut dimension on every edge."""
import sys
import numpy as np
sys.path.insert(0, "experiments/regime_map/pilots")
from interim import run_sweep  # noqa: E402

EXP = "b8_sigma_chi"
SIGMAS = [float(f"{s:.4g}") for s in np.geomspace(0.03, 30, 16)]
SEEDS = range(6)
cfgs = [dict(exp=EXP, kind="binary", n=8, d=3, chi=c, cap=True, sigma=s, seed=k, model="centred")
        for c in [3, 6, 12, 24, 81] for s in SIGMAS for k in SEEDS]
cfgs += [dict(exp=EXP, kind="binary", n=8, d=3, chi=162, cap=False, bonds={2: 18, 4: 162}, sigma=s, seed=k,
              model="centred") for s in SIGMAS for k in SEEDS]
if __name__ == "__main__":
    run_sweep(cfgs, EXP, n_proc=int(sys.argv[1]) if len(sys.argv) > 1 else 340)
