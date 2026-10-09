"""Interim sigma x K sweep: binary N = 4, 8, 16 (K = 3, 7, 15) and MPS N = 4, 5, 6, 8, 10, 12
(K = N); d = 3, chi = 3 (teacher rank), odeco [3, 2, 1], centred model."""
import sys
import numpy as np
sys.path.insert(0, "experiments/regime_map/pilots")
from interim import run_sweep  # noqa: E402

EXP = "sigma_K"
SIGMAS = [float(f"{s:.4g}") for s in np.geomspace(0.05, 10, 15)]
TREES = [("binary", 4), ("binary", 8), ("binary", 16)] + [("mps", n) for n in (4, 5, 6, 8, 10, 12)]
cfgs = [dict(exp=EXP, kind=k, n=n, d=3, chi=3, cap=True, sigma=s, seed=seed, model="centred")
        for k, n in TREES for s in SIGMAS for seed in range(6)]
if __name__ == "__main__":
    run_sweep(cfgs, EXP, n_proc=int(sys.argv[1]) if len(sys.argv) > 1 else 340)
