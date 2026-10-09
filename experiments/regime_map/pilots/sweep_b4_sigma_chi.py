"""Interim sigma x chi sweep, binary N = 4, d = 3, odeco [3, 2, 1]; centred, plain, two-copy."""
import sys
import numpy as np
sys.path.insert(0, "experiments/regime_map/pilots")
from interim import run_sweep  # noqa: E402

EXP = "b4_sigma_chi"
CHIS = [3, 4, 6, 9, 12, 18, 24, 36]
SIGMAS = [float(f"{s:.4g}") for s in np.geomspace(0.01, 40, 19)]
SEEDS = range(6)
cfgs = [dict(exp=EXP, kind="binary", n=4, d=3, chi=c, cap=False, sigma=s, seed=k, model=m)
        for m in ["centred", "plain", "twocopy"] for c in CHIS for s in SIGMAS for k in SEEDS]
if __name__ == "__main__":
    run_sweep(cfgs, EXP)
