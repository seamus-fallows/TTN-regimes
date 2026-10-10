"""Lazy-limit predictions for the b8_sigma_chi grid (per seed): residual and kappa."""
import sys
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, "experiments/regime_map/pilots")
from interim import lazy_row  # noqa: E402
import torch  # noqa: E402


def job(cfg):
    torch.set_num_threads(4)
    p = lazy_row(cfg)
    p.update({k: v for k, v in cfg.items() if k != "bonds"})
    return p


cfgs = [dict(kind="binary", n=8, d=3, chi=c, cap=True, seed=s) for c in [3, 6, 12, 24, 81] for s in range(6)]
cfgs += [dict(kind="binary", n=8, d=3, chi=162, cap=False, bonds={2: 18, 4: 162}, seed=s) for s in range(6)]
if __name__ == "__main__":
    with Pool(12) as pool:
        rows = pool.map(job, cfgs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv("experiments/regime_map/results/lazy_pred_b8.csv", index=False)
    print(df.groupby("chi")[["residual", "kappa", "rel_dtheta1", "cg_iter"]].median())
