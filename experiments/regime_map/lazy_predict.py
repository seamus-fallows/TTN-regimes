"""Per-seed lazy-limit predictions (lazy residual, kappa = sigma^K delta_K at first order) for a grid
of (tree, chi, seed); written to results/lazy_pred_<name>.csv. Usage:
uv run python experiments/regime_map/lazy_predict.py <name> <kind> <n> <d> <chis> <n_seeds> <cap> [lr]"""
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, "experiments/regime_map/pilots")
from interim import lazy_row  # noqa: E402


def job(cfg):
    p = lazy_row(cfg)
    p.update(cfg)
    return p


if __name__ == "__main__":
    name, kind, n, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    chis = [int(c) for c in sys.argv[5].split(",")]
    seeds, cap = int(sys.argv[6]), sys.argv[7] == "cap"
    lr = sys.argv[8] if len(sys.argv) > 8 else "sp"
    cfgs = [dict(kind=kind, n=n, d=d, chi=c, cap=cap, seed=s, lr=lr) for c in chis for s in range(seeds)]
    with Pool(min(len(cfgs), 24)) as pool:
        rows = pool.map(job, cfgs)
    df = pd.DataFrame(rows)
    df.to_csv(f"experiments/regime_map/results/lazy_pred_{name}.csv", index=False)
    print(df.groupby("chi")[["residual", "kappa", "rel_dtheta1"]].agg(["median", "mean"]))
