"""Per-seed lazy-limit predictions (lazy residual, kappa = sigma^K delta_K at first order) for a grid
of (tree, chi, seed); written to <out>/lazy_pred_<name>.csv (refuses to overwrite). Usage:
uv run python experiments/regime_map/lazy_predict.py <name> <kind> <n> <d> <chis> <n_seeds> <cap> [lr] [out] [n_proc]"""
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, "experiments/regime_map/pilots")
from interim import lazy_row  # noqa: E402  (same init/teacher/probes as the runs)


def job(cfg):
    p = lazy_row(cfg)
    p.update(cfg)
    return p


if __name__ == "__main__":
    name, kind, n, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    chis = [int(c) for c in sys.argv[5].split(",")]
    seeds, cap = int(sys.argv[6]), sys.argv[7] == "cap"
    lr = sys.argv[8] if len(sys.argv) > 8 else "sp"
    out = sys.argv[9] if len(sys.argv) > 9 else "experiments/regime_map/results"
    n_proc = int(sys.argv[10]) if len(sys.argv) > 10 else 24
    path = f"{out}/lazy_pred_{name}.csv"
    if __import__("os").path.exists(path):
        raise FileExistsError(path)
    cfgs = [dict(kind=kind, n=n, d=d, chi=c, cap=cap, seed=s, lr=lr) for c in chis for s in range(seeds)]
    with Pool(min(len(cfgs), n_proc)) as pool:
        rows = pool.map(job, cfgs)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(df.groupby("chi")[["residual", "kappa", "rel_dtheta1"]].agg(["median", "mean"]))
