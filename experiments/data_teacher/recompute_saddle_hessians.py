"""Recompute the exact-saddle Hessian tables (r2_saddle.yaml -> results/r2_saddle_hessian.csv,
r2_saddle_b.yaml -> results/r2_saddle_b_hessian.csv): gauge projection of library commit 6c42d85 (default
rtol 1e-3), raw lambda_min, saddle type by sign, per-edge rates gamma_e (fix round 2).
Usage: uv run python experiments/data_teacher/recompute_saddle_hessians.py [--procs 4]"""
import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import yaml

from common import RESULTS
from r2_saddle_runs import hessian_row, store_saddles

HERE = Path(__file__).parent
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    for cfgname, out in (("r2_saddle", "r2_saddle_hessian"), ("r2_saddle_b", "r2_saddle_b_hessian")):
        cfg = yaml.safe_load(open(HERE / "configs" / f"{cfgname}.yaml"))
        combos = [(tname, tcfg, n, tree) for n, block in cfg["teachers"].items()
                  for tname, tcfg in block.items() for tree in cfg["trees"][n]]
        store_saddles(combos)  # computed once here, loaded by the (spawned) workers
        with mp.get_context("spawn").Pool(a.procs) as pool:
            rows = pool.map(hessian_row, combos, chunksize=1)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / f"{out}.csv", index=False)
        print(df[["teacher", "tree", "lambda_min", "n_negative", "n_negative_lib", "edge_gammas"]].to_string())
