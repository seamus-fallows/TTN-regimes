"""Recompute the exact-saddle Hessian tables of r2_saddle.yaml with the fixed gauge projection
(library commit 6c42d85, default rtol 1e-3) and the raw lambda_min. Writes results/r2_saddle_hessian.csv."""
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

from common import RESULTS
from r2_saddle_runs import hessian_row

HERE = Path(__file__).parent
if __name__ == "__main__":
    cfg = yaml.safe_load(open(HERE / "configs/r2_saddle.yaml"))
    combos = [(tname, tcfg, n, tree) for n, block in cfg["teachers"].items()
              for tname, tcfg in block.items() for tree in cfg["trees"][n]]
    with Pool(4) as pool:
        rows = pool.map(hessian_row, combos, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "r2_saddle_hessian.csv", index=False)
    print(df[["teacher", "tree", "l_saddle", "lambda_min", "lambda_min_raw", "n_negative"]].to_string())
