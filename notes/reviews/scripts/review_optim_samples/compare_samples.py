"""Reviewer: per-seed finite-sample results of the re-run (new data/init seeds 5-7) vs committed seeds 0-4."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
rows = []
for f in Path(sys.argv[1]).glob("*.jsonl"):
    for line in f.read_text().splitlines():
        d = json.loads(line)
        tk = d["task"]
        base = dict(chi=tk["chi"], n=tk["n_train"], seed=tk["seed"])
        rows.append(
            dict(
                base,
                method="small GD",
                test=d["test_rel"],
                train=d["train_loss"],
                rank=np.mean(d["eff_ranks"]),
                status=d["status"],
            )
        )
        rows.append(
            dict(
                base,
                method="lazy",
                test=d["lazy_test_rel"],
                train=np.nan,
                rank=np.mean(d["lazy_eff_ranks"]),
                status=f"rankJ={d['rank_J']}",
            )
        )
df = pd.DataFrame(rows).sort_values(["chi", "method", "n", "seed"])
print(df.to_string(float_format=lambda v: f"{v:.3g}"))
g = df.groupby(["chi", "method", "n"]).agg(
    mean=("test", "mean"),
    median=("test", "median"),
    max=("test", "max"),
    rec=("test", lambda v: f"{(v < 1e-2).sum()}/{len(v)}"),
    rank=("rank", "mean"),
)
print(g.to_string(float_format=lambda v: f"{v:.3g}"))
