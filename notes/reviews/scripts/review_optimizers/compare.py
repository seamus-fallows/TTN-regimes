"""Compare reproduced runs (agent code, data/review_optimizers/raw) with the committed summaries.

uv run python compare.py [sweep ...]
"""

import os
import sys
from pathlib import Path

REPO = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
OUT = REPO / "data/review_optimizers"
os.environ["OPTIMIZERS_OUT"] = str(OUT / "raw")
os.environ["OPTIMIZERS_RESULTS"] = str(OUT / "results")
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.optimizers.analysis import collect  # noqa: E402
from experiments.optimizers.plots import scaling_table, with_keys  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

NUM = [
    "t_first",
    "t_0",
    "t_1",
    "t_2",
    "gap_0",
    "gap_1",
    "meta_n_steps",
    "loss_end",
    "pop_loss_end",
    "balance_end_max",
    "t_loss_0.001",
    "cons_drift_end",
]
CAT = ["order", "converged", "eff_ranks_end", "meta_stop_reason", "t_first_resolved", "s2s"]


def committed(name):
    return pd.read_csv(
        REPO / f"experiments/optimizers/results/{name}_summary.csv",
        dtype={"order": str},
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )


def per_run(name):
    mine = with_keys(collect(name, refresh=True))
    theirs = committed(name)
    m = mine.merge(theirs, on="file", suffixes=("", "_c"))
    rows = []
    for meth, g in m.groupby("method"):
        r = {"sweep": name, "method": meth, "n": len(g)}
        for c in NUM:
            if c in g and c + "_c" in g:
                a, b = g[c].astype(float), g[c + "_c"].astype(float)
                both = np.isfinite(a) & np.isfinite(b)
                nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
                rel = (np.abs(a - b) / np.maximum(np.abs(b), 1e-300))[both]
                r[f"{c}_maxrel"] = rel.max() if len(rel) else np.nan
                if nan_mismatch:
                    r[f"{c}_nanmismatch"] = nan_mismatch
        for c in CAT:
            if c in g and c + "_c" in g:
                r[f"{c}_same"] = float(
                    np.mean(g[c].astype(str).values == g[c + "_c"].astype(str).values)
                )
        rows.append(r)
    return mine, m, pd.DataFrame(rows)


def exponents(name, mine, quantities=("t_first", "gap_0", "gap_1", "t_loss_0.001")):
    """Per-seed exponents (agent's scaling_table) on my reproduced subset, on the committed rows of the
    same subset, and the committed full-sweep values."""
    theirs = with_keys(committed(name))
    sub = theirs[theirs["file"].isin(mine["file"])]
    a = scaling_table(mine, quantities)[
        [
            "method",
            "quantity",
            "exponent",
            "exponent_se",
            "n_seeds",
            "per_seed",
            "n_unresolved",
            "n_other_order",
        ]
    ]
    b = scaling_table(sub, quantities)[
        ["method", "quantity", "exponent", "exponent_se", "per_seed"]
    ]
    full = (
        pd.read_csv(REPO / f"experiments/optimizers/results/{name}_exponents.csv")
        if (REPO / f"experiments/optimizers/results/{name}_exponents.csv").exists()
        else None
    )
    out = a.merge(b, on=["method", "quantity"], suffixes=("", "_committed_subset"))
    if full is not None:
        out = out.merge(
            full[["method", "quantity", "exponent", "exponent_se", "n_seeds"]],
            on=["method", "quantity"],
            how="left",
            suffixes=("", "_committed_full"),
        )
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ["scaling_b4", "later_b8", "eps_mps8", "deep", "implicit_bias_b4"]
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    for name in names:
        if not (OUT / "raw" / name).exists():
            continue
        mine, merged, table = per_run(name)
        table.to_csv(OUT / "results" / f"cmp_{name}_per_run.csv", index=False)
        merged.to_csv(OUT / "results" / f"cmp_{name}_merged.csv", index=False)
        print("=====", name, len(mine), "runs")
        cols = [
            c
            for c in table.columns
            if c.endswith("_maxrel")
            or c.endswith("_same")
            or c in ("method", "n")
            or c.endswith("_nanmismatch")
        ]
        print(table[cols].round(4).to_string())
        if name in ("scaling_b4", "later_b8"):
            ex = exponents(name, mine)
            ex.to_csv(OUT / "results" / f"cmp_{name}_exponents.csv", index=False)
            print(ex.round(3).to_string())
