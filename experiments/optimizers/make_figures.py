"""Regenerate every figure and summary table of the optimizer comparison from saved runs.

    uv run python -m experiments.optimizers.make_figures [b4] [deep] [eps] [bias] [probe] [precond]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments.optimizers.analysis import RESULTS, collect
from experiments.optimizers.plots import fig_balance, fig_probe, fig_scaling, fig_side_by_side, scaling_table, with_keys

B4_SCALING = ["GF", "heavy ball, lr 0.002", "Adam, lr 0.001", "Adam, lr 0.001, b2 0.9", "RMSProp, lr 0.001",
              "GD + re-gauge, lr 0.02, every 1", "Riemannian GD, lr 1", "ALS (1-site)"]
B4_SIDE = ["GF", "Adam, lr 0.001", "GD + re-gauge, lr 0.02, every 1", "Riemannian GD, lr 0.1", "ALS (1-site)"]


def order_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per method and sigma: fraction of runs with weight order (012), S2S fraction (rho > 3 rule),
    median separation ratios."""
    g = df.groupby(["method", "sigma"])
    return g.agg(n=("seed", "count"), converged=("converged", "mean"),
                 weight_order=("order", lambda x: float(np.mean([str(o) == "o012" for o in x]))),
                 s2s=("s2s", "mean"), rho0_median=("rho_0", "median"), rho1_median=("rho_1", "median"),
                 balance_end=("balance_end_max", "median")).reset_index()


def b4():
    df = with_keys(collect("scaling_b4"))
    st = scaling_table(df)
    st.to_csv(RESULTS / "scaling_b4_exponents.csv", index=False)
    order_table(df).to_csv(RESULTS / "scaling_b4_order.csv", index=False)
    fig_scaling("scaling_b4", B4_SCALING)
    fig_side_by_side("scaling_b4", 0.008, 0, B4_SIDE, edge_index=0)


def balance():
    ex = [("scaling_b4", {"sigma": 0.008, "seed": 0},
           ["GF", "GD, lr 0.02", "heavy ball, lr 0.002", "Adam, lr 0.001", "RMSProp, lr 0.001"])]
    scatter = [("scaling_b4", {})]
    fig_balance(ex, scatter)


FIGURES = {"b4": b4, "probe": fig_probe, "balance": balance}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
