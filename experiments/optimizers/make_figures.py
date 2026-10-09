"""Regenerate every figure and summary table of the optimizer comparison from saved runs.

    uv run python -m experiments.optimizers.make_figures [b4] [deep] [eps] [bias] [probe] [precond]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments.optimizers.analysis import RESULTS, collect
from experiments.optimizers.plots import fig_adam_escape, fig_implicit_bias, fig_balance, fig_probe, fig_scaling, fig_side_by_side, scaling_table, with_keys

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
    ex.append(("deep", {"tree": "mps", "teacher": "odeco", "init": "balanced", "sigma": 0.8, "seed": 0},
               ["GF", "GD, lr 0.01", "heavy ball, lr 0.001", "Adam, lr 0.001", "RMSProp, lr 0.001"]))
    scatter = [("scaling_b4", {}),
               ("deep", {"tree": "binary", "teacher": "odeco", "init": "balanced"}),
               ("deep", {"tree": "mps", "teacher": "odeco", "init": "balanced"})]
    fig_balance(ex, scatter)


def summary_table(name: str, by=("tree", "teacher", "init", "sigma")) -> pd.DataFrame:
    """Per problem and method: convergence, first escape and gaps (method clock), S2S fraction, orders,
    rank-growth pattern, balance, cost."""
    df = with_keys(collect(name))
    df["T_first"] = df["t_first"] * df["clock"]
    df["gap0"] = df["gap_0"] * df["clock"]
    df["gap1"] = df["gap_1"] * df["clock"] if "gap_1" in df else np.nan
    df["evals"] = df["meta_n_evals"].fillna(df.get("meta_nfev", np.nan)).fillna(df.get("meta_n_micro", np.nan))
    out = df.groupby(list(by) + ["method"]).agg(
        n=("seed", "count"), converged=("converged", "mean"),
        T_first=("T_first", "median"), gap0=("gap0", "median"), gap1=("gap1", "median"),
        s2s=("s2s", "mean"), orders=("order", lambda x: " ".join(f"{k[1:]}:{v}" for k, v in
                                                                  x.astype(str).value_counts().items())),
        rank_incremental=("spec_incremental", "mean"), edge_spread=("spec_edge_spread", "median"),
        eff_ranks=("eff_ranks_end", lambda x: x.astype(str).value_counts().index[0]),
        balance_end=("balance_end_max", "median"), wall_s=("meta_wall_seconds", "median"),
        evals=("evals", "median")).reset_index()
    out.to_csv(RESULTS / f"{name}_table.csv", index=False)
    return out


def deep():
    summary_table("deep")
    side = ["GF", "Adam, lr 0.001", "GD + re-gauge, lr 0.01, every 1", "Riemannian GD, lr 1", "DMRG (2-site)"]
    for tree, teacher, edge in (("binary", "odeco", 2), ("mps", "odeco", 3), ("mps", "pattern", 3),
                                ("binary", "pattern", 2)):
        fig_side_by_side("deep", 0.3, 0, side, edge_index=edge, tag=f"{tree}_{teacher}_",
                         extra_filter={"tree": tree, "teacher": teacher, "init": "balanced"})


def later():
    df = with_keys(collect("later_b8"))
    scaling_table(df, quantities=("t_first", "gap_0", "gap_1", "t_loss_0.001")).to_csv(
        RESULTS / "later_b8_exponents.csv", index=False)
    fig_scaling("later_b8", ["GF", "Adam, lr 0.001", "Adam, lr 0.01", "GD + re-gauge, lr 0.01, every 1"])


FIGURES = {"b4": b4, "probe": fig_probe, "balance": balance, "escape": fig_adam_escape, "deep": deep, "bias": fig_implicit_bias, "later": later}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
