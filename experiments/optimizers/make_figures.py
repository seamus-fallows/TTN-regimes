"""Regenerate every figure and summary table of the optimizer comparison from saved runs.

    uv run python -m experiments.optimizers.make_figures [--results-dir DIR] [figure ...]

Figures: b4 probe balance escape deep bias later cost precond (default: all). Order: run the sweeps
(run.py, configs/*.yaml) and adam_probe.py first; summaries are built on demand from the raw outputs
($OPTIMIZERS_OUT, default /root/outputs/optimizers). Output goes to --results-dir (default
$OPTIMIZERS_RESULTS or /root/outputs/optimizers/results, outside git); the committed results were made
with --results-dir experiments/optimizers/results. Single process (each figure takes seconds).
"""

from __future__ import annotations

import os
import sys

if "--results-dir" in sys.argv:  # must be set before the experiment modules read it
    i = sys.argv.index("--results-dir")
    os.environ["OPTIMIZERS_RESULTS"] = sys.argv[i + 1]
    del sys.argv[i : i + 2]

import numpy as np
import pandas as pd

from experiments.optimizers.analysis import RESULTS, collect
from experiments.optimizers.plots import fig_adam_escape, fig_implicit_bias, fig_balance, fig_probe, fig_scaling, fig_side_by_side, scaling_table, with_keys

# Riemannian GD and ALS / DMRG are left out: their events fall within the first 1-2 steps / micro-steps
# at every sigma (unresolved); their time to convergence has p = 0 (scaling_b4_exponents.csv).
B4_SCALING = ["GF", "heavy ball, lr 0.002", "Adam, lr 0.001", "Adam, lr 0.001, b2 0.9", "Adam, lr 0.01",
              "RMSProp, lr 0.001", "GD + re-gauge, lr 0.02, every 1"]
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


def cost_table() -> pd.DataFrame:
    """Cost per method: own-clock time to L < 1e-3 L0, evaluations (loss + gradient for first-order
    methods; f and Jacobian evaluations for LSODA; micro-steps for sweeps; steps for Riemannian GD, each a
    dense Jacobian + least squares) and wall seconds (including recording), median over converged runs."""
    parts = []
    for name, flt in (("scaling_b4", {"sigma": 0.008}), ("deep", {"init": "balanced", "sigma": 0.3})):
        df = with_keys(collect(name))
        for k, v in flt.items():
            df = df[df[k] == v]
        df["problem"] = df["tree"] + df["n_sites"].astype(int).astype(str) + " " + df["teacher"]
        evals = df["meta_n_evals"].copy() if "meta_n_evals" in df else pd.Series(np.nan, index=df.index)
        if "meta_nfev" in df:
            evals = evals.fillna(df["meta_nfev"] + df.get("meta_njev", 0))
        if "meta_n_micro" in df:
            evals = evals.fillna(df["meta_n_micro"])
        df["evals"] = evals
        df["t_conv"] = df["t_loss_0.001"] * df["clock"]
        g = df.groupby(["problem", "method"])
        parts.append(g.agg(n=("seed", "count"), converged=("converged", "mean"), t_conv=("t_conv", "median"),
                           evals=("evals", "median"), wall_s=("meta_wall_seconds", "median")).reset_index())
    out = pd.concat(parts)
    out.to_csv(RESULTS / "cost_table.csv", index=False)
    return out


def order_agreement() -> pd.DataFrame:
    """Per method: fraction of inits (same problem, sigma, seed) where the method learns the components in
    the same order as GF, and where its first learned component is GF's."""
    rows = []
    for name in ("scaling_b4", "deep", "later_b8"):
        df = with_keys(collect(name))
        df["init"] = df["init"].fillna("fanin") if "init" in df else "fanin"
        key = ["tree", "n_sites", "teacher", "init", "sigma", "seed"]
        gf = df[df["optimizer"] == "gf"].set_index(key)["order"].astype(str)
        for m, g in df[df["optimizer"] != "gf"].groupby("method"):
            g = g.set_index(key)
            common = g.index.intersection(gf.index)
            if len(common) == 0:
                continue
            mine = g.loc[common, "order"].astype(str)
            ref = gf.loc[common]
            rows.append({"sweep": name, "method": m, "n_inits": len(common),
                         "same_order": float(np.mean(mine.values == ref.values)),
                         "same_first": float(np.mean([a[1:2] == b[1:2] for a, b in zip(mine, ref)]))})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "order_agreement_with_gf.csv", index=False)
    return out


def eps():
    """Adam's first escape vs sigma for eps = 1e-8 and ~0 (mps8), and the eps prediction."""
    df = with_keys(collect("eps_mps8"))
    t = df.groupby(["lr", "eps", "sigma"]).agg(
        n=("seed", "count"), escaped=("t_first", lambda x: float(np.mean(np.isfinite(x)))),
        t_first_median=("t_first", "median"), t_first_min=("t_first", "min"), t_first_max=("t_first", "max"),
        converged=("converged", "mean")).reset_index()
    t.to_csv(RESULTS / "eps_mps8_first_escape.csv", index=False)
    # per seed: measured / predicted, prediction (eps / lr) T_GF(sigma) + n_sign with T_GF(sigma) =
    # (that seed's GF first-escape time at sigma = 0.3, deep sweep, same init direction) (0.3 / sigma)^(K-2)
    deep = collect("deep")
    ref = deep[(deep["optimizer"] == "gf") & (deep["tree"] == "mps") & (deep["teacher"] == "odeco")
               & (deep["init"] == "balanced") & (deep["sigma"] == 0.3)].set_index("seed")["t_first"]
    sign = {(round(lr, 8), int(sd)): v for (lr, sd), v in
            df[df["eps"] == 1e-30].groupby(["lr", "seed"])["t_first"].median().items()}
    rows = []
    for _, r in df[(df["eps"] == 1e-8) & (df["sigma"] <= 0.05)].iterrows():
        pred = 1e-8 / r["lr"] * ref[r["seed"]] * (0.3 / r["sigma"]) ** 6 + sign[(round(r["lr"], 8), int(r["seed"]))]
        rows.append({"lr": r["lr"], "sigma": r["sigma"], "seed": r["seed"], "measured": r["t_first"],
                     "predicted": pred, "ratio": r["t_first"] / pred, "budget": r["meta_n_steps"]})
    per = pd.DataFrame(rows)
    per.to_csv(RESULTS / "eps_mps8_prediction_per_seed.csv", index=False)
    fig_adam_escape()
    return t


def precond():
    df = with_keys(collect("precond_b4"))
    df["method"] = df["method"] + df["damping_abs"].fillna(0).map(lambda v: f", abs damping {v:g}" if v > 0 else "")
    scaling_table(df, quantities=("t_loss_0.1", "t_loss_0.001")).to_csv(RESULTS / "precond_b4_exponents.csv",
                                                                      index=False)
    df.groupby(["method", "sigma"]).agg(
        n=("seed", "count"), converged=("converged", "mean"), t_conv=("t_loss_0.001", "median"),
        max_loss_over_L0=("loss_max_rel", "median")).reset_index().to_csv(RESULTS / "precond_b4_table.csv",
                                                                         index=False)


FIGURES = {"b4": b4, "probe": fig_probe, "balance": balance, "deep": deep, "bias": fig_implicit_bias, "later": later, "cost": cost_table,
           "order": order_agreement, "eps": eps, "precond": precond}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
