"""Consistency of this experiment's GF / GD / Adam baselines with notes/theory_optimizer.md (section 9).

    uv run python -m experiments.optimizers.consistency
        -> results/theory_optimizer_P1P2.csv, results/theory_optimizer_P6.csv

P1/P2: Hessian lambda_max at the end of converged runs (theta_end) vs K s_1^(2-2/K) (balanced minimum);
       GF ends near balance, Adam / RMSProp at imbalanced minima, which A2/P2 says are at least as sharp.
P6:    GD event shifts t_GD - t_GF (same seed and sigma) vs (lr/2) ln(|grad L(t_r)| / |grad L(0)|), the
       gradient norms read off the GF trajectory (the recorder's `speed`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from experiments.optimizers.analysis import RESULTS, collect, load
from experiments.optimizers.common import make_problem
from experiments.optimizers.plots import with_keys


def lambda_max(row) -> float:
    cfg = {k: row[k] for k in ("tree", "n_sites", "d", "chi", "teacher", "teacher_seed")}
    for k in ("n_sites", "d", "chi", "teacher_seed"):
        cfg[k] = int(cfg[k])
    cfg["weights"] = [float(w) for w in json.loads(row["weights"])]
    if cfg["teacher"] == "pattern":
        cfg["patterns"] = json.loads(row["patterns"])
    p = make_problem(cfg)
    theta = torch.from_numpy(load(row["sweep"], row["file"]).data["theta_end"])
    return torch.linalg.eigvalsh(p.hessian(theta))[-1].item()


def p1p2(sweeps=("scaling_b4", "deep"), per_group: int = 3) -> pd.DataFrame:
    rows = []
    for sweep in sweeps:
        df = with_keys(collect(sweep))
        df["sweep"] = sweep
        df = df[df["converged"] & df["optimizer"].isin(["gf", "gd", "adam", "rmsprop", "momentum"])
                & (df["teacher"] == "odeco")]
        keys = [k for k in ("tree", "n_sites", "init", "sigma", "method") if k in df]
        for _, g in df.groupby(keys):
            for _, r in g.head(per_group).iterrows():
                s1 = max(json.loads(r["weights"]))
                k = {"binary": int(r["n_sites"]) - 1, "mps": int(r["n_sites"])}[r["tree"]]
                pred = k * s1 ** (2 - 2 / k)
                lam = lambda_max(r)
                rows.append({"sweep": sweep, "tree": r["tree"], "n_sites": r["n_sites"],
                             "init": r.get("init", "fanin"), "sigma": r["sigma"], "seed": r["seed"],
                             "method": r["method"], "lambda_max": lam, "pattern_rule": pred,
                             "ratio": lam / pred, "balance_end": r["balance_end_max"]})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "theory_optimizer_P1P2.csv", index=False)
    return out


def p6(sweep="scaling_b4") -> pd.DataFrame:
    df = with_keys(collect(sweep))
    gf = df[df["optimizer"] == "gf"].set_index(["sigma", "seed"])
    gd = df[df["optimizer"] == "gd"]
    rows = []
    for _, r in gd.iterrows():
        key = (r["sigma"], r["seed"])
        if key not in gf.index:
            continue
        g = gf.loc[key]
        d = load(sweep, g["file"]).data
        lr = r["lr"]
        for c in range(3):
            t_gf, t_gd = g[f"t_{c}"], r[f"t_{c}"] * lr
            if not (np.isfinite(t_gf) and np.isfinite(t_gd)):
                continue
            speed = np.interp(t_gf, d["t"], d["speed"])
            pred = lr / 2 * np.log(speed / d["speed"][0])
            rows.append({"sigma": r["sigma"], "seed": r["seed"], "component": c, "lr": lr,
                         "dt_measured": t_gd - t_gf, "dt_predicted": pred,
                         "dt_over_lr": (t_gd - t_gf) / lr, "pred_over_lr": pred / lr})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "theory_optimizer_P6.csv", index=False)
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    pd.set_option("display.width", 200)
    import sys

    if "p6" in sys.argv:
        a = p6()
        print(a.groupby(["sigma", "component"])[["dt_over_lr", "pred_over_lr"]].mean().round(2).to_string())
    b = p1p2()
    print(b.groupby(["tree", "n_sites", "init", "method"])[["ratio", "balance_end"]].agg(["mean", "min", "max"])
          .round(3).to_string())
