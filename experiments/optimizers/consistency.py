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


def p6_slope(sweep="scaling_b4") -> pd.DataFrame:
    """P6, asymptotic form: the GD - GF event shift grows by lr (K - 1) / 2 per e-fold of 1/sigma. Per seed,
    fit the shift (GF events located exactly, GD from its fine trace) against ln(1/sigma); mean +- s.e.
    over seeds. Heavy ball is reported alongside (lr -> lr / (1 - beta) in the clock)."""
    df = with_keys(collect(sweep))
    gf = df[df["optimizer"] == "gf"].set_index(["sigma", "seed"])
    rows = []
    for opt in ("gd", "momentum"):
        o = df[df["optimizer"] == opt].set_index(["sigma", "seed"])
        for comp in ("t_first", "t_1", "t_2"):
            slopes = []
            for seed in sorted(df["seed"].unique()):
                sig = sorted(df["sigma"].unique())
                y = np.array([o.loc[(sg, seed), comp] * o.loc[(sg, seed), "clock"] - gf.loc[(sg, seed), comp]
                              for sg in sig])
                x = np.log(1 / np.array(sig))
                ok = np.isfinite(y)
                slopes.append(np.polyfit(x[ok], y[ok], 1)[0])
            slopes = np.array(slopes)
            lr = float(o["lr"].iloc[0])
            rows.append({"method": opt, "event": comp, "slope_per_efold": slopes.mean(),
                         "slope_se": slopes.std(ddof=1) / np.sqrt(len(slopes)), "n_seeds": len(slopes),
                         "p6_prediction_gd": lr * 2 / 2 if opt == "gd" else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "theory_optimizer_P6_slope.csv", index=False)
    return out


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

    print(p6_slope().round(4).to_string())
    if "p6" in sys.argv:  # path form (needs |grad L| at the events, read off the GF output grid: coarse)
        a = p6()
        print(a.groupby(["sigma", "component"])[["dt_over_lr", "pred_over_lr"]].mean().round(2).to_string())
    b = p1p2()
    print(b.groupby(["tree", "n_sites", "init", "method"])[["ratio", "balance_end"]].agg(["mean", "min", "max"])
          .round(3).to_string())


def r12_route_ends(files: list, sweep_dir) -> pd.DataFrame:
    """Test of the conservation-law trap (notes/theory_saddles.md R12) at the two ends of MPS N=8.

    On the end edges (edge 0: leaf core 0 / core 1; edge 6: core 6 / root core 7) take nu = the unit bond
    direction orthogonal to the bond vectors of the components learned at the end (two of three). R12
    predicts for a stalled run: the lighter side's column in nu has died (norm^2 ~ 0) and the heavier side
    keeps norm^2 = |nu^T Q_e(0) nu| (Q is conserved by GF); fan-in gives E[Q_0] = sigma^2 (1 - 4/3) < 0
    (leaf lighter) and E[Q_6] = sigma^2 (1 - 1/3) > 0 (root lighter)."""
    from pathlib import Path

    from experiments.optimizers.common import initial_point
    from ttn.network import conserved_quantities

    rows = []
    for f in files:
        tr = load_traj(Path(sweep_dir) / f)
        cfg = tr.meta["cfg"]
        p = make_problem(cfg)
        spec = p.spec
        th0 = initial_point(p, cfg)
        th1 = torch.from_numpy(tr.data["theta_end"])
        c0, c1 = spec.unflatten(th0), spec.unflatten(th1)
        q0 = conserved_quantities(spec, c0)
        a_end = p.target.component_overlaps(spec, c1).detach().numpy()
        learned = [r for r in range(len(a_end)) if a_end[r] > 0.5]
        u = p.target.factors  # (R, N, d)
        row = {"file": f, "init": cfg.get("init", "fanin"), "sigma": cfg["sigma"], "seed": cfg["seed"],
               "a_end": " ".join(f"{x:.2g}" for x in a_end)}
        # edge 0: child = leaf core 0 (site x bond), parent = core 1 (site, child bond, parent bond)
        g0 = c1[0]
        b = torch.stack([g0.T @ u[q, 0] for q in learned], 1)
        g1 = torch.movedim(c1[1], 1, 0).reshape(c1[1].shape[1], -1)
        # edge 6: child = core 6 (site, child bond, parent bond), parent = root core 7 (site, child bond, 1)
        g6 = c1[6].reshape(-1, c1[6].shape[-1])
        g7 = torch.movedim(c1[7], 1, 0).reshape(c1[7].shape[1], -1)
        c = torch.stack([c1[7][:, :, 0].T @ u[q, 7] for q in learned], 1)
        for name, vecs, child, parent, e in (("e0", b, g0, g1, 0), ("e6", c, g6, g7, 6)):
            qmat, _ = torch.linalg.qr(vecs, mode="complete")
            nu = qmat[:, -1]
            row[f"{name}_child2"] = (child @ nu).pow(2).sum().item()
            row[f"{name}_parent2"] = (nu @ parent).pow(2).sum().item()
            row[f"{name}_Qnunu0"] = (nu @ q0[e] @ nu).item()
        row["converged"] = bool(len(learned) == len(a_end))
        rows.append(row)
    return pd.DataFrame(rows)


def load_traj(path):
    from ttn.dynamics import Trajectory

    return Trajectory.load(path)
