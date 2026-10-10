"""Fair comparison of a logarithmic and a power law for gap(sigma), per seed (review of round 1).

Both laws have two parameters and are fitted by least squares in the SAME space (log of the gap):
  log law:   log t = log(A ln(1/sigma) + B)
  power law: log t = log C - p log sigma
The score is the rms of the log residuals. Fits are per seed (the same seed, i.e. the same normalized
init direction, is used at every sigma), and errors are standard errors over seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def fit_power(sig, t):
    x, y = np.log(sig), np.log(t)
    a = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - a @ coef) ** 2)))
    return dict(p=-coef[0], logC=coef[1], rms_pow=rms)


def fit_log(sig, t):
    x, y = np.log(1 / sig), np.log(t)
    # start from the linear least-squares fit in t-space
    a = np.vstack([x, np.ones_like(x)]).T
    (A0, B0), *_ = np.linalg.lstsq(a, t, rcond=None)
    A0 = max(A0, 1e-6)
    B0 = B0 if np.all(A0 * x + B0 > 0) else -A0 * x.min() + 0.5 * t.min()

    def res(p):
        v = p[0] * x + p[1]
        return np.where(v > 0, np.log(np.maximum(v, 1e-300)), -50.0) - y

    r = least_squares(res, [A0, B0])
    return dict(A=float(r.x[0]), B=float(r.x[1]), rms_log=float(np.sqrt(np.mean(r.fun**2))))


def per_seed(df, group_cols, sigma_col="sigma", t_col="gap", min_points=4):
    """Per-seed fits of both laws; returns (per_seed_table, summary_table)."""
    rows = []
    for key, g in df.groupby(group_cols + ["seed"]):
        g = g[np.isfinite(g[t_col]) & (g[t_col] > 0)].sort_values(sigma_col)
        if g[sigma_col].nunique() < min_points:
            continue
        s, t = g[sigma_col].values, g[t_col].values
        r = dict(zip(group_cols + ["seed"], key if isinstance(key, tuple) else (key,)))
        r.update(n_sigma=len(g), sigma_min=s.min(), sigma_max=s.max(), **fit_log(s, t), **fit_power(s, t))
        r["log_wins"] = r["rms_log"] < r["rms_pow"]
        rows.append(r)
    ps = pd.DataFrame(rows)
    if not len(ps):
        return ps, ps
    summ = ps.groupby(group_cols).agg(
        n_seeds=("seed", "size"), A_mean=("A", "mean"), A_sem=("A", "sem"),
        p_mean=("p", "mean"), p_sem=("p", "sem"), rms_log_median=("rms_log", "median"),
        rms_pow_median=("rms_pow", "median"), log_wins=("log_wins", "sum")).reset_index()
    return ps, summ


def per_seed_local(df, group_cols, sigma_col="sigma", t_col="gap", n_small=3, edge_col=None):
    """Asymptotic slopes (theory_teachers.md section 9 protocol): per seed, a straight line through the
    n_small smallest sigma, t = A ln(1/sigma) + B (slope A_loc) and log t = log C - p log sigma (p_loc).
    With edge_col: the edge that grows first at each of those sigma; edge_loc = that edge if it is the
    same at all of them (else -1: the seed's fit mixes routes). Returns (per_seed_table, summary_table)."""
    rows = []
    for key, g in df.groupby(group_cols + ["seed"]):
        g = g[np.isfinite(g[t_col]) & (g[t_col] > 0)].sort_values(sigma_col).head(n_small)
        if g[sigma_col].nunique() < n_small:
            continue
        x, t = np.log(1 / g[sigma_col].values), g[t_col].values
        r = dict(zip(group_cols + ["seed"], key if isinstance(key, tuple) else (key,)))
        r.update(n_loc=len(g), sigma_loc_max=float(g[sigma_col].max()),
                 A_loc=float(np.polyfit(x, t, 1)[0]), p_loc=float(np.polyfit(x, np.log(t), 1)[0]))
        if edge_col is not None:
            e = g[edge_col].unique()
            r["edge_loc"] = int(e[0]) if len(e) == 1 else -1
        rows.append(r)
    ps = pd.DataFrame(rows)
    if not len(ps):
        return ps, ps
    summ = ps.groupby(group_cols).agg(
        n_seeds_loc=("seed", "size"), sigma_loc_max=("sigma_loc_max", "max"),
        A_loc_mean=("A_loc", "mean"), A_loc_sem=("A_loc", "sem"),
        p_loc_mean=("p_loc", "mean"), p_loc_sem=("p_loc", "sem")).reset_index()
    return ps, summ
