"""Exponent / log-law fits of the escape time t2 - t1 vs sigma (per seed, then mean +- SE)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd


def load(path):
    rows = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(rows)
    df["S"] = df["S"].apply(lambda s: "".join(map(str, s)))
    ta, tb = df["a0_0.5"].astype(float), df["a1_0.5"].astype(float)
    df["greedy"] = ta <= tb  # the larger component crossed 1/2 first
    df["t1"] = np.minimum(ta, tb)
    df["t2"] = np.maximum(ta, tb)
    df["rotation"] = False
    if "all_a0_0.5" in df:
        # With every crossing recorded: t2 = last *final* upward crossing (a component that is
        # learned, lost and relearned, as in the H1b piggyback rotation, counts when relearned).
        def final_up(c):
            return c[-1] if isinstance(c, list) and len(c) % 2 == 1 else np.nan
        fa = df["all_a0_0.5"].apply(final_up)
        fb = df["all_a1_0.5"].apply(final_up)
        has = df["all_a0_0.5"].apply(lambda c: isinstance(c, list))
        df.loc[has, "t2"] = np.maximum(fa, fb)[has]
        df["rotation"] = has & (df["all_a0_0.5"].apply(lambda c: isinstance(c, list) and len(c) > 1)
                                | df["all_a1_0.5"].apply(lambda c: isinstance(c, list) and len(c) > 1))
    df["dt"] = df["t2"] - df["t1"]
    w0 = df["a0_0.9"].astype(float) - df["a0_0.1"].astype(float)
    w1 = df["a1_0.9"].astype(float) - df["a1_0.1"].astype(float)
    df["rho"] = df["dt"] / np.maximum(w0, w1)
    return df


def slope_se(x, y):
    """Least-squares slope and its standard error."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    n = len(x)
    if n <= 2:
        return coef[0], np.nan
    s2 = np.sum((y - A @ coef) ** 2) / (n - 2)
    return coef[0], np.sqrt(s2 / np.sum((x - x.mean()) ** 2))


def mean_se(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)


def summarize(df, group=("kind", "n", "S", "d", "chi"), only_greedy=True):
    out = []
    for key, g in df.groupby(list(group), dropna=False):
        row = dict(zip(group, key))
        row["K"] = int(g["K"].iloc[0])
        row["k_pred"] = int(g["k_pred"].iloc[0])
        s1, s2 = g["weights"].iloc[0][:2]
        lam2 = s2 * s1 ** ((row["K"] - 2) / row["K"])
        sig = sorted(g["sigma"].unique())
        row["sigmas"] = f"{min(sig):.3g}-{max(sig):.3g} ({len(sig)})"
        row["seeds"] = g["seed"].nunique()
        row["greedy_frac"] = g["greedy"].mean()
        p, lg = [], []
        row["rotations"] = int(g["rotation"].sum())
        for _, gs in g.groupby("seed"):
            gs = gs.dropna(subset=["dt"])
            gs = gs[(gs["dt"] > 0) & (gs["greedy"] | (not only_greedy))]
            if len(gs) < 3:
                continue
            x = np.log(1 / gs["sigma"].values)
            p.append(slope_se(x, np.log(gs["dt"].values))[0])
            lg.append(slope_se(x, gs["dt"].values)[0])
        row["p_fit"], row["p_se"] = mean_se(p)
        row["p_pred"] = row["k_pred"] - 2 if row["k_pred"] >= 3 else 0
        row["logslope"], row["logslope_se"] = mean_se(lg)
        row["logslope_pred"] = (1 - s2 / s1) / lam2 if row["k_pred"] == 2 else np.nan
        row["logslope_naive"] = 1 / lam2 if row["k_pred"] == 2 else np.nan
        small = g[g["sigma"] == min(sig)]
        row["dt_smallest_sigma"] = small["dt"].median()
        row["rho_smallest_sigma"] = small["rho"].median()
        if "hess_min" in g:
            row["hess_min_smallest"], row["hess_min_smallest_se"] = mean_se(small["hess_min"])
            row["hess_neg_smallest"] = small["hess_neg"].median()
            row["lambda2_pred"] = -lam2 if row["k_pred"] == 2 else np.nan
            hp = []
            for _, gs in g.groupby("seed"):
                gs = gs.dropna(subset=["hess_min"])
                gs = gs[gs["hess_min"] < 0]
                if len(gs) >= 3:
                    hp.append(slope_se(np.log(gs["sigma"]), np.log(-gs["hess_min"]))[0])
            row["hess_exp"], row["hess_exp_se"] = mean_se(hp)
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = load(sys.argv[1])
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    s = summarize(df)
    print(s.drop(columns=["d", "chi"]).to_string(index=False))
