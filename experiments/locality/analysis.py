"""Plateau table and fits for runs written by the locality runner (one JSON line per run).

Every run has learning events (final upward crossing of a_r = 1/2, in time order); consecutive events
bound a plateau. For each plateau we attach the predictions of three candidate rules for the escape
degree (literal H2, the bond-rank rule, Furman's N) and fit, per seed, the exponent of the plateau
length vs sigma; then report mean +- SE over seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rules import furman_N, get_tree, naive_k, rank_rule, steiner_k  # noqa: E402


def slope_se(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    if len(x) <= 2:
        return coef[0], np.nan
    s2 = np.sum((y - A @ coef) ** 2) / (len(x) - 2)
    return coef[0], np.sqrt(s2 / np.sum((x - x.mean()) ** 2))


def mean_se(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)


def case_label(rec):
    tree = get_tree(rec["kind"], rec.get("n"))
    name = rec.get("name") or "S=" + "".join(map(str, rec["S"]))
    return f"{tree.name}", name


_pred_cache: dict = {}


def predictions(rec, learned, r):
    key = (rec["kind"], rec.get("n"), json.dumps(rec["patterns"]), json.dumps(rec["weights"]),
           tuple(sorted(learned)), r)
    if key not in _pred_cache:
        tree = get_tree(rec["kind"], rec.get("n"))
        k, nodes, rising = rank_rule(tree, rec["patterns"], rec["weights"], sorted(learned), r)
        _pred_cache[key] = dict(
            k_rank=k,
            k_naive=naive_k(tree, rec["patterns"], sorted(learned), r),
            N_furman=furman_N(tree, rec["patterns"], rec["weights"], sorted(learned), rising),
            rising=rising)
    return _pred_cache[key]


def load_records(paths):
    """Records from .jsonl files or from runner prefixes (<prefix>.part<pid> files)."""
    import glob
    recs = []
    for path in paths:
        files = [path] if Path(path).exists() else sorted(glob.glob(path + ".part*"))
        for f in files:
            recs += [json.loads(line) for line in open(f)]
    return recs


def plateau_table(paths) -> pd.DataFrame:
    rows = []
    for rec in load_records(paths):
        if True:
            if rec.get("status") == "timeout" or "events" not in rec:
                continue
            if "patterns" not in rec or rec["patterns"] is None:
                continue
            tree_name, case = case_label(rec)
            ev = rec["events"]
            order = ">".join(str(r) for _, r in ev)
            n05 = {r: len(rec["crossings"][f"a{r}_0.5"]) for r in range(len(rec["weights"]))}
            first = {r: (rec["crossings"][f"a{r}_0.5"] or [np.inf])[0] for r in n05}
            base = dict(tree=tree_name, case=case, kind=rec["kind"], start=rec.get("start", "small"),
                        d=rec["d"], chi=rec["chi"], w=tuple(rec["weights"]), sigma=rec["sigma"],
                        seed=rec["seed"], K=rec["K"], order=order,
                        rotation=any(c > 1 for c in n05.values()), Qdrift=rec.get("Qdrift"),
                        n_events=len(ev), n_comp=len(rec["weights"]),
                        complete=len(ev) == len(rec["weights"]))
            if len(rec["weights"]) == 2:
                base["greedy"] = first[0] <= first[1]
            learned = [ev[0][1]] if ev else []
            for j, ((ta, ra), (tb, rb)) in enumerate(zip(ev[:-1], ev[1:])):
                p = rec["plateaus"][j]
                pred = predictions(rec, learned, rb)
                row = dict(base, j=j, after=ra, before=rb, learned="+".join(map(str, sorted(learned))),
                           t_a=ta, t_b=tb, gap=tb - ta, **{k: v for k, v in pred.items() if k != "rising"})
                for key in ("t_mid", "hess_min", "hess_max", "hess_neg"):
                    row[key] = p.get(key, np.nan)
                row["a_mid"] = p["a_mid"][rb] if "a_mid" in p else np.nan
                row["L_mid"] = p.get("L_mid", np.nan)
                # float64 floor: the dormant overlap of the escaping component fell below 1e-14
                row["dormant_min"] = rec.get("dormant_min", {}).get(str(rb), np.nan)
                row["floor_flag"] = bool(row["dormant_min"] < 1e-14)
                row["remaining"] = tb - p["t_mid"] if "t_mid" in p else np.nan
                rows.append(row)
                learned.append(rb)
    return pd.DataFrame(rows)


def lambda2(w, K, r_new=1, r_learned=0):
    """Hessian eigenvalue of the k = 2 route at the balanced two-component saddle."""
    return w[r_new] * w[r_learned] ** ((K - 2) / K)


def summarize(df: pd.DataFrame, group=("tree", "case", "start", "d", "chi", "w", "order", "j")):
    out = []
    for key, g in df.groupby(list(group), dropna=False):
        row = dict(zip(group, key))
        row.update(K=int(g["K"].iloc[0]), before=int(g["before"].iloc[0]),
                   learned=g["learned"].iloc[0])
        for col in ("k_rank", "k_naive", "N_furman"):
            vals = g[col].unique()
            row[col] = int(vals[0]) if len(vals) == 1 else -1
        k = row["k_rank"]
        sig = sorted(g["sigma"].unique())
        row["sigma_range"] = f"{min(sig):.3g}-{max(sig):.3g}"
        row["n_sigma"] = len(sig)
        row["seeds"] = g["seed"].nunique()
        row["runs"] = len(g)
        p, lg, hp = [], [], []
        for _, gs in g.groupby("seed"):
            gs = gs[gs["gap"] > 0]
            if gs["sigma"].nunique() >= 3:
                x = np.log(1 / gs["sigma"].values)
                p.append(slope_se(x, np.log(gs["gap"].values))[0])
                lg.append(slope_se(x, gs["gap"].values)[0])
            gh = gs[gs["hess_min"] < 0]
            if gh["sigma"].nunique() >= 3:
                hp.append(slope_se(np.log(gh["sigma"]), np.log(-gh["hess_min"]))[0])
        row["fit_seeds"] = len(p)
        row["p_fit"], row["p_se"] = mean_se(p)
        row["p_pred"] = max(k - 2, 0)
        row["logslope"], row["logslope_se"] = mean_se(lg)
        w = row["w"]
        if k == 2 and len(w) == 2:
            lam = lambda2(w, row["K"], row["before"], 1 - row["before"])
            row["lam2_pred"] = lam
            row["logslope_pred"] = (1 - w[row["before"]] / w[1 - row["before"]]) / lam \
                if row["start"] == "small" else 1 / lam
            row["logslope_naive"] = 1 / lam
        small = g[g["sigma"] == min(sig)]
        row["gap_small_med"] = small["gap"].median()
        row["hess_min_small"], row["hess_min_small_se"] = mean_se(small["hess_min"])
        row["hess_neg_small"] = small["hess_neg"].median()
        row["hess_exp"], row["hess_exp_se"] = mean_se(hp)
        # dormant-norm test: remaining time vs overlap of the next component at the midpoint
        gd = g[(g["a_mid"] > 0) & (g["remaining"] > 0)]
        if len(gd) >= 5:
            if k >= 3:
                row["dormant_slope"], row["dormant_slope_se"] = slope_se(
                    np.log(gd["a_mid"]), np.log(gd["remaining"]))
                row["dormant_pred"] = -(k - 2) / k
            elif k == 2:
                row["dormant_slope"], row["dormant_slope_se"] = slope_se(
                    np.log(1 / gd["a_mid"]), gd["remaining"])
                if len(w) == 2:
                    row["dormant_pred"] = 1 / (2 * lambda2(w, row["K"], row["before"],
                                                           1 - row["before"]))
        row["Qdrift_max"] = g["Qdrift"].max()
        row["floor_flags"] = int(g["floor_flag"].sum()) if "floor_flag" in g else 0
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = plateau_table(sys.argv[1:])
    s = summarize(df)
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.precision", 3,
                  "display.max_rows", 500)
    cols = ["tree", "case", "start", "order", "j", "k_naive", "k_rank", "N_furman", "sigma_range",
            "seeds", "fit_seeds", "p_pred", "p_fit", "p_se", "logslope", "logslope_pred",
            "hess_min_small", "hess_neg_small", "hess_exp", "dormant_slope", "dormant_pred"]
    print(s[[c for c in cols if c in s]].to_string(index=False))
