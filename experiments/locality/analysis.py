"""Plateau table and fits for runs written by the locality runner (one JSON line per run).

Every run has learning events (final upward crossing of a_r = 1/2, in time order); consecutive events
bound a plateau. For each plateau we attach the predictions of candidate rules for the escape degree
(literal H2, the bond-rank rule, Furman's N) and fit, per seed, the exponent of the plateau length vs
sigma; then report mean +- SE (and median) over seeds.

Analysis standards (notes/briefs/README.md): runs are classified before pooling (greedy / other order /
rotation / stalled / timeout; rotation runs are excluded from fits), fits are per seed, log and power laws
are compared in the same space, and records can be restricted to the sigma grids and seeds of a committed
config (`config=`), so every table is regenerable from the configs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rules import furman_N, get_tree, naive_k, rank_rule  # noqa: E402


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


def config_keys(config_paths):
    """Run keys of every (case, seed, sigma) of the given YAML configs (same keys as run.run_key)."""
    import yaml

    keys = set()
    for path in config_paths:
        for c in yaml.safe_load(open(path))["cases"]:
            for seed in c["seeds"]:
                for s in c["sigmas"]:
                    keys.add(json.dumps([c.get("name"), c.get("S"), c["kind"], c.get("n"),
                                         c.get("start", "small"), c.get("d", 2), c.get("chi", 2),
                                         c.get("weights", [1.0, 0.5]), c.get("teacher_seed", 1000),
                                         seed, float(s)]))
    return keys


def load_records(paths, config=None):
    """Records from .jsonl files or from runner prefixes (<prefix>.part<pid> files). A run retried after
    an error / timeout appears more than once: its last record is kept. With `config` (list of YAML
    paths) only runs of those configs are kept (provenance: tables regenerable from committed configs)."""
    import glob
    recs = []
    for path in paths:
        files = [path] if Path(path).exists() else sorted(glob.glob(path + ".part*"))
        for f in files:
            recs += [json.loads(line) for line in open(f, encoding="utf-8")]
    last = {}
    for i, r in enumerate(recs):
        last[r.get("_key", i)] = r
    recs = list(last.values())
    if config:
        keys = config_keys(config)
        recs = [r for r in recs if r.get("_key") in keys]
    return recs


def final_rise_width(cross, r):
    """Final-rise width of component r: t(a = 0.9) - t(a = 0.1) around its final 1/2 crossing."""
    c05 = cross.get(f"a{r}_0.5", [])
    if len(c05) % 2 == 0:
        return np.nan
    t_half = c05[-1]
    lo = [t for t in cross.get(f"a{r}_0.1", []) if t <= t_half]
    hi = [t for t in cross.get(f"a{r}_0.9", []) if t >= t_half]
    return (hi[0] - lo[-1]) if lo and hi else np.nan


def load_hessians(path):
    """Recomputed plateau Hessians (hess_recompute.py): {(run key, plateau j): row}."""
    if not path or not Path(path).exists():
        return {}
    h = pd.read_csv(path)
    return {(k, int(j)): r for k, j, r in zip(h["_key"], h["j"], h.to_dict("records"))}


def plateau_table(paths, config=None, hessians=None) -> pd.DataFrame:
    hess = load_hessians(hessians)
    rows = []
    for rec in load_records(paths, config):
        if rec.get("status") in ("timeout", "error", "stalled") or "events" not in rec:
            continue
        if rec.get("patterns") is None:
            continue
        tree_name, case = case_label(rec)
        ev = rec["events"]
        cross = rec["crossings"]
        order = ">".join(str(r) for _, r in ev)
        n05 = {r: len(cross[f"a{r}_0.5"]) for r in range(len(rec["weights"]))}
        first = {r: (cross[f"a{r}_0.5"] or [np.inf])[0] for r in n05}
        base = dict(tree=tree_name, case=case, kind=rec["kind"], start=rec.get("start", "small"),
                    d=rec["d"], chi=rec["chi"], w=tuple(rec["weights"]), sigma=rec["sigma"],
                    seed=rec["seed"], K=rec["K"], order=order, key=rec.get("_key"),
                    rotation=any(c > 1 for c in n05.values()), Qdrift=rec.get("Qdrift"),
                    Qinit=rec.get("Qinit"), n_events=len(ev), n_comp=len(rec["weights"]),
                    complete=len(ev) == len(rec["weights"]))
        if len(rec["weights"]) == 2:
            base["greedy"] = first[0] <= first[1]
        learned = [ev[0][1]] if ev else []
        for j, ((ta, ra), (tb, rb)) in enumerate(zip(ev[:-1], ev[1:])):
            p = rec["plateaus"][j]
            pred = predictions(rec, learned, rb)
            row = dict(base, j=j, after=ra, before=rb, learned="+".join(map(str, sorted(learned))),
                       t_a=ta, t_b=tb, gap=tb - ta, **{k: v for k, v in pred.items() if k != "rising"})
            # separation ratio with final-rise widths (notes/diagnostics.md)
            wa = final_rise_width(cross, ra) if ta > 0 else 0.0
            row["rho"] = (tb - ta) / np.nanmax([wa, final_rise_width(cross, rb)])
            for key in ("t_mid", "hess_min", "hess_max", "hess_neg"):
                row[key] = p.get(key, np.nan)
            h = hess.get((rec.get("_key"), j))
            if h is not None:  # recomputed: raw Hessian and well-conditioned gauge projection
                for key in ("hess_raw_min", "hess_wc_min", "hess_wc_neg", "hess_raw_neg", "hess_raw_max"):
                    row[key] = h.get(key, np.nan)
            row["a_mid"] = p["a_mid"][rb] if "a_mid" in p else np.nan
            row["L_mid"] = p.get("L_mid", np.nan)
            row["remaining"] = tb - p["t_mid"] if "t_mid" in p else np.nan
            rows.append(row)
            learned.append(rb)
    return pd.DataFrame(rows)


def law_rms(x, gap, law):
    """RMS residual of log(gap) for the best fit of a log law (gap = a + b x) or a power law
    (log gap = c + p x), x = ln(1/sigma). The log law is fitted by least squares on log residuals."""
    y = np.log(gap)
    if law == "power":
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(np.sqrt(np.mean((y - A @ coef) ** 2)))
    from scipy.optimize import least_squares

    b0 = max((gap[-1] - gap[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 1.0, 1e-9)
    a0 = gap.min() - b0 * x.min() * 0.5

    def res(c):
        m = c[0] + c[1] * x
        return np.log(np.maximum(m, 1e-300)) - y

    fit = least_squares(res, [a0, b0], bounds=([-np.inf, 0], [np.inf, np.inf]))
    return float(np.sqrt(np.mean(fit.fun ** 2)))


def classify(rec):
    st = rec.get("status")
    if st in ("stalled", "timeout", "error"):
        return st
    ev = rec.get("events", [])
    n05 = [len(c) for k, c in rec.get("crossings", {}).items() if k.endswith("_0.5")]
    if len(ev) < len(rec["weights"]):
        return "incomplete"
    if any(c > 1 for c in n05):
        return "rotation"
    if [r for _, r in ev] == sorted(range(len(rec["weights"])), key=lambda r: -rec["weights"][r]):
        return "greedy"
    return "other_order"


CLASSES = ("greedy", "other_order", "rotation", "stalled", "timeout", "incomplete", "error")


def run_classes(paths, config=None) -> pd.DataFrame:
    """Per (tree, case, start, d, chi, w): number of runs in each class (analysis standard 3). Cases in
    which no run escapes are kept."""
    rows = []
    for rec in load_records(paths, config):
        if rec.get("patterns") is None and rec.get("S") is None:
            continue
        tree, case = case_label(rec)
        rows.append(dict(tree=tree, case=case, start=rec.get("start", "small"), d=rec.get("d", 2),
                         chi=rec.get("chi", 2), w=tuple(rec.get("weights", [1.0, 0.5])), cls=classify(rec)))
    df = pd.DataFrame(rows)
    out = df.groupby(["tree", "case", "start", "d", "chi", "w", "cls"]).size().unstack(fill_value=0)
    for c in CLASSES:
        if c not in out:
            out[c] = 0
    return out[list(CLASSES)]


def stall_dropout(paths, config=None) -> pd.DataFrame:
    """Sigma-dependent dropout: per case, seeds with stalls at every sigma vs at some sigmas only."""
    rows = []
    recs = load_records(paths, config)
    by = {}
    for r in recs:
        if r.get("patterns") is None:
            continue
        tree, case = case_label(r)
        by.setdefault((tree, case, r.get("start", "small"), r.get("chi", 2)), []).append(r)
    for key, rs in by.items():
        seeds = {}
        for r in rs:
            seeds.setdefault(r["seed"], []).append(r.get("status") == "stalled")
        n_all = sum(all(v) for v in seeds.values())
        n_some = sum(any(v) and not all(v) for v in seeds.values())
        n_runs = sum(len(v) for v in seeds.values())
        n_st = sum(sum(v) for v in seeds.values())
        if n_st:
            rows.append(dict(tree=key[0], case=key[1], start=key[2], chi=key[3], runs=n_runs, stalled=n_st,
                             seeds=len(seeds), seeds_stall_all_sigma=n_all, seeds_stall_some_sigma=n_some))
    return pd.DataFrame(rows)


def lambda2(w, K, r_new=1, r_learned=0):
    """Hessian eigenvalue of the k = 2 route at the balanced two-component saddle."""
    return w[r_new] * w[r_learned] ** ((K - 2) / K)


def summarize(df: pd.DataFrame, group=("tree", "case", "start", "d", "chi", "w", "order", "j"),
              exclude_rotation=True):
    if exclude_rotation and "rotation" in df:
        df = df[~df["rotation"]]
    out = []
    for key, g in df.groupby(list(group), dropna=False):
        row = dict(zip(group, key))
        row.update(K=int(g["K"].iloc[0]), before=int(g["before"].iloc[0]), learned=g["learned"].iloc[0])
        for col in ("k_rank", "k_naive", "N_furman"):
            vals = g[col].unique()
            row[col] = int(vals[0]) if len(vals) == 1 else -1
        k = row["k_rank"]
        sig = sorted(g["sigma"].unique())
        row["sigma_range"] = f"{min(sig):.3g}-{max(sig):.3g}"
        row["n_sigma"] = len(sig)
        row["decades"] = float(np.log10(max(sig) / min(sig)))
        row["seeds"] = g["seed"].nunique()
        row["runs"] = len(g)
        hcol = "hess_wc_min" if "hess_wc_min" in g and g["hess_wc_min"].notna().any() else "hess_min"
        p, lg, hp, hp_raw, ploc, rl, rp, dslope = [], [], [], [], [], [], [], []
        for _, gs in g.groupby("seed"):
            gs = gs[gs["gap"] > 0].sort_values("sigma")
            x = np.log(1 / gs["sigma"].values)
            if gs["sigma"].nunique() >= 3:
                p.append(slope_se(x, np.log(gs["gap"].values))[0])
                lg.append(slope_se(x, gs["gap"].values)[0])
                ploc.append(slope_se(x[:3], np.log(gs["gap"].values[:3]))[0])
            if gs["sigma"].nunique() >= 4:  # competing laws, same space and weights (standard 5)
                rl.append(law_rms(x, gs["gap"].values, "log"))
                rp.append(law_rms(x, gs["gap"].values, "power"))
            for col, store in ((hcol, hp), ("hess_raw_min", hp_raw)):
                if col in gs:
                    gh = gs[gs[col] < 0]
                    if gh["sigma"].nunique() >= 3:
                        store.append(slope_se(np.log(gh["sigma"]), np.log(-gh[col]))[0])
            # dormant-amplitude test, per seed: remaining time vs overlap at the plateau midpoint
            gd = gs[(gs["a_mid"] > 0) & (gs["remaining"] > 0)]
            if len(gd) >= 3:
                if k >= 3:
                    dslope.append(slope_se(np.log(gd["a_mid"]), np.log(gd["remaining"]))[0])
                elif k == 2:
                    dslope.append(slope_se(np.log(1 / gd["a_mid"]), gd["remaining"])[0])
        row["fit_seeds"] = len(p)
        row["p_fit"], row["p_se"] = mean_se(p)
        row["p_median"] = float(np.median(p)) if p else np.nan
        row["p_pred"] = max(k - 2, 0)
        row["p_local"], row["p_local_se"] = mean_se(ploc)
        row["logslope"], row["logslope_se"] = mean_se(lg)
        row["rms_log_law"] = float(np.median(rl)) if rl else np.nan
        row["rms_power_law"] = float(np.median(rp)) if rp else np.nan
        row["law"] = ("log" if row["rms_log_law"] < row["rms_power_law"] else "power") if rl else "n/a"
        w = row["w"]
        if k == 2 and len(w) == 2:
            lam = lambda2(w, row["K"], row["before"], 1 - row["before"])
            row["lam2_pred"] = lam
            row["logslope_pred"] = ((1 - w[row["before"]] / w[1 - row["before"]]) / lam
                                    if row["start"] != "saddle" else 1 / lam)
            row["logslope_naive"] = 1 / lam
        small = g[g["sigma"] == min(sig)]
        row["gap_small_med"] = small["gap"].median()
        row["rho_small_med"] = small["rho"].median()
        row["hess_col"] = hcol
        row["hess_min_small"], row["hess_min_small_se"] = mean_se(small[hcol])
        if "hess_raw_min" in small:
            row["hess_raw_min_small"], _ = mean_se(small["hess_raw_min"])
        row["hess_neg_small"] = small["hess_wc_neg"].median() if "hess_wc_neg" in small else small["hess_neg"].median()
        row["hess_exp"], row["hess_exp_se"] = mean_se(hp)
        row["hess_raw_exp"], row["hess_raw_exp_se"] = mean_se(hp_raw)
        row["dormant_slope"], row["dormant_slope_se"] = mean_se(dslope)
        if k >= 3:
            row["dormant_pred"] = -(k - 2) / k
        elif k == 2 and len(w) == 2:
            row["dormant_pred"] = 1 / (2 * lambda2(w, row["K"], row["before"], 1 - row["before"]))
        row["Qdrift_max"] = g["Qdrift"].max()
        row["Qdrift_rel_max"] = float((g["Qdrift"] / g["Qinit"]).max()) if "Qinit" in g else np.nan
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
