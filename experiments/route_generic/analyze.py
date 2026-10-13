"""Tables of the route_generic experiment (definitions and decision rules: notes/route_generic.md Part I).

usage: uv run python experiments/route_generic/analyze.py <synthetic_dir> <mnist_dir> <results_dir> [workers]

Writes runs.csv, events.csv, rises.csv (one row per run / event / singular-value rise), exponents.csv
(one row per series = target, tree, teacher seed, init seed, event signature) and summary.csv (one row per
target x tree x event class) into <results_dir>. Either input directory may be "-" (skipped).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import math
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import get_tree  # noqa: E402
from events import amplitude_events, load, run_events  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
WINDOW_POINTS, MIN_POINTS, MIN_SPAN = 7, 5, 1.5
TRAP_REL = 1e-10


def _pattern_info(case):
    from ttn.routes import RouteTarget

    return RouteTarget(get_tree(case["tree"]), case["patterns"], case["weights"])


def synth_one(path: str):
    import torch

    torch.set_num_threads(1)
    run = load(path)
    m = run["meta"]
    case = m["case"]
    tree = get_tree(case["tree"])
    ident = dict(target=case["target"], tree=case["tree"], d=case["d"], chi=max(m["chi"]), sigma=case["sigma"],
                 seed=case["seed"], teacher_seed=case.get("teacher_seed", 0))
    events, rs = run_events(run, tree)
    L = run["loss"]
    learned = [r for r in rs if r["learned"]]
    dead = [r for r in rs if not r["learned"] and r["final"] < TRAP_REL]
    run_row = dict(ident, status=m["status"], trapped=bool(m["status"] != "converged" and dead),
                   n_steps=m["n_steps"], restarts=m["restarts"], solver_fails=len(m["fails"]),
                   wall_s=round(m["wall"], 1), t_a=m["t_a"], final_loss_rel=float(L[-1] / m["L0"]),
                   n_rises_target=len(rs), n_rises_learned=len(learned), n_events=len(events),
                   n_floor_flagged_events=int(sum(e["floor_flag"] for e in events)),
                   n_relearn=int(sum(e["relearn"] for e in events)),
                   nonstiff_at_end=bool(run["method"][-1] == 1))
    ev_rows = []
    amp_info = None
    if case["family"] in ("pattern", "cp"):
        from common import build_target

        spec, teacher, _ = build_target(case)
        tgt = (teacher.gram() @ teacher.weights).numpy()
        amp_info = amplitude_events(run, tgt)
    rt = _pattern_info(case) if case["family"] == "pattern" else None
    for ev in events:
        row = dict(ident, **ev, status=m["status"])
        if amp_info is not None:
            done = [a for a in amp_info if a["learned"] and np.isfinite(a["t_half"])]
            if ev["event"] == 0:
                first = min(done, key=lambda a: a["t_half"]) if done else None
                match = first
            else:
                match = min(done, key=lambda a: abs(a["t_half"] - ev["t_event"])) if done else None
            if match is not None:
                row["component"] = match["component"]
                row["amp_dt"] = match["t_half"] - (ev["t_event"] if ev["event"] else 0.0) if ev["event"] else math.nan
                row["eff_depth_amp"] = match.get("eff_depth", math.nan)
                row["forgotten"] = match["forgotten"]
                if rt is not None:
                    J = [a["component"] for a in done if a["t_half"] < match["t_half"]]
                    r = match["component"]
                    row["J_before"] = " ".join(map(str, sorted(J)))
                    row["c_pred"] = rt.depth(J, r) if J else tree.n_nodes
                    region = rt.region(J, r) if J else frozenset(range(tree.n_nodes))
                    pred_bonds = sorted(e for e in tree.edges if e in region and tree.parent[e] in region)
                    row["pred_bonds"] = " ".join(map(str, pred_bonds))
                    row["bonds_match"] = row["pred_bonds"] == ev["bonds"]
                    if J:
                        ranked = rt.greedy_next(J)
                        row["greedy_next"] = ranked[0]["component"]
        ev_rows.append(row)
    rise_rows = [dict(ident, **{k: r.get(k) for k in ("edge", "k", "target", "learned", "n_up", "final", "t_on",
                                                     "t_half", "event", "depth", "depth_points", "depth_wide",
                                                     "floor_min")}) for r in rs]
    return run_row, ev_rows, rise_rows


def mnist_one(path: str):
    from mnist_events import load as mload, mnist_run_events

    run = mload(path)
    m = run["meta"]
    case = m["case"]
    events, rs, tree = mnist_run_events(run)
    tree_name = f"{case['tree']}16_{case['ordering']}"
    ident = dict(target="mnist", tree=tree_name, d=2, chi=case["chi"], sigma=case["sigma"], seed=case["seed"],
                 teacher_seed=0)
    p2 = run["p2"]
    run_row = dict(ident, status=m.get("stop", "?"), trapped=False, n_steps=m.get("phase2", {}).get("n_steps"),
                   restarts=0, solver_fails=0, wall_s=m.get("wall_total"), t_a=m["t_a"],
                   final_loss_rel=float(p2["loss"][-1] / m["loss0"]), n_rises_target=math.nan,
                   n_rises_learned=len(rs), n_events=len(events), n_floor_flagged_events=0,
                   n_relearn=int(sum(e["relearn"] for e in events)), nonstiff_at_end=False,
                   tau_end=float(p2["t"][-1]))
    ev_rows = [dict(ident, **ev, status=run_row["status"]) for ev in events]
    rise_rows = [dict(ident, **{k: r.get(k) for k in ("edge", "k", "n_up", "final", "t_on", "t_half", "event",
                                                     "depth", "depth_wide", "floor_min")}) for r in rs]
    return run_row, ev_rows, rise_rows


def _safe(fn, path):
    try:
        return fn(path)
    except Exception as exc:  # noqa: BLE001
        print("FAILED", path, repr(exc), flush=True)
        return None


def _synth(path):
    return _safe(synth_one, path)


def _mnist(path):
    return _safe(mnist_one, path)


# ---- fits ----------------------------------------------------------------------------------


def fit_series(pts: pd.DataFrame) -> dict:
    """Per-series fits in the pre-registered window (smallest 7 usable sigma, >= 5 points, >= 1.5 dec)."""
    use = pts[(~pts["floor_flag"].astype(bool)) & (pts["duration"] > 0) & np.isfinite(pts["duration"])]
    use = use.sort_values("sigma").head(WINDOW_POINTS)
    out = dict(n_points=len(use), n_flagged=int(pts["floor_flag"].astype(bool).sum()),
               sigma_min=float(use["sigma"].min()) if len(use) else math.nan,
               sigma_max=float(use["sigma"].max()) if len(use) else math.nan)
    out["span_dec"] = math.log10(out["sigma_max"] / out["sigma_min"]) if len(use) else 0.0
    d_eff = use["eff_depth"].dropna()
    out["eff_depth"] = float(d_eff.median()) if len(d_eff) else math.nan
    out["n_eff_depth"] = int(len(d_eff))
    d_w = use["eff_depth_wide"].dropna()
    out["eff_depth_wide"] = float(d_w.median()) if len(d_w) else math.nan
    if "eff_depth_amp" in use:
        d_a = use["eff_depth_amp"].dropna()
        out["eff_depth_amp"] = float(d_a.median()) if len(d_a) else math.nan
    ok = len(use) >= MIN_POINTS and out["span_dec"] >= MIN_SPAN - 1e-9
    out["fitted"] = bool(ok)
    if not ok:
        out["reason"] = "too few points" if len(use) < MIN_POINTS else "span < 1.5 decades"
        return out
    x = np.log(use["sigma"].to_numpy())
    y = np.log(use["duration"].to_numpy())
    A = np.vstack([np.ones_like(x), x]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    cov = (resid @ resid / dof) * np.linalg.inv(A.T @ A)
    out["p"] = float(-coef[1])
    out["p_se"] = float(math.sqrt(cov[1, 1]))
    out["rms_pow"] = float(np.sqrt(np.mean(resid**2)))
    # log law with relative residuals: minimize sum ((A l + B - g) / g)^2, l = ln(1/sigma)
    g = use["duration"].to_numpy()
    lx = -x
    M = np.vstack([lx / g, 1 / g]).T
    ab, *_ = np.linalg.lstsq(M, np.ones_like(g), rcond=None)
    fit = ab[0] * lx + ab[1]
    out["log_A"], out["log_B"] = float(ab[0]), float(ab[1])
    out["rms_log"] = float(np.sqrt(np.mean(np.log(np.maximum(fit, 1e-300) / g) ** 2))) if (fit > 0).all() else math.inf
    out["log_preferred"] = bool(out["rms_log"] < out["rms_pow"])
    out["reason"] = ""
    return out


def exponents(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "tree", "d", "chi", "teacher_seed", "seed", "prior", "rises"]
    for key, g in events.groupby(keys, dropna=False, sort=False):
        first = g.sort_values("sigma").iloc[0]
        row = dict(zip(keys, key))
        row.update(event=int(g["event"].median()), event_min=int(g["event"].min()), event_max=int(g["event"].max()),
                   bonds=first["bonds"], n_bonds=int(first["n_bonds"]), k_region=int(first["k_region"]),
                   n_components=int(first["n_components"]), k_max_component=int(first["k_max_component"]),
                   n_sigma_present=int(g["sigma"].nunique()))
        for c in ("component", "c_pred", "J_before", "pred_bonds", "bonds_match"):
            if c in g and g[c].notna().any():
                row[c] = g[c].dropna().iloc[0]
        row.update(fit_series(g))
        row["p_pred"] = row["k_region"] - 2
        row["dp"] = row.get("p", math.nan) - row["p_pred"]
        row["dd"] = row["eff_depth"] - row["k_region"]
        row["dp_vs_depth"] = row.get("p", math.nan) - (row["eff_depth"] - 2)
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(dd: pd.Series, dp: pd.Series, n: int) -> str:
    if n < 10:
        return "INCONCLUSIVE"
    md, mp = dd.abs().median(), dp.abs().median()
    fd, fp = (dd.abs() <= 0.3).mean(), (dp.abs() <= 0.3).mean()
    if md <= 0.3 and mp <= 0.3 and fd >= 0.7 and fp >= 0.7:
        return "HOLDS"
    if md > 0.5 or mp > 0.5:
        return "FAILS"
    return "PARTIAL"


def summarize(ex: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fitted = ex[ex["fitted"]].copy()
    fitted["cls"] = np.where(fitted["event"] == 0, "event0", "later")
    groups = []
    for (t, tr, cls), g in fitted.groupby(["target", "tree", "cls"]):
        groups.append(((t, tr, cls, "all"), g))
        if cls == "later":
            for k, gk in g.groupby("k_region"):
                groups.append(((t, tr, cls, f"k={k}"), gk))
    for (t, tr, cls, sub), g in groups:
        both = g.dropna(subset=["dd", "dp"])
        row = dict(target=t, tree=tr, events=cls, k_class=sub, n_series=len(g), n_series_both=len(both))
        for name, col in (("dd", "dd"), ("dp", "dp"), ("p", "p"), ("eff_depth", "eff_depth"), ("k_region", "k_region")):
            v = g[col].dropna()
            row[f"{name}_median"] = float(v.median()) if len(v) else math.nan
            row[f"{name}_q25"] = float(v.quantile(0.25)) if len(v) else math.nan
            row[f"{name}_q75"] = float(v.quantile(0.75)) if len(v) else math.nan
        row["frac_dd_within_0.3"] = float((g["dd"].dropna().abs() <= 0.3).mean()) if g["dd"].notna().any() else math.nan
        row["frac_dp_within_0.3"] = float((g["dp"].dropna().abs() <= 0.3).mean()) if g["dp"].notna().any() else math.nan
        row["frac_p_vs_depth_within_0.3"] = (float((g["dp_vs_depth"].dropna().abs() <= 0.3).mean())
                                             if g["dp_vs_depth"].notna().any() else math.nan)
        row["frac_log_preferred"] = float(g["log_preferred"].mean())
        row["frac_disconnected"] = float((g["n_components"] > 1).mean())
        row["verdict"] = verdict(both["dd"], both["dp"], len(both)) if sub == "all" or cls == "later" else ""
        rows.append(row)
    return pd.DataFrame(rows)


def collect(paths, fn, workers):
    runs, evs, rises = [], [], []
    with Pool(workers) as pool:
        for res in pool.imap_unordered(fn, paths, chunksize=4):
            if res is None:
                continue
            runs.append(res[0])
            evs.extend(res[1])
            rises.extend(res[2])
    return runs, evs, rises


def main():
    synth_dir, mnist_dir, out_dir = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, evs, rises = [], [], []
    if synth_dir != "-":
        paths = sorted(str(p) for p in Path(synth_dir).glob("*.npz"))
        r, e, s = collect(paths, _synth, workers)
        runs += r
        evs += e
        rises += s
    if mnist_dir != "-":
        paths = sorted(str(p) for p in Path(mnist_dir).glob("*.npz"))
        r, e, s = collect(paths, _mnist, workers)
        runs += r
        evs += e
        rises += s
    order = ["target", "tree", "teacher_seed", "seed", "sigma"]
    runs_df = pd.DataFrame(runs).sort_values(order)
    ev_df = pd.DataFrame(evs).sort_values(order + ["event"])
    rise_df = pd.DataFrame(rises).sort_values(order + ["edge", "k"])
    runs_df.to_csv(out / "runs.csv", index=False, float_format="%.6g")
    ev_df.to_csv(out / "events.csv", index=False, float_format="%.6g")
    rise_df.to_csv(out / "rises.csv", index=False, float_format="%.6g")
    ex = exponents(ev_df)
    ex.to_csv(out / "exponents.csv", index=False, float_format="%.5g")
    sm = summarize(ex)
    sm.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(runs_df.groupby(["target", "tree", "status"]).size())
    cols = ["target", "tree", "events", "k_class", "n_series", "dd_median", "dp_median", "p_median",
            "eff_depth_median", "frac_dd_within_0.3", "frac_dp_within_0.3", "frac_log_preferred", "verdict"]
    if len(sm):
        print(sm[cols].to_string(index=False))


if __name__ == "__main__":
    main()
