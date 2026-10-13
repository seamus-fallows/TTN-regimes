"""Raw per-run JSON -> tidy CSVs and the registered decisions (notes/route_smallinit.md sec. 6).

usage: uv run python experiments/route_smallinit/analyze.py [--raw DIR] [--out DIR]
Writes runs.csv, events.csv, plateaus.csv, floor_check.csv, exponents.csv, cells.csv, c1.csv,
summary.json into results/.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import SpareManifold, build_target  # noqa: E402
from run import route_time  # noqa: E402

ID = ["target", "tree", "d", "chi", "sigma", "log10_sigma", "seed"]
WINDOW = (-5.0, -3.0)  # log10 sigma, registered fit window
SMALL = -3.0  # "sigma <= 1e-3" for the pooled decisions


def num(x):
    if isinstance(x, str):
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}.get(x, x)
    return x


def load(raw: Path):
    targets = {t["name"]: t for t in yaml.safe_load(open(HERE / "configs/targets.yaml"))["targets"]}
    chi = {n: max(SpareManifold(build_target(t), int(t["d"])).spec.bond_dims) for n, t in targets.items()}
    runs, events, plats = [], [], []
    for f in sorted(glob.glob(str(raw / "*" / "*" / "*.json"))):
        d = json.load(open(f))
        m = d["meta"]
        base = dict(target=m["target"], tree=m["tree"], d=m["d"], chi=chi[m["target"]], sigma=m["sigma"],
                    log10_sigma=m["log10_sigma"], seed=m["seed"], variant=m["variant"])
        runs.append(base | {k: num(v) for k, v in d["run"].items()} | dict(wall=m["wall"]))
        for e in d["events"]:
            events.append(base | {k: num(v) for k, v in e.items()})
        for p in d["plateaus"]:
            plats.append(base | {k: num(v) for k, v in p.items()})
    return targets, pd.DataFrame(runs), pd.DataFrame(events), pd.DataFrame(plats)


def floor_check(runs, events):
    """dense vs duplicate with other arithmetic ("reorder": other contraction order; "lib": library
    norm-form gradient): flag if order/status differ or any duration > 1 differs by more than 1e-3
    relative."""
    lib = runs[runs.variant.isin(["reorder", "lib"])]
    rows = []
    ev = events.set_index(["variant", "target", "log10_sigma", "seed"]).sort_index()
    for _, r in lib.iterrows():
        key = (r.target, r.log10_sigma, r.seed)
        dup = r.variant
        dn = runs[(runs.variant == "dense") & (runs.target == r.target) & (runs.log10_sigma == r.log10_sigma)
                  & (runs.seed == r.seed)]
        if dn.empty:
            continue
        dn = dn.iloc[0]
        try:
            d1 = ev.loc[("dense",) + key].sort_values("event").duration.to_numpy()
            d2 = ev.loc[(dup,) + key].sort_values("event").duration.to_numpy()
        except KeyError:
            d1 = d2 = np.array([])
        same = len(d1) == len(d2) and dn.order == r.order and dn.status == r.status
        rel = float(np.max(np.abs(d1 - d2) / np.maximum(np.abs(d1), 1e-300))) if same and len(d1) else math.nan
        # durations of unresolved plateaus (two events in one step, duration ~ 0) are not compared
        if same and len(d1):
            big = np.abs(d1) > 1.0
            rel = float(np.max(np.abs(d1 - d2)[big] / np.abs(d1[big]))) if big.any() else 0.0
        # a duplicate (or original) stopped by its time cap gives no verdict
        complete = r.status in ("done", "trapped") and dn.status in ("done", "trapped")
        flag = complete and ((not same) or (rel > 1e-3))
        rows.append(dict(target=r.target, tree=r.tree, d=r.d, chi=r.chi, sigma=r.sigma,
                         log10_sigma=r.log10_sigma, seed=r.seed, check=dup, order_dense=dn.order,
                         order_check=r.order, status_dense=dn.status, status_check=r.status,
                         max_rel_duration_diff=rel, complete=complete,
                         floor_flag=flag))
    return pd.DataFrame(rows)


def history(raw: Path, runs, targets):
    """Full s/2 crossing history of every dense/ref run from its saved trajectory, in step order:
    a string like "+10101 +00201 +00000 -10101 ..." (+ up, - down), and per learning event whether the
    interval since the previous learning event contains any other s/2 crossing or a forgetting
    (then the event's J_before and duration do not describe a single plateau: "contaminated")."""
    hist, contaminated, passes = {}, {}, []
    for r in runs.itertuples():
        if r.variant not in ("dense", "ref"):
            continue
        tag = f"ls{r.log10_sigma:+.2f}" + ("" if r.seed < 0 else f"_seed{int(r.seed):02d}")
        f = raw / r.variant / r.target / f"{tag}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        s = np.array(targets[r.target]["weights"], float)
        x = z["P"] / s
        pats = targets[r.target]["patterns"]
        cross = []  # (step, pattern, +1/-1, level)
        for lev in (0.5, 0.25):
            above = x >= lev
            for a in range(x.shape[1]):
                ch = np.nonzero(above[1:, a] != above[:-1, a])[0] + 1
                cross += [(int(i), a, 1 if above[i, a] else -1, lev) for i in ch]
        cross.sort()
        half = [c for c in cross if c[3] == 0.5]
        # passes: every upward s/2 crossing, timed by linear interpolation of the stored step times
        t = z["t"]
        ups = [c for c in half if c[2] > 0]
        prev_t = 0.0
        final_step = {}
        for i, a, d, _ in half:
            if d > 0:
                final_step[a] = i
        for n_pass, (i, a, _, _) in enumerate(ups):
            x0, x1 = x[i - 1, a], x[i, a]
            tc = t[i - 1] + (0.5 - x0) / (x1 - x0) * (t[i] - t[i - 1])
            J = [b for b in range(x.shape[1]) if b != a and x[i - 1, b] >= 0.5]
            dur = tc - prev_t
            res = np.spacing(max(tc, 1.0)) * 4 / dur if dur > 0 else np.inf  # float64 time resolution
            passes.append(dict(variant=r.variant, target=r.target, tree=r.tree, d=r.d, chi=r.chi, sigma=r.sigma,
                               log10_sigma=r.log10_sigma, seed=r.seed, n_pass=n_pass, pattern=a,
                               pattern_str=pats[a], J_before="{" + ",".join(map(str, J)) + "}",
                               J_before_str=" ".join(pats[b] for b in J), t_pass=tc, duration=dur,
                               time_resolution=res, final=(final_step.get(a) == i and x[-1, a] >= 0.5),
                               forgets_after=False))
            prev_t = tc
        hist[(r.variant, r.target, r.log10_sigma, r.seed)] = " ".join(
            ("+" if d > 0 else "-") + pats[a] for _, a, d, _ in half)
        # final upward crossing per pattern = learning events
        last_up = {}
        for i, a, d, lev in half:
            if d > 0:
                last_up[a] = i
        final = {a: i for a, i in last_up.items() if x[-1, a] >= 0.5}
        order = sorted(final.items(), key=lambda kv: kv[1])
        prev = 0
        for k, (a, i) in enumerate(order):
            other = [c for c in cross if prev < c[0] <= i and not (c[1] == a and c[0] == i and c[3] == 0.5)
                     and (c[3] == 0.5 or c[2] < 0)]
            contaminated[(r.variant, r.target, r.log10_sigma, r.seed, a)] = bool(other)
            prev = i
    return hist, contaminated, pd.DataFrame(passes)


def per_seed_fit(x, y):
    a, b = np.polyfit(x, y, 1)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/outputs/route-smallinit/runs")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    targets, runs, events, plats = load(Path(args.raw))
    pred = pd.read_csv(HERE / "results/predictions.csv", dtype={"J": str, "J_str": str, "missing_str": str})

    # ---- floor check and run-level flags ------------------------------------------------------
    fc = floor_check(runs, events)
    fc.to_csv(out / "floor_check.csv", index=False, float_format="%.8g")
    flagged = set(zip(fc[fc.floor_flag].target, fc[fc.floor_flag].log10_sigma, fc[fc.floor_flag].seed))
    runs["floor_flag_bound"] = runs["floor_flag"]
    runs["floor_flag"] = [(t, ls, s) in flagged for t, ls, s in zip(runs.target, runs.log10_sigma, runs.seed)]
    runs["floor_checked"] = [(t, ls, s) in set(zip(fc.target, fc.log10_sigma, fc.seed))
                             for t, ls, s in zip(runs.target, runs.log10_sigma, runs.seed)]

    # ---- events: rule agreement, canonical, one-step B --------------------------------------------
    ev = events.copy()
    ev["agree_greedy"] = ev.pattern == ev.pred_greedy
    ev["agree_pairwise"] = ev.pattern == ev.pred_pairwise
    ev["agree_weight"] = ev.pattern == ev.pred_weight
    ev["agree_onestep"] = np.where(ev.onestep_winner >= 0, ev.pattern == ev.onestep_winner, np.nan)
    ev["canonical"] = ev.rel_dist_mg < 0.01
    ev["unseparated"] = ev.sep_other_at_r90 > 0.1
    ev["floor_flag_bound"] = ev["floor_flag"].astype("boolean")
    ev["floor_flag"] = [(t, ls, s) in flagged for t, ls, s in zip(ev.target, ev.log10_sigma, ev.seed)]
    pats = {n: t["patterns"] for n, t in targets.items()}
    wts = {n: t["weights"] for n, t in targets.items()}
    ev["pattern_weight"] = [wts[t][int(p)] for t, p in zip(ev.target, ev.pattern)]
    ev["J_before_str"] = [" ".join(pats[t][int(a)] for a in j.strip("{}").split(",") if a != "")
                          for t, j in zip(ev.target, ev.J_before)]
    for rule in ("greedy", "pairwise", "weight"):
        ev[f"pred_{rule}_str"] = [pats[t][int(p)] for t, p in zip(ev.target, ev[f"pred_{rule}"])]
    oc = pred.dropna(subset=["one_component_order"]).drop_duplicates(["target", "J"])
    occ = {(t, j): o for t, j, o in zip(oc.target, oc.J, oc.one_component_order)}
    ev["one_component_order"] = [occ.get((t, j), np.nan) for t, j in zip(ev.target, ev.J_before)]
    st = runs.set_index(["variant", "target", "log10_sigma", "seed"]).status
    ev["run_status"] = [st.get((v, t, ls, s)) for v, t, ls, s in zip(ev.variant, ev.target, ev.log10_sigma, ev.seed)]
    ev["transition"] = ev.J_before + "->" + ev.pattern.astype(str)

    # one-step prediction B (route-entry norms) on plateau rows, then onto events
    pl = plats.copy()
    if not pl.empty:
        tw = {n: np.array(t["weights"], float) for n, t in targets.items()}
        pl["onestep_pred_time_B"] = [
            route_time(g * (b * sg) ** c, int(c), g, tw[t][q]) if np.isfinite(b) else math.nan
            for t, q, c, g, b, sg in zip(pl.target, pl.missing, pl.depth, pl.gamma, pl.ren_gmean_entry, pl.sigma)]
        pl["gamma_ratio_mg"] = pl.gamma_obs_mg / pl.gamma
        pl["ren_spread_entry"] = pl.ren_max_entry / pl.ren_min_entry
        winB = pl.dropna(subset=["onestep_pred_time_B"]).sort_values("onestep_pred_time_B").drop_duplicates(
            ["variant", "target", "log10_sigma", "seed", "event"])
        wB = {(t, ls, s, e): m for t, ls, s, e, m in zip(winB.target, winB.log10_sigma, winB.seed, winB.event, winB.missing)}
        nxt = pl[pl.learned_next]
        tB = {(t, ls, s, e): x for t, ls, s, e, x in zip(nxt.target, nxt.log10_sigma, nxt.seed, nxt.event, nxt.onestep_pred_time_B)}
        key = list(zip(ev.target, ev.log10_sigma, ev.seed, ev.event))
        ev["onestep_pred_time_B"] = [tB.get(k, np.nan) for k in key]
        ev["onestep_ratio_B"] = ev.t_entry_to_event / ev.onestep_pred_time_B
        ev["onestep_winner_B"] = [wB.get(k, -1) for k in key]
        ev.loc[ev.variant == "ref", ["onestep_pred_time_B", "onestep_ratio_B"]] = np.nan
        pl.to_csv(out / "plateaus.csv", index=False, float_format="%.8g")

    # ---- crossing histories; contaminated events ------------------------------------------------
    hist, cont, ps = history(Path(args.raw), runs, targets)
    runs["history"] = [hist.get(k) for k in zip(runs.variant, runs.target, runs.log10_sigma, runs.seed)]
    ev["contaminated"] = [cont.get(k, False) for k in zip(ev.variant, ev.target, ev.log10_sigma, ev.seed, ev.pattern)]
    # clean: nothing else crossed or was forgotten since the previous event, the pattern crossed once,
    # and (k >= 1) the plateau before it was resolved
    ev["clean"] = ~ev.contaminated & (ev.n_up == 1) & ev.has_entry.astype(bool)

    if len(ps):
        pk = pred.set_index(["target", "J", "missing"])
        oc1 = {(t, j): o for t, j, o in zip(oc.target, oc.J, oc.one_component_order)}
        def lookup(t, J, r, col):
            if J == "{}":
                return {"depth": ps_K[t], "pairwise": ps_K[t]}.get(col, np.nan)
            try:
                return pk.loc[(t, J, r), col]
            except KeyError:
                return np.nan
        ps_K = {n: int(targets[n]["n_sites"]) for n in targets}
        ps_K.update({n: v for n, v in zip(*np.unique(pred.target, return_index=True))} if False else {})
        from model import build_target as _bt
        ps_K = {n: _bt(targets[n]).tree.n_nodes for n in targets}
        for col in ("depth", "pairwise", "pred_greedy", "pred_pairwise", "pred_weight", "tie", "gamma"):
            ps[col] = [lookup(t, J, r, col) for t, J, r in zip(ps.target, ps.J_before, ps.pattern)]
        ps["one_component_order"] = [oc1.get((t, j), np.nan) for t, j in zip(ps.target, ps.J_before)]
        for rule in ("greedy", "pairwise", "weight"):
            ps[f"agree_{rule}"] = ps.pattern == ps[f"pred_{rule}"]
        ps["transition"] = ps.J_before + "->" + ps.pattern.astype(str)
        ps["precise"] = ps.time_resolution < 1e-3
        ps.to_csv(out / "passes.csv", index=False, float_format="%.8g")

    # ---- run classification ----------------------------------------------------------------------
    g = ev[ev.event > 0].groupby(["variant", "target", "log10_sigma", "seed"])
    nonc = g.apply(lambda x: int(((~x.canonical) & x.has_entry).sum()), include_groups=False)
    allg = g.apply(lambda x: bool(x.agree_greedy.all()), include_groups=False)
    runs["n_noncanonical"] = [nonc.get((v, t, ls, s), 0) for v, t, ls, s in
                             zip(runs.variant, runs.target, runs.log10_sigma, runs.seed)]
    runs["all_greedy"] = [allg.get((v, t, ls, s), np.nan) for v, t, ls, s in
                          zip(runs.variant, runs.target, runs.log10_sigma, runs.seed)]
    later = ev[ev.event > 0].groupby(["variant", "target", "log10_sigma", "seed"])
    unsep_later = later.unseparated.sum()
    noentry_later = later.has_entry.apply(lambda h: int((~h.astype(bool)).sum()))
    unsep0 = ev[ev.event == 0].set_index(["variant", "target", "log10_sigma", "seed"]).unseparated
    keyr = list(zip(runs.variant, runs.target, runs.log10_sigma, runs.seed))
    runs["n_unseparated_later"] = [int(unsep_later.get(k, 0)) for k in keyr]
    runs["n_no_entry"] = [int(noentry_later.get(k, 0)) for k in keyr]
    runs["first_escape_mixed"] = [bool(unsep0.get(k, False)) for k in keyr]
    first = ev[ev.event == 0].set_index(["variant", "target", "log10_sigma", "seed"]).pattern_str
    runs["first_pattern"] = [first.get((v, t, ls, s)) for v, t, ls, s in
                             zip(runs.variant, runs.target, runs.log10_sigma, runs.seed)]

    def behaviour(r):
        tags = []
        if r.status != "done":
            tags.append(str(r.status))
        if r.first_escape_mixed:
            tags.append("first_escape_mixed")
        if r.n_unseparated_later > 0 or r.n_no_entry > 0:
            tags.append("two_at_once")
        if r.n_forgotten > 0:
            tags.append("forgetting")
        if r.n_noncanonical > 0:
            tags.append("noncanonical_plateau")
        if r.floor_flag:
            tags.append("floor_flag")
        return "+".join(tags) if tags else "clean"

    runs["behaviour"] = runs.apply(behaviour, axis=1)
    cols = ID + ["variant", "status", "behaviour", "n_learned", "order", "order_idx", "first_pattern", "history",
                 "all_greedy", "n_forgotten", "forgotten", "first_escape_mixed", "n_unseparated", "n_unseparated_later",
                 "n_no_entry", "n_noncanonical",
                 "L_final", "t_final", "max_offtarget", "floor_flag", "floor_checked", "floor_flag_bound",
                 "steps", "segments", "restarts", "vode_segments", "fails", "secs"]
    runs = runs.sort_values(["variant", "target", "log10_sigma", "seed"])
    runs[[c for c in cols if c in runs]].to_csv(out / "runs.csv", index=False, float_format="%.8g")
    ev = ev.sort_values(["variant", "target", "log10_sigma", "seed", "event"])
    ev.to_csv(out / "events.csv", index=False, float_format="%.8g")

    # ---- exponents per seed and cell (three bases) ------------------------------------------------
    # event: the registered fits on every learning event; event_clean: only events whose interval
    # since the previous event holds no other crossing or forgetting; pass: every upward crossing
    def fits(df, basis):
        rows = []
        df = df[(df.duration > 0) & (df.log10_sigma >= WINDOW[0] - 1e-9) & (df.log10_sigma <= WINDOW[1] + 1e-9)]
        df = df[[(t_, ls_, sd_) not in flagged for t_, ls_, sd_ in zip(df.target, df.log10_sigma, df.seed)]]
        for (v, t, J, r, sd), x in df.groupby(["variant", "target", "J_before", "pattern", "seed"]):
            x = x.drop_duplicates("log10_sigma")
            if len(x) < 5 or x.log10_sigma.max() - x.log10_sigma.min() < 1.5 - 1e-9:
                continue
            ls = x.log10_sigma.to_numpy()
            slope, _ = per_seed_fit(ls, np.log10(x.duration.to_numpy()))
            lslope, _ = per_seed_fit(-ls * math.log(10), x.duration.to_numpy())  # duration vs ln(1/sigma)
            c = int(x.depth.iloc[0])
            rows.append(dict(variant=v, basis=basis, target=t, tree=x.tree.iloc[0], d=x.d.iloc[0],
                             chi=x.chi.iloc[0], seed=sd, J_before=J, pattern=r, pattern_str=x.pattern_str.iloc[0],
                             transition=f"{J}->{r}", depth=c, pairwise=int(x.pairwise.iloc[0]),
                             one_component_order=x.one_component_order.iloc[0], gamma=x.gamma.iloc[0],
                             n_points=len(x), log10_sigma_min=ls.min(), log10_sigma_max=ls.max(),
                             exponent=-slope, pred_route=c - 2, pred_pairwise=x.pairwise.iloc[0] - 2,
                             log_slope=lslope,
                             pred_log_slope=(1.0 / (targets[t]["weights"][r] * x.gamma.iloc[0]) if J != "{}" else np.nan)))
        return rows

    base = ev[ev.run_status == "done"]
    erows = fits(base, "event") + fits(base[base.clean], "event_clean")
    if len(ps):
        erows += fits(ps[ps.precise], "pass")
    ex = pd.DataFrame(erows)
    ex.to_csv(out / "exponents.csv", index=False, float_format="%.8g")

    # ---- cells: decisions ---------------------------------------------------------------------------
    def decide(e, c):
        q = e.exponent.quantile([0.25, 0.5, 0.75]).to_numpy()
        row = dict(n_fitted=len(e), exp_q1=q[0], exp_median=q[1], exp_q3=q[2],
                   frac_within_0p3_route=float((abs(e.exponent - (c - 2)) <= 0.3).mean()),
                   frac_within_0p3_pairwise=float((abs(e.exponent - e.pred_pairwise) <= 0.3).mean()),
                   log_slope_median=float(e.log_slope.median()), pred_log_slope=float(e.pred_log_slope.iloc[0]))
        o = e.one_component_order.iloc[0]
        if not math.isnan(o):
            row["frac_within_0p3_order"] = float((abs(e.exponent - (o - 2)) <= 0.3).mean())
        if c >= 3:
            if abs(q[1] - (c - 2)) <= 0.15 and row["frac_within_0p3_route"] >= 0.8:
                dec = "SUPPORTED"
            elif abs(q[1] - (c - 2)) > 0.5 and len(e) >= 5:
                dec = "REFUTED"
            else:
                dec = "INCONCLUSIVE"
        elif math.isnan(row["pred_log_slope"]):
            dec = "n/a"
        else:
            ok = abs(row["log_slope_median"] / row["pred_log_slope"] - 1) <= 0.25 and q[1] < 0.3
            dec = "SUPPORTED" if ok else "NOT SUPPORTED"
        row["decision_exponent"] = dec
        return row

    crows = []
    for basis, src in (("event", ev), ("event_clean", ev[ev.clean]), ("pass", ps)):
        if not len(src):
            continue
        small = src[(src.log10_sigma <= SMALL + 1e-9)]
        small = small[[(t_, ls_, sd_) not in flagged for t_, ls_, sd_ in zip(small.target, small.log10_sigma, small.seed)]]
        for (v, t, J, r), x in src.groupby(["variant", "target", "J_before", "pattern"]):
            xs = small[(small.variant == v) & (small.target == t) & (small.J_before == J) & (small.pattern == r)]
            c = int(x.depth.iloc[0])
            row = dict(variant=v, basis=basis, target=t, J_before=J, J_before_str=x.J_before_str.iloc[0],
                       pattern=r, pattern_str=x.pattern_str.iloc[0], transition=f"{J}->{r}", depth=c,
                       pairwise=x.pairwise.iloc[0], one_component_order=x.one_component_order.iloc[0],
                       pred_greedy=x.pred_greedy.iloc[0], pred_pairwise=x.pred_pairwise.iloc[0],
                       pred_weight=x.pred_weight.iloc[0], n_events=len(x), n_events_small=len(xs),
                       n_seeds=x.seed.nunique(), n_fitted=0)
            e = ex[(ex.basis == basis) & (ex.variant == v) & (ex.target == t) & (ex.J_before == J) & (ex.pattern == r)] if len(ex) else ex
            if len(e):
                row.update(decide(e, c))
            if len(xs) and basis != "pass":
                row.update(
                    eff_depth_median=float(xs.eff_depth.median()),
                    frac_eff_match=float((abs(xs.eff_depth - c) <= 0.3).mean()),
                    frac_canonical=float(xs.canonical[xs.has_entry].mean()) if xs.has_entry.any() else math.nan,
                    frac_unseparated=float(xs.unseparated.mean()),
                    peak_off_median=float(xs.peak_offtarget.median()),
                    onestep_ratio_median=float(xs.onestep_ratio.median()),
                    onestep_ratio_B_median=float(xs.onestep_ratio_B.median()),
                    onestep_winner_agree=float(xs.agree_onestep.dropna().mean()) if xs.agree_onestep.notna().any() else math.nan,
                    frac_P_entry_negative=float((xs.P_r_entry <= 0).mean()),
                )
            crows.append(row)
    cells = pd.DataFrame(crows)
    cells.to_csv(out / "cells.csv", index=False, float_format="%.8g")

    # ---- C.1 item 1: route-entry norm vs sigma per (target, J, missing) --------------------------
    c1 = []
    if not pl.empty:
        p2 = pl[(pl.variant == "dense") & (pl.log10_sigma <= -2 + 1e-9) & pl.ren_gmean_entry.notna()
                & (pl.ren_gmean_entry > 0)]
        for (t, J, q), x in p2.groupby(["target", "J", "missing"]):
            if x.log10_sigma.nunique() < 4:
                continue
            slope, _ = np.polyfit(x.log10_sigma, np.log10(x.ren_gmean_entry * x.sigma), 1)
            per_seed = []
            for _, y in x.groupby("seed"):
                y = y.drop_duplicates("log10_sigma")
                if len(y) >= 5 and y.log10_sigma.max() - y.log10_sigma.min() >= 1.5 - 1e-9:
                    per_seed.append(np.polyfit(y.log10_sigma, np.log10(y.ren_gmean_entry * y.sigma), 1)[0])
            per_seed = np.array(per_seed)
            c1.append(dict(target=t, J=J, missing=q, missing_str=x.missing_str.iloc[0], depth=int(x.depth.iloc[0]),
                           learned_next_frac=float(x.learned_next.mean()), n_rows=len(x),
                           slope_log_ren_vs_log_sigma=slope, n_seed_slopes=len(per_seed),
                           seed_slope_median=float(np.median(per_seed)) if len(per_seed) else math.nan,
                           seed_slope_q1=float(np.quantile(per_seed, 0.25)) if len(per_seed) else math.nan,
                           seed_slope_q3=float(np.quantile(per_seed, 0.75)) if len(per_seed) else math.nan,
                           frac_seed_slope_within_0p1=float((abs(per_seed - 1) <= 0.1).mean()) if len(per_seed) else math.nan,
                           ren_gmean_over_sigma_median=float(x.ren_gmean_entry.median()),
                           spread_max_over_min_median=float(x.ren_spread_entry.median()),
                           gamma_ratio_mg_median=float(x.gamma_ratio_mg.median()),
                           frac_gamma_within_5pct=float((abs(x.gamma_ratio_mg[x.log10_sigma <= SMALL + 1e-9] - 1) <= 0.05).mean()),
                           frac_P_entry_negative=float((x.P_entry <= 0).mean())))
    pd.DataFrame(c1).to_csv(out / "c1.csv", index=False, float_format="%.8g")

    # ---- summary ----------------------------------------------------------------------------------
    dn = runs[runs.variant == "dense"]
    summ = dict(
        n_runs=int(len(dn)), status=dn.status.value_counts().to_dict(),
        behaviour=dn.behaviour.value_counts().to_dict(),
        floor_checked=int(len(fc)), floor_flagged=int(fc.floor_flag.sum()) if len(fc) else 0,
    )
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
