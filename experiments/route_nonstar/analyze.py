"""Aggregate the per-run records into the tidy CSVs of results/ and score the predictors.

usage: uv run python experiments/route_nonstar/analyze.py [--raw DIR] [--out DIR]

Reads <raw>/raw/<target>/*.json (written by run.py) and configs/targets.yaml. Writes (columns
documented in the README): runs.csv, transitions.csv, events.csv, plateaus.csv, fits.csv,
fit_summary.csv, event_fits.csv, event_fit_summary.csv, escape_type.csv, sequence.csv,
run_classes.csv. Definitions and decision rules: notes/route_nonstar.md sections 3-5.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import load_targets  # noqa: E402

from ttn.routes import RouteTarget  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

WINDOW_MAX = -1.5  # fit window: log10 sigma <= -1.5
MIN_POINTS, MIN_SPAN = 5, 1.5
TOL = 0.3
LEVEL_MATCH = 0.05

# registered by-hand orders (note section 2): rank-1 optimum of the recombinant family = K
HAND_RANK1 = {"R3": 3, "R3d": 3, "R4": 4, "R4d": 4}


class TargetInfo:
    def __init__(self, t: dict):
        self.t = t
        self.name = t["name"]
        self.tree = make_tree(t["kind"], t["n"])
        self.pats = t["patterns"]
        self.w = t["weights"]
        self.rt = RouteTarget(self.tree, [[int(c) for c in p] for p in self.pats], self.w)
        self.levels = [s["loss"] for s in t["route_sequence"] if s["k_star"] > 0]
        self.levels_k = [s["k_star"] for s in t["route_sequence"] if s["k_star"] > 0]
        self.K = self.tree.n_nodes

    def idx(self, strings):
        return [self.pats.index(s) for s in strings]

    def jstr(self, J):
        return "{" + ",".join(self.pats[a] for a in sorted(J)) + "}"

    def level_tag(self, L):
        for j, lv in enumerate(self.levels):
            if lv > 0 and abs(L - lv) / lv < LEVEL_MATCH:
                return f"RS{j}"
        return f"L{L:.2g}"

    def paper(self, J):
        """(A): min route depth over the missing patterns and the greedy next (None if complete)."""
        miss = self.rt.missing(J)
        if not miss:
            return None, None, {}
        if not J:
            return self.K, max(miss, key=lambda r: self.w[r]), {r: self.K for r in miss}
        depths = {r: self.rt.depth(J, r) for r in miss}
        nxt = min(miss, key=lambda r: (depths[r], -self.w[r]))
        return depths[nxt], nxt, depths

    def pairwise(self, J):
        miss = self.rt.missing(J)
        if not miss:
            return None, None, {}
        if not J:
            return self.K, max(miss, key=lambda r: self.w[r]), {}
        costs = {r: min(self.rt.pair_cost(a, r) for a in J) for r in miss}
        nxt = min(miss, key=lambda r: (costs[r], -self.w[r]))
        return costs[nxt], nxt, costs

    def weight_next(self, J):
        miss = self.rt.missing(J)
        return max(miss, key=lambda r: self.w[r]) if miss else None

    def hand_order(self, key_canonical, J, level_tag):
        if self.name in HAND_RANK1 and not key_canonical and level_tag == "RS0":
            return HAND_RANK1[self.name]
        if key_canonical and len(J) == 1:
            j = J[0]
            ks = [self.rt.pair_cost(r, j) for r in range(len(self.pats)) if r != j]
            if min(ks) >= 2:
                return self.rt.one_component_order(j)["order"]
        return None


def exp_of(order):
    """Plateau exponent predicted by an order: p - 2 for p >= 3, 0 for p <= 2 (log law / none)."""
    if order is None:
        return 0.0
    return float(max(order - 2, 0))


def load_records(raw: Path):
    recs = []
    for f in sorted(raw.glob("raw/*/*.json")):
        recs.append(json.loads(f.read_text()))
    return recs


def ident(r):
    return dict(target=r["target"], tree=r["tree"], d=r["d"], chi=r["chi"], sigma=r["sigma"],
                log10_sigma=r["log10_sigma"], seed=r["seed"])


def build_tables(recs, infos):
    runs, trans, events, plats = [], [], [], []
    for r in recs:
        info = infos[r["target"]]
        segs, drops = r["segments"], r["drops"]
        flagged = any(s and s.get("floor_flag") for s in segs)
        n_forgot = len(r["forgotten"])
        runs.append(dict(**ident(r), family=r["family"], K=r["K"], n_params=r["n_params"],
                         status=r["status"], order_learned=" ".join(r["order"]),
                         n_learned=len(r["order"]), n_drops=len(drops),
                         n_plateaus=sum(1 for s in segs[1:] if s and "J_amp" in s),
                         forgotten=";".join(f"{f['pattern']}@{f['t_forgotten']:.6g}"
                                            for f in r["forgotten"]), n_forgotten=n_forgot,
                         floor_flagged=flagged, L_end=r["L_end"], off_end=r["off_end"],
                         t_first_escape=r.get("t_first_escape_abs"), t_end_rel=r["t_end"],
                         n_steps=r["n_steps"], n_restart=r["n_restart"], n_fails=len(r["fails"]),
                         secs=r["secs"]))
        # ---- plateaus = resolved segments (deviation 1: an unresolved segment between two drops
        # is a transient inside one escape, merged with the drops around it)
        def is_plateau(k, s):
            if s is None:
                return False
            if s.get("final") and r["status"] == "converged":
                return False
            return k == 0 or bool(s.get("resolved")) or (s.get("final") and "J_amp" in s)

        plateau_ks = [k for k, s in enumerate(segs) if is_plateau(k, s)]
        n_transient = sum(1 for k, s in enumerate(segs) if s is not None and not is_plateau(k, s)
                          and not (s.get("final") and r["status"] == "converged"))
        runs[-1]["n_transients"] = n_transient
        runs[-1]["n_plateaus"] = sum(1 for k in plateau_ks if k >= 1)
        P_end = np.array(r["P_end"])
        J_end = [a for a in range(len(info.pats)) if P_end[a] > info.w[a] / 2]
        for k, s in enumerate(segs):
            if k not in plateau_ks:
                continue
            final = s.get("final", False)
            converged_end = final and r["status"] == "converged"
            if converged_end:
                continue
            row = dict(**ident(r), family=r["family"], K=r["K"], status=r["status"], segment=k,
                       t_start=s["t_start"], t_end=s["t_end"], duration=s["t_end"] - s["t_start"],
                       censored=final, resolved=s.get("resolved"), t_mid=s["t_mid"], L=s["L"])
            if k == 0:
                row.update(key="origin", J="{}", canonical=None, level_tag="origin",
                           pred_paper=info.K, pred_route=info.K, pred_pair=info.K,
                           pred_hand=info.K, floor_flag=False)
            else:
                J = s["J_amp_idx"]
                tag = info.level_tag(s["L"])
                key = s["J_amp"] + ("" if s["canonical"] else "*") + "@" + tag
                cA, nA, depths = info.paper(J)
                cC, nC, costs = info.pairwise(J)
                nW = info.weight_next(J)
                route = s.get("route") or {}
                row.update(key=key, J=s["J_amp"], n_J=len(J), canonical=s["canonical"],
                           level_tag=tag, d_near=s["d_near"], J_near=s["J_near"],
                           d_amp=s["d_amp"], off_mass=s["off"],
                           off_strings=" ".join(f"{o['string']}:{o['amp']:+.3f}"
                                                for o in s["off_strings"]),
                           off_recombinants=" ".join(f"{o['string']}={o['tag']}"
                                                     for o in s["off_strings"] if o["tag"]),
                           n_off_strings=len(s["off_strings"]),
                           lam_min=s["lam_min"], lam_2=s["lam_2"], n_neg=s["n_neg"],
                           gnorm=s["gnorm"], grad_bound=s["grad_bound"],
                           floor_flag=s["floor_flag"], route_k=s["route_k"],
                           route_nodes=" ".join(map(str, route.get("nodes", []))),
                           route_string=route.get("string"),
                           route_string_is_target=route.get("is_target"),
                           t_escape=s["t_escape"],
                           pred_paper=cA, pred_paper_next=info.pats[nA] if nA is not None else None,
                           paper_depths=" ".join(f"{info.pats[q]}:{c}" for q, c in depths.items()),
                           pred_route=s["route_k"],
                           pred_route_next=route.get("string"),
                           pred_pair=cC, pred_pair_next=info.pats[nC] if nC is not None else None,
                           pred_weight_next=info.pats[nW] if nW is not None else None,
                           pred_hand=info.hand_order(s["canonical"], J, tag))
            # what the escape does: the drops up to the next plateau (or to the end), merged
            later = [q for q in plateau_ks if q > k]
            k2 = later[0] if later else len(segs) - 1
            group = [d for d in drops[k:k2] if d is not None] if k < len(drops) else []
            if group and (later or r["status"] == "converged"):
                J_this = set(s.get("J_amp_idx", []))
                J_next = set(segs[k2]["J_amp_idx"]) if later else set(J_end)
                big = max(group, key=lambda d: d["L_before"] - d["L_after"])
                amp = {}
                for d in group:
                    amp.update(d["eff_depth_amp"])
                learned = [info.pats[a] for a in sorted(J_next - J_this)]
                row.update(next_learned=" ".join(learned),
                           next_lost=" ".join(info.pats[a] for a in sorted(J_this - J_next)),
                           next_n_drops=len(group), next_top_change=big["top_change"],
                           next_top_change_is_target=big["top_change_is_target"],
                           next_L_after=group[-1]["L_after"],
                           next_off_peak=max(d["off_peak"] for d in group),
                           next_eff_depth_loss=group[0]["eff_depth_loss"],
                           next_eff_depth_amp=" ".join(f"{p}:{v:.3f}" if v else f"{p}:nan"
                                                       for p, v in amp.items()),
                           next_kind="learn" if learned else "cleanup")
            trans.append(row)
            # plateaus.csv: one row per (plateau, missing pattern)
            if k >= 1:
                J = s["J_amp_idx"]
                miss = info.rt.missing(J)
                base = dict(**ident(r), segment=k, key=row["key"], J=s["J_amp"],
                            J_near=s["J_near"], d_near=s["d_near"], canonical=s["canonical"],
                            off_mass=s["off"], off_strings=row["off_strings"],
                            off_recombinants=row["off_recombinants"], L=s["L"],
                            L_J=info.rt.partial_loss(J), route_k=s["route_k"])
                ent = s.get("entries") or {}
                for q in miss or [None]:
                    pr = dict(base)
                    if q is None:
                        pr.update(missing="none")
                    else:
                        p = info.pats[q]
                        norms = (ent.get("norms") or {}).get(p)
                        vals = list(norms.values()) if isinstance(norms, dict) else []
                        sig = r["sigma"]
                        pr.update(missing=p, weight=info.w[q], depth_paper=depths.get(q),
                                  pair_cost=costs.get(q), P_r=s["P"][q],
                                  entry_min=min(vals) / sig if vals else None,
                                  entry_max=max(vals) / sig if vals else None,
                                  entry_geo=float(np.exp(np.mean(np.log(np.maximum(vals, 1e-300)))))
                                  / sig if vals else None,
                                  align_quality=ent.get("quality"))
                    plats.append(pr)
        # ---- learning events (brief's events.csv)
        seg_of_drop = {}
        for e in r["events"]:
            J = e["J_before"]
            q = e["a"]
            k_drop = e["drop"]
            prior = [segs[q] for q in plateau_ks if q >= 1 and segs[q]["t_mid"] < e["t"]]
            before = prior[-1] if prior else None
            depth = info.rt.depth(J, q) if J else info.K
            pair = min((info.rt.pair_cost(a, q) for a in J), default=info.K)
            cA, nA, _ = info.paper(J)
            cC, nC, _ = info.pairwise(J)
            nW = info.weight_next(J)
            d = drops[k_drop - 1] if k_drop and drops[k_drop - 1] else None
            events.append(dict(**ident(r), pattern=e["pattern"], weight=info.w[q], t_event=e["t"],
                               J_before=info.jstr(J), n_J_before=len(J),
                               duration=e["duration"], drop=k_drop,
                               depth_paper=depth, pair_cost=pair,
                               pred_paper_next=info.pats[nA] if nA is not None else None,
                               pred_pair_next=info.pats[nC] if nC is not None else None,
                               pred_weight_next=info.pats[nW] if nW is not None else None,
                               route_k_before=before.get("route_k") if before else None,
                               pred_route_next=(before.get("route") or {}).get("string")
                               if before else None,
                               eff_depth=e["eff_depth"],
                               off_peak=d["off_peak"] if d else None,
                               plateau_before_L=before.get("L") if before else None,
                               plateau_before_L_J=info.rt.partial_loss(J),
                               plateau_before_d_canon=_dist_to(before, info, J)
                               if before else None))
    return (pd.DataFrame(runs), pd.DataFrame(trans), pd.DataFrame(events), pd.DataFrame(plats))


def _dist_to(seg, info, J):
    """||W - sum_J s_a U_a|| / ||W*|| at the plateau point, from its amplitudes and off mass."""
    if not seg or "P" not in seg:
        return None
    P = np.array(seg["P"])
    w = np.array(info.w)
    on = np.where(np.isin(np.arange(len(w)), J), w, 0.0)
    tot = float((P**2).sum()) / max(1 - seg["off"], 1e-300) if seg["off"] < 1 else None
    off2 = seg["off"] * tot if tot is not None else 0.0
    return math.sqrt(off2 + float(((P - on) ** 2).sum())) / math.sqrt(float((w**2).sum()))


def per_seed_fits(df, group_cols, dur_col, extra_cols, min_span=MIN_SPAN, min_points=MIN_POINTS):
    """Per-seed least-squares slopes of -log10(duration) on log10(sigma) inside the window."""
    out = []
    win = df[(df.log10_sigma <= WINDOW_MAX + 1e-9) & (df[dur_col] > 0)]
    for keys, g in win.groupby(group_cols + ["seed"], dropna=False):
        g = g.sort_values("log10_sigma")
        g = g.groupby("log10_sigma", as_index=False).first()  # one visit per sigma (first one)
        n = len(g)
        span = g.log10_sigma.max() - g.log10_sigma.min() if n else 0
        row = dict(zip(group_cols + ["seed"], keys))
        row.update(n_points=n, span=span)
        if n >= min_points and span >= min_span - 1e-9:
            x, y = g.log10_sigma.values, np.log10(g[dur_col].values)
            slope, icpt = np.polyfit(x, y, 1)
            resid = y - (slope * x + icpt)
            row.update(exponent=-slope, rms=float(np.sqrt((resid**2).mean())), fitted=True)
        else:
            row.update(exponent=np.nan, rms=np.nan, fitted=False)
        for c in extra_cols:
            row[c] = g[c].median() if c in g and g[c].notna().any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def decide(med, frac, pred):
    if pred is None or np.isnan(pred) or np.isnan(med):
        return "n/a"
    if abs(med - pred) <= TOL and frac >= 0.7:
        return "holds"
    if abs(med - pred) > 0.5 or frac < 0.3:
        return "fails"
    return "partial"


def summarize_fits(fits, group_cols, preds, rows_all=None):
    """Per group: distribution of the per-seed exponents and, per predictor, the predicted
    exponent, the fraction of fitted seeds within TOL and the decision. Predictions are medians
    over every window row of the group (rows_all), so groups without any fit still appear."""
    src = rows_all if rows_all is not None else fits
    rows = []
    for keys, g in src.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        sel = np.ones(len(fits), dtype=bool)
        for c, v in zip(group_cols, keys):
            sel &= (fits[c] == v).values
        f = fits[sel & fits.fitted.values] if len(fits) else fits
        row = dict(zip(group_cols, keys))
        e = f.exponent.values if len(f) else np.array([])
        row.update(n_seeds=int(sel.sum()), n_fitted=len(e),
                   median=float(np.median(e)) if len(e) else np.nan,
                   q25=float(np.percentile(e, 25)) if len(e) else np.nan,
                   q75=float(np.percentile(e, 75)) if len(e) else np.nan)
        for name, col in preds.items():
            p = g[col].median() if col in g and g[col].notna().any() else np.nan
            pe = exp_of(p) if not np.isnan(p) else np.nan
            if name in ("paper", "pair") and np.isnan(pe) and "origin" not in str(keys):
                pe = 0.0  # every pattern already in J: these rules predict no plateau
            frac = float(np.mean(np.abs(e - pe) <= TOL)) if len(e) and not np.isnan(pe) else np.nan
            row[f"order_{name}"] = p
            row[f"exp_{name}"] = pe
            row[f"frac_within_{name}"] = frac
            row[f"decision_{name}"] = decide(row["median"], frac if not np.isnan(frac) else 0, pe)
            # deviation 3: an observed (resolved) plateau where a rule predicts none (order <= 1,
            # or nothing missing) refutes the rule whatever the exponent
            none = (name in ("paper", "pair") and (np.isnan(p) or p <= 1)) or \
                (name == "route" and not np.isnan(p) and p <= 1)
            row[f"predicts_plateau_{name}"] = not none
            if none and "origin" not in str(keys):
                row[f"decision_{name}"] = "fails (no plateau predicted)"
        rows.append(row)
    return pd.DataFrame(rows)


def escape_types(trans):
    rows = []
    t = trans[(trans.segment >= 1) & (trans.log10_sigma <= WINDOW_MAX + 1e-9)
              & trans.lam_min.notna() & (trans.floor_flag == False)]  # noqa: E712
    for (target, key), g in t.groupby(["target", "key"]):
        med = g.groupby("log10_sigma").lam_min.median()
        neg = (med < 0)
        row = dict(target=target, key=key, n_points=len(g), n_sigmas=len(med),
                   lam_min_at_smallest=float(med.iloc[0]), lam_min_at_largest=float(med.iloc[-1]),
                   all_neg=bool(neg.all()))
        if len(med) >= 3:
            x = med.index.values
            y = np.log10(np.abs(med.values) + 1e-300)
            row["slope_x"] = float(np.polyfit(x, y, 1)[0])
        else:
            row["slope_x"] = np.nan
        strict = bool((med < -1e-4).all()) and row["slope_x"] < 0.25
        degenerate = row["slope_x"] >= 0.5
        row["type"] = "strict" if strict else ("degenerate" if degenerate else "unclear")
        row["n_neg_median"] = float(g.n_neg.median())
        rows.append(row)
    return pd.DataFrame(rows)


def sequence_scores(trans):
    t = trans[(trans.segment >= 1) & trans.next_kind.notna()].copy()

    def hit(row, col):
        learned = (row.next_learned or "").split()
        pred = row[col]
        if row.next_kind == "learn":
            return pred in learned if pred else False
        if col == "pred_route_next":
            return bool(pred) and pred == row.next_top_change and not row.next_top_change_is_target
        return False

    for col in ["pred_paper_next", "pred_route_next", "pred_pair_next", "pred_weight_next"]:
        t["hit_" + col.replace("pred_", "").replace("_next", "")] = t.apply(lambda r: hit(r, col),
                                                                         axis=1)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/outputs/route-nonstar")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, targets = load_targets()
    infos = {n: TargetInfo(t) for n, t in targets.items()}
    recs = load_records(Path(a.raw))
    print(f"{len(recs)} records")
    runs, trans, events, plats = build_tables(recs, infos)
    first = ["target", "tree", "d", "chi", "sigma", "log10_sigma", "seed"]
    for df, name in [(runs, "runs"), (trans, "transitions"), (events, "events"),
                     (plats, "plateaus")]:
        df.sort_values(first + ([c for c in ["segment", "t_event"] if c in df]),
                       inplace=True)
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    # ---- run classes
    cls = runs.groupby(["target", "log10_sigma", "status"]).size().unstack(fill_value=0)
    cls["floor_flagged"] = runs.groupby(["target", "log10_sigma"]).floor_flagged.sum()
    cls["forgetting"] = runs.groupby(["target", "log10_sigma"]).n_forgotten.apply(
        lambda x: (x > 0).sum())
    cls.reset_index().to_csv(out / "run_classes.csv", index=False)
    # ---- duration fits per plateau key (transitions) and per event identity
    ok = trans[(trans.status == "converged") & (trans.floor_flag != True)  # noqa: E712
               & (trans.censored == False)]  # noqa: E712
    fits = per_seed_fits(ok, ["target", "key"], "duration",
                         ["pred_paper", "pred_route", "pred_pair", "pred_hand"])
    fits.to_csv(out / "fits.csv", index=False, float_format="%.6g")
    # deviation 2 (secondary, labelled): depth-5 plateaus are floor-flagged at sigma <= 1e-3, which
    # leaves < 1.5 decades; (a) unflagged points with a 1-decade span, (b) flagged points included
    conv = trans[(trans.status == "converged") & (trans.censored == False)]  # noqa: E712
    sec = []
    for variant, rows_, span in [("unflagged_span1", ok, 1.0), ("with_flagged", conv, MIN_SPAN)]:
        f2 = per_seed_fits(rows_, ["target", "key"], "duration",
                           ["pred_paper", "pred_route", "pred_pair", "pred_hand"], min_span=span)
        f2["variant"] = variant
        sec.append(f2)
    sec = pd.concat(sec)
    sec.to_csv(out / "fits_secondary.csv", index=False, float_format="%.6g")
    preds = dict(paper="pred_paper", route="pred_route", pair="pred_pair", hand="pred_hand")
    win_rows = trans[trans.log10_sigma <= WINDOW_MAX + 1e-9]
    summ = summarize_fits(fits, ["target", "key"], preds, win_rows)
    # attach what the plateau is (P1) and what leaves it
    desc = trans[(trans.log10_sigma <= WINDOW_MAX + 1e-9)].groupby(["target", "key"]).agg(
        n_points=("seed", "size"), frac_canonical=("canonical", "mean"),
        d_near_median=("d_near", "median"), off_mass_median=("off_mass", "median"),
        L_median=("L", "median"), off_recombinants=("off_recombinants", _mode),
        next_kind=("next_kind", _mode), next_learned=("next_learned", _mode),
        eff_depth_loss_median=("next_eff_depth_loss", "median"),
        route_k_median=("route_k", "median"), floor_flagged=("floor_flag", "sum"))
    summ = summ.merge(desc.reset_index(), on=["target", "key"], how="left")
    for variant in ["unflagged_span1", "with_flagged"]:
        sv = summarize_fits(sec[sec.variant == variant], ["target", "key"], preds, win_rows)
        sv = sv[["target", "key", "n_fitted", "median", "q25", "q75", "frac_within_route",
                 "frac_within_paper", "frac_within_hand"]]
        sv.columns = ["target", "key"] + [f"{c}_{variant}" for c in sv.columns[2:]]
        summ = summ.merge(sv, on=["target", "key"], how="left")
    summ.to_csv(out / "fit_summary.csv", index=False, float_format="%.4g")
    ev_ok = events.merge(runs[["target", "sigma", "seed", "status", "floor_flagged"]],
                         on=["target", "sigma", "seed"])
    ev_ok = ev_ok[(ev_ok.status == "converged") & (ev_ok.floor_flagged == False)]  # noqa: E712
    ev_ok = ev_ok.rename(columns={"depth_paper": "pred_paper", "pair_cost": "pred_pair",
                                  "route_k_before": "pred_route"})
    ev_ok["ident"] = ev_ok.J_before + "->" + ev_ok.pattern
    efits = per_seed_fits(ev_ok, ["target", "ident"], "duration",
                          ["pred_paper", "pred_route", "pred_pair", "eff_depth"])
    efits.to_csv(out / "event_fits.csv", index=False, float_format="%.6g")
    esumm = summarize_fits(efits, ["target", "ident"],
                           dict(paper="pred_paper", route="pred_route", pair="pred_pair"),
                           ev_ok[ev_ok.log10_sigma <= WINDOW_MAX + 1e-9])
    esumm.to_csv(out / "event_fit_summary.csv", index=False, float_format="%.4g")
    # ---- escape type and sequence agreement
    escape_types(trans).to_csv(out / "escape_type.csv", index=False, float_format="%.4g")
    seq = sequence_scores(trans)
    cols = [c for c in seq.columns if c.startswith("hit_")]
    seq_s = seq.groupby(["target", "log10_sigma"])[cols].mean().reset_index()
    seq_s["n_transitions"] = seq.groupby(["target", "log10_sigma"]).size().values
    seq_s.to_csv(out / "sequence.csv", index=False, float_format="%.4g")
    esc = pd.read_csv(out / "escape_type.csv") if (out / "escape_type.csv").stat().st_size > 1 \
        else pd.DataFrame()
    write_summary(out, infos, runs, trans, summ, esc, seq_s)
    print(summ[["target", "key", "n_fitted", "median", "exp_paper", "exp_route", "exp_pair",
                "exp_hand", "decision_paper", "decision_route", "decision_pair",
                "frac_canonical"]].to_string())


def write_summary(out: Path, infos, runs, trans, summ, esc, seq):
    """results/summary_by_target.md: per target, its plateaus, how each predictor scored, the
    escape type, the sequence scores and the run counts (the tables quoted in the report)."""
    esc = esc.set_index(["target", "key"]) if len(esc) else esc
    lines = ["# Per-target summary (generated by analyze.py)", "",
             "Window log10 sigma <= -1.5. Exponent = per-seed slope of -log10(plateau duration) on "
             "log10(sigma): median [quartiles] (n fitted seeds). Predictions: paper = max(c*_J - 2, "
             "0) (0 also when nothing is missing), route = k* - 2 of the numerical route rule at "
             "the plateau point, pair = pairwise rule, hand = registered by-hand order - 2; in "
             "brackets the fraction of seeds within 0.3 and the decision. canon = fraction of "
             "plateau points within 0.03 of a partial target. Escape: strict / degenerate with x "
             "= slope of log|lambda_min| on log sigma. eff = median effective depth of the escape "
             "(loss estimator).", ""]
    seq_w = seq[seq.log10_sigma <= WINDOW_MAX + 1e-9]
    for name, info in infos.items():
        r = runs[runs.target == name]
        if not len(r):
            continue
        viol = "; ".join(f"e{e}: {'/'.join(info.pats[a] for a in m)}"
                         for e, m in info.rt.star_violations) or "none (star condition holds)"
        lines += [f"## {name} ({info.t['kind']}{info.t['n']}, K = {info.K})", "",
                  "patterns: " + ", ".join(f"{p} ({w:g})" for p, w in zip(info.pats, info.w)),
                  f"star violations: {viol}", ""]
        st = r.status.value_counts().to_dict()
        lines.append(f"runs: {len(r)}; status {st}; floor-flagged at some plateau: "
                     f"{int(r.floor_flagged.sum())}; with a forgetting event: "
                     f"{int((r.n_forgotten > 0).sum())}")
        orders = r[r.log10_sigma <= WINDOW_MAX + 1e-9].order_learned.value_counts()
        lines.append("learning orders (window): " + "; ".join(
            f"{o} x{c}" for o, c in orders.head(4).items()))
        lines += ["", "| plateau key | canon | nearest partial target, median distance | "
                  "off-target strings (recombinant tag) | level L | exits by | exponent | "
                  "paper | route | pair | hand | escape | eff |", "|" + "---|" * 13]
        sm = summ[(summ.target == name)].copy()
        sm = sm[(sm.n_points.fillna(0) >= 20) | (sm.key == "origin")]
        sm = sm.sort_values("L_median", ascending=False)
        for _, x in sm.iterrows():
            def pr(p):
                e = x.get(f"exp_{p}")
                if e is None or (isinstance(e, float) and np.isnan(e)):
                    return "-"
                fr = x.get(f"frac_within_{p}")
                fr = "-" if fr is None or np.isnan(fr) else f"{fr:.2f}"
                return f"{e:g} ({fr}, {x.get(f'decision_{p}')})"
            et = "-"
            if (name, x.key) in getattr(esc, "index", []):
                e_ = esc.loc[(name, x.key)]
                et = f"{e_['type']} (x = {e_['slope_x']:.2f}, lam_min {e_['lam_min_at_smallest']:.2g})"
            tp = trans[(trans.target == name) & (trans.key == x.key)]
            near = tp.J_near.mode().iloc[0] if "J_near" in tp and tp.J_near.notna().any() else "-"
            dn = f"{x.d_near_median:.3g}" if not np.isnan(x.d_near_median) else "-"
            ex = f"{x['median']:.2f} [{x.q25:.2f}, {x.q75:.2f}] ({int(x.n_fitted)})" \
                if x.n_fitted else f"- (0 of {int(x.n_seeds)})"
            for v, lab in [("unflagged_span1", "1-decade"), ("with_flagged", "incl. flagged")]:
                nf = x.get(f"n_fitted_{v}")
                if not x.n_fitted and nf and not np.isnan(nf) and nf > 0:
                    ex += (f"; {lab}: {x[f'median_{v}']:.2f} [{x[f'q25_{v}']:.2f}, "
                           f"{x[f'q75_{v}']:.2f}] ({int(nf)})")
            canon = "-" if np.isnan(x.frac_canonical) else f"{x.frac_canonical:.2f}"
            eff = "-" if np.isnan(x.eff_depth_loss_median) else f"{x.eff_depth_loss_median:.2f}"
            nxt = x.next_kind if isinstance(x.next_kind, str) else "-"
            if nxt == "learn":
                nxt = "learning " + str(x.next_learned)
            lines.append(f"| {x.key} | {canon} | {near}, {dn} | {x.off_recombinants or '-'} | "
                         f"{x.L_median:.4g} | {nxt} | {ex} | {pr('paper')} | {pr('route')} | "
                         f"{pr('pair')} | {pr('hand')} | {et} | {eff} |")
        sw = seq_w[seq_w.target == name]
        if len(sw):
            tot = sw.n_transitions.sum()
            sc = {c: float((sw[c] * sw.n_transitions).sum() / tot) for c in
                  ["hit_paper", "hit_route", "hit_pair", "hit_weight"]}
            lines += ["", f"next-step agreement over {int(tot)} escapes (window): "
                      + ", ".join(f"{k[4:]} {v:.2f}" for k, v in sc.items())]
        lines.append("")
    (out / "summary_by_target.md").write_text("\n".join(lines))


def _mode(x):
    x = x.dropna()
    if not len(x):
        return None
    return x.astype(str).value_counts().index[0]


if __name__ == "__main__":
    main()
