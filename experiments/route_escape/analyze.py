"""Tidy CSVs from the raw JSON of the route-escape sweep (notes/route_escape.md sections 3-5).

Writes to experiments/route_escape/results/: cells.csv, static_orders.csv, static_curves.csv,
thm92.csv, escape_runs.csv, escape_fits.csv, decisions.csv. Column meanings: README.md.

Usage: uv run python experiments/route_escape/analyze.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, OUT, all_cells, get_cell  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GRID = yaml.safe_load((HERE / "configs" / "sweep.yaml").read_text())
WIN = GRID["window"]
FLOOR = 0.01


def target_string(cell):
    return " ".join(f"{p}:{w:g}" for p, w in zip(cell.pattern_strings, cell.weights))


def ident(cell):
    return dict(cell=cell.name, target=target_string(cell), tree=cell.tree_name, d=cell.d, chi=cell.chi,
                J=" ".join(cell.pattern_strings[a] for a in cell.J))


# ---- cells -----------------------------------------------------------------------------------


def cells_table():
    rows = []
    for name in all_cells():
        c = get_cell(name)
        t92 = c.thm92()
        for m in c.describe_missing():
            parent = None
            for a, p in t92["per_parent"].items():
                if m["r"] in p["residuals"]:
                    parent = a
            pp = t92["per_parent"][parent]
            rows.append(dict(
                **ident(c), role=c.cfg["role"], n_params=c.spec.n_params, n_entries=c.man.n_entries,
                L_J=c.L_J, c_star=c.c_star, greedy=c.pattern_strings[c.greedy()],
                missing=m["pattern"], weight=m["weight"], depth=m["depth"],
                region=" ".join(map(str, m["region"])), pairwise=m["pairwise"], A=m["A"], sA=m["sA"],
                shared_dormant=m["n_shared_dormant"],
                parent=c.pattern_strings[parent], parent_k_star=pp["k_star"],
                parent_l_star=pp["l_star"] if math.isfinite(pp["l_star"]) else None,
                parent_order=pp["order"], thm92_formula=t92["order"],
                thm92_family=bool(c.cfg.get("parents") is not None),
            ))
    return pd.DataFrame(rows)


# ---- static ----------------------------------------------------------------------------------


def static_tables():
    curves, summ = [], []
    for name in all_cells():
        c = get_cell(name)
        for space in ("Mstar", "full"):
            p = OUT / "static" / f"{name}_{space}.json"
            if not p.exists():
                continue
            o = json.loads(p.read_text())
            rows = o["rows"]
            for r in rows:
                curves.append(dict(**ident(c), space=space, t=r["t"], m=r["m"], m_over_tc=r["m_over_tc"],
                                   local_slope=r.get("slope"), w_active=r["w_active"],
                                   w_dormant=r["w_dormant"], w_offM=r["w_offM"],
                                   D_offtarget_frac=r["D_offtarget_frac"]))
            sl = [r.get("slope") for r in rows]
            last, prev = sl[-1], sl[-2]
            resolved = last is not None and prev is not None and abs(last - prev) <= 0.1
            last_row = rows[-1]
            pred = c.c_star
            row = dict(**ident(c), space=space, c_star=c.c_star, order=last, order_prev=prev,
                       resolved=resolved,
                       m_over_tc_smallest=last_row["m_over_tc"],
                       w_offM_smallest=last_row["w_offM"], w_dormant_smallest=last_row["w_dormant"],
                       D_offtarget_frac_smallest=last_row["D_offtarget_frac"],
                       D_top_offtarget=" ".join(f"{s}:{v:.2e}" for s, v in last_row["D_top_offtarget"]),
                       D_target=" ".join(f"{s}:{v:.2e}" for s, v in last_row["D_target"].items()))
            if space == "Mstar":
                row["decision"] = "P1 holds" if last is not None and abs(last - pred) <= 0.2 else "P1 fails"
            else:
                if last is None:
                    row["decision"] = "no descent"
                elif abs(last - pred) <= 0.2:
                    row["decision"] = "= c*_J"
                elif last < pred - 0.5:
                    row["decision"] = "shortcut"
                else:
                    row["decision"] = "unresolved"
                h = o.get("hessian", {})
                row.update({f"hess_{k}": v for k, v in h.items()})
            summ.append(row)
    return pd.DataFrame(curves), pd.DataFrame(summ)


def thm92_table(static_summary):
    rows = []
    for name in all_cells():
        c = get_cell(name)
        if c.cfg.get("parents") is None:
            continue
        t92 = c.thm92()
        full = static_summary[(static_summary.cell == name) & (static_summary.space == "full")]
        mst = static_summary[(static_summary.cell == name) & (static_summary.space == "Mstar")]
        measured = float(full.order.iloc[0]) if len(full) and full.order.iloc[0] is not None else None
        for a, p in t92["per_parent"].items():
            rows.append(dict(
                **ident(c), parent=c.pattern_strings[a], parent_weight=float(c.weights[a]),
                residuals=" ".join(f"{c.pattern_strings[r]}:{c.weights[r]:g}" for r in p["residuals"]),
                k_star=p["k_star"], l_star=p["l_star"] if math.isfinite(p["l_star"]) else None,
                mu=" ".join(f"{lv}:{m:.3f}" for lv, m in p["mu"].items()),
                parent_order=p["order"], mechanism=p["mechanism"],
                formula=t92["order"], c_star=c.c_star,
                order_full=measured, order_full_resolved=bool(full.resolved.iloc[0]) if len(full) else None,
                order_Mstar=float(mst.order.iloc[0]) if len(mst) and mst.order.iloc[0] is not None else None,
                holds=(measured is not None and abs(measured - t92["order"]) <= 0.2),
            ))
    return pd.DataFrame(rows)


# ---- escape runs -----------------------------------------------------------------------------


def outcome(o):
    if o["t_esc"] is not None:
        return "floor_flagged" if o["floor_phi"] > FLOOR else "escaped"
    if o["status"] == "trapped":
        return "trapped"
    if o["status"] == "failed":
        return "failed"
    return "unfinished"


def diag_rows(o, r):
    """Route-entry diagnostics of pattern r at t = 0, the switch checkpoint (t ~ 200) and the last
    checkpoint before t_esc / 3 (late plateau)."""
    d = o.get("diag", [])
    key = f"route_{r}"
    k = o["missing"].index(r)
    out = {}
    if not d:
        return out

    def summ(row, tag):
        norms = np.array(list(row[key].values()))
        out[f"route_min_{tag}"] = float(norms.min()) / o["eps"]
        out[f"route_max_{tag}"] = float(norms.max()) / o["eps"]
        out[f"route_gm_{tag}"] = float(np.exp(np.log(np.maximum(norms, 1e-300)).mean())) / o["eps"]
        out[f"P_{tag}"] = row["P"][k]
        out[f"t_{tag}"] = row["t"]
        out[f"align_{tag}"] = row["align_min"]

    summ(d[0], "t0")
    sw = [x for x in d if x["t"] >= 199]
    if sw:
        summ(sw[0], "sw")
    if o["t_esc"] is not None:
        late = [x for x in d if x["t"] <= o["t_esc"] / 3]
    else:
        late = d
    if late:
        summ(late[-1], "late")
    return out


def escape_runs():
    rows = []
    for name in all_cells():
        c = get_cell(name)
        if c.cfg["role"] != "dyn":
            continue
        for p in sorted((OUT / "escape" / name).glob("*.json")):
            o = json.loads(p.read_text())
            g = c.greedy()
            row = dict(**ident(c), kind=o["kind"], seed=o["seed"], log_eps=o["log_eps"], eps=o["eps"],
                       c_star=c.c_star, greedy=c.pattern_strings[g], greedy_depth=c.depth(g),
                       outcome=outcome(o), status=o["status"], t_esc=o["t_esc"],
                       t_esc_1e4=o["esc_other"].get("1e-04"), t_esc_1e2=o["esc_other"].get("1e-02"),
                       floor_phi=o["floor_phi"],
                       first_learned=c.pattern_strings[o["first_learned"]] if o["first_learned"] is not None else None,
                       first_depth=c.depth(o["first_learned"]) if o["first_learned"] is not None else None,
                       t_learn=o["t_learn"],
                       first_is_greedy=(o["first_learned"] == g) if o["first_learned"] is not None else None,
                       esc_pattern=c.pattern_strings[o["esc_pattern"]] if o.get("esc_pattern") is not None else None,
                       offmass_esc=o.get("offmass_esc"),
                       eff_depth=o.get("eff_depth"),
                       forgotten=" ".join(c.pattern_strings[a] for a in o["forgotten"]),
                       J_end=" ".join(c.pattern_strings[a] for a in o["J_end"]),
                       P_greedy_0=o["P0"][g], n_steps=o["n_steps"], n_restart=o["n_restart"],
                       n_fails=len(o["fails"]), secs=o["secs"])
            if "reduced" in o:
                red = o["reduced"]
                row["t_red"] = red["t_esc"]
                row["first_red"] = c.pattern_strings[red["first_learned"]] if red["first_learned"] is not None else None
                row["status_red"] = red["status"]
                row["rel_full_red"] = (o["t_esc"] / red["t_esc"] - 1) if (o["t_esc"] and red["t_esc"]) else None
                iso = {int(k): v for k, v in o["iso"].items()}
                row["t_iso"] = iso[g]["t_iso"]
                row["t_616"] = iso[g]["t_616"]
                win = min(iso, key=lambda r: iso[r]["t_iso"])
                row["iso_winner"] = c.pattern_strings[win]
                row["t_iso_winner"] = iso[win]["t_iso"]
                # the isolated-route prediction for the pattern that is actually learned first
                fl = o["first_learned"]
                row["t_iso_first"] = iso[fl]["t_iso"] if fl is not None else None
                row["ratio_iso"] = o["t_esc"] / iso[fl]["t_iso"] if (fl is not None and o["t_esc"]) else None
                row["ratio_616"] = o["t_esc"] / iso[g]["t_616"] if o["t_esc"] else None
                row["P_greedy_0_sign"] = int(np.sign(iso[g]["P0"]))
            for k, v in diag_rows(o, g).items():
                row[k] = v
            rows.append(row)
    return pd.DataFrame(rows)


def static_order_for(static_summary, cell):
    s = static_summary[(static_summary.cell == cell) & (static_summary.space == "full")]
    if not len(s) or s.order.iloc[0] is None or not np.isfinite(s.order.iloc[0]):
        return None
    return float(s.order.iloc[0])


def escape_fits(runs, static_summary):
    rows = []
    for (cell, kind, seed), g in runs.groupby(["cell", "kind", "seed"]):
        c = get_cell(cell)
        gr = c.greedy()
        p_M = c.c_star
        p_full = static_order_for(static_summary, cell)
        p_full_int = int(round(p_full)) if p_full is not None else None
        p_pred = p_M if kind != "C" else (p_full_int if p_full_int is not None else p_M)
        w = g[(g.log_eps >= WIN[0] - 1e-9) & (g.log_eps <= WIN[1] + 1e-9)].sort_values("log_eps")
        n_win = len(w)
        complete = n_win == 7 and (w.outcome == "escaped").all()
        if (w.outcome == "trapped").any():
            cls = "trapped"
        elif not complete:
            cls = "incomplete"
        else:
            cls = None
        row = dict(**ident(c), kind=kind, seed=seed, c_star=p_M, order_full_static=p_full,
                   p_pred=p_pred, gamma_pred=p_pred - 2, n_window=n_win,
                   n_escaped=int((w.outcome == "escaped").sum()),
                   n_trapped=int((w.outcome == "trapped").sum()),
                   n_flagged=int((w.outcome == "floor_flagged").sum()),
                   n_unfinished=int((w.outcome == "unfinished").sum()),
                   n_failed=int((w.outcome == "failed").sum()),
                   greedy_frac_window=float(w.first_is_greedy.fillna(False).astype(bool).mean()) if n_win else None,
                   complete=complete)
        if complete:
            x = -np.log(w.eps.values)
            y = np.log(w.t_esc.values.astype(float))
            gam = float(np.polyfit(x, y, 1)[0])
            loc = np.diff(y) / np.diff(x)
            row.update(gamma=gam, local_last=float(loc[-1]) if len(loc) else None,
                       local_first=float(loc[0]) if len(loc) else None)
            row["log_slope"] = float(np.polyfit(x, w.t_esc.values.astype(float), 1)[0])
            if c.depth(gr) == 2:
                row["log_slope_pred"] = 1.0 / (c.weights[gr] * c.A(gr))
            dev = gam - (p_pred - 2)
            row["dev"] = dev
            cls = "at_prediction" if abs(dev) <= 0.3 else "shifted"
            row["shift_int"] = int(round(dev))
            smallest = w.iloc[0]
            row["first_at_smallest"] = smallest.first_learned
            if kind in ("A", "A_bal", "B"):
                row["ratio_iso_smallest"] = smallest.ratio_iso
                row["ratio_616_smallest"] = smallest.ratio_616
                row["max_abs_ratio_iso_minus1"] = float(np.nanmax(np.abs(w.ratio_iso.values.astype(float) - 1)))
                row["max_rel_full_red"] = float(np.nanmax(np.abs(w.rel_full_red.values.astype(float))))
            row["offmass_esc_smallest"] = smallest.offmass_esc
            row["eff_depth_smallest"] = smallest.eff_depth
        # diagnostics at the smallest window eps for all directions
        if n_win:
            s0 = w.iloc[0]
            for k in ("P_greedy_0", "P_sw", "route_min_sw", "route_max_sw", "route_min_late", "route_gm_sw",
                      "route_gm_late", "P_greedy_0_sign"):
                if k in w.columns:
                    row[k + "_smallest"] = s0.get(k)
        row["class"] = cls
        rows.append(row)
    return pd.DataFrame(rows)


def decisions(fits, runs):
    rows = []
    for (cell, kind), f in fits.groupby(["cell", "kind"]):
        comp = f[f.complete]
        n = len(f)
        row = dict(cell=cell, kind=kind, n_directions=n, n_complete=len(comp),
                   n_at_pred=int((f["class"] == "at_prediction").sum()),
                   n_shifted=int((f["class"] == "shifted").sum()),
                   n_trapped=int((f["class"] == "trapped").sum()),
                   n_incomplete=int((f["class"] == "incomplete").sum()),
                   gamma_pred=float(f.gamma_pred.iloc[0]))
        if len(comp):
            gam = comp.gamma.values
            row.update(gamma_median=float(np.median(gam)), gamma_q1=float(np.quantile(gam, 0.25)),
                       gamma_q3=float(np.quantile(gam, 0.75)), gamma_min=float(gam.min()),
                       gamma_max=float(gam.max()),
                       frac_within_03=float(np.mean(np.abs(gam - row["gamma_pred"]) <= 0.3)))
            ok = (row["frac_within_03"] >= 0.8 and abs(row["gamma_median"] - row["gamma_pred"]) <= 0.15
                  and len(comp) >= 10 if kind != "A_bal" else abs(row["gamma_median"] - row["gamma_pred"]) <= 0.15)
            ref = len(comp) >= 10 and abs(row["gamma_median"] - row["gamma_pred"]) > 0.5
            row["P5"] = "SUPPORTED" if ok else ("REFUTED" if ref else "INCONCLUSIVE")
        else:
            row["P5"] = "INCONCLUSIVE (no complete direction)"
        r = runs[(runs.cell == cell) & (runs.kind == kind)]
        rw = r[(r.log_eps >= WIN[0] - 1e-9) & (r.log_eps <= WIN[1] + 1e-9) & (r.outcome == "escaped")]
        row["greedy_frac_window"] = float(rw.first_is_greedy.astype(bool).mean()) if len(rw) else None
        row["P6"] = ("SUPPORTED" if row["greedy_frac_window"] is not None and row["greedy_frac_window"] >= 0.9
                     else "NOT SUPPORTED")
        if kind in ("A", "A_bal", "B"):
            both = r[r.t_red.notna() & r.t_esc.notna()]
            row["P3_runs"] = len(both)
            row["P3_max_rel"] = float(np.abs(both.rel_full_red).max()) if len(both) else None
            row["P3_first_agree"] = float((both.first_learned == both.first_red).mean()) if len(both) else None
            row["P3"] = ("holds" if len(both) and row["P3_max_rel"] < 1e-4 and row["P3_first_agree"] == 1.0
                         else "fails")
            rwi = rw[rw.ratio_iso.notna()]
            row["iso_frac_within_5pct"] = float((np.abs(rwi.ratio_iso - 1) <= 0.05).mean()) if len(rwi) else None
            row["iso_ratio_median"] = float(rwi.ratio_iso.median()) if len(rwi) else None
            row["iso_winner_frac_window"] = float((rw.first_learned == rw.iso_winner).mean()) if len(rw) else None
        for o in ("escaped", "floor_flagged", "trapped", "unfinished", "failed"):
            row[f"runs_{o}"] = int((r.outcome == o).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    RES.mkdir(exist_ok=True)
    cells = cells_table()
    cells.to_csv(RES / "cells.csv", index=False)
    curves, summ = static_tables()
    curves.to_csv(RES / "static_curves.csv", index=False)
    summ.to_csv(RES / "static_orders.csv", index=False)
    thm = thm92_table(summ)
    thm.to_csv(RES / "thm92.csv", index=False)
    runs = escape_runs()
    if len(runs):
        runs.to_csv(RES / "escape_runs.csv", index=False)
        fits = escape_fits(runs, summ)
        fits.to_csv(RES / "escape_fits.csv", index=False)
        dec = decisions(fits, runs)
        dec.to_csv(RES / "decisions.csv", index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 30)
        print(dec[["cell", "kind", "n_complete", "n_at_pred", "n_shifted", "n_trapped", "n_incomplete",
                   "gamma_pred", "gamma_median", "frac_within_03", "P5", "greedy_frac_window"]].to_string())
    print(summ[["cell", "space", "c_star", "order", "order_prev", "decision"]].to_string())
    if len(thm):
        print(thm[["cell", "parent", "k_star", "l_star", "parent_order", "formula", "order_full", "holds"]].to_string())


if __name__ == "__main__":
    main()
