"""Markdown tables for notes/route_escape.md Part II, printed from results/*.csv.

Usage: uv run python experiments/route_escape/tables.py > /tmp/tables.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_cell  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
STR = {c: str for c in ("greedy", "first_learned", "iso_winner", "esc_pattern", "first_red", "missing",
                        "parent", "J", "J_end", "forgotten", "first_at_smallest")}
DYN = ["EX87_J01", "TWO23", "S3", "U3HT", "TWO34", "TIE3", "S4", "U4B8", "EX87_J0", "U5B8", "PS2"]
WIN = (-4.5, -3.0)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.1e}"
        return f"{v:.3f}"
    return str(v)


def static():
    o = pd.read_csv(RES / "static_orders.csv", dtype=STR)
    rows = []
    for cell in o.cell.unique():
        m = o[(o.cell == cell) & (o.space == "Mstar")].iloc[0]
        f = o[(o.cell == cell) & (o.space == "full")].iloc[0]
        c = get_cell(cell)
        rows.append(dict(cell=cell, tree=c.tree_name, c_star=c.c_star, order_Mstar=m.order,
                         order_full=f.order, decision=f.decision, hess_lam_min=f.hess_lam_min,
                         hess_n_neg=int(f.hess_n_neg), minimiser_off_M=f.w_offM_smallest))
    print("### Static orders\n")
    print(md(pd.DataFrame(rows)))
    t = pd.read_csv(RES / "thm92.csv", dtype=STR)
    print("\n### Theorem 9.2 family\n")
    print(md(t[["cell", "parent", "parent_weight", "residuals", "k_star", "l_star", "parent_order", "formula",
                "c_star", "order_full", "holds"]]))


def decisions():
    d = pd.read_csv(RES / "decisions.csv")
    d["gamma [q1, q3]"] = [f"{m:.3f} [{a:.3f}, {b:.3f}]" if not np.isnan(m) else ""
                           for m, a, b in zip(d.gamma_median, d.gamma_q1, d.gamma_q3)]
    d["order"] = d.cell.map({c: i for i, c in enumerate(DYN)})
    d = d.sort_values(["order", "kind"])
    print("\n### Escape exponents per cell and kind (fit window eps = 1e-4.5 ... 1e-3)\n")
    print(md(d[["cell", "kind", "gamma_pred", "n_complete", "n_at_pred", "n_shifted", "n_trapped", "n_incomplete",
                "gamma [q1, q3]", "frac_within_03", "P5", "greedy_frac_window", "P6"]]))
    m = d[d.kind.isin(["A", "A_bal", "B", "Cp"])]
    print("\n### Inside M*: full network vs reduced flow (P3) and the isolated-route formula (P4)\n")
    print(md(m[["cell", "kind", "P3_runs", "P3_max_rel", "P3_first_agree", "P3", "iso_frac_within_5pct",
                "iso_ratio_median", "iso_winner_frac_window"]]))
    print("\n### Run inventory (all eps)\n")
    print(md(d[["cell", "kind", "runs_escaped", "runs_floor_flagged", "runs_trapped", "runs_unfinished",
                "runs_failed"]]))


def balanced():
    r = pd.read_csv(RES / "escape_runs.csv", dtype=STR)
    b = r[r.kind == "A_bal"].copy()
    rows = []
    for cell in DYN:
        s = b[(b.cell == cell) & (b.outcome == "escaped")].sort_values("log_eps")
        if not len(s):
            continue
        c = get_cell(cell)
        g = c.greedy()
        sm = s.iloc[0]
        row = dict(cell=cell, depth=c.depth(g), sA=c.weights[g] * c.A(g), smallest_eps=sm.eps,
                   t_esc=sm.t_esc, t_iso=sm.t_iso, ratio_iso=sm.ratio_iso, t_616=sm.t_616,
                   ratio_616=sm.ratio_616)
        w = s[(s.log_eps >= WIN[0] - 1e-9) & (s.log_eps <= WIN[1] + 1e-9)]
        if c.depth(g) == 2 and len(w) >= 3:
            x = np.log(1 / w.eps.values)
            row["log_slope"] = float(np.polyfit(x, w.t_esc.values, 1)[0])
            row["log_slope_pred"] = 1 / (c.weights[g] * c.A(g))
        rows.append(row)
    print("\n### Balanced starts (A_bal) against eq. 6.16\n")
    print(md(pd.DataFrame(rows)))


def first_learned():
    r = pd.read_csv(RES / "escape_runs.csv", dtype=STR)
    rows = []
    for cell in ("TWO23", "TWO34", "TIE3", "PS2"):
        for kind in ("A", "B", "C"):
            s = r[(r.cell == cell) & (r.kind == kind) & r.first_learned.notna()]
            for le, g in s.groupby("log_eps"):
                rows.append(dict(cell=cell, kind=kind, log_eps=le,
                                 greedy=g.greedy.iloc[0], frac_greedy=g.first_is_greedy.astype(bool).mean(),
                                 first=" ".join(f"{k}:{v}" for k, v in g.first_learned.value_counts().items())))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["cell", "kind"], columns="log_eps", values="frac_greedy")
    piv = piv[sorted(piv.columns, reverse=True)]
    print("\n### Fraction of runs that learn the greedy pattern first, by log10 eps\n")
    piv = piv.reset_index()
    piv.columns = [str(c) for c in piv.columns]
    print(md(piv))


def shifted():
    f = pd.read_csv(RES / "escape_fits.csv", dtype=STR)
    s = f[f["class"].isin(["shifted", "trapped", "incomplete"]) & (f.kind.isin(["C", "Cp", "A", "B"]))]
    cols = ["cell", "kind", "seed", "class", "gamma", "dev", "n_escaped", "n_flagged", "n_trapped", "n_unfinished",
            "route_depletion_smallest", "route_depleted_node_smallest", "P_sw_smallest"]
    print("\n### Directions not at the prediction (all kinds)\n")
    print(md(s[[c for c in cols if c in s.columns]]))


if __name__ == "__main__":
    static()
    decisions()
    balanced()
    first_learned()
    shifted()
