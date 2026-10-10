"""Markdown tables for notes/saddle_dyn.md Part II, from the analysis CSVs (run analyze.py first).

Usage: uv run python experiments/saddle_dyn/tables.py [--res experiments/saddle_dyn/results]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def pm(m, s, nd=3):
    return "-" if not math.isfinite(m) else (f"{m:.{nd}f}" + ("" if not math.isfinite(s) else f" ± {s:.{nd}f}"))


def ha_table(dec):
    rows = ["| cell | p, k* | pred. gamma (alt.) | fitted seeds | gamma mean ± SE | median [min, max] | within 0.3 | RSS pred / alt | local exp. (2 smallest pairs) | decision |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in dec.iterrows():
        go, gr = int(r.k_order) - 2 if "k_order" in r else None, None
        rows.append(
            f"| {r.cell} | {r.order}, {r.k_star} | {r.pred_label} | {r.n_fitted} | {pm(r.get('gamma_mean', float('nan')), r.get('gamma_se', float('nan')))} | "
            f"{f(r.get('gamma_median', float('nan')))} [{f(r.get('gamma_min', float('nan')), 2)}, {f(r.get('gamma_max', float('nan')), 2)}] | "
            f"{f(r.get('frac_within_0p3', float('nan')), 2)} | {r.rss_label} | {pm(r.get('loc1_mean', float('nan')), r.get('loc1_se', float('nan')))}, "
            f"{pm(r.get('loc0_mean', float('nan')), r.get('loc0_se', float('nan')))} | {r.decision} |")
    return "\n".join(rows)


def inventory(dec, kind="HA"):
    if kind == "HA":
        rows = ["| cell | runs | escaped | trapped | failed | wall | floor-flagged | window runs: escaped / trapped / failed | window escapes: split / mixed / route | q forgotten after escape |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in dec.iterrows():
            rows.append(f"| {r.cell} | {r.n_runs} | {r.n_escaped} | {r.n_trapped} | {int(r.get('n_failed', 0))} | {r.n_wall} | {r.n_flagged} | "
                        f"{r.n_window_escaped} / {r.n_window_trapped} / {r.n_window_failed} | {r.n_split} / {r.n_mixed} / {r.n_route} | "
                        f"{f(r.frac_q_forgotten, 2)} |")
        return "\n".join(rows)
    return ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    res = Path(args.res)
    runs = pd.read_csv(res / "ha_runs.csv")
    dec = pd.read_csv(res / "ha_decisions.csv")
    main = runs[runs.tag.isna() | (runs.tag == "")]
    info = main.groupby("cell").agg(order=("cell", lambda c: None)).reset_index()
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from analyze import gam  # noqa: E402
    from runs import CELLS  # noqa: E402

    dec["order"] = [CELLS[c]["order"] for c in dec.cell]
    dec["k_star"] = [gam(c)[1] + 2 for c in dec.cell]
    dec["pred_label"] = [f"{gam(c)[0]} ({gam(c)[1]})" if gam(c)[0] != gam(c)[1] else
                         f"{gam(c)[0]} ({gam(c)[0] - 1}{', 4' if c == 'PS06' else ''})" for c in dec.cell]

    def rss_label(r):
        if not math.isfinite(r.get("rss_ord", float("nan"))):
            return "-"
        go, gr = gam(r.cell)
        if go != gr:
            return f"{r.rss_ord:.3g} / {r.rss_route:.3g}"
        alts = [r.rss_ord_minus1] + ([r.rss_four] if r.cell == "PS06" else [])
        return f"{r.rss_route:.3g} / " + ", ".join(f"{a:.3g}" for a in alts)

    dec["rss_label"] = [rss_label(r) for _, r in dec.iterrows()]
    dec["n_failed"] = [int(main[(main.cell == c)].status.astype(str).str.startswith("failed").sum()) for c in dec.cell]
    print("### H-A decisions\n")
    print(ha_table(dec))
    print("\n### H-A run inventory\n")
    print(inventory(dec))
    cr = res / "ha_crossover.csv"
    if cr.exists():
        c = pd.read_csv(cr)
        print("\n### Two-mechanism crossover (descriptive)\n")
        print("| cell | seeds | median log10 eps_c | mean ± SE | RSS two-mech / single power law (summed) |\n|---|---|---|---|---|")
        for cell, g in c.groupby("cell", sort=False):
            print(f"| {cell} | {len(g)} | {g.log10_eps_c.median():.2f}{' (above the grid: no crossover in range)' if g.log10_eps_c.median() > 0 else ''} | {g.log10_eps_c.mean():.2f} ± {g.log10_eps_c.std(ddof=1) / math.sqrt(len(g)):.2f} | "
                  f"{g.rss_two_mech.sum():.3g} / {g.rss_power.sum():.3g} |")
    if "above_mean" in dec:
        for _, r in dec[dec.cell.isin(["B8W", "B8W05"])].iterrows():
            print(f"\n{r.cell} H-A local exponent over pairs with sigma >= 0.1: {pm(r.above_mean, r.get('above_se', float('nan')))} (n = {r.get('above_n')})")
    chk = pd.read_csv(res / "ha_checks.csv")
    print("\n### Checks\n")
    for tag, g in chk.groupby("tag"):
        v = g.rel_diff.dropna()
        print(f"- {tag}: {len(g)} reruns, {len(v)} comparable; max |rel diff t_esc| = {v.abs().max():.2e}; non-comparable: {len(g) - len(v)}")
    hb = res / "hb_decisions.csv"
    if hb.exists():
        h = pd.read_csv(hb)
        print("\n### H-B decisions\n")
        print("| cell | seeds | visiting (any sigma) | fitted | gamma mean ± SE | median [min, max] | pred | RSS pred / alt | local exp. (2 smallest pairs) | decision |\n|---|---|---|---|---|---|---|---|---|---|")
        for _, r in h.iterrows():
            go, gr = gam(r.cell)
            rl = f"{r.rss_ord:.3g} / {r.rss_route:.3g}" if go != gr and math.isfinite(r.get('rss_ord', float('nan'))) else (
                f"{r.rss_route:.3g} / {r.rss_ord_minus1:.3g}, {r.rss_four:.3g}" if math.isfinite(r.get('rss_route', float('nan'))) else "-")
            print(f"| {r.cell} | {r.n_seeds} | {r.n_seeds_visiting_any} | {r.n_fitted} | {pm(r.get('gamma_mean', float('nan')), r.get('gamma_se', float('nan')))} | "
                  f"{f(r.get('gamma_median', float('nan')))} [{f(r.get('gamma_min', float('nan')), 2)}, {f(r.get('gamma_max', float('nan')), 2)}] | "
                  f"{r.pred} | {rl} | {pm(r.get('loc1_mean', float('nan')), r.get('loc1_se', float('nan')))}, {pm(r.get('loc0_mean', float('nan')), r.get('loc0_se', float('nan')))} | {r.decision} |")
        runs_hb = pd.read_csv(res / "hb_runs.csv")
        print("\nH-B run status counts:\n")
        ct = pd.crosstab(runs_hb.cell, runs_hb.status)
        print("| cell | " + " | ".join(ct.columns) + " |\n|---|" + "---|" * len(ct.columns))
        for cell, row in ct.iterrows():
            print(f"| {cell} | " + " | ".join(str(v) for v in row.values) + " |")
