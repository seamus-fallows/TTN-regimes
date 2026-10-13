"""Markdown tables for the note and the report, from the CSVs in results/ (no new numbers).

usage: uv run python experiments/route_generic/tables.py <results_dir>   -> <results_dir>/summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = {"pattern": "pattern (control)", "cp03": "CP, overlap 0.3", "cp06": "CP, overlap 0.6", "ttn": "random TTN",
         "mnist": "MNIST"}
ORDER = list(NAMES)


def q(row, name, fmt="{:.2f}"):
    m, a, b = row[f"{name}_median"], row[f"{name}_q25"], row[f"{name}_q75"]
    if not np.isfinite(m):
        return "-"
    return f"{fmt.format(m)} [{fmt.format(a)}, {fmt.format(b)}]"


def pct(x):
    return "-" if not np.isfinite(x) else f"{100 * x:.0f}%"


def main():
    res = Path(sys.argv[1])
    sm = pd.read_csv(res / "summary.csv")
    runs = pd.read_csv(res / "runs.csv")
    ev = pd.read_csv(res / "events.csv")
    ex = pd.read_csv(res / "exponents.csv")
    out = []
    sm["o"] = sm["target"].map({t: i for i, t in enumerate(ORDER)})
    sm = sm.sort_values(["o", "tree", "events", "k_class"])

    out.append("### Later events (all events after the escape from the origin), per target and tree\n")
    out.append("Medians [quartiles] over fitted series (seed x event). d_eff: growth-law depth; p: duration ~ "
               "sigma^-p; prediction d_eff = k_region, p = k_region - 2.\n")
    out.append("| target | tree | events | series | k_region | d_eff | p | d_eff within 0.3 of k_region | "
               "p within 0.3 of k_region - 2 | p within 0.3 of d_eff - 2 | log law preferred | verdict |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in sm[(sm["events"] == "later")].iterrows():
        out.append(f"| {NAMES[r['target']]} | {r['tree']} | {r['k_class']} | {r['n_series']} | "
                   f"{q(r, 'k_region', '{:.0f}')} | {q(r, 'eff_depth')} | {q(r, 'p')} | "
                   f"{pct(r['frac_dd_within_0.3'])} | {pct(r['frac_dp_within_0.3'])} | "
                   f"{pct(r['frac_p_vs_depth_within_0.3'])} | {pct(r['frac_log_preferred'])} | {r['verdict']} |")
    out.append("")
    out.append("### Event 0 (escape from the origin; p = K - 2 is an exact identity: a check of the estimators)\n")
    out.append("| target | tree | series | k_region | d_eff | p |")
    out.append("|---|---|---|---|---|---|")
    for _, r in sm[(sm["events"] == "event0")].iterrows():
        out.append(f"| {NAMES[r['target']]} | {r['tree']} | {r['n_series']} | {q(r, 'k_region', '{:.0f}')} | "
                   f"{q(r, 'eff_depth', '{:.3f}')} | {q(r, 'p', '{:.3f}')} |")
    out.append("")
    out.append("### Runs and counts\n")
    out.append("| target | tree | runs | converged / completed | trapped | runs with a relearned rise | "
               "floor-flagged events (runs) | series fitted / all |")
    out.append("|---|---|---|---|---|---|---|---|")
    for (t, tr), g in sorted(runs.groupby(["target", "tree"]), key=lambda x: (ORDER.index(x[0][0]), x[0][1])):
        e = ev[(ev["target"] == t) & (ev["tree"] == tr)]
        x = ex[(ex["target"] == t) & (ex["tree"] == tr)]
        ok = g["status"].isin(["converged", "completed"]).sum()
        out.append(f"| {NAMES[t]} | {tr} | {len(g)} | {ok} | {int(g['trapped'].sum())} | "
                   f"{int((g['n_relearn'] > 0).sum())} | {int(e['floor_flag'].sum())} "
                   f"({int((g['n_floor_flagged_events'] > 0).sum())}) | {int(x['fitted'].sum())} / {len(x)} |")
    (res / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
