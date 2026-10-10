"""Markdown tables for notes/theory_trap.md section 7 from results/locality_pilot_joined.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).parent / "results"


def cell(g):
    return f"{int(g.bad.sum())}/{len(g)}" if len(g) else "-"


def main(by):
    df = pd.read_csv(RES / "locality_pilot_joined.csv.gz", dtype={"S": str})
    df = df[df.obs.isin(["stall", "escaped", "dead", "pinned"]) & (df.k >= 4) & (df.q == 0)].copy()
    df["bad"] = df.obs.isin(["stall", "dead", "pinned"])
    lines = ["| " + " | ".join(by) + " | pred. dead | pred. pinned | pred. alive |",
             "|" + "---|" * (len(by) + 3)]
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        cells = [cell(g[g.verdict == v]) for v in ("dead", "pinned", "alive")]
        lines.append("| " + " | ".join(str(x) for x in key) + " | " + " | ".join(cells) + " |")
    print("\n".join(lines))




def s2s_table():
    """s2s_core: init-only phase-2 verdict vs observed class, per (tree, chi, teacher, init)."""
    df = pd.read_csv(RES / "s2s_predictions.csv")
    df = df[df.obs.isin(["trapped", "stalled", "complete"])]
    lines = ["| tree | chi | teacher | init | pred. dead: t/s/c | pred. pinned: t/s/c | pred. alive: t/s/c |",
             "|---|---|---|---|---|---|---|"]
    for key, g in df.groupby(["tree", "chi", "teacher", "init"]):
        cells = []
        for v in ("dead", "pinned", "alive"):
            h = g[g.verdict == v]
            cells.append("/".join(str(int((h.obs == o).sum())) for o in ("trapped", "stalled", "complete"))
                         if len(h) else "-")
        lines.append("| " + " | ".join(str(x) for x in key) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "s2s":
        print(s2s_table())
    else:
        main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["source", "tree", "chi", "start"])
