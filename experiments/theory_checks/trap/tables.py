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
    df = df[df.obs.isin(["stall", "escaped", "dead", "pinned"]) & (df.k >= 4)].copy()
    df["bad"] = df.obs.isin(["stall", "dead", "pinned"])
    lines = ["| " + " | ".join(by) + " | pred. dead | pred. pinned | pred. alive |",
             "|" + "---|" * (len(by) + 3)]
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        cells = [cell(g[g.verdict == v]) for v in ("dead", "pinned", "alive")]
        lines.append("| " + " | ".join(str(x) for x in key) + " | " + " | ".join(cells) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["source", "tree", "chi", "start"])
