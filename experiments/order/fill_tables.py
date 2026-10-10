"""Fill the generated tables of notes/order.md from results/*.csv (placeholders MULTI3TABLE and
NOSTOP_SUMMARY, or the blocks they produced, which are replaced again on re-runs).

Usage: uv run python experiments/order/fill_tables.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RES  # noqa: E402

NOTES = Path(__file__).parents[2] / "notes" / "order.md"
M_OF = {"binary4": "2", "mps4": "4", "mps6": "6", "binary8": "4"}


def multi3_table():
    s = pd.read_csv(RES / "multi3_summary.csv")
    lines = []
    for r in s.itertuples():
        lines.append(
            f"| {r.tree} ({M_OF[r.tree]}), sigma {r.sigma:g} | {r.rho} | {r.n_used} of {r.n} "
            f"({r.n_stalled} stalled, {r.n_tmax} unfinished) | {r.p_greedy_order:.3f} +- "
            f"{r.p_greedy_order_se:.3f} vs {r.p_greedy_order_model:.3f} | {r.z_vs_model:+.2f} | "
            f"{r.kendall_greedy:.2f} +- {r.kendall_greedy_se:.2f} vs {r.kendall_greedy_model:.2f} | "
            f"{r.race_first_agree:.2f} / {r.race_exact_order:.2f} |")
    return "\n".join(lines)


def nostop_summary():
    s = pd.read_csv(RES / "multi_nostop_summary.csv")
    parts = []
    for r in s.itertuples():
        parts.append(f"{r.tree}: {r.n_converged} of {r.n} runs learn all four components; learned at "
                     f"t = 1e9 (count:runs) {r.learned_at_stop}; unlearned component (rank:runs) "
                     f"{r.unlearned}")
    return "; ".join(parts) + "."


HEADER = "| tree (m), sigma | rho | used (excluded) |"


def main():
    lines = NOTES.read_text().split("\n")
    # R = 3 table: the rows after the header and separator, up to the next blank line
    h = next(i for i, l in enumerate(lines) if l.startswith(HEADER))
    end = next(i for i in range(h + 2, len(lines)) if not lines[i].startswith("|"))
    if (RES / "multi3_summary.csv").exists():
        lines[h + 2:end] = multi3_table().split("\n")
    text = "\n".join(lines)
    if (RES / "multi_nostop_summary.csv").exists():
        text = re.sub(r"(`results/multi_nostop_summary.csv`\): ).*?(\n- How many)",
                      lambda m: m.group(1) + nostop_summary() + m.group(2), text, flags=re.S)
    NOTES.write_text(text)


if __name__ == "__main__":
    main()
