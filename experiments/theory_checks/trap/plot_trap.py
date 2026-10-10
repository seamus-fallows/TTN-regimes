"""Figures for notes/theory_trap.md (style and palette from ttn.plotting).

fig_criteria: fraction of runs that stall, per predicted class, for the three criteria (this note's
C_u criterion, R12's Q + |y_route|^2, Q alone); locality two-component runs with k >= 4, fan-in init
(saddle and small starts).
fig_probability: Monte Carlo trap probabilities (sigma -> 0 verdicts) against the measured stall
fractions of the same cases.

usage: PYTHONPATH=. python experiments/theory_checks/trap/plot_trap.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ttn import plotting as tp

RES = Path(__file__).parent / "results"
CRIT = [
    ("verdict", "this note ($C_u$)"),
    ("verdict_r12", r"R12 ($Q_{\nu\nu}+|y_{\rm route}|^2$)"),
    ("verdict_q", r"$Q_{\nu\nu}$ alone"),
]
CLASSES = ["dead", "pinned", "alive"]


def fig_criteria():
    df = pd.read_csv(RES / "locality_pilot_joined.csv.gz", dtype={"S": str})
    d = df[
        (df.source == "locality")
        & df.start.isin(["saddle", "small"])
        & (df.k >= 4) & (df.q == 0)
        & df.obs.isin(["stall", "escaped", "dead", "pinned"])
    ].copy()
    d["bad"] = d.obs.isin(["stall", "dead", "pinned"])
    fig, ax = tp.subplots(1, 1, width=0.62 * tp.TEXT_WIDTH, panel_aspect=0.68)
    colors = tp.palette(len(CRIT))
    width = 0.26
    for i, (col, label) in enumerate(CRIT):
        for j, cls in enumerate(CLASSES):
            g = d[d[col] == cls]
            if not len(g):
                continue
            x = j + (i - 1) * width
            ax.bar(x, g.bad.mean(), width * 0.92, color=colors[i], label=label if j == 0 else None)
            ax.text(
                x,
                g.bad.mean() + 0.02,
                f"{int(g.bad.sum())}/{len(g)}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=tp.INK_SECONDARY,
                rotation=90,
            )
    ax.set_xticks(range(len(CLASSES)), [f"predicted {c}" for c in CLASSES])
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("fraction of runs that stall")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    tp.savefig(fig, RES / "fig_criteria")


def fig_probability():
    mc = pd.read_csv(RES / "montecarlo_summary.csv", dtype={"S": str})
    obs = pd.read_csv(RES / "locality_pilot_joined.csv.gz", dtype={"S": str})
    obs = obs[
        (obs.source == "locality") & obs.obs.isin(["stall", "escaped", "dead", "pinned"])
    ].copy()
    obs["bad"] = obs.obs.isin(["stall", "dead", "pinned"])
    obs["kind"] = obs.tree.str.extract(r"^([a-z]+)")[0].where(obs.tree != "midroot6", "midroot6")
    rows = []
    for r in mc.itertuples():
        g = obs[(obs.kind == r.kind) & (obs.S == r.S) & (obs.start == r.start) & (obs.chi == r.chi)]
        if r.kind != "midroot6":
            g = g[g.tree == f"{r.kind}{int(r.n)}"]
        if len(g):
            rows.append(
                dict(
                    label=f"{r.kind}{'' if r.kind == 'midroot6' else int(r.n)} {{{','.join(r.S)}}} "
                    f"$\\chi={r.chi}$ {r.start}",
                    k=r.k,
                    pd=r.P_dead,
                    pdp=r.P_dead + r.P_pinned,
                    obs=g.bad.mean(),
                    n=len(g),
                )
            )
    t = pd.DataFrame(rows).sort_values(["k", "label"]).reset_index(drop=True)
    fig, ax = tp.subplots(1, 1, width=tp.TEXT_WIDTH, panel_aspect=0.5)
    c = tp.palette(3)
    y = np.arange(len(t))
    big = (t.k >= 5).to_numpy()  # at k = 4 a pinned end is O(sigma): pinned runs escape normally
    ax.hlines(y[big], t.pd[big], t.pdp[big], color=tp.GRID, lw=3, zorder=1)
    ax.plot(t.pd, y, "o", color=c[0], ms=4, label="predicted P(dead)", zorder=2)
    ax.plot(t.pdp[big], y[big], "o", color=c[1], ms=4, label="predicted P(dead or pinned), k >= 5", zorder=2)
    ax.plot(t.obs, y, "D", color=c[2], ms=4, label="measured stall fraction", zorder=3)
    ax.set_yticks(y, [f"{lb} (k={int(k)})" for lb, k in zip(t.label, t.k)], fontsize=6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("probability per run")
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    tp.savefig(fig, RES / "fig_probability")
    t.to_csv(RES / "probability_vs_measured.csv", index=False)


if __name__ == "__main__":
    tp.set_style()
    fig_criteria()
    if (RES / "montecarlo_summary.csv").exists():
        fig_probability()
