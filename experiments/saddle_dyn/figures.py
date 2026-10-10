"""Figures from the analysis CSVs (run analyze.py first).

Usage: uv run python experiments/saddle_dyn/figures.py [--res experiments/saddle_dyn/results]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analyze import CONTROL, HB_WINDOW, TEST, gam  # noqa: E402
from runs import CELLS  # noqa: E402
from ttn.plotting import INK, MUTED, TEXT_WIDTH, palette, savefig, set_style  # noqa: E402

TYPE_COLORS = dict(zip(["split", "mixed", "route"], palette(3)))


def ref_lines(ax, x0, t0, go, gr, xr):
    xs = np.array(xr)
    ax.plot(xs, t0 * (xs / x0) ** (-go), color=INK, lw=1.0, zorder=1)
    if gr != go:
        ax.plot(xs, t0 * (xs / x0) ** (-gr), color=MUTED, lw=1.0, ls="--", zorder=1)


def escape_panels(df, cells, x_col, t_col, windows, xlabel, ylabel, path, title_fn):
    n = len(cells)
    ncol = 4 if n > 4 else n
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXT_WIDTH, 1.75 * nrow + 0.4), squeeze=False)
    for ax, cell in zip(axes.flat, cells):
        d = df[(df.cell == cell) & df[t_col].notna()]
        lo, hi = windows[cell]
        ax.axvspan(10**lo, 10**hi, color="#f0efec", zorder=0, lw=0)
        for seed, g in d.groupby("seed"):
            g = g.sort_values(x_col)
            x = 10 ** g[x_col].to_numpy()
            ax.plot(x, g[t_col], color=MUTED, lw=0.5, alpha=0.6, zorder=2)
            for kind, c in TYPE_COLORS.items():
                m = (g.esc_type == kind).to_numpy()
                ax.scatter(x[m], g[t_col].to_numpy()[m], s=7, color=c, zorder=3, lw=0)
            fl = g.flagged.to_numpy().astype(bool)
            ax.scatter(x[fl], g[t_col].to_numpy()[fl], s=18, facecolors="none", edgecolors=INK, lw=0.6, zorder=4)
        go, gr = gam(cell)
        w = d[np.isclose(d[x_col], lo)]
        if len(w):
            t0 = float(np.exp(np.log(w[t_col].astype(float)).median()))
            ref_lines(ax, 10**lo, t0, go, gr, [10**lo, 10**hi])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title_fn(cell), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8)
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=4, label=f"{k} escape") for k, c in TYPE_COLORS.items()]
    handles += [Line2D([], [], color=INK, lw=1, label="theorem order: slope -(p-2)"),
                Line2D([], [], color=MUTED, lw=1, ls="--", label="route law: slope -(k*-2)"),
                Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=5, label="float64-floor flag")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, path)
    plt.close(fig)


def local_panels(loc, cells, windows, xlabel, path, title_fn):
    n = len(cells)
    ncol = 4 if n > 4 else n
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXT_WIDTH, 1.6 * nrow + 0.4), squeeze=False, sharex=False)
    for ax, cell in zip(axes.flat, cells):
        d = loc[loc.cell == cell].copy()
        lo, hi = windows[cell]
        ax.axvspan(10**lo, 10**hi, color="#f0efec", zorder=0, lw=0)
        d["xm"] = 10 ** (0.5 * (d.x_hi + d.x_lo))
        ax.scatter(d.xm, d.gamma_loc, s=5, color=palette(1)[0], alpha=0.5, lw=0, zorder=2)
        m = d.groupby("xm").gamma_loc.median()
        ax.plot(m.index, m.values, color=INK, lw=1.0, zorder=3)
        go, gr = gam(cell)
        ax.axhline(go, color=INK, lw=0.8, ls=":", zorder=1)
        if gr != go:
            ax.axhline(gr, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_ylim(min(go, gr) - 1.5, max(go, gr) + 1.5)
        ax.set_title(title_fn(cell), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("local exponent", fontsize=8)
    handles = [Line2D([], [], marker="o", ls="", color=palette(1)[0], ms=3, label="seed (consecutive grid pair)"),
               Line2D([], [], color=INK, lw=1, label="median over seeds"),
               Line2D([], [], color=INK, lw=0.8, ls=":", label="p - 2 (theorem order)"),
               Line2D([], [], color=MUTED, lw=0.8, ls="--", label="k* - 2 (route law)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, path)
    plt.close(fig)


def ha_title(cell):
    c = CELLS[cell]
    go, gr = gam(cell)
    return f"{cell}: p = {c['order']}, k* = {gr + 2}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    res = Path(args.res)
    set_style()
    runs = pd.read_csv(res / "ha_runs.csv")
    main = runs[runs.tag.isna() | (runs.tag == "")].copy()
    main["t_plot"] = main.t_esc.where(main.escaped)
    cells = [c for c in TEST + CONTROL if c in set(main.cell)]
    wins = {c: tuple(CELLS[c]["window"]) for c in cells}
    escape_panels(main, cells, "lx", "t_plot", wins, r"perturbation $\varepsilon$ (B8W*: $\sigma$)",
                  r"$t_{\rm esc}$", res / "ha_escape", ha_title)
    loc = pd.read_csv(res / "ha_local_exponents.csv")
    local_panels(loc, cells, wins, r"$\varepsilon$ (B8W*: $\sigma$)", res / "ha_local", ha_title)
    if (res / "hb_runs.csv").exists():
        hb = pd.read_csv(res / "hb_runs.csv")
        hb["t_plot"] = hb.dt.where((hb.status == "escaped") & (hb.dt > 0))
        hcells = [c for c in ["PS08", "PS06", "B8W"] if c in set(hb.cell)]
        hw = {c: HB_WINDOW[c] for c in hcells}
        escape_panels(hb.rename(columns={"ls": "lx"}), hcells, "lx", "t_plot", hw, r"init scale $\sigma$",
                      r"plateau $\Delta t$", res / "hb_dt", lambda c: f"{c} from small init: p = {CELLS[c]['order']}")
        hloc = pd.read_csv(res / "hb_local_exponents.csv")
        local_panels(hloc, hcells, hw, r"$\sigma$", res / "hb_local", lambda c: f"{c} from small init")
