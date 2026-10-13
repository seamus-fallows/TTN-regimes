"""Draft figures of the route-escape experiment from results/*.csv (run analyze.py first).

One message per figure, stated in its title; axes in words with units; predictions as labelled
reference lines or markers; at most six colours (ttn.plotting palette).

Usage: uv run python experiments/route_escape/figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, get_cell  # noqa: E402

from ttn.plotting import INK, MUTED, palette, savefig, set_style  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DYN = ["EX87_J01", "TWO23", "S3", "U3HT", "TWO34", "TIE3", "S4", "U4B8", "EX87_J0", "U5B8", "PS2"]
BLUE, ORANGE, GREEN, YELLOW, PINK, DGREEN = palette(6)
KIND_COLOR = {"A": BLUE, "A_bal": INK, "B": ORANGE, "C": GREEN}
KIND_LABEL = {"A": "A: inside M*, positive", "A_bal": "A_bal: inside M*, balanced",
              "B": "B: inside M*, random signs", "C": "C: Gaussian on all parameters"}


def big():
    set_style("sans")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                         "figure.titlesize": 13})


def cell_label(name):
    c = get_cell(name)
    return f"{name}\n{c.tree_name}, c*={c.c_star}"


# ---- 1. static orders -----------------------------------------------------------------------


def fig_static(orders, thm):
    cells = DYN + ["T92a", "T92b", "T92c", "T92d"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(cells))
    for i, name in enumerate(cells):
        c = get_cell(name)
        ax.plot([i - 0.32, i + 0.32], [c.c_star] * 2, color=MUTED, lw=2.5, solid_capstyle="butt",
                label="route depth c*_J (Thm 8.4)" if i == 0 else None, zorder=1)
        for space, col, dx, mk in (("Mstar", BLUE, -0.12, "o"), ("full", ORANGE, 0.12, "s")):
            r = orders[(orders.cell == name) & (orders.space == space)]
            if len(r) and pd.notna(r.order.iloc[0]):
                ax.plot(i + dx, r.order.iloc[0], mk, color=col, ms=9, mec="white", mew=1.2, zorder=3,
                        label=("measured inside M*" if space == "Mstar" else "measured in the full space")
                        if i == 0 else None)
        if c.cfg.get("parents") is not None:
            ax.plot(i + 0.12, c.thm92()["order"], "D", mfc="none", mec=INK, ms=13, mew=1.3, zorder=4,
                    label="Thm 9.2 formula (family cells)" if name == "PS2" else None)
    ax.set_xticks(x, [cell_label(n) for n in cells], rotation=0, fontsize=8)
    ax.set_ylabel("saddle order\n(power of the first loss decrease)")
    ax.set_ylim(1.5, 6.5)
    ax.axvline(len(DYN) - 0.5, color=MUTED, lw=0.8, ls=":")
    ax.text(len(DYN) - 0.4, 6.3, "static-only Thm 9.2 family", fontsize=9, color=MUTED, va="top")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_title("Static saddle order: inside M* it equals the route depth c*_J in every cell; "
                 "in the full space it drops below c*_J exactly where a spectral shortcut is predicted",
                 fontsize=11)
    savefig(fig, RES / "fig1_static_orders")
    plt.close(fig)


# ---- 2. inside M*: full = reduced = isolated route ------------------------------------------


def fig_inside(runs):
    r = runs[runs.kind.isin(["A", "A_bal", "B"]) & (runs.outcome == "escaped") & runs.ratio_iso.notna()]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    for k in ("A", "B", "A_bal"):
        s = r[r.kind == k]
        ax.loglog(s.t_iso_first, s.t_esc, "o" if k != "A_bal" else "*", ms=3 if k != "A_bal" else 8,
                  color=KIND_COLOR[k], alpha=0.5 if k != "A_bal" else 1, label=KIND_LABEL[k], mec="none")
    lo, hi = r.t_esc.min() / 3, r.t_esc.max() * 3
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--", label="prediction (equal times)")
    ax.set_xlabel("predicted escape time: isolated route, eq. 6.14-6.16\n(gradient-flow time units)")
    ax.set_ylabel("measured escape time of the full network\n(first L < L_J - 0.001)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Every start inside M* (A, B): measured = predicted")
    ax = axs[1]
    full_red = np.abs(r.rel_full_red.astype(float))
    iso = np.abs(r.ratio_iso.astype(float) - 1)
    bins = np.logspace(-12, 1, 40)
    ax.hist(np.clip(full_red, 1e-12, None), bins=bins, color=BLUE, alpha=0.8,
            label="full network vs reduced flow on M*")
    ax.hist(np.clip(iso, 1e-12, None), bins=bins, color=ORANGE, alpha=0.6,
            label="full network vs isolated-route formula")
    ax.axvline(1e-4, color=MUTED, ls="--", lw=1)
    ax.axvline(0.05, color=MUTED, ls=":", lw=1)
    ax.text(1e-4, ax.get_ylim()[1] * 0.95, " P3 bound 1e-4", color=MUTED, fontsize=9, va="top")
    ax.text(0.05, ax.get_ylim()[1] * 0.80, " P4 bound 5 %", color=MUTED, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("relative difference of escape times |t_a / t_b - 1|")
    ax.set_ylabel("number of runs")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("The full network reproduces the reduced flow to ~1e-8")
    savefig(fig, RES / "fig2_inside_Mstar")
    plt.close(fig)


# ---- 3. per-direction exponents ------------------------------------------------------------


def fig_exponents(fits):
    fig, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.default_rng(0)
    for i, name in enumerate(DYN):
        f = fits[fits.cell == name]
        pred_M = get_cell(name).c_star - 2
        predC = f[f.kind == "C"].gamma_pred.iloc[0] if (f.kind == "C").any() else pred_M
        ax.plot([i - 0.4, i + 0.4], [pred_M] * 2, color=MUTED, lw=2.5, solid_capstyle="butt",
                label="prediction c*_J - 2 (inside M*)" if i == 0 else None)
        if predC != pred_M:
            ax.plot([i + 0.05, i + 0.4], [predC] * 2, color=INK, lw=2.5, ls=(0, (2, 1)),
                    label="prediction for C: static full-space order - 2" if name == "PS2" else None)
        for k, dx in (("A", -0.27), ("B", -0.09), ("C", 0.18)):
            s = f[(f.kind == k) & f.complete]
            ax.plot(i + dx + rng.uniform(-0.05, 0.05, len(s)), s.gamma, "o", ms=5, color=KIND_COLOR[k],
                    alpha=0.75, mec="none", label=KIND_LABEL[k] if i == 0 else None)
            nt = int((f[f.kind == k]["class"] == "trapped").sum())
            ni = int((f[f.kind == k]["class"] == "incomplete").sum())
            if nt or ni:
                ax.text(i + dx, -0.75, f"{nt}T\n{ni}I" if ni else f"{nt}T", ha="center", va="top",
                        fontsize=7, color=KIND_COLOR[k])
    ax.set_xticks(range(len(DYN)), [cell_label(n) for n in DYN], fontsize=8)
    ax.set_ylabel("escape exponent gamma\n(t_esc ~ eps^-gamma, fit over eps = 1e-4.5 ... 1e-3)")
    ax.set_ylim(-1.1, 4.6)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_title("Escape exponent per direction: inside M* always c*_J - 2; Gaussian starts (C) mostly too"
                 " (T = trapped, I = incomplete directions)", fontsize=11)
    savefig(fig, RES / "fig3_exponents")
    plt.close(fig)


# ---- 4. which pattern is learned first -------------------------------------------------------


def fig_first(runs):
    cells = ["TWO23", "TWO34", "TIE3"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, name in zip(axs, cells):
        c = get_cell(name)
        g = c.pattern_strings[c.greedy()]
        for k in ("A", "B", "C"):
            s = runs[(runs.cell == name) & (runs.kind == k) & runs.first_learned.notna()]
            fr = s.groupby("eps").first_is_greedy.mean()
            ax.semilogx(fr.index, fr.values, "o-", color=KIND_COLOR[k], label=KIND_LABEL[k], ms=5)
        if name != "TIE3":
            other = [c.pattern_strings[r] for r in c.missing if r != c.greedy()][0]
            ax.set_title(f"{name}: greedy {g} (c={c.depth(c.greedy())}, s={c.weights[c.greedy()]:g})\n"
                         f"vs {other} (c={c.depth(c.missing[1] if c.missing[0] == c.greedy() else c.missing[0])},"
                         f" larger weight)", fontsize=10)
        else:
            other = [r for r in c.missing if r != c.greedy()][0]
            ax.set_title(f"TIE3: tie at depth 3; greedy {g} (sA={c.weights[c.greedy()] * c.A(c.greedy()):.2f})\n"
                         f"vs {c.pattern_strings[other]} (sA={c.weights[other] * c.A(other):.2f})", fontsize=10)
        ax.axhline(1, color=MUTED, ls="--", lw=1)
        ax.set_xlabel("perturbation size eps")
        ax.invert_xaxis()
    axs[0].set_ylabel("fraction of runs that learn\nthe greedy pattern first")
    axs[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Which missing pattern is learned first: the greedy rule (shorter route first) holds at small eps",
                 fontsize=12)
    savefig(fig, RES / "fig4_first_learned")
    plt.close(fig)


# ---- 5. Corollary 8.6 ------------------------------------------------------------------------


def fig_cor86(runs):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, col in (("EX87_J0", ORANGE), ("EX87_J01", BLUE)):
        c = get_cell(name)
        r = c.greedy()
        for k, mk in (("A", "o"), ("C", "s")):
            s = runs[(runs.cell == name) & (runs.kind == k) & (runs.outcome == "escaped")]
            med = s.groupby("eps").t_esc.median()
            ax.loglog(med.index, med.values, mk, color=col, ms=6, mfc=col if k == "A" else "none",
                      label=f"J = {{{', '.join(c.pattern_strings[a] for a in c.J)}}}: kind {k} (median)")
        e = np.logspace(-4.5, -1, 50)
        b = runs[(runs.cell == name) & (runs.kind == "A_bal")]
        pred = [c.weights[r] * c.A(r)]
        from common import leading_616
        ax.loglog(e, [leading_616(x, c.depth(r), c.A(r), c.weights[r]) for x in e], color=col, lw=1.2, ls="--",
                  label=f"eq. 6.16, depth {c.depth(r)}" + (" (log law)" if c.depth(r) == 2 else ""))
    ax.set_xlabel("perturbation size eps")
    ax.set_ylabel("escape time (gradient-flow time units)")
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Corollary 8.6: once 1111111 is learned, 0022111 needs 2 new cores instead of 5\n"
                 "(escape time eps^-3 becomes log(1/eps))", fontsize=11)
    savefig(fig, RES / "fig5_cor86")
    plt.close(fig)


# ---- 6. Theorem 9.2 family -----------------------------------------------------------------------


def fig_thm92(curves, thm):
    names = ["T92a", "T92b", "T92c", "T92d", "PS2"]
    fig, axs = plt.subplots(1, len(names), figsize=(15, 4), sharey=False)
    for ax, name in zip(axs, names):
        c = get_cell(name)
        for space, col in (("Mstar", BLUE), ("full", ORANGE)):
            s = curves[(curves.cell == name) & (curves.space == space)]
            ax.loglog(s.t, -s.m, "o-", color=col, ms=4, label="inside M*" if space == "Mstar" else "full space")
        s = curves[(curves.cell == name) & (curves.space == "full")]
        t0, m0 = s.t.iloc[-1], -s.m.iloc[-1]
        f = c.thm92()["order"]
        tt = np.array([s.t.min(), s.t.max()])
        ax.loglog(tt, m0 * (tt / t0) ** f, color=INK, ls="--", lw=1, label=f"slope {f} (Thm 9.2 formula)")
        if f != c.c_star:
            ax.loglog(tt, m0 * (tt / t0) ** c.c_star, color=MUTED, ls=":", lw=1, label=f"slope {c.c_star} (c*_J)")
        ax.set_title(f"{name}: formula {f}, c*_J = {c.c_star}", fontsize=10)
        ax.set_xlabel("distance t from theta_J")
        ax.legend(fontsize=8, loc="lower right")
    axs[0].set_ylabel("largest loss decrease on the sphere\n-(min L - L_J)")
    fig.suptitle("Theorem 9.2 family: the full-space order follows min over parents of min(k*, 2 l*)",
                 fontsize=12)
    savefig(fig, RES / "fig6_thm92")
    plt.close(fig)


def main():
    big()
    orders = pd.read_csv(RES / "static_orders.csv")
    curves = pd.read_csv(RES / "static_curves.csv")
    thm = pd.read_csv(RES / "thm92.csv")
    fig_static(orders, thm)
    fig_thm92(curves, thm)
    if (RES / "escape_runs.csv").exists():
        runs = pd.read_csv(RES / "escape_runs.csv")
        fits = pd.read_csv(RES / "escape_fits.csv")
        fig_inside(runs)
        fig_exponents(fits)
        fig_first(runs)
        fig_cor86(runs)


if __name__ == "__main__":
    main()
