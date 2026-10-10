"""Report figures of Sec. 3 (map of regimes) and Sec. 5.1-5.4 (lazy, rich, stalled, large chi).

Both figures read only committed tables, never raw runs:
  report_map    (a) sigma x chi boundaries, binary N = 4 (per-seed S2S, lazy and floor-stall
                boundaries; theory P5, P4); (b) sigma x K phase diagram (centred, d = 2, chi = 4) with
                per-tree S2S boundaries and brackets; (c) root imbalance at chi = 24; (d) kernel change
                vs root scale at the P20 check point.
                Tables: sensitivity_per_seed_sigma_chi_b4_centred.csv, boundaries_per_seed_chi_b4.csv,
                fix1/lazy_per_seed_chi_b4.csv, fix1/lazy_summary_chi_b4.csv, sigma_c_vs_K.csv,
                fix1/imbalance_P20.csv and report/{floor_sigma_chi_b4, k_sweep_cells,
                imbalance_map_b4}.csv
  report_other  (a) final loss against the Eckart-Young floor at deficient bonds (binary N = 4);
                (b) kernel change and (c) core changes against chi under SP, NTK and muP-TTN.
                Tables: report/runs_sigma_chi_b4.csv, report/large_chi_runs.csv

report/*.csv are written from the raw runs by report_tables.py. Run from the repository root:
    uv run python experiments/regime_map/plot_report.py [--out DIR]
Writes report_map.pdf/.png and report_other.pdf/.png (default: experiments/regime_map/results).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from ttn import plotting as tp  # noqa: E402

RES = os.path.join(HERE, "results")
TAB = os.path.join(RES, "report")
REGIMES = ["s2s", "rich", "lazy"]
COL = tp.regime_colors(REGIMES)
COL["stalled"] = "#cfcec8"  # neutral: no regime
NAME = {
    "s2s": "saddle-to-saddle",
    "rich": "rich, non-incremental",
    "lazy": "lazy",
    "stalled": "stalled",
}


def gmean(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.log(v).mean())) if v.size else np.nan


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RES, name))


def edges(vals) -> np.ndarray:
    """Cell edges at geometric midpoints of sorted positive values."""
    v = np.log(np.asarray(vals, float))
    if v.size == 1:
        return np.exp(np.array([v[0] - 0.15, v[0] + 0.15]))
    mid = (v[1:] + v[:-1]) / 2
    return np.exp(np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]]))


# ---- report_map ---------------------------------------------------------------------------------


def panel_sigma_chi(ax):
    sc = read("sensitivity_per_seed_sigma_chi_b4_centred.csv")
    cb = read("boundaries_per_seed_chi_b4.csv")
    cb = cb[cb.model == "centred"]
    lz = read("fix1/lazy_per_seed_chi_b4.csv")
    lz = lz[lz.model == "centred"]
    p5 = read("fix1/lazy_summary_chi_b4.csv")
    p5 = p5[p5.model == "centred"]
    fl = pd.read_csv(os.path.join(TAB, "floor_sigma_chi_b4.csv"))

    s2s_a = sc.groupby("chi").s2s_3.apply(gmean)  # sigma_chi_b4: rho = 3 crossing, chi = 3-36
    s2s_b = cb.groupby("chi").sigma_c.apply(gmean)  # chi_b4: label-rule bracket, chi = 9-128
    lazy_t = lz.groupby("chi")["tensor_0.1"].apply(gmean)  # tensor-space kernel (theory P5)
    lazy_p = lz.groupby("chi")["probe_0.1"].apply(gmean)
    x0, x1, y0, y1 = 2.6, 160.0, 0.008, 30.0
    split = np.sqrt(6 * 9)  # bonds are full (chi >= d^2 = 9) to the right

    # regions on a dense grid: S2S below the S2S boundary; above the upper boundary lazy (full bonds,
    # chi >= d^2 = 9) or stalled at the floor of the unrealizable centred target (deficient bonds)
    s2s = pd.concat([s2s_a, s2s_b[s2s_b.index > 36]]).sort_index()
    gx = np.geomspace(x0, x1, 400)
    lo = np.exp(np.interp(np.log(gx), np.log(s2s.index), np.log(s2s.values)))
    lo = np.where(gx > s2s.index[-1], s2s.values[-1] * np.sqrt(s2s.index[-1] / gx), lo)
    lzy = np.exp(np.interp(np.log(gx), np.log(lazy_t.index), np.log(lazy_t.values)))
    lzy = np.where(gx > lazy_t.index[-1], lazy_t.values[-1] * np.sqrt(lazy_t.index[-1] / gx), lzy)
    fly = np.exp(np.interp(np.log(gx), np.log(fl.chi), np.log(fl.sigma_all_stalled)))
    up = np.where(gx < split, fly, lzy)
    ax.fill_between(gx, y0, lo, color=COL["s2s"], alpha=0.16, lw=0)
    ax.fill_between(gx, lo, up, color=COL["rich"], alpha=0.16, lw=0)
    full = gx >= split
    ax.fill_between(gx[full], lzy[full], y1, color=COL["lazy"], alpha=0.16, lw=0)
    ax.fill_between(
        gx[~full], fly[~full], y1, facecolor=COL["stalled"], edgecolor="white", hatch="////", lw=0
    )
    ax.axvline(split, color=tp.MUTED, lw=0.6, ls=":")

    # theory: P5 bracket and asymptote; slope -1/2 guide through the S2S boundary
    ax.fill_between(p5.chi, p5.p5_lo, p5.p5_hi, color=COL["lazy"], alpha=0.35, lw=0)
    xx = np.geomspace(9, x1, 50)
    ax.plot(xx, 6.50 / np.sqrt(xx), ls=(0, (1, 1.5)), color=tp.INK, lw=0.9)
    ax.plot(xx, 0.30 / np.sqrt(xx), ls=(0, (4, 2)), color=tp.INK_SECONDARY, lw=0.7)

    # per-seed values (light) and geometric means
    ax.plot(sc.chi, sc.s2s_3, ".", ms=2.5, color=COL["s2s"], alpha=0.45)
    ax.plot(cb.chi, cb.sigma_c, ".", ms=2.5, color=COL["s2s"], alpha=0.45)
    ax.plot(lz.chi, lz["tensor_0.1"], ".", ms=2.5, color=COL["lazy"], alpha=0.45)
    ax.plot(s2s_a.index, s2s_a.values, "o-", ms=3.5, lw=1.1, color=COL["s2s"])
    ax.plot(s2s_b.index, s2s_b.values, "o--", ms=3.5, lw=0.9, mfc="white", color=COL["s2s"])
    ax.plot(lazy_t.index, lazy_t.values, "D-", ms=3.2, lw=1.1, color=COL["lazy"])
    ax.plot(lazy_p.index, lazy_p.values, "D--", ms=3.0, lw=0.8, mfc="white", color=COL["lazy"])
    ax.plot(fl.chi, fl.sigma_all_stalled, "v-", ms=3.5, lw=1.1, color=tp.INK_SECONDARY)
    ax.plot(fl.chi, fl.sigma_floor_P4, "v", ms=3.5, mfc="white", color=tp.INK_SECONDARY)

    ax.text(4.2, 0.03, "S2S", ha="center", fontsize=8)
    ax.text(16, 0.45, "rich", ha="center", fontsize=8)
    ax.text(45, 6.0, "lazy", ha="center", fontsize=8)
    ax.text(4.4, 5.0, "floor", ha="center", fontsize=8, bbox=dict(fc="white", ec="none", pad=0.8))
    ax.set(xscale="log", yscale="log", xlim=(x0, x1), ylim=(y0, y1))
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"per-core scale $\sigma$")
    ax.set_title(r"(a) $\sigma\times\chi$: binary $N=4$ ($K=3$), $d=3$", fontsize=8.5)
    ax.grid(False)


def panel_sigma_K(ax):
    cells = pd.read_csv(os.path.join(TAB, "k_sweep_cells.csv"))
    cells = cells[cells.series == "centred, chi = 4"]
    sk = read("sigma_c_vs_K.csv")
    sk = sk[sk.exp == "sigma_K_centred_d2"].set_index("tree")
    trees = cells.drop_duplicates("tree").sort_values("K")[["tree", "K"]].values.tolist()
    for j, (tree, K) in enumerate(trees):
        g = cells[cells.tree == tree].sort_values("sigma")
        xe = edges(g.sigma.values)
        b = sk.loc[tree]
        for i, r in enumerate(g.itertuples()):
            lab = r.majority
            # stalled above the S2S block: the floor of the unrealizable centred target (chi = 4 is
            # deficient from K = 6 on); below it: a partial fit whose plateau outlasts the budget
            floor = lab == "stalled" and r.sigma > b.bracket_lo_median
            ax.add_patch(
                Rectangle(
                    (xe[i], j - 0.42),
                    xe[i + 1] - xe[i],
                    0.84,
                    facecolor=COL[lab],
                    edgecolor="white",
                    lw=0.4,
                    hatch="////" if floor else None,
                    alpha=1.0 if r.n_majority >= 4 else 0.55,
                )
            )
        if b.n_resolved >= b.n_seeds / 2:
            ax.plot(b.sigma_c_median, j, "o", ms=3.6, color=tp.INK, zorder=5)
        else:
            ax.plot(
                [b.bracket_lo_median, b.bracket_hi_median],
                [j, j],
                "-",
                color=tp.INK,
                lw=1.4,
                zorder=5,
            )
            ax.plot(
                [b.bracket_lo_median, b.bracket_hi_median],
                [j, j],
                "|",
                color=tp.INK,
                ms=5,
                zorder=5,
            )
    a0c = [sk.loc[t].a0c for t, _ in trees]
    ax.plot(a0c, range(len(trees)), ls=(0, (3, 1.5)), color=tp.INK_SECONDARY, lw=0.8)
    ax.set_yticks(range(len(trees)), [f"{t.replace('mps', 'MPS')} ({K})" for t, K in trees])
    ax.set(xscale="log", xlim=(0.015, 2.6), ylim=(-0.6, len(trees) - 0.4))
    ax.set_xlabel(r"per-core scale $\sigma$")
    ax.set_ylabel(r"tree ($K$)")
    ax.set_title(r"(b) $\sigma\times K$: centred, $d=2$, $\chi=4$", fontsize=8.5)
    ax.grid(False)


def panel_imbalance(ax):
    m = pd.read_csv(os.path.join(TAB, "imbalance_map_b4.csv"))
    m = m[m.chi == 24]
    lam = np.sort(m.lam_rel.unique())
    alp = np.sort(m.alpha.unique())
    xe, ye = edges(lam), edges(alp)
    for r in m.itertuples():
        i, j = np.searchsorted(lam, r.lam_rel), np.searchsorted(alp, r.alpha)
        ax.add_patch(
            Rectangle(
                (xe[i], ye[j]),
                xe[i + 1] - xe[i],
                ye[j + 1] - ye[j],
                facecolor=COL[r.majority],
                edgecolor="white",
                lw=0.4,
                alpha=1.0 if r.n_majority >= 4 else 0.55,
            )
        )
    ax.axvline(1.0, color=tp.INK, lw=0.8, ls=(0, (3, 1.5)))
    ax.axvline(1 / np.sqrt(24), color=tp.INK, lw=0.8, ls=(0, (1, 1.5)))
    ax.set(xscale="log", yscale="log", xlim=(xe[0], xe[-1]), ylim=(ye[0], ye[-1]))
    ax.set_xlabel(r"root imbalance $\lambda/\sqrt{\chi}$")
    ax.set_ylabel(r"function-space scale $\alpha=\lambda\sigma^3$")
    ax.set_title(r"(c) root scale $\lambda\sigma$: binary $N=4$, $\chi=24$", fontsize=8.5)
    ax.grid(False)


def panel_p20(ax):
    p = read("fix1/imbalance_P20.csv")
    t, q = p[p.kernel == "dK_tensor"], p[p.kernel == "dK"]
    ax.axhspan(0.6, 0.7, color=tp.GRID, lw=0)
    ax.fill_between(t.lam, t.lo, t.hi, color=COL["lazy"], alpha=0.18, lw=0)
    ax.plot(t.lam, t["median"], "D-", ms=3.2, lw=1.1, color=COL["lazy"], label="tensor kernel")
    ax.plot(
        q.lam,
        q["median"],
        "D--",
        ms=3.0,
        lw=0.8,
        mfc="white",
        color=COL["lazy"],
        label="probe kernel",
    )
    lam = t.lam.values[:4]
    ax.plot(
        lam,
        t["median"].values[1] * (lam / lam[1]) ** 2,
        color=tp.INK,
        lw=0.7,
        ls=(0, (3, 1.5)),
        label=r"$\propto\lambda^2$",
    )
    ax.axhline(0.1, color=tp.MUTED, lw=0.7)
    ax.axvline(4.0, color=tp.INK, lw=0.8, ls=(0, (3, 1.5)))
    ax.set(xscale="log", yscale="log", ylim=(1e-5, 2))
    ax.set_xlabel(r"root scale $\lambda$ ($\chi=16$, $\alpha=3.375$)")
    ax.set_ylabel(r"kernel change $\delta_\Theta(T)$")
    ax.set_title(r"(d) kernel change against $\lambda$", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=8)


def fig_map(out: str):
    tp.set_style()
    fig, axes = plt.subplots(2, 2, figsize=(tp.TEXT_WIDTH, 4.55))
    panel_sigma_chi(axes[0, 0])
    panel_sigma_K(axes[0, 1])
    panel_imbalance(axes[1, 0])
    panel_p20(axes[1, 1])
    handles = [Patch(facecolor=COL[r], label=NAME[r]) for r in ("s2s", "rich", "lazy")]
    handles += [
        Patch(facecolor=COL["stalled"], label="stalled at a partial fit"),
        Patch(
            facecolor=COL["stalled"], hatch="////", edgecolor="white", label="stalled at a floor"
        ),
        Line2D(
            [],
            [],
            color=tp.INK,
            ls=(0, (1, 1.5)),
            lw=0.9,
            label=r"lazy asymptote $6.50\,\chi^{-1/2}$",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    tp.savefig(fig, os.path.join(out, "report_map"))
    plt.close(fig)


# ---- report_other -------------------------------------------------------------------------------


def panel_floor(ax):
    r = pd.read_csv(os.path.join(TAB, "runs_sigma_chi_b4.csv"))
    r = r[r.model == "centred"]
    cols = tp.ordinal_colors(6)[2:]
    for c, chi in zip(cols, (3, 4, 6, 9)):
        g = r[r.chi == chi].groupby("sigma")
        med = g.l_end.median()
        ax.plot(med.index, med.values, "o", ms=2.8, color=c, label=rf"$\chi={chi}$")
        if chi < 9:
            eb = g.l_best.median()
            ax.plot(eb.index, eb.values, "-", lw=0.9, color=c)
            dim_m, D = 18 * chi - chi * chi, 81
            s = np.geomspace(0.01, 0.6, 30)
            ax.plot(s, 0.7 * (1 - dim_m / D) * s**6 / 14, ls=(0, (1, 1.2)), lw=0.8, color=c)
        else:
            ax.plot(med.index, med.values, "-", lw=0.6, color=c)
    ax.axhline(1e-3, color=tp.MUTED, lw=0.7, ls=(0, (3, 1.5)))
    ax.set(xscale="log", yscale="log", ylim=(1e-14, 2), xlim=(0.008, 50))
    ax.set_xlabel(r"per-core scale $\sigma$")
    ax.set_ylabel(r"$L(T)/L(0)$")
    ax.set_title(r"(a) deficient bonds: the floor", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=8, ncol=2, handletextpad=0.2, columnspacing=0.6)


PARAM = [("sp", "SP"), ("ntk", "NTK"), ("mup", r"$\mu$P-TTN")]


def panel_large_chi(ax, ax2):
    r = pd.read_csv(os.path.join(TAB, "large_chi_runs.csv"))
    r = r[(r.model == "centred") & np.isclose(r.sigma, 1.5)]
    cols = tp.palette(3)
    for c, (lr, name) in zip(cols, PARAM):
        g = r[r.lr == lr]
        med = g.groupby("chi").dK_tensor.median()
        ax.plot(g.chi, g.dK_tensor, ".", ms=2.2, color=c, alpha=0.4)
        ax.plot(med.index, med.values, "o-", ms=3.0, lw=1.1, color=c, label=name)
        root = g.groupby("chi").root_change.median()
        leaf = g.groupby("chi").leaf_change.median()
        ax2.plot(root.index, root.values, "o-", ms=3.0, lw=1.1, color=c)
        ax2.plot(leaf.index, leaf.values, "s--", ms=2.8, lw=0.9, mfc="white", color=c)
    x = np.array([16, 128.0])
    for c, (lr, _), p in zip(cols, PARAM, (-3, -1, 0)):
        y128 = r[(r.lr == lr) & (r.chi == 128)].dK_tensor.median()
        ax.plot(x, y128 * (x / 128) ** p, color=tp.INK, lw=0.7, ls=(0, (3, 1.5)))
    ax.plot([64, 128], [0.240, 0.120], "x", ms=5, color=tp.INK, label="NTK, predicted")
    ax.axhspan(6.2, 6.3, color=tp.GRID, lw=0)
    ax.set(xscale="log", yscale="log", ylim=(1e-4, 30))
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"kernel change $\delta_\Theta(T)$")
    ax.set_title(r"(b) large $\chi$: kernel change", fontsize=8.5)
    ax.legend(loc="lower left", fontsize=8, handlelength=1.5, labelspacing=0.25)
    # colours as in (b); line style tells the cores apart
    style = [
        Line2D([], [], color=tp.INK_SECONDARY, marker="o", ms=3.0, lw=1.1, label="root"),
        Line2D(
            [],
            [],
            color=tp.INK_SECONDARY,
            marker="s",
            ms=2.8,
            mfc="white",
            lw=0.9,
            ls="--",
            label="leaf cores",
        ),
    ]
    ax2.set(xscale="log", yscale="log")
    ax2.set_xlabel(r"bond dimension $\chi$")
    ax2.set_ylabel("relative change of the cores")
    ax2.set_title(r"(c) large $\chi$: root and leaf cores", fontsize=8.5)
    ax2.legend(handles=style, loc="lower left", fontsize=8, handlelength=2.2)


def fig_other(out: str):
    tp.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(tp.TEXT_WIDTH, 2.45))
    panel_floor(axes[0])
    panel_large_chi(axes[1], axes[2])
    tp.savefig(fig, os.path.join(out, "report_other"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=RES)
    a = ap.parse_args()
    fig_map(a.out)
    fig_other(a.out)
    print("wrote report_map and report_other to", a.out)
