"""Report figures for Sec. 4 (s2s_core), at final size for a 6.5 in text width.

- report_runs: two representative saddle-to-saddle runs (MPS4 and binary8), one row each: loss in
  linear time, overlaps, the bond spectrum of one internal edge, the effective bond rank of every edge.
  Reads the raw run files (.json + .npz) of sweeps main and large.
- report_prefactor: leading-order first-escape prefactor (median over initializations) against the
  physical dimension d at full bond, the bond dimension chi and the number of cores K, with the
  Kumar-Haupt time and the RMS-overlap heuristic. Reads results/lo_d.csv, lo_chi_table.csv, lo_K_table.csv.

    uv run python experiments/s2s_core/plot_report.py --raw-main data/s2s_core/main \
        --raw-large data/s2s_core/large --results experiments/s2s_core/results --out experiments/s2s_core/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402

from ttn import diagnostics as dg  # noqa: E402
from ttn import plotting as tp  # noqa: E402

HERE = Path(__file__).parent
# (run name, raw folder option, edge whose spectrum is shown: the middle / root edge, as named by
# tp.edge_label with sites counted from 0, and as printed in the report: sites from 0, bipartitions as
# 01|23 or 0-3|4-7, notes/report_style.md)
RUNS = [
    ("mps4_chi3_w3-2-1_fanin_s0.01_seed0", "main", "{0,1} | {2,3}", r"$01|23$"),
    ("binary8_chi6_w3-2-1_fanin_s0.1_seed0", "large", "{0–3} | {4–7}", r"$0$–$3|4$–$7$"),
]
TREE_NAMES = {"mps": "MPS", "binary": "binary", "ht": "HT"}


def load_run(raw: Path, name: str):
    summ = json.loads((raw / f"{name}.json").read_text())
    return summ, np.load(raw / f"{name}.npz")


def fig_runs(raws: dict[str, Path], out: Path):
    fig, axes = plt.subplots(len(RUNS), 4, figsize=(tp.TEXT_WIDTH, 3.7), layout="constrained")
    for i, (name, sweep, edge_name, edge_title) in enumerate(RUNS):
        row = axes[i]
        summ, d = load_run(raws[sweep], name)
        spec, _ = s2s.setup(
            summ["tree"], summ["N"], summ["d"], summ["chi"], summ["weights"], summ["teacher_seed"]
        )
        w = np.asarray(summ["weights"])
        t, a, sp = d["t"], d["overlaps"], d["spectra"]
        # Loss and overlaps: log grid merged with the refined event windows.
        ts, ls, as_ = [t], [d["loss"]], [a]
        for r in range(len(w)):
            if f"fine_t_{r}" in d:
                ts.append(d[f"fine_t_{r}"])
                ls.append(d[f"fine_loss_{r}"])
                as_.append(d[f"fine_a_{r}"])
        tm = np.concatenate(ts)
        idx = np.argsort(tm, kind="stable")
        tm, lm, am = tm[idx], np.concatenate(ls)[idx], np.concatenate(as_)[idx]
        ev = np.array([np.nan if x is None else x for x in summ["events"]])

        ax = row[0]
        ax.plot(tm, lm, color=tp.PALETTE[0], lw=1.2)
        for k in range(len(w) + 1):
            lev = 0.5 * float(np.sum(w[k:] ** 2))
            if lev > 0:
                ax.axhline(lev, color=tp.MUTED, lw=0.7, ls=":")
        ax.set_xlim(0, 1.1 * np.nanmax(ev))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_xlabel("$t$")
        ax.set_ylabel("$L(t)$")
        label = f"{TREE_NAMES[summ['tree']]}{summ['N']}, $K={summ['K']}$, $\\chi={summ['chi']}$, $\\sigma={summ['sigma']:g}$"
        ax.set_title(label, loc="left", fontsize=8)

        ax = row[1]
        tp.plot_overlaps(ax, tm, am, legend=False)

        ax = row[2]
        edges = list(spec.tree.edges)
        j = [tp.edge_label(spec.tree, v) for v in edges].index(edge_name)
        tp.plot_bond_spectra(
            [ax], t, {edges[j]: sp[:, j, :]}, tree=spec.tree, teacher=w, legend=False
        )
        ax.set_title(f"spectrum, edge {edge_title}", fontsize=8)
        ax.set_ylabel("singular value")

        ax = row[3]
        ranks = dg.effective_rank(np.nan_to_num(sp), 0.1 * w.min())
        cols = tp.ordinal_colors(len(edges))
        for k in range(len(edges)):
            ax.plot(t, ranks[:, k] + 0.05 * (k - (len(edges) - 1) / 2), lw=1.0, color=cols[k])
        ax.set_xscale("log")
        ax.set_yticks(range(len(w) + 1))
        ax.set_xlabel("$t$")
        ax.set_ylabel("effective bond rank")
        ax.set_title(f"all {len(edges)} edges", fontsize=8)
        for x in ev[np.isfinite(ev)]:
            for axx in row[1:]:
                axx.axvline(x, color=tp.MUTED, lw=0.6, ls="--", zorder=0)
    # One legend for the figure: component r and singular-value index k share colour slot r = k.
    handles = [
        plt.Line2D(
            [],
            [],
            color=tp.PALETTE[k],
            lw=1.5,
            label=f"$a_{k + 1}$, $s^e_{k + 1}$ ($s_{k + 1}={x:g}$)",
        )
        for k, x in enumerate((3, 2, 1))
    ]
    handles += [
        plt.Line2D([], [], color=tp.MUTED, lw=0.9, label="other $s^e_j$"),
        plt.Line2D([], [], color=tp.MUTED, lw=0.9, ls="--", label="teacher values / events"),
        plt.Line2D([], [], color=tp.MUTED, lw=0.7, ls=":", label="partial-fit levels $L_S$"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, fontsize=8)
    tp.savefig(fig, out / "report_runs")


def median_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x))), axis=1)
    return np.median(x), np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def fig_prefactor(res: Path, out: Path):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(tp.TEXT_WIDTH, 2.9),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.25, 0.75, 1.1]},
    )
    trees = ["binary4", "binary8", "mps4", "mps8", "ht4"]
    # colour = tree family (MPS blue, binary orange, HT aqua, as in every report figure); marker = N
    fam_col = dict(zip(["mps", "binary", "ht"], tp.palette(3)))
    colors = {t: fam_col[t.rstrip("0123456789")] for t in trees}
    marker = {t: "o" if t.endswith("4") else "s" for t in trees}
    names = {
        "binary4": "binary4 ($K=3$)",
        "mps4": "MPS4 ($K=4$)",
        "binary8": "binary8 ($K=7$)",
        "ht4": "HT4 ($K=7$)",
        "mps8": "MPS8 ($K=8$)",
    }

    # (a) bond dimension, single component s = 1
    ax = axes[0]
    tab = pd.read_csv(res / "lo_chi_table.csv")
    tab = tab[tab.weights.astype(str) == "1"]
    for tree in trees:
        g = tab[tab.tree == tree].sort_values("chi")
        ax.errorbar(
            g.chi,
            g.T_median,
            yerr=[g.T_median - g.T_ci_lo, g.T_ci_hi - g.T_median],
            color=colors[tree],
            marker=marker[tree],
            ms=3,
            lw=1.1,
            capsize=1.2,
        )
        ax.plot(
            g.chi,
            g.tKH_median,
            color=colors[tree],
            lw=0.8,
            ls="--",
            marker=marker[tree],
            ms=2.5,
            mfc="white",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"bond dimension $\chi$ ($d=4$, $s=1$)")
    ax.set_ylabel(r"$\mathcal{T}=\lim_{\sigma\to0}\sigma^{K-2}t_1$")
    ax.set_title("(a)", loc="left", fontsize=8)

    # (b) physical dimension at full bond chi = d^2, single component s = 1
    ax = axes[1]
    ld = pd.read_csv(res / "lo_d.csv")
    for tree in ("binary4", "mps4"):
        g = ld[ld.tree == tree]
        ds = sorted(g.d.unique())
        st = np.array([median_ci(g[g.d == x]["T"]) for x in ds])
        ax.errorbar(
            ds,
            st[:, 0],
            yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]],
            color=colors[tree],
            marker=marker[tree],
            ms=3,
            lw=1.1,
            capsize=1.2,
        )
        ax.plot(
            ds,
            [np.median(g[g.d == x].t_KH) for x in ds],
            color=colors[tree],
            lw=0.8,
            ls="--",
            marker=marker[tree],
            ms=2.5,
            mfc="white",
        )
    ax.axhline(np.pi / 2, color=tp.INK, lw=0.7, ls=":")
    ax.text(3.0, np.pi / 2 * 1.12, r"$\pi/2$", fontsize=8, color=tp.INK, ha="center")
    ax.set_yscale("log")
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel(r"$d$ (full bond, $\chi=d^2$)")
    ax.set_ylabel(r"$\mathcal{T}$")
    ax.set_title("(b)", loc="left", fontsize=8)

    # (c) number of cores, chi = 3, d = 4, s = (3, 2, 1)
    ax = axes[2]
    kt = pd.read_csv(res / "lo_K_table.csv")
    for fam in ("binary", "mps", "ht"):
        g = kt[kt.family == fam].sort_values("K")
        ax.errorbar(
            g.K,
            g.T_median,
            yerr=[g.T_median - g.T_ci_lo, g.T_ci_hi - g.T_median],
            color=fam_col[fam],
            marker="D",
            ms=2.8,
            lw=1.1,
            capsize=1.2,
        )
        ax.plot(g.K, g.P3, color=fam_col[fam], lw=0.9, ls=":")
        ax.plot(g.K, g.tKH_median, color=fam_col[fam], lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"cores $K$ ($\chi=3$, $d=4$, $s=(3,2,1)$)")
    ax.set_ylabel(r"$\mathcal{T}$")
    ax.set_title("(c)", loc="left", fontsize=8)

    # One legend below the panels: trees of (a), (b) (marker = N; in (c) diamonds, colour = family)
    # and the line styles.
    h = [
        plt.Line2D([], [], color=colors[t], marker=marker[t], ms=3, lw=1.1, label=names[t])
        for t in trees
    ]
    h += [
        plt.Line2D([], [], color=tp.INK, lw=1.1, label=r"median $\mathcal{T}$ (95% CI)"),
        plt.Line2D([], [], color=tp.INK, lw=0.8, ls="--", label=r"median $t_{\rm KH}$"),
        plt.Line2D([], [], color=tp.INK, lw=0.9, ls=":", label=r"heuristic $P_3$"),
    ]
    fig.legend(
        handles=h,
        loc="outside lower center",
        ncols=4,
        fontsize=8,
        columnspacing=1.2,
        handlelength=1.8,
    )
    tp.savefig(fig, out / "report_prefactor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-main", required=True, help="raw folder of sweep main (e.g. data/s2s_core/main)"
    )
    ap.add_argument(
        "--raw-large", required=True, help="raw folder of sweep large (e.g. data/s2s_core/large)"
    )
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument(
        "--out", required=True, help="output directory (e.g. experiments/s2s_core/results)"
    )
    args = ap.parse_args()
    tp.set_style()
    out = Path(args.out)
    fig_runs({"main": Path(args.raw_main), "large": Path(args.raw_large)}, out)
    fig_prefactor(Path(args.results), out)


if __name__ == "__main__":
    main()
