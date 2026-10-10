"""Report figures of Sec. 4.5 (order of learning), from committed tables in results/.

Two figures at the report's text width in the `ttn.plotting` style:
  report_order_first  which component is learned first: (a) P(larger first) against rho at sigma = 0.01,
                      gradient flow and the leading-order race (main_summary.csv); (b) P(smaller first)
                      against sigma at rho = 0.8 for matrix-like (m = 2) and tensor-like (m >= 3) pairs,
                      with the fitted power laws (sigma_table.csv, sigma_fits.csv); (c) uncapped bond
                      dimension: race P(larger first) at rho = 0.8 against chi, with gradient-flow checks
                      (chi_scan.csv, chi_gf.csv)
  report_order_wrong  after a wrong first escape, started at the wrong saddle: overlaps of one run per
                      branch (rotation, adding, kept; report_examples.csv) and the escape time, in units
                      of the predicted 1/|lambda_min|, against ln(1/sigma) for the two strict branches
                      (h1b_saddle_b4.csv, h1b_saddle_mps4_sep.csv, h1b_saddle_mps4_adj.csv)
report_examples.csv holds the three example trajectories, downsampled from the raw runs; it is rewritten
only with --raw DIR (the folder holding h1b_saddle_b4/, h1b_saddle_mps4_adj/, h1b_saddle_mps4/).

Run: uv run python experiments/order/plot_report.py [--raw data/order] [--res experiments/order/results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ttn import plotting as tp

RES_DEFAULT = Path(__file__).parent / "results"
# The tight bounding box adds the outer labels and the legend: this keeps the saved figure within the text width.
FIG_WIDTH = tp.TEXT_WIDTH - 0.17

# One colour per entity (tree, or tree and differing cores), the same in both figures.
COLORS = dict(
    zip(
        ["binary4", "mps4", "mps6", "binary8", "mps8", "mps4_sep", "mps4_adj", "mps6_m3"],
        tp.palette(8),
    )
)
LABELS = {  # odeco teachers, or pattern teachers whose two components differ on the sites D
    "binary4": r"binary4 ($m=2$)",
    "mps4": r"MPS4 ($m=4$)",
    "mps6": r"MPS6 ($m=6$)",
    "binary8": r"binary8 ($m=4$)",
    "mps8": r"MPS8 ($m=8$)",
    "mps4_sep": r"MPS4, $D=\{0,2\}$ ($m=2$)",
    "mps4_adj": r"MPS4, $D=\{1,2\}$ ($m=2$, adj.)",
    "mps6_m3": r"MPS6, $D=\{0,1,2\}$ ($m=3$)",
}
# Example runs from the wrong saddle (rho = 0.8, seed 0): (case, raw folder, file).
EXAMPLES = [
    ("binary4", "h1b_saddle_b4", "rho0.8_s0.0001_seed0.npz"),
    ("mps4_adj", "h1b_saddle_mps4_adj", "rho0.8_s0.0001_seed0.npz"),
    ("mps4", "h1b_saddle_mps4", "rho0.8_s0.01_seed0.npz"),
]


def extract_examples(raw: Path, res: Path, n_points: int = 600) -> None:
    """Write report_examples.csv: t, loss, a1, a2 of the example runs on a log-time grid."""
    rows = []
    for case, folder, name in EXAMPLES:
        traj = np.load(raw / folder / name)["traj"]  # columns t, L, a1, a2, |grad|, s1, s2 (h1b.py)
        t = traj[:, 0]
        grid = np.geomspace(max(t[t > 0].min(), 1e-2), t[-1], n_points)
        # keep every recorded point inside the escapes (steep parts), plus the log grid elsewhere
        steep = np.abs(np.gradient(traj[:, 2])) + np.abs(np.gradient(traj[:, 3])) > 1e-3
        keep = np.unique(np.concatenate([grid, t[steep & (t > 0)]]))
        sample = {
            key: np.interp(keep, t, traj[:, col])
            for col, key in [(1, "loss"), (2, "a1"), (3, "a2")]
        }
        rows.append(pd.DataFrame(dict(case=case, t=keep, **sample)))
    pd.concat(rows).to_csv(
        res / "report_examples.csv", index=False, float_format="%.8g", lineterminator="\n"
    )


# ---- figure 1: which component is learned first -------------------------------------------------


def panel_vs_rho(ax, res: Path) -> None:
    """(a) P(larger first) against rho, sigma = 0.01, d = 2, chi = 4: GF (Wilson 95% intervals) and the
    race of the leading-order blow-up times of the same initializations (lines; dotted for binary4,
    where the race is not the dynamics)."""
    df = pd.read_csv(res / "main_summary.csv")
    df = df[(df.d == 2) & (df.chi == 4)]
    trees = ["binary4", "mps4", "mps6", "binary8", "mps8"]
    for i, tree in enumerate(trees):
        s = df[df.tree == tree].sort_values("rho")
        x = s.rho.to_numpy() + (i - 2) * 0.004  # small offsets so that the intervals do not overlap
        ls = ":" if tree == "binary4" else "-"
        ax.plot(s.rho, s.race_curve, color=COLORS[tree], lw=1.0, ls=ls)
        err = [s.p_greedy - s.ci_lo, s.ci_hi - s.p_greedy]
        ax.errorbar(x, s.p_greedy, yerr=err, fmt="o", color=COLORS[tree], ms=3.2, lw=0.8, capsize=0)
    ax.axhline(0.5, color=tp.MUTED, lw=0.6)
    ax.set_xlim(0.47, 1.01)
    ax.set_ylim(0.44, 1.02)
    ax.set_xlabel(r"weight ratio $w=s_2/s_1$")
    ax.set_ylabel("P(larger component first)")
    ax.set_title(r"(a) $\sigma=0.01$, $d=2$, $\chi=4$")


def panel_vs_sigma(ax, res: Path) -> None:
    """(b) P(smaller first) against sigma at rho = 0.8 (Wilson 95% intervals): m = 2 pairs with the
    fitted power law C sigma^gamma (sigma_fits.csv), m >= 3 pairs joined by thin lines."""
    table = pd.read_csv(res / "sigma_table.csv")
    fits = pd.read_csv(res / "sigma_fits.csv")
    sets = [  # (entity, sweep, chi, m)
        ("binary4", "b4_sigma", 4, 2),
        ("mps4_sep", "pat_mps4_sep", 4, 2),
        ("mps4_adj", "pat_mps4_adj", 4, 2),
        ("mps6_m3", "pat_mps6_m3", 4, 3),
        ("mps4", "m4_sigma", 4, 4),
        ("binary8", "b8_sigma", 4, 4),
    ]
    rho = 0.8
    for entity, sweep, chi, m in sets:
        s = table[(table.sweep == sweep) & (table.chi == chi) & np.isclose(table.rho, rho)]
        s = s[s.p_wrong > 0].sort_values("sigma")  # zero counts cannot be drawn on a log axis
        c = COLORS[entity]
        err = [s.p_wrong - s.ci_lo, s.ci_hi - s.p_wrong]
        ax.errorbar(
            s.sigma, s.p_wrong, yerr=err, fmt="o", color=c, ms=3.2, lw=0.8, capsize=0, zorder=3
        )
        if m == 2:
            f = fits[(fits.sweep == sweep) & (fits.chi == chi) & np.isclose(fits.rho, rho)].iloc[0]
            sig = table[table.sweep == sweep].sigma
            xs = np.geomspace(sig.min(), sig.max(), 50)
            ax.plot(xs, f.C * xs**f.gamma, color=c, lw=1.0)
        else:
            ax.plot(s.sigma, s.p_wrong, color=c, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 0.7)
    ax.set_xlabel(r"initialization scale $\sigma$")
    ax.set_ylabel("P(smaller component first)")
    ax.set_title(r"(b) $w=0.8$, $d=2$, $\chi=4$")


def panel_vs_chi(ax, res: Path) -> None:
    """(c) Uncapped bond dimension: race P(larger first) at rho = 0.8 against chi (lines), and
    gradient flow at chi = 16, 32 (markers, Wilson 95% intervals)."""
    scan = pd.read_csv(res / "chi_scan.csv")
    gf = pd.read_csv(res / "chi_gf.csv")
    for tree in ["mps4", "mps6", "binary8", "mps8"]:
        s = scan[scan.tree == tree].sort_values("chi")
        ax.plot(s.chi, s["race_0.8"], color=COLORS[tree], lw=1.0)
        g = gf[(gf.tree == tree) & np.isclose(gf.rho, 0.8)]
        if len(g):
            err = [g.p_greedy - g.ci_lo, g.ci_hi - g.p_greedy]
            ax.errorbar(
                g.chi, g.p_greedy, yerr=err, fmt="o", color=COLORS[tree], ms=3.2, lw=0.8, capsize=0
            )
    ax.axhline(0.5, color=tp.MUTED, lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], labels=["1", "2", "4", "8", "16", "32"])
    ax.set_ylim(0.44, 1.02)
    ax.set_xlabel(r"bond dimension $\chi$ (uncapped)")
    ax.set_ylabel("P(larger component first)")
    ax.set_title(r"(c) $w=0.8$, $d=2$, uncapped bonds")


def figure_first(res: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.35))
    panel_vs_rho(axes[0], res)
    panel_vs_sigma(axes[1], res)
    panel_vs_chi(axes[2], res)
    handles = [
        Line2D([], [], color=tp.INK, marker="o", ms=3.2, ls="none", label="gradient flow"),
        Line2D([], [], color=tp.INK, lw=1.0, label="LO race / fit"),
    ]
    handles += [
        Line2D([], [], color=COLORS[k], marker="o", ms=3.2, lw=1.0, label=LABELS[k]) for k in COLORS
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=5,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.3,
    )
    tp.savefig(fig, res / "report_order_first")
    plt.close(fig)


# ---- figure 2: after a wrong first escape --------------------------------------------------------

BRANCHES = {  # case: panel title
    "binary4": "(a) rotation: binary4",
    "mps4_adj": r"(b) adding: MPS4, $D=\{1,2\}$",
    "mps4": "(c) kept: MPS4, odeco",
}


def lam_pred(kind: str, rho: float, n_cores: int) -> float:
    """|lambda_min| at the wrong saddle (weights 1, rho): rotation (m = 2, k >= 3) or adding (k = 2)."""
    q = rho ** ((n_cores - 2) / n_cores)
    return (1 - rho) * q if kind == "rotation" else q


def panel_example(ax, ex: pd.DataFrame, case: str) -> None:
    s = ex[ex.case == case]
    c = COLORS[case]
    ax.plot(s.t, s.a1, color=c, lw=1.2, label=r"$a_1$ (larger)")
    ax.plot(s.t, s.a2, color=c, lw=1.2, ls="--", label=r"$a_2$ (smaller)")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("$t$")
    ax.set_title(BRANCHES[case])


def panel_escape_time(ax, res: Path) -> None:
    """(d) Escape time t_1 of the larger component from the wrong saddle, in units of the predicted
    1/|lambda_min|, against ln(1/sigma): every rho of each case, mean over 8 seeds per (rho, sigma).
    Lines: slope 1, intercept from the points at sigma <= 1e-3."""
    sets = [
        ("binary4", "h1b_saddle_b4", 3, "rotation", "o"),
        ("mps4_sep", "h1b_saddle_mps4_sep", 4, "rotation", "s"),
        ("mps4_adj", "h1b_saddle_mps4_adj", 4, "add", "^"),
    ]
    for case, name, n_cores, kind, marker in sets:
        df = pd.read_csv(res / f"{name}.csv")
        df["scaled"] = df.t1 * [lam_pred(kind, r, n_cores) for r in df.rho]
        g = df.groupby(["rho", "sigma"]).scaled.mean().reset_index()
        x = np.log(1 / g.sigma)
        ax.plot(
            x,
            g.scaled,
            marker,
            color=COLORS[case],
            mfc="white" if case == "mps4_sep" else COLORS[case],
            ms=3.4,
            ls="none",
            zorder=3,
        )
        small = g.sigma <= 1e-3
        intercept = np.mean(g.scaled[small] - x[small])
        xs = np.array([x.min() - 0.6, x.max() + 0.6])
        ax.plot(xs, intercept + xs, color=COLORS[case], lw=0.9)
    ax.set_xlabel(r"$\ln(1/\sigma)$")
    ax.set_ylabel(r"$|\lambda_{\min}|\,t_1$")
    ax.set_title(r"(d) escape from the saddle")


def figure_wrong(res: Path) -> None:
    ex = pd.read_csv(res / "report_examples.csv")
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, 1.95))
    for ax, case in zip(axes[:3], BRANCHES):
        panel_example(ax, ex, case)
    axes[0].set_ylabel(r"overlap $a_r(t)$")
    style = [
        Line2D([], [], color=tp.INK, lw=1.2, label=r"$a_1$ (larger)"),
        Line2D([], [], color=tp.INK, lw=1.2, ls="--", label=r"$a_2$ (smaller)"),
    ]
    axes[0].legend(handles=style, loc="center left")
    panel_escape_time(axes[3], res)
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[k],
            marker=m,
            mfc="white" if k == "mps4_sep" else COLORS[k],
            ms=3.4,
            lw=1.0,
            label=LABELS[k],
        )
        for k, m in [("binary4", "o"), ("mps4_sep", "s"), ("mps4_adj", "^"), ("mps4", "")]
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False)
    tp.savefig(fig, res / "report_order_wrong")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--res", type=Path, default=RES_DEFAULT, help="committed tables; figures are written here"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="raw outputs of the order experiment; if given, report_examples.csv is rewritten",
    )
    args = parser.parse_args()
    tp.set_style()
    if args.raw is not None:
        extract_examples(args.raw, args.res)
    figure_first(args.res)
    figure_wrong(args.res)


if __name__ == "__main__":
    main()
