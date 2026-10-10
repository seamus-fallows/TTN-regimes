"""Report figures of Sec. 6 (input covariance and teacher structure), from committed CSVs only.

Two figures at the report's text width in the `ttn.plotting` style, plus three small tables:
  report_contrast   the tree-shape contrast for generic teachers: (a) saddle start, MPS students,
                    escape time rescaled by the predicted slope 1/lambda_e (the report's name for
                    the edge rate gamma_e of the notes; 13 teachers); (b) saddle
                    start, binary and HT students, eps * t_esc (flat = eps^-1); (c) random init, N = 4,
                    lambda_min at the rank-one plateau against sigma; (d) random init, N = 4, second
                    escape against sigma (r2_saddle_runs_all.csv, r2_saddle_fits.csv,
                    r2_random_v2_runs.csv)
  report_mps_laws   MPS-4, random init: the second escape against sigma when the teacher's components
                    overlap (e4a_sigma_v2_runs.csv), when the covariance is misaligned
                    (e2_sigma_v2_runs.csv, e2_sigma_v2_summary.csv) and when the sites are correlated
                    (r2_corr_runs.csv, r2_corr_fits.csv), each with its orthogonal baseline
  report_first_escape.csv  per-seed exponent of the first escape t_1 against sigma in the random-init
                    sweep r2_random (all 9 sigma, and the 3 smallest), mean and range per (teacher, tree)
  report_lambda_scaling.csv  aligned covariance, one-component teachers: per-seed exponent of t_1/2
                    against Lambda s within each eigenvector direction (e1_scaling_lambda_summary.csv)
  report_float_floor_refit.csv  odeco baselines of e4a_sigma and e2_sigma: per-seed power-law exponent
                    of the second escape with and without the runs flagged by check_float_floor.py
Selection rules are those of the analysis scripts: runs on the major rank-one saddle with a resolved
plateau (analyze_law_v2.py, analyze_r2_corr.py), onset events at f = 0.05; misaligned8_20 on mps8 is
timed at its intermediate level (t_int, valid where s_2(0) <= 1.8e-5; analyze_r2_saddle.py).

Run: uv run python experiments/data_teacher/plot_report.py
Reads and writes $DATA_TEACHER_RESULTS (default experiments/data_teacher/results).
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ttn import plotting as tp

RES = Path(os.environ.get("DATA_TEACHER_RESULTS", Path(__file__).parent / "results"))

TREE_COLOR = dict(zip(["mps", "binary", "ht"], tp.palette(3)))
TREE_NAME = {"mps": "MPS", "binary": "binary", "ht": "HT"}
FAMILY_MARKER = {"nonorth": "o", "misaligned": "s", "random": "^"}
FAMILY_NAME = {
    "nonorth": "non-orthogonal CP",
    "misaligned": "misaligned covariance",
    "random": "random TTN teacher",
}
F = 0.05  # onset threshold of the escape event, as a fraction of the teacher's s_2 on the edge
M20_LEVEL_VALID = 1.8e-5  # t_int of misaligned8_20 is valid where s_2(0) is below this
# binary / HT escape times are shown relative to this noise level, at which every seed escapes
EPS_REF = 1e-3


def family(teacher: str) -> str:
    if teacher.startswith("nonorth"):
        return "nonorth"
    if teacher.startswith("misaligned"):
        return "misaligned"
    return "random"


def gmean_sem(x):
    """Geometric mean and the factor exp(s.e. of the mean of log x)."""
    lx = np.log(np.asarray(x, dtype=float))
    if len(lx) < 2:
        return np.exp(lx.mean()), 1.0
    return np.exp(lx.mean()), np.exp(lx.std(ddof=1) / np.sqrt(len(lx)))


# ---- saddle start ------------------------------------------------------------------------------


def saddle_start_times():
    """Escape times of the saddle-start runs: one row per (teacher, n, tree, eps, seed) with t."""
    runs = pd.read_csv(RES / "r2_saddle_runs_all.csv")
    runs["t"] = runs[f"t_esc_{F}"]
    m20 = runs.teacher == "misaligned8_20"
    # misaligned8_20: the f-rule times the next, slower escape; use the intermediate-level event.
    runs.loc[m20, "t"] = np.where(
        runs.loc[m20, "s2_start"] <= M20_LEVEL_VALID, runs.loc[m20, "t_int"], np.nan
    )
    return runs.dropna(subset=["t"])


def predicted_slopes():
    """{(teacher, tree): 1/gamma_e of the realized edge, mean over the fitted seeds} for MPS students."""
    fits = pd.read_csv(RES / "r2_saddle_fits.csv")
    fits = fits[(fits.tree == "mps") & ((fits.f == F) | (fits.teacher == "misaligned8_20"))]
    fits = fits[(fits.teacher != "misaligned8_20") | fits.f.isna()]
    return {(r.teacher, r.tree): r.pred_A_edge for r in fits.itertuples()}


def panel_saddle_mps(ax, runs):
    """gamma_e (t(eps) - t(eps_min)) + ln(1/eps_min) against ln(1/eps): the prediction is y = x."""
    pred = predicted_slopes()
    mps = runs[runs.tree == "mps"]
    for (teacher, n), g in mps.groupby(["teacher", "n"]):
        mean = g.groupby("sigma").t.mean().sort_index()
        x = np.log(1 / mean.index.values)
        y = (mean.values - mean.values[0]) / pred[(teacher, "mps")] + x[0]
        ax.plot(
            x,
            y,
            color=TREE_COLOR["mps"],
            lw=0.8,
            alpha=0.8,
            marker=FAMILY_MARKER[family(teacher)],
            ms=3.2,
            mfc=TREE_COLOR["mps"] if n == 4 else "white",
            mec=TREE_COLOR["mps"],
            mew=0.7,
        )
    lim = np.array([0.0, 19.5])
    ax.plot(lim, lim, color=tp.INK, lw=0.8, ls="--", zorder=0)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"$\ln(1/\varepsilon)$")
    ax.set_ylabel("escape time rescaled\nby the predicted slope")
    ax.set_title("(a) saddle start, MPS students", fontsize=9)


def panel_saddle_binary_ht(ax, runs):
    """eps t_esc relative to its value at eps = 1e-3, per seed, then the geometric mean over seeds
    (per-seed ratios: nonorth8_c0.45 on ht8 has small-eps runs for some seeds only)."""
    for tree in ("binary", "ht"):
        sub = runs[(runs.tree == tree) & (runs.teacher != "misaligned8_20")]
        for (teacher, n), g in sub.groupby(["teacher", "n"]):
            ref = g[g.sigma == EPS_REF].set_index("seed").t
            ratio = g.sigma * g.t / (EPS_REF * g.seed.map(ref))
            gm = ratio.groupby(g.sigma).apply(lambda v: gmean_sem(v)[0]).sort_index()
            ax.plot(
                gm.index.values,
                gm.values,
                color=TREE_COLOR[tree],
                lw=0.8,
                alpha=0.8,
                ls="-" if tree == "binary" else (0, (3, 1.5)),
                marker=FAMILY_MARKER[family(teacher)],
                ms=3.2,
                mfc=TREE_COLOR[tree] if n == 4 else "white",
                mec=TREE_COLOR[tree],
                mew=0.7,
            )
    ax.axhline(1.0, color=tp.INK, lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(0.8, 1.1)
    ax.set_xlabel(r"noise at the saddle $\varepsilon$")
    ax.set_ylabel(r"$\varepsilon\,t_{\rm esc}$ relative to $\varepsilon=10^{-3}$")
    ax.set_title(r"(b) saddle start, binary and HT students", fontsize=9)


# ---- random initialization, N = 4 ----------------------------------------------------------------


def random_init_runs():
    """r2_random runs on the major rank-one saddle with a resolved plateau."""
    runs = pd.read_csv(RES / "r2_random_v2_runs.csv")
    return runs[(runs.saddle == 0) & runs.resolved]


def panel_random(ax, runs, column, ylabel, title, sign=1.0, guide=None):
    """Geometric mean over seeds of sign * column against sigma, one line per (teacher, tree)."""
    for (case, tree), g in runs.groupby(["case", "tree"]):
        stats = g.groupby("sigma")[column].apply(lambda v: gmean_sem(sign * v.values))
        sig = stats.index.values
        mean = np.array([m for m, _ in stats.values])
        ax.plot(
            sig,
            mean,
            color=TREE_COLOR[tree],
            lw=0.8,
            alpha=0.85,
            marker=FAMILY_MARKER[family(case)],
            ms=3.2,
            mfc=TREE_COLOR[tree],
            mec=TREE_COLOR[tree],
            mew=0.7,
        )
    if guide is not None:
        x0, y0, slope, label = guide
        x = np.array([x0, x0 * 30])
        ax.plot(x, y0 * (x / x0) ** slope, color=tp.INK, lw=0.8, ls=":", zorder=0)
        ax.annotate(
            label,
            (x[1], y0 * 30**slope),
            fontsize=8,
            color=tp.INK_SECONDARY,
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"initialization scale $\sigma$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


def fig_contrast():
    runs = saddle_start_times()
    rand = random_init_runs()
    fig, axes = plt.subplots(2, 2, figsize=(tp.TEXT_WIDTH, 0.66 * tp.TEXT_WIDTH))
    panel_saddle_mps(axes[0, 0], runs)
    panel_saddle_binary_ht(axes[0, 1], runs)
    panel_random(
        axes[1, 0],
        rand,
        "lambda_min",
        r"$-\lambda_{\min}$ at the rank-one plateau",
        r"(c) random init, $N=4$: curvature at the plateau",
        sign=-1.0,
        guide=(1.5e-4, 2.5e-5, 1.0, r"$\propto\sigma$"),
    )
    panel_random(
        axes[1, 1],
        rand,
        f"gap_{F}",
        r"second escape $\Delta t_2$",
        r"(d) random init, $N=4$: second escape",
        guide=(3e-4, 3e5, -1.0, r"$\propto\sigma^{-1}$"),
    )
    trees = [
        Line2D([], [], color=TREE_COLOR[t], lw=1.2, label=f"{TREE_NAME[t]} student")
        for t in ("mps", "binary", "ht")
    ]
    families = [
        Line2D([], [], color=tp.MUTED, lw=0, marker=FAMILY_MARKER[f], ms=4, label=FAMILY_NAME[f])
        for f in ("nonorth", "misaligned", "random")
    ]
    sizes = [
        Line2D([], [], color=tp.MUTED, lw=0, marker="o", ms=4, label=r"$N=4$"),
        Line2D([], [], color=tp.MUTED, lw=0, marker="o", ms=4, mfc="white", label=r"$N=8$"),
    ]
    # legend columns fill top to bottom: row 1 = trees and N = 4, row 2 = teacher families and N = 8
    handles = [h for pair in zip(trees + sizes[:1], families + sizes[1:]) for h in pair]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=8)
    tp.savefig(fig, RES / "report_contrast")
    plt.close(fig)


# ---- MPS-4, random init: three ways to break orthogonality -----------------------------------------


def law_panel(ax, groups, title, legend_title):
    """groups: list of (label, runs with columns sigma and gap, slope A of the predicted log law or
    None, is_baseline). Geometric mean over seeds with its standard-error factor; dashed: the
    predicted law anchored at the smallest sigma (the intercept is not predicted)."""
    n_ord = sum(not base for _, _, _, base in groups)
    shades = iter(tp.ordinal_colors(n_ord))
    for label, runs, slope, base in groups:
        color = tp.INK if base else next(shades)
        stats = runs.groupby("sigma").gap.apply(lambda v: gmean_sem(v.values)).sort_index()
        sig = stats.index.values
        gm = np.array([m for m, _ in stats.values])
        fac = np.array([f for _, f in stats.values])
        ax.errorbar(
            sig,
            gm,
            yerr=[gm - gm / fac, gm * fac - gm],
            color=color,
            lw=1.0,
            ms=3.5,
            marker="o",
            label=label,
            capsize=1.5,
            elinewidth=0.7,
        )
        # open markers: some run at this sigma is flagged by the float64-floor scan
        flagged = runs.groupby("sigma").flag.any().reindex(sig).values
        if flagged.any():
            ax.plot(
                sig[flagged],
                gm[flagged],
                ls="none",
                marker="o",
                ms=3.5,
                mfc="white",
                mec=color,
                mew=0.9,
                zorder=5,
            )
        # the predicted laws are asymptotic (sigma -> 0): drawn up to sigma = 0.05 only
        x = np.geomspace(sig[0], min(sig[-1], 0.05), 50)
        if base:
            ax.plot(x, gm[0] * (x / sig[0]) ** -2.0, color=color, lw=0.8, ls="--")
        elif slope is not None:
            ax.plot(x, gm[0] - slope * np.log(x / sig[0]), color=color, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel(r"initialization scale $\sigma$")
    ax.set_title(title, fontsize=9)
    ax.legend(
        title=legend_title,
        fontsize=8,
        title_fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=len(groups),
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.4,
    )


def floor_flags():
    """Run names flagged by the float64-floor scan (check_float_floor.py)."""
    ff = pd.read_csv(RES / "float_floor_check.csv")
    return set(ff[ff.flag].run)


def dense_runs(name):
    """Major-saddle runs with a resolved plateau of a dense sigma sweep, with the float-floor flag."""
    runs = pd.read_csv(RES / name)
    runs = runs[(runs.saddle == 0) & runs.resolved].rename(columns={f"gap_{F}": "gap"}).copy()
    names = [f"{r.case}_{r.tree}_s{r.sigma:g}_seed{int(r.seed)}" for r in runs.itertuples()]
    runs["flag"] = np.isin(names, list(floor_flags()))
    return runs


def fig_mps_laws():
    fig, axes = plt.subplots(1, 3, figsize=(tp.TEXT_WIDTH, 0.39 * tp.TEXT_WIDTH))

    runs = dense_runs("e4a_sigma_v2_runs.csv")
    groups = []
    for c in (0.0, 0.05, 0.2, 0.45):
        g = runs[runs.case == f"overlap{c}"]
        groups.append((f"{c:g}", g, None if c == 0 else g.A_pred.iloc[0], c == 0))
    law_panel(axes[0], groups, "(a) overlapping CP components", r"overlap $\cos\phi$")
    axes[0].set_ylabel(r"second escape $\Delta t_2$")

    runs = dense_runs("e2_sigma_v2_runs.csv")
    summ = pd.read_csv(RES / "e2_sigma_v2_summary.csv")
    summ = summ[summ.f == F].set_index("case")
    groups = []
    for th in (0, 5, 20):
        case = f"theta_deg{th}"
        bound = None if th == 0 else 1.0 / abs(summ.loc[case, "lam_small_sigma"])
        groups.append((f"{th}", runs[runs.case == case], bound, th == 0))
    law_panel(axes[1], groups, "(b) misaligned covariance", "misalignment angle (deg)")

    runs = pd.read_csv(RES / "r2_corr_runs.csv")
    runs = runs[(runs.tree == "mps") & (runs.saddle == "major")].copy()
    runs["flag"] = False  # the scan flags no correlated-input run (float_floor_check.csv, r2_corr)
    fits = pd.read_csv(RES / "r2_corr_fits.csv")
    pred = fits[fits.tree == "mps"].set_index("rho").pred_inc_per_halving
    groups = []
    for rho in (0.0, 0.1, 0.2, 0.3, 0.4):
        inc = pred.get(rho, np.nan)
        slope = None if not np.isfinite(inc) else inc / np.log(2.0)
        groups.append((f"{rho:g}", runs[runs.rho == rho], slope, rho == 0))
    law_panel(axes[2], groups, "(c) correlated sites", r"correlation $c$")
    ticks = [0.002, 0.005, 0.01, 0.02, 0.05]  # sigma spans 1.5 decades only
    axes[2].set_xticks(ticks, [f"{t:g}" for t in ticks])
    tp.savefig(fig, RES / "report_mps_laws")
    plt.close(fig)


# ---- first escape: still sigma^-(K-2) --------------------------------------------------------------


def first_escape_table():
    """Per-seed slope of ln t_1 against ln(1/sigma) in r2_random (all runs, both saddles)."""
    runs = pd.read_csv(RES / "r2_random_v2_runs.csv").dropna(subset=["t1"])
    rows = []
    for (case, tree, seed), g in runs.groupby(["case", "tree", "seed"]):
        g = g.sort_values("sigma")
        x, y = np.log(1 / g.sigma.values), np.log(g.t1.values)
        rows.append(
            dict(
                case=case,
                tree=tree,
                seed=seed,
                n_sigma=len(g),
                p_all=np.polyfit(x, y, 1)[0],
                p_small3=np.polyfit(x[:3], y[:3], 1)[0],
            )
        )
    per_seed = pd.DataFrame(rows)
    out = per_seed.groupby(["case", "tree"]).agg(
        n_seeds=("seed", "size"),
        p_all_mean=("p_all", "mean"),
        p_all_min=("p_all", "min"),
        p_all_max=("p_all", "max"),
        p_small3_mean=("p_small3", "mean"),
        p_small3_sem=("p_small3", "sem"),
    )
    out.to_csv(RES / "report_first_escape.csv")
    return out


def lambda_scaling_table():
    """Aligned covariance, one-component teachers: per-seed slope of ln t_1/2 against ln(Lambda s)
    within each teacher direction (e1: Lambda 1-33; e4: 0.008-0.53). The pooled fit of
    e1_scaling_fits.csv mixes the two directions, whose prefactors differ."""
    runs = pd.read_csv(RES / "e1_scaling_lambda_summary.csv")
    runs["direction"] = runs.teacher.str[:2]
    rows = []
    for (tree, variant, direction), g in runs.groupby(["tree", "variant", "direction"]):
        slopes = [
            np.polyfit(np.log(h.Lambda_0 * h.s_0), np.log(h.t_half_0), 1)[0]
            for _, h in g.groupby("seed")
        ]
        rows.append(
            dict(
                tree=tree,
                variant=variant,
                direction=direction,
                lambda_min=g.Lambda_0.min(),
                lambda_max=g.Lambda_0.max(),
                n_seeds=len(slopes),
                slope_mean=np.mean(slopes),
                slope_sem=np.std(slopes, ddof=1) / np.sqrt(len(slopes)),
                slope_min=np.min(slopes),
                slope_max=np.max(slopes),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(RES / "report_lambda_scaling.csv", index=False)
    return out


def float_floor_refit():
    """Odeco baselines (c = 0 of e4a_sigma, theta = 0 of e2_sigma): per-seed power-law exponent of the
    second escape with and without the runs flagged by the float64-floor scan."""
    rows = []
    for name, case in (
        ("e4a_sigma_v2_runs.csv", "overlap0.0"),
        ("e2_sigma_v2_runs.csv", "theta_deg0"),
    ):
        runs = dense_runs(name)
        runs = runs[runs.case == case]
        p_all, p_clean = [], []
        for _, g in runs.groupby("seed"):
            x, y = np.log(1 / g.sigma.values), np.log(g.gap.values)
            p_all.append(np.polyfit(x, y, 1)[0])
            ok = ~g.flag.values
            if ok.sum() >= 3:
                p_clean.append(np.polyfit(x[ok], y[ok], 1)[0])
        rows.append(
            dict(
                sweep=name.replace("_v2_runs.csv", ""),
                case=case,
                n_runs=len(runs),
                n_flagged=int(runs.flag.sum()),
                sigma_flagged_max=runs[runs.flag].sigma.max(),
                n_seeds=len(p_all),
                p_all_mean=np.mean(p_all),
                p_all_sem=np.std(p_all, ddof=1) / np.sqrt(len(p_all)),
                n_seeds_clean=len(p_clean),
                p_clean_mean=np.mean(p_clean),
                p_clean_sem=np.std(p_clean, ddof=1) / np.sqrt(len(p_clean)),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(RES / "report_float_floor_refit.csv", index=False)
    return out


if __name__ == "__main__":
    tp.set_style()
    fig_contrast()
    fig_mps_laws()
    print(first_escape_table().to_string())
    print(lambda_scaling_table().to_string())
    print(float_floor_refit().to_string())
