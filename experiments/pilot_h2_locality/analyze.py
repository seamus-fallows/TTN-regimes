"""Fits, tables and figures for the H2 locality pilot.

    uv run python experiments/pilot_h2_locality/analyze.py

Reads data/pilot_h2_locality/runs.csv and saddle_runs.csv (one row per run, written by
run_sweep.py) and, for the loss-curve figures, the trajectories in data/pilot_h2_locality/runs/.
Writes results/runs_derived.csv (compact per-run table), results/fits.{csv,md},
results/saddle_fits.{csv,md} and results/fig_*.png.

Definitions (per run): t_r = first time a_r crosses 0.5 (r = 1: weight 1.0, r = 2: weight 0.5);
t_first = min(t1, t2), gap = |t2 - t1| (time from the first to the second learning event; the
order is reversed when t2 < t1). A run is "stuck" if it ended (t_max or wall-clock limit) with one
component unlearned.
Exponents are slopes of log t vs log sigma, fitted per seed; we report mean, std, min, max over seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
DATA = REPO / "data" / "pilot_h2_locality"
RUNS_DIR = DATA / "runs"

TREES = [("mps", 4), ("binary", 4), ("mps", 6), ("binary", 8)]
KIND_ORDER = ["local", "far", "global", "sites0-2", "sites0-3", "sites0-4"]
# Reference categorical palette (dataviz skill), fixed order; color follows the kind everywhere.
KIND_COLOR = dict(
    zip(KIND_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"])
)
KIND_MARKER = dict(zip(KIND_ORDER, ["o", "s", "D", "^", "v", "P"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N_ASYM = 4  # number of smallest sigma values used for the "asymptotic" slope


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def kind_label(kind: str, k: int) -> str:
    name = {"sites0-2": "sites {0,2}", "sites0-3": "sites {0,3}", "sites0-4": "sites {0,4}"}
    return f"{name.get(kind, kind)} (k={k})"


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(DATA / "runs.csv")
    df = df[(df["method"] == "LSODA") & (df["rtol"] == 1e-10)].copy()
    df["npz"] = [str(RUNS_DIR / f"{t}.npz") for t in df["tag"]]
    df["extended"] = False
    ext_path = DATA / "extend_runs.csv"
    if ext_path.exists():  # censored runs rerun with a longer horizon replace the originals
        ext = pd.read_csv(ext_path)
        ext["npz"] = [str(DATA / "extend" / f"{t}.npz") for t in ext["tag"]]
        ext["extended"] = True
        df = pd.concat([df[~df["tag"].isin(ext["tag"])], ext], ignore_index=True)
    df["t1"], df["t2"] = df["t_a1_0.5"], df["t_a2_0.5"]
    df["reversed"] = (df["t2"] < df["t1"]) | (df["t1"].isna() & df["t2"].notna())
    df["t_first"] = np.fmin(df["t1"], df["t2"])
    df["t_second"] = np.where(df["reversed"], df["t1"], df["t2"])
    df["gap"] = df["t_second"] - df["t_first"]
    unlearned_end = np.where(df["reversed"], df["a1_end"], df["a2_end"])
    df["stuck"] = (df["status"] != "converged") & (unlearned_end < 0.5)
    # Unlearned runs: "collapsed" (rule fired, or rank-2 content at round-off), "decaying" (rank-2
    # content still falling over the last decade of time) or "growing" (a slow escape cut off by
    # t_max / the wall-clock limit: censored).
    df["stuck_type"] = ""
    for i, r in df[df["stuck"]].iterrows():
        d = np.load(r.npz)
        t, s2 = d["t"], d["spectra"][:, :, 1].max(axis=1)
        j = np.searchsorted(t, t[-1] / 10)
        if r.status == "collapsed" or s2[-1] <= 1e-12:
            df.loc[i, "stuck_type"] = "collapsed"
        else:
            df.loc[i, "stuck_type"] = "growing" if s2[-1] > s2[j] else "decaying"
    # Intermediate plateau: time from the first component being learned (a = 0.9) to the second
    # being half learned (a = 0.5), compared with the width of the first transition (a: 0.1 -> 0.9).
    # (Crossings of a = 0.1 are not used for the second component: it can overshoot 0.1 transiently
    # during the first escape and decay again.)
    first_01 = np.where(df["reversed"], df["t_a2_0.1"], df["t_a1_0.1"])
    first_09 = np.where(df["reversed"], df["t_a2_0.9"], df["t_a1_0.9"])
    df["plateau_len"] = df["t_second"] - first_09
    df["has_mid_plateau"] = df["plateau_len"] > 2 * (first_09 - first_01)
    # Loss level of the intermediate plateau (at the geometric-mean time between the two events;
    # expected 0.125 = s2^2/2, or 0.5 = s1^2/2 if component 2 was learned first).
    df["L_mid"] = np.nan
    for i, r in df[df["has_mid_plateau"]].iterrows():
        d = np.load(r.npz)
        df.loc[i, "L_mid"] = np.interp(np.sqrt(r.t_first * r.t_second), d["t"], d["loss"])
    return df


def per_seed_slopes(sub: pd.DataFrame, ycol: str, asym: bool) -> np.ndarray:
    out = []
    for _, g in sub.groupby("seed"):
        g = g[np.isfinite(g[ycol]) & (g[ycol] > 0)].sort_values("sigma")
        if asym:
            g = g.head(N_ASYM)
        if len(g) >= 3:
            out.append(np.polyfit(np.log(g["sigma"]), np.log(g[ycol]), 1)[0])
    return np.array(out)


def per_seed_log_fit(sub: pd.DataFrame, asym: bool = False) -> np.ndarray:
    """Slope of gap vs ln(1/sigma), per seed (the k = 2 prediction is linear in ln(1/sigma)).
    asym: use only the N_ASYM smallest sigma values."""
    out = []
    for _, g in sub.groupby("seed"):
        g = g[np.isfinite(g["gap"])].sort_values("sigma")
        if asym:
            g = g.head(N_ASYM)
        if len(g) >= 3:
            out.append(np.polyfit(np.log(1 / g["sigma"]), g["gap"], 1)[0])
    return np.array(out)


def pooled_slope(sub: pd.DataFrame, ycol: str) -> tuple[float, float]:
    """Common slope with one intercept per seed (fixed effects), and its standard error."""
    xs, ys = [], []
    for _, g in sub.groupby("seed"):
        g = g[np.isfinite(g[ycol]) & (g[ycol] > 0)]
        if len(g) >= 3:
            x, y = np.log(g["sigma"].to_numpy()), np.log(g[ycol].to_numpy())
            xs.append(x - x.mean())
            ys.append(y - y.mean())
    if not xs:
        return np.nan, np.nan
    x, y = np.concatenate(xs), np.concatenate(ys)
    slope = (x @ y) / (x @ x)
    dof = len(x) - len(xs) - 1
    se = np.sqrt(((y - slope * x) ** 2).sum() / max(dof, 1) / (x @ x))
    return slope, se


def stats(a: np.ndarray) -> dict:
    if len(a) == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, n=0)
    return dict(
        mean=a.mean(), std=a.std(ddof=1) if len(a) > 1 else 0.0, min=a.min(), max=a.max(), n=len(a)
    )


def prediction_gap(k: int) -> str:
    if k == 1:
        return "no plateau"
    if k == 2:
        return "log(1/sigma)"
    return f"{-(k - 2)}"


def gap_fit_runs(esc: pd.DataFrame, k: int) -> pd.DataFrame:
    """Runs used to fit the gap exponent. For k >= 3 only runs with a distinct intermediate plateau
    (the second component starts after the first is learned): at large sigma the two escapes can
    overlap, and their time difference is then not a second-escape time (pre-asymptotic)."""
    if k >= 3 and "has_mid_plateau" in esc:
        return esc[esc["has_mid_plateau"]]
    return esc


def per_seed_table(df: pd.DataFrame, time_col: str = "gap") -> pd.DataFrame:
    """One row per (tree, kind, seed): how many sigma values escaped / got stuck, the order of
    learning, and the per-seed exponents."""
    rows = []
    for (tree, n, kind, seed), g in df.groupby(["tree", "n", "kind", "seed"]):
        esc = g[~g["stuck"]].sort_values("sigma")
        row = dict(
            tree=f"{tree}{n}",
            kind=kind,
            k=int(g["k_pred"].iloc[0]),
            seed=int(seed),
            n_sigma=len(g),
            n_stuck=int(g["stuck"].sum()),
            stuck_sigmas="-".join(f"{x:g}" for x in sorted(g.loc[g["stuck"], "sigma"])),
        )
        if "reversed" in g:
            row["n_reversed"] = int((g["reversed"] & ~g["stuck"]).sum())
        for col, name in (("t_first", "t1"), (time_col, "gap")):
            if col not in esc or (col == "t_first" and time_col != "gap"):
                continue
            src = gap_fit_runs(esc, row["k"]) if name == "gap" else esc
            ok = src[np.isfinite(src[col]) & (src[col] > 0)]
            for label, sub in (("", ok), ("_asym", ok.head(N_ASYM))):
                row[f"slope_{name}{label}"] = (
                    np.polyfit(np.log(sub["sigma"]), np.log(sub[col]), 1)[0]
                    if len(sub) >= 3
                    else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tree, n), _ in [((t, n), None) for t, n in TREES]:
        for kind in KIND_ORDER:
            sub = df[(df["tree"] == tree) & (df["n"] == n) & (df["kind"] == kind)]
            if sub.empty:
                continue
            k, n_cores = int(sub["k_pred"].iloc[0]), int(sub["n_cores"].iloc[0])
            esc = sub[~sub["stuck"]]
            row = dict(
                tree=f"{tree}{n}",
                K=n_cores,
                kind=kind,
                k=k,
                n_runs=len(sub),
                n_seeds=sub["seed"].nunique(),
                sigma_min=sub["sigma"].min(),
                sigma_max=sub["sigma"].max(),
                n_stuck=int(sub["stuck"].sum()),
                n_collapsed=int((sub["stuck_type"] == "collapsed").sum()),
                n_decaying=int((sub["stuck_type"] == "decaying").sum()),
                n_growing=int((sub["stuck_type"] == "growing").sum()),
                n_reversed=int((sub["reversed"] & ~sub["stuck"]).sum()),
                pred_t1=-(n_cores - 2),
                pred_gap=prediction_gap(k),
            )
            gap_runs = gap_fit_runs(esc, k)
            row["n_gap_excluded"] = len(esc) - len(gap_runs)
            for name, col, runs in (("t1", "t_first", esc), ("gap", "gap", gap_runs)):
                for asym, suffix in ((False, ""), (True, "_asym")):
                    s = stats(per_seed_slopes(runs, col, asym))
                    row.update({f"slope_{name}{suffix}_{key}": v for key, v in s.items()})
                p, se = pooled_slope(runs, col)
                row[f"slope_{name}_pooled"], row[f"slope_{name}_pooled_se"] = p, se
            lin = stats(per_seed_log_fit(esc))
            row.update({f"gap_vs_ln_{key}": v for key, v in lin.items()})
            lin = stats(per_seed_log_fit(esc, asym=True))
            row.update({f"gap_vs_ln_asym_{key}": v for key, v in lin.items()})
            row["median_gap"] = esc["gap"].median()
            row["median_gap_over_t1"] = (esc["gap"] / esc["t_first"]).median()
            row["max_abs_gap"] = esc["gap"].abs().max()
            row["frac_mid_plateau"] = esc["has_mid_plateau"].mean()
            row["median_wall_s"] = sub["wall_s"].median()
            row["max_wall_s"] = sub["wall_s"].max()
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(m, s, digits=2):
    if not np.isfinite(m):
        return "-"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def markdown_table(fits: pd.DataFrame) -> str:
    lines = [
        "| tree | K | kind | k | runs (no 2nd escape: collapsed/decaying/growing) | t_first: pred | "
        f"t_first: fit (all σ) | t_first: fit (smallest {N_ASYM} σ) | gap: pred | gap: fit (all σ) | "
        f"gap: fit (smallest {N_ASYM} σ) | runs excluded from gap fit | gap vs ln(1/σ) | "
        "median gap / t_first |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in fits.iterrows():
        stuck = (
            f"{r.n_runs} ({r.n_collapsed}/{r.n_decaying}/{r.n_growing})"
            if r.n_stuck
            else f"{r.n_runs} (0)"
        )
        lin = (
            f"{fmt(r.gap_vs_ln_mean, r.gap_vs_ln_std)} (smallest {N_ASYM} σ: "
            f"{fmt(r.gap_vs_ln_asym_mean, r.gap_vs_ln_asym_std)})"
            if r.k <= 2
            else ""
        )
        gap_all = fmt(r.slope_gap_mean, r.slope_gap_std)
        gap_asym = fmt(r.slope_gap_asym_mean, r.slope_gap_asym_std)
        if r.k == 1:  # gap ~ 0: a log-log slope is not meaningful
            gap_all = gap_asym = f"max abs(gap) = {r.max_abs_gap:.2g}"
        lines.append(
            f"| {r.tree} | {r.K} | {r.kind} | {r.k} | {stuck} | {r.pred_t1} | "
            f"{fmt(r.slope_t1_mean, r.slope_t1_std)} | {fmt(r.slope_t1_asym_mean, r.slope_t1_asym_std)} | "
            f"{r.pred_gap} | {gap_all} | {gap_asym} | {r.n_gap_excluded} | {lin} | "
            f"{r.median_gap_over_t1:.3g} |"
        )
    return "\n".join(lines)


# ---- figures ------------------------------------------------------------------------------------


def _median_curve(sub: pd.DataFrame, col: str):
    g = sub[np.isfinite(sub[col]) & (sub[col] > 0)].groupby("sigma")[col]
    med = g.median()
    count = g.count()
    return med[count >= max(1, sub["seed"].nunique() // 2)]


def fig_scaling(df: pd.DataFrame, col: str, fname: str, ylabel: str, k2_coef: float = 1.0):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11.5))
    for ax, (tree, n) in zip(axes.flat, TREES):
        sub_t = df[(df["tree"] == tree) & (df["n"] == n)]
        if sub_t.empty:
            ax.set_visible(False)
            continue
        n_cores = int(sub_t["n_cores"].iloc[0])
        y_top = 0
        for kind in KIND_ORDER:
            sub = sub_t[sub_t["kind"] == kind]
            if sub.empty:
                continue
            k = int(sub["k_pred"].iloc[0])
            esc = sub[~sub["stuck"]]
            c, m = KIND_COLOR[kind], KIND_MARKER[kind]
            if col == "gap" and k == 1:
                ax.scatter(
                    esc["sigma"],
                    esc["gap"].abs(),
                    s=14,
                    facecolors="none",
                    edgecolors=c,
                    marker=m,
                    lw=0.9,
                    label=f"{kind_label(kind, k)}: |gap|",
                )
                continue
            ax.scatter(esc["sigma"], esc[col], s=10, color=c, alpha=0.35, marker=m, lw=0)
            med = _median_curve(esc, col)
            ax.plot(
                med.index, med.values, color=c, lw=1.6, marker=m, ms=5, label=kind_label(kind, k)
            )
            if len(esc):
                y_top = max(y_top, np.nanmax(esc[col]))
            stuck = sub[sub["stuck"]]
            if col == "gap" and len(stuck):
                lower = stuck["t_end"] - stuck["t_first"]
                ax.scatter(
                    stuck["sigma"],
                    lower,
                    marker=r"$\uparrow$",
                    s=60,
                    color=c,
                    lw=0,
                    label=f"{kind_label(kind, k)}: stuck (gap > shown)",
                )
            # reference slope anchored at the median at the smallest sigma
            if len(med) >= 2 and (col == "t_first" or k >= 2):
                s_ref = np.geomspace(med.index.min(), med.index.max(), 20)
                x0, y0 = med.index.min(), med.loc[med.index.min()]
                if col == "t_first":
                    ref = y0 * (s_ref / x0) ** (-(n_cores - 2))
                elif k >= 3:
                    ref = y0 * (s_ref / x0) ** (-(k - 2))
                else:  # k = 2: gap = A ln(1/sigma) + B, slope A = (1 - s2/s1)/s2 = 1 (balanced estimate)
                    ref = y0 + k2_coef * (np.log(1 / s_ref) - np.log(1 / x0))
                # drawn a factor 3 below the data so that it does not hide under the median line
                if col == "gap":
                    ax.plot(s_ref, ref / 3, color=c, lw=0.9, ls=(0, (4, 3)), alpha=0.9)
                elif kind == "local":
                    ax.plot(
                        s_ref,
                        ref / 3,
                        color=MUTED,
                        lw=1.0,
                        ls=(0, (4, 3)),
                        label=f"slope −{n_cores - 2} (predicted)",
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("per-core init scale σ", color=INK, fontsize=9)
        sig = sorted(sub_t["sigma"].unique())
        ax.set_xticks(sig)
        ax.set_xticklabels([f"{x:g}" for x in sig], fontsize=7)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        title = f"{tree.upper() if tree == 'mps' else 'binary'} N={n} (K={n_cores} cores)"
        if col == "t_first":
            title += f": predicted slope −{n_cores - 2}"
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        style_axes(ax)
        ax.legend(
            fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2
        )
    note = (
        f"dashed: predicted slope (k≥3: −(k−2); k=2: {k2_coef:g}·ln(1/σ)), anchored at the "
        "smallest σ and shifted down 3x.\nPoints: runs; lines: median over the seeds that escape; "
        "arrows: runs without a second escape (lower bound)."
    )
    if col == "t_first":
        note = (
            "dashed: predicted slope −(K−2), anchored at the smallest σ, shifted down 3x. "
            "Points: runs; lines: median over seeds."
        )
    fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


def fig_kscan(fits: pd.DataFrame, fname: str, saddle_fits: pd.DataFrame | None = None):
    """Fitted second-escape exponent vs k. Filled: from small init (gap); hollow: saddle start."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ks = np.arange(3, 8)
    ax.plot(ks, -(ks - 2), color=MUTED, lw=1.2, ls=(0, (4, 3)), label="H2: −(k−2)")
    tree_style = {
        "mps4": ("#2a78d6", "o"),
        "binary4": ("#eb6834", "s"),
        "mps6": ("#1baf7a", "D"),
        "binary8": ("#eda100", "^"),
    }
    offsets = {"mps4": -0.21, "binary4": -0.07, "mps6": 0.07, "binary8": 0.21}
    sources = [(fits, "slope_gap_asym", True, "from small init")]
    if saddle_fits is not None:
        sources.append((saddle_fits, "slope_esc_asym", False, "saddle start"))
    for tree, (c, m) in tree_style.items():
        for table, col, filled, what in sources:
            sub = table[(table["tree"] == tree) & (table["k"] >= 3)]
            if sub.empty:
                continue
            dx = offsets[tree] + (0 if filled else 0.035)
            ok = np.isfinite(sub[f"{col}_mean"])
            ax.errorbar(
                sub["k"][ok] + dx,
                sub[f"{col}_mean"][ok],
                yerr=sub[f"{col}_std"][ok],
                color=c,
                marker=m,
                ms=6,
                lw=0,
                elinewidth=1.1,
                capsize=2.5,
                mfc=c if filled else "white",
                mec=c,
                label=f"{tree}, {what}",
            )
            for _, r in sub.iterrows():
                if r.n_stuck > 0 and filled:  # saddle-start counts: see saddle_fits.md
                    y = r[f"{col}_mean"] if np.isfinite(r[f"{col}_mean"]) else -(r.k - 2)
                    ax.annotate(
                        f"{r.n_stuck}/{r.n_runs} stuck",
                        (r.k + dx, y),
                        xytext=(0, -16 if r.tree in ("mps4", "mps6") else 12),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6.5,
                        color=INK,
                    )
                    if not np.isfinite(r[f"{col}_mean"]):
                        ax.scatter([r.k + dx], [y], marker="x", s=40, color=c, lw=1.4)
    ax.set_xlabel("k = number of cores that must acquire a new direction", color=INK, fontsize=9)
    ax.set_ylabel(
        f"fitted exponent of second-escape time vs σ\n(smallest {N_ASYM} σ, mean ± std over seeds)",
        color=INK,
        fontsize=9,
    )
    ax.set_title(
        "Second-escape exponent vs k (labels: runs from small init with no second escape)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    style_axes(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


def load_saddle() -> pd.DataFrame | None:
    path = DATA / "saddle_runs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["t_escape"] = df["t_a2_0.5"]  # component 1 is learned at t = 0
    df["stuck"] = (df["status"] != "converged") & (df["a2_end"] < 0.5)
    return df


def saddle_fit_table(sd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tree, n in TREES:
        for kind in KIND_ORDER:
            sub = sd[(sd["tree"] == tree) & (sd["n"] == n) & (sd["kind"] == kind)]
            if sub.empty:
                continue
            k = int(sub["k_pred"].iloc[0])
            esc = sub[~sub["stuck"]].copy()
            esc["gap"] = esc["t_escape"]
            row = dict(
                tree=f"{tree}{n}",
                K=int(sub["n_cores"].iloc[0]),
                kind=kind,
                k=k,
                n_runs=len(sub),
                n_stuck=int(sub["stuck"].sum()),
                sigma_min=sub["sigma"].min(),
                sigma_max=sub["sigma"].max(),
                pred=prediction_gap(k),
            )
            for asym, suffix in ((False, ""), (True, "_asym")):
                st = stats(per_seed_slopes(esc, "t_escape", asym))
                row.update({f"slope_esc{suffix}_{key}": v for key, v in st.items()})
            lin = stats(per_seed_log_fit(esc))
            row.update({f"esc_vs_ln_{key}": v for key, v in lin.items()})
            lin = stats(per_seed_log_fit(esc, asym=True))
            row.update({f"esc_vs_ln_asym_{key}": v for key, v in lin.items()})
            row["median_t_escape"] = esc["t_escape"].median()
            row["median_wall_s"] = sub["wall_s"].median()
            rows.append(row)
    return pd.DataFrame(rows)


def saddle_markdown(sf: pd.DataFrame) -> str:
    lines = [
        "| tree | kind | k | runs (stuck) | pred | escape-time exponent (all σ) | "
        f"(smallest {N_ASYM} σ) | t_esc vs ln(1/σ) | median t_esc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sf.iterrows():
        lin = (
            f"{fmt(r.esc_vs_ln_mean, r.esc_vs_ln_std)} (smallest {N_ASYM} σ: "
            f"{fmt(r.esc_vs_ln_asym_mean, r.esc_vs_ln_asym_std)})"
            if r.k <= 2
            else ""
        )
        lines.append(
            f"| {r.tree} | {r.kind} | {r.k} | {r.n_runs} ({r.n_stuck}) | {r.pred} | "
            f"{fmt(r.slope_esc_mean, r.slope_esc_std)} | "
            f"{fmt(r.slope_esc_asym_mean, r.slope_esc_asym_std)} | {lin} | "
            f"{r.median_t_escape:.3g} |"
        )
    return "\n".join(lines)


def fig_local(df: pd.DataFrame, fname: str):
    """k = 2 (MPS local): gap vs ln(1/sigma), per seed. k = 1 (binary local): |gap| vs sigma."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for (tree, n), c in ((("mps", 4), "#2a78d6"), (("mps", 6), "#1baf7a")):
        sub = df[(df["tree"] == tree) & (df["n"] == n) & (df["kind"] == "local") & ~df["stuck"]]
        for i, (_, g) in enumerate(sub.groupby("seed")):
            g = g.sort_values("sigma")
            ax.plot(
                np.log(1 / g["sigma"]),
                g["gap"],
                color=c,
                lw=1.0,
                marker="o",
                ms=3,
                alpha=0.8,
                label=f"MPS N={n} (one line per seed)" if i == 0 else None,
            )
    xs = np.linspace(*ax.get_xlim(), 10)
    ax.plot(
        xs,
        xs - xs[0] + ax.get_ylim()[0],
        color=MUTED,
        lw=1,
        ls=(0, (4, 3)),
        label="unit slope (balanced estimate (1−s₂/s₁)/s₂)",
    )
    ax.set_xlabel("ln(1/σ)", color=INK, fontsize=9)
    ax.set_ylabel("gap = t₂ − t₁", color=INK, fontsize=9)
    ax.set_title("'local' on MPS (k = 2): gap vs ln(1/σ)", color=INK, fontsize=10, loc="left")
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    n_points = 0
    for (tree, n), c in ((("binary", 4), "#eb6834"), (("binary", 8), "#eda100")):
        sub = df[(df["tree"] == tree) & (df["n"] == n) & (df["kind"] == "local") & ~df["stuck"]]
        rel = (sub["gap"] / sub["t_first"]).abs()
        ok = np.isfinite(rel) & (rel > 0)
        n_points += int(ok.sum())
        ax.scatter(sub["sigma"][ok], rel[ok], s=14, color=c, lw=0, label=f"binary N={n}")
    if n_points:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("σ", color=INK, fontsize=9)
    ax.set_ylabel("|t₂ − t₁| / t_first", color=INK, fontsize=9)
    ax.set_title("'local' on binary tree (k = 1): relative gap", color=INK, fontsize=10, loc="left")
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


def _sigma_colors(sigmas):
    ramp = [
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    ]
    idx = np.linspace(0, len(ramp) - 1, len(sigmas)).round().astype(int)
    # smallest sigma = darkest
    return {s: ramp[i] for s, i in zip(sorted(sigmas, reverse=True), idx)}


def fig_loss_curves(df: pd.DataFrame, fname: str, seed: int = 0):
    kinds = ["local", "far", "global"]
    fig, axes = plt.subplots(len(TREES), len(kinds), figsize=(12, 11), sharey=True)
    for i, (tree, n) in enumerate(TREES):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            sub = df[
                (df["tree"] == tree) & (df["n"] == n) & (df["kind"] == kind) & (df["seed"] == seed)
            ]
            if sub.empty:
                ax.set_visible(False)
                continue
            colors = _sigma_colors(sub["sigma"].unique())
            for _, r in sub.sort_values("sigma", ascending=False).iterrows():
                path = Path(r.npz)
                if not path.exists():
                    continue
                d = np.load(path)
                t, loss = d["t"][1:], d["loss"][1:]
                ax.plot(t, loss, color=colors[r.sigma], lw=1.3, label=f"σ={r.sigma:g}")
            ax.set_xscale("log")
            for lev in (0.625, 0.125):
                ax.axhline(lev, color=GRID, lw=0.8)
            k = int(sub["k_pred"].iloc[0])
            ax.set_title(
                f"{tree}{n} '{kind}' (k={k}), seed {seed}", color=INK, fontsize=9, loc="left"
            )
            ax.set_xlabel("t", color=INK, fontsize=8)
            if j == 0:
                ax.set_ylabel("population loss", color=INK, fontsize=8)
            style_axes(ax)
            if j == len(kinds) - 1:
                ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.text(
        0.01,
        0.003,
        "gray lines: L = 0.625 (start) and 0.125 (component 1 learned, component 2 "
        "not). Darker = smaller σ.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(RESULTS / fname, dpi=130)
    plt.close(fig)


def fig_bond_s2(df: pd.DataFrame, fname: str, cases: list[tuple[str, int, str, float, int]]):
    """Second singular value of W across every bond vs time: growth (escape) or collapse to rank 1."""
    tags = set(df["tag"])
    cases = [c for c in cases if f"{c[0]}{c[1]}_{c[2]}_sig{c[3]:.6g}_seed{c[4]}" in tags]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.5), squeeze=False)
    edge_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for ax, (tree, n, kind, sigma, seed) in zip(axes[0], cases):
        tag = f"{tree}{n}_{kind}_sig{sigma:.6g}_seed{seed}"
        r = df[df["tag"] == tag].iloc[0]
        d = np.load(r.npz)
        t, s2 = d["t"][1:], d["spectra"][1:, :, 1]
        for e in range(s2.shape[1]):
            ax.plot(
                t,
                np.maximum(s2[:, e], 1e-18),
                color=edge_colors[e % 6],
                lw=1.2,
                label=f"bond {d['edges'][e]}",
            )
        ax.axvline(r.t_first, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-18, 1)
        status = "stuck" if r.stuck else f"escapes at t={r.t_second:.3g}"
        ax.set_title(
            f"{tree}{n} '{kind}' (k={r.k_pred}) σ={sigma:g}\nseed {seed}: {status}",
            color=INK,
            fontsize=8.5,
            loc="left",
        )
        ax.set_xlabel("t (gray line: t_first)", color=INK, fontsize=8)
        style_axes(ax)
    axes[0, 0].set_ylabel("2nd singular value of W across bond", color=INK, fontsize=8)
    axes[0, -1].legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=140)
    plt.close(fig)


def fig_loss_aligned(df: pd.DataFrame, fname: str, seed: int = 0):
    """Loss vs t - t_first for the 'local' teacher: one plateau (binary) or two (MPS)?"""
    fig, axes = plt.subplots(1, len(TREES), figsize=(13, 3.6), sharey=True)
    for ax, (tree, n) in zip(axes, TREES):
        sub = df[
            (df["tree"] == tree) & (df["n"] == n) & (df["kind"] == "local") & (df["seed"] == seed)
        ]
        colors = _sigma_colors(sub["sigma"].unique())
        for _, r in sub.sort_values("sigma", ascending=False).iterrows():
            path = Path(r.npz)
            if not path.exists() or not np.isfinite(r.t_first):
                continue
            d = np.load(path)
            ax.plot(
                d["t"] - r.t_first, d["loss"], color=colors[r.sigma], lw=1.3, label=f"σ={r.sigma:g}"
            )
        ax.axhline(0.125, color=GRID, lw=0.8)
        ax.set_xlim(-4, 16)
        k = int(sub["k_pred"].iloc[0]) if len(sub) else 0
        ax.set_title(f"{tree}{n} 'local' (k={k}), seed {seed}", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("t − t_first", color=INK, fontsize=8)
        style_axes(ax)
    axes[0].set_ylabel("population loss", color=INK, fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=130)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_runs()
    cols = [
        "tag",
        "tree",
        "n",
        "n_cores",
        "kind",
        "k_pred",
        "sigma",
        "seed",
        "status",
        "t1",
        "t2",
        "t_first",
        "gap",
        "reversed",
        "stuck",
        "stuck_type",
        "extended",
        "plateau_len",
        "has_mid_plateau",
        "L_mid",
        "t_end",
        "loss_end",
        "a1_end",
        "a2_end",
        "s2_end_max",
        "q_drift",
        "nfev",
        "n_steps",
        "wall_s",
    ]
    df[[c for c in cols if c in df]].to_csv(
        RESULTS / "runs_derived.csv", index=False, float_format="%.8g"
    )
    per_seed_table(df).to_csv(RESULTS / "per_seed.csv", index=False, float_format="%.4g")
    fits = fit_table(df)
    fits.to_csv(RESULTS / "fits.csv", index=False, float_format="%.4g")
    (RESULTS / "fits.md").write_text(markdown_table(fits) + "\n", encoding="utf-8")
    print(markdown_table(fits))
    fig_scaling(df, "t_first", "fig_t_first.png", "t_first (first component learned)")
    fig_scaling(df, "gap", "fig_gap.png", "gap = t_second − t_first")
    sd = load_saddle()
    saddle_fits = None
    if sd is not None:
        saddle_fits = saddle_fit_table(sd)
        sd_ps = sd.assign(gap=sd["t_escape"])
        per_seed_table(sd_ps, time_col="gap").drop(
            columns=["slope_t1", "slope_t1_asym"], errors="ignore"
        ).to_csv(RESULTS / "saddle_per_seed.csv", index=False, float_format="%.4g")
        saddle_fits.to_csv(RESULTS / "saddle_fits.csv", index=False, float_format="%.4g")
        (RESULTS / "saddle_fits.md").write_text(
            saddle_markdown(saddle_fits) + "\n", encoding="utf-8"
        )
        print("\nsaddle start:")
        print(saddle_markdown(saddle_fits))
        sd2 = sd.copy()
        sd2["gap"], sd2["t_first"] = sd2["t_escape"], 0.0
        fig_scaling(
            sd2, "gap", "fig_saddle_escape.png", "escape time from the rank-one saddle", k2_coef=2.0
        )
    fig_kscan(fits, "fig_kscan.png", saddle_fits)
    fig_local(df, "fig_local.png")
    fig_loss_curves(df, "fig_loss_curves.png")
    fig_loss_aligned(df, "fig_loss_local_aligned.png")
    fig_bond_s2(
        df,
        "fig_bond_s2.png",
        [
            ("mps", 4, "far", 0.01, 0),
            ("mps", 4, "far", 0.01, 1),
            ("mps", 6, "far", 0.1, 0),
            ("mps", 6, "far", 0.1, 2),
            ("binary", 8, "global", 0.15, 1),
        ],
    )
    # Text summary of surprises
    print("\nreversed order (component 2 learned first), escaped runs only:")
    rev = df[df["reversed"] & ~df["stuck"]]
    print(
        rev.groupby(["tree", "n", "kind"])["seed"].agg(lambda s: sorted(set(s))).to_string()
        if len(rev)
        else "  none"
    )
    print("\nstuck runs (ended with one component unlearned):")
    st = df[df["stuck"]].assign(t_end_over_t_first=lambda x: x["t_end"] / x["t_first"])
    print(
        st.groupby(["tree", "n", "kind", "stuck_type"])
        .agg(
            n=("seed", "size"),
            seeds=("seed", lambda s: sorted(set(s))),
            s2_end_max=("s2_end_max", "max"),
            min_t_end_over_t_first=("t_end_over_t_first", "min"),
        )
        .to_string()
        if len(st)
        else "  none"
    )
    if len(st):
        print("stuck sigmas per seed, and t_end / t_first (how long the run was followed):")
        for (tree, n, kind), g in st.groupby(["tree", "n", "kind"]):
            n_sig = df[(df["tree"] == tree) & (df["n"] == n) & (df["kind"] == kind)][
                "sigma"
            ].nunique()
            per_seed = {int(s): sorted(gg["sigma"].tolist()) for s, gg in g.groupby("seed")}
            ratio = (g["t_end"] / g["t_first"]).min()
            print(
                f"  {tree}{n} {kind}: {n_sig} sigma values; {per_seed}; min t_end/t_first = {ratio:.3g}"
            )
    print("\nintermediate-plateau loss level (runs with a distinct plateau):")
    mid = df[df["has_mid_plateau"]]
    # only plateaus long compared with the O(1) width of a transition
    long_mid = mid[mid["plateau_len"] > 20]
    odd = long_mid[
        ((long_mid["L_mid"] - 0.125).abs() > 0.01) & ((long_mid["L_mid"] - 0.5).abs() > 0.01)
    ]
    print(
        mid.groupby(["tree", "n", "kind"])["L_mid"]
        .agg(["min", "median", "max", "count"])
        .to_string()
    )
    print("runs with a plateau longer than 20 not at 0.125 or 0.5:", "none" if odd.empty else "")
    if len(odd):
        print(odd[["tag", "L_mid", "t_first", "t_second"]].to_string())
    print(
        "loss at the end of stuck runs:",
        df.loc[df["stuck"], "loss_end"].round(4).value_counts().to_dict(),
    )
    print("\nnon-converged, non-stuck runs:")
    other = df[(df["status"] != "converged") & ~df["stuck"]]
    print(
        other[["tag", "status", "loss_end", "a1_end", "a2_end"]].to_string()
        if len(other)
        else "  none"
    )
    print(
        f"\nconservation drift: max {df['q_drift'].max():.2e}; wall time per run: median "
        f"{df['wall_s'].median():.1f}s, max {df['wall_s'].max():.1f}s"
    )


if __name__ == "__main__":
    main()
