"""Figure functions for the regime map (read saved runs via analysis.load). Run them through
make_figures.py (--out DIR, --force); existing files are never overwritten without --force.
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import analysis  # noqa: E402
from analysis import K_THRESHOLDS, REGIMES, RHO_THRESHOLDS, boundary, fractions, load  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")  # default output dir (set by --out)
FORCE = False  # --force: allow overwriting existing outputs


def _target(name):
    path = os.path.join(RES, name)
    if os.path.exists(path) and not FORCE:
        raise FileExistsError(f"{path} exists; pass --out <new dir> or --force")
    os.makedirs(RES, exist_ok=True)
    return path


def save_fig(fig, name):
    fig.savefig(_target(name), bbox_inches="tight")
    plt.close(fig)


def save_csv(df, name):
    df.to_csv(_target(name), index=False)


# categorical slots of the reference palette (validated: first three all-pairs), stalled = neutral
COLORS = {"lazy": "#2a78d6", "s2s": "#eb6834", "rich": "#1baf7a", "stalled": "#b9b8b2"}
NAMES = {
    "lazy": "lazy",
    "s2s": "saddle-to-saddle",
    "rich": "rich, non-incremental",
    "stalled": "stalled",
}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 160,
    }
)


def _edges(vals):
    v = np.log(np.asarray(vals, float))
    mid = (v[1:] + v[:-1]) / 2
    return np.exp(np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]]))


def phase_panel(
    ax, df, ycol, label_col="label", hatch_lazy_stall=True, xcol="sigma", ylabels=None, logx=True
):
    """Categorical map: cell colour = majority label; text = majority count / seeds when not
    unanimous; hatched = stalled with delta_K < 0.1 (kernel-like stall). Rows are the sorted values
    of `ycol` (categorical axis), columns the sorted values of `xcol` (log-spaced edges)."""
    xs = np.sort(df[xcol].unique())
    ys = np.sort(df[ycol].unique()) if ylabels is None else list(ylabels)
    xe = (
        _edges(xs)
        if logx
        else np.concatenate([[xs[0] - 0.5], (xs[1:] + xs[:-1]) / 2, [xs[-1] + 0.5]])
    )
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            sub = df[(df[xcol] == x) & (df[ycol] == y)]
            if sub.empty:
                continue
            counts = sub[label_col].value_counts()
            lab = counts.index[0]
            hatch = None
            if lab == "stalled" and hatch_lazy_stall and (sub.dK < 0.1).mean() > 0.5:
                hatch = "////"
            ax.add_patch(
                plt.Rectangle(
                    (xe[i], j - 0.5),
                    xe[i + 1] - xe[i],
                    1,
                    facecolor=COLORS[lab],
                    edgecolor="white",
                    lw=1.0,
                    hatch=hatch,
                )
            )
            if counts.iloc[0] < len(sub):
                ax.text(
                    np.sqrt(xe[i] * xe[i + 1]) if logx else (xe[i] + xe[i + 1]) / 2,
                    j,
                    f"{counts.iloc[0]}/{len(sub)}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=INK,
                )
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(-0.5, len(ys) - 0.5)
    ax.set_yticks(range(len(ys)), [str(y) for y in ys])
    ax.set_xlabel(r"per-core init scale $\sigma$" if xcol == "sigma" else xcol)
    return ys


def heat_panel(
    ax,
    df,
    ycol,
    value,
    cmap,
    norm,
    title,
    xcol="sigma",
    ylabels=None,
    contour=None,
    transform=None,
    fig=None,
):
    """Median over seeds of `value` per (y, x) cell as a heatmap (log x)."""
    piv = df.groupby([ycol, xcol])[value].median().unstack()
    if ylabels is not None:
        piv = piv.reindex(ylabels)
    vals = piv.values.astype(float)
    if transform is not None:
        vals = transform(vals)
    im = ax.pcolormesh(
        _edges(piv.columns),
        np.arange(len(piv.index) + 1) - 0.5,
        vals,
        norm=norm,
        cmap=cmap,
        shading="flat",
    )
    if contour is not None:
        ax.contour(
            piv.columns, np.arange(len(piv.index)), vals, levels=[contour], colors=INK, linewidths=1
        )
    ax.set_xscale("log")
    ax.set_yticks(range(len(piv.index)), [str(c) for c in piv.index])
    ax.set_xlabel(r"$\sigma$" if xcol == "sigma" else xcol)
    ax.set_title(title, fontsize=9, loc="left")
    if fig is not None:
        fig.colorbar(im, ax=ax, fraction=0.05)
    return im


def legend_handles(with_hatch=True):
    h = [Patch(facecolor=COLORS[r], label=NAMES[r]) for r in ("s2s", "rich", "lazy", "stalled")]
    if with_hatch:
        h.append(
            Patch(
                facecolor=COLORS["stalled"],
                hatch="////",
                label=r"stalled, $\delta_K<0.1$ (lazy residual)",
            )
        )
    return h


def scaling_sigma_c(df, ycol, K, rho_thr=3.0):
    """Predicted S2S boundary from the small-sigma law rho = R sigma^-(K-2) + c (theory_regimes
    section 3.2), fitted to the per-sigma median of min_i rho_i on the S2S side (S2S fraction >= 1/2,
    sigma below the first sigma where it drops under 1/2); solves R sigma^-(K-2) + c = rho_thr."""
    out = {}
    for y, sub in df.groupby(ycol):
        sig = np.sort(sub.sigma.unique())
        frac = np.array([(sub[sub.sigma == s].label == "s2s").mean() for s in sig])
        stop = np.argmax(frac < 0.5) if (frac < 0.5).any() else len(sig)
        use = sig[:stop]
        med = np.array([np.nanmedian(sub[sub.sigma == s].min_rho) for s in use])
        ok = np.isfinite(med)
        if ok.sum() < 3:
            out[y] = np.nan
            continue
        A = np.stack([use[ok] ** (-(K - 2.0)), np.ones(ok.sum())], 1)
        (R, c), *_ = np.linalg.lstsq(A, med[ok], rcond=None)
        out[y] = (R / (rho_thr - c)) ** (1.0 / (K - 2)) if R > 0 and rho_thr > c else np.nan
    return out


def aligned_boundaries(df, ycol, K):
    """Median over seeds of the per-seed S2S boundary predicted from each seed's deepest run."""
    from analysis import aligned_sigma_c

    out = {}
    for y, sub in df.groupby(ycol):
        vals = [aligned_sigma_c(rows, K)[0] for _, rows in sub.groupby("seed")]
        out[y] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
    return out


def fig_sigma_chi(exp, model, lazy_csv=None, title=""):
    df = load(exp)
    df = df[df.model == model]
    K = int(df.K.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    ys = phase_panel(ax, df, "chi")
    ax.set_ylabel(r"bond dimension $\chi$")
    # predicted boundaries
    yi = {y: j for j, y in enumerate(ys)}
    sc = aligned_boundaries(df, "chi", K)
    ax.plot(
        [sc[y] for y in ys],
        range(len(ys)),
        "k--",
        lw=1.2,
        marker="o",
        ms=3,
        label="S2S boundary predicted from the deepest run\n(aligned-mode extrapolation, median of seeds)",
    )
    if "l_best" in df.columns and df.l_best.notna().any():
        rb = {}
        for y, sub in df.groupby("chi"):
            sig = np.sort(sub.sigma.unique())
            med = np.array([np.nanmedian(sub[sub.sigma == s].l_best.astype(float)) for s in sig])
            rb[y] = boundary(sig, (med >= 1e-3).astype(float)) if np.nanmax(med) >= 1e-3 else np.nan
        pts = [(yi[y], rb[y]) for y in ys if np.isfinite(rb.get(y, np.nan))]
        if pts:
            ax.plot(
                [p[1] for p in pts],
                [p[0] for p in pts],
                ls="none",
                marker="v",
                ms=5,
                color=INK,
                label=r"realizability: best fit of $W^*+W_0$ reaches $10^{-3}L_0$",
            )
    if lazy_csv is not None and model in ("centred", "twocopy"):
        lp = pd.read_csv(lazy_csv)
        lp = lp[lp.residual < 1e-6]
        pts = [
            (yi[c], np.median((g.kappa / 0.1) ** (1.0 / K)))
            for c, g in lp.groupby("chi")
            if c in yi
        ]
        if model == "centred" and pts:
            ax.plot(
                [p[1] for p in pts],
                [p[0] for p in pts],
                "k:",
                lw=1.5,
                marker="s",
                ms=3,
                label=r"lazy boundary, first order $(\kappa/0.1)^{1/K}$",
            )
    ax.legend(
        handles=legend_handles() + ax.get_legend_handles_labels()[0],
        fontsize=6.5,
        loc="upper left",
        bbox_to_anchor=(0, -0.18),
        ncol=2,
        frameon=False,
    )
    ax.set_title(f"{title}regime (majority of {df.seed.nunique()} seeds)", fontsize=9, loc="left")

    heat_panel(
        axes[1],
        df,
        "chi",
        "dK",
        "Blues",
        LogNorm(1e-4, 1e4),
        r"median kernel change $\delta_K(T)$ (line: 0.1)",
        contour=0.1,
        fig=fig,
    )
    heat_panel(
        axes[2],
        df,
        "chi",
        "min_rho",
        "RdBu_r",
        TwoSlopeNorm(vcenter=np.log10(3), vmin=-2, vmax=2.5),
        r"median $\log_{10}\min_i\rho_i$ (white: $\rho=3$)",
        transform=np.log10,
        fig=fig,
    )
    fig.tight_layout()
    save_fig(fig, f"phase_{exp}_{model}.png")
    return df


def boundaries_table(df, ycol):
    """S2S and lazy boundaries per y for every threshold (fraction-0.5 crossings in sigma)."""
    rows = []
    for y, sub in df.groupby(ycol):
        sig = np.sort(sub.sigma.unique())
        for kt in K_THRESHOLDS:
            for rt in RHO_THRESHOLDS:
                fr = fractions(sub, ["sigma"], f"label_{kt}_{rt}").reindex(sig).fillna(0)
                rows.append(
                    dict(
                        y=y,
                        k_thr=kt,
                        rho_thr=rt,
                        sigma_s2s=boundary(sig, fr["s2s"].values),
                        sigma_lazy=boundary(sig, fr["lazy"].values),
                    )
                )
    return pd.DataFrame(rows).rename(columns={"y": ycol})


def fig_sensitivity(exp, model, ycol="chi"):
    """Threshold sensitivity from continuous per-seed boundaries (median over seeds, bars: quartiles):
    S2S boundary where min_i rho_i crosses 2 / 3 / 5, kernel boundary where delta_K crosses
    0.05 / 0.1 / 0.2. Also writes the label-fraction boundaries (grid-limited) as a table."""
    from analysis import sensitivity_table

    df = load(exp)
    df = df[df.model == model]
    K = int(df.K.iloc[0])
    save_csv(boundaries_table(df, ycol), f"boundaries_{exp}_{model}.csv")
    st = sensitivity_table(df, [ycol])
    save_csv(st, f"sensitivity_per_seed_{exp}_{model}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    blues = ["#86b6ef", "#2a78d6", "#104281"]
    oranges = ["#f5b394", "#eb6834", "#a3401a"]
    for ax, cols, colors, labs in (
        (
            axes[0],
            [f"s2s_{t:g}" for t in RHO_THRESHOLDS],
            oranges,
            [rf"$\rho$ = {t:g}" for t in RHO_THRESHOLDS],
        ),
        (
            axes[1],
            [f"lazy_{t:g}" for t in K_THRESHOLDS],
            blues,
            [rf"$\delta_K$ = {t:g}" for t in K_THRESHOLDS],
        ),
    ):
        for col, c, lab in zip(cols, colors, labs):
            g = st.groupby(ycol)[col]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ok = med.notna()
            ax.errorbar(
                med.index[ok],
                med[ok],
                yerr=[med[ok] - q1[ok], q3[ok] - med[ok]],
                fmt="o-",
                color=c,
                ms=4,
                lw=1.3,
                capsize=2,
                label=lab,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(ycol)
        ax.set_ylabel(r"boundary $\sigma$ (median of seeds)")
        ax.legend(fontsize=7, frameon=False)
        xs = np.sort(st[ycol].unique())
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
    r1 = (st["s2s_2"] / st["s2s_5"]).median()
    r2 = (st["lazy_0.05"] / st["lazy_0.2"]).median()
    axes[0].set_title(
        f"S2S boundary vs rho threshold: ratio 2->5 = {r1:.2f}\n(predicted (5/2)^(1/(K-2)) = "
        f"{2.5 ** (1 / (K - 2)):.2f})",
        fontsize=8.5,
        loc="left",
    )
    axes[1].set_title(
        f"kernel boundary vs delta_K threshold: ratio 0.05->0.2 = {r2:.2f}\n(first order 4^(1/K) = "
        f"{4 ** (1 / K):.2f}; second order 4^(1/(2K)) = {4 ** (1 / (2 * K)):.2f})",
        fontsize=8.5,
        loc="left",
    )
    fig.tight_layout()
    save_fig(fig, f"sensitivity_{exp}_{model}.png")
    return st


def load_traj(exp, cfg):
    """(t, L/L0, overlaps) for a run of either driver (library Trajectory or interim npz)."""

    path = analysis._traj_path(exp, cfg)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if "overlaps" in z.files:
            return z["t"], z["loss"] / z["loss"][0], z["overlaps"]
        return z["t"], z["L"], z["A"]
    from pilots.interim import run_id as rid

    z = np.load(os.path.join(analysis.OUT, exp, "traj", rid(cfg) + ".npz"))
    return z["t"], z["L"], z["A"]


def fig_examples(exp, picks, name):
    """Representative runs: list of (title, cfg-dict). Top: L / L(0) vs t (log-log; the S2S run also in
    linear time, inset, since plateaus must be judged in linear time). Bottom: overlaps a_r vs t."""
    fig, axes = plt.subplots(2, len(picks), figsize=(3.3 * len(picks), 4.8), sharex=False)
    comp_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, (title, cfg) in enumerate(picks):
        t, L, A = load_traj(exp, cfg)
        t, L, A = t[1:], L[1:], A[1:]
        ax = axes[0, k]
        ax.loglog(t, L, color=INK, lw=1.5)
        ax.set_title(title, fontsize=8.5, loc="left")
        if cfg.get("linear_inset"):
            ins = ax.inset_axes([0.12, 0.12, 0.45, 0.45])
            ins.plot(t, L, color=INK, lw=1.2)
            ins.set_xlim(0, t[-1])
            ins.tick_params(labelsize=6)
            ins.set_title("linear time", fontsize=6)
        ws = cfg.get("weights", [3, 2, 1])
        for r in range(A.shape[1]):
            axes[1, k].semilogx(
                t, A[:, r], color=comp_colors[r], lw=1.5, label=f"$a_{r + 1}$ ($s={ws[r]:g}$)"
            )
        axes[1, k].set_xlabel("gradient-flow time t")
        axes[1, k].set_ylim(-0.1, 1.15)
    axes[0, 0].set_ylabel(r"$L/L(0)$")
    axes[1, 0].set_ylabel(r"overlap $a_r$")
    axes[1, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_fig(fig, f"examples_{name}.png")


# ---- boundaries vs bond dimension (chi_b4) ------------------------------------------------------


def chi_boundaries(df: pd.DataFrame, K: int = 3):
    """Per-seed S2S boundary (label-based, with status), aligned-extrapolation prediction and
    kernel boundary (delta_K = 0.1 crossing) for every chi; returns the per-seed table."""
    from analysis import per_seed_table

    return per_seed_table(df, ["model", "chi"], K)


def fig_chi_b4(exp="chi_b4", lazy_csv=None):
    """Boundaries vs chi: S2S (per seed, mean +- SE of log sigma_c) with the aligned prediction,
    lazy (delta_K = 0.1 crossing per seed) with theory_regimes P5/P6 (6.50 and 5.16 chi^-1/2), and the
    fitted exponents (fit to the per-seed values, error from seed-to-seed scatter)."""
    from analysis import load, mean_se

    df = load(exp)
    tab = chi_boundaries(df)
    save_csv(tab, f"boundaries_per_seed_{exp}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    out = {}
    for model, color, marker in (("centred", "#2a78d6", "o"), ("twocopy", "#eb6834", "s")):
        t = tab[tab.model == model]
        if t.empty:
            continue
        chis = np.sort(t.chi.unique())
        cols = [("sigma_c", axes[0], "S2S boundary"), ("sigma_K01", axes[1], "probe kernel")]
        if "sigma_K01_tensor" in t.columns and t.sigma_K01_tensor.notna().any():
            cols.append(("sigma_K01_tensor", axes[1], "tensor-space kernel"))  # fix round 1
        for col, ax, lab in cols:
            if col == "sigma_c" and model == "twocopy":
                continue
            m, e = [], []
            for c in chis:
                mu, se, n = mean_se(np.log(t[t.chi == c][col]))
                m.append(mu)
                e.append(se)
            m, e = np.array(m), np.array(e)
            ok = np.isfinite(m)
            ax.errorbar(
                chis[ok],
                np.exp(m[ok]),
                yerr=[np.exp(m[ok]) - np.exp(m[ok] - e[ok]), np.exp(m[ok] + e[ok]) - np.exp(m[ok])],
                fmt=marker,
                color=color,
                mfc="white" if col == "sigma_K01_tensor" else color,
                ms=4,
                lw=1.2,
                label=f"{model}, {lab}: geometric mean over seeds",
            )
            # exponent: fit on all per-seed points (chi >= 16 for the lazy boundary: isotropic regime)
            sel = t[np.isfinite(t[col]) & (t.chi >= (9 if col == "sigma_c" else 16))]
            if len(sel) >= 5:
                x, y = np.log(sel.chi.values), np.log(sel[col].values)
                A = np.stack([x, np.ones_like(x)], 1)
                coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
                dof = max(len(x) - 2, 1)
                s2 = ((y - A @ coef) ** 2).sum() / dof
                cov = s2 * np.linalg.inv(A.T @ A)
                out[(model, col)] = (coef[0], np.sqrt(cov[0, 0]))
                xx = np.geomspace(sel.chi.min(), sel.chi.max(), 50)
                ax.plot(
                    xx,
                    np.exp(coef[1]) * xx ** coef[0],
                    "--" if col == "sigma_K01_tensor" else "-",
                    color=color,
                    lw=1,
                    label=f"fit slope {coef[0]:.2f} +- {np.sqrt(cov[0, 0]):.2f}",
                )
        if model == "centred":
            pred = t.groupby("chi").sigma_c_pred.median()
            axes[0].plot(
                pred.index, pred.values, "k--", lw=1, label="aligned-mode prediction (median)"
            )
    xx = np.geomspace(16, 128, 50)
    axes[1].plot(
        xx,
        6.50 * xx**-0.5,
        ":",
        color="#2a78d6",
        lw=1.5,
        label=r"theory P5: $6.50\,\chi^{-1/2}$ (centred)",
    )
    axes[1].plot(
        xx,
        5.16 * xx**-0.5,
        ":",
        color="#eb6834",
        lw=1.5,
        label=r"theory P6: $5.16\,\chi^{-1/2}$ (two-copy)",
    )
    if lazy_csv:
        lp = pd.read_csv(lazy_csv)
        fo = lp[lp.chi >= 9].groupby("chi").kappa.median()  # full bonds only (below: lazy residual)
        axes[1].plot(
            fo.index,
            (fo.values / 0.1) ** (1 / 3),
            "k-.",
            lw=1,
            label=r"first order $(\kappa/0.1)^{1/3}$",
        )
    for ax, title in zip(axes, ("S2S boundary (per seed)", "lazy boundary (per seed)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"bond dimension $\chi$")
        ax.set_ylabel(r"boundary $\sigma$")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save_fig(fig, f"boundaries_vs_chi_{exp}.png")
    return tab, out


# ---- sigma x K ---------------------------------------------------------------------------------


def tree_label(kind, n, K):
    return f"{kind}{n} (K={K})"


def phase_rows(ax, df, rowkey, xcol, rows, hatch_lazy_stall=True):
    """Categorical map with one row per entry of `rows` (values of `rowkey`) and per-row log-x
    cells (each row keeps its own x grid)."""
    for j, rk in enumerate(rows):
        sub_r = df[df[rowkey] == rk]
        xs = np.sort(sub_r[xcol].unique())
        if len(xs) == 0:
            continue
        xe = _edges(xs) if len(xs) > 1 else np.array([xs[0] / 1.2, xs[0] * 1.2])
        for i, x in enumerate(xs):
            sub = sub_r[sub_r[xcol] == x]
            counts = sub.label.value_counts()
            lab = counts.index[0]
            hatch = (
                "////"
                if (lab == "stalled" and hatch_lazy_stall and (sub.dK < 0.1).mean() > 0.5)
                else None
            )
            ax.add_patch(
                plt.Rectangle(
                    (xe[i], j - 0.5),
                    xe[i + 1] - xe[i],
                    1,
                    facecolor=COLORS[lab],
                    edgecolor="white",
                    lw=0.8,
                    hatch=hatch,
                )
            )
            if counts.iloc[0] < len(sub):
                ax.text(
                    np.sqrt(xe[i] * xe[i + 1]),
                    j,
                    f"{counts.iloc[0]}/{len(sub)}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=INK,
                )
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    allx = df[xcol].values
    ax.set_xlim(allx.min() / 1.3, allx.max() * 1.3)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_yticks(range(len(rows)), [str(r) for r in rows])


def fig_sigma_K(exp="sigma_K", chi=6):
    """Phase diagram over trees (rows, sorted by K) vs sigma and vs alpha = sigma^K, with per-seed
    S2S boundaries, the aligned-extrapolation prediction and the exact aligned boundary a0c(K)."""
    from analysis import aligned_boundary_a0, load, per_seed_table

    df = load(exp)
    df = df[df.chi == chi].copy()
    df["tree"] = [tree_label(k, n, K) for k, n, K in zip(df.kind, df.n, df.K)]
    df["alpha"] = df.sigma**df.K
    df["u0"] = df.alpha / np.sqrt(
        df.d.astype(float) ** df.n
    )  # RMS overlap of W_0 with a unit rank-one tensor
    d = int(df.d.iloc[0])
    weights = (
        tuple(df.weights.iloc[0])
        if "weights" in df.columns and isinstance(df.weights.iloc[0], list)
        else (3.0, 2.0, 1.0)
    )
    order = df.drop_duplicates("tree").sort_values(["K", "kind"]).tree.tolist()
    Ks = {t: int(df[df.tree == t].K.iloc[0]) for t in order}
    Ns = {t: int(df[df.tree == t].n.iloc[0]) for t in order}
    tab = per_seed_table(df, ["tree"])
    save_csv(tab, f"boundaries_per_seed_{exp}_chi{chi}.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for ax, xcol in zip(axes, ("sigma", "alpha", "u0")):
        phase_rows(ax, df, "tree", xcol, order)
        med = [np.nanmedian(tab[tab.tree == t].sigma_c) for t in order]
        pred = [np.nanmedian(tab[tab.tree == t].sigma_c_pred) for t in order]
        a0c = [aligned_boundary_a0(Ks[t], weights) for t in order]
        f = {
            "sigma": lambda s, t: s,
            "alpha": lambda s, t: s ** Ks[t],
            "u0": lambda s, t: s ** Ks[t] / float(d) ** (Ns[t] / 2),
        }[xcol]
        ys = np.arange(len(order))
        ax.plot(
            [f(m, t) for m, t in zip(med, order)],
            ys,
            "o",
            color=INK,
            ms=4,
            label="per-seed S2S boundary (median)",
        )
        ax.plot(
            [f(m, t) for m, t in zip(pred, order)],
            ys,
            "x",
            color=INK,
            ms=5,
            label="predicted from deepest run (aligned extrapolation)",
        )
        ax.plot(
            [f(m, t) for m, t in zip(a0c, order)],
            ys,
            "--",
            color=MUTED,
            lw=1,
            label=r"aligned-init boundary $a_{0c}(K)$ ($\kappa_K\to1$ limit)",
        )
        ax.set_xlabel(
            {
                "sigma": r"per-core init scale $\sigma$",
                "alpha": r"function-space scale $\alpha=\sigma^K$",
                "u0": r"per-component amplitude $\sigma^K/\sqrt{D}$",
            }[xcol]
        )
    axes[0].legend(
        handles=legend_handles(False) + axes[0].get_legend_handles_labels()[0],
        fontsize=6.5,
        loc="upper left",
        bbox_to_anchor=(0, -0.15),
        ncol=2,
        frameon=False,
    )
    axes[0].set_title(
        f"{exp}: {df.model.iloc[0]} model, chi = {chi}, d = {d}, teacher {weights}",
        fontsize=9,
        loc="left",
    )
    fig.tight_layout()
    save_fig(fig, f"phase_{exp}_chi{chi}.png")
    return tab


# ---- N = 8: centred and plain rows --------------------------------------------------------------


def fig_sigma_chi_b8(exp="sigma_chi_b8", lazy_csv=None, full_label=None, floor=0.75):
    """Rows (model, chi): centred at every chi (chi = 162 means bonds 18 / 162, twice the cut
    dimension), plain at deficient chi. Overlays: centred floor sigma_floor ~ 0.74-0.78 (theory P4),
    per-seed S2S boundaries, first-order lazy boundary (kappa/0.1)^(1/7) for full bonds."""
    from analysis import load, per_seed_table

    df = load(exp)
    full_label = full_label or {162: "18/162"}
    df["row"] = [f"{m} chi={full_label.get(c, c)}" for m, c in zip(df.model, df.chi)]
    chis = sorted(df.chi.unique())
    order = [f"{m} chi={full_label.get(c, c)}" for m in ("centred", "plain") for c in chis]
    order = [r for r in order if r in set(df.row)]
    d = int(df.d.iloc[0])
    tab = per_seed_table(df, ["row"], 7)
    save_csv(tab, f"boundaries_per_seed_{exp}.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    phase_rows(ax, df, "row", "sigma", order)
    ys = np.arange(len(order))
    med = [np.nanmedian(tab[tab.row == r].sigma_c) for r in order]
    pred = [np.nanmedian(tab[tab.row == r].sigma_c_pred) for r in order]
    ax.plot(med, ys, "o", color=INK, ms=4, label="per-seed S2S boundary (median)")
    ax.plot(
        pred, ys, "x", color=INK, ms=5, label="predicted from deepest run (aligned extrapolation)"
    )
    if floor is not None:
        for j, r in enumerate(order):
            if r.startswith("centred") and int(str(r.split("=")[1]).split("/")[-1]) < d**4:
                ax.plot([floor, floor], [j - 0.45, j + 0.45], color=MUTED, lw=2)
        ax.plot(
            [],
            [],
            color=MUTED,
            lw=2,
            label=rf"centred floor $\sigma_{{\rm floor}}\approx{floor}$ (theory P4)",
        )
    if lazy_csv:
        lp = pd.read_csv(lazy_csv)
        for j, r in enumerate(order):
            c = 81 if r.endswith("81") else (162 if r.endswith("18/162") else None)
            if r.startswith("centred") and c is not None:
                k = lp[lp.chi == c].kappa.median()
                ax.plot((k / 0.1) ** (1 / 7), j, "s", color=INK, ms=4, mfc="none")
        ax.plot(
            [],
            [],
            "s",
            color=INK,
            ms=4,
            mfc="none",
            label=r"lazy boundary, first order $(\kappa/0.1)^{1/7}$",
        )
    ax.set_xlabel(r"per-core init scale $\sigma$")
    ax.set_title(f"{exp}: binary N = 8 (K = 7), d = {d}, 6 seeds", fontsize=9, loc="left")
    ax.legend(
        handles=legend_handles() + ax.get_legend_handles_labels()[0],
        fontsize=6.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
    )
    fig.tight_layout()
    save_fig(fig, f"phase_{exp}.png")
    return tab


# ---- imbalance -----------------------------------------------------------------------------------


def fig_imbalance(exp="imbalance_b4"):
    """Phase diagrams over (lambda / sqrt(chi), alpha) for chi = 24 and 3, and delta_K vs lambda at the
    theory check point (chi = 16, alpha = 3.375; theory_regimes P20: delta_K ~ lambda^2, saturating)."""
    from analysis import load

    df = load(exp)
    # early rows stored the recomputed product; 4 digits keep the check point 3.375 (3 digits gave
    # 3.38 and an empty P20 panel, fix round 1)
    df["alpha"] = [float(f"{a:.4g}") for a in df.alpha]
    df["lam_rel"] = df.root_scale / np.sqrt(df.chi)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, chi in zip(axes[:2], (24, 3)):
        sub = df[(df.chi == chi) & ~np.isclose(df.alpha, 3.375)]
        if sub.empty:
            continue
        alphas = sorted(sub.alpha.unique())
        sub = sub.assign(alpha_lab=[f"{a:g}" for a in sub.alpha])
        phase_rows(ax, sub, "alpha_lab", "lam_rel", [f"{a:g}" for a in alphas])
        ax.axvline(1.0, color=INK, lw=1, ls="--")
        ax.axvline(1 / np.sqrt(chi), color=MUTED, lw=1, ls=":")
        ax.set_xlabel(r"root imbalance $\lambda/\sqrt{\chi}$ (1: $\mathbb{E}Q=0$ on root edges)")
        ax.set_ylabel(r"function-space scale $\alpha=\lambda\sigma^K$")
        ax.set_title(
            f"binary N = 4, chi = {chi} (dotted: fan-in default lambda = 1)", fontsize=9, loc="left"
        )
    ax = axes[2]
    pt = df[(df.chi == 16) & np.isclose(df.alpha, 3.375)]
    if not pt.empty:
        g = pt.groupby("root_scale").dK
        lam = np.array(sorted(pt.root_scale.unique()))
        ax.loglog(
            lam,
            g.median().reindex(lam).values,
            "o-",
            color="#2a78d6",
            ms=4,
            label=r"median $\delta_K$, probe kernel (6 seeds)",
        )
        ax.fill_between(
            lam,
            g.min().reindex(lam).values,
            g.max().reindex(lam).values,
            color="#2a78d6",
            alpha=0.2,
            lw=0,
        )
        ref = g.median().reindex(lam).values[0]
        if (
            "dK_tensor" in pt.columns and pt.dK_tensor.notna().any()
        ):  # the kernel of P20 (fix round 1)
            gt = pt.groupby("root_scale").dK_tensor
            ax.loglog(
                lam,
                gt.median().reindex(lam).values,
                "D-",
                color="#104281",
                ms=4,
                label=r"median $\delta_K$, tensor-space kernel",
            )
            ax.axhspan(0.6, 0.7, color=MUTED, alpha=0.15, lw=0, label="P20 saturation 0.6-0.7")
        ax.loglog(
            lam[:4], ref * (lam[:4] / lam[0]) ** 2, "k--", lw=1, label=r"slope 2 (theory P20)"
        )
        ax.axhline(0.1, color=MUTED, lw=0.8)
        ax.axvline(4.0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel(r"root scale $\lambda$ ($\chi=16$, $\alpha=3.375$)")
        ax.set_ylabel(r"$\delta_K(T)$")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(
            r"kernel change vs root scale; dotted: $\lambda=\sqrt{\chi}$", fontsize=9, loc="left"
        )
    axes[0].legend(
        handles=legend_handles(),
        fontsize=6.5,
        loc="upper left",
        bbox_to_anchor=(0, -0.18),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    save_fig(fig, f"phase_{exp}.png")


# ---- large bond dimension ----------------------------------------------------------------------

PARAM_COLORS = {"sp": "#2a78d6", "ntk": "#eb6834", "mup": "#1baf7a"}
PARAM_NAMES = {"sp": "SP (fan-in, eta = 1)", "ntk": "NTK (eta_v = 1/F_v)", "mup": "muP-TTN"}
THEORY_EXP = {  # theory_regimes P16 (N = 4, centred): (delta_K, delta_theta)
    "sp": (-3.0, -1.5),
    "ntk": (-1.0, -0.5),
    "mup": (0.0, 0.0),
}


def pooled_slope(x, y):
    """Slope of log y vs log x over independent runs (each (chi, seed) is its own init), with the
    standard error from the residual scatter."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 4:
        return np.nan, np.nan
    X = np.stack([np.log(x[ok]), np.ones(ok.sum())], 1)
    coef, *_ = np.linalg.lstsq(X, np.log(y[ok]), rcond=None)
    r = np.log(y[ok]) - X @ coef
    cov = (r @ r) / max(ok.sum() - 2, 1) * np.linalg.inv(X.T @ X)
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def fig_large_chi(exp="large_chi_b4", model="centred", sigma=1.5, chi_fit_min=16):
    """delta_K, delta_theta and per-core changes vs chi for the three parameterizations; fitted
    exponents (chi >= chi_fit_min) against theory_regimes P16."""
    from analysis import load

    df = load(exp)
    df = df[(df.model == model) & (np.isclose(df.sigma, sigma))]
    if df.empty:
        return None
    root = {"binary": {4: 2, 8: 6}}  # root index of binary_tree(N)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for lr in ("sp", "ntk", "mup"):
        s = df[df.lr == lr]
        if s.empty:
            continue
        c = PARAM_COLORS[lr]
        ridx = root[s.kind.iloc[0]][int(s.n.iloc[0])]
        s = s.assign(
            root_change=[d[ridx] for d in s.dth_core],
            leaf_change=[np.mean([v for i, v in enumerate(d) if i != ridx]) for d in s.dth_core],
        )
        for ax, col in zip(axes, ("dK", "dth", "root_change")):
            med = s.groupby("chi")[col].median()
            ax.loglog(med.index, med.values, "o-", color=c, ms=4, lw=1.2, label=PARAM_NAMES[lr])
            ax.scatter(s.chi, s[col], s=6, color=c, alpha=0.35, lw=0)
        fit = s[s.chi >= chi_fit_min]
        # theory slopes (P16) as dotted reference lines through the largest-chi median
        for ax, col, th in zip(
            axes,
            ("dK", "dth", "root_change"),
            (THEORY_EXP[lr][0], THEORY_EXP[lr][1], {"sp": -1.0, "ntk": -1.0, "mup": -0.5}[lr]),
        ):
            med = s.groupby("chi")[col].median()
            if len(med) and model == "centred" and s.n.iloc[0] == 4:
                xx = np.array([chi_fit_min, med.index.max()], float)
                ax.loglog(xx, med.iloc[-1] * (xx / xx[-1]) ** th, ":", color=c, lw=1.2)
        for col in ("dK", "dK_tensor", "dth", "root_change", "leaf_change"):
            if col not in fit.columns:  # dK_tensor: runs of the fixed driver (fix round 1)
                continue
            b, e = pooled_slope(fit.chi, fit[col])
            rows.append(
                dict(
                    exp=exp,
                    model=model,
                    sigma=sigma,
                    lr=lr,
                    quantity=col,
                    slope=b,
                    se=e,
                    n_runs=int(np.isfinite(fit[col]).sum()),
                    theory={
                        "dK": THEORY_EXP[lr][0],
                        "dK_tensor": THEORY_EXP[lr][0],
                        "dth": THEORY_EXP[lr][1],
                    }.get(col, np.nan),
                )
            )
    for ax, t in zip(
        axes,
        (
            r"kernel change $\delta_K(T)$",
            r"$\delta_\theta(T)$",
            "root core: relative Frobenius change",
        ),
    ):
        ax.set_xlabel(r"bond dimension $\chi$")
        ax.set_title(t, fontsize=9, loc="left")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"{exp}, {model}, sigma = {sigma} (dots: seeds; lines: medians; dotted: theory_regimes P16 slopes)",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    save_fig(fig, f"large_chi_{exp}_{model}_sigma{sigma}.png")
    tab = pd.DataFrame(rows)
    save_csv(tab, f"large_chi_exponents_{exp}_{model}_sigma{sigma}.csv")
    return tab


def fig_stall_fraction(specs, name="stalls_vs_K"):
    """Fraction of runs labelled stalled vs sigma / a0c(K) (a0c = exact aligned S2S boundary), one
    line per tree, one panel per (exp, chi-class). specs: list of (title, exp, chi or 'full')."""
    from analysis import aligned_boundary_a0, load

    fig, axes = plt.subplots(1, len(specs), figsize=(4.6 * len(specs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("viridis")
    for ax, (title, exp, chi) in zip(axes, specs):
        df = load(exp)
        if chi != "full":
            df = df[df.chi == chi]
        w = (
            tuple(df.weights.iloc[0])
            if "weights" in df.columns and isinstance(df.weights.iloc[0], list)
            else (3.0, 2.0, 1.0)
        )
        trees = (
            df.drop_duplicates(["kind", "n"]).sort_values("K")[["kind", "n", "K"]].values.tolist()
        )
        Kmax = max(t[2] for t in trees)
        for kind, n, K in trees:
            g = df[(df.kind == kind) & (df.n == n)]
            fr = g.groupby("sigma").apply(lambda x: (x.label == "stalled").mean())
            a0c = aligned_boundary_a0(int(K), w)
            ax.semilogx(
                fr.index / a0c,
                fr.values,
                "o-",
                ms=3,
                lw=1.2,
                color=cmap((K - 3) / max(Kmax - 3, 1)),
                label=f"{kind}{n} (K={K})",
            )
        ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel(r"$\sigma / a_{0c}(K)$")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=6, frameon=False)
    axes[0].set_ylabel("fraction of seeds stalled")
    fig.tight_layout()
    save_fig(fig, f"{name}.png")


def k_series_rows(tab, label, exp, chi, w):
    """Per tree of a per-seed table (`analysis.per_seed_table`): counts by label status and the
    boundary medians. Resolved = status 'ok...' (the next run up is rich or lazy); 'unresolved...'
    = the next run stalled, so the boundary is only bracketed by [s2s_lo, s2s_hi] (fix round 1: the
    pre-fix table counted only the exact status 'unresolved' and missed 'unresolved+stall_below').
    sigma_c_median and kappa use resolved seeds only; bracket_* summarize every seed."""
    from analysis import aligned_boundary_a0

    rows = []
    for (kind, n), g in tab.groupby(["kind", "n"]):
        K = int(g.K.iloc[0])
        st = g.s2s_status.astype(str)
        res = st.str.startswith("ok")
        sc = g.sigma_c[res].dropna()
        a0c = aligned_boundary_a0(K, w)
        rows.append(
            dict(
                series=label,
                exp=exp,
                chi=chi,
                tree=f"{kind}{n}",
                K=K,
                n_seeds=len(g),
                n_resolved=int(res.sum()),
                n_unresolved=int(st.str.startswith("unresolved").sum()),
                n_never=int(st.str.startswith("never").sum()),
                n_all_s2s=int(st.str.startswith("all_s2s").sum()),
                n_stall_below=int(st.str.contains("stall_below").sum()),
                sigma_c_median=float(sc.median()) if len(sc) else np.nan,
                sigma_c_q1=float(sc.quantile(0.25)) if len(sc) else np.nan,
                sigma_c_q3=float(sc.quantile(0.75)) if len(sc) else np.nan,
                bracket_median=float(g.sigma_c.median()),
                bracket_lo_median=float(g.s2s_lo.median()),
                bracket_hi_median=float(g.s2s_hi.median()),
                a0c=a0c,
                kappa=a0c / float(sc.median()) if len(sc) else np.nan,
            )
        )
    return rows


def fig_sigma_c_vs_K(specs, name="sigma_c_vs_K", tables=None):
    """S2S boundary vs K: per-seed boundaries for each (exp, chi) series, with the exact aligned-init
    boundary a0c(K) (the kappa_K -> 1 limit of H4) and the effective per-core amplitude
    kappa_K = a0c(K) / sigma_c (theory_regimes R9; H4 holds iff kappa_K -> 1). Filled: median over
    resolved seeds (quartiles), shown when at least half the seeds are resolved; open triangles:
    otherwise, the median bracket [last S2S sigma, next sigma] (a lower bound and a step above it).
    specs: list of (label, exp, chi or 'full'); tables: optional {label: (per-seed table, weights)}
    used instead of the raw runs (fix round 1: committed per-seed CSVs). Also writes the table."""
    from analysis import aligned_boundary_a0, load, per_seed_table

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for (label, exp, chi), c in zip(specs, colors):
        if tables is not None:
            if label not in tables:
                continue
            tab, w = tables[label]
        else:
            df = load(exp)
            if chi != "full":
                df = df[df.chi == chi]
            w = (
                tuple(df.weights.iloc[0])
                if "weights" in df.columns and isinstance(df.weights.iloc[0], list)
                else (3.0, 2.0, 1.0)
            )
            tab = per_seed_table(df, ["kind", "n"])
        rows += k_series_rows(tab, label, exp, chi, w)
        t = pd.DataFrame([r for r in rows if r["series"] == label]).sort_values("K")
        ok = t.sigma_c_median.notna() & (t.n_resolved >= t.n_seeds / 2)
        axes[0].errorbar(
            t.K[ok],
            t.sigma_c_median[ok],
            yerr=[t.sigma_c_median[ok] - t.sigma_c_q1[ok], t.sigma_c_q3[ok] - t.sigma_c_median[ok]],
            fmt="o",
            color=c,
            ms=4,
            capsize=2,
            label=f"{label} (resolved seeds)",
        )
        br = ~ok & t.bracket_lo_median.notna()
        if br.any():
            lo = t.bracket_lo_median[br]
            hi = t.bracket_hi_median[br].fillna(lo * 1.5)
            axes[0].errorbar(
                t.K[br] + 0.15,
                lo,
                yerr=[0 * lo, hi - lo],
                fmt="^",
                color=c,
                mfc="white",
                ms=4,
                capsize=2,
                label=f"{label}: unresolved, median bracket",
            )
        axes[1].plot(t.K[ok], t.kappa[ok], "o", color=c, ms=4, label=label)
    Ks = np.arange(3, 16)
    axes[0].plot(
        Ks,
        [aligned_boundary_a0(k, (3.0, 2.0)) for k in Ks],
        "--",
        color=MUTED,
        lw=1,
        label=r"aligned init $a_{0c}(K)$ (teacher 3,2)",
    )
    axes[0].axhline(1.0, color=MUTED, lw=0.8, ls=":")
    axes[1].axhline(1.0, color=MUTED, lw=0.8, ls=":")
    axes[0].set_ylabel(r"S2S boundary $\sigma_c$ (per seed: median, quartiles)")
    axes[1].set_ylabel(r"$\kappa_K=a_{0c}(K)/\sigma_c$")
    for ax in axes:
        ax.set_xlabel("number of cores K")
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save_fig(fig, f"{name}.png")
    out = pd.DataFrame(rows)
    save_csv(out, f"{name}.csv")
    return out


def fig_summary(name="summary"):
    """Summary figure from the raw runs: per-seed boundaries of chi_b4 (centred) and the K sweep
    (sigma_K_centred_d2, chi = 4) with floor-stalled runs classified by min_i rho_i
    (`analysis.seed_rho_boundary_stalls`); see `summary_figure`."""
    from analysis import load, per_seed_table

    df = load("chi_b4")
    chi_tab = per_seed_table(df[df.model == "centred"], ["chi"], 3)
    fix1 = os.path.join(os.path.dirname(__file__), "results", "fix1", "k_sweep_per_seed.csv")
    if os.path.exists(fix1):  # fix round 1: label brackets, unresolved seeds walked up by rho
        t = pd.read_csv(fix1)
    else:
        k = load("sigma_K_centred_d2")
        t = per_seed_table(k[k.chi == 4], ["kind", "n"])
        t["lo"], t["hi"] = t.rho_lo, t.rho_hi
        t["status"] = np.where(t.rho_status.isin(["ok", "gap"]), "rho", "bracket (open)")
    save_csv(t, f"{name}_sigma_c_per_seed.csv")
    summary_figure(chi_tab, t, name)


def summary_figure(chi_tab, k_tab, name="summary"):
    """(left) binary N = 4 (K = 3), centred, fan-in, chi >= 9: per-seed S2S boundary (last-crossing
    events) and lazy boundaries (probe kernel, the label rule; tensor-space kernel, the theory's)
    times sqrt(chi), with theory P5's asymptote 6.50; (right) S2S boundary vs K (centred, d = 2,
    chi = 4): per seed the bracket [last S2S-like sigma, first rich-like sigma] by min_i rho_i
    (floor-stalled runs included), open where no rich-like run was found.
    chi_tab: chi, seed, sigma_c, sigma_K01, sigma_K01_tensor; k_tab: kind, n, K, seed, lo, hi, status."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    t = chi_tab[chi_tab.chi >= 9]
    for col, c, m, lab in (
        ("sigma_c", COLORS["s2s"], "o", "S2S boundary (rho = 3)"),
        ("sigma_K01", COLORS["lazy"], "s", r"lazy boundary, probe kernel ($\delta_K$ = 0.1)"),
        ("sigma_K01_tensor", "#104281", "D", "lazy boundary, tensor-space kernel"),
    ):
        if col not in t.columns or t[col].isna().all():
            continue
        y = t[col] * np.sqrt(t.chi)
        ax.scatter(
            t.chi * np.exp(0.04 * (t.seed - 2.5)), y, s=10, color=c, marker=m, alpha=0.5, lw=0
        )
        med = y.groupby(t.chi).median()
        ax.plot(med.index, med.values, "-", color=c, marker=m, ms=4, lw=1.2, label=lab + ", median")
    # regime bands between the median boundaries (S2S below, rich between, lazy above; probe kernel)
    s2s = (t.sigma_c * np.sqrt(t.chi)).groupby(t.chi).median()
    lazy = (t.sigma_K01 * np.sqrt(t.chi)).groupby(t.chi).median()
    ok = s2s.notna() & lazy.notna()
    xs = s2s.index[ok]
    ax.fill_between(xs, s2s[ok] / 3, s2s[ok], color=COLORS["s2s"], alpha=0.08, lw=0)
    ax.fill_between(xs, s2s[ok], lazy[ok], color=COLORS["rich"], alpha=0.08, lw=0)
    ax.fill_between(xs, lazy[ok], lazy[ok] * 2, color=COLORS["lazy"], alpha=0.08, lw=0)
    if ok.any():
        x0 = xs[len(xs) // 2]
        for yv, txt in (
            (s2s[x0] / 1.8, "S2S"),
            (np.sqrt(s2s[x0] * lazy[x0]), "rich"),
            (lazy[x0] * 1.4, "lazy"),
        ):
            ax.text(x0, yv, txt, fontsize=8, color=MUTED, ha="center", va="center")
    ax.axhline(6.50, color=INK, ls=":", lw=1, label="theory P5 asymptote 6.50 (tensor kernel)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"boundary $\sigma\sqrt{\chi}$")
    ax.set_title("binary N = 4 (K = 3), centred, fan-in init (6 seeds)", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="center right")
    ax = axes[1]
    for r in k_tab.itertuples():
        c = "#2a78d6" if r.kind == "binary" else "#eb6834"
        x = r.K + 0.08 * (r.seed - 2.5)
        if np.isfinite(r.hi):
            ax.plot([x, x], [r.lo, r.hi], color=c, lw=1.2, alpha=0.8)
            ok = str(getattr(r, "status", "")).startswith(
                ("label", "rho", "kernel", "no-learning", "ok")
            )
            ax.plot(x, np.sqrt(r.lo * r.hi), "o", color=c, ms=3, mfc=c if ok else "white")
        elif np.isfinite(r.lo):
            ax.annotate(
                "", (x, r.lo * 1.25), (x, r.lo), arrowprops=dict(arrowstyle="->", color=c, lw=1)
            )
    ax.axhline(np.sqrt(2), color=MUTED, ls=":", lw=1)
    ax.axhline(1.0, color=MUTED, ls="--", lw=0.8)
    ax.text(15.6, np.sqrt(2) * 1.01, r"$\sqrt{d}$", fontsize=8, color=MUTED)
    ax.text(15.6, 1.01, "1", fontsize=8, color=MUTED)
    ax.plot([], [], "o-", color="#2a78d6", ms=3, label="binary tree: per-seed bracket, midpoint")
    ax.plot(
        [],
        [],
        "o",
        color=MUTED,
        mfc="white",
        ms=3,
        label="hollow: bracket wider than one grid step",
    )
    ax.plot([], [], "o-", color="#eb6834", ms=3, label="MPS")
    ax.plot(
        [],
        [],
        marker=r"$\uparrow$",
        ls="none",
        color=MUTED,
        label="open: no rich-like run found (lower bound)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("number of cores K")
    ax.set_ylabel(r"S2S boundary $\sigma_c$ (classified by $\min_i\rho_i$)")
    ax.set_title(
        "S2S boundary vs K: centred, d = 2, chi = 4 (2x teacher rank)", fontsize=9, loc="left"
    )
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    save_fig(fig, f"{name}.png")
