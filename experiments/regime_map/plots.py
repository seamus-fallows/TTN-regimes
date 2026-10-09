"""Figures for the regime map (reads saved runs via analysis.load). Usage:
uv run python experiments/regime_map/plots.py sigma_chi <exp> <model> [--lazy CSV] [--out DIR] [--force]
Outputs go to --out (default results/); existing files are never overwritten without --force.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from analysis import K_THRESHOLDS, OUT, REGIMES, RHO_THRESHOLDS, boundary, fractions, load  # noqa: E402

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
NAMES = {"lazy": "lazy", "s2s": "saddle-to-saddle", "rich": "rich, non-incremental", "stalled": "stalled"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def _edges(vals):
    v = np.log(np.asarray(vals, float))
    mid = (v[1:] + v[:-1]) / 2
    return np.exp(np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]]))


def phase_panel(ax, df, ycol, label_col="label", hatch_lazy_stall=True):
    """Categorical map: cell colour = majority label; text = majority count / seeds."""
    sig = np.sort(df.sigma.unique())
    ys = np.sort(df[ycol].unique())
    xe = _edges(sig)
    for j, y in enumerate(ys):
        for i, s in enumerate(sig):
            sub = df[(df.sigma == s) & (df[ycol] == y)]
            if sub.empty:
                continue
            counts = sub[label_col].value_counts()
            lab = counts.index[0]
            hatch = None
            if lab == "stalled" and hatch_lazy_stall and (sub.dK < 0.1).mean() > 0.5:
                hatch = "////"
            ax.add_patch(plt.Rectangle((xe[i], j - 0.5), xe[i + 1] - xe[i], 1, facecolor=COLORS[lab],
                                       edgecolor="white", lw=1.0, hatch=hatch))
            if counts.iloc[0] < len(sub):
                ax.text(np.sqrt(xe[i] * xe[i + 1]), j, f"{counts.iloc[0]}/{len(sub)}", ha="center",
                        va="center", fontsize=6, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(-0.5, len(ys) - 0.5)
    ax.set_yticks(range(len(ys)), [str(y) for y in ys])
    ax.set_xlabel(r"per-core init scale $\sigma$")
    return ys


def legend_handles(with_hatch=True):
    h = [Patch(facecolor=COLORS[r], label=NAMES[r]) for r in ("s2s", "rich", "lazy", "stalled")]
    if with_hatch:
        h.append(Patch(facecolor=COLORS["stalled"], hatch="////", label=r"stalled, $\delta_K<0.1$ (lazy residual)"))
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
    ax.plot([sc[y] for y in ys], range(len(ys)), "k--", lw=1.2, marker="o", ms=3,
            label="S2S boundary predicted from the deepest run\n(aligned-mode extrapolation, median of seeds)")
    if "l_best" in df.columns and df.l_best.notna().any():
        rb = {}
        for y, sub in df.groupby("chi"):
            sig = np.sort(sub.sigma.unique())
            med = np.array([np.nanmedian(sub[sub.sigma == s].l_best.astype(float)) for s in sig])
            rb[y] = boundary(sig, (med >= 1e-3).astype(float)) if np.nanmax(med) >= 1e-3 else np.nan
        pts = [(yi[y], rb[y]) for y in ys if np.isfinite(rb.get(y, np.nan))]
        if pts:
            ax.plot([p[1] for p in pts], [p[0] for p in pts], ls="none", marker="v", ms=5, color=INK,
                    label=r"realizability: best fit of $W^*+W_0$ reaches $10^{-3}L_0$")
    if lazy_csv is not None and model in ("centred", "twocopy"):
        lp = pd.read_csv(lazy_csv)
        lp = lp[lp.residual < 1e-6]
        pts = [(yi[c], np.median((g.kappa / 0.1) ** (1.0 / K))) for c, g in lp.groupby("chi") if c in yi]
        if model == "centred" and pts:
            ax.plot([p[1] for p in pts], [p[0] for p in pts], "k:", lw=1.5, marker="s", ms=3,
                    label=r"lazy boundary, first order $(\kappa/0.1)^{1/K}$")
    ax.legend(handles=legend_handles() + ax.get_legend_handles_labels()[0], fontsize=6.5,
              loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, frameon=False)
    ax.set_title(f"{title}regime (majority of {df.seed.nunique()} seeds)", fontsize=9, loc="left")

    piv = df.groupby(["chi", "sigma"]).dK.median().unstack()
    ax = axes[1]
    im = ax.pcolormesh(_edges(piv.columns), np.arange(len(piv.index) + 1) - 0.5, piv.values,
                       norm=LogNorm(1e-4, 1e4), cmap="Blues", shading="flat")
    ax.contour(piv.columns, np.arange(len(piv.index)), piv.values, levels=[0.1], colors=INK, linewidths=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(piv.index)), [str(c) for c in piv.index])
    ax.set_xlabel(r"$\sigma$")
    ax.set_title(r"median kernel change $\delta_K(T)$ (line: 0.1)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.05)

    piv = df.groupby(["chi", "sigma"]).min_rho.median().unstack()
    ax = axes[2]
    lr = np.log10(piv.values.astype(float))
    im = ax.pcolormesh(_edges(piv.columns), np.arange(len(piv.index) + 1) - 0.5, lr,
                       norm=TwoSlopeNorm(vcenter=np.log10(3), vmin=-2, vmax=2.5), cmap="RdBu_r",
                       shading="flat")
    ax.set_xscale("log")
    ax.set_yticks(range(len(piv.index)), [str(c) for c in piv.index])
    ax.set_xlabel(r"$\sigma$")
    ax.set_title(r"median $\log_{10}\min_i\rho_i$ (white: $\rho=3$)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.05)
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
                rows.append(dict(y=y, k_thr=kt, rho_thr=rt,
                                 sigma_s2s=boundary(sig, fr["s2s"].values),
                                 sigma_lazy=boundary(sig, fr["lazy"].values)))
    return pd.DataFrame(rows).rename(columns={"y": ycol})


def fig_sensitivity(exp, model, ycol="chi"):
    df = load(exp)
    df = df[df.model == model]
    K = int(df.K.iloc[0])
    bt = boundaries_table(df, ycol)
    save_csv(bt, f"boundaries_{exp}_{model}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    blues = ["#86b6ef", "#2a78d6", "#104281"]
    oranges = ["#f5b394", "#eb6834", "#a3401a"]
    for c, rt in zip(oranges, RHO_THRESHOLDS):
        b = bt[(bt.k_thr == 0.1) & (bt.rho_thr == rt)]
        axes[0].plot(b[ycol], b.sigma_s2s, "o-", color=c, ms=4, lw=1.5, label=rf"$\rho$ threshold {rt:g}")
    for c, kt in zip(blues, K_THRESHOLDS):
        b = bt[(bt.k_thr == kt) & (bt.rho_thr == 3.0)]
        axes[1].plot(b[ycol], b.sigma_lazy, "s-", color=c, ms=4, lw=1.5, label=rf"$\delta_K$ threshold {kt:g}")
    axes[0].set_title(f"S2S boundary (S2S fraction = 1/2); predicted shift\n"
                      rf"$\rho$ 2$\to$5: $\times(5/2)^{{1/(K-2)}}$ = {2.5 ** (1 / (K - 2)):.2f}", fontsize=8.5, loc="left")
    axes[1].set_title(f"lazy boundary (lazy fraction = 1/2); predicted shift\n"
                      rf"$\delta_K$ 0.05$\to$0.2: $\times4^{{-1/K}}$ = {4 ** (-1 / K):.2f}", fontsize=8.5, loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(ycol)
        ax.set_ylabel(r"boundary $\sigma$")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_fig(fig, f"sensitivity_{exp}_{model}.png")
    return bt


def load_traj(exp, cfg):
    """(t, L/L0, overlaps) for a run of either driver (library Trajectory or interim npz)."""
    from run_regime import row_cfg, run_id

    path = os.path.join(OUT, exp, "traj", run_id(row_cfg(cfg)) + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if "overlaps" in z.files:
            return z["t"], z["loss"] / z["loss"][0], z["overlaps"]
        return z["t"], z["L"], z["A"]
    from pilots.interim import run_id as rid

    z = np.load(os.path.join(OUT, exp, "traj", rid(cfg) + ".npz"))
    return z["t"], z["L"], z["A"]


def fig_examples(exp, picks, name):
    """Representative runs: list of (title, cfg-dict); L/L0 and overlaps a_r vs t."""
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 4.6), sharex=False)
    comp_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, (title, cfg) in enumerate(picks):
        t, L, A = load_traj(exp, cfg)
        t, L, A = t[1:], L[1:], A[1:]
        axes[0, k].loglog(t, L, color=INK, lw=1.5)
        axes[0, k].set_title(title, fontsize=8.5, loc="left")
        for r in range(A.shape[1]):
            axes[1, k].semilogx(t, A[:, r], color=comp_colors[r], lw=1.5, label=f"$a_{r + 1}$ ($s={[3, 2, 1][r]}$)")
        axes[1, k].set_xlabel("gradient-flow time t")
        axes[1, k].set_ylim(-0.1, 1.15)
    axes[0, 0].set_ylabel(r"$L/L_0$")
    axes[1, 0].set_ylabel(r"overlap $a_r$")
    axes[1, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_fig(fig, f"examples_{name}.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["sigma_chi"])
    ap.add_argument("exp")
    ap.add_argument("model")
    ap.add_argument("--lazy", help="per-seed lazy predictions CSV (lazy_predict.py)")
    ap.add_argument("--out", default=RES, help="output directory (default: results/)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    a = ap.parse_args()
    RES, FORCE = a.out, a.force
    if a.what == "sigma_chi":
        fig_sigma_chi(a.exp, a.model, a.lazy, title=f"{a.exp}, {a.model}: ")
        bt = fig_sensitivity(a.exp, a.model)
        print(bt[(bt.k_thr == 0.1) & (bt.rho_thr == 3.0)])
