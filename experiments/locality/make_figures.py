"""Exponent table and summary figures for the two-component sweep (steps 1, 2, 5).

usage: uv run python experiments/locality/make_figures.py <prefix-or-jsonl> [...] --out experiments/locality/results/<tag>
writes <tag>_table.csv, <tag>_table.md, <tag>_exponents.png, <tag>_hessian.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analysis import lambda2, plateau_table, run_classes, summarize  # noqa: E402
from rules import get_tree  # noqa: E402

from ttn import plotting as tp  # noqa: E402

# Tree family colours of every report figure (MPS blue, binary orange, HT green); the order sets the
# horizontal jitter of the exponent figure.
TREE_COLOR = {
    "mps": "#2a78d6",
    "binary": "#eb6834",
    "ht": "#1baf7a",
    "star3": "#4a3aa7",
    "midroot6": "#e87ba4",
}
FAMILY_NAME = {
    "mps": "MPS",
    "binary": "binary",
    "ht": "HT",
    "star3": "star3",
    "midroot6": "midroot6",
}
FIG_WIDTH = 0.86 * tp.TEXT_WIDTH  # the report includes these figures at 0.86 of the text width


def _tree_args(name):
    """(kind, n) for get_tree from a tree name such as 'mps8' or 'star3'."""
    if name.startswith(("star", "midroot")):
        return name, None
    kind = name.rstrip("0123456789")
    return kind, int(name[len(kind) :])


def tree_kind(name):
    for k in ("mps", "binary", "ht", "star3", "midroot6"):
        if name.startswith(k):
            return k
    return name


def main_rows(s):
    """Rows of the two-component table: the greedy-order plateau (U1 learned, then U2)."""
    return s[(s["order"] == "0>1") & (s["j"] == 0)].copy()


def table_md(t):
    lines = [
        "| tree | S | protocol | d | chi | weights | k (Steiner) | σ range (n; decades) | seeds fitted | runs: greedy / "
        "other order / rotation / stalled / timeout | predicted | measured p ± SE (median) | p, 3 smallest σ | "
        "log-law slope ± SE (pred.) | RMS log-res. log / power (law) | λ_min proj. / raw, smallest σ | "
        "λ_min ∝ σ^x (proj. / raw) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        k = int(r["k_rank"]) if np.isfinite(r.get("k_rank", np.nan)) else int(r["k_steiner"])
        pred = {1: "no plateau", 2: "log"}.get(k, f"p = {k - 2}")
        runs = f"{r['greedy']} / {r['other_order']} / {r['rotation']} / {r['stalled']} / {r['timeout']}"
        if (
            not np.isfinite(r.get("p_fit", np.nan))
            and k != 1
            and not np.isfinite(r.get("logslope", np.nan))
        ):
            lines.append(
                f"| {r['tree']} | {r['case'].replace('S=', '')} | {r['start']} | {r['d']} | {r['chi']} | "
                f"{'/'.join(f'{x:g}' for x in r['w'])} | {k} | – | 0 | {runs} | {pred} | no escaping run | "
                "– | – | – | – | – |"
            )
            continue
        p = (
            f"{r['p_fit']:.3f} ± {r['p_se']:.3f} ({r['p_median']:.2f})"
            if np.isfinite(r["p_fit"])
            else "–"
        )
        pl = (
            f"{r['p_local']:.2f} ± {r['p_local_se']:.2f}"
            if np.isfinite(r["p_local"]) and k >= 3
            else "–"
        )
        if k == 1:
            p = f"gap {r['gap_small_med']:.2g}, ρ = {r['rho_small_med']:.2g}"
        ls = (
            f"{r['logslope']:.3f} ± {r['logslope_se']:.3f} ({r['logslope_pred']:.3f})"
            if k == 2 and np.isfinite(r.get("logslope_pred", np.nan))
            else "–"
        )
        law = (
            f"{r['rms_log_law']:.3f} / {r['rms_power_law']:.3f} ({r['law']})"
            if np.isfinite(r["rms_log_law"])
            else "–"
        )
        raw = r.get("hess_raw_min_small", np.nan)
        hm = f"{r['hess_min_small']:.4g} / {raw:.4g}" if np.isfinite(r["hess_min_small"]) else "–"
        hx = f"{r['hess_exp']:.2f} / {r['hess_raw_exp']:.2f}" if np.isfinite(r["hess_exp"]) else "–"
        lines.append(
            f"| {r['tree']} | {r['case'].replace('S=', '')} | {r['start']} | {r['d']} | {r['chi']} | "
            f"{'/'.join(f'{x:g}' for x in r['w'])} | {k} | {r['sigma_range']} ({r['n_sigma']}; "
            f"{r['decades']:.1f}) | {r['fit_seeds']}/{r['seeds']} | {runs} | {pred} | {p} | {pl} | {ls} | "
            f"{law} | {hm} | {hx} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--config", nargs="*", default=None, help="keep only runs of these YAML configs"
    )
    ap.add_argument("--hessians", default=str(Path(__file__).parent / "results" / "hessians.csv"))
    args = ap.parse_args()
    df = plateau_table(args.inputs, config=args.config, hessians=args.hessians)
    df = df[df["n_comp"] == 2]
    s = summarize(df)
    t = main_rows(s)
    cls = run_classes(args.inputs, config=args.config).reset_index()
    # outer join: cases in which no run escapes stay in the table
    t = cls.merge(t, on=["tree", "case", "start", "d", "chi", "w"], how="left")
    t = t[t["case"].str.startswith("S=")]
    t["k_steiner"] = [
        len(get_tree(*_tree_args(tr)).steiner_nodes([int(ch) for ch in c[2:]]))
        for tr, c in zip(t["tree"], t["case"])
    ]
    for c in ("greedy", "other_order", "rotation", "stalled", "timeout", "incomplete", "error"):
        t[c] = t[c].fillna(0).astype(int)
    t = t.sort_values(["d", "chi", "w", "start", "k_steiner", "tree", "case"])
    t.to_csv(args.out + "_table.csv", index=False, float_format="%.5g")
    Path(args.out + "_table.md").write_text(table_md(t) + "\n", encoding="utf-8")

    # --- exponents: measured vs predicted ---
    # Figures at their printed size (0.86 of the report's text width) in the report style, so that
    # the text is 8-9 pt in print.
    tp.set_style()
    t = t[t["p_fit"].notna() | t["logslope"].notna()].copy()
    t["k_rank"] = t["k_rank"].astype(int)
    base = t[(t["d"] == 2) & (t["chi"] == 2) & (t["w"] == (1.0, 0.5))]
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, 2.9), layout="constrained", width_ratios=[1.3, 1]
    )
    ax = axes[0]
    for (kind, start), g in base[base["k_rank"] >= 3].groupby(
        [base["tree"].map(tree_kind), "start"]
    ):
        jit = 0.08 * (list(TREE_COLOR).index(kind) - 2) + (0.03 if start == "saddle" else -0.03)
        ax.errorbar(
            g["p_pred"] + jit,
            g["p_fit"],
            yerr=g["p_se"],
            fmt="o" if start == "small" else "s",
            ms=3,
            color=TREE_COLOR[kind],
            mfc="white" if start == "saddle" else TREE_COLOR[kind],
            lw=0.8,
            capsize=0,
        )
    lim = [0.5, base["p_pred"].max() + 0.5]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel(r"predicted exponent $k-2$")
    ax.set_ylabel(r"measured exponent $p$ of $\Delta t_2$")
    ax = axes[1]
    k2 = t[(t["k_rank"] == 2)]
    for start, mk in (("small", "o"), ("saddle", "s")):
        g = k2[k2["start"] == start]
        for _, r in g.iterrows():
            kind = tree_kind(r["tree"])
            ax.errorbar(
                r["logslope_pred"],
                r["logslope"],
                yerr=r["logslope_se"],
                fmt=mk,
                ms=3,
                lw=0.8,
                color=TREE_COLOR[kind],
                mfc="white" if start == "saddle" else TREE_COLOR[kind],
            )
    m = np.nanmax(k2[["logslope_pred", "logslope"]].values) * 1.1 if len(k2) else 1
    ax.plot([0, m], [0, m], "k--", lw=0.8)
    ax.annotate(
        "small initialization:\n$(1-s_2/s_1)/\\lambda_2=1$",
        (1, 1),
        xytext=(1.12, 0.45),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color=tp.MUTED, lw=0.5),
    )
    ax.annotate(
        "saddle start:\n$1/\\lambda_2=2$",
        (2, 2),
        xytext=(1.25, 1.75),
        fontsize=8,
        ha="right",
        arrowprops=dict(arrowstyle="-", color=tp.MUTED, lw=0.5),
    )
    ax.set_xlabel(r"predicted slope of $\Delta t_2$ against $\ln(1/\sigma)$")
    ax.set_ylabel(r"measured slope ($k=2$ rows)")
    handles = [
        Line2D([], [], color=c, marker="o", ls="none", ms=4, label=FAMILY_NAME[kind])
        for kind, c in TREE_COLOR.items()
    ]
    handles += [
        Line2D([], [], color=tp.INK_SECONDARY, marker="o", ls="none", ms=4, label="small init."),
        Line2D(
            [],
            [],
            color=tp.INK_SECONDARY,
            marker="s",
            mfc="white",
            ls="none",
            ms=4,
            label="saddle start",
        ),
        Line2D([], [], color="k", ls="--", lw=0.8, label="equality"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    fig.savefig(args.out + "_exponents.png", dpi=300)

    # --- Hessian: -lambda_min at the plateau midpoint vs sigma, by k ---
    hc = "hess_lib_min" if "hess_lib_min" in df and df["hess_lib_min"].notna().any() else "hess_min"
    h = df[
        (df["order"] == "0>1")
        & (~df["rotation"])
        & (df["d"] == 2)
        & (df["chi"] == 2)
        & (df["w"] == (1.0, 0.5))
        & (df["k_rank"] >= 2)
        & df[hc].notna()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.8), sharey=True, layout="constrained")
    cmap = plt.get_cmap("viridis")
    kmax = max(int(h["k_rank"].max()), 3) if len(h) else 3

    def kcol(k):  # viridis without its pale yellow end, which is hard to see on white
        return cmap((k - 2) / max(kmax - 2, 1) * 0.85)

    for ax, start in zip(axes, ("small", "saddle")):
        hs = h[h["start"] == start]
        for (k, tree, case), g in hs.groupby(["k_rank", "tree", "case"]):
            med = g.groupby("sigma")[hc].median()
            ax.plot(med.index, -med.values, "o-", ms=2.2, lw=0.8, color=kcol(k), alpha=0.85)
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_title("small initialization" if start == "small" else "saddle start", fontsize=8)
    axes[0].set_ylabel(r"$-\lambda_{\min}(H_\perp)$ at the plateau midpoint")
    handles = [
        Line2D([], [], color=kcol(k), marker="o", ms=3, lw=0.8, label=f"$k={k}$")
        for k in sorted(h["k_rank"].unique())
    ]
    handles.append(
        Line2D([], [], color="k", ls=":", lw=0.8, label=r"$\lambda_2=s_2s_1^{(K-2)/K}=0.5$")
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    fig.savefig(args.out + "_hessian.png", dpi=300)
    print(len(t), "escaping rows")


if __name__ == "__main__":
    main()
