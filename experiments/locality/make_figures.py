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

sys.path.insert(0, str(Path(__file__).parent))
from analysis import lambda2, plateau_table, summarize  # noqa: E402

TREE_COLOR = {"mps": "#2a78d6", "binary": "#eb6834", "ht": "#1baf7a", "star3": "#4a3aa7",
              "midroot6": "#e87ba4"}


def tree_kind(name):
    for k in ("mps", "binary", "ht", "star3", "midroot6"):
        if name.startswith(k):
            return k
    return name


def main_rows(s):
    """Rows of the two-component table: the greedy-order plateau (U1 learned, then U2)."""
    return s[(s["order"] == "0>1") & (s["j"] == 0)].copy()


def table_md(t):
    lines = ["| tree | S | protocol | d | chi | weights | k (Steiner) | σ range (n) | seeds | predicted | "
             "measured p ± SE | log-law slope ± SE (pred.) | λ_min at smallest σ | λ_min ∝ σ^x |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        k = r["k_rank"]
        pred = {1: "no plateau", 2: "log"}.get(k, f"p = {k - 2}")
        p = f"{r['p_fit']:.3f} ± {r['p_se']:.3f}" if np.isfinite(r["p_fit"]) else "–"
        if k == 1:
            p = f"gap {r['gap_small_med']:.2g} (ρ<3)"
        ls = (f"{r['logslope']:.3f} ± {r['logslope_se']:.3f} ({r['logslope_pred']:.3f})"
              if k == 2 and np.isfinite(r.get("logslope_pred", np.nan)) else "–")
        hm = f"{r['hess_min_small']:.3g}" if np.isfinite(r["hess_min_small"]) else "–"
        hx = (f"{r['hess_exp']:.2f} ± {r['hess_exp_se']:.2f}" if np.isfinite(r["hess_exp"]) else "–")
        lines.append(f"| {r['tree']} | {r['case'].replace('S=', '')} | {r['start']} | {r['d']} | {r['chi']} | "
                     f"{'/'.join(f'{x:g}' for x in r['w'])} | {k} | {r['sigma_range']} ({r['n_sigma']}) | "
                     f"{r['fit_seeds']}/{r['seeds']} | {pred} | {p} | {ls} | {hm} | {hx} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = plateau_table(args.inputs)
    df = df[df["n_comp"] == 2]
    s = summarize(df)
    t = main_rows(s).sort_values(["d", "chi", "w", "start", "k_rank", "tree", "case"])
    t.to_csv(args.out + "_table.csv", index=False, float_format="%.5g")
    Path(args.out + "_table.md").write_text(table_md(t) + "\n")

    # --- exponents: measured vs predicted ---
    base = t[(t["d"] == 2) & (t["chi"] == 2) & (t["w"] == (1.0, 0.5))]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    ax = axes[0]
    for (kind, start), g in base[base["k_rank"] >= 3].groupby(
            [base["tree"].map(tree_kind), "start"]):
        jit = 0.08 * (list(TREE_COLOR).index(kind) - 2) + (0.03 if start == "saddle" else -0.03)
        ax.errorbar(g["p_pred"] + jit, g["p_fit"], yerr=g["p_se"], fmt="o" if start == "small" else "s",
                    ms=4, color=TREE_COLOR[kind], mfc="white" if start == "saddle" else TREE_COLOR[kind],
                    lw=1, label=f"{kind}, {start} init")
    lim = [0.5, base["p_pred"].max() + 0.5]
    ax.plot(lim, lim, "k--", lw=1, label="p = k − 2")
    ax.set_xlabel("predicted exponent k − 2 (k = Steiner subtree size)")
    ax.set_ylabel("measured exponent of t2 − t1 vs 1/σ")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax = axes[1]
    k2 = t[(t["k_rank"] == 2)]
    for start, mk in (("small", "o"), ("saddle", "s")):
        g = k2[k2["start"] == start]
        for _, r in g.iterrows():
            kind = tree_kind(r["tree"])
            ax.errorbar(r["logslope_pred"], r["logslope"], yerr=r["logslope_se"], fmt=mk, ms=4,
                        color=TREE_COLOR[kind], mfc="white" if start == "saddle" else TREE_COLOR[kind])
    for start, mk in (("small", "o"), ("saddle", "s")):
        ax.plot([], [], mk, color="gray", mfc="white" if start == "saddle" else "gray",
                label="small init: (1−s2/s1)/λ2" if start == "small" else "saddle start: 1/λ2")
    m = np.nanmax(k2[["logslope_pred", "logslope"]].values) * 1.1 if len(k2) else 1
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.set_xlabel("predicted slope of t2 − t1 vs ln(1/σ)")
    ax.set_ylabel("measured slope (k = 2 rows)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out + "_exponents.png", dpi=150)

    # --- Hessian: -lambda_min at the plateau midpoint vs sigma, by k ---
    h = df[(df["order"] == "0>1") & (df["d"] == 2) & (df["chi"] == 2) & (df["w"] == (1.0, 0.5))
           & (df["k_rank"] >= 2) & df["hess_min"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
    cmap = plt.get_cmap("viridis")
    kmax = max(int(h["k_rank"].max()), 3) if len(h) else 3
    for ax, start in zip(axes, ("small", "saddle")):
        hs = h[h["start"] == start]
        for (k, tree, case), g in hs.groupby(["k_rank", "tree", "case"]):
            med = g.groupby("sigma")["hess_min"].median()
            ax.plot(med.index, -med.values, "o-", ms=3, lw=1, color=cmap((k - 2) / max(kmax - 2, 1) * 0.9),
                    alpha=0.8)
        for k in sorted(hs["k_rank"].unique()):
            ax.plot([], [], "o-", color=cmap((k - 2) / max(kmax - 2, 1) * 0.9), label=f"k = {k}")
        ax.axhline(0.5, color="k", ls=":", lw=1, label="λ2 = s2 s1^((K−2)/K) = 0.5")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("σ")
        ax.set_title(f"{start} init", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("−λ_min (gauge-projected Hessian, plateau midpoint)")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out + "_hessian.png", dpi=150)
    print(table_md(t))


if __name__ == "__main__":
    main()
