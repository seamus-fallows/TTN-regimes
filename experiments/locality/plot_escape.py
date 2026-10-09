"""Figures for step 1/2: escape time t2 - t1 vs sigma by predicted k, and the Hessian's most negative
eigenvalue at the plateau midpoint vs sigma.

usage: uv run python experiments/locality/plot_escape.py <results.jsonl> [more.jsonl ...] --out <prefix>
"""
import argparse
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fit import load  # noqa: E402

TREE_COLOR = {"mps": "#2a78d6", "binary": "#eb6834", "ht": "#1baf7a", "star3": "#4a3aa7",
              "midroot6": "#e87ba4"}
MARKERS = "osD^vP*X<>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pandas as pd

    df = pd.concat([load(f) for f in args.files], ignore_index=True)
    df = df[df["dt"].notna()]
    ks = sorted(df["k_pred"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.3 * len(ks), 3.2), squeeze=False)
    for ax, k in zip(axes[0], ks):
        g = df[df["k_pred"] == k]
        for i, ((kind, n, S), gc) in enumerate(g.groupby(["kind", "n", "S"], dropna=False)):
            med = gc.groupby("sigma")["dt"].agg(["median", "min", "max"]).reset_index()
            x = 1 / med["sigma"]
            label = f"{kind}{'' if pd.isna(n) else int(n)} S={S}"
            ax.plot(x, med["median"], marker=MARKERS[i % len(MARKERS)], ms=4, lw=1.2,
                    color=TREE_COLOR.get(kind, "gray"), label=label)
            ax.fill_between(x, med["min"], med["max"], color=TREE_COLOR.get(kind, "gray"), alpha=0.12,
                            lw=0)
        ax.set_xscale("log")
        if k >= 3:
            ax.set_yscale("log")
            xs = np.array([x.min(), x.max()]) if len(g) else None
            x0 = (1 / g["sigma"]).min()
            y0 = g[g["sigma"] == g["sigma"].max()]["dt"].median()
            xr = np.logspace(np.log10(x0), np.log10((1 / g["sigma"]).max()), 10)
            ax.plot(xr, y0 * (xr / x0) ** (k - 2), "k--", lw=1, label=f"slope {k - 2}")
        if k == 2:
            s1, s2 = g["weights"].iloc[0][:2]
            for (kind, n), gk in g.groupby(["kind", "n"], dropna=False):
                K = gk["K"].iloc[0]
                lam = s2 * s1 ** ((K - 2) / K)
                xr = np.logspace(np.log10((1 / gk["sigma"]).min()), np.log10((1 / gk["sigma"]).max()), 10)
                base = gk[gk["sigma"] == gk["sigma"].max()]["dt"].median()
                ax.plot(xr, base + (1 - s2 / s1) / lam * np.log(xr / xr[0]), "k--", lw=1)
                ax.plot(xr, base + 1 / lam * np.log(xr / xr[0]), ":", color="gray", lw=1)
            ax.plot([], [], "k--", lw=1, label="(1-s2/s1)/λ log")
            ax.plot([], [], ":", color="gray", lw=1, label="naive 1/λ log")
        if k == 1:
            ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"predicted k = {k}")
        ax.set_xlabel("1/σ")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6, frameon=False)
    axes[0][0].set_ylabel("t2 − t1 (gradient-flow time)")
    fig.tight_layout()
    fig.savefig(args.out + "_escape.png", dpi=150)

    if "hess_min" in df:
        h = df[df["hess_min"].notna() & (df["k_pred"] >= 2)]
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        cmap = plt.get_cmap("viridis")
        kmax = max(h["k_pred"].max(), 3)
        for k, gk in h.groupby("k_pred"):
            med = gk.groupby("sigma")["hess_min"].median().reset_index()
            col = cmap((k - 2) / max(kmax - 2, 1) * 0.9)
            ax.plot(med["sigma"], -med["hess_min"], marker="o", ms=4, lw=1.2, color=col,
                    label=f"k = {k}")
            sig = np.array(sorted(gk["sigma"].unique()))
            if k >= 3:
                y0 = -med["hess_min"].iloc[-1]
                ax.plot(sig, y0 * (sig / sig[-1]) ** (k - 2), "--", color=col, lw=0.8)
        s1, s2 = h["weights"].iloc[0][:2]
        ax.axhline(s2, color="k", ls=":", lw=1, label="λ2 = s2 (s1 = 1)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("σ")
        ax.set_ylabel("−λ_min of gauge-projected Hessian\nat plateau midpoint")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(args.out + "_hessian.png", dpi=150)


if __name__ == "__main__":
    main()
