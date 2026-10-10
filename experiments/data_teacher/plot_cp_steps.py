"""Figures for steps 2, 3, 4(a) from results/<name>_plateaus.csv and the rank-1 predictions.

Usage: uv run python experiments/data_teacher/plot_cp_steps.py <config name> <sweep key> [pred csv]
Panel 1: first-plateau level (L / L(0)) per run vs the swept parameter, with every predicted rank-1
critical value (lines; marker size = how many of the random starts found it).
Panel 2: plateau duration t_2 - t_1. Panel 3 (if available): alignment of W with the race and saddle
tensors early in the escape and on the plateau. Writes results/<name>_summary.png.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import RESULTS  # noqa: E402

HERE = Path(__file__).parent


def main():
    name, key = sys.argv[1], sys.argv[2]
    pred_csv = sys.argv[3] if len(sys.argv) > 3 else "rank1_predictions.csv"
    df = pd.read_csv(RESULTS / f"{name}_plateaus.csv")
    pr = pd.read_csv(RESULTS / pred_csv)
    pr = pr[(pr.exp == name) & (pr.kind == "saddle")]
    groups = [k for k in ("alpha", "s2") if k in df and df[k].nunique() > 1]
    gvals = sorted(df[groups[0]].unique()) if groups else [None]
    variants = sorted(df["variant"].unique()) if "variant" in df else [None]
    trees = sorted(df["tree"].unique())
    has_cos = "cos_race_early" in df
    ncol = 3 if has_cos else 2
    fig, axes = plt.subplots(len(trees) * len(gvals), ncol, figsize=(5 * ncol, 3.6 * len(trees) * len(gvals)),
                             squeeze=False)
    row = 0
    for tree in trees:
        for gv in gvals:
            sub = df[df.tree == tree]
            if gv is not None:
                sub = sub[sub[groups[0]] == gv]
            ax = axes[row, 0]
            for v, mk, c in zip(variants, "os", ("C3", "C4")):
                s = sub if v is None else sub[sub.variant == v]
                jitter = (0.004 if v == variants[0] else -0.004) * (s[key].max() - s[key].min() + 1e-9)
                ax.plot(s[key] + jitter, s["plateau_rel"], mk, color=c, ms=3, alpha=0.5,
                        label=f"runs ({v})" if v else "runs")
            p = pr[pr.tree == tree]
            if gv is not None:
                p = p[p.tag.str.contains(f"{groups[0]}{gv}_") | p.tag.str.endswith(f"{groups[0]}{gv}")]
            for _, q in p.iterrows():
                x = float(q.tag.split(key)[1].split("_")[0])
                ax.plot(x, q.rel_loss, "k_", ms=14, mew=2 if q["count"] > 2 else 0.8)
            ax.set(xlabel=key, ylabel="first plateau L / L(0)",
                   title=f"{name}: {tree}-4" + (f", {groups[0]}={gv}" if gv is not None else ""))
            ax.legend(fontsize=7)
            ax = axes[row, 1]
            for v, mk, c in zip(variants, "os", ("C3", "C4")):
                s = sub if v is None else sub[sub.variant == v]
                m = s.groupby(key)["plateau_duration"].agg(["median", "min", "max"])
                ax.errorbar(m.index, m["median"], [m["median"] - m["min"], m["max"] - m["median"]],
                            fmt=mk + "-", color=c, capsize=2, label=v)
            ax.set(xlabel=key, ylabel="t_2 - t_1 (median, range)", yscale="log",
                   title="rank-1 plateau duration")
            ax.legend(fontsize=7)
            if has_cos:
                ax = axes[row, 2]
                s = sub if variants[0] is None else sub[sub.variant == variants[0]]
                for col, mk, c in [("cos_race_early", "o", "C0"), ("cos_saddle_early", "s", "C1"),
                                   ("cos_saddle_plateau", "^", "C2")]:
                    if col in s:
                        m = s.groupby(key)[col].median()
                        ax.plot(m.index, m.values, mk + "-", color=c, label=col)
                ax.set(xlabel=key, ylabel="|cos(W, X)| (median)", ylim=(0, 1.02),
                       title="alignment: race vs saddle tensors")
                ax.legend(fontsize=7)
            row += 1
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_summary.png", dpi=100)


if __name__ == "__main__":
    main()
