"""Step-3 analysis and figure: test error and effective bond rank vs training-set size n.

Reads /root/outputs/optim_samples/samples/*.jsonl (samples_sweep.py). Test error is relative to the
zero predictor, test_rel = ||W_learned - W*||^2 / ||W*||^2. Effective bond rank: number of bond
singular values of the learned tensor above 0.1 x (smallest teacher weight), averaged over edges.

Run: uv run python experiments/optim_samples/analyze_samples.py
Writes results/samples_runs.csv, samples_summary.csv, fig_samples.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps

RAW = Path("/root/outputs/optim_samples/samples")
OUT = Path(__file__).parent / "results"


def load():
    rows, curves = [], []
    for f in RAW.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            tk = d["task"]
            base = dict(block=tk["block"], chi=tk["chi"], centred=tk["centred"], sigma=tk["sigma"],
                        seed=tk["seed"], n=tk["n_train"])
            rows.append(dict(base, method="GD", status=d["status"], test_rel=d["test_rel"],
                             train_rel=d["train_loss"], eff_rank=np.mean(d["eff_ranks"]),
                             eff_ranks=str(d["eff_ranks"]), ent_rank=np.mean(d["entropy_ranks"]),
                             rel_change=d["rel_param_change"], t_end=d["t_end"]))
            if "lazy_test_rel" in d:
                rows.append(dict(base, method="lazy limit", status="exact",
                                 test_rel=d["lazy_test_rel"], train_rel=np.nan,
                                 eff_rank=np.mean(d["lazy_eff_ranks"]),
                                 eff_ranks=str(d["lazy_eff_ranks"]),
                                 ent_rank=np.mean(d["lazy_entropy_ranks"]), rank_A=d["rank_A"],
                                 rank_J=d["rank_J"]))
            for t, tr, te in zip(d["curve_t"], d["curve_train"], d["curve_test"]):
                curves.append(dict(base, t=t, train_rel=tr, test_rel=te / (0.5 * 13.0)))
    return pd.DataFrame(rows), pd.DataFrame(curves)


def label(row):
    if row.method == "lazy limit":
        return "lazy limit (kernel regression, exact)"
    if row.centred:
        return f"centred, σ = {row.sigma:g} (GD)"
    return f"small init, σ = {row.sigma:g} (GD)"


def main():
    df, curves = load()
    if df.empty:
        return
    df["label"] = df.apply(label, axis=1)
    df.to_csv(OUT / "samples_runs.csv", index=False)
    summ = (df.groupby(["chi", "label", "n"])
            .agg(test_mean=("test_rel", "mean"), test_se=("test_rel", "sem"),
                 test_median=("test_rel", "median"), eff_rank=("eff_rank", "mean"),
                 eff_rank_se=("eff_rank", "sem"), ent_rank=("ent_rank", "mean"),
                 frac_converged=("status", lambda s: np.mean(s.isin(["converged", "exact"]))),
                 train_rel=("train_rel", "mean"), rel_change=("rel_change", "mean"),
                 n_seeds=("seed", "size"))
            .reset_index())
    summ.to_csv(OUT / "samples_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(summ.to_string(float_format=lambda v: f"{v:.3g}"))
    curves.to_csv(OUT / "samples_curves.csv", index=False)

    ps.apply()
    labels = sorted(df.label.unique(), key=lambda s: (s.startswith("lazy"), s.startswith("centred"), s))
    colors = {lab: ps.CAT[i] for i, lab in enumerate(labels)}
    chis = sorted(df.chi.unique())
    fig, axes = plt.subplots(2, len(chis), figsize=(4.6 * len(chis), 6.2), squeeze=False)
    marks = {4: [(28, "rank-2 manifold dim 28"), (52, "teacher-rank params 52"),
                 (112, "tangent dim 112")],
             16: [(28, "28"), (52, "52"), (256, "d^N = 256 = tangent dim")]}
    for j, chi in enumerate(chis):
        sub = summ[summ.chi == chi]
        ax = axes[0, j]
        for lab in labels:
            g = sub[sub.label == lab]
            if g.empty:
                continue
            ax.errorbar(g.n, g.test_mean, yerr=g.test_se, fmt="o-", color=colors[lab], ms=3.5,
                        lw=1.2, label=lab)
        for x, txt in marks.get(chi, []):
            ax.axvline(x, color=ps.MUTED, lw=0.7, ls="--")
            ax.text(x * 1.04, 1.02, txt, transform=ax.get_xaxis_transform(), fontsize=6.5,
                    color=ps.INK2, rotation=90, va="bottom", ha="left")
        ax.axhline(1, color=ps.INK2, lw=0.6)
        ax.set(xscale="log", yscale="log", xlabel="training-set size n",
               ylabel="test error / ||W*||² (mean ± s.e., 5 seeds)",
               title=f"student χ = {chi}: generalization")
        ax.legend(loc="lower left", fontsize=6.5)
        ax = axes[1, j]
        for lab in labels:
            g = sub[sub.label == lab]
            if g.empty:
                continue
            ax.errorbar(g.n, g.eff_rank, yerr=g.eff_rank_se, fmt="o-", color=colors[lab], ms=3.5,
                        lw=1.2, label=lab)
        ax.axhline(2, color=ps.INK2, lw=0.6)
        ax.text(0.99, 2, " teacher rank 2", transform=ax.get_yaxis_transform(), fontsize=6.5,
                color=ps.INK2, va="bottom", ha="right")
        ax.set(xscale="log", xlabel="training-set size n",
               ylabel="effective bond rank (mean over edges)",
               title=f"student χ = {chi}: rank of the learned tensor")
    fig.tight_layout()
    fig.savefig(OUT / "fig_samples.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
