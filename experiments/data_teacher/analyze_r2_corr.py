"""Round 2, item 3: sigma law of the second escape under cross-site correlated inputs (r2_corr.py runs).

Per (tree, rho): gap = t(s_2 event) - t(s_1 event) on the 01|23 cut (final-rise events,
theory_covariance.md 10.10), per seed vs sigma (6 values, factor 2 apart, 1.5 decades, 5 seeds):
- per-seed log and power law fits in log space (lawfit.py), mean +- s.e.m.;
- increments per halving of sigma: per-seed slope of the gap vs log2(1/sigma) (actual sigma values),
  over all sigma and over the 3 smallest, mean +- s.e.m. over seeds (theory_covariance.md: MPS4,
  d = 3, s = (1, 0.6): 2.128 at rho = 0.4 and 14.39 at rho = 0.2, two-stage law through the
  junk-rich saddle; this repeats theory_covariance.md check N14 with the same grid and events);
- separation ratio rho_sep = gap / max(final-rise widths) (S2S needs > 3), over all runs;
- runs are classified by the first saddle they visit (major / minor; see main()); fits use major-saddle runs;
- gaps also in units of 1/L(0): gap x L(0)(rho) / L(0)(rho = 0) (the review: t_1 falls 14x while
  L(0) grows 12x between rho = 0 and 0.5).
Usage: uv run python experiments/data_teacher/analyze_r2_corr.py
Writes results/r2_corr_runs.csv (refreshed from the raw files), r2_corr_fits.csv,
r2_corr_increments_perseed.csv, r2_corr_law.png.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_ROOT, RESULTS  # noqa: E402
from lawfit import per_seed  # noqa: E402

PRED_INC = {("mps", 0.4): 2.128, ("mps", 0.2): 14.39}


def main():
    rows = [json.loads(str(np.load(f)["summary"])) for f in sorted((OUT_ROOT / "r2_corr").glob("*.npz"))]
    df = pd.DataFrame(rows)
    l00 = df[df.rho == 0.0].groupby("tree")["loss0"].first()
    df["gap_norm"] = df["gap"] * df["loss0"] / df["tree"].map(l00)
    # visited first saddle: if the top singular value's final rise spans the gap, the first plateau was
    # the minor one (the 0.6 component learned first; on MPS the order is then not corrected and the
    # "gap" is not a plateau length). Fits use major-saddle runs only.
    df["saddle"] = np.where(df["width1"] > 0.5 * df["gap"], "minor", "major")
    df.to_csv(RESULTS / "r2_corr_runs.csv", index=False)
    print(df.groupby(["tree", "rho", "saddle"]).size().unstack(fill_value=0).to_string())
    df_all = df
    df = df[df.saddle == "major"]
    ps, summ = per_seed(df, ["tree", "rho"], min_points=4)
    # increment per halving of sigma = per-seed slope of gap vs log2(1/sigma) with the actual (rounded)
    # sigma values (review of round 2: raw differences ignore the grid ratios 1.94-2.03, and pooling 20
    # increments as independent understates the error); all sigma and the 3 smallest; s.e.m. over seeds
    inc = []
    for (tree, rho, seed), g in df.groupby(["tree", "rho", "seed"]):
        g = g.sort_values("sigma")
        x, y = np.log2(1 / g["sigma"].values), g["gap"].values
        if len(g) >= 3:
            inc.append(dict(tree=tree, rho=rho, seed=seed, n_sigma=len(g), inc_all=np.polyfit(x, y, 1)[0],
                            inc_small3=np.polyfit(x[:3], y[:3], 1)[0]))
    inc = pd.DataFrame(inc)
    inc.to_csv(RESULTS / "r2_corr_increments_perseed.csv", index=False)
    inc_s = inc.groupby(["tree", "rho"]).agg(
        n_seeds_inc=("seed", "size"), inc_mean=("inc_all", "mean"), inc_sem=("inc_all", "sem"),
        inc_small3_mean=("inc_small3", "mean"), inc_small3_sem=("inc_small3", "sem")).reset_index()
    sep = df_all.groupby(["tree", "rho"]).agg(sep_min=("sep_ratio", "min"), sep_median=("sep_ratio", "median"),
                                          frac_sep_gt3=("sep_ratio", lambda x: float((x > 3).mean())),
                                          loss0=("loss0", "first")).reset_index()
    summ = summ.merge(inc_s, on=["tree", "rho"]).merge(sep, on=["tree", "rho"])
    summ["pred_inc_per_halving"] = [PRED_INC.get((t, r), np.nan) for t, r in zip(summ.tree, summ.rho)]
    summ.to_csv(RESULTS / "r2_corr_fits.csv", index=False)
    print(summ.round(4).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, tree in enumerate(("mps", "binary")):
        ax = axes[j]
        for i, (rho, g) in enumerate(df[df.tree == tree].groupby("rho")):
            m = g.groupby("sigma")["gap"].agg(["mean", "sem"])
            s = summ[(summ.tree == tree) & (summ.rho == rho)].iloc[0]
            ax.errorbar(m.index, m["mean"], m["sem"], fmt="o-", color=f"C{i}", ms=4,
                        label=f"rho={rho}: p={s.p_mean:.2f}, A={s.A_mean:.1f}, log wins {s.log_wins}/{s.n_seeds}")
        ax.set(xscale="log", yscale="log", xlabel="sigma", ylabel="gap (s_2 event - s_1 event)",
               title=f"{tree}4, correlated inputs (d = 3, s = (1, 0.6))")
        ax.legend(fontsize=6)
    ax = axes[2]
    for i, (rho, g) in enumerate(df[df.tree == "mps"].groupby("rho")):
        if rho == 0.0:
            continue
        m = g.groupby("sigma")["gap"].mean()
        ax.plot(np.log2(1 / m.index.values), m.values, "o-", color=f"C{i}", label=f"mps4 rho={rho}")
        if ("mps", rho) in PRED_INC:
            x = np.log2(1 / m.index.values)
            ax.plot(x, m.values[-1] + PRED_INC[("mps", rho)] * (x - x[-1]), "k--", lw=0.8)
    ax.set(xlabel="log2(1/sigma)", ylabel="gap (linear)",
           title="mps4: gap vs halvings of sigma (dashed: predicted increments 2.128, 14.39)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_corr_law.png", dpi=100)


if __name__ == "__main__":
    main()
