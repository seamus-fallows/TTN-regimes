"""Learning order of GD vs gradient flow below eta_max (binary N=4, init seeds 0-44).

Events are persistent (last upward crossing of 1/2; fix round 1: round-1 first-crossing events
counted transient excursions as events). For each GF reference: the order, the gap between the
first two events, and the peak overlap of the eventually-second component during the first event
(near-tie measure). For each GD run below eta_max: whether its persistent order differs from GF,
and whether the eventually-second component makes a transient crossing of 1/2 (n_cross >= 2).

Run after analyze_gd.py: uv run python experiments/optim_samples/analyze_order.py
-> results/order_flips.csv, fig_order_flips.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps

from common import RAW as RAW_ROOT
from common import RESULTS as RES

RAW = RAW_ROOT / "gd"


def near_tie(sigma, seed):
    base = f"binary4_sig{sigma}_seed{seed}_gf"
    d = json.loads((RAW / f"{base}.json").read_text())
    z = np.load(RAW / f"{base}.npz")
    t50 = np.array([np.nan if v is None else v for v in d["t50"]], float)
    order = np.argsort(np.where(np.isfinite(t50), t50, np.inf))
    first, second = order[0], order[1]
    before = z["t"] < t50[first]
    rival = float(np.max(z["a"][before, second])) if before.any() else np.nan
    # peak of the eventual second component during the first event (until t1 + 2 widths)
    window = z["t"] <= t50[first] + 2 * (d["t90"][first] - d["t10"][first])
    rival_event = float(np.max(z["a"][window, second]))
    return dict(rival_before=rival, rival_peak=rival_event,
                gap_rel=(t50[second] - t50[first]) / t50[first])


def main():
    ps.apply()
    df = pd.read_csv(RES / "gd_sweep.csv")
    df = df[(df.tree == "binary") & (df.n == 4) & ~df.centred & df.root_scale.isna()
            & (df.ratio < 1)]
    ties = {(s, sd): near_tie(s, sd) for s, sd in df[["sigma", "seed"]].drop_duplicates().itertuples(index=False)}
    for key in ("rival_peak", "gap_rel"):
        df[key] = [ties[(s, sd)][key] for s, sd in zip(df.sigma, df.seed)]
    df["flip"] = ~df.same_order.astype(bool)
    df["transient"] = [max(int(x) for x in str(c).strip("[]").split(",")) >= 2 for c in df.n_cross]
    out = df[["sigma", "seed", "ratio", "flip", "transient", "n_cross", "rival_peak", "gap_rel",
              "t1_gf", "t2_gf", "t1_gd", "t2_gd"]]
    out.to_csv(RES / "order_flips.csv", index=False)
    summ = df.groupby(["sigma", "ratio"]).agg(n=("flip", "size"), p_flip=("flip", "mean"),
                                               p_transient=("transient", "mean"))
    summ["p_flip_se"] = np.sqrt(summ.p_flip * (1 - summ.p_flip) / summ.n)
    summ["p_transient_se"] = np.sqrt(summ.p_transient * (1 - summ.p_transient) / summ.n)
    print(summ.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    big = summ[summ.n >= 45].reset_index()
    for i, sig in enumerate(sorted(big.sigma.unique())):
        g = big[big.sigma == sig]
        ax.errorbar(g.ratio, g.p_transient, yerr=g.p_transient_se, fmt="o-", ms=3.5, lw=1,
                    color=ps.BLUES[2 * i + 1], label=f"transient crossing, σ = {sig:g} ({int(g.n.iloc[0])} seeds)")
        ax.plot(g.ratio, g.p_flip, "x--", ms=5, lw=0.8, color=ps.BLUES[2 * i + 1],
                label=f"persistent order differs, σ = {sig:g}")
    ax.set(xscale="log", xlabel="η / η_max", ylabel="fraction of inits",
           title="(a) below η_max: transient crossings, no order change")
    ax.legend(fontsize=6, loc="upper left")
    ax = axes[1]
    per_seed = df[np.isclose(df.ratio, 0.9)].groupby(["sigma", "seed"]).agg(
        flip=("transient", "max"), rival=("rival_peak", "first"))
    for i, sig in enumerate(sorted(df.sigma.unique())):
        g = per_seed.loc[sig]
        ax.scatter(g.rival, g.flip.astype(float) + np.random.default_rng(i).uniform(-0.04, 0.04, len(g)),
                   s=14, color=ps.BLUES[i + 1], label=f"σ = {sig:g}")
    ax.set(xlabel="GF: peak overlap of the eventual second component during the first event",
           ylabel="transient crossing at 0.9 η_max", yticks=[0, 1], yticklabels=["no", "yes"],
           title="(b) transients happen at near-ties")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(RES / "fig_order_flips.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
