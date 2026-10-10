"""Accuracy of the first-order prediction of GD event shifts, per event and per seed.

For every GD run below eta_max (plain model, chi = 3, binary N=4 / binary N=8 / MPS N=8) and every
event present in both GD and GF (persistent events), the measured shift (t^GD - t^GF)/eta is compared
with c_e from predict_shift.py (variational equation along the GF path). Reported: relative deviation
|measured/c_e - 1| per run, its distribution per (tree, event, eta/eta_max), the fraction of
comparisons within 0.3%, 1%, 5%, and a Richardson extrapolation to eta -> 0 from the 0.01 and
0.03 eta_max runs (linear in eta), which tests c_e independently of the O(eta^2) term.

Run after analyze_gd.py: uv run python experiments/optim_samples/analyze_shift.py
Writes shift_accuracy_runs.csv, shift_accuracy.csv, shift_richardson.csv, fig_shift_accuracy.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps
from common import RESULTS as RES


def main():
    df = pd.read_csv(RES / "gd_sweep.csv")
    df = df[~df.centred & df.root_scale.isna() & (df.chi == 3) & (df.ratio < 1)
            & (df.ratio_ref == "teacher")]
    rows = []
    for _, r in df.iterrows():
        for e in (1, 2, 3):
            meas, pred = r.get(f"shift_over_eta{e}"), r.get(f"pred_shift_over_eta{e}")
            if pd.isna(meas) or pd.isna(pred):
                continue
            rows.append(dict(tree=f"{r.tree}{r.n}", sigma=r.sigma, seed=r.seed, ratio=r.ratio,
                             event=e, measured=meas, predicted=pred,
                             rel_dev=abs(meas / pred - 1) if pred != 0 else np.inf,
                             n_cross=r.n_cross))
    runs = pd.DataFrame(rows)
    runs.to_csv(RES / "shift_accuracy_runs.csv", index=False)
    summ = (runs.groupby(["tree", "event", "ratio"])
            .agg(n=("rel_dev", "size"), median=("rel_dev", "median"), max=("rel_dev", "max"),
                 within_0p3pct=("rel_dev", lambda v: np.mean(v <= 0.003)),
                 within_1pct=("rel_dev", lambda v: np.mean(v <= 0.01)),
                 within_5pct=("rel_dev", lambda v: np.mean(v <= 0.05)))
            .reset_index())
    summ.to_csv(RES / "shift_accuracy.csv", index=False)
    low = runs[runs.ratio <= 0.1]
    print(f"eta <= 0.1 eta_max: {len(low)} comparisons; within 0.3%: {np.mean(low.rel_dev <= 0.003):.2f},"
          f" within 1%: {np.mean(low.rel_dev <= 0.01):.2f}, within 5%: {np.mean(low.rel_dev <= 0.05):.2f}")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summ.to_string(float_format=lambda v: f"{v:.3g}"))
    # Richardson: shift/eta = c + b eta  =>  c ~ (3 s(0.01) - s(0.03)) / 2 (ratios 0.01, 0.03 of eta_max)
    a = runs[np.isclose(runs.ratio, 0.01)].set_index(["tree", "sigma", "seed", "event"])
    b = runs[np.isclose(runs.ratio, 0.03)].set_index(["tree", "sigma", "seed", "event"])
    j = a.join(b[["measured"]], rsuffix="_03", how="inner")
    j["richardson"] = (3 * j.measured - j.measured_03) / 2
    j["rel_dev_richardson"] = (j.richardson / j.predicted - 1).abs()
    j.reset_index().to_csv(RES / "shift_richardson.csv", index=False)
    if len(j):
        print(f"Richardson (eta -> 0) vs c_e: {len(j)} events, median rel. dev "
              f"{j.rel_dev_richardson.median():.2e}, max {j.rel_dev_richardson.max():.2e}")
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax = axes[0]
    jitter = np.random.default_rng(0)
    for i, (tree, g) in enumerate(runs.groupby("tree")):
        for e, mk in ((1, "o"), (2, "s"), (3, "^")):
            ge = g[g.event == e]
            if ge.empty:
                continue
            x = ge.ratio * np.exp(jitter.uniform(-0.08, 0.08, len(ge)))
            ax.scatter(x, ge.rel_dev.clip(lower=1e-6), s=9, marker=mk, color=ps.CAT[i], alpha=0.6,
                       label=f"{tree}, event {e}")
    ax.axhline(0.003, color=ps.MUTED, lw=0.8, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="η / η_max",
           ylabel="|measured shift / c_e − 1| (every run)",
           title="(a) accuracy of the O(η) shift, per run")
    ax.legend(fontsize=5.8, ncol=2)
    ax = axes[1]
    for i, (tree, g) in enumerate(runs[runs.ratio <= 0.1].groupby("tree")):
        ax.scatter(g.predicted, g.measured, s=10, color=ps.CAT[i], alpha=0.7, label=tree)
    lim = runs[["measured", "predicted"]].abs().max().max() * 1.2
    ax.plot([-lim, lim], [-lim, lim], color=ps.MUTED, lw=0.8, zorder=0)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set(xlabel="predicted c_e", ylabel="measured (t^GD − t^GF)/η",
           title="(b) all events, η ≤ 0.1 η_max")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(RES / "fig_shift_accuracy.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
