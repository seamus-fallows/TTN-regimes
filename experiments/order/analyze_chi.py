"""Over-parameterization: does a larger bond dimension restore the greedy order?

Usage: uv run python experiments/order/analyze_chi.py
Reads /root/outputs/order/chi_scan (LO blow-up times, uncapped chi = 1..32) and chi_gf (GF check).
Per (tree, chi): sd of log tau, race-model P(greedy) at rho = 0.5..0.99 (U-statistic over pooled
blow-up times), GF P(greedy) where available; power-law fit sd(log tau) ~ chi^-alpha over chi >= 4.
Writes results/chi_scan.csv and results/chi_scan.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from analyze_main import K_OF, load, race_curve, wilson  # noqa: E402

RES = Path(__file__).parent / "results"
RHOS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def main():
    seeds, _ = load("chi_scan")
    rows = []
    for (tree, chi), g in seeds.groupby(["tree", "chi"]):
        lt = np.log(np.concatenate([g.tau1, g.tau2]))
        rec = dict(tree=tree, K=K_OF[tree], chi=chi, n=len(g), sd_log_tau=lt.std(ddof=1),
                   sd_se=lt.std(ddof=1) / np.sqrt(2 * (len(lt) - 1)))
        for rho, p in zip(RHOS, race_curve(g.tau1.values, g.tau2.values, RHOS)):
            rec[f"race_{rho}"] = p
        rows.append(rec)
    df = pd.DataFrame(rows)
    gf_path = Path("/root/outputs/order/chi_gf/runs.jsonl")
    gf = None
    if gf_path.exists():
        gseeds, gruns = load("chi_gf")
        gf = gruns.groupby(["tree", "chi", "rho"]).winner.agg(
            k=lambda w: int((w == 0).sum()), n="count").reset_index()
        gf["p_greedy"] = gf.k / gf.n
        gf["ci_lo"], gf["ci_hi"] = wilson(gf.k.values, gf.n.values)
        merged = gruns.merge(gseeds[["tree", "chi", "seed", "tau1", "tau2"]], on=["tree", "chi", "seed"])
        agree = merged.assign(ok=(merged.tau1 < merged.tau2 / merged.rho) == (merged.winner == 0))
        gf = gf.merge(agree.groupby(["tree", "chi", "rho"]).ok.mean().rename("race_agree").reset_index())
        gf.to_csv(RES / "chi_gf.csv", index=False, float_format="%.4g")
        print(gf.to_string(index=False))
    fits = []
    for tree, g in df.groupby("tree"):
        big = g[g.chi >= 4]
        if len(big) >= 3:
            slope, icpt = np.polyfit(np.log(big.chi), np.log(big.sd_log_tau), 1)
            fits.append(dict(tree=tree, alpha=-slope))
    df = df.merge(pd.DataFrame(fits), on="tree", how="left")
    df.to_csv(RES / "chi_scan.csv", index=False, float_format="%.4g")
    print(df.to_string(index=False))

    plt = style.plt
    style.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    for i, (tree, g) in enumerate(df.groupby("tree", sort=False)):
        col = style.SERIES[i]
        axes[0].errorbar(g.chi, g.sd_log_tau, yerr=g.sd_se, fmt="o-", color=col, ms=3.5, lw=1,
                         label=f"{tree} (K={K_OF[tree]})")
        axes[1].plot(g.chi, g["race_0.8"], "o-", color=col, ms=3.5, lw=1, label=f"{tree}, race")
        if gf is not None:
            h = gf[(gf.tree == tree) & (gf.rho == 0.8)]
            if len(h):
                axes[1].errorbar(h.chi, h.p_greedy, yerr=[h.p_greedy - h.ci_lo, h.ci_hi - h.p_greedy],
                                 fmt="s", color=col, mfc="white", ms=4, lw=0, elinewidth=0.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bond dimension chi (uncapped)")
    axes[0].set_ylabel("sd of log blow-up time")
    axes[0].legend(fontsize=6)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("bond dimension chi (uncapped)")
    axes[1].set_ylabel("P(greedy) at rho = 0.8")
    axes[1].set_title("lines: LO race model; open squares: full GF", fontsize=7)
    axes[1].axhline(0.5, color=style.MUTED, lw=0.6)
    fig.savefig(RES / "chi_scan.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
