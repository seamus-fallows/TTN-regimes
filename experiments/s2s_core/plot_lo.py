"""Figure (e): first-escape prefactor t_hat_1 = lim sigma^(K-2) t_1 vs bond dimension chi (leading order,
`lo_sweep.py`), median with bootstrap 95% CI over inits, against the Kumar-Haupt time t_KH and the
RMS-overlap heuristic P3; and the K dependence (lo_K).

    uv run python experiments/s2s_core/plot_lo.py [--results results] [--out results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttn import plotting as tp  # noqa: E402

HERE = Path(__file__).parent
TREES = ["binary4", "mps4", "binary8", "ht4", "mps8"]


def median_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x))), axis=1)
    return np.median(x), np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def chi_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tree, K, d, chi, w), g in df.groupby(["tree", "K", "d", "chi", "weights"]):
        m, lo, hi = median_ci(g["T"])
        mk = np.median(g.t_KH)
        rows.append({"tree": tree, "K": K, "d": d, "chi": chi, "weights": w, "n": len(g), "T_median": m,
                     "T_ci_lo": lo, "T_ci_hi": hi, "sd_lnT": np.log(g["T"]).std(ddof=1),
                     "tKH_median": mk, "T_over_tKH_median": np.median(g["T"] / g.t_KH),
                     "P3": g.P3.iloc[0], "frac_dev_max": g.frac_dev_max.max()})
    return pd.DataFrame(rows)


def slopes(tab: pd.DataFrame, chi_min: int) -> pd.DataFrame:
    """Log-log slope of the median t_hat_1 and of the median t_KH against chi (chi >= chi_min)."""
    rows = []
    for (tree, w), g in tab.groupby(["tree", "weights"]):
        g = g[g.chi >= chi_min]
        if len(g) < 3:
            continue
        x = np.log(g.chi.values)
        rows.append({"tree": tree, "weights": w, "chi_range": f"{g.chi.min()}-{g.chi.max()}",
                     "slope_T": np.polyfit(x, np.log(g.T_median.values), 1)[0],
                     "slope_tKH": np.polyfit(x, np.log(g.tKH_median.values), 1)[0]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    tp.set_style()
    df = pd.read_csv(res / "lo_chi.csv")
    tab = chi_table(df)
    tab.to_csv(out / "lo_chi_table.csv", index=False)
    sl = slopes(tab, chi_min=2)
    sl.to_csv(out / "lo_chi_slopes.csv", index=False)
    print(tab.round(4).to_string())
    print(sl.round(3).to_string())

    trees = [t for t in TREES if t in set(tab.tree)]
    fig, axes = plt.subplots(1, len(trees), figsize=(tp.TEXT_WIDTH, 2.3), layout="constrained", squeeze=False)
    colors = tp.palette(2)
    for ax, tree in zip(axes[0], trees):
        for j, w in enumerate(["1", "3-2-1"]):
            g = tab[(tab.tree == tree) & (tab.weights == w)].sort_values("chi")
            if not len(g):
                continue
            lab = "single comp., $s=1$" if w == "1" else "$s=(3,2,1)$"
            ax.errorbar(g.chi, g.T_median, yerr=[g.T_median - g.T_ci_lo, g.T_ci_hi - g.T_median],
                        color=colors[j], marker="o", ms=3, lw=1.2, capsize=1.5, label=f"$\\hat t_1$, {lab}")
            ax.plot(g.chi, g.tKH_median, color=colors[j], lw=0.8, ls="--",
                    label="Kumar-Haupt $t_{KH}$" if j == 0 else None)
            ax.axhline(g.P3.iloc[0], color=colors[j], lw=0.7, ls=":", label="RMS-overlap P3" if j == 0 else None)
        K = int(tab[tab.tree == tree].K.iloc[0])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\chi$")
        ax.set_title(f"{tree} (K={K})", fontsize=8)
    axes[0][0].set_ylabel(r"$\hat t_1=\lim\,\sigma^{K-2}t_1$")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=4, fontsize=6.5)
    tp.savefig(fig, out / "fig_e_chi")

    if (res / "lo_K.csv").exists():
        dk = pd.read_csv(res / "lo_K.csv")
        rows = []
        for (tree, N, K), g in dk.groupby(["tree", "N", "K"]):
            m, lo, hi = median_ci(g["T"])
            rows.append({"tree": tree, "family": tree.rstrip("0123456789"), "N": N, "K": K, "n": len(g),
                         "T_median": m, "T_ci_lo": lo, "T_ci_hi": hi, "P3": g.P3.iloc[0],
                         "tKH_median": np.median(g.t_KH), "T_over_P3": m / g.P3.iloc[0],
                         "T_over_tKH_median": np.median(g["T"] / g.t_KH),
                         "frac_dev_max": g.frac_dev_max.max()})
        kt = pd.DataFrame(rows).sort_values(["family", "K"])
        kt.to_csv(out / "lo_K_table.csv", index=False)
        print(kt.round(4).to_string())
        fig, ax = plt.subplots(figsize=(tp.TEXT_WIDTH / 2, 2.6), layout="constrained")
        for j, (fam, g) in enumerate(kt.groupby("family")):
            c = tp.palette(3)[j]
            ax.errorbar(g.K, g.T_median, yerr=[g.T_median - g.T_ci_lo, g.T_ci_hi - g.T_median], color=c,
                        marker="o", ms=3, lw=1.2, capsize=1.5, label=f"{fam}: median $\\hat t_1$")
            ax.plot(g.K, g.P3, color=c, lw=0.8, ls=":")
            ax.plot(g.K, g.tKH_median, color=c, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("number of cores $K$")
        ax.set_ylabel(r"$\hat t_1$ ($\chi=3$, $d=4$, $s=(3,2,1)$)")
        h, lab = ax.get_legend_handles_labels()
        h += [plt.Line2D([], [], color="k", lw=0.8, ls=":"), plt.Line2D([], [], color="k", lw=0.8, ls="--")]
        lab += ["RMS-overlap P3", "Kumar-Haupt $t_{KH}$ (median)"]
        ax.legend(h, lab, fontsize=6)
        tp.savefig(fig, out / "fig_e2_K")


if __name__ == "__main__":
    main()
