"""P(greedy) versus init scale sigma: m = 2 (matrix-like pairs) vs m >= 3 (tensor-like pairs).

Usage: uv run python experiments/order/analyze_sigma.py b4_sigma m4_sigma b8_sigma pat_mps6_m2end ...
For each sweep: table of P(wrong order) vs (rho, sigma) with Wilson 95% CIs; for every (setting,
rho) a power-law fit P(wrong) = C sigma^gamma (binomial maximum likelihood, points with sigma in
the configured range) and the per-seed stability of the winner across sigma. Writes
results/sigma_fits.csv, results/sigma_table.csv and results/sigma_scan.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from analyze_main import wilson  # noqa: E402

from common import RAW, RES  # noqa: E402
M_OF = {"b4_pilot": 2, "b4_sigma": 2, "m4_sigma": 4, "b8_sigma": 4, "pat_mps6_m2end": 2, "pat_mps6_m2adj": 2,
        "pat_mps6_m3": 3, "pat_mps6_m4": 4, "pat_mps4_adj": 2, "pat_mps4_sep": 2}
# number of unstable directions at the smaller component (notes/theory_escape.md 3.3): chi_e for
# two adjacent differing cores sharing bond e, else 1
N_OF = {"pat_mps6_m2adj": 4, "pat_mps4_adj": 4}


def load(name):
    rows = []
    for line in open(RAW / name / "runs.jsonl"):
        r = json.loads(line)
        for run in r["runs"]:
            rows.append(dict(sweep=name, tree=f"{r['kind']}{r['n']}", d=r["d"], chi=r["chi"],
                             seed=r["seed"], rho=run["rho"], sigma=run["sigma"],
                             wrong=int(run["winner"] != 0)))
    return pd.DataFrame(rows)


def fit_power(sig, k, n):
    """Binomial ML fit of p = min(1, C sigma^gamma); returns gamma, its SE (observed information)."""
    ls = np.log(sig)

    def nll(par):
        lc, g = par
        p = np.clip(np.exp(lc + g * ls), 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))

    x0 = np.polyfit(ls, np.log(np.maximum(k, 0.5) / n), 1)[::-1]
    res = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12))
    h = np.zeros((2, 2))
    eps = 1e-4
    for i in range(2):
        for j in range(2):
            e_i, e_j = np.eye(2)[i] * eps, np.eye(2)[j] * eps
            h[i, j] = (nll(res.x + e_i + e_j) - nll(res.x + e_i - e_j) - nll(res.x - e_i + e_j)
                       + nll(res.x - e_i - e_j)) / (4 * eps**2)
    cov = np.linalg.inv(h)
    return res.x[1], np.sqrt(cov[1, 1]), np.exp(res.x[0])


def main():
    names = sys.argv[1:] or list(M_OF)
    df = pd.concat([load(n) for n in names if (RAW / n / "runs.jsonl").exists()])
    keys = ["sweep", "tree", "d", "chi", "rho"]
    tab = df.groupby(keys + ["sigma"]).wrong.agg(["sum", "count"]).reset_index()
    tab["p_wrong"] = tab["sum"] / tab["count"]
    tab["ci_lo"], tab["ci_hi"] = wilson(tab["sum"].values, tab["count"].values)
    tab.to_csv(RES / "sigma_table.csv", index=False, float_format="%.5g")
    fits = []
    for key, g in tab.groupby(keys):
        rho = key[-1]
        m = M_OF.get(key[0])
        stab = df[(df[keys] == pd.Series(dict(zip(keys, key)))).all(axis=1)]
        same = stab.groupby("seed").wrong.nunique().eq(1).mean()
        n_unst = N_OF.get(key[0], 1)
        rec = dict(zip(keys, key), m=m, n=n_unst if m == 2 else 0,
                   gamma_pred=n_unst * (1 - rho) / rho if m == 2 else 0.0,
                   seed_stable_frac=same)
        if g["sum"].sum() >= 5:
            gam, se, c = fit_power(g.sigma.values, g["sum"].values, g["count"].values)
            # the same seeds are reused at every sigma, so the binomial SE (independent points) is
            # too small: bootstrap over seeds instead
            wide = stab.pivot_table(index="seed", columns="sigma", values="wrong")
            sig = wide.columns.values
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(200):
                w = wide.values[rng.integers(0, len(wide), len(wide))]
                k = w.sum(0)
                if k.sum() >= 5:
                    boot.append(fit_power(sig, k, np.full(len(sig), len(w)))[0])
            rec.update(gamma=gam, gamma_se_binomial=se, gamma_se=np.std(boot, ddof=1), C=c)
            # asymptotic regime only (sigma <= 0.01), where P(wrong) is further from saturation
            small = wide.columns.values <= 0.01
            if wide.values[:, small].sum() >= 5 and small.sum() >= 3:
                boot_s = []
                for _ in range(200):
                    w = wide.values[rng.integers(0, len(wide), len(wide))][:, small]
                    if w.sum() >= 5:
                        boot_s.append(fit_power(sig[small], w.sum(0), np.full(small.sum(), len(w)))[0])
                g_s = fit_power(sig[small], wide.values[:, small].sum(0),
                                np.full(small.sum(), len(wide)))[0]
                rec.update(gamma_small_sigma=g_s, gamma_small_sigma_se=np.std(boot_s, ddof=1))
        fits.append(rec)
    fits = pd.DataFrame(fits)
    fits.to_csv(RES / "sigma_fits.csv", index=False, float_format="%.4g")
    print(fits.to_string(index=False))

    plt = style.plt
    style.setup()
    groups = [s for s in names if s in set(tab.sweep)]
    ncol = min(3, len(groups))
    nrow = -(-len(groups) // ncol)
    fig, ax_grid = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow), squeeze=False,
                                gridspec_kw=dict(hspace=0.55, wspace=0.3))
    for extra in ax_grid.flat[len(groups):]:
        extra.set_visible(False)
    axes = [list(ax_grid.flat[: len(groups)])]
    rho_list = sorted(tab.rho.unique())
    for ax, name in zip(axes[0], groups):
        sub = tab[tab.sweep == name]
        chis = sorted(sub.chi.unique())
        for (chi, rho), g in sub.groupby(["chi", "rho"]):
            col = style.SERIES[rho_list.index(rho)]  # colour = rho, marker fill = chi
            filled = chi == chis[0]
            ok = g["sum"] > 0
            ax.errorbar(g.sigma[ok], g.p_wrong[ok],
                        yerr=[(g.p_wrong - g.ci_lo)[ok], (g.ci_hi - g.p_wrong)[ok]],
                        fmt="o" if filled else "s", color=col, mfc=col if filled else "white",
                        ms=3.5, lw=0, elinewidth=0.8,
                        label=f"rho={rho}" if filled else None)
            f = fits[(fits.sweep == name) & (fits.chi == chi) & (fits.rho == rho)]
            if M_OF.get(name) == 2 and len(f) and np.isfinite(f.gamma.iloc[0]):
                ss = np.geomspace(g.sigma.min(), g.sigma.max(), 50)
                ax.plot(ss, f.C.iloc[0] * ss ** f.gamma.iloc[0], color=col, lw=0.9,
                        ls="-" if filled else "--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 1)
        ax.set_xlabel("sigma (per-core init scale)")
        chi_txt = ", ".join(f"chi={c}" + (" (filled)" if c == chis[0] else " (open)") for c in chis)
        ax.set_title(f"{name} (m={M_OF.get(name)}); {chi_txt}", fontsize=7)
    handles = [plt.Line2D([], [], color=style.SERIES[i], marker="o", lw=0, ms=4, label=f"rho={r}")
               for i, r in enumerate(rho_list)]
    axes[0][-1].legend(handles=handles, fontsize=6, loc="lower right")
    for a in ax_grid[:, 0]:
        a.set_ylabel("P(smaller first)")
    fig.savefig(RES / "sigma_scan.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
