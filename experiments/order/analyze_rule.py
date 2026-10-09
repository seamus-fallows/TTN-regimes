"""Which statistic of the init decides the order? Selection rules "s_r X_r^q" vs gradient flow.

Usage: uv run python experiments/order/analyze_rule.py [name]   (default: main)
Uses the per-seed init statistics and blow-up times of the main sweep (sweep.py with proxies and
lo_race). For every setting and candidate statistic X (slice norm, |T_r(theta_0)| = |amp0|,
|grad T_r|^2, KKT projection, product of per-core overlaps):
  - fit log tau_r = c - q log X_r (OLS over components and seeds) and the rank correlation;
  - predicted winner of each GF run: argmax_r s_r X_r^q (q fitted), accuracy vs the GF winner;
  - the heuristic `ttn.theory.winner_probability` (decoupled balanced modes with effective
    amplitude c_r = |P_r(0)|^(1/K), and with the product-Gaussian sampler), as P(greedy).
Writes results/<name>_rules.csv and results/<name>_rules.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from analyze_main import load  # noqa: E402
from ttn.theory import escape_time_asymptotic, product_gaussian_sampler, winner_probability  # noqa: E402

from common import RAW, RES  # noqa: E402
PROXIES = {"slice_norm": "slice norm", "amp0": "|T_r(theta0)|", "grad2": "|grad T_r|^2",
           "kkt_proj": "KKT projection", "core_prod": "prod. per-core overlaps"}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "main"
    seeds, runs = load(name)
    keys = ["tree", "K", "d", "chi", "bonds"]
    rows = []
    for key, s in seeds.groupby(keys):
        K = key[1]
        r = runs.merge(s, on=keys + ["seed"])
        logtau = np.log(np.concatenate([s.tau1, s.tau2]))
        for px in PROXIES:
            x = np.abs(np.concatenate([s[f"{px}1"], s[f"{px}2"]]))
            ok = x > 0
            q, c = np.polyfit(-np.log(x[ok]), logtau[ok], 1)
            rho_s = spearmanr(x[ok], -logtau[ok]).statistic
            x1, x2 = np.abs(r[f"{px}1"].values), np.abs(r[f"{px}2"].values)
            with np.errstate(divide="ignore"):
                pred_greedy = np.log(x1) * q > np.log(r.rho.values) + np.log(x2) * q
            acc = np.mean(pred_greedy == (r.winner.values == 0))
            rows.append(dict(zip(keys, key), proxy=px, q_fit=q, q_over_Km2=q / max(K - 2, 1),
                             spearman=rho_s, order_accuracy=acc))
        # exact race (tau) accuracy for reference
        acc = np.mean((r.tau1 < r.tau2 / r.rho).values == (r.winner.values == 0))
        rows.append(dict(zip(keys, key), proxy="tau (LO race)", q_fit=np.nan, q_over_Km2=np.nan,
                         spearman=1.0, order_accuracy=acc))
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"{name}_rules.csv", index=False, float_format="%.4g")
    print(df.pivot_table(index=["tree", "d", "chi"], columns="proxy",
                         values="order_accuracy").round(3).to_string())
    print(df.pivot_table(index=["tree", "d", "chi"], columns="proxy", values="q_fit").round(2)
          .to_string())

    # library heuristic (decoupled balanced modes) vs GF, P(greedy) per setting and rho
    heur = []
    for key, s in seeds.groupby(keys):
        K = key[1]
        amp = np.abs(np.stack([s.amp01, s.amp02], 1)) ** (1.0 / K)
        g = runs[(runs[keys] == pd.Series(dict(zip(keys, key)))).all(axis=1)]
        for rho, gr in g.groupby("rho"):
            w = np.array([1.0, rho])
            t = escape_time_asymptotic(K, w, amp, correction=False)
            p_amp = np.mean(np.argmin(t, axis=1) == 0)
            p_pg = winner_probability(w, K, product_gaussian_sampler(K, 1e-3), 200000)[0]
            heur.append(dict(zip(keys, key), rho=rho, p_gf=np.mean(gr.winner == 0),
                             p_heur_measured_amp=p_amp, p_heur_product_gaussian=p_pg))
    heur = pd.DataFrame(heur)
    heur.to_csv(RES / f"{name}_heuristic.csv", index=False, float_format="%.4g")

    plt = style.plt
    style.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    ax = axes[0]
    piv = df.pivot_table(index="proxy", values="order_accuracy", aggfunc="mean").sort_values(
        "order_accuracy")
    ax.barh(range(len(piv)), piv.order_accuracy, color=style.SERIES[0])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([PROXIES.get(p, p) for p in piv.index], fontsize=7)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("per-run accuracy of the predicted winner (mean over settings)")
    ax = axes[1]
    for i, col in enumerate(["p_heur_measured_amp", "p_heur_product_gaussian"]):
        ax.plot(heur.p_gf, heur[col], "o", ms=3, color=style.SERIES[i + 1], alpha=0.7,
                label={"p_heur_measured_amp": "heuristic, c_r = |T_r(0)|^(1/K)",
                       "p_heur_product_gaussian": "heuristic, product-Gaussian c_r"}[col])
    ax.plot([0.45, 1], [0.45, 1], color=style.MUTED, lw=0.8)
    ax.set_xlabel("P(greedy), gradient flow")
    ax.set_ylabel("P(greedy), decoupled-mode heuristic")
    ax.legend(fontsize=6)
    fig.savefig(RES / f"{name}_rules.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
