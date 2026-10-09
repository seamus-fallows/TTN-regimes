"""Step 4 analysis: full learning order with R >= 3 components.

Usage: uv run python experiments/order/analyze_multi.py [name]   (default: multi)
Per (tree, R, rho): fraction of greedy orders, mean Kendall tau with the greedy order, P(first = r),
per-seed agreement with the leading-order race order (argsort tau_r / s_r), and the race-model
(Thurstone) distribution of orders obtained by resampling the pooled blow-up times as iid.
Runs with a component that never crossed 1/2 ("stalled") or whose dormant overlap fell below 1e-14
before its event ("floor") are counted and excluded from the order statistics.
Writes results/<name>_summary.csv, results/<name>_perms.csv and results/<name>_orders.png.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

RES = Path(__file__).parent / "results"
FLOOR = 1e-14


def load(name):
    rows = []
    for line in open(f"/root/outputs/order/{name}/runs.jsonl"):
        r = json.loads(line)
        times = np.array([np.inf if x is None else x for x in r["times"]])
        weights = r["rho"] ** np.arange(r["R"])
        rows.append(dict(tree=r["tree"].replace(":", ""), R=r["R"], rho=r["rho"], seed=r["seed"],
                         times=times, tau=np.array(r["tau"]), weights=weights,
                         stalled=bool(np.any(~np.isfinite(times))),
                         floor=bool(np.nanmin(r["min_dormant"]) < FLOOR),
                         drift=r["drift"]))
    return pd.DataFrame(rows)


def perm(x):
    return tuple(int(i) for i in np.argsort(x, kind="stable"))


def race_perm_distribution(taus, weights, n_samples=200000, seed=0):
    """Order distribution if every component's blow-up time were an independent draw from the
    pooled sample of tau (iid race, Thurstone-type model), ranked by tau_r / s_r."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(taus, size=(n_samples, len(weights)))
    order = np.argsort(draws / weights, axis=1)
    return Counter(map(tuple, order.tolist())), n_samples


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "multi"
    df = load(name)
    summary, perms = [], []
    for (tree, n_comp, rho), g in df.groupby(["tree", "R", "rho"]):
        ok = g[~g.stalled & ~g.floor]
        greedy = tuple(range(n_comp))
        gf_perm = [perm(t) for t in ok.times]
        race_perm = [perm(tau / w) for tau, w in zip(ok.tau, ok.weights)]
        ktau = [kendalltau(np.arange(n_comp), t).statistic for t in ok.times]
        ktau_race = [kendalltau(t, tau / w).statistic for t, tau, w in
                     zip(ok.times, ok.tau, ok.weights)]
        pooled = np.concatenate(g.tau.values)
        model, n_mod = race_perm_distribution(pooled, rho ** np.arange(n_comp))
        model_ktau = sum(c * kendalltau(np.arange(n_comp), np.argsort(p)).statistic
                         for p, c in model.items()) / n_mod
        first = Counter(p[0] for p in gf_perm)
        rec = dict(tree=tree, R=n_comp, rho=rho, n=len(g), n_used=len(ok),
                   stalled=int(g.stalled.sum()), floor=int(g.floor.sum()),
                   p_greedy_order=np.mean([p == greedy for p in gf_perm]),
                   p_greedy_order_model=model[greedy] / n_mod,
                   kendall_greedy=np.mean(ktau), kendall_greedy_se=np.std(ktau) / np.sqrt(len(ktau)),
                   kendall_greedy_model=model_ktau,
                   race_exact_order=np.mean([a == b for a, b in zip(gf_perm, race_perm)]),
                   race_kendall=np.mean(ktau_race), max_drift=g.drift.max())
        for r in range(n_comp):
            rec[f"p_first_{r + 1}"] = first[r] / max(len(ok), 1)
        summary.append(rec)
        counts = Counter(gf_perm)
        for p in itertools.permutations(range(n_comp)):
            perms.append(dict(tree=tree, R=n_comp, rho=rho, order="".join(str(i + 1) for i in p),
                              gf=counts[p] / max(len(ok), 1), race_model=model[p] / n_mod))
    summary, perms = pd.DataFrame(summary), pd.DataFrame(perms)
    summary.to_csv(RES / f"{name}_summary.csv", index=False, float_format="%.4g")
    perms.to_csv(RES / f"{name}_perms.csv", index=False, float_format="%.4g")
    print(summary.to_string(index=False))

    plt = style.plt
    style.setup()
    groups = list(summary.groupby(["R", "rho"]))
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.5, 2.2 * len(groups)), squeeze=False)
    for ax, ((n_comp, rho), sg) in zip(axes[:, 0], groups):
        trees = list(sg.tree)
        sub = perms[(perms.R == n_comp) & (perms.rho == rho)]
        top = sub.groupby("order").gf.max().sort_values(ascending=False).index[:8]
        x = np.arange(len(top))
        width = 0.8 / len(trees)
        for i, tree in enumerate(trees):
            s = sub[sub.tree == tree].set_index("order").reindex(top)
            ax.bar(x + i * width, s.gf, width * 0.9, color=style.SERIES[i], label=f"{tree} GF")
            ax.plot(x + i * width, s.race_model, "_", color=style.INK, ms=8, mew=1.5)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(top)
        ax.set_ylabel("frequency")
        ax.set_title(f"R = {n_comp}, weights rho^(r-1), rho = {rho}  (bars: GF; black ticks: iid race model)",
                     fontsize=8)
        ax.legend(fontsize=6, ncol=len(trees))
    axes[-1, 0].set_xlabel("learning order (component indices, 1 = largest)")
    fig.savefig(RES / f"{name}_orders.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
