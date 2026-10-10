"""Summaries and figures for the order-statistics sweep (configs/main.yaml).

Usage: uv run python experiments/order/analyze_main.py [name]   (default: main)
Reads /root/outputs/order/<name>/runs.jsonl; writes results/<name>_summary.csv,
results/<name>_settings.csv and figures results/<name>_*.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

HERE = Path(__file__).parent
from common import RAW, RES  # noqa: E402
TREE_ORDER = ["binary4", "mps4", "mps6", "binary8", "mps8"]
K_OF = {"binary4": 3, "mps4": 4, "mps6": 6, "binary8": 7, "mps8": 8}


def load(name: str):
    seeds, runs = [], []
    for line in open(RAW / name / "runs.jsonl"):
        r = json.loads(line)
        tree = f"{r['kind']}{r['n']}"
        base = dict(tree=tree, K=K_OF.get(tree), d=r["d"], chi=r["chi"],
                    bonds="-".join(map(str, r["bonds"][:-1])), seed=r["seed"])
        srec = dict(base)
        if "tau" in r:
            srec.update(tau1=r["tau"][0], tau2=r["tau"][1])
        for i, p in enumerate(r.get("proxies", [])):
            for k, v in p.items():
                srec[f"{k}{i + 1}"] = v
        seeds.append(srec)
        for run in r["runs"]:
            runs.append(dict(base, rho=run["rho"], sigma=run["sigma"], winner=run["winner"],
                             t1=run["t1"], a1=run["a"][0], a2=run["a"][1], nfev=run["nfev"],
                             status=run["status"], lo_winner=run.get("lo_winner")))
    return pd.DataFrame(seeds), pd.DataFrame(runs)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def race_curve(tau1, tau2, rhos):
    """P(tau_b / tau_a > rho) for iid tau, as a U-statistic over all ordered pairs of the pooled
    sample (the independent-race prediction; exact at leading order for odeco on an MPS)."""
    t = np.log(np.concatenate([tau1, tau2]))
    diff = t[None, :] - t[:, None]  # log(tau_j / tau_i)
    off = ~np.eye(len(t), dtype=bool)
    return np.array([np.mean(diff[off] > np.log(r)) for r in rhos])


def summarize(seeds: pd.DataFrame, runs: pd.DataFrame):
    keys = ["tree", "K", "d", "chi", "bonds"]
    s = seeds.copy()
    s["log_tau1"], s["log_tau2"] = np.log(s.tau1), np.log(s.tau2)
    per_setting = []
    for key, g in s.groupby(keys):
        lt = np.concatenate([g.log_tau1, g.log_tau2])
        per_setting.append(dict(zip(keys, key), n_seeds=len(g), sd_log_tau=lt.std(ddof=1),
                                mean_log_tau=lt.mean()))
    per_setting = pd.DataFrame(per_setting)

    rows = []
    merged = runs.merge(s[keys + ["seed", "tau1", "tau2"]], on=keys + ["seed"])
    for key, g in merged.groupby(keys + ["rho", "sigma"]):
        n = len(g)
        k = int((g.winner == 0).sum())
        lo, hi = wilson(k, n)
        pred = (g.tau1 < g.tau2 / g.rho).astype(int)  # 1 = greedy predicted
        curve = race_curve(g.tau1.values, g.tau2.values, [key[-2]])[0]
        rows.append(dict(zip(keys + ["rho", "sigma"], key), n=n, p_greedy=k / n,
                         se=np.sqrt(k / n * (1 - k / n) / n), ci_lo=lo, ci_hi=hi,
                         race_pred_seedwise=pred.mean(),
                         race_agree=np.mean(pred.values == (g.winner == 0).values),
                         race_curve=curve, failed=int((g.status != 1).sum())))
    summ = pd.DataFrame(rows).merge(per_setting[keys + ["sd_log_tau"]], on=keys)
    summ["lognormal_pred"] = norm.cdf(np.log(1 / summ.rho) / (np.sqrt(2) * summ.sd_log_tau))
    return summ, per_setting


def plot_pg_vs_rho(summ, seeds, name):
    plt = style.plt
    style.setup()
    trees = [t for t in TREE_ORDER if t in set(summ.tree)]
    fig, axes = plt.subplots(1, len(trees), figsize=(2.6 * len(trees), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    rr = np.linspace(0.45, 1.0, 120)
    for ax, tree in zip(axes, trees):
        sub = summ[summ.tree == tree]
        for ci, chi in enumerate(sorted(sub.chi.unique())):
            for d, mk, fill in [(2, "o", True), (4, "s", False)]:
                g = sub[(sub.chi == chi) & (sub.d == d)].sort_values("rho")
                if g.empty:
                    continue
                col = style.SERIES[ci]
                ax.errorbar(g.rho, g.p_greedy, yerr=[g.p_greedy - g.ci_lo, g.ci_hi - g.p_greedy],
                            fmt=mk, color=col, mfc=col if fill else "white", mec=col, ms=4,
                            lw=0, elinewidth=0.8, label=f"chi={chi}, d={d} ({g.bonds.iloc[0]})")
                sd = seeds[(seeds.tree == tree) & (seeds.chi == chi) & (seeds.d == d)]
                # m = 2 (binary4): the independent race is not the theory (unique local max);
                # drawn faintly for reference only
                ax.plot(rr, race_curve(sd.tau1.values, sd.tau2.values, rr), color=col,
                        lw=0.9, ls="-" if fill else "--", alpha=0.3 if tree == "binary4" else 1)
        ax.axhline(0.5, color=style.MUTED, lw=0.6)
        m = 2 if tree == "binary4" else (4 if tree == "binary8" else K_OF[tree])
        ax.set_title(f"{tree} (K={K_OF[tree]}, m={m})", fontsize=8)
        ax.set_xlabel("rho = s2/s1")
        ax.set_xlim(0.45, 1.01)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=5, loc="upper right", handletextpad=0.2, borderaxespad=0.2)
    axes[0].set_ylabel("P(larger component first)")
    fig.suptitle("Full GF, sigma = 0.01 (markers, 95% Wilson CI) vs independent-race model "
                 "(lines; filled/solid d = 2, open/dashed d = 4)", fontsize=8, y=1.03)
    fig.savefig(RES / f"{name}_pg_vs_rho.png")
    plt.close(fig)


def plot_spread(per_setting, summ, name):
    plt = style.plt
    style.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    ax = axes[0]
    ps = per_setting.copy()
    ps["mean_chi"] = [np.mean([int(b) for b in s.split("-")]) for s in ps.bonds]
    for i, tree in enumerate(TREE_ORDER):
        g = ps[ps.tree == tree].sort_values("mean_chi")
        if g.empty:
            continue
        ax.plot(g.mean_chi, g.sd_log_tau, "o", color=style.SERIES[i], label=f"{tree} (K={K_OF[tree]})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("mean bond dimension")
    ax.set_ylabel("sd of log tau (blow-up time)")
    ax.legend(fontsize=6)
    ax = axes[1]
    for i, tree in enumerate(TREE_ORDER):
        g = summ[(summ.tree == tree) & (summ.rho == 0.8)]
        if g.empty:
            continue
        ax.errorbar(g.K + 0.08 * (np.arange(len(g)) - len(g) / 2), g.p_greedy, yerr=1.96 * g.se,
                    fmt="o", color=style.SERIES[i], ms=3.5, lw=0, elinewidth=0.8, label=tree)
    ax.set_xlabel("K (number of cores)")
    ax.set_ylabel("P(greedy) at rho = 0.8")
    ax.legend(fontsize=6)
    fig.savefig(RES / f"{name}_spread.png")
    plt.close(fig)


def d_pairs(summ):
    """Settings that differ only in d (same tree, same capped bond dimensions): is P(greedy) equal?
    Prediction: on an MPS d only rescales every core slice by the same factor (cancels); on binary
    trees it also changes the leaf-to-internal scale ratio of the fan-in init."""
    rows = []
    for (tree, bonds, rho), g in summ.groupby(["tree", "bonds", "rho"]):
        if set(g.d) >= {2, 4}:
            a, b = g[g.d == 2].iloc[0], g[g.d == 4].iloc[0]
            diff = b.p_greedy - a.p_greedy
            rows.append(dict(tree=tree, bonds=bonds, rho=rho, p_d2=a.p_greedy, p_d4=b.p_greedy,
                             diff=diff, z=diff / np.sqrt(a.se**2 + b.se**2 + 1e-12),
                             race_d2=a.race_curve, race_d4=b.race_curve))
    return pd.DataFrame(rows)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "main"
    seeds, runs = load(name)
    summ, per_setting = summarize(seeds, runs)
    RES.mkdir(exist_ok=True)
    summ.to_csv(RES / f"{name}_summary.csv", index=False, float_format="%.5g")
    per_setting.to_csv(RES / f"{name}_settings.csv", index=False, float_format="%.5g")
    dp = d_pairs(summ)
    dp.to_csv(RES / f"{name}_d_pairs.csv", index=False, float_format="%.4g")
    if len(dp):
        print(dp.groupby("tree").z.agg(["count", "mean", lambda z: np.sum(z**2)]).rename(
            columns={"<lambda_0>": "chi2"}).to_string())
    plot_pg_vs_rho(summ, seeds, name)
    plot_spread(per_setting, summ, name)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summ[["tree", "d", "chi", "bonds", "rho", "n", "p_greedy", "se", "race_curve",
                    "race_agree", "lognormal_pred", "failed"]].to_string(index=False))


if __name__ == "__main__":
    main()
