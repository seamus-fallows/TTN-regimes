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

from common import RAW, RES  # noqa: E402
FLOOR = 1e-14  # multi2 only: overlap-product flag (superseded; see SLICE_FLAG)
# multi3: completed runs whose component slice ratio fell below this before its event are flagged
# as possibly roundoff-seeded (the stall stop itself fires at 1e-14)
SLICE_FLAG = 1e-12


def load(name):
    rows = []
    for line in open(RAW / name / "runs.jsonl"):
        r = json.loads(line)
        times = np.array([np.inf if x is None else x for x in r["times"]])
        weights = r["rho"] ** np.arange(r["R"])
        rows.append(dict(tree=r["tree"].replace(":", ""), R=r["R"], rho=r["rho"], seed=r["seed"],
                         times=times, tau=np.array(r["tau"]), weights=weights,
                         stalled=bool(np.any(~np.isfinite(times))),
                         floor=(bool(np.nanmin(r["min_slice"]) < SLICE_FLAG) if "min_slice" in r
                                else bool(np.nanmin(r["min_dormant"]) < FLOOR)),
                         min_slice=np.nanmin(r["min_slice"]) if "min_slice" in r else np.nan,
                         n_learned=int(np.isfinite(times).sum()),
                         tie=bool(unresolved(times, exact="n_steps" in r)),
                         status=r.get("status"),
                         drift=r.get("drift", np.nan)))
    return pd.DataFrame(rows)


# records are log-spaced with 1500 points over [1e-2, 1e9]: one interval is a factor 1.017 in t;
# events closer than two intervals cannot be ordered reliably
TIE = 2 * np.log(10 ** (11 / 1500))


def unresolved(times, exact=False):
    """Events too close to order: within two record intervals for log-grid runs (multi); with exact
    event location (multi2, `common.gf_events`) only coincident times (relative 1e-9)."""
    t = np.sort(times[np.isfinite(times)])
    return len(t) > 1 and np.any(np.diff(np.log(t)) < (1e-9 if exact else TIE))


def n_ok_check(ok):
    return len(ok) > 0


def perm(x):
    return tuple(int(i) for i in np.argsort(x, kind="stable"))


def race_perm_distribution(taus, weights, n_samples=200000, seed=0):
    """Order distribution if every component's blow-up time were an independent draw from the
    pooled sample of tau (iid race, Thurstone-type model), ranked by tau_r / s_r."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(taus, size=(n_samples, len(weights)))
    order = np.argsort(draws / weights, axis=1)
    return Counter(map(tuple, order.tolist())), n_samples


SIGMA_OF = {"binary4": 0.05, "mps4": 0.1, "mps6": 0.2, "binary8": 0.25}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "multi3"
    df = load(name)
    summary, perms = [], []
    for (tree, n_comp, rho), g in df.groupby(["tree", "R", "rho"]):
        ok = g[~g.stalled & ~g.floor & ~g.tie]
        excl = g[g.stalled | g.floor | g.tie]
        greedy = tuple(range(n_comp))
        weights = rho ** np.arange(n_comp)
        gf_perm = [perm(t) for t in ok.times]
        race_perm = [perm(tau / w) for tau, w in zip(ok.tau, ok.weights)]
        race_perm_excl = [perm(tau / w) for tau, w in zip(excl.tau, excl.weights)]
        ktau = [kendalltau(np.arange(n_comp), t).statistic for t in ok.times]
        ktau_race = [kendalltau(t, tau / w).statistic for t, tau, w in
                     zip(ok.times, ok.tau, ok.weights)]
        # iid race model from the blow-up times of (a) the same runs as GF, (b) all runs
        model, n_mod = race_perm_distribution(
            np.concatenate(ok.tau.values) if n_ok_check(ok) else np.concatenate(g.tau.values), weights)
        model_all, n_all = race_perm_distribution(np.concatenate(g.tau.values), weights)
        model_ktau = sum(c * kendalltau(np.arange(n_comp), np.argsort(p)).statistic
                         for p, c in model.items()) / n_mod
        first = Counter(p[0] for p in gf_perm)
        p_gf = np.mean([p == greedy for p in gf_perm]) if gf_perm else np.nan
        p_mod = model[greedy] / n_mod
        n_ok = len(ok)
        rec = dict(tree=tree, R=n_comp, rho=rho, sigma=SIGMA_OF.get(tree, np.nan), n=len(g),
                   n_used=n_ok, n_converged=int((g.status == "converged").sum()),
                   n_stalled=int((g.status == "stalled").sum()),
                   n_tmax=int((g.status == "t_max").sum()),
                   n_floor_flag=int((g.floor & ~g.stalled).sum()),
                   unresolved_ties=int(g.tie.sum()),
                   learned_at_stop=" ".join(f"{k}:{int((g.n_learned == k).sum())}"
                                            for k in range(n_comp + 1) if (g.n_learned == k).any()),
                   p_greedy_order=p_gf,
                   p_greedy_order_se=np.sqrt(p_gf * (1 - p_gf) / max(n_ok, 1)),
                   p_greedy_order_model=p_mod,
                   p_greedy_order_model_all=model_all[greedy] / n_all,
                   z_vs_model=(p_gf - p_mod) / np.sqrt(max(p_mod * (1 - p_mod) / max(n_ok, 1), 1e-12)),
                   race_greedy_used=np.mean([p == greedy for p in race_perm]) if race_perm else np.nan,
                   race_greedy_excluded=(np.mean([p == greedy for p in race_perm_excl])
                                         if race_perm_excl else np.nan),
                   race_first_excluded=" ".join(
                       f"{r + 1}:{sum(p[0] == r for p in race_perm_excl)}" for r in range(n_comp)),
                   race_first_used=" ".join(
                       f"{r + 1}:{sum(p[0] == r for p in race_perm)}" for r in range(n_comp)),
                   # which components were not learned by the stop (1 = largest)
                   unlearned=" ".join(
                       f"{r + 1}:{int(sum(not np.isfinite(t[r]) for t in g.times))}"
                       for r in range(n_comp)),
                   kendall_greedy=np.mean(ktau) if ktau else np.nan,
                   kendall_greedy_se=np.std(ktau) / np.sqrt(max(len(ktau), 1)),
                   kendall_greedy_model=model_ktau,
                   race_exact_order=np.mean([a == b for a, b in zip(gf_perm, race_perm)]),
                   race_first_agree=np.mean([a[0] == b[0] for a, b in zip(gf_perm, race_perm)]),
                   race_kendall=np.mean(ktau_race) if ktau_race else np.nan)
        for r in range(n_comp):
            rec[f"p_first_{r + 1}"] = first[r] / max(n_ok, 1)
        summary.append(rec)
        counts = Counter(gf_perm)
        for p in itertools.permutations(range(n_comp)):
            perms.append(dict(tree=tree, R=n_comp, rho=rho, order="".join(str(i + 1) for i in p),
                              gf=counts[p] / max(n_ok, 1), race_model=model[p] / n_mod,
                              race_model_all=model_all[p] / n_all))
    summary, perms = pd.DataFrame(summary), pd.DataFrame(perms)
    summary.to_csv(RES / f"{name}_summary.csv", index=False, float_format="%.4g")
    perms.to_csv(RES / f"{name}_perms.csv", index=False, float_format="%.4g")
    print(summary.to_string(index=False))

    plt = style.plt
    style.setup()
    # only groups with enough completed runs (R = 4 at chi = 4 mostly stalls, see the summary)
    groups = [(k, g[g.n_used >= 30]) for k, g in summary.groupby(["R", "rho"])]
    groups = [(k, g) for k, g in groups if len(g)]
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.5, 2.4 * len(groups)), squeeze=False,
                             gridspec_kw=dict(hspace=0.6))
    for ax, ((n_comp, rho), sg) in zip(axes[:, 0], groups):
        trees = list(sg.tree)
        sub = perms[(perms.R == n_comp) & (perms.rho == rho)]
        top = sub.groupby("order").gf.max().sort_values(ascending=False).index[:8]
        x = np.arange(len(top))
        width = 0.8 / len(trees)
        for i, tree in enumerate(trees):
            s = sub[sub.tree == tree].set_index("order").reindex(top)
            col = style.SERIES[["binary4", "mps4", "mps6", "binary8"].index(tree)]  # colour = tree
            ax.bar(x + i * width, s.gf, width * 0.9, color=col, label=f"{tree} GF")
            ax.plot(x + i * width, s.race_model, "_", color=style.INK, ms=8, mew=1.5)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(top)
        ax.set_ylabel("frequency")
        ax.set_title(f"R = {n_comp}, weights rho^(r-1), rho = {rho}  (bars: GF; black ticks: iid race "
                     f"model; groups with >= 30 completed runs)", fontsize=7)
        ax.legend(fontsize=6, ncol=len(trees))
    axes[-1, 0].set_xlabel("learning order (component indices, 1 = largest)")
    fig.savefig(RES / f"{name}_orders.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
