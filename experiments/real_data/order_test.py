"""Analysis of the pre-registered order test (notes/real_data.md section 9.5).

Usage: uv run python experiments/real_data/order_test.py <predictor.json> <extra_orders_summary.json> <out_prefix>
           [--main <main_summary.json>] [--routes <routes_raw.json>] [--plateau <routes_extra.json>]

Per order: S (data-only predictor, order_predictor.py); O1 = mean over seeds 0-2 of the train loss at tau = 1e4;
O2 = mean over seeds of log10 tau(L <= 0.25), seeds that never reach 0.25 set to tau = 1e4 (censored). Spearman
rho(S, O1), rho(S, O2) across all orders and within the new random orders, with 95% bootstrap CIs over orders
(10,000 resamples) and two-sided permutation p-values (10,000 permutations). Secondary: (a) without the orders
seen before registration, (b) with the round-0 random orders scrambled0-2 added (main sweep, seeds 0-2), (c) S vs
the plateau-state share of the round-0 orders. --plateau: an exploratory predictor, the plateau-state share of the
test runs themselves (route_amplitudes.py on the extra-order runs), averaged over seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SIGMA, SEEDS, HORIZON = 0.1, (0, 1, 2), 1e4
SEEN = {"quadtree", "quadtreeT", "rowmajor"}  # outcomes seen before registration
N_BOOT, N_PERM = 10_000, 10_000


def outcomes(rows):
    """{order: {O1, O2, n_seeds, censored, per_seed}} from summarize.py rows at sigma = 0.1, seeds 0-2."""
    by = {}
    for r in rows:
        if (
            r.get("tree") != "binary"
            or abs(r["sigma"] - SIGMA) > 1e-9
            or r["seed"] not in SEEDS
            or "loss_end" not in r
        ):
            continue
        tau = r.get("t_loss_0.25", np.nan)
        by.setdefault(r["ordering"], []).append(
            {
                "seed": r["seed"],
                "loss_end": r["loss_end"],
                "tau_025": tau,
                "censored": not np.isfinite(tau),
                "t2": r["t2"],
                "stop": r["stop"],
            }
        )
    out = {}
    for o, v in by.items():
        v.sort(key=lambda x: x["seed"])
        logt = [np.log10(x["tau_025"]) if not x["censored"] else np.log10(HORIZON) for x in v]
        out[o] = {
            "O1": float(np.mean([x["loss_end"] for x in v])),
            "O2": float(np.mean(logt)),
            "n_seeds": len(v),
            "censored": sum(x["censored"] for x in v),
            "per_seed": v,
        }
    return out


def spearman_ci(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y).statistic)
    boots = []
    n = len(x)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        boots.append(spearmanr(x[i], y[i]).statistic)
    perm = np.array([spearmanr(rng.permutation(x), y).statistic for _ in range(N_PERM)])
    return {
        "n": n,
        "rho": rho,
        "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
        "p_perm": float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (N_PERM + 1)),
        "n_boot_used": len(boots),
        "supported": bool(rho < 0 and np.nanpercentile(boots, 97.5) < 0),
    }


def analyse(table, orders, rng, key="S"):
    sel = [
        o
        for o in orders
        if o in table and table[o].get(key) is not None and table[o]["n_seeds"] == len(SEEDS)
    ]
    return {
        out: spearman_ci([table[o][key] for o in sel], [table[o][out] for o in sel], rng)
        | {"orders": sel}
        for out in ("O1", "O2")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictor")
    ap.add_argument("summary")
    ap.add_argument("out_prefix")
    ap.add_argument("--main", default=None)
    ap.add_argument("--routes", default=None)
    ap.add_argument("--plateau", default=None)
    a = ap.parse_args()
    pred = {p["order"]: p for p in json.load(open(a.predictor))["orders"]}
    rows = [r for r in json.load(open(a.summary)) if "error" not in r]
    if a.main:  # secondary (b): round-0 random orders from the main sweep
        rows += [
            r
            for r in json.load(open(a.main))
            if "error" not in r
            and r.get("features") == "raw"
            and r.get("ordering") in ("scrambled0", "scrambled1", "scrambled2")
        ]
    out = outcomes(rows)
    table = {
        o: {**out[o], "S": pred[o]["S"], "best_rank1_loss": pred[o]["best_loss"]}
        for o in out
        if o in pred
    }
    if a.plateau:  # exploratory: plateau-state share of the test runs, averaged over seeds
        sh = {}
        for r in json.load(open(a.plateau)):
            if "pairs" in r and abs(r["sigma"] - SIGMA) < 1e-9 and r["seed"] in SEEDS:
                lam = np.array([p["lam"] for p in r["pairs"]])
                k = np.array([p["k"] for p in r["pairs"]])
                sh.setdefault(r["ordering"], []).append(
                    float((lam[k <= 5] ** 2).sum() / (lam**2).sum())
                )
        for o in table:
            v = sh.get(o, [])
            table[o]["S_plateau"] = float(np.mean(v)) if len(v) == len(SEEDS) else None
    new_random = sorted(
        o for o in table if o.startswith("scrambled") and int(o[len("scrambled") :]) >= 100
    )
    structured = sorted(o for o in table if not o.startswith("scrambled"))
    old_random = sorted(
        o for o in table if o.startswith("scrambled") and int(o[len("scrambled") :]) < 100
    )
    rng = np.random.default_rng(20261010)
    res = {
        "table": table,
        "orders": {"new_random": new_random, "structured": structured, "old_random": old_random},
        "primary": {
            "all": analyse(table, new_random + structured, rng),
            "random": analyse(table, new_random, rng),
        },
        "secondary_a_unseen": analyse(
            table, [o for o in new_random + structured if o not in SEEN], rng
        ),
        "secondary_b_with_round0": {
            "all": analyse(table, new_random + structured + old_random, rng),
            "random": analyse(table, new_random + old_random, rng),
        },
    }
    if a.plateau:
        res["exploratory_plateau_share"] = {
            "all": analyse(table, new_random + structured, rng, "S_plateau"),
            "random": analyse(table, new_random, rng, "S_plateau"),
        }
    if a.routes:  # secondary (c): predictor vs the plateau-state share of the round-0 orders
        val = {}
        for r in json.load(open(a.routes)):
            if "pairs" in r and abs(r["sigma"] - SIGMA) < 1e-9 and r["features"] == "raw":
                lam = np.array([p["lam"] for p in r["pairs"]])
                k = np.array([p["k"] for p in r["pairs"]])
                val.setdefault(r["ordering"], []).append(
                    {
                        "seed": r["seed"],
                        "plateau_loss": r["train_loss"],
                        "share": float((lam[k <= 5] ** 2).sum() / (lam**2).sum()),
                    }
                )
        res["secondary_c_validity"] = {
            o: {"S": pred[o]["S"], "best_rank1_loss": pred[o]["best_loss"], "plateau": v}
            for o, v in sorted(val.items())
            if o in pred
        }
    out_p = Path(a.out_prefix)
    json.dump(res, open(out_p.with_suffix(".json"), "w"), indent=1, default=float)
    open(out_p.with_suffix(".md"), "w", encoding="utf-8").write(markdown(res) + "\n")
    print(markdown(res))


def _fmt(r):
    lo, hi = r["ci95"]
    return (
        f"rho = {r['rho']:+.2f} [{lo:+.2f}, {hi:+.2f}], p_perm = {r['p_perm']:.3g}, n = {r['n']}"
        f" -> {'supported' if r['supported'] else 'not supported'}"
    )


def markdown(res):
    md = [
        "# Pre-registered order test (notes/real_data.md section 9): results",
        "",
        "| order | group | S | best rank-1 loss | O1 = mean L(1e4) | O2 = mean log10 tau(L<=0.25) | censored | per-seed L(1e4) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    t = res["table"]
    groups = {o: g for g, os in res["orders"].items() for o in os}
    for o in sorted(t, key=lambda o: t[o]["S"]):
        v = t[o]
        md.append(
            f"| {o} | {groups.get(o, '-')} | {v['S']:.3f} | {v['best_rank1_loss']:.4f} | {v['O1']:.4f} | {v['O2']:.3f} "
            f"| {v['censored']}/{v['n_seeds']} | {', '.join(f'{x['loss_end']:.4f}' for x in v['per_seed'])} |"
        )
    md += ["", "## Primary analysis (Spearman over orders, 95% bootstrap CI, permutation p)", ""]
    for scope in ("all", "random"):
        for o in ("O1", "O2"):
            md.append(f"- {scope} orders, S vs {o}: {_fmt(res['primary'][scope][o])}")
    md += ["", "## Secondary analyses (registered)", ""]
    for o in ("O1", "O2"):
        md.append(
            f"- (a) without the orders seen before registration, S vs {o}: {_fmt(res['secondary_a_unseen'][o])}"
        )
    for scope in ("all", "random"):
        for o in ("O1", "O2"):
            md.append(
                f"- (b) with scrambled0-2 added, {scope} orders, S vs {o}: {_fmt(res['secondary_b_with_round0'][scope][o])}"
            )
    if "secondary_c_validity" in res:
        md += [
            "",
            "(c) predictor validity: S and best rank-1 loss vs the plateau state reached by the runs (sigma = 0.1):",
            "",
        ]
        for o, v in res["secondary_c_validity"].items():
            pl = ", ".join(
                f"seed {x['seed']}: L {x['plateau_loss']:.3f} share {x['share']:.2f}"
                for x in v["plateau"]
            )
            md.append(
                f"- {o}: S = {v['S']:.3f} at best rank-1 loss {v['best_rank1_loss']:.4f}; plateau states: {pl}"
            )
    if "exploratory_plateau_share" in res:
        md += [
            "",
            "## Exploratory (not registered): plateau-state share of the test runs, averaged over seeds",
            "",
        ]
        for scope in ("all", "random"):
            for o in ("O1", "O2"):
                md.append(
                    f"- {scope} orders, S_plateau vs {o}: {_fmt(res['exploratory_plateau_share'][scope][o])}"
                )
    return "\n".join(md)


if __name__ == "__main__":
    main()
