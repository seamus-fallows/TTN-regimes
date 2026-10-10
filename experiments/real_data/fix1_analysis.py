"""Fix round 1 of the real-data review (notes/reviews/real_data_round1.md), from committed outputs.

Usage: uv run python experiments/real_data/fix1_analysis.py <results_dir> <out_prefix>
           [--local <summary.json>...] [--audit <audit_table.json>...] [--logs <agent-runs/real-data/logs>] [--local-runs <dir>]

The raw main-sweep trajectories are not available locally, so every number here is recomputed from the
committed per-run / per-series JSON files in <results_dir> (events_table, routes_raw, hessians_raw,
regime_analysis, scaling, chi8_scaling), plus optional extra summaries:
  --local   summarize.py outputs of the fix-round reruns (configs/fix1_local.yaml, fix1_mps.yaml)
  --audit   the reviewers' audit tables (data/review_real_data/audit/*audit_table.json; one row per run
            with final loss, times to loss levels and route-amplitude shares, from the experiment's code)
  --logs    run logs of the remote sweeps (stop reasons of the chi = 8 runs)
Sections (one key each in <out_prefix>.json; markdown tables in <out_prefix>.md):
  first_route   clean k = 3 series (events_table: the same sibling pair rises inside drop 1 at every sigma
                of the series); power law, log law and A / sigma + B, all with relative residuals
  routes        route-amplitude rule after the rank-1 filter, de-duplicated into critical-point units
  shares        share of sum lambda^2 on k <= 5 routes, one value per (order, seed)
  locality      per (order, seed) at sigma = 0.1: share, final loss, times to loss levels; Spearman
  hessians      per-seed lambda_min exponents at the common first plateau
  mps_first     MPS first rank increase relative to the first drop (resolved vs at tau = 0)
  regime        lazy floors per seed, convergence of the lazy runs, threshold sensitivity
  chi8          two-point slopes of the rank-clock gap, wall-limited runs
  clock         first-escape times t_a per seed (absolute clock), sigma = 0.1
  repro         fix1 reruns vs committed values (quadtree t_a, final loss; MPS first-rise timing)
  mps_reruns    35 MPS reruns (tau <= 1000): first rise at tau = 0 or resolved, per-edge power vs log fits
  creep         loss decrease between drops relative to the next drop (binary reruns; needs --local-runs)
  two_point_*   chi = 8 and Fashion: two-point slopes of the rank-clock gap, first rises at tau = 0
  with --main (full main-sweep summary): first_route_full (window-checked fits over all binary runs, from
                scaling.json), locality_all_sigma (per-seed final losses), mps_full (per-edge laws, full horizon)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from analysis import ERR_MAX, route_units  # noqa: E402


def rel_fit(cols, y):
    """Coefficients c minimising sum((cols . c) / y - 1)^2 (relative residuals) and the rms."""
    a = np.vstack(cols).T / y[:, None]
    c, *_ = np.linalg.lstsq(a, np.ones_like(y), rcond=None)
    return c, float(np.sqrt(np.mean((a @ c - 1) ** 2)))


def three_laws(sig, gap):
    """Power law (log space), log law and A / sigma + B (relative residuals) for one series."""
    sig, gap = np.asarray(sig, float), np.asarray(gap, float)
    x = np.log(1 / sig)
    p, a = np.polyfit(x, np.log(gap), 1)
    pow_rms = float(np.sqrt(np.mean((np.exp(a + p * x) / gap - 1) ** 2)))
    (la, lb), log_rms = rel_fit([np.ones_like(x), x], gap)
    (kb, ka), k3_rms = rel_fit([np.ones_like(x), 1 / sig], gap)
    b_unw, a_unw = np.polyfit(x, gap, 1)  # the round-0 (unweighted) log-law fit, for comparison
    log_rms_unw = float(np.sqrt(np.mean(((a_unw + b_unw * x) / gap - 1) ** 2)))
    return {
        "power_p": float(p),
        "power_rms": pow_rms,
        "log_slope": float(lb),
        "log_rms": log_rms,
        "log_rms_unweighted": log_rms_unw,
        "A": float(ka),
        "B": float(kb),
        "k3_rms": k3_rms,
    }


def mean_se(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return [np.nan, np.nan, 0]
    return [
        float(v.mean()),
        float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
        len(v),
    ]


def first_route_clean(events):
    """Clean k = 3 first-route series: raw features, binary trees, the signature of drop 1 is one sibling
    pair gaining its 2nd singular value at every sigma of the series (>= 4 sigma over >= 0.5 decade, as
    in events_table.py). The rank crossing lies inside drop 1's window by construction, so the gap
    drop 1 - drop 0 belongs to that route (the round-0 `first_route` could take a later drop)."""
    out = []
    for s in events:
        sig = s["signature"]
        if (
            s["features"] != "raw"
            or not s["tree"].startswith("binary")
            or s["k"] != 3
            or len(sig) != 2
            or any(k != 2 for _, k in sig)
            or set(s["drop_index"]) != {1}
        ):
            continue
        fit = three_laws(s["sigmas"], s["dt"])
        # same fits on the sigma grid of main.yaml's reduced orders only (0.05, 0.1, 0.2, 0.5)
        sub = [
            (a, g)
            for a, g in zip(s["sigmas"], s["dt"])
            if any(abs(a - b) < 1e-9 for b in (0.05, 0.1, 0.2, 0.5))
        ]
        fit4 = three_laws(*zip(*sub)) if len(sub) >= 3 else None
        out.append(
            {
                "tree": s["tree"],
                "seed": s["seed"],
                "edges": [e for e, _ in sig],
                "sigmas": s["sigmas"],
                "gaps": s["dt"],
                **fit,
                "power_p_main_grid": fit4["power_p"] if fit4 else None,
                "B_over_A": fit["B"] / fit["A"],
            }
        )
    summary = {}
    for tree in sorted({o["tree"] for o in out}) + ["all"]:
        sel = [o for o in out if tree in ("all", o["tree"])]
        summary[tree] = {
            "n_series": len(sel),
            "power_p": mean_se([o["power_p"] for o in sel]),
            "power_p_range": [min(o["power_p"] for o in sel), max(o["power_p"] for o in sel)],
            "power_rms_max": max(o["power_rms"] for o in sel),
            "log_rms_range": [min(o["log_rms"] for o in sel), max(o["log_rms"] for o in sel)],
            "log_rms_unweighted_range": [
                min(o["log_rms_unweighted"] for o in sel),
                max(o["log_rms_unweighted"] for o in sel),
            ],
            "k3_rms_max": max(o["k3_rms"] for o in sel),
            "B_negative": sum(o["B"] < 0 for o in sel),
        }
    return {"series": out, "summary": summary}


def refit_all_series(events):
    """Every events_table series refitted with relative residuals for both laws (round 0 fitted the
    log law unweighted): power exponent and log slope with their rms."""
    out = []
    for s in events:
        f = three_laws(s["sigmas"], s["dt"])
        out.append(
            {
                k: s[k]
                for k in ("tree", "features", "seed", "signature", "k", "connected", "drop_index")
            }
            | {
                k: f[k]
                for k in ("power_p", "power_rms", "log_slope", "log_rms", "log_rms_unweighted")
            }
            | {
                "sigma_range": [min(s["sigmas"]), max(s["sigmas"])],
                "n_sigma": len(s["sigmas"]),
                "preferred": "power" if f["power_rms"] < f["log_rms"] else "log",
            }
        )
    return out


def shares_per_seed(states, sigma=0.05):
    """Share of sum lambda^2 on k <= 5 routes (and on k = 3) per (order, seed) at one sigma: one value per
    independent init, instead of pooling the same critical point over sigma."""
    rows = []
    for r in states:
        if "pairs" not in r or abs(r["sigma"] - sigma) > 1e-9 or r["features"] != "raw":
            continue
        lam = np.array([p["lam"] for p in r["pairs"]])
        k = np.array([p["k"] for p in r["pairs"]])
        rows.append(
            {
                "ordering": r["ordering"],
                "seed": r["seed"],
                "rank1_rel_err": r["rank1_rel_err"],
                "share_k3": float((lam[k == 3] ** 2).sum() / (lam**2).sum()),
                "share_k5": float((lam[k <= 5] ** 2).sum() / (lam**2).sum()),
                "best_k3": float(max(lam[k == 3])),
                "plateau_loss": r["train_loss"],
            }
        )
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["ordering"], r["seed"])):
        by[r["ordering"]].append(r["share_k5"])
        if r["ordering"] != "quadtree":
            by["scrambled (3 orders)"].append(r["share_k5"])
    n3, n5 = 4, 12  # sibling pairs (k = 3) and pairs with k <= 5 among the 28 pairs of bottom cores
    summary = {k: {"mean_se": mean_se(v), "values": v} for k, v in sorted(by.items())}
    return {
        "sigma": sigma,
        "rows": rows,
        "summary": summary,
        "pair_count_chance_k5": n5 / 28,
        "pair_count_chance_k3": n3 / 28,
    }


LEVELS = (0.25, 0.2, 0.15)


def locality_units(states, local_rows, audit_rows, sigma=0.1):
    """One row per (order, seed) at sigma: k <= 5 share at the first plateau (routes_raw for the four
    committed orders, the audit tables for the reviewers' extra orders), final train loss and test
    accuracy at tau_end, times to loss levels (tau clock), escape time t_a. Sources: 'fix1' = this round's
    reruns of main.yaml cases (configs/fix1_local.yaml), 'audit' = review runs with the same code and
    settings (seed 0: 6 random orders, row-major, transposed quadtree; seed 1: the last two)."""
    share = {}
    for r in states:
        if "pairs" in r and abs(r["sigma"] - sigma) < 1e-9 and r["features"] == "raw":
            lam = np.array([p["lam"] for p in r["pairs"]])
            k = np.array([p["k"] for p in r["pairs"]])
            share[(r["ordering"], r["seed"])] = (
                float((lam[k <= 5] ** 2).sum() / (lam**2).sum()),
                r["rank1_rel_err"],
            )
    rows = {}
    for r in local_rows:
        if r.get("tree") != "binary" or abs(r["sigma"] - sigma) > 1e-9 or "loss_end" not in r:
            continue
        key = (r["ordering"], r["seed"])
        sh = share.get(key, (np.nan, np.nan))
        rows[key] = {
            "ordering": r["ordering"],
            "seed": r["seed"],
            "source": "fix1",
            "share_k5": sh[0],
            "rank1_rel_err": sh[1],
            "loss_end": r["loss_end"],
            "test_acc": r["test_acc_end"],
            "tau_end": r["t2"],
            "stop": r["stop"],
            "t_a": r["t_a"],
            **{f"tau_L{lv}": r.get(f"t_loss_{lv}", np.nan) for lv in LEVELS},
        }
    for r in audit_rows:
        if r.get("tree") != "binary":
            continue
        seed = int(re.search(r"seed(\d+)", r["name"]).group(1))
        key = (r["ordering"], seed)
        if key in rows:  # same case as a fix1 rerun: keep the rerun, record the agreement
            rows[key]["audit_loss_end"] = r["L_end"]
            continue
        rows[key] = {
            "ordering": r["ordering"],
            "seed": seed,
            "source": "audit",
            "share_k5": r["share_k5"] + r["share_k3"],
            "rank1_rel_err": r["rank1_err"],
            "loss_end": r["L_end"],
            "test_acc": r["test_acc"],
            "tau_end": 1e4,
            "stop": "completed",
            "t_a": r["t_a"],
            **{f"tau_L{lv}": r.get(f"tau_L{lv}", np.nan) for lv in LEVELS},
        }
    rows = sorted(rows.values(), key=lambda r: (r["seed"], r["loss_end"]))
    ok = [r for r in rows if np.isfinite(r["share_k5"])]

    def rho(sel):
        if len(sel) < 4:
            return None
        res = spearmanr([r["share_k5"] for r in sel], [r["loss_end"] for r in sel])
        return {"n": len(sel), "rho": float(res.statistic), "p": float(res.pvalue)}

    committed = {"quadtree", "scrambled0", "scrambled1", "scrambled2"}
    out = {
        "sigma": sigma,
        "rows": rows,
        "spearman_all": rho(ok),
        "spearman_committed_orders": rho([r for r in ok if r["ordering"] in committed]),
        "spearman_seed0": rho([r for r in ok if r["seed"] == 0]),
    }
    by = defaultdict(list)
    for r in rows:
        by[r["ordering"]].append(r)
    out["per_order"] = {
        o: {
            "n": len(v),
            "loss_end": mean_se([r["loss_end"] for r in v]),
            "loss_values": [r["loss_end"] for r in v],
            "share_k5": mean_se([r["share_k5"] for r in v]),
            **{
                f"tau_L{lv}_median": float(np.nanmedian([r[f"tau_L{lv}"] for r in v]))
                if any(np.isfinite(r[f"tau_L{lv}"]) for r in v)
                else None
                for lv in LEVELS
            },
        }
        for o, v in sorted(by.items())
    }
    # rank of the quadtree among the orders run at the same seed (final loss and time to L <= 0.25)
    ranks = {}
    for seed in sorted({r["seed"] for r in rows}):
        sel = [r for r in rows if r["seed"] == seed]
        q = [r for r in sel if r["ordering"] == "quadtree"]
        if not q or len(sel) < 2:
            continue
        q = q[0]
        ranks[seed] = {
            "n_orders": len(sel),
            "lower_final_loss_than_quadtree": sum(r["loss_end"] < q["loss_end"] for r in sel),
            "reach_L0.25_before_quadtree": sum(
                np.nan_to_num(r["tau_L0.25"], nan=np.inf) < q["tau_L0.25"] for r in sel
            ),
        }
    out["quadtree_rank_per_seed"] = ranks
    return out


def hessian_exponents(hess, levels=(("binary", 0.265, 0.275), ("mps", 0.225, 0.235))):
    """Per-seed exponent of |lambda_min| (gauge-projected) vs sigma at the common first plateau (same
    saddle: plateau loss inside the band), every seed with >= 3 sigma values; sigma range per seed."""
    out = {}
    for tree, lo, hi in levels:
        sel = [
            r for r in hess if "proj_min" in r and r["tree"] == tree and lo <= r["loss_mid"] <= hi
        ]
        per = {}
        for seed in sorted({r["seed"] for r in sel}):
            rs = sorted((r for r in sel if r["seed"] == seed), key=lambda r: r["sigma"])
            if len(rs) < 3:
                continue
            s = np.array([r["sigma"] for r in rs])
            lam = -np.array([r["proj_min"] for r in rs])
            p = float(np.polyfit(np.log(s), np.log(lam), 1)[0])
            per[seed] = {
                "exponent": p,
                "sigmas": s.tolist(),
                "decades": float(np.log10(s[-1] / s[0])),
                "n_neg": [r["proj_n_neg"] for r in rs],
            }
        out[tree] = {
            "band": [lo, hi],
            "n_runs": len(sel),
            "per_seed": per,
            "mean_se": mean_se([v["exponent"] for v in per.values()]),
        }
    return out


def mps_first_rise(scaling, local_rows=()):
    """Timing of the MPS first rank increase relative to the first loss drop's peak, from the committed
    rank-clock gaps (scaling.json grank_mixed: t_first_rank2 - t_drop0 per sigma 0.05 ... 0.5). A gap
    near -t_drop0 (< -5 here; drop 0 peaks at tau ~ 10-12) means the event was recorded at tau = 0: s2/s1
    was >= 0.05 from the start of phase 2 (not a resolved rank increase). Local reruns, if given, report
    the same directly (`first_rank2_at_start`)."""
    sigmas = [0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5]
    rows = []
    for m in scaling["mps_snake/raw"]["grank_mixed"]:
        seed = m["seed"][1]
        for s, g in zip(sigmas, m["gaps"]):
            rows.append({"seed": seed, "sigma": s, "gap": g, "at_start": g < -5})
    resolved = [r for r in rows if not r["at_start"]]
    local = [
        {
            "seed": r["seed"],
            "sigma": r["sigma"],
            "first": r.get("first_rank2_edges"),
            "at_start": r.get("first_rank2_at_start"),
            "gap": r["t_first_rank2"] - r["drops"][0]["t"],
            "drop0_peak": r["drops"][0]["t"],
        }
        for r in local_rows
        if r.get("tree") == "mps" and r.get("drops") and r["name"].endswith("_short")
    ]
    return {
        "rows": rows,
        "n": len(rows),
        "at_start": sum(r["at_start"] for r in rows),
        "resolved_gap_range": [min(r["gap"] for r in resolved), max(r["gap"] for r in resolved)],
        "resolved_before_peak": sum(r["gap"] < 0 for r in resolved),
        "n_resolved": len(resolved),
        "local_reruns": local,
    }


def regime_checks(reg):
    """Lazy floors per seed vs where the other runs end; convergence of the lazy runs at their horizon;
    the sigma = 1.4 'rich' label; label sensitivity to the thresholds (labels stored per run)."""
    floors = {
        r["seed"]: r["L_lin_floor"] for r in reg if r["model"] == "centred" and r["sigma"] == 4.0
    }
    lazy = [r for r in reg if r["label"] == "lazy"]
    plain = [r for r in reg if r["model"] == "plain" and r["sigma"] >= 0.1]
    rich = [r for r in reg if r["model"] == "centred" and abs(r["sigma"] - 1.4) < 1e-9]
    sens = defaultdict(lambda: defaultdict(int))
    for r in reg:
        for key, lab in r["labels"].items():
            sens[key][f"{r['sigma']:g}|{lab}"] += 1
    per_seed = {}
    for seed, fl in sorted(floors.items()):
        mine = [r for r in plain if r["seed"] == seed]
        per_seed[seed] = {
            "lazy_floor": fl,
            "plain_end_sigma_ge_0.1": [
                min(r["loss_end"] for r in mine),
                max(r["loss_end"] for r in mine),
            ],
            "plain_end_below_floor": sum(r["loss_end"] < fl for r in mine),
            "n_plain": len(mine),
            "rich_1.4_end": [r["loss_end"] for r in rich if r["seed"] == seed],
        }
    return {
        "floors": floors,
        "floor_mean": float(np.mean(list(floors.values()))),
        "per_seed": per_seed,
        "lazy_runs": [
            {
                "sigma": r["sigma"],
                "seed": r["seed"],
                "L_T": r["loss_end"],
                "L_lin_T": r["L_lin_T"],
                "floor": r["L_lin_floor"],
                "excess_over_floor": r["loss_end"] - r["L_lin_floor"],
                "test_acc": r["test_acc_end"],
            }
            for r in lazy
        ],
        "rich_sigma_1.4": [
            {
                "seed": r["seed"],
                "n_drops": r["n_drops"],
                "delta_K": r["delta_K_end"],
                "L_T": r["loss_end"],
                "L_lin_T": r["L_lin_T"],
                "floor": r["L_lin_floor"],
                "t2": r["t2"],
            }
            for r in rich
        ],
        "label_sensitivity": {k: dict(v) for k, v in sorted(sens.items())},
    }


def chi8_checks(chi8, logs_dir=None):
    """Two-point slopes (sigma = 0.05 -> 0.14) of the rank-clock gap t_first_rank2 - t_drop0 per seed
    (chi8_scaling.json grank_mixed; the first route's identity per sigma is not stored there, only the set
    of routes per seed), and the runs stopped by the wall limit (run logs)."""
    out = {}
    for tree in ("binary_quadtree/raw", "binary_scrambled/raw"):
        sl = []
        for m in chi8[tree]["grank_mixed"]:
            g05, g14 = m["gaps"][0], m["gaps"][1]
            sl.append(
                {
                    "seed": m["seed"][1],
                    "slope": float(np.log(g05 / g14) / np.log(0.14 / 0.05)),
                    "routes": m["first_edges"],
                    "single_route": len(m["first_edges"]) == 1,
                }
            )
        out[tree] = {
            "two_point": sl,
            "mean_se": mean_se([s["slope"] for s in sl]),
            "mean_se_single_route": mean_se([s["slope"] for s in sl if s["single_route"]]),
        }
    if logs_dir:
        stops = defaultdict(int)
        for fn in ("chi8.log", "chi8b.log"):
            for line in open(Path(logs_dir) / fn):
                if line.startswith("{"):
                    d = json.loads(line)
                    if d.get("status") not in (None, "exists"):
                        tree = (
                            "mps"
                            if "_mps_" in d["name"]
                            else ("quadtree" if "quadtree" in d["name"] else "scrambled")
                        )
                        stops[f"{tree}|{d['status']}"] += 1
        out["stops"] = dict(stops)
    return out


def two_point(scaling, sig=(0.05, 0.14)):
    """Robustness sweeps (3 sigma values): per seed, two-point slope ln(g(sigma_0) / g(sigma_1)) /
    ln(sigma_1 / sigma_0) of the rank-clock gap (first s_2/s_1 >= 0.05 minus the first drop; grank_mixed),
    the set of first-rising edges over the three sigma values, and MPS runs whose first rise is recorded at
    tau = 0 (gap < -5, i.e. minus the first drop's time)."""
    out = {}
    for tree in ("binary_quadtree/raw", "binary_scrambled/raw", "mps_snake/raw"):
        rows = []
        for m in scaling[tree]["grank_mixed"]:
            g = m["gaps"]
            ok = g[0] > 0 and g[1] > 0
            rows.append({"seed": m["seed"][1], "routes": m["first_edges"], "gaps": g,
                         "slope": float(np.log(g[0] / g[1]) / np.log(sig[1] / sig[0])) if ok else None,
                         "at_start": sum(x < -5 for x in g)})
        out[tree] = {"rows": rows, "slope": mean_se([r["slope"] for r in rows if r["slope"] is not None]),
                     "at_start": sum(r["at_start"] for r in rows)}
    return out


def clock(reg, local_rows, audit_rows, sigma=0.1):
    """First-escape time t_a per seed at sigma (absolute clock): binary quadtree (plain main-sweep runs in
    regime_analysis.json), the fix1 reruns (scrambled orders, MPS snake) and the audit runs. Since
    t_a ~ sigma^-(K-2) per seed (K = 15 binary, 16 MPS), the MPS / binary ratio grows like 1/sigma."""
    rows = [
        {
            "tree": "binary",
            "ordering": "quadtree",
            "seed": r["seed"],
            "t_a": r["t_a"],
            "source": "regime_analysis",
        }
        for r in reg
        if r["model"] == "plain" and abs(r["sigma"] - sigma) < 1e-9
    ]
    rows += [
        {
            "tree": r["tree"],
            "ordering": r["ordering"],
            "seed": r["seed"],
            "t_a": r["t_a"],
            "source": "fix1",
        }
        for r in local_rows
        if abs(r["sigma"] - sigma) < 1e-9
        and not (r["tree"] == "binary" and r["ordering"] == "quadtree")
    ]
    uniq = {}
    for r in rows:
        uniq.setdefault((r["tree"], r["ordering"], r["seed"]), r)
    rows = list(uniq.values())
    for r in audit_rows:
        seed = int(re.search(r"seed(\d+)", r["name"]).group(1))
        if not any(
            q["ordering"] == r["ordering"] and q["tree"] == r["tree"] and q["seed"] == seed
            for q in rows
        ):
            rows.append(
                {
                    "tree": r["tree"],
                    "ordering": r["ordering"],
                    "seed": seed,
                    "t_a": r["t_a"],
                    "source": "audit",
                }
            )
    quad = {r["seed"]: r["t_a"] for r in rows if r["ordering"] == "quadtree"}
    mps = {r["seed"]: r["t_a"] for r in rows if r["tree"] == "mps" and r["ordering"] == "snake"}
    ratio = {s: mps[s] / quad[s] for s in mps if s in quad}
    return {
        "sigma": sigma,
        "rows": rows,
        "mps_over_quadtree": ratio,
        "binary_orders_faster_than_mps": {
            s: sum(r["t_a"] < mps[s] for r in rows if r["tree"] == "binary" and r["seed"] == s)
            for s in mps
        },
    }


def repro_check(reg, local_rows, sigma=0.1):
    """fix1 reruns of main.yaml cases vs the committed values of the same cases (quadtree at sigma, from
    regime_analysis.json): t_a and the final train loss."""
    ref = {r["seed"]: r for r in reg if r["model"] == "plain" and abs(r["sigma"] - sigma) < 1e-9}
    out = []
    for r in local_rows:
        if r.get("ordering") == "quadtree" and abs(r["sigma"] - sigma) < 1e-9 and r["seed"] in ref:
            q = ref[r["seed"]]
            out.append(
                {
                    "seed": r["seed"],
                    "t_a_rel_diff": abs(r["t_a"] / q["t_a"] - 1),
                    "loss_end": r["loss_end"],
                    "loss_end_committed": q["loss_end"],
                }
            )
    return out


def creep(local_rows, runs_dir):
    """Loss decrease between consecutive drops (from the end of drop i's widened window to the start of drop
    i+1's), relative to the decrease inside drop i+1's widened window; binary fix1 reruns (tau = 1e4)."""
    from analysis import load

    out = []
    for r in local_rows:
        if r.get("tree") != "binary" or len(r.get("drops", [])) < 2:
            continue
        p2 = load(Path(runs_dir) / f"{r['name']}.npz")["p2"]
        t_, loss = p2["t"], np.minimum.accumulate(p2["loss"])
        at = lambda x: float(np.interp(x, t_, loss))  # noqa: E731
        for d0, d1 in zip(r["drops"][:-1], r["drops"][1:]):
            a, b = d0["t_hi"] + d0["width"], d1["t_lo"] - d1["width"]
            nxt = at(b) - at(d1["t_hi"] + d1["width"])
            if b > a and nxt > 0:
                out.append(
                    {
                        "ordering": r["ordering"],
                        "seed": r["seed"],
                        "creep_over_next_drop": (at(a) - at(b)) / nxt,
                    }
                )
    v = [o["creep_over_next_drop"] for o in out]
    return {
        "intervals": out,
        "n": len(v),
        "range": [min(v), max(v)] if v else None,
        "median": float(np.median(v)) if v else None,
    }


def mps_reruns(local_rows, tag="_t1000", features="raw"):
    """The 35 snake-MPS reruns (configs/fix1_mps.yaml, tau <= 1000): per run the first-escape time t_a, the
    first drop's rate peak, each edge's first s_2/s_1 >= 0.05 time (tau clock; equal to tau = 0 when the
    ratio is already above threshold at the start of phase 2) and the first rising edges; then per edge
    and seed, rise time after the first drop T_e vs sigma fitted by a power law and by a log law, both with
    relative residuals (`scaling.fit_seed`; seeds with T_e > 0 at >= 5 sigma values spanning a decade)."""
    from scaling import fit_seed

    runs = []
    for r in local_rows:
        if (r.get("tree") != "mps" or not r["name"].endswith(tag) or not r.get("drops")
                or r.get("features") != features or r["name"].endswith("_short")):
            continue
        ev = np.array(r["rank_events"], float)[:, 0]
        runs.append(
            {
                "sigma": r["sigma"],
                "seed": r["seed"],
                "t_a": r["t_a"],
                "drop0": r["drops"][0]["t"],
                "drop0_window": [
                    r["drops"][0]["t_lo"] - r["drops"][0]["width"],
                    r["drops"][0]["t_hi"] + r["drops"][0]["width"],
                ],
                "rise": [None if not np.isfinite(x) else float(x) for x in ev],
                "at_start": r.get("rank2_at_start", []),
                "first": r.get("first_rank2_edges"),
                "first_at_start": r.get("first_rank2_at_start"),
                "loss_end": r["loss_end"],
            }
        )
    runs.sort(key=lambda r: (r["seed"], r["sigma"]))
    if not runs:
        return None
    resolved = [r for r in runs if not r["first_at_start"]]
    before_peak = [r for r in resolved if r["rise"][r["first"][0]] < r["drop0"]]
    inside0 = [
        r
        for r in resolved
        if r["drop0_window"][0] <= r["rise"][r["first"][0]] <= r["drop0_window"][1]
    ]
    n_edges = len(runs[0]["rise"])
    edges = []
    for e in range(n_edges):
        fits = []
        for seed in sorted({r["seed"] for r in runs}):
            pts = sorted(
                (r["sigma"], r["rise"][e] - r["drop0"])
                for r in runs
                if r["seed"] == seed and r["rise"][e] is not None and r["rise"][e] - r["drop0"] > 0
            )
            if len(pts) >= 5 and pts[-1][0] / pts[0][0] >= 9.9:
                f = fit_seed(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
                fits.append(
                    {
                        "seed": seed,
                        "n_sigma": len(pts),
                        **{
                            k: float(f[k]) for k in ("power_p", "power_rms", "log_slope", "log_rms")
                        },
                    }
                )
        if fits:
            edges.append(
                {
                    "edge": e,
                    "n_seeds": len(fits),
                    "per_seed": fits,
                    "log_preferred": sum(f["log_rms"] < f["power_rms"] for f in fits),
                    **{
                        k: mean_se([f[k] for f in fits])
                        for k in ("power_p", "power_rms", "log_slope", "log_rms")
                    },
                }
            )
    return {
        "runs": runs,
        "n": len(runs),
        "first_at_start": len(runs) - len(resolved),
        "first_single_edge_resolved": sum(len(r["first"] or []) == 1 for r in resolved),
        "n_resolved": len(resolved),
        "first_edges_resolved": sorted({tuple(r["first"]) for r in resolved}),
        "resolved_before_drop0_peak": len(before_peak),
        "resolved_inside_drop0_window": len(inside0),
        "edge_fits": edges,
    }


def mps_repro(scaling, mr):
    """MPS reruns (tau <= 1000) vs the committed full-horizon runs: first rank-rise time minus the first
    drop's peak, per (seed, sigma) (scaling.json grank_mixed lists them in sigma order 0.05 ... 0.5)."""
    if not mr:
        return None
    sigmas = [0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5]
    ref = {}
    for m in scaling["mps_snake/raw"]["grank_mixed"]:
        for s, gap in zip(sigmas, m["gaps"]):
            ref[(m["seed"][1], s)] = gap
    diffs = []
    for r in mr["runs"]:
        key = (r["seed"], r["sigma"])
        if key in ref and r["first"]:
            mine = (0.0 if r["first_at_start"] else r["rise"][r["first"][0]]) - r["drop0"]
            diffs.append({"seed": key[0], "sigma": key[1], "rerun": mine, "committed": ref[key]})
    d = [abs(x["rerun"] - x["committed"]) for x in diffs]
    return {
        "pairs": diffs,
        "n": len(d),
        "max_abs_diff": max(d) if d else None,
        "median_abs_diff": float(np.median(d)) if d else None,
    }


def first_route_full(scaling):
    """Window-checked first-route fits over all binary runs of the main sweep (scaling.json first_route,
    regenerated from the raw outputs): per seed power exponent, A / sigma + B and log-law rms."""
    out, allp = {}, []
    for tree, d in sorted(scaling["first_route"].items()):
        if not tree.endswith("/raw"):
            continue
        ps = d.get("per_seed", [])
        allp += [p["power_p"] for p in ps]
        out[tree] = {
            "n_seeds": len(ps),
            "per_seed": ps,
            "power_p": mean_se([p["power_p"] for p in ps]),
            "B_negative": sum(p["k3_B"] < 0 for p in ps),
            "max_rms": {k: max([p[k] for p in ps], default=None) for k in ("power_rms", "k3_rms")},
            "log_rms_range": [
                min([p["log_rms"] for p in ps], default=None),
                max([p["log_rms"] for p in ps], default=None),
            ],
            "excluded": d["excluded"],
        }
    out["all"] = {"power_p": mean_se(allp), "range": [min(allp), max(allp)] if allp else None}
    return out


def locality_all_sigma(main_rows):
    """Per sigma and seed, the final train loss of the four round-0 binary orders and of the MPS (main sweep):
    how often each order ends below the quadtree of the same seed (orders share the init cores of a seed)."""
    out = {}
    for s in sorted({r["sigma"] for r in main_rows}):
        rows = [
            r
            for r in main_rows
            if abs(r["sigma"] - s) < 1e-9 and "loss_end" in r and r["features"] == "raw"
        ]
        per_seed = {}
        for seed in sorted({r["seed"] for r in rows}):
            byo = {
                (r["tree"] if r["tree"] == "mps" else r["ordering"]): r["loss_end"]
                for r in rows
                if r["seed"] == seed
            }
            if "quadtree" in byo:
                per_seed[seed] = byo
        binary = [
            (seed, o, v)
            for seed, byo in per_seed.items()
            for o, v in byo.items()
            if o.startswith("scrambled")
        ]
        out[f"{s:g}"] = {
            "per_seed": per_seed,
            "scrambled_below_quadtree": [
                sum(v < per_seed[seed]["quadtree"] for seed, _, v in binary),
                len(binary),
            ],
            "mps_below_quadtree": [
                sum(byo.get("mps", 9) < byo["quadtree"] for byo in per_seed.values()),
                sum("mps" in byo for byo in per_seed.values()),
            ],
        }
    return out


def load_json(p):
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("out_prefix")
    ap.add_argument("--local", nargs="*", default=[])
    ap.add_argument("--audit", nargs="*", default=[])
    ap.add_argument("--logs", default=None)
    ap.add_argument("--local-runs", default=None)
    ap.add_argument("--main", default=None, help="summarize.py output of the full main sweep")
    a = ap.parse_args()
    R = Path(a.results_dir)
    events, routes = load_json(R / "events_table.json"), load_json(R / "routes_raw.json")
    hess, reg = load_json(R / "hessians_raw.json"), load_json(R / "regime_analysis.json")
    scaling, chi8, tables = (
        load_json(R / "scaling.json"),
        load_json(R / "chi8_scaling.json"),
        load_json(R / "tables.json"),
    )
    fashion = load_json(R / "fashion_scaling.json")
    local = [r for p in a.local for r in load_json(p) if "error" not in r]
    audit = [r for p in a.audit for r in load_json(p)]
    white = tables["white"]["s2s"]
    small = [
        v
        for k, v in white.items()
        if k.startswith("binary") and float(k.split("|")[1]) <= 0.1 + 1e-9
    ]
    res = {
        "first_route": first_route_clean(events),
        "events_refit": refit_all_series(events),
        "routes": route_units(routes),
        "shares_s0.05": shares_per_seed(routes, 0.05),
        "locality": locality_units(routes, local, audit),
        "hessians": hessian_exponents(hess),
        "mps_first": mps_first_rise(scaling, local),
        "regime": regime_checks(reg),
        "chi8": chi8_checks(chi8, a.logs),
        "repro": repro_check(reg, local),
        "two_point_chi8": two_point(chi8),
        "two_point_fashion": two_point(fashion),
        "mps_reruns": mps_reruns(local),
        "creep": creep(local, a.local_runs) if a.local_runs else None,
        "clock": clock(reg, local, audit),
        "white_s2s_sigma_le_0.1": {
            "s2s_runs": round(sum(v["s2s_fraction"] * v["runs"] for v in small)),
            "runs": sum(v["runs"] for v in small),
        },
    }
    res["mps_repro"] = mps_repro(scaling, res["mps_reruns"])
    if a.main:
        main_rows = [r for r in load_json(a.main) if "error" not in r and not r.get("centred")]
        res["first_route_full"] = first_route_full(scaling)
        res["locality_all_sigma"] = locality_all_sigma(main_rows)
        res["mps_full"] = mps_reruns(main_rows, tag="")
    out = Path(a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=1, default=float)
    open(out.with_suffix(".md"), "w", encoding="utf-8").write(markdown(res) + "\n")
    print(markdown(res))


def _f(x, fmt=".3g"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return format(x, fmt)


def _ms(v, fmt=".3f"):
    m, se, n = v
    if n > 1:
        return f"{m:{fmt}} ± {se:{fmt}} (n = {n})"
    return f"{m:{fmt}} (n = 1)" if n else "-"


def _md_first_route(res):
    fr = res["first_route"]
    md = [
        "## First route (k = 3): clean series",
        "",
        "Series = one seed whose first route (one sibling pair) rises inside drop 1 at every sigma of the "
        "series (events_table.json). Gap = drop 1 - drop 0 (tau clock). All laws fitted with relative "
        "residuals; 'unw.' = the round-0 unweighted log-law fit.",
        "",
        "| tree | seed | edges | sigma (n) | power p (rms) | log law rms (unw.) | A / sigma + B: A, B (rms) "
        "| B/A | p on 0.05/0.1/0.2/0.5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in fr["series"]:
        md.append(
            f"| {s['tree']} | {s['seed']} | {s['edges']} | {min(s['sigmas']):g}-{max(s['sigmas']):g} "
            f"({len(s['sigmas'])}) | {s['power_p']:.3f} ({s['power_rms']:.3f}) | {s['log_rms']:.3f} "
            f"({s['log_rms_unweighted']:.3f}) | {s['A']:.1f}, {s['B']:.1f} ({s['k3_rms']:.3f}) "
            f"| {s['B_over_A']:.2f} | {_f(s['power_p_main_grid'], '.3f')} |"
        )
    md += [
        "",
        "| tree | series | power p (mean ± SE) | range | log rms range (unw.) | max rms power / A/sigma+B "
        "| B < 0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, v in fr["summary"].items():
        lr, lu = v["log_rms_range"], v["log_rms_unweighted_range"]
        md.append(
            f"| {t} | {v['n_series']} | {_ms(v['power_p'])} | {v['power_p_range'][0]:.2f}-"
            f"{v['power_p_range'][1]:.2f} | {lr[0]:.2f}-{lr[1]:.2f} ({lu[0]:.2f}-{lu[1]:.2f}) "
            f"| {v['power_rms_max']:.3f} / {v['k3_rms_max']:.3f} | {v['B_negative']}/{v['n_series']} |"
        )
    md += [
        "",
        "## All events_table series refitted (both laws with relative residuals)",
        "",
        "| tree | features | seed | drop # | signature | k | sigma (n) | power p (rms) "
        "| log slope (rms; unw. rms) | preferred |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in res["events_refit"]:
        k = f"{s['k']}" + ("" if s["connected"] else " (disc.)")
        md.append(
            f"| {s['tree']} | {s['features']} | {s['seed']} | {sorted(set(s['drop_index']))} "
            f"| {s['signature']} | {k} | {s['sigma_range'][0]:g}-{s['sigma_range'][1]:g} ({s['n_sigma']}) "
            f"| {s['power_p']:.2f} ({s['power_rms']:.3f}) | {s['log_slope']:.3g} ({s['log_rms']:.3f}; "
            f"{s['log_rms_unweighted']:.2f}) | {s['preferred']} |"
        )
    return md


def _md_routes(res):
    ru = res["routes"]
    per = ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in ru["per_state_hits"].items())
    tog = ru["pairs_rising_together_in_kept"]
    md = [
        "",
        "## Route-amplitude rule, rank-1 states, critical-point units",
        "",
        f"States: {ru['n_states']['all']}, dropped (rank1_rel_err > {ERR_MAX}): {ru['n_states']['dropped']}. "
        f"Hits per kept state: {per}. Units: {ru['n_units']}; outcomes {ru['outcomes']} (chance per unit "
        f"{ru['chance']}). The first pair is two edges crossing together in {tog[0]}/{tog[1]} kept states.",
        "",
        "| ordering | seed | sigmas | plateau L | predicted pair | first pair | outcome |",
        "|---|---|---|---|---|---|---|",
    ]
    for u in ru["units"]:
        extra = f" ({sum(u['hits'])}/{len(u['hits'])})" if u["outcome"] == "mixed" else ""
        sig = ", ".join(f"{s:g}" for s in u["sigmas"])
        md.append(
            f"| {u['ordering']} | {u['seed']} | {sig} | {u['plateau_loss']:.3f} | {u['pred']} | {u['obs']} "
            f"| {u['outcome']}{extra} |"
        )
    sh = res["shares_s0.05"]
    md += [
        "",
        f"## Share of sum lambda^2 on k <= 5 routes, one value per (order, seed), sigma = {sh['sigma']}",
        "",
        f"Pair-count chance level: {sh['pair_count_chance_k5']:.2f} (12 of 28 pairs have k <= 5).",
        "",
        "| order | mean ± SE over seeds | values (seeds 0-4) |",
        "|---|---|---|",
    ]
    for k, v in sh["summary"].items():
        md.append(
            f"| {k} | {_ms(v['mean_se'], '.2f')} | {', '.join(f'{x:.3f}' for x in v['values'])} |"
        )
    return md


def _md_locality(res):
    lo = res["locality"]
    md = [
        "",
        f"## Locality per (order, seed), sigma = {lo['sigma']} (tau_end = 1e4)",
        "",
        "| seed | order | source | k <= 5 share | L(tau_end) | test acc | tau(L<=0.25) | tau(L<=0.2) "
        "| tau(L<=0.15) | t_a |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in lo["rows"]:
        md.append(
            f"| {r['seed']} | {r['ordering']} | {r['source']} | {_f(r['share_k5'], '.2f')} "
            f"| {r['loss_end']:.4f} | {r['test_acc']:.3f} | {_f(r['tau_L0.25'], '.4g')} "
            f"| {_f(r['tau_L0.2'], '.4g')} | {_f(r['tau_L0.15'], '.4g')} | {r['t_a']:.3g} |"
        )
    for k in ("spearman_all", "spearman_committed_orders", "spearman_seed0"):
        v = lo[k]
        if v:
            md.append(
                f"\n{k}: share vs L(tau_end): rho = {v['rho']:.2f}, p = {v['p']:.2g}, n = {v['n']}"
            )
    md += [
        "",
        "| order | n | L(tau_end) mean ± SE | values | median tau(L<=0.25) / (0.2) / (0.15) |",
        "|---|---|---|---|---|",
    ]
    for o, v in lo["per_order"].items():
        vals = ", ".join(f"{x:.3f}" for x in v["loss_values"])
        taus = " / ".join(_f(v[f"tau_L{lv}_median"], ".4g") for lv in LEVELS)
        md.append(f"| {o} | {v['n']} | {_ms(v['loss_end'], '.4f')} | {vals} | {taus} |")
    ranks = "; ".join(
        f"seed {s}: {v['lower_final_loss_than_quadtree']}/{v['n_orders'] - 1} lower, "
        f"{v['reach_L0.25_before_quadtree']}/{v['n_orders'] - 1} earlier"
        for s, v in lo["quadtree_rank_per_seed"].items()
    )
    md.append(
        f"\nOther orders with a lower final loss than the quadtree / reaching L <= 0.25 first: {ranks}"
    )
    return md


def _md_rest(res):
    md = [
        "",
        "## Hessian at the first plateau: per-seed exponent of |lambda_min| vs sigma",
        "",
        "| tree | plateau band | seed | exponent | sigmas (decades) | n_neg (projected) |",
        "|---|---|---|---|---|---|",
    ]
    for t, v in res["hessians"].items():
        for seed, w in v["per_seed"].items():
            sig = ", ".join(f"{s:g}" for s in w["sigmas"])
            md.append(
                f"| {t} | {v['band'][0]}-{v['band'][1]} | {seed} | {w['exponent']:.3f} | {sig} "
                f"({w['decades']:.2f}) | {w['n_neg']} |"
            )
        md.append(f"| {t} | | mean ± SE | {_ms(v['mean_se'])} | | |")
    mp = res["mps_first"]
    g0, g1 = mp["resolved_gap_range"]
    md += [
        "",
        "## MPS first rank increase vs the first drop",
        "",
        f"{mp['n']} runs: {mp['at_start']} recorded at tau = 0 (s2/s1 >= 0.05 from the start of phase 2, "
        f"gap < -5); {mp['n_resolved']} resolved, {mp['resolved_before_peak']} of them before the first drop's "
        f"rate peak (gap {g0:.2f} to {g1:.2f}).",
    ]
    if mp["local_reruns"]:
        md.append(
            "Local reruns: "
            + "; ".join(
                f"seed {r['seed']} sigma {r['sigma']:g}: {r['first']} at_start={r['at_start']} gap {r['gap']:.2f}"
                for r in mp["local_reruns"]
            )
        )
    rg = res["regime"]
    floors = ", ".join(f"{s}: {v:.4f}" for s, v in rg["floors"].items())
    md += [
        "",
        "## Regime map checks",
        "",
        f"Lazy floor L_lin(inf) per seed: {floors} (mean {rg['floor_mean']:.3f}).",
        "",
        "| seed | lazy floor | plain runs sigma 0.1-0.3: L(1e4) range | below floor | rich sigma 1.4: L(T) |",
        "|---|---|---|---|---|",
    ]
    for s, v in rg["per_seed"].items():
        lo_, hi_ = v["plain_end_sigma_ge_0.1"]
        rich = ", ".join(f"{x:.4f}" for x in v["rich_1.4_end"])
        md.append(
            f"| {s} | {v['lazy_floor']:.4f} | {lo_:.4f}-{hi_:.4f} | {v['plain_end_below_floor']}/{v['n_plain']} "
            f"| {rich} |"
        )
    lz = rg["lazy_runs"]
    exc = [r["excess_over_floor"] for r in lz]
    md.append(
        f"\nLazy-labelled runs ({len(lz)}): L(T) = {min(r['L_T'] for r in lz):.3f}-{max(r['L_T'] for r in lz):.3f}, "
        f"L(T) - floor = {min(exc):.3f}-{max(exc):.3f}: not converged at the horizon."
    )
    md.append(
        "sigma = 1.4 runs: "
        + "; ".join(
            f"seed {r['seed']}: {r['n_drops']} drops, delta_K {r['delta_K']:.3g}, L(T) {r['L_T']:.4f} vs floor "
            f"{r['floor']:.4f}, T {r['t2']:.4g}"
            for r in rg["rich_sigma_1.4"]
        )
    )
    md += [
        "",
        "Label counts per sigma under each threshold pair (dK = delta_K threshold, rho = separation threshold):",
        "",
    ]
    for key, v in rg["label_sensitivity"].items():
        items = sorted(v.items(), key=lambda kv: (float(kv[0].split("|")[0]), kv[0]))
        md.append(f"- {key}: " + ", ".join(f"{k} {n}" for k, n in items))
    c8 = res["chi8"]
    md += ["", "## chi = 8", ""]
    for t in ("binary_quadtree/raw", "binary_scrambled/raw"):
        v = c8[t]
        sl = ", ".join(
            f"seed {s['seed']}: {s['slope']:.2f}"
            + ("" if s["single_route"] else f" (routes {s['routes']})")
            for s in v["two_point"]
        )
        md.append(
            f"- {t}: two-point slopes (0.05 -> 0.14, rank clock) {sl}; mean {_ms(v['mean_se'], '.2f')}; "
            f"seeds with one route {_ms(v['mean_se_single_route'], '.2f')}"
        )
    if "stops" in c8:
        md.append(f"- stop reasons (run logs): {c8['stops']}")
    ck = res["clock"]
    md += [
        "",
        f"## Absolute clock: first-escape time t_a at sigma = {ck['sigma']}",
        "",
        "| tree | order | seed | t_a | source |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(ck["rows"], key=lambda r: (r["seed"], r["t_a"])):
        md.append(
            f"| {r['tree']} | {r['ordering']} | {r['seed']} | {r['t_a']:.3g} | {r['source']} |"
        )
    ratio = ", ".join(f"{s}: {v:.2f}" for s, v in sorted(ck["mps_over_quadtree"].items()))
    faster = ", ".join(f"{s}: {v}" for s, v in sorted(ck["binary_orders_faster_than_mps"].items()))
    md.append(
        f"\nt_a(MPS) / t_a(quadtree) per seed: {ratio}; binary runs with t_a below the MPS's, per seed: {faster}"
    )
    rp = res.get("repro") or []
    if rp:
        md += ["", "## Reproduction: fix1 reruns vs committed values (quadtree, sigma = 0.1)", ""]
        md += [
            f"- seed {r['seed']}: |t_a / t_a(committed) - 1| = {r['t_a_rel_diff']:.1e}, L(1e4) {r['loss_end']:.5f} vs "
            f"{r['loss_end_committed']:.5f}"
            for r in rp
        ]
    cr = res.get("creep")
    if cr and cr["n"]:
        md += [
            "",
            f"## Creep between drops (binary fix1 reruns, sigma = 0.1): {cr['n']} intervals, loss decrease "
            f"between drops / next drop: median {cr['median']:.2f}, range {cr['range'][0]:.2f}-{cr['range'][1]:.2f}",
        ]
    for key, name in (("two_point_chi8", "chi = 8"), ("two_point_fashion", "Fashion-MNIST")):
        tp = res.get(key)
        if not tp:
            continue
        md += ["", f"## {name}: two-point slopes (sigma 0.05 -> 0.14) of the rank-clock gap; first rises at tau = 0", ""]
        for tree, v in tp.items():
            sl = ", ".join(f"seed {r['seed']}: {_f(r['slope'], '.2f')} (routes {r['routes']})" for r in v["rows"])
            md.append(f"- {tree}: {sl}; mean {_ms(v['slope'], '.2f')}; runs with the first rise at tau = 0: {v['at_start']}")
    w = res["white_s2s_sigma_le_0.1"]
    md += [
        "",
        f"## Whitened features: S2S fraction of binary runs at sigma <= 0.1: {w['s2s_runs']}/{w['runs']} "
        f"= {w['s2s_runs'] / w['runs']:.2f}",
    ]
    return md


def _md_mps_reruns(res):
    mr = res.get("mps_reruns")
    if not mr:
        return []
    md = [
        "",
        "## MPS reruns (configs/fix1_mps.yaml, tau <= 1000): first rise and per-edge laws",
        "",
        f"{mr['n']} runs; first rise already at tau = 0 (threshold met at the start of phase 2): {mr['first_at_start']}; "
        f"resolved: {mr['n_resolved']}, of which a single edge {mr['first_single_edge_resolved']} (edges "
        f"{mr['first_edges_resolved']}), before the first drop's rate peak {mr['resolved_before_drop0_peak']}, inside "
        f"the first drop's widened window {mr['resolved_inside_drop0_window']}.",
        "",
        "| edge | seeds | power p (mean ± SE) | power rms | log slope | log rms | seeds preferring log |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in mr["edge_fits"]:
        md.append(
            f"| {e['edge']} | {e['n_seeds']} | {_ms(e['power_p'], '.2f')} | {e['power_rms'][0]:.3f} "
            f"| {e['log_slope'][0]:.3g} ± {e['log_slope'][1]:.2g} | {e['log_rms'][0]:.3f} | {e['log_preferred']}/{e['n_seeds']} |"
        )
    md += [
        "",
        "| seed | sigma | t_a | first drop peak | first rising edges | at tau = 0 | rise of edges 5 / 4 / 3 / 2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in mr["runs"]:
        rise = " / ".join(_f(r["rise"][e], ".4g") for e in (5, 4, 3, 2))
        md.append(
            f"| {r['seed']} | {r['sigma']:g} | {r['t_a']:.4g} | {r['drop0']:.2f} | {r['first']} | {r['first_at_start']} | {rise} |"
        )
    rp = res.get("mps_repro")
    if rp and rp["n"]:
        md += [
            "",
            f"Reproduction: first rise - first drop peak, reruns vs committed full-horizon runs ({rp['n']} runs): "
            f"median |difference| {rp['median_abs_diff']:.3g}, max {rp['max_abs_diff']:.3g} (time units).",
        ]
    return md


def _md_full(res):
    md = []
    fr = res.get("first_route_full")
    if fr:
        md += [
            "",
            "## First route over all binary runs (window-checked attribution, full main sweep)",
            "",
            "| tree | seeds fitted | power p (mean ± SE) | max rms power / A/sigma+B | log rms range | B < 0 |",
            "|---|---|---|---|---|---|",
        ]
        for t, v in fr.items():
            if t == "all":
                continue
            lr = v["log_rms_range"]
            md.append(
                f"| {t} | {v['n_seeds']} | {_ms(v['power_p'])} | {_f(v['max_rms']['power_rms'], '.3f')} / "
                f"{_f(v['max_rms']['k3_rms'], '.3f')} | {_f(lr[0], '.2f')}-{_f(lr[1], '.2f')} | {v['B_negative']}/{v['n_seeds']} |"
            )
        md.append(
            f"\nAll fitted seeds: power p {_ms(fr['all']['power_p'])}, range {fr['all']['range'][0]:.2f}-{fr['all']['range'][1]:.2f}."
        )
        md += [
            "",
            "Excluded (seed, sigma, reason): "
            + "; ".join(f"{t}: {v['excluded']}" for t, v in fr.items() if t != "all"),
        ]
    la = res.get("locality_all_sigma")
    if la:
        md += [
            "",
            "## Final train loss per seed, main sweep (round-0 orders and the MPS)",
            "",
            "| sigma | scrambled runs ending below the quadtree of the same seed | MPS below quadtree |",
            "|---|---|---|",
        ]
        for s, v in la.items():
            md.append(
                f"| {s} | {v['scrambled_below_quadtree'][0]}/{v['scrambled_below_quadtree'][1]} | "
                f"{v['mps_below_quadtree'][0]}/{v['mps_below_quadtree'][1]} |"
            )
    mf = res.get("mps_full")
    if mf:
        md += [
            "",
            "## MPS per-edge laws over the full horizon (main sweep, relative residuals for both laws)",
            "",
            f"First rise at tau = 0: {mf['first_at_start']}/{mf['n']}; resolved: single edge {mf['first_single_edge_resolved']}/"
            f"{mf['n_resolved']} (edges {mf['first_edges_resolved']}), before the first drop's peak {mf['resolved_before_drop0_peak']}.",
            "",
            "| edge | seeds | power p (mean ± SE) | power rms | log slope | log rms | seeds preferring log |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in mf["edge_fits"]:
            md.append(
                f"| {e['edge']} | {e['n_seeds']} | {_ms(e['power_p'], '.2f')} | {e['power_rms'][0]:.3f} "
                f"| {e['log_slope'][0]:.3g} ± {e['log_slope'][1]:.2g} | {e['log_rms'][0]:.3f} | {e['log_preferred']}/{e['n_seeds']} |"
            )
    return md
def markdown(res):
    md = ["# Real data, fix round 1: re-analysis (`fix1_analysis.py`; committed outputs, local reruns, full raw sweep)", ""]
    md += (
        _md_first_route(res)
        + _md_full(res)
        + _md_routes(res)
        + _md_locality(res)
        + _md_rest(res)
        + _md_mps_reruns(res)
    )
    return "\n".join(md)


if __name__ == "__main__":
    main()
