"""Per-seed scaling fits of escape times and gaps against sigma (summarize.py output -> JSON).

Usage: uv run python experiments/real_data/scaling.py <summary.json>... <out.json>

Quantities per run: t1 (absolute time of the first loss drop) and g2 = gap between the first two loss
drops (tau clock). Runs are classified before pooling: a seed enters the g2 fit only if every one of its
sigma values has >= 2 drops (otherwise it is listed under "excluded").
Fits per seed (same init direction rescaled by sigma), all with relative residuals (g_fit - g) / g:
  t1:  log t1 = a + p log(1/sigma)                      (prediction p = K - 2; an identity, see notes 4.1)
  g2:  power   log g2 = a + p log(1/sigma)               (least squares in log g = relative, to first order)
       log-law g2 = a + b log(1/sigma)                   (k = 2 prediction; weighted least squares, weights 1/g)
       k3      g2 = A / sigma + B                        (k = 3 prediction with head start; weights 1/g)
  The three g2 models are compared by their RMS relative residual. (Fix round 1: the log law and A/sigma + B
  were fitted unweighted before, which inflated their residuals 2-4x; notes/real_data.md section 8.)
Reported: mean +- SE over seeds of each fitted exponent / slope, and of each model's RMS residual.

First-route attribution (fix round 1): the loss drop that carries a route is the drop whose widened window
[t_lo - width, t_hi + width] contains the route's rank crossing (first s_2/s_1 >= 0.05 on its edges). A
crossing inside drop 0's window (merged with the escape) or inside no window (risen during the creep
between drops) is excluded and listed with its reason; before, the next drop after the crossing was taken,
which could belong to a different route.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def _fit_linear(x, y):
    a = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return coef


def _fit_relative(x, y):
    """(a, b) minimising sum ((a + b x) / y - 1)^2: least squares with relative residuals."""
    a = np.vstack([np.ones_like(x), x]).T / y[:, None]
    coef, *_ = np.linalg.lstsq(a, np.ones_like(y), rcond=None)
    return coef


def fit_seed(sig, val):
    sig, val = np.asarray(sig, float), np.asarray(val, float)
    x = np.log(1 / sig)
    out = {}
    a, p = _fit_linear(x, np.log(val))
    out["power_p"], out["power_rms"] = p, np.sqrt(np.mean((np.exp(a + p * x) / val - 1) ** 2))
    a, b = _fit_relative(x, val)
    out["log_slope"], out["log_rms"] = b, np.sqrt(np.mean(((a + b * x) / val - 1) ** 2))
    a, b = _fit_relative(1 / sig, val)
    out["k3_A"], out["k3_B"] = b, a
    out["k3_rms"] = np.sqrt(np.mean(((a + b / sig) / val - 1) ** 2))
    return out


def mean_se(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return (
        float(v.mean()),
        float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
        len(v),
    )


def edge_scaling(rows, min_sigmas=5):
    """Per edge: T_e = tau of the edge's first s_2/s_1 >= 0.05 minus tau of the first loss drop,
    per seed against sigma (seeds with T_e > 0 at >= min_sigmas sigma values spanning a decade).
    Fits as in fit_seed; mean +- SE over seeds of the power exponent and log slope, and mean rms."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r.get("drops"):
            continue
        tree = f"{r['tree']}_{r['ordering']}/{r['features']}"
        ev = np.array(r["rank_events"], float)[:, 0] - r["drops"][0]["t"]
        for e, t in enumerate(ev):
            groups[(tree, e, r["edge_levels"][e])][r["seed"]].append((r["sigma"], t))
    out = []
    for (tree, e, lv), seeds in sorted(groups.items()):
        fits = []
        for seed, pts in seeds.items():
            pts = sorted((s, t) for s, t in pts if np.isfinite(t) and t > 0)
            if len(pts) >= min_sigmas and pts[-1][0] / pts[0][0] >= 9.9:
                fits.append(fit_seed(np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
        if fits:
            out.append(
                {
                    "tree": tree,
                    "edge": e,
                    "level": lv,
                    "n_seeds": len(fits),
                    **{
                        k: mean_se([f[k] for f in fits])
                        for k in ("power_p", "power_rms", "log_slope", "log_rms")
                    },
                }
            )
    return out


def carrier_drop(drops, t_cross):
    """(index, reason) of the loss drop whose widened window [t_lo - width, t_hi + width] contains the
    rank crossing t_cross: reason 'merged' if that is drop 0 (the route rose with the escape),
    'between drops' if no window contains it (it rose during the creep), else 'ok'."""
    inside = [
        i
        for i, d in enumerate(drops)
        if d["t_lo"] - d["width"] <= t_cross <= d["t_hi"] + d["width"]
    ]
    if not np.isfinite(t_cross) or not inside:
        return None, "between drops"
    if inside[0] == 0:
        return 0, "merged"
    return inside[0], "ok"


def first_route_points(rows):
    """Per (tree, seed) of the binary runs: the route R* (edges that gain rank first at the smallest
    sigma) and, per sigma, the drop-to-drop gap (drop carrying R*'s rank crossing minus drop 0) and the
    rank-clock gap (R*'s crossing minus drop 0). A sigma is kept only if its first route is R* and the
    crossing lies inside the window of a drop after drop 0 (`carrier_drop`); else excluded with a reason."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("drops") and r["tree"] == "binary":
            groups[(f"binary_{r['ordering']}/{r['features']}", r["seed"])].append(r)
    out = {}
    for (tree, seed), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["sigma"])
        ref = rs[0].get("first_rank2_edges")
        if not ref:
            continue
        keep, excl = [], []
        for r in rs:
            t_cross = r.get("t_first_rank2", np.nan)
            if r.get("first_rank2_edges") != ref:
                excl.append((r["sigma"], "other route"))
                continue
            i, reason = carrier_drop(r["drops"], t_cross)
            if reason != "ok":
                excl.append((r["sigma"], reason))
                continue
            t0 = r["drops"][0]["t"]
            keep.append((r["sigma"], r["drops"][i]["t"] - t0, t_cross - t0, i))
        out[(tree, seed)] = {"route": ref, "kept": keep, "excluded": excl}
    return out


def first_route(rows, min_sigmas=4):
    """Classified first-route gap (binary trees), per seed, from `first_route_points`; fitted per seed
    over >= min_sigmas kept values spanning >= 0.5 decade (drop-to-drop gaps; the rank-clock gaps are
    fitted alongside as `rank_*`)."""
    by_tree = defaultdict(
        lambda: {"fits": [], "rank_fits": [], "excluded": [], "routes": [], "points": []}
    )
    for (tree, seed), d in first_route_points(rows).items():
        keep = d["kept"]
        by_tree[tree]["excluded"] += [(seed, s, why) for s, why in d["excluded"]]
        by_tree[tree]["routes"].append((seed, d["route"]))
        by_tree[tree]["points"].append((seed, keep))
        if len(keep) >= min_sigmas and keep[-1][0] / keep[0][0] >= 10**0.5:
            sig = np.array([k[0] for k in keep])
            f = fit_seed(sig, np.array([k[1] for k in keep]))
            f["sigma_range"] = (keep[0][0], keep[-1][0], len(keep))
            f["seed"] = seed
            by_tree[tree]["fits"].append(f)
            g_rank = np.array([k[2] for k in keep])
            if np.all(g_rank > 0):
                by_tree[tree]["rank_fits"].append(fit_seed(sig, g_rank))
    out = {}
    keys = ("power_p", "power_rms", "log_slope", "log_rms", "k3_A", "k3_B", "k3_rms")
    for tree, d in by_tree.items():
        out[tree] = {
            "n_seeds": len(d["fits"]),
            "excluded": d["excluded"],
            "routes": d["routes"],
            "points": d["points"],
            "ranges": [f["sigma_range"] for f in d["fits"]],
            "per_seed": [{k: f[k] for k in ("seed", "sigma_range", *keys)} for f in d["fits"]],
            **{k: mean_se([f[k] for f in d["fits"]]) for k in keys},
            **{f"rank_{k}": mean_se([f[k] for f in d["rank_fits"]]) for k in keys},
        }
    return out


def main(*args):
    *paths, out_path = args
    rows = [
        r for p in paths for r in json.load(open(p)) if "error" not in r and not r.get("centred")
    ]
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tree = (
            r["tree"]
            + "_"
            + ("scrambled" if r["ordering"].startswith("scrambled") else r["ordering"])
        )
        groups[(tree, r["features"])][(r["ordering"], r["seed"])].append(r)
    result = {}
    for (tree, feats), seeds in sorted(groups.items()):
        t1_fits, g2_fits, excluded, gr_fits, gr_mixed = [], [], [], [], []
        for key, runs in seeds.items():
            runs = sorted(runs, key=lambda r: r["sigma"])
            sig = np.array([r["sigma"] for r in runs])
            t1 = np.array([r.get("t1", np.nan) for r in runs])
            if len(sig) >= 3 and np.all(np.isfinite(t1)):
                x = np.log(1 / sig)
                t1_fits.append(_fit_linear(x, np.log(t1))[1])
            g2 = np.array([r["gaps"][0] if r.get("gaps") else np.nan for r in runs])
            if len(sig) >= 4 and np.all(np.isfinite(g2)) and np.all(g2 > 0):
                g2_fits.append(fit_seed(sig, g2))
            else:
                excluded.append(
                    {
                        "seed": key,
                        "sigmas_without_second_drop": [
                            float(s) for s, g in zip(sig, g2) if not np.isfinite(g)
                        ],
                    }
                )
            # rank-based gap: first s_2/s_1 >= 0.05 event minus the first drop, if the same edges
            # rise first at every sigma (same route); else the seed is listed as mixed
            ids = {tuple(r.get("first_rank2_edges") or []) for r in runs}
            gr = np.array(
                [
                    r["t_first_rank2"] - r["drops"][0]["t"] if r.get("drops") else np.nan
                    for r in runs
                ]
            )
            if len(ids) == 1 and len(sig) >= 4 and np.all(np.isfinite(gr)) and np.all(gr > 0):
                gr_fits.append(fit_seed(sig, gr))
            else:
                gr_mixed.append(
                    {
                        "seed": key,
                        "first_edges": sorted(map(list, ids)),
                        "gaps": [float(g) for g in gr],
                    }
                )
        res = {
            "t1_exponent": mean_se(t1_fits),
            "n_seeds": len(seeds),
            "g2_excluded": excluded,
            "grank_mixed": gr_mixed,
        }
        for k in ("power_p", "power_rms", "log_slope", "log_rms", "k3_A", "k3_B", "k3_rms"):
            res[f"g2_{k}"] = mean_se([f[k] for f in g2_fits])
            res[f"grank_{k}"] = mean_se([f[k] for f in gr_fits])
        result[f"{tree}/{feats}"] = res
    result["edge_scaling"] = edge_scaling(rows)
    result["first_route"] = first_route(rows)
    json.dump(result, open(out_path, "w"), indent=1)
    for k, v in result.items():
        if k in ("edge_scaling", "first_route"):
            continue
        print(
            k,
            "t1 exp %.3f +- %.3f (n=%d)" % v["t1_exponent"],
            "| g2 power p %.2f +- %.2f (n=%d)" % v["g2_power_p"],
            "rms: power %.3f log %.3f k3 %.3f"
            % (v["g2_power_rms"][0], v["g2_log_rms"][0], v["g2_k3_rms"][0]),
            "| excluded",
            len(v["g2_excluded"]),
            "| rank gap p %.2f +- %.2f (n=%d)" % v["grank_power_p"],
            "rms: power %.3f log %.3f k3 %.3f"
            % (v["grank_power_rms"][0], v["grank_log_rms"][0], v["grank_k3_rms"][0]),
            "mixed",
            len(v["grank_mixed"]),
        )
    print("classified first route (binary): gap to the first rank increase on the seed's route")
    for tree, d in sorted(result["first_route"].items()):
        print(
            f"  {tree:28s} seeds {d['n_seeds']} power p {d['power_p'][0]:.2f} +- {d['power_p'][1]:.2f}"
            f" rms power {d['power_rms'][0]:.3f} log {d['log_rms'][0]:.3f} k3 {d['k3_rms'][0]:.3f}"
            f" | excluded (seed, sigma) {d['excluded']} | ranges {d['ranges']}"
        )
    print("per-edge first rank increase (tau after the first drop):")
    for d in result["edge_scaling"]:
        print(
            f"  {d['tree']:28s} edge {d['edge']:2d} level {d['level']:2d} seeds {d['n_seeds']}"
            f"  power p {d['power_p'][0]:.2f} +- {d['power_p'][1]:.2f} (rms {d['power_rms'][0]:.3f})"
            f"  log slope {d['log_slope'][0]:.3g} +- {d['log_slope'][1]:.2g} (rms {d['log_rms'][0]:.3f})"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
