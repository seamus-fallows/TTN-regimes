"""Per-seed scaling fits of escape times and gaps against sigma (summarize.py output -> JSON).

Usage: uv run python experiments/real_data/scaling.py <summary.json>... <out.json>

Quantities per run: t1 (absolute time of the first loss drop) and g2 = gap between the first two loss
drops (tau clock). Runs are classified before pooling: a seed enters the g2 fit only if every one of its
sigma values has >= 2 drops (otherwise it is listed under "excluded").
Fits per seed (same init direction rescaled by sigma), in the same space with the same weights:
  t1:  log t1 = a + p log(1/sigma)                      (prediction p = K - 2)
  g2:  power   log g2 = a + p log(1/sigma)
       log-law g2 = a + b log(1/sigma)                   (k = 2 prediction)
       k3      g2 = A / sigma + B                        (k = 3 prediction with head start)
  The three g2 models are compared by their RMS relative residual (g_fit - g) / g, all in linear g.
Reported: mean +- SE over seeds of each fitted exponent / slope, and of each model's RMS residual.
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


def fit_seed(sig, val):
    x = np.log(1 / sig)
    out = {}
    a, p = _fit_linear(x, np.log(val))
    out["power_p"], out["power_rms"] = p, np.sqrt(np.mean((np.exp(a + p * x) / val - 1) ** 2))
    a, b = _fit_linear(x, val)
    out["log_slope"], out["log_rms"] = b, np.sqrt(np.mean(((a + b * x) - val) ** 2 / val**2))
    a, b = _fit_linear(1 / sig, val)
    out["k3_A"], out["k3_B"] = b, a
    out["k3_rms"] = np.sqrt(np.mean(((a + b / sig) - val) ** 2 / val**2))
    return out


def mean_se(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan, len(v))


def main(*args):
    *paths, out_path = args
    rows = [r for p in paths for r in json.load(open(p)) if "error" not in r and not r.get("centred")]
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tree = r["tree"] + "_" + ("scrambled" if r["ordering"].startswith("scrambled") else r["ordering"])
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
                excluded.append({"seed": key, "sigmas_without_second_drop":
                                 [float(s) for s, g in zip(sig, g2) if not np.isfinite(g)]})
            # rank-based gap: first s_2/s_1 >= 0.05 event minus the first drop, if the same edges
            # rise first at every sigma (same route); else the seed is listed as mixed
            ids = {tuple(r.get("first_rank2_edges") or []) for r in runs}
            gr = np.array([r["t_first_rank2"] - r["drops"][0]["t"] if r.get("drops") else np.nan
                           for r in runs])
            if len(ids) == 1 and len(sig) >= 4 and np.all(np.isfinite(gr)) and np.all(gr > 0):
                gr_fits.append(fit_seed(sig, gr))
            else:
                gr_mixed.append({"seed": key, "first_edges": sorted(map(list, ids)),
                                 "gaps": [float(g) for g in gr]})
        res = {"t1_exponent": mean_se(t1_fits), "n_seeds": len(seeds), "g2_excluded": excluded,
               "grank_mixed": gr_mixed}
        for k in ("power_p", "power_rms", "log_slope", "log_rms", "k3_A", "k3_B", "k3_rms"):
            res[f"g2_{k}"] = mean_se([f[k] for f in g2_fits])
            res[f"grank_{k}"] = mean_se([f[k] for f in gr_fits])
        result[f"{tree}/{feats}"] = res
    json.dump(result, open(out_path, "w"), indent=1)
    for k, v in result.items():
        print(k, "t1 exp %.3f +- %.3f (n=%d)" % v["t1_exponent"],
              "| g2 power p %.2f +- %.2f (n=%d)" % v["g2_power_p"],
              "rms: power %.3f log %.3f k3 %.3f" % (v["g2_power_rms"][0], v["g2_log_rms"][0], v["g2_k3_rms"][0]),
              "| excluded", len(v["g2_excluded"]),
              "| rank gap p %.2f +- %.2f (n=%d)" % v["grank_power_p"],
              "rms: power %.3f log %.3f k3 %.3f" % (v["grank_power_rms"][0], v["grank_log_rms"][0], v["grank_k3_rms"][0]),
              "mixed", len(v["grank_mixed"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
