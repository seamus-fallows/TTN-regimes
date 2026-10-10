"""Independent recomputation: k = 3 first-route gaps (window-checked attribution) and MPS per-edge laws.

Input: the per-run summary rows (data/real_data_local/full/main_summary.json); own attribution and fits.
"""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

rows = [r for r in json.load(open(sys.argv[1])) if "error" not in r and not r.get("centred")]


def fits(sig, g):
    sig, g = np.asarray(sig, float), np.asarray(g, float)
    x = np.log(1 / sig)
    # power law: true relative residuals (nonlinear), started from the log-space fit
    p0 = np.polyfit(x, np.log(g), 1)
    rp = least_squares(lambda c: np.exp(c[0] + c[1] * x) / g - 1, [p0[1], p0[0]])
    p_log = p0[0]
    rms_pow = np.sqrt(np.mean(rp.fun**2))
    rms_pow_logfit = np.sqrt(np.mean((np.exp(p0[1] + p0[0] * x) / g - 1) ** 2))
    # log law a + b x, relative residuals
    rl = least_squares(lambda c: (c[0] + c[1] * x) / g - 1, [g.mean(), 1.0])
    # k = 3: A / sigma + B, relative residuals
    rk = least_squares(lambda c: (c[0] / sig + c[1]) / g - 1, [g.mean() * sig.mean(), 0.0])
    return dict(
        p=p_log,
        p_rel=rp.x[1],
        rms_pow=rms_pow_logfit,
        rms_pow_rel=rms_pow,
        log_b=rl.x[1],
        rms_log=np.sqrt(np.mean(rl.fun**2)),
        A=rk.x[0],
        B=rk.x[1],
        rms_k3=np.sqrt(np.mean(rk.fun**2)),
    )


def route(r):
    ev = np.array(r["rank_events"], float)[:, 0]
    if not np.isfinite(ev).any():
        return None, np.nan
    t = np.nanmin(ev)
    tol = max(1.0, 0.05 * t)
    return tuple(int(e) for e in np.flatnonzero(ev <= t + tol)), t


# ---- first route, binary raw
groups = defaultdict(list)
for r in rows:
    if r["tree"] == "binary" and r.get("drops"):
        groups[(r["ordering"], r["features"], r["seed"])].append(r)
series = []
excl_count = defaultdict(int)
for (o, f, seed), rs in sorted(groups.items()):
    rs.sort(key=lambda r: r["sigma"])
    ref, _ = route(rs[0])
    kept = []
    for r in rs:
        rt, tc = route(r)
        if rt != ref:
            excl_count[(f, "other")] += 1
            continue
        inside = [
            i
            for i, d in enumerate(r["drops"])
            if d["t_lo"] - d["width"] <= tc <= d["t_hi"] + d["width"]
        ]
        if not inside:
            excl_count[(f, "between")] += 1
            continue
        if inside[0] == 0:
            excl_count[(f, "merged")] += 1
            continue
        kept.append((r["sigma"], r["drops"][inside[0]]["t"] - r["drops"][0]["t"]))
        excl_count[(f, "kept")] += 1
    if len(kept) >= 4 and kept[-1][0] / kept[0][0] >= 10**0.5 - 1e-9:
        s, g = zip(*kept)
        series.append(
            dict(order=o, feat=f, seed=seed, route=ref, n=len(s), lo=s[0], hi=s[-1], **fits(s, g))
        )
print("attribution counts:", dict(excl_count))
for f in ("raw", "white"):
    S = [x for x in series if x["feat"] == f]
    print(f"--- first-route series, {f}: {len(S)}")
    for x in S:
        print(
            f"  {x['order']:11s} s{x['seed']} {x['route']} {x['lo']}-{x['hi']} ({x['n']}) p={x['p']:.3f} "
            f"p_rel={x['p_rel']:.3f} rms_pow={x['rms_pow']:.3f} rms_log={x['rms_log']:.3f} "
            f"A={x['A']:.1f} B={x['B']:.1f} rms_k3={x['rms_k3']:.3f}"
        )
    if S:
        p = np.array([x["p"] for x in S])
        print(
            f"  p mean {p.mean():.3f} +- {p.std(ddof=1) / np.sqrt(len(p)) if len(p) > 1 else np.nan:.3f}, range {p.min():.2f}-{p.max():.2f}; "
            f"B<0 {sum(x['B'] < 0 for x in S)}/{len(S)}; log rms {min(x['rms_log'] for x in S):.3f}-{max(x['rms_log'] for x in S):.3f}; "
            f"k3 rms max {max(x['rms_k3'] for x in S):.3f}; pow rms max {max(x['rms_pow'] for x in S):.3f}"
        )

# ---- MPS per-edge laws (raw), T_e = first s2/s1 >= 0.05 minus first drop, T_e > 0, >= 5 sigma over a decade
mg = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["tree"] == "mps" and r["features"] == "raw" and r.get("drops"):
        ev = np.array(r["rank_events"], float)[:, 0] - r["drops"][0]["t"]
        for e, t in enumerate(ev):
            mg[e][r["seed"]].append((r["sigma"], t))
tot_log = tot = 0
print("--- MPS edges (raw)")
for e in sorted(mg):
    res = []
    for seed, pts in sorted(mg[e].items()):
        pts = sorted((s, t) for s, t in pts if np.isfinite(t) and t > 0)
        if len(pts) >= 5 and pts[-1][0] / pts[0][0] >= 9.9:
            s, t = zip(*pts)
            res.append(fits(s, t))
    if not res:
        continue
    nlog = sum(x["rms_log"] < x["rms_pow"] for x in res)
    nlog_rel = sum(x["rms_log"] < x["rms_pow_rel"] for x in res)
    print(
        f"  edge {e:2d}: seeds {len(res)} prefer log {nlog}/{len(res)} (vs rel-power: {nlog_rel}) "
        f"p {np.mean([x['p'] for x in res]):.2f} log slope {np.mean([x['log_b'] for x in res]):.1f} "
        f"+- {np.std([x['log_b'] for x in res], ddof=1) / np.sqrt(len(res)) if len(res) > 1 else np.nan:.1f} "
        f"rms log {np.mean([x['rms_log'] for x in res]):.3f} pow {np.mean([x['rms_pow'] for x in res]):.3f}"
    )
    if e != 5:
        tot += len(res)
        tot_log += nlog
print(f"  later edges (not 5): {tot_log}/{tot} seed fits prefer log")
