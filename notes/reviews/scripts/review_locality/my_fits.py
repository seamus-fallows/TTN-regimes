"""Reviewer's own fits from the runner's JSON records (independent of analysis.py / make_figures.py).

Per (tree, S, start): run classes; per seed (>= 3 sigmas) the slope of log(t2 - t1) vs log(1/sigma)
(all sigmas and the 3 smallest), the k = 2 slope of (t2 - t1) vs ln(1/sigma), and the slope of
log(-lambda_min) vs log(sigma) at the recorded plateau midpoints; mean +- SE over seeds.
Two pools: 'all0>1' (every run whose final-crossing order is 0>1, as the agent's table) and 'clean'
(greedy runs with exactly one 1/2-crossing per component)."""

import glob
import json
import sys

import numpy as np

recs = {}
for f in sorted(glob.glob(sys.argv[1] + ".part*")):
    for line in open(f):
        r = json.loads(line)
        recs[r["_key"]] = r
recs = list(recs.values())


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.polyfit(x, y, 1)[0]


def mse(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) == 0:
        return "-"
    return f"{v.mean():.3f} +- {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan'):.3f} (n={len(v)}, med {np.median(v):.3f})"


groups = {}
for r in recs:
    if r.get("S") is None:
        continue
    groups.setdefault((r["kind"] + str(r["n"]), "".join(map(str, r["S"])), r["start"]), []).append(
        r
    )

for key in sorted(groups):
    rs = groups[key]
    cls = {}
    for r in rs:
        st = r.get("status")
        if st != 1:
            c = str(st)
        else:
            n05 = [len(r["crossings"][f"a{i}_0.5"]) for i in range(2)]
            order = [x[1] for x in r["events"]]
            c = (
                "rotation"
                if max(n05) > 1
                else ("greedy" if order == [0, 1] else "reversed" if order == [1, 0] else "other")
            )
        r["_cls"] = c
        cls[c] = cls.get(c, 0) + 1
    print(key, "classes:", cls)
    for pool in ("all0>1", "clean"):
        p_all, p_loc, ls, hx, hmin_small = [], [], [], [], []
        by_seed = {}
        for r in rs:
            if r.get("status") != 1 or [x[1] for x in r["events"]] != [0, 1]:
                continue
            if pool == "clean" and r["_cls"] != "greedy":
                continue
            by_seed.setdefault(r["seed"], []).append(r)
        for seed, g in by_seed.items():
            g = sorted(g, key=lambda r: r["sigma"])
            sig = np.array([r["sigma"] for r in g])
            gap = np.array([r["events"][1][0] - r["events"][0][0] for r in g])
            hm = np.array([r["plateaus"][0].get("hess_min", np.nan) for r in g])
            if len(sig) >= 3 and np.all(gap > 0):
                x = np.log(1 / sig)
                p_all.append(slope(x, np.log(gap)))
                p_loc.append(slope(x[:3], np.log(gap[:3])))
                ls.append(slope(x, gap))
            ok = np.isfinite(hm) & (hm < 0)
            if ok.sum() >= 3:
                hx.append(slope(np.log(sig[ok]), np.log(-hm[ok])))
            if np.isfinite(hm[0]):
                hmin_small.append(hm[0])
        print(
            f"  [{pool}] seeds={len(by_seed)}  p={mse(p_all)}  p_loc3={mse(p_loc)}  logslope={mse(ls)}  "
            f"hess_exp={mse(hx)}  lam_min(smallest sigma)={mse(hmin_small)}"
        )
    if key[2] == "small" or True:
        gaps_small = [
            r["events"][1][0] - r["events"][0][0]
            for r in rs
            if r.get("status") == 1
            and len(r["events"]) == 2
            and r["sigma"] == min(x["sigma"] for x in rs)
        ]
        if gaps_small:
            print(f"  median |t2-t1| at smallest sigma: {np.median(np.abs(gaps_small)):.4g}")
    stalled_seeds = sorted({r["seed"] for r in rs if r.get("status") == "stalled"})
    if stalled_seeds:
        print(
            "  stalled seeds:",
            stalled_seeds,
            "stalled runs:",
            sum(r.get("status") == "stalled" for r in rs),
        )
