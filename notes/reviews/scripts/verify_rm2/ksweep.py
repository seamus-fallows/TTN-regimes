"""Independent K-sweep walk (centred, chi = 4, d = 2): label-rule brackets, then floor-stalled runs by min rho."""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out")
LOCAL = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/regime_map_local/sigma_K_centred_d2/results.jsonl"


def ratios(times, widths):
    t, w = np.asarray(times, float), np.asarray(widths, float)
    k = np.isfinite(t)
    o = np.argsort(t[k], kind="stable")
    t, w = t[k][o], w[k][o]
    if len(t) < 2:
        return np.array([])
    return np.diff(t) / np.maximum(w[:-1], w[1:])


def lab(l_end, dK, rh):
    if not np.isfinite(l_end) or l_end >= 1e-3:
        return "stalled"
    if dK < 0.1:
        return "lazy"
    return "s2s" if rh.size and np.all(rh > 3) else "rich"


r = pd.read_csv(os.path.join(OUTD, "relabel_sigma_K_centred_d2.csv"))
r = r[r.chi == 4].copy()
r["src"] = "remote"
r["times"] = [json.loads(x) for x in r.times_new]
r["wid"] = [json.loads(x) for x in r.widths]
loc = [json.loads(line) for line in open(LOCAL)]
add = []
have = {(a, b, c, round(float(s), 4)) for a, b, c, s in zip(r.kind, r.n, r.seed, r.sigma)}
for q in loc:
    if q.get("chi") != 4 or (q["kind"], q["n"], q["seed"], round(float(q["sigma"]), 4)) in have:
        continue
    add.append(
        dict(
            kind=q["kind"],
            n=q["n"],
            K=q["K"],
            seed=q["seed"],
            sigma=q["sigma"],
            l_end=q["l_end"],
            dK=q["dK"],
            times=[np.nan if x is None else x for x in q["event_times"]],
            wid=[np.nan if x is None else x for x in q["event_widths"]],
            src="local:" + str(q.get("event_rule")),
        )
    )
r = pd.concat([r, pd.DataFrame(add)], ignore_index=True)
print(
    "local rows added:",
    len(add),
    pd.DataFrame(add)[["kind", "n", "seed", "sigma", "src"]].to_string() if add else "",
)
r["rh"] = [ratios(t, w) for t, w in zip(r.times, r.wid)]
r["minrho"] = [float(np.min(x)) if x.size else np.nan for x in r.rh]
r["nev"] = [int(np.sum(np.isfinite(np.asarray(t, float)))) for t in r.times]
r["lab"] = [lab(a, b, c) for a, b, c in zip(r.l_end, r.dK, r.rh)]

out = []
for (kind, n, seed), g in r.groupby(["kind", "n", "seed"]):
    g = g.sort_values("sigma")
    sig, lb = g.sigma.values, g.lab.values
    K = int(g.K.iloc[0])
    grid = np.sort(np.unique(sig))
    ratio = float(np.median(grid[1:] / grid[:-1]))
    s = np.flatnonzero(lb == "s2s")
    if s.size == 0:
        out.append(dict(tree=f"{kind}{n}", K=K, seed=seed, status="never"))
        continue
    e = s[0]
    while e + 1 < len(lb) and lb[e + 1] == "s2s":
        e += 1
    if e == len(lb) - 1:
        out.append(dict(tree=f"{kind}{n}", K=K, seed=seed, status="all_s2s", lo=sig[e]))
        continue
    lo, hi = sig[e], sig[e + 1]
    if lb[e + 1] != "stalled":
        out.append(
            dict(
                tree=f"{kind}{n}",
                K=K,
                seed=seed,
                status="label",
                lo=lo,
                hi=hi,
                sc=np.sqrt(lo * hi),
                first_above_rho=np.nan,
            )
        )
        continue
    status, hi, first = "open", np.nan, None
    for _, q in g[g.sigma > lo * (1 + 1e-9)].iterrows():
        if first is None:
            first = q.minrho
        if np.isfinite(q.minrho) and q.minrho <= 3:
            hi, status = q.sigma, "rho"
            break
        if not np.isfinite(q.minrho) and q.dK < 0.1:
            hi, status = q.sigma, "kernel-like stall"
            break
        if not np.isfinite(q.minrho) and q.nev == 0:
            hi, status = q.sigma, "no-learning stall"
            break
        if np.isfinite(q.minrho):
            lo = q.sigma
    steps = np.log(hi / lo) / np.log(ratio) if np.isfinite(hi) else np.nan
    if np.isfinite(steps) and steps > 1.5:
        status = "wide"
    out.append(
        dict(
            tree=f"{kind}{n}",
            K=K,
            seed=seed,
            status=status,
            lo=lo,
            hi=hi,
            sc=np.sqrt(lo * hi) if np.isfinite(hi) else np.nan,
            steps=steps,
            first_above_rho=first,
        )
    )
t = pd.DataFrame(out)
t.to_csv(os.path.join(OUTD, "ksweep_mine.csv"), index=False)
RES = ("label", "rho", "kernel-like stall", "no-learning stall")
for tree, g in t.groupby("tree", sort=False):
    res = g[g.status.isin(RES)]
    strict = g[g.status.isin(("label", "rho"))]
    print(
        f"{tree:9s} K={g.K.iloc[0]:2d} statuses {g.status.value_counts().to_dict()} | median(resolved) "
        f"{res.sc.median():.4f} (n={len(res)}) values {np.round(np.sort(res.sc.values), 3).tolist()} | "
        f"label/rho only: {strict.sc.median():.4f} (n={len(strict)})"
    )
un = t[~t.status.isin(("label", "all_s2s", "never"))]
mps = un[un.tree.str.startswith("mps") & un.K.between(6, 12)]
print(
    "unresolved MPS seeds K=6-12:",
    len(mps),
    "first run above S2S-like (rho>3):",
    int((mps.first_above_rho > 3).sum()),
)
