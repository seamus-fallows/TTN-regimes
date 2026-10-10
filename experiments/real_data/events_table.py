"""Per-event route degrees and gap scaling (summarize.py output -> JSON + markdown).

Usage: uv run python experiments/real_data/events_table.py <summary.json> <out_prefix>

Each loss drop after the first escape gets a signature: the set of (edge, singular-value index) whose
s_k/s_1 crosses 0.05 inside the drop's window (summarize.py, `drop_edges`). Its route is the subgraph
of the tree spanned by those edges; for a connected edge set the route degree (number of cores) is
k = #edges + 1 (theory_saddles.md R1-R2: the edges whose rank must grow are the Steiner edges of the route).
For each seed, a signature that occurs at >= 4 sigma values (spanning >= 0.5 decade) gives a gap series
Delta t(sigma) = t(drop) - t(previous drop). Prediction: Delta t ∝ sigma^-(k-2) for k >= 3, linear in
log(1/sigma) for k = 2. Per series we fit the power exponent and the log slope (same space: linear
Delta t, relative residuals) and report both residuals.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def route_degree(edge_nodes, parent):
    """(k, connected): number of cores spanned by the edges (child nodes) and whether they connect."""
    nodes, adj = set(), defaultdict(set)
    for v in edge_nodes:
        p = parent[v]
        nodes |= {v, p}
        adj[v].add(p)
        adj[p].add(v)
    if not nodes:
        return 0, False
    start = next(iter(nodes))
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for w in adj[u] - seen:
            seen.add(w)
            stack.append(w)
    return len(nodes), seen == nodes


def fit(sig, dt):
    x = np.log(1 / np.asarray(sig))
    dt = np.asarray(dt)
    a, p = np.polyfit(x, np.log(dt), 1)[::-1]
    pow_rms = np.sqrt(np.mean((np.exp(a + p * x) / dt - 1) ** 2))
    b, a2 = np.polyfit(x, dt, 1)
    log_rms = np.sqrt(np.mean(((a2 + b * x) / dt - 1) ** 2))
    return float(p), float(pow_rms), float(b), float(log_rms)


def main(summary_json, out_prefix):
    rows = [r for r in json.load(open(summary_json)) if "drops" in r and not r.get("centred")]
    series = defaultdict(list)
    for r in rows:
        tree = r["tree"] + ":" + r["ordering"]
        edges, parent = r["edges"], r["parent"]
        for i in range(1, len(r["drops"])):
            sigset = r["drop_edges"][i]
            if not sigset:
                continue
            nodes = sorted({edges[e] for e, k in sigset})
            k, conn = route_degree(nodes, parent)
            key = (tree, r["features"], r["seed"], tuple(sorted(map(tuple, sigset))))
            series[key].append({"sigma": r["sigma"], "dt": r["drops"][i]["t"] - r["drops"][i - 1]["t"],
                                "k": k, "connected": conn, "index": i, "n_edges": len(nodes)})
    out, md = [], ["| tree | features | seed | drop # | edges (edge, k-th value) | route k | n sigma | sigma range | power exponent (rms) | log slope (rms) | predicted |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (tree, feats, seed, sig), pts in sorted(series.items()):
        sigmas = sorted({p["sigma"] for p in pts})
        if len(pts) != len(sigmas) or len(sigmas) < 4 or max(sigmas) / min(sigmas) < 10 ** 0.5:
            continue
        pts.sort(key=lambda p: p["sigma"])
        if any(p["dt"] <= 0 for p in pts):
            continue
        p_exp, p_rms, l_slope, l_rms = fit([p["sigma"] for p in pts], [p["dt"] for p in pts])
        k, conn = pts[0]["k"], pts[0]["connected"]
        pred = ("log" if k == 2 else f"p = {k - 2}") if conn else "several routes"
        d = {"tree": tree, "features": feats, "seed": seed, "signature": sig, "k": k, "connected": conn,
             "sigmas": sigmas, "dt": [p["dt"] for p in pts], "drop_index": [p["index"] for p in pts],
             "power_exponent": p_exp, "power_rms": p_rms, "log_slope": l_slope, "log_rms": l_rms}
        out.append(d)
        md.append(f"| {tree} | {feats} | {seed} | {sorted(set(d['drop_index']))} | {list(sig)} | {k}{'' if conn else ' (disc.)'} "
                  f"| {len(sigmas)} | {min(sigmas):g}-{max(sigmas):g} | {p_exp:.2f} ({p_rms:.2f}) | {l_slope:.3g} ({l_rms:.2f}) | {pred} |")
    json.dump(out, open(f"{out_prefix}.json", "w"), indent=1)
    open(f"{out_prefix}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:])
