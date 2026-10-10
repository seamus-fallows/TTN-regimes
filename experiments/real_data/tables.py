"""Tables for the real-data report from summarize.py output (markdown to stdout and a JSON file).

Usage: uv run python experiments/real_data/tables.py <summary.json> <out_prefix> [routes.json]

Tables (per tree = binary quadtree / binary scrambled (orders pooled, also per order) / MPS snake,
and per feature map):
  S2S       per sigma: runs, fraction with the S2S signature (>= 2 drops, all rho > 3), mean drops
  first     first rank increase after the escape: fraction on exactly two sibling bottom edges
            (binary; the k = 3 route) or on a single edge (MPS; k = 2)
  order     Kendall tau between edge level and the time of its first s_2/s_1 >= 0.05 (bottom-up > 0)
  locality  per sigma: tau to reach train loss 0.25 / 0.2 / 0.15 / 0.12 (median, with the number of runs
            that reach it), train / test loss and test accuracy at tau = 1e4 (mean +- SE)
  routes    (if routes.json): fraction of binary runs whose first sibling pair to grow is the pair with
            the largest route amplitude at the rank-1 plateau
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau


def tree_label(r, pool_scrambled=True):
    if r["tree"] == "mps":
        return "MPS snake"
    if r["ordering"] == "quadtree":
        return "binary quadtree"
    return "binary scrambled" if pool_scrambled else f"binary {r['ordering']}"


def mse(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.3g}"
    return f"{v.mean():.3g} ± {v.std(ddof=1) / np.sqrt(len(v)):.2g}"


def sibling_pair(r) -> bool:
    edges = r.get("first_rank2_edges") or []
    if len(edges) != 2:
        return False
    lv = r["edge_levels"]
    return all(lv[e] == 1 for e in edges)  # two bottom edges rising together


def first_two_siblings(r) -> bool:
    """Binary trees: the first two edges to gain a 2nd singular value are both bottom edges with the same
    parent (one k = 3 route), whatever the gap between their threshold crossings."""
    ev = np.array(r["rank_events"], float)[:, 0]
    ok = np.flatnonzero(np.isfinite(ev))
    if len(ok) < 2:
        return False
    a, b = ok[np.argsort(ev[ok])[:2]]
    lv, edges, parent = r["edge_levels"], r["edges"], r["parent"]
    return lv[a] == 1 and lv[b] == 1 and parent[edges[a]] == parent[edges[b]]


def main(summary_json, out_prefix, routes_json=None):
    rows = [r for r in json.load(open(summary_json)) if "error" not in r and not r.get("centred")
            and "drops" in r]
    out, md = {}, []
    for feats in sorted({r["features"] for r in rows}):
        sub = [r for r in rows if r["features"] == feats]
        md.append(f"\n### features = {feats}\n")
        # S2S and drops per sigma
        md.append("| tree | sigma | runs | S2S (rho > 3) | rho > 2 | rho > 5 | drops (mean) | min rho (median) |\n|---|---|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for r in sub:
            groups[(tree_label(r), r["sigma"])].append(r)
        for (tree, s), rs in sorted(groups.items()):
            s2s = np.mean([r["s2s"] for r in rs])
            minrho = [min(r["rho"]) if r["rho"] else np.nan for r in rs]
            sens = [np.mean([r["n_drops"] >= 2 and min(r["rho"]) > th for r in rs]) for th in (2.0, 5.0)]
            md.append(f"| {tree} | {s:g} | {len(rs)} | {s2s:.2f} | {sens[0]:.2f} | {sens[1]:.2f} | "
                      f"{np.mean([r['n_drops'] for r in rs]):.1f} | {np.nanmedian(minrho):.2g} |")
            out.setdefault(feats, {}).setdefault("s2s", {})[f"{tree}|{s:g}"] = {
                "runs": len(rs), "s2s_fraction": float(s2s),
                "drops_mean": float(np.mean([r["n_drops"] for r in rs]))}
        # first rank increase and bottom-up order
        md.append("\n| tree | runs | first rise = sibling bottom pair | first two rising edges = siblings | = single edge | Kendall tau(level, time) |\n|---|---|---|---|---|---|")
        by_tree = defaultdict(list)
        for r in sub:
            by_tree[tree_label(r)].append(r)
        for tree, rs in sorted(by_tree.items()):
            sib = np.mean([sibling_pair(r) for r in rs])
            single = np.mean([len(r.get("first_rank2_edges") or []) == 1 for r in rs])
            taus = []
            for r in rs:
                ev = np.array(r["rank_events"], float)[:, 0]
                lv = np.array(r["edge_levels"])
                ok = np.isfinite(ev)
                if ok.sum() >= 4:
                    taus.append(kendalltau(lv[ok], ev[ok]).statistic)
            two = np.mean([first_two_siblings(r) for r in rs]) if rs[0]["tree"] == "binary" else np.nan
            md.append(f"| {tree} | {len(rs)} | {sib:.2f} | {two:.2f} | {single:.2f} | {mse(taus)} |")
            out[feats].setdefault("first", {})[tree] = {"sibling": float(sib), "single": float(single),
                                                       "kendall": mse(taus), "n": len(rs)}
        # locality
        levels = (0.25, 0.2, 0.15, 0.12)
        md.append("\n| tree | sigma | " + " | ".join(f"tau(L<={lv})" for lv in levels)
                  + " | train L(1e4) | test L(1e4) | test acc |\n|---|---|" + "---|" * (len(levels) + 3))
        for (tree, s), rs in sorted(groups.items()):
            cells = []
            for lv in levels:
                ts = [r[f"t_loss_{lv}"] for r in rs if np.isfinite(r.get(f"t_loss_{lv}", np.nan))]
                cells.append(f"{np.median(ts):.3g} ({len(ts)}/{len(rs)})" if ts else f"- (0/{len(rs)})")
            md.append(f"| {tree} | {s:g} | " + " | ".join(cells)
                      + f" | {mse([r['loss_end'] for r in rs])} | {mse([r['test_loss_end'] for r in rs])}"
                      f" | {mse([r['test_acc_end'] for r in rs])} |")
    if routes_json:
        res = [r for r in json.load(open(routes_json)) if "pairs" in r]
        hits = defaultdict(list)
        for r in res:
            sib = [p for p in r["pairs"] if p["k"] == 3]
            best = max(sib, key=lambda p: p["lam"])
            edges = r["edges"]
            pred = sorted([edges.index(best["u"]), edges.index(best["w"])])
            obs = sorted(r.get("first_rank2_edges") or [])
            hits[(r["ordering"], r["features"])].append(pred == obs)
        md.append("\n| ordering | features | runs | first sibling pair = argmax lambda |\n|---|---|---|---|")
        for k, v in sorted(hits.items()):
            md.append(f"| {k[0]} | {k[1]} | {len(v)} | {np.mean(v):.2f} ({sum(v)}/{len(v)}) |")
            out.setdefault("routes", {})[f"{k[0]}|{k[1]}"] = {"n": len(v), "hit": float(np.mean(v))}
        # share of the squared route amplitude by route degree (where the residual interaction sits)
        share = defaultdict(lambda: defaultdict(list))
        for r in res:
            lam = np.array([p_["lam"] for p_ in r["pairs"]])
            kk = np.array([p_["k"] for p_ in r["pairs"]])
            key = "quadtree" if r["ordering"] == "quadtree" else "scrambled (3 orders)"
            for k in sorted(set(kk)):
                share[key][int(k)].append(float((lam[kk == k] ** 2).sum() / (lam ** 2).sum()))
        md.append("\n| tree order | runs | share of sum lambda^2 on k = 3 | k = 5 | k = 7 |\n|---|---|---|---|---|")
        for key, d in sorted(share.items()):
            md.append(f"| {key} | {len(d[3])} | " + " | ".join(mse(d[k]) for k in (3, 5, 7)) + " |")
            out.setdefault("route_share", {})[key] = {k: mse(v) for k, v in d.items()}
    text = "\n".join(md)
    print(text)
    open(f"{out_prefix}.md", "w").write(text + "\n")
    json.dump(out, open(f"{out_prefix}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
