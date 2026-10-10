"""Tables for the real-data report from summarize.py output (markdown to stdout and a JSON file).

Usage: uv run python experiments/real_data/tables.py <summary.json> <out_prefix> [routes.json]

Tables (per tree = binary quadtree / binary scrambled (orders pooled, also per order) / MPS snake,
and per feature map):
  S2S       per sigma: runs, fraction with the S2S signature (>= 2 drops, all rho > 3), mean drops
  first     first rank increase after the escape: fraction on exactly two sibling bottom edges
            (binary; the k = 3 route) or on a single edge (MPS; k = 2); runs whose first "rise" is the
            threshold already met at tau = 0 are counted separately (unresolved)
  order     Kendall tau between edge level and the time of its first s_2/s_1 >= 0.05 (bottom-up > 0;
            on the MPS the "level" is only the position along the chain)
  locality  per sigma: tau to reach train loss 0.25 / 0.2 / 0.15 / 0.12 (median, with the number of runs
            that reach it), train / test loss and test accuracy at the end of the run tau_end (mean +- SE;
            tau_end = t2 unless the wall limit stopped the run, counted in the last column)
  routes    (if routes.json): route-amplitude rule per (order, seed, critical point) unit on states near
            rank 1 (`analysis.route_units`), and the share of sum lambda^2 per route degree, one state per
            (order, seed)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
from analysis import ERR_MAX, route_units  # noqa: E402


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
    """The first rank increase is on exactly two bottom edges with the same parent, crossing together."""
    edges = r.get("first_rank2_edges") or []
    if len(edges) != 2:
        return False
    lv, nodes, parent = r["edge_levels"], r["edges"], r["parent"]
    return all(lv[e] == 1 for e in edges) and parent[nodes[edges[0]]] == parent[nodes[edges[1]]]


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
    rows = [
        r
        for r in json.load(open(summary_json))
        if "error" not in r and not r.get("centred") and "drops" in r
    ]
    out, md = {}, []
    for feats in sorted({r["features"] for r in rows}):
        sub = [r for r in rows if r["features"] == feats]
        md.append(f"\n### features = {feats}\n")
        # S2S and drops per sigma
        md.append(
            "| tree | sigma | runs | S2S (rho > 3) | rho > 2 | rho > 5 | drops (mean) | min rho (median) |\n|---|---|---|---|---|---|---|---|"
        )
        groups = defaultdict(list)
        for r in sub:
            groups[(tree_label(r), r["sigma"])].append(r)
        for (tree, s), rs in sorted(groups.items()):
            s2s = np.mean([r["s2s"] for r in rs])
            minrho = [min(r["rho"]) if r["rho"] else np.nan for r in rs]
            sens = [
                np.mean([r["n_drops"] >= 2 and min(r["rho"]) > th for r in rs]) for th in (2.0, 5.0)
            ]
            md.append(
                f"| {tree} | {s:g} | {len(rs)} | {s2s:.2f} | {sens[0]:.2f} | {sens[1]:.2f} | "
                f"{np.mean([r['n_drops'] for r in rs]):.1f} | {np.nanmedian(minrho):.2g} |"
            )
            out.setdefault(feats, {}).setdefault("s2s", {})[f"{tree}|{s:g}"] = {
                "runs": len(rs),
                "s2s_fraction": float(s2s),
                "drops_mean": float(np.mean([r["n_drops"] for r in rs])),
            }
        # first rank increase and bottom-up order
        md.append(
            "\n| tree | runs | first rise = sibling bottom pair | first two rising edges = siblings | = single edge"
            " | first rise at tau = 0 (unresolved) | single edge, resolved runs"
            " | Kendall tau(level, time) (MPS: level = position along the chain) |\n|---|---|---|---|---|---|---|---|"
        )
        by_tree = defaultdict(list)
        for r in sub:
            by_tree[tree_label(r)].append(r)
        for tree, rs in sorted(by_tree.items()):
            sib = np.mean([sibling_pair(r) for r in rs])
            single = np.mean([len(r.get("first_rank2_edges") or []) == 1 for r in rs])
            # rank "events" recorded at tau = 0 are the threshold, not a resolved increase (fix round 1)
            res_runs = [r for r in rs if not r.get("first_rank2_at_start")]
            n_single_res = sum(len(r.get("first_rank2_edges") or []) == 1 for r in res_runs)
            taus = []
            for r in rs:
                ev = np.array(r["rank_events"], float)[:, 0]
                lv = np.array(r["edge_levels"])
                ok = np.isfinite(ev)
                if ok.sum() >= 4:
                    taus.append(kendalltau(lv[ok], ev[ok]).statistic)
            two = (
                np.mean([first_two_siblings(r) for r in rs])
                if rs[0]["tree"] == "binary"
                else np.nan
            )
            md.append(
                f"| {tree} | {len(rs)} | {sib:.2f} | {two:.2f} | {single:.2f} | {len(rs) - len(res_runs)}/{len(rs)}"
                f" | {n_single_res}/{len(res_runs)} | {mse(taus)} |"
            )
            out[feats].setdefault("first", {})[tree] = {
                "sibling": float(sib),
                "single": float(single),
                "at_start": len(rs) - len(res_runs),
                "single_resolved": [n_single_res, len(res_runs)],
                "kendall": mse(taus),
                "n": len(rs),
            }
        # locality
        levels = (0.25, 0.2, 0.15, 0.12)
        # final values are at the end of each run, tau_end (= t2 unless the wall limit stopped it)
        md.append(
            "\n| tree | sigma | "
            + " | ".join(f"tau(L<={lv})" for lv in levels)
            + " | train L(tau_end) | test L(tau_end) | test acc | tau_end: median (runs stopped early) |\n|---|---|"
            + "---|" * (len(levels) + 4)
        )
        for (tree, s), rs in sorted(groups.items()):
            cells = []
            for lv in levels:
                ts = [r[f"t_loss_{lv}"] for r in rs if np.isfinite(r.get(f"t_loss_{lv}", np.nan))]
                cells.append(
                    f"{np.median(ts):.3g} ({len(ts)}/{len(rs)})" if ts else f"- (0/{len(rs)})"
                )
            early = sum(r.get("stop") != "completed" for r in rs)
            md.append(
                f"| {tree} | {s:g} | "
                + " | ".join(cells)
                + f" | {mse([r['loss_end'] for r in rs])} | {mse([r['test_loss_end'] for r in rs])}"
                f" | {mse([r['test_acc_end'] for r in rs])} | {np.median([r['t2'] for r in rs]):.4g} ({early}) |"
            )
    if routes_json:
        # fix round 1: only states near rank 1 (the amplitude formula's premise), and runs of the same
        # (order, seed) that sit at the same critical point counted once (`analysis.route_units`)
        states = json.load(open(routes_json))
        ru = route_units(states)
        hits = defaultdict(lambda: defaultdict(int))
        for u in ru["units"]:
            hits[(u["ordering"], u["features"])][u["outcome"]] += 1
        md.append(
            f"\n{ru['n_states']['dropped']} of {ru['n_states']['all']} states dropped (rank1_rel_err > {ERR_MAX});"
            " units = (order, seed, critical point); chance per unit 0.25"
            "\n\n| ordering | features | units | first sibling pair = argmax lambda in all states | mixed | none |"
            "\n|---|---|---|---|---|---|"
        )
        for k, v in sorted(hits.items()):
            n = sum(v.values())
            md.append(f"| {k[0]} | {k[1]} | {n} | {v['hit']} | {v['mixed']} | {v['miss']} |")
            out.setdefault("routes", {})[f"{k[0]}|{k[1]}"] = {"units": n, **dict(v)}
        # share of the squared route amplitude by route degree, one state per (order, seed): the
        # smallest sigma (closest to rank 1); pooling over sigma would repeat the same critical point
        first = {}
        for r in sorted(
            (r for r in states if "pairs" in r), key=lambda r: r["sigma"], reverse=True
        ):
            first[(r["ordering"], r["features"], r["seed"])] = r
        share = defaultdict(lambda: defaultdict(list))
        for r in first.values():
            lam = np.array([p_["lam"] for p_ in r["pairs"]])
            kk = np.array([p_["k"] for p_ in r["pairs"]])
            key = "quadtree" if r["ordering"] == "quadtree" else "scrambled (pooled orders)"
            for k in sorted(set(kk)):
                share[key][int(k)].append(float((lam[kk == k] ** 2).sum() / (lam**2).sum()))
        md.append(
            "\n| tree order | (order, seed) pairs | share of sum lambda^2 on k = 3 | k = 5 | k = 7 |\n|---|---|---|---|---|"
        )
        for key, d in sorted(share.items()):
            md.append(f"| {key} | {len(d[3])} | " + " | ".join(mse(d[k]) for k in (3, 5, 7)) + " |")
            out.setdefault("route_share", {})[key] = {k: mse(v) for k, v in d.items()}
    text = "\n".join(md)
    print(text)
    open(f"{out_prefix}.md", "w", encoding="utf-8").write(text + "\n")
    json.dump(out, open(f"{out_prefix}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
