"""Choose the targets of experiment route_nonstar and write configs/targets.yaml.

Hand-picked families (brief section 4.3): the smallest recombinant family on a path of 3 and 4
nodes, the nested chain on mps4 / binary4 / ht4, TD-like targets with two separate local edits,
and the star-satisfying control B4. Random d = 2 targets on mps6 and binary8 come from a seeded
generator: the first three that violate the star condition and the first one that satisfies it,
among those whose route depths along the paper's greedy sequence (from the largest weight) and
whose numerical route-rule sequence (experiments/locality/rules.route_sequence) stay <= 5 and whose
teacher bond ranks are <= 3 (a usable sigma window above the float64 floor, small bonds).

usage: uv run python experiments/route_nonstar/make_targets.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "locality"))
from rules import edge_rank, route_sequence  # noqa: E402

from ttn.routes import RouteTarget  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

GRID_SMALL = [round(-1.0 - 0.25 * k, 2) for k in range(13)]  # log10 sigma: -1 ... -4 (K <= 4)
GRID_LARGE = [round(-1.0 - 0.25 * k, 2) for k in range(11)]  # -1 ... -3.5 (K = 6, 7)

HAND = [
    # name, family, kind, n, patterns, weights
    ("R3", "recombinant", "mps", 3, ["000", "100", "001"], [1.0, 0.6, 0.4]),
    ("R3p", "recombinant", "mps", 3, ["000", "100", "001", "101"], [1.0, 0.6, 0.4, 0.24]),
    ("R3d", "recombinant", "mps", 3, ["000", "100", "001", "101"], [1.0, 0.6, 0.4, 0.5]),
    ("R4", "recombinant", "mps", 4, ["0000", "1000", "0001"], [1.0, 0.6, 0.4]),
    ("R4p", "recombinant", "mps", 4, ["0000", "1000", "0001", "1001"], [1.0, 0.6, 0.4, 0.24]),
    ("R4d", "recombinant", "mps", 4, ["0000", "1000", "0001", "1001"], [1.0, 0.6, 0.4, 0.5]),
    ("NC_mps4", "nested", "mps", 4, ["0000", "1100", "1110", "1111"], [1.0, 0.6, 0.35, 0.2]),
    ("NC_binary4", "nested", "binary", 4, ["0000", "1100", "1110", "1111"],
     [1.0, 0.6, 0.35, 0.2]),
    ("NC_ht4", "nested", "ht", 4, ["0000", "1100", "1110", "1111"], [1.0, 0.6, 0.35, 0.2]),
    ("TD_mps6", "two_edits", "mps", 6, ["000000", "000110", "110000"], [1.0, 0.6, 0.35]),
    ("TD_binary8", "two_edits", "binary", 8, ["00000000", "11000000", "00001100"],
     [1.0, 0.6, 0.35]),
    ("B4_mps4", "control", "mps", 4, ["0000", "0110", "1001"], [1.0, 0.3, 0.6]),
]


def describe(kind, n, patterns, weights) -> dict:
    tree = make_tree(kind, n)
    pats = [[int(c) for c in p] for p in patterns]
    target = RouteTarget(tree, pats, weights)
    ranks = {e: edge_rank(tree, pats, weights, list(range(len(pats))), e) for e in tree.edges}
    greedy = target.greedy_sequence()
    seq = route_sequence(tree, pats, weights, d=2, chi_max=4)
    return dict(
        K=tree.n_nodes,
        star=target.is_star,
        star_violations=[[int(e), [patterns[a] for a in m]] for e, m in target.star_violations],
        ranks={int(e): int(r) for e, r in ranks.items()},
        greedy=[dict(learned_before=[patterns[a] for a in s["learned_before"]],
                     next=patterns[s["component"]], depth=int(s["depth"]),
                     pairwise=int(s["pairwise"]), tie=bool(s["tie"])) for s in greedy],
        route_sequence=[dict(loss=round(float(s["loss"]), 6), k_star=int(s["k_star"]))
                        for s in seq],
    )


def random_targets(kind, n, want_nonstar=3, want_star=1, seed=0):
    rng = random.Random(f"{kind}{n}-{seed}")
    nonstar, star = [], []
    tried = 0
    while len(nonstar) < want_nonstar or len(star) < want_star:
        tried += 1
        m = rng.choice([3, 4, 5])
        pats = set()
        while len(pats) < m:
            pats.add("".join(rng.choice("01") for _ in range(n)))
        pats = sorted(pats)
        rng.shuffle(pats)
        weights = [1.0] + sorted((round(rng.uniform(0.3, 0.9), 2) for _ in range(m - 1)),
                                 reverse=True)
        if len(set(weights)) < m:
            continue
        tree = make_tree(kind, n)
        target = RouteTarget(tree, [[int(c) for c in p] for p in pats], weights)
        if not target.is_star and len(nonstar) >= want_nonstar:
            continue
        if target.is_star and len(star) >= want_star:
            continue
        greedy = target.greedy_sequence()
        if max(s["depth"] for s in greedy) > 5:
            continue
        ranks = [edge_rank(tree, [[int(c) for c in p] for p in pats], weights, list(range(m)), e)
                 for e in tree.edges]
        if max(ranks) > 3:
            continue
        info = describe(kind, n, pats, weights)
        if max(s["k_star"] for s in info["route_sequence"]) > 5:
            continue
        (nonstar if not target.is_star else star).append((pats, weights, info))
    return nonstar, star, tried


def main():
    out = []
    for name, family, kind, n, pats, w in HAND:
        info = describe(kind, n, pats, w)
        out.append(dict(name=name, family=family, kind=kind, n=n, d=2, patterns=pats, weights=w,
                        **info))
    for kind, n in [("mps", 6), ("binary", 8)]:
        nonstar, star, tried = random_targets(kind, n)
        print(f"{kind}{n}: {tried} random draws", flush=True)
        for i, (pats, w, info) in enumerate(nonstar):
            out.append(dict(name=f"RND{i + 1}_{kind}{n}", family="random", kind=kind, n=n, d=2,
                            patterns=pats, weights=w, **info))
        for i, (pats, w, info) in enumerate(star):
            out.append(dict(name=f"CTL{i + 1}_{kind}{n}", family="control", kind=kind, n=n, d=2,
                            patterns=pats, weights=w, **info))
    for t in out:
        t["log10_sigmas"] = GRID_SMALL if t["K"] <= 4 else GRID_LARGE
        print(f"{t['name']:14s} K={t['K']} star={t['star']} ranks={t['ranks']} "
              f"greedy depths={[s['depth'] for s in t['greedy']]} "
              f"route-rule seq={[(s['loss'], s['k_star']) for s in t['route_sequence']]}")
        print(f"{'':14s} patterns={t['patterns']} weights={t['weights']}")
        print(f"{'':14s} violations={t['star_violations']}")
    (HERE / "configs").mkdir(exist_ok=True)
    cfg = dict(seeds=list(range(40)), spare=2, rtol=1e-10, atol=1e-14, t_max=1e22, wall=1800,
               targets=out)
    (HERE / "configs" / "targets.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False,
                                                                  width=110))


if __name__ == "__main__":
    np.random.seed(0)
    main()
