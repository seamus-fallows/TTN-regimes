"""Two-component cases for the exponent table: small-init (brief protocol, where the first plateau
is affordable) and saddle-start (exact balanced saddle s1 U1 + sigma-scale noise, every tree)."""
import json
import sys

import numpy as np

sys.path.insert(0, "experiments/locality")
from rules import get_tree, steiner_k  # noqa: E402

SETS = {
    ("mps", 4): [[0, 1], [1, 2], [0, 2], [0, 1, 2], [1, 2, 3], [0, 3], [0, 1, 2, 3]],
    ("binary", 4): [[0, 1], [2, 3], [1, 2], [0, 3], [0, 2], [0, 1, 2], [0, 1, 2, 3]],
    ("ht", 4): [[0, 1], [2, 3], [1, 2], [0, 3], [0, 2], [0, 1, 2], [0, 1, 2, 3]],
    ("mps", 6): [[0, 1], [2, 3], [0, 2], [2, 3, 4], [0, 3], [1, 4], [1, 3, 5], [0, 5], list(range(6))],
    ("mps", 8): [[0, 1], [3, 4], [2, 3, 4], [2, 5], [1, 5], [1, 6], [0, 7]],
    ("binary", 8): [[0, 1], [1, 2], [0, 2], [0, 1, 2, 3], [4, 5, 6], [3, 4], [0, 7], [2, 3, 4, 5],
                    [0, 2, 4]],
    ("ht", 8): [[0, 1], [6, 7], [1, 2], [0, 1, 2, 3], [3, 4]],
    ("star3", None): [[0, 1], [1, 2], [0, 5], [0, 2, 4], list(range(6))],
    ("midroot6", None): [[2, 3], [0, 2], [0, 3], [0, 5]],
}
# sigma range by predicted k: escape ~ sigma^-(k-2) must stay affordable (< ~5e4 time units)
RANGE_SMALL = {1: (1e-3, 0.1), 2: (1e-3, 0.1), 3: (1e-3, 0.1), 4: (0.03, 0.3), 5: (0.03, 0.3),
               6: (0.07, 0.6), 7: (0.12, 0.6), 8: (0.17, 0.6)}
RANGE_SADDLE = {1: (1e-3, 0.1), 2: (1e-4, 0.1), 3: (1e-3, 0.1), 4: (0.01, 0.1), 5: (0.03, 0.3),
                6: (0.06, 0.4), 7: (0.1, 0.5), 8: (0.15, 0.5)}
T1_MAX = 1e11  # first plateau ~ 3 sigma^-(K-2) must stay below this (time resolution, cost)

cases = []
for (kind, n), sets in SETS.items():
    tree = get_tree(kind, n)
    K = tree.n_nodes
    for S in sets:
        k = steiner_k(tree, S)
        if k > 8:
            continue
        for start, rng in (("small", RANGE_SMALL), ("saddle", RANGE_SADDLE)):
            lo, hi = rng[k]
            if start == "small":
                if kind == "ht" and n == 8:
                    continue
                lo = max(lo, (T1_MAX / 3) ** (-1 / (K - 2)))
                if hi / lo < 3:
                    continue
            n_sig = 7 if hi / lo > 20 else 6
            sig = [float(f"{x:.4g}") for x in np.logspace(np.log10(hi), np.log10(lo), n_sig)]
            cases.append(dict(kind=kind, n=n, S=S, start=start, sigmas=sig, seeds=list(range(8)),
                              weights=[1.0, 0.5], cost=k + (K if start == "small" else 0), k=k, K=K))
json.dump(cases, open(sys.argv[1], "w"), indent=1)
for c in cases:
    print(c["kind"], c["n"], c["S"], c["start"], "K", c["K"], "k", c["k"], c["sigmas"][-1], c["sigmas"][0])
print(len(cases), "cases,", sum(len(c["seeds"]) for c in cases), "tasks")
