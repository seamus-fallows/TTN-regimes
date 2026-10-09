"""Case list for the step-1 exponent table beyond MPS/binary N = 4 (pilot integrator)."""
import json
import sys

import numpy as np

sys.path.insert(0, "experiments/locality")
from rules import get_tree, steiner_k  # noqa: E402

SIGMA_MIN = {1: 1e-3, 2: 1e-3, 3: 1e-3, 4: 0.03, 5: 0.03, 6: 0.07, 7: 0.12, 8: 0.17}
SETS = {
    ("ht", 4): [[0, 1], [2, 3], [1, 2], [0, 3], [0, 2], [0, 1, 2], [0, 1, 2, 3]],
    ("mps", 6): [[0, 1], [2, 3], [0, 2], [2, 3, 4], [0, 3], [1, 4], [1, 3, 5], [0, 5], list(range(6))],
    ("mps", 8): [[0, 1], [3, 4], [2, 3, 4], [2, 5], [1, 5], [1, 6], [0, 7]],
    ("binary", 8): [[0, 1], [1, 2], [0, 2], [0, 1, 2, 3], [4, 5, 6], [3, 4], [0, 7], [2, 3, 4, 5],
                    [0, 2, 4]],
    ("ht", 8): [[0, 1], [6, 7], [1, 2], [0, 1, 2, 3]],
    ("star3", None): [[0, 1], [1, 2], [0, 5], [0, 2, 4], list(range(6))],
    ("midroot6", None): [[2, 3], [0, 2], [0, 3], [0, 5]],
}

cases = []
for (kind, n), sets in SETS.items():
    tree = get_tree(kind, n)
    K = tree.n_nodes
    for S in sets:
        k = steiner_k(tree, S)
        if k > 8:
            continue
        lo = max(SIGMA_MIN[k], 1e14 ** (-1 / (K - 2)) if K > 2 else 0)
        hi = min(lo * (100 if k <= 3 else 10), 0.3 if k <= 5 else 0.6)
        if hi / lo < 2:
            hi = min(lo * 3, 0.6)
        sig = [float(f"{x:.4g}") for x in np.logspace(np.log10(hi), np.log10(lo), 6)]
        cases.append(dict(kind=kind, n=n, S=S, sigmas=sig, seeds=list(range(8)),
                          cost=k + K / 4, k=k, K=K))
json.dump(cases, open("experiments/locality/pilot/cases_step1.json", "w"), indent=1)
for c in cases:
    print(c["kind"], c["n"], c["S"], "K", c["K"], "k", c["k"], c["sigmas"][0], c["sigmas"][-1])
print(len(cases), "cases,", sum(len(c["seeds"]) for c in cases), "tasks")
