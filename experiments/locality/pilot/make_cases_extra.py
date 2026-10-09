"""Case lists for step 3 at N = 8 / rule tests (nested teachers) and step 5 (robustness)."""
import json

import numpy as np


def logs(hi, lo, n=6):
    return [float(f"{x:.4g}") for x in np.logspace(np.log10(hi), np.log10(lo), n)]


SEEDS = list(range(8))
# --- nested teachers at N = 6, 8 (rank rule vs literal H2 vs Furman N) ---
TC = dict(name="TC", patterns=[[0] * 8, [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1] * 8],
          weights=[1.0, 0.6, 0.35, 0.2])
TD = dict(name="TD", patterns=[[0] * 8, [0, 0, 0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]],
          weights=[1.0, 0.6, 0.35])
TD6 = dict(name="TD6", patterns=[[0] * 6, [0, 0, 0, 1, 1, 0], [1, 1, 0, 0, 0, 0]],
           weights=[1.0, 0.6, 0.35])
nested = []
for kind in ("mps", "binary"):
    nested.append(dict(kind=kind, n=8, sigmas=logs(0.2, 0.03), seeds=SEEDS, cost=20, **TC))
nested.append(dict(kind="ht", n=8, start="saddle", sigmas=logs(0.3, 0.03), seeds=SEEDS, cost=20,
                   wall_limit=900, **TC))
nested.append(dict(kind="mps", n=8, sigmas=logs(0.2, 0.02), seeds=SEEDS, cost=15, **TD))
nested.append(dict(kind="mps", n=6, sigmas=logs(0.2, 0.01), seeds=SEEDS, cost=10, **TD6))
json.dump(nested, open("experiments/locality/pilot/cases_nested.json", "w"), indent=1)

# --- robustness: chi, d, weight ratio on N = 4 ---
base = [("mps", [1, 2], 2), ("mps", [0, 1, 2], 3), ("binary", [1, 2], 3), ("mps", [0, 3], 4),
        ("binary", [0, 1], 1), ("ht", [0, 1], 3)]
rob = []


def sig(k, kind):
    if k == 4:
        return logs(0.3, 0.03)
    if kind == "ht":
        return logs(0.1, 0.006)
    return logs(0.1, 0.001, 7)


for kind, S, k in base:
    rob.append(dict(kind=kind, n=4, S=S, d=2, chi=4, sigmas=sig(k, kind), seeds=SEEDS, cost=k))
    for chi in (2, 4, 8):
        rob.append(dict(kind=kind, n=4, S=S, d=4, chi=chi, sigmas=sig(k, kind), seeds=SEEDS, cost=k))
    for s2 in (0.25, 0.75):
        rob.append(dict(kind=kind, n=4, S=S, weights=[1.0, s2], sigmas=sig(k, kind), seeds=SEEDS,
                        cost=k))
for s2 in (0.25, 0.75):
    rob.append(dict(kind="mps", n=4, S=[1, 2], start="saddle", weights=[1.0, s2],
                    sigmas=logs(0.1, 1e-4, 7), seeds=SEEDS, cost=1))
json.dump(rob, open("experiments/locality/pilot/cases_robust.json", "w"), indent=1)
print(len(nested), "nested cases;", len(rob), "robustness cases")
