"""Independent recomputation of the pre-registered order test from the committed CSV + predictor JSON.

Written from the registration text (notes/real_data.md 9.4-9.5), not from order_test.py.
"""

import csv
import json
import math
import sys

import numpy as np
from scipy.stats import rankdata

REPO = sys.argv[1]
R = f"{REPO}/experiments/real_data/results"
pred = {p["order"]: p for p in json.load(open(f"{R}/extra_orders_predictor.json"))["orders"]}
rows = list(csv.DictReader(open(f"{R}/extra_orders_summary.csv")))
print("rows", len(rows), "stops", {r["stop"] for r in rows}, "t2", {r["t2"] for r in rows})

by = {}
for r in rows:
    assert r["tree"] == "binary" and abs(float(r["sigma"]) - 0.1) < 1e-12
    by.setdefault(r["ordering"], []).append(r)
O1, O2, cens = {}, {}, {}
for o, v in by.items():
    seeds = sorted(int(r["seed"]) for r in v)
    assert seeds == [0, 1, 2], (o, seeds)
    O1[o] = np.mean([float(r["loss_end"]) for r in v])
    t = [float(r["t_loss_0.25"]) for r in v]
    cens[o] = sum(not math.isfinite(x) for x in t)
    O2[o] = np.mean([math.log10(x) if math.isfinite(x) else 4.0 for x in t])
print("orders", len(by), "censored seeds", sum(cens.values()), {o: c for o, c in cens.items() if c})

new_random = sorted(o for o in by if o.startswith("scrambled") and int(o[9:]) >= 100)
structured = sorted(o for o in by if not o.startswith("scrambled"))
print("new random", len(new_random), "structured", structured)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return np.corrcoef(rx, ry)[0, 1]


def test(orders, out, rng, nb=10000, npm=10000):
    x = np.array([pred[o]["S"] for o in orders])
    y = np.array([out[o] for o in orders])
    rho = spearman(x, y)
    n = len(x)
    b = []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        b.append(spearman(x[i], y[i]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    perm = np.array([spearman(rng.permutation(x), y) for _ in range(npm)])
    p = (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (npm + 1)
    return rho, lo, hi, p, n


for seed in (1, 2, 3):
    rng = np.random.default_rng(seed)
    print(f"--- rng seed {seed}")
    for name, orders in (
        ("all", new_random + structured),
        ("random", new_random),
        (
            "unseen",
            [o for o in new_random + structured if o not in ("quadtree", "quadtreeT", "rowmajor")],
        ),
    ):
        for oname, out in (("O1", O1), ("O2", O2)):
            rho, lo, hi, p, n = test(orders, out, rng)
            print(f"{name:7s} {oname}: rho={rho:+.3f} CI=[{lo:+.3f},{hi:+.3f}] p={p:.4f} n={n}")

# ties in S: quadtree/rowmajor/snake share S; quadtreeT/colmajor share S
print("S ties:", sorted({round(pred[o]["S"], 6) for o in structured}))
print("O1 min/max", min(O1.values()), max(O1.values()), min(O1, key=O1.get))
print("O2 min/max", min(O2.values()), max(O2.values()))
json.dump({"O1": O1, "O2": O2, "cens": cens}, open(sys.argv[2], "w"), indent=1)
