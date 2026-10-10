"""Reviewer's own TD level-occupation check: L(t) from the saved trajectory with the reviewer's dense
contraction (indep.Net), time within +-2% of each predicted level, per-seed power fits vs sigma."""

import glob
import json
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(
    0,
    "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality",
)
from indep import Net, teacher  # noqa: E402

LEVELS = [
    0.24125,
    0.06125,
    0.015184582173045497,
]  # 1/2 (0.36 + 0.1225), 1/2 0.1225, route-rule optimum
recs = {}
for f in sorted(glob.glob(sys.argv[1] + ".part*")):
    for line in open(f):
        r = json.loads(line)
        if r.get("name") == "TD":
            recs[r["_key"]] = r
net = Net("mps", 8)
rows = []
for r in recs.values():
    if r.get("status") != 1:
        rows.append((r["seed"], r["sigma"], "status", r.get("status")))
        continue
    F, U, Wst = teacher(8, None, r["weights"], patterns=r["patterns"])
    z = np.load(r["traj"])
    t, y = z["t"], torch.from_numpy(z["y"])
    L = np.array(
        [float(0.5 * ((net.dense(net.unflat(y[:, i])) - Wst) ** 2).sum()) for i in range(len(t))]
    )
    order = ">".join(str(e[1]) for e in r["events"])
    durs, nins = [], []
    for lev in LEVELS:
        x = np.log(L / lev)
        lo, hi = np.log(0.98), np.log(1.02)
        dur = 0.0
        for i in range(len(t) - 1):
            a, b = x[i], x[i + 1]
            if a == b:
                dur += (t[i + 1] - t[i]) if lo < a < hi else 0.0
                continue
            s0, s1 = min(a, b), max(a, b)
            ov = max(0.0, min(s1, hi) - max(s0, lo))
            dur += (t[i + 1] - t[i]) * ov / (s1 - s0)
        durs.append(dur)
        nins.append(int(((x > lo) & (x < hi)).sum()))
    rows.append((r["seed"], r["sigma"], order, durs, nins))
json.dump(rows, open(sys.argv[2], "w"))
by = {}
for row in rows:
    if row[2] == "0>1>2":
        by.setdefault(row[0], []).append(row)
for j in range(3):
    ps = []
    for seed, g in by.items():
        g = [x for x in g if x[3][j] > 0 and x[4][j] >= 3]
        if len({x[1] for x in g}) >= 3:
            sig = np.array([x[1] for x in g])
            d = np.array([x[3][j] for x in g])
            ps.append(np.polyfit(np.log(1 / sig), np.log(d), 1)[0])
    ps = np.array(ps)
    print(
        f"level {j} L={LEVELS[j]:.5f}: per-seed power p = {ps.mean():.3f} +- {ps.std(ddof=1) / np.sqrt(len(ps)):.3f} (n={len(ps)})"
        if len(ps) > 1
        else f"level {j}: n={len(ps)}"
    )
print("orders:", {o: sum(1 for x in rows if x[2] == o) for o in {x[2] for x in rows}})
