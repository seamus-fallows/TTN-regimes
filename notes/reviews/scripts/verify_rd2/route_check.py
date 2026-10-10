"""Route-amplitude rule: own unit grouping; observed first pair from the round-2 summary; out-of-sample review runs."""

import glob
import json
import sys
from collections import defaultdict

import numpy as np

REPO = sys.argv[1]
routes = json.load(open(f"{REPO}/experiments/real_data/results/routes_raw.json"))
summ = {
    r["name"]: r for r in json.load(open(f"{REPO}/data/real_data_local/full/main_summary.json"))
}


def pred_pair(r):
    sib = [p for p in r["pairs"] if p["k"] == 3]
    b = max(sib, key=lambda p: p["lam"])
    e = r["edges"]
    return tuple(sorted([e.index(b["u"]), e.index(b["w"])]))


def obs_pair(name):
    s = summ[name]
    ev = np.array(s["rank_events"], float)[:, 0]
    t = np.nanmin(ev)
    return tuple(int(x) for x in np.flatnonzero(ev <= t + max(1.0, 0.05 * t))), t == 0 or s.get(
        "first_rank2_at_start"
    )


states = [r for r in routes if "pairs" in r]
kept = [r for r in states if r["rank1_rel_err"] <= 0.05]
print("states", len(states), "kept", len(kept), "dropped", len(states) - len(kept))
mismatch_obs = 0
per_state = defaultdict(lambda: [0, 0])
by = defaultdict(list)
for r in kept:
    o, _ = obs_pair(r["name"])
    if list(o) != sorted(r.get("first_rank2_edges") or []):
        mismatch_obs += 1
    r["_obs"], r["_pred"] = o, pred_pair(r)
    k = "quadtree" if r["ordering"] == "quadtree" else "scrambled"
    per_state[k][0] += r["_obs"] == r["_pred"]
    per_state[k][1] += 1
    by[(r["ordering"], r["seed"])].append(r)
print("observed pair differs from routes_raw first_rank2_edges in", mismatch_obs, "kept states")
print("per state:", dict(per_state))
print("pairs observed = 2 edges:", sum(len(r["_obs"]) == 2 for r in kept), "/", len(kept))

# own units: consecutive sigma states of an (order, seed) with the same predicted pair and plateau loss within 0.01
for name, key in (("pred+loss", lambda r: r["_pred"]), ("loss only", lambda r: 0)):
    units = []
    for (o, s), rs in sorted(by.items()):
        rs.sort(key=lambda r: r["sigma"])
        cur = [rs[0]]
        for r in rs[1:]:
            if key(r) == key(cur[-1]) and abs(r["train_loss"] - cur[-1]["train_loss"]) < 0.01:
                cur.append(r)
            else:
                units.append(cur)
                cur = [r]
        units.append(cur)
    out = defaultdict(int)
    for u in units:
        h = [r["_obs"] == r["_pred"] for r in u]
        out["hit" if all(h) else "mixed" if any(h) else "miss"] += 1
    print(f"units ({name}): {len(units)} {dict(out)}; (order, seed) pairs {len(by)}")

# majority-per-(order, seed): one trial per (order, seed) at its smallest sigma
first = [sorted(rs, key=lambda r: r["sigma"])[0] for rs in by.values()]
print(
    "one state per (order, seed) at smallest sigma: hits",
    sum(r["_obs"] == r["_pred"] for r in first),
    "/",
    len(first),
)

# out of sample: review runs
rows = []
for p in glob.glob(f"{REPO}/data/review_real_data/audit/**/audit_table.json", recursive=True) + [
    f"{REPO}/data/review_real_data/audit/audit_table.json"
]:
    for r in json.load(open(p)):
        rows.append((p, r))
seen = set()
oos = []
for p, r in rows:
    if r["name"] in seen or r.get("tree") != "binary":
        continue
    seen.add(r["name"])
    oos.append(r)
print("review binary runs:", len(oos))
for r in oos:
    b3 = r.get("best3_pair")
    print(
        f"  {r['name'][27:]:45s} best3 {b3} first {r.get('first_rank2_edges')} rank1_err {r.get('rank1_err', float('nan')):.3g} "
        f"hit {sorted(b3 or []) == sorted(r.get('first_rank2_edges') or [])}"
    )
