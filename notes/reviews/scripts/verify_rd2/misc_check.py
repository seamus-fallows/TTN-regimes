"""Pixel orders of the test runs vs predictor; MPS first-rise counts; whitened S2S; Fashion sibling counts;
per-seed ranks in the order test."""

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = sys.argv[1]
R = f"{REPO}/experiments/real_data/results"
pred = {p["order"]: p for p in json.load(open(f"{R}/extra_orders_predictor.json"))["orders"]}
bad = 0
n = 0
for f in sorted(glob.glob(f"{REPO}/data/real_data_local/extra_orders/*.npz")):
    with np.load(f, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
    o = meta["case"]["ordering"]
    n += 1
    if list(meta["order"]) != list(pred[o]["pixel_order"]):
        bad += 1
        print("pixel order mismatch", o, meta["order"], pred[o]["pixel_order"])
    c = meta["case"]
    if (c["sigma"], c["chi"], c["t2"], c["features"], c.get("rtol"), c.get("atol")) != (
        0.1,
        4,
        1e4,
        "raw",
        c.get("rtol"),
        c.get("atol"),
    ):
        print("settings differ", o, c)
print(
    f"pixel orders checked: {n} runs, mismatches {bad}; example case keys:",
    sorted(meta["case"].keys()),
)
print(
    "rtol/atol example:",
    meta["case"].get("rtol"),
    meta["case"].get("atol"),
    "n_train",
    meta["case"].get("n_train"),
)

# per-seed rank of quadtree in the order test
rows = list(csv.DictReader(open(f"{R}/extra_orders_summary.csv")))
by = defaultdict(dict)
for r in rows:
    by[int(r["seed"])][r["ordering"]] = float(r["loss_end"])
for s, d in sorted(by.items()):
    srt = sorted(d, key=d.get)
    print(
        f"seed {s}: lowest {srt[:3]} ({[round(d[o], 4) for o in srt[:3]]}); quadtreeT rank {srt.index('quadtreeT') + 1}/{len(srt)}"
    )
O1 = {o: np.mean([by[s][o] for s in by]) for o in by[0]}
srt = sorted(O1, key=O1.get)
print(
    "O1 rank of quadtreeT",
    srt.index("quadtreeT") + 1,
    "colmajor",
    srt.index("colmajor") + 1,
    "hilbert",
    srt.index("hilbert") + 1,
    "of",
    len(srt),
)

# MPS first rises (main sweep, raw)
main = [
    r
    for r in json.load(open(f"{REPO}/data/real_data_local/full/main_summary.json"))
    if "error" not in r
]
mps = [r for r in main if r["tree"] == "mps" and r["features"] == "raw"]
at0 = [r for r in mps if r.get("first_rank2_at_start")]
res = [r for r in mps if not r.get("first_rank2_at_start")]
print(
    f"MPS raw runs {len(mps)}: first rise at tau = 0 in {len(at0)} ({sorted((r['seed'], r['sigma']) for r in at0)})"
)
print(
    "resolved first edges:",
    defaultdict(int, {tuple(r["first_rank2_edges"]): 0 for r in res}).keys(),
    sum(r["first_rank2_edges"] == [5] for r in res),
    "/",
    len(res),
    "are edge 5 alone",
)
before = sum(r["t_first_rank2"] < r["drops"][0]["t"] for r in res)
print("resolved first rise before the first drop's peak:", before, "/", len(res))

# whitened S2S at sigma <= 0.1 (binary)
w = [
    r
    for r in main
    if r["tree"] == "binary" and r["features"] == "white" and r["sigma"] <= 0.1 + 1e-12
]
print(
    "whitened binary sigma <= 0.1: S2S",
    sum(bool(r.get("s2s")) for r in w),
    "/",
    len(w),
    {
        o: f"{sum(bool(r.get('s2s')) for r in w if r['ordering'] == o)}/{sum(r['ordering'] == o for r in w)}"
        for o in {r["ordering"] for r in w}
    },
)

# Fashion: siblings crossing together (first_rank2_edges = 2 bottom edges with the same parent)
fa = [
    r
    for r in json.load(open(f"{REPO}/data/real_data_local/full/fashion_summary.json"))
    if "error" not in r
]
for o in ("quadtree", "scrambled0"):
    rs = [r for r in fa if r["tree"] == "binary" and r["ordering"] == o]
    tog = 0
    for r in rs:
        e = r.get("first_rank2_edges") or []
        if len(e) == 2:
            lv, nodes, par = r["edge_levels"], r["edges"], r["parent"]
            tog += all(lv[x] == 1 for x in e) and par[nodes[e[0]]] == par[nodes[e[1]]]
    print(f"Fashion {o}: siblings cross together {tog}/{len(rs)}")
