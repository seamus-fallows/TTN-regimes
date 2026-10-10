"""MPS vs binary: tau-clock medians, absolute-clock t_a ratios, final-loss comparisons, Fashion."""

import json
import sys
from collections import defaultdict

import numpy as np

REPO = sys.argv[1]
LV = (0.3, 0.25, 0.2, 0.15, 0.12, 0.1)


def load(p):
    return [
        r
        for r in json.load(open(p))
        if "error" not in r and not r.get("centred") and r.get("features", "raw") == "raw"
    ]


def key(r):
    o = r["ordering"]
    return "mps" if r["tree"] == "mps" else ("scr" if o.startswith("scrambled") else o)


for label, path in (
    ("MNIST main", f"{REPO}/data/real_data_local/full/main_summary.json"),
    ("Fashion", f"{REPO}/data/real_data_local/full/fashion_summary.json"),
):
    rows = load(path)
    g = defaultdict(list)
    for r in rows:
        g[(key(r), r["sigma"])].append(r)
    sigmas = sorted({r["sigma"] for r in rows})
    print(f"=== {label}: tau-clock medians (inf = not reached); MPS first at level?")
    viol = []
    for s in sigmas:
        line = []
        for lv in LV:
            med = {}
            for t in ("mps", "quadtree", "scr"):
                v = [r.get(f"t_loss_{lv}", np.nan) for r in g.get((t, s), [])]
                v = [x if np.isfinite(x) else np.inf for x in v]
                med[t] = np.median(v) if v else np.nan
            first = med["mps"] <= min(med["quadtree"], med["scr"])
            if (
                not first
                and np.isfinite(med["mps"])
                or (not first and np.isfinite(min(med["quadtree"], med["scr"])))
            ):
                viol.append((s, lv, med))
            line.append(f"{lv}: {med['mps']:.0f}/{med['quadtree']:.0f}/{med['scr']:.0f}")
        print(f"  sigma {s}: " + "  ".join(line))
    print("  levels where the MPS median is NOT first:", [(s, lv) for s, lv, _ in viol])
    # absolute clock: t_a ratios MPS / quadtree per seed
    print("  t_a(MPS)/t_a(quadtree) per seed:")
    for s in sigmas:
        m = {r["seed"]: r["t_a"] for r in g.get(("mps", s), [])}
        q = {r["seed"]: r["t_a"] for r in g.get(("quadtree", s), [])}
        rat = {sd: m[sd] / q[sd] for sd in sorted(m) if sd in q}
        print(
            f"    sigma {s}: "
            + ", ".join(f"s{sd} {v:.3g}" for sd, v in rat.items())
            + f"  later in {sum(v > 1 for v in rat.values())}/{len(rat)}"
        )
    # final loss: MPS below quadtree, same seed
    print("  final loss: MPS below quadtree (same seed):")
    for s in sigmas:
        m = {r["seed"]: r["loss_end"] for r in g.get(("mps", s), []) if "loss_end" in r}
        q = {r["seed"]: r["loss_end"] for r in g.get(("quadtree", s), []) if "loss_end" in r}
        c = [m[sd] < q[sd] for sd in m if sd in q]
        sc = defaultdict(dict)
        for r in rows:
            if (
                r["tree"] == "binary"
                and r["ordering"].startswith("scrambled")
                and r["sigma"] == s
                and "loss_end" in r
            ):
                sc[r["ordering"]][r["seed"]] = r["loss_end"]
        cs = [sc[o][sd] < q[sd] for o in sc for sd in sc[o] if sd in q]
        print(f"    sigma {s}: MPS<quad {sum(c)}/{len(c)}; scrambled<quad {sum(cs)}/{len(cs)}")
