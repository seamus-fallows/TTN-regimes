"""Reviewer's independent gradient flow (indep.py flow, Radau rtol 1e-10) vs the runner (LSODA rtol 1e-9)."""

import glob
import json

import numpy as np

R = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_locality/"
run = {}
for f in glob.glob(R + "out/main.part*") + glob.glob(R + "out/timing*.part*"):
    for line in open(f):
        r = json.loads(line)
        if r.get("S") is not None and r.get("status") == 1:
            run[
                (
                    r["kind"],
                    r["n"],
                    "".join(map(str, r["S"])),
                    r["start"],
                    float(r["sigma"]),
                    r["seed"],
                )
            ] = r
rows = []
for f in sorted(glob.glob(R + "indep_flow/*.json")):
    txt = open(f).read().strip()
    if not txt:
        print("EMPTY", f)
        continue
    m = json.loads(txt)
    key = (m["kind"], m["n"], "".join(map(str, m["S"])), m["start"], float(m["sigma"]), m["seed"])
    r = run.get(key)
    ev_m = m["events"]
    line = f"{key}: indep events {[(round(t, 6), c) for t, c in ev_m]} (nfev {m['nfev']}, {m['wall']:.0f}s)"
    if r:
        ev_r = r["events"]
        line += f"\n      runner events {[(round(t, 6), c) for t, c in ev_r]}"
        if len(ev_m) == len(ev_r) == 2:
            g_m, g_r = ev_m[1][0] - ev_m[0][0], ev_r[1][0] - ev_r[0][0]
            line += f"\n      gap t2-t1: indep {g_m:.8g} runner {g_r:.8g} rel.diff {abs(g_m - g_r) / abs(g_r):.1e}"
            rows.append((key, g_m, g_r))
    print(line)
# k = 2 log slopes for seed 0 from the independent runs
for start in ("small", "saddle"):
    pts = sorted((k[4], gm) for k, gm, gr in rows if k[:4] == ("mps", 4, "12", start))
    if len(pts) >= 3:
        x = np.log(1 / np.array([p[0] for p in pts]))
        y = np.array([p[1] for p in pts])
        print(
            f"mps4 S=12 {start} seed 0: indep slope of (t2-t1) vs ln(1/sigma) = {np.polyfit(x, y, 1)[0]:.4f} "
            f"over sigma {[p[0] for p in pts]}"
        )
        pr = sorted((k[4], gr) for k, gm, gr in rows if k[:4] == ("mps", 4, "12", start))
        print(f"   runner same points: {np.polyfit(x, np.array([p[1] for p in pr]), 1)[0]:.4f}")
