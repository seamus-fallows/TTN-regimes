"""Step 4(c) teacher scan (fix round 2): random MPS-8 teachers (ranks 2, decay 0.6) whose root cut on
the binary-8 student has a teacher s_2 above EVERY bottom bond by > 4% (a weight-order rule then predicts
the root bond first, P11 predicts it after a bottom bond on each half). Prints the teacher seeds used in
configs/r2_cross_b.yaml. Usage: uv run python experiments/data_teacher/scan_cross_teachers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch
torch.set_num_threads(1)
from setups import build
from run_generic import dense_bond_spectra
base = dict(kind="ttn_teacher", n_sites=8, d=2, chi=4, decay=0.6, pair="mps_on_binary", tree="binary")
rows = []
for ts in range(200, 1000):
    spec, teacher = build(dict(base, teacher_seed=ts), 0)[:2]
    tr = spec.tree
    sp = dense_bond_spectra(teacher.dense(), tr, None)
    s2 = {v: sp[v][1] for v in tr.edges}
    root_edges = [v for v in tr.edges if tr.parent[v] == tr.root]
    bottoms = [v for v in tr.edges if not tr.children[v]]
    halves = {}
    for v in bottoms:
        halves.setdefault(tr.parent[v], []).append(v)
    r = s2[root_edges[0]]
    disc = any(all(r > s2[b] for b in bs) for bs in halves.values())
    strong = all(r > s2[b] for b in bottoms)
    if strong and r > 1.04 * max(s2[b] for b in bottoms):
        rows.append((ts, round(r, 4), [round(s2[b], 4) for b in bottoms], strong))
for x in rows: print(x)
