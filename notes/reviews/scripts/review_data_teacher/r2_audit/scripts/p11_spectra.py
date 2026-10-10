"""Teacher s_2 on every binary8 edge for the 4c / r2_cross teachers: is P11 discriminating vs weight order?"""
import os, sys
os.environ["DATA_TEACHER_RESULTS"] = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/results"
os.environ["DATA_TEACHER_OUT"] = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/raw"
sys.path.insert(0, r"C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher")
import torch
torch.set_num_threads(1)
import yaml
from setups import build
from run_generic import expand, dense_bond_spectra

for cfgname in ("r2_cross", "e4c_cross"):
    cfg = yaml.safe_load(open(rf"C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher/configs/{cfgname}.yaml"))
    for case in expand(cfg):
        if case.get("pair") != "mps_on_binary":
            continue
        built = build(dict(case, tree="binary"), 0)
        spec, teacher, metrics = built[:3]
        tr = spec.tree
        sp = dense_bond_spectra(teacher.dense(), tr, metrics)
        s2 = {v: float(sp[v][1]) if len(sp[v]) > 1 else 0.0 for v in tr.edges}
        root_edges = [v for v in tr.edges if tr.parent[v] == tr.root]
        bottoms = [v for v in tr.edges if not tr.children[v]]
        halves = {}
        for v in bottoms:
            halves.setdefault(tr.parent[v], []).append(v)
        root = min(s2[v] for v in root_edges)
        # strong discrimination: root s2 exceeds BOTH bottom bonds of at least one half
        strong = any(root > max(s2[v] for v in vs) for vs in halves.values())
        print(cfgname, case["tag"], "root", round(root, 4),
              "bottoms", {v: round(s2[v], 4) for v in bottoms},
              "halves", {p: [round(s2[v], 4) for v in vs] for p, vs in halves.items()},
              "root>min(all)", root > min(s2.values()) + 1e-9, "strong", strong,
              "root is max", root >= max(s2.values()) - 1e-12)
