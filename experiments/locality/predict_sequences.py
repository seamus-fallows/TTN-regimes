"""Predicted saddle sequences (route rule at rank-constrained optima) for the multi-component teachers.

Writes experiments/locality/results/route_sequences.json: per (teacher, tree) the list of predicted
plateau levels L_j and escape degrees k*_j with the minimal routes."""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from rules import get_tree, route_sequence  # noqa: E402

cfg = yaml.safe_load(open(Path(__file__).parent / "configs" / "multicomp.yaml"))["cases"]
out_path = Path(__file__).parent / "results" / "route_sequences.json"
out = json.loads(out_path.read_text()) if out_path.exists() else {}
seen = set()
for c in cfg:
    tree = get_tree(c["kind"], c["n"])
    key = f"{c['name']}_{tree.name}"
    if key in seen or key in out:
        continue
    seen.add(key)
    t0 = time.time()
    st = route_sequence(tree, c["patterns"], c["weights"], d=c.get("d", 2), chi_max=c["chi"])
    out[key] = [dict(loss=s["loss"], k_star=s["k_star"], profile={str(k): v for k, v in s["profile"].items()},
                     routes=[(h[0], float(h[1])) for h in s["routes"]]) for s in st]
    print(key, " -> ".join(f"L={s['loss']:.4f} k*={s['k_star']}" for s in st), f"{time.time() - t0:.0f}s",
          flush=True)
    out_path.write_text(json.dumps(out, indent=1))
