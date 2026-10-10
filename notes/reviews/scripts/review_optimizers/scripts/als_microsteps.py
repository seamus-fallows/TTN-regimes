"""ALS / DMRG / Riemannian GD on the deep problems: overlaps a_r at every micro-step / step, to see whether the
'learning order' reported for these methods is resolved (crossings of 1/2 in different micro-steps) or not.
usage: python als_microsteps.py OUTDIR"""
import json
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
from experiments.optimizers.common import initial_point, make_problem  # noqa: E402
from experiments.optimizers.optimizers import run  # noqa: E402

rows = []
for tree, n in (("binary", 4), ("binary", 8), ("mps", 8)):
    for opt, extra in (("als", {"n_sweeps": 3}), ("dmrg", {"n_sweeps": 3}), ("riemannian", {"lr": 1.0, "n_steps": 20})):
        for seed in range(6):
            cfg = dict(tree=tree, n_sites=n, d=3, chi=4, teacher="odeco", weights=[3.0, 2.0, 1.0], teacher_seed=100,
                       sigma=0.3 if n == 8 else 0.008, seed=seed, init="balanced" if n == 8 else "fanin",
                       optimizer=opt, **extra)
            p = make_problem(cfg)
            x0 = initial_point(p, cfg)
            tr = run(p, x0, cfg)
            d = tr.data
            a = d["fine_overlaps"]
            # index of the first micro-step (sample) at which each a_r >= 1/2 (stays learned: use last upward)
            first_idx = []
            for r in range(a.shape[1]):
                above = a[:, r] >= 0.5
                idx = np.flatnonzero(above)
                first_idx.append(int(idx[0]) if idx.size else -1)
            same_interval = len(set(first_idx)) < len(first_idx)
            row = dict(tree=f"{tree}{n}", opt=opt, seed=seed, first_sample_ge_half=first_idx,
                       n_distinct_samples=len(set(first_idx)), a_at_samples=np.round(a[: max(first_idx) + 2], 3).tolist())
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "a_at_samples"}), flush=True)
with open(sys.argv[1] + "/als_microsteps.json", "w") as f:
    json.dump(rows, f)
