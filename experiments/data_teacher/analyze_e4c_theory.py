"""Step 4(c), MPS-8 student on binary-8 teachers: predicted first rank increase (theory_teachers.md
R5): at each rank-1 critical point lambda (x) w_i, M_e = T contracted with w_i off edge (i, i+1), the
first escape route is e* = argmax_e s2(M_e). Compared with the observed first edge
(results/e4c_cross_summary.csv). Usage: uv run python experiments/data_teacher/analyze_e4c_theory.py
Writes results/e4c_theory.csv."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from constrained import constrained_minima, metric_loss_of_dense, ranked_spec
from run_generic import expand
from setups import build

HERE = Path(__file__).parent
torch.set_num_threads(8)
cfg = yaml.safe_load(open(HERE / "configs/e4c_cross.yaml"))
obs = pd.read_csv(HERE / "results/e4c_cross_summary.csv")
rows = []
for case in expand(cfg):
    if case["pair"] != "binary_on_mps":
        continue
    spec, teacher = build(dict(case, tree="mps"), 0)[:2]
    T = teacher.dense()
    n = T.ndim
    spec1 = ranked_spec(spec, {v: 1 for v in spec.tree.edges})
    for m in constrained_minima(spec1, metric_loss_of_dense(T, None), n_starts=12):
        X = m["W"]
        w = [torch.linalg.svd(torch.movedim(X, i, 0).reshape(T.shape[i], -1))[0][:, 0]
             for i in range(n)]
        s2 = []
        for i in range(n - 1):
            M = T
            for k in sorted([k for k in range(n) if k not in (i, i + 1)], reverse=True):
                M = torch.tensordot(M, w[k], dims=([k], [0]))
            s2.append(float(torch.linalg.svdvals(M)[1]))
        o = obs[(obs.pair == "binary_on_mps") & (obs.teacher_seed == case["teacher_seed"])]
        rows.append(dict(teacher_seed=case["teacher_seed"], saddle_loss=m["loss"],
                         count=m["count"], s2_M=[round(x, 4) for x in s2],
                         edge_star=int(np.argmax(s2)),
                         observed_first_edges=o["first_edge"].tolist(),
                         frac_match=float((o["first_edge"] == int(np.argmax(s2))).mean())))
df = pd.DataFrame(rows)
df.to_csv(HERE / "results/e4c_theory.csv", index=False)
print(df.to_string())
