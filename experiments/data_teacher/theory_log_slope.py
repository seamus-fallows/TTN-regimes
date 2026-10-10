"""Predicted slope of the second plateau, Dt_2 = A log(1/sigma) + O(1), for an MPS-4 student at the
rank-1 saddle of a two-component non-orthogonal teacher (theory_teachers.md R5/R6):
A_e = (1 - s2(M_e)/lambda) / (lambda^((N-2)/N) s2(M_e)), M_e = T contracted with the saddle's site
vectors outside edge e, lambda = <T, (x) w_i>; the escape uses e* = argmax_e s2(M_e).
Uses the rank-1 critical points from constrained.py (all local minima; the one reached depends on
the init basin). Usage: uv run python experiments/data_teacher/theory_log_slope.py
Writes results/e4a_theory_slopes.csv."""
from pathlib import Path

import pandas as pd
import torch

from constrained import constrained_minima, metric_loss_of_dense, ranked_spec
from setups import build
from common import RESULTS  # noqa: E402

HERE = Path(__file__).parent
rows = []
for c in (0.05, 0.2, 0.45):
    case = dict(kind="nonorth", n_sites=4, d=4, chi=4, s2=0.7, overlap=c, tree="mps")
    spec, teacher, _, _, _ = build(case, 0)
    T = teacher.dense()
    spec1 = ranked_spec(spec, {v: 1 for v in spec.tree.edges})
    for m in constrained_minima(spec1, metric_loss_of_dense(T, None), n_starts=16):
        X = m["W"]
        w = [torch.linalg.svd(torch.movedim(X, i, 0).reshape(4, -1))[0][:, 0] for i in range(4)]
        lam = float(torch.einsum("abcd,a,b,c,d->", T, *w))
        if lam < 0:
            w[0], lam = -w[0], -lam
        out = dict(overlap=c, saddle_loss=m["loss"], count=m["count"], lam=lam)
        best = None
        for e, (i, j) in enumerate([(0, 1), (1, 2), (2, 3)]):
            others = [k for k in range(4) if k not in (i, j)]
            M = T
            for k in sorted(others, reverse=True):
                M = torch.tensordot(M, w[k], dims=([k], [0]))
            s = torch.linalg.svdvals(M)
            s2 = float(s[1])
            a = (1 - s2 / lam) / (lam ** 0.5 * s2)
            out[f"s2_M{e}"] = s2
            out[f"A_{e}"] = a
            if best is None or s2 > best[0]:
                best = (s2, e, a)
        out.update(edge_star=best[1], A_pred=best[2])
        rows.append(out)
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "e4a_theory_slopes.csv", index=False)
print(df.round(4).to_string())
