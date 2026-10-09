"""Predicted first saddles for steps 2 and 4(a): rank-1 constrained critical points.

For every case of configs/e2_misaligned.yaml and configs/e4a_nonorth.yaml (variant 'sigma' only:
the absorbed white problem has the same critical values), find the distinct local minima of L over
TTNs on the student's tree with all bond ranks 1 (constrained.py), and for step 2 also the race
direction: the best bond-rank-1 approximation of Sigma W* in the Frobenius norm (the maximizer of
g = <W, Sigma W*> on unit bond-rank-1 tensors, which sets the escape direction from the origin).

Usage: uv run python experiments/data_teacher/predict_rank1.py [config names]
Writes results/rank1_predictions_<names>.csv and /root/outputs/data_teacher/rank1_predictions_<names>.pt.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
import yaml

from common import OUT_ROOT
from constrained import (constrained_minima, matrix_loss_of_dense, metric_loss_of_dense,
                         ranked_spec)
from correlated import moment_matrix, site_cross_cov
from run_generic import expand
from setups import build

from ttn.network import apply_on_leg

HERE = Path(__file__).parent


def one(args):
    name, case, tree = args
    torch.set_num_threads(1)
    if case.get("variant", "sigma") != "sigma":
        return []
    spec, teacher, metrics, loss_fn, _ = build(dict(case, tree=tree), 0)
    w_star = teacher.dense()
    spec1 = ranked_spec(spec, {v: 1 for v in spec.tree.edges})
    rows, states = [], []
    full = None
    if case["kind"] == "correlated":
        full = moment_matrix(case["n_sites"], case["d"], site_cross_cov(case["n_sites"], case["rho"]))
        loss_dense = matrix_loss_of_dense(w_star, full)
    else:
        loss_dense = metric_loss_of_dense(w_star, metrics)
    l0 = loss_dense(torch.zeros_like(w_star)).item()
    for m in constrained_minima(spec1, loss_dense, n_starts=12):
        rows.append(dict(exp=name, tag=case["tag"], tree=tree, kind="saddle", loss=m["loss"],
                         rel_loss=m["loss"] / l0, count=m["count"], grad_norm=m["grad_norm"]))
        states.append(m["W"])
    if metrics is not None or full is not None:
        sw = w_star
        if full is not None:
            sw = (full @ w_star.reshape(-1)).reshape(w_star.shape)
        else:
            for i, mat in enumerate(metrics):
                sw = apply_on_leg(sw, mat, i)
        for m in constrained_minima(spec1, metric_loss_of_dense(sw, None), n_starts=12):
            rows.append(dict(exp=name, tag=case["tag"], tree=tree, kind="race", loss=m["loss"],
                             rel_loss=float("nan"), count=m["count"], grad_norm=m["grad_norm"]))
            states.append(m["W"])
    return list(zip(rows, states))


if __name__ == "__main__":
    jobs = []
    names = sys.argv[1:] or ["e2_misaligned", "e4a_nonorth", "e3_correlated"]
    for cfg_name in names:
        cfg = yaml.safe_load(open(HERE / "configs" / f"{cfg_name}.yaml"))
        for case in expand(cfg):
            for tree in cfg["trees"]:
                jobs.append((cfg["name"], case, tree))
    with Pool(3) as pool:
        out = [x for res in pool.map(one, jobs) for x in res]
    df = pd.DataFrame([r for r, _ in out])
    tag = "_".join(names)
    df.to_csv(HERE / "results" / f"rank1_predictions_{tag}.csv", index=False)
    torch.save([(r, w) for r, w in out], OUT_ROOT / f"rank1_predictions_{tag}.pt")
    print(df.to_string())
