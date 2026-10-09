"""Predicted first saddles for steps 2 and 4(a): rank-1 constrained critical points.

For every case of configs/e2_misaligned.yaml and configs/e4a_nonorth.yaml (variant 'sigma' only:
the absorbed white problem has the same critical values), find the distinct local minima of L over
TTNs on the student's tree with all bond ranks 1 (constrained.py), and for step 2 also the race
direction: the best bond-rank-1 approximation of Sigma W* in the Frobenius norm (the maximizer of
g = <W, Sigma W*> on unit bond-rank-1 tensors, which sets the escape direction from the origin).

Usage: uv run python experiments/data_teacher/predict_rank1.py
Writes results/rank1_predictions.csv and /root/outputs/data_teacher/rank1_predictions.pt.
"""

from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
import yaml

from common import OUT_ROOT
from constrained import constrained_minima, metric_loss_of_dense, ranked_spec
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
    l0 = metric_loss_of_dense(w_star, metrics)(torch.zeros_like(w_star)).item()
    for m in constrained_minima(spec1, metric_loss_of_dense(w_star, metrics), n_starts=40):
        rows.append(dict(exp=name, tag=case["tag"], tree=tree, kind="saddle", loss=m["loss"],
                         rel_loss=m["loss"] / l0, count=m["count"], grad_norm=m["grad_norm"]))
        states.append(m["W"])
    if metrics is not None:
        sw = w_star
        for i, mat in enumerate(metrics):
            sw = apply_on_leg(sw, mat, i)
        for m in constrained_minima(spec1, metric_loss_of_dense(sw, None), n_starts=40):
            rows.append(dict(exp=name, tag=case["tag"], tree=tree, kind="race", loss=m["loss"],
                             rel_loss=float("nan"), count=m["count"], grad_norm=m["grad_norm"]))
            states.append(m["W"])
    return list(zip(rows, states))


if __name__ == "__main__":
    jobs = []
    for cfg_name in ("e2_misaligned", "e4a_nonorth"):
        cfg = yaml.safe_load(open(HERE / "configs" / f"{cfg_name}.yaml"))
        for case in expand(cfg):
            for tree in cfg["trees"]:
                jobs.append((cfg["name"], case, tree))
    with Pool(8) as pool:
        out = [x for res in pool.map(one, jobs) for x in res]
    df = pd.DataFrame([r for r, _ in out])
    df.to_csv(HERE / "results" / "rank1_predictions.csv", index=False)
    torch.save([(r, w) for r, w in out], OUT_ROOT / "rank1_predictions.pt")
    print(df.to_string())
