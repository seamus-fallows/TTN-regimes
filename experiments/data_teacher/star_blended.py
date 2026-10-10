"""Star teachers near the threshold (fix round 2): the escape from theta_q ends at the blended rank-1
critical point (the best rank-1 approximation). Its a_q and its Hessian (zero-padded, balanced,
gauge-projected) per tree and w. Where its a_q > 0.9, the saddle-start event "a_q < 0.9" also times the
escape from the blended point: slope 1/|lambda_q| + 1/|lambda_b| if that point is strict (mps4), a power
of 1/eps if it is not (ht4). Compared with results/r2_star_rotation_law.csv.
Usage: uv run python experiments/data_teacher/star_blended.py
Writes results/r2_star_blended.csv.
"""

import math

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from r2_common import balance_rank1, embed, rank1_saddles
from r2_star import spec_for, star

from ttn.diagnostics import gauge_directions, hessian, project_out_gauge
from ttn.dynamics import Problem


def main():
    torch.set_num_threads(1)
    rot = pd.read_csv(RESULTS / "r2_star_rotation_law.csv")
    rows = []
    for tree, n in (("mps", 4), ("binary", 4), ("ht", 4)):
        for w in (1.42, 1.44, 1.48, 1.6):
            spec = spec_for(tree, n)
            teacher = star(n, 2.0, w, w)
            sads = rank1_saddles(spec, teacher, None, n_starts=16)
            best = sads[0]
            spec1 = best["spec1"]
            aq = float(teacher.component_overlaps(spec1, best["cores"])[0])
            cores = embed(spec, balance_rank1(best["cores"]))
            prob = Problem(spec, teacher)
            th = spec.flatten(cores)
            h = hessian(prob.loss, th)
            _, eigs = project_out_gauge(h, gauge_directions(spec, th))
            lam_b = float(eigs[0])
            k = spec.n_cores
            lam_q = -(2.0 ** ((k - 2) / k)) * (math.sqrt(2) * w - 2.0)
            two_stage = 1 / abs(lam_q) + (1 / abs(lam_b) if lam_b < -1e-6 else np.inf)
            m = rot[(rot.tree == f"{tree}{n}") & (rot.w == w)]
            rows.append(dict(tree=f"{tree}{n}", w=w, loss_blended=float(best["loss"]), aq_blended=abs(aq),
                             lambda_b=lam_b, lambda_q=lam_q, pred_one_stage=1 / abs(lam_q),
                             pred_two_stage=two_stage,
                             A_measured=float(m.A_mean.iloc[0]) if len(m) else np.nan,
                             p_measured=float(m.p_mean.iloc[0]) if len(m) else np.nan))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "r2_star_blended.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
