"""Step 1, check of the derived swap criterion (notes 1.3) with the library's gauge-projected Hessian.

Exact critical point "component a learned, b not" under an aligned covariance: aligned_init with
balanced amplitudes s_a^(1/K) for a and 0 for b (gradient norm <= 1e-15). The gauge directions are
linearly dependent at such special points; `project_out_gauge` removes their span (rtol 1e-10). Prediction on binary-4 (two bottom cores
of two sites each, same eigenvalues on both cores): the most negative Hessian eigenvalue in
parameter units is
    lambda_min = s_a sqrt(Lambda_b) (s_a sqrt(Lambda_a) - s_b sqrt(Lambda_b)) / q^2,  q = s_a^(1/3),
negative iff Lambda_b s_b^2 > Lambda_a s_a^2. On MPS-4 the swap route needs 4 cores: no negative
eigenvalue (non-strict). Usage: uv run python experiments/data_teacher/e1_hessian_check.py
Writes results/e1_hessian_check.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from common import RESULTS, aligned_setup, make_spec

from ttn.diagnostics import gauge_directions, hessian, project_out_gauge, saddle_type
from ttn.dynamics import Problem
from ttn.network import aligned_init

HERE = Path(__file__).parent


def main():
    torch.set_num_threads(4)
    cfg = yaml.safe_load(open(HERE / "configs/e1_order.yaml"))
    rows = []
    for tname, tcfg in cfg["teachers"].items():
        for variant in ("sigma", "white_Ls", "white_sqrtLs"):
            for tree in ("binary", "mps"):
                spec = make_spec(tree, 4, 4, 4)
                teacher, metrics, big_lam = aligned_setup(4, 4, tcfg["weights"], tcfg["eig_index"],
                                                          tcfg["alpha"], variant)
                w = teacher.weights.numpy()
                k = spec.n_cores
                for a in (0, 1):
                    b = 1 - a
                    amps = torch.zeros(k, 2, dtype=torch.float64)
                    amps[:, a] = float(w[a]) ** (1.0 / k)
                    theta = spec.flatten(aligned_init(spec, teacher.factors, amps))
                    prob = Problem(spec, teacher, site_metrics=metrics)
                    grad = float(torch.linalg.norm(prob.grad(theta)))
                    h = hessian(prob.loss, theta)
                    _, eigs = project_out_gauge(h, gauge_directions(spec, theta))
                    st = saddle_type(eigs)
                    lam_eff = big_lam if variant == "sigma" else np.ones(2)
                    sa, sb = w[a], w[b]
                    la, lb = lam_eff[a], lam_eff[b]
                    q2 = sa ** (2.0 / 3.0)
                    pred = sa * np.sqrt(lb) * (sa * np.sqrt(la) - sb * np.sqrt(lb)) / q2
                    rows.append(dict(teacher=tname, variant=variant, tree=tree, learned=a,
                                     grad_norm=grad, lambda_min=st["lambda_min"],
                                     n_negative=st["n_negative"], type=st["type"] if "type" in st
                                     else st.get("kind"),
                                     pred_strict=bool(lb * sb**2 > la * sa**2),
                                     pred_lambda_min_binary=pred if tree == "binary" else np.nan))
                    print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e1_hessian_check.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
