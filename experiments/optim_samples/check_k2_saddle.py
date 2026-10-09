"""Hessian check of the k = 2 route used by collapse_test.py (MPS N=4, pattern teacher [3, 0.3]).

Gradient flow (lsoda) from fan-in init sigma = 0.1, seed 0; at the midpoint of the second plateau
(minimum speed between the two events) the gauge-projected Hessian spectrum is computed with
`ttn.diagnostics`. Prediction for a k = 2 route: one negative direction of size
s2 * s1^{(K-2)/K} = 0.52 (theory notes).

Run: uv run python experiments/optim_samples/check_k2_saddle.py [--out-dir DIR]
Writes k2_saddle_hessian.json.
"""

import json

import numpy as np
import torch

from common import RESULTS, TEACHER_SEED
from ttn.diagnostics import (
    crossing_times,
    gauge_directions,
    hessian,
    plateau_midpoints,
    project_out_gauge,
    saddle_type,
)
from ttn.dynamics import Problem, Recorder, integrate_gf
from ttn.network import TTNSpec
from ttn.teachers import pattern_teacher
from ttn.tree import make_tree


def main():
    torch.set_num_threads(1)
    spec = TTNSpec.uniform(make_tree("mps", 4), 3, 3)
    teacher = pattern_teacher([3.0, 0.3], [[0, 0, 0, 0], [1, 1, 0, 0]], 3, seed=TEACHER_SEED)
    prob = Problem(spec, teacher)
    traj = integrate_gf(prob, spec.init(0.1, seed=0), 60.0, method="lsoda",
                        log_times=np.linspace(0.05, 60, 1200),
                        recorder=Recorder(spectra=False, save_params=True))
    events = crossing_times(traj["t"], traj["overlaps"], 0.5)
    mid = plateau_midpoints(traj["t"], traj["speed"], np.sort(events))[0]
    i = int(np.argmin(np.abs(traj["t"] - mid)))
    theta = torch.from_numpy(traj["params"][i])
    h_proj, _ = project_out_gauge(hessian(prob.loss, theta), gauge_directions(spec, theta))
    eigs = torch.linalg.eigvalsh(h_proj)
    out = dict(events=events.tolist(), midpoint=float(traj["t"][i]),
               overlaps_at_midpoint=traj["overlaps"][i].tolist(),
               lowest_eigs=eigs[:4].tolist(), top_eig=eigs[-1].item(),
               saddle=saddle_type(eigs), predicted_lambda_min=-0.3 * 3.0**0.5)
    (RESULTS / "k2_saddle_hessian.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
