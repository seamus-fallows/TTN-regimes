"""Are the GD 'order flips' of near-tie inits real (second plateau removed) or transient crossings of
a_2 = 1/2 (diagnostics.md: use the last upward crossing)? binary N=4, odeco [3,2,1], teacher seed 101.
Records a_r(t) per step for GD at several eta/eta_max and for a fine-step GD proxy of GF (0.01 eta_max)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
from ttn.losses import population_loss
from ttn.network import TTNSpec
from ttn.teachers import odeco_teacher
from ttn.tree import make_tree

OUT = Path("c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optim_samples")
spec = TTNSpec.uniform(make_tree("binary", 4), 3, 3)
teacher = odeco_teacher(4, 3, [3.0, 2.0, 1.0], seed=101)
t2n = teacher.norm2()
eta_max = 2.0 / (3 * 3.0 ** (4 / 3))


def run(sigma, seed, ratio, T):
    p = spec.flatten(spec.init(sigma, seed=seed))
    eta = ratio * eta_max
    steps = int(math.ceil(T / eta))
    a = np.empty((steps + 1, 3))
    for k in range(steps + 1):
        q = p.detach().requires_grad_(True)
        lv = population_loss(spec, spec.unflatten(q), teacher, teacher_norm2=t2n)
        (g,) = torch.autograd.grad(lv, q)
        with torch.no_grad():
            a[k] = teacher.component_overlaps(spec, spec.unflatten(p)).numpy()
        p = p.detach() - eta * g
    t = eta * np.arange(steps + 1)
    res = {}
    for r in range(3):
        up = np.nonzero((a[1:, r] > 0.5) & (a[:-1, r] <= 0.5))[0]
        res[f"a{r + 1}_up_crossings"] = [float(t[i + 1]) for i in up]
        res[f"a{r + 1}_max_before_10"] = float(a[t < 10, r].max())
    # a2 after its first crossing: min over the window [first crossing, first crossing + 10]
    return res


if __name__ == "__main__":
    out = []
    for sigma, seed, T in [(0.1, 1, 40.0), (0.1, 3, 40.0), (0.03, 12, 120.0)]:
        for ratio in (0.01, 0.1, 0.3, 0.9):
            res = run(sigma, seed, ratio, T)
            res.update(sigma=sigma, seed=seed, ratio=ratio)
            print(json.dumps(res), flush=True)
            out.append(res)
    (OUT / "flip_check.json").write_text(json.dumps(out, indent=1))
