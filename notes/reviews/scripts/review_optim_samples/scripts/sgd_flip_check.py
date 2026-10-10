"""SGD 'joint' runs of near-tie inits: transient a_2 crossing or genuine removal of the second plateau?
binary N=4, odeco [3,2,1] (teacher seed 101), fan-in sigma = 0.1 seed 1, online SGD eta = 5e-4,
B in {256, 16}; run to T = 30 (past GF's second event at 23.2) and record all upward crossings of 1/2.
Noise seeds 100+ (new draws, not the agent's)."""

import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
OUT = Path("c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optim_samples")


def run(task):
    torch.set_num_threads(1)
    from ttn.losses import sample_inputs
    from ttn.network import TTNSpec, forward
    from ttn.teachers import odeco_teacher
    from ttn.tree import make_tree

    spec = TTNSpec.uniform(make_tree("binary", 4), 3, 3)
    teacher = odeco_teacher(4, 3, [3.0, 2.0, 1.0], seed=101)
    sigma, seed, B, ns, eta, T = task
    p = spec.flatten(spec.init(sigma, seed=seed))
    gen = torch.Generator().manual_seed(777_000 + ns)
    steps = int(math.ceil(T / eta))
    ts, ovl = [], []
    for k in range(steps + 1):
        if k % 20 == 0:
            with torch.no_grad():
                ovl.append(teacher.component_overlaps(spec, spec.unflatten(p)).numpy())
            ts.append(k * eta)
        if k == steps:
            break
        x = sample_inputs(B, 4, 3, gen)
        q = p.detach().requires_grad_(True)
        loss = 0.5 * (forward(spec, spec.unflatten(q), x) - teacher(x)).pow(2).mean()
        (g,) = torch.autograd.grad(loss, q)
        p = (q - eta * g).detach()
    ts, ovl = np.array(ts), np.array(ovl)
    res = dict(sigma=sigma, seed=seed, B=B, noise_seed=ns, eta=eta)
    for r in range(3):
        up = np.nonzero((ovl[1:, r] > 0.5) & (ovl[:-1, r] <= 0.5))[0]
        res[f"a{r + 1}_up"] = [round(float(ts[i + 1]), 3) for i in up]
        res[f"a{r + 1}_max_t<12"] = round(float(ovl[ts < 12, r].max()), 3)
        res[f"a{r + 1}_at_t15"] = round(float(ovl[np.argmin(np.abs(ts - 15)), r]), 3)
    return res


if __name__ == "__main__":
    tasks = [(0.1, 1, B, ns, 5e-4, 30.0) for B in (256, 16) for ns in range(100, 106)]
    OUT.mkdir(parents=True, exist_ok=True)
    with (
        Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool,
        open(OUT / "sgd_flip_check.jsonl", "a") as f,
    ):
        for res in pool.imap_unordered(run, tasks):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(json.dumps(res), flush=True)
