"""Step 2: online SGD at small step size, batch sizes B = 1 .. 256: plateau times vs B and eta.

Online SGD on the minibatch loss (1/2B) sum_b (f(x_b) - f*(x_b))^2 with fresh Gaussian inputs at every
step (no label noise), time t = eta * steps. For each init (fan-in seed, or the aligned balanced init
with per-core amplitude a0) many noise seeds are run, so the spread of event times over noise and the
shift of their mean relative to gradient flow can be measured separately from init randomness.
Recorded per run: event times t_r (a_r = 1/2, overlaps checked every `rec_every` steps, linear
interpolation), final loss, overlaps, conservation-law drift and tr Q_e, effective bond ranks.

Run:  uv run python experiments/optim_samples/sgd_sweep.py configs/sgd_sweep.yaml [--workers 30]
Raw:  /root/outputs/optim_samples/sgd/<block>.jsonl (one line per run; resumable).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

from common import RAW, Problem, crossing_times, setup
from ttn.losses import sample_inputs
from ttn.network import aligned_init, conserved_quantities, forward

OUT = RAW / "sgd"


def key_of(task):
    return "|".join(f"{k}={task[k]}" for k in sorted(task))


def init_params(spec, teacher, task):
    if task["init"] == "aligned":
        amps = torch.full((spec.n_cores, teacher.n_components), task["sigma"])
        return spec.flatten(aligned_init(spec, teacher.factors, amps))
    return spec.flatten(spec.init(task["sigma"], seed=task["seed"]))


def run(task):
    torch.set_num_threads(1)
    spec, teacher = setup(task["tree"], task["n"])
    prob = Problem(spec, teacher)
    p = init_params(spec, teacher, task)
    q0 = conserved_quantities(spec, spec.unflatten(p))
    trq0 = prob.q_traces(p)
    eta, batch = task["eta"], task["B"]
    steps = int(math.ceil(task["T"] / eta))
    rec = task.get("rec_every", 4)
    n_stop = task.get("stop_after", teacher.n_components)  # stop once this many events happened
    gen = torch.Generator().manual_seed(10_000 + 1000 * task["seed"] + task["noise_seed"])
    n, d = spec.tree.n_sites, spec.phys_dims[0]
    ts, ovl = [], []
    t0 = time.time()
    diverged, k, k_hit = False, 0, None
    for k in range(steps + 1):
        if k % rec == 0:
            a = prob.overlaps(p)
            ts.append(k * eta)
            ovl.append(a)
            if not np.all(np.isfinite(a)) or np.abs(a).max() > 1e6:
                diverged = True
                break
            if k_hit is None and np.sum(np.max(ovl, axis=0) > 0.5) >= n_stop:
                k_hit = k
            if k_hit is not None and k >= k_hit + 25 * rec:  # resolve the last crossing
                break
        if k == steps:
            break
        x = sample_inputs(batch, n, d, gen)
        pr = p.requires_grad_(True)
        loss = 0.5 * (forward(spec, spec.unflatten(pr), x) - teacher(x)).pow(2).mean()
        (g,) = torch.autograd.grad(loss, pr)
        p = (pr - eta * g).detach()
    ts, ovl = np.array(ts), np.array(ovl)
    out = dict(task=task, steps=k, diverged=diverged, t50=crossing_times(ts, ovl, 0.5),
               t10=crossing_times(ts, ovl, 0.1), t90=crossing_times(ts, ovl, 0.9),
               wall=time.time() - t0)
    if not diverged:
        q = conserved_quantities(spec, spec.unflatten(p))
        scale = max(c.pow(2).sum().item() for c in spec.unflatten(p))
        out.update(loss_end=prob.loss(p).item(), a_end=ovl[-1].tolist(),
                   q_drift=max((q[v] - q0[v]).abs().max().item() for v in q) / scale,
                   trq_change=[a - b for a, b in zip(prob.q_traces(p), trq0)],
                   eff_ranks=prob.eff_ranks(p))
    return out


def tasks_from_config(cfg):
    out = []
    for block in cfg["blocks"]:
        grid = itertools.product(block["sigmas"], block["seeds"], block["noise_seeds"],
                                 block["B"], block["eta"])
        for sigma, seed, ns, batch, eta in grid:
            out.append(dict(block=block["name"], tree=block["tree"], n=block["n"],
                            init=block.get("init", "fanin"), sigma=sigma, seed=seed,
                            noise_seed=ns, B=batch, eta=eta,
                            T=block["T"][sigma] if isinstance(block["T"], dict) else block["T"],
                            stop_after=block.get("stop_after", 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_known_args()[0]  # --raw-dir / --out-dir: see common.py
    cfg = yaml.safe_load(Path(args.config).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = tasks_from_config(cfg)
    done = set()
    for f in OUT.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            done.add(key_of(json.loads(line)["task"]))
    todo = [t for t in tasks if key_of(t) not in done]
    todo.sort(key=lambda t: -t["T"] / t["eta"] * (1 + t["B"] / 64))
    print(f"{len(tasks)} tasks, {len(todo)} to run", flush=True)
    files = {}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run, todo)):
            name = res["task"]["block"]
            if name not in files:
                files[name] = open(OUT / f"{name}.jsonl", "a")
            files[name].write(json.dumps(res, default=float) + "\n")
            files[name].flush()
            if i % 50 == 0:
                print(f"[{i + 1}/{len(todo)}] {res['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
