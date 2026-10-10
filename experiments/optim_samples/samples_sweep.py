"""Step 3: finite training sets. Test error vs n for small init (plain model) and large init (centred).

Binary N = 8, d = 2, odeco teacher [3, 2] (teacher seed 101, rank 2 on every bond), student chi = 4.
A training set of n Gaussian inputs (data seed 500 + seed, labels f*(x), no noise) is fitted by
gradient descent on the empirical loss with step eta = 1/lambda_max (re-estimated every 200 steps by
power iteration; time t = sum of steps); stops when the train loss falls below `train_tol` x its
initial value, or at t_max / max_steps. Test error = exact population loss
1/2 ||W - W*||^2 (centred model: 1/2 ||W - W_0 - W*||^2, computed cancellation-free).

Also computed for every (seed, n): the lazy limit, i.e. the minimum-norm solution of the model
linearized at theta_0 (kernel regression with the initial tangent kernel), which the centred model
approaches as sigma -> infinity; it depends on the init only through its direction.

Run:  uv run python experiments/optim_samples/samples_sweep.py configs/samples_sweep.yaml [--workers 30]
Raw:  /root/outputs/optim_samples/samples/<block>.jsonl (resumable).
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

import torch
import yaml

from common import RAW
from ttn.losses import centred_population_loss, population_loss, sample_inputs
from ttn.network import (
    TTNSpec,
    bond_spectra,
    forward,
    sum_difference_cores,
    to_dense,
    two_copy_cores,
)
from ttn.teachers import odeco_teacher
from ttn.tree import make_tree

torch.set_default_dtype(torch.float64)
OUT = RAW / "samples"


def make_problem(task):
    spec = TTNSpec.uniform(make_tree(task["tree"], task["N"]), task["d"], task["chi"])
    teacher = odeco_teacher(task["N"], task["d"], task["weights"], seed=task["teacher_seed"])
    g = torch.Generator().manual_seed(500 + task["seed"])
    x = sample_inputs(task["n_train"], task["N"], task["d"], g)
    return spec, teacher, x, teacher(x)


def model_out(spec, cores, cores0, x, centred):
    if not centred:
        return forward(spec, cores, x)
    spec2, c2 = two_copy_cores(spec, cores, cores0)
    return forward(spec2, sum_difference_cores(spec2, c2), x)


def learned_dense(spec, cores, cores0, centred):
    if not centred:
        return to_dense(spec, cores)
    spec2, c2 = two_copy_cores(spec, cores, cores0)
    return to_dense(spec2, sum_difference_cores(spec2, c2))


def spectra_of(spec, cores, cores0, centred):
    if not centred:
        return bond_spectra(spec, cores)
    spec2, c2 = two_copy_cores(spec, cores, cores0)
    return bond_spectra(spec2, sum_difference_cores(spec2, c2))


def rank_stats(spectra, thr):
    eff = [int((s > thr).sum()) for _, s in sorted(spectra.items())]
    ent = []
    for _, s in sorted(spectra.items()):
        p = s**2 / (s**2).sum()
        p = p[p > 0]
        ent.append(float(torch.exp(-(p * p.log()).sum())))
    return eff, ent


def lazy_limit(spec, cores0, x, y, w_star):
    """Min-norm solution of (Phi_X J_0) dtheta = y; returns test error and the learned tensor."""
    flat0 = spec.flatten(cores0)
    jac = torch.autograd.functional.jacobian(
        lambda p: to_dense(spec, spec.unflatten(p)).reshape(-1), flat0
    )
    phi = x[:, 0]
    for i in range(1, x.shape[1]):
        phi = (phi.unsqueeze(-1) * x[:, i].unsqueeze(1)).reshape(x.shape[0], -1)
    a = phi @ jac
    dtheta = torch.linalg.lstsq(a, y.unsqueeze(-1), rcond=1e-12, driver="gelsd").solution[:, 0]
    w = (jac @ dtheta).reshape(w_star.shape)
    rank_a = int(torch.linalg.matrix_rank(a, rtol=1e-10))
    rank_j = int(torch.linalg.matrix_rank(jac, rtol=1e-10))
    return 0.5 * (w - w_star).pow(2).sum().item(), w, rank_a, rank_j


def dense_spectra(w, tree):
    out = {}
    for v in tree.edges:
        below = list(tree.subtree_sites[v])
        rest = [i for i in range(w.ndim) if i not in below]
        m = w.permute(below + rest).reshape(math.prod(w.shape[i] for i in below), -1)
        out[v] = torch.linalg.svdvals(m)
    return out


def top_eig(loss_fn, p, v0, iters=20):
    """Largest Hessian eigenvalue of loss_fn at p by power iteration with HVPs (warm start v0)."""
    p = p.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(loss_fn(p), p, create_graph=True)
    v = v0 / v0.norm()
    lam = 0.0
    for _ in range(iters):
        (hv,) = torch.autograd.grad(g @ v, p, retain_graph=True)
        lam = (hv @ v).item()
        v = hv / hv.norm()
    return abs(lam), v.detach()


def run(task):
    """Gradient descent with step eta = c / lambda_max, lambda_max re-estimated every `every`
    steps (power iteration), so plateaus are crossed fast and the step stays stable."""
    torch.set_num_threads(1)
    t_start = time.time()
    spec, teacher, x, y = make_problem(task)
    cores0 = [c.detach() for c in spec.init(task["sigma"], seed=task["seed"])]
    centred = task["centred"]
    flat0 = spec.flatten(cores0)
    y_scale = 0.5 * y.pow(2).mean().item()
    t2 = teacher.norm2()

    def train_loss(p):
        return 0.5 * (model_out(spec, spec.unflatten(p), cores0, x, centred) - y).pow(2).mean()

    def test_loss(p):
        cores = spec.unflatten(p)
        if centred:
            return centred_population_loss(spec, cores, cores0, teacher, base_norm2=t2).item()
        return population_loss(spec, cores, teacher, teacher_norm2=t2).item()

    p = flat0.clone()
    v = torch.randn(p.shape, generator=torch.Generator().manual_seed(task["seed"]))
    c, every = task.get("lr_factor", 1.0), task.get("every", 200)
    eta_lam = None
    t, k, status = 0.0, 0, "max_steps"
    curve_t, curve_train, curve_test = [], [], []
    next_rec = 0.0
    lam_prev = None
    while k < task["max_steps"]:
        if k % every == 0:
            lam, v = top_eig(train_loss, p, v)
            lam_prev = max(lam, 1e-300)
            eta_lam = c / lam_prev
        pr = p.detach().requires_grad_(True)
        loss = train_loss(pr)
        (g,) = torch.autograd.grad(loss, pr)
        lv = loss.item()
        if not math.isfinite(lv):
            status = "diverged"
            break
        if t >= next_rec:
            curve_t.append(t)
            curve_train.append(lv / y_scale)
            curve_test.append(test_loss(p))
            next_rec = max(1e-3, t * 1.15)
        if lv < task["train_tol"] * y_scale:
            status = "converged"
            break
        if t > task["t_max"]:
            status = "t_max"
            break
        # Near the origin the Hessian vanishes and 1/lambda is huge: also cap the relative
        # parameter change per step at rel_step (the local time scale |theta|/|grad|).
        eta = min(eta_lam, task["rel_step"] * p.norm().item() / g.norm().item(), task["eta_cap"])
        p = (pr - eta * g).detach()
        t += eta
        k += 1
    cores = spec.unflatten(p)
    test = test_loss(p)
    thr = 0.1 * min(task["weights"])
    eff, ent = rank_stats(spectra_of(spec, cores, cores0, centred), thr)
    rel_change = ((p - flat0).norm() / flat0.norm()).item()
    out = dict(task=task, status=status, t_end=t, steps=k, lam_end=lam_prev,
               train_loss=train_loss(p).item() / y_scale, test=test,
               test_rel=test / (0.5 * t2.item()), eff_ranks=eff, entropy_ranks=ent,
               rel_param_change=rel_change, curve_t=curve_t, curve_train=curve_train,
               curve_test=curve_test)
    if task.get("lazy"):
        lazy_test, w_lazy, rank_a, rank_j = lazy_limit(spec, cores0, x, y, teacher.dense())
        eff_l, ent_l = rank_stats(dense_spectra(w_lazy, spec.tree), thr)
        out.update(lazy_test=lazy_test, lazy_test_rel=lazy_test / (0.5 * t2.item()),
                   lazy_eff_ranks=eff_l, lazy_entropy_ranks=ent_l, rank_A=rank_a,
                   rank_J=rank_j)
    out["wall"] = time.time() - t_start
    return out


def key_of(task):
    return "|".join(f"{k}={task[k]}" for k in sorted(task))


def tasks_from_config(cfg):
    """One task per (n_train, sigma, seed) of every block; block keys override `base`."""
    base = cfg["base"]
    grid_keys = ("name", "n_train", "sigmas", "seeds")
    out = []
    for block in cfg["blocks"]:
        extra = {k: v for k, v in block.items() if k not in grid_keys}
        for n_train, sigma, seed in itertools.product(block["n_train"], block["sigmas"],
                                                      block["seeds"]):
            t = dict(base, **extra)
            t.update(block=block["name"], n_train=n_train, sigma=sigma, seed=seed)
            t.setdefault("lazy", False)
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 1))
    args = ap.parse_known_args()[0]  # --raw-dir / --out-dir: see common.py
    cfg = yaml.safe_load(Path(args.config).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = tasks_from_config(cfg)
    done = set()
    for f in OUT.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            done.add(key_of(json.loads(line)["task"]))
    todo = [t for t in tasks if key_of(t) not in done]
    todo.sort(key=lambda t: -t["n_train"] * t["chi"] * (1 + 2 * t["centred"]))
    print(f"{len(tasks)} tasks, {len(todo)} to run", flush=True)
    files = {}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run, todo)):
            name = res["task"]["block"]
            if name not in files:
                files[name] = open(OUT / f"{name}.jsonl", "a")
            files[name].write(json.dumps(res, default=float) + "\n")
            files[name].flush()
            print(f"[{i + 1}/{len(todo)}] {name} n={res['task']['n_train']} "
                  f"sigma={res['task']['sigma']} {res['status']} test_rel={res['test_rel']:.3g} "
                  f"{res['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
