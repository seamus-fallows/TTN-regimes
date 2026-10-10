"""Reviewer checks of the optim_samples GD claims (independent minimal re-implementation).

mode "edge":    binary N=4, fan-in init, ratios close to eta_max (0.9, 0.97, 0.99, 1.01, 1.03):
                is the GD stability boundary at eta_max to +-1%?
mode "centred": centred model, sigma 1.4 / 2.0, ratios above 2/lambda_GF, run 10x longer than the
                agent (T = 20): do runs classed 'reentry' / 'oscillating' end up converging (catapult)?
                Also a long GF reference (DOP853) for the converged sharpness.
Writes JSON lines to data/review_optim_samples/<mode>.jsonl.
"""

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)

OUT = Path("c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optim_samples")


def setup(task):
    from ttn.network import TTNSpec
    from ttn.teachers import odeco_teacher
    from ttn.tree import make_tree

    spec = TTNSpec.uniform(make_tree("binary", 4), 3, 3)
    teacher = odeco_teacher(4, 3, [3.0, 2.0, 1.0], seed=101)
    theta0 = spec.flatten(spec.init(task["sigma"], seed=task["seed"]))
    return spec, teacher, theta0


def make_loss(spec, teacher, theta0, centred):
    from ttn.losses import centred_population_loss, population_loss

    t2 = teacher.norm2()
    cores0 = [c.detach() for c in spec.unflatten(theta0)]
    if centred:
        return lambda p: centred_population_loss(
            spec, spec.unflatten(p), cores0, teacher, base_norm2=t2
        )
    return lambda p: population_loss(spec, spec.unflatten(p), teacher, teacher_norm2=t2)


def overlaps(spec, teacher, theta0, p, centred):
    from ttn.network import sum_difference_cores, two_copy_cores

    with torch.no_grad():
        if centred:
            s2, c2 = two_copy_cores(
                spec, spec.unflatten(p), [c.detach() for c in spec.unflatten(theta0)]
            )
            return teacher.component_overlaps(s2, sum_difference_cores(s2, c2)).numpy()
        return teacher.component_overlaps(spec, spec.unflatten(p)).numpy()


def sharpness(loss, p):
    h = torch.autograd.functional.hessian(loss, p.detach())
    return torch.linalg.eigvalsh(h)[-1].item()


def run(task):
    torch.set_num_threads(1)
    spec, teacher, p = setup(task)
    centred = task.get("centred", False)
    theta0 = p.clone()
    loss = make_loss(spec, teacher, theta0, centred)
    k = spec.n_cores
    eta_max = 2.0 / (k * 3.0 ** (2 - 2 / k))
    t0 = time.time()
    if task["ratio"] == "gf":

        def rhs(t, y):
            q = torch.from_numpy(y).requires_grad_(True)
            (g,) = torch.autograd.grad(loss(q), q)
            return -g.numpy()

        sol = solve_ivp(
            rhs,
            (0, task["T"]),
            p.numpy(),
            method="DOP853",
            rtol=1e-9,
            atol=1e-12,
            t_eval=np.linspace(0, task["T"], 401),
        )
        losses = [loss(torch.from_numpy(y)).item() for y in sol.y.T]
        pe = torch.from_numpy(sol.y[:, -1])
        out = dict(
            task=task,
            loss0=losses[0],
            loss_end=losses[-1],
            loss_at={
                str(tt): l
                for tt, l in zip(
                    [2, 3, 5, 10, 20],
                    [
                        losses[int(round(x / task["T"] * 400))]
                        for x in (2, 3, 5, 10, 20)
                        if x <= task["T"]
                    ],
                )
            },
            sharp_end=sharpness(loss, pe),
            a_end=overlaps(spec, teacher, theta0, pe, centred).tolist(),
            wall=time.time() - t0,
        )
        return out
    eta = task["ratio"] * eta_max
    steps = int(math.ceil(task["T"] / eta))
    losses = np.empty(steps + 1)
    loss0 = None
    diverged = False
    for i in range(steps + 1):
        q = p.detach().requires_grad_(True)
        lv = loss(q)
        (g,) = torch.autograd.grad(lv, q)
        losses[i] = lv.item()
        loss0 = losses[i] if loss0 is None else loss0
        if not math.isfinite(losses[i]) or losses[i] > 1e6 * loss0:
            diverged = True
            losses = losses[: i + 1]
            break
        if i < steps:
            p = p.detach() - eta * g
    n = len(losses)
    tail = losses[-max(20, min(2000, n // 5)) :]
    tv = float(np.sum(np.abs(np.diff(tail))))
    net = float(abs(tail[-1] - tail[0]))
    out = dict(
        task=task,
        eta=eta,
        eta_max=eta_max,
        steps=n - 1,
        diverged=diverged,
        loss0=loss0,
        loss_end=float(losses[-1]),
        tail_mean=float(np.mean(tail)),
        tail_min=float(np.min(tail)),
        tail_max=float(np.max(tail)),
        tail_tv=tv,
        tail_net=net,
        max_loss_after_start=float(np.max(losses[n // 10 :])) if n > 10 else float("nan"),
        wall=time.time() - t0,
    )
    # loss trajectory at 10 checkpoints (for catapult detection)
    idx = np.linspace(0, n - 1, 11).astype(int)
    out["loss_traj"] = [[float(i * eta), float(losses[i])] for i in idx]
    if not diverged:
        out["sharp_end"] = sharpness(loss, p)
        out["stab_end"] = out["sharp_end"] * eta / 2
        out["a_end"] = overlaps(spec, teacher, theta0, p, centred).tolist()
    return out


def tasks(mode):
    out = []
    if mode == "edge":
        for sigma, T in ((0.03, 300.0), (0.1, 100.0)):
            for seed in range(5):
                for ratio in (0.9, 0.97, 0.99, 1.01, 1.03):
                    out.append(dict(sigma=sigma, seed=seed, ratio=ratio, T=T))
    elif mode == "centred":
        for seed in range(5):
            out.append(dict(sigma=2.0, seed=seed, ratio="gf", T=20.0, centred=True))
            out.append(dict(sigma=1.4, seed=seed, ratio="gf", T=20.0, centred=True))
            for ratio in (0.2, 0.3, 0.5, 0.7):
                out.append(dict(sigma=2.0, seed=seed, ratio=ratio, T=20.0, centred=True))
            for ratio in (0.7, 0.9, 1.05):
                out.append(dict(sigma=1.4, seed=seed, ratio=ratio, T=20.0, centred=True))
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{mode}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            done.add(json.dumps(json.loads(line)["task"], sort_keys=True))
    todo = [t for t in tasks(mode) if json.dumps(t, sort_keys=True) not in done]
    print(len(todo), "tasks", flush=True)
    with Pool(workers) as pool, open(path, "a") as f:
        for res in pool.imap_unordered(run, todo):
            f.write(json.dumps(res) + "\n")
            f.flush()
            tk = res["task"]
            print(tk, res.get("loss_end"), res.get("stab_end"), f"{res['wall']:.0f}s", flush=True)
