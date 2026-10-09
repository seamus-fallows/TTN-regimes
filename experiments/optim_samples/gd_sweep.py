"""Step 1: gradient descent vs gradient flow, stability limit, edge of stability.

For every (tree, sigma, seed) a reference gradient flow is integrated (DOP853, rtol 1e-10,
atol 1e-14, exact event times), then GD from the same init at eta = ratio * eta_max with
eta_max = 2 / (K s_1^{2-2/K}). Time is t = eta * steps. Per run we store the loss and overlaps
(every step, subsampled on disk), sharpness / conservation-law drift / Q traces / effective bond ranks
at 41 times, and a JSON summary.

Run:  uv run python experiments/optim_samples/gd_sweep.py configs/gd_sweep.yaml [--workers 30]
Raw:  /root/outputs/optim_samples/gd/<tag>.{json,npz}; summary table: results/gd_sweep.csv
(written by analyze_gd.py). Resumable: finished runs are skipped.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

from common import RAW, Problem, crossing_times, eta_max, gradient_flow, save_json, setup
from ttn.network import conserved_quantities

OUT = RAW / "gd"


def tag_of(task):
    base = f"{task['tree']}{task['n']}_sig{task['sigma']}_seed{task['seed']}"
    if task.get("root_scale"):
        base += f"_root{task['root_scale']}"
    return base + ("_gf" if task["ratio"] == "gf" else f"_r{task['ratio']}")


def init_params(spec, task):
    scales = None
    if task.get("root_scale"):
        scales = [1.0] * spec.n_cores
        scales[spec.tree.root] = task["root_scale"] / task["sigma"]
    return spec.flatten(spec.init(task["sigma"], seed=task["seed"], core_scales=scales))


def run_gf(task):
    spec, teacher = setup(task["tree"], task["n"])
    prob = Problem(spec, teacher)
    theta0 = init_params(spec, task)
    q0 = conserved_quantities(spec, spec.unflatten(theta0))
    t0 = time.time()
    sol, ev = gradient_flow(prob, theta0, task["T"])
    a = np.array([prob.overlaps(torch.from_numpy(y)) for y in sol.y.T])
    loss = np.array([prob.loss(torch.from_numpy(y)).item() for y in sol.y.T])
    p_end = torch.from_numpy(sol.y[:, -1])
    summary = dict(task=task, eta_max=eta_max(spec, teacher), K=spec.n_cores, **ev,
                   loss_end=loss[-1], loss0=loss[0], a_end=a[-1].tolist(),
                   sharp_end=prob.sharpness(p_end), q_drift=prob.q_drift(p_end, q0),
                   q_traces_end=prob.q_traces(p_end), eff_ranks_end=prob.eff_ranks(p_end),
                   wall=time.time() - t0)
    np.savez_compressed(OUT / f"{tag_of(task)}.npz", t=sol.t, loss=loss, a=a)
    save_json(OUT / f"{tag_of(task)}.json", summary)


def run_gd(task):
    spec, teacher = setup(task["tree"], task["n"])
    prob = Problem(spec, teacher)
    p = init_params(spec, task)
    q0 = conserved_quantities(spec, spec.unflatten(p))
    em = eta_max(spec, teacher)
    eta = task["ratio"] * em
    steps = int(math.ceil(task["T"] / eta))
    checkpoints = set(np.linspace(0, steps, 41).astype(int).tolist())
    losses = np.empty(steps + 1)
    ovl = np.empty((steps + 1, teacher.n_components))
    probe = []
    t0 = time.time()
    loss0 = None
    diverged = False
    k_last = steps
    for k in range(steps + 1):
        loss, g = prob.loss_grad(p)
        loss0 = loss if loss0 is None else loss0
        losses[k] = loss
        ovl[k] = prob.overlaps(p)
        if not math.isfinite(loss) or loss > 1e6 * loss0:
            diverged, k_last = True, k
            break
        if k in checkpoints:
            probe.append(dict(t=k * eta, loss=loss, sharp=prob.sharpness(p),
                              q_drift=prob.q_drift(p, q0), q_traces=prob.q_traces(p),
                              eff_ranks=prob.eff_ranks(p), a=ovl[k].tolist(),
                              core_norms=[c.norm().item() for c in spec.unflatten(p)]))
        if k < steps:
            p = p - eta * g
    losses, ovl = losses[: k_last + 1], ovl[: k_last + 1]
    ts = eta * np.arange(k_last + 1)
    ev50 = crossing_times(ts, ovl, 0.5)
    ev10 = crossing_times(ts, ovl, 0.1)
    ev90 = crossing_times(ts, ovl, 0.9)
    # Count upward crossings of 1/2 per component (re-learning after a kick back to the saddle).
    n_cross = [int(np.sum((ovl[1:, r] > 0.5) & (ovl[:-1, r] <= 0.5))) for r in range(ovl.shape[1])]
    first = np.nanmin(ev50) if np.any(np.isfinite(ev50)) else float("inf")
    reentry = bool(np.any((ts > first) & (losses > 0.9 * loss0))) if math.isfinite(first) else False
    tail = losses[-max(10, len(losses) // 20):]
    summary = dict(task=task, eta=eta, eta_max=em, K=spec.n_cores, steps=k_last,
                   diverged=diverged, t50=ev50, t10=ev10, t90=ev90, n_cross=n_cross,
                   reentry=reentry, loss0=loss0, loss_end=float(losses[-1]),
                   loss_tail_mean=float(np.mean(tail)), loss_tail_min=float(np.min(tail)),
                   loss_tail_max=float(np.max(tail)), a_end=ovl[-1].tolist(), probe=probe,
                   wall=time.time() - t0)
    if not diverged:
        summary["sharp_end"] = prob.sharpness(p)
        summary["q_drift_end"] = prob.q_drift(p, q0)
        summary["q_traces_end"] = prob.q_traces(p)
        summary["eff_ranks_end"] = prob.eff_ranks(p)
        summary["spectra_end"] = prob.spectra(p)
    # Subsample the per-step record for disk: <= 20000 points plus the last 2000 steps in full.
    idx = np.unique(np.concatenate([np.linspace(0, k_last, min(20000, k_last + 1)).astype(int),
                                    np.arange(max(0, k_last - 2000), k_last + 1)]))
    np.savez_compressed(OUT / f"{tag_of(task)}.npz", t=ts[idx], loss=losses[idx], a=ovl[idx])
    save_json(OUT / f"{tag_of(task)}.json", summary)


def run(task):
    torch.set_num_threads(1)
    if (OUT / f"{tag_of(task)}.json").exists():
        return tag_of(task), "skip"
    t0 = time.time()
    (run_gf if task["ratio"] == "gf" else run_gd)(task)
    return tag_of(task), f"{time.time() - t0:.0f}s"


def tasks_from_config(cfg):
    out = []
    for block in cfg["blocks"]:
        for sigma, seed, ratio in itertools.product(block["sigmas"], block["seeds"],
                                                    ["gf"] + list(block["ratios"])):
            out.append(dict(tree=block["tree"], n=block["n"], sigma=sigma, seed=seed, ratio=ratio,
                            T=block["T"][sigma] if isinstance(block["T"], dict) else block["T"],
                            root_scale=block.get("root_scale")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks = tasks_from_config(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    # Longest runs first (smallest ratio, largest T) so the pool drains evenly.
    tasks.sort(key=lambda t: -(t["T"] / (1e-9 if t["ratio"] == "gf" else t["ratio"])))
    print(f"{len(tasks)} tasks", flush=True)
    with Pool(args.workers) as pool:
        for i, (tag, status) in enumerate(pool.imap_unordered(run, tasks)):
            print(f"[{i + 1}/{len(tasks)}] {tag} {status}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
