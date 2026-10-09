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
import json
import math
import os
import time
from types import SimpleNamespace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

from common import RAW, Problem, eta_max, gradient_flow, record_events, save_json, setup
from ttn.dynamics import Problem as LibProblem
from ttn.dynamics import Recorder, integrate_gf
from ttn.network import conserved_quantities
from ttn.teachers import ShiftedTeacher

OUT = RAW / "gd"


def tag_of(task):
    base = f"{task['tree']}{task['n']}_sig{task['sigma']}_seed{task['seed']}"
    if task.get("root_scale"):
        base += f"_root{task['root_scale']}"
    if task.get("centred"):
        base += "_centred"
    if task.get("chi", 3) != 3:
        base += f"_chi{task['chi']}"
    if task.get("ratio_ref", "teacher") == "gf" and task["ratio"] != "gf":
        base += "_relgf"
    return base + ("_gf" if task["ratio"] == "gf" else f"_r{task['ratio']}")


def init_params(spec, task):
    scales = None
    if task.get("root_scale"):
        scales = [1.0] * spec.n_cores
        scales[spec.tree.root] = task["root_scale"] / task["sigma"]
    return spec.flatten(spec.init(task["sigma"], seed=task["seed"], core_scales=scales))


def make_problem(task):
    spec, teacher = setup(task["tree"], task["n"], chi=task.get("chi", 3))
    theta0 = init_params(spec, task)
    cores0 = spec.unflatten(theta0) if task.get("centred") else None
    return spec, teacher, Problem(spec, teacher, cores0), theta0


def gf_lsoda(task, spec, teacher, theta0):
    """GF reference with `ttn.dynamics.integrate_gf(method='lsoda')` (stiff lazy-side runs);
    events from a dense log record (`record_events`)."""
    cores0 = spec.unflatten(theta0)
    target = ShiftedTeacher(teacher, spec, [c.clone() for c in cores0]) if task.get("centred") else teacher
    lib_prob = LibProblem(spec, target)
    t_max = task.get("T_gf", task["T"])
    floor = task.get("gf_loss_floor")
    traj = integrate_gf(lib_prob, theta0, t_max, method="lsoda", n_log=3000,
                        loss_floor=None if floor is None else floor,
                        recorder=Recorder(spectra=False, conservation=False))
    ev = record_events(traj["t"], traj["overlaps"])
    ev.update(nfev=int(traj.meta.get("n_evals", 0) or 0), t_end=float(traj.meta["t_end"]),
              integrator="lsoda")
    return traj["t"], traj["loss"], traj["overlaps"], torch.from_numpy(traj["theta_end"]), ev


def run_gf(task):
    spec, teacher, prob, theta0 = make_problem(task)
    q0 = conserved_quantities(spec, spec.unflatten(theta0))
    t0 = time.time()
    if task.get("gf_method") == "lsoda":
        t_rec, loss, a, p_end, ev = gf_lsoda(task, spec, teacher, theta0)
        sol = SimpleNamespace(t=t_rec)
    else:
        sol, ev = gradient_flow(prob, theta0, task.get("T_gf", task["T"]),
                                loss_floor=task.get("gf_loss_floor"))
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
    spec, teacher, prob, p = make_problem(task)
    q0 = conserved_quantities(spec, spec.unflatten(p))
    em = eta_max(spec, teacher)
    lam_gf = None
    if task.get("ratio_ref", "teacher") == "gf":
        # Step relative to this init's own GF state: eta = ratio x 2/lambda_GF.
        ref = json.loads((OUT / f"{tag_of(dict(task, ratio='gf'))}.json").read_text())
        lam_gf = ref["sharp_end"]
        eta = task["ratio"] * 2.0 / lam_gf
        if task.get("T_from_gf"):  # run as long as the GF reference took to converge (x factor)
            task = dict(task, T=task["T_from_gf"] * ref["t_end"])
    else:
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
    ev = record_events(ts, ovl)  # last-crossing events, final-rise widths (diagnostics.md)
    first = np.nanmin(ev["t50_first"]) if np.any(np.isfinite(ev["t50_first"])) else float("inf")
    reentry = bool(np.any((ts > first) & (losses > 0.9 * loss0))) if math.isfinite(first) else False
    tail = losses[-max(10, len(losses) // 20):]
    finite = losses[np.isfinite(losses)]
    max_rise = float(np.max(np.diff(finite)) / loss0) if len(finite) > 1 else 0.0
    summary = dict(task=task, eta=eta, eta_max=em, lam_gf=lam_gf, K=spec.n_cores, steps=k_last,
                   diverged=diverged, **ev, reentry=reentry, max_rise=max_rise,
                   loss0=loss0, loss_end=float(losses[-1]),
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
            task = dict(tree=block["tree"], n=block["n"], sigma=sigma, seed=seed, ratio=ratio,
                        T=block["T"][sigma] if isinstance(block["T"], dict) else block["T"],
                        root_scale=block.get("root_scale"), centred=block.get("centred", False))
            for key in ("chi", "ratio_ref", "gf_loss_floor", "T_gf", "gf_method", "T_from_gf"):
                if key in block:
                    val = block[key]
                    task[key] = val[sigma] if isinstance(val, dict) else val
            out.append(task)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 1))
    args = ap.parse_known_args()[0]  # --raw-dir / --out-dir: see common.py
    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks = tasks_from_config(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    # Longest runs first (smallest ratio, largest T) so the pool drains evenly.
    tasks.sort(key=lambda t: -(t["T"] / (1e-9 if t["ratio"] == "gf" else t["ratio"])))
    print(f"{len(tasks)} tasks", flush=True)
    # Two phases: GF references first (steps relative to lambda_GF need them), then GD.
    phases = [[t for t in tasks if t["ratio"] == "gf"], [t for t in tasks if t["ratio"] != "gf"]]
    i = 0
    with Pool(args.workers) as pool:
        for phase in phases:
            for tag, status in pool.imap_unordered(run, phase):
                i += 1
                print(f"[{i}/{len(tasks)}] {tag} {status}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
