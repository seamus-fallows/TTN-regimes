"""Backward-error prediction of the GD event-time shift, t_e^GD - t_e^GF = eta * c_e + O(eta^2).

GD with step eta follows, to O(eta^2), the modified flow theta' = -g - (eta/2) H g (g = grad L).
Writing theta_GD(t) = theta_GF(t) + eta * delta(t), the first-order deviation solves the variational
equation delta' = -H (delta + g/2), delta(0) = 0, along the GF path. An event a_r = 1/2 is then
shifted by c_e = -grad a_r . delta(t_e) / (d a_r/dt)(t_e). Also reported: the tangential
approximation c_e ~ (1/2) ln(|g(t_e)| / |g(0)|), which ignores the curvature of the path.

Run: uv run python experiments/optim_samples/predict_shift.py [--workers 8]
Reads the GF references in /root/outputs/optim_samples/gd/*_gf.json (event times) and writes
results/gd_shift_prediction.csv.
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

from gd_sweep import make_problem

RAW = Path("/root/outputs/optim_samples/gd")
OUT = Path(__file__).parent / "results"


def predict(path):
    torch.set_num_threads(1)
    ref = json.loads(Path(path).read_text())
    task = ref["task"]
    spec, teacher, prob, theta0 = make_problem(task)
    n_par = theta0.numel()

    def grad_and_hvp(theta, v):
        p = theta.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(prob.loss(p), p, create_graph=True)
        (hv,) = torch.autograd.grad(g @ v, p)
        return g.detach(), hv

    def rhs(t, z):
        th = torch.from_numpy(z[:n_par])
        de = torch.from_numpy(z[n_par:])
        g = prob.loss_grad(th)[1]
        _, hv = grad_and_hvp(th, de + 0.5 * g)
        return np.concatenate([-g.numpy(), -hv.numpy()])

    events = [t for t in ref["t50"] if t is not None]
    rows = []
    if not events:
        return rows
    z0 = np.concatenate([theta0.numpy(), np.zeros(n_par)])
    t_eval = sorted(events)
    sol = solve_ivp(rhs, (0, max(t_eval) * 1.0000001), z0, method="DOP853", rtol=1e-10,
                    atol=1e-14, t_eval=t_eval)
    g0 = prob.loss_grad(theta0)[1].norm().item()
    for comp, t_e in enumerate(ref["t50"]):
        if t_e is None:
            continue
        i = t_eval.index(t_e)
        th = torch.from_numpy(sol.y[:n_par, i])
        de = torch.from_numpy(sol.y[n_par:, i])
        p = th.detach().requires_grad_(True)
        a = teacher.component_overlaps(*prob.learned(p))[comp]
        (grad_a,) = torch.autograd.grad(a, p)
        g = prob.loss_grad(th)[1]
        rate = -(grad_a @ g).item()
        rows.append(dict(tree=task["tree"], n=task["n"], sigma=task["sigma"], seed=task["seed"],
                         root_scale=task.get("root_scale"), comp=comp, t_e=t_e,
                         a_check=a.item(), pred_shift_over_eta=-(grad_a @ de).item() / rate,
                         tangential_shift_over_eta=0.5 * np.log(g.norm().item() / g0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = sorted(str(p) for p in RAW.glob("*_gf.json"))
    with Pool(args.workers) as pool:
        rows = [r for res in pool.imap_unordered(predict, paths) for r in res]
    df = pd.DataFrame(rows).sort_values(["tree", "n", "sigma", "root_scale", "seed", "comp"])
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "gd_shift_prediction.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
