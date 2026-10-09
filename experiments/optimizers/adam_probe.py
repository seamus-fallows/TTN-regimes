"""Adam's internal state along a trajectory: what sets the time scale of later plateaus?

    uv run python -m experiments.optimizers.adam_probe experiments/optimizers/configs/adam_probe.yaml

For each run (same config format as run.py; optimizer must be adam or rmsprop) records at ~80
log-spaced steps, besides loss and overlaps:
  - the preconditioner P = sqrt(vhat) + eps (median, min, max over coordinates),
  - the step norm |theta_{n+1} - theta_n| and the gradient norm,
  - lambda_max(H) and lambda_max(P^-1/2 H P^-1/2), the preconditioned sharpness, which full-batch
    Adam drives to ~(2 + 2 beta1) / ((1 - beta1) lr) (Cohen et al. 2022, "adaptive edge of
    stability"); for RMSProp (no momentum) the threshold is 2 / lr,
  - the effective learning rate of a slow direction, lr / median(P), in GF time per step.
Writes /root/outputs/optimizers/<name>/<run_id>_probe.npz.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys

import numpy as np
import torch

from experiments.optimizers.run import OUT_ROOT, jobs_from_config, run_id


def probe(cfg: dict) -> dict:
    torch.set_num_threads(1)
    from experiments.optimizers.common import initial_point, make_problem
    from experiments.optimizers.optimizers import make_torch_opt

    p = make_problem(cfg)
    x0 = initial_point(p, cfg)
    n_max = int(cfg["n_steps"])
    probe_at = set(np.unique(np.round(np.geomspace(1, n_max, int(cfg.get("n_probe", 80)))).astype(int)))
    x = x0.clone().requires_grad_(True)
    opt = make_torch_opt(cfg, x)
    beta1 = cfg.get("beta1", 0.9) if cfg["optimizer"] == "adam" else 0.0
    eps = float(cfg.get("eps", 1e-8))
    out = {k: [] for k in ("step", "loss", "overlaps", "p_median", "p_min", "p_max", "step_norm",
                           "grad_norm", "sharpness", "precond_sharpness")}
    loss0 = p.loss(x0).item()
    fine_t, fine_a, fine_loss = [], [], []
    stride = max(1, n_max // 20000)
    for n in range(n_max + 1):
        opt.zero_grad()
        loss = p.loss(x)
        loss.backward()
        if n % stride == 0:
            with torch.no_grad():
                fine_t.append(n)
                fine_loss.append(loss.item())
                fine_a.append(p.target.component_overlaps(p.spec, p.spec.unflatten(x)).numpy())
        if loss.item() < cfg.get("tol", 1e-6) * loss0:
            break
        x_old = x.detach().clone()
        opt.step()
        if n in probe_at and n > 0:
            state = opt.state[x]
            v = state["exp_avg_sq"] if cfg["optimizer"] == "adam" else state["square_avg"]
            k = state["step"].item() if torch.is_tensor(state["step"]) else state["step"]
            bc2 = 1 - cfg.get("beta2", 0.999) ** k if cfg["optimizer"] == "adam" else 1.0
            pre = (v / bc2).sqrt() + eps
            hess = p.hessian(x_old)
            dinv = pre.rsqrt()
            ph = dinv[:, None] * hess * dinv[None, :]
            with torch.no_grad():
                out["step"].append(n)
                out["loss"].append(loss.item())
                out["overlaps"].append(p.target.component_overlaps(p.spec, p.spec.unflatten(x_old)).numpy())
                out["p_median"].append(pre.median().item())
                out["p_min"].append(pre.min().item())
                out["p_max"].append(pre.max().item())
                out["step_norm"].append((x.detach() - x_old).norm().item())
                out["grad_norm"].append(x.grad.norm().item())
                out["sharpness"].append(torch.linalg.eigvalsh(hess)[-1].item())
                out["precond_sharpness"].append(torch.linalg.eigvalsh(ph)[-1].item())
    res = {k: np.array(v) for k, v in out.items()}
    res.update(fine_t=np.array(fine_t), fine_loss=np.array(fine_loss), fine_overlaps=np.array(fine_a),
               threshold=np.array((2 + 2 * beta1) / ((1 - beta1) * cfg["lr"])),
               cfg=np.array(json.dumps(cfg)))
    return res


def work(args):
    cfg, out_dir = args
    path = out_dir / f"{run_id(cfg)}_probe.npz"
    if path.exists():
        return
    np.savez_compressed(path, **probe(cfg))
    print("done", cfg["tree"], cfg["optimizer"], cfg["lr"], cfg["sigma"], cfg["seed"], flush=True)


if __name__ == "__main__":
    name, jobs = jobs_from_config(sys.argv[1])
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(min(len(jobs), 24)) as pool:
        list(pool.imap_unordered(work, [(j, out_dir) for j in jobs]))
