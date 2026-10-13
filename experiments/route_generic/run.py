"""Gradient flow from small random init on the synthetic targets; one .npz per run, every solver step.

usage: uv run python experiments/route_generic/run.py <config.yaml> <out_dir> <workers> [--only SUBSTR]

Integrator (notes/saddle_dyn.md section 8): scipy's LSODA class stepped directly (its dense output is
not used), exact Hessian as Jacobian, rtol/atol from the config. Every accepted step is recorded:
absolute time, the clock tau (re-zeroed at the first escape, L < L0/2, so that later gaps are resolved
in float64), loss, top bond singular values of every edge, target amplitudes <U_a, W> (CP targets).
The solver clock is re-zeroed (fresh solver, first step 1e-2) when the step falls below 1e-11 of the
local time; a failed step is retried from the last good step (at most 5 times). Stops: L < stop_rel L0
("converged"), tau > tau_max ("unfinished"), wall clock ("wall"), max_steps ("stalled").
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import itertools
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.integrate import LSODA
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from common import build_target, dense_teacher, get_tree, target_ranks, teacher_spectra  # noqa: E402

from ttn.network import to_dense  # noqa: E402
from ttn.teachers import CPTeacher  # noqa: E402

torch.set_num_threads(1)
RESTART_H = 1e-2
DEFAULTS = dict(rtol=1e-10, atol=1e-15, tau_max=1e8, stop_rel=1e-7, wall=1800.0, max_steps=400_000)


def case_name(c: dict) -> str:
    return (f"{c['target']}_{c['tree']}_d{c['d']}_t{c.get('teacher_seed', 0)}_"
            f"s{np.log10(c['sigma']):+.2f}_seed{c['seed']}")


def expand(config: dict) -> list[dict]:
    base = {**DEFAULTS, **config.get("defaults", {})}
    out = []
    for case in config["cases"]:
        c = {**base, **case}
        sigmas = c.pop("sigmas")
        seeds = c.pop("seeds")
        tseeds = c.pop("teacher_seeds", [0])
        for ts, sigma, seed in itertools.product(tseeds, sigmas, seeds):
            out.append({**c, "teacher_seed": int(ts), "sigma": float(sigma), "seed": int(seed)})
    return out


class Model:
    def __init__(self, case):
        self.spec, self.teacher, _ = build_target(case)
        self.tree = self.spec.tree
        self.w_star = dense_teacher(self.teacher).detach()
        self.ref = teacher_spectra(self.w_star, self.tree)
        self.ranks = target_ranks(self.ref)
        self.keep = {e: min(len(self.ref[e]), self.ranks[e] + 1) for e in self.tree.edges}
        self.factors = self.teacher.factors if isinstance(self.teacher, CPTeacher) else None
        self.n = self.tree.n_sites
        self.d = self.spec.phys_dims[0]
        self.cut_perm = {}
        for e in self.tree.edges:
            below = sorted(self.tree.subtree_sites[e])
            rest = [s for s in range(self.n) if s not in below]
            self.cut_perm[e] = (below + rest, self.d ** len(below))

    def W(self, theta: torch.Tensor) -> torch.Tensor:
        return to_dense(self.spec, self.spec.unflatten(theta))

    def loss_t(self, theta: torch.Tensor) -> torch.Tensor:
        e = self.W(theta) - self.w_star
        return 0.5 * (e * e).sum()

    def loss_grad(self, y: np.ndarray):
        th = torch.from_numpy(y).requires_grad_(True)
        loss = self.loss_t(th)
        (g,) = torch.autograd.grad(loss, th)
        return float(loss.detach()), g.numpy()

    def hessian(self, y: np.ndarray) -> np.ndarray:
        return torch.func.hessian(self.loss_t)(torch.from_numpy(y)).numpy()

    def measure(self, y: np.ndarray):
        with torch.no_grad():
            w = self.W(torch.from_numpy(y))
            spec = []
            for e in self.tree.edges:
                perm, rows = self.cut_perm[e]
                sv = torch.linalg.svdvals(w.permute(perm).reshape(rows, -1))
                spec.append(sv[: self.keep[e]])
            amps = None
            if self.factors is not None:
                a = w.unsqueeze(0).expand(self.factors.shape[0], *w.shape)
                for i in range(self.n):
                    a = torch.einsum("rj...,rj->r...", a, self.factors[:, i, :])
                amps = a.numpy()
            e = w - self.w_star
            return 0.5 * float((e * e).sum()), torch.cat(spec).numpy(), amps


def run_case(case: dict, out_dir: Path) -> dict:
    name = case_name(case)
    path = out_dir / f"{name}.npz"
    if path.exists():
        return {"name": name, "status": "skipped"}
    wall0 = time.time()
    with threadpool_limits(1):
        model = Model(case)
        theta0 = torch.cat([c.reshape(-1) for c in model.spec.init(case["sigma"], seed=case["seed"])])
        y0 = theta0.numpy().copy()
        cache = {}

        def evalf(y):
            key = y.tobytes()
            if key not in cache:
                if len(cache) > 8:
                    cache.clear()
                cache[key] = model.loss_grad(y)
            return cache[key]

        def rhs(t, y):
            return -evalf(y)[1]

        def jac(t, y):
            return -model.hessian(y)

        rtol, atol = case["rtol"], case["atol"]
        L0, sp0, am0 = model.measure(y0)
        T_abs, TAU, LOSS, SPEC, AMPS, METH = [0.0], [np.nan], [L0], [sp0], [am0], [0]
        TLOC, SEG = [0.0], [0]  # solver-local time and solver segment: exact differences within a segment
        seg = 0
        off_abs, off_tau, phase = 0.0, 0.0, 0
        fails, restarts, status = [], 0, "running"
        t_a = None
        y_a = None
        solver = LSODA(rhs, 0.0, y0, np.inf, rtol=rtol, atol=atol, jac=jac)
        t_loc_old, y_old = 0.0, y0
        n_steps = 0
        while True:
            if time.time() - wall0 > case["wall"]:
                status = "wall"
                break
            if n_steps >= case["max_steps"]:
                status = "stalled"
                break
            msg = solver.step()
            if solver.status == "failed":
                fails.append(float(off_abs + t_loc_old))
                if len(fails) > 5:
                    status = "failed: " + str(msg)
                    break
                off_abs += t_loc_old
                if phase:
                    off_tau += t_loc_old
                t_loc_old = 0.0
                seg += 1
                solver = LSODA(rhs, 0.0, y_old, np.inf, rtol=rtol, atol=atol, jac=jac, first_step=RESTART_H)
                continue
            n_steps += 1
            t_loc, y = solver.t, np.array(solver.y, copy=True)
            L, sp, am = model.measure(y)
            T_abs.append(off_abs + t_loc)
            TAU.append(off_tau + t_loc if phase else np.nan)
            TLOC.append(t_loc)
            SEG.append(seg)
            LOSS.append(L)
            SPEC.append(sp)
            AMPS.append(am)
            try:
                METH.append(int(solver._lsoda_solver._integrator.iwork[19]))
            except Exception:  # noqa: BLE001
                METH.append(-1)
            t_loc_old, y_old = t_loc, y
            if L < case["stop_rel"] * L0:
                status = "converged"
                break
            if phase and off_tau + t_loc > case["tau_max"]:
                status = "unfinished"
                break
            restart = False
            if phase == 0 and L < 0.5 * L0:
                # first escape: the tau clock starts here (exact re-zero of the solver)
                phase = 1
                t_a = off_abs + t_loc
                y_a = y.copy()
                TAU[-1] = 0.0
                off_tau = -t_loc
                restart = True
            elif t_loc > 0 and solver.step_size < 1e-11 * t_loc:
                restart = True
            if restart:
                off_abs += t_loc
                if phase:
                    off_tau += t_loc
                t_loc_old = 0.0
                restarts += 1
                seg += 1
                solver = LSODA(rhs, 0.0, y, np.inf, rtol=rtol, atol=atol, jac=jac, first_step=RESTART_H)
        amps = np.array(AMPS) if AMPS[0] is not None else np.zeros((len(LOSS), 0))
        np.savez_compressed(
            path,
            t=np.array(T_abs), tau=np.array(TAU), loss=np.array(LOSS), spectra=np.array(SPEC),
            amps=amps, method=np.array(METH, dtype=np.int8), tloc=np.array(TLOC), seg=np.array(SEG), theta0=y0, theta_end=y_old,
            theta_a=y_a if y_a is not None else np.zeros(0),
            meta=json.dumps(dict(
                case=case, name=name, status=status, t_a=t_a, L0=L0, n_steps=n_steps, restarts=restarts,
                fails=fails, wall=time.time() - wall0,
                edges=list(model.tree.edges), keep=[model.keep[e] for e in model.tree.edges],
                ref={str(e): model.ref[e].tolist() for e in model.tree.edges},
                ranks={str(e): model.ranks[e] for e in model.tree.edges},
                weights=model.teacher.weights.tolist() if model.factors is not None else None,
                parent=list(model.tree.parent), chi=list(model.spec.bond_dims),
            )),
        )
    return {"name": name, "status": status, "steps": n_steps, "restarts": restarts, "fails": len(fails),
            "wall": round(time.time() - wall0, 1), "t_a": t_a}


def _work(args):
    case, out_dir = args
    try:
        return run_case(case, Path(out_dir))
    except Exception:  # noqa: BLE001
        return {"name": case_name(case), "status": "error", "error": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("workers", type=int)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    config = yaml.safe_load(open(args.config))
    cases = expand(config)
    if args.only:
        cases = [c for c in cases if args.only in case_name(c)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if not (out / f"{case_name(c)}.npz").exists()]
    # longest first: smallest sigma first, deepest trees first
    todo.sort(key=lambda c: (c["sigma"], c["tree"] != "binary8"))
    print(f"{len(cases)} cases, {len(todo)} to run -> {out}", flush=True)
    with Pool(args.workers, maxtasksperchild=4) as pool, open(out / "runs.jsonl", "a") as log:
        for res in pool.imap_unordered(_work, [(c, str(out)) for c in todo]):
            log.write(json.dumps(res) + "\n")
            log.flush()
            print(json.dumps(res)[:300], flush=True)


if __name__ == "__main__":
    main()
