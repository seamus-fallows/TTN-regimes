"""Gradient flow on the empirical loss of a real-data problem; one trajectory .npz per case.

Usage: uv run python experiments/real_data/run.py <config.yaml> <out_dir> <workers>

Protocol (two phases, both `ttn.dynamics.integrate_gf`):
1. First plateau: Dormand-Prince 5(4) from theta_0 until the training loss has dropped by 1%
   (loss_floor = 0.99 L(0)); log grid from 1e-3 tau_0 (tau_0 = |theta_0| / |v(theta_0)|).
2. Everything after: LSODA (stiff once a component is learned) from there for a time T2, on its
   own clock tau = t - t_a (t_a = end of phase 1; t_a ~ sigma^-(K-2) is huge at small sigma, so
   absolute times cannot resolve later gaps in float64). Newton matrix: the exact Hessian (dense
   residual tensor, N <= 16) or its Gauss-Newton part (`hessian: gn`); LSODA's error control does
   not depend on it. Grid: log in tau from 1e-2 plus a uniform grid of T2 / n_linear spacing.
Saved arrays (npz): p1_* = phase-1 Trajectory data (absolute t), p2_* = phase 2 (t = tau, the
phase-2 clock), each with the library Recorder's log rows (t, loss, spectra = Frobenius bond spectra,
conservation_drift, speed, core_norms, ...) and fine rows (fine_t, fine_loss, fine_spectra), plus
test = (test loss, test acc, train acc, rms f on train) and fspectra / fine_fspectra (data-weighted
bond spectra, `data.function_spectra`); theta_end of each phase; theta0. Meta: case, t_a, stop.
Default Newton matrix: Gauss-Newton (`hessian: gn`); `dense` = exact (N <= 16 only).

The config lists `cases` (dicts; missing keys from `defaults`, then DEFAULTS), expanded over the
lists `sigmas` and `seeds`. Finished cases are skipped (resumable); every process uses one thread.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

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

sys.path.insert(0, str(Path(__file__).parent))
from data import DenseHessianProblem, GNProblem, Task, build, function_spectra  # noqa: E402

from ttn.dynamics import Recorder, integrate_gf, log_time_grid  # noqa: E402
from ttn.network import forward  # noqa: E402

TASK_KEYS = ("dataset", "size", "pos", "neg", "n_train", "n_test", "data_seed")
DEFAULTS = dict(dataset="mnist", size=4, pos=[3], neg=[5], n_train=4000, n_test=1600, data_seed=0,
                tree="binary", ordering="quadtree", chi=4, features="raw", centred=False,
                t2=1e4, n_per_decade=60, n_linear=1000, rtol=1e-8, atol=1e-14,
                hessian="gn", escape_drop=0.01, wall_limit=3600, tag="")


def case_name(c: dict) -> str:
    pos = "".join(map(str, c["pos"]))
    neg = "".join(map(str, c["neg"]))
    tag = f"_{c['tag']}" if c["tag"] else ""
    return (f"{c['dataset']}{c['size']}_{pos}v{neg}_n{c['n_train']}_{c['features']}_{c['tree']}_"
            f"{c['ordering']}_chi{c['chi']}{'_centred' if c['centred'] else ''}_"
            f"s{c['sigma']:.4g}_seed{c['seed']}{tag}")


def expand(config: dict) -> list[dict]:
    base = {**DEFAULTS, **config.get("defaults", {})}
    out = []
    for case in config["cases"]:
        c = {**base, **case}
        sigmas = c.pop("sigmas", [c.get("sigma")])
        seeds = c.pop("seeds", [c.get("seed", 0)])
        for sigma, seed in itertools.product(sigmas, seeds):
            out.append({**c, "sigma": float(sigma), "seed": int(seed)})
    return out


class RealRecorder(Recorder):
    """Library Recorder plus, on log rows, `test` = (test loss, test acc, train acc, rms f on train)
    and `fspectra` (data-weighted bond spectra on the training set, `data.function_spectra`), and
    `fine_fspectra` on fine rows (so `merged_rows(data, keys=("loss", "fspectra"))` resolves events)."""

    def __init__(self, rp):
        super().__init__(overlaps=False, extra=[("test", self._test), ("fspectra", self._fspec)])
        self.rp = rp

    def _test(self, prob, theta, t):
        spec, cores = prob.effective(theta)
        with torch.no_grad():
            f_te = forward(spec, cores, self.rp.x_test)
            f_tr = forward(spec, cores, prob.dataset[0])
        return np.array([
            0.5 * (f_te - self.rp.y_test).pow(2).mean().item(),
            (torch.sign(f_te) == self.rp.y_test).double().mean().item(),
            (torch.sign(f_tr) == prob.dataset[1]).double().mean().item(),
            f_tr.pow(2).mean().sqrt().item(),
        ])

    @staticmethod
    def _fspec(prob, theta, t):
        return function_spectra(*prob.effective(theta), prob.dataset[0])

    def _fine_row(self, problem, theta, t, loss):
        super()._fine_row(problem, theta, t, loss)
        self.fine_rows.setdefault("fine_fspectra", []).append(self._fspec(problem, theta, t))


def run_case(c: dict, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    name = case_name(c)
    path = out_dir / f"{name}.npz"
    if path.exists():
        return {"name": name, "status": "exists"}
    wall0 = time.time()
    task = Task(**{k: (tuple(c[k]) if k in ("pos", "neg") else c[k]) for k in TASK_KEYS})
    spec = build(task, c["tree"], c["ordering"], c["chi"], c["features"]).problem.spec
    cores0 = spec.init(c["sigma"], seed=c["seed"])
    rp = build(task, c["tree"], c["ordering"], c["chi"], c["features"],
               centred_cores0=cores0 if c["centred"] else None)
    cls = DenseHessianProblem if c["hessian"] == "dense" else GNProblem
    problem = cls(rp.problem.spec, rp.problem.teacher, dataset=rp.problem.dataset)
    theta0 = spec.flatten(cores0)
    meta = {"case": c, "name": name, "order": rp.order.tolist()}
    data = {"theta0": theta0.numpy()}

    # phase 1: first plateau, DP5 until the loss has dropped by escape_drop
    loss0 = problem.loss(theta0).item()
    tau0 = theta0.norm().item() / problem.velocity(theta0).norm().item()
    grid1 = log_time_grid(1e-3 * tau0, 1e6 * tau0, int(c["n_per_decade"] * 9))
    rec1 = RealRecorder(rp)
    tr1 = integrate_gf(problem, theta0, 1e6 * tau0, log_times=grid1, rtol=c["rtol"], atol=c["atol"],
                       recorder=rec1, method="dopri5", loss_floor=(1 - c["escape_drop"]) * loss0,
                       max_wall_time=c["wall_limit"])
    meta["phase1"] = {k: v for k, v in tr1.meta.items() if k != "problem"}
    data |= {f"p1_{k}": v for k, v in tr1.data.items()}
    stop = None if tr1.meta["stop_reason"] == "converged" else f"phase1_{tr1.meta['stop_reason']}"
    if stop is None:
        # phase 2: LSODA on its own clock tau = t - t_a
        meta["t_a"] = tr1.meta["t_end"]
        n_log = int(c["n_per_decade"] * np.log10(c["t2"] / 1e-2))
        grid2 = np.union1d(log_time_grid(1e-2, c["t2"], n_log),
                           np.linspace(0, c["t2"], c["n_linear"] + 1))
        rec2 = RealRecorder(rp)
        tr2 = integrate_gf(problem, torch.as_tensor(tr1["theta_end"]), c["t2"], log_times=grid2, rtol=c["rtol"],
                           atol=c["atol"], recorder=rec2, method="lsoda",
                           max_wall_time=max(60.0, c["wall_limit"] - (time.time() - wall0)))
        meta["phase2"] = {k: v for k, v in tr2.meta.items() if k != "problem"}
        data |= {f"p2_{k}": v for k, v in tr2.data.items()}
        stop = tr2.meta["stop_reason"]
    meta.update(stop=stop, loss0=loss0, tau0=tau0, wall_total=time.time() - wall0,
                n_params=spec.n_params, bond_dims=list(spec.bond_dims),
                edges=list(spec.tree.edges), parent=list(spec.tree.parent),
                sites=[list(s) for s in spec.tree.sites])
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta, default=str)), **data)
    return {"name": name, "status": stop, "wall": round(time.time() - wall0, 1)}


def _worker(args):
    c, out_dir = args
    try:
        return run_case(c, out_dir)
    except Exception:
        return {"name": case_name(c), "status": "error", "trace": traceback.format_exc()}


def main(config_path: str, out_dir: str, workers: int) -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = expand(config)
    print(f"{len(cases)} cases -> {out}", flush=True)
    with Pool(workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_worker, [(c, out) for c in cases]):
            print(json.dumps(res), flush=True)
            with open(out / "runs.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
