"""Compute-cost benchmark for the compute plan (README.md, section "Benchmark").

    uv run python experiments/00_smoke/benchmark.py [--part grad|gf|both] [--workers 8]

Trees binary N = 4, 8, 16 and MPS N = 8, 16; d = 2, 3, 4; chi = 2, 4, 8 (bonds capped at the
rank their bipartition allows). Teacher: odeco with R = min(3, d, chi) components, weights
3, 2, 1 (teacher seed 0).

Part 1 (this process, one torch thread): median wall time (and mean CPU time) of one
training-loss gradient (what the integrators evaluate: 6 per DP5 step), one Recorder log entry
(loss, speed, spectra, overlaps, conservation) and one exact Hessian (LSODA's Jacobian), at a
fan-in init with sigma = 0.5. -> results/benchmark_grad.csv
Part 2 (ttn.runner sweep): one gradient-flow run per tree, d and chi, at a fixed function-space
init scale sigma^K = 1e-3, with LSODA (t_max = 1e6, stop at L < 1e-10), and for d = 3 also with
DP5 capped at 5000 steps (DP5 becomes stiffness-limited after the first escape; its cost to cross
the first plateau is read off the trajectory). -> results/benchmark_gf.csv; raw runs in
data/00_smoke/benchmark/ (resumable).
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ttn.config import RunConfig
from ttn.dynamics import Recorder
from ttn.runner import build, load_run, make_spec, run_sweep

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TREES = [("binary", 4), ("binary", 8), ("binary", 16), ("mps", 8), ("mps", 16)]
DS, CHIS = (2, 3, 4), (2, 4, 8)


def config(kind: str, n: int, d: int, chi: int) -> RunConfig:
    """The benchmark problem; sigma is set per tree in part 2."""
    n_comp = min(3, d, chi)
    return RunConfig(name=f"bench__{kind}{n}__d={d}__chi={chi}", d=d, chi=chi).with_overrides(
        {
            "tree.kind": kind,
            "tree.n_sites": n,
            "teacher.weights": [3.0, 2.0, 1.0][:n_comp],
            "teacher.seed": 0,
            "init.sigma": 0.5,
        }
    )


def time_call(fn, min_time: float = 0.3, min_reps: int = 5) -> tuple[float, float]:
    """(median wall time, mean CPU time) per call, after one warm-up call. CPU time is averaged
    over all calls because the Windows process clock ticks only every ~16 ms."""
    fn()
    walls, cpu0, total = [], time.process_time(), 0.0
    while len(walls) < min_reps or total < min_time:
        w = time.perf_counter()
        fn()
        walls.append(time.perf_counter() - w)
        total += walls[-1]
    return float(np.median(walls)), (time.process_time() - cpu0) / len(walls)


def part_grad() -> pd.DataFrame:
    torch.set_num_threads(1)
    rows = []
    for (kind, n), d, chi in [(t, d, c) for t in TREES for d in DS for c in CHIS]:
        setup = build(config(kind, n, d, chi))
        problem, theta = setup.problem, setup.theta0
        rec = Recorder()
        rec.start(problem, theta)
        grad = time_call(partial(problem.loss_and_velocity, theta))
        record = time_call(partial(rec.record, problem, theta, 0.0), min_reps=3)
        hess = time_call(partial(problem.hessian, theta), min_time=0.5, min_reps=2)
        rows.append(
            {
                "tree": f"{kind}{n}",
                "K": setup.spec.n_cores,
                "d": d,
                "chi": chi,
                "bond_dims": "-".join(str(b) for b in setup.spec.bond_dims),
                "n_params": setup.spec.n_params,
                "grad_ms": 1e3 * grad[0],
                "grad_cpu_ms": 1e3 * grad[1],
                "record_ms": 1e3 * record[0],
                "hessian_s": hess[0],
                "hessian_cpu_s": hess[1],
            }
        )
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "results" / "benchmark_grad.csv", index=False, float_format="%.4g")
    return table


def part_gf(workers: int) -> pd.DataFrame:
    configs = []
    for (kind, n), d, chi in [(t, d, c) for t in TREES for d in DS for c in CHIS]:
        base = config(kind, n, d, chi)
        k = make_spec(base).n_cores
        common = {"init.sigma": 1e-3 ** (1 / k), "optimizer.t_max": 1e6}
        configs.append(
            base.with_overrides(
                common | {"name": f"{base.name}__lsoda", "optimizer.method": "lsoda"}
            )
        )
        if d == 3:  # DP5's step count depends on the dynamics (K, sigma), hardly on d and chi
            configs.append(
                base.with_overrides(
                    common | {"name": f"{base.name}__dopri5", "optimizer.max_steps": 5000}
                )
            )
    out = REPO / "data" / "00_smoke" / "benchmark"
    # Largest problems first, so that the pool does not end on one long run.
    configs.sort(key=lambda c: -make_spec(c).n_params)
    table = run_sweep(configs, out, workers)
    rows = []
    for _, r in table.iterrows():
        traj = load_run(out, r["name"])
        meta = traj.meta
        # First escape of any component (a smaller one can come first): the first log time with
        # some a_r >= 1/2, and the integrator steps at the log time before it (the cost of the
        # plateau; interpolating steps across that interval would charge the escape with the
        # stiff steps that follow it).
        crossed = np.flatnonzero(np.any(traj["overlaps"] >= 0.5, axis=1))
        t_escape = traj["t"][crossed[0]] if crossed.size else np.nan
        steps_before_escape = traj["step"][crossed[0] - 1] if crossed.size else np.nan
        _, tree, d, chi, method = r["name"].split("__")
        rows.append(
            {
                "tree": tree,
                "K": len(meta["problem"]["parent"]),
                "d": int(d.split("=")[1]),
                "chi": int(chi.split("=")[1]),
                "method": method,
                "n_params": meta["problem"]["n_params"],
                "sigma": r["init.sigma"],
                "stop_reason": r["stop_reason"],
                "t_end": r["t_end"],
                "t_escape_1": t_escape,
                "n_learned": r["n_learned"],
                "n_steps": r["n_steps"],
                "n_evals": r["n_evals"],
                "n_jacobians": meta.get("njev", 0),
                "steps_before_escape_1": steps_before_escape,
                "wall_s": r["wall_time"],
                "cpu_s": r["cpu_time"],
            }
        )
    result = pd.DataFrame(rows).sort_values(["method", "tree", "d", "chi"])
    result.to_csv(HERE / "results" / "benchmark_gf.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=["grad", "gf", "both"], default="both")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    if args.part in ("grad", "both"):
        part_grad()
    if args.part in ("gf", "both"):
        part_gf(args.workers)


if __name__ == "__main__":
    main()
