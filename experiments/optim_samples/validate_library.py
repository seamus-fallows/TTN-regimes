"""Cross-check of this experiment's own GD/SGD loops and DOP853 references against ttn.dynamics.

The step-1/2 sweeps were run with minimal loops written before `ttn.dynamics` landed. This script
reruns a sample of settings with `run_gd`, `run_sgd` and `integrate_gf` (same inits, same SGD
generator seeds) and compares event times (a_r = 1/2) and final losses with the stored results.

Run: uv run python experiments/optim_samples/validate_library.py  -> results/validate_library.csv
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from common import RAW, eta_max, setup
from ttn.diagnostics import crossing_times
from ttn.dynamics import Problem, Recorder, integrate_gf, run_gd, run_sgd

from common import RESULTS as OUT  # noqa: E402
REC = dict(spectra=False, conservation=False)


def gd_case(args):
    tree, n, sigma, seed, ratio = args
    torch.set_num_threads(1)
    spec, teacher = setup(tree, n)
    mine = json.loads((RAW / "gd" / f"{tree}{n}_sig{sigma}_seed{seed}_r{ratio}.json").read_text())
    lr = ratio * eta_max(spec, teacher)
    n_steps = mine["steps"]
    traj = run_gd(Problem(spec, teacher), spec.init(sigma, seed=seed), lr, n_steps,
                  log_steps=np.arange(n_steps + 1), recorder=Recorder(**REC))
    t_lib = crossing_times(traj["t"], traj["overlaps"], 0.5)
    t_mine = np.array([np.nan if v is None else v for v in mine["t50"]], float)
    return dict(kind="gd", case=f"{tree}{n} sigma={sigma} seed={seed} eta={ratio} eta_max",
                max_abs_dt=np.nanmax(np.abs(t_lib - t_mine)), events_mine=str(np.round(t_mine, 6)),
                events_lib=str(np.round(t_lib, 6)), loss_end_mine=mine["loss_end"],
                loss_end_lib=traj.meta["loss_end"], wall=traj.meta["wall_time"])


def gf_case(args):
    tree, n, sigma, seed, method = args
    torch.set_num_threads(1)
    spec, teacher = setup(tree, n)
    mine = json.loads((RAW / "gd" / f"{tree}{n}_sig{sigma}_seed{seed}_gf.json").read_text())
    t_max = mine["task"]["T"]
    t_mine = np.array([np.nan if v is None else v for v in mine["t50"]], float)
    grid = np.unique(np.concatenate([np.linspace(0, t_max, 20001)[1:],
                                     np.geomspace(1e-3, t_max, 2000)]))
    traj = integrate_gf(Problem(spec, teacher), spec.init(sigma, seed=seed), t_max,
                        log_times=grid, rtol=1e-10, atol=1e-14, method=method,
                        recorder=Recorder(spectra=False))
    t_lib = crossing_times(traj["t"], traj["overlaps"], 0.5)
    return dict(kind=f"gf-{method}", case=f"{tree}{n} sigma={sigma} seed={seed}",
                max_abs_dt=np.nanmax(np.abs(t_lib - t_mine)), events_mine=str(np.round(t_mine, 6)),
                events_lib=str(np.round(t_lib, 6)), loss_end_mine=mine["loss_end"],
                loss_end_lib=float(traj["loss"][-1]),
                drift=float(np.max(traj["conservation_drift"])), wall=traj.meta["wall_time"])


def sgd_case(row):
    torch.set_num_threads(1)
    tk = row["task"]
    spec, teacher = setup(tk["tree"], tk["n"])
    gen = torch.Generator().manual_seed(10_000 + 1000 * tk["seed"] + tk["noise_seed"])
    n_steps = row["steps"]
    traj = run_sgd(Problem(spec, teacher), spec.init(tk["sigma"], seed=tk["seed"]), tk["eta"],
                   n_steps, tk["B"], gen, log_steps=np.arange(0, n_steps + 1, 4),
                   recorder=Recorder(**REC))
    t_lib = crossing_times(traj["t"], traj["overlaps"], 0.5)
    t_mine = np.array([np.nan if v is None else v for v in row["t50"]], float)
    return dict(kind="sgd", case=f"sigma={tk['sigma']} seed={tk['seed']} noise={tk['noise_seed']} "
                f"B={tk['B']} eta={tk['eta']}", max_abs_dt=np.nanmax(np.abs(t_lib - t_mine)),
                events_mine=str(np.round(t_mine, 6)), events_lib=str(np.round(t_lib, 6)),
                loss_end_mine=row.get("loss_end"), loss_end_lib=traj.meta["loss_end"],
                wall=traj.meta["wall_time"])


def main():
    gd = [("binary", 4, 0.1, 0, 0.3), ("binary", 4, 0.1, 1, 0.5), ("binary", 4, 0.1, 0, 1.2),
          ("binary", 4, 0.003, 2, 0.9), ("binary", 8, 0.9, 0, 0.5), ("mps", 8, 0.9, 2, 0.3)]
    gf = [("binary", 4, 0.1, 1, "dopri5"), ("binary", 4, 0.1, 1, "lsoda"),
          ("binary", 4, 0.003, 2, "dopri5"), ("binary", 8, 0.9, 0, "lsoda")]
    sgd_rows = [json.loads(line) for line in (RAW / "sgd" / "B_sweep.jsonl").read_text().splitlines()]
    pick = {}
    for r in sorted(sgd_rows, key=lambda r: (r["task"]["B"], r["task"]["sigma"])):
        pick.setdefault((r["task"]["sigma"], r["task"]["B"]), r)
    sgd = list(pick.values())[:2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    workers = ap.parse_known_args()[0].workers
    with Pool(workers) as pool:
        res = pool.map(gd_case, gd) + pool.map(gf_case, gf) + pool.map(sgd_case, sgd)
    df = pd.DataFrame(res)
    df.to_csv(OUT / "validate_library.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(df.to_string())


if __name__ == "__main__":
    main()
