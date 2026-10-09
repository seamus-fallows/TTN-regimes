"""Step 4: full learning order with R >= 3 odeco components (weights 1, rho, rho^2, ...).

Usage: uv run python experiments/order/multi.py experiments/order/configs/multi.yaml [--procs 90]

One job = (tree, d, chi, R, rho, seed). Gradient flow with the library integrator
(`ttn.dynamics.integrate_gf`, method='lsoda', dense log grid) from sigma * u0 until the loss is
below `loss_floor` x L(0) or t_max; event times = first crossing of a_r = 1/2
(`ttn.diagnostics.crossing_times`). Per component it also stores the leading-order blow-up time
tau_r (only component r present, unit-scale init), whose ranking tau_r / s_r is the race-model
prediction of the order, and min |a_r| before its event (float64-floor flag: a dormant overlap
below 1e-14 means a later escape may be seeded by roundoff). Results: JSON lines in
/root/outputs/order/<name>/runs.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW as OUT, single_thread, lo_blowup, make_spec, make_teacher, single_component, unit_init  # noqa: E402
from ttn.diagnostics import crossing_times  # noqa: E402
from ttn.dynamics import Problem, Recorder, integrate_gf, log_time_grid, merged_rows  # noqa: E402



def job(args):
    cfg, tree, d, chi, n_comp, rho, seed = args
    single_thread()
    t0 = time.time()
    kind, n = tree.split(":")
    n = int(n)
    spec = make_spec(kind, n, d, chi, cap=cfg.get("cap", True))
    weights = [rho**r for r in range(n_comp)]
    teacher = make_teacher(n, d, weights)
    u0 = unit_init(spec, seed)
    taus = [lo_blowup(spec, single_component(teacher, r), u0)[0] for r in range(n_comp)]
    sigma = cfg["sigma"][tree] if isinstance(cfg["sigma"], dict) else cfg["sigma"]
    problem = Problem(spec, teacher)
    l0 = 0.5 * float(teacher.norm2())
    grid = log_time_grid(1e-2, cfg["t_max"], cfg.get("n_log", 1500))
    traj = integrate_gf(problem, torch.from_numpy(sigma * u0), t_max=cfg["t_max"],
                        log_times=grid, method="lsoda", rtol=cfg.get("rtol", 1e-9),
                        atol=cfg.get("atol", 1e-14), loss_floor=cfg.get("loss_floor", 1e-8) * l0,
                        recorder=Recorder(spectra=False))
    # event times from log rows merged with the recorder's fine rows (integrator steps where theta
    # or the loss changed by 1%), which resolve every escape
    rows = merged_rows(traj.data, keys=("loss", "overlaps"))
    t, a = rows["t"], rows["overlaps"]
    times = crossing_times(t, a, 0.5)
    min_dormant = []
    for r in range(n_comp):
        before = t < (times[r] if np.isfinite(times[r]) else np.inf)
        min_dormant.append(float(np.abs(a[before, r]).min()) if before.any() else np.nan)
    rec = dict(tree=tree, d=d, chi=chi, bonds=list(spec.bond_dims), R=n_comp, rho=rho,
               sigma=sigma, seed=seed, times=[None if not np.isfinite(x) else float(x) for x in times],
               tau=taus, final_a=a[-1].tolist(), final_loss=float(traj["loss"][-1]),
               min_dormant=min_dormant, drift=float(np.nanmax(traj["conservation_drift"])),
               n_rows=len(t), fine_capped=bool(traj.meta.get("fine_capped", False)),
               status=traj.meta.get("status"), t_end=traj.meta["t_end"], wall=time.time() - t0)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("PROCS", 90)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = OUT / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add((r["tree"], r["d"], r["chi"], r["R"], r["rho"], r["seed"]))
    jobs = [(cfg, *key) for key in itertools.product(cfg["trees"], cfg["d"], cfg["chi"], cfg["R"],
                                                     cfg["rho"], range(cfg["seeds"]))
            if key not in done]
    print(f"{cfg['name']}: {len(jobs)} jobs ({len(done)} done)", flush=True)
    t0 = time.time()
    with Pool(args.procs) as pool, open(path, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
