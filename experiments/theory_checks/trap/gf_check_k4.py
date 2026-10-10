"""Full gradient flow on k = 4 routes where the v1 scalar route ODE and the basis-invariant (tensor)
route ODE disagree (MPS4, d = 2, chi = 4, pattern teachers (1, 0.5), teacher seed 1000).

Library model and loss (ttn.dynamics.Problem, exact population loss), ttn.dynamics.integrate_gf with
LSODA (rtol 1e-9, atol 1e-14 as in the locality runs; atol 1e-16 sits below the gradient's roundoff
and makes LSODA crawl), float64, one thread per process. Init as in the locality runs: small init =
spec.init(sigma, seed); saddle start = aligned saddle of component 0 + spec.init(sigma, seed). Stops at
convergence (loss below 1e-10 of its start) or at t_max. Observed class: 'escape' if the dormant component's
overlap a_r reaches 0.5; 'dead' if not and both route-end entries (route_tensors at the final state,
learned directions from it) are below 1e-8 sigma (s2s-core's 'trapped' level); else 'unresolved'.
Writes results/gf_check_k4.csv; predicted verdicts are read from results/k4_invariance.csv.

usage (from the worktree root):
  PYTHONPATH=. python experiments/theory_checks/trap/gf_check_k4.py --sigma 0.01 --workers 6 \
      03:small:1 0123:small:1 03:small:100005 ... 03:saddle:100205
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from k4_invariance import WEIGHTS, setup  # noqa: E402
from trapcrit import learned_directions, route_tensors  # noqa: E402
from ttn.dynamics import Problem, Recorder, integrate_gf, merged_rows  # noqa: E402
from ttn.network import aligned_init  # noqa: E402


def initial_point(spec, teacher, start, sigma, seed):
    cores = spec.init(sigma, seed=seed)
    if start == "saddle":  # experiments/locality/run.py initial_point, learned = [0]
        K = spec.n_cores
        amps = torch.tensor([[WEIGHTS[0] ** (1 / K)]] * K, dtype=torch.float64)
        sad = aligned_init(spec, teacher.factors[[0]], amps)
        cores = [a + b for a, b in zip(sad, cores)]
    return spec.flatten(cores)


def crossing(t, a, level=0.5):
    i = np.flatnonzero(a >= level)
    return float(t[i[0]]) if len(i) else float("nan")


def one(item):
    key, sigma, t_max, wall = item
    torch.set_num_threads(1)
    S_str, start, seed = key.split(":")
    S, seed = tuple(int(c) for c in S_str), int(seed)
    spec, teacher = setup(S)
    problem = Problem(spec, teacher)
    th0 = initial_point(spec, teacher, start, sigma, seed)
    L0 = float(problem.loss(th0))
    rec = Recorder(spectra=False, conservation=False)
    traj = integrate_gf(
        problem,
        th0,
        t_max,
        method="lsoda",
        rtol=1e-9,
        atol=1e-14,
        loss_floor=1e-10 * L0,
        recorder=rec,
        max_wall_time=wall,
    )
    rows = merged_rows(traj.data, keys=("loss", "overlaps"))
    t, a = rows["t"], rows["overlaps"]
    th = torch.from_numpy(traj.data["theta_end"])
    cores = spec.unflatten(th)
    a_end = teacher.component_overlaps(spec, cores).numpy()
    # learned component = the first to reach a = 0.5 (component 0 at t = 0 for a saddle start)
    q = int(np.nanargmin([np.nan_to_num(crossing(t, a[:, i]), nan=np.inf) for i in range(2)]))
    r = 1 - q
    w = learned_directions(spec, cores)
    nodes = spec.tree.steiner_nodes(list(S))
    _, T, _ = route_tensors(spec, teacher, cores, w, r, nodes)
    ends = [abs(float(T[0].item())), abs(float(T[-1].item()))]  # MPS4 route 0-1-2-3
    if a[:, r].max() >= 0.5:
        obs = "escape"
    elif max(ends) < 1e-8 * sigma:
        obs = "dead"
    else:
        obs = "unresolved"
    return dict(
        key=key,
        S=S_str,
        start=start,
        seed=seed,
        sigma=sigma,
        obs=obs,
        stop=traj.meta.get("stop_reason"),
        t_end=float(traj.meta["t_end"]),
        t_escape_q=crossing(t, a[:, q]),
        t_escape_r=crossing(t, a[:, r]),
        a_r_max=float(a[:, r].max()),
        a_r_end=float(a_end[r]),
        end_entries_over_sigma=[e / sigma for e in ends],
        loss_end=float(rows["loss"][-1]),
        wall=float(traj.meta["wall_time"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+", help="S:start:seed, e.g. 03:small:1")
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--t_max", type=float, default=1e8)
    ap.add_argument("--wall", type=float, default=600.0, help="wall-clock cap per run (s)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        rows = pool.map(one, [(k, a.sigma, a.t_max, a.wall) for k in a.keys], chunksize=1)
    df = pd.DataFrame(rows)
    inv = pd.read_csv(HERE / "results" / "k4_invariance.csv", dtype={"S": str})
    inv["key"] = inv.S + ":" + inv.start + ":" + inv.seed.astype(str)
    df = df.merge(
        inv[["key", "v_first", "v_tensor", "first_rot", "p", "p_first", "cos"]],
        on="key",
        how="left",
    )
    out = Path(a.out) / "gf_check_k4.csv"
    if out.exists():  # keep runs at other sigma
        old = pd.read_csv(out, dtype={"S": str})
        df = pd.concat([old[~(old.key.isin(df.key) & (old.sigma == a.sigma))], df])
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(
            df[
                [
                    "key",
                    "sigma",
                    "v_first",
                    "v_tensor",
                    "obs",
                    "t_escape_q",
                    "t_escape_r",
                    "a_r_end",
                    "end_entries_over_sigma",
                    "stop",
                    "wall",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
