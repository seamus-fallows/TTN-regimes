"""Step 1 sweeps (aligned covariance): order, swaps, event times, scaling. Configs configs/e1_*.yaml.

Usage: uv run python experiments/data_teacher/run_aligned.py configs/e1_order.yaml [--procs 30]
Resumable: finished runs (npz files) are skipped. Writes results/e1_order_summary.csv.
"""

import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
import yaml

from common import OUT_ROOT, aligned_setup, make_spec, run_and_save

HERE = Path(__file__).parent


def job(args):
    cfg, tree, sigma, tname, variant, seed = args
    torch.set_num_threads(1)
    tcfg = cfg["teachers"][tname]
    spec = make_spec(tree, cfg["n_sites"], cfg["d"], cfg["chi"])
    teacher, metrics, big_lam = aligned_setup(
        cfg["n_sites"], cfg["d"], tcfg["weights"], tcfg["eig_index"], tcfg["alpha"], variant
    )
    cores0 = spec.init(sigma, seed=seed)
    run_cfg = dict(cfg, t_max=cfg["t_max"][tree])
    meta = dict(tree=tree, sigma=sigma, teacher=tname, variant=variant, seed=seed,
                **{f"Lambda_{r}": float(x) for r, x in enumerate(big_lam)},
                **{f"s_{r}": float(x) for r, x in enumerate(tcfg["weights"])})
    path = OUT_ROOT / cfg["name"] / f"{tree}_s{sigma}_{tname}_{variant}_seed{seed}.npz"
    summary = run_and_save(path, spec, cores0, teacher, metrics, run_cfg, meta)
    print(tree, sigma, tname, variant, seed, summary["order"], summary["n_unlearn"],
          f"wall {summary['wall']:.0f}s", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=str(HERE / "configs/e1_order.yaml"))
    ap.add_argument("--procs", type=int, default=30)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    jobs = []
    for tree in cfg["trees"]:
        for sigma, tname, variant, seed in itertools.product(
            cfg["sigmas"][tree], cfg["teachers"], cfg["variants"], range(*cfg["seeds"])
        ):
            jobs.append((cfg, tree, sigma, tname, variant, seed))
    # Slow (MPS, small sigma) jobs first so the pool finishes evenly.
    jobs.sort(key=lambda j: (j[1] != "mps", j[2]))
    with Pool(a.procs) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows).drop(columns=["events"])
    out = HERE / "results" / f"{cfg['name']}_summary.csv"
    df.to_csv(out, index=False)
    pd.to_pickle(rows, OUT_ROOT / cfg["name"] / "summary_rows.pkl")
    print("wrote", out)
