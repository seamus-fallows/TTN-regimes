"""Leading-order (correlation-flow) first-escape prefactors over many inits: t_hat_1 = sigma^(K-2) t_1
as sigma -> 0, from `s2s.correlation_flow_blowup` (no gradient-flow run). Used for the chi, d and K
dependence (theory_escape.md section 9, predictions 2-6) where hundreds of inits are needed.

    uv run python experiments/s2s_core/lo_sweep.py experiments/s2s_core/configs/lo_chi.yaml [--workers 60] [--out DIR]

Config: list of `settings` (tree, N, d, chi, weights) x `seeds`. Writes <out>/<name>.csv with one row
per (setting, seed): T (= t_hat_1 for the actual weights), first component r1, |theta_hat|, the
Kumar-Haupt time with delta = |theta_hat|, the RMS-overlap heuristic P3, the core-norm fractions and
relative imbalance at the end of the alignment (stop of the flow).
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402

HERE = Path(__file__).parent


def job(args):
    setting, seed, init, teacher_seed = args
    torch.set_num_threads(1)
    tree, n, d, chi, weights = setting["tree"], setting["N"], setting["d"], setting["chi"], setting["weights"]
    spec, teacher = s2s.setup(tree, n, d, chi, weights, teacher_seed)
    K = spec.n_cores
    theta_hat = spec.flatten(s2s.make_init(spec, teacher, init, 1.0, seed))
    t0 = time.perf_counter()
    p = s2s.correlation_flow_blowup(spec, teacher, theta_hat)
    fr = np.array(p["core_fractions"])
    return {
        "tree": f"{tree}{n}", "N": n, "K": K, "d": d, "chi": chi, "weights": "-".join(f"{w:g}" for w in weights),
        "init": init, "seed": seed, "T": p["T"], "r1": p["r1"], "status": p["status"],
        "theta_hat_norm": theta_hat.norm().item(),
        "t_KH": s2s.kh_prefactor(theta_hat, K, max(weights)),
        "P3": s2s.p3_prefactor(d, n, K, max(weights)),
        "frac_dev_max": float(np.max(np.abs(fr * K - 1))), "imbalance_end": p["imbalance"],
        "wall": time.perf_counter() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--out", required=True, help="output directory (e.g. experiments/s2s_core/results)")
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = []
    for s in config["settings"]:
        for chi in np.atleast_1d(s["chi"]):
            for d in np.atleast_1d(s.get("d", config.get("d", 4))):
                c = int(chi) if chi != "full" else None
                settings.append({**s, "chi": c if c is not None else int(d) ** 2, "d": int(d)})
    jobs = [(s, seed, init, config["teacher_seed"]) for s, seed, init in
            itertools.product(settings, config["seeds"], config.get("inits", ["fanin"]))]
    print(f"{config['name']}: {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=50) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s]", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["tree", "d", "chi", "weights", "init", "seed"])
    df.to_csv(out / f"{config['name']}.csv", index=False)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
