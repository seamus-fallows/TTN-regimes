"""Recompute the Hessian spectra at the saved plateau midpoints of finished runs: the raw
(unprojected) lambda_min and the gauge-projected one with the library's current projection.

    uv run python experiments/s2s_core/recompute_hessians.py --raw DIR [DIR ...] [--workers 60] [--out results] [--tag hess]

Needed because `ttn.diagnostics.project_out_gauge` was found to bias projected eigenvalues near
rank-deficient saddles (fix pending on main): report raw values next to projected ones, and rerun
this script after the fix. Writes <out>/<tag>.csv: one row per (run, plateau).
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402

from ttn import diagnostics as dg  # noqa: E402
from ttn.dynamics import Problem  # noqa: E402

HERE = Path(__file__).parent


def job(path: Path):
    torch.set_num_threads(1)
    s = json.loads(path.read_text())
    d = np.load(path.with_suffix(".npz"))
    if "mid_params" not in d or len(d["mid_params"]) == 0:
        return []
    spec, teacher = s2s.setup(s["tree"], s["N"], s["d"], s["chi"], s["weights"], s["teacher_seed"])
    problem = Problem(spec, teacher)
    rows = []
    for i, p in enumerate(d["mid_params"]):
        theta = torch.as_tensor(p)
        h = problem.hessian(theta)
        raw = torch.linalg.eigvalsh(0.5 * (h + h.T)).numpy()
        _, proj = dg.project_out_gauge(h, dg.gauge_directions(spec, theta))
        proj = proj.numpy()
        rows.append({"sweep": path.parent.name, "tree": f"{s['tree']}{s['N']}", "K": s["K"], "chi": s["chi"],
                     "teacher": s["teacher"], "init": s["init"], "sigma": s["sigma"], "seed": s["seed"],
                     "plateau": i + 1, "order": "-".join(str(r + 1) for r in s["order"]),
                     "lam_min_raw": float(raw[0]), "lam_max_raw": float(raw[-1]),
                     "lam_min_proj": float(proj[0]), "lam_max_proj": float(proj[-1]),
                     "n_neg_raw_1e-3": int(np.sum(raw < -1e-3 * raw[-1])),
                     "n_neg_proj_1e-3": int(np.sum(proj < -1e-3 * proj[-1]))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--tag", default="hess")
    args = ap.parse_args()
    files = [f for r in args.raw for f in sorted(Path(r).glob("*.json")) if f.name != "summary.json"]
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.imap_unordered(job, files, chunksize=4) for r in rs]
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.tag}.csv", index=False)
    print(len(df), "plateaus")


if __name__ == "__main__":
    main()
