"""Recompute the correlation-flow prediction T (and the single-component T_r, first component r1) of
finished first-escape jobs with the current `s2s.correlation_flow_blowup` (direction-convergence stop,
fix of review s2s_core round 1, B2 / issue 6). The previous values are kept in each job's JSON under
`T_v1`, `r1_pred_v1`, `T_single_v1`; `T`, `r1_pred`, `corrflow`, `T_single` are replaced.

    uv run python experiments/s2s_core/recompute_T.py --raw /root/outputs/s2s_core/first [--workers 60]
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

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402

from ttn.teachers import CPTeacher  # noqa: E402


def job(path: Path):
    torch.set_num_threads(1)
    s = json.loads(path.read_text())
    if "T_v1" in s:
        return path.name, "skip"
    spec, teacher = s2s.setup(s["tree"], s["N"], s["d"], s["chi"], s["weights"], s["teacher_seed"])
    theta_hat = spec.flatten(s2s.make_init(spec, teacher, s["init"], 1.0, s["seed"]))
    pred = s2s.correlation_flow_blowup(spec, teacher, theta_hat)
    single = []
    if s.get("T_single"):
        for r in range(teacher.n_components):
            t1 = CPTeacher(teacher.weights[r : r + 1].clone(), teacher.factors[r : r + 1].clone())
            single.append(s2s.correlation_flow_blowup(spec, t1, theta_hat)["T"])
    s.update({"T_v1": s["T"], "r1_pred_v1": s["r1_pred"], "T_single_v1": s.get("T_single"),
              "T": pred["T"], "r1_pred": pred["r1"], "corrflow": pred, "T_single": single})
    path.write_text(json.dumps(s, default=s2s._json))
    return path.name, f"{s['T_v1']:.10g} -> {pred['T']:.10g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--workers", type=int, default=60)
    args = ap.parse_args()
    files = sorted(Path(args.raw).glob("*.json"))
    with Pool(args.workers) as pool:
        for name, msg in pool.imap_unordered(job, files):
            print(name, msg, flush=True)


if __name__ == "__main__":
    main()
