"""Recompute the Hessian at every stored plateau midpoint (review issue 3).

The round-1 runner projected out every gauge generator with singular value > 1e-10 x the largest. Near a
rank-deficient saddle the generators of a nearly unused bond index are tiny (~ sigma) and, normalized,
overlap the route's negative mode, so that projection biased lambda_min (k = 2 saddle start: -0.44..-0.46
instead of -0.5). Here, at the same stored states, we record
  - raw: eigenvalues of the full Hessian (gauge zero modes included; they are ~0 near a critical point),
  - wc: eigenvalues after projecting out only well-conditioned gauge directions (singular value
    > 1e-3 x the largest, the reviewer's check),
  - lib: `ttn.diagnostics.project_out_gauge` as it is on main (to be rerun once its fix lands).
Counts of negative eigenvalues use the single-sigma fallback threshold -1e-3 x max|eigenvalue|.

usage: uv run python experiments/locality/hess_recompute.py <prefix> [...] --out results/hessians.csv --workers 16
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from analysis import load_records  # noqa: E402
from rules import get_tree  # noqa: E402
from ttn.diagnostics import gauge_directions, project_out_gauge  # noqa: E402
from ttn.dynamics import Problem  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

_CACHE: dict = {}


def problem_for(rec):
    key = (rec["kind"], rec.get("n"), rec["d"], rec["chi"], str(rec["weights"]), str(rec["patterns"]),
           rec["teacher_seed"])
    if key not in _CACHE:
        spec = TTNSpec.uniform(get_tree(rec["kind"], rec.get("n")), rec["d"], rec["chi"])
        teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec["teacher_seed"])
        _CACHE[key] = (spec, Problem(spec, teacher))
    return _CACHE[key]


def spectra(spec, problem, theta):
    H = problem.hessian(theta)
    raw = torch.linalg.eigvalsh(0.5 * (H + H.T))
    U = gauge_directions(spec, theta)
    q, s, _ = torch.linalg.svd(U, full_matrices=True)
    keep = int((s > 1e-3 * s.max()).sum())
    C = q[:, keep:]
    wc = torch.linalg.eigvalsh(C.T @ H @ C)
    _, lib = project_out_gauge(H, U)
    return raw.numpy(), wc.numpy(), lib.numpy()


def neg(e):
    return int(np.sum(e < -1e-3 * np.abs(e).max()))


def measure(rec):
    try:
        data = np.load(rec["traj"])
    except (FileNotFoundError, KeyError):
        return []
    spec, problem = problem_for(rec)
    rows = []
    for j, p in enumerate(rec.get("plateaus", [])):
        if p.get("t_mid") is None:
            continue
        i = int(np.argmin(np.abs(data["t"] - p["t_mid"])))
        raw, wc, lib = spectra(spec, problem, torch.from_numpy(data["y"][:, i]))
        rows.append(dict(_key=rec["_key"], j=j, t_mid=float(data["t"][i]), hess_raw_min=float(raw[0]),
                         hess_raw_max=float(raw[-1]), hess_raw_neg=neg(raw), hess_raw_low=str(raw[:3].round(6).tolist()),
                         hess_wc_min=float(wc[0]), hess_wc_neg=neg(wc), hess_wc_low=str(wc[:3].round(6).tolist()),
                         hess_lib_min=float(lib[0]), stored_hess_min=p.get("hess_min")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    recs = [r for r in load_records(args.inputs) if r.get("traj") and r.get("plateaus")
            and r.get("status") not in ("timeout", "error", "stalled")]
    old = pd.read_csv(args.out) if Path(args.out).exists() else None
    done = set(zip(old["_key"], old["j"])) if old is not None else set()
    recs = [r for r in recs if (r["_key"], 0) not in done]
    with Pool(args.workers) as pool:
        rows = [x for xs in pool.imap_unordered(measure, recs, chunksize=8) for x in xs]
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(len(df), "plateau Hessians")


if __name__ == "__main__":
    main()
