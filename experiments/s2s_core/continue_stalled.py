"""Continue stalled runs (all route ends alive, budget exhausted) from their saved final state, to test
whether the dormant component escapes near the balanced-route estimate of the remaining plateau,
b^-(K-2) / ((K-2) s_r) with b = (s_r |a_r|)^(1/K) (theory_saddles R10/R5 with A = 1, k = K).

    uv run python experiments/s2s_core/continue_stalled.py --raw DIR --n 6 [--factor 4] [--out results]
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
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


def candidates(raw: Path):
    out = []
    for f in sorted(raw.glob("*.json")):
        s = json.loads(f.read_text())
        if len(s["order"]) == len(s["weights"]) or s["stop_reason"] != "completed":
            continue
        d = np.load(f.with_suffix(".npz"))
        if "theta_end" not in d:
            continue
        a, w, K = d["overlaps"][-1], np.asarray(s["weights"]), s["K"]
        never = [r for r in range(len(w)) if r not in s["order"]]
        r = min(never, key=lambda r: (w[r] * abs(a[r])) ** (-(K - 2) / K) / w[r])
        pred = (w[r] * abs(a[r])) ** (-(K - 2) / K) / ((K - 2) * w[r])
        out.append((pred / s["t_end"], f, r, pred))
    return sorted(out, key=lambda x: x[0])


def job(args):
    ratio, f, r, pred, factor = args
    torch.set_num_threads(1)
    s = json.loads(f.read_text())
    d = np.load(f.with_suffix(".npz"))
    spec, teacher = s2s.setup(s["tree"], s["N"], s["d"], s["chi"], s["weights"], s["teacher_seed"])
    problem = Problem(spec, teacher)
    theta = torch.as_tensor(d["theta_end"])
    t0 = time.perf_counter()
    tr = s2s.integrate_chunked(problem, theta, factor * pred, n_log=200, wall_limit=3000,
                               max_steps=10**6, spectra=False, method="lsoda",
                               rtol=s.get("rtol", 1e-8), atol=s.get("atol_rel", 1e-11) * s["sigma"],
                               stop_fn=lambda dd: dd["overlaps"][-1, r] >= 0.9)
    t, a = tr["t"], tr["overlaps"]
    t_esc = float(dg.crossing_times(t, a[:, r], 0.5)) if np.any(a[:, r] >= 0.5) else np.nan
    return {"run": f.stem, "component": r, "t_end_before": s["t_end"], "predicted_remaining": pred,
            "measured_remaining": t_esc, "ratio_measured_over_predicted": t_esc / pred,
            "stop": tr.meta["stop_reason"], "wall": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--max_ratio", type=float, default=1e3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cands = candidates(Path(args.raw))
    # args.spread: evenly spaced over the distribution of predicted remaining plateau / elapsed time
    # (else the n shortest), skipping estimates beyond max_ratio x the elapsed time
    cands = [c for c in cands if c[0] <= args.max_ratio]
    pick = cands[:: max(1, len(cands) // args.n)][: args.n] if args.spread else cands[: args.n]
    with Pool(len(pick)) as pool:
        rows = pool.map(job, [(*c, args.factor) for c in pick])
    df = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df["predicted_over_elapsed"] = [c[0] for c in pick]
    df.to_csv(Path(args.out) / f"continued_{Path(args.raw).name}{args.tag}.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
