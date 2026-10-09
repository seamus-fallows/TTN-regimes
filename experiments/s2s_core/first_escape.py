"""Step 3a: first escape t_1 vs sigma, per seed, against the correlation-flow prediction (P1).

For each (tree, chi, teacher, seed): the blow-up time T(theta_hat) of the correlation flow from the
sigma = 1 init (and, optionally, T_r with only component r in the teacher), then gradient flow at
every sigma of the grid until shortly after the first escape, with the first event refined.

    uv run python experiments/s2s_core/first_escape.py experiments/s2s_core/configs/first.yaml

Outputs: /root/outputs/s2s_core/<name>/<job>.json, collected by analyze_first.py.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402

from ttn import diagnostics as dg  # noqa: E402
from ttn.dynamics import Problem  # noqa: E402
from ttn.teachers import CPTeacher  # noqa: E402


def job(args):
    cfg, out_dir = args
    torch.set_num_threads(1)
    try:
        return _job(cfg, out_dir)
    except Exception:
        return {"name": cfg["name"], "error": traceback.format_exc()}


def _job(cfg, out_dir):
    wall0 = time.perf_counter()
    spec, teacher = s2s.setup(cfg["tree"], cfg["N"], cfg["d"], cfg["chi"], cfg["weights"], cfg["teacher_seed"])
    K, R = spec.n_cores, teacher.n_components
    cores1 = s2s.make_init(spec, teacher, cfg["init"], 1.0, cfg["seed"])
    theta_hat = spec.flatten(cores1)
    pred = s2s.correlation_flow_blowup(spec, teacher, theta_hat)
    per_comp = []
    if cfg.get("single_component", True):
        for r in range(R):
            single = CPTeacher(teacher.weights[r : r + 1].clone(), teacher.factors[r : r + 1].clone())
            per_comp.append(s2s.correlation_flow_blowup(spec, single, theta_hat)["T"])
    a_hat = (teacher.component_overlaps(spec, cores1) * teacher.weights).tolist()
    problem = Problem(spec, teacher)
    runs = []
    for sigma in cfg["sigmas"]:
        theta0 = sigma * theta_hat  # fan-in / balanced / aligned inits are all linear in sigma
        t_pred = sigma ** (-(K - 2)) * pred["T"]
        # Stop once some component is learned (a >= 0.999); the run past the escape is stiff.
        t_max = 3.0 * t_pred + 100.0
        gf = dict(method="lsoda", rtol=cfg.get("rtol", 1e-8), atol=cfg.get("atol_rel", 1e-11) * sigma)
        tr = s2s.integrate_chunked(problem, theta0, t_max, n_log=300, chunk=10, wall_limit=600.0,
                                   spectra=False, stop_fn=lambda d: d["overlaps"].max() >= 0.999, **gf)
        t, a, params = tr["t"], tr["overlaps"], tr["params"]
        row = {"sigma": sigma, "t_pred": t_pred, "stop": tr.meta["stop_reason"]}
        reached = [r for r in range(R) if np.any(a[:, r] >= 0.9)]
        if reached:
            first_hi = {r: np.flatnonzero(a[:, r] >= 0.9)[0] for r in reached}
            r1 = min(first_hi, key=first_hi.get)
            hi = first_hi[r1]
            lo = np.flatnonzero(a[: hi + 1, r1] < 0.1)
            i_lo = lo[-1] if lo.size else 0
            z = s2s.zoom_rise(problem, torch.as_tensor(params[i_lo]), t[i_lo], t[hi], r1, **gf)
            if z is not None:
                tz, az, _ = z
                row.update({
                    "r1": int(r1),
                    "t1": float(dg.crossing_times(tz, az[:, r1], 0.5, last=True)),
                    "t1_first_crossing": float(dg.crossing_times(t, a[:, r1], 0.5)),
                    "w1": float(dg.event_widths(tz, az[:, r1])),
                    # other components' overlaps at the end of the first rise
                    "a_end": az[-1].tolist(),
                })
        runs.append(row)
    out = {
        **cfg, "K": K, "P": spec.n_params, "T": pred["T"], "r1_pred": pred["r1"],
        "corrflow": pred, "T_single": per_comp, "a_hat": a_hat,
        "theta_hat_norm": theta_hat.norm().item(),
        "kh": s2s.kh_prefactor(theta_hat, K, max(cfg["weights"])),
        "p3": [s2s.p3_prefactor(cfg["d"], cfg["N"], K, s) for s in cfg["weights"]],
        "c_K": [s2s.c_K(K, s) for s in cfg["weights"]],
        "runs": runs, "wall_time": time.perf_counter() - wall0,
    }
    (out_dir / f"{cfg['name']}.json").write_text(json.dumps(out, default=s2s._json))
    return {"name": cfg["name"], "error": None, "wall": out["wall_time"]}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--raw", default=str(s2s.OUT), help="parent directory of the raw outputs")
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    workers = args.workers
    out_dir = Path(args.raw) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for (tree, n), chi, (tname, w), init, seed in itertools.product(
        config["trees"], config["chis"], config["teachers"].items(), config["inits"], config["seeds"]
    ):
        cfg = {"tree": tree, "N": n, "d": config["d"], "chi": chi, "teacher": tname,
               "weights": [float(x) for x in w], "teacher_seed": config["teacher_seed"],
               "init": init, "seed": seed, "sigmas": config["sigmas"][f"{tree}{n}"],
               "single_component": config.get("single_component", True)}
        cfg["name"] = f"{tree}{n}_chi{chi}_{tname}_{init}_seed{seed}"
        if not (out_dir / f"{cfg['name']}.json").exists():
            jobs.append((cfg, out_dir))
    print(f"{config['name']}: {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with Pool(workers, maxtasksperchild=10) as pool:
        for i, res in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {res['name']} {res.get('wall', 0):.0f}s", flush=True)
            if res["error"]:
                print(res["error"], flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
