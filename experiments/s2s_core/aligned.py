"""Step 1: aligned (Saxe-type) balanced init on an odeco teacher, TTN gradient flow vs the exact
mode solutions of `ttn.theory.odeco_aligned_prediction`.

    uv run python experiments/s2s_core/aligned.py experiments/s2s_core/configs/aligned.yaml

Writes results/aligned.csv (per tree: max |a_r^GF - a_r^theory|, event times GF vs theory, loss
deviation, conservation drift) and the trajectories to /root/outputs/s2s_core/aligned/.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402

from ttn import diagnostics as dg  # noqa: E402
from ttn.dynamics import Problem, Recorder, integrate_gf  # noqa: E402
from ttn.theory import odeco_aligned_prediction  # noqa: E402

HERE = Path(__file__).parent


def run(args):
    cfg, out_dir = args
    torch.set_num_threads(1)
    tree, n = cfg["tree"], cfg["N"]
    spec, teacher = s2s.setup(tree, n, cfg["d"], cfg["chi"], cfg["weights"], cfg["teacher_seed"])
    K = spec.n_cores
    a0 = cfg["a0"]
    cores0 = s2s.make_init(spec, teacher, "aligned", 0.0, 0, a0)
    pred_t = odeco_aligned_prediction(cfg["weights"], K, a0, [1.0])["escape_times"]
    t_max = 1.5 * pred_t.max() + 20
    # Dense linear grid (events have O(1) widths) plus a log grid for the early phase.
    times = np.unique(np.concatenate([np.geomspace(1e-3, t_max, 200), np.linspace(0, t_max, 2001)[1:]]))
    problem = Problem(spec, teacher)
    tr = integrate_gf(problem, cores0, t_max, log_times=times, recorder=Recorder(),
                      method=cfg["method"], rtol=cfg["rtol"], atol=cfg["atol"])
    t = tr["t"]
    pred = odeco_aligned_prediction(cfg["weights"], K, a0, t)
    dev = np.abs(tr["overlaps"] - pred["overlaps"])
    ev_gf = dg.crossing_times(t, tr["overlaps"], 0.5)
    ev_th = dg.crossing_times(t, pred["overlaps"], 0.5)
    np.savez_compressed(out_dir / f"{tree}{n}.npz", t=t, overlaps=tr["overlaps"], loss=tr["loss"],
                        pred_overlaps=pred["overlaps"], pred_loss=pred["loss"],
                        spectra=tr["spectra"], drift=tr["conservation_drift"])
    return {
        "tree": f"{tree}{n}", "K": K, "a0": a0, "method": cfg["method"], "rtol": cfg["rtol"],
        "max_abs_dev_overlap": float(dev.max()),
        "max_abs_dev_loss": float(np.abs(tr["loss"] - pred["loss"]).max()),
        **{f"t{r + 1}_gf": float(ev_gf[r]) for r in range(len(cfg["weights"]))},
        **{f"t{r + 1}_theory": float(pred["escape_times"][r]) for r in range(len(cfg["weights"]))},
        # Same interpolation on the same grid for both curves (the exact escape_times differ from any
        # grid crossing by the interpolation error, ~1e-4 relative here).
        "max_rel_dev_event_same_grid": float(np.max(np.abs(ev_gf - ev_th) / ev_th)),
        "max_rel_dev_event_vs_exact": float(np.max(np.abs(ev_gf - pred["escape_times"]) / pred["escape_times"])),
        "max_conservation_drift": float(np.nanmax(tr["conservation_drift"])),
        "wall_time": tr.meta["wall_time"],
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = s2s.OUT / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for (tree, n) in config["trees"]:
        for method in config["methods"]:
            cfg = {"tree": tree, "N": n, "d": config["d"], "chi": config["chi"],
                   "weights": config["weights"], "teacher_seed": config["teacher_seed"],
                   "a0": config["a0"][f"{tree}{n}"], "method": method,
                   "rtol": config["rtol"], "atol": config["atol"]}
            jobs.append((cfg, out_dir))
    with Pool(len(jobs)) as pool:
        rows = pool.map(run, jobs)
    df = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / "aligned.csv", index=False)
    print(df.to_string())
    (out_dir / "summary.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
