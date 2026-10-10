"""Round 2, item 1: escape law from the first saddle for the same generic teacher on different trees.

Saddle-start protocol (r2_common.py): exact balanced rank-1 saddle (the lowest-loss bond-rank-1
critical point) + fan-in noise of scale sigma; gradient flow with the library integrator
(`integrate_gf`, LSODA, rtol 1e-9, atol 1e-14), integrated in chunks of doubling length until
max_e s_2^e >= 0.2 (the escape is over) or t = t_cap (config; default 1e7). Escape time t_esc = first time
max_e s_2^e >= 0.05 (log-interpolated). Also records the gauge-projected Hessian at the exact saddle
(strict iff lambda_min < 0) once per (teacher, tree).

Prediction (theory_saddles.md R13, theory_teachers.md R9/P9): MPS students: strict saddle (k* = 2),
t_esc = log(1/sigma)/|lambda_min| + const; binary / HT students: non-strict (k* = 3 through an
internal core), t_esc ~ sigma^-1.
Usage: uv run python experiments/data_teacher/r2_saddle_runs.py configs/r2_saddle.yaml [--procs 28]
Writes /root/outputs/data_teacher/r2_saddle/<teacher>_<tree>_s<sigma>_seed<seed>.npz,
results/r2_saddle_runs.csv and results/r2_saddle_hessian.csv.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from common import RESULTS, OUT_ROOT
from r2_common import balance_rank1, embed, rank1_saddles, saddle_start, student, teacher_for

from ttn.diagnostics import gauge_directions, hessian, project_out_gauge, saddle_type
from ttn.dynamics import Problem, Recorder, integrate_gf, merged_rows

HERE = Path(__file__).parent
_SADDLES = {}
T_CAP = 1e7  # set from the config's t_cap


def saddle_for(tname, tcfg, n, tree):
    key = (tname, tree)
    if key not in _SADDLES:
        teacher, metrics = teacher_for(tcfg["kind"], n, **tcfg.get("args", {}))
        spec = student(tree, n)
        sads = rank1_saddles(spec, teacher, metrics, n_starts=12)
        best = sads[0]
        cores = embed(spec, balance_rank1(best["cores"]))
        _SADDLES[key] = (spec, teacher, metrics, cores, best["loss"], len(sads))
    return _SADDLES[key]


def first_cross(t, x, level):
    k = np.nonzero(x >= level)[0]
    if len(k) == 0:
        return np.nan
    k = k[0]
    if k == 0:
        return t[0]
    lx0, lx1 = np.log(max(x[k - 1], 1e-300)), np.log(x[k])
    return t[k - 1] + (np.log(level) - lx0) / (lx1 - lx0) * (t[k] - t[k - 1])


def run(args):
    name, tname, tcfg, n, tree, sigma, seed = args
    torch.set_num_threads(1)
    path = OUT_ROOT / name / f"{tname}_{tree}_s{sigma}_seed{seed}.npz"
    if path.exists():
        return json.loads(str(np.load(path)["summary"]))
    spec, teacher, metrics, cores, l_sad, n_sad = saddle_for(tname, tcfg, n, tree)
    prob = Problem(spec, teacher, site_metrics=metrics)
    theta = saddle_start(spec, cores, sigma, seed)
    t_off, chunk, parts = 0.0, 20.0, []
    while True:
        tr = integrate_gf(prob, theta, t_max=chunk, n_log=600, rtol=1e-9, atol=1e-14,
                          method="lsoda", recorder=Recorder(spectra=True, overlaps=False))
        d = tr.data
        m = merged_rows(d)  # log rows + the Recorder's fine rows (event resolution)
        parts.append((m["t"] + t_off, m["loss"], np.nan_to_num(m["spectra"]),
                      d["conservation_drift"]))
        s2 = np.nan_to_num(m["spectra"])[:, :, 1].max(axis=1)
        theta = torch.as_tensor(d["theta_end"])
        t_off += chunk
        if s2.max() >= 0.2 or t_off > T_CAP:
            break
        chunk *= 2
    t = np.concatenate([p[0] for p in parts])
    loss = np.concatenate([p[1] for p in parts])
    sp = np.concatenate([p[2] for p in parts])
    drift = float(np.nanmax(np.concatenate([p[3] for p in parts])))
    s2 = sp[:, :, 1].max(axis=1)
    t_esc = first_cross(t, s2, 0.05)
    k_esc = int(np.argmax(s2 >= 0.05)) if (s2 >= 0.05).any() else len(t) - 1
    edge = int(np.argmax(sp[k_esc, :, 1]))
    summary = dict(teacher=tname, n=n, tree=tree, sigma=sigma, seed=seed, l_saddle=l_sad,
                   n_rank1_minima=n_sad, t_esc=t_esc, edge=edge, t_end=float(t[-1]),
                   s2_start=float(s2[0]), max_drift=drift)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=t, loss=loss, spectra=sp, summary=json.dumps(summary))
    print(tname, tree, sigma, seed, f"t_esc={t_esc:.4g}", flush=True)
    return summary


def hessian_row(args):
    tname, tcfg, n, tree = args
    torch.set_num_threads(1)
    spec, teacher, metrics, cores, l_sad, n_sad = saddle_for(tname, tcfg, n, tree)
    prob = Problem(spec, teacher, site_metrics=metrics)
    th = spec.flatten(cores)
    h = hessian(prob.loss, th)
    _, eigs = project_out_gauge(h, gauge_directions(spec, th))  # library default rtol (1e-3 after 6c42d85)
    st = saddle_type(eigs)
    return dict(teacher=tname, n=n, tree=tree, l_saddle=l_sad, n_rank1_minima=n_sad,
                grad_norm=float(prob.grad(th).norm()), lambda_min=st["lambda_min"],
                lambda_min_raw=float(torch.linalg.eigvalsh(h)[0]),
                n_negative=st["n_negative"], type=st["type"], lambda_max=st["lambda_max"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=28)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    name = cfg["name"]
    combos = [(tname, tcfg, n, tree) for n, block in cfg["teachers"].items()
              for tname, tcfg in block.items() for tree in cfg["trees"][n]]
    with Pool(a.procs) as pool:
        hrows = pool.map(hessian_row, combos, chunksize=1)
    out_tag = cfg.get("out_tag", name)
    pd.DataFrame(hrows).to_csv(RESULTS / f"{out_tag}_hessian.csv", index=False)
    print(pd.DataFrame(hrows).to_string(), flush=True)
    globals()["T_CAP"] = float(cfg.get("t_cap", 1e7))  # inherited by the forked workers

    def sigmas_for(tree):
        sig = cfg["sigmas"]
        return sig[tree] if isinstance(sig, dict) else sig

    jobs = [(name, tname, tcfg, n, tree, sigma, seed) for tname, tcfg, n, tree in combos
            for sigma in sigmas_for(tree) for seed in range(*cfg["seeds"])]
    jobs.sort(key=lambda j: j[5])  # smallest sigma (longest runs) first
    with Pool(a.procs) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(RESULTS / f"{out_tag}_runs.csv", index=False)
    print("wrote", len(rows))
