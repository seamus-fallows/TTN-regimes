"""Cross-check of the pilot integrator against the library (ttn.dynamics.integrate_gf, LSODA).

For a sample of runs of every product-covariance sweep (step 1, step 2, step 4a, 4b), rerun the
same problem from the same init with `Problem` + `integrate_gf(method="lsoda", rtol=1e-9,
atol=1e-14)` and compare the event times (first a_r = 1/2 crossings for CP teachers, first
s_k^e = half-final crossings for TTN teachers) and the loss at those times. Step 3 (non-product
metric) is not supported by `Problem` and stays pilot-only.
Usage: uv run python experiments/data_teacher/crosscheck_library.py
Writes results/crosscheck_library.csv.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from common import OUT_ROOT, aligned_setup, crossings, make_spec, map_cores
from run_generic import expand
from setups import build

from ttn.dynamics import Problem, Recorder, integrate_gf
from ttn.losses import power_law_cov

HERE = Path(__file__).parent


def first_up(t, a, level=0.5):
    c = [tc for tc, sg in crossings(t, a, level) if sg > 0]
    return c[0] if c else np.nan


def jobs():
    out = []
    cfg = yaml.safe_load(open(HERE / "configs/e1_order.yaml"))
    for tree, sigma in (("binary", 0.05), ("mps", 0.1)):
        for tname in cfg["teachers"]:
            for variant in ("sigma", "white_sqrtLs_mapped"):
                for seed in (0, 1):
                    out.append(("e1_order", dict(cfg=cfg, tree=tree, sigma=sigma, tname=tname,
                                                 variant=variant, seed=seed)))
    for name, picks in (("e2_misaligned", [("alpha1.0_theta_deg20_variantsigma", "mps", 0.1),
                                           ("alpha1.0_theta_deg45_variantsigma", "binary", 0.05)]),
                        ("e4a_nonorth", [("s20.7_overlap0.35", "mps", 0.1),
                                         ("s20.7_overlap0.65", "binary", 0.05)]),
                        ("e4b_ttn", [("teacher_seed103", "mps", 0.0356)])):
        cfg = yaml.safe_load(open(HERE / "configs" / f"{name}.yaml"))
        cases = {c["tag"]: c for c in expand(cfg)}
        for tag, tree, sigma in picks:
            for seed in (0, 1):
                out.append((name, dict(cfg=cfg, case=cases[tag], tree=tree, sigma=sigma, seed=seed)))
    return out


def run(job):
    try:
        return run1(job)
    except Exception as err:  # report and continue
        print("FAILED", job[0], repr(err), flush=True)
        return [dict(run=f"{job[0]}: failed", event=repr(err), t_pilot=np.nan, t_lib=np.nan,
                     rel_diff=np.nan)]


def run1(job):
    name, j = job
    torch.set_num_threads(1)
    if name == "e1_order":
        cfg, tcfg = j["cfg"], j["cfg"]["teachers"][j["tname"]]
        spec = make_spec(j["tree"], cfg["n_sites"], cfg["d"], cfg["chi"])
        teacher, metrics, _ = aligned_setup(cfg["n_sites"], cfg["d"], tcfg["weights"],
                                            tcfg["eig_index"], tcfg["alpha"], j["variant"])
        cores0 = spec.init(j["sigma"], seed=j["seed"])
        if j["variant"] == "white_sqrtLs_mapped":
            cores0 = map_cores(spec, cores0, [power_law_cov(cfg["d"], tcfg["alpha"]).sqrt()] * 4)
        path = OUT_ROOT / name / f"{j['tree']}_s{j['sigma']}_{j['tname']}_{j['variant']}_seed{j['seed']}.npz"
        label = f"{j['tree']} {j['tname']} {j['variant']} seed{j['seed']}"
    else:
        case = j["case"]
        spec, teacher, metrics, _, init_map = build(dict(case, tree=j["tree"]), j["seed"])
        cores0 = spec.init(j["sigma"], seed=j["seed"])
        if init_map is not None:
            cores0 = map_cores(spec, cores0, init_map)
        path = OUT_ROOT / name / f"{case['tag']}_{j['tree']}_s{j['sigma']}_seed{j['seed']}.npz"
        label = f"{case['tag']} {j['tree']} seed{j['seed']}"
    pilot = np.load(path, allow_pickle=True)
    t_end = float(pilot["t"][-1])
    prob = Problem(spec, teacher, site_metrics=metrics)
    is_cp = hasattr(teacher, "factors")
    traj = integrate_gf(prob, cores0, t_max=t_end, log_times=pilot["t"][1:-1], rtol=1e-9,
                        atol=1e-14, method="lsoda", recorder=Recorder(overlaps=is_cp))
    a = traj.data
    t_lib, loss_lib = np.asarray(a["t"]), np.asarray(a["loss"])
    rows = []
    if is_cp:
        # The pilot's Euclidean dual overlaps equal the library's <W, M U_r>/<W*, M U_r> only for
        # components that stay orthogonal under M (step 1); steps 2, 4a are compared on spectra.
        ov_lib = np.asarray(a["overlaps"])
        if name == "e1_order":
            for r in range(ov_lib.shape[1]):
                tp = first_up(pilot["t"], pilot["overlaps"][:, r])
                tl = first_up(t_lib, ov_lib[:, r])
                rows.append(dict(run=f"{name}: {label}", event=f"a_{r}=1/2", t_pilot=tp, t_lib=tl))
    sp_lib = np.nan_to_num(np.asarray(a["spectra"]))
    sp_pil = pilot["spectra"]
    final = sp_pil[-1]
    for e in range(sp_pil.shape[1]):
        for k in range(min(2, sp_pil.shape[2])):
            if final[e, k] < 1e-3:
                continue
            tp = first_up(pilot["t"], sp_pil[:, e, k] / final[e, k])
            tl = first_up(t_lib, sp_lib[:, e, k] / final[e, k])
            rows.append(dict(run=f"{name}: {label}", event=f"s_{k + 1}^{e}=half", t_pilot=tp,
                             t_lib=tl))
    n = min(len(loss_lib), len(pilot["loss"]))
    mid = np.nonzero(pilot["loss"][:n] > 1e-3 * pilot["loss"][0])[0]
    rows.append(dict(run=f"{name}: {label}", event="max rel loss diff (L > 1e-3 L0)",
                     t_pilot=np.nan, t_lib=np.nan,
                     rel_diff=float(np.max(np.abs(loss_lib[mid] - pilot["loss"][mid])
                                           / pilot["loss"][mid]))))
    for r in rows:
        if "rel_diff" not in r:
            r["rel_diff"] = (r["t_lib"] - r["t_pilot"]) / r["t_pilot"]
    print(label, max(abs(r["rel_diff"]) for r in rows if np.isfinite(r["rel_diff"])), flush=True)
    return rows


if __name__ == "__main__":
    with Pool(8) as pool:
        res = pool.map(run, jobs(), chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(HERE / "results" / "crosscheck_library.csv", index=False)
    print(df.groupby(df["run"].str.split(":").str[0])["rel_diff"].agg(
        lambda x: np.nanmax(np.abs(x))).to_string())
