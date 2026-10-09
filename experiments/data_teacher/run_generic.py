"""Sweeps for steps 2-4 (misaligned covariance, non-orthogonal CP, correlated inputs, TTN teachers).

Config: `base` (dict of setup parameters, see setups.py), `sweep` (dict of lists; the cartesian
product is run), `sigmas` (per tree), `seeds` ([start, stop)), integrator settings.
Usage: uv run python experiments/data_teacher/run_generic.py configs/<name>.yaml [--procs 30]
Resumable; writes results/<name>_summary.csv and /root/outputs/data_teacher/<name>/*.npz.
"""

import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from common import OUT_ROOT, component_events, crossings
from pilots.pilot_gf import run_gf
from setups import build

from ttn.network import _sqrt_psd

HERE = Path(__file__).parent


def dense_bond_spectra(w, tree, metrics=None):
    """Singular values of (Sigma^{1/2} W) matricized across every edge of `tree` (dense, small N)."""
    if metrics is not None:
        from ttn.network import apply_on_leg
        for i, m in enumerate(metrics):
            w = apply_on_leg(w, _sqrt_psd(m), i)
    out = {}
    n = w.ndim
    for v in tree.edges:
        below = list(tree.subtree_sites[v])
        rest = [i for i in range(n) if i not in below]
        mat = w.permute(below + rest).reshape(int(np.prod([w.shape[i] for i in below])), -1)
        out[v] = torch.linalg.svdvals(mat).numpy()
    return out


def spectral_events(t, spectra, final):
    """Event (e, k): first time s_k^e crosses half of its final value (diagnostics.md)."""
    ev = []
    for j in range(spectra.shape[1]):
        for k in range(spectra.shape[2]):
            f = final[j, k]
            if f < 1e-6:
                continue
            for tc, sign in crossings(t, spectra[:, j, k] / f, 0.5):
                if sign > 0:
                    ev.append((float(tc), j, k))
                    break
    return sorted(ev)


def job(args):
    name, case, tree, sigma, seed, icfg = args
    torch.set_num_threads(1)
    path = OUT_ROOT / name / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
    if path.exists():
        return np.load(path, allow_pickle=True)["summary"].item()
    cfg = dict(case, tree=tree)
    spec, teacher, metrics, loss_fn = build(cfg, seed)
    cores0 = spec.init(sigma, seed=seed)
    is_cp = hasattr(teacher, "factors")
    rec = run_gf(spec, cores0, teacher, metrics, t_max=float(icfg["t_max"][tree]),
                 n_rec=int(icfg["n_rec"]), rtol=float(icfg["rtol"]), atol=float(icfg["atol"]),
                 stop_rel_loss=float(icfg["stop_rel_loss"]), w_star=teacher.dense(),
                 loss_fn=loss_fn, record_overlaps=is_cp, keep_y=True)
    edges = list(spec.tree.edges)
    chi = max(spec.bond_dims[v] for v in edges)
    spectra = np.zeros((len(rec["t"]), len(edges), chi))
    for k, sp in enumerate(rec["spectra"]):
        for j, v in enumerate(edges):
            spectra[k, j, : len(sp[v])] = sp[v]
    tsp = dense_bond_spectra(teacher.dense(), spec.tree, metrics)
    teacher_spectra = np.zeros((len(edges), chi))
    for j, v in enumerate(edges):
        m = min(chi, len(tsp[v]))
        teacher_spectra[j, :m] = tsp[v][:m]
    summary = dict({k: v for k, v in case.items() if not isinstance(v, (list, dict))},
                   tree=tree, sigma=sigma, seed=seed, q_drift=rec["q_drift"], nfev=rec["nfev"],
                   wall=rec["wall"], status=rec["status"], t_end=rec["t"][-1],
                   loss0=rec["loss"][0], loss_end=rec["loss"][-1],
                   spectral_events=spectral_events(rec["t"], spectra, teacher_spectra))
    if is_cp:
        summary.update(component_events(rec["t"], rec["overlaps"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=rec["t"], loss=rec["loss"],
                        overlaps=rec["overlaps"] if is_cp else np.zeros(0), spectra=spectra,
                        teacher_spectra=teacher_spectra, edges=np.array(edges), y=rec["y"],
                        summary=np.array(summary, dtype=object))
    print(case["tag"], tree, sigma, seed, summary.get("order", ""), f"wall {rec['wall']:.0f}s",
          flush=True)
    return summary


def expand(cfg):
    keys = list(cfg.get("sweep", {}))
    cases = []
    for values in itertools.product(*(cfg["sweep"][k] for k in keys)):
        case = dict(cfg["base"])
        case.update(dict(zip(keys, values)))
        case["tag"] = "_".join(f"{k}{v}" for k, v in zip(keys, values)) or "base"
        cases.append(case)
    return cases


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=30)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    icfg = {k: cfg[k] for k in ("t_max", "n_rec", "rtol", "atol", "stop_rel_loss")}
    jobs = [(cfg["name"], case, tree, sigma, seed, icfg)
            for case in expand(cfg) for tree in cfg["trees"] for sigma in cfg["sigmas"][tree]
            for seed in range(*cfg["seeds"])]
    jobs.sort(key=lambda j: (j[2] != "mps", j[3]))
    with Pool(a.procs) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    pd.to_pickle(rows, OUT_ROOT / cfg["name"] / "summary_rows.pkl")
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in ("events", "spectral_events") if c in df]).to_csv(
        HERE / "results" / f"{cfg['name']}_summary.csv", index=False)
    print("wrote", len(df), "rows")
