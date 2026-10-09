"""Steps 4(b), 4(c): which saddles do runs with TTN teachers visit, and in which order do bonds grow?

Per run:
- plateau midpoints: minima of the parameter speed |d theta/dt| (diagnostics.md) between the first
  escape and L < 1e-4 L(0), kept if < 0.2 x both neighbouring maxima;
- bond ranks at each plateau: #{k: s_k^e > 0.1 x the teacher's s_k^e on that cut} (a per-value version
  of the diagnostics.md threshold, which with teacher values ~1e-2 counted still-rising values);
- the plateau state W_pl (dense) vs (a) the HSVD truncation W_H of W* to the same ranks
  (projectors on the top-r_e left singular subspaces of W*_(e), all computed from W*, applied root-ward
  edges first) and (b) the nearest of the local minima of L restricted to those ranks (constrained.py):
  distances relative to ||W*||, and the loss gaps;
- bond-growth order: the time each edge's k-th singular value first exceeds 0.5 x the teacher's
  (k = 2, 3, ...), from which the order of rank increments is read.
Usage: uv run python experiments/data_teacher/analyze_ttn.py <config name>
Writes results/<name>_plateaus.csv and results/<name>_growth.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import find_peaks

from common import RESULTS, OUT_ROOT
from constrained import constrained_minima, metric_loss_of_dense, ranked_spec
from run_generic import expand
from setups import build

from ttn.dynamics import Trajectory
from ttn.network import to_dense

HERE = Path(__file__).parent


def matricize(w, below):
    rest = [i for i in range(w.ndim) if i not in below]
    rows = int(np.prod([w.shape[i] for i in below]))
    return w.permute(list(below) + rest).reshape(rows, -1), rest


def hsvd(w_star, tree, ranks):
    """Root-to-leaves HSVD truncation (Grasedyck 2010): projectors from W* itself, applied to W*
    starting with the edges with the most sites below (so later projectors act inside earlier blocks
    and do not raise ranks already fixed)."""
    edges = sorted(tree.edges, key=lambda v: -len(tree.subtree_sites[v]))
    projs = {}
    for v in edges:
        m, _ = matricize(w_star, tree.subtree_sites[v])
        u = torch.linalg.svd(m, full_matrices=False)[0][:, : ranks[v]]
        projs[v] = u @ u.T
    w = w_star.clone()
    n = w.ndim
    for v in edges:
        below = list(tree.subtree_sites[v])
        m, rest = matricize(w, below)
        m = projs[v] @ m
        perm = below + rest
        w = m.reshape([w.shape[i] for i in perm]).permute(list(np.argsort(perm)))
        assert w.ndim == n
    return w


def plateaus(t, loss, speed=None):
    """Plateau midpoints (diagnostics.md: minima of the parameter speed |d theta/dt| between
    events): local minima of the speed after the first escape (L < 0.9 L(0)) and before L < 1e-4 L(0)
    that are < 0.2 x both neighbouring maxima. Falls back to the loss rate -d log L/dt without speed."""
    if speed is None:
        speed = -np.gradient(np.log(np.maximum(loss, 1e-300)), t)
    lo = int(np.argmax(loss < 0.9 * loss[0]))
    hi = int(np.argmax(loss < 1e-4 * loss[0])) if (loss < 1e-4 * loss[0]).any() else len(t) - 1
    v = speed[lo:hi + 1]
    maxima, _ = find_peaks(v)
    out = []
    for a, b in zip(maxima[:-1], maxima[1:]):
        m = a + int(np.argmin(v[a:b + 1]))
        if v[m] < 0.2 * min(v[a], v[b]):
            out.append((lo + m, lo + a, lo + b))
    return out, maxima + lo


def analyze(name, case, tree, sigma, seed, spec, teacher, cache):
    dense = OUT_ROOT / f"{name}_dense" / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
    path = OUT_ROOT / name / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
    if dense.exists():  # library two-stage run (run_dense.py), preferred: resolves short plateaus
        tr = Trajectory.load(dense)
        t, loss, tsp, y = tr.data["t"], tr.data["loss"], tr.data["teacher_spectra"], tr.data["params"]
        speed = tr.data["speed"]
        sp = np.nan_to_num(tr.data["spectra"])
        edges = list(spec.tree.edges)
    elif path.exists():
        d = np.load(path, allow_pickle=True)
        t, loss, sp, tsp, y = d["t"], d["loss"], d["spectra"], d["teacher_spectra"], d["y"]
        edges = list(d["edges"])
        speed = None
    else:
        return [], []
    w_star = teacher.dense()
    lfun = metric_loss_of_dense(w_star, None)
    rows = []
    pls, _ = plateaus(t, loss, speed)
    for i, (m, a, b) in enumerate(pls):
        # rank: k-th singular value above 0.1 x the teacher's k-th value on that cut
        ranks = {v: max(1, int(sum(1 for k in range(sp.shape[2]) if tsp[j, k] > 1e-3
                                   and sp[m, j, k] > 0.1 * tsp[j, k]))) for j, v in enumerate(edges)}
        w_pl = to_dense(spec, spec.unflatten(torch.from_numpy(y[m].copy())))
        w_h = hsvd(w_star, spec.tree, ranks)
        key = (case["tag"], tree, tuple(sorted(ranks.items())))
        if key not in cache:
            cache[key] = constrained_minima(ranked_spec(spec, ranks), lfun, n_starts=6)
        mins = cache[key]
        dists = [float((w_pl - mm["W"]).norm()) for mm in mins]
        jc = int(np.argmin(dists))
        nrm = float(w_star.norm())
        rows.append(dict(tag=case["tag"], tree=tree, sigma=sigma, seed=seed, plateau=i,
                         t=t[m], loss=loss[m], ranks=str([ranks[v] for v in edges]),
                         dist_crit=dists[jc] / nrm, dist_hsvd=float((w_pl - w_h).norm()) / nrm,
                         dist_crit_hsvd=float((mins[jc]["W"] - w_h).norm()) / nrm,
                         loss_crit=mins[jc]["loss"], loss_hsvd=float(lfun(w_h)),
                         crit_is_global=jc == 0, n_crit=len(mins)))
    growth = []
    for j, v in enumerate(edges):
        for k in range(1, sp.shape[2]):
            if tsp[j, k] <= 1e-3:
                continue
            hit = np.nonzero(sp[:, j, k] > 0.5 * tsp[j, k])[0]
            growth.append(dict(tag=case["tag"], tree=tree, sigma=sigma, seed=seed, edge=v,
                               k=k + 1, teacher_value=tsp[j, k],
                               t_rise=t[hit[0]] if len(hit) else np.nan))
    return rows, growth


def per_case(args):
    name, case, cfg = args
    torch.set_num_threads(1)
    rows, growth, cache = [], [], {}
    for tree in cfg["trees"]:
        built = build(dict(case, tree=tree), 0)
        if built is None:
            continue
        spec, teacher = built[0], built[1]
        for sigma in cfg["sigmas"][tree]:
            for seed in range(*cfg["seeds"]):
                r, g = analyze(name, case, tree, sigma, seed, spec, teacher, cache)
                rows += r
                growth += g
    return rows, growth


def main():
    name = sys.argv[1]
    cfg = yaml.safe_load(open(HERE / "configs" / f"{name}.yaml"))
    from multiprocessing import Pool
    cases = expand(cfg)
    with Pool(min(len(cases), 8)) as pool:
        res = pool.map(per_case, [(name, c, cfg) for c in cases])
    rows = [r for rr, _ in res for r in rr]
    growth = [g for _, gg in res for g in gg]
    pd.DataFrame(rows).to_csv(RESULTS / f"{name}_plateaus.csv", index=False)
    pd.DataFrame(growth).to_csv(RESULTS / f"{name}_growth.csv", index=False)
    df = pd.DataFrame(rows)
    if len(df):
        print(df.groupby(["tag", "ranks"])[["loss", "loss_crit", "loss_hsvd", "dist_crit",
                                            "dist_hsvd", "dist_crit_hsvd"]].median().to_string())


if __name__ == "__main__":
    main()
