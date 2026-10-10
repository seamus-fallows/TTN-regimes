"""Route amplitudes at the first (bond-rank-1) plateau of binary-tree runs.

Usage: uv run python experiments/real_data/route_amplitudes.py <summary.json> <runs_dir> <out.json> <workers> [filter]

At a bond-rank-1 point a binary TTN computes f1(x) = c prod_v q_v(x) over its bottom cores v
(q_v(x) = g_v . Phi_v(x), Phi_v = phi_s (x) phi_s' of the core's two sites). A route that raises the
rank on the path between bottom cores u and w (k = path length + 1 cores; siblings: k = 3) changes
f by eps^k a.Phi_u(x) b.Phi_w(x) h_uw(x), h_uw = c prod_{v != u, w} q_v, and enters the loss at
degree k through the residual r = f - y:
    M_uw = (1/n) sum_j r_j h_uw(x_j) Phi_u(x_j) Phi_w(x_j)^T     (4 x 4),
projected on the complements of g_u, g_w (the old directions; at an exact critical point M_uw g_w = 0
and g_u^T M_uw = 0). The route amplitude is lambda_uw = |P_u M_uw P_w|_2: the k-core escape rate is
lambda_uw times old-core factors common to all routes of the same k (theory_saddles.md R1,
locality.md section 2). The state is the plateau midpoint (minimum of |d theta/dt| between the first
two loss drops), reached by re-integrating from the saved end of phase 1 (LSODA, same tolerances).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from analysis import load  # noqa: E402
from data import Task, build  # noqa: E402
from run import TASK_KEYS  # noqa: E402

from ttn.dynamics import Recorder, integrate_gf, merged_rows  # noqa: E402
from ttn.network import forward  # noqa: E402


def plateau_state(run: dict, row: dict):
    """(rp, theta) at the speed minimum between the first two loss drops (None if < 2 drops)."""
    if len(row.get("drops", [])) < 2:
        return None, None, None
    c = run["meta"]["case"]
    lg = run["p2_log"]
    d1, d2 = row["drops"][0], row["drops"][1]
    inside = np.flatnonzero((lg["t"] > d1["t_hi"]) & (lg["t"] < d2["t_lo"]))
    if inside.size == 0:
        return None, None, None
    tau_mid = float(lg["t"][inside[np.argmin(lg["speed"][inside])]])
    task = Task(**{k: (tuple(c[k]) if k in ("pos", "neg") else c[k]) for k in TASK_KEYS})
    rp = build(task, c["tree"], c["ordering"], c["chi"], c["features"])
    theta_a = torch.as_tensor(run["p1_log"]["theta_end"])
    tr = integrate_gf(rp.problem, theta_a, tau_mid, log_times=[tau_mid], rtol=c["rtol"],
                      atol=c["atol"], method="lsoda",
                      recorder=Recorder(spectra=False, conservation=False, fine=False,
                                        overlaps=False))
    return rp, torch.as_tensor(tr["theta_end"]), tau_mid


def amplitudes(rp, theta) -> dict:
    spec = rp.problem.spec
    t = spec.tree
    x, y = rp.problem.dataset
    cores = spec.unflatten(theta)
    bottom = [v for v in range(t.n_nodes) if len(t.sites[v]) == 2]
    with torch.no_grad():
        f = forward(spec, cores, x)
        r = (f - y).numpy()
        phis, gs, qs = {}, {}, {}
        for v in bottom:
            s1, s2 = t.sites[v]
            phi = torch.einsum("na,nb->nab", x[:, s1], x[:, s2]).reshape(len(x), 4).numpy()
            mat = cores[v].reshape(4, -1).numpy()
            u, sv, _ = np.linalg.svd(mat)
            g = u[:, 0] * sv[0]
            phis[v], gs[v], qs[v] = phi, g, phi @ g
        prod_all = np.prod(np.stack([qs[v] for v in bottom]), axis=0)
        c = float(f.numpy() @ prod_all / (prod_all @ prod_all))
        rank1_err = float(np.sqrt(np.mean((f.numpy() - c * prod_all) ** 2) / np.mean(f.numpy() ** 2)))
    out = []
    for i, u in enumerate(bottom):
        for w in bottom[i + 1:]:
            h = c * np.prod(np.stack([qs[v] for v in bottom if v not in (u, w)]), axis=0)
            m = (phis[u] * (r * h)[:, None]).T @ phis[w] / len(r)
            pu = np.eye(4) - np.outer(gs[u], gs[u]) / (gs[u] @ gs[u])
            pw = np.eye(4) - np.outer(gs[w], gs[w]) / (gs[w] @ gs[w])
            lam = float(np.linalg.norm(pu @ m @ pw, 2))
            raw = float(np.linalg.norm(m, 2))
            k = len(t.path(u, w))
            out.append({"u": u, "w": w, "k": k, "lam": lam, "lam_unprojected": raw,
                        "sites_u": list(t.sites[u]), "sites_w": list(t.sites[w])})
    return {"pairs": out, "c": c, "rank1_rel_err": rank1_err, "bottom": bottom,
            "train_loss": float(0.5 * np.mean(r ** 2))}


def one(args):
    row, runs_dir = args
    try:
        run = load(Path(runs_dir) / f"{row['name']}.npz")
        rp, theta, tau_mid = plateau_state(run, row)
        if rp is None:
            return {"name": row["name"], "skipped": "fewer than two drops"}
        res = amplitudes(rp, theta)
        res.update(name=row["name"], tau_mid=tau_mid, order=run["meta"]["order"],
                   **{k: row[k] for k in ("tree", "ordering", "features", "sigma", "seed")},
                   first_rank2_edges=row.get("first_rank2_edges"),
                   edges=run["meta"]["edges"])
        return res
    except Exception as exc:
        import traceback
        return {"name": row["name"], "error": traceback.format_exc()}


def main(summary_json, runs_dir, out_json, workers, flt=""):
    rows = [r for r in json.load(open(summary_json))
            if r.get("tree") == "binary" and any(f in r.get("name", "") for f in flt.split(","))
            and "error" not in r]
    with Pool(int(workers)) as pool:
        res = pool.map(one, [(r, runs_dir) for r in rows])
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_json, "w"))
    print(len(res), "runs;", sum("error" in r for r in res), "errors;",
          sum("skipped" in r for r in res), "skipped")


if __name__ == "__main__":
    main(*sys.argv[1:])
