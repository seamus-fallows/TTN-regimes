"""Test of notes/theory_saddle_order.md on the scored teachers: generalized piggyback operator at every
rank-one saddle the hopper predicts.

For every hopper case of the four scored experiments (prediction caches), each first-escape candidate
(main: attractor of the escape problem; rare: a saddle of it) gives a rank-one critical point
W_S = s_q (x)_v q_v of the loss (one learned component, as the theorem assumes). In whitened coordinates
(W' = Sigma^(1/2) W for product covariances) and core bases rotated so that q_v = e_0, compute
k* = min |Steiner(B)| over sectors with R_B != 0 and lambda_max of the generalized piggyback operator S
restricted to the sets of degree <= l, for every l < k*/2 (`experiments/theory_checks/saddle_order/
gp_operator.py`, read only). Order = k* if every lambda_max < s_q (Theorems 1, 1'); 2 l* if
lambda_max(S|L_l*) > s_q (Theorem 2, pattern teachers with a spare bond index).

Saddles with several learned components (every later saddle of a multi-component teacher) are outside
the theorem; `later_saddles()` lists the hopper events with k* >= 5 that leave such saddles.

usage: PYTHONPATH=. python experiments/saddle_hopper/gp_check.py [--workers 8]
Writes results/gp_saddle_order.csv and results/gp_later_saddles.csv.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "experiments" / "theory_checks" / "saddle_order"))
import predict  # noqa: E402
from gp_operator import core_grouped, deg, sector, steiner_cores  # noqa: E402

RES = HERE / "results"
DATASETS = ("locality", "s2s_core", "data_teacher", "order")
PATTERN_KINDS = ("pattern", "odeco", "aligned_cov")  # whitened residual is a pattern tensor
TOL_SECTOR = 1e-8  # relative: a sector counts as nonzero above TOL_SECTOR * ||R||
TOL_CMP = 1e-9  # lambda_max within TOL_CMP * s_q of s_q: borderline (theory silent)


# ---------------------------------------------------------------- linear algebra on core-grouped tensors


def whiten(W: np.ndarray, metrics) -> np.ndarray:
    """Sigma^(1/2) W for a product covariance (None = whitened already)."""
    if metrics is None:
        return W
    for i, m in enumerate(metrics):
        w, u = np.linalg.eigh(0.5 * (m + m.T))
        h = (u * np.sqrt(np.clip(w, 0, None))) @ u.T
        W = np.moveaxis(np.tensordot(h, W, axes=([1], [i])), 0, i)
    return W


def contract_except(T: np.ndarray, ys, v: int) -> np.ndarray:
    """<T, (x)_{u != v} y_u> as a vector on leg v."""
    X = T
    for u in reversed(range(T.ndim)):
        if u != v:
            X = np.tensordot(X, ys[u], axes=([u], [0]))
    return X


def hopm_refine(T: np.ndarray, ys, iters: int = 5000, tol: float = 1e-15):
    """Alternating power iteration on a core-grouped tensor from ys (one unit vector per core)."""
    ys = [y / np.linalg.norm(y) for y in ys]
    for _ in range(iters):
        delta = 0.0
        for v in range(T.ndim):
            if T.shape[v] == 1:
                continue
            g = contract_except(T, ys, v)
            n = np.linalg.norm(g)
            if n == 0:
                break
            g = g / n
            delta = max(delta, float(np.linalg.norm(g - ys[v])))
            ys[v] = g
        if delta < tol:
            break
    s = float(contract_except(T, ys, 0) @ ys[0])
    if s < 0:
        v0 = next(v for v in range(T.ndim) if T.shape[v] > 1)
        ys[v0] = -ys[v0]
        s = -s
    return ys, s


def householder(q: np.ndarray) -> np.ndarray:
    """Orthogonal symmetric H with H e_0 = q (first column q)."""
    e = np.zeros_like(q)
    e[0] = 1.0
    u = e - q
    n = np.linalg.norm(u)
    if n < 1e-14:
        return np.eye(len(q))
    u = u / n
    return np.eye(len(q)) - 2.0 * np.outer(u, u)


def rotate(R: np.ndarray, qs) -> np.ndarray:
    """R in core bases whose index 0 is q_v (index 0 = old/learned direction, >= 1 = q_v^perp)."""
    X = R
    for v, q in enumerate(qs):
        if len(q) > 1:
            X = np.moveaxis(np.tensordot(householder(q), X, axes=([0], [v])), 0, v)
    return X


# ---------------------------------------------------------------- k* and the generalized piggyback operator


def k_star(tree, Rc, tol):
    """min |Steiner(B)| over nonempty B (site cores only: T(B) = 0 otherwise) with ||R_B|| > tol."""
    site_cores = [v for v in range(tree.n_nodes) if Rc.shape[v] > 1]
    best, where = math.inf, None
    for size in range(1, len(site_cores) + 1):
        for B in itertools.combinations(site_cores, size):
            k = len(steiner_cores(tree, B))
            if k < best and np.linalg.norm(sector(Rc, set(B))) > tol:
                best, where = k, B
    return best, where


def low_sets(tree, Rc, ell):
    """Nonempty sets of site cores with degree <= ell (sets containing a site-less core have T(C) = 0)."""
    site_cores = [v for v in range(tree.n_nodes) if Rc.shape[v] > 1]
    return [
        frozenset(C)
        for size in range(1, len(site_cores) + 1)
        for C in itertools.combinations(site_cores, size)
        if deg(tree, C) <= ell
    ]


def s_matrix(tree, Rc, sets):
    """Matrix of S on (+)_{C in sets} T(C): block (C1, C2) = R_{C1 u C2} for disjoint C1, C2."""
    dims = [int(np.prod([Rc.shape[v] - 1 for v in sorted(C)])) for C in sets]
    off = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    S = np.zeros((off[-1], off[-1]))
    for i, C1 in enumerate(sets):
        for j, C2 in enumerate(sets):
            if j <= i or C1 & C2:
                continue
            B = sorted(C1 | C2)
            block = sector(Rc, set(B))
            pos1 = [B.index(v) for v in sorted(C1)]
            pos2 = [B.index(v) for v in sorted(C2)]
            M = np.transpose(block, pos1 + pos2).reshape(dims[i], dims[j])
            S[off[i] : off[i + 1], off[j] : off[j + 1]] = M
            S[off[j] : off[j + 1], off[i] : off[i + 1]] = M.T
    return S, off


def steiner_edges(tree, C):
    P = steiner_cores(tree, C)
    return {v for v in P if tree.parent[v] in P}  # edge v = (v, parent v)


def gp_levels(tree, Rc, s_q, ks):
    """lambda_max(S|L_l) for l < k*/2, the first level above s_q, and its Perron support."""
    lams, top = {}, None
    for ell in range(1, math.ceil(ks / 2)):
        if 2 * ell >= ks:
            break
        sets = low_sets(tree, Rc, ell)
        S, off = s_matrix(tree, Rc, sets)
        if not S.size:
            lams[ell] = 0.0
            continue
        w, V = np.linalg.eigh(S)
        lams[ell] = float(w[-1])
        if top is None and w[-1] > s_q * (1 + TOL_CMP):
            x = V[:, -1]
            supp = np.flatnonzero(np.abs(x) > 1e-6 * np.abs(x).max())
            nodes = [(sets[int(np.searchsorted(off, i, side="right") - 1)], i) for i in supp]
            top = (ell, nodes)
    return lams, top


def index_need(tree, nodes):
    """Max over edges of the number of multi-core support nodes whose Steiner tree contains the edge
    (Theorem 2's index condition: chi_e - 1 >= this number)."""
    count = {}
    for C, _ in nodes:
        if len(C) >= 2:
            for e in steiner_edges(tree, C):
                count[e] = count.get(e, 0) + 1
    return max(count.values(), default=0)


# ---------------------------------------------------------------- one case


def rank_one_factors(Xg: np.ndarray):
    """Unit factors of an (exactly) rank-one core-grouped tensor."""
    ys = []
    for v in range(Xg.ndim):
        if Xg.shape[v] == 1:
            ys.append(np.ones(1))
            continue
        M = np.moveaxis(Xg, v, 0).reshape(Xg.shape[v], -1)
        ys.append(np.linalg.svd(M, full_matrices=False)[0][:, 0])
    return ys


def outer(ys) -> np.ndarray:
    X = ys[0]
    for y in ys[1:]:
        X = np.multiply.outer(X, y)
    return X


def teacher_label(t: dict) -> str:
    if t["kind"] == "pattern":
        return " ".join(
            f"{''.join(map(str, p))}({w:g})" for p, w in zip(t["patterns"], t["weights"])
        )
    if t["kind"] in ("odeco", "nonorth", "misaligned"):
        rest = {k: v for k, v in t.items() if k not in ("kind", "weights")}
        return f"{t['kind']} {[round(w, 4) for w in t['weights']]} {rest if rest else ''}".strip()
    return json.dumps({k: v for k, v in t.items() if k != "site_eigs"})


def run_case(case):
    import torch

    torch.set_num_threads(1)
    from hopper import SaddleHopper

    setting = predict.build(case)
    tree, d = setting.tree, setting.d
    h = SaddleHopper(setting, seed=case.get("seed", 0))
    metrics = None if setting.metrics is None else [m.numpy() for m in setting.metrics]
    Tg = core_grouped(tree, d, whiten(setting.w_star.numpy(), metrics))
    rows = []
    for ci, cand in enumerate(h.first_candidates()):
        X = whiten(h.candidate_dense(cand["ys"]).numpy(), metrics)
        ys, s_q = hopm_refine(Tg, rank_one_factors(core_grouped(tree, d, X)))
        R = Tg - s_q * outer(ys)
        nR = float(np.linalg.norm(R))
        row = dict(
            candidate=ci,
            branch="main" if cand["n_unstable"] == 0 else "rare",
            gamma=cand["gamma"],
            overlaps=None if cand["overlaps"] is None else [round(float(a), 4) for a in cand["overlaps"]],
            s_q=s_q,
            residual_norm=nR,
        )
        if nR <= 1e-10 * float(np.linalg.norm(Tg)):
            rows.append(dict(row, k_star=np.nan, order=np.nan, note="teacher is rank one"))
            continue
        Rc = rotate(R, ys)
        tol = TOL_SECTOR * nR
        crit = max(
            (np.linalg.norm(sector(Rc, {v})) for v in range(tree.n_nodes) if Rc.shape[v] > 1),
            default=0.0,
        )
        ks, where = k_star(tree, Rc, tol)
        lams, top = gp_levels(tree, Rc, s_q, ks)
        need = index_need(tree, top[1]) if top and case["teacher"]["kind"] in PATTERN_KINDS else None
        row.update(
            criticality=float(crit) / nR,
            k_star=ks,
            k_star_sector=",".join(map(str, where)),
            order=2 * top[0] if top else ks,
            order_drop=bool(top),
            level=top[0] if top else None,
            index_need=need,
            index_ok=None if need is None else bool(case["chi"] - 1 >= need),
            borderline=any(abs(lam / s_q - 1) < 1e-6 for lam in lams.values()),
            **{f"lam_l{ell}_over_sq": lam / s_q for ell, lam in lams.items()},
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------- all scored cases


def scored_cases():
    """Unique (tree, d, chi, teacher) cases of the four prediction caches, with their datasets and inits."""
    out = {}
    for ds in DATASETS:
        for key in json.loads((RES / f"predictions_{ds}.json").read_text()):
            c = json.loads(key)
            base = {k: c[k] for k in ("teacher", "tree", "n", "d", "chi")}
            u = out.setdefault(json.dumps(base, sort_keys=True), dict(case=base, datasets=set(), inits=set()))
            u["datasets"].add(ds)
            u["inits"].add(c.get("init", "fanin"))
    return list(out.values())


def later_saddles():
    """Hopper events with k >= 5 that leave a saddle with >= 2 learned components (outside the theorem)."""
    rows = []
    for ds in DATASETS:
        for key, pred in json.loads((RES / f"predictions_{ds}.json").read_text()).items():
            c = json.loads(key)
            for s in pred["sequences"]:
                evs = s["events"]
                for a, b in zip(evs[:-1], evs[1:]):
                    if len(a["learned"]) >= 2 and (b.get("k") or 0) >= 5:
                        rows.append(
                            dict(
                                dataset=ds,
                                tree=f"{c['tree']}{c['n'] or ''}",
                                chi=c["chi"],
                                init=c.get("init"),
                                teacher=teacher_label(c["teacher"]),
                                pclass=s["pclass"],
                                learned_before=",".join(map(str, a["learned"])),
                                event=b["kind"],
                                k=b["k"],
                            )
                        )
    return pd.DataFrame(rows).drop_duplicates()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=predict.workers_default())
    args = ap.parse_args()
    todo = scored_cases()
    print(f"{len(todo)} unique (tree, d, chi, teacher) cases", flush=True)
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run_case, [u["case"] for u in todo], chunksize=1)
    rows = []
    for u, res in zip(todo, results):
        c = u["case"]
        for r in res:
            rows.append(
                dict(
                    datasets="+".join(sorted(u["datasets"])),
                    inits="+".join(sorted(u["inits"])),
                    tree=f"{c['tree']}{c['n'] or ''}",
                    d=c["d"],
                    chi=c["chi"],
                    teacher_kind=c["teacher"]["kind"],
                    teacher=teacher_label(c["teacher"]),
                    **r,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "gp_saddle_order.csv", index=False)
    lat = later_saddles()
    lat.to_csv(RES / "gp_later_saddles.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
    print(df.groupby(["datasets", "branch"]).size().to_string())
    print("criticality max:", df.criticality.max())
    print("k* >= 5 saddles:", int((df.k_star >= 5).sum()), "of", len(df))
    drop = df[df.order_drop.astype(str) == "True"]
    print(f"order below k*: {len(drop)} saddles")
    cols = ["datasets", "tree", "chi", "teacher", "branch", "overlaps", "s_q", "k_star", "order",
            "level", "index_ok"] + [c for c in df.columns if c.startswith("lam_l")]
    print(drop[cols].to_string())
    print(f"later saddles (>= 2 learned) with k >= 5: {len(lat)} events,",
          lat.groupby("dataset").size().to_dict())


if __name__ == "__main__":
    main()
