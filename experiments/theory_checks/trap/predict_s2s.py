"""Approximate per-run trap predictions for the s2s_core runs (odeco teacher, 3 components).

Phase-2 criterion (notes/theory_trap.md, section 5): after two components q1, q2 are learned, the last
component r needs the remaining new bond space nu (complement of the two learned directions) at every
route end (= every tree leaf, since odeco routes use all cores). At a leaf end u with partner w the
conserved GN quantity is C'_u = X^T X - Y^T Y, X = leaf's nu-columns orthogonal to the learned
physical patterns, Y = partner's nu-slice with learned indices on its other legs minus the learned
(diagonal) combinations. Approximations: entries frozen at their values in the correlation-flow state
of the race between q2 and r (q1 removed); learned directions from the correlation flows of the full
teacher (q1) and of {q2, r} (q2, orthogonalized against q1); no phase-1 GN decay.

usage: PYTHONPATH=. python experiments/theory_checks/trap/predict_s2s.py --out <results dir> --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "s2s_core"))
from s2s import gauge_balance  # noqa: E402
from trapcrit import contract, correlation_flow, learned_directions, neighbours, new_block, pattern_vector  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import CPTeacher, odeco_teacher  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

TREES = {
    "binary4": ("binary", 4),
    "mps4": ("mps", 4),
    "binary8": ("binary", 8),
    "ht4": ("ht", 4),
    "mps8": ("mps", 8),
}
WEIGHTS = {"separated": [3.0, 2.0, 1.0], "close": [1.0, 0.85, 0.7]}


def sub_teacher(teacher, keep):
    return CPTeacher(teacher.weights[keep], teacher.factors[keep])


def bases(w1, w2):
    """Per edge: learned basis B (chi x 2, Gram-Schmidt of w1, w2) and new basis N (complement)."""
    out = {}
    for e in w1:
        q, _ = torch.linalg.qr(
            torch.cat(
                [w1[e][:, None], w2[e][:, None], torch.eye(len(w1[e]), dtype=w1[e].dtype)], dim=1
            )
        )
        out[e] = (q[:, :2], q[:, 2 : len(w1[e])])
    return out


def block2(spec, teacher, cores, bs, learned, v, e):
    """Core v with leg e on the new space N_e, its other edges on the learned bases B_f, physical
    legs kept; flattened to (rest, c_e) and projected orthogonally to the learned (diagonal)
    combinations p_g (x) e_g (x) ... (x) e_g, g = index of a learned component."""
    tree = spec.tree
    ops, other = {}, []
    for x, f in neighbours(tree, v):
        ops[("edge", f)] = bs[f][1] if f == e else bs[f][0]
    t = contract(cores[v], tree, v, ops)
    kept = [lg for lg in __import__("trapcrit").legs(tree, v) if lg[0] != "root"]
    pos_e = kept.index(("edge", e))
    t = torch.movedim(t, pos_e, -1)
    m = t.reshape(-1, t.shape[-1])
    n_other = len(kept) - 1 - len(tree.sites[v])
    diag = []
    for g, comp in enumerate(learned):
        vec = pattern_vector(teacher, tree, v, comp) if tree.sites[v] else torch.ones(1, dtype=torch.float64)
        for _ in range(n_other):
            vec = torch.kron(vec, torch.nn.functional.one_hot(torch.tensor(g), 2).double())
        diag.append(vec)
    dmat, _ = torch.linalg.qr(torch.stack(diag, dim=1))
    return m - dmat @ (dmat.T @ m)


def phase2(spec, teacher, cores0):
    """Predicted order (q1, q2), dormant r and per-leaf C' (largest eigenvalue) for the last component."""
    R = teacher.n_components
    c1, q1 = correlation_flow(spec, teacher, cores0)
    w1 = learned_directions(spec, c1)
    keep = [r for r in range(R) if r != q1]
    c2, i2 = correlation_flow(spec, sub_teacher(teacher, keep), cores0)
    q2 = keep[i2]
    r = [x for x in keep if x != q2][0]
    bs = bases(w1, learned_directions(spec, c2))
    tree = spec.tree
    leaves = [v for v in range(tree.n_nodes) if len(neighbours(tree, v)) == 1]
    ends = []
    for u in leaves:
        ((w, e),) = neighbours(tree, u)
        X = block2(spec, teacher, c2, bs, [q1, q2], u, e)
        Y = block2(spec, teacher, c2, bs, [q1, q2], w, e)
        ev = torch.linalg.eigvalsh(X.T @ X - Y.T @ Y)
        # phase 1 (only q1 learned, both other components dormant): j = 2 GN at ends whose partner
        # carries sites; the end keeps n1 = #positive eigenvalues of C1 new directions
        n1 = None
        if tree.sites[w]:
            X1 = new_block(spec, teacher, c1, w1, q1, u, {e})
            Y1 = new_block(spec, teacher, c1, w1, q1, w, {e})
            n1 = int((torch.linalg.eigvalsh(X1.T @ X1 - Y1.T @ Y1) > 0).sum())
        ends.append(dict(u=u, C=float(ev.max()), n1=n1, dying=bool(ev.max() <= 0),
                         dying12=bool(ev.max() <= 0) or (n1 is not None and n1 <= 1)))
    n = sum(x["dying"] for x in ends)
    n12 = sum(x["dying12"] for x in ends)

    def v(k):
        return "dead" if k >= 2 else "pinned" if k == 1 else "alive"

    return dict(q1=q1, q2=q2, r=r, n_dying=n, verdict=v(n), verdict12=v(n12),
                C_ends=json.dumps([round(x["C"], 4) for x in ends]), n1_ends=json.dumps([x["n1"] for x in ends]))


def one(key):
    tree, chi, teacher_name, init, seed = key
    torch.set_num_threads(1)
    kind, n = TREES[tree]
    spec = TTNSpec.uniform(make_tree(kind, n), 4, chi)
    teacher = odeco_teacher(n, 4, WEIGHTS[teacher_name], seed=1000)
    cores0 = spec.init(1.0, seed=seed)
    if init == "balanced":
        cores0 = gauge_balance(spec, cores0)
    try:
        return key, phase2(spec, teacher, cores0)
    except Exception as exc:
        return key, dict(error=repr(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    runs = pd.read_csv(ROOT / "experiments" / "s2s_core" / "results" / "main_runs.csv")
    runs = runs[runs.tree != "binary4"]
    keys = sorted({(r.tree, int(r.chi), r.teacher, r.init, int(r.seed)) for r in runs.itertuples()}, key=str)
    print(f"{len(runs)} runs, {len(keys)} predictions", flush=True)
    with Pool(a.workers) as pool:
        preds = dict(pool.imap_unordered(one, keys))
    rows = []
    for r in runs.itertuples():
        p = preds[(r.tree, int(r.chi), r.teacher, r.init, int(r.seed))]
        rows.append(dict(sweep=r.sweep, tree=r.tree, K=r.K, chi=r.chi, teacher=r.teacher, init=r.init,
                         sigma=r.sigma, seed=r.seed, obs=r._9, n_learned=r.n_learned, order=r.order, **p))
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "s2s_predictions.csv", index=False)
    print(df.groupby(["tree", "chi", "teacher", "init", "verdict", "obs"]).size().to_string())


if __name__ == "__main__":
    main()
