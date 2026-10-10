"""Basis invariance of the k = 4 route verdict (notes/theory_trap.md section 2.6).

MPS4 {0,3} and {0,1,2,3}, d = 2, chi = 4: the end edges have one new direction (c = 1), the inner
edge three (c = 3). For the committed seeds 0-7 (small init) and the Monte Carlo draws of {0,3}
(seeds 100000+, small init and saddle start), every route with two dying ends is decided by the
reduced route ODE in three ways:
  first_vector  the v1 scalar model (first basis vector of the inner new space), in the code's basis
                and after n_rot Haar-random rotations of the inner new space;
  tensor        vector route amplitudes (trapcrit.reduced_route default), in the same bases;
  effective     the scalar model with the effective inner amplitudes
                b1, b2 = (sqrt(S + 2p) +- sqrt(S - 2p)) / 2,  p = <B1, B2>,  S = |B1|^2 + |B2|^2,
                which section 2.6 shows is the same flow (end and route trajectories compared).
Writes results/k4_invariance.csv and prints a summary.

usage (from the worktree root):
  PYTHONPATH=. python experiments/theory_checks/trap/k4_invariance.py --workers 6
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "locality"))
from rules import get_tree  # noqa: E402
from trapcrit import plateau_state, predict, reduced_route, route_tensors  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

WEIGHTS = (1.0, 0.5)


def haar(c, rng):
    q, r = np.linalg.qr(rng.standard_normal((c, c)))
    return torch.from_numpy(q * np.sign(np.diag(r)))


def setup(S, chi=4, d=2):
    tree = get_tree("mps", 4)
    spec = TTNSpec.uniform(tree, d, chi)
    patterns = [[0] * 4, [1 if i in S else 0 for i in range(4)]]
    return spec, pattern_teacher(list(WEIGHTS), patterns, d=d, seed=1000)


def trajectory_gap(a, b):
    """Max deviation of the end entries and of the route overlap P between two reduced-route
    solutions (full=True output), relative to their largest values, on a common log time grid.
    An escape is a finite-time blow-up, whose last stretch amplifies O(rtol) timing differences:
    escapes are compared up to 0.9 of the earlier stop."""
    (_, sa, la), (_, sb, lb) = a, b
    t_hi = min(sa.t[-1], sb.t[-1]) * (0.9 if 1 in (sa.status, sb.status) else 1.0)
    t = np.concatenate([[0.0], np.logspace(np.log10(t_hi) - 8, np.log10(t_hi), 400)])
    ya, yb = sa.sol(t), sb.sol(t)

    def overlap(y, lay):
        offs, shapes, subs = lay["offs"], lay["shapes"], lay["subs"]
        Ts = [y[offs[i] : offs[i + 1]].reshape(shapes[i] + (len(t),)) for i in range(len(shapes))]
        eq = ",".join(s + "z" for s in subs) + "->z"
        return np.einsum(eq, *Ts)

    gaps = [
        np.abs(ya[p] - yb[q]).max() / np.abs(ya[p]).max() for p, q in zip(la["ends"], lb["ends"])
    ]
    Pa, Pb = overlap(ya, la), overlap(yb, lb)
    gaps.append(np.abs(Pa - Pb).max() / np.abs(Pa).max())
    return float(max(gaps))


def one(item):
    S, start, seed, n_rot = item
    torch.set_num_threads(1)
    spec, teacher = setup(S)
    tree = spec.tree
    cores0 = spec.init(1.0, seed=seed)
    cores, q, w = plateau_state(spec, teacher, cores0, start)
    r = 1 - q
    p = predict(spec, teacher, cores, w, q, S, r=r, weights=WEIGHTS)
    row = dict(
        S="".join(map(str, S)),
        start=start,
        seed=seed,
        q=q,
        k=p["k"],
        n_dying=p["n_dying"],
        verdict=p["verdict"],
        C_ends=[round(x["C"], 5) for x in p["ends"]],
    )
    if p["verdict_k4"] is None:
        return row
    args = (spec, teacher, cores, w, q, r, S, WEIGHTS[q], WEIGHTS[r])
    nodes = tree.steiner_nodes(list(S))
    inner = [
        e for e in tree.edges if e in nodes and tree.parent[e] in nodes and spec.bond_dims[e] > 2
    ]
    order, T, subs = route_tensors(spec, teacher, cores, w, r, nodes)
    B1, B2 = T[1].reshape(-1), T[2].reshape(-1)  # MPS4: route 0-1-2-3, inner cores 1 and 2
    pp, SS = float(B1 @ B2), float(B1 @ B1 + B2 @ B2)
    sp, sm = np.sqrt(SS + 2 * pp), np.sqrt(max(SS - 2 * pp, 0.0))
    eff = (order, [T[0], np.array([[(sp + sm) / 2]]), np.array([[(sp - sm) / 2]]), T[3]], subs)
    full_t = reduced_route(*args, full=True)
    full_e = reduced_route(*args, tensors=eff, full=True)
    rng = np.random.default_rng(10_000 + seed)
    rots = [{e: haar(spec.bond_dims[e] - 1, rng) for e in inner} for _ in range(n_rot)]
    row.update(
        v_first=reduced_route(*args, mode="first_vector"),
        v_tensor=full_t[0],
        v_eff=full_e[0],
        first_rot="".join(reduced_route(*args, rot=R, mode="first_vector")[0] for R in rots),
        tensor_rot="".join(reduced_route(*args, rot=R)[0] for R in rots),
        traj_gap=trajectory_gap(full_t, full_e),
        x_u=float(T[0].item()),
        x_u2=float(T[3].item()),
        p=pp,
        S_inner=SS,
        cos=pp / np.sqrt(float(B1 @ B1) * float(B2 @ B2)),
        p_first=float(B1[0] * B2[0]),
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n_rot", type=int, default=4)
    ap.add_argument("--n_small", type=int, default=60)
    ap.add_argument("--n_saddle", type=int, default=400)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    items = [(S, "small", s, a.n_rot) for S in [(0, 3), (0, 1, 2, 3)] for s in range(8)]
    items += [((0, 3), "small", 100000 + i, a.n_rot) for i in range(a.n_small)]
    items += [((0, 3), "saddle", 100000 + i, a.n_rot) for i in range(a.n_saddle)]
    with Pool(a.workers) as pool:
        rows = pool.map(one, items, chunksize=4)
    df = pd.DataFrame(rows)
    df["draws"] = np.where(df.seed < 100000, "committed", "montecarlo")
    df.to_csv(Path(a.out) / "k4_invariance.csv", index=False)
    ode = df[df.v_tensor.notna()].copy()
    ode["first_flips"] = ode.apply(lambda x: len(set(x.first_rot) | {x.v_first[0]}) > 1, axis=1)
    ode["tensor_invariant"] = ode.apply(lambda x: set(x.tensor_rot) == {x.v_tensor[0]}, axis=1)
    print(f"{len(df)} draws, {len(ode)} with two dying ends (k = 4 ODE)")
    g = ode.groupby(["draws", "S", "start"])
    print(
        g.agg(
            n=("seed", "size"),
            first_basis_dependent=("first_flips", "sum"),
            tensor_invariant=("tensor_invariant", "sum"),
            eff_equal=("v_eff", lambda s: int((s == ode.loc[s.index, "v_tensor"]).sum())),
            max_traj_gap=("traj_gap", "max"),
        ).to_string()
    )
    print(pd.crosstab([ode.draws, ode.S, ode.start, ode.v_first], ode.v_tensor).to_string())


if __name__ == "__main__":
    main()
