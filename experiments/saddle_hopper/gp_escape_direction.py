"""Escape direction at the saddles where theory_saddle_order predicts order 2l < k* (gp_check.py).

At a rank-one saddle (learned q, plateau L_S) the escape toward a residual component r can go
- by the k*-route: the full pattern of r grows directly, L - L_S = -s_r^2 a_r + O(a_r^2);
- by a generalized piggyback curve (Theorem 2; here two vertex-disjoint polymers y_1, y_2 on sibling
  pairs, or one polymer and an old rotation): the mixed patterns (r on some cores of the residual's
  support, q on the others) cost (y_1^2 + y_2^2)/2 and the full pattern appears only at order y_1 y_2,
  so for two polymers L - L_S = (y_1^2 + y_2^2)/2 - s_r y_1 y_2 / s_q >= -(s_r - s_q) s_r a_r.
So x = (L_S - L) / (s_r^2 a_r) at small a_r is ~1 for the route and 1 - s_q/s_r (two symmetric
polymers) or less for the generalized curves. Where full trajectories exist (locality) the mixed
patterns are read directly: mixed_ratio = max |<W, mixed pattern>| / sqrt(s_q s_r a_r) at a_r = A_HI is
~1 for symmetric polymers and ~0 for the route. learned_min_after = lowest overlap of the first-learned
component after it reached 0.9 (forgetting).

Stored trajectories only (no new runs): order h1b saddle starts (binary8 = order-6 saddle; MPS4, MPS6 =
controls), s2s_core small-init runs on ht4 / binary8 (second escape; wrong-first runs sit at an order-6
saddle, greedy-first runs are controls), locality two-component runs on tensor-like pairs (m >= 3) with
a stored trajectory (every wrong-first run, two greedy-first controls per setting; the robust sweep is
left out because its trajectory file names do not carry the teacher seed).

usage: PYTHONPATH=. python experiments/saddle_hopper/gp_escape_direction.py
Raw data: data/order/h1b_saddle_*/, data/s2s_core/{main,large}/, data/locality/ (override the root
with DATA_ROOT). Writes results/gp_escape_direction.csv.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = Path(os.environ.get("DATA_ROOT", HERE.parents[1] / "data"))
RES = HERE / "results"
A_LO, A_HI = 1e-4, 0.05  # overlap window of the escaping component (escape started, still small)


def ratio(L, a, L_S, s_r):
    """Median of (L_S - L) / (s_r^2 a) over the points with A_LO < a < A_HI before a first exceeds A_HI."""
    stop = np.argmax(a > A_HI) if (a > A_HI).any() else len(a)
    m = (a[:stop] > A_LO) & (a[:stop] < A_HI)
    if m.sum() < 1:
        return np.nan, int(m.sum())
    return float(np.median((L_S - L[:stop][m]) / (s_r**2 * a[:stop][m]))), int(m.sum())


def order_saddle_starts():
    """order h1b saddle starts: teacher (1, rho), component 2 (rho) learned; traj = t, L, a1, a2, ..."""
    rows = []
    for name, tree in (
        ("h1b_saddle_b8", "binary8"),
        ("h1b_saddle_mps6", "mps6"),
        ("h1b_saddle_mps4", "mps4"),
    ):
        for f in sorted((DATA / "order" / name).glob("*.npz")):
            rho = float(f.stem.split("_")[0][3:])
            sigma = float(f.stem.split("_")[1][1:])
            tr = np.load(f)["traj"]
            x, n = ratio(tr[:, 1], tr[:, 2], 0.5, 1.0)  # L_S = 1/2 s_r^2 with s_r = 1
            rows.append(
                dict(
                    source="order " + name,
                    tree=tree,
                    start="saddle",
                    sigma=sigma,
                    seed=int(f.stem.split("seed")[1]),
                    s_q=rho,
                    s_r=1.0,
                    x=x,
                    points=n,
                    polymer_bound=1 - rho,
                    order_drop=tree == "binary8",
                )
            )
    return rows


def s2s_core_runs():
    """s2s_core small-init runs on ht4 / binary8: the second escape, read in the fine window around it."""
    rows = []
    for sweep in ("main", "large"):
        for f in sorted((DATA / "s2s_core" / sweep).glob("*.json")):
            if not f.name.startswith(("ht4_", "binary8_")):
                continue
            meta = json.loads(f.read_text())
            order = [int(r) for r in meta.get("order", [])]
            if len(order) < 2 or meta.get("init") != "fanin":
                continue
            z = np.load(f.with_suffix(".npz"))
            q, r = order[0], order[1]
            # coarse record + fine window around the second event, between the first and second events
            t = np.concatenate([z["t"]] + ([z[f"fine_t_{r}"]] if f"fine_t_{r}" in z.files else []))
            L = np.concatenate(
                [z["loss"]] + ([z[f"fine_loss_{r}"]] if f"fine_t_{r}" in z.files else [])
            )
            a = np.concatenate(
                [z["overlaps"][:, r]]
                + ([z[f"fine_a_{r}"][:, r]] if f"fine_t_{r}" in z.files else [])
            )
            idx = np.argsort(t, kind="stable")
            t, L, a = t[idx], L[idx], a[idx]
            keep = (t > meta["events"][q]) & (t < meta["events"][r])
            s = np.array(meta["weights"], dtype=float)
            L_S = 0.5 * (np.sum(s**2) - s[q] ** 2)
            x, n = ratio(L[keep], a[keep], L_S, s[r])
            rows.append(
                dict(
                    source=f"s2s_core {sweep}",
                    tree=meta["tree"] + str(meta["N"]),
                    teacher=meta["teacher"],
                    chi=meta["chi"],
                    start="small",
                    sigma=meta["sigma"],
                    seed=meta["seed"],
                    first=q,
                    second=r,
                    s_q=s[q],
                    s_r=s[r],
                    x=x,
                    points=n,
                    polymer_bound=1 - s[q] / s[r] if s[r] > s[q] else np.nan,
                    order_drop=bool(s[r] > s[q]),
                )
            )
    return rows


LOC_SWEEPS = ("twocomp", "twocomp_balanced", "twocomp_chi4", "k5ext", "khigh")


def saddle_class(tree, D, s_learned, s_res):
    """Wrong-first saddle of a two-component pattern teacher (learned = the smaller pattern, residual =
    the larger one on the differing sites D): k*, the order of Theorems 1'-2 and whether a split at
    degree exactly k*/2 exceeds s_q (then a polymer curve has the route's order k*: a tie)."""
    import sys

    sys.path.insert(0, str(HERE))
    from gp_check import gp_levels, k_star, low_sets, s_matrix
    from gp_operator import core_grouped

    R = np.zeros((2,) * tree.n_sites)
    R[tuple(1 if i in D else 0 for i in range(tree.n_sites))] = s_res
    Rc = core_grouped(tree, 2, R)
    ks, _ = k_star(tree, Rc, 1e-12)
    lams, top = gp_levels(tree, Rc, s_learned, ks)
    tie = False
    if ks % 2 == 0 and not top:
        S, _ = s_matrix(tree, Rc, low_sets(tree, Rc, ks // 2))
        tie = bool(S.size) and float(np.linalg.eigvalsh(S).max()) > s_learned
    return ks, (2 * top[0] if top else ks), tie


def locality_runs(controls_per_group: int = 2):
    """locality two-component runs on tensor-like pairs (m >= 3 cores carry the differing sites), small
    init, with a stored trajectory: every wrong-first run and a few greedy-first controls per setting."""
    import sys

    import torch

    sys.path.insert(0, str(HERE))
    from cases import get_tree
    from ttn.network import TTNSpec, to_dense
    from ttn.teachers import pattern_teacher

    torch.set_num_threads(1)
    traj_dir = DATA / "locality" / "traj"
    groups = {}
    for pref in LOC_SWEEPS:
        for f in sorted((DATA / "locality").glob(f"{pref}.part*")):
            for line in open(f):
                r = json.loads(line)
                if r.get("S") is None or len(r.get("weights", [1.0, 0.5])) != 2:
                    continue
                if r.get("start", "small") == "saddle" or not r.get("traj"):
                    continue
                path = traj_dir / Path(r["traj"]).name
                c = r.get("crossings", {})
                if not path.exists() or not c.get("a0_0.5") or not c.get("a1_0.5"):
                    continue
                tree = get_tree(r["kind"], r.get("n"))
                if len({tree.site_node[i] for i in r["S"]}) < 3:
                    continue
                key = (pref, r["kind"], r.get("n"), tuple(r["S"]), r.get("d", 2), r.get("chi", 2),
                       r.get("start", "small"), tuple(r.get("weights", [1.0, 0.5])), r.get("teacher_seed", 1000))
                wrong = c["a1_0.5"][0] < c["a0_0.5"][0]
                groups.setdefault(key, []).append((wrong, r, path))
    rows = []
    for key, runs in groups.items():
        wrong = [x for x in runs if x[0]]
        if not wrong:
            continue
        right = sorted((x for x in runs if not x[0]), key=lambda x: x[1]["sigma"])[:controls_per_group]
        pref, kind, n, S, d, chi, start, _, _ = key
        tree = get_tree(kind, n)
        r0 = runs[0][1]
        s = np.array(r0.get("weights", [1.0, 0.5]), dtype=float)
        pats = [[0] * tree.n_sites, [1 if i in S else 0 for i in range(tree.n_sites)]]
        fac = pattern_teacher(list(s), pats, d=d, seed=r0.get("teacher_seed", 1000)).factors.numpy()
        ks, order6, tie = saddle_class(tree, set(S), s[1], s[0])
        spec = TTNSpec.uniform(tree, d, chi)

        def unit(choice):  # choice[i] = component whose site-i vector is used
            X = fac[choice[0], 0]
            for i in range(1, tree.n_sites):
                X = np.multiply.outer(X, fac[choice[i], i])
            return X.ravel()

        U = np.array([unit([0] * tree.n_sites), unit([1] * tree.n_sites)])
        W_star = s[0] * U[0] + s[1] * U[1]
        mixed = []
        for size in range(1, len(S)):
            for A in itertools.combinations(S, size):
                mixed.append(unit([0 if (i in A or i not in S) else 1 for i in range(tree.n_sites)]))
        mixed = np.array(mixed)
        for is_wrong, r, path in wrong + right:
            z = np.load(path)
            if z["y"].shape[0] != spec.n_params:  # file name shared with a run of another setting
                continue
            with torch.no_grad():
                Ws = np.array(
                    [to_dense(spec, spec.unflatten(torch.from_numpy(y))).numpy().ravel() for y in z["y"].T]
                )
            a = Ws @ U.T / s
            m = np.abs(Ws @ mixed.T).max(axis=1)
            L = 0.5 * ((Ws - W_star) ** 2).sum(axis=1)
            cross = [np.argmax(a[:, c] > 0.5) if (a[:, c] > 0.5).any() else len(a) for c in range(2)]
            q, rr = (1, 0) if is_wrong else (0, 1)
            if cross[rr] == len(a) or cross[q] == len(a):
                continue
            keep = slice(cross[q], max(cross[rr], cross[q] + 1))
            x, npts = ratio(L[keep], a[keep, rr], 0.5 * s[rr] ** 2, s[rr])
            ar = a[keep, rr]
            i = np.argmax(ar > A_HI) if (ar > A_HI).any() else len(ar)
            rows.append(
                dict(
                    source="locality traj",
                    sweep=pref,
                    tree=f"{kind}{n or ''}",
                    teacher="S=" + "".join(map(str, S)),
                    chi=chi,
                    start=start,
                    sigma=r["sigma"],
                    seed=r["seed"],
                    first=q,
                    second=rr,
                    s_q=s[q],
                    s_r=s[rr],
                    x=x,
                    points=npts,
                    polymer_bound=1 - s[q] / s[rr] if s[rr] > s[q] else np.nan,
                    k_star_wrong_saddle=ks,
                    order_wrong_saddle=order6,
                    tie_wrong_saddle=tie,
                    order_drop=bool(is_wrong and order6 < ks),
                    mixed_ratio=float(m[keep][i - 1] / np.sqrt(s[q] * s[rr] * A_HI)) if i > 0 else np.nan,
                    learned_min_after=float(a[np.argmax(a[:, q] > 0.9) :, q].min()),
                )
            )
    return rows


def main():
    df = pd.DataFrame(order_saddle_starts() + s2s_core_runs() + locality_runs())
    df["saddle"] = np.where(
        df.order_drop,
        "order < k*",
        np.where(df.get("tie_wrong_saddle", False).fillna(False).astype(bool) & (df["first"] == 1), "tie", "order k*"),
    )
    df.to_csv(RES / "gp_escape_direction.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    ok = df[df.points > 0]
    print(f"{len(df)} escapes read, {len(ok)} with a point in {A_LO} < a < {A_HI}")
    summ = ok.groupby(["source", "tree", "saddle"]).agg(
        runs=("x", "size"),
        x_median=("x", "median"),
        x_min=("x", "min"),
        x_max=("x", "max"),
        polymer_value=("polymer_bound", "max"),
        mixed_ratio_max=("mixed_ratio", "max"),
        learned_min=("learned_min_after", "min"),
    )
    print(summ.to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
