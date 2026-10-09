"""Route-entry amplitudes at the plateau midpoint (lead's request (c): pinned route ends?).

Two-component teachers, plateau after component 1 is learned. The learned direction on every bond is the
upward component-1 message m_v (the subtree below edge v contracted with component 1's site vectors). For
every core w of the Steiner route T_S, its route entry is the core contracted with component 2's site
vectors on its physical legs, with the learned directions on bonds leaving T_S, and projected onto the
orthogonal complement of the learned direction on bonds inside T_S; its Frobenius norm is the entry
amplitude eps_w. Route ends = leaves of T_S. Under a small-init flow these should be O(sigma) if the route is
alive and balanced; much smaller ends than interiors means pinned / dying ends (theory_saddles.md R12).

usage: uv run python experiments/locality/route_amps.py <prefix> [...] --cases mps6:135 mps8:15 ... --out <csv>
"""

from __future__ import annotations

import argparse
import string
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from analysis import load_records, mean_se, slope_se  # noqa: E402
from rules import get_tree  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402


def upward_messages(spec, cores, vecs):
    """m_v = contraction of the subtree below edge v with the given site vectors (vector on v's parent leg)."""
    t = spec.tree
    msg = [None] * t.n_nodes
    for v in t.postorder:
        g = cores[v]
        for s in t.sites[v]:
            g = torch.tensordot(vecs[s], g, dims=([0], [0]))
        for c in t.children[v]:
            g = torch.tensordot(msg[c], g, dims=([0], [0]))
        msg[v] = g.reshape(-1)
    return msg


def route_entries(spec, cores, teacher, S):
    t = spec.tree
    f = teacher.factors  # (R, N, d)
    u1 = [f[0, i] for i in range(t.n_sites)]
    u2 = [f[1, i] for i in range(t.n_sites)]
    m = upward_messages(spec, cores, u1)
    learned = {v: m[v] / m[v].norm() for v in t.edges}
    route = t.steiner_nodes(sorted(S))
    out = {}
    for w in route:
        g = cores[w]
        legs = []  # (axis operator): ("vec", v) contracts, ("proj", P) projects
        n_phys = len(t.sites[w])
        ops = []
        for k, s in enumerate(t.sites[w]):
            ops.append(("vec", u2[s]))
        for c in t.children[w]:
            if c in route:
                ops.append(("proj", torch.eye(len(learned[c]), dtype=g.dtype) - torch.outer(learned[c], learned[c])))
            else:
                ops.append(("vec", learned[c]))
        if t.parent[w] >= 0:
            if t.parent[w] in route:
                ops.append(("proj", torch.eye(len(learned[w]), dtype=g.dtype) - torch.outer(learned[w], learned[w])))
            else:
                ops.append(("vec", learned[w]))
        else:
            ops.append(("vec", torch.ones(1, dtype=g.dtype)))
        # apply: projections first (keep axes), then contract vectors from the last axis backwards
        for ax, (kind, op) in enumerate(ops):
            if kind == "proj":
                g = torch.movedim(torch.tensordot(g, op, dims=([ax], [1])), -1, ax)
        for ax in reversed(range(len(ops))):
            kind, op = ops[ax]
            if kind == "vec":
                g = torch.tensordot(g, op, dims=([ax], [0]))
        nbrs = [x for x in list(t.children[w]) + [t.parent[w]] if x in route]
        out[w] = dict(eps=float(g.norm()), end=len(nbrs) <= 1, s_node=any(s in S for s in t.sites[w]))
    return out


def measure(rec):
    tree = get_tree(rec["kind"], rec.get("n"))
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec["teacher_seed"])
    data = np.load(rec["traj"])
    p = rec["plateaus"][0]
    rows = []
    for where, tq in (("start", p["t_a"]), ("mid", p.get("t_mid"))):
        if tq is None:
            continue
        i = int(np.argmin(np.abs(data["t"] - tq)))
        cores = spec.unflatten(torch.from_numpy(data["y"][:, i]))
        for w, e in route_entries(spec, cores, teacher, rec["S"]).items():
            rows.append(dict(tree=tree.name, S="".join(map(str, rec["S"])), start=rec["start"], sigma=rec["sigma"],
                             k=len(tree.steiner_nodes(sorted(rec["S"]))),
                             seed=rec["seed"], where=where, node=w, gap=p["t_b"] - p["t_a"], **e))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--cases", nargs="+", required=True, help="tree:S, e.g. mps6:135")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    want = {(c.split(":")[0], c.split(":")[1]) for c in args.cases}
    recs = [r for r in load_records(args.inputs) if r.get("traj") and len(r.get("events", [])) == 2
            and r["events"][0][1] == 0 and r.get("S") is not None
            and (get_tree(r["kind"], r.get("n")).name, "".join(map(str, r["S"]))) in want]
    with Pool(args.workers) as pool:
        df = pd.DataFrame([x for xs in pool.map(measure, recs) for x in xs])
    df.to_csv(args.out, index=False)
    # per case: per-seed slopes of log eps (ends, interior) and of log gap vs log sigma, at the plateau midpoint
    rows = []
    mid = df[df["where"] == "mid"]
    for (tree, S, start), g in mid.groupby(["tree", "S", "start"]):
        se, si, sg, ends_rel = [], [], [], []
        for _, gs in g.groupby("seed"):
            per = gs.groupby("sigma").apply(lambda x: pd.Series(dict(
                end=np.exp(np.log(x[x["end"]]["eps"]).mean()),
                interior=np.exp(np.log(x[~x["end"]]["eps"]).mean()) if (~x["end"]).any() else np.nan,
                gap=x["gap"].iloc[0])), include_groups=False).reset_index()
            if len(per) < 3:
                continue
            ls = np.log(per["sigma"].values)
            se.append(slope_se(ls, np.log(per["end"].values))[0])
            if per["interior"].notna().all():
                si.append(slope_se(ls, np.log(per["interior"].values))[0])
            sg.append(-slope_se(ls, np.log(per["gap"].values))[0])
            ends_rel.append(float(np.median(per["end"] / per["sigma"])))
        k = int(g["k"].iloc[0])
        rows.append(dict(tree=tree, S=S, start=start, k=k, seeds=len(se),
                         end_vs_sigma=mean_se(se), interior_vs_sigma=mean_se(si), gap_exponent=mean_se(sg),
                         end_over_sigma_median=float(np.median(ends_rel)) if ends_rel else np.nan))
    s = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(s.to_string(index=False))
    s.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
