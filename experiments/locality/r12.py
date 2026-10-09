"""Review issue 4: dead vs slow routes, and theory_saddles.md R12's criterion at every route end.

For each run of the stall-diagnosis sweep (stalled runs and escaped controls, long stall patience, saved
trajectories), two-component teachers:
  - plateau start t_s = first time a_1 >= 0.99 (component 1 learned, escape transient over);
  - route = Steiner subtree T_S; route ends = its leaves; for an end u with route edge e (to the
    neighbouring route core w) the new bond direction nu on e = unit vector orthogonal to the learned
    direction (chi = 2) or the dominant new direction of w's route entry on e (chi > 2);
  - Q_end = nu^T Q_e nu if u is the child side of e, -nu^T Q_e nu if u is the parent side (Q_e = conserved
    G_v^T G_v - G_p,(v) G_p,(v)^T); y_route = norm of w's route entry along nu on e;
    R12: the end decays iff c = Q_end + |y_route|^2 < 0; the route dies iff every end has c < 0;
  - outcome: escaped (second event), dead (stalled, second overlap < 1e-20 at the stop) or other;
    end amplitudes at the stop / end of the run (route_amps.route_entries).

usage: uv run python experiments/locality/r12.py <prefix> --out results/r12.csv --workers 12
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from analysis import load_records  # noqa: E402
from route_amps import route_entries, upward_messages  # noqa: E402
from rules import get_tree  # noqa: E402
from ttn.network import TTNSpec, conserved_quantities  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402


def entry_along(spec, cores, teacher, S, w, e, nu, learned, route):
    """Norm of core w's route entry with its leg on edge e contracted with nu (other route legs
    projected onto the new-index complement, other legs contracted with learned directions)."""
    t = spec.tree
    g = cores[w]
    f = teacher.factors
    ops = [("vec", f[1, s]) for s in t.sites[w]]
    for c in t.children[w]:
        if c == e:
            ops.append(("vec", nu))
        elif c in route:
            ops.append(("proj", torch.eye(len(learned[c]), dtype=g.dtype) - torch.outer(learned[c], learned[c])))
        else:
            ops.append(("vec", learned[c]))
    if t.parent[w] >= 0:
        if w == e:
            ops.append(("vec", nu))
        elif t.parent[w] in route:
            ops.append(("proj", torch.eye(len(learned[w]), dtype=g.dtype) - torch.outer(learned[w], learned[w])))
        else:
            ops.append(("vec", learned[w]))
    else:
        ops.append(("vec", torch.ones(1, dtype=g.dtype)))
    for ax, (kind, op) in enumerate(ops):
        if kind == "proj":
            g = torch.movedim(torch.tensordot(g, op, dims=([ax], [1])), -1, ax)
    for ax in reversed(range(len(ops))):
        kind, op = ops[ax]
        if kind == "vec":
            g = torch.tensordot(g, op, dims=([ax], [0]))
    return float(g.norm())


def analyse(rec):
    tree = get_tree(rec["kind"], rec.get("n"))
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec["teacher_seed"])
    data = np.load(rec["traj"])
    t, Y = data["t"], data["y"]
    a1 = np.array([teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(Y[:, i]))).numpy()[0]
                   for i in range(0, len(t), max(1, len(t) // 3000))])
    tt = t[:: max(1, len(t) // 3000)]
    above = np.where(a1 >= 0.99)[0]
    if len(above) == 0:
        return None
    i_s = int(np.argmin(np.abs(t - tt[above[0]])))
    cores = spec.unflatten(torch.from_numpy(Y[:, i_s]))
    S = rec["S"]
    route = tree.steiner_nodes(sorted(S))
    m = upward_messages(spec, cores, [teacher.factors[0, i] for i in range(tree.n_sites)])
    learned = {v: m[v] / m[v].norm() for v in tree.edges}
    Q = conserved_quantities(spec, cores)
    ends = []
    for u in route:
        nbrs = [x for x in list(tree.children[u]) + [tree.parent[u]] if x in route]
        if len(nbrs) != 1:
            continue
        w = nbrs[0]
        e = u if tree.parent[u] == w else w  # edge id = its child node
        lv = learned[e]
        P = torch.eye(len(lv), dtype=lv.dtype) - torch.outer(lv, lv)
        if len(lv) == 2:
            nu = P[:, 0] if P[:, 0].norm() > P[:, 1].norm() else P[:, 1]
            nu = nu / nu.norm()
        else:
            # dominant new direction: w's route entry with its e-leg left open (projected)
            best, nu = -1.0, None
            for k in range(len(lv)):
                cand = P[:, k]
                if cand.norm() < 1e-12:
                    continue
                cand = cand / cand.norm()
                val = entry_along(spec, cores, teacher, S, w, e, cand, learned, route)
                if val > best:
                    best, nu = val, cand
        qnn = float(nu @ Q[e] @ nu)
        q_end = qnn if e == u else -qnn
        y = entry_along(spec, cores, teacher, S, w, e, nu, learned, route)
        ends.append(dict(end=u, q_end=q_end, y_route=y, c=q_end + y ** 2))
    final = spec.unflatten(torch.from_numpy(Y[:, -1]))
    ent = route_entries(spec, final, teacher, S)
    end_amp = min(v["eps"] for v in ent.values() if v["end"])
    status = rec.get("status")
    escaped = len(rec.get("events", [])) == 2
    a_stop = rec.get("a_stall", rec.get("a_final", [np.nan, np.nan]))
    outcome = "escaped" if escaped else ("dead" if status == "stalled" and abs(a_stop[1]) < 1e-20 else str(status))
    return dict(tree=tree.name, S="".join(map(str, S)), start=rec["start"], chi=rec["chi"], sigma=rec["sigma"],
                seed=rec["seed"], outcome=outcome, t_end=float(t[-1]),
                n_unfav=sum(e["c"] < 0 for e in ends), n_ends=len(ends),
                c_ends=str([round(e["c"] / rec["sigma"] ** 2, 4) for e in ends]),
                q_ends=str([round(e["q_end"] / rec["sigma"] ** 2, 4) for e in ends]),
                y_ends=str([f"{e['y_route']:.3g}" for e in ends]),
                end_amp_final=end_amp, interior_amp_final=min((v["eps"] for v in ent.values() if not v["end"]),
                                                               default=np.nan))


def safe(rec):
    try:
        return analyse(rec)
    except Exception as exc:  # noqa: BLE001
        return dict(error=repr(exc)[:200], seed=rec.get("seed"), sigma=rec.get("sigma"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    recs = [r for r in load_records(args.inputs) if r.get("traj") and r.get("S") is not None]
    with Pool(args.workers) as pool:
        rows = [x for x in pool.map(safe, recs) if x]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ok = df[df["outcome"].notna()] if "outcome" in df else df
    ok = ok.assign(pred_dead=ok["n_unfav"] == ok["n_ends"])
    pd.set_option("display.width", 220)
    print(pd.crosstab([ok["start"], ok["pred_dead"]], ok["outcome"]).to_string())
    print(pd.crosstab([ok["tree"], ok["S"], ok["start"], ok["chi"]], [ok["outcome"], ok["pred_dead"]]).to_string())


if __name__ == "__main__":
    main()
