"""Learning events of one MNIST run of the real_data pipeline (definitions: notes/route_generic.md section 3).

No target exists, so a rise is read on the ratio r_k^e = s_k^e / s_1^e of the data-weighted bond spectra
(k >= 1): rise time = last upward crossing of 0.05 (the real_data convention), onset = last upward
crossing of 0.005 before it. Event 0 is the first escape (absolute time t_a, the runner's end of phase 1);
rises whose onset precedes the rate peak of the first loss drop (phase-2 clock) are merged into it.
Effective depth: `ttn.routes.effective_depth` on r_k^e with s = r_k^e at the end of the run.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_data"))
from analysis import load as rd_load, loss_drops  # noqa: E402

from events import _k_lead, bond_components, crossing, depth_on_plateau  # noqa: E402

from ttn.tree import Tree  # noqa: E402

R_RISE = 0.05
R_ONSET = 0.005
MEAS_FLOOR = 1e-6  # data-weighted spectra: eigenvalue-based, accurate to ~1e-8 s_1; 100x margin


def load(path):
    return rd_load(path)


def tree_of(meta) -> Tree:
    return Tree(tuple(meta["parent"]), tuple(tuple(s) for s in meta["sites"]), name=meta["case"]["tree"])


def mnist_run_events(run: dict):
    from common import k_region

    meta = run["meta"]
    tree = tree_of(meta)
    p2 = run["p2"]
    tau, loss, fs = p2["t"], p2["loss"], p2["fspectra"]
    edges = meta["edges"]
    drops = loss_drops(tau, loss)
    if drops:
        t0 = drops[0]["t"]
    else:
        rate = -np.gradient(np.minimum.accumulate(loss), tau)
        t0 = float(tau[int(np.argmax(rate))])
    rs = []
    for j, e in enumerate(edges):
        top = fs[:, j, 0]
        for k in range(1, fs.shape[2]):
            y = fs[:, j, k]
            if not np.isfinite(y).all():
                continue
            ratio = y / top
            i_half, t_half, n_up = crossing(tau, np.maximum(ratio, 1e-300), R_RISE)
            final = float(ratio[-1])
            if i_half is None or final < R_RISE:
                continue
            i_on, t_on, _ = crossing(tau, np.maximum(ratio, 1e-300), R_ONSET, before=i_half)
            if i_on is None:
                i_on, t_on = 0, float(tau[0])
            rs.append(dict(edge=e, k=k, i_half=i_half, t_half=t_half, i_on=i_on, t_on=t_on, n_up=n_up,
                           final=final, ratio=ratio))
    pre = [r for r in rs if r["t_on"] <= t0]
    later = sorted([r for r in rs if r["t_on"] > t0], key=lambda r: (r["t_on"], r["t_half"]))
    groups, end = [], -math.inf
    for r in later:
        if groups and r["t_on"] <= end:
            groups[-1].append(r)
            end = max(end, r["t_half"])
        else:
            groups.append([r])
            end = r["t_half"]
    events = []
    k_tree = tree.n_nodes
    pre_s = "".join(f";{r['edge']}:{r['k']}" for r in sorted(pre, key=lambda r: (r["edge"], r["k"])))
    events.append(dict(
        event=0, rises=";".join(f"{e}:0" for e in edges) + pre_s, prior="", bonds=" ".join(map(str, edges)), n_bonds=len(edges), k_region=k_tree,
        t_event=float(meta["t_a"]), duration=float(meta["t_a"]), eff_depth=math.nan, eff_depth_min=math.nan,
        eff_depth_max=math.nan, n_depth=0, eff_depth_wide=math.nan, n_components=1, k_max_component=k_tree,
        floor_min=math.nan, floor_flag=False, relearn=0, merged_into_escape=len(pre), t_escape_peak=t0,
        k_lead=k_tree))
    prev_t = t0
    prior = [f"{e}:0" for e in edges] + [f"{r['edge']}:{r['k']}" for r in pre]
    for j, g in enumerate(groups, start=1):
        t_ev = min(r["t_half"] for r in g)
        bonds = sorted({r["edge"] for r in g})
        depths, depths2, floors = [], [], []
        for r in g:
            d = depth_on_plateau(tau, r["ratio"], r["final"], prev_t, r["t_half"])
            d2 = depth_on_plateau(tau, r["ratio"], r["final"], prev_t, r["t_half"], lo=1e-3, hi=1e-1)
            r["depth"] = None if d is None else d["depth"]
            r["depth_wide"] = None if d2 is None else d2["depth"]
            if d is not None:
                depths.append(d["depth"])
            if d2 is not None:
                depths2.append(d2["depth"])
            sel = (tau > prev_t) & (tau <= r["t_on"])
            r["floor_min"] = float(np.nanmin(r["ratio"][sel])) if sel.any() else math.nan
            floors.append(r["floor_min"])
            r["event"] = j
        comps = bond_components(tree, bonds)
        events.append(dict(
            event=j,
            rises=";".join(f"{r['edge']}:{r['k']}" for r in sorted(g, key=lambda r: (r["edge"], r["k"]))),
            prior=";".join(sorted(prior)), bonds=" ".join(map(str, bonds)), n_bonds=len(bonds),
            k_region=k_region(tree, bonds), t_event=t_ev, duration=t_ev - prev_t,
            eff_depth=float(np.median(depths)) if depths else math.nan,
            eff_depth_min=float(np.min(depths)) if depths else math.nan,
            eff_depth_max=float(np.max(depths)) if depths else math.nan, n_depth=len(depths),
            eff_depth_wide=float(np.median(depths2)) if depths2 else math.nan,
            n_components=len(comps), k_max_component=max(len(c) for c in comps),
            floor_min=float(np.nanmin(floors)) if np.isfinite(floors).any() else math.nan,
            floor_flag=False, below_meas_floor=bool(any(np.isfinite(f) and f < MEAS_FLOOR for f in floors)),
            relearn=int(sum(max(r["n_up"] - 1, 0) for r in g)), merged_into_escape=0, t_escape_peak=t0,
            first_bond=min(g, key=lambda r: r["t_on"])["edge"],
            onset_spread_rel=(max(r["t_on"] for r in g) - min(r["t_on"] for r in g)) / (t_ev - prev_t)
            if t_ev > prev_t else math.nan,
            k_lead=_k_lead(tree, g, j, t_ev, prev_t)))
        prev_t = t_ev
        prior += [f"{r['edge']}:{r['k']}" for r in g]
    for r in rs:
        r.pop("ratio", None)
    return events, rs, tree
