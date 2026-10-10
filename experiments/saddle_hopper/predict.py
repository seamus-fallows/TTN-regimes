"""Run the hopper on many cases (parallel, cached) and turn predicted sequences into observable orders.

A case is a JSON-able dict: {"teacher": {...}, "tree": kind, "n": N, "d": d, "chi": chi, "init": ...}.
Teacher kinds: pattern (patterns, weights), odeco (weights), nonorth (c, weights), misaligned (theta,
weights, alpha), r2_random (tree kind, teacher_seed, decay), ttn (teacher tree kind, ranks, seed, decay).
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def build(case):
    import cases as C

    t = case["teacher"]
    kind = t["kind"]
    metrics = None
    if kind == "pattern":
        W, comps = C.pattern(t["patterns"], t["weights"], case["d"])
    elif kind == "odeco":
        W, comps = C.odeco(case["n"], case["d"], t["weights"])
    elif kind == "nonorth":
        W, comps = C.nonorth(case["n"], case["d"], t["c"], t["weights"])
    elif kind == "misaligned":
        W, comps, metrics = C.misaligned(
            case["n"], case["d"], t["theta"], t["weights"], t.get("alpha", 1.0)
        )
    elif kind == "r2_random":
        W, comps = (
            C.r2_random_teacher(
                t["tree"], case["n"], t["teacher_seed"], t.get("decay", 0.6), case["d"]
            ),
            None,
        )
    elif kind == "ttn":
        from ttn.tree import make_tree

        tt = make_tree(t["tree"], case["n"])
        ranks = dict(zip(tt.edges, t["ranks"]))
        W, comps = C.ttn_teacher(tt, case["d"], ranks, t["teacher_seed"], t.get("decay", 1.0)), None
    elif kind == "cp_sites":  # theory_teachers 4.3: u_i = e0, v_i = c_i e0 + sqrt(1 - c_i^2) e1
        import math

        import torch

        e = torch.eye(case["d"], dtype=torch.float64)
        v = [t["overlaps"][i] * e[0] + math.sqrt(1 - t["overlaps"][i] ** 2) * e[1] for i in range(case["n"])]
        W, comps = C.cp_dense(t["weights"], [[e[0]] * case["n"], v])
    elif kind == "aligned_cov":
        W, comps = C.odeco(case["n"], case["d"], t["weights"])
        import torch

        metrics = [torch.diag(torch.tensor(lam, dtype=torch.float64)) for lam in t["site_eigs"]]
    else:
        raise ValueError(kind)
    tree = C.get_tree(case["tree"], case["n"])
    return C.setting(
        tree,
        case["d"],
        case["chi"],
        W,
        comps,
        metrics,
        case.get("init", "fanin"),
        name=case.get("name", ""),
        start_component=case.get("start_component"),
    )


def serialize(out):
    cands = []
    for c in out["candidates"]:
        cands.append(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items() if k != "ys"}
        )
    return dict(candidates=cands, sequences=out["sequences"])


def run_case(case):
    import torch

    torch.set_num_threads(1)
    from hopper import SaddleHopper

    t0 = time.time()
    h = SaddleHopper(build(case), seed=case.get("seed", 0))
    out = serialize(
        h.run(max_events=case.get("max_events", 10), max_sequences=case.get("max_sequences", 64))
    )
    out["wall"] = time.time() - t0
    out["K"], out["L0"] = h.K, h.L0
    return out


def key_of(case) -> str:
    return json.dumps(case, sort_keys=True)


def run_many(cases, cache_path, workers=8):
    """Predictions for every case, cached in a JSON file keyed by the case."""
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [c for c in {key_of(c): c for c in cases}.values() if key_of(c) not in cache]
    if todo:
        print(f"hopper: {len(todo)} new cases on {workers} workers", flush=True)
        with get_context("spawn").Pool(workers) as pool:
            for c, out in zip(todo, pool.imap(run_case, todo)):
                cache[key_of(c)] = out
                cache_path.write_text(json.dumps(cache))
    return {key_of(c): cache[key_of(c)] for c in cases}


# ---------------------------------------------------------------- observable orders


def learned_history(seq):
    """Per saddle (after each event): the learned set; plus per event the components that join."""
    return [set(ev["learned"]) for ev in seq["events"]]


def orders(seq, which="final"):
    """All orders of learning compatible with a predicted sequence.

    which = 'final': order of the last time each component joins the learned set (locality's final
    1/2 crossing); 'first': order of the first time (s2s_core's first full rise, order's first event).
    Components joining in the same event may come in any order. Returns (list of tuples, rotation flag)
    where rotation = some component leaves the learned set (or joins twice)."""
    hist = learned_history(seq)
    joins = {}  # component -> list of event indices where it joins
    prev = set()
    rotation = False
    for j, cur in enumerate(hist):
        for r in cur - prev:
            joins.setdefault(r, []).append(j)
        if prev - cur:
            rotation = True
        prev = cur
    if any(len(v) > 1 for v in joins.values()):
        rotation = True
    pick = (lambda v: v[-1]) if which == "final" else (lambda v: v[0])
    final = set(hist[-1]) if hist else set()
    comps = sorted(r for r in joins if which == "first" or r in final)
    by_event = {}
    for r in comps:
        by_event.setdefault(pick(joins[r]), []).append(r)
    groups = [sorted(by_event[j]) for j in sorted(by_event)]
    out = [()]
    for g in groups:
        out = [o + p for o in out for p in itertools.permutations(g)]
    return out, rotation


def allowed(pred, which="final"):
    """{order: set of (pclass, rotation)} over all predicted sequences."""
    res = {}
    for s in pred["sequences"]:
        if not s["end"].startswith("converged"):
            continue
        ords, rot = orders(s, which)
        for o in ords:
            res.setdefault(o, set()).add((s["pclass"], rot))
    return res


def workers_default():
    return max(1, min(8, (os.cpu_count() or 2) - 1))
