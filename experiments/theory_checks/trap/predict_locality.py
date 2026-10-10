"""Per-run trap predictions for the locality and pilot two-component runs, from the init alone.

For every committed run (config + seed) the init is rebuilt with the library's RNG stream
(TTNSpec.init(sigma, seed); teacher pattern_teacher(..., seed=teacher_seed)), the plateau state after
the first escape is computed (saddle start: the init; small init: sigma-free correlation flow) and
trapcrit.predict gives alive / pinned / dead. The prediction does not depend on sigma, so it is
computed once per (case, seed) and joined to the observed class of every sigma.

usage (from the worktree root):
  PYTHONPATH=. python experiments/theory_checks/trap/predict_locality.py --data <data/locality> \
      --pilot <data/pilot_h2_locality> --out experiments/theory_checks/trap/results --workers 6
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
sys.path.insert(0, str(ROOT / "experiments" / "locality"))
sys.path.insert(0, str(ROOT / "experiments" / "s2s_core"))
from analysis import classify, load_records  # noqa: E402
from rules import get_tree  # noqa: E402
from s2s import gauge_balance  # noqa: E402
from trapcrit import plateau_state, predict, reduced_route  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

SWEEPS = [
    ("twocomp", ["twocomp.yaml", "finish_twocomp.yaml"]),
    ("twocomp_balanced", ["twocomp_balanced.yaml"]),
    ("twocomp_chi4", ["twocomp_chi4.yaml", "finish_twocomp_chi4.yaml"]),
    ("robust", ["robust.yaml"]),
    ("k5ext", ["k5_extension.yaml"]),
    ("khigh", ["khigh_extension.yaml"]),
    ("stalldiag", ["stall_diagnosis.yaml"]),
]


def init_cores(spec, start, seed):
    cores = spec.init(1.0, seed=seed)
    if start in ("small_balanced", "balanced"):
        cores = gauge_balance(spec, cores)
    return cores


def one(item):
    key, sigmas = item
    kind, n, S, start, d, chi, weights, tseed, seed = key
    torch.set_num_threads(1)
    tree = get_tree(kind, n)
    N = tree.n_sites
    spec = TTNSpec.uniform(tree, d, chi)
    patterns = [[0] * N, [1 if i in S else 0 for i in range(N)]]
    teacher = pattern_teacher(list(weights), patterns, d=d, seed=tseed)
    cores0 = init_cores(spec, start, seed)
    m = 1.0 if start == "saddle" else float(spec.flatten(cores0).abs().max())
    try:
        cores, q, w = plateau_state(spec, teacher, cores0, "saddle" if start == "saddle" else "small")
        p = predict(spec, teacher, cores, w, q, S, r=1 - q, weights=weights)
        # finite-sigma reduced route model (all ends with partners carrying sites, one new
        # direction on every route edge)
        nodes = tree.steiner_nodes(sorted(S))
        ok = p["k"] >= 4 and all(x["j"] == 2 for x in p["ends"]) and all(
            spec.bond_dims[v] == 2 for v in tree.edges if v in nodes and tree.parent[v] in nodes)
        p["reduced"] = {}
        for s in sigmas:
            try:
                p["reduced"][s] = (reduced_route(spec, teacher, cores, w, q, 1 - q, S, weights[q],
                                                 weights[1 - q], unit=s * m) if ok else None)
            except Exception:  # the finite-sigma model is optional; keep the sigma -> 0 verdict
                p["reduced"][s] = "error"
    except Exception as exc:  # report, do not stop the sweep
        return key, dict(error=repr(exc))
    p.update(q=q, unit_per_sigma=m, K=spec.n_cores)
    return key, p


def flat(p, sigma, weights):
    """Per-run row: verdict as sigma -> 0 (sign test; reduced model at k = 4), reduced model at this
    sigma, comparison criteria and per-end quantities."""
    if "error" in p:
        return dict(error=p["error"])
    ends = p["ends"]
    return dict(k=p["k"], q=p["q"], verdict=p["verdict"], n_dying=p["n_dying"], verdict_k4=p["verdict_k4"],
                reduced_sigma=p["reduced"].get(sigma), verdict_q=p["verdict_q"], verdict_r12=p["verdict_r12"],
                C_ends=json.dumps([round(x["C"], 5) for x in ends]), j_ends=json.dumps([x["j"] for x in ends]),
                qnn_ends=json.dumps([round(x["q_nn"], 5) for x in ends]),
                r12_ends=json.dumps([round(x["r12"], 5) for x in ends]),
                protected=json.dumps([x["protected"] for x in ends]))


def pilot_sites(kind, n):
    if kind == "local":
        return [0, 1]
    if kind == "far":
        return [0, n - 1]
    if kind == "global":
        return list(range(n))
    return [int(s) for s in kind[len("sites"):].split("-")]


def locality_runs(data, configs):
    rows = []
    for sweep, cfgs in SWEEPS:
        for rec in load_records([str(Path(data) / sweep)], config=[str(configs / c) for c in cfgs]):
            if rec.get("S") is None or rec.get("kind") is None:
                continue
            rows.append(dict(source="locality", sweep=sweep, kind=rec["kind"], n=rec.get("n"),
                             S=tuple(rec["S"]), start=rec.get("start", "small"), d=rec.get("d", 2),
                             chi=rec.get("chi", 2), weights=tuple(rec.get("weights", [1.0, 0.5])),
                             tseed=rec.get("teacher_seed", 1000), seed=rec["seed"], sigma=rec["sigma"],
                             obs=classify(rec)))
    return rows


def pilot_runs(pilot):
    rows = []
    for f in ["runs.csv", "saddle_runs.csv", "extend_runs.csv"]:
        df = pd.read_csv(Path(pilot) / f)
        for r in df.itertuples():
            rows.append(dict(source="pilot", sweep=f[:-4], kind=r.tree, n=int(r.n),
                             S=tuple(pilot_sites(r.kind, int(r.n))),
                             start="saddle" if r.start == "saddle" else "small", d=int(r.d), chi=int(r.chi),
                             weights=tuple(float(x) for x in str(r.weights).split("-")),
                             tseed=int(r.teacher_seed), seed=int(r.seed), sigma=float(r.sigma), obs=r.status))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    runs = locality_runs(a.data, ROOT / "experiments" / "locality" / "configs") + pilot_runs(a.pilot)
    sig = {}
    for r in runs:
        k = (r["kind"], r["n"], r["S"], r["start"], r["d"], r["chi"], r["weights"], r["tseed"], r["seed"])
        sig.setdefault(k, set()).add(r["sigma"])
    keys = sorted(((k, sorted(v)) for k, v in sig.items()), key=str)
    print(f"{len(runs)} runs, {len(keys)} distinct (case, seed) predictions", flush=True)
    with Pool(a.workers) as pool:
        preds = dict(pool.imap_unordered(one, keys, chunksize=4))
    out = []
    for r in runs:
        p = preds[(r["kind"], r["n"], r["S"], r["start"], r["d"], r["chi"], r["weights"], r["tseed"], r["seed"])]
        out.append({**r, "S": "".join(map(str, r["S"])), "w2": r["weights"][1], **flat(p, r["sigma"], r["weights"])})
    df = pd.DataFrame(out).drop(columns=["weights"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "locality_pilot_predictions.csv.gz", index=False)
    print(df.groupby(["source", "verdict"]).size())


if __name__ == "__main__":
    main()
