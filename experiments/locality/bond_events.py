"""Step 4: which bonds gain rank at each learning event, compared with the predicted rising edges.

For every run with a saved trajectory, computes the bond spectra s_k^e(t) on the saved time grid, the
effective bond rank r_e(t) = #{k : s_k^e > 0.1 x smallest nonzero teacher Schmidt value on e}
(notes/diagnostics.md), assigns every rank increase to the learning event nearest in time, and compares
the set of edges that gained rank at each event with the rank-rule prediction (rules.rank_rule).

usage: uv run python experiments/locality/bond_events.py <runs.jsonl> [...] --out <csv>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from rules import get_tree, rank_rule  # noqa: E402
from ttn.network import TTNSpec, bond_spectra  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

RAW = Path("/root/outputs/locality")
F_THRESH = 0.1  # effective rank: s_k^e > f x smallest nonzero teacher Schmidt value (diagnostics.md)


def npz_path(rec, raw_dir):
    if rec.get("traj"):
        return Path(rec["traj"])
    tree = get_tree(rec["kind"], rec.get("n"))
    tag = rec.get("name") or ("S" + "".join(map(str, rec["S"])))
    tag += "_saddle" if rec.get("start") == "saddle" else ""
    return Path(raw_dir) / (f"{tag}_{rec['kind']}{tree.n_sites}_chi{rec['chi']}_d{rec['d']}"
                            f"_s{rec['sigma']:.4g}_seed{rec['seed']}.npz")


def teacher_spectra(spec, teacher):
    w = teacher.dense()
    out = {}
    for v in spec.tree.edges:
        below = sorted(spec.tree.subtree_sites[v])
        rest = [i for i in range(w.ndim) if i not in below]
        m = w.permute(below + rest).reshape(math.prod(w.shape[i] for i in below), -1)
        out[v] = torch.linalg.svdvals(m)
    return out


def spectra_trajectory(rec, raw_dir=RAW / "pilot3", max_points: int = 800):
    tree = get_tree(rec["kind"], rec.get("n"))
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    data = np.load(npz_path(rec, raw_dir))
    t, y = data["t"], data["y"]
    if len(t) > max_points:  # accepted steps cluster where the dynamics is fast (events): subsample
        idx = np.unique(np.linspace(0, len(t) - 1, max_points).astype(int))
        t, y = t[idx], y[:, idx]
    spectra = {v: [] for v in tree.edges}
    for i in range(y.shape[1]):
        bs = bond_spectra(spec, spec.unflatten(torch.from_numpy(y[:, i])))
        for v in tree.edges:
            spectra[v].append(bs[v].numpy())
    return spec, t, {v: np.array(s) for v, s in spectra.items()}


def event_rank_changes(rec, raw_dir=RAW / "pilot3"):
    """Edges whose effective rank is higher on the plateau after event j than on the plateau before
    it (plateau midpoints from the run record; the end of the run after the last event)."""
    spec, t, spectra = spectra_trajectory(rec, raw_dir)
    teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec.get("teacher_seed", 1000))
    tsp = teacher_spectra(spec, teacher)
    events = rec["events"]
    mids = [p.get("t_mid") for p in rec["plateaus"]] + [t[-1]]
    ranks = {}
    for v, s in spectra.items():
        nz = tsp[v][tsp[v] > 1e-9 * tsp[v][0]]
        ranks[v] = (s > F_THRESH * nz.min().item()).sum(1)
    tree = spec.tree
    rows = []
    learned = [events[0][1]]
    for j in range(1, len(events)):
        rb = events[j][1]
        before, after = mids[j - 1], mids[j]
        if before is None or after is None:
            learned.append(rb)
            continue
        ib, ia = int(np.argmin(np.abs(t - before))), int(np.argmin(np.abs(t - after)))
        observed = sorted(v for v in tree.edges if ranks[v][ia] > ranks[v][ib])
        _, _, rising = rank_rule(tree, rec["patterns"], rec["weights"], sorted(learned), rb)
        rows.append(dict(event=j, component=rb, learned="+".join(map(str, sorted(learned))),
                         predicted=sorted(rising), observed=observed, match=set(rising) == set(observed)))
        learned.append(rb)
    return rows


def _set_f(f):
    global F_THRESH
    F_THRESH = f


def _rows(arg):
    rec, raw = arg
    tree = get_tree(rec["kind"], rec.get("n"))
    case = rec.get("name") or "S=" + "".join(map(str, rec["S"]))
    return [dict(tree=tree.name, case=case, start=rec.get("start", "small"), sigma=rec["sigma"],
                 seed=rec["seed"], **r) for r in event_rank_changes(rec, raw)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=str(RAW / "pilot3"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--f", type=float, default=0.1, help="rank threshold factor (sensitivity: 0.03-0.1)")
    ap.add_argument("--max-runs", type=int, default=0, help="subsample runs (0 = all)")
    args = ap.parse_args()
    global F_THRESH
    F_THRESH = args.f
    from multiprocessing import Pool

    from analysis import load_records

    recs = [r for r in load_records(args.files)
            if r.get("status") not in ("timeout", "error") and len(r.get("events", [])) >= 2
            and r.get("patterns") and npz_path(r, args.raw).exists()]
    if args.max_runs:
        recs = [recs[i] for i in np.random.default_rng(0).choice(len(recs), min(args.max_runs, len(recs)),
                                                                 replace=False)]
    with Pool(args.workers, initializer=_set_f, initargs=(F_THRESH,)) as pool:
        rows = [x for xs in pool.map(_rows, [(r, args.raw) for r in recs]) for x in xs]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summ = df.groupby(["tree", "case", "start", "event"]).agg(
        runs=("match", "size"), match_frac=("match", "mean"),
        predicted=("predicted", lambda x: str(x.iloc[0])))
    pd.set_option("display.max_rows", 500)
    print(summ.to_string())


if __name__ == "__main__":
    main()
