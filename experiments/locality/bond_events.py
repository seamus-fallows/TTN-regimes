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

sys.path.insert(0, str(Path(__file__).parent))
from rules import get_tree, rank_rule  # noqa: E402
from ttn.network import TTNSpec, bond_spectra  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

RAW = Path("/root/outputs/locality")


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


def spectra_trajectory(rec, raw_dir=RAW / "pilot3"):
    tree = get_tree(rec["kind"], rec.get("n"))
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    data = np.load(npz_path(rec, raw_dir))
    t, y = data["t"], data["y"]
    spectra = {v: [] for v in tree.edges}
    for i in range(y.shape[1]):
        bs = bond_spectra(spec, spec.unflatten(torch.from_numpy(y[:, i])))
        for v in tree.edges:
            spectra[v].append(bs[v].numpy())
    return spec, t, {v: np.array(s) for v, s in spectra.items()}


def event_rank_changes(rec, raw_dir=RAW / "pilot3"):
    spec, t, spectra = spectra_trajectory(rec, raw_dir)
    teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec.get("teacher_seed", 1000))
    tsp = teacher_spectra(spec, teacher)
    events = rec["events"]
    ev_t = np.array([e[0] for e in events])
    rows = []
    learned = [events[0][1]]
    rise_at = {j: set() for j in range(len(events))}
    for v, s in spectra.items():
        nz = tsp[v][tsp[v] > 1e-9 * tsp[v][0]]
        theta = 0.1 * nz.min().item()
        r = (s > theta).sum(1)
        for i in np.where(np.diff(r) > 0)[0]:
            tj = 0.5 * (t[i] + t[i + 1])
            j = int(np.argmin(np.abs(np.log(np.maximum(ev_t, 1e-300) + 1) - np.log(tj + 1))))
            rise_at[j].add(v)
    tree = spec.tree
    for j in range(1, len(events)):
        rb = events[j][1]
        _, _, rising = rank_rule(tree, rec["patterns"], rec["weights"], sorted(learned), rb)
        rows.append(dict(event=j, component=rb, learned="+".join(map(str, sorted(learned))),
                         predicted=sorted(rising), observed=sorted(rise_at[j]),
                         match=set(rising) == rise_at[j]))
        learned.append(rb)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=str(RAW / "pilot3"))
    args = ap.parse_args()
    rows = []
    for f in args.files:
        for line in open(f):
            rec = json.loads(line)
            if rec.get("status") == "timeout" or len(rec.get("events", [])) < 2:
                continue
            if not npz_path(rec, args.raw).exists():
                continue
            tree = get_tree(rec["kind"], rec.get("n"))
            case = rec.get("name") or "S=" + "".join(map(str, rec["S"]))
            for r in event_rank_changes(rec, args.raw):
                rows.append(dict(tree=tree.name, case=case, start=rec.get("start", "small"),
                                 sigma=rec["sigma"], seed=rec["seed"], **r))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summ = df.groupby(["tree", "case", "start", "event"]).agg(
        runs=("match", "size"), match_frac=("match", "mean"),
        predicted=("predicted", lambda x: str(x.iloc[0])))
    print(summ.to_string())


if __name__ == "__main__":
    main()
