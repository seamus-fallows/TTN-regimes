"""Stall bookkeeping tables (review round 1, issue 4; verification round 2, items (d), (g)).

Writes, into the output directory:
  run_classes_by_sweep.csv   per sweep: runs per class (greedy / other order / rotation / stalled / timeout /
                             incomplete / error), restricted to the runs of the committed configs;
  stall_dropout_<sweep>.csv  per case with stalls: seeds that stall at every sigma vs at some sigmas only;
  stall_dropout_bias.csv     per case with stalls: exponent fitted on all escaping seeds vs on seeds that never
                             stall (sigma-dependent dropout bias);
  stall_capped_ends.csv      chi = 2 vs chi = 4 stalls on the same cases, and whether both route ends sit on bonds
                             capped at d.

usage: uv run python experiments/locality/stall_tables.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from analysis import classify, load_records, mean_se, plateau_table, slope_se, stall_dropout  # noqa: E402
from rules import get_tree  # noqa: E402

O = os.environ.get("LOCALITY_OUT", "/root/outputs/locality")
C = str(Path(__file__).parent / "configs")
SWEEPS = [("twocomp", "twocomp"), ("twocomp_balanced", "twocomp_balanced"), ("twocomp_chi4", "twocomp_chi4"),
          ("robust", "robust"), ("k5ext", "k5_extension"), ("khigh", "khigh_extension"), ("lambda2K", "lambda2_K"),
          ("multicomp", "multicomp"), ("multicomp_wide", "multicomp_wide")]


def both_ends_capped(kind, n, S):
    """Do all ends of the Steiner route sit on bonds capped at d (a bond whose smaller side is one site)?"""
    tree = get_tree(kind, n)
    nodes = tree.steiner_nodes(list(S))
    caps = []
    for u in nodes:
        nb = [x for x in list(tree.children[u]) + [tree.parent[u]] if x in nodes]
        if len(nb) != 1:
            continue
        e = u if tree.parent[u] == nb[0] else nb[0]
        below = len(tree.subtree_sites[e])
        caps.append(min(below, tree.n_sites - below) == 1)
    return bool(caps) and all(caps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pref, cfg in SWEEPS:
        recs = load_records([f"{O}/{pref}"], config=[f"{C}/{cfg}.yaml"])
        c = collections.Counter(classify(r) for r in recs)
        rows.append(dict(sweep=pref, runs=len(recs), **{k: c.get(k, 0) for k in
                         ("greedy", "other_order", "rotation", "stalled", "timeout", "incomplete", "error")}))
    pd.DataFrame(rows).to_csv(out / "run_classes_by_sweep.csv", index=False)

    for pref, cfg in SWEEPS[:3]:
        stall_dropout([f"{O}/{pref}"], config=[f"{C}/{cfg}.yaml"]).to_csv(out / f"stall_dropout_{pref}.csv", index=False)

    bias = []
    for pref, cfg in SWEEPS[:3]:
        recs = load_records([f"{O}/{pref}"], config=[f"{C}/{cfg}.yaml"])
        stalled = collections.defaultdict(set)
        for r in recs:
            if r.get("S") is None or r.get("status") == "error":
                continue
            key = (r["kind"], r.get("n"), "".join(map(str, r["S"])), r["start"], r["chi"])
            stalled[key]
            if r.get("status") == "stalled":
                stalled[key].add(r["seed"])
        df = plateau_table([f"{O}/{pref}"], config=[f"{C}/{cfg}.yaml"])
        df = df[(df["order"] == "0>1") & (~df["rotation"]) & (df["n_comp"] == 2)]
        for (kind, n, S, start, chi), bad in stalled.items():
            if not bad:
                continue
            tname = get_tree(kind, n).name
            g = df[(df["tree"] == tname) & (df["case"] == "S=" + S) & (df["start"] == start) & (df["chi"] == chi)]
            pa, pc = [], []
            for seed, gs in g.groupby("seed"):
                if gs["sigma"].nunique() < 3:
                    continue
                p = slope_se(np.log(1 / gs["sigma"].values), np.log(gs["gap"].values))[0]
                pa.append(p)
                if seed not in bad:
                    pc.append(p)
            m_all, m_clean = mean_se(pa)[0], mean_se(pc)
            bias.append(dict(sweep=pref, tree=tname, case="S=" + S, start=start, chi=chi, seeds_fit=len(pa), p_all=m_all,
                             seeds_never_stalled=len(pc), p_never_stalled=m_clean[0], p_never_stalled_se=m_clean[1],
                             diff=m_all - m_clean[0] if pc and pa else np.nan))
    pd.DataFrame(bias).to_csv(out / "stall_dropout_bias.csv", index=False, float_format="%.4g")

    c4 = yaml.safe_load(open(f"{C}/twocomp_chi4.yaml"))["cases"]
    want = {(c["kind"], c.get("n"), tuple(c["S"])) for c in c4}

    def tally(pref, cfg):
        t = collections.defaultdict(lambda: [0, 0])
        for r in load_records([f"{O}/{pref}"], config=[f"{C}/{cfg}.yaml"]):
            key = (r["kind"], r.get("n"), tuple(r["S"]))
            if r.get("start") == "small" and key in want:
                t[key][0] += r.get("status") == "stalled"
                t[key][1] += 1
        return t

    a, b = tally("twocomp", "twocomp"), tally("twocomp_chi4", "twocomp_chi4")
    cap = [dict(tree=get_tree(k[0], k[1]).name, case="S=" + "".join(map(str, k[2])), both_ends_capped=both_ends_capped(*k),
                stalled_chi2=a[k][0], runs_chi2=a[k][1], stalled_chi4=b[k][0], runs_chi4=b[k][1]) for k in sorted(a)]
    pd.DataFrame(cap).to_csv(out / "stall_capped_ends.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
