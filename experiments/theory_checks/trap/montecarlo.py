"""Probability law of the trap under fan-in init: Monte Carlo over init draws (seeds 100000+i,
disjoint from the committed seeds), sigma -> 0 verdicts of trapcrit.predict, two-component pattern
teachers (1, 0.5), d = 2, teacher seed 1000. Saddle starts also get the closed form for one-new-
direction ends with partners carrying sites (theory_trap.md section 8):
  p_u = I_x(n_u/2, n_w/2), x = F_u / (F_u + F_w), P(dead) = P(>= 2 dying ends) (k >= 5).

usage: PYTHONPATH=. python experiments/theory_checks/trap/montecarlo.py --workers 6 [--n_small 60 --n_saddle 400]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import betainc

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from predict_locality import one  # noqa: E402
from trapcrit import neighbours, route_ends  # noqa: E402
from rules import get_tree  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402

CASES = [  # (kind, n, S, chi)
    ("mps", 4, (0, 3), 2),
    ("mps", 4, (0, 3), 4),
    ("mps", 4, (0, 1, 2, 3), 2),
    ("mps", 6, (0, 3), 2),
    ("mps", 6, (1, 4), 2),
    ("mps", 6, (0, 5), 2),
    ("mps", 6, (0, 5), 4),
    ("midroot6", None, (0, 5), 2),
    ("mps", 6, (1, 3, 5), 2),
    ("mps", 8, (1, 5), 2),
    ("mps", 8, (0, 7), 2),
    ("ht", 4, (0, 1, 2, 3), 2),
]


def closed_form(kind, n, S, chi, d=2):
    """Saddle start, sigma -> 0: (P(dead), P(pinned)) from independent end probabilities, or None if
    an end has a siteless partner or more than one new direction."""
    tree = get_tree(kind, n)
    spec = TTNSpec.uniform(tree, d, chi)
    _, ends = route_ends(tree, list(S))
    ps = []
    for u, e, w in ends:
        if not tree.sites[w] or spec.bond_dims[e] != 2:
            return None
        Fu, Fw = spec.fan_in[u], spec.fan_in[w]
        nu, nw = d ** len(tree.sites[u]) - 1, d ** len(tree.sites[w]) - 1
        ps.append(betainc(nu / 2, nw / 2, Fu / (Fu + Fw)))
    if len(ps) != 2:
        return None
    return ps[0] * ps[1], ps[0] + ps[1] - 2 * ps[0] * ps[1], ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n_small", type=int, default=60)
    ap.add_argument("--n_saddle", type=int, default=400)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    items = []
    for kind, n, S, chi in CASES:
        for start, nd in (("small", a.n_small), ("saddle", a.n_saddle)):
            for i in range(nd):
                items.append(((kind, n, S, start, 2, chi, (1.0, 0.5), 1000, 100000 + i), []))
    with Pool(a.workers) as pool:
        res = pool.map(one, items, chunksize=8)
    rows = []
    for key, p in res:
        kind, n, S, start, d, chi, w, ts, seed = key
        rows.append(
            dict(
                kind=kind,
                n=n,
                S="".join(map(str, S)),
                start=start,
                chi=chi,
                seed=seed,
                k=p.get("k"),
                verdict=p.get("verdict"),
                q=p.get("q"),
                error=p.get("error"),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "montecarlo_runs.csv.gz", index=False)
    out = []
    for (kind, n, S, start, chi), g in df.groupby(["kind", "n", "S", "start", "chi"], dropna=False):
        m = len(g)
        pdead, ppin = (g.verdict == "dead").mean(), (g.verdict == "pinned").mean()
        cf = (
            closed_form(kind, None if pd.isna(n) else int(n), tuple(int(c) for c in S), chi)
            if start == "saddle"
            else None
        )
        out.append(
            dict(
                kind=kind,
                n=n,
                S=S,
                start=start,
                chi=chi,
                k=g.k.iloc[0],
                draws=m,
                P_dead=pdead,
                se_dead=np.sqrt(pdead * (1 - pdead) / m),
                P_pinned=ppin,
                P_dead_closed=None if cf is None else cf[0],
                P_pinned_closed=None if cf is None else cf[1],
                p_ends_closed=None if cf is None else [round(x, 4) for x in cf[2]],
                wrong_order=(g.q != 0).mean(),
            )
        )
    summ = pd.DataFrame(out)
    summ.to_csv(Path(a.out) / "montecarlo_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
