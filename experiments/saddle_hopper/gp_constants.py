"""Leading constants m(t) / t^p at the wrong-first saddles of the locality runs that escape along mixed
patterns (gp_escape_direction.py), next to the single-route value.

m(t) = min over ||delta|| = t of L(theta_S + delta) - L_S, by the sphere minimization of
experiments/theory_checks/saddle_order/saddle_order.py (imported read only; its run_case writes into that
folder, so the loop is repeated here). Learned pattern 0...0 with s_q = 0.5, one residual pattern with
weight 1 (the locality pair (1, 0.5) seen from its wrong-first saddle), d = chi = 2.

usage: PYTHONPATH=. python experiments/saddle_hopper/gp_constants.py   (a few minutes on one core)
Writes results/gp_constants.csv.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[0] / "theory_checks" / "saddle_order"))
from saddle_order import Instance, random_starts, sphere_min  # noqa: E402

RES = HERE / "results"
T_GRID = [0.4, 0.3, 0.2, 0.14, 0.1, 0.07, 0.05]
CASES = {  # name: (tree, n, residual pattern, kind of saddle from gp_check.saddle_class)
    "ht4 S=0123": ("ht", 4, "1111", "order 6 < k* = 7"),
    "ht4 S=012": ("ht", 4, "1110", "tie: order 6 = k*"),
    "binary8 S=024": ("binary", 8, "10101000", "tie: order 6 = k*"),
    "mps6 S=012345": ("mps", 6, "111111", "tie: order 6 = k*"),
    "mps4 S=0123": ("mps", 4, "1111", "tie: order 4 = k*"),
    "mps6 S=135": ("mps", 6, "010101", "order k* = 5, no tie"),
}


def main():
    rows = []
    for name, (kind, n, pat, kind_saddle) in CASES.items():
        inst = Instance(kind, n, 2, 2, 0.5, [(tuple(int(c) for c in pat), 1.0)])
        p = 6 if name != "mps4 S=0123" and name != "mps6 S=135" else inst.k_star
        rng = np.random.default_rng(0)
        prev, ms = None, []
        for t in T_GRID:
            starts = random_starts(inst, rng, 12) + ([prev] if prev is not None else [])
            m, prev = sphere_min(inst, t, starts, t**p)
            ms.append(m)
            rows.append(
                dict(
                    saddle=name,
                    kind=kind_saddle,
                    k_star=inst.k_star,
                    p=p,
                    t=t,
                    m=m,
                    m_over_tp=m / t**p,
                    route_over_tp=inst.route_prediction(t) / t**p,
                    slope=(math.log(abs(ms[-2] / m)) / math.log(T_GRID[len(ms) - 2] / t))
                    if len(ms) > 1 and m < 0 and ms[-2] < 0
                    else np.nan,
                )
            )
            print(name, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "gp_constants.csv", index=False)


if __name__ == "__main__":
    main()
