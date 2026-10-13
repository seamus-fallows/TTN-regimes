"""Route-theory predictions for every target and every learned set J (no training):
results/targets.csv (one row per target) and results/predictions.csv (one row per (target, J,
missing pattern)): route depth c_J(r), pairwise cost, A_r(J) at the balanced theta_J, the three
rules' next pattern, and the one-component order of Theorem 3.2 for |J| = 1.

usage: uv run python experiments/route_smallinit/targets.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import SpareManifold, build_target  # noqa: E402
from run import gamma, predictions  # noqa: E402


def main():
    targets = yaml.safe_load(open(HERE / "configs/targets.yaml"))["targets"]
    trows, prows = [], []
    for cfg in targets:
        t = build_target(cfg)
        man = SpareManifold(t, int(cfg["d"]))
        pstr = lambda a: "".join(map(str, t.patterns[a]))  # noqa: E731
        seqs = {}
        for first in range(t.n):
            J, seq = [first], [pstr(first)]
            while len(J) < t.n:
                J.append(predictions(cfg, t, J)["greedy"])
                seq.append(pstr(J[-1]))
            seqs[first] = " ".join(seq)
        worst = max(t.min_depth(J) for k in range(1, t.n) for J in itertools.combinations(range(t.n), k))
        trows.append(dict(
            target=cfg["name"], kind=cfg["kind"], tree=f"{cfg['tree']}{cfg['n_sites']}", d=cfg["d"],
            K=t.tree.n_nodes, patterns=" ".join(cfg["patterns"]), weights=" ".join(map(str, cfg["weights"])),
            bonds=" ".join(map(str, man.spec.bond_dims[:-1])), n_params=man.spec.n_params, star=t.is_star,
            max_min_depth=worst, greedy_seq_from_heaviest=seqs[max(range(t.n), key=lambda a: t.weights[a])],
        ))
        for k in range(1, t.n):
            for J in itertools.combinations(range(t.n), k):
                pr = predictions(cfg, t, list(J))
                oc = t.one_component_order(J[0]) if k == 1 else None
                for r in t.missing(J):
                    prows.append(dict(
                        target=cfg["name"], tree=f"{cfg['tree']}{cfg['n_sites']}", d=cfg["d"],
                        J="{" + ",".join(map(str, J)) + "}", J_str=" ".join(pstr(a) for a in J),
                        missing=r, missing_str=pstr(r), weight=t.weights[r], depth=pr["depth"][r],
                        pairwise=pr["pair"][r], region=" ".join(map(str, sorted(t.region(J, r)))),
                        gamma=gamma(cfg, list(J), r), coef=pr["coef"][r],
                        pred_greedy=pr["greedy"], pred_pairwise=pr["pairwise"], pred_weight=pr["weight"],
                        pred_lib_greedy=pr["lib_greedy"], tie=pr["tie"], min_depth=min(pr["depth"].values()),
                        one_component_order=(oc["order"] if oc else None),
                        one_component_mechanism=(oc["mechanism"] if oc else None),
                    ))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(trows).to_csv(out / "targets.csv", index=False)
    pd.DataFrame(prows).to_csv(out / "predictions.csv", index=False)
    print(pd.DataFrame(trows).to_string())


if __name__ == "__main__":
    main()
