"""Post-hoc diagnostic (not part of the registered grid): are the trapped runs of D4 and B4 caused by
the missing spare channel on the edges next to a single site (bond capped at d)? Run all 40 seeds of D4
and B4 with the cap removed (bond = blocks + 2 on every edge) at sigma = 1e-3 and 1e-4 and compare the
trap rate with the registered (capped) runs. Removing the cap changes the parameter shapes, so a seed
gives a different init: the comparison is of rates, not seed by seed.

usage: uv run python experiments/route_smallinit/diag_trapped.py  -> results/diag_trapped.csv
"""

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TARGETS = ("D4", "B4")
LOG10_SIGMAS = (-3.0, -4.0)


def one(task):
    import run as R

    name, ls, seed, cap = task
    cfg = {c["name"]: c for c in yaml.safe_load(open(HERE / "configs/targets.yaml"))["targets"]}[name]
    model, traj, info = R.run_full(cfg, 10.0**ls, seed, cap=cap, wall=1800)
    run, ev, pl, _ = R.reduce_run(cfg, traj, info, 10.0**ls, model=model)
    return dict(target=name, tree=f"{cfg['tree']}{cfg['n_sites']}", d=cfg["d"], chi=max(model.spec.bond_dims),
                sigma=10.0**ls, log10_sigma=ls, seed=seed, bond_cap=cap, bonds=" ".join(map(str, model.spec.bond_dims[:-1])),
                status=run["status"], order=run["order"], n_learned=run["n_learned"])


if __name__ == "__main__":
    tasks = [(n, ls, s, False) for n in TARGETS for s in range(40) for ls in LOG10_SIGMAS]
    with Pool(len(tasks)) as pool:
        rows = pool.map(one, tasks)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results/diag_trapped.csv", index=False)
    print(df.groupby(["target", "bond_cap", "status"]).size())
