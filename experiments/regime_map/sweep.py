"""Expand a YAML sweep config into run configs and run them (library dynamics, parallel, resumable).

Config schema:
  exp: name (outputs in /root/outputs/regime_map/<exp>/)
  base: {key: value}            defaults for every run (see run_regime.py for keys)
  grid: {key: [values]}         Cartesian product
  sigmas: {geomspace: [lo, hi, n]} or [values]  (optional shortcut for grid.sigma)
  variants: [{key: value}, ...] each variant is crossed with the grid; a variant may carry its own
                                'sigmas' and 'grid' entries that override the top-level ones
  alpha (in grid or base):      fixed function-space scale prod_v sigma_v; then
                                sigma = (alpha / root_scale)^(1/K) (imbalance sweeps)
Usage: uv run python experiments/regime_map/sweep.py configs/a.yaml [configs/b.yaml ...] [--n-proc N] [--out DIR]
Runs already present in <out>/<exp>/results.jsonl are skipped (resumable); nothing is overwritten.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from run_regime import run_sweep  # noqa: E402

from ttn.tree import make_tree  # noqa: E402


def _sigmas(spec):
    if isinstance(spec, dict) and "geomspace" in spec:
        lo, hi, n = spec["geomspace"]
        return [float(f"{s:.4g}") for s in np.geomspace(lo, hi, int(n))]
    return [float(s) for s in spec]


def expand(cfg: dict) -> list[dict]:
    out = []
    for var in cfg.get("variants", [{}]):
        var = dict(var)
        grid = {**cfg.get("grid", {}), **var.pop("grid", {})}
        sig = var.pop("sigmas", cfg.get("sigmas"))
        if sig is not None:
            grid["sigma"] = _sigmas(sig)
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            run = {**cfg.get("base", {}), **var, **dict(zip(keys, values)), "exp": cfg["exp"]}
            if "alpha" in run:  # fixed function-space scale: sigma = (alpha / root_scale)^(1/K)
                K = make_tree(run["kind"], run["n"]).n_nodes
                run["sigma"] = float(f"{(run['alpha'] / run.get('root_scale', 1.0)) ** (1 / K):.6g}")
            out.append(run)
    return out


def expected_cost(c):
    """Order runs longest first: big trees, small sigma (long plateaus), large bond dimension."""
    return (-c["n"], c["sigma"], -c.get("chi", 0))


if __name__ == "__main__":
    import argparse

    import run_regime

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", help="YAML configs (a trailing integer = process count)")
    ap.add_argument("--n-proc", type=int, default=340)
    ap.add_argument("--out", default=run_regime.OUT, help="raw output root (default /root/outputs/regime_map)")
    a = ap.parse_args()
    paths = list(a.configs)
    if paths[-1].isdigit():
        a.n_proc = int(paths.pop())
    run_regime.OUT = a.out
    runs = []
    for path in paths:
        with open(path) as f:
            runs += expand(yaml.safe_load(f))
    run_regime.run_sweep(runs, n_proc=a.n_proc, order=expected_cost)
