"""d test with independent seeds: P(larger first) for d = 2 vs d = 4 at identical bond dimensions.

Usage: uv run python experiments/order/analyze_dtest.py   -> results/d_test.csv
Reads RAW/d_test_<tree>_d2 and _d4 (configs/d_test_*.yaml: seeds 200000+ for d = 2, 300000+ for
d = 4, so no random draws are shared); two-sample z test per (tree, rho).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from analyze_main import load  # noqa: E402
from common import RAW, RES  # noqa: E402


def main():
    rows = []
    for tree in ("binary8", "mps4"):
        parts = {}
        for d in (2, 4):
            name = f"d_test_{tree}_d{d}"
            if (RAW / name / "runs.jsonl").exists():
                _, runs = load(name)
                parts[d] = runs
        if len(parts) < 2:
            continue
        for rho in sorted(parts[2].rho.unique()):
            a = parts[2][parts[2].rho == rho].winner == 0
            b = parts[4][parts[4].rho == rho].winner == 0
            pa, pb = a.mean(), b.mean()
            pp = (a.sum() + b.sum()) / (len(a) + len(b))
            z = (pb - pa) / np.sqrt(pp * (1 - pp) * (1 / len(a) + 1 / len(b)))
            rows.append(dict(tree=tree, rho=rho, bonds=parts[2].bonds.iloc[0], n_d2=len(a),
                             n_d4=len(b), p_d2=pa, p_d4=pb, diff=pb - pa, z=z,
                             p_value=2 * (1 - norm.cdf(abs(z)))))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "d_test.csv", index=False, float_format="%.4g")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
