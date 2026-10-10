"""Predicted order (Theorems 1-2) for every pattern case in saddle_order.CASES, next to k*."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gp_operator import core_grouped, predicted_order  # noqa: E402
from saddle_order import CASES, Instance  # noqa: E402

for name, kw in CASES.items():
    inst = Instance(**kw)
    Rc = core_grouped(inst.tree, kw["d"], inst.R.numpy())
    ks, order, lams = predicted_order(inst.tree, Rc, kw["s_q"])
    print(f"{name:20s} k*={ks}  predicted order={order}  lambda_max(S|L_l)={ {k: round(v, 4) for k, v in lams.items()} }")
