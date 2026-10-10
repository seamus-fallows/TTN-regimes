"""General (non-pattern) residuals: R random on the sectors B with |Steiner(B)| >= kmin, rescaled so
that lambda_max(S | L_lmax) = target * s_q (lmax = largest l < k*/2). Compares the measured saddle
order with Theorems 1-2 (predicted_order).

Usage: uv run python experiments/theory_checks/saddle_order/random_R.py NAME KIND N D KMIN TARGET SEED [T_GRID]
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gp_operator import S_matrix, core_grouped, k_star, predicted_order, steiner_cores  # noqa: E402
from saddle_order import Instance, run_case, show  # noqa: E402
from ttn.tree import make_tree  # noqa: E402


def random_R(tree, d, kmin, rng):
    K = tree.n_nodes
    dims = [d ** len(tree.sites[v]) for v in range(K)]
    Rc = np.zeros(dims)
    for size in range(2, K + 1):
        for B in itertools.combinations(range(K), size):
            if len(steiner_cores(tree, B)) < kmin or any(dims[v] == 1 for v in B):
                continue
            idx = tuple(slice(1, None) if v in B else 0 for v in range(K))
            Rc[idx] = rng.standard_normal(Rc[idx].shape)
    # back to site order
    perm = [s for v in range(K) for s in tree.sites[v]]
    X = Rc.reshape([d] * tree.n_sites)
    return np.transpose(X, np.argsort(perm))


if __name__ == "__main__":
    name, kind, n, d, kmin, target, seed = sys.argv[1:8]
    n, d, kmin, target, seed = int(n), int(d), int(kmin), float(target), int(seed)
    tree = make_tree(kind, n)
    rng = np.random.default_rng(seed)
    R = random_R(tree, d, kmin, rng)
    Rc = core_grouped(tree, d, R)
    ks = k_star(tree, Rc)
    lmax = math.ceil(ks / 2) - 1
    S, _ = S_matrix(tree, Rc, lmax)
    R *= target / np.linalg.eigvalsh(S).max()
    ks, order, lams = predicted_order(tree, core_grouped(tree, d, R), 1.0)
    print(f"{name}: k*={ks} predicted order={order} lambda_max={ {k: round(v, 4) for k, v in lams.items()} }")
    spec = dict(tree_kind=kind, n_sites=n, d=d, chi=2, s_q=1.0, residuals=[], R_dense=R.tolist())
    grid = [float(x) for x in sys.argv[8].split(",")] if len(sys.argv) > 8 else None
    out = run_case(name, spec, t_grid=grid) if grid else run_case(name, spec)
    out["predicted_order"], out["lambdas"] = order, lams
    show(out)
