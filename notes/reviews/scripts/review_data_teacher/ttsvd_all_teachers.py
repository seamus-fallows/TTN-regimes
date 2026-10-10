"""Reviewer: independent check that the rank-(1,1,1) plateau of a random-TTN-teacher run (e4b, teacher 103)
sits at a critical point of the bond-rank-constrained loss.

constrained.py finds the constrained minima from random starts (L-BFGS). Here instead:
 (a) rank-1 critical points via the higher-order power method (HOPM) on W*, started from the plateau
     state's own mode vectors (a rank-1 critical point of 1/2||X - W*||^2 is g (x) w_i with w_i a
     critical point of g = <W*, (x) w_i> on unit spheres; plateau loss = (||W*||^2 - g^2)/2);
 (b) for every detected plateau (any ranks): L-BFGS on the rank-constrained TTN started from the
     TT-SVD truncation of the plateau state itself (nearest critical point by descent), compared with
     the constrained.py minimum recorded by analyze_ttn.py;
 (c) a non-HSVD alternative truncation (TT-SVD of W*) for reference.
"""

import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

sys.path.insert(0, sys.argv[1])  # the copy of experiments/data_teacher (setups.build = the teacher)
from setups import build  # noqa: E402

from ttn.dynamics import Trajectory  # noqa: E402
from ttn.network import TTNSpec, to_dense  # noqa: E402

torch.set_num_threads(1)
copy = sys.argv[1]
cfg = dict(
    kind="ttn_teacher",
    n_sites=4,
    d=3,
    chi=4,
    teacher_ranks=[2, 3, 2],
    decay=0.5,
    teacher_seed=103,
    tree="mps",
)
spec, teacher, *_ = build(cfg, 0)
Ws = teacher.dense()
nrm = float(Ws.norm())


def hopm(T, w, iters=20000, tol=1e-15):
    w = [x / x.norm() for x in w]
    for _ in range(iters):
        old = [x.clone() for x in w]
        for i in range(4):
            M = T
            for k in sorted([k for k in range(4) if k != i], reverse=True):
                M = torch.tensordot(M, w[k], dims=([k], [0]))
            w[i] = M / M.norm()
        if max(float((a - b).norm()) for a, b in zip(w, old)) < tol:
            break
    g = float(torch.einsum("abcd,a,b,c,d->", T, *w))
    return g, w


def tt_svd(W, ranks):
    """Left-to-right TT-SVD truncation of a 4-site tensor to bond ranks (r01, r12, r23); returns dense."""
    d = W.shape
    cores, r_prev, M = [], 1, W.reshape(1, -1)
    for i in range(3):
        M = M.reshape(r_prev * d[i], -1)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        r = ranks[i]
        cores.append(U[:, :r].reshape(r_prev, d[i], r))
        M = S[:r, None] * Vh[:r]
        r_prev = r
    cores.append(M.reshape(r_prev, d[3], 1))
    out = cores[0]
    for c in cores[1:]:
        out = torch.tensordot(out, c, dims=([-1], [0]))
    return out.reshape(d), cores


def descend_from(W_start, ranks):
    """L-BFGS on the rank-constrained loss from the TT-SVD truncation of W_start (MPS cores)."""
    _, cores = tt_svd(W_start, ranks)
    shapes = [c.shape for c in cores]
    sizes = [c.numel() for c in cores]
    y0 = torch.cat([c.reshape(-1) for c in cores]).numpy()

    def dense(y):
        parts, k = [], 0
        for sh, n in zip(shapes, sizes):
            parts.append(y[k : k + n].reshape(sh))
            k += n
        o = parts[0]
        for c in parts[1:]:
            o = torch.tensordot(o, c, dims=([-1], [0]))
        return o.reshape(Ws.shape)

    def fg(y):
        yt = torch.from_numpy(y).requires_grad_(True)
        loss = 0.5 * ((dense(yt) - Ws) ** 2).sum()
        (g,) = torch.autograd.grad(loss, yt)
        return loss.item(), g.numpy().copy()

    res = minimize(
        fg,
        y0,
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=20000, gtol=1e-12, ftol=1e-16, maxcor=50),
    )
    return res.fun, dense(torch.from_numpy(res.x)).detach(), float(np.linalg.norm(res.jac))




import pandas as pd
comm = pd.read_csv("C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher/results/e4b_ttn_plateaus.csv")
for ts in (101, 102, 103, 104, 105):
    cfg_t = dict(cfg, teacher_seed=ts)
    _, teach, *_ = build(cfg_t, 0)
    T = teach.dense()
    n = float(T.norm())
    Wl, _ = tt_svd(T, [1, 1, 1])
    Wr, _ = tt_svd(T.permute(3, 2, 1, 0).contiguous(), [1, 1, 1])
    Wr = Wr.permute(3, 2, 1, 0)
    globals()["Ws"] = T
    lc, Wc, gn = descend_from(Wl, [1, 1, 1])
    r1 = comm[(comm.tag == f"teacher_seed{ts}") & (comm.ranks == "[1, 1, 1]")]
    print(ts, f"crit loss {lc:.6f} (committed {r1.loss_crit.iloc[0] if len(r1) else float('nan'):.6f}); "
          f"|TTSVD_lr - crit| {float((Wl - Wc).norm()) / n:.4f}, |TTSVD_rl - crit| {float((Wr - Wc).norm()) / n:.4f}, "
          f"|HSVD - crit| (committed) {r1.dist_crit_hsvd.iloc[0] if len(r1) else float('nan'):.4f}; "
          f"plateau-crit distances (committed) {list(r1.dist_crit.round(4))}")
