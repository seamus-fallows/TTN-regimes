"""Reviewer's independent check of Pi_r(t; Sigma) = Pi_r^white(Lambda_r t) (notes/data_teacher.md 1.1).

Independent of the agent's scripts: builds the problem directly from the library, integrates with the
library integrator (LSODA, not the pilot DOP853), and compares
  (a) a_r^Sigma(t) with a_r^white(Lambda_r t) from a separate whitened run (I, s) from the same init,
  (b) the half-times with the quadrature formula t_r = T(s_r, a0) / Lambda_r.
Same teacher as e1_exact_aligned.py: s = (1, 0.7, 0.5) on eigenvectors (e3, e2, e1) of power_law_cov(4, 0.5).
"""

import sys

import numpy as np
import torch
from scipy.integrate import quad

from ttn.dynamics import Problem, Recorder, integrate_gf
from ttn.losses import power_law_cov
from ttn.network import TTNSpec, aligned_init
from ttn.teachers import CPTeacher
from ttn.tree import make_tree

torch.set_num_threads(1)
N, D, CHI, ALPHA, A0 = 4, 4, 4, 0.5, 0.05
W, EIG = [1.0, 0.7, 0.5], [2, 1, 0]
out = sys.argv[1]

cov = power_law_cov(D, ALPHA)
lam_site = torch.diagonal(cov)
eye = torch.eye(D, dtype=torch.float64)
factors = torch.stack([eye[k].expand(N, D) for k in EIG]).clone()
teacher = CPTeacher(torch.tensor(W, dtype=torch.float64), factors)
big_lam = np.array([float(lam_site[k] ** N) for k in EIG])


def half_time_quad(s, lam, K):
    f = lambda p: 1.0 / ((s - p) * p ** (2 - 2.0 / K))
    val, _ = quad(f, A0**K, s / 2, limit=500, epsabs=0, epsrel=1e-12)
    return val / (K * lam)


def crossing(t, a, level):
    k = np.nonzero(a >= level)[0][0]
    return t[k - 1] + (level - a[k - 1]) * (t[k] - t[k - 1]) / (a[k] - a[k - 1])


lines = [
    "tree,r,s,Lambda,t_half_sigma,t_half_quad,rel_err_quad,width_sigma,width_white_over_Lambda,"
    "rel_err_width,max_abs_dev_clock,max_drift"
]
for tree in ("binary", "mps"):
    spec = TTNSpec.uniform(make_tree(tree, N), D, CHI)
    K = spec.n_cores
    amps = torch.full((K, 3), A0, dtype=torch.float64)
    theta0 = spec.flatten(aligned_init(spec, teacher.factors, amps))
    t_max = 200.0 if K == 3 else 2000.0
    grid = np.unique(np.r_[np.linspace(0.01, t_max, 20001)])
    ps = Problem(spec, teacher, site_metrics=[cov] * N)
    tr_s = integrate_gf(
        ps,
        theta0,
        t_max=t_max,
        log_times=grid[:-1],
        rtol=1e-11,
        atol=1e-15,
        method="lsoda",
        recorder=Recorder(spectra=False, overlaps=True),
    )
    t_s, a_s = tr_s.data["t"], tr_s.data["overlaps"]
    # whitened run sampled exactly at Lambda_r * t for every r (no interpolation)
    wgrid = np.unique(np.concatenate([lam * grid for lam in big_lam]))
    pw = Problem(spec, teacher)
    tr_w = integrate_gf(
        pw,
        theta0,
        t_max=float(wgrid[-1]) * 1.0000001,
        log_times=wgrid,
        rtol=1e-11,
        atol=1e-15,
        method="lsoda",
        recorder=Recorder(spectra=False, overlaps=True),
    )
    t_w, a_w = tr_w.data["t"], tr_w.data["overlaps"]
    for r in range(3):
        idx = np.searchsorted(t_w, big_lam[r] * t_s)
        idx = np.clip(idx, 0, len(t_w) - 1)
        ok = np.isclose(t_w[idx], big_lam[r] * t_s, rtol=1e-12, atol=1e-12)
        dev = np.max(np.abs(a_s[ok, r] - a_w[idx[ok], r]))
        th = crossing(t_s, a_s[:, r], 0.5)
        thq = half_time_quad(W[r], big_lam[r], K)
        w_s = crossing(t_s, a_s[:, r], 0.9) - crossing(t_s, a_s[:, r], 0.1)
        w_w = (crossing(t_w, a_w[:, r], 0.9) - crossing(t_w, a_w[:, r], 0.1)) / big_lam[r]
        drift = max(
            np.nanmax(tr_s.data["conservation_drift"]), np.nanmax(tr_w.data["conservation_drift"])
        )
        lines.append(
            f"{tree},{r},{W[r]},{big_lam[r]:.6f},{th:.10f},{thq:.10f},{(th - thq) / thq:.2e},"
            f"{w_s:.8f},{w_w:.8f},{(w_s - w_w) / w_w:.2e},{dev:.2e},{drift:.1e}"
        )
        print(lines[-1], flush=True)
open(out, "w").write("\n".join(lines) + "\n")
