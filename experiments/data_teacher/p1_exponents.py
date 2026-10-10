"""Predicted P1 exponent for the step-4(b) teachers (fix round 2: the review noted that only teacher 104
was compared, against a value computed for another teacher).

theory_teachers.md section 9, P1 (heuristic): at the rank-one plateau the state approaches the rank-one
critical point as sigma^alpha, alpha = min(alpha_r, alpha_a), with
  alpha_r = (1 - kappa) mu_r / (mu_r + gamma),  mu_r = K lambda^(2(K-1)/K),  gamma = lambda^((K-2)/K) s2*,
  kappa = s2* / lambda,  s2* = s2(M_e*) = max_e s2(M_e)  (MPS-N student, K = N, whitened),
  alpha_a = min |h_a| / (|h_a| + s2*) over the eigenvalues h_a of the Riemannian Hessian of the rank-one
  problem (g = T(w_1, ..., w_N) on the product of unit spheres) with |h_a| < lambda - s2*.
Check: sym(0.5) (MPS4, d = 2, s = (1, 0.6), overlap 0.5 on every site) gives alpha_r = 0.834,
h = -0.618, alpha_a = 0.811 as in theory_teachers.md 7.6.
Measured: slope of log(distance of the plateau state to the critical point) vs log(sigma), per seed over
the resolved rank-one plateaus of results/e4b_ttn_plateaus.csv (analyze_ttn.py), mean +- s.e.m.
Usage: uv run python experiments/data_teacher/p1_exponents.py
Writes results/e4b_p1_exponents.csv.
"""

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from constrained import constrained_minima, metric_loss_of_dense, ranked_spec
from setups import build

from ttn.teachers import CPTeacher


def site_vectors(W):
    n = W.ndim
    return [torch.linalg.svd(torch.movedim(W, i, 0).reshape(W.shape[i], -1))[0][:, 0] for i in range(n)]


def contract_except(T, w, keep):
    M = T
    for k in sorted([k for k in range(T.ndim) if k not in keep], reverse=True):
        M = torch.tensordot(M, w[k], dims=([k], [0]))
    return M


def alpha_pred(T, W):
    n = T.ndim
    w = site_vectors(W)
    lam = float(contract_except(T, w, ()))
    if lam < 0:
        w[0], lam = -w[0], -lam
    s2 = [float(torch.linalg.svdvals(contract_except(T, w, (i, i + 1)))[1]) for i in range(n - 1)]
    e = int(np.argmax(s2))
    s2s, k = s2[e], n
    kappa, gamma, mu_r = s2s / lam, lam ** ((k - 2) / k) * s2s, k * lam ** (2 * (k - 1) / k)
    alpha_r = (1 - kappa) * mu_r / (mu_r + gamma)
    # Riemannian Hessian on the product of spheres in orthonormal tangent bases
    bases = []
    for wi in w:
        q, _ = torch.linalg.qr(torch.cat([wi[:, None], torch.eye(len(wi), dtype=wi.dtype)], 1))
        bases.append(q[:, 1:len(wi)])
    blocks = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                blocks[i][j] = -lam * torch.eye(bases[i].shape[1], dtype=T.dtype)
            else:
                M = contract_except(T, w, (min(i, j), max(i, j)))
                M = M if i < j else M.T
                blocks[i][j] = bases[i].T @ M @ bases[j]
    h = torch.linalg.eigvalsh(torch.cat([torch.cat(r, 1) for r in blocks], 0)).numpy()
    slow = [abs(x) for x in h if abs(x) < lam - s2s]
    alpha_a = min(abs(x) / (abs(x) + s2s) for x in slow) if slow else np.inf
    return dict(lam=lam, edge_star=e, s2_star=s2s, kappa=kappa, alpha_r=alpha_r,
                h_slowest=-min(slow) if slow else np.nan, alpha_a=alpha_a, alpha=min(alpha_r, alpha_a))


def sym_check(c=0.5):
    d, n = 2, 4
    u = torch.tensor([1.0, 0.0], dtype=torch.float64)
    v = torch.tensor([c, (1 - c * c) ** 0.5], dtype=torch.float64)
    t = CPTeacher(torch.tensor([1.0, 0.6], dtype=torch.float64),
                  torch.stack([u.expand(n, d), v.expand(n, d)]).clone()).dense()
    spec, _ = build(dict(kind="nonorth", n_sites=n, d=d, chi=2, overlap=c, s2=0.6, tree="mps"), 0)[:2]
    spec1 = ranked_spec(spec, {e: 1 for e in spec.tree.edges})
    m = constrained_minima(spec1, metric_loss_of_dense(t, None), n_starts=12)[0]
    return alpha_pred(t, m["W"])


def main():
    torch.set_num_threads(1)
    print("sym(0.5) check:", {k: round(v, 4) for k, v in sym_check().items()})
    plat = pd.read_csv(RESULTS / "e4b_ttn_plateaus.csv")
    plat = plat[(plat.plateau == 0) & (plat.ranks == "[1, 1, 1]")]
    rows = []
    for ts in (101, 102, 103, 104, 105):
        spec, teacher = build(dict(kind="ttn_teacher", n_sites=4, d=3, chi=4, teacher_ranks=[2, 3, 2],
                                   decay=0.5, teacher_seed=ts, tree="mps"), 0)[:2]
        spec1 = ranked_spec(spec, {e: 1 for e in spec.tree.edges})
        T = teacher.dense()
        m = constrained_minima(spec1, metric_loss_of_dense(T, None), n_starts=12)[0]
        r = dict(teacher=f"teacher_seed{ts}", **alpha_pred(T, m["W"]))
        g = plat[plat.tag == r["teacher"]]
        ex = []
        for seed, gg in g.groupby("seed"):
            gg = gg[gg.dist_crit > 0].sort_values("sigma")
            if gg.sigma.nunique() >= 3:
                ex.append(np.polyfit(np.log(gg.sigma), np.log(gg.dist_crit), 1)[0])
        r.update(n_seeds=len(ex), alpha_measured=np.mean(ex) if ex else np.nan,
                 alpha_measured_sem=np.std(ex, ddof=1) / np.sqrt(len(ex)) if len(ex) > 1 else np.nan,
                 alpha_measured_min=min(ex) if ex else np.nan, alpha_measured_max=max(ex) if ex else np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e4b_p1_exponents.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
