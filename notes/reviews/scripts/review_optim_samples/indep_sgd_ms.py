"""Reviewer's independent computation of the online-SGD mean-square stability limit at the balanced
minimum of binary N=4, d=3, chi=3, teacher [3,2,1] (own contraction, own Jacobian, own 4th-moment
operator, symmetric-matrix basis; no code from experiments/optim_samples).

Linearization at an interpolating minimum: e <- (I - eta Theta P_B) e, P_B = (1/B) sum_b Phi_b Phi_b^T,
E[P_B M P_B] = (1 - 1/B) M + F(M)/B, F(M) = E[Phi Phi^T M Phi Phi^T].
Also: Monte Carlo check of F for product Gaussian features.
Usage: python indep_sgd_ms.py OUTDIR
"""

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])
from indep_binary4 import balanced_min, dense, jacobian, random_basis, LAM_BAL  # noqa: E402

N, D = 4, 3


def f_op(m):
    """F(M) for M an 81x81 matrix: per site pair (i_k, j_k): A -> A + A^T + tr(A) I."""
    t = m.reshape((D,) * (2 * N))
    for k in range(N):
        sw = np.swapaxes(t, k, N + k)
        tr = np.trace(t, axis1=k, axis2=N + k)  # removes axes k and N+k
        eye_part = np.multiply.outer(tr, np.eye(D))  # axes: rest..., k, N+k at the end
        eye_part = np.moveaxis(eye_part, [-2, -1], [k, N + k])
        t = t + sw + eye_part
    return t.reshape(m.shape)


def mc_check(rng, n_samp=200000):
    m = rng.standard_normal((D**N, D**N))
    acc = np.zeros_like(m)
    for _ in range(n_samp // 1000):
        x = rng.standard_normal((1000, N, D))
        phi = x[:, 0]
        for i in range(1, N):
            phi = (phi[:, :, None] * x[:, i][:, None, :]).reshape(1000, -1)
        q = np.einsum("bi,ij,bj->b", phi, m, phi)
        acc += np.einsum("b,bi,bj->ij", q, phi, phi)
    acc /= n_samp
    ex = f_op(m)
    return float(np.linalg.norm(acc - ex) / np.linalg.norm(ex))


def main(out):
    rng = np.random.default_rng(7)
    u = random_basis(rng)
    th = balanced_min(u)
    jac = jacobian(th)
    theta = jac @ jac.T
    ev, vec = np.linalg.eigh(theta)
    keep = ev > 1e-10 * ev[-1]
    lam, uu = ev[keep], vec[:, keep]
    m = len(lam)
    print("rank Theta", m, "lam_max", lam[-1], "formula", LAM_BAL, flush=True)
    # symmetric basis
    idx = [(i, j) for i in range(m) for j in range(i, m)]
    nb = len(idx)

    def to_vec(x):
        return np.array([x[i, j] for i, j in idx])

    def from_vec(v):
        x = np.zeros((m, m))
        for (i, j), val in zip(idx, v):
            x[i, j] = val
            x[j, i] = val
        return x

    lm = np.diag(lam)
    a1 = np.zeros((nb, nb))
    a2 = np.zeros((nb, nb))
    a3 = np.zeros((nb, nb))
    for c, (i, j) in enumerate(idx):
        x = np.zeros((m, m))
        x[i, j] = 1.0
        x[j, i] = 1.0
        if i == j:
            x[i, i] = 1.0
        a1[:, c] = to_vec(lm @ x + x @ lm)
        a2[:, c] = to_vec(lm @ x @ lm)
        a3[:, c] = to_vec(lm @ (uu.T @ f_op(uu @ x @ uu.T) @ uu) @ lm)
    # Note: columns use x = E_ij + E_ji (i != j); coordinates read x[i, j]: consistent basis.
    rows = []
    for b in (1, 2, 4, 8, 16, 64, 256, 10**9):

        def rho(eta):
            t = np.eye(nb) - eta * a1 + eta**2 * ((1 - 1 / b) * a2 + a3 / b)
            return np.max(np.abs(np.linalg.eigvals(t)))

        lo, hi = 0.0, 2.0 / lam[-1] * 1.0001
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if rho(mid) < 1 else (lo, mid)
        rows.append(dict(B=b, eta_ms=lo, eta_ms_over_B=lo / b))
        print(rows[-1], flush=True)
    err = mc_check(np.random.default_rng(3))
    print("MC check of F, rel err", err)
    json.dump(
        dict(rank=m, lam_max=lam[-1], rows=rows, mc_rel_err=err),
        open(f"{out}/indep_sgd_ms.json", "w"),
    )


if __name__ == "__main__":
    main(sys.argv[1])
