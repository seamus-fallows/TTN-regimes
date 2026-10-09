"""Exact linear mean-square stability limit of online SGD at the balanced global minimum.

Near an interpolating minimum (realizable teacher, no label noise) the function-space error
e = W - W* evolves as e <- (I - eta Theta Phi_B Phi_B^T) e, with Theta = J J^T the tangent kernel
and Phi_B Phi_B^T = (1/B) sum_b Phi(x_b) Phi(x_b)^T. Its second moment M = E[e e^T] evolves linearly,

    M <- M - eta (Theta M + M Theta) + eta^2 [(1 - 1/B) Theta M Theta + Theta F(M) Theta / B],

with F(M) = E[Phi Phi^T M Phi Phi^T] = (x)_i F_i and F_i(A) = A + A^T + tr(A) I (Gaussian fourth
moments of whitened site features). eta_ms(B) is the largest eta with spectral radius < 1
(restricted to the range of Theta; the null space is invariant with eigenvalue 1). The 1D estimate
uses only the top direction U_1: eta < 2 / (lambda (1 + (3^N - 1) / B)).

Run: uv run python experiments/optim_samples/sgd_stability.py [tree N ...]
Writes results/sgd_stability.csv. Dense method: feasible for d^N <= ~100.
"""

import argparse

import pandas as pd
import torch

from ttn.network import TTNSpec, aligned_init, to_dense
from ttn.teachers import odeco_teacher
from ttn.tree import make_tree

torch.set_default_dtype(torch.float64)
from common import RESULTS as OUT  # noqa: E402
WEIGHTS = [3.0, 2.0, 1.0]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 10**9]


def fourth_moment(m: torch.Tensor, n: int, d: int) -> torch.Tensor:
    t = m.reshape((d,) * (2 * n))
    for k in range(n):
        tr = torch.diagonal(t, dim1=k, dim2=n + k).sum(-1)
        tr_eye = torch.movedim(tr[..., None, None] * torch.eye(d), (-2, -1), (k, n + k))
        t = t + t.transpose(k, n + k) + tr_eye
    return t.reshape(m.shape)


def operator_pieces(kind: str, n: int, d: int = 3):
    spec = TTNSpec.uniform(make_tree(kind, n), d, len(WEIGHTS))
    teacher = odeco_teacher(n, d, WEIGHTS, seed=101)
    k = spec.n_cores
    amps = teacher.weights.unsqueeze(0).expand(k, len(WEIGHTS)) ** (1.0 / k)
    flat = spec.flatten(aligned_init(spec, teacher.factors, amps.clone()))
    jac = torch.autograd.functional.jacobian(
        lambda p: to_dense(spec, spec.unflatten(p)).reshape(-1), flat
    )
    theta = jac @ jac.T
    ev, vecs = torch.linalg.eigh(theta)
    keep = ev > 1e-10 * ev[-1]
    u, lam = vecs[:, keep], ev[keep]
    m = u.shape[1]
    basis = torch.eye(m * m).reshape(m * m, m, m)
    lam_m = torch.diag(lam)
    a1 = torch.stack([(lam_m @ x + x @ lam_m).reshape(-1) for x in basis]).T
    a2 = torch.stack([(lam_m @ x @ lam_m).reshape(-1) for x in basis]).T
    a3 = torch.stack(
        [(lam_m @ (u.T @ fourth_moment(u @ x @ u.T, n, d) @ u) @ lam_m).reshape(-1) for x in basis]
    ).T
    return spec, lam[-1].item(), a1, a2, a3


def spectral_radius(eta, batch, a1, a2, a3):
    t = torch.eye(a1.shape[0]) - eta * a1 + eta**2 * ((1 - 1 / batch) * a2 + a3 / batch)
    return torch.linalg.eigvals(t).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trees", nargs="*", default=["binary", "4", "mps", "4"])
    args = ap.parse_known_args()[0].trees
    rows = []
    for kind, n in zip(args[::2], map(int, args[1::2])):
        spec, lam, a1, a2, a3 = operator_pieces(kind, n)
        for batch in BATCHES:
            lo, hi = 0.0, 2.0 / lam
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if spectral_radius(mid, batch, a1, a2, a3) < 1 else (lo, mid)
            est = 2 / (lam * (1 + (3**n - 1) / batch))
            rows.append(dict(tree=kind, N=n, K=spec.n_cores, B=batch, lam_max=lam,
                             eta_gd=2 / lam, eta_ms=lo, eta_1d=est))
            print(rows[-1], flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "sgd_stability.csv", index=False)


if __name__ == "__main__":
    main()
