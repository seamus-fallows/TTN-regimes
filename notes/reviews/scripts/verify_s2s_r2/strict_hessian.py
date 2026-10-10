"""Exact Hessian at the balanced partial fit W_S of binary4 (own dense contraction).
Teacher close [1, 0.85, 0.7] (repo odeco teacher, seed 1000). Learned sets S given as 0-based lists.
Prints the lowest eigenvalues (incl. gauge zero modes) and the piggyback prediction s_q^(1/3)(s_q - s_r)."""

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
REPO = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
sys.path.insert(0, str(REPO / "experiments/s2s_core"))
sys.path.insert(0, str(REPO))
import s2s  # noqa: E402

for weights in ([1.0, 0.85, 0.7], [3.0, 2.0, 1.0]):
    for chi in (3, 6):
        spec, tch = s2s.setup("binary", 4, 4, chi, weights, 1000)
        U = tch.factors.clone()  # (R, N, d)
        s = tch.weights.clone()
        d = 4
        Wstar = sum(
            s[r] * torch.einsum("i,j,k,l->ijkl", U[r, 0], U[r, 1], U[r, 2], U[r, 3])
            for r in range(3)
        )
        n0, n1, n2 = 4 * 4 * chi, 4 * 4 * chi, chi * chi

        def W_of(x):
            A = x[:n0].reshape(4, 4, chi)
            B = x[n0 : n0 + n1].reshape(4, 4, chi)
            M = x[n0 + n1 :].reshape(chi, chi)
            return torch.einsum("ija,ab,klb->ijkl", A, M, B)

        def L(x):
            W = W_of(x)
            return 0.5 * ((W - Wstar) ** 2).sum()

        for S in ([0, 2], [0], [0, 1], [1], [2], [1, 2]):
            A = torch.zeros(4, 4, chi, dtype=torch.float64)
            B = torch.zeros(4, 4, chi, dtype=torch.float64)
            M = torch.zeros(chi, chi, dtype=torch.float64)
            for j, r in enumerate(S):
                q = float(s[r]) ** (1 / 3)
                A[:, :, j] = q * torch.outer(U[r, 0], U[r, 1])
                B[:, :, j] = q * torch.outer(U[r, 2], U[r, 3])
                M[j, j] = q
            x = torch.cat([A.reshape(-1), B.reshape(-1), M.reshape(-1)])
            g = torch.autograd.functional.jacobian(L, x)
            H = torch.autograd.functional.hessian(L, x)
            ev = torch.linalg.eigvalsh(H).numpy()
            unl = [r for r in range(3) if r not in S]
            preds = [
                float(s[q]) ** (1 / 3) * (float(s[q]) - max(float(s[r]) for r in unl)) for q in S
            ]
            print(
                f"w={weights} chi={chi} S={[r + 1 for r in S]} |grad|={g.norm():.1e} lowest eig={np.round(ev[:3], 6)} "
                f"n(<-1e-9)={int((ev < -1e-9).sum())} piggyback={np.round(min(preds), 6)}"
            )
