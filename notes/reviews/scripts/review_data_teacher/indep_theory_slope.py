"""Reviewer: independent recomputation of the predicted log slopes A (theory_teachers.md R6) for the
e4a teachers (MPS-4, s = (1, 0.7), per-site overlap c), with rank-1 saddles from the higher-order power
method (HOPM) started near each component instead of constrained.py's random-start L-BFGS.
A_e = (1 - s2(M_e)/lam) / (lam^((N-2)/N) s2(M_e)), M_e = W* contracted with the saddle's site vectors
outside the two sites of edge e's 2-core route, lam = <W*, (x) w_i>."""

import math

import torch

torch.set_num_threads(1)
N, D = 4, 4
for c in (0.05, 0.2, 0.45):
    u1 = torch.zeros(D, dtype=torch.float64)
    u1[0] = 1
    u2 = torch.zeros(D, dtype=torch.float64)
    u2[0], u2[1] = c, math.sqrt(1 - c * c)
    T = torch.einsum("a,b,c,d->abcd", u1, u1, u1, u1) + 0.7 * torch.einsum(
        "a,b,c,d->abcd", u2, u2, u2, u2
    )
    L0 = 0.5 * float((T * T).sum())
    for start_name, start in (("near u1", u1), ("near u2", u2)):
        w = [start.clone() for _ in range(N)]
        for _ in range(100000):
            old = [x.clone() for x in w]
            for i in range(N):
                M = T
                for k in sorted([k for k in range(N) if k != i], reverse=True):
                    M = torch.tensordot(M, w[k], dims=([k], [0]))
                w[i] = M / M.norm()
            if max(float((a - b).norm()) for a, b in zip(w, old)) < 1e-15:
                break
        lam = float(torch.einsum("abcd,a,b,c,d->", T, *w))
        saddle_loss = L0 - 0.5 * lam**2
        A = []
        for i, j in ((0, 1), (1, 2), (2, 3)):
            M = T
            for k in sorted([k for k in range(N) if k not in (i, j)], reverse=True):
                M = torch.tensordot(M, w[k], dims=([k], [0]))
            s2 = float(torch.linalg.svdvals(M)[1])
            A.append((1 - s2 / lam) / (lam ** ((N - 2) / N) * s2))
        print(
            f"c={c} start {start_name}: saddle loss {saddle_loss:.6f} (rel {saddle_loss / L0:.4f}), "
            f"lam {lam:.6f}, A_e = {[round(a, 3) for a in A]}, A_pred = {min(A):.3f}"
        )
