"""Reviewer's independent finite-sample check (no code from experiments/optim_samples or ttn/).

Binary N=8 tree, d=2: leaves L0..L3 carry sites (0,1),(2,3),(4,5),(6,7); middles M0 (L0,L1), M1 (L2,L3);
root R (M0, M1). Bond dims: leaf->middle min(chi, 4), middle->root min(chi, 16).
Own odeco teacher [3,2] (random orthonormal basis per site), own data x ~ N(0, I).
(i) lazy limit = min-norm solution of (Phi_X J0) dtheta = y at a fan-in init, test = ||J0 dtheta - W*||^2/||W*||^2;
(ii) small-init GD (sigma = 0.5) on the empirical loss, step = 1/lambda_max (power iteration every 200
steps) capped at 1% relative parameter change; test error and bond ranks of the learned tensor.
Usage: python indep_samples.py OUTDIR lazy|gd chi n seed [steps]
"""

import json
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
N, D = 8, 2
S = np.array([3.0, 2.0])


def shapes(chi):
    a, b = min(chi, 4), min(chi, 16)
    return [(D, D, a)] * 4 + [(a, a, b)] * 2 + [(b, b)]


def split(th, chi):
    out, i = [], 0
    for s in shapes(chi):
        n = int(np.prod(s))
        out.append(th[i : i + n].reshape(s))
        i += n
    return out


def dense(th, chi, lib=np):
    c = split(th, chi)
    m0 = lib.einsum("abx,cdy,xyp->abcdp", c[0], c[1], c[4])
    m1 = lib.einsum("abx,cdy,xyq->abcdq", c[2], c[3], c[5])
    return lib.einsum("abcdp,efghq,pq->abcdefgh", m0, m1, c[6])


def forward(th, chi, x):
    """x: (n, N, D) torch."""
    c = split(th, chi)
    leaves = [torch.einsum("abx,na,nb->nx", c[k], x[:, 2 * k], x[:, 2 * k + 1]) for k in range(4)]
    m0 = torch.einsum("xyp,nx,ny->np", c[4], leaves[0], leaves[1])
    m1 = torch.einsum("xyq,nx,ny->nq", c[5], leaves[2], leaves[3])
    return torch.einsum("pq,np,nq->n", c[6], m0, m1)


def teacher(rng):
    u = np.zeros((2, N, D))
    for i in range(N):
        q, r = np.linalg.qr(rng.standard_normal((D, D)))
        q = q * np.sign(np.diag(r))
        u[:, i] = q.T[:2]
    w = 0
    for r in range(2):
        t = S[r]
        for i in range(N):
            t = np.multiply.outer(t, u[r, i])
        w = w + t
    return w


def init(rng, chi, sigma):
    parts = []
    for s in shapes(chi):
        fan = int(np.prod(s[:-1])) if len(s) == 3 else int(np.prod(s))
        parts.append(sigma * rng.standard_normal(s).ravel() / np.sqrt(fan))
    return np.concatenate(parts)


def features(x):
    phi = x[:, 0]
    for i in range(1, N):
        phi = (phi[:, :, None] * x[:, i][:, None, :]).reshape(len(x), -1)
    return phi


def jacobian(th, chi):
    w0 = dense(th, chi).ravel()
    jac = np.empty((w0.size, th.size))
    for i in range(th.size):
        e = th.copy()
        e[i] += 1.0
        jac[:, i] = dense(e, chi).ravel() - w0
    return jac


def bond_ranks(w, thr=0.2):
    cuts = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1, 2, 3]]
    out = []
    for cut in cuts:
        rest = [i for i in range(N) if i not in cut]
        m = np.transpose(w, cut + rest).reshape(D ** len(cut), -1)
        out.append(int(np.sum(np.linalg.svd(m, compute_uv=False) > thr)))
    return out


def setup(chi, n, seed):
    wstar = teacher(np.random.default_rng(777))  # one fixed teacher for every run
    x = np.random.default_rng(9000 + seed).standard_normal((n, N, D))
    y = features(x) @ wstar.ravel()
    th0 = init(np.random.default_rng(5000 + seed), chi, 0.5)
    return wstar, x, y, th0


def lazy(chi, n, seed):
    wstar, x, y, th0 = setup(chi, n, seed)
    jac = jacobian(th0, chi)
    a = features(x) @ jac
    dth = np.linalg.lstsq(a, y, rcond=1e-12)[0]
    w = (jac @ dth).reshape(wstar.shape)
    return dict(
        mode="lazy",
        chi=chi,
        n=n,
        seed=seed,
        rank_J=int(np.linalg.matrix_rank(jac, tol=1e-10 * np.linalg.norm(jac, 2))),
        test_rel=float(np.sum((w - wstar) ** 2) / np.sum(wstar**2)),
        ranks=bond_ranks(w),
    )


def gd(chi, n, seed, max_steps):
    wstar, x, y, th0 = setup(chi, n, seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    p = torch.from_numpy(th0.copy())

    def loss_fn(q):
        return 0.5 * (forward(q, chi, xt) - yt).pow(2).mean()

    y_scale = 0.5 * float((yt**2).mean())
    v = torch.randn(p.shape, generator=torch.Generator().manual_seed(seed))
    t0 = time.time()
    status = "max_steps"
    for k in range(max_steps):
        if k % 200 == 0:
            q = p.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(loss_fn(q), q, create_graph=True)
            v = v / v.norm()
            for _ in range(20):
                (hv,) = torch.autograd.grad(g @ v, q, retain_graph=True)
                lam = abs(float(hv @ v))
                v = hv / hv.norm()
            eta_lam = 1.0 / max(lam, 1e-300)
            v = v.detach()
        q = p.detach().requires_grad_(True)
        lv = loss_fn(q)
        (g,) = torch.autograd.grad(lv, q)
        if float(lv) < 1e-8 * y_scale:
            status = "converged"
            break
        eta = min(eta_lam, 0.01 * float(p.norm()) / float(g.norm()))
        p = (p - eta * g).detach()
    w = dense(p.numpy(), chi)
    return dict(
        mode="gd",
        chi=chi,
        n=n,
        seed=seed,
        status=status,
        steps=k,
        train_rel=float(loss_fn(p)) / y_scale,
        test_rel=float(np.sum((w - wstar) ** 2) / np.sum(wstar**2)),
        ranks=bond_ranks(w),
        wall=time.time() - t0,
    )


if __name__ == "__main__":
    out, mode, chi, n, seed = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
    )
    if mode == "lazy":
        res = lazy(chi, n, seed)
    else:
        res = gd(chi, n, seed, int(sys.argv[6]) if len(sys.argv) > 6 else 60000)
    print(json.dumps(res), flush=True)
    with open(f"{out}/indep_samples.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
