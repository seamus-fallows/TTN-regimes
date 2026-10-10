"""Independent check of the real-data claims WITHOUT experiments/real_data code.

Own code: IDX parsing, crop + block average, balanced subsample (protocol of notes/real_data.md, data
seed 0), pixel orders, cos/sin features, empirical loss (autograd gradient), chunked DOP853 phase 1 (time
of the 1% loss drop, t_a), RK45 phase 2, rate-peak events (scipy find_peaks, relative prominence 2/3 =
"valley factor 3"), data-weighted bond spectra (up messages by own contraction, down messages by autograd
retain_grad), route amplitudes from the DENSE tensor (bottom-core directions from unfoldings of W).
Library used only for: ttn.tree (binary_tree, mps_tree, path), ttn.network (TTNSpec.uniform, init,
flatten/unflatten, forward, to_dense).

Usage: uv run python -I indep.py <binary|mps> <ordering> <sigma> <seed> <T2 (0 = phase 1 only)> <out.json>
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import gzip
import json
import struct
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from ttn.network import TTNSpec, forward, to_dense
from ttn.tree import binary_tree, mps_tree

torch.set_num_threads(1)
RAW = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_real_data/data/mnist_raw"


def idx(path):
    with gzip.open(path, "rb") as f:
        b = f.read()
    nd = b[3]
    dims = struct.unpack(">" + "I" * nd, b[4 : 4 + 4 * nd])
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def to4x4(img):
    a = img[:, 2:26, 2:26].astype(np.float64) / 255.0
    return a.reshape(len(a), 4, 6, 4, 6).mean(axis=(2, 4)).reshape(len(a), 16)


def task(seed=0):
    xtr, ytr = idx(RAW + "/train-images-idx3-ubyte.gz"), idx(RAW + "/train-labels-idx1-ubyte.gz")
    xte, yte = idx(RAW + "/t10k-images-idx3-ubyte.gz"), idx(RAW + "/t10k-labels-idx1-ubyte.gz")
    rng = np.random.default_rng(seed)
    out = []
    for x, lab, n in ((xtr, ytr, 4000), (xte, yte, 1600)):
        p, q = np.flatnonzero(lab == 3), np.flatnonzero(lab == 5)
        sel = np.concatenate(
            [rng.choice(p, n // 2, replace=False), rng.choice(q, n - n // 2, replace=False)]
        )
        sel = rng.permutation(sel)
        out += [to4x4(x[sel]), np.where(lab[sel] == 3, 1.0, -1.0)]
    return out


def pixel_order(name):
    if (
        name == "quadtree"
    ):  # Z order: bit3 = row bit1, bit2 = col bit1, bit1 = row bit0, bit0 = col bit0

        def key(p):
            r, c = divmod(p, 4)
            return ((r >> 1) & 1) << 3 | ((c >> 1) & 1) << 2 | (r & 1) << 1 | (c & 1)

        return np.array(sorted(range(16), key=key))
    if name == "snake":
        return np.array([0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12])
    if name.startswith("scrambled"):
        return np.random.default_rng(int(name[9:]) + 12345).permutation(16)
    raise ValueError(name)


def feats(x):
    a = 0.5 * np.pi * x
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


class Model:
    def __init__(self, kind, ordering, sigma, seed):
        xtr, ytr, xte, yte = task(0)
        self.order = pixel_order(ordering)
        self.X = torch.tensor(feats(xtr[:, self.order]))
        self.Y = torch.tensor(ytr)
        self.Xte = torch.tensor(feats(xte[:, self.order]))
        self.Yte = torch.tensor(yte)
        self.tree = binary_tree(16) if kind == "binary" else mps_tree(16)
        self.spec = TTNSpec.uniform(self.tree, 2, 4)
        self.theta0 = self.spec.flatten(self.spec.init(sigma, seed=seed)).numpy().copy()
        self.n_eval = 0

    def f(self, th, X=None):
        return forward(self.spec, self.spec.unflatten(th), self.X if X is None else X)

    def loss(self, y):
        with torch.no_grad():
            return 0.5 * ((self.f(torch.as_tensor(y)) - self.Y) ** 2).mean().item()

    def grad(self, y):
        self.n_eval += 1
        th = torch.tensor(y, requires_grad=True)
        L = 0.5 * ((self.f(th) - self.Y) ** 2).mean()
        (g,) = torch.autograd.grad(L, th)
        return g.numpy()

    def rhs(self, t, y):
        return -self.grad(y)

    # data-weighted bond spectra: up messages by own contraction, down = d(sum_j f_j)/d up (autograd)
    def spectra(self, y):
        t, X = self.tree, self.X
        letters = "abcdefgh"
        cores_g = [c.detach().clone() for c in self.spec.unflatten(torch.as_tensor(y))]
        upg = [None] * t.n_nodes
        for v in t.postorder:
            ops = [X[:, s, :] for s in t.sites[v]] + [upg[c] for c in t.children[v]]
            legs = letters[: len(ops) + 1]
            eq = legs + "," + ",".join("z" + legs[i] for i in range(len(ops))) + "->z" + legs[-1]
            if t.parent[v] >= 0 and not any(o.requires_grad for o in ops[len(t.sites[v]):]):
                upg[v] = torch.einsum(eq, cores_g[v], *ops).requires_grad_(True)  # leaf message
            else:
                upg[v] = torch.einsum(eq, cores_g[v], *ops)
                if t.parent[v] >= 0:
                    upg[v].retain_grad()
        upg[t.root][:, 0].sum().backward()
        n = X.shape[0]
        out = []
        for v in t.edges:
            U, D = upg[v].detach(), upg[v].grad.detach()
            gu, gd = (U.T @ U / n).numpy(), (D.T @ D / n).numpy()
            w, V = np.linalg.eigh(gu)
            r = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
            ev = np.sqrt(np.clip(np.linalg.eigvalsh(r @ gd @ r), 0, None))[::-1]
            row = np.full(4, np.nan)
            row[: len(ev)] = ev
            out.append(row)
        return np.array(out)  # (E, 4), NaN-padded for capped bonds


def phase1(m, rtol=1e-10, atol=1e-14):
    """Chunked DOP853: a chunk ends when |theta| grew by 20% (clock re-anchored) or at 2 |theta|/|v|."""
    y = m.theta0.copy()
    L0 = m.loss(y)
    target = 0.99 * L0
    t_off, chunks = 0.0, 0
    while True:
        chunks += 1
        g = m.grad(y)
        r0 = np.linalg.norm(y)
        D = 2.0 * r0 / np.linalg.norm(g)

        def ev_loss(t, z):
            return m.loss(z) - target

        ev_loss.terminal, ev_loss.direction = True, -1

        def ev_norm(t, z):
            return np.linalg.norm(z) - 1.2 * r0

        ev_norm.terminal, ev_norm.direction = True, 1
        sol = solve_ivp(
            m.rhs, (0.0, D), y, method="DOP853", rtol=rtol, atol=atol, events=[ev_loss, ev_norm]
        )
        if sol.t_events[0].size:
            return t_off + sol.t_events[0][0], sol.y_events[0][0], L0, chunks
        if sol.t_events[1].size:
            t_off += sol.t_events[1][0]
            y = sol.y_events[1][0]
        else:
            t_off += sol.t[-1]
            y = sol.y[:, -1]
        if chunks > 5000:
            raise RuntimeError("too many chunks")


def rate_peaks(t, L, L0):
    Lm = np.minimum.accumulate(L)
    tm = 0.5 * (t[1:] + t[:-1])
    r = np.maximum(-np.diff(Lm) / np.diff(t), 0)
    pk, prop = find_peaks(np.concatenate([[0], r, [0]]), prominence=0)
    pk = pk - 1
    keep = [p for p, pr in zip(pk, prop["prominences"]) if pr >= (2 / 3) * r[p]]
    # bracketing minima, drop size >= 1% of L0
    out = []
    for i, p in enumerate(keep):
        lo = keep[i - 1] if i > 0 else 0
        hi = keep[i + 1] if i + 1 < len(keep) else len(r) - 1
        a = lo + int(np.argmin(r[lo : p + 1])) if i > 0 else 0
        b = p + int(np.argmin(r[p : hi + 1])) if i + 1 < len(keep) else len(r) - 1
        size = Lm[a] - Lm[b + 1]
        half = r[p] / 2
        j = p
        while j > 0 and r[j - 1] > half:
            j -= 1
        k = p
        while k < len(r) - 1 and r[k + 1] > half:
            k += 1
        out.append(
            {
                "t": float(tm[p]),
                "rate": float(r[p]),
                "size": float(size),
                "fwhm": float(tm[k] - tm[j]),
                "t_lo": float(tm[j]),
                "t_hi": float(tm[k]),
            }
        )
    return [d for d in out if d["size"] >= 0.01 * L0]


def route_amplitudes(m, y):
    """lambda_uw from the dense tensor: bottom-core directions g_u = top left singular vector of the
    (4 x 2^14) unfolding of W at the core's two sites (unit norm); c = <W, (x) g>; h_uw = c prod q_v."""
    spec, t = m.spec, m.tree
    cores = spec.unflatten(torch.as_tensor(y))
    with torch.no_grad():
        W = to_dense(spec, cores).numpy()
        f = m.f(torch.as_tensor(y)).numpy()
    bottom = [v for v in range(t.n_nodes) if len(t.sites[v]) == 2]
    X = m.X.numpy()
    g, q, phi, core_sv = {}, {}, {}, {}
    for v in bottom:
        s1, s2 = t.sites[v]
        rest = [a for a in range(16) if a not in (s1, s2)]
        unf = np.transpose(W, [s1, s2] + rest).reshape(4, -1)
        u, s, _ = np.linalg.svd(unf, full_matrices=False)
        g[v] = u[:, 0]
        phi[v] = np.einsum("na,nb->nab", X[:, s1], X[:, s2]).reshape(len(X), 4)
        q[v] = phi[v] @ g[v]
        core_sv[v] = float(np.linalg.svd(cores[v].reshape(4, -1).numpy(), compute_uv=False)[0])
    # c = <W, (x)_v g_v> (the bottom cores carry all 16 sites on binary16)
    perm = []
    for v in bottom:
        perm += list(t.sites[v])
    Wp = np.transpose(W, perm).reshape([4] * len(bottom))
    for v in bottom:
        Wp = np.tensordot(Wp, g[v], axes=([0], [0]))
    c = float(Wp)
    # rank-1 relative error on the data
    prod_all = np.prod(np.stack([q[v] for v in bottom]), axis=0)
    rel = float(np.sqrt(np.mean((f - c * prod_all) ** 2) / np.mean(f**2)))
    r = f - m.Y.numpy()
    pairs = []
    for i, u in enumerate(bottom):
        for w in bottom[i + 1 :]:
            h = c * np.prod(np.stack([q[v] for v in bottom if v not in (u, w)]), axis=0)
            M = (phi[u] * (r * h)[:, None]).T @ phi[w] / len(r)
            Pu = np.eye(4) - np.outer(g[u], g[u])
            Pw = np.eye(4) - np.outer(g[w], g[w])
            lam = float(np.linalg.norm(Pu @ M @ Pw, 2))
            pairs.append(
                {
                    "u": u,
                    "w": w,
                    "k": len(t.path(u, w)),
                    "lam_unit": lam,
                    "lam_core_norm": lam / (core_sv[u] * core_sv[w]),
                }
            )
    return {"pairs": pairs, "c": c, "rank1_rel_err": rel, "core_sv": core_sv}


def main(kind, ordering, sigma, seed, T2, out):
    sigma, seed, T2 = float(sigma), int(seed), float(T2)
    wall0 = time.time()
    m = Model(kind, ordering, sigma, seed)
    res = {
        "kind": kind,
        "ordering": ordering,
        "sigma": sigma,
        "seed": seed,
        "order": m.order.tolist(),
        "x_train_checksum": float(m.X.sum()),
        "y_train_sum": float(m.Y.sum()),
    }
    t_a, y_a, L0, chunks = phase1(m)
    res.update(t_a=t_a, L0=L0, chunks=chunks, wall_p1=time.time() - wall0, evals_p1=m.n_eval)
    print(
        json.dumps(
            {k: res[k] for k in ("kind", "ordering", "sigma", "seed", "t_a", "chunks", "wall_p1")}
        ),
        flush=True,
    )
    if T2 > 0:
        dt = 0.05
        grid = np.arange(0.0, T2 + 1e-9, dt)
        sol = solve_ivp(m.rhs, (0.0, T2), y_a, method="RK45", rtol=1e-9, atol=1e-12, t_eval=grid)
        Ls = np.array([m.loss(sol.y[:, i]) for i in range(sol.y.shape[1])])
        tt = sol.t
        res["wall_p2"] = time.time() - wall0
        res["p2_nfev"] = int(sol.nfev)
        drops = rate_peaks(tt, Ls, L0)
        res["drops"] = drops
        # speed on the grid (for the plateau midpoint) and spectra every 0.5 time units
        idx_sp = np.arange(0, len(tt), 10)
        spec_t = tt[idx_sp]
        spectra = np.array([m.spectra(sol.y[:, i]) for i in idx_sp])  # (T, E, chi)
        ratio = spectra[:, :, 1] / spectra[:, :, 0]
        first = []
        for e in range(ratio.shape[1]):
            above = np.flatnonzero(ratio[:, e] >= 0.05)
            if above.size == 0:
                first.append(np.nan)
                continue
            i = above[0]
            if i == 0:
                first.append(float(spec_t[0]))
                continue
            a, b = np.log(ratio[i - 1, e]), np.log(ratio[i, e])
            first.append(
                float(spec_t[i - 1] + (np.log(0.05) - a) / (b - a) * (spec_t[i] - spec_t[i - 1]))
            )
        first = np.array(first)
        res["edges"] = list(m.tree.edges)
        res["rank2_time"] = [None if not np.isfinite(x) else x for x in first]
        if np.isfinite(first).any():
            t0 = np.nanmin(first)
            res["first_rank2_edges"] = [
                int(e) for e in np.flatnonzero(first <= t0 + max(1.0, 0.05 * t0))
            ]
            order_e = np.argsort(np.where(np.isfinite(first), first, np.inf))
            res["first_two_edges"] = [int(order_e[0]), int(order_e[1])]
        res["loss_end"] = float(Ls[-1])
        for lv in (0.25, 0.2, 0.15):
            below = np.flatnonzero(np.minimum.accumulate(Ls) <= lv)
            res[f"t_loss_{lv}"] = float(tt[below[0]]) if below.size else None
        with torch.no_grad():
            fte = m.f(torch.as_tensor(sol.y[:, -1]), m.Xte)
        res["test_acc_end"] = float((torch.sign(fte) == m.Yte).double().mean())
        res["loss_curve"] = {"t": tt[::20].tolist(), "L": Ls[::20].tolist()}
        if kind == "binary" and len(drops) >= 2:
            lo, hi = drops[0]["t_hi"], drops[1]["t_lo"]
            sel = np.flatnonzero((tt > lo) & (tt < hi))[::4]
            if sel.size:
                sp = np.array([np.linalg.norm(m.grad(sol.y[:, i])) for i in sel])
                i_mid = sel[int(np.argmin(sp))]
                res["tau_mid"] = float(tt[i_mid])
                res["loss_mid"] = float(Ls[i_mid])
                res["routes"] = route_amplitudes(m, sol.y[:, i_mid])
        np.savez_compressed(
            out.replace(".json", "_traj.npz"),
            t=tt[::10],
            y=sol.y[:, ::10],
            L=Ls,
            spectra=spectra,
            spec_t=spec_t,
        )
    res["wall_total"] = time.time() - wall0
    json.dump(res, open(out, "w"), default=float)
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("routes", "loss_curve", "order")},
            default=float,
        )[:3000],
        flush=True,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
