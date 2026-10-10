"""Reviewer's independent checks on binary N=4 (K=3), d=3, odeco teacher [3,2,1].

Pure numpy dynamics (own contraction, analytic gradient); no code from experiments/optim_samples.
W[i0,i1,i2,i3] = sum_{x,y} A[i0,i1,x] B[i2,i3,y] C[x,y]  (A, B leaves; C root).

Part A (fully independent: own teacher basis, own init RNG):
  sharpness at the balanced minimum vs K s1^{2-2/K}; GD stability scan around eta_max;
  oscillation excess loss vs s1^2 x 1D-map attractor loss (own map iteration).
Part B (same init and teacher arrays as the agent, taken from the library; own dynamics):
  GF event times (DOP853, rtol 1e-12) and GD event times at eta = r eta_max, r = 0.1, 0.05, 0.025;
  shift/eta and its Richardson extrapolation to eta -> 0 (= c_e, to compare with predict_shift.py).
Usage: python indep_binary4.py A|B OUTDIR
"""

import json
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

D, CHI = 3, 3
S = np.array([3.0, 2.0, 1.0])
K = 3
LAM_BAL = K * S[0] ** (2 - 2 / K)
ETA_MAX = 2 / LAM_BAL
NA = D * D * CHI
NC = CHI * CHI


def unpack(th):
    return (
        th[:NA].reshape(D, D, CHI),
        th[NA : 2 * NA].reshape(D, D, CHI),
        th[2 * NA :].reshape(CHI, CHI),
    )


def dense(th):
    a, b, c = unpack(th)
    return np.einsum("ijx,kly,xy->ijkl", a, b, c, optimize=True)


def grad(th, wstar):
    a, b, c = unpack(th)
    r = np.einsum("ijx,kly,xy->ijkl", a, b, c, optimize=True) - wstar
    ga = np.einsum("ijkl,kly,xy->ijx", r, b, c, optimize=True)
    gb = np.einsum("ijkl,ijx,xy->kly", r, a, c, optimize=True)
    gc = np.einsum("ijkl,ijx,kly->xy", r, a, b, optimize=True)
    return 0.5 * np.sum(r * r), np.concatenate([ga.ravel(), gb.ravel(), gc.ravel()])


def make_teacher(u):
    """u: (R, 4, D) unit vectors. Returns W*, list of U_r."""
    us = [np.einsum("i,j,k,l->ijkl", *u[r]) for r in range(3)]
    return sum(S[r] * us[r] for r in range(3)), us


def overlaps(w, us):
    return np.array([np.sum(w * us[r]) / S[r] for r in range(3)])


def balanced_min(u):
    a = np.zeros((D, D, CHI))
    b = np.zeros((D, D, CHI))
    c = np.zeros((CHI, CHI))
    for r in range(3):
        f = S[r] ** (1 / 3)
        a[:, :, r] = f * np.outer(u[r, 0], u[r, 1])
        b[:, :, r] = f * np.outer(u[r, 2], u[r, 3])
        c[r, r] = f
    return np.concatenate([a.ravel(), b.ravel(), c.ravel()])


def jacobian(th):
    """Exact: W is linear in every single parameter, so W(th + e_i) - W(th) is column i."""
    w0 = dense(th).ravel()
    jac = np.empty((w0.size, th.size))
    for i in range(th.size):
        e = th.copy()
        e[i] += 1.0
        jac[:, i] = dense(e).ravel() - w0
    return jac


def hessian_fd(th, wstar, h=1e-5):
    n = th.size
    hes = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        hes[:, i] = (grad(th + e, wstar)[1] - grad(th - e, wstar)[1]) / (2 * h)
    return 0.5 * (hes + hes.T)


def map1d_loss(ratio, k=3, x0=0.3, n_iter=60000, keep=4000):
    h = ratio * 2.0 / k
    x = x0
    acc = []
    for i in range(n_iter):
        x = x - h * x ** (k - 1) * (x**k - 1)
        if not np.isfinite(x) or abs(x) > 1e6:
            return np.nan, "diverged"
        if i >= n_iter - keep:
            acc.append(x)
    acc = np.array(acc)
    state = "reentry" if np.max(np.abs(acc)) < 0.2 else f"{len(np.unique(np.round(acc, 7)))}pts"
    return 0.5 * np.mean((acc**k - 1) ** 2), state


def run_gd(th, wstar, us, eta, t_end, stop_after_events=None, record_every=1):
    steps = int(np.ceil(t_end / eta))
    loss0 = None
    ts, avals, losses = [], [], []
    a_prev = None
    ev = [np.nan] * 3
    k_hit = None
    for k in range(steps + 1):
        loss, g = grad(th, wstar)
        if loss0 is None:
            loss0 = loss
        if not np.isfinite(loss) or loss > 1e6 * loss0:
            return dict(diverged=True, steps=k, t50=ev, losses=np.array(losses), loss0=loss0)
        a = overlaps(dense(th), us)
        if a_prev is not None:
            for r in range(3):
                if np.isnan(ev[r]) and a[r] > 0.5 >= a_prev[r]:
                    frac = (0.5 - a_prev[r]) / (a[r] - a_prev[r])
                    ev[r] = (k - 1 + frac) * eta
        a_prev = a
        losses.append(loss)
        if stop_after_events is not None and np.sum(np.isfinite(ev)) >= stop_after_events:
            if k_hit is None:
                k_hit = k
            if k >= k_hit + 2:
                break
        if k < steps:
            th = th - eta * g
    return dict(diverged=False, steps=k, t50=ev, losses=np.array(losses), loss0=loss0, th=th)


def gf_events(th0, wstar, us, t_end):
    def rhs(t, y):
        return -grad(y, wstar)[1]

    def mk(r):
        def f(t, y):
            return overlaps(dense(y), us)[r] - 0.5

        f.direction = 1
        return f

    sol = solve_ivp(
        rhs,
        (0, t_end),
        th0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-15,
        events=[mk(r) for r in range(3)],
    )
    ev = [float(e[0]) if len(e) else np.nan for e in sol.t_events]
    return ev, sol.y[:, -1], grad(sol.y[:, -1], wstar)[0]


def random_basis(rng):
    u = np.zeros((3, 4, D))
    for i in range(4):
        q, rr = np.linalg.qr(rng.standard_normal((D, D)))
        q = q * np.sign(np.diag(rr))
        for r in range(3):
            u[r, i] = q[:, r]
    return u


def fanin_init(rng, sigma):
    a = sigma * rng.standard_normal((D, D, CHI)) / np.sqrt(D * D)
    b = sigma * rng.standard_normal((D, D, CHI)) / np.sqrt(D * D)
    c = sigma * rng.standard_normal((CHI, CHI)) / np.sqrt(CHI * CHI)
    return np.concatenate([a.ravel(), b.ravel(), c.ravel()])


def part_a(out):
    rng = np.random.default_rng(20261009)
    u = random_basis(rng)
    wstar, us = make_teacher(u)
    res = {}
    thb = balanced_min(u)
    assert np.allclose(dense(thb), wstar)
    jac = jacobian(thb)
    lam_jtj = np.linalg.eigvalsh(jac.T @ jac)[-1]
    lam_fd = np.linalg.eigvalsh(hessian_fd(thb, wstar))[-1]
    res["sharpness"] = dict(
        lam_JtJ=lam_jtj,
        lam_fd_hessian=lam_fd,
        formula=LAM_BAL,
        rel_err_JtJ=lam_jtj / LAM_BAL - 1,
        rel_err_fd=lam_fd / LAM_BAL - 1,
    )
    # random orthogonal gauge on both bonds
    q1, _ = np.linalg.qr(rng.standard_normal((CHI, CHI)))
    q2, _ = np.linalg.qr(rng.standard_normal((CHI, CHI)))
    a, b, c = unpack(thb.copy())
    a2 = np.einsum("ijx,xz->ijz", a, q1)
    b2 = np.einsum("ijy,yz->ijz", b, q2)
    c2 = q1.T @ c @ q2
    thg = np.concatenate([a2.ravel(), b2.ravel(), c2.ravel()])
    assert np.allclose(dense(thg), wstar)
    res["sharpness"]["lam_gauged"] = np.linalg.eigvalsh(jacobian(thg).T @ jacobian(thg))[-1]
    print(res["sharpness"], flush=True)
    rows = []
    ratios = [0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5]
    for sigma, t_end in ((0.03, 300.0), (0.1, 100.0)):
        for seed in range(5):
            th0 = fanin_init(np.random.default_rng(1000 + seed), sigma)
            ev_gf, th_gf, loss_gf = gf_events(th0, wstar, us, t_end)
            for ratio in ratios:
                eta = ratio * ETA_MAX
                out_gd = run_gd(th0.copy(), wstar, us, eta, t_end)
                ls = out_gd["losses"]
                row = dict(
                    sigma=sigma,
                    seed=seed,
                    ratio=ratio,
                    diverged=out_gd["diverged"],
                    t50_gf=ev_gf,
                    t50_gd=out_gd["t50"],
                    loss_gf_end=loss_gf,
                )
                if not out_gd["diverged"]:
                    tail = ls[-max(10, len(ls) // 20) :]
                    first = (
                        np.nanmin(out_gd["t50"]) if np.any(np.isfinite(out_gd["t50"])) else np.inf
                    )
                    tt = eta * np.arange(len(ls))
                    reentry = bool(np.any((tt > first) & (ls > 0.9 * out_gd["loss0"])))
                    last = ls[-min(2000, max(20, len(ls) // 5)) :]
                    tv = np.sum(np.abs(np.diff(last)))
                    net = abs(last[-1] - last[0])
                    th_end = out_gd["th"]
                    jj = jacobian(th_end)
                    lam_end = np.linalg.eigvalsh(hessian_fd(th_end, wstar))[-1]
                    row.update(
                        loss_end=ls[-1],
                        tail_mean=float(np.mean(tail)),
                        reentry=reentry,
                        tv=tv,
                        net=net,
                        lam_end=lam_end,
                        stab=lam_end * eta / 2,
                        rank_j=int(np.linalg.matrix_rank(jj)),
                    )
                rows.append(row)
            print(sigma, seed, "done", flush=True)
    res["scan"] = rows
    res["map1d"] = {str(r): map1d_loss(r) for r in ratios}
    with open(f"{out}/indep_A.json", "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def part_b(out, which):
    import torch
    from ttn.network import TTNSpec
    from ttn.teachers import odeco_teacher
    from ttn.tree import make_tree

    torch.set_num_threads(1)
    spec = TTNSpec.uniform(make_tree("binary", 4), 3, 3)
    teacher = odeco_teacher(4, 3, [3.0, 2.0, 1.0], seed=101)
    u = teacher.factors.numpy()  # (R, N, d)
    wstar, us = make_teacher(u)
    assert np.allclose(wstar, teacher.dense().numpy())
    rows = []
    for sigma, seed in which:
        cores = spec.init(sigma, seed=seed)
        th0 = np.concatenate(
            [cores[0].numpy().ravel(), cores[1].numpy().ravel(), cores[2].numpy()[..., 0].ravel()]
        )
        from ttn.network import to_dense

        assert np.allclose(dense(th0), to_dense(spec, cores).numpy(), atol=1e-14)
        t0 = time.time()
        ev_gf, _, _ = gf_events(
            th0, wstar, us, {0.003: 2000.0, 0.01: 800.0, 0.03: 300.0, 0.1: 100.0, 0.3: 40.0}[sigma]
        )
        row = dict(sigma=sigma, seed=seed, t50_gf=ev_gf, gd={})
        t_last = np.nanmax(ev_gf) * 1.05 + 1
        for ratio in (0.1, 0.05, 0.025):
            eta = ratio * ETA_MAX
            r = run_gd(th0.copy(), wstar, us, eta, t_last, stop_after_events=3)
            row["gd"][str(ratio)] = dict(
                t50=r["t50"], shift_over_eta=[(a - b) / eta for a, b in zip(r["t50"], ev_gf)]
            )
        s1 = np.array(row["gd"]["0.05"]["shift_over_eta"])
        s0 = np.array(row["gd"]["0.1"]["shift_over_eta"])
        s2 = np.array(row["gd"]["0.025"]["shift_over_eta"])
        row["richardson_c"] = (2 * s2 - s1).tolist()
        row["richardson_c_coarse"] = (2 * s1 - s0).tolist()
        row["wall"] = time.time() - t0
        print(json.dumps(row, default=float), flush=True)
        rows.append(row)
    tag = "_".join(f"{s}-{d}" for s, d in which)
    with open(f"{out}/indep_B_{tag}.json", "w") as f:
        json.dump(rows, f, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "A":
        part_a(sys.argv[2])
    else:
        pairs = [tuple(p.split(":")) for p in sys.argv[3:]]
        part_b(sys.argv[2], [(float(s), int(d)) for s, d in pairs])
