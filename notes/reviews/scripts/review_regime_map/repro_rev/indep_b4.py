"""Independent re-implementation (numpy + scipy only, no ttn / regime_map code) of the binary N = 4
(K = 3) centred-model gradient flow and the regime diagnostics, for the reproduction review.

Model (notes/model.md): W = A R B^T across the 01|23 cut (A, B: 9 x chi leaf cores, R: chi x chi root),
fan-in init A, B = sigma Z / 3, R = lambda sigma Z / chi; odeco teacher [3, 2, 1] at d = 3; centred
target W* + W_0; loss 1/2 ||W - W_0 - W*||^2; per-core rates eta (sp / ntk / mup as theory_regimes 4.3).
Diagnostics (notes/diagnostics.md): a_r = <W - W_0, U_r> / s_r; t_r = first upward crossing of 1/2;
width = first reach of 0.9 minus last upward crossing of 0.1 before it; rho_i on time-sorted events;
probe kernel on 128 Gaussian inputs, delta_K = ||Theta_T - Theta_0||_F / ||Theta_0||_F.
Own RNG streams (numpy), own integrator (scipy DOP853 with root-found events), own kernel formula.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

D_SITE = 3
WEIGHTS = np.array([3.0, 2.0, 1.0])


def teacher(seed):
    rng = np.random.default_rng([seed, 4242])
    Q = [np.linalg.qr(rng.standard_normal((D_SITE, D_SITE)))[0] for _ in range(4)]
    U = [
        np.outer(np.kron(Q[0][:, r], Q[1][:, r]), np.kron(Q[2][:, r], Q[3][:, r])) for r in range(3)
    ]
    return sum(s * u for s, u in zip(WEIGHTS, U)), U


def probes(n=128, seed=99):
    rng = np.random.default_rng([seed, 31337])
    x = rng.standard_normal((n, 4, D_SITE))
    Pa = np.einsum("ni,nj->nij", x[:, 0], x[:, 1]).reshape(n, -1)
    Pb = np.einsum("ni,nj->nij", x[:, 2], x[:, 3]).reshape(n, -1)
    return Pa, Pb


PA, PB = probes()
GA, GB = PA @ PA.T, PB @ PB.T


def kernel(A, R, B, eta):
    """Probe tangent kernel Theta = J diag(eta) J^T (derived by hand for W = A R B^T)."""
    eA, eR, eB = eta
    MA = B @ (R.T @ R) @ B.T
    MB = A @ (R @ R.T) @ A.T
    th = eA * GA * (PB @ MA @ PB.T)
    th += eR * (PA @ (A @ A.T) @ PA.T) * (PB @ (B @ B.T) @ PB.T)
    th += eB * (PA @ MB @ PA.T) * GB
    return th


def setup(chi, sigma, seed, lr="sp", root_scale=1.0):
    rng = np.random.default_rng([seed, 2718, chi])
    rs = root_scale * ((2 * chi) ** -0.5 if lr == "mup" else 1.0)
    A = sigma * rng.standard_normal((9, chi)) / 3.0
    B = sigma * rng.standard_normal((9, chi)) / 3.0
    R = rs * sigma * rng.standard_normal((chi, chi)) / chi
    eta = {
        "sp": (1.0, 1.0, 1.0),
        "ntk": (1 / 9, 1 / chi**2, 1 / 9),
        "mup": (chi / 9, 1 / chi**2, chi / 9),
    }[lr]
    return A, R, B, eta


def run(
    chi,
    sigma,
    seed,
    lr="sp",
    root_scale=1.0,
    rtol=1e-10,
    atol=1e-14,
    l_floor=1e-6,
    t_max=None,
    target="centred",
):
    Ws, U = teacher(seed)
    A0, R0, B0, eta = setup(chi, sigma, seed, lr, root_scale)
    W0 = A0 @ R0 @ B0.T
    Y = Ws + W0 if target == "centred" else Ws
    L_ref = 0.5 * float((Ws**2).sum())
    nA, nR = A0.size, R0.size
    shapes = [(9, chi), (chi, chi), (9, chi)]

    def unpack(y):
        return (
            y[:nA].reshape(shapes[0]),
            y[nA : nA + nR].reshape(shapes[1]),
            y[nA + nR :].reshape(shapes[2]),
        )

    def rhs(t, y):
        A, R, B = unpack(y)
        E = Y - A @ R @ B.T
        return np.concatenate(
            [
                (eta[0] * (E @ B @ R.T)).ravel(),
                (eta[1] * (A.T @ E @ B)).ravel(),
                (eta[2] * (E.T @ A @ R)).ravel(),
            ]
        )

    def ov(y, r):
        A, R, B = unpack(y)
        return float(((A @ R @ B.T - W0) * U[r]).sum()) / WEIGHTS[r]

    events = []
    for r in range(3):
        for lev in (0.1, 0.5, 0.9):
            f = (lambda r, lev: lambda t, y: ov(y, r) - lev)(r, lev)
            f.direction = 1
            events.append(f)

    def floor_ev(t, y):
        A, R, B = unpack(y)
        return 0.5 * float(((A @ R @ B.T - Y) ** 2).sum()) - l_floor * L_ref

    floor_ev.terminal, floor_ev.direction = True, -1
    events.append(floor_ev)

    # time budget like the agent's rule but computed here: 50 t_esc + 1e4 / max(lam_bar, 1)
    lam_bar = (
        eta[0] * 9 * np.trace(B0 @ R0.T @ R0 @ B0.T)
        + eta[1] * np.trace(A0 @ A0.T) * np.trace(B0 @ B0.T)
        + eta[2] * 9 * np.trace(A0 @ R0 @ R0.T @ A0.T)
    ) / 81.0
    alpha = sigma**3 * root_scale * ((2 * chi) ** -0.5 if lr == "mup" else 1.0)
    if t_max is None:
        t_max = 50 * (0.3 * sigma) ** -1 / WEIGHTS.min() + 1e4 / max(lam_bar, 1.0)
    y0 = np.concatenate([A0.ravel(), R0.ravel(), B0.ravel()])
    sol = solve_ivp(rhs, (0.0, t_max), y0, method="DOP853", rtol=rtol, atol=atol, events=events)
    yT = sol.y[:, -1]
    A, R, B = unpack(yT)
    l_end = 0.5 * float(((A @ R @ B.T - Y) ** 2).sum()) / L_ref
    K0, KT = kernel(A0, R0, B0, eta), kernel(A, R, B, eta)
    dK = float(np.linalg.norm(KT - K0) / np.linalg.norm(K0))
    # events
    times, widths = [], []
    for r in range(3):
        lo, mid, hi = (np.asarray(sol.t_events[3 * r + j]) for j in range(3))
        t_mid = mid[0] if mid.size else np.nan
        if hi.size:
            t_hi = hi[0]
            before = lo[lo < t_hi]
            t_lo = before[-1] if before.size else 0.0
            w = t_hi - t_lo
        else:
            w = np.nan
        times.append(t_mid)
        widths.append(w)
    order = np.argsort(times)
    ts, ws = np.array(times)[order], np.array(widths)[order]
    rhos = [(ts[i + 1] - ts[i]) / max(ws[i], ws[i + 1]) for i in range(2)]
    min_rho = float(np.min(rhos)) if np.all(np.isfinite(rhos)) else np.nan
    # diagnostics.md rule for transients above 1/2: last upward crossing of 1/2 (before reaching 0.9 for good)
    n_up = [int(np.asarray(sol.t_events[3 * r + 1]).size) for r in range(3)]
    times_last = []
    for r in range(3):
        mid = np.asarray(sol.t_events[3 * r + 1])
        times_last.append(float(mid[-1]) if mid.size else np.nan)
    o2 = np.argsort(times_last)
    tl, wl = np.array(times_last)[o2], np.array(widths)[o2]
    rhos_last = [(tl[i + 1] - tl[i]) / max(wl[i], wl[i + 1]) for i in range(2)]
    min_rho_last = float(np.min(rhos_last)) if np.all(np.isfinite(rhos_last)) else np.nan
    if l_end >= 1e-3:
        label = "stalled"
    elif dK < 0.1:
        label = "lazy"
    elif np.isfinite(min_rho) and min_rho > 3:
        label = "s2s"
    else:
        label = "rich"
    # Eckart-Young floor of the centred target at rank chi (only meaningful for chi < 9)
    sv = np.linalg.svd(Y, compute_uv=False)
    l_best = float((sv[chi:] ** 2).sum() / (2 * L_ref)) if chi < 9 else 0.0
    rel_root = float(np.linalg.norm(R - R0) / np.linalg.norm(R0))
    rel_leaf = float(
        0.5
        * (
            np.linalg.norm(A - A0) / np.linalg.norm(A0)
            + np.linalg.norm(B - B0) / np.linalg.norm(B0)
        )
    )
    dth = float(np.linalg.norm(yT - y0) / np.linalg.norm(y0))
    return dict(
        chi=chi,
        sigma=sigma,
        seed=seed,
        lr=lr,
        root_scale=root_scale,
        alpha=alpha,
        t_end=float(sol.t[-1]),
        t_max=t_max,
        status=int(sol.status),
        nfev=int(sol.nfev),
        l_end=l_end,
        l_best=l_best,
        dK=dK,
        times=[float(x) for x in times],
        widths=[float(x) for x in widths],
        rhos=[float(x) for x in rhos],
        min_rho=min_rho,
        n_up=n_up,
        min_rho_last=min_rho_last,
        label=label,
        rel_root=rel_root,
        rel_leaf=rel_leaf,
        dth=dth,
    )


def bisect_log(f, lo, hi, target, n_iter=7):
    """Find x in [lo, hi] (log scale) with f(x) = target, f monotone; returns (x, evaluations).
    Uses log-linear interpolation between the bracketing evaluations after bisection steps."""
    evals = {}

    def g(x):
        if x not in evals:
            evals[x] = f(x)
        return evals[x]

    flo, fhi = g(lo), g(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or (flo - target) * (fhi - target) > 0:
        return np.nan, evals
    for _ in range(n_iter):
        mid = math.sqrt(lo * hi)
        fm = g(mid)
        if not np.isfinite(fm):
            return np.nan, evals
        if (flo - target) * (fm - target) <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    w = (math.log(target) - math.log(flo)) / (math.log(fhi) - math.log(flo))
    return math.exp(math.log(lo) + w * (math.log(hi) - math.log(lo))), evals
