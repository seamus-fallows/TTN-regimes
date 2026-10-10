"""Independent re-implementation (reviewer) of the key s2s-core measurements on binary4 / mps4.

Does NOT use experiments/s2s_core (s2s.py etc.) and uses the ttn library only as a data source for
the identical random draws (TTNSpec.init for the fan-in init, odeco_teacher for the teacher factors).
Everything else is written here: dense contraction of the 4-site TTN, the loss 1/2 |W - W*|^2,
gradient flow with scipy LSODA + exact Hessian Jacobian (autograd), event location with solve_ivp
event functions (root finding on the dense interpolant), plateau midpoints, dense SVD bond spectra,
my own gauge balancing, and the correlation-flow blow-up time by a different method (projected
flow on the unit sphere, T = rho0^-(K-2) int exp(-(K-2) Lambda) dtau), plus c_K by quadrature.

    uv run python indep.py first  binary4 3 SEED SIGMA[,SIGMA...] OUT.json
    uv run python indep.py full   mps4 3 SEED SIGMA OUT.json [fanin|balanced]
    uv run python indep.py corr   binary4 3 SEEDS OUT.json
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
from scipy.integrate import quad, solve_ivp

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

from ttn.network import TTNSpec  # noqa: E402  (data source only)
from ttn.teachers import odeco_teacher  # noqa: E402  (data source only)
from ttn.tree import make_tree  # noqa: E402

D, N, WEIGHTS = 4, 4, [3.0, 2.0, 1.0]


class Model:
    def __init__(self, tree: str, chi: int):
        self.tree = tree
        if tree == "binary4":
            c = min(chi, 16)
            self.shapes = [(D, D, c), (D, D, c), (c, c, 1)]
            self.eq = "ija,klb,abz->ijkl"
            self.cuts = [(16, 16)]  # both edges: {0,1}|{2,3}
            self.phys = {0: [0, 1], 1: [2, 3]}  # node -> sites
        elif tree == "mps4":
            c0, c1, c2 = min(chi, 4), min(chi, 16), min(chi, 4)
            self.shapes = [(D, c0), (D, c0, c1), (D, c1, c2), (D, c2, 1)]
            self.eq = "ia,jab,kbc,lcz->ijkl"
            self.cuts = [(4, 64), (16, 16), (64, 4)]
            self.phys = {0: [0], 1: [1], 2: [2], 3: [3]}
        else:
            raise ValueError(tree)
        self.K = len(self.shapes)
        self.sizes = [math.prod(s) for s in self.shapes]
        kind = "binary" if tree.startswith("binary") else "mps"
        self.spec = TTNSpec.uniform(make_tree(kind, N), D, chi)
        assert tuple(map(tuple, self.spec.shapes)) == tuple(self.shapes), (
            self.spec.shapes,
            self.shapes,
        )
        t = odeco_teacher(N, D, WEIGHTS, seed=1000)
        self.u = t.factors.clone()  # (R, N, d)
        self.s = torch.tensor(WEIGHTS)
        self.R = len(WEIGHTS)
        self.U = []
        for r in range(self.R):
            u = self.u[r]
            self.U.append(torch.einsum("i,j,k,l->ijkl", u[0], u[1], u[2], u[3]))
        self.Wstar = sum(self.s[r] * self.U[r] for r in range(self.R))
        # sanity: orthonormal factors
        for i in range(N):
            g = self.u[:, i, :] @ self.u[:, i, :].T
            assert torch.allclose(g, torch.eye(self.R), atol=1e-12)

    def cores(self, theta):
        out, i = [], 0
        for sh, sz in zip(self.shapes, self.sizes):
            out.append(theta[i : i + sz].reshape(sh))
            i += sz
        return out

    def W(self, theta):
        return torch.einsum(self.eq, *self.cores(theta))

    def loss(self, theta):
        return 0.5 * (self.W(theta) - self.Wstar).pow(2).sum()

    def corr(self, theta):
        return (self.W(theta) * self.Wstar).sum()

    def overlaps(self, theta):
        w = self.W(theta)
        return torch.stack([(w * self.U[r]).sum() / self.s[r] for r in range(self.R)])

    def grad(self, theta):
        th = theta.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(self.loss(th), th)
        return g

    def hess(self, theta):
        return torch.func.hessian(self.loss)(theta.detach())

    def spectra(self, theta):
        w = self.W(theta).detach()
        return [torch.linalg.svdvals(w.reshape(a, b)).numpy() for a, b in self.cuts]

    def slices(self, theta):
        """(K, R) norm of core v contracted with u_r on its physical legs (nan for site-less cores)."""
        cs = self.cores(theta.detach())
        out = np.full((self.K, self.R), np.nan)
        for v, sites in self.phys.items():
            for r in range(self.R):
                x = cs[v]
                for s in sites:
                    x = torch.tensordot(self.u[r, s], x, dims=([0], [0]))
                out[v, r] = x.norm().item()
        return out


def init(model: Model, seed: int, sigma: float, kind: str = "fanin"):
    cores = model.spec.init(sigma, seed=seed)  # data source: the identical fan-in draw
    if kind == "balanced":
        cores = my_balance(model, cores)
    return torch.cat([c.reshape(-1) for c in cores])


def my_balance(model: Model, cores, tol=1e-13, sweeps=20000):
    """Own gauge balancing: on every edge v -> parent p, replace (G_v, G_p) by (G_v A, A^-1 G_p) with
    A A^T = X the minimiser of tr(M X) + tr(N X^-1) (M = G_v^T G_v, N = G_p,(v) G_p,(v)^T), i.e. the
    matrix geometric mean X = M^-1 # N; sweep until every Q_e = M - N vanishes."""
    tr = model.spec.tree
    cores = [c.clone() for c in cores]

    def psd_pow(m, p):
        vals, vecs = torch.linalg.eigh(0.5 * (m + m.T))
        return (vecs * vals.clamp_min(1e-300) ** p) @ vecs.T

    for _ in range(sweeps):
        worst = 0.0
        scale = max(c.pow(2).sum().item() for c in cores)
        for v in tr.edges:
            p = tr.parent[v]
            chi = model.spec.bond_dims[v]
            leg = len(tr.sites[p]) + tr.children[p].index(v)
            gv = cores[v].reshape(-1, chi)
            gp = torch.movedim(cores[p], leg, 0).reshape(chi, -1)
            m, n = gv.T @ gv, gp @ gp.T
            worst = max(worst, (m - n).abs().max().item())
            mh, mih = psd_pow(m, 0.5), psd_pow(m, -0.5)
            x = mih @ psd_pow(mh @ n @ mh, 0.5) @ mih  # geometric mean M^-1 # N
            a, ainv = psd_pow(x, 0.5), psd_pow(x, -0.5)
            cores[v] = (gv @ a).reshape(cores[v].shape)
            gp_new = (ainv @ gp).reshape(
                [chi] + [s for i, s in enumerate(cores[p].shape) if i != leg]
            )
            cores[p] = torch.movedim(gp_new, 0, leg)
        if worst < tol * scale:
            return cores
    raise RuntimeError("balance did not converge")


def gf(model: Model, theta0, t_max, rtol=1e-10, atol=None, wall=900.0, loss_floor_rel=1e-9):
    """Gradient flow with LSODA + exact Hessian; events: a_r crossing 0.1 / 0.5 / 0.9 (both
    directions), terminal at loss < loss_floor_rel * L0 or wall clock."""
    sigma_scale = theta0.abs().max().item()
    atol = atol if atol is not None else 1e-12 * sigma_scale
    L0 = model.loss(theta0).item()
    wall0 = time.perf_counter()

    def fun(t, y):
        return -model.grad(torch.from_numpy(y)).numpy()

    def jac(t, y):
        return -model.hess(torch.from_numpy(y)).numpy()

    evs = []
    for r in range(model.R):
        for lvl in (0.1, 0.5, 0.9):

            def e(t, y, r=r, lvl=lvl):
                return model.overlaps(torch.from_numpy(y))[r].item() - lvl

            evs.append(e)

    def conv(t, y):
        return model.loss(torch.from_numpy(y)).item() - loss_floor_rel * L0

    conv.terminal = True

    def wallev(t, y):
        return wall - (time.perf_counter() - wall0)

    wallev.terminal = True
    evs += [conv, wallev]
    sol = solve_ivp(
        fun,
        (0.0, t_max),
        theta0.numpy(),
        method="LSODA",
        jac=jac,
        rtol=rtol,
        atol=atol,
        events=evs,
        dense_output=True,
    )
    ev = {}
    k = 0
    for r in range(model.R):
        for lvl in (0.1, 0.5, 0.9):
            ts = sol.t_events[k]
            ys = sol.y_events[k]
            # direction of each crossing from the overlap slope (finite difference on the interpolant)
            dirs = []
            for tt in ts:
                h = max(1e-9 * tt, 1e-9)
                a1 = model.overlaps(torch.from_numpy(sol.sol(min(tt + h, sol.t[-1]))))[r].item()
                a0 = model.overlaps(torch.from_numpy(sol.sol(max(tt - h, 0.0))))[r].item()
                dirs.append(1 if a1 > a0 else -1)
            ev[(r, lvl)] = (np.asarray(ts), np.asarray(dirs))
            k += 1
    stop = "t_max"
    if len(sol.t_events[-2]):
        stop = "converged"
    if len(sol.t_events[-1]):
        stop = "wall"
    return sol, ev, stop, time.perf_counter() - wall0


def final_rise(ev, r):
    """Event = last upward crossing of 1/2; width = first 0.9 crossing after it minus the last
    upward 0.1 crossing before it (final rise, diagnostics.md)."""
    t5, d5 = ev[(r, 0.5)]
    up = t5[d5 > 0]
    if up.size == 0:
        return np.nan, np.nan
    te = up[-1]
    t9, d9 = ev[(r, 0.9)]
    t1, d1 = ev[(r, 0.1)]
    hi = t9[(d9 > 0) & (t9 >= te)]
    lo = t1[(d1 > 0) & (t1 <= te)]
    w = (hi[0] - lo[-1]) if hi.size and lo.size else np.nan
    return float(te), float(w)


def c_K_quad(K, s):
    """O(1) offset of the balanced-mode a = 1/2 crossing vs the correlation-flow blow-up (own quadrature)."""
    I, _ = quad(lambda x: x / (1 - x**K), 0, 2 ** (-1 / K), epsabs=1e-14, epsrel=1e-13)
    return s ** (-(2 - 2 / K)) * (I - 2 ** (1 - 2 / K) / (K - 2))


def corr_blowup_sphere(model: Model, theta_hat, tau_max=None, rtol=1e-11):
    """Blow-up time of theta' = grad <W, W*> from theta_hat, by the projected flow on the unit sphere:
    u' = grad g(u) - K g(u) u, Lambda' = K g(u), I' = exp(-(K-2) Lambda); T = rho0^-(K-2) (I(tau_end) +
    exp(-(K-2) Lambda_end) / ((K-2) K g(u_end))). Integrate until u has converged."""
    K = model.K
    rho0 = theta_hat.norm().item()
    u0 = (theta_hat / rho0).numpy()
    n = u0.size

    def g_and_grad(u):
        th = torch.from_numpy(u).requires_grad_(True)
        g = model.corr(th)
        (gr,) = torch.autograd.grad(g, th)
        return g.item(), gr.numpy()

    def rhs(tau, y):
        u = y[:n]
        g, gr = g_and_grad(u)
        du = gr - K * g * u
        return np.concatenate([du, [K * g, math.exp(-(K - 2) * y[n])]])

    y0 = np.concatenate([u0, [0.0, 0.0]])
    tau, y = 0.0, y0
    total_tau = 0.0
    last_T = None
    for chunk in range(200):
        sol = solve_ivp(rhs, (0, 50.0), y, method="DOP853", rtol=rtol, atol=1e-14)
        y = sol.y[:, -1]
        total_tau += 50.0
        u = y[:n]
        g, gr = g_and_grad(u)
        resid = np.linalg.norm(gr - K * g * u)
        T = (
            rho0 ** (-(K - 2)) * (y[n + 1] + math.exp(-(K - 2) * y[n]) / ((K - 2) * K * g))
            if g > 0
            else np.nan
        )
        if (
            last_T is not None
            and np.isfinite(T)
            and abs(T - last_T) < 1e-12 * abs(T)
            and resid < 1e-10
        ):
            break
        last_T = T
    ov = model.overlaps(torch.from_numpy(u)).numpy() * model.s.numpy()
    r1 = int(np.argmax(np.abs(ov)))
    return {
        "T": float(T),
        "r1": r1,
        "g_end": g,
        "g_end_over_gmax": g / (model.s[r1].item() * K ** (-K / 2)),
        "sphere_residual": float(resid),
        "tau": total_tau,
        "rho0": rho0,
    }


def t_kh(K, s1, rho0):
    return 1.0 / (K * (K - 2) * s1 * K ** (-K / 2) * rho0 ** (K - 2))


def main():
    mode, tree, chi, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    model = Model(tree, chi)
    K = model.K
    out = {"mode": mode, "tree": tree, "chi": chi, "K": K}
    if mode == "corr":
        rows = []
        for sd in [int(x) for x in seed.split(",")]:
            th = init(model, sd, 1.0)
            c = corr_blowup_sphere(model, th)
            c.update({"seed": sd, "t_KH": t_kh(K, max(WEIGHTS), c["rho0"])})
            c["T_over_KH"] = c["T"] / c["t_KH"]
            rows.append(c)
            print(json.dumps(c), flush=True)
        out["rows"] = rows
        path = sys.argv[5]
    elif mode == "first":
        sd = int(seed)
        sigmas = [float(x) for x in sys.argv[5].split(",")]
        path = sys.argv[6]
        th = init(model, sd, 1.0)
        c = corr_blowup_sphere(model, th)
        out["corr"] = c
        rows = []
        for sg in sigmas:
            t_pred = sg ** (-(K - 2)) * c["T"]
            sol, ev, stop, wall = gf(model, sg * th, 2.0 * t_pred + 50.0, wall=600)
            # first component to reach 0.5 for good (final rise)
            fr = {r: final_rise(ev, r) for r in range(model.R)}
            done = {r: v for r, v in fr.items() if np.isfinite(v[0])}
            if not done:
                rows.append({"sigma": sg, "stop": stop})
                continue
            r1 = min(done, key=lambda r: done[r][0])
            t1, w1 = done[r1]
            pred = t_pred + c_K_quad(K, WEIGHTS[r1])
            row = {
                "sigma": sg,
                "r1": r1,
                "t1": t1,
                "w1": w1,
                "t_pred_leading": t_pred,
                "ratio_with_offset": t1 / pred,
                "ratio_leading": t1 / t_pred,
                "stop": stop,
                "nfev": int(sol.nfev),
                "wall": wall,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        out["seed"] = sd
        out["rows"] = rows
        out["c_K_quad"] = [c_K_quad(K, s) for s in WEIGHTS]
    elif mode == "full":
        sd = int(seed)
        sg = float(sys.argv[5])
        path = sys.argv[6]
        kind = sys.argv[7] if len(sys.argv) > 7 else "fanin"
        th0 = init(model, sd, sg, kind)
        P3 = D ** (N * (K - 2) / (2 * K)) / ((K - 2) * min(WEIGHTS))
        t_max = 1e3 * P3 * sg ** (-(K - 2))
        sol, ev, stop, wall = gf(
            model, th0, t_max, wall=float(sys.argv[8]) if len(sys.argv) > 8 else 900
        )
        fr = {r: final_rise(ev, r) for r in range(model.R)}
        events = np.array([fr[r][0] for r in range(model.R)])
        widths = np.array([fr[r][1] for r in range(model.R)])
        order = [int(r) for r in np.argsort(events) if np.isfinite(events[r])]
        ts = [events[r] for r in order]
        rho = [
            (ts[i + 1] - ts[i]) / max(widths[order[i]], widths[order[i + 1]])
            for i in range(len(ts) - 1)
        ]
        plateaus = []
        for i in range(len(ts) - 1):
            grid = np.linspace(ts[i], ts[i + 1], 3001)[1:-1]
            speeds = np.array(
                [model.grad(torch.from_numpy(sol.sol(tt))).norm().item() for tt in grid]
            )
            j = int(np.argmin(speeds))
            # refine the minimum on a finer local grid
            lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
            g2 = np.linspace(lo, hi, 401)
            sp2 = np.array([model.grad(torch.from_numpy(sol.sol(tt))).norm().item() for tt in g2])
            tm = float(g2[int(np.argmin(sp2))])
            th = torch.from_numpy(sol.sol(tm))
            learned = order[: i + 1]
            level = 0.5 * sum(WEIGHTS[q] ** 2 for q in range(model.R) if q not in learned)
            L = model.loss(th).item()
            spec = model.spectra(th)
            want = sorted([WEIGHTS[q] for q in learned], reverse=True)
            spec_dev = max(
                max(abs(sv[k] - want[k]) / want[k] for k in range(len(want))) for sv in spec
            )
            nxt = max(sv[len(want)] for sv in spec) / min(WEIGHTS)
            eig = torch.linalg.eigvalsh(model.hess(th)).numpy()
            plateaus.append(
                {
                    "t_mid": tm,
                    "loss": L,
                    "level": level,
                    "rel_level_err": abs(L - level) / level,
                    "spectrum_dev": float(spec_dev),
                    "next_sv_over_smin": float(nxt),
                    "lam_min_raw": float(eig[0]),
                    "lam_max": float(eig[-1]),
                    "overlaps": model.overlaps(th).tolist(),
                }
            )
        # rank steps: number of singular values above f*s_r on each cut, before vs after each event
        rank_steps = []
        for j, r in enumerate(order):
            prev = [WEIGHTS[q] for q in order[:j]]
            f = next(
                (
                    fr_
                    for fr_ in (0.5, 0.75, 0.3, 0.85, 0.2)
                    if all(abs(fr_ * WEIGHTS[r] - x) >= 0.15 * WEIGHTS[r] for x in prev)
                ),
                0.5,
            )
            w = widths[r]
            before = [
                int((sv > f * WEIGHTS[r]).sum())
                for sv in model.spectra(torch.from_numpy(sol.sol(max(events[r] - 3 * w, 0))))
            ]
            after = [
                int((sv > f * WEIGHTS[r]).sum())
                for sv in model.spectra(
                    torch.from_numpy(sol.sol(min(events[r] + 3 * w, sol.t[-1])))
                )
            ]
            # time at which each cut first reaches its final count, on a fine grid over the rise window
            tg = np.linspace(max(events[r] - 3 * w, 0), min(events[r] + 3 * w, sol.t[-1]), 1201)
            counts = np.array(
                [
                    [
                        int((sv > f * WEIGHTS[r]).sum())
                        for sv in model.spectra(torch.from_numpy(sol.sol(tt)))
                    ]
                    for tt in tg
                ]
            )
            step_t = [
                float(tg[np.flatnonzero(counts[:, j] >= counts[-1, j])[0]])
                for j in range(counts.shape[1])
            ]
            spread = (max(step_t) - min(step_t)) / w if np.isfinite(w) and w > 0 else np.nan
            rank_steps.append(
                {
                    "component": r,
                    "before": before,
                    "after": after,
                    "f": f,
                    "step_times": step_t,
                    "spread_over_width": spread,
                    "grid_dt_over_width": float((tg[1] - tg[0]) / w),
                }
            )
        # dormant slices of never-learned components after the first event
        never = [r for r in range(model.R) if r not in order]
        slice_end = model.slices(torch.from_numpy(sol.y[:, -1]))
        min_never_slice = float(np.nanmin(slice_end[:, never])) / sg if never else np.nan
        # decay of the smallest never-learned slice over the plateau after the first escape
        decay = []
        if never and order:
            tgrid = np.geomspace(ts[-1] * 1.5, sol.t[-1], 12)
            for tt in tgrid:
                sl = model.slices(torch.from_numpy(sol.sol(tt)))
                decay.append([float(tt), float(np.nanmin(sl[:, never]))])
        out.update(
            {
                "seed": sd,
                "sigma": sg,
                "init": kind,
                "stop": stop,
                "t_end": float(sol.t[-1]),
                "wall": wall,
                "events": events.tolist(),
                "widths": widths.tolist(),
                "order": order,
                "sep_ratios": rho,
                "s2s": bool(len(order) >= 2 and all(x > 3 for x in rho)),
                "plateaus": plateaus,
                "rank_steps": rank_steps,
                "never_learned": never,
                "min_never_slice_over_sigma_end": min_never_slice,
                "never_slice_decay": decay,
                "loss_end": model.loss(torch.from_numpy(sol.y[:, -1])).item(),
                "nfev": int(sol.nfev),
            }
        )
        print(
            json.dumps(
                {
                    k: out[k]
                    for k in (
                        "seed",
                        "sigma",
                        "init",
                        "stop",
                        "events",
                        "order",
                        "s2s",
                        "min_never_slice_over_sigma_end",
                        "wall",
                    )
                }
            ),
            flush=True,
        )
    with open(path, "w") as f:
        json.dump(out, f, default=float)


if __name__ == "__main__":
    main()
