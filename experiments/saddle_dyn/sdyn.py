"""Gradient flow near a learned rank-one saddle theta_S, with a cancellation-free gradient.

Contraction and topology: the verifier's independent `vttn.Net` (core legs: own sites, child bonds,
parent bond). Teachers are pattern teachers in the standard site basis with the learned component
q = 0...0 (odeco teachers are relabelled so that the learned component is e_0 at every site).

Cancellation-free loss. With reference cores S (rank one, W(S) = c e_q), R' = W* - W(S) is computed
once and Delta = W(G) - W(S) by the exact telescoping sum
    Delta = sum_j W(S_1..S_{j-1}, G_j - S_j, G_{j+1}..G_K),
so E = W - W* = Delta - R' has no O(1) cancellation at the q entry. Near S every term is small.
Far from S (runs from the origin) the plain E = W - W* is used instead (`ref=None`).
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import LSODA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory_checks" / "saddle_order_verify"))
from vttn import Net  # noqa: E402

torch.set_num_threads(1)
DT = torch.float64
U = 2.0**-53
OUT = Path("/root/outputs/saddle_dyn")


TRAP_DL = 1e-30  # trap rule: at 10 log-spaced checks spanning a decade with t > 1e6, either
TRAP_R = 1e-45  # 0 < L - L_S < TRAP_DL, or every residual amplitude |Delta[r]| < TRAP_R (dead routes)


def pattern(s):
    return tuple(int(c) for c in s)


class Model:
    """Pattern teacher s_q e_q + sum_r s_r e_r on a vttn.Net; theta_S balanced at q = 0...0."""

    def __init__(self, tree, n, d, chi, s_q, residuals):
        self.net = net = Net(tree, n, d=d, chi=chi)
        self.K, self.P, self.n = net.K, net.P, n
        self.s_q = float(s_q)
        self.residuals = [(pattern(p), float(w)) for p, w in residuals]
        self.q = (0,) * n
        Ws = torch.zeros((d,) * n, dtype=DT)
        for p, w in self.residuals:
            Ws[p] += w
        self.R = Ws.clone()  # residual (exact: zero at q)
        Ws[self.q] += self.s_q
        self.W_star = Ws
        self.L_S = 0.5 * float((self.R**2).sum())
        self.a = self.s_q ** (1.0 / self.K)
        theta = torch.zeros(self.P, dtype=DT)
        for v, c in enumerate(net.cores(theta)):
            c[(0,) * c.dim()] = self.a
        self.theta_S = theta
        core, old, _ = net.entry_info()
        self.old_mask = torch.tensor(old)
        self.new_mask = ~self.old_mask
        self.split_patterns = self._split_patterns()
        # einsum with E as an extra operand, for gradients with absolute values (floor bound)
        ins, out = net.eq.split("->")
        self.eq_env = ins + "," + out[:-1]  # E has no root-bond leg

    # ---------------------------------------------------------------- structure
    def k_route(self, p):
        sc = self.net.site_core()
        T = sorted({sc[i] for i, x in enumerate(p) if x != self.q[i]})
        return T, len(self.net.steiner(T))

    @property
    def k_star(self):
        return min(self.k_route(p)[1] for p, _ in self.residuals)

    def _deg(self, C):
        return 1 if len(C) == 1 else len(self.net.steiner(C))

    def _split_patterns(self):
        """Multi-core partial patterns r|C of the low splits T_r = C1 + C2 (deg C1, deg C2 < k*/2)."""
        ks = self.k_star
        sc = self.net.site_core()
        out = set()
        for p, _ in self.residuals:
            T, _ = self.k_route(p)
            for m in range(1, len(T)):
                for C1 in itertools.combinations(T, m):
                    C2 = [v for v in T if v not in C1]
                    if self._deg(C1) < ks / 2 and self._deg(C2) < ks / 2:
                        for C in (C1, C2):
                            if self._deg(C) >= 2:
                                out.add(tuple(p[i] if sc[i] in C else self.q[i] for i in range(self.n)))
        return sorted(out)

    # ---------------------------------------------------------------- tensors
    def W(self, theta):
        return self.net.dense(theta)

    def set_ref(self, ref_theta):
        """Reference cores S (or None: plain loss). Returns the constant L_ref - L_S."""
        if ref_theta is None:
            self.ref = None
            self.Rp = None
            self.c0 = 0.0
            return
        self.ref = ref_theta.clone()
        self.ref_cores = self.net.cores(self.ref)
        WS = self.W(self.ref)
        self.Rp = self.W_star - WS
        c = float(WS[self.q])
        # 1/2 |R'|^2 - L_S: R' and R differ only at q (W(S) is rank one at q); computed exactly
        assert float((WS.abs().sum() - abs(c))) == 0.0
        self.c0 = 0.5 * (self.s_q - c) ** 2

    def delta(self, theta):
        """Delta = W(G) - W(S) by the telescoping sum (no cancellation near S)."""
        G = self.net.cores(theta)
        S = self.ref_cores
        out = torch.zeros_like(self.W_star)
        for j in range(self.K):
            ops = list(S[:j]) + [G[j] - S[j]] + list(G[j + 1 :])
            out = out + torch.einsum(self.net.eq, *ops)[..., 0]
        return out

    def delta_abs(self, theta):
        G = [g.abs() for g in self.net.cores(theta)]
        S = [s.abs() for s in self.ref_cores]
        Gr = self.net.cores(theta)
        out = torch.zeros_like(self.W_star)
        for j in range(self.K):
            ops = list(S[:j]) + [(Gr[j] - self.ref_cores[j]).abs()] + list(G[j + 1 :])
            out = out + torch.einsum(self.net.eq, *ops)[..., 0]
        return out

    def residual(self, theta):
        """E = W - W*, and L - L_S."""
        if self.ref is None:
            E = self.W(theta) - self.W_star
            return E, 0.5 * float((E * E).sum()) - self.L_S
        D = self.delta(theta)
        dL = 0.5 * float((D * D).sum()) - float((self.Rp * D).sum()) + self.c0
        return D - self.Rp, dL

    def grad_from_E(self, theta, E):
        th = theta.clone().requires_grad_(True)
        val = (E * self.W(th)).sum()
        (g,) = torch.autograd.grad(val, th)
        return g

    def grad(self, theta):
        E, dL = self.residual(theta)
        return self.grad_from_E(theta, E), dL, E

    def grad_err_bound(self, theta, E):
        """Upper bound (10 u x sum of |terms|) on the roundoff in the gradient."""
        A = E.abs()
        if self.ref is not None:
            A = A + self.delta_abs(theta) + self.Rp.abs()
        else:
            A = A + self.W_star.abs()
        th = theta.abs().clone().requires_grad_(True)
        val = (A * self.W(th)).sum()
        (g,) = torch.autograd.grad(val, th)
        return 10 * U * g

    def hessian(self, theta):
        def f(th):
            E = self.W(th) - self.W_star
            return 0.5 * (E * E).sum()

        return torch.func.hessian(f)(theta)

    def conserved(self, theta):
        """Bond conserved quantities Q_c = G_c^T G_c - G_p,(c) G_p,(c)^T (all edges, stacked)."""
        net = self.net
        G = net.cores(theta)
        out = []
        for c in range(self.K):
            p = net.parent[c]
            if p < 0:
                continue
            gc = G[c].reshape(-1, net.bond[c])
            leg = net.n_phys[p] + net.children[p].index(c)
            gp = torch.movedim(G[p], leg, 0).reshape(net.bond[c], -1)
            out.append((gc.T @ gc - gp @ gp.T).reshape(-1))
        return torch.cat(out)

    # ---------------------------------------------------------------- gauge
    def regauge(self, theta):
        """Exact orthogonal gauge map sending each bond's q-message to |x| e_0 (Householder)."""
        net = self.net
        G = [c.clone() for c in net.cores(theta.clone())]
        msgs = {}

        def up(c):
            """q-message up edge c: the subtree of c contracted with e_0 on its sites."""
            if c in msgs:
                return msgs[c]
            g = G[c]
            for _ in range(net.n_phys[c]):
                g = g[0]
            for ch in net.children[c]:
                g = torch.tensordot(up(ch), g, dims=([0], [0]))
            msgs[c] = g  # vector over the parent bond of c
            return g

        for c in range(self.K):
            if net.parent[c] >= 0:
                up(c)
        for c in range(self.K):
            p = net.parent[c]
            if p < 0:
                continue
            x = msgs[c]
            nx = x.norm()
            e = torch.zeros_like(x)
            e[0] = 1.0
            sgn = 1.0 if x[0] >= 0 else -1.0
            w = x + sgn * nx * e
            H = torch.eye(len(x), dtype=DT) - 2 * torch.outer(w, w) / (w @ w)  # H x = -sgn |x| e_0
            if sgn > 0:
                H = -H  # now H x = +|x| e_0; H symmetric orthogonal
            G[c] = torch.tensordot(G[c], H, dims=([G[c].dim() - 1], [0]))  # G_c H, parent gets H G_p
            leg = net.n_phys[p] + net.children[p].index(c)
            G[p] = torch.movedim(torch.tensordot(H, G[p], dims=([1], [leg])), 0, leg)
        return torch.cat([g.reshape(-1) for g in G])

    def old_part(self, theta):
        S = torch.zeros_like(theta)
        for v, (c, s) in enumerate(zip(self.net.cores(theta), self.net.cores(S))):
            s[(0,) * c.dim()] = c[(0,) * c.dim()]
        return S


def amplitudes(model, W):
    out = {"q": float(W[model.q])}
    for p, w in model.residuals:
        out["r" + "".join(map(str, p))] = float(W[p])
    for p in model.split_patterns:
        out["p" + "".join(map(str, p))] = float(W[p])
    rest = W.clone()
    rest[model.q] = 0
    for p, _ in model.residuals:
        rest[p] = 0
    flat = rest.abs().reshape(-1)
    top = torch.topk(flat, 4)
    out["top_other"] = [
        ("".join(str(int(x)) for x in np.unravel_index(int(i), W.shape)), float(rest.reshape(-1)[i])) for i in top.indices
    ]
    return out


def escape_type(model, amps):
    rmax = max(abs(amps["r" + "".join(map(str, p))]) for p, _ in model.residuals)
    pmax = max([abs(amps["p" + "".join(map(str, p))]) for p in model.split_patterns], default=0.0)
    ratio = pmax / rmax if rmax > 0 else math.inf
    kind = "split" if ratio >= 5 else ("route" if ratio <= 1 else "mixed")
    return kind, ratio


THRESH = (1e-4, 1e-3, 1e-2)


class Event:
    """g(t, y) = 0 crossing on the step values; direction +1 (up), -1 (down), 0 (both)."""

    def __init__(self, name, g, direction=0, terminal=False):
        self.name, self.g, self.direction, self.terminal = name, g, direction, terminal

    def crossed(self, ga, gb):
        up = ga < 0 <= gb
        down = ga > 0 >= gb
        return up if self.direction > 0 else (down if self.direction < 0 else (up or down))


class Result:
    def __init__(self):
        self.t, self.y = [], []
        self.events = {}  # name -> list of (t, y)
        self.status = "running"

    def event_t(self, name):
        ev = self.events.get(name, [])
        return ev[0][0] if ev else None

    def event_y(self, name):
        ev = self.events.get(name, [])
        return ev[0][1] if ev else None


def advance(fun, jac, t0, y0, t1, rtol, atol):
    """Integrate exactly from t0 to t1 with a fresh LSODA (lands on t1: scipy passes it as tcrit).
    The ODE is autonomous, so the clock is re-zeroed (t0 -> 0) to keep full time resolution."""
    h = t1 - t0
    if h <= 0:
        return np.array(y0, copy=True)
    if h <= 1e-12 * max(abs(t0), 1.0):  # below any meaningful resolution: one Euler step
        return np.asarray(y0) + h * fun(0.0, y0)
    s = LSODA(fun, 0.0, y0, h, rtol=rtol, atol=atol, jac=jac)
    while s.status == "running":
        s.step()
    if s.status != "finished":
        raise RuntimeError("LSODA failed in advance")
    return np.array(s.y, copy=True)


def refine(ev, fun, jac, ta, ya, ga, tb, gb, rtol, atol, rel=1e-9, max_iter=40):
    """Root of ev.g in (ta, tb] by bisection on exact re-integrations from (ta, ya); the root time is
    resolved to rel * tb and the state is integrated to it (never interpolated)."""
    for _ in range(max_iter):
        if tb - ta <= rel * abs(tb):
            break
        tm = 0.5 * (ta + tb)
        ym = advance(fun, jac, ta, ya, tm, rtol, atol)
        gm = ev.g(tm, ym)
        if ev.crossed(ga, gm):
            tb, gb = tm, gm
        else:
            ta, ya, ga = tm, ym, gm
    tr = ta + (tb - ta) * (ga / (ga - gb)) if ga != gb else tb
    return tr, advance(fun, jac, ta, ya, tr, rtol, atol)


def integrate(model, y0, t_max, rtol, atol, events_extra=(), wall=3600.0, stop_after=None, trap=True):
    """LSODA (exact Hessian as Jacobian) driven step by step; events on the step values (scipy's LSODA
    dense output is not reliable here), refined by re-integration. Escape events "esc_<thr>" at
    L - L_S = -thr. Stops at t_max, a terminal event, the wall clock, or the trap rule."""
    t_wall = time.time()
    cache = {}

    def evalf(y):
        key = y.tobytes()
        if key not in cache:
            if len(cache) > 16:
                cache.clear()
            g, dL, _ = model.grad(torch.from_numpy(y))
            cache[key] = (g.numpy(), dL)
        return cache[key]

    nfev = [0]

    def rhs(t, y):
        nfev[0] += 1
        return -evalf(y)[0]

    def jac(t, y):
        return -model.hessian(torch.from_numpy(y)).numpy()

    evs = [Event(f"esc_{thr:g}", (lambda t, y, thr=thr: evalf(y)[1] + thr), -1, thr == stop_after) for thr in THRESH]
    evs += list(events_extra)
    res = Result()
    y0 = np.array(y0, dtype=np.float64)
    res.t.append(0.0)
    res.y.append(y0.copy())
    g_old = [ev.g(0.0, y0) for ev in evs]
    # The ODE is autonomous: the solver runs on a local clock (absolute t = off + local t) that is
    # re-zeroed when its step falls below 1e-7 of the local time, so that fast transitions at late
    # absolute times (escapes at t ~ 1e13-1e20, where float64 spacing in t is O(0.01-1e4)) stay resolved.
    off, t_loc_old, n_restart = 0.0, 0.0, 0
    solver = LSODA(rhs, 0.0, y0, t_max, rtol=rtol, atol=atol, jac=jac)
    nxt, count = 1e6, 0
    while True:
        if time.time() - t_wall > wall:
            res.status = "wall"
            break
        msg = solver.step()
        if solver.status == "failed":
            res.status = "failed: " + str(msg)
            break
        t_loc, y = solver.t, np.array(solver.y, copy=True)
        t = off + t_loc
        g_new = [ev.g(t, y) for ev in evs]
        hits = []
        for k, ev in enumerate(evs):
            if ev.crossed(g_old[k], g_new[k]):
                tr, yr = refine(ev, rhs, jac, t_loc_old, res.y[-1], g_old[k], t_loc, g_new[k], rtol, atol)
                hits.append((off + tr, k, yr))
        hits.sort(key=lambda h: h[0])
        stop = None
        for tr, k, yr in hits:
            res.events.setdefault(evs[k].name, []).append((tr, yr))
            if evs[k].terminal:
                stop = (tr, yr)
                break
        if stop is not None:
            res.t.append(stop[0])
            res.y.append(stop[1])
            res.status = "event"
            break
        res.t.append(t)
        res.y.append(y)
        g_old = g_new
        t_loc_old = t_loc
        if solver.status == "finished":
            res.status = "t_max"
            break
        if t_loc > 0 and solver.step_size < 1e-7 * t_loc:
            off += t_loc
            t_loc_old = 0.0
            n_restart += 1
            solver = LSODA(rhs, 0.0, y, t_max - off, rtol=rtol, atol=atol, jac=jac)
        if trap and t >= nxt:
            nxt = t * 10**0.1
            dL = evalf(y)[1]
            dead = False
            if model.ref is not None:
                D = model.delta(torch.from_numpy(y))
                dead = max(abs(float(D[p])) for p, _ in model.residuals) < TRAP_R
            count = count + 1 if (0 < dL < TRAP_DL or dead) else 0
            if count >= 10:
                res.status = "trapped"
                break
    res.t = np.array(res.t)
    res.y = np.array(res.y).T
    return res, dict(status=res.status, nfev=nfev[0], n_steps=len(res.t) - 1, n_restart=n_restart,
                     secs=time.time() - t_wall)
