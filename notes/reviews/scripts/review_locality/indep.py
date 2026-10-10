"""Independent re-implementation (reviewer): dense TTN contraction, loss, gradient flow, learning events and
gauge-projected Hessian, written from the model definition in notes/model.md without ttn.network /
ttn.losses / ttn.dynamics / ttn.diagnostics or the agent's runner. Only the teacher's site vectors are
taken from ttn.teachers.pattern_teacher (as numbers); the fan-in init is regenerated here with the same
documented RNG convention and checked against TTNSpec.init.

usage: python indep.py flow <kind> <n> <S as 12> <start> <sigma> <seed>     independent gradient flow
       python indep.py hess <record-jsonl> <filter-json>                      Hessian at recorded midpoints
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality")


# ---------------- trees (own construction, compared with the library's below) ----------------
def own_tree(kind, n):
    """(parent, sites) per node, same node numbering as the library builders (checked)."""
    if kind == "mps":
        return [k + 1 for k in range(n - 1)] + [-1], [(k,) for k in range(n)]
    parent, sites = [], []
    if kind == "binary":

        def build(group):
            if len(group) <= 2:
                parent.append(-1)
                sites.append(tuple(group))
                return len(parent) - 1
            half = (len(group) + 1) // 2
            own, kids = [], []
            for part in (group[:half], group[half:]):
                if len(part) == 1:
                    own.append(part[0])
                else:
                    kids.append(build(part))
            parent.append(-1)
            sites.append(tuple(own))
            v = len(parent) - 1
            for c in kids:
                parent[c] = v
            return v
    elif kind == "ht":

        def build(group):
            if len(group) == 1:
                parent.append(-1)
                sites.append((group[0],))
                return len(parent) - 1
            half = (len(group) + 1) // 2
            l, r = build(group[:half]), build(group[half:])
            parent.append(-1)
            sites.append(())
            v = len(parent) - 1
            parent[l] = parent[r] = v
            return v

    build(list(range(n)))
    return parent, sites


class Net:
    def __init__(self, kind, n, d=2, chi=2):
        self.parent, self.sites = own_tree(kind, n)
        self.K, self.N, self.d = len(self.parent), n, d
        self.children = [[c for c in range(self.K) if self.parent[c] == v] for v in range(self.K)]
        self.root = self.parent.index(-1)
        below = [None] * self.K

        def sub(v):
            s = list(self.sites[v])
            for c in self.children[v]:
                s += sub(c)
            below[v] = s
            return s

        sub(self.root)
        self.bond = [
            1 if v == self.root else min(chi, d ** len(below[v]), d ** (n - len(below[v])))
            for v in range(self.K)
        ]
        self.shapes = [
            tuple(
                [d] * len(self.sites[v]) + [self.bond[c] for c in self.children[v]] + [self.bond[v]]
            )
            for v in range(self.K)
        ]
        self.sizes = [math.prod(s) for s in self.shapes]
        self.P = sum(self.sizes)

    def unflat(self, x):
        out, i = [], 0
        for sh, sz in zip(self.shapes, self.sizes):
            out.append(x[i : i + sz].reshape(sh))
            i += sz
        return out

    def dense(self, cores):
        def contract(v):
            T = cores[v]
            labels = (
                [("s", s) for s in self.sites[v]]
                + [("c", c) for c in self.children[v]]
                + [("p", v)]
            )
            for c in self.children[v]:
                Tc, lc = contract(c)
                ax = labels.index(("c", c))
                T = torch.tensordot(T, Tc, dims=([ax], [len(lc) - 1]))
                labels = labels[:ax] + labels[ax + 1 :] + lc[:-1]
            pa = labels.index(("p", v))
            T = T.movedim(pa, -1)
            labels = labels[:pa] + labels[pa + 1 :] + [("p", v)]
            return T, labels

        T, lab = contract(self.root)
        T = T[..., 0]
        order = [lab.index(("s", i)) for i in range(self.N)]
        return T.permute(order)

    def init(self, sigma, seed):
        g = torch.Generator().manual_seed(seed)
        cores = []
        for sh in self.shapes:
            z = torch.randn(sh, generator=g, dtype=torch.float64)
            cores.append(sigma / math.sqrt(math.prod(sh[:-1])) * z)
        return cores

    def saddle(self, u, amp):
        """Exact balanced saddle amp^K U1: every core = amp * (x) u at its sites (x) e0 on every bond leg."""
        cores = []
        for v in range(self.K):
            g = torch.tensor(amp, dtype=torch.float64)
            vecs = [u[s] for s in self.sites[v]]
            for c in self.children[v]:
                e = torch.zeros(self.bond[c], dtype=torch.float64)
                e[0] = 1
                vecs.append(e)
            e = torch.zeros(self.bond[v], dtype=torch.float64)
            e[0] = 1
            vecs.append(e)
            for vec in vecs:
                g = g.unsqueeze(-1) * vec
            cores.append(g)
        return cores

    def gauge_basis(self, x):
        """Columns: tangent vectors of G_v -> G_v (1 + eE) on v's parent leg, G_p -> (1 - eE) on p's v-leg."""
        cores = self.unflat(x)
        offs = np.cumsum([0] + self.sizes)
        cols = []
        for v in range(self.K):
            p = self.parent[v]
            if p < 0:
                continue
            chi = self.bond[v]
            leg = len(self.sites[p]) + self.children[p].index(v)
            for a in range(chi):
                for b in range(chi):
                    E = torch.zeros(chi, chi, dtype=torch.float64)
                    E[a, b] = 1
                    col = torch.zeros(self.P, dtype=torch.float64)
                    dv = torch.tensordot(cores[v], E, dims=([cores[v].ndim - 1], [0]))  # G_v E
                    dp = torch.movedim(
                        torch.tensordot(-E, cores[p], dims=([1], [leg])), 0, leg
                    )  # -E on v-leg
                    col[offs[v] : offs[v + 1]] = dv.reshape(-1)
                    col[offs[p] : offs[p + 1]] += dp.reshape(-1)
                    cols.append(col)
        return torch.stack(cols, 1)


def teacher(n, S, weights, d=2, seed=1000, patterns=None):
    from ttn.teachers import pattern_teacher  # site vectors only (numbers)

    if patterns is None:
        patterns = [[0] * n, [1 if i in S else 0 for i in range(n)]]
    tch = pattern_teacher(weights, patterns, d=d, seed=seed)
    F = tch.factors.clone()  # (R, N, d)
    U = []
    for r in range(F.shape[0]):
        t = torch.tensor(1.0, dtype=torch.float64)
        for i in range(n):
            t = t.unsqueeze(-1) * F[r, i]
        U.append(t)
    Wstar = sum(w * u for w, u in zip(weights, U))
    return F, U, Wstar


def make_problem(net, U, Wstar, weights):
    def loss(x):
        W = net.dense(net.unflat(x))
        return 0.5 * ((W - Wstar) ** 2).sum()

    gradf = torch.func.grad(loss)
    hessf = torch.func.hessian(loss)

    def overl(x):
        W = net.dense(net.unflat(x))
        return np.array([float((W * u).sum()) / w for u, w in zip(U, weights)])

    return loss, gradf, hessf, overl


def flow(
    kind,
    n,
    S,
    start,
    sigma,
    seed,
    weights=(1.0, 0.5),
    rtol=1e-10,
    atol=1e-14,
    method="Radau",
    patterns=None,
    t_max=1e14,
):
    net = Net(kind, n)
    F, U, Wstar = teacher(n, S, list(weights), patterns=patterns)
    loss, gradf, hessf, overl = make_problem(net, U, Wstar, list(weights))
    cores = net.init(sigma, seed)
    # check own init == library init
    from rules import get_tree
    from ttn.network import TTNSpec

    spec = TTNSpec.uniform(get_tree(kind, n), 2, 2)
    lib = spec.init(sigma, seed=seed)
    assert all(torch.equal(a, b) for a, b in zip(cores, lib)), "own init differs from TTNSpec.init"
    assert tuple(net.shapes) == tuple(spec.shapes)
    if start == "saddle":
        sad = net.saddle(F[0], weights[0] ** (1 / net.K))
        cores = [a + b for a, b in zip(sad, cores)]
    x0 = torch.cat([c.reshape(-1) for c in cores]).numpy()
    R = len(weights)
    evs = []
    for r in range(R):
        e = (lambda r: lambda t, y: overl(torch.from_numpy(y))[r] - 0.5)(r)
        evs.append(e)
    done = lambda t, y: overl(torch.from_numpy(y)).min() - 0.97  # noqa: E731
    done.terminal = True
    evs.append(done)
    t0 = time.time()
    sol = solve_ivp(
        lambda t, y: -gradf(torch.from_numpy(y)).numpy(),
        (0, t_max),
        x0,
        method=method,
        jac=lambda t, y: -hessf(torch.from_numpy(y)).numpy(),
        rtol=rtol,
        atol=atol,
        events=evs,
    )
    cross = [list(map(float, te)) for te in sol.t_events[:R]]
    # final upward crossing of 1/2 per component (odd number of crossings), as in the runner's rule
    a0 = overl(torch.from_numpy(x0))
    ev = sorted(
        [(0.0, r) for r in range(R) if a0[r] > 0.5]
        + [(c[-1], r) for r, c in enumerate(cross) if len(c) % 2 == 1]
    )
    return dict(
        kind=kind,
        n=n,
        S=S,
        start=start,
        sigma=sigma,
        seed=seed,
        method=method,
        rtol=rtol,
        status=int(sol.status),
        events=ev,
        crossings=cross,
        nfev=int(sol.nfev),
        wall=time.time() - t0,
        a_final=overl(torch.from_numpy(sol.y[:, -1])).tolist(),
    )


def hess_at(rec, t_target):
    """Own gauge-projected Hessian spectrum at the stored trajectory point closest to t_target."""
    S = rec["S"]
    net = Net(rec["kind"], rec["n"])
    F, U, Wstar = teacher(net.N, S, rec["weights"], patterns=rec.get("patterns"))
    loss, gradf, hessf, overl = make_problem(net, U, Wstar, rec["weights"])
    z = np.load(rec["traj"])
    j = int(np.argmin(np.abs(z["t"] - t_target)))
    x = torch.from_numpy(z["y"][:, j].copy())
    H = hessf(x)
    G = net.gauge_basis(x)
    # sanity: gauge directions leave W unchanged to first order (directional derivative of W)
    _, jvp = torch.func.jvp(lambda y: net.dense(net.unflat(y)), (x,), (G[:, 0] / G[:, 0].norm(),))
    u, s, _ = torch.linalg.svd(G, full_matrices=True)
    rank = int((s > 1e-10 * s.max()).sum())
    C = u[:, rank:]
    h = C.T @ H @ C
    ev = torch.linalg.eigvalsh(0.5 * (h + h.T))
    full = torch.linalg.eigvalsh(H)
    # same projection, keeping only well-conditioned gauge directions (singular value > 1e-3 x max):
    # near an aligned saddle some generators (on unused bond indices) give O(sigma) columns
    rank3 = int((s > 1e-3 * s.max()).sum())
    C3 = u[:, rank3:]
    h3 = C3.T @ H @ C3
    ev3 = torch.linalg.eigvalsh(0.5 * (h3 + h3.T))
    # same point moved along its gauge orbit to the balanced representative (agent's run.balance_gauge,
    # W unchanged), then the same own projection: the flat-metric projection is gauge dependent
    sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_locality/code")
    from run import balance_gauge
    from rules import get_tree
    from ttn.network import TTNSpec

    spec = TTNSpec.uniform(get_tree(rec["kind"], rec["n"]), 2, 2)
    assert tuple(spec.shapes) == tuple(net.shapes)
    try:
        xb = torch.cat([c.reshape(-1) for c in balance_gauge(spec, [c.clone() for c in net.unflat(x)])])
        dW = float((net.dense(net.unflat(xb)) - net.dense(net.unflat(x))).abs().max())
        Hb = hessf(xb)
        ub, sb, _ = torch.linalg.svd(net.gauge_basis(xb), full_matrices=True)
        Cb = ub[:, int((sb > 1e-10 * sb.max()).sum()) :]
        evb = torch.linalg.eigvalsh(0.5 * (Cb.T @ Hb @ Cb + (Cb.T @ Hb @ Cb).T))
    except Exception:  # balance_gauge overflows on very ill-conditioned bond Grams
        dW, evb = float("nan"), torch.tensor([float("nan")])
    return dict(
        lam_min_balanced=float(evb[0]),
        balance_dW=dW,
        lam_min_cut3=float(ev3[0]),
        n_gauge_cut3=rank3,
        gauge_sv_min=float(s[rank - 1] / s.max()),
        t=float(z["t"][j]),
        dt_rel=abs(float(z["t"][j]) - t_target) / max(abs(t_target), 1e-300),
        lam_min=float(ev[0]),
        lam_2=float(ev[1]),
        lam_max=float(ev[-1]),
        n_gauge=rank,
        gauge_jvp=float(jvp.abs().max()),
        lam_min_unprojected=float(full[0]),
        loss=float(loss(x)),
        a=overl(x).tolist(),
    )


if __name__ == "__main__":
    if sys.argv[1] == "flow":
        kind, n, S, start, sigma, seed = (
            sys.argv[2],
            int(sys.argv[3]),
            [int(c) for c in sys.argv[4]],
            sys.argv[5],
            float(sys.argv[6]),
            int(sys.argv[7]),
        )
        print(json.dumps(flow(kind, n, S, start, sigma, seed)))
    elif sys.argv[1] == "hess":
        flt = json.loads(open(sys.argv[3][1:]).read() if sys.argv[3].startswith("@") else sys.argv[3])
        for line in open(sys.argv[2]):
            rec = json.loads(line)
            if any(rec.get(k) != v for k, v in flt.items()) or rec.get("status") != 1:
                continue
            for p in rec["plateaus"]:
                if "t_mid" not in p:
                    continue
                h = hess_at(rec, p["t_mid"])
                print(
                    json.dumps(
                        dict(
                            kind=rec["kind"],
                            n=rec["n"],
                            S=rec["S"],
                            start=rec["start"],
                            sigma=rec["sigma"],
                            seed=rec["seed"],
                            hess_min_runner=p["hess_min"],
                            **h,
                        )
                    )
                )
