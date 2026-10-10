"""Saddle hopper: predict the saddle-to-saddle (S2S) sequence of gradient flow on a TTN.

The rules are fixed in notes/saddle_hopper.md section 1 (rule numbers R0-R8 below refer to it). They
come from notes/theory_saddles.md (routes, local residual R_C, k*, strictness, head start, piggyback
block, stars, rule 5.1, traps), theory_escape.md (first escape, attractors of the spherical flow,
race vs greedy), theory_teachers.md (generic teachers, accumulated-utility race) and
theory_covariance.md (product covariance: landscape of Sigma^1/2 W*, escape driven by Sigma W*).

Everything is computed on dense tensors (N <= 8 sites), so one code path covers pattern, CP and TTN
teachers and product covariances. A saddle is a balanced critical point of the loss restricted to a
bond-rank profile; the Hessian of the true loss at it gives the gradient-flow rates (in the original
coordinates, so a covariance needs no separate preconditioner).

Main entry: `SaddleHopper(setting).run()` -> first-escape candidates and the predicted sequences
(branches where the theory says the order is random or rare).
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize

from ttn.network import TTNSpec, apply_on_leg, to_dense
from ttn.tree import Tree

DT = torch.float64
LETTERS = string.ascii_letters


# ---------------------------------------------------------------- setting and dense helpers


@dataclass
class Setting:
    """One problem: student tree and bond dimension, teacher, input covariance, init protocol.

    w_star: dense teacher (axes = sites 0..N-1). metrics: per-site covariances Sigma_i (None =
    whitened). components: optional labels [(s_r, U_r)] with U_r a dense unit-norm rank-one tensor,
    used only to name the events (overlap a_r = <W, Sigma U_r> / <W*, Sigma U_r>).
    init: 'fanin' | 'balanced' | 'saddle' (exact first saddle + noise: no first escape, no head start).
    """

    tree: Tree
    d: int
    chi: int
    w_star: torch.Tensor
    metrics: list | None = None
    components: list | None = None
    init: str = "fanin"
    name: str = ""
    # init='saddle': labelled component to start from (None: the lowest rank-one attractor)
    start_component: int | None = None


def apply_metric(x: torch.Tensor, metrics) -> torch.Tensor:
    """(Sigma_1 (x) ... (x) Sigma_N) x for a dense tensor (metrics None = identity)."""
    if metrics is None:
        return x
    for i, m in enumerate(metrics):
        x = torch.movedim(torch.tensordot(m, x, dims=([1], [i])), 0, i)
    return x


class RankDeficient(Exception):
    """A bond direction is unused (singular Gram matrix): the TTN cannot be balanced at this rank."""


def sym_pow(m: torch.Tensor, a: float) -> torch.Tensor:
    w, u = torch.linalg.eigh(0.5 * (m + m.T))
    return (u * w.clamp_min(1e-300) ** a) @ u.T


def balance(spec: TTNSpec, cores, sweeps: int = 2000, tol: float = 1e-14):
    """Gauge-balance a TTN with full-rank bonds (every Q_e = 0, W unchanged): theory_saddles A3.

    Same update as experiments/locality/run.py: on edge v (parent p), P = G_v^T G_v, Q = G_p,(v) G_p,(v)^T,
    X = P^-1/2 (P^1/2 Q P^1/2)^1/2 P^-1/2 (X P X = Q), A = X^1/2: G_v -> G_v A, G_p -> A^-1 G_p."""
    t = spec.tree
    cores = [c.detach().clone() for c in cores]
    for _ in range(sweeps):
        worst = 0.0
        for v in t.edges:
            p, r = t.parent[v], spec.bond_dims[v]
            leg = len(t.sites[p]) + t.children[p].index(v)
            gv = cores[v].reshape(-1, r)
            gp = torch.movedim(cores[p], leg, 0).reshape(r, -1)
            P, Q = gv.T @ gv, gp @ gp.T
            scale = max(float(P.abs().max()), float(Q.abs().max()), 1e-300)
            if (
                float(torch.linalg.eigvalsh(P)[0]) < 1e-13 * scale
                or float(torch.linalg.eigvalsh(Q)[0]) < 1e-13 * scale
            ):
                raise RankDeficient(v)
            worst = max(worst, float((P - Q).abs().max()) / scale)
            ph, pmh = sym_pow(P, 0.5), sym_pow(P, -0.5)
            X = pmh @ sym_pow(ph @ Q @ ph, 0.5) @ pmh
            A, Ainv = sym_pow(X, 0.5), sym_pow(X, -0.5)
            cores[v] = (gv @ A).reshape(cores[v].shape)
            cores[p] = apply_on_leg(cores[p], Ainv, leg)
        if worst < tol:
            break
    return cores


def connected_subsets(tree: Tree, max_size: int) -> list[frozenset]:
    """All connected node sets with 2 <= size <= max_size, sorted by size."""
    nbrs = {
        v: set(tree.children[v]) | ({tree.parent[v]} if tree.parent[v] >= 0 else set())
        for v in range(tree.n_nodes)
    }
    level = {frozenset([v]) for v in range(tree.n_nodes)}
    found = []
    for _ in range(max_size - 1):
        nxt = set()
        for c in level:
            for v in c:
                for u in nbrs[v] - c:
                    nxt.add(c | {u})
        found.extend(sorted(nxt, key=lambda s: sorted(s)))
        level = nxt
    return found


def hopm(
    T: torch.Tensor,
    n_starts: int = 8,
    starts=None,
    iters: int = 3000,
    tol: float = 1e-14,
    rng: np.random.Generator | None = None,
):
    """Local maxima of |<T, y_1 (x) ... (x) y_k>| over unit vectors (alternating power iteration).

    Returns a list of (value, [y_j]) with value > 0, one per start (not deduplicated)."""
    k = T.ndim
    rng = rng or np.random.default_rng(0)
    letters = LETTERS[:k]
    all_starts = list(starts or [])
    for _ in range(n_starts):
        all_starts.append([torch.from_numpy(rng.standard_normal(n)).to(DT) for n in T.shape])
    out = []
    for ys in all_starts:
        ys = [y / y.norm() for y in ys]
        val = 0.0
        for _ in range(iters):
            for j in range(k):
                eq = (
                    letters
                    + ","
                    + ",".join(letters[i] for i in range(k) if i != j)
                    + "->"
                    + letters[j]
                )
                z = torch.einsum(eq, T, *[ys[i] for i in range(k) if i != j])
                nz = z.norm()
                if nz == 0:
                    break
                ys[j] = z / nz
            new = float(torch.einsum(letters + "," + ",".join(letters) + "->", T, *ys))
            if abs(abs(new) - abs(val)) <= tol * max(abs(new), 1e-300):
                val = new
                break
            val = new
        if val < 0:
            ys[0] = -ys[0]
            val = -val
        out.append((val, ys))
    return out


def spectral_norm_grouped(T: torch.Tensor, n_starts: int = 8, rng=None):
    """Injective norm of T (legs already grouped per core) and its maximizer: SVD for 2 legs."""
    if T.ndim == 1:
        n = T.norm()
        return float(n), [T / n if n > 0 else T]
    if T.ndim == 2:
        u, s, vh = torch.linalg.svd(T, full_matrices=False)
        return float(s[0]), [u[:, 0], vh[0]]
    # HOSVD start (top left singular vector of each unfolding) plus random starts
    start = []
    for j in range(T.ndim):
        m = torch.movedim(T, j, 0).reshape(T.shape[j], -1)
        start.append(torch.linalg.svd(m, full_matrices=False)[0][:, 0])
    res = hopm(T, n_starts=n_starts, starts=[start], rng=rng)
    return max(res, key=lambda r: r[0])


def rank_one_flat(vecs) -> torch.Tensor:
    X = vecs[0]
    for v in vecs[1:]:
        X = torch.einsum("...,j->...j", X, v)
    return X.reshape(-1)


def spherical_classify(spec: TTNSpec, T: torch.Tensor, cores_unit, hessian_fn):
    """Riemannian Hessian of g(theta) = <W(theta), T> on the unit sphere at the balanced bond-rank-one
    point `cores_unit` (theory_escape 3.3): its positive eigenvalues h_u are the unstable directions.
    Returns g, n_unstable, the h_u, and gamma = sum_u h_u / (K g) (exponent of P ~ sigma^gamma)."""
    K = spec.n_cores
    u = spec.flatten(cores_unit)

    def gfun(x):
        return (to_dense(spec, spec.unflatten(x)) * T).sum()

    gval = float(gfun(u))
    H = hessian_fn(spec, u, fn=gfun)
    n = u.numel()
    P = torch.eye(n, dtype=DT) - torch.outer(u, u)
    HR = P @ (H - K * gval * torch.eye(n, dtype=DT)) @ P
    w = torch.linalg.eigvalsh(0.5 * (HR + HR.T))
    pos = w[w > 1e-6 * K * abs(gval)]
    return dict(
        g=gval,
        n_unstable=int(pos.numel()),
        unstable=[float(x) for x in pos],
        gamma=float(pos.sum()) / (K * gval) if pos.numel() else 0.0,
    )


# ---------------------------------------------------------------- saddles


@dataclass
class Saddle:
    """A balanced critical point of the loss restricted to the bond-rank profile `profile`."""

    profile: tuple  # rank per edge, in tree.edges order
    cores: list  # balanced cores with exactly these bond dimensions
    loss: float
    grad: float  # gradient norm (criticality check)
    overlaps: np.ndarray | None = None
    lam_old: float = float("nan")  # most negative eigenvalue of the fixed-rank (old) block
    v_old: torch.Tensor | None = None
    lam_old_max: float = float("nan")
    routes: list = field(default_factory=list)  # Route objects with room, all sizes
    no_room: list = field(default_factory=list)  # (C, mu) of routes blocked by chi
    W: torch.Tensor | None = None


@dataclass
class Route:
    C: frozenset
    k: int
    mu: float  # coupling: sigma_max(R_C) for k = 2 (the Hessian rate), injective norm for k >= 3
    blocks: list  # maximizer, one vector per core of C (sorted), legs grouped as in local_residual
    groups: list  # per core u of C (sorted): (u, list of output legs) for embedding


class Hopper:
    def __init__(
        self,
        setting: Setting,
        seed: int = 0,
        n_random_starts: int = 32,
        tol_route: float = 1e-6,
        tol_neg: float = 1e-7,
        max_route: int | None = None,
    ):
        s = setting
        self.s = s
        self.tree = s.tree
        self.K, self.N = s.tree.n_nodes, s.tree.n_sites
        self.full = TTNSpec.uniform(s.tree, s.d, s.chi)
        self.edges = list(s.tree.edges)
        self.chi = {v: self.full.bond_dims[v] for v in self.edges}
        self.W = s.w_star.to(DT)
        self.M = s.metrics
        self.T = apply_metric(self.W, self.M)  # escape tensor Sigma W* (theory_covariance C4)
        self.L0 = 0.5 * float((self.W * self.T).sum())
        self.rng = np.random.default_rng(seed)
        self.n_random_starts = n_random_starts
        self.tol_route, self.tol_neg = tol_route, tol_neg
        self.max_route = max_route or self.K
        t = self.tree
        self.subtree_nodes = {v: self._subtree(v) for v in range(self.K)}
        self.site_l = {i: LETTERS[i] for i in range(self.N)}
        self.edge_l = {v: LETTERS[self.N + j] for j, v in enumerate(self.edges)}
        self.root_l = LETTERS[self.N + len(self.edges)]
        self.subsets = connected_subsets(t, self.max_route)
        self.teacher_rank = {v: self.matricization_rank(self.W, v) for v in self.edges}
        comps = s.components or []
        self.comp_w = [float(c[0]) for c in comps]
        self.comp_U = [c[1].to(DT) for c in comps]
        self.comp_target = [float((self.T * U).sum()) for U in self.comp_U]  # <W*, Sigma U_r>
        self.comp_factors = [
            spectral_norm_grouped(U, n_starts=1, rng=self.rng)[1] for U in self.comp_U
        ]

    # ---- tree helpers

    def _subtree(self, v):
        out, stack = set(), [v]
        while stack:
            u = stack.pop()
            out.add(u)
            stack.extend(self.tree.children[u])
        return out

    def core_labels(self, v):
        t = self.tree
        lab = "".join(self.site_l[i] for i in t.sites[v])
        lab += "".join(self.edge_l[c] for c in t.children[v])
        lab += self.edge_l[v] if t.parent[v] >= 0 else self.root_l
        return lab

    def matricization_rank(self, X, v, rtol=1e-9):
        below = list(self.tree.subtree_sites[v])
        above = [i for i in range(self.N) if i not in below]
        m = X.permute(below + above).reshape(self.s.d ** len(below), -1)
        sv = torch.linalg.svdvals(m)
        return int((sv > rtol * sv[0]).sum()) if sv[0] > 0 else 0

    # ---- profiles, loss, fixed-rank fits

    def spec_for(self, profile) -> TTNSpec:
        bonds = [1] * self.K
        for v, r in zip(self.edges, profile):
            bonds[v] = int(r)
        return TTNSpec(self.tree, (self.s.d,) * self.N, tuple(bonds))

    def loss_dense(self, W):
        D = W - self.W
        return 0.5 * (D * apply_metric(D, self.M)).sum()

    def loss_of(self, spec, flat):
        return self.loss_dense(to_dense(spec, spec.unflatten(flat)))

    def _fg(self, spec):
        def f(x):
            xt = torch.from_numpy(x).requires_grad_(True)
            L = self.loss_of(spec, xt)
            (g,) = torch.autograd.grad(L, xt)
            return float(L.detach()), g.numpy().copy()

        return f

    def fit(self, profile, cores0, gtol=1e-12, rounds=4):
        """Descend the loss restricted to `profile` from cores0 (L-BFGS, re-balanced between rounds).
        Returns (cores, loss, gradient norm, effective profile). If the optimum leaves a bond direction
        unused (effective profile < profile), the cores are returned unbalanced for truncation."""
        spec = self.spec_for(profile)
        f = self._fg(spec)
        x = spec.flatten([c.detach().to(DT) for c in cores0]).numpy().copy()
        L, gn = float("nan"), float("nan")
        for _ in range(rounds):
            res = minimize(
                f,
                x,
                jac=True,
                method="L-BFGS-B",
                options=dict(maxiter=100000, maxfun=200000, gtol=gtol, ftol=0.0, maxcor=30),
            )
            cores = spec.unflatten(torch.from_numpy(res.x.copy()))
            eff = self.effective_profile(to_dense(spec, cores).detach())
            if eff != tuple(profile):
                return cores, float(res.fun), float("nan"), eff
            try:
                cores = balance(spec, cores)
            except RankDeficient:
                eff = self.effective_profile(to_dense(spec, cores).detach(), rtol=1e-5)
                return cores, float(res.fun), float("nan"), eff
            x = spec.flatten(cores).numpy().copy()
            L, g = f(x)
            gn = float(np.linalg.norm(g))
            if gn < 1e-10 * max(1.0, abs(L)) ** 0.5:
                break
        return cores, L, gn, tuple(profile)

    def hessian(self, spec, flat, fn=None):
        """Dense Hessian of fn (default: the loss on `spec`) at flat, by vmapped Hessian-vector products."""
        fn = fn or (lambda x: self.loss_of(spec, x))
        g = torch.func.grad(fn)
        n = flat.numel()

        def hvp(v):
            return torch.func.jvp(g, (flat,), (v,))[1]

        H = torch.func.vmap(hvp, chunk_size=64)(torch.eye(n, dtype=DT))
        return 0.5 * (H + H.T)

    def overlaps(self, W):
        if not self.comp_U:
            return None
        return np.array(
            [
                float((W * apply_metric(U, self.M)).sum()) / tgt
                for U, tgt in zip(self.comp_U, self.comp_target)
            ]
        )

    # ---- contractions: hanging subtrees and local residuals (theory_saddles section 2)

    def subnet(self, cores, nodes):
        """Contraction of the cores in `nodes`: legs = their sites (sorted) + open bond legs (sorted)."""
        nodes = sorted(nodes)
        t = self.tree
        terms = [self.core_labels(v) for v in nodes]
        open_edges = [e for e in self.edges if (e in nodes) != (t.parent[e] in nodes)]
        out_sites = sorted(i for v in nodes for i in t.sites[v])
        out = "".join(self.site_l[i] for i in out_sites) + "".join(
            self.edge_l[e] for e in open_edges
        )
        res = torch.einsum(",".join(terms) + "->" + out, *[cores[v] for v in nodes])
        return res, out_sites, open_edges

    def local_residual(self, cores, C, Rres):
        """R_C = Rres contracted with the hanging learned subtrees T_f of every boundary edge f of C,
        legs grouped per core of C (sorted): its sites and its boundary bond legs (old indices).
        <R, W_C[delta]> = <R_C, Phi_C(delta)> for route entries (new indices on C's internal edges)."""
        t = self.tree
        C = set(C)
        bnd = [e for e in self.edges if (e in C) != (t.parent[e] in C)]
        terms, ops = ["".join(self.site_l[i] for i in range(self.N))], [Rres]
        for f in bnd:
            hang = (
                self.subtree_nodes[f] if f not in C else set(range(self.K)) - self.subtree_nodes[f]
            )
            Tf, sites_f, _ = self.subnet(cores, hang)
            terms.append("".join(self.site_l[i] for i in sites_f) + self.edge_l[f])
            ops.append(Tf)
        out_sites = sorted(i for v in C for i in t.sites[v])
        out = "".join(self.site_l[i] for i in out_sites) + "".join(self.edge_l[f] for f in bnd)
        RC = torch.einsum(",".join(terms) + "->" + out, *ops)
        groups, perm, dims = [], [], []
        for u in sorted(C):
            legs = [out.index(self.site_l[i]) for i in t.sites[u]]
            legs += [out.index(self.edge_l[f]) for f in bnd if f == u or t.parent[f] == u]
            groups.append((u, [out[j] for j in legs], [RC.shape[j] for j in legs]))
            perm += legs
            dims.append(int(np.prod([RC.shape[j] for j in legs])) if legs else 1)
        return RC.permute(perm).reshape(dims), groups

    def routes_at(self, cores, profile, Rres, extra: int = 2):
        """Routes with non-zero coupling, by size, up to (smallest size found) + extra.
        Returns (routes with room, routes blocked because an internal edge has chi_e = r_e)."""
        prof = dict(zip(self.edges, profile))
        t = self.tree
        tol = self.tol_route * float(Rres.norm())
        found, blocked, kmin = [], [], None
        for C in self.subsets:
            if kmin is not None and len(C) > kmin + extra:
                break
            RC, groups = self.local_residual(cores, C, Rres)
            if float(RC.norm()) <= tol:
                continue
            mu, blocks = spectral_norm_grouped(RC, n_starts=4, rng=self.rng)
            if mu <= tol:
                continue
            internal = [e for e in self.edges if e in C and t.parent[e] in C]
            r = Route(C, len(C), mu, blocks, groups)
            if all(self.chi[e] > prof[e] for e in internal):
                found.append(r)
                kmin = len(C) if kmin is None else kmin
            else:
                blocked.append(r)
        return found, blocked

    # ---- saddles

    def effective_profile(self, W, rtol=1e-6):
        return tuple(self.matricization_rank(W, v, rtol) for v in self.edges)

    def make_saddle(self, profile, cores0):
        """Fixed-rank descent from cores0, then the saddle's fixed-rank block and its routes. If the
        optimum does not use every bond direction, re-fit with the effective ranks."""
        profile = tuple(int(r) for r in profile)
        cores = cores0
        for _ in range(4):
            cores, L, g, eff = self.fit(profile, cores)
            if eff == profile:
                break
            cores = truncate(self.spec_for(profile), cores, dict(zip(self.edges, eff)))
            profile = eff
        spec = self.spec_for(profile)
        W = to_dense(spec, cores).detach()
        S = Saddle(profile=profile, cores=cores, loss=L, grad=g, W=W, overlaps=self.overlaps(W))
        flat = spec.flatten(cores).detach()
        H = self.hessian(spec, flat)
        w, V = torch.linalg.eigh(H)
        S.lam_old, S.v_old, S.lam_old_max = float(w[0]), V[:, 0], float(w[-1])
        if L > 1e-12 * self.L0:
            Rres = apply_metric(self.W - W, self.M)
            S.routes, S.no_room = self.routes_at(cores, profile, Rres)
        return S

    def old_strict(self, S):
        return S.lam_old < -self.tol_neg * max(1.0, abs(S.lam_old_max))

    # ---- first escape (rule R2)

    def groups_first(self):
        """Site-carrying cores and their sites: the leg grouping of the bond-rank-one problem."""
        return [(v, list(self.tree.sites[v])) for v in range(self.K) if self.tree.sites[v]]

    def grouped(self, X):
        g = self.groups_first()
        return X.permute([i for _, ss in g for i in ss]).reshape(
            [self.s.d ** len(ss) for _, ss in g]
        )

    def rank_one_cores(self, ys, amp, spec=None, sign=1.0):
        """Bond-rank-one cores (bond index 0): core v = amp * y_v on its sites, amp on site-less cores."""
        spec = spec or self.full
        t = self.tree
        yv = {v: y for (v, _), y in zip(self.groups_first(), ys)}
        cores = []
        for v in range(self.K):
            g = torch.zeros(spec.shapes[v], dtype=DT)
            n_ph = len(t.sites[v])
            idx = (slice(None),) * n_ph + (0,) * (len(spec.shapes[v]) - n_ph)
            g[idx] = (
                (amp * yv[v]).reshape([self.s.d] * n_ph) if n_ph else torch.tensor(amp, dtype=DT)
            )
            cores.append(g)
        cores[0] = cores[0] * sign
        return cores

    def candidate_dense(self, ys):
        X = ys[0]
        for y in ys[1:]:
            X = torch.einsum("...,j->...j", X, y)
        g = self.groups_first()
        X = X.reshape([self.s.d] * self.N)
        order = [i for _, ss in g for i in ss]
        return X.permute(list(np.argsort(order)))

    def first_candidates(self):
        """Fixed points of the grouped rank-one problem max <Sigma W*, (x)_v y_v> (HOPM from random
        starts and from every labelled component), each classified by the Riemannian Hessian of
        g = <W(theta), Sigma W*> at its balanced bond-rank-one point on the unit sphere (full chi):
        n_unstable = 0 -> attractor of the first escape; else a saddle reached with P ~ sigma^gamma."""
        Tg = self.grouped(self.T)
        starts = []
        for U in self.comp_U:
            starts.append(spectral_norm_grouped(self.grouped(U), n_starts=1, rng=self.rng)[1])
        res = hopm(Tg, n_starts=self.n_random_starts, starts=starts, rng=self.rng)
        cands = []
        for j, (val, ys) in enumerate(res):
            if val <= 1e-12 * float(self.T.norm()):
                continue
            for c in cands:
                if abs(c["value"] - val) <= 1e-6 * val and all(
                    abs(float(a @ b)) > 1 - 1e-4 for a, b in zip(c["ys"], ys)
                ):
                    c["random_hits"] += int(j >= len(starts))
                    break
            else:
                cands.append(dict(value=val, ys=ys, random_hits=int(j >= len(starts))))
        for c in cands:
            X = self.candidate_dense(c["ys"])
            c["overlaps"] = self.overlaps(X * c["value"]) if self.comp_U else None
        if self.comp_U:
            # symmetry copies of one fixed point (e.g. sign flips that leave W* invariant) have the same
            # value and the same component overlaps: keep one, they predict the same sequence
            merged = []
            for c in cands:
                twin = next(
                    (
                        m
                        for m in merged
                        if abs(m["value"] - c["value"]) <= 1e-6 * c["value"]
                        and np.allclose(m["overlaps"], c["overlaps"], atol=1e-4)
                    ),
                    None,
                )
                if twin is None:
                    merged.append(c)
                else:
                    twin["random_hits"] += c["random_hits"]
            cands = merged
        for c in cands:
            c.update(self.classify_candidate(c["ys"]))
        cands.sort(key=lambda c: -c["value"])
        return cands

    def classify_candidate(self, ys):
        return spherical_classify(
            self.full, self.T, self.rank_one_cores(ys, self.K**-0.5), self.hessian
        )

    def saddle_from_candidate(self, cand):
        """First saddle: the rank-one critical point of the true loss in the candidate's basin."""
        prof1 = tuple(1 for _ in self.edges)
        spec1 = self.spec_for(prof1)
        X = self.candidate_dense(cand["ys"])
        c = float((X * self.T).sum()) / float((X * apply_metric(X, self.M)).sum())
        cores = self.rank_one_cores(cand["ys"], abs(c) ** (1.0 / self.K), spec1, sign=np.sign(c))
        return self.make_saddle(prof1, cores)

    def saddle_start(self):
        """init='saddle': the exact balanced saddle of a labelled component (start_component), or, if
        none is given, the lowest rank-one attractor of the first escape (the rank-one minimum)."""
        r = self.s.start_component
        if r is None:
            sads = [
                self.saddle_from_candidate(c)
                for c in self.first_candidates()
                if c["n_unstable"] == 0
            ]
            return min(sads, key=lambda S: S.loss)
        ys = spectral_norm_grouped(self.grouped(self.comp_U[r]), n_starts=1, rng=self.rng)[1]
        return self.saddle_from_candidate(dict(ys=ys))


def truncate(spec: TTNSpec, cores, new_profile: dict):
    """Drop bond directions with the smallest Gram eigenvalues (child side) to reach new_profile."""
    t = spec.tree
    cores = [c.clone() for c in cores]
    for v in t.edges:
        r, r_new = spec.bond_dims[v], new_profile[v]
        if r_new >= r:
            continue
        p = t.parent[v]
        leg = len(t.sites[p]) + t.children[p].index(v)
        gv = cores[v].reshape(-1, r)
        w, U = torch.linalg.eigh(gv.T @ gv)
        U = U[:, -r_new:]
        cores[v] = (gv @ U).reshape(cores[v].shape[:-1] + (r_new,))
        cores[p] = apply_on_leg(cores[p], U.T, leg)
    return cores


# ---------------------------------------------------------------- hopping (rules R3-R7)


def law_str(kind, k=None, slope=None, K=None):
    if kind == "first":
        return f"sigma^-{K - 2}"
    if kind == "rotation":
        return "log (rotation)"
    if k == 2:
        return f"log, slope {slope:.4g}" if slope is not None else "log"
    return f"sigma^-{k - 2}"


class _Hops:
    """Mixin: the depth-first hop over predicted saddles (kept separate for readability)."""

    def internal_edges(self, C):
        return [e for e in self.edges if e in C and self.tree.parent[e] in C]

    def component_route_starts(self, cores, C, groups):
        """HOPM starts at each labelled component's route blocks: its site vectors on the sites of the
        route's cores, and on each boundary leg the hanging subtree contracted with the component."""
        if not self.comp_U:
            return []
        t = self.tree
        C = set(C)
        bnd = [e for e in self.edges if (e in C) != (t.parent[e] in C)]
        vf_all = []
        for fac in self.comp_factors:
            vf = {}
            for f in bnd:
                hang = (
                    self.subtree_nodes[f]
                    if f not in C
                    else set(range(self.K)) - self.subtree_nodes[f]
                )
                Tf, sites_f, _ = self.subnet(cores, hang)
                eq = (
                    "".join(LETTERS[j] for j in range(len(sites_f)))
                    + "z,"
                    + ",".join(LETTERS[j] for j in range(len(sites_f)))
                    + "->z"
                )
                vf[self.edge_l[f]] = torch.einsum(eq, Tf, *[fac[i] for i in sites_f])
            vf_all.append(vf)
        starts = []
        for fac, vf in zip(self.comp_factors, vf_all):
            blocks = []
            for u, letters, dims in groups:
                vecs = [fac[i] for i in t.sites[u]] + [vf[ch] for ch in letters[len(t.sites[u]) :]]
                z = rank_one_flat(vecs) if vecs else torch.ones(1, dtype=DT)
                blocks.append(z)
            if all(float(b.norm()) > 1e-12 for b in blocks):
                starts.append([b / b.norm() for b in blocks])
        return starts

    def route_classify(self, S, C, groups, RC, ys):
        """Attractor or saddle of the route sub-TTN's spherical problem (internal bond dimensions
        chi_e - r_e, one site per core with a non-trivial group)."""
        prof = dict(zip(self.edges, S.profile))
        nodes = sorted(C)
        loc = {u: j for j, u in enumerate(nodes)}
        t = self.tree
        parent = [loc[t.parent[u]] if t.parent[u] in C else -1 for u in nodes]
        sites, dims, keep = [], [], []
        for j, (u, letters, gd) in enumerate(groups):
            D = int(np.prod(gd)) if gd else 1
            if D > 1:
                sites.append((len(dims),))
                dims.append(D)
                keep.append(j)
            else:
                sites.append(())
        sub = Tree(tuple(parent), tuple(sites), name="route")
        bonds = [1] * len(nodes)
        for u in nodes:
            if t.parent[u] in C:
                bonds[loc[u]] = self.chi[u] - prof[u]
        spec = TTNSpec(sub, tuple(dims), tuple(bonds))
        Tsub = RC.reshape(dims)
        k = len(nodes)
        cores = []
        for j, u in enumerate(nodes):
            g = torch.zeros(spec.shapes[j], dtype=DT)
            n_ph = len(sub.sites[j])
            idx = (slice(None),) * n_ph + (0,) * (len(spec.shapes[j]) - n_ph)
            g[idx] = (
                ys[j] * k**-0.5
                if n_ph
                else torch.tensor(k**-0.5, dtype=DT) * float(ys[j].reshape(-1)[0])
            )
            cores.append(g)
        return spherical_classify(spec, Tsub, cores, self.hessian)

    def k3_options(self, S, kmin):
        """Fixed points of <R_C, (x) Z> for every route C of size kmin, each classified as an attractor
        of the route's own escape (tie among attractors: random) or a saddle (rare branch)."""
        Rres = apply_metric(self.W - S.W, self.M)
        opts = []
        for r in S.routes:
            if r.k != kmin:
                continue
            RC, groups = self.local_residual(S.cores, r.C, Rres)
            starts = [r.blocks] + self.component_route_starts(S.cores, r.C, groups)
            res = hopm(RC, n_starts=12, starts=starts, rng=self.rng)
            seen = []
            for val, ys in sorted(res, key=lambda x: -x[0]):
                if val <= self.tol_route * float(Rres.norm()):
                    continue
                if any(
                    abs(v - val) <= 1e-6 * val
                    and all(abs(float(a @ b)) > 1 - 1e-5 for a, b in zip(z, ys))
                    for v, z in seen
                ):
                    continue
                seen.append((val, ys))
            for val, ys in seen:
                cls = self.route_classify(S, r.C, groups, RC, ys)
                opts.append((Route(r.C, r.k, val, ys, groups), cls))
        return opts

    def embed_route(self, cores, prof_old, routes, b):
        """Add route blocks (amplitude b per core) on the new bond indices r_e of the internal edges."""
        cores = [c.clone() for c in cores]
        for r in routes:
            for (u, letters, dims), z in zip(r.groups, r.blocks):
                Zb = z.reshape(dims) if dims else z.reshape(())
                idx, perm = [], []
                for ch in self.core_labels(u):
                    if ch in letters:
                        j = letters.index(ch)
                        idx.append(slice(0, dims[j]))
                        perm.append(j)
                    elif ch == self.root_l:
                        idx.append(0)
                    else:
                        e = next(v for v in self.edges if self.edge_l[v] == ch)
                        idx.append(prof_old[e])
                block = Zb.permute(perm) if perm else Zb
                cores[u][tuple(idx)] = cores[u][tuple(idx)] + b * block
        return cores

    def pad(self, cores, spec_new):
        out = []
        for g, shape in zip(cores, spec_new.shapes):
            h = torch.zeros(shape, dtype=DT)
            h[tuple(slice(0, s) for s in g.shape)] = g
            out.append(h)
        return out

    def route_next(self, S, routes):
        prof = dict(zip(self.edges, S.profile))
        new = dict(prof)
        for r in routes:
            for e in self.internal_edges(r.C):
                new[e] = prof[e] + 1
        new_prof = tuple(new[e] for e in self.edges)
        spec_new = self.spec_for(new_prof)
        a = float(S.W.norm()) ** (1.0 / self.K)
        cores = self.embed_route(self.pad(S.cores, spec_new), prof, routes, 0.2 * a)
        return self.make_saddle(new_prof, cores)

    def rotation_next(self, S):
        spec = self.spec_for(S.profile)
        flat = spec.flatten(S.cores)
        a = float(S.W.norm()) ** (1.0 / self.K)
        outs = []
        for sgn in (1.0, -1.0):
            nxt = self.make_saddle(S.profile, spec.unflatten(flat + sgn * 0.2 * a * S.v_old))
            if nxt.loss < S.loss - 1e-9 * self.L0 and not any(
                abs(nxt.loss - o.loss) <= 1e-8 * self.L0 and nxt.profile == o.profile for o in outs
            ):
                outs.append(nxt)
        return outs


class SaddleHopper(Hopper, _Hops):
    """Hopper + the depth-first hop. `run()` returns candidates (first escape) and sequences."""

    def saddle_type(self, S):
        k2 = [r.mu for r in S.routes if r.k == 2]
        if S.loss <= 1e-10 * self.L0:
            return "global minimum"
        if self.old_strict(S):
            return "strict (fixed-rank block)"
        if k2:
            return "strict (k*=2 route)"
        if S.routes:
            return f"higher-order (k*={min(r.k for r in S.routes)})"
        return "stall (no route with room)" if S.no_room else "stall (no route)"

    def learned(self, S):
        return set() if S.overlaps is None else {int(i) for i in np.flatnonzero(S.overlaps > 0.5)}

    def event(self, j, kind, S_from, S_to, **kw):
        lf = self.learned(S_from) if S_from is not None else set()
        lt = self.learned(S_to)
        k2 = [r.mu for r in S_from.routes if r.k == 2] if S_from is not None else []
        ev = dict(
            j=j,
            kind=kind,
            from_level=S_from.loss if S_from is not None else self.L0,
            to_level=S_to.loss,
            from_profile=list(S_from.profile) if S_from is not None else None,
            to_profile=list(S_to.profile),
            type=self.saddle_type(S_from) if S_from is not None else f"origin (order {self.K})",
            lam_old=S_from.lam_old if S_from is not None else None,
            gamma_k2_max=max(k2) if k2 else 0.0,
            learned=sorted(lt),
            newly_learned=sorted(lt - lf),
            forgotten=sorted(lf - lt),
            overlaps=None if S_to.overlaps is None else [round(float(x), 4) for x in S_to.overlaps],
            grad_to=S_to.grad,
            flags=[],
        )
        ev.update(kw)
        return ev

    def run(self, max_events: int = 10, max_sequences: int = 64, rare: bool = True):
        self.sequences = []
        self.max_events, self.max_sequences = max_events, max_sequences
        if self.s.init == "saddle":
            S1 = self.saddle_start()
            ev = self.event(
                0,
                "saddle start",
                None,
                S1,
                k=None,
                law="exact saddle + noise",
                branch="deterministic",
            )
            state = dict(n=1, ell={}, kappa_ok=False, lamT=None, a1=None, transient=False)
            self._dfs(S1, state, [ev], ["deterministic"], "main")
            return dict(candidates=[], sequences=self.sequences)
        cands = self.first_candidates()
        attr = [c for c in cands if c["n_unstable"] == 0]
        for c in cands:
            if c["n_unstable"] and not rare:
                continue
            S1 = self.saddle_from_candidate(c)
            if c["n_unstable"] == 0:
                branch = (
                    "deterministic"
                    if len(attr) == 1
                    else f"random (race among {len(attr)} attractors)"
                )
                pclass = "main"
            else:
                branch = "rare: P ~ sigma^%.3g (n_unstable=%d)" % (c["gamma"], c["n_unstable"])
                pclass = "rare"
            ev = self.event(
                0,
                "first",
                None,
                S1,
                k=self.K,
                C=list(range(self.K)),
                rate=c["value"],
                law=law_str("first", K=self.K),
                exponent=self.K - 2,
                branch=branch,
                gamma_wrong=c["gamma"] if c["n_unstable"] else 0.0,
            )
            state = dict(
                n=1,
                ell={},
                kappa_ok=True,
                lamT=c["value"],
                a1=float(S1.W.norm()) ** (1 / self.K),
                transient=c["n_unstable"] > 0,
            )
            self._dfs(S1, state, [ev], [branch], pclass)
        return dict(candidates=cands, sequences=self.sequences)

    def _finish(self, events, branches, pclass, end):
        if len(self.sequences) < self.max_sequences:
            self.sequences.append(dict(events=events, branches=branches, pclass=pclass, end=end))

    def kappa(self, state, r):
        """Head start accumulated during the first escape (theory_saddles 3.2-3.3, theory_teachers 3.3):
        kappa_C = mu_C / (lambda_T a1^(K-k)), the route's integrated coupling over log(1/sigma)."""
        if not state["kappa_ok"] or state["n"] != 1:
            return 0.0
        return r.mu / (state["lamT"] * state["a1"] ** (self.K - r.k))

    def trap_flag(self, k):
        if self.s.init == "balanced":
            return None
        if k >= 5:
            return "trap risk: dead route possible under unbalanced init (k>=5: dead state stable, R12)"
        if k == 4:
            return "trap risk: k=4 undecided (R12)"
        return None

    def _dfs(self, S, state, events, branches, pclass):
        if len(self.sequences) >= self.max_sequences:
            return
        j = len(events)
        if S.loss <= 1e-10 * self.L0:
            return self._finish(events, branches, pclass, "converged")
        if j > self.max_events:
            return self._finish(events, branches, pclass, "max events")
        k2 = [r for r in S.routes if r.k == 2]
        gmax = max((r.mu for r in k2), default=0.0)
        if self.old_strict(S) and -S.lam_old >= gmax:
            return self._rotation(S, state, events, branches, pclass)
        if not S.routes:
            if S.no_room:
                end = "stall: routes blocked by chi %s" % [sorted(r.C) for r in S.no_room]
            else:
                end = "stall: no route"
            return self._finish(events, branches, pclass, end)
        if k2:
            return self._k2_event(S, state, events, branches, pclass, k2)
        return self._k3_event(S, state, events, branches, pclass)

    def _rotation(self, S, state, events, branches, pclass):
        j = len(events)
        nxts = self.rotation_next(S)
        transient = state.get("transient", False) and state["n"] == 1
        for nxt in nxts:
            br = "deterministic" if len(nxts) == 1 else "random (sign of the unstable direction)"
            alt = min(S.routes, key=lambda r: (r.k, -r.mu)) if S.routes else None
            ev = self.event(
                j,
                "rotation",
                S,
                nxt,
                k=None,
                C=None,
                rate=-S.lam_old,
                law=law_str("rotation"),
                slope_saddle=1.0 / -S.lam_old,
                slope_small=0.0 if transient else 1.0 / -S.lam_old,
                branch=br,
                alt_k=alt.k if alt else None,
                alt_mu=alt.mu if alt else None,
            )
            if transient:
                ev["flags"].append("plateau unresolved in natural runs (O(1) deviation on arrival)")
            if ev["forgotten"]:
                ev["flags"].append(
                    "forgetting: learned component(s) %s fall below 1/2" % ev["forgotten"]
                )
            st = dict(state, n=state["n"] + 1, transient=False)
            self._dfs(nxt, st, events + [ev], branches + [br], pclass)
        if not nxts:
            self._finish(events, branches, pclass, "rotation found no lower critical point")

    def _k2_event(self, S, state, events, branches, pclass, k2):
        """Accumulated-utility race of the k = 2 routes (theory_teachers 3.4): log-amplitudes ell in
        units of log(1/sigma), -1 + kappa after the first escape, growing at rate gamma per plateau."""
        j = len(events)
        ell = dict(state["ell"])
        for r in k2:
            if r.C not in ell:
                ell[r.C] = -1.0 + self.kappa(state, r)
        tau = {r.C: max(0.0, -ell[r.C]) / r.mu for r in k2}
        best = min(tau.values())
        tied = [r for r in k2 if tau[r.C] <= best * (1 + 1e-6) + 1e-12]
        near = [r for r in k2 if r not in tied and tau[r.C] <= 1.1 * best + 1e-12]
        nxt = self.route_next(S, tied)
        for r in k2:
            if r not in tied:
                ell[r.C] = ell[r.C] + r.mu * best
        for r in tied:
            ell.pop(r.C, None)
        gam = max(r.mu for r in tied)
        ev = self.event(
            j,
            "route",
            S,
            nxt,
            k=2,
            C=sorted(set().union(*[r.C for r in tied])),
            rate=gam,
            law=law_str("route", 2, best),
            slope_small=best,
            slope_saddle=1.0 / gam,
            n_joint=len(tied),
            branch="deterministic (AGF race)",
        )
        if near:
            ev["flags"].append(
                "near tie in the k=2 race (within 10 percent): %s" % [sorted(r.C) for r in near]
            )
        if best == 0.0:
            ev["flags"].append("head start >= 1: no plateau (escapes during the previous one)")
        st = dict(state, n=state["n"] + 1, ell=ell, transient=False)
        self._dfs(nxt, st, events + [ev], branches + ["deterministic"], pclass)

    def _k3_event(self, S, state, events, branches, pclass):
        """k* >= 3 (rule R4c): attractors of the route problems tie (random order); fixed points that
        are saddles of the route problem are rare branches (theory_saddles Prop. 1 + theory_escape E9)."""
        j = len(events)
        kmin = min(r.k for r in S.routes)
        opts = self.k3_options(S, kmin)
        attr = [o for o in opts if o[1]["n_unstable"] == 0]
        for r, cls in opts:
            if cls["n_unstable"] == 0:
                br = (
                    "deterministic"
                    if len(attr) == 1
                    else "random (tie: %d routes with k=%d)" % (len(attr), kmin)
                )
                pc = pclass
            else:
                br = "rare: P ~ sigma_eff^%.3g (route saddle, n_unstable=%d)" % (
                    cls["gamma"],
                    cls["n_unstable"],
                )
                pc = "rare"
            kap = self.kappa(state, r)
            nxt = self.route_next(S, [r])
            # aligned-init prefactor of the gap (units sigma^-(k-2)): known for the first escape after the
            # origin plateau (head start kappa, theory_saddles 3.3); a route using every core (k = K)
            # never sees the learned cores and escapes at the absolute time sigma^-(K-2) / ((K-2) mu).
            pref = (1 - kap) / ((kmin - 2) * r.mu) if state["n"] == 1 else None
            absolute = 1.0 / ((kmin - 2) * r.mu) if kmin == self.K else None
            ev = self.event(
                j,
                "route",
                S,
                nxt,
                k=kmin,
                C=sorted(r.C),
                rate=r.mu,
                law=law_str("route", kmin),
                exponent=kmin - 2,
                aligned_prefactor=pref,
                absolute_prefactor=absolute,
                kappa=kap,
                branch=br,
                route_gamma_wrong=cls["gamma"],
            )
            fl = self.trap_flag(kmin)
            if fl:
                ev["flags"].append(fl)
            if kap >= 1:
                ev["flags"].append("head start >= 1: escapes during the first escape")
            st = dict(
                state,
                n=state["n"] + 1,
                ell=dict(state["ell"]),
                transient=pc == "rare" and pclass != "rare",
            )
            self._dfs(nxt, st, events + [ev], branches + [br], pc)


def summarize(seq) -> str:
    """One line per sequence: [law] level, learned-set changes."""
    parts = []
    for ev in seq["events"]:
        if ev["kind"] == "first":
            law = ev["law"]
        elif ev["kind"] == "saddle start":
            law = "saddle"
        elif ev["kind"] == "rotation":
            law = "rot %.3g" % ev["rate"]
        elif ev["k"] == 2:
            law = "log %.3g" % ev["slope_small"]
        else:
            law = ev["law"]
        lab = "+" + ",".join(map(str, ev["newly_learned"])) if ev["newly_learned"] else ""
        lab += ("-" + ",".join(map(str, ev["forgotten"]))) if ev["forgotten"] else ""
        parts.append(f"[{law}] L={ev['to_level']:.4g} {lab}".strip())
    return " -> ".join(parts) + f"  ({seq['end']})"
