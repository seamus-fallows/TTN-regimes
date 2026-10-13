"""Route geometry of pattern targets (sums of orthogonal product-basis tensors) on a tree.

Implements the objects of the saddle-order / aligned-dynamics paper
(`report/ttn_saddle_order_and_aligned_training_dynamics.pdf`):

- disagreement sets V_ab and Steiner subtrees St(.) (Sec. 2.2);
- the one-component order min(k*, 2 l*) of Theorem 3.2 (`RouteTarget.one_component_order`);
- the edgewise star condition and its blocks (Sec. 7.1), local types (Sec. 7.2);
- the routing region R_J(r) and route depth c_J(r) (Sec. 8.2), the greedy ordering rule (App. C);
- the block-aligned manifold M*, its exact reduced gradient flow (Theorem 7.7), and the gauge
  alignment that reads route entries off a generic network (`AlignedManifold`).

Conventions. A "node string" x_a(v) is the tuple of pattern values of component a on the sites of
node v (empty for nodes without sites). An edge is named by its lower node e (it joins e to
tree.parent[e]). Patterns are in the standard site basis (a Gaussian init is rotation invariant on
every physical leg, so this loses no generality).
"""

from __future__ import annotations

import itertools
import math
from functools import cached_property

import numpy as np
import scipy.integrate
import torch

from ttn.network import Cores, TTNSpec, forward, norm2
from ttn.teachers import CPTeacher, pattern_teacher
from ttn.tree import Tree


class RouteTarget:
    """W* = sum_a s_a U_a, U_a the product-basis tensor of pattern a."""

    def __init__(self, tree: Tree, patterns, weights):
        self.tree = tree
        self.patterns = tuple(tuple(int(i) for i in p) for p in patterns)
        self.weights = tuple(float(w) for w in weights)
        if len(set(self.patterns)) != len(self.patterns):
            raise ValueError("patterns must be distinct")
        if len(self.weights) != len(self.patterns) or min(self.weights) <= 0:
            raise ValueError("need one positive weight per pattern")
        if any(len(p) != tree.n_sites for p in self.patterns):
            raise ValueError("patterns need one value per site")
        self.n = len(self.patterns)
        self.x = tuple(
            tuple(tuple(p[s] for s in tree.sites[v]) for v in range(tree.n_nodes))
            for p in self.patterns
        )
        self.blocks: dict[int, tuple[int, ...]] = {}  # edge -> block id of every component
        self.n_blocks: dict[int, int] = {}
        self.star_violations: list[tuple[int, tuple[int, ...]]] = []  # (edge, block members)
        for e in tree.edges:
            below = self._below(e)
            left = [tuple(x[v] for v in sorted(below)) for x in self.x]
            right = [tuple(x[v] for v in range(tree.n_nodes) if v not in below) for x in self.x]
            # connected components of the edge support graph: a ~ b if they share a substring
            label = list(range(self.n))
            for a, b in itertools.combinations(range(self.n), 2):
                if (left[a] == left[b] or right[a] == right[b]) and label[a] != label[b]:
                    old, new = label[b], label[a]
                    label = [new if x == old else x for x in label]
            ids = {x: i for i, x in enumerate(dict.fromkeys(label))}
            self.blocks[e] = tuple(ids[x] for x in label)
            self.n_blocks[e] = len(ids)
            for blk in range(len(ids)):
                members = tuple(a for a in range(self.n) if self.blocks[e][a] == blk)
                # a star has a centre: every member shares the left or the right substring
                if len({left[a] for a in members}) > 1 and len({right[a] for a in members}) > 1:
                    self.star_violations.append((e, members))

    # ---- tree geometry ---------------------------------------------------------------------

    def _below(self, e: int) -> frozenset[int]:
        out, stack = set(), [e]
        while stack:
            u = stack.pop()
            out.add(u)
            stack.extend(self.tree.children[u])
        return frozenset(out)

    def steiner(self, nodes) -> frozenset[int]:
        """Nodes of the smallest subtree containing `nodes` (empty for no nodes)."""
        nodes = list(nodes)
        if not nodes:
            return frozenset()
        out = {nodes[0]}
        for n in nodes[1:]:
            out.update(self.tree.path(nodes[0], n))
        return frozenset(out)

    def diff(self, a: int, b: int) -> frozenset[int]:
        """V_ab: nodes where components a and b disagree."""
        return frozenset(v for v in range(self.tree.n_nodes) if self.x[a][v] != self.x[b][v])

    def pair_cost(self, a: int, b: int) -> int:
        """|St(V_ab)|: cores needed to grow b out of a alone."""
        return len(self.steiner(self.diff(a, b)))

    # ---- star condition and local types ----------------------------------------------------

    @property
    def is_star(self) -> bool:
        """Assumption 7.1: every component of every edge support graph is a star."""
        return not self.star_violations

    def local_type(self, v: int, a: int) -> tuple[int, ...]:
        """tau_v(a) as the index of core v: site values, child-edge blocks, parent-edge block."""
        t = self.tree
        parent = (self.blocks[v][a],) if v != t.root else (0,)
        return self.x[a][v] + tuple(self.blocks[c][a] for c in t.children[v]) + parent

    # ---- routing regions ---------------------------------------------------------------------

    def region(self, J, r: int) -> frozenset[int]:
        """R_J(r): nodes where r cannot reuse a local entry of the learned set J (all nodes if J
        is empty). Under the star condition it is a nonempty connected subtree (Cor. 7.4)."""
        out = frozenset(range(self.tree.n_nodes))
        for a in J:
            out &= self.steiner(self.diff(r, a))
        return out

    def depth(self, J, r: int) -> int:
        """c_J(r) = |R_J(r)|, the number of dormant factors of P_r at the partial-target point."""
        return len(self.region(J, r))

    def missing(self, J) -> list[int]:
        return [r for r in range(self.n) if r not in set(J)]

    def min_depth(self, J) -> int:
        return min(self.depth(J, r) for r in self.missing(J))

    def partial_loss(self, J) -> float:
        """L_J = 1/2 sum of s_r^2 over the missing components."""
        return 0.5 * sum(self.weights[r] ** 2 for r in self.missing(J))

    def greedy_next(self, J) -> list[dict]:
        """Missing components ranked by the Appendix C rule: smallest route depth first, larger
        weight on ties (the tie-break coefficient s_r A_r(J) with unit active entries)."""
        J = list(J)
        rows = []
        for r in self.missing(J):
            rows.append(
                dict(
                    component=r,
                    depth=self.depth(J, r),
                    pairwise=min((self.pair_cost(a, r) for a in J), default=self.tree.n_nodes),
                    region=sorted(self.region(J, r)),
                    weight=self.weights[r],
                )
            )
        return sorted(rows, key=lambda x: (x["depth"], -x["weight"]))

    def greedy_sequence(self, first: int | None = None) -> list[dict]:
        """The predicted learning sequence after `first` (default: the largest weight)."""
        J = [max(range(self.n), key=lambda a: self.weights[a]) if first is None else first]
        steps = []
        while len(J) < self.n:
            ranked = self.greedy_next(J)
            step = dict(ranked[0], learned_before=tuple(J), tie=False)
            if len(ranked) > 1 and ranked[1]["depth"] == ranked[0]["depth"]:
                step["tie"] = True
            steps.append(step)
            J.append(ranked[0]["component"])
        return steps

    # ---- one learned component (Theorem 3.2) -----------------------------------------------

    def one_component_order(self, j: int, residuals=None) -> dict:
        """Saddle order at W = s_j U_j: k*, the eigenvalues mu_l of the coupling matrices M_l,
        the first level l* with mu_l > s_j, and min(k*, 2 l*).

        An atom is (B, restriction of a residual to B). Two atoms on disjoint sets couple with
        weight s_r when together they make residual r; the pair enters M_l when both Steiner
        sizes are <= l."""
        residuals = [r for r in range(self.n) if r != j] if residuals is None else list(residuals)
        k = {r: self.pair_cost(r, j) for r in residuals}
        k_star = min(k.values())
        pairs = []  # (atom 1, atom 2, weight, level at which the pair enters)
        for r in residuals:
            nodes = sorted(self.diff(r, j))
            for m in range(1, len(nodes) // 2 + 1):
                for b1 in itertools.combinations(nodes, m):
                    b2 = tuple(v for v in nodes if v not in b1)
                    if len(b1) == len(b2) and b1 > b2:
                        continue
                    atoms = [(b, tuple(self.x[r][v] for v in b)) for b in (b1, b2)]
                    level = max(len(self.steiner(b1)), len(self.steiner(b2)))
                    pairs.append((atoms[0], atoms[1], self.weights[r], level))
        mu = {}
        for level in range(1, max((p[3] for p in pairs), default=0) + 1):
            used = [p for p in pairs if p[3] <= level]
            index = {a: i for i, a in enumerate(dict.fromkeys(x for p in used for x in p[:2]))}
            mat = np.zeros((len(index), len(index)))
            for a1, a2, w, _ in used:
                mat[index[a1], index[a2]] = mat[index[a2], index[a1]] = w
            mu[level] = float(np.linalg.eigvalsh(mat)[-1]) if len(index) else 0.0
        l_star = min((lv for lv, m in mu.items() if m > self.weights[j]), default=math.inf)
        order = min(k_star, 2 * l_star)
        return dict(
            k_star=k_star,
            k=k,
            mu=mu,
            l_star=l_star,
            order=order,
            mechanism="spectral" if 2 * l_star < k_star else "direct",
        )

    # ---- tensors ---------------------------------------------------------------------------

    def teacher(self, d: int) -> CPTeacher:
        return pattern_teacher(
            list(self.weights), [list(p) for p in self.patterns], d, basis="standard"
        )


def amplitudes(spec: TTNSpec, cores: Cores, teacher: CPTeacher) -> torch.Tensor:
    """P_a = <U_a, W> for every teacher component (gauge invariant, valid off the manifold)."""
    return forward(spec, cores, [teacher.factors[:, i, :] for i in range(teacher.n_sites)])


def off_target_mass(spec: TTNSpec, cores: Cores, teacher: CPTeacher) -> float:
    """Fraction of ||W||^2 outside the span of the target components (0 on M*, Theorem 7.6)."""
    total = float(norm2(spec, cores))
    on = float((amplitudes(spec, cores, teacher) ** 2).sum())
    return max(total - on, 0.0) / total if total > 0 else 0.0


def crossing_time(t, y, level: float) -> float:
    """First time y rises through `level` (linear interpolation between samples); nan if never."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    above = np.nonzero((y[1:] >= level) & (y[:-1] < level))[0]
    if len(above) == 0:
        return math.nan
    i = above[0]
    return float(t[i] + (level - y[i]) * (t[i + 1] - t[i]) / (y[i + 1] - y[i]))


def effective_depth(
    t, p, s: float, lo: float = 1e-4, hi: float = 1e-2, min_points: int = 6
) -> dict | None:
    """Depth d of the scalar route equation dy/dt = d gamma^(2/d) y^(2-2/d) (s - y) (eq. 6.15) read
    off one amplitude curve: the slope of log[(dy/dt)/(s - y)] against log y for y/s in [lo, hi]
    is 2 - 2/d. Same estimator and window as experiments/locality/growth.py."""
    t, p = np.asarray(t, dtype=float), np.asarray(p, dtype=float)
    rate = np.gradient(p, t)
    ok = (p > lo * s) & (p < hi * s) & (rate > 0)
    if ok.sum() < min_points:
        return None
    slope = float(np.polyfit(np.log(p[ok]), np.log(rate[ok] / (s - p[ok])), 1)[0])
    return dict(slope=slope, depth=2 / (2 - slope) if slope < 2 else math.inf, points=int(ok.sum()))


class AlignedManifold:
    """The block-aligned coordinate subspace M* of a TTN (Sec. 7.2).

    Its coordinates are the "entries" g[k], one per (node, local type); component a uses entry
    route[a, v] at node v, and on M* the network is W = sum_a P_a U_a with P_a the product of the
    entries on its route (Theorem 7.6). Bond e has dimension max(chi, number of blocks on e), so
    chi larger than needed gives spare channels.
    """

    def __init__(
        self, target: RouteTarget, d: int, chi: int | None = None, require_star: bool = True
    ):
        if require_star and not target.is_star:
            raise ValueError(
                f"target violates the edgewise star condition: {target.star_violations}"
            )
        self.target = target
        tree = target.tree
        bonds = [1] * tree.n_nodes
        for e in tree.edges:
            bonds[e] = target.n_blocks[e] if chi is None else max(chi, target.n_blocks[e])
        self.spec = TTNSpec(tree, (d,) * tree.n_sites, tuple(bonds))
        self.teacher = target.teacher(d)
        self.weights = np.array(target.weights)
        slices = self.spec.core_slices()
        index: dict[tuple, int] = {}
        self.entries: list[tuple[int, tuple[int, ...]]] = []  # (node, core index)
        flat = []
        self.route = np.zeros((target.n, tree.n_nodes), dtype=int)
        for v in range(tree.n_nodes):
            for a in range(target.n):
                idx = target.local_type(v, a)
                if (v, idx) not in index:
                    index[(v, idx)] = len(self.entries)
                    self.entries.append((v, idx))
                    flat.append(
                        slices[v].start + int(np.ravel_multi_index(idx, self.spec.shapes[v]))
                    )
                self.route[a, v] = index[(v, idx)]
        self.flat = np.array(flat)
        self.n_entries = len(self.entries)

    # ---- coordinates -------------------------------------------------------------------------

    @cached_property
    def mask(self) -> torch.Tensor:
        """Flat boolean vector: True on the allowed entries."""
        m = torch.zeros(self.spec.n_params, dtype=torch.bool)
        m[torch.from_numpy(self.flat)] = True
        return m

    def theta(self, g) -> torch.Tensor:
        """Flat parameter vector of the manifold point with entries g."""
        out = torch.zeros(self.spec.n_params, dtype=torch.float64)
        out[torch.from_numpy(self.flat)] = torch.as_tensor(np.asarray(g), dtype=torch.float64)
        return out

    def g(self, theta: torch.Tensor) -> np.ndarray:
        return theta.detach()[torch.from_numpy(self.flat)].numpy().copy()

    def amplitudes(self, g) -> np.ndarray:
        """P_a = product of the entries on the route of a (eq. 7.9)."""
        return np.asarray(g, dtype=float)[self.route].prod(1)

    def used_by(self, J) -> np.ndarray:
        """Boolean per entry: used by at least one component of J."""
        out = np.zeros(self.n_entries, dtype=bool)
        out[self.route[list(J)].reshape(-1)] = True
        return out

    def partial_target(
        self, J, dormant: float = 0.0, relax_from: float | None = None
    ) -> np.ndarray:
        """Entries of a partial-target point theta_J (W = sum over J of s_a U_a).

        Default: the representative of Lemma 8.1 (active entries 1, one private entry per learned
        component scaled by s_a). With relax_from = sigma: the balanced representative that the
        flow on M* reaches from uniform active entries sigma (the one small-init training
        approaches). Entries not used by J are set to `dormant`."""
        J = list(J)
        active = self.used_by(J)
        if relax_from is None:
            g = np.where(active, 1.0, 0.0)
        else:
            res = self.flow(
                np.where(active, relax_from, 0.0),
                t_max=1e30,
                stop=lambda t, g, p: np.abs(p[J] - self.weights[J]).max() < 1e-9,
            )
            g = res["g"][-1]
        # one private entry per learned component (exists by Cor. 7.4) sets P_a = s_a exactly
        for a in J:
            others = self.used_by([b for b in J if b != a])
            private = [k for k in self.route[a] if not others[k]]
            g[private[0]] *= self.target.weights[a] / self.amplitudes(g)[a]
        return np.where(active, g, dormant)

    # ---- exact reduced flow (Theorem 7.7) ----------------------------------------------------

    def _products(self, g):
        on_route = g[self.route]  # (n, K)
        p = on_route.prod(1)
        q = np.stack(
            [np.delete(on_route, v, axis=1).prod(1) for v in range(on_route.shape[1])], axis=1
        )
        return on_route, p, q

    def velocity(self, g: np.ndarray) -> np.ndarray:
        """dg/dt on M*: dg_i/dt = sum over components a through i of (s_a - P_a) P_a / g_i."""
        _, p, q = self._products(g)
        out = np.zeros_like(g)
        np.add.at(out, self.route, (self.weights - p)[:, None] * q)
        return out

    def jacobian(self, g: np.ndarray) -> np.ndarray:
        on_route, p, q = self._products(g)
        n_nodes = on_route.shape[1]
        jac = np.zeros((self.n_entries, self.n_entries))
        for a in range(self.target.n):
            ra = self.route[a]
            jac[np.ix_(ra, ra)] -= np.outer(q[a], q[a])
            gap = self.weights[a] - p[a]
            for i, j in itertools.combinations(range(n_nodes), 2):
                qij = gap * np.delete(on_route[a], [i, j]).prod()
                jac[ra[i], ra[j]] += qij
                jac[ra[j], ra[i]] += qij
        return jac

    def flow(
        self,
        g0,
        t_max: float,
        rtol: float = 1e-10,
        atol: float = 1e-30,
        method: str = "LSODA",
        stop=None,
        max_steps: int = 10**7,
    ) -> dict:
        """Gradient flow on M* from entries g0. Returns the solver's own steps: t (n,), g (n, m),
        P (n, components). No dense output is used (scipy's LSODA interpolant is unreliable at
        these tolerances, see notes/saddle_dyn.md deviation 3). `stop(t, g, P)` ends the run early."""
        g0 = np.asarray(g0, dtype=float)
        solver = getattr(scipy.integrate, method)(
            lambda t, y: self.velocity(y),
            0.0,
            g0,
            t_max,
            rtol=rtol,
            atol=atol,
            jac=lambda t, y: self.jacobian(y),
        )
        ts, gs = [0.0], [g0.copy()]
        while solver.status == "running" and len(ts) < max_steps:
            solver.step()
            ts.append(solver.t)
            gs.append(solver.y.copy())
            if stop is not None and stop(solver.t, solver.y, self.amplitudes(solver.y)):
                break
        gs = np.array(gs)
        return dict(t=np.array(ts), g=gs, P=gs[:, self.route].prod(2), status=solver.status)

    # ---- reading a generic network -----------------------------------------------------------

    def _lower_message(self, cores: Cores, v: int, a: int) -> torch.Tensor:
        """Subtree of v contracted with the one-hot site vectors of component a (vector on v's
        parent bond). It does not depend on the gauge of the bonds below v."""
        out = cores[v][self.target.x[a][v]]
        for c in self.target.tree.children[v]:
            out = torch.tensordot(self._lower_message(cores, c, a), out, dims=([0], [0]))
        return out

    def align(self, cores: Cores, J) -> tuple[Cores, dict[int, float]]:
        """Orthogonal bond rotations that put the channels used by the learned set J on their
        block indices. Gradient flow commutes with orthogonal gauge maps, and from small init the
        flow stays balanced, so a network near theta_J in some gauge is mapped near theta_J itself.

        On every edge the lower messages of the learned components span the active channels; the
        rotation sends their (orthonormalised) directions to the block indices and an arbitrary
        orthonormal basis of the rest to the remaining indices. Returns the rotated cores and, per
        edge, the smallest singular value of the normalised message matrix (1 = orthogonal
        messages; near 0 = the network is not near theta_J)."""
        tree, target = self.target.tree, self.target
        cores = [c.clone() for c in cores]
        quality = {}
        for e in tree.edges:
            chi = self.spec.bond_dims[e]
            active = sorted({target.blocks[e][a] for a in J})
            cols = []
            for blk in active:
                msgs = [self._lower_message(cores, e, a) for a in J if target.blocks[e][a] == blk]
                cols.append(sum(m / m.norm() for m in msgs))
                cols[-1] = cols[-1] / cols[-1].norm()
            m = torch.stack(cols, dim=1)  # (chi, active)
            u, s, vt = torch.linalg.svd(m, full_matrices=True)
            quality[e] = float(s.min())
            q = torch.zeros(chi, chi, dtype=m.dtype)
            q[:, active] = u[:, : len(active)] @ vt  # polar factor: closest orthonormal frame
            rest = [i for i in range(chi) if i not in active]
            q[:, rest] = u[:, len(active) :]
            p = tree.parent[e]
            leg = len(tree.sites[p]) + tree.children[p].index(e)
            cores[e] = torch.tensordot(cores[e], q, dims=([cores[e].ndim - 1], [0]))
            cores[p] = torch.movedim(torch.tensordot(cores[p], q, dims=([leg], [0])), -1, leg)
        return cores, quality

    def route_entry_norms(self, aligned: Cores, J, r: int) -> dict[int, float]:
        """Per node of R_J(r): the size of the dormant entries that component r needs there, read
        off cores returned by `align(cores, J)`. On edges inside R_J(r) the channel is new, so the
        norm is taken over the whole dormant subspace of that bond (invariant under rotations of
        it); on edges leaving R_J(r) the channel is the active block of r. On M* with a single
        dormant channel this is |g| of the dormant entry (the b of eq. C.4)."""
        tree, target = self.target.tree, self.target
        region = target.region(J, r)
        active = {e: {target.blocks[e][a] for a in J} for e in tree.edges}
        out = {}
        for v in sorted(region):
            idx = [[i] for i in target.x[r][v]]
            for e, other in [(c, c) for c in tree.children[v]] + (
                [(v, tree.parent[v])] if v != tree.root else []
            ):
                if other in region:
                    idx.append([i for i in range(self.spec.bond_dims[e]) if i not in active[e]])
                else:
                    if target.blocks[e][r] not in active[e]:
                        raise ValueError(
                            "boundary channel of the route is not active (star condition?)"
                        )
                    idx.append([target.blocks[e][r]])
            if v == tree.root:
                idx.append([0])
            out[v] = float(np.linalg.norm(aligned[v].detach().numpy()[np.ix_(*idx)]))
        return out
