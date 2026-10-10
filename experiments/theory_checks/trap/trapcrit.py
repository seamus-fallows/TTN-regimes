"""Conservation-law trap: prediction of dead / pinned / alive routes from the initial condition.

Theory in notes/theory_trap.md. On a plateau at a learned rank-one point, the single-new-index
entries of a route END and of its neighbours form small "fit-zero" products whose Gauss-Newton (GN)
cost is the part of the product normal to the tangent space of the learned manifold. Along the
product the Gram difference C_u (end minus partner, restricted to the new bond space and to physical
directions orthogonal to the learned pattern) is conserved, so the end's route entry survives iff
C_u > 0 (c_e = 1). Partners without sites (HT / binary internal nodes) give no GN cost at second
order; there the end decays through a three-core cluster (end, partner, sibling with sites).

All quantities are evaluated at the start of the plateau after the first escape, which is a
sigma-free function of the init direction: the init itself for a saddle start, and the
correlation-flow blow-up state (rescaled units) for a small init.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ttn.network import TTNSpec

torch.set_num_threads(1)


# ---- legs and contractions ----------------------------------------------------------------------


def legs(tree, v):
    """Legs of core v in library order: ('site', i), ('edge', c) for each child c, then its own
    edge ('edge', v) or ('root', v) for the trailing dimension-1 leg of the root."""
    out = [("site", s) for s in tree.sites[v]]
    out += [("edge", c) for c in tree.children[v]]
    out.append(("edge", v) if tree.parent[v] >= 0 else ("root", v))
    return out


def neighbours(tree, v):
    """(neighbour core, edge name) pairs of core v."""
    out = [(c, c) for c in tree.children[v]]
    if tree.parent[v] >= 0:
        out.append((tree.parent[v], v))
    return out


def contract(core, tree, v, ops):
    """Contract core v leg by leg. ops maps a leg to a 1-D tensor (contracted away) or a 2-D tensor
    M (leg dim x m: the leg is replaced by m, kept). Legs not in ops are kept. The root's trailing
    leg is always contracted with [1]. Returns the tensor with kept legs in their original order."""
    t = core
    pos = 0
    for leg in legs(tree, v):
        if leg[0] == "root":
            t = t.select(pos, 0)
            continue
        op = ops.get(leg)
        if op is None:
            pos += 1
        elif op.dim() == 1:
            t = torch.tensordot(t, op, dims=([pos], [0]))
        else:
            t = torch.movedim(torch.tensordot(t, op, dims=([pos], [0])), -1, pos)
            pos += 1
    return t


# ---- state at the start of the plateau after the first escape ----------------------------------


def correlation_flow(spec: TTNSpec, teacher, cores0, M: float = 1e10, rtol: float = 1e-10):
    """Rescaled correlation flow d phi/dtau = grad <W*, W(phi)> from cores0 until the largest
    component overlap a_r = <W, U_r>/s_r reaches M (theory_escape: the sigma -> 0 limit of the first
    plateau; homogeneous, so the result does not depend on the scale of cores0). Returns
    (cores at the stop, index of the escaping component)."""

    def field(_, y):
        th = torch.from_numpy(y).requires_grad_(True)
        (g,) = torch.autograd.grad(teacher.inner_ttn(spec, spec.unflatten(th)), th)
        return g.numpy()

    def overlaps(y):
        return teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y))).numpy()

    def event(_, y):
        return math.log(np.abs(overlaps(y)).max() + 1e-300) - math.log(M)

    event.terminal = True
    y0 = spec.flatten(cores0).numpy()
    y0 = y0 / np.abs(y0).max()
    sol = solve_ivp(field, (0.0, 1e12), y0, method="DOP853", rtol=rtol, atol=1e-13, events=event)
    if sol.status != 1:
        raise RuntimeError(f"correlation flow did not blow up: {sol.message}")
    y = sol.y[:, -1]
    return spec.unflatten(torch.from_numpy(y.copy())), int(np.abs(overlaps(y)).argmax())


def learned_directions(spec: TTNSpec, cores):
    """Learned bond direction on every edge = top right singular vector of the child core
    (fan-in x chi_e); valid once the learned rank-one part dominates the core."""
    w = {}
    for c in spec.tree.edges:
        g = cores[c].reshape(-1, cores[c].shape[-1])
        _, _, vh = torch.linalg.svd(g, full_matrices=False)
        w[c] = vh[0]
    return w


def complement(w):
    """Orthonormal basis (chi x (chi - 1)) of the complement of the unit vector w."""
    q, _ = torch.linalg.qr(torch.cat([w[:, None], torch.eye(len(w), dtype=w.dtype)], dim=1))
    return q[:, 1 : len(w)]


def plateau_state(spec: TTNSpec, teacher, cores0, start: str, learned: int = 0, M: float = 1e10):
    """(cores, q, w): state at the start of the plateau after the first escape, in units where the
    small entries are O(1), the learned component q and its bond directions w.
    start = 'saddle': the init already sits at the aligned saddle of component `learned` (bond
    index 0 on every edge); 'small' / 'small_balanced' / 'fanin' / 'balanced': correlation flow."""
    if start == "saddle":
        w = {
            c: torch.nn.functional.one_hot(torch.tensor(0), spec.bond_dims[c]).double()
            for c in spec.tree.edges
        }
        return cores0, learned, w
    cores, q = correlation_flow(spec, teacher, cores0, M=M)
    return cores, q, learned_directions(spec, cores)


# ---- route ends and their conserved Gram differences ------------------------------------------


def route_ends(tree, D):
    """(Steiner nodes of the cores carrying the sites D, list of (end u, route edge e, partner w))."""
    nodes = tree.steiner_nodes(sorted(D))
    ends = []
    if len(nodes) < 2:
        return nodes, ends
    for u in sorted(nodes):
        nb = [(x, e) for x, e in neighbours(tree, u) if x in nodes]
        if len(nb) == 1:
            ends.append((u, nb[0][1], nb[0][0]))
    return nodes, ends


def pattern_vector(teacher, tree, v, q):
    """Learned physical pattern of core v (Kronecker product of u_q over its sites, flattened)."""
    vec = torch.ones(1, dtype=torch.float64)
    for s in tree.sites[v]:
        vec = torch.kron(vec, teacher.factors[q, s])
    return vec


def new_block(spec, teacher, cores, w, q, v, open_edges):
    """Core v with every edge not in open_edges contracted with its learned direction, the open
    edges projected on their new spaces, and the physical legs projected orthogonally to the learned
    pattern. Returns a matrix (physical dim x prod of new dims), or (1 x ...) for a siteless core
    (no projection)."""
    tree = spec.tree
    ops = {}
    for x, e in neighbours(tree, v):
        ops[("edge", e)] = complement(w[e]) if e in open_edges else w[e]
    t = contract(cores[v], tree, v, ops)
    n_sites = len(tree.sites[v])
    phys = int(np.prod(t.shape[:n_sites])) if n_sites else 1
    m = t.reshape(phys, -1)
    if n_sites:
        p = pattern_vector(teacher, tree, v, q)
        m = m - torch.outer(p, p @ m)
    return m


def end_terms(spec, teacher, cores, w, q, u, e, partner):
    """Conserved GN quantities of route end u (route edge e to `partner`).
    j = 2 (partner with sites): C = X^T X - Y^T Y (c_e x c_e).
    j = 3 (siteless partner): for each neighbour u2 of the partner with sites, scalars
    c1 = |X_u|^2 - Z^2 and c2 = |X_u2|^2 - Z^2 of the three-core product (c_e = c_e2 = 1 only)."""
    tree = spec.tree
    X = new_block(spec, teacher, cores, w, q, u, {e})
    if tree.sites[partner]:
        Y = new_block(spec, teacher, cores, w, q, partner, {e})
        return dict(j=2, C=(X.T @ X - Y.T @ Y))
    clusters = []
    for u2, e2 in neighbours(tree, partner):
        if u2 == u or not tree.sites[u2]:
            continue
        Z = new_block(spec, teacher, cores, w, q, partner, {e, e2})
        X2 = new_block(spec, teacher, cores, w, q, u2, {e2})
        if X.shape[1] != 1 or X2.shape[1] != 1:
            clusters.append(dict(u2=u2, c1=float("nan"), c2=float("nan")))
            continue
        z2 = float(Z.pow(2).sum())
        clusters.append(dict(u2=u2, c1=float(X.pow(2).sum()) - z2, c2=float(X2.pow(2).sum()) - z2))
    return dict(j=3 if clusters else 4, clusters=clusters)


def gram_on_edge(spec, cores, e):
    """Q_e = G_c^T G_c - G_p,(c) G_p,(c)^T (child minus parent), the conserved matrix of edge e."""
    tree = spec.tree
    p = tree.parent[e]
    gc = cores[e].reshape(-1, cores[e].shape[-1])
    leg = len(tree.sites[p]) + tree.children[p].index(e)
    gp = torch.movedim(cores[p], leg, 0).reshape(cores[p].shape[leg], -1)
    return gc.T @ gc - gp @ gp.T


def route_entry_norm2(spec, teacher, cores, w, r, nodes, v, e):
    """|y_route|^2 of theory_saddles R12: core v's route entry for component r with its leg on the
    route edge e kept (projected on the new space), other route edges projected on their new
    spaces, non-route edges contracted with learned directions, sites contracted with u_r."""
    tree = spec.tree
    ops = {("site", s): teacher.factors[r, s] for s in tree.sites[v]}
    for x, f in neighbours(tree, v):
        ops[("edge", f)] = complement(w[f]) if (x in nodes) else w[f]
    return float(contract(cores[v], tree, v, ops).pow(2).sum())


def predict(spec, teacher, cores, w, q, D, r=1, weights=(1.0, 0.5)):
    """Per-end GN quantities and the route verdict for the dormant component r (differing sites D).
    Returns dict(k, verdict in {'alive', 'pinned', 'dead'}, n_dying, ends=[...]) plus the Q-only
    and R12 comparison verdicts. An end counts as dying only if its GN decay outruns the route
    (k >= 2j; k = 2j is the marginal case, counted)."""
    tree = spec.tree
    nodes, ends = route_ends(tree, D)
    k = len(nodes)
    rows = []
    for u, e, partner in ends:
        t = end_terms(spec, teacher, cores, w, q, u, e, partner)
        sign = 1.0 if u == e else -1.0
        nu = complement(w[e])
        Qn = sign * (nu.T @ gram_on_edge(spec, cores, e) @ nu)
        qnn = float(torch.linalg.eigvalsh(Qn).max())
        yr = route_entry_norm2(spec, teacher, cores, w, r, nodes, partner, e)
        Qfull = sign * gram_on_edge(spec, cores, e)
        row = dict(
            u=u,
            e=e,
            partner=partner,
            j=t["j"],
            q_nn=qnn,
            r12=qnn + yr,
            protected=bool(torch.linalg.eigvalsh(Qfull).min() > 0),
        )
        if t["j"] == 2:
            ev = torch.linalg.eigvalsh(t["C"])
            row.update(C=float(ev.max()), n_pos=int((ev > 0).sum()), dying=bool(ev.max() <= 0))
        elif t["j"] == 3:
            cl = t["clusters"]
            row.update(
                C=min(c["c1"] - min(0.0, c["c2"]) for c in cl),
                dying=any(c["c1"] < 0 and c["c1"] < c["c2"] for c in cl),
            )
        else:
            row.update(C=float("nan"), dying=False)
        row["active"] = k >= 2 * t["j"]
        rows.append(row)

    def classify(flags):
        n = sum(flags)
        return n, ("dead" if n >= 2 else "pinned" if n == 1 else "alive")

    n, v = classify([x["dying"] and x["active"] for x in rows])
    nq, vq = classify([x["q_nn"] < 0 for x in rows]) if k >= 4 else (0, "alive")
    nr, vr = classify([x["r12"] < 0 for x in rows]) if k >= 4 else (0, "alive")
    amps = route_amplitudes(spec, teacher, cores, w, r, nodes)
    v_k4 = None
    # k = 4, two dying ends: the sigma-free competition decides (reduced route ODE with tensor route
    # entries, basis invariant; ends need one new direction, inner edges may have spare ones)
    if (
        k == 4
        and n == 2
        and all(x["j"] == 2 for x in rows)
        and all(spec.bond_dims[e] == 2 for e in [x["e"] for x in rows])
    ):
        v_k4 = reduced_route(spec, teacher, cores, w, q, r, D, weights[q], weights[r])
        v = "dead" if v_k4 == "dead" else "alive"
    return dict(
        k=k,
        verdict=v,
        n_dying=n,
        verdict_q=vq,
        verdict_r12=vr,
        ends=rows,
        amps=amps,
        verdict_k4=v_k4,
    )


def nonvisible_gram(spec, cores, w, v, e):
    """Gram (c_e x c_e) of core v's entries with leg e in the new space of e and at least one other
    bond leg in its new space (any physical index): v's full new-slice Gram minus the part with all
    other legs on their learned directions. These entries pair with nothing at second order."""
    tree = spec.tree
    N = complement(w[e])
    kept = [lg for lg in legs(tree, v) if lg[0] != "root"]

    def gram(t, kept_legs):
        m = torch.movedim(t, kept_legs.index(("edge", e)), -1).reshape(-1, N.shape[1])
        return m.T @ m

    full = contract(cores[v], tree, v, {("edge", e): N})
    ops = {("edge", f): (N if f == e else w[f]) for _, f in neighbours(tree, v)}
    vis = contract(cores[v], tree, v, ops)
    vis_legs = [lg for lg in kept if lg[0] == "site" or lg == ("edge", e)]
    return gram(full, kept) - gram(vis, vis_legs)


def route_amplitudes(spec, teacher, cores, w, r, nodes):
    """|b_v| for every route core v: core v contracted with u_r on its sites, the new spaces of its
    route edges (kept, then the Frobenius norm is taken) and the learned directions elsewhere."""
    tree = spec.tree
    out = {}
    for v in nodes:
        ops = {("site", s): teacher.factors[r, s] for s in tree.sites[v]}
        for x, f in neighbours(tree, v):
            ops[("edge", f)] = complement(w[f]) if x in nodes else w[f]
        out[v] = float(contract(cores[v], tree, v, ops).norm())
    return out


def verdict_at(pred, unit, s_q, s_r, K):
    """Verdict at a finite scale. pred: output of predict (with 'amps'); unit = physical size of one
    unit of the small entries (sigma for a saddle start, sigma * max|Z| for the correlation-flow
    state); a = s_q^(1/K) the learned core amplitude. Two dying ends are a stable dead state iff
    gamma_u gamma_u' > kappa^2, gamma = a^(2(K-2)) |C| unit^2, kappa = s_r a^(K-k) prod_(other route
    cores) b unit^(k-2) (theory_trap.md, section 2.5); three or more dying ends are always dead."""
    k = pred["k"]
    dying = [x for x in pred["ends"] if x["dying"] and x["active"]]
    if len(dying) < 2:
        return ("pinned" if dying else "alive"), float("nan")
    if len(dying) > 2 or k < 4 or any(x["j"] != 2 for x in dying):
        return "dead", float("inf")  # no linear mutual drive, or j = 3 ends (sign test only)
    a = s_q ** (1.0 / K)
    u1, u2 = dying[0]["u"], dying[1]["u"]
    b = [pred["amps"][v] for v in pred["amps"] if v not in (u1, u2)]
    kappa = s_r * a ** (K - k) * math.prod(b) * unit ** (k - 2)
    gam = [a ** (2 * (K - 2)) * abs(x["C"]) * unit**2 for x in dying]
    ratio = gam[0] * gam[1] / max(kappa**2, 1e-300)
    return ("dead" if ratio > 1 else "unstable"), ratio


def route_tensors(spec, teacher, cores, w, r, nodes, rot=None):
    """Route entry of every route core as a tensor with one leg per route edge (in the order of
    neighbours(tree, v)), each leg in the new space of its edge: core v contracted with u_r on its
    sites and with the learned direction on its non-route edges. rot (edge -> orthogonal matrix)
    rotates the new-space basis of an edge (basis-invariance tests).
    Returns (route cores sorted, tensors as numpy arrays, einsum subscripts: one letter per edge)."""
    tree = spec.tree
    rot = rot or {}
    order = sorted(nodes)
    letters, tensors, subs = {}, [], []
    for v in order:
        ops = {("site", s): teacher.factors[r, s] for s in tree.sites[v]}
        sub = ""
        for x, f in neighbours(tree, v):
            if x in nodes:
                ops[("edge", f)] = complement(w[f]) @ rot[f] if f in rot else complement(w[f])
                sub += letters.setdefault(f, chr(ord("a") + len(letters)))
            else:
                ops[("edge", f)] = w[f]
        tensors.append(contract(cores[v], tree, v, ops).numpy())
        subs.append(sub)
    return order, tensors, subs


def reduced_route(
    spec,
    teacher,
    cores,
    w,
    q,
    r,
    D,
    s_q,
    s_r,
    unit=1.0,
    t_max=1e4,
    rot=None,
    mode="tensor",
    tensors=None,
    full=False,
):
    """Reduced plateau model of one route whose ends all have partners carrying sites and one new
    direction on their route edge (c_e = 1); inner route edges may have any number of new directions.
    Integrated from the plateau-start values scaled by `unit` (the physical size of one unit of the
    small entries; the result does not depend on it for k = 4):
      T_v' = c dP/dT_v - [v end] A^2 |y_v|^2 T_v,   |y_u|^2' = -2 A^2 |X_u|^2 |y_u|^2  (each end u),
    T_v = route entry of core v as a tensor over the new spaces of its route edges (route_tensors),
    P = full contraction of all T_v (the route's overlap with U_r), A^2 = a^(2(K-2)), c = s_r a^(K-k),
    a = s_q^(1/K); T_u at an end is its (signed) route entry, |X_u|^2 its whole normal-visible block,
    y_u the partner's normal-visible block. The flow is equivariant under rotations of every new-space
    basis, so the verdict is basis invariant (theory_trap.md section 2.6). mode='first_vector' keeps
    only the first basis vector of every new space (the v1 scalar model; basis dependent when an inner
    edge has c > 1, kept for the comparison). tensors: (order, tensors, subs) to use instead of
    route_tensors (tests). Returns 'escape' (the route entries grew 1e3-fold), 'dead' (two end entries
    below 1e-12 of their start) or 'pinned' (neither within t_max decay times); with full=True also
    the solve_ivp solution (dense output) and the state layout."""
    tree = spec.tree
    K = spec.n_cores
    nodes, ends = route_ends(tree, D)
    order, T, subs = tensors or route_tensors(spec, teacher, cores, w, r, nodes, rot)
    if mode == "first_vector":
        T = [t[(slice(0, 1),) * t.ndim] for t in T]
    T = [np.asarray(t, dtype=float) * unit for t in T]
    shapes = [t.shape for t in T]
    offs = np.cumsum([0] + [t.size for t in T])
    nb = int(offs[-1])
    a = s_q ** (1.0 / K)
    A2, c = a ** (2 * (K - 2)), s_r * a ** (K - len(nodes))
    ix = {v: i for i, v in enumerate(order)}
    endinfo = []  # (position of the end's route entry in the state, rest2, |y|^2)
    for u, e, partner in ends:
        if T[ix[u]].size != 1:
            raise ValueError("reduced_route needs one new direction on every end edge")
        X = new_block(spec, teacher, cores, w, q, u, {e})
        Y = new_block(spec, teacher, cores, w, q, partner, {e})
        bu = float(T[ix[u]].reshape(-1)[0])
        rest2 = max(float(X.pow(2).sum()) * unit**2 - bu**2, 0.0)
        endinfo.append((int(offs[ix[u]]), rest2, float(Y.pow(2).sum()) * unit**2))
    m = len(ends)
    y0 = np.concatenate(
        [
            np.concatenate([t.reshape(-1) for t in T]),
            [e[1] for e in endinfo],
            [e[2] for e in endinfo],
        ]
    )
    # basis-invariant scale; equal to v1's max |b_v| when every route entry is a scalar
    scale = max(float(np.linalg.norm(t)) for t in T)
    eqs = [
        ",".join(subs[j] for j in range(len(T)) if j != i) + "->" + subs[i] for i in range(len(T))
    ]

    def rhs(_, z):
        Ts = [z[offs[i] : offs[i + 1]].reshape(shapes[i]) for i in range(len(T))]
        db = np.concatenate(
            [c * np.einsum(eqs[i], *(Ts[:i] + Ts[i + 1 :])).reshape(-1) for i in range(len(T))]
        )
        rest2, y2 = z[nb : nb + m], z[nb + m :]
        drest, dy2 = np.zeros(m), np.zeros(m)
        for j, (p, _, _) in enumerate(endinfo):
            db[p] -= A2 * y2[j] * z[p]
            drest[j] = -2 * A2 * y2[j] * rest2[j]
            dy2[j] = -2 * A2 * (z[p] ** 2 + rest2[j]) * y2[j]
        return np.concatenate([db, drest, dy2])

    def big(_, z):  # smooth, so event location is reliable
        return float(np.sum(z[:nb] ** 2)) - (1e3 * scale) ** 2

    big.terminal = True
    t_end = t_max / (A2 * scale**2)
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        method="LSODA",
        rtol=1e-9,
        atol=1e-16 * scale,
        events=big,
        dense_output=full,
    )
    if sol.status == 1:
        verdict = "escape"
    else:
        z = sol.y[:, -1]
        ratios = sorted(abs(z[p]) / max(abs(y0[p]), 1e-300) for p, _, _ in endinfo)
        verdict = "dead" if ratios[1] < 1e-12 else "pinned"
    if full:
        layout = dict(offs=offs, shapes=shapes, subs=subs, ends=[p for p, _, _ in endinfo])
        return verdict, sol, layout
    return verdict
