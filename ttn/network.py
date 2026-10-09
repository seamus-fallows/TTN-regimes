"""TTN architectures and contractions.

A `TTNSpec` fixes the tree, the physical dimension of every site and the bond dimension of every
edge. Parameters ("cores") are plain lists of tensors, one per node, so that integrators can work on
flat parameter vectors. Core v has legs

    [physical legs of v (in `tree.sites[v]` order)] + [child bond legs (in `tree.children[v]` order)]
    + [parent bond leg]

and the root's parent leg has dimension 1 (the scalar output). W is the contraction of all cores.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from functools import cached_property

import torch

from ttn.tree import Tree

Cores = list[torch.Tensor]
_LETTERS = string.ascii_letters
_BATCH = "Z"  # batch letter; core legs never need 52 letters


@dataclass(frozen=True)
class TTNSpec:
    tree: Tree
    phys_dims: tuple[int, ...]  # one per site
    bond_dims: tuple[int, ...]  # one per node: dimension of the edge to its parent (root: 1)

    def __post_init__(self):
        if len(self.phys_dims) != self.tree.n_sites:
            raise ValueError("need one physical dimension per site")
        if len(self.bond_dims) != self.tree.n_nodes or self.bond_dims[self.tree.root] != 1:
            raise ValueError("need one bond dimension per node, with 1 at the root")

    @classmethod
    def uniform(cls, tree: Tree, d: int, chi: int, cap: bool = True) -> TTNSpec:
        """Same physical dimension d at every site and bond dimension chi on every edge.

        With cap=True each bond is capped at the largest rank its bipartition allows,
        min(d^{#sites below}, d^{#sites above}), as is standard for tensor networks.
        """
        bonds = [1] * tree.n_nodes
        for v in tree.edges:
            below = len(tree.subtree_sites[v])
            max_rank = min(d**below, d ** (tree.n_sites - below))
            bonds[v] = min(chi, max_rank) if cap else chi
        return cls(tree, (d,) * tree.n_sites, tuple(bonds))

    @cached_property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        t = self.tree
        return tuple(
            tuple(self.phys_dims[s] for s in t.sites[v])
            + tuple(self.bond_dims[c] for c in t.children[v])
            + (self.bond_dims[v],)
            for v in range(t.n_nodes)
        )

    @cached_property
    def fan_in(self) -> tuple[int, ...]:
        return tuple(math.prod(s[:-1]) for s in self.shapes)

    @cached_property
    def sizes(self) -> tuple[int, ...]:
        return tuple(math.prod(s) for s in self.shapes)

    @cached_property
    def n_params(self) -> int:
        return sum(self.sizes)

    @property
    def n_cores(self) -> int:
        return self.tree.n_nodes

    # ---- parameter vectors -------------------------------------------------------------

    def flatten(self, cores: Cores) -> torch.Tensor:
        return torch.cat([c.reshape(-1) for c in cores])

    def unflatten(self, flat: torch.Tensor) -> Cores:
        """Views into `flat` shaped like the cores (gradients flow back to `flat`)."""
        out, i = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            out.append(flat[i : i + size].view(shape))
            i += size
        return out

    def core_slices(self) -> list[slice]:
        out, i = [], 0
        for size in self.sizes:
            out.append(slice(i, i + size))
            i += size
        return out

    # ---- initialization ----------------------------------------------------------------

    def init(
        self,
        sigma: float,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        core_scales: list[float] | tuple[float, ...] | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> Cores:
        """Fan-in initialization G_v = sigma_v Z_v / sqrt(F_v), Z_v iid N(0, 1).

        sigma_v = sigma * core_scales[v] (core_scales default to 1), so E||W||_F^2 = prod sigma_v^2.
        """
        if generator is None:
            generator = torch.Generator().manual_seed(0 if seed is None else seed)
        scales = core_scales if core_scales is not None else [1.0] * self.n_cores
        cores = []
        for shape, fan_in, s in zip(self.shapes, self.fan_in, scales):
            z = torch.randn(shape, generator=generator, dtype=dtype)
            cores.append((sigma * s / math.sqrt(fan_in) * z).to(device))
        return cores


# ---- helpers ---------------------------------------------------------------------------


def apply_on_leg(t: torch.Tensor, m: torch.Tensor, leg: int) -> torch.Tensor:
    """t'[..., a, ...] = sum_b m[a, b] t[..., b, ...] on axis `leg`."""
    return torch.movedim(torch.tensordot(t, m, dims=([leg], [1])), -1, leg)


def _sqrt_psd(m: torch.Tensor) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(m)
    return (vecs * vals.clamp_min(0).sqrt()) @ vecs.T


def site_vectors(x: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    """Accept a (B, N, d) tensor or a list of N tensors of shape (B, d_i)."""
    if isinstance(x, torch.Tensor):
        return [x[:, i, :] for i in range(x.shape[1])]
    return list(x)


# ---- contractions ------------------------------------------------------------------------


def forward(spec: TTNSpec, cores: Cores, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """f(x) = <W, x_1 (x) ... (x) x_N> for a batch; x is (B, N, d) or a list of (B, d_i)."""
    t = spec.tree
    vecs = site_vectors(x)
    msgs: list[torch.Tensor | None] = [None] * t.n_nodes
    for v in t.postorder:
        n_phys, n_kids = len(t.sites[v]), len(t.children[v])
        legs = string.ascii_lowercase[: n_phys + n_kids + 1]
        terms = [legs] + [_BATCH + legs[k] for k in range(n_phys + n_kids)]
        ops = [cores[v]] + [vecs[s] for s in t.sites[v]] + [msgs[c] for c in t.children[v]]
        msgs[v] = torch.einsum(",".join(terms) + f"->{_BATCH}{legs[-1]}", *ops)
    return msgs[t.root][:, 0]


def to_dense(spec: TTNSpec, cores: Cores) -> torch.Tensor:
    """The full tensor W with axes in site order 0..N-1 (small N only)."""
    t = spec.tree
    dense: list[torch.Tensor | None] = [None] * t.n_nodes
    for v in t.postorder:
        out = cores[v]
        n_phys = len(t.sites[v])
        # Contract children one at a time; child c's subtree axes are appended after the core legs.
        for k, c in enumerate(t.children[v]):
            leg = n_phys + k
            dc = dense[c]  # (subtree dims of c..., chi_c)
            out = torch.tensordot(out, dc, dims=([leg - k], [dc.ndim - 1]))
            # Move the parent leg back to the end so leg indices of later children stay valid.
            parent_axis = out.ndim - (dc.ndim - 1) - 1
            out = torch.movedim(out, parent_axis, -1)
        # Axis order is now: phys of v, child subtree axes (children in order), parent.
        dense[v] = out
    w = dense[t.root][..., 0]
    order = t.subtree_sites[t.root]
    return w.permute([order.index(s) for s in range(t.n_sites)])


def inner(
    spec: TTNSpec,
    cores_a: Cores,
    cores_b: Cores,
    site_metrics: list[torch.Tensor] | None = None,
    spec_b: TTNSpec | None = None,
) -> torch.Tensor:
    """<W_a, (M_1 (x) ... (x) M_N) W_b> for two TTNs on the same tree (bond dims may differ)."""
    t = spec.tree
    if spec_b is not None and spec_b.tree != t:
        raise ValueError("inner() needs the same tree; use ttn.general.inner_general otherwise")
    env: list[torch.Tensor | None] = [None] * t.n_nodes
    for v in t.postorder:
        n_phys, n_kids = len(t.sites[v]), len(t.children[v])
        n = n_phys + n_kids + 1
        la, lb = _LETTERS[:n], _LETTERS[n : 2 * n]
        terms, ops = [la, lb], [cores_a[v], cores_b[v]]
        for k, s in enumerate(t.sites[v]):
            if site_metrics is None:
                lb = lb[:k] + la[k] + lb[k + 1 :]
            else:
                terms.append(la[k] + lb[k])
                ops.append(site_metrics[s])
        terms[1] = lb
        for k, c in enumerate(t.children[v]):
            j = n_phys + k
            terms.append(la[j] + lb[j])
            ops.append(env[c])
        env[v] = torch.einsum(",".join(terms) + f"->{la[-1]}{lb[-1]}", *ops)
    return env[t.root][0, 0]


def norm2(spec: TTNSpec, cores: Cores, site_metrics: list[torch.Tensor] | None = None):
    return inner(spec, cores, cores, site_metrics)


# ---- bond spectra --------------------------------------------------------------------------


def bond_factors(
    spec: TTNSpec, cores: Cores, site_metrics: list[torch.Tensor] | None = None
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    """QR sweeps giving, for every edge v, factors R_v (below) and U_v (above) such that the
    matricization of W across edge v is Iso_below @ R_v @ U_v^T @ Iso_above^T.

    With site metrics M_i, the factors are those of (M_1^{1/2} (x) ... (x) M_N^{1/2}) W, i.e. the
    function-space (L2) matricization under an anisotropic product input measure.
    """
    t = spec.tree
    sqrt_m = None if site_metrics is None else [_sqrt_psd(m) for m in site_metrics]

    def with_metrics(v: int) -> torch.Tensor:
        g = cores[v]
        if sqrt_m is not None:
            for k, s in enumerate(t.sites[v]):
                g = apply_on_leg(g, sqrt_m[s], k)
        return g

    r_below: list[torch.Tensor | None] = [None] * t.n_nodes
    for v in t.postorder:
        g = with_metrics(v)
        n_phys = len(t.sites[v])
        for k, c in enumerate(t.children[v]):
            g = apply_on_leg(g, r_below[c], n_phys + k)
        _, r = torch.linalg.qr(g.reshape(-1, g.shape[-1]))
        r_below[v] = r

    u_above: list[torch.Tensor | None] = [None] * t.n_nodes
    u_above[t.root] = torch.ones(1, 1, dtype=cores[t.root].dtype, device=cores[t.root].device)
    for p in t.preorder:
        n_phys = len(t.sites[p])
        g_p = with_metrics(p)
        g_p = apply_on_leg(g_p, u_above[p], g_p.ndim - 1)
        for k, v in enumerate(t.children[p]):
            g = g_p
            for k2, c in enumerate(t.children[p]):
                if c != v:
                    g = apply_on_leg(g, r_below[c], n_phys + k2)
            g = torch.movedim(g, n_phys + k, -1)
            _, u = torch.linalg.qr(g.reshape(-1, g.shape[-1]))
            u_above[v] = u
    return r_below, u_above


def bond_spectra(
    spec: TTNSpec, cores: Cores, site_metrics: list[torch.Tensor] | None = None
) -> dict[int, torch.Tensor]:
    """Singular values of W matricized across every edge (keyed by the edge's child node),
    sorted in decreasing order. Accurate down to ~1e-15 relative to the largest."""
    r_below, u_above = bond_factors(spec, cores, site_metrics)
    return {v: torch.linalg.svdvals(r_below[v] @ u_above[v].T) for v in spec.tree.edges}


# ---- gauge conservation laws -----------------------------------------------------------------


def conserved_quantities(
    spec: TTNSpec, cores: Cores, lr_multipliers: list[float] | None = None
) -> dict[int, torch.Tensor]:
    """Q_v = G_v^T G_v / eta_v - G_p,(v) G_p,(v)^T / eta_p for every edge v with parent p.

    G_v is matricized as (fan-in x chi_v); G_p,(v) is the parent core with its v-leg as rows.
    Exactly conserved by gradient flow with per-core learning rates eta (default 1).
    """
    t = spec.tree
    eta = lr_multipliers if lr_multipliers is not None else [1.0] * t.n_nodes
    out = {}
    for v in t.edges:
        p = t.parent[v]
        gv = cores[v].reshape(-1, cores[v].shape[-1])
        leg = len(t.sites[p]) + t.children[p].index(v)
        gp = torch.movedim(cores[p], leg, 0).reshape(cores[p].shape[leg], -1)
        out[v] = gv.T @ gv / eta[v] - gp @ gp.T / eta[p]
    return out


# ---- special initializations ------------------------------------------------------------------


def antisymmetric_double(spec: TTNSpec, cores: Cores) -> tuple[TTNSpec, Cores, torch.Tensor]:
    """Two-copy model W(theta+) - W(theta-) with theta+ = theta- (Woodworth et al. 2020).

    Represented as a TTN with doubled bond dimensions and block-diagonal cores: every non-root core
    holds G_v on its "+" and "-" blocks, the root holds +G and -G, so W = 0 exactly at any scale.
    Returns (spec, cores, mask). Gradient flow does NOT preserve the block structure by itself (the
    root's mixed blocks couple the + subtree of one child to the - subtree of another), so the
    gradient must be multiplied by `mask` (a flat 0/1 vector) to train the two copies.
    """
    t = spec.tree
    bonds = tuple(1 if v == t.root else 2 * spec.bond_dims[v] for v in range(t.n_nodes))
    new_spec = TTNSpec(t, spec.phys_dims, bonds)
    new, masks = [], []
    for v in range(t.n_nodes):
        g = cores[v]
        n_phys = len(t.sites[v])
        out = torch.zeros(new_spec.shapes[v], dtype=g.dtype, device=g.device)
        mask = torch.zeros_like(out)
        for sign, block in ((1.0, 0), (-1.0, 1)):
            idx = [slice(None)] * n_phys
            for c in t.children[v]:
                chi = spec.bond_dims[c]
                idx.append(slice(block * chi, (block + 1) * chi))
            if v == t.root:
                idx.append(slice(None))
                out[tuple(idx)] = sign * g
            else:
                chi = spec.bond_dims[v]
                idx.append(slice(block * chi, (block + 1) * chi))
                out[tuple(idx)] = g
            mask[tuple(idx)] = 1.0
        new.append(out)
        masks.append(mask)
    return new_spec, new, new_spec.flatten(masks)


def aligned_init(spec: TTNSpec, factors: torch.Tensor, amplitudes: torch.Tensor) -> Cores:
    """Cores diagonal in a CP teacher's basis ("decoupled" Saxe-type init).

    factors: (R, N, d) unit vectors u_r^(i); amplitudes: (K, R) per-core, per-component scalars a_{v,r}.
    Requires every bond dimension >= R. The represented tensor is
    sum_r (prod_v a_{v,r}) u_r^(1) (x) ... (x) u_r^(N). For orthonormal factors (odeco teacher) the
    gradient flow stays on this diagonal family and reduces to one scalar ODE per component.
    """
    t = spec.tree
    n_comp = factors.shape[0]
    cores = []
    for v in range(t.n_nodes):
        g = torch.zeros(spec.shapes[v], dtype=factors.dtype)
        for r in range(n_comp):
            vectors = [factors[r, s] for s in t.sites[v]]
            for c in t.children[v]:
                vectors.append(_onehot(spec.bond_dims[c], r, factors.dtype))
            vectors.append(_onehot(spec.bond_dims[v], 0 if v == t.root else r, factors.dtype))
            term = amplitudes[v, r]
            for vec in vectors:
                term = term.unsqueeze(-1) * vec
            g = g + term
        cores.append(g)
    return cores


def _onehot(n: int, i: int, dtype: torch.dtype) -> torch.Tensor:
    e = torch.zeros(n, dtype=dtype)
    e[i] = 1.0
    return e
