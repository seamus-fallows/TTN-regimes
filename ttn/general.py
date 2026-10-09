"""Inner products between TTNs on different trees (general tensor-network contraction).

Gluing two trees along their physical legs gives a graph with loops, contracted with an
opt_einsum path that is computed once per (spec_a, spec_b, metrics?) and cached.
"""

from __future__ import annotations

from functools import lru_cache

import opt_einsum
import torch

from ttn.network import Cores, TTNSpec


def _leg_symbols(spec: TTNSpec, phys_sym: list[str], bond_offset: int) -> list[str]:
    """einsum subscripts for every core of one TTN."""
    t = spec.tree
    bond_sym = {v: opt_einsum.get_symbol(bond_offset + v) for v in range(t.n_nodes)}
    out = []
    for v in range(t.n_nodes):
        legs = [phys_sym[s] for s in t.sites[v]]
        legs += [bond_sym[c] for c in t.children[v]]
        legs.append(bond_sym[v])  # root: its own size-1 index, summed out
        out.append("".join(legs))
    return out


@lru_cache(maxsize=64)
def _expression(spec_a: TTNSpec, spec_b: TTNSpec, with_metrics: bool):
    n = spec_a.tree.n_sites
    phys_a = [opt_einsum.get_symbol(i) for i in range(n)]
    phys_b = [opt_einsum.get_symbol(n + i) for i in range(n)] if with_metrics else phys_a
    offset = 2 * n
    subs_a = _leg_symbols(spec_a, phys_a, offset)
    subs_b = _leg_symbols(spec_b, phys_b, offset + spec_a.tree.n_nodes)
    subs = subs_a + subs_b
    shapes = list(spec_a.shapes) + list(spec_b.shapes)
    if with_metrics:
        subs += [phys_a[i] + phys_b[i] for i in range(n)]
        shapes += [(spec_a.phys_dims[i], spec_b.phys_dims[i]) for i in range(n)]
    eq = ",".join(subs) + "->"
    return opt_einsum.contract_expression(eq, *shapes, optimize="auto-hq")


def inner_general(
    spec_a: TTNSpec,
    cores_a: Cores,
    spec_b: TTNSpec,
    cores_b: Cores,
    site_metrics: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """<W_a, (M_1 (x) ... (x) M_N) W_b> for TTNs on arbitrary trees over the same sites."""
    expr = _expression(spec_a, spec_b, site_metrics is not None)
    ops = list(cores_a) + list(cores_b)
    if site_metrics is not None:
        ops += list(site_metrics)
    return expr(*ops, backend="torch")
