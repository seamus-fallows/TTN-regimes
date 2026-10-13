"""Causal test of the exponent shift in Gaussian (C) starts (notes/route_escape.md, Part II).

Hypothesis: a route entry at a leaf u of the routing region R_J(r) is drained on the eps^-2 time scale
by off-target two-core clusters formed with its neighbour w in the region: entries of w whose leg
towards u is a spare channel and whose other legs are active channels ("partners"). The cluster
G_u[x, k] G_w[k, active...] makes non-target strings (a learned pattern with u, w changed), costs
loss, and decays; the route entry G_u[x_r(u), k] decays with it.

Test: take a C direction, set its partner entries to zero (everything else unchanged), and rerun.
If the hypothesis holds, the drain disappears and the escape time returns to the eps^-(c-2) law.

Usage: uv run python experiments/route_escape/diag_partner.py CELL SEED LOG_EPS [LOG_EPS ...]
Writes /root/outputs/route-escape/diag/partner_{CELL}_{SEED}_{LOG_EPS}.json and prints a summary.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escape  # noqa: E402
from common import OUT, get_cell  # noqa: E402


def partner_mask(cell, r: int) -> np.ndarray:
    """Flat bool mask of the partner entries for every leaf u of R_J(r) (one neighbour in the region)."""
    tree, target, spec = cell.tree, cell.target, cell.spec
    region = target.region(cell.J, r)
    active = {e: sorted({target.blocks[e][a] for a in cell.J}) for e in tree.edges}
    slices = spec.core_slices()
    out = np.zeros(spec.n_params, dtype=bool)
    for u in region:
        nbrs = [v for v in region if v == tree.parent[u] or tree.parent[v] == u]
        if len(nbrs) != 1:
            continue
        w = nbrs[0]
        e_uw = u if tree.parent[u] == w else w  # edge named by its lower node
        legs = []  # index lists for core w: physical, children, parent
        for _ in tree.sites[w]:
            legs.append(list(range(cell.d)))
        for c in tree.children[w]:
            legs.append([i for i in range(spec.bond_dims[c]) if i not in active[c]] if c == e_uw
                        else active[c])
        if w != tree.root:
            legs.append([i for i in range(spec.bond_dims[w]) if i not in active[w]] if w == e_uw
                        else active[w])
        else:
            legs.append([0])
        for idx in itertools.product(*legs):
            out[slices[w].start + int(np.ravel_multi_index(idx, spec.shapes[w]))] = True
    return out


def main():
    cell_name, seed = sys.argv[1], int(sys.argv[2])
    cell = get_cell(cell_name)
    r = cell.greedy()
    pm = partner_mask(cell, r)
    orig = escape.direction

    def no_partners(c, kind, i):
        xi = orig(c, "C", i)
        xi[pm] = 0.0
        return xi

    escape.direction = no_partners
    for le in map(float, sys.argv[3:]):
        o = escape.run(cell_name, "C", seed, le, wall=3000)
        o["kind"] = "C_nopartner"
        o["n_partner_entries"] = int(pm.sum())
        p = OUT / "diag" / f"partner_{cell_name}_{seed:02d}_{le:+.2f}.json"
        escape.save(o, p)
        ref = json.loads(escape.path_for(cell_name, "C", seed, le).read_text()) \
            if escape.path_for(cell_name, "C", seed, le).exists() else {}
        d = o["diag"]
        node_min = {}
        for x in d:
            for v, val in x[f"route_{r}"].items():
                node_min[v] = min(node_min.get(v, np.inf), val / o["eps"])
        print(f"{cell_name} seed {seed} eps 1e{le}: partners zeroed = {pm.sum()}, t_esc = {o['t_esc']:.4g} "
              f"(with partners: {ref.get('t_esc', float('nan')):.4g}), phi = {o['floor_phi']:.2g}, "
              f"lowest route-entry norm / eps per node = { {v: round(x, 3) for v, x in node_min.items()} }",
              flush=True)


if __name__ == "__main__":
    main()
