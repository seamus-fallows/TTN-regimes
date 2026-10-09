"""Rooted trees describing TTN topologies.

Node v carries one core. `parent[v]` is its parent (-1 for the root) and `sites[v]` the physical
sites attached to it. Sites are numbered 0..N-1 and each is attached to exactly one node. The edge
between node v and its parent is called "edge v"; it splits the sites into `subtree_sites[v]` and
the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


@dataclass(frozen=True)
class Tree:
    parent: tuple[int, ...]
    sites: tuple[tuple[int, ...], ...]
    name: str = field(default="custom", compare=False)  # equality and hash are structural

    def __post_init__(self):
        # Accept lists (e.g. from YAML); tuples keep the dataclass hashable and concatenations valid.
        object.__setattr__(self, "parent", tuple(int(p) for p in self.parent))
        object.__setattr__(self, "sites", tuple(tuple(int(s) for s in ss) for ss in self.sites))
        if len(self.parent) != len(self.sites):
            raise ValueError("parent and sites must have one entry per node")
        if sum(p == -1 for p in self.parent) != 1:
            raise ValueError("tree must have exactly one root")
        all_sites = sorted(s for ss in self.sites for s in ss)
        if all_sites != list(range(len(all_sites))):
            raise ValueError("sites must be 0..N-1, each attached to exactly one node")
        if len(self.postorder) != self.n_nodes:
            raise ValueError("parent array does not describe a connected tree")

    @cached_property
    def n_nodes(self) -> int:
        return len(self.parent)

    @cached_property
    def n_sites(self) -> int:
        return sum(len(s) for s in self.sites)

    @cached_property
    def root(self) -> int:
        return self.parent.index(-1)

    @cached_property
    def children(self) -> tuple[tuple[int, ...], ...]:
        kids: list[list[int]] = [[] for _ in range(len(self.parent))]
        for v, p in enumerate(self.parent):
            if p >= 0:
                kids[p].append(v)
        return tuple(tuple(k) for k in kids)

    @cached_property
    def postorder(self) -> tuple[int, ...]:
        """Children before parents; the root is last."""
        order: list[int] = []
        stack = [(self.parent.index(-1), False)]
        seen = set()
        while stack:
            v, expanded = stack.pop()
            if expanded:
                order.append(v)
                continue
            if v in seen:
                return ()
            seen.add(v)
            stack.append((v, True))
            for c in reversed(self.children[v]):
                stack.append((c, False))
        return tuple(order)

    @cached_property
    def preorder(self) -> tuple[int, ...]:
        """Parents before children; the root is first."""
        return tuple(reversed(self.postorder))

    @cached_property
    def subtree_sites(self) -> tuple[tuple[int, ...], ...]:
        """Sites below each node, in the axis order produced by dense contraction:
        the node's own sites, then each child's subtree sites in child order."""
        out: list[tuple[int, ...]] = [()] * self.n_nodes
        for v in self.postorder:
            out[v] = self.sites[v] + tuple(s for c in self.children[v] for s in out[c])
        return tuple(out)

    @cached_property
    def depth(self) -> tuple[int, ...]:
        d = [0] * self.n_nodes
        for v in self.preorder:
            if self.parent[v] >= 0:
                d[v] = d[self.parent[v]] + 1
        return tuple(d)

    @cached_property
    def edges(self) -> tuple[int, ...]:
        """Non-root nodes; edge v connects v to parent[v]."""
        return tuple(v for v in range(self.n_nodes) if self.parent[v] >= 0)

    @cached_property
    def site_node(self) -> tuple[int, ...]:
        where = [0] * self.n_sites
        for v, ss in enumerate(self.sites):
            for s in ss:
                where[s] = v
        return tuple(where)

    def path(self, u: int, v: int) -> list[int]:
        """Nodes on the path from u to v, inclusive."""
        up_u, up_v = [u], [v]
        while self.parent[up_u[-1]] >= 0:
            up_u.append(self.parent[up_u[-1]])
        while self.parent[up_v[-1]] >= 0:
            up_v.append(self.parent[up_v[-1]])
        common = set(up_u) & set(up_v)
        lca = next(x for x in up_u if x in common)
        left = up_u[: up_u.index(lca) + 1]
        right = up_v[: up_v.index(lca)]
        return left + list(reversed(right))

    def steiner_nodes(self, sites: list[int] | tuple[int, ...]) -> set[int]:
        """Nodes of the smallest subtree containing the nodes that carry the given sites."""
        nodes = [self.site_node[s] for s in sites]
        out = {nodes[0]}
        for n in nodes[1:]:
            out.update(self.path(nodes[0], n))
        # Paths from one terminal to all others cover the Steiner tree in a tree graph.
        return out

    def describe(self) -> str:
        lines = [
            f"Tree '{self.name}': {self.n_nodes} cores, {self.n_sites} sites, root {self.root}"
        ]
        for v in self.preorder:
            lines.append(
                "  " * self.depth[v]
                + f"node {v}: sites {list(self.sites[v])}, children {list(self.children[v])}"
            )
        return "\n".join(lines)


def binary_tree(n_sites: int) -> Tree:
    """Balanced binary tree with N - 1 cores; every core has two inputs (sites or children).

    Bottom nodes carry two adjacent sites; when a split leaves a single site, that site is attached
    to the parent core instead of getting its own (redundant d x d) leaf core.
    """
    if n_sites < 2:
        raise ValueError("need at least 2 sites")
    parent: list[int] = []
    sites: list[tuple[int, ...]] = []

    def build(group: list[int]) -> int:
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

    build(list(range(n_sites)))
    return Tree(tuple(parent), tuple(sites), name=f"binary{n_sites}")


def mps_tree(n_sites: int) -> Tree:
    """Matrix product state (caterpillar): node k carries site k; node k's parent is k+1."""
    if n_sites < 2:
        raise ValueError("need at least 2 sites")
    parent = tuple(k + 1 for k in range(n_sites - 1)) + (-1,)
    sites = tuple((k,) for k in range(n_sites))
    return Tree(parent, sites, name=f"mps{n_sites}")


def ht_tree(n_sites: int) -> Tree:
    """Hierarchical Tucker: balanced binary tree over leaf cores, one leaf core per site."""
    if n_sites < 2:
        raise ValueError("need at least 2 sites")
    parent: list[int] = []
    sites: list[tuple[int, ...]] = []

    def build(group: list[int]) -> int:
        if len(group) == 1:
            parent.append(-1)
            sites.append((group[0],))
            return len(parent) - 1
        half = (len(group) + 1) // 2
        left, right = build(group[:half]), build(group[half:])
        parent.append(-1)
        sites.append(())
        v = len(parent) - 1
        parent[left] = parent[right] = v
        return v

    build(list(range(n_sites)))
    return Tree(tuple(parent), tuple(sites), name=f"ht{n_sites}")


def make_tree(kind: str, n_sites: int) -> Tree:
    builders = {"binary": binary_tree, "mps": mps_tree, "ht": ht_tree}
    return builders[kind](n_sites)
