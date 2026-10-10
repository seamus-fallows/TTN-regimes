"""Independent check of Steiner counts k for every (tree, S) row in the committed locality tables."""
import sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality")
from rules import get_tree, route_rule, dense_pattern  # route rule only for cross-check
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"

def my_steiner(tree, S):
    # prune: repeatedly delete degree-1 nodes that carry no site of S (independent of Tree.path)
    adj = {v: set() for v in range(tree.n_nodes)}
    for v in range(tree.n_nodes):
        p = tree.parent[v]
        if p >= 0:
            adj[v].add(p); adj[p].add(v)
    term = {v for v in range(tree.n_nodes) if any(s in S for s in tree.sites[v])}
    alive = set(adj)
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            if v not in term and len(adj[v] & alive) <= 1:
                alive.discard(v); changed = True
    return alive

rows = []
for f in ["twocomp_table.csv", "twocomp_balanced_table.csv", "twocomp_chi4_table.csv", "robust_table.csv"]:
    t = pd.read_csv(R + f)
    for _, r in t.iterrows():
        tr = r["tree"]
        kind = "mps" if tr.startswith("mps") else "binary" if tr.startswith("binary") else "ht" if tr.startswith("ht") else tr
        n = int(tr[len(kind):]) if kind in ("mps", "binary", "ht") else None
        tree = get_tree(kind, n)
        S = [int(c) for c in r["case"].replace("S=", "")]
        k_mine = len(my_steiner(tree, S))
        k_lib = len(tree.steiner_nodes(S))
        rows.append(dict(file=f, tree=tr, S="".join(map(str, S)), k_table=int(r["k_rank"]), k_mine=k_mine, k_lib=k_lib))
df = pd.DataFrame(rows).drop_duplicates(["tree", "S", "k_table"])
bad = df[(df.k_table != df.k_mine) | (df.k_lib != df.k_mine)]
print("distinct (tree,S):", len(df), " mismatches:", len(bad))
print(bad.to_string())
# route rule at the exact saddle s1 U1 for each distinct (tree, S) (d = 2 pattern tensors)
out = []
for _, r in df.drop_duplicates(["tree", "S"]).iterrows():
    tr = r["tree"]; kind = "mps" if tr.startswith("mps") else "binary" if tr.startswith("binary") else "ht" if tr.startswith("ht") else tr
    n = int(tr[len(kind):]) if kind in ("mps", "binary", "ht") else None
    tree = get_tree(kind, n)
    S = [int(c) for c in r["S"]]
    pats = [[0] * tree.n_sites, [1 if i in S else 0 for i in range(tree.n_sites)]]
    w = [1.0, 0.5]
    ws = dense_pattern(pats, w, [0], 2); wst = dense_pattern(pats, w, [0, 1], 2)
    k_route, hits = route_rule(tree, ws, wst, max_size=min(tree.n_nodes, 9))
    out.append(dict(tree=tr, S=r["S"], k=r["k_mine"], k_route=k_route))
o = pd.DataFrame(out)
print("route-rule mismatches:", (o.k != o.k_route).sum(), "of", len(o))
print(o[o.k != o.k_route].to_string())
print(o.groupby("k").tree.apply(lambda x: sorted(set(x))).to_string())
