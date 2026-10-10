"""Reviewer's bond-spectrum check (independent of bond_events.py / ttn.network.bond_spectra): dense W from
the saved trajectory with the reviewer's contraction, s_2 of every edge matricization by SVD.
Per greedy two-component run:
  - non-Steiner edges: max over the whole run of s_2^e / (0.1 x teacher Schmidt value) (< 1 = never rank 2);
  - Steiner edges: time at which s_2^e first reaches 0.5 x s2 (half the teacher value) after t1;
    spread of these times across the Steiner edges, in units of the event width w = t(a2=0.9) - t(a2=0.1);
  - s_2^e at the plateau midpoint / s2 (dormant level) on the Steiner edges.
"""

import glob
import json
import sys

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(
    0,
    "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality",
)
from indep import Net, teacher  # noqa: E402


def own_steiner(parent, sites_of_node, S):
    """Prune non-terminal leaves until none is left (own Steiner subtree, as in steiner_check.py)."""
    n = len(parent)
    adj = {v: set() for v in range(n)}
    for v, p in enumerate(parent):
        if p >= 0:
            adj[v].add(p)
            adj[p].add(v)
    node_of = {s: v for v, ss in enumerate(sites_of_node) for s in ss}
    terms = {node_of[s] for s in S}
    alive = set(range(n))
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            if v not in terms and sum(1 for u in adj[v] if u in alive) <= 1:
                alive.remove(v)
                changed = True
    return alive


recs = {}
for f in sorted(glob.glob(sys.argv[1] + ".part*")):
    for line in open(f):
        r = json.loads(line)
        if (
            r.get("S") is not None
            and r.get("status") == 1
            and [e[1] for e in r["events"]] == [0, 1]
        ):
            tag = f"{r['kind']}{r['n']}_{''.join(map(str, r['S']))}_{r['start']}"
            if len(sys.argv) > 3 and tag not in sys.argv[3].split(","):
                continue
            recs[r["_key"]] = r
nets = {}
out = []
for r in recs.values():
    key = (r["kind"], r["n"])
    if key not in nets:
        nets[key] = Net(*key)
    net = nets[key]
    below = {}
    for v in range(net.K):

        def sub(u):
            s = list(net.sites[u])
            for c in net.children[u]:
                s += sub(c)
            return s

        below[v] = sorted(sub(v))
    edges = [v for v in range(net.K) if net.parent[v] >= 0]
    st = own_steiner(net.parent, net.sites, r["S"])
    steiner_edges = [v for v in edges if v in st and net.parent[v] in st]
    F, U, Wst = teacher(net.N, r["S"], r["weights"])
    z = np.load(r["traj"])
    t, y = z["t"], torch.from_numpy(z["y"])
    s2 = np.zeros((len(t), len(edges)))
    for i in range(len(t)):
        W = net.dense(net.unflat(y[:, i])).numpy()
        for j, v in enumerate(edges):
            rest = [a for a in range(net.N) if a not in below[v]]
            m = np.transpose(W, below[v] + rest).reshape(2 ** len(below[v]), -1)
            sv = np.linalg.svd(m, compute_uv=False)
            s2[i, j] = sv[1] if len(sv) > 1 else 0.0
    t1, t2 = r["events"][0][0], r["events"][1][0]
    c = r["crossings"]
    w = (c["a1_0.9"][-1] if c["a1_0.9"] else np.nan) - (c["a1_0.1"][-1] if c["a1_0.1"] else np.nan)
    s2_teacher = r["weights"][1]
    non = [j for j, v in enumerate(edges) if v not in steiner_edges]
    ste = [j for j, v in enumerate(edges) if v in steiner_edges]
    max_non = (
        float(
            (s2[:, non].max(0) / (0.1 * np.sqrt(r["weights"][0] ** 2 + r["weights"][1] ** 2))).max()
        )
        if non
        else np.nan
    )
    rise = []
    for j in ste:
        after = np.where((t > t1) & (s2[:, j] >= 0.5 * s2_teacher))[0]
        rise.append(t[after[0]] if len(after) else np.nan)
    rise = np.array(rise)
    tm = r["plateaus"][0].get("t_mid")
    im = int(np.argmin(np.abs(t - tm))) if tm is not None else None
    dorm = (s2[im, ste] / s2_teacher).max() if im is not None and ste else np.nan
    out.append(
        dict(
            tree=f"{r['kind']}{r['n']}",
            S="".join(map(str, r["S"])),
            start=r["start"],
            sigma=r["sigma"],
            seed=r["seed"],
            n_steiner=len(ste),
            n_other=len(non),
            max_other_over_thr=max_non,
            rise_spread_over_w=float((np.nanmax(rise) - np.nanmin(rise)) / w) if ste else np.nan,
            rise_minus_t2_over_w=float((np.nanmedian(rise) - t2) / w) if ste else np.nan,
            dormant_mid_max=float(dorm),
            steiner_all_rise=bool(np.all(np.isfinite(rise))),
        )
    )
json.dump(out, open(sys.argv[2], "w"))
import pandas as pd  # noqa: E402

df = pd.DataFrame(out)
pd.set_option("display.width", 220)
print(
    df.groupby(["tree", "S", "start"])
    .agg(
        runs=("seed", "size"),
        n_steiner=("n_steiner", "first"),
        n_other=("n_other", "first"),
        other_never_rank2=(
            "max_other_over_thr",
            lambda x: float(np.mean(np.asarray(x) < 1)) if np.isfinite(x).any() else np.nan,
        ),
        max_other=("max_other_over_thr", "max"),
        all_steiner_rise=("steiner_all_rise", "mean"),
        spread_med=("rise_spread_over_w", "median"),
        spread_max=("rise_spread_over_w", "max"),
        rise_vs_t2_med=("rise_minus_t2_over_w", "median"),
        dormant_mid_max=("dormant_mid_max", "max"),
    )
    .to_string()
)
