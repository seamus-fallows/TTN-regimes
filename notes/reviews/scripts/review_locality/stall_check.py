"""Is a 'stalled' run (runner's dead-route detector) really dead? Reviewer's own model (indep.Net), LSODA in
segments up to t = 1e12 with a wall budget; prints a_r, max |grad|, and s_2 of the Steiner edges (route
amplitude) at checkpoints. usage: stall_check.py <kind> <n> <S> <sigma> <seed> [wall_s]"""

import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
sys.path.insert(
    0,
    "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality",
)
from indep import Net, make_problem, teacher  # noqa: E402

kind, n, S, sigma, seed = (
    sys.argv[1],
    int(sys.argv[2]),
    [int(c) for c in sys.argv[3]],
    float(sys.argv[4]),
    int(sys.argv[5]),
)
wall = float(sys.argv[6]) if len(sys.argv) > 6 else 300
net = Net(kind, n)
F, U, Wst = teacher(n, S, [1.0, 0.5])
loss, gradf, hessf, overl = make_problem(net, U, Wst, [1.0, 0.5])
x = torch.cat([c.reshape(-1) for c in net.init(sigma, seed)]).numpy()


def s2_edges(x):
    W = net.dense(net.unflat(torch.from_numpy(x))).numpy()
    out = []
    for v in range(net.K):
        if net.parent[v] < 0:
            continue
        below = []
        stack = [v]
        while stack:
            u = stack.pop()
            below += list(net.sites[u])
            stack += [c for c in range(net.K) if net.parent[c] == u]
        below = sorted(below)
        rest = [a for a in range(n) if a not in below]
        sv = np.linalg.svd(
            np.transpose(W, below + rest).reshape(2 ** len(below), -1), compute_uv=False
        )
        out.append(sv[1])
    return out


t0, t = time.time(), 0.0
print(f"{kind}{n} S={S} sigma={sigma} seed={seed}")
for t_next in [10.0**e for e in range(1, 13)]:
    sol = solve_ivp(
        lambda t, y: -gradf(torch.from_numpy(y)).numpy(),
        (t, t_next),
        x,
        method="LSODA",
        jac=lambda t, y: -hessf(torch.from_numpy(y)).numpy(),
        rtol=1e-9,
        atol=1e-14,
    )
    x, t = sol.y[:, -1], t_next
    g = np.abs(gradf(torch.from_numpy(x)).numpy()).max()
    print(
        f"t=1e{int(np.log10(t)):d}  a={np.round(overl(torch.from_numpy(x)), 6).tolist()}  max|grad|={g:.2e}  "
        f"s2(edges)={['%.1e' % s for s in s2_edges(x)]}  wall={time.time() - t0:.0f}s",
        flush=True,
    )
    if overl(torch.from_numpy(x)).min() > 0.97 or time.time() - t0 > wall:
        break
