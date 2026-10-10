"""Independent blow-up time T of the correlation flow theta' = grad g, g = <W(theta), W*>, by the
spherical decomposition (no amplitude stop, no extrapolation): u' = grad g(u) - K g(u) u,
G' = g(u), t' = rho0^-(K-2) exp(-(K-2) K G) in tau; stop when (K-2) K G > 45 (tail < 1e-19).
Own contraction for binary4 / mps4; only the init (s2s.make_init) and teacher vectors come from the repo.
Usage: python sphere_T.py OUT.csv tree chi teacher init seeds...
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
REPO = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
sys.path.insert(0, str(REPO / "experiments/s2s_core"))
sys.path.insert(0, str(REPO))
import s2s  # noqa: E402

W = {"separated": [3.0, 2.0, 1.0], "close": [1.0, 0.85, 0.7]}


def make_g(tree, shapes, U, s):
    sizes = [int(np.prod(sh)) for sh in shapes]
    offs = np.cumsum([0] + sizes)

    def cores(x):
        return [x[offs[i] : offs[i + 1]].reshape(shapes[i]) for i in range(len(shapes))]

    def comps(x):
        c = cores(x)
        out = []
        for r in range(U.shape[0]):
            u = U[r]
            if tree == "binary4":
                A = torch.einsum("ijk,i,j->k", c[0], u[0], u[1])
                B = torch.einsum("ijk,i,j->k", c[1], u[2], u[3])
                M = c[2][:, :, 0]
                out.append(A @ M @ B)
            else:  # mps4: core0 (d, b), core1 (d, b_in, b_out), core2 (d, b_in, b_out), core3 (d, b_in, 1)
                v = torch.einsum("ia,i->a", c[0], u[0])
                v = torch.einsum("a,iab,i->b", v, c[1], u[1])
                v = torch.einsum("a,iab,i->b", v, c[2], u[2])
                out.append(torch.einsum("a,iab,i->b", v, c[3], u[3])[0])
        return torch.stack(out)

    def g(x):
        return (comps(x) * s).sum()

    return g, comps


def T_sphere(tree, chi, teacher, init, seed):
    name = tree.replace("4", "")
    spec, tch = s2s.setup(name, 4, 4, chi, W[teacher], 1000)
    K = spec.n_cores
    th = s2s.make_init(spec, tch, init, 1.0, seed)
    shapes = [tuple(c.shape) for c in th]
    x0 = torch.cat([c.reshape(-1) for c in th])
    U = tch.factors.clone()
    s = tch.weights.clone()
    g, comps = make_g(tree, shapes, U, s)
    rho0 = x0.norm().item()
    u0 = (x0 / rho0).numpy()
    n = len(u0)

    def rhs(tau, y):
        u = torch.tensor(y[:n], requires_grad=True)
        gv = g(u)
        (gr,) = torch.autograd.grad(gv, u)
        uu = u.detach()
        nrm2 = (uu * uu).sum()
        # keep |u| = 1 against drift: project the gradient and add a small normalising term
        du = gr - (gr @ uu) / nrm2 * uu - 0.0 * uu
        G = y[n]
        dt = rho0 ** (-(K - 2)) * np.exp(-(K - 2) * K * G)
        return np.concatenate([du.numpy(), [gv.item() / nrm2.item() ** (K / 2)], [dt]])

    def done(tau, y):
        return (K - 2) * K * y[n] - 45.0

    done.terminal = True
    y0 = np.concatenate([u0, [0.0, 0.0]])
    sol = solve_ivp(rhs, (0, 1e5), y0, method="DOP853", rtol=1e-12, atol=1e-14, events=done)
    y = sol.y[:, -1]
    u = torch.tensor(y[:n])
    gstar = g(u / u.norm()).item()
    tail = rho0 ** (-(K - 2)) * np.exp(-(K - 2) * K * y[n]) / ((K - 2) * K * gstar)
    T = y[n + 1] + tail
    a = (comps(u / u.norm()) * s).abs().numpy()
    order = np.argsort(a)[::-1]
    return {
        "tree": tree,
        "chi": chi,
        "teacher": teacher,
        "init": init,
        "seed": seed,
        "T_sphere": T,
        "r1_sphere": int(order[0]),
        "ratio_end": float(a[order[1]] / a[order[0]]),
        "status": sol.status,
        "tau_end": sol.t[-1],
        "norm_u_end": float(u.norm()),
    }


if __name__ == "__main__":
    out, tree, chi, teacher, init = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4],
        sys.argv[5],
    )
    seeds = [int(x) for x in sys.argv[6:]]
    rows = []
    for sd in seeds:
        r = T_sphere(tree, chi, teacher, init, sd)
        js = json.loads(
            (
                REPO / f"data/s2s_core/first/{tree}_chi{chi}_{teacher}_{init}_seed{sd}.json"
            ).read_text()
        )
        r.update(
            {
                "T_repo": js["T"],
                "T_v1": js.get("T_v1"),
                "r1_repo": js["r1_pred"],
                "ratio_stop_repo": js["corrflow"].get("ratio_stop"),
            }
        )
        r["rel_diff"] = r["T_sphere"] / r["T_repo"] - 1
        r["rel_diff_v1"] = r["T_sphere"] / r["T_v1"] - 1
        print(r, flush=True)
        rows.append(r)
    import pandas as pd

    pd.DataFrame(rows).to_csv(out, index=False)
