"""Throwaway pilot (before ttn.dynamics / ttn.diagnostics land): gradient flow with scipy solve_ivp
(RK45, rtol 1e-9, atol 1e-14) on the flat parameter vector, compiled autograd gradient of
ttn.losses.population_loss. Two-component pattern teachers. Records event times a_r = 0.1/0.5/0.9,
the plateau midpoint (min |grad| between the two events) and a gauge-projected Hessian there.

usage: uv run python pilot2.py <cases.json> <out.jsonl>   (cases: list of {kind,n,S,sigmas,seeds})
"""
import json, sys, time
import numpy as np, torch
from multiprocessing import Pool
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
torch.backends.opt_einsum.enabled = False
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rules import get_tree, steiner_k, two_component_patterns  # noqa: E402
from ttn.network import TTNSpec, conserved_quantities  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402
from ttn.losses import population_loss  # noqa: E402


def gauge_basis(spec, cores):
    """Orthonormal basis (columns) of the gauge tangent space at `cores`."""
    t = spec.tree
    vecs = []
    for v in t.edges:
        p = t.parent[v]
        chi = spec.bond_dims[v]
        leg = len(t.sites[p]) + t.children[p].index(v)
        for a in range(chi):
            for b in range(chi):
                A = torch.zeros(chi, chi, dtype=torch.float64); A[a, b] = 1.0
                d = [torch.zeros_like(c) for c in cores]
                d[v] = torch.tensordot(cores[v], A, dims=([cores[v].ndim - 1], [0]))
                d[p] = -torch.movedim(torch.tensordot(cores[p], A, dims=([leg], [1])), -1, leg)
                vecs.append(spec.flatten(d))
    G = torch.stack(vecs, 1)
    u, s, _ = torch.linalg.svd(G, full_matrices=False)
    return u[:, s > 1e-10 * s.max()]


def projected_hessian_eigs(spec, loss, y):
    x = torch.from_numpy(y)
    H = torch.func.hessian(loss)(x)
    B = gauge_basis(spec, spec.unflatten(x))
    P = torch.eye(len(y), dtype=torch.float64) - B @ B.T
    # restrict to the complement of the gauge directions
    u, s, _ = torch.linalg.svd(P)
    Q = u[:, s > 0.5]
    return torch.linalg.eigvalsh(Q.T @ H @ Q).numpy()


def run_case(case):
    kind, n, S, seed = case["kind"], case.get("n"), case["S"], case["seed"]
    d, chi, weights = case.get("d", 2), case.get("chi", 2), case.get("weights", [1.0, 0.5])
    tree = get_tree(kind, n)
    N = tree.n_sites
    spec = TTNSpec.uniform(tree, d, chi)
    teacher = pattern_teacher(weights, two_component_patterns(N, S), d=d, seed=0)
    t2 = teacher.norm2()

    def loss(x):
        return population_loss(spec, spec.unflatten(x), teacher, teacher_norm2=t2)

    grad_c = torch.compile(torch.func.grad(loss), fullgraph=True, dynamic=False)
    outs = []
    for sigma in case["sigmas"]:
        y0 = spec.flatten(spec.init(sigma, seed=seed)).numpy()
        L0 = loss(torch.from_numpy(y0)).item()
        memo = {}

        def stats(y):
            key = y.tobytes()
            if key not in memo:
                memo.clear()
                c = spec.unflatten(torch.from_numpy(y))
                memo[key] = teacher.component_overlaps(spec, c).numpy()
            return memo[key]

        events, names = [], []
        for r in range(2):
            for lev in (0.1, 0.5, 0.9):
                events.append((lambda r, lev: lambda t, y: stats(y)[r] - lev)(r, lev))
                names.append(f"a{r}_{lev}")
        done = lambda t, y: min(stats(y)) - 0.97
        done.terminal = True
        events.append(done)
        t0 = time.time()
        wall_limit = case.get("wall_limit", 1200)

        def rhs(t, y):
            if time.time() - t0 > wall_limit:
                raise TimeoutError
            return -grad_c(torch.from_numpy(y)).numpy()

        try:
            sol = solve_ivp(rhs, (0, case.get("tmax", 1e14)), y0, method="RK45", rtol=1e-9,
                            atol=1e-14, events=events)
        except TimeoutError:
            out = {k: case[k] for k in ("kind", "n", "S", "seed")}
            out.update(sigma=sigma, d=d, chi=chi, weights=weights, K=spec.n_cores,
                       k_pred=steiner_k(tree, S), status="timeout", wall=time.time() - t0)
            outs.append(out)
            continue
        out = {k: case[k] for k in ("kind", "n", "S", "seed")}
        out.update(sigma=sigma, d=d, chi=chi, weights=weights, K=spec.n_cores,
                   k_pred=steiner_k(tree, S), wall=time.time() - t0, nfev=int(sol.nfev),
                   nsteps=len(sol.t), status=int(sol.status), L0=L0)
        for name, te in zip(names, sol.t_events[:6]):
            out[name] = float(te[0]) if len(te) else None
        q0 = conserved_quantities(spec, spec.unflatten(torch.from_numpy(y0)))
        q1 = conserved_quantities(spec, spec.unflatten(torch.from_numpy(sol.y[:, -1])))
        out["Qdrift"] = max(float((q1[v] - q0[v]).abs().max()) for v in q0)
        ta, tb = out["a0_0.5"], out["a1_0.5"]
        if ta is not None and tb is not None:
            t1, t2_ = sorted((ta, tb))
            # plateau midpoint: minimum speed over accepted steps strictly between the events
            idx = np.where((sol.t > t1) & (sol.t < t2_))[0]
            if len(idx) > 2:
                sub = idx[np.linspace(0, len(idx) - 1, min(len(idx), 400)).astype(int)]
                speed = [float(np.linalg.norm(grad_c(torch.from_numpy(sol.y[:, i])).numpy())) for i in sub]
                j = sub[int(np.argmin(speed))]
                ev = projected_hessian_eigs(spec, loss, sol.y[:, j])
                out.update(t_mid=float(sol.t[j]), speed_mid=float(min(speed)),
                           L_mid=loss(torch.from_numpy(sol.y[:, j])).item(),
                           a_mid=stats(sol.y[:, j]).tolist(),
                           hess_min=float(ev[0]), hess_max=float(ev[-1]),
                           hess_neg=int((ev < -1e-3 * abs(ev[-1])).sum()),
                           hess_low=ev[:4].tolist())
        sub = np.unique(np.concatenate([np.linspace(0, len(sol.t) - 1, 3000).astype(int)]))
        np.savez_compressed(
            f"/root/outputs/locality/pilot2/{kind}{N}_S{''.join(map(str, S))}_s{sigma:.4g}_seed{seed}.npz",
            t=sol.t[sub], y=sol.y[:, sub])
        outs.append(out)
    return outs


if __name__ == "__main__":
    import os
    os.makedirs("/root/outputs/locality/pilot2", exist_ok=True)
    cases = json.load(open(sys.argv[1]))
    tasks = [dict(c, seed=s) for c in cases for s in c["seeds"]]
    tasks.sort(key=lambda c: -c.get("cost", 1))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 44) as pool, open(sys.argv[2], "a") as fh:
        for outs in pool.imap_unordered(run_case, tasks):
            for out in outs:
                fh.write(json.dumps(out) + "\n")
            fh.flush()
            o = outs[-1]
            print(o["kind"], o["n"], o["S"], o["seed"], f"{sum(x['wall'] for x in outs):.0f}s", flush=True)
