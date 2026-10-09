"""Throwaway pilot (before ttn.dynamics lands): gradient flow with scipy solve_ivp on the flat
parameter vector, two-component pattern teachers, event times a_r = 0.1/0.5/0.9."""
import json, sys, time
import numpy as np, torch
from multiprocessing import Pool
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
torch.backends.opt_einsum.enabled = False
from ttn.tree import make_tree
from ttn.network import TTNSpec, conserved_quantities
from ttn.teachers import pattern_teacher
from ttn.losses import population_loss


def run(cfg):
    kind, n, S, sigma, seed = cfg["kind"], cfg["n"], cfg["S"], cfg["sigma"], cfg["seed"]
    d, chi = cfg.get("d", 2), cfg.get("chi", 2)
    tree = make_tree(kind, n)
    spec = TTNSpec.uniform(tree, d, chi)
    p2 = [1 if i in S else 0 for i in range(n)]
    teacher = pattern_teacher(cfg.get("weights", [1.0, 0.5]), [[0] * n, p2], d=d, seed=0)
    t2 = teacher.norm2()
    y0 = spec.flatten(spec.init(sigma, seed=seed)).numpy()
    L0 = population_loss(spec, spec.unflatten(torch.from_numpy(y0)), teacher, teacher_norm2=t2).item()

    def rhs(t, y):
        x = torch.from_numpy(y).requires_grad_(True)
        L = population_loss(spec, spec.unflatten(x), teacher, teacher_norm2=t2)
        return -torch.autograd.grad(L, x)[0].numpy()

    memo = {}
    def stats(y):
        key = y.tobytes()
        if key not in memo:
            memo.clear()
            x = torch.from_numpy(y)
            c = spec.unflatten(x)
            a = teacher.component_overlaps(spec, c).numpy()
            L = population_loss(spec, c, teacher, teacher_norm2=t2).item()
            memo[key] = (a, L)
        return memo[key]

    events = []
    for r in range(2):
        for lev in (0.1, 0.5, 0.9):
            f = (lambda r, lev: lambda t, y: abs(stats(y)[0][r]) - lev)(r, lev)
            events.append(f)
    stop = lambda t, y: np.log(stats(y)[1] / L0) + np.log(1e8)
    stop.terminal = True
    events.append(stop)
    t0 = time.time()
    sol = solve_ivp(rhs, (0, cfg.get("tmax", 1e12)), y0, method="RK45", rtol=1e-9, atol=1e-14,
                    events=events)
    out = dict(cfg)
    out["wall"] = time.time() - t0
    out["nfev"] = int(sol.nfev)
    out["status"] = int(sol.status)
    names = [f"a{r}_{lev}" for r in range(2) for lev in (0.1, 0.5, 0.9)]
    for name, te in zip(names, sol.t_events[:6]):
        out[name] = float(te[0]) if len(te) else None
    q0 = conserved_quantities(spec, spec.unflatten(torch.from_numpy(y0)))
    q1 = conserved_quantities(spec, spec.unflatten(torch.from_numpy(sol.y[:, -1])))
    out["Qdrift"] = max(float((q1[v] - q0[v]).abs().max()) for v in q0)
    out["Lfinal"] = stats(sol.y[:, -1])[1] / L0
    out["tfinal"] = float(sol.t[-1])
    return out


if __name__ == "__main__":
    cases = [("mps", 4, [1, 2]), ("mps", 4, [0, 1, 2]), ("mps", 4, [0, 3]),
             ("binary", 4, [0, 1]), ("binary", 4, [1, 2]), ("ht", 4, [0, 1])]
    cfgs = [dict(kind=k, n=n, S=S, sigma=s, seed=seed) for k, n, S in cases
            for s in (0.3, 0.1, 0.03, 0.01) for seed in range(3)]
    with Pool(40) as pool:
        with open(sys.argv[1], "w") as fh:
            for out in pool.imap_unordered(run, cfgs):
                fh.write(json.dumps(out) + "\n"); fh.flush()
                print(out["kind"], out["n"], out["S"], out["sigma"], out["seed"], f"{out['wall']:.0f}s",
                      out["a0_0.5"], out["a1_0.5"], flush=True)
