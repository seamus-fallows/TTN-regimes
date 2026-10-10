"""Independent check of the gauge-projected Hessian claims (H3) on mps4, d = chi = 2, weights (1, 0.5)."""
import json, sys, time
import numpy as np, torch
from scipy.integrate import solve_ivp
torch.set_num_threads(1)
from ttn.tree import make_tree
from ttn.network import TTNSpec, aligned_init
from ttn.teachers import pattern_teacher
from ttn.dynamics import Problem
from ttn.diagnostics import gauge_directions, project_out_gauge

def setup(S, n=4):
    tree = make_tree("mps", n)
    spec = TTNSpec.uniform(tree, 2, 2)
    pats = [[0] * n, [1 if i in S else 0 for i in range(n)]]
    teacher = pattern_teacher([1.0, 0.5], pats, d=2, seed=1000)
    return spec, teacher, Problem(spec, teacher)

def proj_eigs(spec, prob, th):
    _, e = project_out_gauge(prob.hessian(th), gauge_directions(spec, th))
    return e.numpy()

def saddle(spec, teacher):
    amps = torch.ones(spec.n_cores, 1, dtype=torch.float64)
    return aligned_init(spec, teacher.factors[[0]], amps)

out = {}
for S in ([1, 2], [0, 2], [0, 3]):
    spec, teacher, prob = setup(S)
    th = spec.flatten(saddle(spec, teacher))
    e = proj_eigs(spec, prob, th)
    out[f"exact_saddle_S{''.join(map(str,S))}"] = dict(lam_min=float(e.min()), n_neg=int((e < -1e-10).sum()),
        low=[float(x) for x in e[:4]], grad=float(prob.velocity(th).abs().max()))
    print(S, "exact saddle: lowest eigs", np.round(e[:5], 6), "max|grad|", float(prob.velocity(th).abs().max()), flush=True)

def run(S, sigma, seed, t_max=1e9):
    spec, teacher, prob = setup(S)
    sad = saddle(spec, teacher)
    noise = spec.init(sigma, seed=seed)
    y0 = spec.flatten([a + b for a, b in zip(sad, noise)]).numpy()
    def f(t, y): return prob.velocity(torch.from_numpy(y)).numpy()
    def jac(t, y): return -prob.hessian(torch.from_numpy(y)).numpy()
    def ev(t, y): return float(teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y)))[1]) - 0.5
    ev.terminal = True
    sol = solve_ivp(f, (0, t_max), y0, method="LSODA", jac=jac, rtol=1e-9, atol=1e-14, events=[ev])
    t2 = float(sol.t_events[0][0]) if len(sol.t_events[0]) else np.nan
    sp = np.array([np.linalg.norm(f(0, sol.y[:, i])) for i in range(len(sol.t))])
    inside = np.where((sol.t > 0) & (sol.t < t2))[0]
    j = inside[np.argmin(sp[inside])]
    e = proj_eigs(spec, prob, torch.from_numpy(sol.y[:, j]))
    return dict(t2=t2, t_mid=float(sol.t[j]), n_steps=len(sol.t), lam_min=float(e.min()), low=[float(x) for x in e[:3]])

res = []
for S, sigmas in (([1, 2], [1e-4, 1e-3, 1e-2]), ([0, 2], [1e-3, 3e-3, 1e-2])):
    for sigma in sigmas:
        for seed in (0, 1, 2):
            t0 = time.time()
            r = run(S, sigma, seed)
            r.update(S="".join(map(str, S)), sigma=sigma, seed=seed, wall=round(time.time() - t0, 1))
            res.append(r)
            print(r, flush=True)
out["saddle_start_runs"] = res
json.dump(out, open("c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_locality/claims_audit/hess_check.json", "w"), indent=1)
