"""Raw vs gauge-projected lambda_min at the plateau midpoint (min speed between events), saddle start,
mps4 S = {1,2}, weights (1, s2): the runner's procedure, plus the raw Hessian."""
import json
import numpy as np, torch
from scipy.integrate import solve_ivp
torch.set_num_threads(1)
from ttn.tree import make_tree
from ttn.network import TTNSpec, aligned_init
from ttn.teachers import pattern_teacher
from ttn.dynamics import Problem
from ttn.diagnostics import gauge_directions, project_out_gauge
res = []
for s2 in (0.25, 0.5, 0.75):
    tree = make_tree("mps", 4); spec = TTNSpec.uniform(tree, 2, 2)
    teacher = pattern_teacher([1.0, s2], [[0, 0, 0, 0], [0, 1, 1, 0]], d=2, seed=1000)
    prob = Problem(spec, teacher)
    sad = aligned_init(spec, teacher.factors[[0]], torch.ones(4, 1, dtype=torch.float64))
    for sigma in (1e-4, 1e-2):
        for seed in range(4):
            y0 = spec.flatten([a + b for a, b in zip(sad, spec.init(sigma, seed=seed))]).numpy()
            f = lambda t, y: prob.velocity(torch.from_numpy(y)).numpy()
            jac = lambda t, y: -prob.hessian(torch.from_numpy(y)).numpy()
            ev = lambda t, y: float(teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y)))[1]) - 0.5
            ev.terminal = True
            sol = solve_ivp(f, (0, 1e8), y0, method="LSODA", jac=jac, rtol=1e-9, atol=1e-14, events=[ev])
            t2 = sol.t_events[0][0]
            idx = np.where((sol.t > 0) & (sol.t < t2))[0]
            sp = [np.linalg.norm(f(0, sol.y[:, i])) for i in idx]
            th = torch.from_numpy(sol.y[:, idx[int(np.argmin(sp))]])
            H = prob.hessian(th)
            raw = float(torch.linalg.eigvalsh(0.5 * (H + H.T)).min())
            proj = float(project_out_gauge(H, gauge_directions(spec, th))[1].min())
            res.append(dict(s2=s2, sigma=sigma, seed=seed, raw=raw, proj=proj, lam2=s2))
            print(f"s2 {s2} sigma {sigma:g} seed {seed}: raw {raw:.4f} projected {proj:.4f} (pred {-s2})", flush=True)
json.dump(res, open("c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_locality/claims_audit/hess_raw_vs_proj.json", "w"), indent=1)
