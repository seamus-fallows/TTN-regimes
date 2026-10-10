"""Why is the k = 2 gauge-projected lambda_min seed-dependent (-0.44..-0.50) instead of -lambda2 = -0.5?
Hypothesis: the E_11 gauge generator on the route edge at a noisy point is (eps_u, -eps_v) / norm, which
overlaps the route's negative mode (1, 1)/sqrt2 unless eps_u = eps_v; projecting it out leaves
-lambda2 * 2 eps_u eps_v / (eps_u^2 + eps_v^2)."""
import numpy as np, torch
from scipy.integrate import solve_ivp
torch.set_num_threads(1)
from ttn.tree import make_tree
from ttn.network import TTNSpec, aligned_init
from ttn.teachers import pattern_teacher
from ttn.dynamics import Problem
from ttn.diagnostics import gauge_directions, project_out_gauge

tree = make_tree("mps", 4); spec = TTNSpec.uniform(tree, 2, 2)
teacher = pattern_teacher([1.0, 0.5], [[0, 0, 0, 0], [0, 1, 1, 0]], d=2, seed=1000)
prob = Problem(spec, teacher)
u = teacher.factors  # (R, N, d)
sad = aligned_init(spec, u[[0]], torch.ones(4, 1, dtype=torch.float64))
print("core shapes", [tuple(c.shape) for c in sad])
for seed in (0, 1, 2):
    for sigma in (1e-3,):
        noise = spec.init(sigma, seed=seed)
        y0 = spec.flatten([a + b for a, b in zip(sad, noise)]).numpy()
        f = lambda t, y: prob.velocity(torch.from_numpy(y)).numpy()
        jac = lambda t, y: -prob.hessian(torch.from_numpy(y)).numpy()
        sol = solve_ivp(f, (0, 1.0), y0, method="LSODA", jac=jac, rtol=1e-9, atol=1e-14, dense_output=False)
        th = torch.from_numpy(sol.y[:, -1])
        H = prob.hessian(th)
        raw = torch.linalg.eigvalsh(0.5 * (H + H.T)).numpy()
        _, pe = project_out_gauge(H, gauge_directions(spec, th))
        cores = spec.unflatten(th)
        # route entries: core 1 (site 1, child leg from core 0, parent leg to core 2); core 2 (site 2, child leg from 1, parent to 3)
        g1, g2 = cores[1], cores[2]
        # leg order in a core: sites, children, parent (library convention)
        a = float(torch.einsum("i,ijk->jk", u[1, 1], g1)[0, 1])   # phys u2 at site 1, child idx 0, parent idx 1
        b = float(torch.einsum("i,ijk->jk", u[1, 2], g2)[1, 0])   # phys u2 at site 2, child idx 1, parent idx 0
        print(f"seed {seed} sigma {sigma}: raw lam_min {raw.min():.4f}  projected {pe.min().item():.4f}  "
              f"pred -0.5*2ab/(a2+b2) = {-0.5 * 2 * a * b / (a * a + b * b):.4f}  (a, b) = ({a:.2e}, {b:.2e})")
