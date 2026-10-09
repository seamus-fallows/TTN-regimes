import sys, time, json
import numpy as np, torch
from scipy.integrate import solve_ivp
sys.path.insert(0, "experiments/locality/pilot"); sys.path.insert(0, "experiments/locality")
import pilot3
kind, n, S, sigma, seed = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
tree = pilot3.get_tree(kind, n); N = tree.n_sites
patterns = [[0] * N, [1 if i in S else 0 for i in range(N)]]
spec, teacher, loss, grad_c = pilot3._model(kind, n, 2, 2, [1.0, 0.5], patterns, 1000)
hess_c = torch.compile(torch.func.hessian(loss), dynamic=False)
y0 = spec.flatten(spec.init(sigma, seed=seed)).numpy()
x = torch.from_numpy(y0); grad_c(x); hess_c(x)
t0 = time.perf_counter(); [hess_c(x) for _ in range(50)]; print("hessian ms", (time.perf_counter() - t0) / 50 * 1e3)
ov = lambda y: teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y))).numpy()
evs = [(lambda r: (lambda t, y: ov(y)[r] - 0.5))(r) for r in range(2)]
done = lambda t, y: min(ov(y)) - 0.97; done.terminal = True
for method, kw in (("LSODA", dict(jac=lambda t, y: -hess_c(torch.from_numpy(y)).numpy())),
                   ("Radau", dict(jac=lambda t, y: -hess_c(torch.from_numpy(y)).numpy())),
                   ("RK45", {})):
    for rtol, atol in ((1e-9, 1e-14), (1e-10, 1e-15)):
        if method == "RK45" and rtol < 1e-9:
            continue
        t0 = time.time()
        sol = solve_ivp(lambda t, y: -grad_c(torch.from_numpy(y)).numpy(), (0, 1e14), y0, method=method,
                        rtol=rtol, atol=atol, events=evs + [done], **kw)
        print(method, rtol, f"{time.time() - t0:.1f}s", "steps", len(sol.t), "nfev", sol.nfev, "njev", sol.njev,
              "t_a0", sol.t_events[0][:2], "t_a1", sol.t_events[1][:2], flush=True)
