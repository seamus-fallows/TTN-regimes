"""Runner for the locality experiments (library v0 model + scipy LSODA with event location).

Model, teacher, loss, Hessian, conservation laws and gauge projection come from `ttn`. Gradient flow is
integrated with scipy's LSODA (the ODEPACK solver behind `ttn.dynamics.integrate_gf(method="lsoda")`)
through `solve_ivp`, because the experiment needs root-finding of the learning events a_r = 0.1/0.5/0.9:
the second escape can be 1e10 times shorter than the first, which a recording grid cannot resolve.
The vector field and its Jacobian are the library's population loss, differentiated by autograd and
compiled with torch.compile (checked against `Problem.velocity` / `Problem.hessian` at the start of every
run).

usage: uv run python experiments/locality/run.py <config.yaml> <out_prefix> [n_workers]
Each run appends one JSON line to <out_prefix>.part<pid> as soon as it finishes; finished runs are
skipped on restart. Trajectories (every accepted step) go to /root/outputs/locality/traj/.

Case keys: kind, n (tree; custom trees: star3, midroot6), S (two-component teacher differing on S) or
patterns + weights, name, d, chi, start (small | saddle | small_balanced), sigmas, seeds,
teacher_seed (default 1000), rtol (1e-9), atol (1e-14), t_max (1e14), wall_limit (s, 1500).
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
import ttn.network  # noqa: E402
from rules import get_tree  # noqa: E402
from ttn.diagnostics import gauge_directions, project_out_gauge, saddle_type  # noqa: E402
from ttn.dynamics import Problem  # noqa: E402
from ttn.network import TTNSpec, aligned_init, conserved_quantities  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402

OUT = Path("/root/outputs/locality/traj")
LEVELS = (0.1, 0.5, 0.9)

# torch.compile cannot trace the library's cached opt_einsum expressions; while compiling, plain
# torch.einsum computes the same contraction (the path search is compiled away).
_lib_einsum = ttn.network._einsum


def _traceable_einsum(eq, *ops):
    if torch.compiler.is_compiling():
        return torch.einsum(eq, *ops)
    return _lib_einsum(eq, *ops)


ttn.network._einsum = _traceable_einsum


def teacher_patterns(case, n_sites):
    if case.get("patterns") is not None:
        return case["patterns"], case["weights"]
    S = case["S"]
    return [[0] * n_sites, [1 if i in S else 0 for i in range(n_sites)]], case.get("weights", [1.0, 0.5])


_MODELS: dict = {}


def model(kind, n, d, chi, weights, patterns, teacher_seed):
    """(spec, teacher, problem, compiled velocity, compiled Jacobian), cached per worker process."""
    key = json.dumps([kind, n, d, chi, weights, patterns, teacher_seed])
    if key not in _MODELS:
        spec = TTNSpec.uniform(get_tree(kind, n), d, chi)
        teacher = pattern_teacher(weights, patterns, d=d, seed=teacher_seed)
        problem = Problem(spec, teacher)
        t2 = teacher.norm2()
        from ttn.losses import population_loss

        def loss(x):
            return population_loss(spec, spec.unflatten(x), teacher, teacher_norm2=t2)

        grad = torch.compile(torch.func.grad(loss), fullgraph=True, dynamic=False)
        hess = torch.compile(torch.func.hessian(loss), dynamic=False)
        _MODELS[key] = (spec, teacher, problem, grad, hess)
    return _MODELS[key]


def balance_gauge(spec: TTNSpec, cores, sweeps: int = 200, tol: float = 1e-13):
    """Move cores along their gauge orbit (W unchanged) until every Q_e = 0 (balanced, min-norm).

    Per edge v with parent p: P = G_v^T G_v, Q = G_p,(v) G_p,(v)^T; with X = P^-1/2 (P^1/2 Q P^1/2)^1/2
    P^-1/2 (so X P X = Q) and A = X^1/2, the gauge G_v -> G_v A, G_p -> A^-1 G_p makes both Gram
    matrices equal. Sweeping over edges converges to the balanced point of the orbit."""
    t = spec.tree
    cores = [c.clone() for c in cores]

    def sym_pow(m, a):
        w, u = torch.linalg.eigh(0.5 * (m + m.T))
        return (u * w.clamp_min(1e-300) ** a) @ u.T

    for _ in range(sweeps):
        worst = 0.0
        for v in t.edges:
            p, chi = t.parent[v], spec.bond_dims[v]
            leg = len(t.sites[p]) + t.children[p].index(v)
            gv = cores[v].reshape(-1, chi)
            gp = torch.movedim(cores[p], leg, 0).reshape(chi, -1)
            P, Q = gv.T @ gv, gp @ gp.T
            worst = max(worst, float((P - Q).abs().max() / max(P.abs().max(), Q.abs().max())))
            ph, pmh = sym_pow(P, 0.5), sym_pow(P, -0.5)
            X = pmh @ sym_pow(ph @ Q @ ph, 0.5) @ pmh
            A, Ainv = sym_pow(X, 0.5), sym_pow(X, -0.5)
            cores[v] = (gv @ A).reshape(cores[v].shape)
            cores[p] = ttn.network.apply_on_leg(cores[p], Ainv, leg)
        if worst < tol:
            break
    return cores


def initial_point(case, spec, teacher, weights, sigma, seed):
    cores = spec.init(sigma, seed=seed)
    start = case.get("start", "small")
    if start == "saddle":
        learned = case.get("learned", [0])
        amps = torch.tensor([[weights[r] ** (1 / spec.n_cores) for r in learned]] * spec.n_cores,
                            dtype=torch.float64)
        sad = aligned_init(spec, teacher.factors[learned], amps)
        cores = [a + b for a, b in zip(sad, cores)]
    elif start == "small_balanced":
        cores = balance_gauge(spec, cores)
    return spec.flatten(cores)


def run_one(case, sigma):
    kind, n, seed = case["kind"], case.get("n"), case["seed"]
    d, chi = case.get("d", 2), case.get("chi", 2)
    tree = get_tree(kind, n)
    patterns, weights = teacher_patterns(case, tree.n_sites)
    R = len(weights)
    teacher_seed = case.get("teacher_seed", 1000)
    spec, teacher, problem, grad, hess = model(kind, n, d, chi, weights, patterns, teacher_seed)
    y0t = initial_point(case, spec, teacher, weights, sigma, seed)
    y0 = y0t.numpy()
    # the compiled field must be the library's (agreement to roundoff)
    v_lib = problem.velocity(y0t)
    v_c = -grad(y0t)
    assert torch.allclose(v_c, v_lib, rtol=1e-10, atol=1e-30 + 1e-12 * float(v_lib.abs().max()))

    memo = {}

    def overlaps(y):
        key = y.tobytes()
        if key not in memo:
            memo.clear()
            memo[key] = teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y))).numpy()
        return memo[key]

    events, names = [], []
    for r in range(R):
        for lev in LEVELS:
            events.append((lambda r, lev: lambda t, y: overlaps(y)[r] - lev)(r, lev))
            names.append(f"a{r}_{lev}")
    done = lambda t, y: min(overlaps(y)) - 0.97  # noqa: E731
    done.terminal = True
    events.append(done)
    t0 = time.time()
    wall_limit = case.get("wall_limit", 1500)

    def fun(t, y):
        if time.time() - t0 > wall_limit:
            raise TimeoutError
        return -grad(torch.from_numpy(y)).numpy()

    def jac(t, y):
        return -hess(torch.from_numpy(y)).numpy()

    out = dict(name=case.get("name"), kind=kind, n=n, S=case.get("S"), seed=seed, sigma=sigma, d=d,
               chi=chi, weights=weights, patterns=patterns, K=spec.n_cores, start=case.get("start", "small"),
               teacher_seed=teacher_seed, method="LSODA", rtol=case.get("rtol", 1e-9),
               atol=case.get("atol", 1e-14))
    a_init = overlaps(y0)
    try:
        sol = solve_ivp(fun, (0, case.get("t_max", 1e14)), y0, method="LSODA", jac=jac,
                        rtol=out["rtol"], atol=out["atol"], events=events)
    except TimeoutError:
        out.update(status="timeout", wall=time.time() - t0)
        return out, None
    out.update(status=int(sol.status), message=sol.message, wall=time.time() - t0, nfev=int(sol.nfev),
               njev=int(sol.njev), nsteps=len(sol.t), t_end=float(sol.t[-1]))
    out["crossings"] = {nm: [float(x) for x in te] for nm, te in zip(names, sol.t_events[:-1])}
    cores0 = spec.unflatten(torch.from_numpy(y0))
    cores1 = spec.unflatten(torch.from_numpy(sol.y[:, -1]))
    q0, q1 = conserved_quantities(spec, cores0), conserved_quantities(spec, cores1)
    out["Qdrift"] = max(float((q1[v] - q0[v]).abs().max()) for v in q0)
    out["Qinit"] = max(float(q0[v].abs().max()) for v in q0)
    out["a_final"] = overlaps(sol.y[:, -1]).tolist()
    out["L_final"] = float(problem.loss(torch.from_numpy(sol.y[:, -1])))
    out["L0"] = float(problem.loss(y0t))
    # learning events: final upward crossing of 1/2 (components already learned at t = 0 count at 0)
    fin = [(0.0, r) for r in range(R) if a_init[r] > 0.5]
    for r in range(R):
        c = out["crossings"][f"a{r}_0.5"]
        if len(c) % 2 == 1:
            fin.append((c[-1], r))
    fin.sort()
    out["events"] = fin
    # overlaps on the stored steps: float64-floor check for dormant components
    a_steps = np.array([overlaps(sol.y[:, i]) for i in range(len(sol.t))])
    out["dormant_min"] = {}
    for t_ev, r in fin:
        before = sol.t < t_ev
        if before.any():
            out["dormant_min"][str(r)] = float(np.abs(a_steps[before, r]).min())
    plateaus = []
    for (ta, ra), (tb, rb) in zip(fin[:-1], fin[1:]):
        idx = np.where((sol.t > ta) & (sol.t < tb))[0]
        p = dict(after=ra, before=rb, t_a=ta, t_b=tb)
        if len(idx) > 2:
            speed = np.array([float(np.linalg.norm(fun(0, sol.y[:, i]))) for i in idx])
            j = idx[int(np.argmin(speed))]
            th = torch.from_numpy(sol.y[:, j])
            _, eigs = project_out_gauge(problem.hessian(th), gauge_directions(spec, th))
            st = saddle_type(eigs)
            p.update(t_mid=float(sol.t[j]), speed_mid=float(speed.min()),
                     L_mid=float(problem.loss(th)), a_mid=a_steps[j].tolist(),
                     hess_min=st["lambda_min"], hess_max=st["lambda_max"], hess_neg=st["n_negative"],
                     hess_type=st["type"], hess_low=eigs[:4].tolist())
        plateaus.append(p)
    out["plateaus"] = plateaus
    return out, (sol.t, sol.y)


def traj_path(out):
    tag = out.get("name") or ("S" + "".join(map(str, out["S"])))
    tree = get_tree(out["kind"], out.get("n"))
    return OUT / (f"{tag}_{out['start']}_{tree.name}_d{out['d']}_chi{out['chi']}"
                  f"_w{'-'.join(f'{w:g}' for w in out['weights'])}_s{out['sigma']:.4g}_seed{out['seed']}.npz")


def run_key(c, sigma):
    return json.dumps([c.get("name"), c.get("S"), c["kind"], c.get("n"), c.get("start", "small"),
                       c.get("d", 2), c.get("chi", 2), c.get("weights", [1.0, 0.5]),
                       c.get("teacher_seed", 1000), c["seed"], float(sigma)])


def run_task(task):
    path = f"{task['_out']}.part{os.getpid()}"
    t0 = time.time()
    for sigma in task["sigmas"]:
        key = run_key(task, sigma)
        if key in task["_done"]:
            continue
        out, traj = run_one(task, sigma)
        out["_key"] = key
        if traj is not None:
            t, y = traj
            np.savez_compressed(traj_path(out), t=t, y=y.astype(np.float64))
            out["traj"] = str(traj_path(out))
        with open(path, "a") as fh:
            fh.write(json.dumps(out) + "\n")
    return task, time.time() - t0


def load_results(prefix):
    import glob
    recs = []
    for f in sorted(glob.glob(prefix + ".part*")):
        recs += [json.loads(line) for line in open(f)]
    return recs


def load_cases(path):
    text = Path(path).read_text()
    cases = yaml.safe_load(text) if path.endswith((".yaml", ".yml")) else json.loads(text)
    return cases["cases"] if isinstance(cases, dict) else cases


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases(sys.argv[1])
    prefix = sys.argv[2]
    done = {r["_key"] for r in load_results(prefix)}
    tasks = [dict(c, seed=s, _out=prefix, _done=done) for c in cases for s in c["seeds"]]
    tasks.sort(key=lambda c: -c.get("cost", 1))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 44) as pool:
        for task, wall in pool.imap_unordered(run_task, tasks):
            print(task.get("name"), task.get("S"), task["kind"], task.get("n"), task.get("start"),
                  task["seed"], f"{wall:.0f}s", flush=True)
