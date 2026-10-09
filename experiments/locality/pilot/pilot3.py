"""Throwaway pilot runner for multi-component pattern teachers (before ttn.dynamics lands).

Gradient flow with scipy solve_ivp (RK45, rtol 1e-9, atol 1e-14) on the flat parameter vector,
compiled autograd gradient of ttn.losses.population_loss. Records every crossing of
a_r = 0.1/0.5/0.9 for every component r, a gauge-projected Hessian at the minimum-speed point of
every plateau between consecutive learning events, and a subsampled trajectory (for bond spectra).

usage: uv run python pilot3.py <cases.json> <out.jsonl> [n_workers]
case: {name, kind, n, patterns, weights, d, chi, sigmas, seeds}
"""
import json
import os
import sys
import time

import numpy as np
import torch
from multiprocessing import Pool
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
torch.backends.opt_einsum.enabled = False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot2 import projected_hessian_eigs  # noqa: E402
from rules import get_tree  # noqa: E402
from ttn.losses import population_loss  # noqa: E402
from ttn.network import TTNSpec, aligned_init, conserved_quantities  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402
import ttn.network  # noqa: E402

# torch.compile cannot trace the library's cached opt_einsum expressions; inside compiled code the
# plain torch.einsum is equivalent (same contraction) and the path search is compiled away.
_lib_einsum = ttn.network._einsum


def _traceable_einsum(eq, *ops):
    if torch.compiler.is_compiling():
        return torch.einsum(eq, *ops)
    return _lib_einsum(eq, *ops)


ttn.network._einsum = _traceable_einsum

OUT = "/root/outputs/locality/pilot3"
LEVELS = (0.1, 0.5, 0.9)


_MODELS: dict = {}


def _model(kind, n, d, chi, weights, patterns, teacher_seed):
    """Spec, teacher, loss and compiled gradient, cached per worker (compiling takes 5-25 s)."""
    key = json.dumps([kind, n, d, chi, weights, patterns, teacher_seed])
    if key not in _MODELS:
        spec = TTNSpec.uniform(get_tree(kind, n), d, chi)
        teacher = pattern_teacher(weights, patterns, d=d, seed=teacher_seed)
        t2 = teacher.norm2()

        def loss(x):
            return population_loss(spec, spec.unflatten(x), teacher, teacher_norm2=t2)

        grad_c = torch.compile(torch.func.grad(loss), fullgraph=True, dynamic=False)
        _MODELS[key] = (spec, teacher, loss, grad_c)
    return _MODELS[key]


def run_case(case):
    kind, n, seed = case["kind"], case.get("n"), case["seed"]
    d, chi, weights = case.get("d", 2), case.get("chi", 2), case.get("weights", [1.0, 0.5])
    patterns = case.get("patterns")
    if patterns is None:  # two-component teacher differing on S
        nn = get_tree(kind, n).n_sites
        patterns = [[0] * nn, [1 if i in case["S"] else 0 for i in range(nn)]]
    R = len(weights)
    tree = get_tree(kind, n)
    teacher_seed = case.get("teacher_seed", 1000)
    spec, teacher, loss, grad_c = _model(kind, n, d, chi, weights, patterns, teacher_seed)
    outs = []
    start = case.get("start", "small")
    for sigma in case["sigmas"]:
        cores0 = spec.init(sigma, seed=seed)
        if start == "saddle":
            # exact balanced saddle with the components in `learned` fitted (one bond index each),
            # plus the usual fan-in noise of scale sigma on every core
            learned = case.get("learned", [0])
            amps = torch.tensor([[weights[r] ** (1 / spec.n_cores) for r in learned]] * spec.n_cores,
                                dtype=torch.float64)
            sad = aligned_init(spec, teacher.factors[learned], amps)
            cores0 = [a + b for a, b in zip(sad, cores0)]
        y0 = spec.flatten(cores0).numpy()
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
        done = lambda t, y: min(overlaps(y)) - 0.97
        done.terminal = True
        events.append(done)
        t0 = time.time()
        wall_limit = case.get("wall_limit", 1500)

        def rhs(t, y):
            if time.time() - t0 > wall_limit:
                raise TimeoutError
            return -grad_c(torch.from_numpy(y)).numpy()

        out = {k: case[k] for k in ("name", "kind", "n", "seed") if k in case}
        out.update(sigma=sigma, d=d, chi=chi, weights=weights, patterns=patterns, K=spec.n_cores,
                   start=start, S=case.get("S"), teacher_seed=teacher_seed)
        a_init = overlaps(y0)
        try:
            sol = solve_ivp(rhs, (0, case.get("tmax", 1e14)), y0, method="RK45", rtol=1e-9,
                            atol=1e-14, events=events)
        except TimeoutError:
            out.update(status="timeout", wall=time.time() - t0)
            outs.append(out)
            continue
        out.update(status=int(sol.status), wall=time.time() - t0, nfev=int(sol.nfev),
                   nsteps=len(sol.t))
        out["crossings"] = {name: [float(x) for x in te] for name, te in zip(names, sol.t_events[:-1])}
        q0 = conserved_quantities(spec, spec.unflatten(torch.from_numpy(y0)))
        q1 = conserved_quantities(spec, spec.unflatten(torch.from_numpy(sol.y[:, -1])))
        out["Qdrift"] = max(float((q1[v] - q0[v]).abs().max()) for v in q0)
        out["a_final"] = overlaps(sol.y[:, -1]).tolist()
        # learning events: final upward crossing of 1/2 per component, in time order
        fin = [(0.0, r) for r in range(R) if a_init[r] > 0.5]  # already learned at the start
        for r in range(R):
            c = out["crossings"][f"a{r}_0.5"]
            if len(c) % 2 == 1:
                fin.append((c[-1], r))
        fin.sort()
        out["events"] = fin
        plateaus = []
        for (ta, ra), (tb, rb) in zip(fin[:-1], fin[1:]):
            idx = np.where((sol.t > ta) & (sol.t < tb))[0]
            p = dict(after=ra, before=rb, t_a=ta, t_b=tb)
            if len(idx) > 2:
                sub = idx[np.linspace(0, len(idx) - 1, min(len(idx), 300)).astype(int)]
                speed = [float(np.linalg.norm(grad_c(torch.from_numpy(sol.y[:, i])).numpy())) for i in sub]
                j = sub[int(np.argmin(speed))]
                ev = projected_hessian_eigs(spec, loss, sol.y[:, j])
                p.update(t_mid=float(sol.t[j]), speed_mid=float(min(speed)),
                         a_mid=overlaps(sol.y[:, j]).tolist(), hess_min=float(ev[0]),
                         hess_max=float(ev[-1]), hess_neg=int((ev < -1e-3 * abs(ev[-1])).sum()),
                         hess_low=ev[:4].tolist())
            plateaus.append(p)
        out["plateaus"] = plateaus
        sub = np.unique(np.linspace(0, len(sol.t) - 1, case.get("n_save", 4000)).astype(int))
        tag = case.get("name") or ("S" + "".join(map(str, case["S"])))
        tag += "_saddle" if start == "saddle" else ""
        np.savez_compressed(f"{OUT}/{tag}_{kind}{tree.n_sites}_chi{chi}_d{d}_s{sigma:.4g}_seed{seed}.npz",
                            t=sol.t[sub], y=sol.y[:, sub])
        outs.append(out)
    return outs


def run_key(c, sigma):
    return json.dumps([c.get("name"), c.get("S"), c["kind"], c.get("n"), c.get("start", "small"),
                       c.get("d", 2), c.get("chi", 2), c.get("weights", [1.0, 0.5]),
                       c.get("teacher_seed", 1000), c["seed"], float(sigma)])


def run_and_store(task):
    """Run one (case, seed) task; append each sigma's result at once to a per-worker part file."""
    path = f"{task['_out']}.part{os.getpid()}"
    todo = [s for s in task["sigmas"] if run_key(task, s) not in task["_done"]]
    if not todo:
        return task, 0.0
    t0 = time.time()
    for sigma in todo:  # one at a time, so finished sigmas are saved even if a later one is slow
        for out in run_case(dict(task, sigmas=[sigma])):
            out["_key"] = run_key(task, sigma)
            with open(path, "a") as fh:
                fh.write(json.dumps(out) + "\n")
    return task, time.time() - t0


def load_results(prefix):
    import glob
    recs = []
    for f in sorted(glob.glob(prefix + ".part*")):
        recs += [json.loads(line) for line in open(f)]
    return recs


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cases = json.load(open(sys.argv[1]))
    out_prefix = sys.argv[2]
    done = {r["_key"] for r in load_results(out_prefix)}
    tasks = [dict(c, seed=s, _out=out_prefix, _done=done) for c in cases for s in c["seeds"]]
    tasks.sort(key=lambda c: -c.get("cost", 1))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 44) as pool:
        for task, wall in pool.imap_unordered(run_and_store, tasks):
            print(task.get("name"), task.get("S"), task["kind"], task.get("n"), task.get("start"),
                  task["seed"], f"{wall:.0f}s", flush=True)
