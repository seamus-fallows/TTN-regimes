"""Review check v2: GF (LSODA + exact Hessian, agent's functions) WITHOUT the stall stop, integrated in
log-spaced segments with a wall-clock budget per run. At every segment end: overlaps a_r, loss and the
per-core slice norms of every component (MPS: slice = physical index r). Records when |a_r| < 1e-25
(the agent's 'stalled' rule) and whether component r is learned afterwards."""
import os, sys, json, time
os.environ.setdefault("ORDER_RES", "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/claims_audit")
os.environ.setdefault("ORDER_RAW", "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/claims_audit")
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/order")
import numpy as np, torch
torch.set_num_threads(1)
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from common import loss_grad_fn, make_spec, make_teacher, overlaps, unit_init, single_thread

ENDS = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]


def run(args):
    kind, n, d, chi, R, rho, sigma, seed, budget = args
    single_thread()
    t0 = time.time()
    spec = make_spec(kind, n, d, chi)
    teacher = make_teacher(n, d, [rho ** r for r in range(R)])
    loss, rhs = loss_grad_fn(spec, teacher)
    y = sigma * unit_init(spec, seed)
    l0 = loss(torch.from_numpy(y)).item()

    def up(r):
        def ev(t, y):
            return overlaps(spec, teacher, y)[r] - 0.5
        ev.direction = 1
        return ev

    def dead(r):
        def ev(t, y):
            return abs(overlaps(spec, teacher, y)[r]) - 1e-25
        ev.direction = -1
        return ev

    def done(t, y):
        with torch.no_grad():
            return loss(torch.from_numpy(y)).item() - 1e-8 * l0
    done.terminal = True

    def jac(t, y):
        return -torch.autograd.functional.hessian(loss, torch.from_numpy(y), vectorize=True).numpy()

    ups, deads, snaps = [[] for _ in range(R)], [[] for _ in range(R)], []
    t_now, converged, status = 0.0, False, "budget"
    for t_end in ENDS:
        sol = solve_ivp(rhs, (t_now, t_end), y, method="LSODA", rtol=1e-9, atol=1e-14, jac=jac,
                        events=[up(r) for r in range(R)] + [dead(r) for r in range(R)] + [done])
        for r in range(R):
            ups[r] += [float(x) for x in sol.t_events[r]]
            deads[r] += [float(x) for x in sol.t_events[R + r]]
        y, t_now = sol.y[:, -1], float(sol.t[-1])
        cores = spec.unflatten(torch.from_numpy(y))
        with torch.no_grad():
            snaps.append(dict(t=t_now, loss=loss(torch.from_numpy(y)).item(),
                              a=overlaps(spec, teacher, y).tolist(),
                              slices=[[float(c[r].norm()) for c in cores] for r in range(R)]))
        if len(sol.t_events[2 * R]):
            converged, status = True, "converged"
            break
        if sol.status < 0:
            status = "failed"
            break
        if time.time() - t0 > budget:
            break
    else:
        status = "t_max"
    return dict(kind=kind, n=n, d=d, chi=chi, R=R, rho=rho, sigma=sigma, seed=seed, ups=ups,
                dead25=deads, status=status, t_end=t_now, snaps=snaps, wall=time.time() - t0)


if __name__ == "__main__":
    kind, n, R, rho, sigma = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    seeds = [int(s) for s in sys.argv[6].split(",")]
    procs, budget = int(sys.argv[7]), float(sys.argv[8])
    out = f"c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/claims_audit/stall2_{kind}{n}_R{R}_rho{rho}_s{sigma}.jsonl"
    jobs = [(kind, n, 4, 4, R, rho, sigma, s, budget) for s in seeds]
    with Pool(procs) as pool, open(out, "a") as f:
        for rec in pool.imap_unordered(run, jobs):
            f.write(json.dumps(rec) + "\n"); f.flush()
            first = lambda xs: [round(x[0], 1) if x else None for x in xs]
            print(rec["seed"], rec["status"], "up", first(rec["ups"]), "dead25", first(rec["dead25"]),
                  "t_end %.3g wall %.0f" % (rec["t_end"], rec["wall"]), flush=True)
