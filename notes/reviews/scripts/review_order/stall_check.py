"""Review check: does the order agent's stall rule (some |a_r| < 1e-25 -> stop, 'stalled') ever
fire on runs that would later learn every component? Integrates GF (LSODA + exact Hessian, the
agent's own functions) WITHOUT the stall stop, records when the rule would have fired, and whether
the component is learned afterwards. Output: JSON lines under data/review_order/."""
import os, sys, json, time
os.environ.setdefault("ORDER_RES", "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/res")
os.environ.setdefault("ORDER_RAW", "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/raw")
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/order")
import numpy as np, torch
torch.set_num_threads(1)
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from common import loss_grad_fn, make_spec, make_teacher, overlaps, unit_init, single_thread


def run(args):
    kind, n, d, chi, R, rho, sigma, seed, t_max = args
    single_thread()
    t0 = time.time()
    spec = make_spec(kind, n, d, chi)
    teacher = make_teacher(n, d, [rho ** r for r in range(R)])
    loss, rhs = loss_grad_fn(spec, teacher)
    y0 = sigma * unit_init(spec, seed)
    l0 = loss(torch.from_numpy(y0)).item()

    def up(r):
        def ev(t, y):
            return overlaps(spec, teacher, y)[r] - 0.5
        ev.direction = 1
        return ev

    def dead(r, floor):
        def ev(t, y):
            return abs(overlaps(spec, teacher, y)[r]) - floor
        ev.direction = -1
        return ev

    def done(t, y):
        with torch.no_grad():
            return loss(torch.from_numpy(y)).item() - 1e-8 * l0
    done.terminal = True

    def jac(t, y):
        return -torch.autograd.functional.hessian(loss, torch.from_numpy(y), vectorize=True).numpy()

    evs = [up(r) for r in range(R)] + [dead(r, 1e-25) for r in range(R)] + [dead(r, 1e-14) for r in range(R)] + [done]
    sol = solve_ivp(rhs, (0.0, t_max), y0, method="LSODA", rtol=1e-9, atol=1e-14, jac=jac, events=evs)
    te = [list(map(float, x)) for x in sol.t_events]
    a_steps = np.array([overlaps(spec, teacher, y) for y in sol.y.T])
    with torch.no_grad():
        l_end = loss(torch.from_numpy(sol.y[:, -1])).item()
    # per-core slice norms of each component at the end (component r = physical index r)
    cores = spec.unflatten(torch.from_numpy(sol.y[:, -1]))
    slice_norms = [[float(c[r].norm()) for c in cores] for r in range(R)]
    rec = dict(kind=kind, n=n, d=d, chi=chi, R=R, rho=rho, sigma=sigma, seed=seed,
               up=te[:R], dead25=te[R:2 * R], dead14=te[2 * R:3 * R], converged=bool(len(te[3 * R])),
               t_end=float(sol.t[-1]), status=int(sol.status), final_a=a_steps[-1].tolist(),
               min_abs_a=np.abs(a_steps).min(0).tolist(), final_loss=l_end, slice_norms_end=slice_norms,
               n_steps=len(sol.t), wall=time.time() - t0)
    return rec


if __name__ == "__main__":
    kind, n, R, rho, sigma = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    seeds = range(int(sys.argv[6]), int(sys.argv[7]))
    procs = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    t_max = float(sys.argv[9]) if len(sys.argv) > 9 else 1e9
    out = f"c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/stall_{kind}{n}_R{R}_rho{rho}_s{sigma}.jsonl"
    jobs = [(kind, n, 4, 4, R, rho, sigma, s, t_max) for s in seeds]
    with Pool(procs) as pool, open(out, "a") as f:
        for rec in pool.imap_unordered(run, jobs):
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(rec["seed"], "up", [x[:1] for x in rec["up"]], "dead25", [x[:1] for x in rec["dead25"]],
                  "conv", rec["converged"], "t_end %.3g" % rec["t_end"], "wall %.0f" % rec["wall"], flush=True)
