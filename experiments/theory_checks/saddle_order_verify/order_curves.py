"""Saddle order at theta_S from optimized polynomial curves (independent check).

For each t: m(t) = min over curves theta_S + t a1 + t^2 a2 (deg 2) or theta_S + t a1 (deg 1) of
L - L_S, with ||a1|| = 1, ||a2|| < B, and a1, a2 orthogonal to the gauge-orbit tangent at theta_S.
L-BFGS, random restarts (full / new-only / cluster / new+old) + continuation from the previous t.
Order = slope of log|m| vs log t. Usage: order_curves.py CASE [CASE ...] (see CASES).
"""
import json, sys, time, pathlib
import numpy as np, torch
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vttn import Net, PatternSaddle, gauge_basis, DT

OUT = pathlib.Path(__file__).parent / "results"
# name: (tree, N, residuals, theory-predicted order, predicted leading constant of m/t^p)
r5 = 5 ** -2.5
CASES = {
    "mps4_1100": ("mps", 4, [("1100", 0.8)], 2, -0.8 / 2),
    "mps4_1010": ("mps", 4, [("1010", 0.8)], 3, -0.8 * 3**-1.5),
    "binary4_1010": ("binary", 4, [("1010", 0.8)], 3, -0.8 * 3**-1.5),
    "binary6_100001": ("binary", 6, [("100001", 0.8)], 4, -0.8 / 16),
    "ht4_1100": ("ht", 4, [("1100", 0.8)], 3, -0.8 * 3**-1.5),
    "ht4_1010": ("ht", 4, [("1010", 0.8)], 5, -0.8 * r5),
    "mps5_10001": ("mps", 5, [("10001", 0.8)], 5, -0.8 * r5),
    "binary6_100100": ("binary", 6, [("100100", 0.8)], 5, -0.8 * r5),
    "mps6_pairstar_06": ("mps", 6, [("110001", 0.6), ("110010", 0.6)], 5, -0.6 * r5),
    "mps6_pairstar_08": ("mps", 6, [("110001", 0.8), ("110010", 0.8)], 4, (1 - 2 * 0.64) / 8),
    "mps6_110001_15": ("mps", 6, [("110001", 1.5)], 4, (1 - 1.5**2) / 8),
    "mps5_10001_15": ("mps", 5, [("10001", 1.5)], 2, -(1.5 - 1) / 2),
}
TS = [0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025, 0.0175, 0.0125]
B = 3.0


def setup(name):
    kind, n, res, p_pred, c_pred = CASES[name]
    net = Net(kind, n)
    ps = PatternSaddle(net, [(tuple(int(c) for c in r), w) for r, w in res])
    G = torch.tensor(gauge_basis(net, ps.theta_S))
    core, old, _ = net.entry_info()
    return net, ps, G, core, old, p_pred, c_pred


def make_curve(ps, G, deg):
    P = ps.net.P

    def proj(u):
        return u - G @ (G.T @ u)

    def curve(x, t):
        u1 = proj(x[:P])
        a1 = u1 / u1.norm()
        th = ps.theta_S + t * a1
        a2 = None
        if deg == 2:
            u2 = proj(x[P:])
            u2 = u2 - a1 * (a1 @ u2)  # a2 orthogonal to a1: no radial stretch, ||delta|| = t (1 + O(t^2))
            a2 = B * u2 / torch.sqrt(B**2 + (u2**2).sum())
            th = th + t**2 * a2
        return th, a1, a2

    return curve


def run_opt(ps, curve, x0, t, scale):
    def fun(xn):
        x = torch.tensor(xn, dtype=DT, requires_grad=True)
        f = ps.dL(curve(x, t)[0]) / scale
        f.backward()
        return f.item(), x.grad.numpy().copy()

    r = minimize(fun, x0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=4000, maxfun=10000, ftol=1e-16, gtol=1e-13, maxcor=50))
    return r.fun * scale, r.x


def starts(rng, net, core, old, deg, n):
    P, out = net.P, []
    nbrs = [[x for x in range(net.K) if net.parent[x] == v or net.parent[v] == x] for v in range(net.K)]
    for i in range(n):
        kind = ["full", "new", "cluster", "newold"][i % 4]
        u2 = np.zeros(P)
        if kind == "full":
            u1, u2 = rng.standard_normal(P), 0.3 * rng.standard_normal(P)
        elif kind == "new":
            u1 = rng.standard_normal(P) * (~old)
        elif kind == "cluster":
            v = int(rng.integers(net.K))
            sel = {v} | set(int(c) for c in rng.choice(nbrs[v], size=min(len(nbrs[v]), int(rng.integers(1, 3))), replace=False))
            u1 = rng.standard_normal(P) * ((~old) & np.isin(core, list(sel)))
            u2 = 0.5 * rng.standard_normal(P) * old
        else:
            u1, u2 = rng.standard_normal(P) * (~old), 0.5 * rng.standard_normal(P) * old
        if deg == 1:
            x = u1 + 0.3 * u2
        else:
            x = np.concatenate([u1, u2])
        out.append((kind, x))
    return out


def composition(a, old, core, K):
    a = a.detach().numpy()
    w = a**2
    return dict(old_frac=float(w[old].sum() / w.sum()),
                new_by_core=[round(float(w[(~old) & (core == v)].sum() / w.sum()), 4) for v in range(K)],
                old_by_core=[round(float(w[old & (core == v)].sum() / w.sum()), 4) for v in range(K)])


def slopes(ts, ms):
    return [float(np.log(ms[i] / ms[i - 1]) / np.log(ts[i] / ts[i - 1])) for i in range(1, len(ts))]


def study(name, deg, n_restarts, seed=0):
    net, ps, G, core, old, p_pred, c_pred = setup(name)
    rng = np.random.default_rng(1000 + seed)
    curve = make_curve(ps, G, deg)
    rows, best_x, t0 = [], None, time.time()
    for t in TS:
        scale = t**ps.k_star
        cands = starts(rng, net, core, old, deg, n_restarts)
        if best_x is not None:
            cands.append(("continue", best_x))
        vals = []
        for kind, x0 in cands:
            m, x = run_opt(ps, curve, x0, t, scale)
            vals.append((m, kind, x))
        vals.sort(key=lambda z: z[0])
        m, kind, best_x = vals[0]
        _, a1, a2 = curve(torch.tensor(best_x), t)
        n_close = sum(1 for v in vals if v[0] <= m * 0.99) if m < 0 else 0
        row = dict(t=t, m=m, best_from=kind, n_within_1pct=n_close, n_starts=len(vals),
                   second_best=vals[1][0], a1=composition(a1, old, core, net.K))
        if a2 is not None:
            row["a2_norm"] = float(a2.norm())
            row["a2"] = composition(a2, old, core, net.K) if a2.norm() > 1e-12 else None
        row["r"] = float((curve(torch.tensor(best_x), t)[0] - ps.theta_S).norm())
        rows.append(row)
        print(name, deg, f"t={t:.4f} m={m:.6e} m/t^p={m / t**p_pred:.6f} from={kind} close={n_close}/{len(vals)}"
              f" old_frac={row['a1']['old_frac']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    ts = np.array([r["t"] for r in rows])
    ms = np.array([r["m"] for r in rows])
    sl = slopes(ts, -ms) if np.all(ms < 0) else None
    rs = np.array([r["r"] for r in rows])
    sl_r = slopes(rs, -ms) if np.all(ms < 0) else None
    fit = float(np.polyfit(np.log(ts[-4:]), np.log(-ms[-4:]), 1)[0]) if np.all(ms[-4:] < 0) else None
    # fixed curve: the best (a1, a2) found at the smallest t, followed to smaller t without re-optimizing
    tf = np.array(TS + [0.008, 0.005, 0.003])
    mf = np.array([ps.dL(curve(torch.tensor(best_x), float(t))[0]).item() for t in tf])
    slf = slopes(tf, -mf) if np.all(mf < 0) else None
    out = dict(name=name, deg=deg, tree=net.kind, N=net.n, K=net.K, residuals=ps.residuals,
               k_star=ps.k_star, k_r=ps.k_r, p_pred=p_pred, c_pred=c_pred, rows=rows,
               local_slopes=sl, local_slopes_vs_r=sl_r, fit_last4=fit, c_meas=float(ms[-1] / ts[-1] ** p_pred),
               fixed_curve=dict(t=tf.tolist(), m=mf.tolist(), local_slopes=slf), secs=time.time() - t0)
    (OUT / f"order_{name}_deg{deg}.json").write_text(json.dumps(out, indent=1))
    print(name, deg, "k*", ps.k_star, "pred", p_pred, "slopes", np.round(sl, 3) if sl else None, "fit", fit,
          "c_meas", out["c_meas"], "c_pred", c_pred, "fixed-curve slopes", np.round(slf, 3) if slf else None, flush=True)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, deg = arg.split(":") if ":" in arg else (arg, "2")
        study(name, int(deg), 12 if deg == "2" else 8)
