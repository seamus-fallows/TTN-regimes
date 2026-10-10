"""Follow ONE curve to smaller scales: optimize at t0, split delta into new entries n and old entries o,
then evaluate L - L_S along theta_S + (tau/t0) n + (tau/t0)^2 o (curved), theta_S + (tau/t0)(n + o)
(straight) and theta_S + (tau/t0) n (new only). Usage: fixed_curve.py CASE [CASE ...]"""
import json, sys, pathlib
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from order_curves import setup, make_curve, starts, run_opt, slopes, OUT

T0 = 0.02
TAUS = [0.02, 0.014, 0.01, 0.007, 0.005]

for name in sys.argv[1:]:
    net, ps, G, core, old, p_pred, c_pred = setup(name)
    curve = make_curve(ps, G, 2)
    rng = np.random.default_rng(7)
    best = min((run_opt(ps, curve, x0, T0, T0**ps.k_star) for _, x0 in starts(rng, net, core, old, 2, 12)),
               key=lambda z: z[0])
    delta = curve(torch.tensor(best[1]), T0)[0] - ps.theta_S
    o = delta * torch.tensor(old, dtype=delta.dtype)
    n = delta - o
    out = dict(name=name, t0=T0, m_t0=best[0], new_norm=float(n.norm()), old_norm=float(o.norm()), tau=TAUS)
    for label, f in [("curved", lambda s: s * n + s**2 * o), ("straight", lambda s: s * (n + o)), ("new_only", lambda s: s * n)]:
        ms = [ps.dL(ps.theta_S + f(tau / T0)).item() for tau in TAUS]
        out[label] = dict(m=ms, m_over_tpred=[m / tau**p_pred for m, tau in zip(ms, TAUS)],
                          local_slopes=slopes(np.array(TAUS), -np.array(ms)) if all(m < 0 for m in ms) else None)
        print(name, label, ["%.3e" % m for m in ms], "slopes", np.round(out[label]["local_slopes"], 3) if out[label]["local_slopes"] else "sign change", flush=True)
    (OUT / f"fixed_{name}.json").write_text(json.dumps(out, indent=1))
