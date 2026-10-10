"""Gradient flow from small fan-in init on the MPS6 pair-star teacher (independent check).

W* = e_{000000} + s (e_{110001} + e_{110010}), d = chi = 2, s = 0.8 (counterexample) or 0.6 (control).
Fan-in init G_v = sigma Z_v / sqrt(F_v) (same Z for every sigma at a given seed). LSODA, rtol 1e-10.
Events: q half learned (W[q] = 1/2); escape from the q plateau (L = L_S - 1e-3 and L_S - 1e-2).
Usage: gf_origin.py S SEEDS SIGMAS   e.g.  gf_origin.py 0.8 0-4 0.2,0.1,0.05
"""
import json, sys, time, pathlib
import numpy as np, torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vttn import Net, PatternSaddle, DT
from mps_fast import FastMPS

OUT = pathlib.Path(__file__).parent / "results"
PATS = {"q": "000000", "r1": "110001", "r2": "110010", "p110000": "110000", "p000001": "000001",
        "p000010": "000010", "p000011": "000011", "p110011": "110011"}


def amps(W):
    out = {k: float(W[tuple(int(c) for c in v)]) for k, v in PATS.items()}
    rest = W.clone()
    for v in PATS.values():
        rest[tuple(int(c) for c in v)] = 0
    i = int(rest.abs().argmax())
    out["top_other"] = ("".join(str(x) for x in np.unravel_index(i, W.shape)), float(rest.reshape(-1)[i]))
    return out


def run(s, seed, sigma, rtol=1e-10, atol=1e-14):
    net = Net("mps", 6)
    ps = PatternSaddle(net, [((1, 1, 0, 0, 0, 1), s), ((1, 1, 0, 0, 1, 0), s)])
    g = torch.Generator().manual_seed(12345 + seed)
    th0 = torch.cat([(sigma / np.sqrt(np.prod(sh[:-1])) * torch.randn(sh, generator=g, dtype=DT)).reshape(-1)
                     for sh in net.shapes]).numpy()
    nfev = [0]

    fm = FastMPS(net, ps.W_star.numpy())

    def rhs(t, y):
        nfev[0] += 1
        return -fm.loss_grad(y)[1]

    def loss_np(y):
        return fm.loss_grad(y)[0]

    def ev_q(t, y):
        return float(fm.W(y)[0]) - 0.5

    def ev_e3(t, y):
        return loss_np(y) - (ps.L_S - 1e-3)

    def ev_e2(t, y):
        return loss_np(y) - (ps.L_S - 1e-2)

    def ev_conv(t, y):
        return loss_np(y) - 1e-9
    ev_conv.terminal = True
    t_cap = 3e3 * sigma**-4
    t_eval = np.minimum(np.concatenate([[0.0], np.logspace(-1, np.log10(t_cap), 500)]), t_cap)
    t0 = time.time()
    sol = solve_ivp(rhs, (0, t_cap), th0, method="LSODA", rtol=rtol, atol=atol, t_eval=t_eval,
                    events=[ev_q, ev_e3, ev_e2, ev_conv], dense_output=True)
    first = lambda arr: float(arr[0]) if len(arr) else None
    tq, te3, te2, tconv = (first(sol.t_events[i]) for i in range(4))
    rec = dict(s=s, seed=seed, sigma=sigma, L_S=ps.L_S, t_q=tq, t_esc3=te3, t_esc2=te2, t_conv=tconv,
               t_end=float(sol.t_events[3][0]) if sol.status == 1 else float(sol.t[-1]),
               L_end=1e-9 if sol.status == 1 else loss_np(sol.y[:, -1]), status=sol.status, nfev=nfev[0],
               secs=time.time() - t0)
    # state when the q plateau is reached (L first within 0.02 of L_S, from above) and at the escape
    Ls = np.array([loss_np(sol.y[:, i]) for i in range(sol.y.shape[1])])
    rec["traj"] = dict(t=sol.t.tolist(), L=Ls.tolist())
    snaps = {}
    if tq is not None:
        snaps["t_q+"] = tq
    if te3 is not None:
        snaps["esc3"] = te3
        if tq is not None:
            for f in (1.5, 2, 4, 10, 30):
                tt = te3 + (f - 1) * (te3 - tq)
                if tt <= sol.t[-1]:
                    snaps[f"esc3+{f}x"] = tt
    snaps["end"] = rec["t_end"]
    rec["snaps"] = {k: dict(t=v, L=loss_np(sol.sol(v)), amps=amps(net.dense(torch.from_numpy(sol.sol(v)))))
                    for k, v in snaps.items()}
    return rec


if __name__ == "__main__":
    s = float(sys.argv[1])
    a, b = (int(x) for x in sys.argv[2].split("-"))
    sigmas = [float(x) for x in sys.argv[3].split(",")]
    for seed in range(a, b + 1):
        for sigma in sigmas:
            f = OUT / f"gf_s{s}_seed{seed}_sigma{sigma}.json"
            if f.exists():
                continue
            rec = run(s, seed, sigma)
            f.write_text(json.dumps(rec))
            sn = rec["snaps"]
            print(f"s={s} seed={seed} sigma={sigma} t_q={rec['t_q']} t_esc3={rec['t_esc3']} L_end={rec['L_end']:.3g}"
                  f" nfev={rec['nfev']} {rec['secs']:.0f}s", {k: (round(v['L'], 4), {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v['amps'].items()}) for k, v in sn.items() if k in ('t_q+', 'esc3')}, flush=True)
