"""One saddle-start (H-A) or small-init (H-B) run; results as JSON under /root/outputs/saddle_dyn/.

H-A: theta_0 = theta_S + eps xi (xi iid N(0,1) entries, or fan-in for B8W*), gradient flow to the escape
(L - L_S < -1e-2; times at -1e-4, -1e-3, -1e-2 recorded), then on to 3 t_esc + 100 (plain loss).
H-B: fan-in init at sigma, plain loss until L <= L_S + 1e-4 (visit test), exact orthogonal re-gauge,
then cancellation-free to the escape as in H-A.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sdyn import OUT, THRESH, Model, amplitudes, escape_type, integrate

CELLS = yaml.safe_load((Path(__file__).parent / "configs" / "cells.yaml").read_text())
T_MAX = 1e30


def make_model(cell):
    c = CELLS[cell]
    return Model(c["tree"], c["n"], c["d"], c["chi"], c["s_q"], c["residuals"])


def fanin(model, seed, sigma):
    g = torch.Generator().manual_seed(seed)
    parts = [sigma / math.sqrt(math.prod(sh[:-1])) * torch.randn(sh, generator=g, dtype=torch.float64)
             for sh in model.net.shapes]
    return torch.cat([p.reshape(-1) for p in parts])


def ha_init(model, cell, seed, eps):
    if CELLS[cell]["perturb"] == "fanin":
        xi = fanin(model, 7000 + seed, 1.0)
    else:
        xi = torch.from_numpy(np.random.default_rng(7000 + seed).standard_normal(model.P))
    return model.theta_S + eps * xi


def floor_index(model, sol, t_end, n=200):
    """Phi = max_t int_0^t |g_err,new| ds / |theta_new(t)| (roundoff bound, sdyn.grad_err_bound)."""
    ts = np.concatenate([[0.0], np.logspace(-3, math.log10(max(t_end, 1e-2)), n)])
    ts = ts[ts <= t_end]
    errs, news = [], []
    for t in ts:
        th = torch.from_numpy(sol.sol(t))
        E, _ = model.residual(th)
        ge = model.grad_err_bound(th, E)
        errs.append(float(ge[model.new_mask].norm()))
        news.append(float(th[model.new_mask].norm()))
    errs, news = np.array(errs), np.array(news)
    acc = np.concatenate([[0.0], np.cumsum(0.5 * (errs[1:] + errs[:-1]) * np.diff(ts))])
    ratio = acc / np.maximum(news, 1e-300)
    return float(ratio.max()), float(news.min())


def trajectory(model, sol, t_end, n=120):
    ts = np.concatenate([[0.0], np.logspace(-2, math.log10(max(t_end, 1e-1)), n)])
    ts = ts[ts <= t_end]
    rows = []
    for t in ts:
        th = torch.from_numpy(sol.sol(t))
        E, dL = model.residual(th)
        W = E + model.W_star
        a = amplitudes(model, W)
        a.pop("top_other")
        rows.append(dict(t=float(t), dL=dL, **a))
    return rows


def stopped_record(model, info):
    """Diagnostics of a run stopped by the wall clock or the trap rule (state at the last RHS call)."""
    y = torch.from_numpy(info.pop("y_last"))
    E, dL = model.residual(y)
    out = dict(t_last=info["t_last"], dL_last=dL, new_norm_last=float(y[model.new_mask].norm()))
    if model.ref is not None:
        D = model.delta(y)
        out["delta_r_last"] = [float(D[p]) for p, _ in model.residuals]
        out["delta_split_last"] = [float(D[p]) for p in model.split_patterns]
    return out


def escape_record(model, sol, info, t0=0.0):
    """Escape times, amplitudes and type at the -1e-3 crossing."""
    rec = {}
    for thr, te in zip(THRESH, sol.t_events[: len(THRESH)]):
        rec[f"t_esc_{thr:g}"] = float(te[0]) - t0 if len(te) else None
    t3 = rec["t_esc_0.001"]
    if t3 is not None:
        th = torch.from_numpy(sol.sol(t3 + t0))
        E, dL = model.residual(th)
        amps = amplitudes(model, E + model.W_star)
        rec["amps_esc"] = amps
        rec["esc_type"], rec["esc_ratio"] = escape_type(model, amps)
    return rec


def continue_after(model, y, t_from, t_to, rtol, atol):
    model.set_ref(None)
    sol, info = integrate(model, y, t_to - t_from, rtol, atol, wall=1200.0)
    if sol is None or info["status"] != "ok":
        return dict(status=info["status"])
    yt = torch.from_numpy(sol.y[:, -1])
    E, dL = model.residual(yt)
    a = amplitudes(model, E + model.W_star)
    return dict(status="ok", t=t_to, dL=dL, L=dL + model.L_S, amps=a, q_forgotten=bool(a["q"] < 0.1 * model.s_q))


def run_ha(cell, seed, eps, rtol=1e-10, atol=1e-15, plain=False, wall=3600.0, follow=True):
    model = make_model(cell)
    y0 = ha_init(model, cell, seed, eps)
    model.set_ref(None if plain else model.theta_S)
    Q0 = model.conserved(y0)
    sol, info = integrate(model, y0.numpy(), T_MAX, rtol, atol, wall=wall, stop_after=1e-2)
    rec = dict(kind="HA", cell=cell, seed=seed, eps=eps, rtol=rtol, atol=atol, plain=plain, k_star=model.k_star,
               order=CELLS[cell]["order"], L_S=model.L_S)
    if sol is None:
        rec.update(stopped_record(model, info))
        rec.update(info)
        rec["escaped"] = False
        return rec
    rec.update(info)
    rec.update(escape_record(model, sol, info))
    t_end = float(sol.t[-1])
    rec["t_end"] = t_end
    rec["escaped"] = rec["t_esc_0.001"] is not None
    model.set_ref(None if plain else model.theta_S)
    y_end = torch.from_numpy(sol.y[:, -1])
    rec["cons_drift"] = float((model.conserved(y_end) - Q0).abs().max())
    t_floor = rec["t_esc_0.001"] if rec["escaped"] else t_end
    rec["floor_phi"], rec["min_new_norm"] = floor_index(model, sol, t_floor)
    rec["traj"] = trajectory(model, sol, t_end)
    if follow and rec["escaped"]:
        rec["after"] = continue_after(model, sol.y[:, -1], t_end, 3 * rec["t_esc_0.001"] + 100, rtol, atol)
    return rec


def hb_init(model, cell, seed, sigma):
    base = 12345 if cell in ("PS08", "PS06") else 5000  # PS: the verifier's gf_origin.py init
    return fanin(model, base + seed, sigma)


def run_hb(cell, seed, sigma, rtol=1e-10, atol=1e-15, wall=3600.0, follow=False):
    """Small init -> (visit test at L <= L_S + 1e-4) -> re-gauge -> cancellation-free escape."""
    model = make_model(cell)
    y0 = hb_init(model, cell, seed, sigma)
    model.set_ref(None)
    t_wall = time.time()
    s_q = model.s_q

    def ev_q(t, y):
        return float(model.W(torch.from_numpy(y))[model.q]) - 0.5 * s_q
    ev_q.direction = 1

    def ev_visit(t, y):
        return model.residual(torch.from_numpy(y))[1] - 1e-4
    ev_visit.terminal = True
    ev_visit.direction = -1

    ev_res = []
    for p, w in model.residuals:
        def ev(t, y, p=p, w=w):
            return abs(float(model.W(torch.from_numpy(y))[p])) - 0.5 * w
        ev.terminal = True
        ev.direction = 1
        ev_res.append(ev)
    sol, info = integrate(model, y0.numpy(), T_MAX, rtol, atol, events_extra=[ev_q, ev_visit] + ev_res, wall=wall,
                          trap=False)
    rec = dict(kind="HB", cell=cell, seed=seed, sigma=sigma, rtol=rtol, atol=atol, k_star=model.k_star,
               order=CELLS[cell]["order"], L_S=model.L_S)
    if sol is None:
        rec.update(stopped_record(model, info))
        rec["phase1"] = info
        rec["status"] = "phase1_" + info["status"]
        return rec
    rec["phase1"] = info
    nT = len(THRESH)
    tq = sol.t_events[nT]
    rec["t_q"] = float(tq[0]) if len(tq) else None
    tv = sol.t_events[nT + 1]
    rec["t_v"] = float(tv[0]) if len(tv) else None
    rec["traj1"] = trajectory(model, sol, float(sol.t[-1]), n=60)
    y = torch.from_numpy(sol.y[:, -1])
    W = model.W(y)
    amps = amplitudes(model, W)
    rec["amps_v"] = amps
    res_hit = [len(sol.t_events[nT + 2 + i]) > 0 for i in range(len(model.residuals))]
    if any(res_hit) or rec["t_v"] is None:
        rec["status"] = "residual_first" if any(res_hit) else ("phase1_" + ("t_max" if info["status"] == "ok" else info["status"]))
        return rec
    visit = amps["q"] > 0.5 * s_q and all(abs(amps["r" + "".join(map(str, p))]) < 0.5 * w for p, w in model.residuals)
    if not visit:
        rec["status"] = "no_visit"
        return rec
    # exact orthogonal gauge map, then cancellation-free continuation
    t_v = rec["t_v"]
    yg = model.regauge(y)
    rec["regauge_err"] = float((model.W(yg) - W).abs().max())
    model.set_ref(model.old_part(yg))
    Q0 = model.conserved(yg)
    sol2, info2 = integrate(model, yg.numpy(), T_MAX, rtol, atol, wall=max(60.0, wall - (time.time() - t_wall)),
                            stop_after=1e-2)
    if sol2 is None:
        rec.update(stopped_record(model, info2))
        rec["phase2"] = info2
        rec["status"] = info2["status"]
        rec["escaped"] = False
        return rec
    rec["phase2"] = info2
    esc = escape_record(model, sol2, info2)
    for k, v in esc.items():
        rec[k] = (v + t_v) if (k.startswith("t_esc") and v is not None) else v
    rec["escaped"] = rec["t_esc_0.001"] is not None
    rec["status"] = "escaped" if rec["escaped"] else "no_escape"
    rec["dt"] = rec["t_esc_0.001"] - rec["t_q"] if rec["escaped"] and rec["t_q"] is not None else None
    t2 = float(sol2.t[-1])
    rec["cons_drift"] = float((model.conserved(torch.from_numpy(sol2.y[:, -1])) - Q0).abs().max())
    t_floor = esc["t_esc_0.001"] if rec["escaped"] else t2
    rec["floor_phi"], rec["min_new_norm"] = floor_index(model, sol2, t_floor)
    rec["traj2"] = trajectory(model, sol2, t2, n=80)
    for r in rec["traj2"]:
        r["t"] += t_v
    if follow and rec["escaped"]:
        rec["after"] = continue_after(model, sol2.y[:, -1], t2, 3 * esc["t_esc_0.001"] + 100, rtol, atol)
    return rec


def out_path(kind, cell, seed, x, tag=""):
    return OUT / kind / cell / f"s{seed:03d}_x{x:.6g}{('_' + tag) if tag else ''}.json"


def save(rec, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    tmp.rename(path)
