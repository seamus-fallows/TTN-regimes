"""One saddle-start (H-A) or small-init (H-B) run; results as JSON under /root/outputs/saddle_dyn/.

H-A: theta_0 = theta_S + eps xi (xi iid N(0,1) entries, or fan-in for B8W*), gradient flow to the escape
(L - L_S < -1e-2; times at -1e-4, -1e-3, -1e-2 recorded), then on to 3 t_esc + 100 (plain loss).
H-B: fan-in init at sigma, plain loss until L <= L_S + 1e-4 (visit test), exact orthogonal re-gauge,
then cancellation-free to the escape as in H-A.
All states used below are integration states (step values or exact re-integrations), never
interpolated.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sdyn import OUT, THRESH, Event, Model, amplitudes, escape_type, integrate

CELLS = yaml.safe_load((Path(__file__).parent / "configs" / "cells.yaml").read_text())
T_MAX = 1e30
RELAX = 50.0  # H-B: time between the visit and the re-gauge (old directions relax at O(1) rates)
RELAX_HA = 200.0  # H-A: time from theta_S + eps xi to the re-gauge / re-reference


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


def pick_steps(ts, t_end, n):
    """Indices of steps closest (from below) to n log-spaced times in (0, t_end], plus 0 and the last."""
    idx = {0}
    if t_end > 0:
        grid = np.logspace(math.log10(max(ts[1] if len(ts) > 1 else t_end, 1e-12)), math.log10(t_end), n)
        idx |= set(np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1).tolist())
    last = int(np.searchsorted(ts, t_end, side="right") - 1)
    idx.add(max(last, 0))
    return sorted(idx)


def floor_index(model, res, t_end, n=200):
    """Phi = max_t int_0^t |g_err,new| ds / |theta_new(t)| (roundoff bound, sdyn.grad_err_bound), by the
    trapezoid rule on steps log-spaced in t and in t_end - t (the escape at the end is fast and its
    bound is ~1e10 times the plateau's, so it must be resolved, not spread over a long interval)."""
    idx = set(pick_steps(res.t, t_end, n))
    last = int(np.searchsorted(res.t, t_end, side="right") - 1)
    if last > 0:
        back = t_end - res.t[: last + 1]  # decreasing to 0
        pos = back[back > 0]
        if len(pos):
            grid = np.logspace(math.log10(max(pos.min(), 1e-12)), math.log10(pos.max()), n)
            j = np.searchsorted(-back, -grid)  # first step with t_end - t <= grid value
            idx |= set(np.clip(j, 0, last).tolist())
    idx = sorted(idx)
    ts = res.t[idx]
    errs, news = [], []
    for i in idx:
        th = torch.from_numpy(res.y[:, i])
        E, _ = model.residual(th)
        errs.append(float(model.grad_err_bound(th, E)[model.new_mask].norm()))
        news.append(float(th[model.new_mask].norm()))
    errs, news = np.array(errs), np.array(news)
    acc = np.concatenate([[0.0], np.cumsum(0.5 * (errs[1:] + errs[:-1]) * np.diff(ts))])
    return float(np.max(acc / np.maximum(news, 1e-300))), float(news.min())


def trajectory(model, res, n=120, t_shift=0.0):
    rows = []
    for i in pick_steps(res.t, res.t[-1], n):
        th = torch.from_numpy(res.y[:, i])
        E, dL = model.residual(th)
        a = amplitudes(model, E + model.W_star)
        a.pop("top_other")
        rows.append(dict(t=float(res.t[i]) + t_shift, dL=dL, **a))
    return rows


def last_state_record(model, res):
    y = torch.from_numpy(res.y[:, -1])
    E, dL = model.residual(y)
    out = dict(t_last=float(res.t[-1]), dL_last=dL, new_norm_last=float(y[model.new_mask].norm()))
    if model.ref is not None:
        D = model.delta(y)
        out["delta_r_last"] = [float(D[p]) for p, _ in model.residuals]
        out["delta_split_last"] = [float(D[p]) for p in model.split_patterns]
    return out


def escape_record(model, res, t0=0.0):
    """Escape times (shifted by t0), amplitudes and type at the -1e-3 crossing."""
    rec = {}
    for thr in THRESH:
        te = res.event_t(f"esc_{thr:g}")
        rec[f"t_esc_{thr:g}"] = te + t0 if te is not None else None
    y3 = res.event_y("esc_0.001")
    if y3 is not None:
        E, _ = model.residual(torch.from_numpy(y3))
        amps = amplitudes(model, E + model.W_star)
        rec["amps_esc"] = amps
        rec["esc_type"], rec["esc_ratio"] = escape_type(model, amps)
    return rec


def continue_after(model, y, duration, rtol, atol):
    model.set_ref(None)
    res, info = integrate(model, y, duration, rtol, atol, wall=1200.0, trap=False)
    if res.status != "t_max":
        return dict(status=res.status)
    yt = torch.from_numpy(res.y[:, -1])
    E, dL = model.residual(yt)
    a = amplitudes(model, E + model.W_star)
    return dict(status="ok", duration=duration, dL=dL, L=dL + model.L_S, amps=a,
                q_forgotten=bool(a["q"] < 0.1 * model.s_q))


def escape_phase(model, y, t_start, rtol, atol, wall):
    """Exact orthogonal re-gauge at y (each bond's q-message -> e_0), reference S = the rank-one old
    part there (the saddle point the run has settled on), then the cancellation-free flow to the
    escape. Times are absolute (t_start + local)."""
    yg = model.regauge(torch.from_numpy(np.asarray(y)))
    out = dict(regauge_err=float((model.W(yg) - model.W(torch.from_numpy(np.asarray(y)))).abs().max()))
    model.set_ref(model.old_part(yg))
    out["ref_q_dev"] = float(model.W(model.ref)[model.q]) - model.s_q
    Q0 = model.conserved(yg)
    res, info = integrate(model, yg.numpy(), T_MAX, rtol, atol, wall=wall, stop_after=1e-2)
    out["info"] = info
    out["esc"] = escape_record(model, res, t0=t_start)
    out["last"] = last_state_record(model, res)
    out["last"]["t_last"] += t_start
    out["cons_drift"] = float((model.conserved(torch.from_numpy(res.y[:, -1])) - Q0).abs().max())
    te = out["esc"]["t_esc_0.001"]
    t_floor = (te - t_start) if te is not None else float(res.t[-1])
    out["floor_phi"], out["min_new_norm"] = floor_index(model, res, max(t_floor, 0.0))
    out["traj"] = trajectory(model, res, n=100, t_shift=t_start)
    out["res"] = res
    return out


def merge_escapes(first, second):
    """First crossing of each threshold over two consecutive phases (escape records)."""
    rec = {}
    for thr in THRESH:
        k = f"t_esc_{thr:g}"
        rec[k] = first[k] if first[k] is not None else second[k]
    src = first if first.get("amps_esc") is not None else second
    for k in ("amps_esc", "esc_type", "esc_ratio"):
        rec[k] = src.get(k)
    return rec


def run_ha(cell, seed, eps, rtol=1e-10, atol=1e-15, plain=False, wall=3600.0, follow=True):
    """Phase 1: RELAX_HA time units from theta_S + eps xi (cancellation-free about theta_S, or plain);
    phase 2: re-gauge and re-reference to the saddle point reached, then on to the escape."""
    t_wall = time.time()
    model = make_model(cell)
    y0 = ha_init(model, cell, seed, eps)
    model.set_ref(None if plain else model.theta_S)
    Q0 = model.conserved(y0)
    res1, info1 = integrate(model, y0.numpy(), RELAX_HA, rtol, atol, wall=600.0, stop_after=1e-2, trap=False)
    rec = dict(kind="HA", cell=cell, seed=seed, eps=eps, rtol=rtol, atol=atol, plain=plain, k_star=model.k_star,
               order=CELLS[cell]["order"], L_S=model.L_S, phase1=info1)
    esc1 = escape_record(model, res1)
    drift1 = float((model.conserved(torch.from_numpy(res1.y[:, -1])) - Q0).abs().max())
    if res1.status == "event" or plain:
        # escaped during the relaxation (large eps), or the plain-loss cross-check: no re-reference
        if plain and res1.status == "t_max":
            res2, info2 = integrate(model, res1.y[:, -1], T_MAX, rtol, atol, wall=wall, stop_after=1e-2)
            rec["phase2"] = info2
            esc1 = merge_escapes(esc1, escape_record(model, res2, t0=RELAX_HA))
            rec.update(last_state_record(model, res2))
            rec["t_last"] += RELAX_HA
            res_last = res2
        else:
            rec.update(last_state_record(model, res1))
            res_last = res1
        rec.update(esc1)
        rec["status"] = "escaped" if esc1["t_esc_0.001"] is not None else res_last.status
        rec["escaped"] = esc1["t_esc_0.001"] is not None
        rec["cons_drift"] = drift1
        rec["floor_phi"] = None
        rec["traj"] = trajectory(model, res1, n=60)
        y_end = res_last.y[:, -1]
    else:
        ph = escape_phase(model, res1.y[:, -1], RELAX_HA, rtol, atol, max(60.0, wall - (time.time() - t_wall)))
        rec["phase2"] = ph["info"]
        rec.update(merge_escapes(esc1, ph["esc"]))
        rec.update(ph["last"])
        rec["escaped"] = rec["t_esc_0.001"] is not None
        rec["status"] = "escaped" if rec["escaped"] else ph["info"]["status"]
        rec["cons_drift"] = max(drift1, ph["cons_drift"])
        rec["regauge_err"], rec["ref_q_dev"] = ph["regauge_err"], ph["ref_q_dev"]
        rec["floor_phi"], rec["min_new_norm"] = ph["floor_phi"], ph["min_new_norm"]
        rec["traj"] = trajectory(model, res1, n=30) + ph["traj"]
        y_end = ph["res"].y[:, -1]
    if follow and rec["escaped"] and rec["t_esc_0.01"] is not None:
        rec["after"] = continue_after(model, y_end, 2 * rec["t_esc_0.001"] + 100, rtol, atol)
    return rec


def hb_init(model, cell, seed, sigma):
    base = 12345 if cell in ("PS08", "PS06") else 5000  # PS: the verifier's gf_origin.py init
    return fanin(model, base + seed, sigma)


def run_hb(cell, seed, sigma, rtol=1e-10, atol=1e-15, wall=3600.0, follow=False):
    """Small init -> (visit test at L <= L_S + 1e-4) -> re-gauge -> cancellation-free escape."""
    model = make_model(cell)
    y0 = hb_init(model, cell, seed, sigma)
    model.set_ref(None)
    s_q = model.s_q
    W = lambda y: model.W(torch.from_numpy(y))  # noqa: E731
    extra = [Event("q", lambda t, y: float(W(y)[model.q]) - 0.5 * s_q, +1),
             Event("visit", lambda t, y: model.residual(torch.from_numpy(y))[1] - 1e-4, -1, terminal=True)]
    for p, w in model.residuals:
        extra.append(Event("r" + "".join(map(str, p)), (lambda t, y, p=p, w=w: abs(float(W(y)[p])) - 0.5 * w), +1,
                           terminal=True))
    res, info = integrate(model, y0.numpy(), T_MAX, rtol, atol, events_extra=extra, wall=wall, trap=False)
    rec = dict(kind="HB", cell=cell, seed=seed, sigma=sigma, rtol=rtol, atol=atol, k_star=model.k_star,
               order=CELLS[cell]["order"], L_S=model.L_S, phase1=info)
    rec["t_q"], rec["t_v"] = res.event_t("q"), res.event_t("visit")
    rec["traj1"] = trajectory(model, res, n=60)
    y = torch.from_numpy(res.y[:, -1])
    amps = amplitudes(model, model.W(y))
    rec["amps_v"] = amps
    res_hit = [p for p, _ in model.residuals if res.event_t("r" + "".join(map(str, p))) is not None]
    if res.status != "event":
        rec["status"] = "phase1_" + res.status
        return rec
    if res_hit:
        rec["status"] = "residual_first"
        return rec
    visit = amps["q"] > 0.5 * s_q and all(abs(amps["r" + "".join(map(str, p))]) < 0.5 * w for p, w in model.residuals)
    if not visit:
        rec["status"] = "no_visit"
        return rec
    # relax the learned component for RELAX time units (plain loss) so that the reference below is the
    # converged saddle, then an exact orthogonal gauge map and the cancellation-free continuation
    t_v = rec["t_v"]
    res1b, info1b = integrate(model, res.y[:, -1], RELAX, rtol, atol, wall=600.0, stop_after=1e-2, trap=False)
    rec["phase1b"] = info1b
    esc1b = escape_record(model, res1b, t0=t_v)
    if res1b.status == "event":  # escaped within RELAX of the visit: no plateau to speak of
        rec.update(esc1b)
        rec["escaped"] = True
        rec["status"] = "escaped"
        rec["dt"] = rec["t_esc_0.001"] - rec["t_q"] if rec["t_q"] is not None else None
        rec["floor_phi"] = None
        return rec
    t_g = t_v + RELAX
    ph = escape_phase(model, res1b.y[:, -1], t_g, rtol, atol, max(60.0, wall - info["secs"]))
    rec["phase2"] = ph["info"]
    rec.update(merge_escapes(esc1b, ph["esc"]))
    rec.update(ph["last"])
    rec["escaped"] = rec["t_esc_0.001"] is not None
    rec["status"] = "escaped" if rec["escaped"] else ph["info"]["status"]
    rec["dt"] = rec["t_esc_0.001"] - rec["t_q"] if rec["escaped"] and rec["t_q"] is not None else None
    rec["cons_drift"] = ph["cons_drift"]
    rec["regauge_err"], rec["ref_q_dev"] = ph["regauge_err"], ph["ref_q_dev"]
    rec["floor_phi"], rec["min_new_norm"] = ph["floor_phi"], ph["min_new_norm"]
    rec["traj2"] = ph["traj"]
    if follow and rec["escaped"]:
        rec["after"] = continue_after(model, ph["res"].y[:, -1], 2 * (rec["t_esc_0.001"] - t_v) + 100, rtol, atol)
    return rec
def out_path(kind, cell, seed, x, tag=""):
    return OUT / kind / cell / f"s{seed:03d}_x{x:.6g}{('_' + tag) if tag else ''}.json"


def save(rec, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    tmp.rename(path)
