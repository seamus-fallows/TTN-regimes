"""Step-driven LSODA for gradient flow near a plateau (after experiments/saddle_dyn/sdyn.py).

Lessons from notes/saddle_dyn.md section 8 built in: events are detected on the solver's own steps
and the escape time is refined by bisection with exact re-integrations (scipy's LSODA dense output is
off by percents, deviation 3); the clock is re-zeroed when the step falls below 1e-11 of the local time
(deviation 5, addendum); a failed step restarts from the last good step with a small first step
(deviation 6b); the reference switch happens inside the running solver, never by starting a fresh
LSODA on a plateau (deviation 7); a trap rule stops dead routes (deviation 1).
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import BDF, LSODA

RESTART_H = 1e-2
# Restarted solvers are BDF (always stiff): a restarted LSODA starts in nonstiff (Adams) mode and can
# stay pinned at the stability limit of the learned directions on a plateau (sdyn deviation 7; seen
# here in 3 runs of the first sweep, notes/route_escape.md section 7).
RESTART_STIFF = True
TRAP_DL = 1e-30
TRAP_P = 1e-45


def advance(fun, jac, y0, h, rtol, atol):
    """Integrate exactly over a local time h from y0 with a fresh LSODA (lands on h)."""
    if h <= 0:
        return np.array(y0, copy=True)
    s = LSODA(fun, 0.0, y0, h, rtol=rtol, atol=atol, jac=jac)
    while s.status == "running":
        s.step()
    if s.status != "finished":
        raise RuntimeError("LSODA failed in advance")
    return np.array(s.y, copy=True)


class Flow:
    """What the driver integrates. `evaluate(y)` returns (velocity, L - L_J, amplitudes P)."""

    def evaluate(self, y):
        raise NotImplementedError

    def jac(self, y):
        raise NotImplementedError

    def switch(self, y):  # optional reference switch inside the solver
        pass


def integrate(flow: Flow, y0, *, weights, J, missing, t_max=1e25, rtol=1e-10, atol=1e-15,
              thr=1e-3, wall=1800.0, switch_at=None, after_learn=200.0, on_step=None,
              trap=True):
    """Gradient flow from y0 until the first missing pattern is learned (+ `after_learn`), a trap,
    t_max or the wall clock. Records the escape times (L - L_J < -1e-4, -thr, -1e-2; -thr refined by
    bisection), every crossing of P_r = s_r/2 (missing r, up and down) and P_a = s_a/4 (learned a,
    down and up), and the solver steps (t, L - L_J, P)."""
    t_wall = time.time()
    cache: dict = {}
    weights = np.asarray(weights, dtype=float)
    J, missing = list(J), list(missing)

    def ev(y):
        key = y.tobytes()
        if key not in cache:
            if len(cache) > 32:
                cache.clear()
            cache[key] = flow.evaluate(y)
        return cache[key]

    def rhs(t, y):
        return ev(y)[0]

    def jac(t, y):
        return flow.jac(y)

    res = dict(t=[0.0], dL=[], P=[], esc={}, cross=[], status="running", n_restart=0, fails=[],
               switch_t=None)
    y0 = np.array(y0, dtype=float)
    v0, dL0, P0 = ev(y0)
    res["dL"].append(dL0)
    res["P"].append(P0)
    ys_last = y0.copy()
    esc_levels = {"1e-04": 1e-4, "1e-03": thr, "1e-02": 1e-2}
    half = {r: weights[r] / 2 for r in missing}
    quarter = {a: weights[a] / 4 for a in J}
    off, t_loc_old = 0.0, 0.0
    solver = LSODA(rhs, 0.0, y0, t_max, rtol=rtol, atol=atol, jac=jac)
    nxt, count = 1e6, 0
    t_learn = None
    t_esc = None
    while True:
        if time.time() - t_wall > wall:
            res["status"] = "wall"
            break
        msg = solver.step()
        if solver.status == "failed":
            res["fails"].append((off + t_loc_old, str(msg)))
            res["solver"] = "BDF" if RESTART_STIFF else "LSODA"
            if len(res["fails"]) > 5:
                res["status"] = "failed"
                break
            off += t_loc_old
            t_loc_old = 0.0
            solver = (BDF if RESTART_STIFF else LSODA)(rhs, 0.0, ys_last, t_max - off, rtol=rtol,
                                                        atol=atol, jac=jac, first_step=RESTART_H)
            continue
        t_loc, y = solver.t, np.array(solver.y, copy=True)
        t = off + t_loc
        _, dL, P = ev(y)
        dL_old, P_old, t_old = res["dL"][-1], res["P"][-1], res["t"][-1]
        # escape events (down-crossings of L - L_J = -level)
        for name, level in esc_levels.items():
            if name in res["esc"] or not (dL_old > -level >= dL):
                continue
            if name == "1e-03":
                ta, ya, ga = t_loc_old, ys_last, dL_old + level
                tb = t_loc
                for _ in range(60):
                    if tb - ta <= 1e-10 * max(abs(off + tb), 1.0):
                        break
                    tm = 0.5 * (ta + tb)
                    ym = advance(rhs, jac, ya, tm - ta, rtol, atol)
                    gm = ev(ym)[1] + level
                    if gm <= 0:
                        tb = tm
                    else:
                        ta, ya, ga = tm, ym, gm
                yr = advance(rhs, jac, ya, tb - ta, rtol, atol)
                _, dLr, Pr = ev(yr)
                res["esc"][name] = dict(t=off + tb, P=Pr.tolist(), y=yr)
            else:
                f = (dL_old + level) / (dL_old - dL)
                res["esc"][name] = dict(t=t_old + f * (t - t_old))
        if "1e-03" in res["esc"] and t_esc is None:
            t_esc = res["esc"]["1e-03"]["t"]
        # learning / forgetting crossings (linear interpolation between steps)
        for r in missing:
            for up in (True, False):
                a0, a1 = P_old[r] - half[r], P[r] - half[r]
                if (a0 < 0 <= a1) if up else (a0 >= 0 > a1):
                    f = a0 / (a0 - a1)
                    res["cross"].append(dict(kind="learn" if up else "unlearn", pattern=r,
                                             t=t_old + f * (t - t_old)))
                    if up and t_learn is None:
                        t_learn = t
        for a in J:
            for up in (False, True):
                a0, a1 = P_old[a] - quarter[a], P[a] - quarter[a]
                if (a0 >= 0 > a1) if not up else (a0 < 0 <= a1):
                    f = a0 / (a0 - a1)
                    res["cross"].append(dict(kind="forget" if not up else "relearn", pattern=a,
                                             t=t_old + f * (t - t_old)))
        res["t"].append(t)
        res["dL"].append(dL)
        res["P"].append(P)
        if on_step is not None:
            on_step(t, y, dL, P)
        ys_last = y
        t_loc_old = t_loc
        if switch_at is not None and res["switch_t"] is None and t >= switch_at:
            flow.switch(y)
            cache.clear()
            res["switch_t"] = t
            _, dLs, Ps = ev(y)
            res["dL"][-1], res["P"][-1] = dLs, Ps
        if t_learn is not None and t >= t_learn + after_learn:
            res["status"] = "learned"
            break
        if t_esc is not None and t_learn is None and t >= 10 * t_esc + 1000:
            res["status"] = "escaped_no_learn"
            break
        if solver.status == "finished":
            res["status"] = "t_max"
            break
        if t_loc > 0 and solver.step_size < 1e-11 * t_loc:
            off += t_loc
            t_loc_old = 0.0
            res["n_restart"] += 1
            solver = (BDF if RESTART_STIFF else LSODA)(rhs, 0.0, y, t_max - off, rtol=rtol, atol=atol,
                                                        jac=jac, first_step=RESTART_H)
        if trap and t >= nxt and t_esc is None:
            nxt = t * 10**0.1
            dead = max(abs(P[r]) for r in missing) < TRAP_P
            count = count + 1 if (0 < dL < TRAP_DL or dead) else 0
            if count >= 10:
                res["status"] = "trapped"
                break
    res["t"] = np.array(res["t"])
    res["dL"] = np.array(res["dL"])
    res["P"] = np.array(res["P"])
    res["y_end"] = ys_last
    res["secs"] = time.time() - t_wall
    res["t_esc"] = t_esc
    return res
