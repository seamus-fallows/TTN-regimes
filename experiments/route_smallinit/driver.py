"""Step-driven stiff integration of an autonomous ODE with exact time bookkeeping.

Lessons from notes/saddle_dyn.md section 8 built in:
- every recorded state is an accepted solver step (no dense output: scipy's LSODA interpolant is off
  by percents here); events are located afterwards on the step values;
- the solver runs on a local clock that is re-zeroed (new segment) when its step falls below 1e-11 of
  the local time, so fast transitions at late absolute times stay resolved; segment offsets are kept
  as exact fractions, so time differences between any two steps are exact;
- restarts and retries start with a small first step (a huge automatic first step on a flat plateau
  makes the fresh nonstiff solver fail against the O(1) eigenvalues of learned directions);
- a fresh LSODA can stay in nonstiff (Adams) mode with its step pinned at the stability limit of the
  learned directions: if it takes STALL_STEPS consecutive Adams steps with step < 1e-6 of the local
  time, the segment continues with VODE in BDF mode (always stiff), which cannot stall that way.
"""

from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
from scipy.integrate import LSODA, ode

RESTART_H = 1e-2
CLOCK_REL = 1e-11
STALL_STEPS = 3000


class _Vode:
    """VODE (BDF, exact Jacobian) in one-step mode with the attributes the driver reads."""

    def __init__(self, fun, jac, y0, t_bound, rtol, atol, first_step):
        self.r = ode(lambda t, y: fun(t, y), lambda t, y: jac(t, y))
        self.r.set_integrator(
            "vode", method="bdf", rtol=rtol, atol=atol, first_step=first_step, nsteps=10**6
        )
        self.r.set_initial_value(y0, 0.0)
        self.t_bound = t_bound
        self.status = "running"
        self.t, self.y = 0.0, np.array(y0, copy=True)

    def step(self):
        self.r.integrate(self.t_bound, step=True)
        if not self.r.successful():
            self.status = "failed"
            return
        self.t, self.y = self.r.t, np.array(self.r.y, copy=True)
        if self.t >= self.t_bound:
            self.status = "finished"

    @property
    def step_size(self):
        return float(self.r._integrator.rwork[10])

    @property
    def adams(self):
        return False


class _Lsoda:
    def __init__(self, fun, jac, y0, t_bound, rtol, atol, first_step):
        kw = {} if first_step is None else dict(first_step=first_step)
        self.s = LSODA(fun, 0.0, y0, t_bound, rtol=rtol, atol=atol, jac=jac, **kw)
        self.status = "running"
        self.t, self.y = 0.0, np.array(y0, copy=True)

    def step(self):
        self.s.step()
        self.status = self.s.status
        if self.status != "failed":
            self.t, self.y = self.s.t, np.array(self.s.y, copy=True)

    @property
    def step_size(self):
        return float(self.s.step_size or 0.0)

    @property
    def adams(self):
        return int(self.s._lsoda_solver._integrator.iwork[18]) == 1


class Trajectory:
    """Accepted steps: segment index, local time, state, observables. Absolute time of step i is
    offsets[seg[i]] + tloc[i] (exact as a Fraction)."""

    def __init__(self):
        self.seg, self.tloc, self.y, self.obs = [], [], [], []
        self.offsets = [Fraction(0)]

    def append(self, seg, tloc, y, obs):
        self.seg.append(seg)
        self.tloc.append(tloc)
        self.y.append(y)
        self.obs.append(obs)

    def __len__(self):
        return len(self.tloc)

    def exact(self, i) -> Fraction:
        return self.offsets[self.seg[i]] + Fraction(self.tloc[i])

    def t(self) -> np.ndarray:
        return np.array([float(self.exact(i)) for i in range(len(self))])

    def rel(self, ref: Fraction) -> np.ndarray:
        """Times relative to an exact reference time, each rounded once."""
        return np.array([float(self.exact(i) - ref) for i in range(len(self))])


def integrate(
    fun,
    jac,
    y0,
    observe,
    stop,
    rtol=1e-10,
    atol=1e-15,
    t_max=1e40,
    wall=1800.0,
    max_fail=8,
    start="lsoda",
):
    """Integrate dy/dt = fun(y) from y0. observe(y) -> obs (stored per step); stop(traj) -> None or
    a status string ("done", "trapped", ...). Returns (traj, info)."""
    t_wall = time.time()
    traj = Trajectory()
    y0 = np.array(y0, dtype=np.float64)
    traj.append(0, 0.0, y0.copy(), observe(y0))
    f = lambda t, y: fun(y)  # noqa: E731
    jf = lambda t, y: jac(y)  # noqa: E731
    seg = 0
    kind = start
    solver = (_Vode if kind == "vode" else _Lsoda)(f, jf, y0, t_max, rtol, atol, None if kind == "lsoda" else RESTART_H)
    fails, restarts, adams_run, vode_segments = [], 0, 0, 0
    status = None
    while status is None:
        if time.time() - t_wall > wall:
            status = "unfinished_wall"
            break
        solver.step()
        if solver.status == "failed":
            fails.append(float(traj.exact(len(traj) - 1)))
            if len(fails) > max_fail:
                status = "failed"
                break
            seg = _new_segment(traj)
            solver = _Vode(f, jf, traj.y[-1], t_max, rtol, atol, RESTART_H)
            kind, vode_segments = "vode", vode_segments + 1
            continue
        y = solver.y
        traj.append(seg, solver.t, y, observe(y))
        status = stop(traj)
        if status is not None:
            break
        if solver.status == "finished":
            status = "unfinished_tmax"
            break
        adams_run = adams_run + 1 if (solver.adams and solver.step_size < 1e-6 * solver.t) else 0
        if adams_run > STALL_STEPS:  # pinned nonstiff solver on a plateau: continue with BDF
            seg = _new_segment(traj)
            solver = _Vode(f, jf, y, t_max, rtol, atol, RESTART_H)
            kind, vode_segments, adams_run = "vode", vode_segments + 1, 0
        elif solver.t > 0 and solver.step_size < CLOCK_REL * solver.t:
            seg = _new_segment(traj)
            restarts += 1
            solver = (_Vode if kind == "vode" else _Lsoda)(f, jf, y, t_max, rtol, atol, RESTART_H)
    info = dict(
        status=status,
        steps=len(traj) - 1,
        segments=len(traj.offsets),
        restarts=restarts,
        vode_segments=vode_segments,
        fails=len(fails),
        secs=time.time() - t_wall,
    )
    return traj, info


def _new_segment(traj: Trajectory) -> int:
    """Start a new local clock at the last accepted step (which keeps its old segment label)."""
    traj.offsets.append(traj.exact(len(traj) - 1))
    return len(traj.offsets) - 1
