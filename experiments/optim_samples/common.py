"""Shared helpers for the optim_samples experiments (setup, recording, event extraction).

The GD/SGD loops here are deliberately minimal (written before `ttn.dynamics` landed); every run
stores raw trajectories in /root/outputs/optim_samples/ and a small JSON summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ttn.diagnostics import crossing_times, rise_times
from ttn.losses import centred_population_loss, population_loss
from ttn.network import (
    TTNSpec,
    bond_spectra,
    conserved_quantities,
    sum_difference_cores,
    two_copy_cores,
)
from ttn.teachers import odeco_teacher
from ttn.tree import make_tree

torch.set_default_dtype(torch.float64)
def _path_option(flag: str, default: Path) -> Path:
    """`flag PATH` from the command line, parsed leniently so that scripts keep their own options."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(flag, type=Path, default=default)
    ns, _ = ap.parse_known_args()
    return getattr(ns, flag.lstrip("-").replace("-", "_"))


# Every script takes --raw-dir (raw per-run outputs) and --out-dir (summary tables and figures). The
# default output directory is outside git, so re-running never overwrites the committed results; pass
# --out-dir experiments/optim_samples/results to regenerate them.
# Defaults can also be set with the environment variables OPTIM_SAMPLES_RAW / OPTIM_SAMPLES_OUT.
RAW = _path_option("--raw-dir", Path(os.environ.get("OPTIM_SAMPLES_RAW", "/root/outputs/optim_samples")))
RESULTS = _path_option("--out-dir", Path(os.environ.get("OPTIM_SAMPLES_OUT", str(RAW / "results"))))
RESULTS.mkdir(parents=True, exist_ok=True)
try:  # Greek letters in printed tables must not crash non-UTF-8 consoles
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass


TEACHER_SEED = 101  # independent of init seeds (0-4) and data seeds


def setup(tree: str, n: int, d: int = 3, chi: int = 3, weights=(3.0, 2.0, 1.0),
          teacher_seed=TEACHER_SEED):
    spec = TTNSpec.uniform(make_tree(tree, n), d, chi)
    teacher = odeco_teacher(n, d, list(weights), seed=teacher_seed)
    return spec, teacher


def eta_max(spec, teacher) -> float:
    """GD stability limit at the balanced global minimum, 2 / (K s_1^{2-2/K})."""
    k = spec.n_cores
    return 2.0 / (k * teacher.weights.max().item() ** (2 - 2 / k))


class Problem:
    """Population loss, its gradient and recorded diagnostics for one (spec, teacher).

    With `cores0` given, the centred model is used: loss 1/2 ||W(theta) - W(theta0) - W*||^2
    (cancellation-free), and overlaps / bond spectra refer to the learned function W - W_0.
    """

    def __init__(self, spec, teacher, cores0=None):
        self.spec, self.teacher = spec, teacher
        self.t2 = teacher.norm2()
        self.cores0 = None if cores0 is None else [c.detach() for c in cores0]

    def learned(self, p):
        """(spec, cores) of the learned function: W(theta), or W(theta) - W(theta0)."""
        if self.cores0 is None:
            return self.spec, self.spec.unflatten(p)
        spec2, c2 = two_copy_cores(self.spec, self.spec.unflatten(p), self.cores0)
        return spec2, sum_difference_cores(spec2, c2)

    def loss(self, p):
        if self.cores0 is not None:
            return centred_population_loss(self.spec, self.spec.unflatten(p), self.cores0,
                                           self.teacher, base_norm2=self.t2)
        return population_loss(self.spec, self.spec.unflatten(p), self.teacher, teacher_norm2=self.t2)

    def loss_grad(self, p):
        p = p.detach().requires_grad_(True)
        loss = self.loss(p)
        (g,) = torch.autograd.grad(loss, p)
        return loss.item(), g

    def overlaps(self, p) -> np.ndarray:
        with torch.no_grad():
            return self.teacher.component_overlaps(*self.learned(p)).numpy()

    def sharpness(self, p) -> float:
        h = torch.autograd.functional.hessian(self.loss, p.detach())
        return torch.linalg.eigvalsh(h)[-1].item()

    def q_drift(self, p, q0) -> float:
        """max_e max|Q_e - Q_e(0)| relative to the largest squared core norm (diagnostics.md)."""
        q = conserved_quantities(self.spec, self.spec.unflatten(p))
        scale = max(c.pow(2).sum().item() for c in self.spec.unflatten(p))
        scale += max(v.abs().max().item() for v in q0.values())
        return max((q[v] - q0[v]).abs().max().item() for v in q) / scale

    def q_traces(self, p) -> list[float]:
        q = conserved_quantities(self.spec, self.spec.unflatten(p))
        return [q[v].trace().item() for v in sorted(q)]

    def spectra(self, p) -> dict[int, list[float]]:
        with torch.no_grad():
            sp = bond_spectra(*self.learned(p))
        return {int(v): s.tolist() for v, s in sp.items()}

    def eff_ranks(self, p, thresh_frac=0.1) -> list[int]:
        """#{k : s_k^e > 0.1 x smallest teacher Schmidt value} per edge (diagnostics.md)."""
        thr = thresh_frac * self.teacher.weights.min().item()
        return [int(sum(x > thr for x in s)) for _, s in sorted(self.spectra(p).items())]


def record_events(ts: np.ndarray, a: np.ndarray) -> dict:
    """Events of gridded overlap records a (n_t, R) following notes/diagnostics.md.

    t50: the event, i.e. the last upward crossing of 1/2 after which a_r stays >= 1/2 until the end
    of the record (robust to transient excursions; NaN if a_r ends below 1/2); t50_first: the first
    crossing (kept for comparison); n_cross: number of upward crossings of 1/2; t10, t90: the final
    rise (t90 = first time a_r >= 0.9, t10 = last upward crossing of 0.1 before it). Interpolation:
    `ttn.diagnostics.crossing_times`.
    """
    ts, a = np.asarray(ts, float), np.asarray(a, float)
    t10, t90 = rise_times(ts, a, 0.1, 0.9)
    n_cross = [int(np.sum((a[1:, r] >= 0.5) & ~(a[:-1, r] >= 0.5))) + int(a[0, r] >= 0.5)
               for r in range(a.shape[1])]
    return dict(t50=crossing_times(ts, a, 0.5, last=True).tolist(),
                t50_first=crossing_times(ts, a, 0.5).tolist(), n_cross=n_cross,
                t10=t10.tolist(), t90=t90.tolist())


def gradient_flow(prob: Problem, theta0, t_end, rtol=1e-10, atol=1e-14, n_rec=2000,
                  loss_floor=None):
    """Reference GF (DOP853) with exact event location: every upward crossing of a_r = 1/2, 1/10 and
    9/10 is found by the integrator; the event is the last crossing of 1/2 (a_r >= 1/2 at t_end),
    the width the final rise (first 9/10 crossing minus the last 1/10 crossing before it)."""
    r = prob.teacher.n_components

    def rhs(t, y):
        return -prob.loss_grad(torch.from_numpy(y))[1].numpy()

    def make_event(rr, level):
        def ev(t, y):
            return prob.overlaps(torch.from_numpy(y))[rr] - level

        ev.direction = 1
        return ev

    events = [make_event(rr, lv) for lv in (0.5, 0.1, 0.9) for rr in range(r)]
    if loss_floor is not None:  # stop once converged (terminal event on the loss)
        def converged(t, y):
            return prob.loss(torch.from_numpy(y)).item() - loss_floor

        converged.terminal, converged.direction = True, -1
        events.append(converged)
    t_rec = np.unique(np.concatenate([[0.0], np.geomspace(1e-3, t_end, n_rec // 2),
                                      np.linspace(0, t_end, n_rec // 2)]))
    sol = solve_ivp(rhs, (0, t_end), theta0.numpy(), method="DOP853", rtol=rtol, atol=atol,
                    t_eval=t_rec, events=events)
    a_end = prob.overlaps(torch.from_numpy(sol.y[:, -1]))
    out = dict(t50=[], t50_first=[], n_cross=[], t10=[], t90=[], nfev=int(sol.nfev),
               t_end=float(sol.t[-1]))
    for rr in range(r):
        up50, up10, up90 = (np.asarray(sol.t_events[k * r + rr]) for k in range(3))
        out["n_cross"].append(int(len(up50)))
        out["t50_first"].append(float(up50[0]) if len(up50) else float("nan"))
        out["t50"].append(float(up50[-1]) if len(up50) and a_end[rr] >= 0.5 else float("nan"))
        t90 = float(up90[0]) if len(up90) else float("nan")
        before = up10[up10 <= t90] if np.isfinite(t90) else np.array([])
        out["t90"].append(t90)
        out["t10"].append(float(before[-1]) if len(before) else float("nan"))
    return sol, out


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return clean(x.item())
        return x

    path.write_text(json.dumps(clean(obj)))
