"""One escape run: gradient flow from theta_J + eps xi (notes/route_escape.md section 3).

Kinds of perturbation xi (iid N(0, 1) draws z from numpy.random.default_rng(seed)):
  A     dormant entries of M* set to eps |z| (positive), everything else at theta_J;
  A_bal dormant entries of M* all equal to eps (the literal setting of eq. 6.16);
  B     dormant entries of M* set to eps z (random signs);
  C     eps z on every parameter (the realistic case).
The same z (per kind and direction) is used at every eps. For A, A_bal and B the reduced flow on M*
(Theorem 7.7) is integrated from the same start as well, and the isolated-route prediction
(eqs. 6.14-6.16 with the start's own dormant factors) is computed for every missing pattern.

Usage: uv run python experiments/route_escape/escape.py CELL KIND DIRECTION LOG10_EPS  (one run, prints)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, FullModel, ReducedModel, get_cell, isolated_route_time, leading_616  # noqa: E402
from driver import Flow, integrate  # noqa: E402

from ttn.routes import effective_depth  # noqa: E402

torch.set_num_threads(1)
KINDS = {"A": 0, "A_bal": 1, "B": 2, "C": 3, "Cp": 4}
# Cp (added after the first sweep, notes/route_escape.md section 7): the direction of C with the same
# seed, projected onto M* (entries outside M* left at theta_J). A causal test of shifts in C.
RTOL, ATOL = 1e-10, 1e-15
SWITCH_AT = 200.0  # reference switch to the active part of the relaxed state (sdyn deviation 6/7)


class FullFlow(Flow):
    def __init__(self, model: FullModel):
        self.m = model

    def evaluate(self, y):
        th = torch.from_numpy(y)
        g, dL, D = self.m.grad(th)
        return -g.numpy(), dL, self.m.amplitudes_from_delta(D)

    def jac(self, y):
        return -self.m.hessian(torch.from_numpy(y)).numpy()

    def switch(self, y):
        self.m.set_ref(self.m.active_part(torch.from_numpy(y)))


class ReducedFlow(Flow):
    def __init__(self, model: ReducedModel):
        self.m = model

    def evaluate(self, g):
        return self.m.velocity(g), self.m.dL(g), self.m.man.amplitudes(g)

    def jac(self, g):
        return self.m.jacobian(g)


def direction(cell, kind: str, i: int) -> np.ndarray:
    """Flat perturbation xi (eps = 1) of the given kind and direction index."""
    rng = np.random.default_rng(1000 * KINDS["C" if kind == "Cp" else kind] + i)
    xi = np.zeros(cell.spec.n_params)
    nd = int(cell.dormant_flat.sum())
    if kind == "C":
        xi = rng.standard_normal(cell.spec.n_params)
    elif kind == "Cp":
        xi = rng.standard_normal(cell.spec.n_params) * cell.mask
    elif kind == "A":
        xi[cell.man.flat[cell.dormant]] = np.abs(rng.standard_normal(nd))
    elif kind == "A_bal":
        xi[cell.man.flat[cell.dormant]] = 1.0
    elif kind == "B":
        xi[cell.man.flat[cell.dormant]] = rng.standard_normal(nd)
    return xi


def summarize_crossings(res, missing):
    learn = [c for c in res["cross"] if c["kind"] == "learn"]
    first = learn[0] if learn else None
    return first


def run(cell_name: str, kind: str, i: int, log_eps: float, wall: float = 1800.0) -> dict:
    t0 = time.time()
    cell = get_cell(cell_name)
    eps = 10.0**log_eps
    xi = direction(cell, kind, i)
    theta0 = cell.theta_J.numpy() + eps * xi
    out = dict(driver="bdf_restart+stall", cell=cell_name, target=cell.name, tree=cell.tree_name, d=cell.d, chi=cell.chi,
               kind=kind, seed=i, log_eps=log_eps, eps=eps, c_star=cell.c_star, L_J=cell.L_J,
               missing=cell.missing, greedy=cell.greedy())
    model = FullModel(cell)
    flow = FullFlow(model)
    nonactive = ~cell.active_flat
    diag = []
    floor = dict(last_t=0.0, last_g=0.0, integral=0.0, phi=0.0, next_logt=-math.inf)

    def checkpoint(t, y, P):
        th = torch.from_numpy(y)
        cores = cell.spec.unflatten(th)
        aligned, quality = cell.man.align(cores, cell.J)
        row = dict(t=t, P=[float(P[r]) for r in cell.missing], align_min=min(quality.values()),
                   norm_new=float(np.linalg.norm(y[nonactive])))
        for r in cell.missing:
            norms = cell.man.route_entry_norms(aligned, cell.J, r)
            row[f"route_{r}"] = {int(v): float(x) for v, x in norms.items()}
        diag.append(row)

    state = dict(next_ck=0.0)

    def on_step(t, y, dL, P):
        # floor index (sdyn 5.3): accumulated roundoff bound on the new entries relative to their size
        if out.get("_escaped"):
            return
        lt = math.log10(max(t, 1e-300))
        if lt >= floor["next_logt"] or dL < -1e-3:
            th = torch.from_numpy(y)
            E, _, _ = model.residual(th)
            gerr = float(model.grad_err_bound(th, E)[torch.from_numpy(nonactive)].norm())
            floor["integral"] += 0.5 * (gerr + floor["last_g"]) * (t - floor["last_t"])
            floor["last_t"], floor["last_g"] = t, gerr
            nn = float(np.linalg.norm(y[nonactive]))
            if nn > 0:
                floor["phi"] = max(floor["phi"], floor["integral"] / nn)
            floor["next_logt"] = lt + 0.02
        if dL < -1e-3:
            out["_escaped"] = True
        if t >= state["next_ck"]:
            checkpoint(t, y, P)
            state["next_ck"] = max(SWITCH_AT, t * 10**0.25) if t < SWITCH_AT else t * 10**0.25

    with threadpool_limits(limits=1, user_api="blas"):
        checkpoint(0.0, theta0, model.amplitudes_from_delta(model.delta(torch.from_numpy(theta0))))
        state["next_ck"] = 1.0
        res = integrate(flow, theta0, weights=cell.weights, J=cell.J, missing=cell.missing,
                        rtol=RTOL, atol=ATOL, wall=wall, switch_at=SWITCH_AT, on_step=on_step)
    out.pop("_escaped", None)
    out.update(full_summary(cell, model, res))
    out["floor_phi"] = floor["phi"]
    out["diag"] = diag
    out["P0"] = res["P"][0].tolist()
    # reduced flow and isolated-route predictions on M*
    if kind in ("A", "A_bal", "B", "Cp"):
        red = ReducedModel(cell)
        g0 = cell.man.g(torch.from_numpy(theta0))
        with threadpool_limits(limits=1, user_api="blas"):
            rres = integrate(ReducedFlow(red), g0, weights=cell.weights, J=cell.J,
                             missing=cell.missing, rtol=RTOL, atol=1e-30, wall=wall)
        rs = reduced_summary(cell, rres)
        out["reduced"] = rs
        iso = {}
        for r in cell.missing:
            b0 = g0[cell.route_dormant(r)]
            iso[r] = dict(
                t_iso=isolated_route_time(b0, cell.A(r), cell.weights[r]),
                t_616=leading_616(eps, cell.depth(r), cell.A(r), cell.weights[r]),
                P0=float(cell.man.amplitudes(g0)[r]),
            )
        out["iso"] = iso
    out["secs"] = time.time() - t0
    return out


def full_summary(cell, model, res) -> dict:
    out = dict(status=res["status"], t_esc=res["t_esc"], n_steps=len(res["t"]) - 1,
               n_stall_switch=res.get("n_stall_switch", 0),
               n_restart=res["n_restart"], fails=res["fails"], solver_secs=res["secs"],
               t_end=float(res["t"][-1]), dL_end=float(res["dL"][-1]), P_end=res["P"][-1].tolist())
    out["esc_other"] = {k: v["t"] for k, v in res["esc"].items()}
    out["crossings"] = res["cross"]
    first = summarize_crossings(res, cell.missing)
    out["first_learned"] = first["pattern"] if first else None
    out["t_learn"] = first["t"] if first else None
    if "1e-03" in res["esc"]:
        e = res["esc"]["1e-03"]
        y = torch.from_numpy(e["y"])
        D = model.delta(y)
        out["offmass_esc"] = model.off_target_mass(y, D)
        out["P_esc"] = e["P"]
        # which missing pattern carries the escape: the one with the largest s_r P_r at escape
        gain = {r: cell.weights[r] * e["P"][r] for r in cell.missing}
        out["esc_pattern"] = max(gain, key=gain.get)
        out["esc_target_share"] = float(sum(max(v, 0) for v in gain.values()) / 1e-3)
    # effective depth of the first learned pattern over the plateau before its event
    if first is not None:
        r = first["pattern"]
        ok = res["t"] <= first["t"]
        est = effective_depth(res["t"][ok], res["P"][ok, r], cell.weights[r])
        out["eff_depth"] = est["depth"] if est else None
        out["eff_depth_points"] = est["points"] if est else 0
    # learned patterns at the end, forgetting
    w = cell.weights
    P = res["P"][-1]
    out["J_end"] = [a for a in range(len(w)) if P[a] > w[a] / 2]
    out["forgotten"] = sorted({c["pattern"] for c in res["cross"] if c["kind"] == "forget"})
    # thinned trajectory (t, dL, P) for figures: keep steps where log t or any P moved by 1 %
    keep, last = [0], 0
    for k in range(1, len(res["t"])):
        dt = math.log10(max(res["t"][k], 1e-300)) - math.log10(max(res["t"][last], 1e-300))
        dP = np.abs(res["P"][k] - res["P"][last]) / np.maximum(np.abs(res["P"][last]), 1e-300)
        if dt > 0.01 or dP.max() > 0.02 or k == len(res["t"]) - 1:
            keep.append(k)
            last = k
    out["traj"] = dict(t=res["t"][keep].tolist(), dL=res["dL"][keep].tolist(),
                       P=res["P"][keep].tolist())
    return out


def reduced_summary(cell, res) -> dict:
    first = summarize_crossings(res, cell.missing)
    out = dict(status=res["status"], t_esc=res["t_esc"], first_learned=first["pattern"] if first else None,
               t_learn=first["t"] if first else None, n_steps=len(res["t"]) - 1,
               P_end=res["P"][-1].tolist(), secs=res["secs"])
    if "1e-03" in res["esc"]:
        out["P_esc"] = res["esc"]["1e-03"]["P"]
    if first is not None:
        r = first["pattern"]
        ok = res["t"] <= first["t"]
        est = effective_depth(res["t"][ok], res["P"][ok, r], cell.weights[r])
        out["eff_depth"] = est["depth"] if est else None
    return out


def path_for(cell: str, kind: str, i: int, log_eps: float) -> Path:
    return OUT / "escape" / cell / f"{kind}_{i:02d}_{log_eps:+.2f}.json"


def save(out: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, default=_json))
    tmp.rename(path)


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    cell, kind, i, le = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    o = run(cell, kind, i, le)
    o.pop("traj"), o.pop("diag")
    print(json.dumps(o, default=_json, indent=1)[:4000])
