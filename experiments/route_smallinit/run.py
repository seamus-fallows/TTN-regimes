"""One gradient-flow run from small random init (full TTN) or on the aligned manifold (reference),
and its reduction to learning events and plateau diagnostics.

Definitions (brief sec. 3, used verbatim): amplitude P_a = <U_a, W>; learned set J(t) = {a: P_a >
s_a/2}; learning event of a = its last upward crossing of s_a/2; forgotten = falls below s_a/4 after
an upward s_a/2 crossing; plateau k = between learning events k and k+1; escape time from a plateau
start = first time L < L_J - 1e-3.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import scipy.integrate
import torch

from driver import integrate
from model import Model, build_target
from ttn.routes import AlignedManifold, effective_depth

torch.set_num_threads(1)

LEVELS = (0.1, 0.25, 0.5, 0.9)
ENTRY_TOL = 0.01  # plateau entry: every learned amplitude within 1% of its weight
DONE_LOSS = 1e-8
DEAD_AMP = 1e-35  # trap rule: every unlearned |P_r|/s_r below this for a decade of time
FLOOR_RATIO = 100.0  # floor flag: escape signal within 100x of its roundoff bound


# ---------------------------------------------------------------------------------- predictions


@lru_cache(maxsize=None)
def _manifold(key):
    cfg = dict(key)
    return AlignedManifold(build_target(cfg), int(cfg["d"]))


def _key(cfg):
    return tuple(
        (k, tuple(v) if isinstance(v, list) else v)
        for k, v in sorted(cfg.items())
        if k in ("tree", "n_sites", "d", "patterns", "weights")
    )


@lru_cache(maxsize=None)
def _balanced(key, J: tuple):
    """Balanced partial-target entries theta_J (relax_from = 1e-2, as small-init flow)."""
    return _manifold(key).partial_target(list(J), relax_from=1e-2)


def gamma(cfg, J, r) -> float:
    """A_r(J): product of the active entries of r at the balanced theta_J (eq. C.3)."""
    man = _manifold(_key(cfg))
    if not J:
        return 1.0
    g = _balanced(_key(cfg), tuple(sorted(J)))
    region = man.target.region(J, r)
    return float(np.prod([g[man.route[r, v]] for v in range(man.target.tree.n_nodes) if v not in region]))


def predictions(cfg, target, J) -> dict:
    """Next pattern by the greedy rule (depth, then s_r A_r(J)), the pairwise (Prim) rule (min over
    a in J of |St(V_ar)|, then weight) and weight order; plus the library's greedy_next."""
    J = sorted(J)
    miss = target.missing(J)
    w = target.weights
    depth = {r: target.depth(J, r) for r in miss}
    pair = {r: (min(target.pair_cost(a, r) for a in J) if J else target.tree.n_nodes) for r in miss}
    coef = {r: w[r] * gamma(cfg, J, r) for r in miss}
    greedy = min(miss, key=lambda r: (depth[r], -coef[r]))
    pairwise = min(miss, key=lambda r: (pair[r], -w[r]))
    weight = max(miss, key=lambda r: w[r])
    lib = target.greedy_next(J)[0]["component"]
    tie = sum(depth[r] == depth[greedy] for r in miss) > 1
    return dict(
        depth=depth, pair=pair, coef=coef, greedy=greedy, pairwise=pairwise, weight=weight,
        lib_greedy=lib, tie=tie,
    )


def route_time(p0: float, c: int, g: float, s: float, level: float = 0.5) -> float:
    """Time for the isolated balanced route dy/dt = c g^(2/c) y^(2-2/c) (s - y) (eq. 6.15) to rise
    from y = p0 to level * s. inf if p0 <= 0 (the route must first change sign)."""
    if not (p0 > 0):
        return math.inf
    if p0 >= level * s:
        return 0.0
    # in u = log y: dt = du / (c g^(2/c) y^(1-2/c) (s - y))
    f = lambda u: 1.0 / (c * g ** (2 / c) * math.exp(u * (1 - 2 / c)) * (s - math.exp(u)))  # noqa
    val, _ = scipy.integrate.quad(f, math.log(p0), math.log(level * s), limit=200, epsrel=1e-10)
    return float(val)


# ----------------------------------------------------------------------------------- runs


def run_full(cfg, sigma, seed, rtol=1e-10, atol=1e-15, wall=1800.0, arith="dense"):
    model = Model(cfg, arith=arith)
    theta0 = model.spec.flatten(model.spec.init(sigma, seed=seed))

    def fun(y):
        return model.evaluate(torch.from_numpy(y))[2].numpy()

    def jac(y):
        return -model.hessian(torch.from_numpy(y)).numpy()

    def observe(y):
        W, E, v = model.evaluate(torch.from_numpy(y))
        L, P, off = model.loss_terms(W, E)
        return (L, P, off, float(v.norm()))

    traj, info = integrate(fun, jac, theta0.numpy(), observe, _stopper(model.s), rtol, atol, wall=wall)
    return model, traj, info


def run_reference(cfg, sigma, rtol=1e-10, atol=1e-30, wall=1800.0):
    """Reduced flow on M* from uniform entries sigma (Theorem 7.7)."""
    man = _manifold(_key(cfg))
    s = man.weights

    def observe(g):
        P = man.amplitudes(g)
        return (0.5 * float(((P - s) ** 2).sum()), P, 0.0, float(np.linalg.norm(man.velocity(g))))

    g0 = np.full(man.n_entries, float(sigma))
    traj, info = integrate(man.velocity, man.jacobian, g0, observe, _stopper(s), rtol, atol, wall=wall)
    return man, traj, info


def _stopper(s):
    state = dict(dead_since=None)

    def stop(traj):
        L, P, _, _ = traj.obs[-1]
        if L < DONE_LOSS:
            return "done"
        unl = P < 0.5 * s
        # dead routes only after something was learned (at the origin every amplitude ~ sigma^K)
        if unl.any() and not unl.all() and np.max(np.abs(P[unl]) / s[unl]) < DEAD_AMP:
            i = len(traj) - 1
            if state["dead_since"] is None:
                state["dead_since"] = traj.exact(i)
            elif traj.exact(i) > 10 * state["dead_since"] + 1000:
                return "trapped"
        else:
            state["dead_since"] = None
        return None

    return stop


# ------------------------------------------------------------------------------ reduction


def crossings(traj, P, level, a):
    """Every crossing of P[:, a] through level: (exact time, +1/-1, index of the step after)."""
    out = []
    x = P[:, a]
    for i in range(1, len(x)):
        x0, x1 = x[i - 1], x[i]
        if (x0 < level) != (x1 < level):
            t0, t1 = traj.exact(i - 1), traj.exact(i)
            frac = Fraction(float(level - x0)) / Fraction(float(x1 - x0))
            out.append((t0 + frac * (t1 - t0), 1 if x1 > x0 else -1, i))
    return out


def learning_events(traj, P, s):
    """Learning events (last upward s/2 crossing of patterns that end learned), forgetting events
    and all crossings at the LEVELS."""
    cross = {(a, lev): crossings(traj, P, lev * s[a], a) for a in range(len(s)) for lev in LEVELS}
    events, forgotten = [], []
    for a in range(len(s)):
        ups = [c for c in cross[(a, 0.5)] if c[1] > 0]
        if ups and P[-1, a] > 0.5 * s[a]:
            events.append(dict(pattern=a, t=ups[-1][0], i=ups[-1][2], n_up=len(ups)))
        if ups:
            t_first = ups[0][0]
            forgotten += [
                dict(pattern=a, t=c[0]) for c in cross[(a, 0.25)] if c[1] < 0 and c[0] > t_first
            ]
    events.sort(key=lambda e: e["t"])
    forgotten.sort(key=lambda e: e["t"])
    return events, forgotten, cross


def _first_after(cross, t, direction=1):
    for c in cross:
        if c[0] > t and c[1] == direction:
            return c
    return None


def reduce_run(cfg, traj, info, sigma, model=None, man=None):
    """Events and plateau diagnostics of one run. model: full network (None for the reference)."""
    target = build_target(cfg)
    s = np.array(target.weights)
    n = target.n
    L = np.array([o[0] for o in traj.obs])
    P = np.array([o[1] for o in traj.obs])
    off = np.array([o[2] for o in traj.obs])
    gn = np.array([o[3] for o in traj.obs])
    T = traj.t()
    events, forgotten, cross = learning_events(traj, P, s)
    ev_rows, pl_rows = [], []
    floor_flag = False
    for k, ev in enumerate(events):
        r, i_k, t_k = ev["pattern"], ev["i"], ev["t"]
        t_prev = events[k - 1]["t"] if k else Fraction(0)
        i_prev = events[k - 1]["i"] if k else 0
        J = sorted(a for a in range(n) if P[i_k - 1, a] > 0.5 * s[a] and a != r)
        J_prev_events = sorted(e["pattern"] for e in events[:k])
        pred = predictions(cfg, target, J)
        LJ = target.partial_loss(J)
        # steps of the plateau before the event (incl. its start); two events in one solver step
        # leave no step in between
        span = list(range(i_prev, i_k)) or [max(i_k - 1, 0)]
        # plateau entry: every learned amplitude within ENTRY_TOL of its weight
        entry = None
        for i in (span[1:] if k else [0]):
            if all(abs(P[i, a] - s[a]) < ENTRY_TOL * s[a] for a in J):
                entry = i
                break
        inner = [i for i in span if entry is not None and i >= entry] or span[1:] or span
        mg = inner[int(np.argmin(gn[inner]))]
        # escape: first time after the plateau start with L < L_J - 1e-3
        esc = None
        for i in range(max(i_prev, 1), len(L)):
            if L[i] < LJ - 1e-3 and L[i - 1] >= LJ - 1e-3:
                t0, t1 = traj.exact(i - 1), traj.exact(i)
                esc = t0 + Fraction(float(LJ - 1e-3 - L[i - 1])) / Fraction(float(L[i] - L[i - 1])) * (t1 - t0)
                break
        # separation (C.1 items 3-4): when r passes 0.9 s_r, is another unlearned pattern above 0.1?
        c90 = _first_after(cross[(r, 0.9)], t_k)
        others = [q for q in range(n) if q not in J and q != r]
        if c90 is not None and others:
            sep_other = float(max(P[c90[2], q] / s[q] for q in others))
        else:
            sep_other = math.nan
        # effective depth over the plateau before the event (times relative to the event, exact)
        # effective depth (times relative to the event, exact): over the plateau proper (entry ->
        # event; primary) and over the whole interval since the previous event (brief's wording)
        trel = np.array([float(traj.exact(i) - t_k) for i in range(i_prev, i_k + 1)])
        eff_full = _eff(trel, P[i_prev : i_k + 1, r], s[r])
        i0 = entry if entry is not None else i_prev
        eff = _eff(trel[max(i0 - i_prev, 0) :], P[max(i0, i_prev) : i_k + 1, r], s[r])
        # peak off-target mass over the transition (midpoint of the plateau before -> after)
        t_next = events[k + 1]["t"] if k + 1 < len(events) else traj.exact(len(traj) - 1)
        m1, m2 = (t_prev + t_k) / 2, (t_k + t_next) / 2
        win = [i for i in range(len(T)) if m1 <= traj.exact(i) <= m2]
        peak_off = float(off[win].max()) if win else math.nan
        c = target.depth(J, r)
        row = dict(
            event=k, pattern=r, pattern_str=_pstr(target, r), t_event=float(t_k), n_up=ev["n_up"],
            J_before=_jstr(J), J_prev_events=_jstr(J_prev_events), n_learned_before=len(J),
            duration=float(t_k - t_prev), t_escape=(float(esc - t_prev) if esc is not None else math.nan),
            depth=c, pairwise=pred["pair"][r], pred_greedy=pred["greedy"], pred_pairwise=pred["pairwise"],
            pred_weight=pred["weight"], pred_lib_greedy=pred["lib_greedy"], tie=pred["tie"],
            min_depth=min(pred["depth"].values()),
            gamma=gamma(cfg, J, r), eff_depth=(eff["depth"] if eff else math.nan),
            eff_slope=(eff["slope"] if eff else math.nan), eff_points=(eff["points"] if eff else 0),
            eff_depth_full=(eff_full["depth"] if eff_full else math.nan),
            peak_offtarget=peak_off, sep_other_at_r90=sep_other,
            has_entry=entry is not None, L_minus_LJ_mg=float(L[mg] - LJ), gradnorm_mg=float(gn[mg]),
            t_entry_to_event=(float(t_k - traj.exact(entry)) if entry is not None and k else math.nan),
            P_r_entry=(float(P[entry, r]) if entry is not None else math.nan), P_r_mg=float(P[mg, r]),
        )
        if model is not None:
            row.update(_point_dist(model, traj, mg, J))
        # one-step-ahead prediction of the scalar route equation from the entry point
        if k and entry is not None:
            tp = {q: route_time(float(P[entry, q]), target.depth(J, q), gamma(cfg, J, q), s[q])
                  for q in target.missing(J)}
            row.update(onestep_pred_time=tp[r], onestep_winner=min(tp, key=tp.get),
                       onestep_ratio=(row["t_entry_to_event"] / tp[r] if 0 < tp[r] < math.inf else math.nan))
        else:
            tp = {}
            row.update(onestep_pred_time=math.nan, onestep_winner=-1, onestep_ratio=math.nan)
        # plateau rows (J nonempty): one per missing pattern, at the entry and min-gradient points
        if k and model is not None:
            for q in target.missing(J):
                prow = dict(event=k, plateau=k - 1, J=_jstr(J), missing=q, missing_str=_pstr(target, q),
                            learned_next=(q == r), depth=target.depth(J, q), pairwise=pred["pair"][q],
                            gamma=gamma(cfg, J, q), onestep_pred_time=tp.get(q, math.nan))
                for tag, i in (("entry", entry), ("mg", mg)):
                    prow.update(_route_diag(model, traj, i, J, q, sigma, tag))
                pl_rows.append(prow)
            row["floor_flag"] = any(p.get("floor_flag_mg") or p.get("floor_flag_entry")
                                    for p in pl_rows if p["event"] == k and p["learned_next"])
            floor_flag |= row["floor_flag"]
        ev_rows.append(row)
    # unfinished last plateau (trapped / unfinished runs): rows for the patterns never learned
    missing_end = [q for q in range(n) if P[-1, q] <= 0.5 * s[q]]
    if model is not None and missing_end and events:
        k = len(events)
        i_prev = events[-1]["i"]
        J = sorted(a for a in range(n) if P[-1, a] > 0.5 * s[a])
        entry = next((i for i in range(i_prev, len(L)) if all(abs(P[i, a] - s[a]) < ENTRY_TOL * s[a] for a in J)), None)
        inner = list(range(entry if entry is not None else i_prev, len(L)))
        mg = inner[int(np.argmin(gn[inner]))]
        for q in missing_end:
            prow = dict(event=k, plateau=k - 1, J=_jstr(J), missing=q, missing_str=_pstr(target, q),
                        learned_next=False, final=True, depth=target.depth(J, q),
                        pairwise=min(target.pair_cost(a, q) for a in J), gamma=gamma(cfg, J, q),
                        onestep_pred_time=math.nan, P_end=float(P[-1, q]))
            for tag, i in (("entry", entry), ("mg", mg)):
                prow.update(_route_diag(model, traj, i, J, q, sigma, tag))
            pl_rows.append(prow)
    order = [e["pattern"] for e in events]
    run = dict(
        status=info["status"], n_learned=len(events), order=" ".join(_pstr(target, a) for a in order),
        order_idx=" ".join(map(str, order)), n_forgotten=len(forgotten),
        forgotten=" ".join(f"{_pstr(target, f['pattern'])}@{float(f['t']):.6g}" for f in forgotten),
        L_final=float(L[-1]), t_final=float(T[-1]), floor_flag=floor_flag,
        n_unseparated=int(sum(1 for e in ev_rows if e["sep_other_at_r90"] > 0.1)),
        n_no_entry=int(sum(1 for e in ev_rows[1:] if not e["has_entry"])),
        max_offtarget=float(off.max()), **{k: v for k, v in info.items() if k != "status"},
    )
    traj_small = dict(t=T, L=L, P=P, off=off, gn=gn)
    return run, ev_rows, pl_rows, traj_small


def _eff(t, p, s):
    return effective_depth(t, p, s) if len(t) >= 3 else None


def _pstr(target, a):
    return "".join(map(str, target.patterns[a]))


def _jstr(J):
    return "{" + ",".join(map(str, sorted(J))) + "}"


def _point_dist(model, traj, i, J):
    """Distance of the min-gradient point to the canonical partial target sum_J s_a U_a."""
    W = model.W(torch.from_numpy(traj.y[i]))
    WJ = torch.zeros_like(W)
    for a in J:
        WJ[model.idx[a]] = model.s[a]
    return dict(rel_dist_mg=float((W - WJ).norm()) / model.norm_star)


def _route_diag(model, traj, i, J, q, sigma, tag):
    """At step i: signed P_q, route-entry norms of q (aligned to J) / sigma, alignment quality,
    active-factor product of q, and the floor ratio (escape signal / roundoff bound)."""
    out = {}
    if i is None:
        return {f"{k}_{tag}": math.nan for k in
                ("P", "ren_min", "ren_max", "ren_gmean", "align_q", "gamma_obs", "floor_ratio")} | {
                    f"floor_flag_{tag}": False}
    man, target = model.man, model.target
    theta = torch.from_numpy(traj.y[i])
    out["P"] = float(traj.obs[i][1][q])
    try:
        aligned, quality = man.align(model.cores(theta), J)
        norms = man.route_entry_norms(aligned, J, q)
        vals = np.array(list(norms.values()))
        out["ren_min"] = float(vals.min()) / sigma
        out["ren_max"] = float(vals.max()) / sigma
        out["ren_gmean"] = float(np.exp(np.log(np.maximum(vals, 1e-300)).mean())) / sigma
        out["align_q"] = float(min(quality.values()))
        region = target.region(J, q)
        act = [float(aligned[v][target.local_type(v, q)]) for v in range(target.tree.n_nodes) if v not in region]
        out["gamma_obs"] = float(np.prod(act))
    except Exception:  # noqa: BLE001 (diagnostic only; recorded as nan)
        for k in ("ren_min", "ren_max", "ren_gmean", "align_q", "gamma_obs"):
            out[k] = math.nan
    W, E, v = model.evaluate(theta)
    gP = model.grad_amplitude(theta, q)
    bound = model.grad_roundoff_bound(theta, E)
    signal = abs(float(gP @ v))
    noise = float(gP.abs() @ bound)
    out["floor_ratio"] = signal / noise if noise > 0 else math.inf
    res = {f"{k}_{tag}": val for k, val in out.items()}
    res[f"floor_flag_{tag}"] = bool(out["floor_ratio"] < FLOOR_RATIO)
    return res
