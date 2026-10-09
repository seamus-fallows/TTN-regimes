"""Shared code for the s2s_core experiment: setup, inits, one gradient-flow run with refined events,
dormant-component diagnostics, plateau Hessians, and the correlation-flow escape prediction.

Everything here is specific to this experiment; the model, integrator and diagnostics are `ttn`.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ttn import diagnostics as dg
from ttn.dynamics import Problem, Recorder, integrate_gf
from ttn.network import TTNSpec, aligned_init, apply_on_leg, conserved_quantities
from ttn.teachers import CPTeacher, odeco_teacher
from ttn.tree import make_tree

OUT = Path("/root/outputs/s2s_core")


# ---- setup -----------------------------------------------------------------------------------


def setup(tree: str, n_sites: int, d: int, chi: int, weights, teacher_seed: int):
    spec = TTNSpec.uniform(make_tree(tree, n_sites), d, chi)
    teacher = odeco_teacher(n_sites, d, list(weights), seed=teacher_seed)
    return spec, teacher


def gauge_balance(spec: TTNSpec, cores, tol: float = 1e-13, max_sweeps: int = 10_000):
    """Same tensor W, rescaled by bond gauge transformations so that Q_e = 0 on every edge.

    Per edge v (parent p): with M = G_v^T G_v and N = G_p,(v) G_p,(v)^T, the SPD matrix
    P = M^-1/2 (M^1/2 N M^1/2)^1/2 M^-1/2 solves P M P = N, and A = P^1/2 (G_v <- G_v A,
    G_p <- A^-1 on the v-leg) makes the two Grams equal. Balancing one edge changes its neighbours,
    so sweep until max_e |Q_e|_max < tol * max_v |G_v|^2.
    """
    t = spec.tree
    cores = [c.clone() for c in cores]

    def sqrtm(m):
        vals, vecs = torch.linalg.eigh(0.5 * (m + m.T))
        return (vecs * vals.clamp_min(0).sqrt()) @ vecs.T, vecs, vals

    for _ in range(max_sweeps):
        q = conserved_quantities(spec, cores)
        scale = max(c.pow(2).sum().item() for c in cores)
        if max(x.abs().max().item() for x in q.values()) < tol * scale:
            return cores
        for v in t.edges:
            p, chi = t.parent[v], spec.bond_dims[v]
            leg = len(t.sites[p]) + t.children[p].index(v)
            gv = cores[v].reshape(-1, chi)
            gp = torch.movedim(cores[p], leg, 0).reshape(chi, -1)
            m, n = gv.T @ gv, gp @ gp.T
            m_half, vecs, vals = sqrtm(m)
            m_inv_half = (vecs / vals.sqrt()) @ vecs.T
            middle, _, _ = sqrtm(m_half @ n @ m_half)
            pmat = m_inv_half @ middle @ m_inv_half
            a, avecs, avals = sqrtm(pmat)
            a_inv = (avecs / avals.sqrt()) @ avecs.T
            cores[v] = (gv @ a).reshape(cores[v].shape)
            cores[p] = apply_on_leg(cores[p], a_inv, leg)
    raise RuntimeError("gauge balancing did not converge")


def make_init(spec: TTNSpec, teacher: CPTeacher, kind: str, sigma: float, seed: int, a0=None):
    """'fanin' (default model init), 'balanced' (fan-in, then gauge-balanced: same W_0, Q_e = 0) or
    'aligned' (Saxe-type, amplitude a0 per core and component; a0 scalar, (R,) or (K, R))."""
    if kind == "fanin":
        return spec.init(sigma, seed=seed)
    if kind == "balanced":
        return gauge_balance(spec, spec.init(sigma, seed=seed))
    if kind == "aligned":
        amps = torch.as_tensor(np.broadcast_to(a0, (spec.n_cores, teacher.n_components)).copy())
        return aligned_init(spec, teacher.factors, amps)
    raise ValueError(kind)


# ---- dormant-component diagnostics -----------------------------------------------------------


def slice_norms(spec: TTNSpec, teacher: CPTeacher, cores) -> np.ndarray:
    """(K, R): Frobenius norm of core v contracted with u_r on all its physical legs (the part of
    core v that component r's overlap depends on). NaN for cores without physical legs."""
    t = spec.tree
    out = np.full((spec.n_cores, teacher.n_components), np.nan)
    for v in range(t.n_nodes):
        if not t.sites[v]:
            continue
        for r in range(teacher.n_components):
            sl = cores[v]
            for s in t.sites[v]:
                sl = torch.tensordot(teacher.factors[r, s], sl, dims=([0], [0]))
            out[v, r] = sl.norm().item()
    return out


def slice_norm_recorder(teacher: CPTeacher):
    def fn(problem, theta, t):
        return slice_norms(problem.spec, teacher, problem.spec.unflatten(theta))

    return ("slice_norms", fn)


def balance_recorder():
    """Per-edge relative imbalance |Q_e|_F / |theta|^2 (constant |Q_e| under exact GF)."""

    def fn(problem, theta, t):
        q = conserved_quantities(problem.spec, problem.spec.unflatten(theta))
        n2 = theta.pow(2).sum().item()
        return np.array([q[v].norm().item() / n2 for v in problem.spec.tree.edges])

    return ("imbalance", fn)


# ---- refined events ---------------------------------------------------------------------------


def zoom_rise(problem, theta, t0, t1, r, n=100, min_points=40, max_levels=10, spectra=False, **gf):
    """Resolve the final rise (0.1 -> 0.9) of overlap r inside [t0, t1], given theta at t0.

    Re-integrates the window on a linear grid of n points, then repeatedly zooms onto
    [last a_r < 0.1, first a_r >= 0.9] until that interval holds >= min_points samples.
    Returns (t, overlaps (n, R), loss) on the finest window, or None if the rise is not inside; with
    spectra=True also the bond spectra (n, E, chi_max) on that window (recorded at every level).
    """
    for _ in range(max_levels):
        grid = np.linspace(0.0, t1 - t0, n + 1)[1:]
        rec = Recorder(spectra=spectra, conservation=False, save_params=True)
        tr = integrate_gf(problem, theta, t1 - t0, log_times=grid, recorder=rec, **gf)
        t, a, params = t0 + tr["t"], tr["overlaps"], tr["params"]
        hi = np.flatnonzero(a[:, r] >= 0.9)
        if hi.size == 0:
            return None
        i_hi = hi[0]
        lo = np.flatnonzero(a[: i_hi + 1, r] < 0.1)
        i_lo = lo[-1] if lo.size else 0
        if i_hi - i_lo >= min_points or i_lo == 0 and i_hi <= 1:
            break
        theta, t0, t1 = torch.as_tensor(params[i_lo]), t[i_lo], t[i_hi]
    if spectra:
        return t, a, tr["loss"], tr["spectra"]
    return t, a, tr["loss"]


# ---- chunked integration with a wall-clock limit ----------------------------------------------


class Run:
    """Concatenated chunks of `integrate_gf` (data arrays with the log index first, plus meta)."""

    def __init__(self, data, meta):
        self.data, self.meta = data, meta

    def __getitem__(self, key):
        return self.data[key]


def integrate_chunked(problem, theta0, t_max, n_log=400, chunk=50, wall_limit=1200.0,
                      loss_floor=None, extra=(), max_steps=20_000, stop_fn=None, spectra=True, **gf):
    """`integrate_gf` on the default log grid, in chunks of `chunk` log times, stopping early when
    the loss floor is reached, a chunk needs more than max_steps steps, or the wall time exceeds
    wall_limit (stop_reason 'wall_limit'), or stop_fn(chunk data) is true (stop_reason 'stop_fn').
    Saves params at every log time. Conservation drift is measured per chunk (relative to the
    chunk's start); 'conservation_drift' keeps the per-chunk values."""
    wall0 = time.perf_counter()
    theta = problem.spec.flatten(theta0) if isinstance(theta0, list) else theta0
    speed = problem.velocity(theta).norm().item()
    tau0 = theta.norm().item() / speed
    grid = np.geomspace(1e-3 * min(tau0, t_max), t_max, n_log)
    parts, t_off, stop, metas = [], 0.0, None, []
    for i in range(0, n_log, chunk):
        times = grid[i : i + chunk] - t_off
        rec = Recorder(save_params=True, spectra=spectra, extra=list(extra))
        tr = integrate_gf(problem, theta, float(times[-1]), log_times=times, recorder=rec,
                          loss_floor=loss_floor, max_steps=max_steps, **gf)
        d = dict(tr.data)
        d["t"] = d["t"] + t_off
        keep = slice(0, None) if not parts else slice(1, None)  # drop the repeated start point
        parts.append({k: v[keep] for k, v in d.items() if k != "theta_end"})
        metas.append(tr.meta)
        theta, t_off = torch.as_tensor(tr.data["theta_end"]), float(d["t"][-1])
        stop = tr.meta["stop_reason"]
        if stop != "completed":
            break
        if stop_fn is not None and stop_fn(d):
            stop = "stop_fn"
            break
        if time.perf_counter() - wall0 > wall_limit:
            stop = "wall_limit"
            break
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    meta = {"stop_reason": stop, "t_end": t_off, "wall_time": time.perf_counter() - wall0,
            "nfev": sum(m.get("nfev", 0) or 0 for m in metas), "chunks": len(metas),
            "integrator": metas[-1].get("integrator"), "rtol": metas[-1].get("rtol"),
            "atol": metas[-1].get("atol"), "problem": metas[-1].get("problem")}
    return Run(data, meta)


# ---- one run ----------------------------------------------------------------------------------


def run_one(cfg: dict, out_dir: Path, hessians: bool = True) -> dict:
    """Gradient flow for one (tree, chi, teacher, init, sigma, seed); saves the coarse trajectory and
    the refined event windows to out_dir, returns a JSON-able summary."""
    torch.set_num_threads(1)
    wall0 = time.perf_counter()
    spec, teacher = setup(cfg["tree"], cfg["N"], cfg["d"], cfg["chi"], cfg["weights"], cfg["teacher_seed"])
    K, R = spec.n_cores, teacher.n_components
    sigma = cfg["sigma"]
    cores0 = make_init(spec, teacher, cfg["init"], sigma, cfg["seed"], cfg.get("a0"))
    problem = Problem(spec, teacher)
    theta0 = spec.flatten(cores0)
    # atol scales with sigma: parameters that matter for an escape are O(sigma) or larger; anything
    # far below atol is at the float64 floor anyway (dormant amplitudes ~1e-14 relative to O(1)).
    gf = dict(method=cfg.get("method", "lsoda"), rtol=cfg.get("rtol", 1e-8),
              atol=cfg.get("atol_rel", 1e-11) * sigma)
    loss0 = problem.loss(theta0).item()
    # Budget: 1e3 x the RMS-overlap escape time of the weakest component (gaps are seed-dependent
    # multiples of it), and a wall-clock limit for runs that stall (LSODA crawls on stalled plateaus).
    s_min = min(cfg["weights"])
    t_max = cfg.get("t_max") or 1e3 * p3_prefactor(cfg["d"], cfg["N"], K, s_min) * sigma ** (-(K - 2))
    weights = np.asarray(cfg["weights"])

    def at_floor(d):
        # Every component that is not learned has stayed below 1e-15 for the whole chunk: whatever
        # happens next is seeded by float64 roundoff, so stop (class 'floor' in the analysis).
        amps = np.abs(d["overlaps"]) * weights
        learned = d["overlaps"][-1] >= 0.9
        return learned.any() and not learned.all() and amps[:, ~learned].max() < 1e-15

    tr = integrate_chunked(problem, theta0, t_max, n_log=cfg.get("n_log", 400),
                           wall_limit=cfg.get("wall_limit", 1200.0),
                           max_steps=cfg.get("max_steps", 20_000), stop_fn=at_floor,
                           loss_floor=cfg.get("loss_floor", 1e-9) * loss0,
                           extra=[slice_norm_recorder(teacher), balance_recorder()], **gf)
    t, a, params = tr["t"], tr["overlaps"], tr["params"]

    # Events: final rise of every component, refined in linear time.
    events, widths, first_cross, fine = np.full(R, np.nan), np.full(R, np.nan), np.full(R, np.nan), {}
    edge_steps, edge_rank_change = {}, {}
    for r in range(R):
        first_cross[r] = dg.crossing_times(t, a[:, r], 0.5)
        hi = np.flatnonzero(a[:, r] >= 0.9)
        if hi.size == 0:
            continue
        lo = np.flatnonzero(a[: hi[0] + 1, r] < 0.1)
        i_lo = lo[-1] if lo.size else 0
        z = zoom_rise(problem, torch.as_tensor(params[i_lo]), t[i_lo], t[hi[0]], r, spectra=True, **gf)
        if z is None:
            continue
        tz, az, lz, sz = z
        events[r] = dg.crossing_times(tz, az[:, r], 0.5, last=True)
        widths[r] = dg.event_widths(tz, az[:, r])
        fine[f"t_{r}"], fine[f"a_{r}"], fine[f"loss_{r}"], fine[f"spectra_{r}"] = tz, az, lz, sz
        # Per edge: time at which the number of singular values above s_r / 2 (half the final value
        # of the new one, diagnostics.md) reaches its end-of-window value; one step per event and
        # edge, at the same time on every edge if the component is learned on all bonds at once.
        rk = dg.effective_rank(np.nan_to_num(sz), 0.5 * cfg["weights"][r])
        edge_steps[r] = [float(tz[np.flatnonzero(rk[:, j] >= rk[-1, j])[0]]) for j in range(rk.shape[1])]
        edge_rank_change[r] = (rk[-1] - rk[0]).tolist()
    order = [int(r) for r in np.argsort(events) if np.isfinite(events[r])]
    rho = dg.separation_ratios(events, widths)

    # Plateau midpoints between consecutive events, their loss, spectra and Hessians.
    mids = dg.plateau_midpoints(t, tr["speed"], events)
    plateaus = []
    for i, tm in enumerate(mids):
        if not np.isfinite(tm):
            continue
        j = int(np.flatnonzero(t == tm)[0])
        theta = torch.as_tensor(params[j])
        learned = order[: i + 1]
        entry = {
            "t_mid": float(tm),
            "loss": float(tr["loss"][j]),
            "partial_fit_level": 0.5 * float(sum(cfg["weights"][q] ** 2 for q in range(R) if q not in learned)),
            "overlaps": a[j].tolist(),
            "spectra": [[float(x) for x in row if np.isfinite(x)] for row in tr["spectra"][j]],
            "grad_norm": float(tr["speed"][j]),
        }
        if hessians:
            h = problem.hessian(theta)
            _, eig = dg.project_out_gauge(h, dg.gauge_directions(spec, theta))
            eig = eig.numpy()
            entry["saddle"] = dg.saddle_type(eig)
            entry["eig_lowest"] = eig[:8].tolist()
            entry["eig_highest"] = eig[-4:].tolist()
            entry["n_projected"] = int(len(eig))
        plateaus.append(entry)

    # Dormant components during the first escape: overlap and slice norms of the components that are
    # not learned first, from t = 0 up to the second event.
    sn = tr["slice_norms"]
    dormant = {}
    if order:
        t_next = events[order[1]] if len(order) > 1 else t[-1]
        window = t < t_next
        for r in range(R):
            if r == order[0]:
                continue
            ar = np.abs(a[window, r]) * cfg["weights"][r]
            dormant[r] = {
                "amp0": float(ar[0]),
                "amp_min": float(ar.min()),
                "t_amp_min": float(t[window][np.argmin(ar)]),
                "amp_at_first_event": float(np.interp(events[order[0]], t, np.abs(a[:, r]) * cfg["weights"][r])),
                "slice0": np.nan_to_num(sn[0, :, r], nan=-1).tolist(),
                "slice_min": np.nan_to_num(np.nanmin(sn[window, :, r], axis=0), nan=-1).tolist(),
            }

    # Smallest |<W, U_r>| of every not-yet-learned component between consecutive events. After the
    # first escape the gradient carries O(1e-16) absolute roundoff, so a dormant amplitude below
    # ~1e-14 means its later escape is seeded by roundoff (flagged in the analysis).
    amps = np.abs(a) * np.asarray(cfg["weights"])
    ev_sorted = [events[r] for r in order]
    bounds = [0.0] + ev_sorted + [t[-1]]
    dormant_min = []
    for i in range(len(bounds) - 1):
        inside = (t > bounds[i]) & (t < bounds[i + 1])
        row = {}
        for r in range(R):
            if r in order[:i] or not inside.any():
                continue
            row[r] = float(amps[inside, r].min())
        dormant_min.append(row)

    np.savez_compressed(
        out_dir / f"{cfg['name']}.npz",
        **{k: v for k, v in tr.data.items() if k not in ("params",)},
        **{f"fine_{k}": v for k, v in fine.items()},
        mid_params=np.array([params[int(np.flatnonzero(t == tm)[0])] for tm in mids if np.isfinite(tm)]),
        theta_end=params[-1], theta0=theta0.numpy(),
        meta=np.array(json.dumps(tr.meta, default=_json)),
    )
    summary = {
        **cfg,
        "K": K,
        "P": spec.n_params,
        "loss0": loss0,
        "loss_end": float(tr["loss"][-1]),
        "stop_reason": tr.meta["stop_reason"],
        "t_end": float(t[-1]),
        "events": events.tolist(),
        "first_crossings": first_cross.tolist(),
        "widths": widths.tolist(),
        "order": order,
        "separation_ratios": rho.tolist(),
        "edge_rank_step_times": edge_steps,
        "edge_rank_change": edge_rank_change,
        "s2s": bool(len(order) >= 2 and np.all(rho > 3)),
        "plateaus": plateaus,
        "dormant": dormant,
        "dormant_min_between_events": dormant_min,
        "max_conservation_drift": float(np.nanmax(tr["conservation_drift"])),
        "q0_max": max(x.abs().max().item() for x in conserved_quantities(spec, cores0).values()),
        "wall_time": time.perf_counter() - wall0,
        "nfev": tr.meta.get("nfev"),
        "t_max": t_max,
    }
    (out_dir / f"{cfg['name']}.json").write_text(json.dumps(summary, default=_json))
    return summary


def _json(x):
    if isinstance(x, np.integer | np.floating):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ---- correlation flow (first-escape prediction) ----------------------------------------------


def correlation_flow_blowup(spec: TTNSpec, teacher: CPTeacher, theta_hat: torch.Tensor,
                            amp_stop: float = 1e6, rtol: float = 1e-10):
    """Blow-up time T of theta' = grad <W(theta), W*> from theta_hat (prediction P1 in
    notes/s2s_core.md: t_1 = sigma^-(K-2) T(theta_hat) + O(1) for the init sigma * theta_hat).

    Integrates until some |<W, U_r>| reaches amp_stop and adds the remaining blow-up time of a
    balanced aligned mode, q^-(K-2) / ((K-2) s_r) with q = |<W, U_r>|^(1/K). Also returns the
    component that blows up (r1) and, at the stop, the core-norm fractions |G_v|^2 / |theta|^2 and
    the largest relative imbalance |Q_e|_F / |theta|^2 (KKT points are balanced, core norms 1/K).
    """
    K = spec.n_cores
    w = teacher.weights

    def rhs(_, x):
        th = torch.tensor(x, requires_grad=True)
        (g,) = torch.autograd.grad(teacher.inner_ttn(spec, spec.unflatten(th)), th)
        return g.numpy()

    def amps(x):
        return (teacher.component_overlaps(spec, spec.unflatten(torch.tensor(x))) * w).numpy()

    def big(_, x):
        return np.max(np.abs(amps(x))) - amp_stop

    big.terminal = True
    x0 = theta_hat.numpy().copy()
    sol = solve_ivp(rhs, (0, 1e12), x0, method="DOP853", rtol=rtol, atol=1e-14 * np.abs(x0).max(), events=big)
    x = sol.y[:, -1]
    a = amps(x)
    r1 = int(np.argmax(np.abs(a)))
    q = abs(a[r1]) ** (1 / K)
    T = sol.t[-1] + q ** (-(K - 2)) / ((K - 2) * w[r1].item())
    cores = spec.unflatten(torch.tensor(x))
    n2 = float(np.sum(x**2))
    fractions = [c.pow(2).sum().item() / n2 for c in cores]
    imbalance = max(qq.norm().item() for qq in conserved_quantities(spec, cores).values()) / n2
    return {"T": float(T), "r1": r1, "status": int(sol.status), "core_fractions": fractions,
            "imbalance": imbalance, "nfev": int(sol.nfev)}


def p3_prefactor(d: int, n_sites: int, K: int, s: float) -> float:
    """RMS-overlap heuristic (P3): T ~ d^(N(K-2)/(2K)) / ((K-2) s)."""
    return d ** (n_sites * (K - 2) / (2 * K)) / ((K - 2) * s)


def kh_prefactor(theta_hat: torch.Tensor, K: int, s1: float) -> float:
    """Kumar-Haupt escape time from an init along the balanced rank-one KKT point, with
    delta = |theta_hat|: 1 / (K (K-2) s1 K^(-K/2) delta^(K-2)) (a lower bound on T, B1)."""
    return 1.0 / (K * (K - 2) * s1 * K ** (-K / 2) * theta_hat.norm().item() ** (K - 2))


def c_K(K: int, s: float, level: float = 0.5) -> float:
    """O(1) offset between the a = level crossing of a balanced mode and the blow-up of its
    correlation part (s^-(2-2/K) c_K(level), as in ttn.theory.escape_time_asymptotic)."""
    from scipy.special import hyp2f1

    f = hyp2f1(1, 2 / K, 1 + 2 / K, level)
    return s ** (-(2 - 2 / K)) * (level ** (2 / K) * f / 2 - level ** (-(K - 2) / K) / (K - 2))


def name_of(cfg: dict) -> str:
    w = "-".join(f"{x:g}" for x in cfg["weights"])
    return f"{cfg['tree']}{cfg['N']}_chi{cfg['chi']}_w{w}_{cfg['init']}_s{cfg['sigma']:g}_seed{cfg['seed']}"


def is_finite(x) -> bool:
    return x is not None and math.isfinite(x)
