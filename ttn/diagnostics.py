"""Regime diagnostics (definitions in notes/diagnostics.md).

Pure functions, on single snapshots or on recorded trajectories.

Conventions
- Trajectory quantities are numpy arrays; functions of the parameters take and return float64 torch
  tensors (except `lanczos_extremes`, which returns numpy).
- Bond spectra have shape (..., max_bond), sorted decreasingly and NaN-padded (NaN = absent singular
  value), e.g. the recorder's `spectra` of shape (n_log, n_edges, max_bond). Rank functions reduce the
  last axis, so they work on one spectrum or on a whole trajectory at once.
- theta is the flat parameter vector `spec.flatten(cores)` (functions that also take `spec` accept a
  list of cores too).
- Loss functions are callables loss_fn(theta_flat) -> scalar tensor, so this module does not depend on
  ttn.dynamics. Build one as
      loss_fn = lambda th: population_loss(spec, spec.unflatten(th), teacher)
  (or use the `loss` method of a ttn.dynamics problem).
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.sparse.linalg import LinearOperator, eigsh
from threadpoolctl import threadpool_limits

from ttn.network import TTNSpec, _sqrt_psd, apply_on_leg, forward, site_vectors, to_dense

LossFn = Callable[[torch.Tensor], torch.Tensor]


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _flat(spec: TTNSpec, theta) -> torch.Tensor:
    if isinstance(theta, (list, tuple)):
        return spec.flatten([c.detach() for c in theta])
    return theta.detach()


# ---- ranks -----------------------------------------------------------------------------------


def effective_rank(sv, threshold) -> np.ndarray:
    """#{k : s_k > threshold} over the last axis (absolute threshold; NaN entries never count).

    sv: (..., k). threshold: scalar or array broadcastable to sv.shape[:-1] (one per spectrum, e.g.
    one per edge for a (n_log, n_edges, max_bond) trajectory). Returns ints of shape sv.shape[:-1].
    """
    return np.sum(_np(sv) > _np(threshold)[..., None], axis=-1)


def smallest_nonzero(sv, rtol: float = 1e-8) -> np.ndarray:
    """Smallest s_k > rtol * max_k s_k of each spectrum (last axis); NaN for an all-zero spectrum.

    Applied to teacher spectra this gives the per-edge reference of notes/diagnostics.md.
    """
    sv = _np(sv)
    top = np.fmax.reduce(sv, axis=-1, keepdims=True)  # ignores NaN padding
    out = np.where(sv > rtol * top, sv, np.inf).min(axis=-1)
    return np.where(np.isfinite(out), out, np.nan)


def effective_rank_relative(sv, rel: float = 0.1, reference=None) -> np.ndarray:
    """#{k : s_k > rel * reference} over the last axis.

    reference: scalar or array broadcastable to sv.shape[:-1]; the default convention is the smallest
    nonzero teacher Schmidt value on each edge, `smallest_nonzero(teacher_spectra)`, with rel = 0.1.
    None means each spectrum's own largest value (rank relative to s_1).
    """
    sv = _np(sv)
    ref = np.fmax.reduce(sv, axis=-1) if reference is None else _np(reference)
    return effective_rank(sv, rel * ref)


def entropy_rank(sv) -> np.ndarray:
    """exp(H(p)) with p_k = s_k^2 / sum_j s_j^2 over the last axis (NaN = absent).

    Between 1 and the number of nonzero values; 0 for an all-zero (or all-NaN) spectrum.
    """
    s = np.nan_to_num(_np(sv))
    top = s.max(axis=-1, keepdims=True)
    # Normalize by s_1 first: squaring raw values ~1e-160 (tiny init) would underflow.
    s = np.divide(s, top, out=np.zeros_like(s), where=top > 0)
    p = s**2 / np.maximum(np.sum(s**2, axis=-1, keepdims=True), np.finfo(float).tiny)
    plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return np.where(top[..., 0] > 0, np.exp(-plogp.sum(axis=-1)), 0.0)


# ---- learning events -------------------------------------------------------------------------


def _crossing(t: np.ndarray, c: np.ndarray, level: float, last: bool) -> float:
    """Upward crossing of one curve (see `crossing_times`)."""
    if last:
        below = np.flatnonzero(~(c >= level))  # NaN counts as below
        if below.size and below[-1] == len(c) - 1:
            return np.nan
        if below.size:
            t, c = t[below[-1] :], c[below[-1] :]
    above = np.flatnonzero(c >= level)
    if above.size == 0:
        return np.nan
    i = above[0]
    if i == 0 or not np.isfinite(c[i - 1]):
        return float(t[i])
    frac = (level - c[i - 1]) / (c[i] - c[i - 1])
    if t[i - 1] > 0:
        return float(np.exp(np.log(t[i - 1]) + frac * (np.log(t[i]) - np.log(t[i - 1]))))
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def crossing_times(t, curves, level, last: bool = False) -> np.ndarray:
    """First time each curve is >= `level`, interpolated between the bracketing samples.

    t: (n_t,) increasing. curves: (n_t, n_curves) (or (n_t,), giving a 0-d result). level: scalar or
    (n_curves,). Interpolation is linear in log t, or linear in t when the bracket starts at t <= 0.
    A curve already at or above the level at t[0] gives t[0]; one that never reaches it gives NaN.
    NaN samples count as below the level. last=True gives the last upward crossing instead (the
    curve stays >= level afterwards; NaN if it ends below): robust event times when overlaps make
    transient excursions (in GF on binary4, a_2 rises to 0.4 and falls back while a_1 is learned).
    """
    t, c = _np(t), _np(curves)
    single = c.ndim == 1
    c = c.reshape(len(t), -1)
    levels = np.broadcast_to(_np(level), (c.shape[1],))
    out = np.array([_crossing(t, c[:, j], levels[j], last) for j in range(c.shape[1])])
    return out[0] if single else out


def rise_times(t, curves, lo: float = 0.1, hi: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """(t_lo, t_hi) of each column's final rise (NaN if hi is never reached; see `event_widths`).

    t_hi is the first crossing of hi and t_lo the last upward crossing of lo before it.
    """
    t, c = _np(t), _np(curves)
    single = c.ndim == 1
    c = c.reshape(len(t), -1)
    his = np.broadcast_to(_np(hi), (c.shape[1],))
    los = np.broadcast_to(_np(lo), (c.shape[1],))
    t_lo, t_hi = np.full(c.shape[1], np.nan), np.full(c.shape[1], np.nan)
    for j in range(c.shape[1]):
        reached = np.flatnonzero(c[:, j] >= his[j])
        if reached.size:
            end = reached[0] + 1
            t_lo[j] = _crossing(t[:end], c[:end, j], los[j], last=True)
            t_hi[j] = _crossing(t, c[:, j], his[j], last=False)
    return (t_lo[0], t_hi[0]) if single else (t_lo, t_hi)


def event_widths(t, curves, lo: float = 0.1, hi: float = 0.9) -> np.ndarray:
    """w = t(curve = hi) - t(curve = lo) per column, on the final rise (NaN if hi is never reached).

    t(hi) is the first crossing of hi and t(lo) the last upward crossing of lo before it, so a
    transient bump above lo that falls back (see `crossing_times`) does not inflate the width; for
    monotone curves both are first crossings. Curves are normalized so that 1 means learned
    (overlaps a_r, or s_k^e / s_k^e(final)). A rise that happens between two samples gets a width of
    about 0.8 sample intervals whatever its true width: compare with `sample_spacing`.
    """
    t_lo, t_hi = rise_times(t, curves, lo, hi)
    return t_hi - t_lo


def sample_spacing(t, start, end) -> np.ndarray:
    """Largest sample interval of t that overlaps [start, end], elementwise (NaN where start or end
    is NaN): the time resolution of an event spanning [start, end], e.g. a rise from `rise_times`.

    A width below ~3 times this spacing is not resolved: it is then an overestimate (a rise inside
    one interval gets ~0.8 intervals), and separation ratios built from it are lower bounds.
    """
    t = _np(t)
    start, end = np.broadcast_arrays(_np(start), _np(end))
    dt = np.diff(t)
    out = np.full(start.shape, np.nan)
    last = len(dt) - 1
    for idx in np.ndindex(start.shape):
        a, b = start[idx], end[idx]
        if np.isfinite(a) and np.isfinite(b) and last >= 0:
            i = int(np.clip(np.searchsorted(t, a, side="right") - 1, 0, last))
            j = int(np.clip(np.searchsorted(t, b, side="left") - 1, i, last))
            out[idx] = dt[i : j + 1].max()
    return out


def upward_crossings(curves, level) -> np.ndarray:
    """Number of upward crossings of `level` per column (a curve starting at or above it counts
    one; NaN counts as below). More than one means a transient excursion above the level."""
    c = _np(curves)
    single = c.ndim == 1
    c = c.reshape(c.shape[0], -1)
    above = c >= np.broadcast_to(_np(level), (c.shape[1],))
    out = above[0].astype(int) + np.sum(above[1:] & ~above[:-1], axis=0)
    return out[0] if single else out


def _sorted_events(times, widths) -> tuple[np.ndarray, np.ndarray]:
    times, widths = _np(times).ravel(), _np(widths).ravel()
    keep = np.isfinite(times)
    order = np.argsort(times[keep], kind="stable")
    return times[keep][order], widths[keep][order]


def separation_ratios(times, widths) -> np.ndarray:
    """rho_i = (t_{i+1} - t_i) / max(w_i, w_{i+1}) for events sorted by time.

    Events with a NaN time (never happened) are dropped first. Returns n_events - 1 values (inf when
    both widths are 0; NaN when a width is NaN, e.g. an event that never reached 0.9). S2S needs
    every rho_i > 3 (notes/diagnostics.md).
    """
    t, w = _sorted_events(times, widths)
    scale = np.maximum(w[:-1], w[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scale > 0, np.diff(t) / scale, np.where(np.isnan(scale), np.nan, np.inf))


def ratio_lower_bounds(times, unresolved) -> np.ndarray:
    """For the separation ratios of `separation_ratios(times, ...)` (same event order): True where
    either event of the pair is unresolved (width below ~3 sample intervals, `sample_spacing`), so
    that rho_i is only a lower bound."""
    times, flags = _np(times).ravel(), np.asarray(unresolved, dtype=bool).ravel()
    keep = np.isfinite(times)
    flags = flags[keep][np.argsort(times[keep], kind="stable")]
    return flags[:-1] | flags[1:]


def merge_events(times, widths, return_groups: bool = False):
    """Merge events closer than their widths (rho <= 1; single linkage after sorting by time).

    A merged event has the mean time and the largest width of its members (NaN widths ignored).
    Used for general teachers, where one component raises several bond spectra at once.
    Returns (times, widths), plus with return_groups=True the members of each merged event as
    index arrays into the inputs.
    """
    times = _np(times).ravel()
    finite = np.flatnonzero(np.isfinite(times))
    index = finite[np.argsort(times[finite], kind="stable")]
    t, w = _sorted_events(times, widths)
    groups: list[list[int]] = []
    for i in range(len(t)):
        if groups and t[i] - t[groups[-1][-1]] <= np.fmax(w[groups[-1][-1]], w[i]):
            groups[-1].append(i)
        else:
            groups.append([i])
    merged_t = np.array([t[g].mean() for g in groups])
    merged_w = np.array([np.fmax.reduce(w[g]) for g in groups])
    if return_groups:
        return merged_t, merged_w, [index[g] for g in groups]
    return merged_t, merged_w


def spectral_events(
    t, spectra, reference=None, rank_rtol: float = 1e-8, lo: float = 0.1, hi: float = 0.9
) -> dict[str, np.ndarray]:
    """Learning events of the bond spectra (notes/diagnostics.md, general teachers), one per edge e
    and index k that the target needs, before merging (`merge_events`).

    t: (T,). spectra: (T, E, k) NaN-padded, e.g. a recorder's `spectra`. reference: the target's
    Schmidt values per edge, (E, k') NaN-padded or a list of E sequences (e.g. a teacher's
    `bond_spectra`); (e, k) is used iff k < the number of reference values on edge e above
    rank_rtol times the largest. None: the final spectra are the reference (an over-parameterized
    student's decaying extra values then count too: raise rank_rtol). Each used curve is
    c(t) = s_k^e(t) / s_k^e(t[-1]); its event time is the last upward crossing of 1/2 (the first,
    unless a transient crossed 1/2 and fell back) and its width the final rise lo -> hi.
    Returns arrays edge (index along the E axis), k, time, width, t_lo, t_hi, n_cross (upward
    crossings of 1/2).
    """
    t, s = _np(t), _np(spectra)
    final = s[-1]
    if reference is None:
        ref = final
    else:
        ref = np.full((s.shape[1], max(len(r) for r in reference)), np.nan)
        for e, r in enumerate(reference):
            ref[e, : len(r)] = _np(r)
    top = np.fmax.reduce(np.abs(ref), axis=1)
    edges, ks = [], []
    for e in range(s.shape[1]):
        rank = int(np.sum(np.abs(ref[e]) > rank_rtol * top[e])) if top[e] > 0 else 0
        for k in range(min(rank, s.shape[2])):
            if np.isfinite(final[e, k]) and final[e, k] > 0:
                edges.append(e)
                ks.append(k)
    curves = (s[:, edges, ks] / final[edges, ks]).reshape(len(t), len(edges))
    t_lo, t_hi = rise_times(t, curves, lo, hi)
    return {
        "edge": np.array(edges, dtype=int),
        "k": np.array(ks, dtype=int),
        "time": np.atleast_1d(crossing_times(t, curves, 0.5, last=True)),
        "width": np.atleast_1d(t_hi - t_lo),
        "t_lo": np.atleast_1d(t_lo),
        "t_hi": np.atleast_1d(t_hi),
        "n_cross": np.atleast_1d(upward_crossings(curves, 0.5)),
    }


# ---- plateaus and drops of the loss ----------------------------------------------------------


class Plateau(NamedTuple):
    t_start: float
    t_end: float
    level: float  # loss at the flattest point (slowest relative decrease)


class Drop(NamedTuple):
    t_half: float  # loss crosses (loss_before + loss_after) / 2
    t_fastest: float  # largest relative decrease rate -d log L / dt (linear time)
    width: float  # t(10% of the drop done) to t(90% done), linear time
    loss_before: float
    loss_after: float


def find_plateaus(
    t,
    loss,
    min_drop: float = 0.2,
    valley_factor: float = 10.0,
    min_separation: float = 3.0,
    floor: float = 1e-12,
) -> tuple[list[Plateau], list[Drop]]:
    """Plateaus and drops of a loss curve L(t), judged in linear time (t strictly increasing).

    Criterion (thresholds are the arguments):
    1. Use the running minimum of L (GD/SGD curves are not monotone), clipped below at
       floor * L(t[0]) (float noise of the expanded population loss sits far below). Rate
       r = -d log L / dt on each sample interval (r = 0 on exactly flat intervals; isolated
       rounding steps on a plateau become tiny drops, removed by step 2's min_drop merge).
    2. Drop candidates are the peaks of log r with prominence >= log(valley_factor): two drops are
       distinct only if r falls by that factor between them, below the smaller peak. A drop's
       segment ends at the flattest points (minima of r) separating it from its neighbours (the
       curve's ends for the first / last drop). Drops lowering L by less than a fraction `min_drop`
       over their segment are merged into a neighbour, smallest first.
    3. For a drop from L_b to L_a (the losses at its segment ends), progress p = (L_b - L)/(L_b - L_a)
       (linear in L, so the final drop to ~0 is well defined); t_half = t(p = 1/2),
       width = t(p = 0.9) - t(p = 0.1) (interpolated as in `crossing_times`); t_fastest is the
       midpoint of the sample interval at the drop's rate peak.
    4. Plateau between drops i, i+1: [t(p_i = 0.9), t(p_{i+1} = 0.1)] iff
       rho_i = (t_half_{i+1} - t_half_i) / max(w_i, w_{i+1}) > min_separation (the notes' rho).
       Initial plateau [t[0], t(p_0 = 0.1)] iff r falls by valley_factor below the first peak before
       it and (t_half_0 - t[0]) / w_0 > min_separation; final plateau [t(p = 0.9), t[-1]] likewise,
       and only above the floor (a converged run has none). A curve without drops is one plateau.
       Level = loss at the flattest point.

    Resolves: drops in linear time whose rates separate by valley_factor (two logistic drops of
    width w separate for a midpoint gap >~ 2w, so drops at t = 200 and 210 of width ~1 are distinct
    and have a plateau between them), and log-time staircases. Cannot resolve: anything narrower
    than the local sample spacing (on a log grid dt ~ t * dlog t; a drop inside one interval gets
    width ~0.8 dt, which caps rho, so record extra points in linear time where events happen);
    overlapping drops (gap <~ 2 widths: one drop; 2-3 widths: two drops without a plateau);
    log-time steps that are wide in linear time (rho <= 3 is no plateau, by the notes' definition);
    drops smaller than min_drop, e.g. a weak component learned while a strong one is not (use
    overlaps / bond spectra); noise above min_drop (SGD: smooth or average first, as the running
    minimum turns noise into small steps); plateau levels below the floor.
    """
    t, raw = _np(t), _np(loss)
    lm = np.minimum.accumulate(np.where(np.isfinite(raw), raw, np.inf))
    f_abs = floor * lm[0]
    lm = np.maximum(lm, f_abs)
    u = np.log(lm)
    du = u[:-1] - u[1:]
    flat = du <= 0
    deep = np.log(valley_factor)
    lr = np.full(len(t) - 1, -2 * deep - 1)
    if not flat.all():
        lr[~flat] = np.log(du[~flat] / np.diff(t)[~flat])
        lr[flat] = lr[~flat].min() - 2 * deep - 1
    padded = np.concatenate([[lr.min() - 1], lr, [lr.min() - 1]])  # lets edge intervals be peaks
    peaks = list(find_peaks(padded, prominence=deep)[0] - 1)

    def boundaries(pk):
        inner = [a + 1 + int(np.argmin(lr[a + 1 : b])) for a, b in pairwise(pk)]
        return [0] + inner + [len(t) - 1]

    min_du = -np.log1p(-min_drop)
    while peaks:
        b = boundaries(peaks)
        sizes = u[b[:-1]] - u[b[1:]]
        j = int(np.argmin(sizes))
        if sizes[j] >= min_du:
            break
        del peaks[j]
    if not peaks:
        level = lm[int(np.argmin(lr))]
        return ([Plateau(float(t[0]), float(t[-1]), float(level))] if level > f_abs else []), []

    drops, t10, t90 = [], [], []
    for i, k in enumerate(peaks):
        seg = slice(b[i], b[i + 1] + 1)
        lb, la = lm[b[i]], lm[b[i + 1]]
        p = (lb - lm[seg]) / (lb - la)
        q10, q50, q90 = (float(crossing_times(t[seg], p, q)) for q in (0.1, 0.5, 0.9))
        drops.append(Drop(q50, float(t[k] + t[k + 1]) / 2, q90 - q10, float(lb), float(la)))
        t10.append(q10)
        t90.append(q90)

    plateaus = []
    first, last = drops[0], drops[-1]
    if peaks[0] > 0:
        j = int(np.argmin(lr[: peaks[0]]))
        if lr[j] <= lr[peaks[0]] - deep and first.t_half - t[0] > min_separation * first.width:
            plateaus.append(Plateau(float(t[0]), t10[0], float(lm[j])))
    rho = separation_ratios([d.t_half for d in drops], [d.width for d in drops])
    for i in range(len(drops) - 1):
        if rho[i] > min_separation:
            plateaus.append(Plateau(t90[i], t10[i + 1], float(lm[b[i + 1]])))
    if peaks[-1] < len(lr) - 1:
        j = peaks[-1] + 1 + int(np.argmin(lr[peaks[-1] + 1 :]))
        long_enough = t[-1] - last.t_half > min_separation * last.width
        if lr[j] <= lr[peaks[-1]] - deep and long_enough and lm[j] > f_abs:
            plateaus.append(Plateau(t90[-1], float(t[-1]), float(lm[j])))
    return plateaus, drops


def plateau_midpoints(t, speed, event_times) -> np.ndarray:
    """Time of minimum parameter speed |d theta/dt| strictly between consecutive events (sorted,
    NaN events dropped): where the plateau Hessian is evaluated. NaN if no sample lies between."""
    t, speed = _np(t), _np(speed)
    ev = np.sort(_np(event_times)[np.isfinite(_np(event_times))])
    out = np.full(max(len(ev) - 1, 0), np.nan)
    for i in range(len(ev) - 1):
        inside = np.flatnonzero((t > ev[i]) & (t < ev[i + 1]))
        if inside.size:
            out[i] = t[inside[np.argmin(speed[inside])]]
    return out


# ---- regime assignment -----------------------------------------------------------------------

REGIME_THRESHOLDS = {
    "delta_K": 0.1,  # lazy iff delta_K(T) < this
    "separation": 3.0,  # S2S iff every rho_i > this
    "converged": 1e-3,  # stalled iff final loss >= this * L(0)
    "diverged": 1e6,  # diverged iff final loss non-finite or > this * L(0)
    "floor": 0.1,  # lazy at the projected fit iff final loss <= (1 + this) * the lazy floor
}


def classify_regime(
    delta_K: float,
    separation_ratios,
    final_loss: float,
    initial_loss: float,
    thresholds: dict | None = None,
    lower_bounds=None,
    lazy_floor: float | None = None,
) -> str:
    """One run's regime: 'diverged', 'stalled', 'lazy', 'saddle-to-saddle' or 'rich', plus
    'lazy (projected fit)' and 'undetermined' when the optional arguments are given.

    Rules of notes/diagnostics.md, checked in this order: diverged (final loss non-finite or
    > 1e6 L(0)); stalled (final loss >= 1e-3 L(0)); lazy (delta_K < 0.1); saddle-to-saddle (at least
    two events, i.e. at least one ratio, and every rho_i > 3); otherwise rich (overlapping
    transitions, or a single event). The sweep-level scaling check (iii) is `plateau_scaling`.
    `thresholds` overrides entries of REGIME_THRESHOLDS. A NaN ratio (an event that never completed)
    is not > 3, so it rules out S2S.

    lazy_floor (the loss of the lazy fit, `projected_fit_loss`, > 0 when a bond is deficient): a run
    with delta_K < 0.1 whose final loss is within 10% of it is 'lazy (projected fit)', not
    'stalled' (theory_regimes R7; that run cannot converge). Checked before 'stalled'.
    lower_bounds (`ratio_lower_bounds`, one flag per ratio): if every rho_i <= 3 is only a lower
    bound (an event narrower than the sample spacing), the label is 'undetermined', not 'rich'.
    """
    th = {**REGIME_THRESHOLDS, **(thresholds or {})}
    if not np.isfinite(final_loss) or final_loss > th["diverged"] * initial_loss:
        return "diverged"
    converged = final_loss < th["converged"] * initial_loss
    if (
        not converged
        and lazy_floor is not None
        and delta_K < th["delta_K"]
        and final_loss <= (1 + th["floor"]) * lazy_floor
    ):
        return "lazy (projected fit)"
    if not converged:
        return "stalled"
    if delta_K < th["delta_K"]:
        return "lazy"
    rho = _np(separation_ratios).ravel()
    if rho.size >= 1 and np.all(rho > th["separation"]):
        return "saddle-to-saddle"
    if rho.size >= 1 and lower_bounds is not None:
        low = ~(rho > th["separation"])
        if np.all(np.asarray(lower_bounds, dtype=bool)[low] & np.isfinite(rho[low])):
            return "undetermined"
    return "rich"


def plateau_scaling(sigmas, durations, z: float = 3.0) -> dict:
    """S2S scaling check (iii): do plateau durations T grow as the init scale sigma -> 0?

    Least-squares fits over all (sigma, T) pairs with finite T > 0 (seeds as separate points):
      log model    T = intercept + slope * log(1/sigma)
      power model  log T = log_prefactor + exponent * log(1/sigma)
    Returns both fits with standard errors and R^2, the number of points used, and
    grows = (slope > z * slope_se), i.e. a significant increase as sigma decreases (both growth laws
    give a positive slope). Needs at least 3 points and 2 distinct sigmas for standard errors.
    """
    sig, dur = _np(sigmas).ravel(), _np(durations).ravel()
    keep = np.isfinite(sig) & np.isfinite(dur) & (sig > 0) & (dur > 0)
    x = np.log(1 / sig[keep])
    log_fit = _linear_fit(x, dur[keep])
    pow_fit = _linear_fit(x, np.log(dur[keep]))
    return {
        "n": int(keep.sum()),
        "log": {
            "slope": log_fit[0],
            "slope_se": log_fit[1],
            "intercept": log_fit[2],
            "r2": log_fit[3],
        },
        "power": {
            "exponent": pow_fit[0],
            "exponent_se": pow_fit[1],
            "log_prefactor": pow_fit[2],
            "r2": pow_fit[3],
        },
        "grows": bool(log_fit[0] > z * log_fit[1]),
    }


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Least squares y = a + b x: (b, se(b), a, R^2); NaN where undefined."""
    n = len(x)
    sxx = np.sum((x - x.mean()) ** 2) if n else 0.0
    if n < 2 or sxx == 0:
        return np.nan, np.nan, np.nan, np.nan
    b = np.sum((x - x.mean()) * (y - y.mean())) / sxx
    a = y.mean() - b * x.mean()
    ss_res = np.sum((y - a - b * x) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    se = np.sqrt(ss_res / (n - 2) / sxx) if n > 2 else np.nan
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(b), float(se), float(a), float(r2)


# ---- laziness: parameter and kernel changes --------------------------------------------------


def relative_change(a, b) -> float:
    """|a - b| / |b| (Frobenius / 2-norm of the flattened arrays or tensors)."""
    a, b = _np(a), _np(b)
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


def _gram(jac: torch.Tensor, mask) -> torch.Tensor:
    if mask is None:
        return jac @ jac.T
    return (jac * torch.as_tensor(mask, dtype=jac.dtype)) @ jac.T


def tangent_kernel(spec: TTNSpec, theta, probe_x, mask=None) -> torch.Tensor:
    """Empirical tangent kernel K = J diag(mask) J^T, J = d f(probe_x) / d theta (exact, autodiff).

    probe_x: (P, N, d) or a list of N (P, d_i). mask: optional flat (n_params,) weights, e.g. the
    0/1 gradient mask of `antisymmetric_double` or per-parameter learning rates (the kernel of the
    preconditioned flow). Returns (P, P).
    """
    # All rows in one batched backward pass (vmap over the identity): 3-20x faster than one
    # backward pass per probe point, and much faster than torch.func.jacrev on these small einsums.
    th = _flat(spec, theta).clone().requires_grad_(True)
    f = forward(spec, spec.unflatten(th), probe_x)
    eye = torch.eye(f.numel(), dtype=f.dtype)
    (jac,) = torch.autograd.grad(f, th, grad_outputs=eye, is_grads_batched=True)
    return _gram(jac, mask)


def tensor_jacobian(spec: TTNSpec, theta) -> torch.Tensor:
    """J = d vec(W) / d theta, shape (D, n_params), D = prod d_i; vec(W) row-major in site order
    (site 0 slowest), matching `product_features`. Small N only."""
    cores = spec.unflatten(_flat(spec, theta))
    # W is linear in each core, so d vec(W) / d G_v[i] is vec(W) with G_v replaced by the unit
    # tensor e_i: exact, n_params contractions, cost independent of D (unlike reverse mode).
    cols = []
    for v, shape in enumerate(spec.shapes):
        for unit in torch.eye(spec.sizes[v], dtype=cores[v].dtype):
            cols.append(to_dense(spec, cores[:v] + [unit.view(shape)] + cores[v + 1 :]).reshape(-1))
    return torch.stack(cols, dim=1)


def tensor_tangent_kernel(spec: TTNSpec, theta, mask=None) -> torch.Tensor:
    """Tensor-space kernel J diag(mask) J^T with J = `tensor_jacobian`, shape (D, D), D = prod d_i.

    vec(W) is row-major in site order (site 0 slowest), matching `product_features`, so the probe
    kernel is Phi K_tensor Phi^T with Phi = product_features(probe_x). Small N only.
    """
    return _gram(tensor_jacobian(spec, theta), mask)


def projected_fit_loss(
    spec: TTNSpec, theta, residual, mask=None, site_metrics=None, rtol: float = 1e-10
) -> float:
    """Loss at the end of a lazy run: 1/2 ||(I - P) r||_M^2, the part of the residual r that the
    model linearized at theta cannot fit.

    residual: dense tensor (site order) target minus fitted tensor at theta (for the centred and
    two-copy models simply W*). P: the M-orthogonal projector (M = Sigma_1 (x) ... (x) Sigma_N, the
    site metrics; identity if None) onto the tangent space range(J diag(mask)) at theta, J =
    `tensor_jacobian`; mask: trained coordinates (nonzero entries; e.g. a problem's precond).
    Gradient flow of the linearized model, with any positive learning rates, converges to
    W_lin = W(theta) + P r (theory_regimes R2, R7): this is the floor of the lazy regime, 0 with full
    bonds and > 0 when a bond is deficient (then the run is 'stalled' by the convergence rule, see
    `classify_regime(..., lazy_floor=...)`). Rank of J: singular values above rtol times the largest.
    Small N only (J is dense, D x n_params).
    """
    jac = tensor_jacobian(spec, theta)
    if mask is not None:
        jac = jac[:, torch.as_tensor(mask) != 0]
    dims = spec.phys_dims
    jac = jac.reshape(*dims, -1)
    r = torch.as_tensor(residual, dtype=jac.dtype).reshape(dims)
    if site_metrics is not None:
        for i, m in enumerate(site_metrics):
            root = _sqrt_psd(torch.as_tensor(m, dtype=jac.dtype))
            jac, r = apply_on_leg(jac, root, i), apply_on_leg(r, root, i)
    u, s, _ = torch.linalg.svd(jac.reshape(r.numel(), -1), full_matrices=False)
    q = u[:, : int((s > rtol * s.max()).sum())] if s.numel() else u[:, :0]
    b = r.reshape(-1)
    return 0.5 * (b - q @ (q.T @ b)).pow(2).sum().item()


def product_features(x) -> torch.Tensor:
    """Phi(x) = vec(x_1 (x) ... (x) x_N) per sample, (B, N, d) -> (B, d^N): f(x) = Phi(x) vec(W)."""
    vecs = site_vectors(x)
    out = vecs[0]
    for v in vecs[1:]:
        out = (out[:, :, None] * v[:, None, :]).reshape(out.shape[0], -1)
    return out


def kernel_alignment(K, y, centered: bool = False) -> float:
    """A = y^T K y / (|K|_F |y|^2). centered=True: Cortes et al. (2012) centered alignment, the same
    with K -> H K H and y -> H y, H = I - 1 1^T / P."""
    K, y = _np(K), _np(y).ravel()
    if centered:
        K = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
        y = y - y.mean()
    return float(y @ K @ y / (np.linalg.norm(K) * (y @ y)))


def centered_kernel_alignment(K, y) -> float:
    """Centered kernel-target alignment (Cortes et al. 2012); see `kernel_alignment`."""
    return kernel_alignment(K, y, centered=True)


def kernel_change(K_t, K_0) -> float:
    """delta_K = |K_t - K_0|_F / |K_0|_F."""
    return relative_change(K_t, K_0)


# ---- Hessian ---------------------------------------------------------------------------------


def hessian(loss_fn: LossFn, theta) -> torch.Tensor:
    """Full Hessian of loss_fn at theta, (n_params, n_params), symmetrized.

    Vectorized double backward (all rows in one batched pass): 0.002 s at P = 63 and 0.1 s at
    P = 1376, against 0.04 s and 2.3 s with one backward pass per row; the first call in a process
    costs ~0.3 s more (torch.vmap set-up).
    """
    h = torch.autograd.functional.hessian(loss_fn, theta.detach(), vectorize=True)
    return 0.5 * (h + h.T)


def hessian_eigs(loss_fn: LossFn, theta) -> torch.Tensor:
    """All Hessian eigenvalues, ascending."""
    return torch.linalg.eigvalsh(hessian(loss_fn, theta))


def _grad_with_graph(loss_fn: LossFn, theta) -> tuple[torch.Tensor, torch.Tensor]:
    th = theta.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(loss_fn(th), th, create_graph=True)
    return th, g


def hvp(loss_fn: LossFn, theta, v) -> torch.Tensor:
    """Hessian-vector product H(theta) v by double backpropagation."""
    th, g = _grad_with_graph(loss_fn, theta)
    return torch.autograd.grad(g @ torch.as_tensor(v, dtype=th.dtype), th)[0]


def lanczos_extremes(
    loss_fn: LossFn, theta, k: int = 6, which: str = "both", tol: float = 0.0, vectors: bool = False
):
    """Extreme Hessian eigenvalues by Lanczos (scipy eigsh on a LinearOperator of HVPs).

    which: 'largest' (k largest algebraic), 'smallest' (k smallest algebraic) or 'both' (2k values).
    Returns eigenvalues ascending (numpy), plus the (n_params, m) eigenvectors if vectors=True.
    Needs k < n_params. The gradient graph is built once; each matvec is one backward pass.
    """
    th, g = _grad_with_graph(loss_fn, theta)

    def matvec(v):
        vt = torch.as_tensor(np.ravel(v), dtype=th.dtype)
        return torch.autograd.grad(g @ vt, th, retain_graph=True)[0].numpy()

    n = th.numel()
    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    v0 = np.random.default_rng(0).standard_normal(n)  # deterministic start vector
    modes = {"largest": ["LA"], "smallest": ["SA"], "both": ["SA", "LA"]}[which]
    # One BLAS thread: OpenBLAS ignores torch.set_num_threads, and its default pool is 100x slower
    # on a loaded machine.
    with threadpool_limits(limits=1, user_api="blas"):
        results = [eigsh(op, k=k, which=mode, tol=tol, v0=v0) for mode in modes]
    vals = np.concatenate([r[0] for r in results])
    vecs = np.concatenate([r[1] for r in results], axis=1)
    order = np.argsort(vals)
    return (vals[order], vecs[:, order]) if vectors else vals[order]


def saddle_type(eigs, rel_eps: float = 1e-3) -> dict:
    """Raw curvature report of a (gauge-projected) Hessian spectrum (notes/diagnostics.md, "Saddle
    type"); NaN entries (padding, e.g. a recorder's `hessian_eigs_gauge` row) are dropped.

    eps = rel_eps * max |lambda|. 'strict' if lambda_min < -eps, else 'non-strict'. Also returns
    n_negative (eigenvalues < -eps), lambda_min, lambda_max, ratio = lambda_min / lambda_max, eps.
    This relative threshold does NOT separate strict from higher-order saddles: near the origin
    H(sigma theta) = sigma^(K-2) H(theta), so the ratio does not depend on sigma and the first
    plateau (an order-K saddle) is 'strict' at every sigma, and k = 3 plateaus are 'strict' for
    sigma >~ 0.01-0.04 (theory_saddles R11, P7). Decide strictness across a sweep with
    `saddle_scaling` (how lambda_min scales with sigma).
    """
    e = _np(eigs)
    e = e[np.isfinite(e)]
    eps = rel_eps * np.abs(e).max()
    n_negative = int(np.sum(e < -eps))
    return {
        "type": "strict" if n_negative else "non-strict",
        "n_negative": n_negative,
        "lambda_min": float(e.min()),
        "lambda_max": float(e.max()),
        "ratio": float(e.min() / e.max()),
        "eps": float(eps),
    }


def saddle_scaling(sigmas, lambda_mins, threshold: float = 0.5) -> dict:
    """Strict vs higher-order saddle from how the most negative gauge-projected Hessian eigenvalue
    at the same plateau scales with the init scale across a sweep (theory_saddles R11, P6, P7).

    At a strict saddle lambda_min tends to a sigma-independent negative value (p = 0); at a saddle
    of order k (a route of k cores, or the origin with k = K) lambda_min ~ -sigma^(k-2) (p = k - 2),
    measured at equivalent points of the plateau (e.g. midpoints). Fits
    log|lambda_min| = c + p log sigma by least squares over the points with lambda_min < 0 (seeds as
    separate points). Returns n, exponent p, exponent_se, order = p + 2 (the k this suggests), and
    type: 'strict' if p < threshold, 'higher-order' otherwise, 'undetermined' with fewer than two
    distinct sigmas.
    """
    sig, lam = _np(sigmas).ravel(), _np(lambda_mins).ravel()
    keep = np.isfinite(sig) & np.isfinite(lam) & (sig > 0) & (lam < 0)
    p, se, _, _ = _linear_fit(np.log(sig[keep]), np.log(-lam[keep]))
    if np.unique(sig[keep]).size < 2:
        kind = "undetermined"
    else:
        kind = "strict" if p < threshold else "higher-order"
    return {"n": int(keep.sum()), "exponent": p, "exponent_se": se, "order": p + 2, "type": kind}


# ---- gauge directions ------------------------------------------------------------------------


def gauge_directions(spec: TTNSpec, theta) -> torch.Tensor:
    """(n_params, sum_v chi_v^2) matrix whose columns span the gauge tangent space at theta.

    Inserting (1 + eps E) (1 + eps E)^{-1} on edge v (parent p) gives, for each generator
    E = E_ab (a, b < chi_v), dG_v = G_v E (G_v matricized as (fan_in, chi_v)) and dG_p = -E applied
    on p's v-leg; all other cores 0. W is unchanged to first order. Columns are not normalized and
    are linearly independent for generic theta (not at special points such as aligned inits).
    """
    cores = spec.unflatten(_flat(spec, theta))
    t, slices = spec.tree, spec.core_slices()
    n_gauge = sum(spec.bond_dims[v] ** 2 for v in t.edges)
    out = torch.zeros(spec.n_params, n_gauge, dtype=cores[0].dtype)
    col = 0
    for v in t.edges:
        p, chi = t.parent[v], spec.bond_dims[v]
        g_v = cores[v].reshape(-1, chi)
        leg = len(t.sites[p]) + t.children[p].index(v)
        for a in range(chi):
            for b in range(chi):
                e = torch.zeros(chi, chi, dtype=g_v.dtype)
                e[a, b] = 1.0
                out[slices[v], col] = (g_v @ e).reshape(-1)
                out[slices[p], col] = apply_on_leg(cores[p], -e, leg).reshape(-1)
                col += 1
    return out


def gauge_complement(U, rtol: float = 1e-10) -> torch.Tensor:
    """Orthonormal basis (n, n - r) of the orthogonal complement of span(U), r = numerical rank of U
    (singular values > rtol * the largest), so rank-deficient U (special points) is handled."""
    # SVD rather than plain QR: QR of a rank-deficient U does not give a basis of its span.
    q, s, _ = torch.linalg.svd(U, full_matrices=True)
    rank = int((s > rtol * s.max()).sum())
    return q[:, rank:]


def project_out_gauge(H, U, rtol: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict H to the orthogonal complement of span(U): returns (C^T H C, its eigenvalues
    ascending) with C = gauge_complement(U). Gauge zero modes are removed, so the remaining zero /
    negative eigenvalues are genuine flat / descent directions (map eigenvectors back with C)."""
    c = gauge_complement(U, rtol)
    h = c.T @ torch.as_tensor(H, dtype=c.dtype) @ c
    h = 0.5 * (h + h.T)
    return h, torch.linalg.eigvalsh(h)
