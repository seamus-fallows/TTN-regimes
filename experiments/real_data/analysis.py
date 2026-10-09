"""Events, ranks and plateaus of real-data trajectories (output of run.py).

Definitions (notes/diagnostics.md, adapted to data without a teacher; see notes/real_data.md section 3):
- Spectra are the data-weighted bond spectra `fspectra` (Frobenius spectra of W are dominated, late in
  training, by directions the data barely sees).
- Reference values: there is no teacher, so the reference for s_k^e is its value at the end of the run,
  s_k^e(T). (e, k) is "learned" if s_k^e(T) >= LEARNED x s_1^e(T).
- Event (e, k): first time s_k^e(t) >= f x s_k^e(T) (f = 0.05; sensitivity 0.03, 0.1). Width: final
  rise from f to 0.9 of s_k^e(T) (`event_widths`). An event already above threshold at the start of
  phase 2 for k >= 2 is "pre-escape" (not resolved; reported separately).
- First escape: the k = 1 events of all edges, on the absolute clock (t1 = t_a + tau), for the
  sigma^-(K-2) scaling. Later events use the phase-2 clock tau.
- Events are merged when closer than their widths (`merge_events`); separation ratios rho_i.
- Times are taken from merged log + fine rows (`ttn.dynamics.merged_rows`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ttn.diagnostics import (
    crossing_times,
    event_widths,
    merge_events,
    separation_ratios,
)
from ttn.dynamics import merged_rows

LEARNED = 0.02
F_EVENT = 0.05


def load(path) -> dict:
    """meta, phase-1 rows (absolute t), phase-2 merged rows (tau clock) of one run."""
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["__meta__"]))
        raw = {k: f[k] for k in f.files if k != "__meta__"}
    out = {"meta": meta, "path": str(path)}
    for ph in ("p1", "p2"):
        sub = {k[len(ph) + 1:]: v for k, v in raw.items() if k.startswith(ph + "_")}
        if sub:
            out[ph] = merged_rows(sub, keys=("loss", "fspectra", "spectra"))
            out[ph + "_log"] = sub
    out["theta0"] = raw.get("theta0")
    return out


def tree_levels(meta) -> np.ndarray:
    """Height of each edge's child node above the bottom (bottom cores = 1), per edge."""
    parent = meta["parent"]
    n = len(parent)
    kids = [[c for c in range(n) if parent[c] == v] for v in range(n)]
    height = [0] * n

    def h(v):
        if height[v] == 0:
            height[v] = 1 + max((h(c) for c in kids[v]), default=0)
        return height[v]

    return np.array([h(v) for v in meta["edges"]])


def edge_sites(meta) -> list[list[int]]:
    """Sites below each edge."""
    parent, sites = meta["parent"], meta["sites"]
    n = len(parent)
    below = [list(sites[v]) for v in range(n)]
    changed = True
    while changed:
        changed = False
        for c in range(n):
            p = parent[c]
            if p >= 0 and not set(below[c]) <= set(below[p]):
                below[p] = sorted(set(below[p]) | set(below[c]))
                changed = True
    return [below[v] for v in meta["edges"]]


def events(run: dict, f: float = F_EVENT, learned: float = LEARNED) -> dict:
    """Spectral events of one run (see module docstring)."""
    p2 = run["p2"]
    t, fs = p2["t"], p2["fspectra"]
    ref = fs[-1]  # (E, chi)
    E, chi = ref.shape
    is_learned = np.nan_to_num(ref) >= learned * ref[:, :1]
    curves = (fs / ref[None]).reshape(len(t), -1)
    times = crossing_times(t, curves, f).reshape(E, chi)
    widths = event_widths(t, curves, lo=f, hi=0.9).reshape(E, chi)
    start_above = (curves[0] >= f).reshape(E, chi)
    out = {"ref": ref, "learned": is_learned, "times": times, "widths": widths,
           "pre_escape": start_above & (np.arange(chi)[None] >= 1) & is_learned}
    # first escape, absolute clock: k = 1 events of all edges
    t_a = run["meta"]["t_a"]
    out["t1_edges"] = t_a + times[:, 0]
    out["t1"] = float(np.nanmean(out["t1_edges"]))
    out["w1"] = float(np.nanmax(widths[:, 0]))
    # later events (k >= 2), learned and resolved
    sel = is_learned.copy()
    sel[:, 0] = False
    sel &= ~out["pre_escape"]
    ek = np.argwhere(sel)
    out["later"] = [(int(e), int(k), float(times[e, k]), float(widths[e, k])) for e, k in ek
                    if np.isfinite(times[e, k])]
    out["later"].sort(key=lambda r: r[2])
    tt = np.array([times[:, 0].mean()] + [r[2] for r in out["later"]])
    ww = np.array([out["w1"]] + [r[3] for r in out["later"]])
    out["merged_t"], out["merged_w"] = merge_events(tt, ww)
    out["rho"] = separation_ratios(out["merged_t"], out["merged_w"])
    return out


def effective_ranks(fs: np.ndarray, ref: np.ndarray, rel: float = 0.1) -> np.ndarray:
    """r_e(t) = #{k : s_k^e(t) > rel x (smallest learned reference value on e)}."""
    learned = np.where(np.nan_to_num(ref) >= LEARNED * ref[:, :1], ref, np.nan)
    thr = rel * np.nanmin(learned, axis=1)
    return np.sum(np.nan_to_num(fs) > thr[None, :, None], axis=-1)


def time_to_loss(t: np.ndarray, loss: np.ndarray, level: float) -> float:
    """First time the (running-minimum) loss is <= level, interpolated linearly in t; NaN if never."""
    lm = np.minimum.accumulate(loss)
    below = np.flatnonzero(lm <= level)
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(t[0])
    frac = (lm[i - 1] - level) / (lm[i - 1] - lm[i])
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def runs_in(directory) -> list[dict]:
    return [load(p) for p in sorted(Path(directory).glob("*.npz"))]


def loss_steps(run: dict, **kw) -> dict:
    """Plateaus and drops of the training loss after the first escape (phase-2 clock, merged rows),
    `ttn.diagnostics.find_plateaus` (judged in linear time; separation ratio > 3 for a plateau)."""
    from ttn.diagnostics import find_plateaus

    p2 = run["p2"]
    plateaus, drops = find_plateaus(p2["t"], p2["loss"], **kw)
    return {"plateaus": plateaus, "drops": drops}


def first_rank_growth(ev: dict) -> tuple[float, list[int]]:
    """Time of the first k >= 2 event (any edge, phase-2 clock) and the edges whose s_2 crosses within
    the first event's width window of it."""
    later = [r for r in ev["later"] if r[1] == 1]
    if not later:
        return np.nan, []
    t0 = later[0][2]
    return t0, sorted(r[0] for r in later if r[2] <= t0 + max(1.0, 0.1 * t0))


def loss_drops(t: np.ndarray, loss: np.ndarray, valley_factor: float = 3.0,
               min_drop: float = 0.01, rel_height: float = 0.5) -> list[dict]:
    """Drops of a (creeping) loss curve, judged in linear time.

    On real data the loss keeps decreasing slowly between learning events (ill-conditioned directions
    inside the current bond-rank profile), so `find_plateaus`' progress-based widths (10% -> 90% of a
    drop whose end is the next flattest point) measure that tail. Here a drop is a peak of the rate
    r = -dL/dt that stands out by `valley_factor` against the lowest rate between it and its
    neighbouring peaks (in both directions) and lowers L by at least `min_drop` x L(start) between those
    minima. Its time is the rate peak, its width the full width at `rel_height` of the peak rate
    (linear interpolation between samples), its size the loss change between the bracketing minima.
    Separation ratios rho_i = gap / max(width_i, width_i+1) as in notes/diagnostics.md.
    Returns one dict per drop: t, width, loss_before, loss_after, n_samples (inside the width).
    """
    t, loss = np.asarray(t, float), np.minimum.accumulate(np.asarray(loss, float))
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, loss = t[keep], loss[keep]
    tm = 0.5 * (t[1:] + t[:-1])
    r = np.maximum(-(np.diff(loss) / np.diff(t)), 0.0)
    if r.max() <= 0:
        return []
    # candidate peaks: local maxima of r
    cand = [i for i in range(len(r)) if (i == 0 or r[i] >= r[i - 1]) and (i == len(r) - 1 or r[i] > r[i + 1])]
    # keep peaks that rise by valley_factor above the minima separating them from higher neighbours
    peaks = []
    for i in cand:
        left = [j for j in cand if j < i and r[j] >= r[i]]
        right = [j for j in cand if j > i and r[j] >= r[i]]
        lo_l = r[left[-1]:i + 1].min() if left else r[: i + 1].min()
        lo_r = r[i:right[0] + 1].min() if right else r[i:].min()
        if r[i] >= valley_factor * max(lo_l, lo_r, 1e-300):
            peaks.append(i)
    # bracketing minima and sizes; merge drops that are too small into the larger neighbour
    def brackets(pk):
        b = [0] + [a + int(np.argmin(r[a:c + 1])) for a, c in zip(pk[:-1], pk[1:])] + [len(r) - 1]
        return b
    while peaks:
        b = brackets(peaks)
        sizes = [loss[b[j]] - loss[b[j + 1] + 1 if j + 1 == len(peaks) else b[j + 1]] for j in range(len(peaks))]
        j = int(np.argmin(sizes))
        if sizes[j] >= min_drop * loss[0]:
            break
        del peaks[j]
    out = []
    b = brackets(peaks)
    for j, i in enumerate(peaks):
        half = rel_height * r[i]
        a = i
        while a > 0 and r[a - 1] > half:
            a -= 1
        c = i
        while c < len(r) - 1 and r[c + 1] > half:
            c += 1
        def cross(k_in, k_out):
            # time where r crosses `half` between tm[k_in] (above) and tm[k_out] (below)
            if k_out < 0 or k_out >= len(r):
                return tm[k_in]
            f = (r[k_in] - half) / (r[k_in] - r[k_out])
            return tm[k_in] + f * (tm[k_out] - tm[k_in])
        t_lo, t_hi = cross(a, a - 1), cross(c, c + 1)
        end = b[j + 1] + 1 if j + 1 == len(peaks) else b[j + 1]
        out.append({"t": float(tm[i]), "width": float(t_hi - t_lo), "t_lo": float(t_lo),
                    "t_hi": float(t_hi), "loss_before": float(loss[b[j]]),
                    "loss_after": float(loss[end]), "n_samples": int(c - a + 1)})
    return out
