"""Learning events of one run, read off the bond singular values (definitions: notes/route_generic.md).

A "rise" is the k-th singular value s_k^e of edge e growing from the noise floor to its target value:
rise time = last upward crossing of s_k^e = f_half x (target value), onset = last upward crossing of
f_onset x (target value) before it. Rises whose [onset, rise] intervals overlap form one learning event.
For an event: the bonds that gain rank, k_region = cores of the smallest subtree containing their end
nodes, the event time (first rise time in the group), the duration of the plateau before it, and the
effective depth of every rise (`ttn.routes.effective_depth` on s_k^e over the plateau before the event).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ttn.routes import effective_depth

F_HALF = 0.5
F_ONSET = 0.05
FLOOR = 1e-14  # floor flag: emerging value below 1e-14 s_1 (100x the ~1e-16 SVD accuracy) on its plateau


def load(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    run = {k: z[k] for k in z.files if k != "meta"}
    run["meta"] = json.loads(str(z["meta"]))
    return run


def relative_clock(run: dict) -> np.ndarray:
    """Time since the first escape (L < L0/2): exact local-clock differences in the escape's segment
    and after it; t_abs - t_a (float64-limited) in earlier segments."""
    t, tau = run["t"], run["tau"]
    meta = run["meta"]
    out = np.array(tau, dtype=float)
    if meta["t_a"] is None:
        return t.copy()
    ia = int(np.nanargmin(np.where(np.isnan(tau), np.inf, np.abs(tau))))  # step with tau = 0
    seg_a, tloc_a = run["seg"][ia], run["tloc"][ia]
    pre = np.isnan(out)
    same = pre & (run["seg"] == seg_a)
    out[same] = run["tloc"][same] - tloc_a
    other = pre & ~same
    out[other] = t[other] - meta["t_a"]
    return out


def crossing(t: np.ndarray, y: np.ndarray, level: float, last: bool = True, before: int | None = None):
    """Index i and interpolated time of an upward crossing y[i-1] < level <= y[i] (last or first)."""
    yy = y if before is None else y[: before + 1]
    up = np.nonzero((yy[1:] >= level) & (yy[:-1] < level))[0] + 1
    if len(up) == 0:
        return None, math.nan, 0
    i = int(up[-1] if last else up[0])
    y0, y1 = math.log(max(y[i - 1], 1e-300)), math.log(y[i])
    frac = (math.log(level) - y0) / (y1 - y0) if y1 > y0 else 1.0
    return i, float(t[i - 1] + frac * (t[i] - t[i - 1])), len(up)


def columns(meta: dict) -> dict[tuple[int, int], int]:
    out, c = {}, 0
    for e, keep in zip(meta["edges"], meta["keep"]):
        for k in range(keep):
            out[(e, k)] = c
            c += 1
    return out


def rises(run: dict, tt: np.ndarray, f_half=F_HALF, f_onset=F_ONSET) -> list[dict]:
    meta = run["meta"]
    cols = columns(meta)
    spec = run["spectra"]
    out = []
    for e in meta["edges"]:
        ref = np.array(meta["ref"][str(e)])
        for k in range(meta["ranks"][str(e)]):
            y = spec[:, cols[(e, k)]]
            s = ref[k]
            i_half, t_half, n_up = crossing(tt, y, f_half * s)
            learned = bool(y[-1] >= f_half * s)
            row = dict(edge=e, k=k, target=float(s), learned=learned, n_up=n_up, final=float(y[-1] / s))
            if i_half is None or not learned:
                row.update(i_half=None, t_half=math.nan, t_on=math.nan, i_on=None)
                out.append(row)
                continue
            i_on, t_on, _ = crossing(tt, y, f_onset * s, before=i_half)
            if i_on is None:  # already above the onset level at the start
                i_on, t_on = 0, float(tt[0])
            _, t_half_abs, _ = crossing(run["t"], y, f_half * s)
            row.update(i_half=i_half, t_half=t_half, i_on=i_on, t_on=t_on, t_half_abs=t_half_abs)
            out.append(row)
    return out


def group(rs: list[dict]) -> list[list[dict]]:
    """Union of rises with overlapping [onset, rise] intervals, in time order."""
    done = sorted([r for r in rs if r["i_half"] is not None], key=lambda r: (r["t_on"], r["t_half"]))
    groups, end = [], -math.inf
    for r in done:
        if groups and r["t_on"] <= end:
            groups[-1].append(r)
            end = max(end, r["t_half"])
        else:
            groups.append([r])
            end = r["t_half"]
    return groups


def depth_on_plateau(tt, y, s, t_lo, t_hi):
    sel = (tt > t_lo) & (tt <= t_hi)
    if sel.sum() < 6:
        return None
    return effective_depth(tt[sel], y[sel], s)


def run_events(run: dict, tree) -> tuple[list[dict], list[dict]]:
    """Events (one dict per event) and the rises with their per-rise effective depth and floor value."""
    from common import k_region

    meta = run["meta"]
    tt = relative_clock(run)
    rs = rises(run, tt)
    groups = group(rs)
    cols = columns(meta)
    spec = run["spectra"]
    events = []
    prev_t, prev_rises = None, []
    for j, g in enumerate(groups):
        t_ev = min(r["t_half"] for r in g)
        bonds = sorted({r["edge"] for r in g})
        depths, floors = [], []
        for r in g:
            y = spec[:, cols[(r["edge"], r["k"])]]
            top = spec[:, cols[(r["edge"], 0)]]
            if j == 0:
                # escape from the origin: absolute clock (the relative clock is float64-limited there)
                t_lo, t_hi, clock = -math.inf, r["t_half_abs"], run["t"]
            else:
                t_lo, t_hi, clock = prev_t, r["t_half"], tt
            d = depth_on_plateau(clock, y, r["target"], t_lo, t_hi)
            r["depth"] = None if d is None else d["depth"]
            r["depth_points"] = 0 if d is None else d["points"]
            if d is not None:
                depths.append(d["depth"])
            if j > 0:
                sel = (tt > prev_t) & (tt <= r["t_on"])
                rel = (y[sel] / top[sel]) if sel.any() else np.array([np.nan])
                r["floor_min"] = float(np.nanmin(rel)) if np.isfinite(rel).any() else math.nan
            else:
                r["floor_min"] = math.nan
            r["event"] = j
            floors.append(r["floor_min"])
        flagged = any(np.isfinite(f) and f < FLOOR for f in floors)
        ev = dict(
            event=j,
            rises=";".join(f"{r['edge']}:{r['k']}" for r in sorted(g, key=lambda r: (r["edge"], r["k"]))),
            prior=";".join(sorted(f"{r['edge']}:{r['k']}" for r in prev_rises)),
            bonds=" ".join(map(str, bonds)),
            n_bonds=len(bonds),
            k_region=k_region(tree, bonds),
            t_event=t_ev if j > 0 else min(r["t_half_abs"] for r in g),
            duration=(t_ev - prev_t) if j > 0 else min(r["t_half_abs"] for r in g),
            eff_depth=float(np.median(depths)) if depths else math.nan,
            eff_depth_min=float(np.min(depths)) if depths else math.nan,
            eff_depth_max=float(np.max(depths)) if depths else math.nan,
            n_depth=len(depths),
            floor_min=float(np.nanmin(floors)) if np.isfinite(floors).any() else math.nan,
            floor_flag=bool(flagged),
            relearn=int(sum(max(r["n_up"] - 1, 0) for r in g)),
        )
        events.append(ev)
        prev_t = t_ev
        prev_rises = prev_rises + g
    return events, rs


def amplitude_events(run: dict, weights_target: np.ndarray) -> list[dict]:
    """Pattern / CP amplitudes P_a = <U_a, W>: learning event = last upward crossing of P_a*/2
    (P_a* = <U_a, W*>), effective depth on P_a over the plateau before; forgotten if P_a later falls
    below P_a*/4 after an earlier crossing."""
    tt = relative_clock(run)
    amps = run["amps"]
    out = []
    for a in range(amps.shape[1]):
        s = float(weights_target[a])
        y = amps[:, a]
        i, t_half, n_up = crossing(tt, y, 0.5 * s)
        learned = bool(y[-1] >= 0.5 * s)
        forgotten = False
        if n_up > 1:
            up = np.nonzero((y[1:] >= 0.5 * s) & (y[:-1] < 0.5 * s))[0] + 1
            forgotten = bool(np.any(y[up[0]:] < 0.25 * s))
        out.append(dict(component=a, target=s, t_half=t_half if learned else math.nan, learned=learned,
                        n_up=n_up, forgotten=forgotten, i_half=i if learned else None))
    order = sorted([r for r in out if r["learned"]], key=lambda r: r["t_half"])
    prev = None
    for r in order:
        y = amps[:, r["component"]]
        if prev is None:
            d = depth_on_plateau(run["t"], y, r["target"], -math.inf, run["t"][r["i_half"]])
        else:
            d = depth_on_plateau(tt, y, r["target"], prev, r["t_half"])
        r["eff_depth"] = math.nan if d is None else d["depth"]
        prev = r["t_half"]
    return out
