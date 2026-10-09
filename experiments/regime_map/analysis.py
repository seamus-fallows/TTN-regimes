"""Regime labels, boundaries and threshold sensitivity from saved runs (notes/diagnostics.md).

Input: results.jsonl (one summary row per run) + traj/<run_id>.npz (t, L/L0, overlaps A) under
/root/outputs/regime_map/<exp>/. Output: a per-run table with events, separation ratios and labels
for every (delta_K threshold, separation threshold) pair.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

OUT = "/root/outputs/regime_map"
K_THRESHOLDS = (0.05, 0.1, 0.2)
RHO_THRESHOLDS = (2.0, 3.0, 5.0)
CONVERGED = 1e-3  # L(T) < 1e-3 L(0)
REGIMES = ("lazy", "rich", "s2s", "stalled")


def _interp_cross(t, a, i, level):
    """Time where a crosses `level` between samples i-1 and i (linear in t)."""
    if i == 0:
        return t[0]
    return t[i - 1] + (level - a[i - 1]) / (a[i] - a[i - 1]) * (t[i] - t[i - 1])


def component_events(t, A, lo=0.1, mid=0.5, hi=0.9):
    """Per component r: (t_r, w_r). t_r = first crossing of 1/2; w_r = final-rise width,
    t(first reach 0.9) - t(last upward crossing of 0.1 before that) (notes/diagnostics.md).
    nan when the component never reaches the level."""
    out = []
    for r in range(A.shape[1]):
        a = A[:, r]
        above = np.nonzero(a >= mid)[0]
        if len(above) == 0:
            out.append((np.nan, np.nan))
            continue
        t_mid = _interp_cross(t, a, above[0], mid)
        hi_idx = np.nonzero(a >= hi)[0]
        if len(hi_idx) == 0:
            out.append((t_mid, np.nan))
            continue
        i_hi = hi_idx[0]
        t_hi = _interp_cross(t, a, i_hi, hi)
        below = np.nonzero(a[:i_hi] < lo)[0]
        if len(below) == 0:
            t_lo = t[0]
        else:
            j = below[-1]
            t_lo = t[j] + (lo - a[j]) / (a[j + 1] - a[j]) * (t[j + 1] - t[j])
        out.append((t_mid, t_hi - t_lo))
    return out


def separation_ratios(events):
    """rho_i = (t_{i+1} - t_i) / max(w_i, w_{i+1}) for events ordered by time (nan-free only)."""
    ev = sorted((t, w) for t, w in events if np.isfinite(t) and np.isfinite(w))
    return [(ev[i + 1][0] - ev[i][0]) / max(ev[i][1], ev[i + 1][1]) for i in range(len(ev) - 1)]


def label(l_end, dK, rhos, k_thr=0.1, rho_thr=3.0):
    """Regime rule of notes/diagnostics.md (order: stalled, then lazy / s2s / rich)."""
    if not np.isfinite(l_end) or l_end >= CONVERGED:
        return "stalled"
    if dK < k_thr:
        return "lazy"
    if len(rhos) >= 1 and min(rhos) > rho_thr:
        return "s2s"
    return "rich"


def load(exp: str) -> pd.DataFrame:
    """Per-run table with events, rho's and labels at every threshold pair."""
    rows = []
    with open(os.path.join(OUT, exp, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    from pilots.interim import run_id  # same id as the driver

    ev_col, rho_col, minrho = [], [], []
    for _, r in df.iterrows():
        z = np.load(os.path.join(OUT, exp, "traj", run_id(r.to_dict()) + ".npz"))
        ev = component_events(z["t"], z["A"])
        rho = separation_ratios(ev)
        ev_col.append(ev)
        rho_col.append(rho)
        minrho.append(min(rho) if rho else np.nan)
    df["events"] = ev_col
    df["rhos"] = rho_col
    df["min_rho"] = minrho
    df["n_events"] = [sum(np.isfinite(t) for t, _ in ev) for ev in ev_col]
    for kt in K_THRESHOLDS:
        for rt in RHO_THRESHOLDS:
            df[f"label_{kt}_{rt}"] = [
                label(le, dk, rh, kt, rt) for le, dk, rh in zip(df.l_end, df.dK, df.rhos)
            ]
    df["label"] = df["label_0.1_3.0"]
    # stalled sub-type: kernel-like (delta_K below threshold) vs feature-learning stall
    df["stall_type"] = np.where(
        df.label != "stalled", "", np.where(df.dK < 0.1, "stalled-lazy", "stalled-rich")
    )
    return df


def fractions(df: pd.DataFrame, by: list[str], col: str = "label") -> pd.DataFrame:
    """Fraction of seeds per regime at every grid point."""
    tab = df.groupby(by)[col].value_counts(normalize=True).unstack(fill_value=0.0)
    for r in REGIMES:
        if r not in tab.columns:
            tab[r] = 0.0
    return tab[list(REGIMES)]


def boundary(xs, frac, level=0.5):
    """Geometric interpolation of where `frac` (ordered by x) first crosses `level` (nan if never)."""
    xs, frac = np.asarray(xs, float), np.asarray(frac, float)
    for i in range(len(xs) - 1):
        f0, f1 = frac[i], frac[i + 1]
        if (f0 - level) * (f1 - level) < 0 or (f0 != f1 and f1 == level):
            w = (level - f0) / (f1 - f0)
            return float(np.exp(np.log(xs[i]) + w * (np.log(xs[i + 1]) - np.log(xs[i]))))
    return np.nan
