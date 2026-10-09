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
    rh = np.asarray(rhos, dtype=float)
    if rh.size >= 1 and np.all(rh > rho_thr):  # a NaN ratio (unfinished event) rules out S2S
        return "s2s"
    return "rich"


def load(exp: str) -> pd.DataFrame:
    """Per-run table with events, rho's and labels at every threshold pair.

    Library runs (run_regime.py) carry event_times / event_widths / rhos in their rows; interim runs
    (pilots/interim.py) are re-analysed from their saved trajectories."""
    rows = []
    with open(os.path.join(OUT, exp, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    if "rhos" in df.columns:
        df["events"] = [list(zip(t, w)) for t, w in zip(df.event_times, df.event_widths)]
        df["rhos"] = [[x for x in r if x is not None] if isinstance(r, list) else [] for r in df.rhos]
    else:
        from pilots.interim import run_id  # same id as the interim driver

        ev_col, rho_col = [], []
        for _, r in df.iterrows():
            z = np.load(os.path.join(OUT, exp, "traj", run_id(r.to_dict()) + ".npz"))
            ev = component_events(z["t"], z["A"])
            ev_col.append(ev)
            rho_col.append(separation_ratios(ev))
        df["events"] = ev_col
        df["rhos"] = rho_col
    df["min_rho"] = [float(np.min(np.asarray(r, float))) if len(r) else np.nan for r in df.rhos]
    df["n_events"] = [sum(np.isfinite(t) for t, _ in ev) for ev in df.events]
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


# ---- S2S boundary predicted from the deepest sigma by aligned-mode extrapolation ---------------


def _I(K, v0, v1):
    """I_K(v0, v1) = (1/K) int_{v0}^{v1} v^(-2+2/K) / (1-v) dv (theory_regimes section 3.1)."""
    from scipy.integrate import quad

    p = 2 - 2 / K  # split 1/(1-v) = 1 + v/(1-v): the singular part is analytic
    sing = np.log(v1 / v0) if K == 2 else (v0 ** (1 - p) - v1 ** (1 - p)) / (p - 1)
    reg = quad(lambda v: v ** (1 - p) / (1 - v), v0, v1, limit=200)[0]
    return (sing + reg) / K


def aligned_sigma_c(rows: pd.DataFrame, K: int, weights=(3.0, 2.0, 1.0), rho_thr=3.0):
    """Predict one seed's S2S boundary from its deepest run only.

    Each component r is treated as an aligned balanced mode u' = K u^(2-2/K) (s_r - u) with its own
    effective initial value u0_r, calibrated from the measured event time t_r at the smallest sigma
    (u0_r scales as sigma^K). This reproduces the event times and widths of the aligned theory at
    every sigma, so min_i rho_i(sigma) and the crossing rho = rho_thr follow. Returns (sigma_c, R)
    with R = rho * sigma^(K-2) at the deepest sigma (nan if the deepest run is not S2S-like)."""
    from scipy.optimize import brentq

    deep = rows.sort_values("sigma").iloc[0]
    s0 = deep.sigma
    ev = deep.events
    w = np.array(weights)
    expo = 2 - 2 / K
    omega = _I(K, 0.1, 0.9)
    v0 = []
    for r, (t_r, _) in enumerate(ev):
        if not np.isfinite(t_r):
            return np.nan, np.nan
        target = t_r * w[r] ** expo
        f = lambda lv: _I(K, np.exp(lv), 0.5) - target  # noqa: E731
        try:
            v0.append(np.exp(brentq(f, np.log(1e-300), np.log(0.4999))))
        except ValueError:
            return np.nan, np.nan

    def min_rho(sig):
        scale = (sig / s0) ** K
        times = [w[r] ** -expo * _I(K, min(v0[r] * scale, 0.4999), 0.5) for r in range(len(w))]
        order = np.argsort(times)
        widths = [w[r] ** -expo * omega for r in order]
        ts = [times[r] for r in order]
        return min((ts[i + 1] - ts[i]) / max(widths[i], widths[i + 1]) for i in range(len(ts) - 1))

    grid = np.geomspace(s0, s0 * 1e3, 400)
    vals = np.array([min_rho(s) for s in grid])
    below = np.nonzero(vals < rho_thr)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan, min_rho(s0) * s0 ** (K - 2)
    i = below[0]
    return float(np.sqrt(grid[i - 1] * grid[i])), min_rho(s0) * s0 ** (K - 2)


# ---- per-seed boundaries and scaling fits (analysis standards 3-4) ------------------------------


def seed_s2s_boundary(rows: pd.DataFrame, label_col: str = "label"):
    """One seed's S2S boundary: geometric mean of the largest sigma below which every run is S2S
    and the next sigma (first non-S2S run). Returns (sigma_c, status): status 'ok', 'never_s2s'
    (deepest run not S2S), 'all_s2s' (no exit within the grid), or 'unresolved' (first non-S2S run
    did not finish: stalled at the budget, so the label may be a budget artefact)."""
    r = rows.sort_values("sigma")
    lab = r[label_col].values
    sig = r.sigma.values
    if lab[0] != "s2s":
        return np.nan, "never_s2s"
    non = np.flatnonzero(lab != "s2s")
    if non.size == 0:
        return np.nan, "all_s2s"
    i = non[0]
    status = "unresolved" if lab[i] == "stalled" else "ok"
    return float(np.sqrt(sig[i - 1] * sig[i])), status


def seed_kernel_boundary(rows: pd.DataFrame, thr: float = 0.1, converged_only: bool = True):
    """One seed's kernel boundary: sigma where delta_K(T) crosses `thr` from above (log-log
    interpolation between the bracketing runs; the last crossing if delta_K is not monotone).
    With converged_only, stalled runs are excluded (their delta_K is not an end-of-training value)."""
    r = rows.sort_values("sigma")
    if converged_only:
        r = r[r.label != "stalled"]
    sig, dk = r.sigma.values, r.dK.values
    idx = np.flatnonzero((dk[:-1] >= thr) & (dk[1:] < thr))
    if idx.size == 0:
        return np.nan
    i = idx[-1]
    w = (np.log(thr) - np.log(dk[i])) / (np.log(dk[i + 1]) - np.log(dk[i]))
    return float(np.exp(np.log(sig[i]) + w * (np.log(sig[i + 1]) - np.log(sig[i]))))


def seed_power_fit(x, y):
    """Least-squares slope and intercept of log y vs log x (finite, positive values)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    b, a = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return float(b), float(a)


def mean_se(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan, int(v.size)


def per_seed_table(df: pd.DataFrame, by: list[str], K: int | None = None) -> pd.DataFrame:
    """Per (group, seed): S2S boundary (+status), aligned-extrapolation prediction, kernel boundary,
    and the rho exponent on the S2S side (slope of log min_rho vs log sigma over the S2S runs)."""
    out = []
    for key, g in df.groupby(by + ["seed"]):
        k = K or int(g.K.iloc[0])
        sc, status = seed_s2s_boundary(g)
        pred, _ = aligned_sigma_c(g, k)
        s2s = g[g.label == "s2s"]
        slope, _ = seed_power_fit(s2s.sigma, s2s.min_rho) if len(s2s) >= 3 else (np.nan, np.nan)
        row = dict(zip(by + ["seed"], key if isinstance(key, tuple) else (key,)))
        row.update(K=k, sigma_c=sc, s2s_status=status, sigma_c_pred=pred, sigma_K01=seed_kernel_boundary(g),
                   rho_slope=slope, n_s2s=len(s2s))
        out.append(row)
    return pd.DataFrame(out)


def stall_type(exp: str, row: dict, R: int = 3) -> dict:
    """Classify a stalled run from its bond spectra (theory_regimes section 7): the R-th singular
    value of every edge at the end; 'dead route' if its minimum over edges is < 1e-12 (at roundoff:
    the route cannot be revived by the float64 gradient), 'long plateau' if >= 1e-4, 'decaying'
    in between."""
    from run_regime import row_cfg, run_id

    from ttn.dynamics import Trajectory

    tr = Trajectory.load(os.path.join(OUT, exp, "traj", run_id(row_cfg(row)) + ".npz"))
    t, spec = tr["t"], tr["spectra"]  # (n, E, chi_max), NaN-padded
    if spec.shape[2] < R:
        return dict(sR_end=np.nan, stall_kind="n/a")
    sR = np.nanmin(spec[:, :, R - 1], axis=1)  # smallest R-th singular value over edges
    end = sR[-1]
    if end < 1e-12:
        kind = "dead route"
    elif end >= 1e-4:
        kind = "long plateau"
    else:
        kind = "decaying"
    return dict(sR_end=float(end), stall_kind=kind)
