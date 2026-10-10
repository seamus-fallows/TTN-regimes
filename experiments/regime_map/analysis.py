"""Regime labels, boundaries and threshold sensitivity from saved runs (notes/diagnostics.md).

Input: results.jsonl (one summary row per run) + traj/<run_id>.npz (t, L/L0, overlaps A) under
<OUT>/<exp>/, OUT = $REGIME_MAP_OUT (default /root/outputs/regime_map; `make_figures.py --raw`).
Output: a per-run table with events, separation ratios and labels for every (delta_K threshold,
separation threshold) pair.

Fix round 1: event times are last upward crossings of 1/2 (notes/diagnostics.md). Rows written
before the fix (no 'event_rule') are relabelled from their trajectories (`relabel_last_crossing`);
the pre-fix values are kept in the columns *_old.
"""

from __future__ import annotations

import json
import warnings
import os

import numpy as np
import pandas as pd

OUT = os.environ.get("REGIME_MAP_OUT", "/root/outputs/regime_map")
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


def rid_of(row: dict) -> str:
    """run_id of a result row taken from a DataFrame: keys that are missing for this row (NaN
    because other rows have them, e.g. 'balance') are dropped first."""
    from run_regime import row_cfg, run_id

    return run_id(row_cfg({k: v for k, v in row.items() if not _is_missing(v)}))


def _traj_path(exp: str, row: dict) -> str:
    return os.path.join(OUT, exp, "traj", rid_of(row) + ".npz")


def relabel_last_crossing(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Event times by the fixed rule of `run_regime.final_rise` for every run: the last upward
    crossing of 1/2 before the component first reaches 0.9 (notes/diagnostics.md: a transient above
    1/2 that falls back before the component is learned does not set the event).

    Starts from the pre-fix event times (the stored `event_times` of pre-fix rows, `event_times_old`
    of rows written by the fixed driver) and replaces a component's time when it crossed 1/2 upward
    more than once before first reaching 0.9 on the recorded log + fine rows of its trajectory (fine
    rows resolve a rise with ~20 or more samples). The stored refined widths are kept: their window
    (last 0.1 before the first 0.9) is the one of the fixed rule. Adds n_up, bump (any n_up > 1), the
    pre-fix event_times_old / rhos_old, and relabelled = whether the times changed."""
    from run_regime import final_rise

    from ttn.dynamics import merged_rows

    df = df.copy()
    if "event_rule" not in df.columns:
        df["event_rule"] = None
    new_t, new_rho, old_t, old_rho, nup, changed = [], [], [], [], [], []
    for _, r in df.iterrows():
        fixed = r.event_rule == "last"
        times = np.asarray(r.event_times_old if fixed else r.event_times, float)
        widths = np.asarray(r.event_widths, float)
        old_t.append(list(times))
        old_rho.append(list(r.rhos_old if fixed else r.rhos))
        path = _traj_path(exp, r.to_dict())
        if not os.path.exists(path):  # cannot be checked: keep the pre-fix values (flagged)
            new_t.append(list(times))
            new_rho.append(list(old_rho[-1]))
            nup.append([-1] * len(times))
            changed.append(False)
            continue
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in ("t", "overlaps", "fine_t", "fine_overlaps") if k in z.files}
        m = merged_rows(data, keys=("overlaps",))
        t2, n2 = times.copy(), []
        for c in range(m["overlaps"].shape[1]):
            t_mid, _, n = final_rise(m["t"], m["overlaps"][:, c])
            n2.append(n)
            if n > 1:
                t2[c] = t_mid
        new_t.append(list(t2))
        new_rho.append([float(x) for x in _ratios(t2, widths)])
        nup.append(n2)
        changed.append(bool(np.any(~np.isclose(t2, times, equal_nan=True))))
    df["event_times"], df["rhos"] = new_t, new_rho
    df["event_times_old"], df["rhos_old"], df["n_up"] = old_t, old_rho, nup
    df["bump"] = [bool(np.any(np.asarray(n) > 1)) for n in nup]
    df["relabelled"] = changed
    return df


TENSOR_CACHE = os.environ.get("REGIME_MAP_TENSOR_CACHE")  # dir of <exp>.csv (run_id, dK_tensor)
# kernel of the label rule: "probe" (128-probe kernel, fix round 1) or "tensor" (tensor-space kernel
# J D J^T of notes/theory_regimes.md; the lead's choice after round 2). With "tensor", dK holds the
# tensor-space value and the probe value moves to dK_probe.
LABEL_KERNEL = os.environ.get("REGIME_MAP_LABEL_KERNEL", "probe")


def tensor_kernel_row(args) -> tuple[str, float]:
    """(run_id, dK_tensor) of one saved run: theta_end from its trajectory, the initial point rebuilt
    from the config (same seeds). Module-level so that spawned workers can run it."""
    import torch
    from run_regime import build, row_cfg, tensor_kernel_change

    raw, exp, r = args
    torch.set_num_threads(1)
    rid = rid_of(r)
    path = os.path.join(raw, exp, "traj", rid + ".npz")
    if not os.path.exists(path):
        return rid, np.nan
    with np.load(path, allow_pickle=False) as z:
        theta_end = torch.from_numpy(z["theta_end"])
    cfg = {k: v for k, v in row_cfg(r).items() if not _is_missing(v)}
    problem, theta0, _, _ = build(cfg)
    return rid, float(tensor_kernel_change(problem, theta0, theta_end))


def _cache_path(exp: str):
    return os.path.join(TENSOR_CACHE, f"{exp}.csv") if TENSOR_CACHE else None


def read_tensor_cache(exp: str) -> dict:
    path = _cache_path(exp)
    if path is None or not os.path.exists(path):
        return {}
    c = pd.read_csv(path)
    return dict(zip(c.run_id, c.dK_tensor))


def add_tensor_kernel(df: pd.DataFrame, exp: str, mask=None, n_proc: int = 1) -> pd.DataFrame:
    """Fill dK_tensor (tensor-space kernel change, `run_regime.tensor_kernel_change`) for pre-fix
    rows from theta_end of their trajectories (`tensor_kernel_row`). mask: optional boolean Series
    selecting the rows to compute (a few seconds per run at chi = 128); n_proc > 1: spawned pool.
    New values are appended to the cache TENSOR_CACHE/<exp>.csv when TENSOR_CACHE is set."""
    df = df.copy()
    if "dK_tensor" not in df.columns:
        df["dK_tensor"] = np.nan
    cached = read_tensor_cache(exp)
    ids = [rid_of(r) for r in df.to_dict("records")]
    df["dK_tensor"] = [
        v if np.isfinite(v) else cached.get(i, np.nan)
        for v, i in zip(df.dK_tensor.astype(float), ids)
    ]
    sel = df.dK_tensor.isna() if mask is None else (df.dK_tensor.isna() & mask)
    todo = [(OUT, exp, df.iloc[i].to_dict()) for i in np.flatnonzero(sel.values)]
    if not todo:
        return df
    if n_proc > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(min(n_proc, len(todo))) as pool:
            res = pool.map(tensor_kernel_row, todo, chunksize=1)
    else:
        res = [tensor_kernel_row(a) for a in todo]
    new = dict(res)
    df["dK_tensor"] = [
        new.get(i, v) if not np.isfinite(v) else v for v, i in zip(df.dK_tensor.astype(float), ids)
    ]
    path = _cache_path(exp)
    if path is not None:
        os.makedirs(TENSOR_CACHE, exist_ok=True)
        pd.DataFrame({"run_id": list(new), "dK_tensor": list(new.values())}).to_csv(
            path, mode="a", header=not os.path.exists(path), index=False
        )
    return df


def _is_missing(v):
    return v is None or (isinstance(v, float) and not np.isfinite(v))


def _ratios(times, widths):
    from ttn.diagnostics import separation_ratios as sr

    return sr(np.asarray(times, float), np.asarray(widths, float))


_LOADED: dict = {}  # (OUT, exp, cache size) -> table; relabelling reads every trajectory once


def load(exp: str) -> pd.DataFrame:
    """Per-run table with events, rho's and labels at every threshold pair.

    Library runs (run_regime.py) carry event_times / event_widths / rhos in their rows (pre-fix rows
    are relabelled with the last-crossing rule, `relabel_last_crossing`); interim runs
    (pilots/interim.py) are re-analysed from their saved trajectories. Memoized per raw root."""
    key = (OUT, exp, len(read_tensor_cache(exp)), LABEL_KERNEL)
    if key not in _LOADED:
        _LOADED[key] = _load(exp)
    return _LOADED[key].copy()


def _load(exp: str) -> pd.DataFrame:
    rows = []
    with open(os.path.join(OUT, exp, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    # de-duplicate reruns of the same config (early rows stored a recomputed alpha, so the resume
    # check missed them): key on the config with sigma / alpha rounded, keep the latest row
    key = pd.DataFrame(
        {
            c: df[c] if c in df.columns else None
            for c in ("model", "kind", "n", "d", "chi", "seed", "root_scale", "lr", "balance")
        }
    )
    key["sigma_r"] = [float(f"{x:.4g}") for x in df.sigma]
    key["alpha_r"] = [
        float(f"{x:.3g}") if isinstance(x, float) and np.isfinite(x) else None
        for x in (
            df.alpha if "alpha" in df.columns and "root_scale" in df.columns else [None] * len(df)
        )
    ]
    df = df.loc[~key.astype(str).duplicated(keep="last").values].reset_index(drop=True)
    if "rhos" in df.columns:
        for c in ("event_times", "event_widths", "rhos", "event_times_old", "rhos_old"):
            if c in df.columns:  # JSON nulls -> NaN
                df[c] = [
                    [np.nan if x is None else float(x) for x in v] if isinstance(v, list) else v
                    for v in df[c]
                ]
        df = relabel_last_crossing(df, exp)
        df["events"] = [list(zip(t, w)) for t, w in zip(df.event_times, df.event_widths)]
        df["min_rho_old"] = [
            float(np.min(np.asarray(r, float))) if len(r) else np.nan for r in df.rhos_old
        ]
        df["label_old"] = [label(le, dk, rh) for le, dk, rh in zip(df.l_end, df.dK, df.rhos_old)]
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
    cached = read_tensor_cache(exp)  # tensor-space kernel computed for pre-fix rows (fix round 1)
    if cached:
        cur = df.dK_tensor.astype(float) if "dK_tensor" in df.columns else pd.Series(np.nan, index=df.index)
        ids = [rid_of(r) for r in df.to_dict("records")]
        df["dK_tensor"] = [v if np.isfinite(v) else cached.get(i, np.nan) for v, i in zip(cur, ids)]
    df["min_rho"] = [float(np.min(np.asarray(r, float))) if len(r) else np.nan for r in df.rhos]
    df["n_events"] = [sum(np.isfinite(t) for t, _ in ev) for ev in df.events]
    df["dK_probe"] = df.dK
    if LABEL_KERNEL == "tensor":
        tens = df.dK_tensor.astype(float) if "dK_tensor" in df.columns else pd.Series(np.nan, index=df.index)
        need = (df.l_end < 1e-3) & tens.isna()  # converged runs: the lazy label depends on the kernel
        if need.any():
            warnings.warn(f"{exp}: {int(need.sum())} converged runs lack dK_tensor; their labels use NaN")
        df["dK"] = tens
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
    """One seed's S2S boundary from its labels along sigma.

    Uses the first contiguous block of S2S runs (runs below it that stalled -- plateaus longer than the
    budget or dead routes at small sigma -- are skipped and flagged): sigma_c = geometric mean of the
    block's largest sigma and the next sigma. Returns (sigma_c, status, lo, hi): status 'ok' (the next
    run is rich or lazy), 'unresolved' (the next run stalled, so the S2S block may end anywhere in
    [lo, hi]), 'never_s2s', 'all_s2s'; a suffix '+stall_below' marks stalled runs below the block."""
    r = rows.sort_values("sigma")
    lab, sig = r[label_col].values, r.sigma.values
    s2s = np.flatnonzero(lab == "s2s")
    if s2s.size == 0:
        return np.nan, "never_s2s", np.nan, np.nan
    start = s2s[0]
    below = "+stall_below" if (lab[:start] == "stalled").any() else ""
    if (
        lab[:start] != "stalled"
    ).any():  # a rich/lazy run below the first S2S run: not a clean block
        start_status = "mixed_below"
        below = "+" + start_status
    end = start
    while end + 1 < len(lab) and lab[end + 1] == "s2s":
        end += 1
    if end == len(lab) - 1:
        return np.nan, "all_s2s" + below, sig[end], np.nan
    lo, hi = sig[end], sig[end + 1]
    status = ("unresolved" if lab[end + 1] == "stalled" else "ok") + below
    return float(np.sqrt(lo * hi)), status, float(lo), float(hi)


def seed_kernel_boundary(
    rows: pd.DataFrame, thr: float = 0.1, converged_only: bool = True, col: str = "dK"
):
    """One seed's kernel boundary: sigma where delta_K(T) crosses `thr` from above (log-log
    interpolation between the bracketing runs; the last crossing if delta_K is not monotone).
    With converged_only, stalled runs are excluded (their delta_K is not an end-of-training value).
    col: 'dK' (128-probe kernel, the label rule) or 'dK_tensor' (tensor-space kernel, theory)."""
    r = rows.sort_values("sigma")
    if converged_only:
        r = r[r.label != "stalled"]
    if col not in r.columns:
        return np.nan
    r = r[np.isfinite(r[col].astype(float).values)]
    sig, dk = r.sigma.values, r[col].astype(float).values
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
    return (
        float(v.mean()),
        float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan,
        int(v.size),
    )


def per_seed_table(df: pd.DataFrame, by: list[str], K: int | None = None) -> pd.DataFrame:
    """Per (group, seed): S2S boundary (+status), aligned-extrapolation prediction, kernel boundary,
    and the rho exponent on the S2S side (slope of log min_rho vs log sigma over the S2S runs)."""
    out = []
    for key, g in df.groupby(by + ["seed"]):
        k = K or int(g.K.iloc[0])
        sc, status, lo, hi = seed_s2s_boundary(g)
        w = (
            g.weights.iloc[0]
            if "weights" in g.columns and isinstance(g.weights.iloc[0], list)
            else (3.0, 2.0, 1.0)
        )
        pred, _ = aligned_sigma_c(g, k, weights=tuple(w))
        s2s = g[g.label == "s2s"]
        slope, _ = seed_power_fit(s2s.sigma, s2s.min_rho) if len(s2s) >= 3 else (np.nan, np.nan)
        rc, rstatus, rlo, rhi = seed_rho_boundary_stalls(g)
        row = dict(zip(by + ["seed"], key if isinstance(key, tuple) else (key,)))
        row.update(
            K=k,
            sigma_c=sc,
            s2s_status=status,
            s2s_lo=lo,
            s2s_hi=hi,
            sigma_c_rho=rc,
            rho_status=rstatus,
            rho_lo=rlo,
            rho_hi=rhi,
            sigma_c_pred=pred,
            sigma_K01=seed_kernel_boundary(g),
            sigma_K01_tensor=seed_kernel_boundary(g, col="dK_tensor"),
            rho_slope=slope,
            n_s2s=len(s2s),
            n_runs=len(g),
            n_bump=int(g.bump.sum()) if "bump" in g.columns else 0,
            n_relabelled=int(g.relabelled.sum()) if "relabelled" in g.columns else 0,
        )
        out.append(row)
    return pd.DataFrame(out)


def seed_rho_boundary_stalls(rows: pd.DataFrame, thr: float = 3.0):
    """One seed's S2S boundary in rho when some runs stall (fix round 1, K sweep).

    Runs are ordered by sigma and classified by min_i rho_i whatever their label, so a run that
    learned every component but ended on the floor of an unrealizable centred target counts as
    S2S-like (rho > thr) or rich-like. Runs without a ratio (fewer than two completed events: a long
    or dead route) carry no information on the separation and are skipped. Returns
    (sigma_c, status, lo, hi): the log-log interpolated crossing of thr between the last run above
    it (lo) and the next run with a ratio (hi); status 'ok', 'gap' (a run without a ratio lies
    between lo and hi), 'never_above' or 'all_above' (sigma_c = NaN, lo = largest sigma above)."""
    r = rows.sort_values("sigma")
    sig, rho = r.sigma.values, r.min_rho.values.astype(float)
    has = np.isfinite(rho)
    if not has.any() or not (rho[has] > thr).any():
        return np.nan, "never_above", np.nan, np.nan
    idx = np.flatnonzero(has)
    first = idx[np.argmax(rho[idx] > thr)]
    after = idx[idx > first]
    drop = after[rho[after] <= thr]
    if drop.size == 0:
        return np.nan, "all_above", float(sig[idx[-1]]), np.nan
    j = drop[0]
    i = idx[idx < j][-1]
    status = "gap" if (~has[i + 1 : j]).any() else "ok"
    w = (np.log(thr) - np.log(rho[i])) / (np.log(rho[j]) - np.log(rho[i])) if rho[j] > 0 else 0.5
    sc = float(np.exp(np.log(sig[i]) + w * (np.log(sig[j]) - np.log(sig[i]))))
    return sc, status, float(sig[i]), float(sig[j])


def stall_type(exp: str, row: dict, R: int = 3) -> dict:
    """Classify a stalled run from its bond spectra (theory_regimes section 7): the R-th singular
    value of every edge at the end; 'dead route' if its minimum over edges is < 1e-12 (at roundoff:
    the route cannot be revived by the float64 gradient), 'long plateau' if >= 1e-4, 'decaying'
    in between."""
    from ttn.dynamics import Trajectory

    tr = Trajectory.load(_traj_path(exp, row))
    # spectra of W itself: 'spectra' for plain runs; centred runs record W - W_0 there and W in
    # 'spectra_W' (runs from 2026-10-09 21:00 on); older centred runs cannot be classified
    key = (
        "spectra_W"
        if "spectra_W" in tr.data
        else ("spectra" if row.get("model") != "centred" else None)
    )
    if key is None:
        return dict(sR_end=np.nan, stall_kind="n/a")
    t, spec = tr["t"], tr[key]  # (n, E, chi_max), NaN-padded
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


def aligned_boundary_a0(K: int, weights=(3.0, 2.0, 1.0), rho_thr: float = 3.0) -> float:
    """Exact S2S boundary for aligned balanced init (theory_regimes section 3.1): the per-core
    amplitude a0 at which min_i rho_i = rho_thr, every component starting from u0 = a0^K."""
    from scipy.optimize import brentq

    w = np.sort(np.asarray(weights, float))[::-1]
    expo = 2 - 2 / K
    omega = _I(K, 0.1, 0.9)

    def min_rho(log_a0):
        u0 = np.exp(K * log_a0)
        t = [s**-expo * _I(K, min(u0 / s, 0.4999), 0.5) for s in w]
        wid = [s**-expo * omega for s in w]
        return min((t[i + 1] - t[i]) / max(wid[i], wid[i + 1]) for i in range(len(w) - 1))

    return float(
        np.exp(
            brentq(
                lambda la: min_rho(la) - rho_thr,
                np.log(1e-4),
                np.log(0.999 * (0.5 * w[-1]) ** (1 / K)),
            )
        )
    )


def seed_rho_boundary(rows: pd.DataFrame, thr: float = 3.0):
    """One seed's S2S boundary in rho: the sigma where min_i rho_i first drops below `thr` coming
    from small sigma (log-log interpolation between the bracketing runs; runs without a ratio count
    as below the threshold)."""
    r = rows.sort_values("sigma")
    sig = r.sigma.values
    rho = np.where(np.isfinite(r.min_rho.values), r.min_rho.values, 0.0)
    above = rho > thr
    if not above[0]:
        return np.nan
    i = np.argmax(~above) if (~above).any() else -1
    if i <= 0:
        return np.nan
    if rho[i] <= 0:
        return float(np.sqrt(sig[i - 1] * sig[i]))
    w = (np.log(thr) - np.log(rho[i - 1])) / (np.log(rho[i]) - np.log(rho[i - 1]))
    return float(np.exp(np.log(sig[i - 1]) + w * (np.log(sig[i]) - np.log(sig[i - 1]))))


def sensitivity_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per (group, seed): continuous S2S boundaries at rho thresholds 2/3/5 and kernel boundaries
    at delta_K thresholds 0.05/0.1/0.2 (converged runs)."""
    out = []
    for key, g in df.groupby(by + ["seed"]):
        row = dict(zip(by + ["seed"], key if isinstance(key, tuple) else (key,)))
        for rt in RHO_THRESHOLDS:
            row[f"s2s_{rt:g}"] = seed_rho_boundary(g, rt)
        for kt in K_THRESHOLDS:
            row[f"lazy_{kt:g}"] = seed_kernel_boundary(g, kt)
        if "dK_tensor" in g.columns:
            for kt in K_THRESHOLDS:
                row[f"lazy_tensor_{kt:g}"] = seed_kernel_boundary(g, kt, col="dK_tensor")
        row["n_runs"] = len(g)
        row["n_bump"] = int(g.bump.sum()) if "bump" in g.columns else 0
        out.append(row)
    return pd.DataFrame(out)


def width_resolution(exp: str, row: dict, n_dense: int = 200) -> list[float]:
    """Time resolution of each refined event width: (t[i1] - t[i0]) / n_dense for the window that
    run_regime.refine_events re-integrated (last a_r < 0.1 before the first a_r >= 0.9, coarse log
    grid). A width below ~5 resolutions is an upper bound, so the run's rho values are lower bounds."""
    path = _traj_path(exp, row)
    if not os.path.exists(path):
        return []
    with np.load(path, allow_pickle=False) as z:
        if "overlaps" not in z.files:
            return []
        t, A = z["t"], z["overlaps"]
    out = []
    for r in range(A.shape[1]):
        hi = np.flatnonzero(A[:, r] >= 0.9)
        if hi.size == 0:
            out.append(np.nan)
            continue
        i1 = hi[0]
        lo = np.flatnonzero(A[:i1, r] < 0.1)
        i0 = lo[-1] if lo.size else 0
        out.append((t[i1] - t[i0]) / n_dense)
    return out


def add_resolution_flags(df: pd.DataFrame, exp: str, factor: float = 5.0) -> pd.DataFrame:
    """Columns: width_res (per event), rho_resolved (every width >= factor x its resolution), and
    label_resolved: the label, or 'undetermined' when the S2S/rich decision rests on an unresolved
    width (rho <= 3 with an unresolved width could be S2S; rho > 3 is a lower bound and stays S2S)."""
    res, ok = [], []
    for _, r in df.iterrows():
        wr = width_resolution(exp, r.to_dict())
        widths = [w for _, w in r.events]
        good = all(
            (not np.isfinite(w)) or (not np.isfinite(q)) or w >= factor * q
            for w, q in zip(widths, wr)
        )
        res.append(wr)
        ok.append(good)
    df = df.copy()
    df["width_res"] = res
    df["rho_resolved"] = ok
    df["label_resolved"] = np.where(
        (df.label == "rich") & (~df.rho_resolved) & (df.n_events >= 2), "undetermined", df.label
    )
    return df
