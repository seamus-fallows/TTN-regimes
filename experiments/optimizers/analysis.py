"""Events, plateaus and summary tables from saved trajectories (definitions of notes/diagnostics.md,
implemented in ttn.diagnostics).

    uv run python -m experiments.optimizers.analysis <sweep-name> [--refresh]
        -> $OPTIMIZERS_RESULTS/<sweep-name>_summary.csv

Summaries, tables and figures go to $OPTIMIZERS_RESULTS (default /root/outputs/optimizers/results, outside
git, so re-running never overwrites the committed results); the committed ones in
experiments/optimizers/results were made with `make_figures.py --results-dir experiments/optimizers/results`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.optimizers.run import OUT_ROOT
from ttn.diagnostics import crossing_times, entropy_rank, event_widths, separation_ratios
from ttn.dynamics import Trajectory

RESULTS = Path(os.environ.get("OPTIMIZERS_RESULTS", "/root/outputs/optimizers/results"))


def event_trace(traj: Trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, overlaps, loss): the fine trace of discrete methods, else the recorded points."""
    d = traj.data
    if "fine_t" in d:
        return d["fine_t"], d["fine_overlaps"], d["fine_loss"]
    return d["t"], d["overlaps"], d["loss"]


def local_spacing(t: np.ndarray, te: float) -> float:
    """Output interval of the trace around time te."""
    i = int(np.clip(np.searchsorted(t, te), 1, len(t) - 1))
    return float(t[i] - t[i - 1])


def component_events(t: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Event time per component on a sampled trace: the last upward crossing of a_r = 1/2 after which
    a_r stays >= 1/2 (NaN if the component ends below 1/2, so transient crossings are never events),
    and the width of the final rise (0.1 -> 0.9)."""
    return crossing_times(t, a, 0.5, last=True), event_widths(t, a)


def exact_events(d: dict, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    """Events from root-found crossings (GF runs with `xing_<r>_<level>`): a_r starts below every level,
    so crossings alternate up / down; the event is the last crossing of 1/2 if their number is odd (ends
    above), its width = (first crossing of 0.9 after it) - (last crossing of 0.1 before it)."""
    times, widths = np.full(n_comp, np.nan), np.full(n_comp, np.nan)
    for r in range(n_comp):
        c10, c50, c90 = (np.asarray(d.get(f"xing_{r}_{lev}", np.array([])), dtype=float)
                         for lev in (0.1, 0.5, 0.9))
        if len(c50) % 2 == 0:
            continue
        te = c50[-1]
        times[r] = te
        lo = c10[c10 <= te]
        hi = c90[c90 >= te]
        if len(lo) and len(hi):
            widths[r] = hi[0] - lo[-1]
    return times, widths


def overshoot_return(t: np.ndarray, loss: np.ndarray) -> int:
    """Index of the first sample after the last one with L > L(0): events are only defined once a
    loss overshoot (Gram-preconditioned GD's first step) has returned; 0 if there is none."""
    over = np.flatnonzero(loss > loss[0] * (1 + 1e-9))
    return int(over[-1] + 1) if over.size else 0


def effective_ranks(spectra: np.ndarray, teacher_spectra: np.ndarray) -> np.ndarray:
    """Per edge: #{k: s_k > 0.1 x smallest nonzero teacher Schmidt value on that edge}."""
    out = []
    for sp, ts in zip(spectra, teacher_spectra):
        ts = ts[np.isfinite(ts)]
        nonzero = ts[ts > 1e-10 * ts.max()]
        out.append(int(np.sum(np.nan_to_num(sp) > 0.1 * nonzero.min())))
    return np.array(out)


def spectral_events(t: np.ndarray, spectra: np.ndarray, teacher: np.ndarray) -> dict:
    """Rank-growth pattern from the bond spectra (notes/diagnostics.md, general-teacher events):
    event (e, k) when s_k^e first reaches half its teacher value, width 10% -> 90% of it.

    Returns: incremental = fraction of edges whose events come in the order k = 1, 2, ... with every
    separation ratio > 3 (one singular value at a time); rho_min = median over edges of the smallest
    separation ratio between consecutive k; edge_spread = median over k of
    (latest - earliest edge time) / latest edge time (0: all edges at once; ~1: edge by edge)."""
    n_edges = spectra.shape[1]
    times = np.full((n_edges, spectra.shape[2]), np.nan)
    inc, rho_mins = [], []
    for e in range(n_edges):
        ts = teacher[e][np.isfinite(teacher[e])]
        ranks = int(np.sum(ts > 1e-8 * ts.max()))
        curves = spectra[:, e, :ranks] / ts[:ranks]
        te = crossing_times(t, curves, 0.5)
        we = event_widths(t, curves)
        times[e, :ranks] = te
        if ranks >= 2 and np.all(np.isfinite(te)):
            rho = separation_ratios(te, we)
            inc.append(bool(np.all(np.diff(te) > 0) and np.all(rho > 3)))
            rho_mins.append(float(np.nanmin(rho)))
    spread = []
    for k in range(times.shape[1]):
        col = times[:, k][np.isfinite(times[:, k])]
        if len(col) >= 2 and col.max() > 0:
            spread.append((col.max() - col.min()) / col.max())
    return {"incremental": float(np.mean(inc)) if inc else np.nan,
            "rho_min": float(np.median(rho_mins)) if rho_mins else np.nan,
            "edge_spread": float(np.median(spread)) if spread else np.nan}


def summarize(path: Path) -> dict:
    traj = Trajectory.load(path)
    d, meta = traj.data, traj.meta
    cfg = meta["cfg"]
    row = {"file": path.name}
    row.update({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in cfg.items()})
    for k in ("stop_reason", "n_steps", "n_evals", "n_sweeps", "n_micro", "nfev", "njev",
              "wall_seconds", "cpu_seconds", "t_end"):
        if k in meta:
            row[f"meta_{k}"] = meta[k]
    t, a, loss = event_trace(traj)
    row["loss0"], row["loss_end"] = d["loss"][0], d["loss"][-1]
    row["pop_loss_end"] = d["pop_loss"][-1]
    row["converged"] = bool(d["loss"][-1] < 1e-3 * d["loss"][0])
    rel = loss / loss[0]
    for q in (1e-1, 1e-3):  # time to reduce the loss by 10x and to the 'converged' level 1e-3
        hit = np.flatnonzero(rel < q)
        row[f"t_loss_{q:g}"] = t[hit[0]] if hit.size else np.nan
    row["loss_max_rel"] = float(np.nanmax(rel))
    n_comp = a.shape[1]
    exact = "xing_0_0.5" in d
    i0 = 0 if exact else overshoot_return(t, loss)
    row["t_return"] = t[min(i0, len(t) - 1)] if i0 > 0 else 0.0
    row["event_location"] = "exact" if exact else "trace"
    if exact:
        times, widths = exact_events(d, n_comp)
        sample = np.full(n_comp, -1)
    else:
        tt, aa = t[i0:], a[i0:]
        times, widths = component_events(tt, aa) if len(tt) > 1 else (np.full(n_comp, np.nan),) * 2
        # sample from which a_r stays >= 1/2 (ties = several components in the same step / micro-step)
        sample = np.full(n_comp, -1)
        for r in range(n_comp):
            below = np.flatnonzero(aa[:, r] < 0.5)
            if np.isfinite(times[r]):
                sample[r] = (below[-1] + 1 if below.size else 0) + i0
        # learned already when the overshoot returned: no resolved event
        row["n_events_at_return"] = int(np.sum([(sample[r] == i0) and i0 > 0 for r in range(n_comp)]))
    for r, (tr, wr) in enumerate(zip(times, widths)):
        row[f"t_{r}"], row[f"w_{r}"] = tr, wr
        row[f"a_end_{r}"] = a[-1, r]
        row[f"a_min_{r}"] = np.min(np.abs(a[:, r]))
        row[f"sample_{r}"] = sample[r]
    ok = np.isfinite(times)
    order = np.argsort(np.where(ok, times, np.inf))[: ok.sum()]
    row["order"] = "o" + "".join(map(str, order))  # prefix keeps leading zeros through CSV
    learned = sample[ok]
    # the temporal order is resolved only if no two events fall into the same sample (step)
    row["order_resolved"] = bool(exact or len(set(learned.tolist())) == len(learned))
    if not exact and i0 > 0:
        row["order_resolved"] = row["order_resolved"] and row["n_events_at_return"] == 0
    row["n_events"] = int(ok.sum())
    rho = separation_ratios(times, widths)
    sorted_t = np.sort(times[ok])
    spacing = [0.0 if exact else local_spacing(t, te) for te in sorted_t]
    for i in range(len(sorted_t) - 1):
        row[f"gap_{i}"] = sorted_t[i + 1] - sorted_t[i]
        row[f"rho_{i}"] = rho[i]
        # a gap shorter than two output intervals at its ends is not a measurement
        row[f"gap_{i}_resolved"] = bool(row[f"gap_{i}"] > 2 * max(spacing[i], spacing[i + 1]))
    row["t_first"] = sorted_t[0] if ok.any() else np.nan
    row["t_first_resolved"] = bool(ok.any() and sorted_t[0] > 2 * spacing[0])
    row["s2s"] = bool(ok.sum() >= 2 and np.all(rho > 3))
    for k, v in spectral_events(d["t"], d["spectra"], d["teacher_spectra"]).items():
        row[f"spec_{k}"] = v
    sp_end = d["spectra"][-1]
    row["eff_ranks_end"] = "".join(map(str, effective_ranks(sp_end, d["teacher_spectra"])))
    row["entropy_rank_mean_end"] = float(np.mean(entropy_rank(sp_end)))
    bal = d["balance"]
    row["balance0_max"], row["balance_end_max"] = bal[0].max(), bal[-1].max()
    row["balance_end_mean"] = bal[-1].mean()
    row["balance_max_over_run"] = bal.max()
    row["cons_drift_end"] = d["conservation_drift"][-1]
    row["cons_drift_max"] = np.nanmax(d["conservation_drift"])
    row["dtheta_end"] = d["rel_dist_from_init"][-1]
    return row


def collect(name: str, refresh: bool = False) -> pd.DataFrame:
    out_dir = OUT_ROOT / name
    cache = RESULTS / f"{name}_summary.csv"
    old = pd.read_csv(cache, dtype={"order": str}, keep_default_na=False, na_values=[""],
                      float_precision="round_trip") if (cache.exists() and not refresh) else pd.DataFrame()
    seen = set(old["file"]) if len(old) else set()
    rows = [summarize(p) for p in sorted(out_dir.glob("*.npz"))
            if not p.name.endswith(".tmp.npz") and p.name not in seen]
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def load(name: str, file: str) -> Trajectory:
    return Trajectory.load(OUT_ROOT / name / file)


if __name__ == "__main__":
    df = collect(sys.argv[1], refresh="--refresh" in sys.argv)
    print(df.shape)
