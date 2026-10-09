"""Events, plateaus and summary tables from saved trajectories (definitions of notes/diagnostics.md,
implemented in ttn.diagnostics).

    uv run python -m experiments.optimizers.analysis <sweep-name> [--refresh]
        -> experiments/optimizers/results/<sweep-name>_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.optimizers.run import OUT_ROOT
from ttn.diagnostics import crossing_times, entropy_rank, event_widths, separation_ratios
from ttn.dynamics import Trajectory

RESULTS = Path(__file__).parent / "results"


def event_trace(traj: Trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, overlaps, loss): the fine trace of discrete methods, else the recorded points."""
    d = traj.data
    if "fine_t" in d:
        return d["fine_t"], d["fine_overlaps"], d["fine_loss"]
    return d["t"], d["overlaps"], d["loss"]


def component_events(t: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Event time per component (a_r = 1/2; the last upward crossing if a transient above 1/2
    fell back) and width of the final rise (0.1 -> 0.9)."""
    first = crossing_times(t, a, 0.5)
    last = crossing_times(t, a, 0.5, last=True)
    times = np.where(np.isfinite(last), last, first)
    return times, event_widths(t, a)


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
    times, widths = component_events(t, a)
    for r, (tr, wr) in enumerate(zip(times, widths)):
        row[f"t_{r}"], row[f"w_{r}"] = tr, wr
        row[f"a_end_{r}"] = a[-1, r]
        row[f"a_min_{r}"] = np.min(np.abs(a[:, r]))
    ok = np.isfinite(times)
    order = np.argsort(np.where(ok, times, np.inf))[: ok.sum()]
    row["order"] = "o" + "".join(map(str, order))  # prefix keeps leading zeros through CSV
    row["n_events"] = int(ok.sum())
    rho = separation_ratios(times, widths)
    sorted_t = np.sort(times[ok])
    for i in range(len(sorted_t) - 1):
        row[f"gap_{i}"] = sorted_t[i + 1] - sorted_t[i]
        row[f"rho_{i}"] = rho[i]
    row["t_first"] = sorted_t[0] if ok.any() else np.nan
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
    old = pd.read_csv(cache, dtype={"order": str}, keep_default_na=False, na_values=[""]) \
        if (cache.exists() and not refresh) else pd.DataFrame()
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
