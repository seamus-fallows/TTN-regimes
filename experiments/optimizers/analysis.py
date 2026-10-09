"""Events, plateaus and summary tables from saved trajectories (definitions of notes/diagnostics.md).

    uv run python -m experiments.optimizers.analysis <sweep-name>   -> results/<sweep-name>_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.optimizers.run import OUT_ROOT

RESULTS = Path(__file__).parent / "results"


def crossing(t: np.ndarray, a: np.ndarray, level: float, start: int = 0) -> tuple[float, int]:
    """First time at or after index `start` where a reaches `level` (linear interpolation)."""
    idx = np.where(a[start:] >= level)[0]
    if len(idx) == 0:
        return np.nan, -1
    i = start + idx[0]
    if i == 0:
        return float(t[0]), 0
    a0, a1 = a[i - 1], a[i]
    frac = (level - a0) / (a1 - a0) if a1 != a0 else 1.0
    return float(t[i - 1] + frac * (t[i] - t[i - 1])), i


def last_up_crossing(t, a, level, before: int) -> float:
    """Last upward crossing of `level` before index `before` (start of the final rise)."""
    below = np.where(a[:before] < level)[0]
    if len(below) == 0:
        return float(t[0])
    i = below[-1]
    a0, a1 = a[i], a[i + 1]
    frac = (level - a0) / (a1 - a0) if a1 != a0 else 0.0
    return float(t[i] + frac * (t[i + 1] - t[i]))


def component_events(t: np.ndarray, a: np.ndarray) -> list[dict]:
    """Event per component r: time a_r first reaches 1/2, width of the final rise 0.1 -> 0.9."""
    out = []
    for r in range(a.shape[1]):
        t_half, i_half = crossing(t, a[:, r], 0.5)
        t90, i90 = crossing(t, a[:, r], 0.9)
        width = np.nan
        if i90 >= 0:
            width = t90 - last_up_crossing(t, a[:, r], 0.1, i90)
        # a transient above 1/2 that falls back: use the last upward crossing (diagnostics.md)
        if i_half >= 0 and i90 >= 0 and np.any(a[i_half:i90, r] < 0.5):
            t_half = last_up_crossing(t, a[:, r], 0.5, i90)
        out.append({"component": r, "t": t_half, "width": width})
    return out


def separation(events: list[dict]) -> tuple[list[int], list[float], list[float]]:
    ev = sorted([e for e in events if np.isfinite(e["t"])], key=lambda e: e["t"])
    order = [e["component"] for e in ev]
    gaps, rhos = [], []
    for e1, e2 in zip(ev[:-1], ev[1:]):
        gap = e2["t"] - e1["t"]
        gaps.append(gap)
        rhos.append(gap / max(e1["width"], e2["width"]))
    return order, gaps, rhos


def effective_ranks(spectra: np.ndarray, teacher_spectra: np.ndarray) -> np.ndarray:
    """Per edge: #{k: s_k > 0.1 x smallest nonzero teacher Schmidt value on that edge}."""
    out = []
    for sp, ts in zip(spectra, teacher_spectra):
        ts = ts[np.isfinite(ts)]
        nonzero = ts[ts > 1e-10 * ts.max()]
        out.append(int(np.sum(np.nan_to_num(sp) > 0.1 * nonzero.min())))
    return np.array(out)


def entropy_rank(sp: np.ndarray) -> float:
    s = np.nan_to_num(sp)
    p = s**2 / np.sum(s**2)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def summarize(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    info = eval(str(z["info"]), {"np": np, "nan": np.nan})  # repr of a dict of plain numbers
    t, a, loss = z["time"], z["overlaps"], z["loss"]
    events = component_events(t, a)
    order, gaps, rhos = separation(events)
    row = {"file": path.name, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in cfg.items()},
           **{f"info_{k}": v for k, v in info.items()}}
    row["loss0"], row["loss_end"] = loss[0], loss[-1]
    row["converged"] = bool(loss[-1] < 1e-3 * loss[0])
    for e in events:
        row[f"t_{e['component']}"] = e["t"]
        row[f"w_{e['component']}"] = e["width"]
    row["order"] = "".join(map(str, order))
    row["n_events"] = len(order)
    for i, (g, r) in enumerate(zip(gaps, rhos)):
        row[f"gap_{i}"], row[f"rho_{i}"] = g, r
    row["t_first"] = min((e["t"] for e in events if np.isfinite(e["t"])), default=np.nan)
    row["s2s"] = bool(len(order) >= 2 and all(r > 3 for r in rhos))
    ts = z["teacher_spectra"]
    sp_end = z["h_spectra"][-1]
    ranks = effective_ranks(sp_end, ts)
    row["eff_ranks_end"] = "".join(map(str, ranks))
    row["entropy_rank_mean_end"] = np.mean([entropy_rank(s) for s in sp_end])
    bal = z["h_balance"]
    row["balance0_max"], row["balance_end_max"] = bal[0].max(), bal[-1].max()
    row["balance_end_mean"] = bal[-1].mean()
    row["q_drift_end_max"] = z["h_q_drift"][-1].max()
    row["dtheta_end"] = z["dtheta"][-1]
    return row


def collect(name: str, refresh: bool = False) -> pd.DataFrame:
    out_dir = OUT_ROOT / name
    cache = RESULTS / f"{name}_summary.csv"
    old = pd.read_csv(cache) if (cache.exists() and not refresh) else pd.DataFrame()
    seen = set(old["file"]) if len(old) else set()
    rows = [summarize(p) for p in sorted(out_dir.glob("*.npz"))
            if not p.name.endswith(".tmp.npz") and p.name not in seen]
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def load(name: str, file: str) -> dict:
    z = np.load(OUT_ROOT / name / file)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    df = collect(sys.argv[1], refresh="--refresh" in sys.argv)
    print(df.shape)
