"""Per-run summary of real-data trajectories -> CSV + JSON.

Usage: uv run python experiments/real_data/summarize.py <runs_dir> <out_prefix> [workers]

Per run (definitions in analysis.py): t_a (end of phase 1), t1 = absolute time of the first loss drop
(the escape from the origin), the loss drops after it (rate peaks, FWHM widths, sizes), separation
ratios, the S2S signature (i)-(ii) of notes/diagnostics.md on these drops, gaps between consecutive
drops, which edges gain a singular value (s_k/s_1 crossing) inside each drop's window, time to
reach train-loss levels, final train/test loss and accuracy, the integrator's conservation drift.
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis import edge_sites, load, loss_drops, time_to_loss, tree_levels  # noqa: E402

from ttn.diagnostics import separation_ratios  # noqa: E402

LOSS_LEVELS = (0.3, 0.25, 0.2, 0.15, 0.12, 0.1)
RATIO_LEVEL = 0.05  # s_k/s_1 threshold for "edge e has gained its k-th singular value"


def rank_events(t, fs, level=RATIO_LEVEL):
    """First time s_k/s_1 >= level on each edge, per k >= 2 (NaN if never; t[0] if already)."""
    ratio = fs[:, :, 1:] / fs[:, :, :1]
    out = np.full(ratio.shape[1:], np.nan)
    for e in range(ratio.shape[1]):
        for k in range(ratio.shape[2]):
            above = np.flatnonzero(np.nan_to_num(ratio[:, e, k]) >= level)
            if above.size:
                i = above[0]
                if i == 0:
                    out[e, k] = t[0]
                else:
                    a, b = ratio[i - 1, e, k], ratio[i, e, k]
                    f = (np.log(level) - np.log(max(a, 1e-300))) / (np.log(b) - np.log(max(a, 1e-300)))
                    out[e, k] = t[i - 1] + f * (t[i] - t[i - 1])
    return out


def summarize(path) -> dict:
    run = load(path)
    m, c = run["meta"], run["meta"]["case"]
    row = {k: c[k] for k in ("tree", "ordering", "features", "chi", "sigma", "seed", "centred",
                             "n_train")}
    row.update(name=m["name"], stop=m["stop"], wall=m["wall_total"], t_a=m.get("t_a", np.nan))
    if "p2" not in run:
        return row
    p2, lg = run["p2"], run["p2_log"]
    t, loss, fs = p2["t"], p2["loss"], p2["fspectra"]
    drops = loss_drops(t, loss)
    row["n_drops"] = len(drops)
    row["drops"] = drops
    times = [d["t"] for d in drops]
    widths = [d["width"] for d in drops]
    rho = separation_ratios(times, widths) if len(drops) > 1 else np.array([])
    row["rho"] = rho.tolist()
    row["s2s"] = bool(len(drops) >= 2 and np.all(rho > 3))
    row["t1"] = m["t_a"] + times[0] if drops else np.nan
    row["gaps"] = np.diff(times).tolist()
    row["unresolved"] = [d["n_samples"] < 3 for d in drops]
    ev = rank_events(t, fs)
    row["rank_events"] = ev.tolist()
    lv = tree_levels(m)
    # edges whose next singular value crosses inside each drop window [t_lo - w, t_hi + w]
    per_drop = []
    for d in drops:
        lo, hi = d["t_lo"] - d["width"], d["t_hi"] + d["width"]
        per_drop.append(sorted({(int(e), int(k) + 2) for e, k in np.argwhere((ev >= lo) & (ev <= hi))}))
    row["drop_edges"] = per_drop
    first_k2 = np.nanmin(ev[:, 0]) if np.isfinite(ev[:, 0]).any() else np.nan
    row["t_first_rank2"] = float(first_k2)
    if np.isfinite(first_k2):
        tol = max(1.0, 0.05 * first_k2)
        row["first_rank2_edges"] = [int(e) for e in np.flatnonzero(ev[:, 0] <= first_k2 + tol)]
        row["first_rank2_levels"] = [int(lv[e]) for e in row["first_rank2_edges"]]
    row["edge_levels"] = lv.tolist()
    row["edge_sites"] = edge_sites(m)
    for lvl in LOSS_LEVELS:
        row[f"t_loss_{lvl}"] = time_to_loss(t, loss, lvl)
    test = lg["test"]
    row.update(loss_end=float(lg["loss"][-1]), test_loss_end=float(test[-1, 0]),
               test_acc_end=float(test[-1, 1]), train_acc_end=float(test[-1, 2]),
               drift_max=float(np.nanmax(np.concatenate([run["p1_log"]["conservation_drift"],
                                                         lg["conservation_drift"]]))),
               t2=float(t[-1]), n_fine=int(len(lg.get("fine_t", []))))
    if "delta_K" in lg:
        row.update(delta_K_end=float(lg["delta_K"][-1, 0]), kernel_scale_end=float(lg["delta_K"][-1, 1]),
                   kernel_scale_0=float(run["p1_log"]["delta_K"][0, 1]), t2_unit=m.get("t2_unit"))
    return row


def _safe(p):
    try:
        return summarize(p)
    except Exception as exc:  # keep going; report
        return {"name": Path(p).stem, "error": repr(exc)}


def main(runs_dir: str, out_prefix: str, workers: int = 8) -> None:
    paths = sorted(Path(runs_dir).glob("*.npz"))
    with Pool(workers) as pool:
        rows = pool.map(_safe, paths)
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(rows, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    import csv
    flat_keys = [k for k in rows[0] if not isinstance(rows[0][k], list | dict)] if rows else []
    keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, list | dict)},
                  key=lambda k: flat_keys.index(k) if k in flat_keys else 999)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{len(rows)} runs -> {out}.csv/.json; errors: {sum('error' in r for r in rows)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
