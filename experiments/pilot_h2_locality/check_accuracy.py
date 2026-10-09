"""Compare crossing times of the accuracy-check runs against the reference integration.

    uv run python experiments/pilot_h2_locality/run_sweep.py experiments/pilot_h2_locality/configs/accuracy.yaml
    uv run python experiments/pilot_h2_locality/check_accuracy.py

Reference = LSODA with rtol 1e-10 (the sweep setting). For each case, every variant (tighter rtol,
tighter atol, Radau/BDF, explicit DOP853/RK45, capped step) is compared on t1 and t2 (first crossing
of a_r = 0.5). Writes results/accuracy.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CASE = ["tree", "n", "kind", "sigma", "seed", "t_max"]


def variant_name(r: dict) -> str:
    parts = [r["method"], f"rtol={r['rtol']:g}"]
    if r["atol_factor"] != 1e-3:
        parts.append(f"atol x{r['atol_factor'] / 1e-3:g}")
    if np.isfinite(r["max_step"]):
        parts.append(f"max_step={r['max_step']:g}")
    if not r["collapse_stop"]:
        parts.append("no collapse stop")
    return " ".join(parts)


def main():
    df = pd.read_csv(HERE.parents[1] / "data" / "pilot_h2_locality" / "accuracy_runs.csv")
    df["t_max"] = df["t_max"].fillna(-1.0)
    df["variant"] = [variant_name(r) for r in df.to_dict("records")]
    is_ref = (
        (df["method"] == "LSODA")
        & (df["rtol"] == 1e-10)
        & (df["atol_factor"] == 1e-3)
        & (~np.isfinite(df["max_step"]))
        & df["collapse_stop"].astype(bool)
    )
    ref = df[is_ref].drop_duplicates(subset=CASE).set_index(CASE)
    rows = []
    for r in df[~is_ref].to_dict("records"):
        key = tuple(r[c] for c in CASE)
        if key not in ref.index:
            continue
        rr = ref.loc[key]
        row = dict(case=rr["tag"], variant=r["variant"], status=r["status"], ref_status=rr["status"],
                   wall_s=r["wall_s"], ref_wall_s=rr["wall_s"], nfev=r["nfev"], ref_nfev=rr["nfev"],
                   q_drift=r["q_drift"])
        for col in ("t_a1_0.5", "t_a2_0.5"):
            a, b = r[col], rr[col]
            row[f"{col}_ref"], row[col] = b, a
            row[f"rel_diff_{col}"] = abs(a - b) / b if np.isfinite(a) and np.isfinite(b) else np.nan
        row["s2_end_max"], row["s2_end_max_ref"] = r["s2_end_max"], rr["s2_end_max"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "accuracy.csv", index=False, float_format="%.12g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    show = out[
        [
            "case",
            "variant",
            "status",
            "ref_status",
            "rel_diff_t_a1_0.5",
            "rel_diff_t_a2_0.5",
            "wall_s",
            "ref_wall_s",
            "s2_end_max",
            "s2_end_max_ref",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    both = out[["rel_diff_t_a1_0.5", "rel_diff_t_a2_0.5"]].to_numpy()
    print(f"\nmax relative difference in t1, t2 over all variants: {np.nanmax(both):.2e}")
    print(
        f"status mismatches: {(out['status'] != out['ref_status']).sum()} "
        f"(timeouts of slow explicit runs are expected)"
    )


if __name__ == "__main__":
    main()
