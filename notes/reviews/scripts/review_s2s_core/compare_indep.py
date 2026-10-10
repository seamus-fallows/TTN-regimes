"""Compare the reviewer's independent implementation (indep.py outputs) with the committed CSVs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("C:/Users/seamu/Iliad-ML/TTN_regimes")
IND = REPO / "data/review_s2s_core/indep"
RES = REPO / "experiments/s2s_core/results"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

fr = pd.read_csv(RES / "first_runs.csv")
fr = fr[(fr.chi == 3) & (fr.teacher == "separated") & (fr.init == "fanin")]
runs = pd.read_csv(RES / "main_runs.csv")

print("== correlation-flow blow-up time T (sphere method) vs committed T (first_runs.csv)")
for tree in ("binary4", "mps4"):
    p = IND / f"corr_{tree}.json"
    if not p.exists():
        continue
    rows = json.loads(p.read_text())["rows"]
    df = pd.DataFrame(rows)
    c = fr[fr.tree == tree].drop_duplicates("seed").set_index("seed")
    df["T_committed"] = df.seed.map(c["T"])
    df["KH_committed"] = df.seed.map(c["t_KH"])
    df["r1_committed"] = df.seed.map(c["r1_corrflow"])
    df["T_rel_diff"] = (df["T_committed"] - df["T"]) / df["T"]
    df["KH_rel_diff"] = (df["KH_committed"] - df["t_KH"]) / df["t_KH"]
    print(tree)
    print(
        df[
            [
                "seed",
                "T",
                "T_committed",
                "T_rel_diff",
                "r1",
                "r1_committed",
                "t_KH",
                "KH_rel_diff",
                "T_over_KH",
                "g_end_over_gmax",
            ]
        ].to_string(float_format=lambda x: f"{x:.6g}")
    )
    print(
        f"  T/t_KH: median {df.T_over_KH.median():.3g}, range {df.T_over_KH.min():.3g}-{df.T_over_KH.max():.3g}; "
        f"max |T rel diff| {df.T_rel_diff.abs().max():.2e}"
    )

print("\n== first escape from own GF vs committed t1 and prediction")
for name in ("first_binary4_seed0", "first_mps4_seed2"):
    p = IND / f"{name}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    tree = d["tree"]
    for r in d["rows"]:
        c = fr[(fr.tree == tree) & (fr.seed == d["seed"]) & np.isclose(fr.sigma, r["sigma"])]
        t1c = c.t1.iloc[0] if len(c) else np.nan
        print(
            f"{tree} seed {d['seed']} sigma {r['sigma']}: t1 own {r.get('t1', np.nan):.10g}, committed {t1c:.10g}, "
            f"rel diff {(t1c - r.get('t1', np.nan)) / r.get('t1', np.nan):.2e}; r1 own {r.get('r1')} committed "
            f"{c.r1.iloc[0] if len(c) else None}; own ratio t1/(sigma^-(K-2) T + c_K) - 1 = {r.get('ratio_with_offset', np.nan) - 1:.2e} "
            f"(committed {c.ratio_with_offset.iloc[0] - 1 if len(c) else np.nan:.2e})"
        )
    print("  own c_K by quadrature:", d.get("c_K_quad"))

print("\n== full S2S runs, own implementation vs committed main_runs.csv")
for p in sorted(IND.glob("full_*.json")):
    d = json.loads(p.read_text())
    c = runs[
        (runs.sweep.isin(["main", "balanced"]))
        & (runs.tree == d["tree"])
        & (runs.chi == d["chi"])
        & (runs.teacher == "separated")
        & (runs.init == d["init"])
        & np.isclose(runs.sigma, d["sigma"])
        & (runs.seed == d["seed"])
    ]
    c = c.iloc[0] if len(c) else None
    ev = d["events"]
    order = d["order"]
    ts = [ev[r] for r in order]
    print(
        f"\n{p.stem}: stop {d['stop']}, order {[o + 1 for o in order]} (committed {c['order'] if c is not None else None}, "
        f"class {c['class'] if c is not None else None}), s2s {d['s2s']}, sep ratios {np.round(d['sep_ratios'], 1).tolist()}"
    )
    if c is not None:
        for i, key in enumerate(("t1", "t2", "t3")):
            if i < len(ts) and np.isfinite(c[key]):
                print(
                    f"  {key}: own {ts[i]:.10g} committed {c[key]:.10g} rel diff {(c[key] - ts[i]) / ts[i]:.2e}"
                )
    for i, pl in enumerate(d["plateaus"]):
        print(
            f"  plateau {i + 1}: loss/level - 1 = {pl['loss'] / pl['level'] - 1:.2e}, top-|S| sv dev {pl['spectrum_dev']:.2e}, "
            f"next sv/s_min {pl['next_sv_over_smin']:.2e}, raw lambda_min {pl['lam_min_raw']:.3e}"
            + (
                f" (committed proj {c[f'plateau{i + 1}_lam_min']:.3e}, raw {c[f'plateau{i + 1}_lam_min_raw']:.3e})"
                if c is not None and i < 2
                else ""
            )
        )
    for rs in d["rank_steps"]:
        print(
            f"  event comp {rs['component'] + 1}: count above f*s_r per cut before {rs['before']} after {rs['after']} "
            f"(f={rs['f']}), spread of cut step times / width = {rs['spread_over_width']:.3g} (grid {rs['grid_dt_over_width']:.3g})"
        )
    if d["never_learned"]:
        print(
            f"  never learned {[r + 1 for r in d['never_learned']]}: min end slice / sigma = {d['min_never_slice_over_sigma_end']:.2e}"
            f" (committed {c['min_dormant_slice_over_sigma'] if c is not None else None}); decay (t, min slice):"
        )
        dec = np.array(d["never_slice_decay"])
        if len(dec):
            print("   ", "; ".join(f"{t:.3g}: {s:.2e}" for t, s in dec))
            ok = dec[:, 1] > 1e-13
            if ok.sum() >= 3:
                rate = -np.polyfit(dec[ok, 0], np.log(dec[ok, 1]), 1)[0]
                print(
                    f"   fitted decay rate {rate:.3e}, rate / sigma^2 = {rate / d['sigma'] ** 2:.3g}"
                )
