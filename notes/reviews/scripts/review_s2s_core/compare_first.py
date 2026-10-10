"""Compare the reviewer's first-escape rerun (analyze_first.py on data/review_s2s_core/raw/first) and the
lo_sweep rerun with the committed CSVs, row by row."""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("C:/Users/seamu/Iliad-ML/TTN_regimes")
ANA = REPO / "data/review_s2s_core/analysis"
RES = REPO / "experiments/s2s_core/results"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

if (ANA / "first_runs.csv").exists():
    mine = pd.read_csv(ANA / "first_runs.csv")
    com = pd.read_csv(RES / "first_runs.csv")
    keys = ["tree", "chi", "teacher", "init", "seed", "sigma"]
    m = mine.merge(com, on=keys, suffixes=("", "_c"), how="left")
    m["t1_rel"] = (m.t1 - m.t1_c) / m.t1_c
    m["T_rel"] = (m["T"] - m["T_c"]) / m["T_c"]
    m["KH_rel"] = (m.t_KH - m.t_KH_c) / m.t_KH_c
    cols = [
        "tree",
        "seed",
        "sigma",
        "r1",
        "r1_c",
        "t1",
        "t1_rel",
        "T",
        "T_rel",
        "KH_rel",
        "ratio_with_offset",
        "ratio_with_offset_c",
    ]
    print(m[cols].to_string(float_format=lambda x: f"{x:.6g}"))
    print(
        "max |t1 rel| =",
        m.t1_rel.abs().max(),
        " max |T rel| =",
        m.T_rel.abs().max(),
        " r1 agree:",
        (m.r1 == m.r1_c).sum(),
        "/",
        len(m),
    )
    small = m[m.sigma == m.groupby(["tree", "seed"]).sigma.transform("min")]
    print(
        "smallest sigma: |ratio_with_offset - 1| max =", (small.ratio_with_offset - 1).abs().max()
    )
    print(
        "T / t_KH per seed:",
        (m.drop_duplicates(["tree", "seed"]).assign(r=lambda d: d["T"] / d.t_KH))[
            ["tree", "seed", "r"]
        ].to_string(),
    )
    m.to_csv(ANA / "compare_first.csv", index=False)
    fits = pd.read_csv(ANA / "first_fits.csv")
    fc = pd.read_csv(RES / "first_fits.csv")
    f = fits.merge(fc, on=["tree", "K", "chi", "teacher", "init", "seed"], suffixes=("", "_c"))
    print(
        f[
            [
                "tree",
                "seed",
                "exponent",
                "exponent_c",
                "exponent_offset_removed",
                "exponent_offset_removed_c",
            ]
        ].to_string()
    )

for name, ref in (("lo_K_sub", "lo_K"), ("lo_K_sub", "lo_chi")):
    p = ANA / f"{name}.csv"
    if not p.exists():
        continue
    mine = pd.read_csv(p)
    com = pd.read_csv(RES / f"{ref}.csv")
    keys = ["tree", "d", "chi", "weights", "init", "seed"]
    mine["weights"] = mine["weights"].astype(str)
    com["weights"] = com["weights"].astype(str)
    m = mine.merge(com, on=keys, suffixes=("", "_c"), how="inner")
    if not len(m):
        continue
    m["T_rel"] = (m["T"] - m["T_c"]) / m["T_c"]
    print(
        f"\n{name} vs {ref}: {len(m)} matched rows; max |T rel| = {m.T_rel.abs().max():.2e}; r1 agree {(m.r1 == m.r1_c).sum()}"
    )
    g = (
        m.assign(ratio=m["T"] / m.t_KH)
        .groupby(["tree", "chi"])
        .agg(
            n=("seed", "size"),
            T_med=("T", "median"),
            T_over_KH_med=("ratio", "median"),
            T_over_KH_min=("ratio", "min"),
            T_over_KH_max=("ratio", "max"),
            P3=("P3", "first"),
        )
    )
    print(g.to_string(float_format=lambda x: f"{x:.4g}"))
