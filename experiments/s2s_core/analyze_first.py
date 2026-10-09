"""Analysis of the first-escape sweep (first_escape.py): per-seed exponents and the per-seed
correlation-flow prediction t_1 = sigma^-(K-2) T(theta_hat) + c_K(s_r1).

    uv run python experiments/s2s_core/analyze_first.py [--raw /root/outputs/s2s_core/first] [--out results]

Writes <out>/first_runs.csv (one row per seed and sigma), <out>/first_fits.csv (one row per seed:
fitted exponent of t_1 vs sigma, with and without the predicted O(1) offset) and
<out>/first_summary.csv (per tree, chi, teacher, init: exponent mean +- SE over seeds vs K - 2,
ratio t_1 / prediction, T / t_KH, first-component agreement).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def load(raw: Path):
    rows = []
    for f in sorted(raw.glob("*.json")):
        s = json.loads(f.read_text())
        K, w = s["K"], np.asarray(s["weights"])
        T = s["T"]
        race = int(np.argmin(s["T_single"])) if s.get("T_single") else -1
        for r in s["runs"]:
            if "t1" not in r:
                rows.append({"tree": f"{s['tree']}{s['N']}", "K": K, "chi": s["chi"], "teacher": s["teacher"],
                             "init": s["init"], "seed": s["seed"], "sigma": r["sigma"], "resolved": False})
                continue
            r1 = r["r1"]
            offset = s["c_K"][r1]
            pred = r["sigma"] ** (-(K - 2)) * T + offset
            rows.append({
                "tree": f"{s['tree']}{s['N']}", "K": K, "chi": s["chi"], "teacher": s["teacher"],
                "init": s["init"], "seed": s["seed"], "sigma": r["sigma"], "resolved": True,
                "t1": r["t1"], "w1": r["w1"], "r1": r1, "r1_corrflow": s["r1_pred"], "r1_race": race,
                "T": T, "t_KH": s["kh"], "P3": s["p3"][0],
                "ratio_leading": r["t1"] * r["sigma"] ** (K - 2) / T,
                "ratio_with_offset": r["t1"] / pred,
                "offset_term": offset * r["sigma"] ** (K - 2) / T,
                "relearned": r["t1_first_crossing"] < r["t1"] * (1 - 1e-6),
            })
    return pd.DataFrame(rows)


def per_seed_fits(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    keys = ["tree", "K", "chi", "teacher", "init", "seed"]
    for key, g in df[df.resolved].groupby(keys):
        if g.sigma.nunique() < 3:
            continue
        x = np.log(g.sigma.values)
        slope = np.polyfit(x, np.log(g.t1.values), 1)[0]
        corrected = g.t1.values - (g.t1.values / g.ratio_with_offset.values - g.sigma.values ** (-(g.K.values - 2)) * g["T"].values)
        slope_c = np.polyfit(x, np.log(corrected), 1)[0]
        out.append({**dict(zip(keys, key)), "n_sigma": len(g), "exponent": -slope,
                    "exponent_offset_removed": -slope_c,
                    "sigma_min": g.sigma.min(), "sigma_max": g.sigma.max()})
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    keys = ["tree", "K", "chi", "teacher", "init"]
    res = df[df.resolved]
    small = res[res.sigma == res.groupby(keys).sigma.transform("min")]
    a = fits.groupby(keys).agg(n_seeds=("seed", "size"), exponent_mean=("exponent", "mean"),
                               exponent_se=("exponent", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                               exponent_offset_removed_mean=("exponent_offset_removed", "mean"),
                               exponent_offset_removed_se=("exponent_offset_removed",
                                                           lambda x: x.std(ddof=1) / np.sqrt(len(x))))
    b = small.groupby(keys).agg(
        sigma_min=("sigma", "first"),
        max_abs_ratio_leading_minus1=("ratio_leading", lambda x: np.max(np.abs(x - 1))),
        max_abs_ratio_offset_minus1=("ratio_with_offset", lambda x: np.max(np.abs(x - 1))),
        T_median=("T", "median"), T_over_KH_median=("T", lambda x: np.nan),
        r1_agree_corrflow=("r1", lambda x: np.nan), n_unresolved=("seed", lambda x: 0))
    for key, g in small.groupby(keys):
        b.loc[key, "T_over_KH_median"] = float(np.median(g["T"] / g["t_KH"]))
        b.loc[key, "r1_agree_corrflow"] = float(np.mean(g.r1 == g.r1_corrflow))
    for key, g in df.groupby(keys):
        if key in b.index:
            b.loc[key, "n_unresolved"] = int((~g.resolved).sum())
    s = a.join(b, how="outer").reset_index()
    s["K_minus_2"] = s.K - 2
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/outputs/s2s_core/first")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.raw))
    fits = per_seed_fits(df)
    summ = summarize(df, fits)
    df.to_csv(out / "first_runs.csv", index=False)
    fits.to_csv(out / "first_fits.csv", index=False)
    summ.to_csv(out / "first_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
