"""Review issue 5: are steep exponents a power shift or a log correction from small route ends?

If the route ends enter the plateau at eps_end ~ sigma^alpha (alpha > 1) while the interior is ~ sigma, the
conservation laws (eps_u^2 - eps_v^2 constant along the route) let small ends catch up, and the escape time
becomes T ~ sigma^-(k-2) [ (alpha - 1) ln(1/sigma) + 1/(k-2) ] / lambda (reviewer's derivation), a log
correction rather than a power shift. Per seed we fit, by least squares on log T:
  P:  log T = c + p ln(1/sigma)                                  (free power, 2 parameters)
  Lf: T = A sigma^-(k-2) [ a ln(1/sigma) + 1/(k-2) ], a >= 0       (log correction, 2 parameters)
  La: same with a = alpha - 1 from the measured route-end exponent (1 parameter)
and report the median RMS residuals and the fitted a.

usage: uv run python experiments/locality/logcorr.py <prefix> [...] --amps results/route_amps_summary.csv --out results/logcorr.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from analysis import mean_se, plateau_table  # noqa: E402


def fit_logcorr(x, T, k, a_fixed=None):
    y = np.log(T)

    def model(c):
        a = a_fixed if a_fixed is not None else c[1]
        return c[0] + (k - 2) * x + np.log(np.maximum(a * x + 1 / (k - 2), 1e-300))

    c0 = [np.mean(y - (k - 2) * x)] + ([] if a_fixed is not None else [0.1])
    lb = [-np.inf] + ([] if a_fixed is not None else [0.0])
    fit = least_squares(lambda c: model(c) - y, c0, bounds=(lb, [np.inf] * len(c0)))
    return float(np.sqrt(np.mean(fit.fun ** 2))), (fit.x[1] if a_fixed is None else a_fixed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--amps", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    amps = pd.read_csv(args.amps, dtype={"S": str})
    df = plateau_table(args.inputs)
    df = df[(df["order"] == "0>1") & (~df["rotation"]) & (df["k_rank"] >= 3)]
    rows = []
    for _, a in amps.iterrows():
        S = a["S"].zfill(1)
        g = df[(df["tree"] == a["tree"]) & (df["case"] == "S=" + S) & (df["start"] == a["start"])]
        if g.empty:
            continue
        k = int(g["k_rank"].iloc[0])
        alpha = float(re.findall(r"[-+]?\d*\.\d+(?:e[-+]?\d+)?", str(a["end_vs_sigma"]))[0])
        rp, rlf, rla, af, p = [], [], [], [], []
        for _, gs in g.groupby("seed"):
            gs = gs.sort_values("sigma")
            if gs["sigma"].nunique() < 4:
                continue
            x, T = np.log(1 / gs["sigma"].values), gs["gap"].values
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, np.log(T), rcond=None)
            rp.append(float(np.sqrt(np.mean((np.log(T) - A @ coef) ** 2))))
            p.append(coef[0])
            r, aa = fit_logcorr(x, T, k)
            rlf.append(r)
            af.append(aa)
            rla.append(fit_logcorr(x, T, k, a_fixed=max(alpha - 1, 0.0))[0])
        if not p:
            continue
        rows.append(dict(tree=a["tree"], S=S, start=a["start"], k=k, alpha_end=alpha, seeds=len(p),
                         p_power=mean_se(p)[0], rms_power=np.median(rp), rms_logcorr_free=np.median(rlf),
                         a_fit=np.median(af), alpha_minus_1=alpha - 1, rms_logcorr_alpha=np.median(rla)))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
