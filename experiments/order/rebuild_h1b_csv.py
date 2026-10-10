"""Rebuild results/<name>.csv of an h1b.py run from its saved trajectories, for runs whose pool did
not finish (one slow job kept the CSV from being written).

Usage: uv run python experiments/order/rebuild_h1b_csv.py h1b_saddle_b8 [k_cores]
t1 is read from the stored trajectory (fine record window around the events; ~0.1% resolution),
lambda_min from the stored gauge-projected spectrum at the minimum-speed point (raw values are not
stored in the npz: lam_min_raw = NaN). Trajectories that do not exist are listed as missing.
"""

import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/order")
from common import RAW, RES  # noqa: E402


def main():
    name = sys.argv[1]
    rows = []
    for f in sorted((RAW / name).glob("*.npz")):
        rho, sigma, seed = re.match(r"rho([\d.]+)_s([\de.-]+)_seed(\d+)\.npz", f.name).groups()
        z = np.load(f)
        tr, eigs = z["traj"], z["eigs"]
        t, a1, a2 = tr[:, 0], tr[:, 2], tr[:, 3]
        up = np.nonzero((a1[:-1] < 0.5) & (a1[1:] >= 0.5))[0]
        t1 = (t[up[0]] + (0.5 - a1[up[0]]) * (t[up[0] + 1] - t[up[0]]) / (a1[up[0] + 1] - a1[up[0]])
              if len(up) else np.inf)
        rows.append(dict(rho=float(rho), sigma=float(sigma), seed=int(seed), t1=t1, t2=0.0,
                         wrong=True, a2_min_after_t1=a2[t >= t1].min() if np.isfinite(t1) else np.nan,
                         lam_min=eigs[0] if len(eigs) else np.nan, lam_min_raw=np.nan,
                         n_neg=int((eigs < -1e-3 * eigs[-1]).sum()) if len(eigs) else np.nan,
                         lam_pred=0.0, t_end=t[-1], rebuilt=True))
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"{name}.csv", index=False, float_format="%.15g")
    print(df.groupby("sigma").size().to_string())


if __name__ == "__main__":
    main()
