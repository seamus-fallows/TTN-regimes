"""Bifurcation diagram of GD on one balanced aligned mode (the reduced model of an odeco component).

Per-core amplitude x = a / s^{1/K}, GD step x -> x - h x^{K-1} (x^K - 1) with h = eta s^{2-2/K};
the fixed point x = 1 is stable iff h < h_c = 2/K. For each h/h_c we iterate from x0 and record the
attractor (last 2000 iterates): its distinct values, mean loss / s^2 = mean (x^K - 1)^2 / 2, and
whether the orbit diverged or fell back to the plateau near x = 0 ("re-entry").

Run: uv run python experiments/optim_samples/map1d.py
Writes results/map1d_attractor.csv (per h/h_c and K) and results/map1d_points.csv (scatter of
attractor points for the bifurcation figure).
"""


import numpy as np
import pandas as pd

from common import RESULTS as OUT  # noqa: E402


def attractor(k, ratio, x0=0.3, n_iter=60000, keep=2000):
    h = ratio * 2.0 / k
    x = x0
    pts = np.empty(keep)
    for i in range(n_iter):
        x = x - h * x ** (k - 1) * (x**k - 1)
        if not np.isfinite(x) or abs(x) > 1e6:
            return None
        if i >= n_iter - keep:
            pts[i - (n_iter - keep)] = x
    return pts


def main():
    rows, scatter = [], []
    ratios = np.round(np.arange(0.80, 3.2001, 0.005), 4)
    for k in (3, 4, 7, 8):
        for r in ratios:
            pts = attractor(k, r)
            if pts is None:
                rows.append(dict(K=k, ratio=r, state="diverged", n_distinct=0, loss=np.nan,
                                 xmin=np.nan, xmax=np.nan))
                continue
            loss = 0.5 * np.mean((pts**k - 1) ** 2)
            distinct = len(np.unique(np.round(pts, 7)))
            if np.max(np.abs(pts)) < 0.2:
                state = "reentry"
            elif distinct == 1:
                state = "fixed"
            elif distinct <= 64:
                state = f"period{distinct}"
            else:
                state = "chaotic"
            rows.append(dict(K=k, ratio=r, state=state, n_distinct=distinct, loss=loss,
                             xmin=pts.min(), xmax=pts.max()))
            for x in np.unique(np.round(pts[-300:], 6)):
                scatter.append(dict(K=k, ratio=r, x=x))
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "map1d_attractor.csv", index=False)
    pd.DataFrame(scatter).to_csv(OUT / "map1d_points.csv", index=False)
    # Thresholds: first ratio at which each state appears.
    for k, g in df.groupby("K"):
        first = {s: g.ratio[g.state == s].min() for s in g.state.unique()}
        print(k, dict(sorted(first.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
