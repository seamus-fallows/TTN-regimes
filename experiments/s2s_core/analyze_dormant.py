"""Dormant-component suppression (lead's question): how far do the amplitudes |<W, U_r>| of
components that are not yet learned fall during earlier escapes and plateaus, and how fast do the
dormant end-core slices decay on a stalled plateau?

    uv run python experiments/s2s_core/analyze_dormant.py [--raw DIR ...] [--out results] [--tag dormant]

Per run and per dormant component r, for each phase (plateau i = between events i and i+1):
amplitude at the start of the phase, its minimum and its end value (coarse log grid). For floor-class
runs (some dormant amplitude < 1e-14 after the first escape), also the exponential decay rate of every
end-core slice of the stuck component on the last plateau, divided by sigma^2 (the trap of
notes/theory_saddles.md section 8 predicts a rate s_q^(2(K-2)/K) |Q_nu nu + |y_route|^2| ~ sigma^2).
Writes <out>/<tag>_phases.csv and <out>/<tag>_decay.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FLOOR = 1e-14


def analyze(raw: Path):
    phases, decays = [], []
    for f in sorted(raw.glob("*.json")):
        s = json.loads(f.read_text())
        d = np.load(f.with_suffix(".npz"))
        t, a, sn = d["t"], d["overlaps"], d["slice_norms"]
        w = np.asarray(s["weights"])
        amps = np.abs(a) * w
        R = len(w)
        order = s["order"]
        ev = [s["events"][r] for r in order]
        bounds = [0.0] + ev + [t[-1]]
        base = {"tree": f"{s['tree']}{s['N']}", "K": s["K"], "chi": s["chi"], "teacher": s["teacher"],
                "init": s["init"], "sigma": s["sigma"], "seed": s["seed"], "n_learned": len(order)}
        for i in range(len(bounds) - 1):
            inside = (t >= bounds[i]) & (t <= bounds[i + 1])
            if inside.sum() < 2:
                continue
            for r in range(R):
                if r in order[: i + 1]:
                    continue  # learned before this phase, or the one escaping at its end
                seg = amps[inside, r]
                phases.append({**base, "phase": i, "component": r, "amp_start": seg[0],
                               "amp_min": seg.min(), "amp_end": seg[-1],
                               "amp_start_over_sigmaK": seg[0] / s["sigma"] ** s["K"]})
        stuck = [r for r in range(R) if r not in order]
        after = [v for row in s["dormant_min_between_events"][1:] for v in row.values()]
        if stuck and after and min(after) < FLOOR and ev:
            r = stuck[0]
            last = (t > 1.5 * ev[-1]) & (amps[:, r] > 1e-28)
            for core in range(sn.shape[1]):
                y = sn[last, core, r]
                ok = np.isfinite(y) & (y > 1e-13)
                if ok.sum() >= 4:
                    rate = -np.polyfit(t[last][ok], np.log(y[ok]), 1)[0]
                    decays.append({**base, "component": r, "core": core, "rate": rate,
                                   "rate_over_sigma2": rate / s["sigma"] ** 2,
                                   "slice_start": y[ok][0], "slice_end": y[ok][-1]})
    return pd.DataFrame(phases), pd.DataFrame(decays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["/root/outputs/s2s_core/main"])
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--tag", default="dormant")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ph, de = zip(*(analyze(Path(r)) for r in args.raw))
    ph, de = pd.concat(ph), pd.concat(de)
    ph.to_csv(out / f"{args.tag}_phases.csv", index=False)
    de.to_csv(out / f"{args.tag}_decay.csv", index=False)
    pd.set_option("display.width", 250)
    if len(de):
        print(de.groupby(["tree", "chi", "teacher", "init", "seed", "core"]).rate_over_sigma2.agg(["min", "max", "size"]).to_string())


if __name__ == "__main__":
    main()
