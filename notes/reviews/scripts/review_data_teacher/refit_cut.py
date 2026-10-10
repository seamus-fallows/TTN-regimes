"""Reviewer: analyze_sigma_law.events() with the plateau window cut raised from 1e-2 L(0) to 0.1 L(0)
(plateau levels here are >= 0.2 L(0)); runs without a plateau then drop out instead of entering as tail
times. Same fit() as the agent's script (per-sigma means)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])
import analyze_sigma_law as asl
from ttn.dynamics import Trajectory

def events_cut(path, cut):
    tr = Trajectory.load(path)
    t, loss = tr.data["t"], tr.data["loss"]
    k1 = int(np.argmax(loss < 0.9 * loss[0]))
    win = np.nonzero((np.arange(len(t)) > k1) & (loss > cut * loss[0]))[0]
    speed = np.abs(np.gradient(loss, t))
    if len(win) < 3:
        return t[k1], np.nan, np.nan
    kp = win[int(np.argmin(speed[win]))]
    if kp == win[-1]:          # minimum at the window edge: no interior plateau
        return t[k1], np.nan, np.nan
    after = np.nonzero((np.arange(len(t)) > kp) & (loss < 0.5 * loss[kp]))[0]
    return t[k1], (t[after[0]] if len(after) else np.nan), loss[kp] / loss[0]

rows = []
for p in sorted(Path(sys.argv[2], "e4a_sigma_dense").glob("*.npz")):
    s = Trajectory.load(p).meta["summary"]
    for cut in (1e-2, 0.1):
        t1, t2, pr = events_cut(p, cut)
        rows.append(dict(overlap=s["overlap"], sigma=s["sigma"], seed=s["seed"], cut=cut, dt2=t2 - t1, pl=pr))
df = pd.DataFrame(rows)
for (c, cut), g in df.groupby(["overlap", "cut"]):
    for label, gg in (("all saddles", g), ("s=1 saddle only", g[(g.pl > 0.2) & (g.pl < 0.5)])):
        gg = gg.dropna(subset=["dt2"])
        gg = gg[gg.dt2 > 0]
        m = gg.groupby("sigma").dt2.mean()
        if c == 0.0:
            med = gg.groupby("sigma").dt2.median()
            (q, _), (qse, _), _ = asl.fit(np.log(med.index.values), np.log(med.values))
            print(f"c={c} cut={cut} {label:16s}: n={len(gg)}, medians exponent {-q:.3f} +- {qse:.3f}")
        else:
            (a, _), (ase, _), rms = asl.fit(-np.log(m.index.values), m.values)
            print(f"c={c} cut={cut} {label:16s}: n={len(gg)}, A(means) {a:.2f} +- {ase:.2f} (rel rms {rms / m.mean():.3f})")
