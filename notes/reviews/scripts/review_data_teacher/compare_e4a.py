"""Reviewer: step 4(a) plateau law. (1) rerun vs committed per-run t1, t2, dt2 and fits;
(2) robust re-analysis of the rerun: classify each run by the saddle it visits, an event definition
that does not depend on the speed-minimum plateau detection, per-seed slopes (the same seeds are used
at every sigma, so the per-sigma means are correlated and the 6-point regression SE is too small)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ttn.dynamics import Trajectory

repo, copy, raw, out = sys.argv[1:5]
pd.set_option("display.width", 250)
L = []

mine = pd.read_csv(f"{copy}/results/e4a_sigma_law.csv")
theirs = pd.read_csv(f"{repo}/results/e4a_sigma_law.csv")
m = mine.merge(theirs, on=["overlap", "tree", "sigma", "seed"], suffixes=("_mine", "_theirs"))
L.append(f"per-run match: {len(m)} rerun runs matched to committed runs")
for col in ("t1", "t2", "dt2", "plateau_rel"):
    rel = (m[f"{col}_mine"] - m[f"{col}_theirs"]).abs() / m[f"{col}_theirs"].abs()
    L.append(f"  {col}: max rel diff {rel.max():.2e}, median {rel.median():.2e}")
bad = m[((m["dt2_mine"] - m["dt2_theirs"]).abs() / m["dt2_theirs"].abs()) > 1e-2]
if len(bad):
    L.append(
        "  runs with dt2 differing by > 1%:\n"
        + bad[
            [
                "overlap",
                "sigma",
                "seed",
                "dt2_mine",
                "dt2_theirs",
                "plateau_rel_mine",
                "plateau_rel_theirs",
            ]
        ].to_string()
    )
fm = pd.read_csv(f"{copy}/results/e4a_sigma_law_fits.csv").set_index("overlap")
ft = pd.read_csv(f"{repo}/results/e4a_sigma_law_fits.csv").set_index("overlap")
cols = [
    "n_runs",
    "power_exponent_medians",
    "power_medians_se",
    "power_exponent",
    "log_slope_A",
    "log_slope_se",
    "log_rel_rms",
]
L.append("fits (analyze_sigma_law.py): rerun\n" + fm[cols].round(4).to_string())
L.append("fits: committed\n" + ft.loc[fm.index, cols].round(4).to_string())

# (2) robust re-analysis on the rerun's raw trajectories
th = pd.read_csv(f"{repo}/results/e4a_theory_slopes.csv")
rows = []
for p in sorted(Path(raw, "e4a_sigma_dense").glob("*.npz")):
    tr = Trajectory.load(p)
    s = tr.meta["summary"]
    c, sigma, seed = s["overlap"], s["sigma"], s["seed"]
    t, loss = tr.data["t"], tr.data["loss"]
    l0 = loss[0]
    if c == 0.0:
        levels = {"big": 0.5 * 0.49, "small": 0.5 * 1.0}  # odeco: the unlearned component's s^2 / 2
    else:
        tt = th[th.overlap == c].sort_values("saddle_loss")
        levels = {"big": float(tt.saddle_loss.iloc[0]), "small": float(tt.saddle_loss.iloc[1])}
    # time spent within 2% of each saddle level (the plateau, in linear time)
    dt = np.diff(t)
    occ = {}
    for name, lv in levels.items():
        near = np.abs(loss[:-1] - lv) < 0.02 * lv
        occ[name] = float(dt[near].sum())
    visited = max(occ, key=occ.get)
    lv = levels[visited]
    # events independent of the speed minimum: midpoint of the first drop, and L < lv / 2
    ta = t[np.argmax(loss < 0.5 * (l0 + lv))]
    tb = t[np.argmax(loss < 0.5 * lv)]
    # a real plateau: at least 30% of the time between the two drops is spent within 2% of the level
    frac = occ[visited] / max(tb - ta, 1e-12)
    if frac < 0.3:
        visited = "none"
    rows.append(
        dict(
            overlap=c,
            sigma=sigma,
            seed=seed,
            visited=visited,
            occ_big=occ["big"],
            occ_small=occ["small"],
            occ_frac=frac,
            dt_mid=tb - ta,
        )
    )
df = pd.DataFrame(rows)
mm = df.merge(mine, on=["overlap", "sigma", "seed"])
mm.to_csv(Path(out).with_suffix(".runs.csv"), index=False)
L.append("visited saddle per (overlap, seed) [same at every sigma?]:")
for (c, seed), g in mm.groupby(["overlap", "seed"]):
    L.append(
        f"  c={c} seed={seed}: {sorted(set(g.visited))} plateau_rel(agent)={sorted(set(g.plateau_rel.round(3)))}"
        f" occ_big(sigma=0.03..0.3)={list(g.sort_values('sigma').occ_big.round(1))}"
    )


def seed_slopes(g, x_of, ycol):
    out = []
    for seed, h in g.groupby("seed"):
        h = h.dropna(subset=[ycol])
        if len(h) >= 4:
            out.append(np.polyfit(x_of(h.sigma.values), h[ycol].values, 1)[0])
    return np.array(out)


for c in sorted(mm.overlap.unique()):
    g = mm[mm.overlap == c]
    for subset, gg in [
        ("all seeds", g),
        ("big-saddle seeds", g[g.groupby("seed").visited.transform(lambda v: (v == "big").all())]),
        (
            "small-saddle seeds",
            g[g.groupby("seed").visited.transform(lambda v: (v == "small").all())],
        ),
    ]:
        if gg.seed.nunique() < 1:
            continue
        if c == 0.0:  # power law: per-seed exponent of log dt vs log sigma, and per-sigma medians
            for ycol in ("dt2", "dt_mid"):
                ok = gg[gg[ycol] > 0].assign(ly=lambda d: np.log(d[ycol]))
                sl = -seed_slopes(ok, np.log, "ly")
                med = ok.groupby("sigma")[ycol].median()
                pm = -np.polyfit(np.log(med.index.values), np.log(med.values), 1)[0]
                L.append(
                    f"c=0 {subset:18s} {ycol:6s}: per-seed exponent {sl.mean():.3f} +- "
                    f"{sl.std(ddof=1) / np.sqrt(len(sl)) if len(sl) > 1 else np.nan:.3f} (n={len(sl)}); "
                    f"medians exponent {pm:.3f}"
                )
        else:  # log law
            for ycol in ("dt2", "dt_mid"):
                sl = seed_slopes(gg, lambda s: -np.log(s), ycol)
                mean = gg.groupby("sigma")[ycol].mean()
                med = gg.groupby("sigma")[ycol].median()
                a_mean = np.polyfit(-np.log(mean.index.values), mean.values, 1)[0]
                a_med = np.polyfit(-np.log(med.index.values), med.values, 1)[0]
                L.append(
                    f"c={c} {subset:18s} {ycol:6s}: per-seed A {sl.mean():.2f} +- "
                    f"{sl.std(ddof=1) / np.sqrt(len(sl)) if len(sl) > 1 else np.nan:.2f} (n={len(sl)}, sd "
                    f"{sl.std(ddof=1) if len(sl) > 1 else np.nan:.2f}); A(means) {a_mean:.2f}; A(medians) {a_med:.2f}"
                )
txt = "\n".join(L)
print(txt)
open(out, "w").write(txt + "\n")
