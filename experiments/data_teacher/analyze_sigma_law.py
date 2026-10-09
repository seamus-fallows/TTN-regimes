"""Plateau law of the second escape vs init scale (dense runs of configs/e4a_sigma.yaml and
configs/e2_sigma.yaml; library integrator, LSODA rtol 1e-9).

t_1 = first time L < 0.9 L(0) (first drop); t_2 = first time the loss falls below half
of the plateau level that follows (see events()). (A fixed small threshold on s_2, as suggested in
theory_teachers.md 3.3, fails at sigma = 0.3, where the init's own singular values are ~1e-2.)
Dt_2 = t_2 - t_1 per run. For every case two fits over sigma (6 values, 1 decade, 10 seeds):
power law Dt_2 = C sigma^-p (log-log least squares on per-sigma means of log Dt_2) and logarithmic
Dt_2 = A log(1/sigma) + B (least squares on per-sigma means); the residual ratio says which fits.
Predictions: odeco (c = 0, theta = 0) p = K - 2 = 2 on MPS-4; c > 0 or theta > 0: logarithmic.
Usage: uv run python experiments/data_teacher/analyze_sigma_law.py e4a_sigma overlap
Writes results/<name>_law.csv, results/<name>_law_fits.csv, results/<name>_law.png.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_ROOT  # noqa: E402
from run_generic import expand  # noqa: E402

from ttn.dynamics import Trajectory  # noqa: E402

HERE = Path(__file__).parent


def events(path):
    """t_1: first time L < 0.9 L(0) (the first drop is a steep blow-up, so this is within O(1) of
    its midpoint; the dense grid starts just before it). Plateau: the point of minimum |dL/dt| after t_1 among points with L > 1e-2 L(0)
    (all plateau levels here are >= 0.2 L(0)). t_2: first time after it that L < half the plateau
    level (a level crossing, robust to the coarser grid late in the run)."""
    tr = Trajectory.load(path)
    t, loss = tr.data["t"], tr.data["loss"]
    k1 = int(np.argmax(loss < 0.9 * loss[0]))
    win = np.nonzero((np.arange(len(t)) > k1) & (loss > 1e-2 * loss[0]))[0]
    if len(win) < 3:
        return t[k1], t[k1], np.nan, tr.meta["summary"]["max_drift"]
    speed = np.abs(np.gradient(loss, t))
    kp = win[int(np.argmin(speed[win]))]
    after = np.nonzero((np.arange(len(t)) > kp) & (loss < 0.5 * loss[kp]))[0]
    t2 = t[after[0]] if len(after) else np.nan
    return t[k1], t2, loss[kp] / loss[0], tr.meta["summary"]["max_drift"]


def fit(x, y):
    a = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    res = y - a @ coef
    dof = max(len(x) - 2, 1)
    cov = (res @ res / dof) * np.linalg.inv(a.T @ a)
    return coef, np.sqrt(np.diag(cov)), np.sqrt(np.mean(res**2))


def main():
    name, key = sys.argv[1], sys.argv[2]
    cfg = yaml.safe_load(open(HERE / "configs" / f"{name}.yaml"))
    rows = []
    for case in expand(cfg):
        for tree in cfg["trees"]:
            for sigma in cfg["sigmas"][tree]:
                for seed in range(*cfg["seeds"]):
                    p = OUT_ROOT / f"{name}_dense" / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
                    if p.exists():
                        t1, t2, s2_t1, drift = events(p)
                        rows.append({key: case[key], "tree": tree, "sigma": sigma, "seed": seed,
                                     "t1": t1, "t2": t2, "dt2": t2 - t1, "plateau_rel": s2_t1,
                                     "max_drift": drift})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / f"{name}_law.csv", index=False)
    fits = []
    # theory slopes (theory_log_slope.py, lowest-loss rank-1 saddle) and which cases to draw where
    theory_a, log_cases, power_cases = {}, set(), set()
    if name == "e4a_sigma":
        th = pd.read_csv(HERE / "results" / "e4a_theory_slopes.csv")
        theory_a = {r.overlap: r.A_pred for r in th.sort_values("saddle_loss")
                    .drop_duplicates("overlap").itertuples()}
        log_cases, power_cases = {0.2, 0.45}, {0.0}
    elif name == "e2_sigma":
        log_cases, power_cases = {5, 20}, {0}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (val, g) in enumerate(df.groupby(key)):
        g = g[g["dt2"] > 0]
        # keep sigma values whose plateau is resolved (median plateau level at least half the
        # largest one in this case; at sigma = 0.3, c = 0.45 the drops merge)
        pl = g.groupby("sigma")["plateau_rel"].median()
        keep = pl.index[pl >= 0.5 * pl.max()]
        g = g[g["sigma"].isin(keep)]
        m = g.groupby("sigma")["dt2"].agg(["mean", "sem", "size"])
        lm = g.assign(l=np.log(g["dt2"])).groupby("sigma")["l"].mean()
        x_s = np.log(lm.index.values)
        (p_slope, p_icpt), (p_se, _), p_rms = fit(x_s, lm.values)
        (a_slope, a_icpt), (a_se, _), a_rms = fit(-np.log(m.index.values), m["mean"].values)
        rel_rms_log = a_rms / m["mean"].mean()
        med = g.groupby("sigma")["dt2"].median()
        (q_slope, _), (q_se, _), _ = fit(np.log(med.index.values), np.log(med.values))
        fits.append({key: val, "n_runs": len(g), "sigmas_used": len(m),
                     "power_exponent_medians": -q_slope, "power_medians_se": q_se,
                     "power_exponent": -p_slope, "power_se": p_se,
                     "power_rms_log": p_rms, "log_slope_A": a_slope, "log_slope_se": a_se,
                     "log_rel_rms": rel_rms_log,
                     "dt2_range": f"{m['mean'].min():.3g}-{m['mean'].max():.3g}"})
        c = f"C{i}"
        axes[0].errorbar(m.index, m["mean"], m["sem"], fmt="o", color=c,
                         label=f"{key}={val}: p = {-p_slope:.2f} +- {p_se:.2f}")
        xs = np.array(sorted(m.index))
        axes[0].plot(xs, np.exp(p_icpt) * xs**p_slope, "-", color=c, lw=0.8)
        if val in log_cases:
            axes[1].errorbar(-np.log(m.index), m["mean"], m["sem"], fmt="o", color=c,
                             label=f"{key}={val}: fitted A = {a_slope:.2f} +- {a_se:.2f}")
            axes[1].plot(-np.log(xs), a_slope * -np.log(xs) + a_icpt, "-", color=c, lw=0.8)
            if val in theory_a:  # predicted slope through the data's centre of mass
                x0, y0 = np.mean(-np.log(xs)), m["mean"].mean()
                axes[1].plot(-np.log(xs), y0 + theory_a[val] * (-np.log(xs) - x0), "--", color=c,
                             lw=1, label=f"{key}={val}: predicted A = {theory_a[val]:.2f}")
        if val in power_cases:
            axes[0].plot(xs, m["mean"].iloc[-1] * (xs / xs[-1]) ** -2.0, ":", color=c, lw=1,
                         label=f"{key}={val}: sigma^-2 guide")
    axes[0].set(xscale="log", yscale="log", xlabel="sigma",
                ylabel="Dt_2 = t_2 - t_1 (mean +- s.e., 10 seeds)",
                title=f"{name}: power-law fit")
    axes[1].set(xlabel="log(1/sigma)", ylabel="Dt_2 (linear scale)",
                title="logarithmic fit Dt_2 = A log(1/sigma) + B")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"{name}_law.png", dpi=110)
    out = pd.DataFrame(fits)
    out.to_csv(HERE / "results" / f"{name}_law_fits.csv", index=False)
    print(out.to_string())
    print("max conservation drift", df["max_drift"].max())


if __name__ == "__main__":
    main()
