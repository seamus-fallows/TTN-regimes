"""H1b figures and fits.

Usage: uv run python experiments/order/analyze_h1b.py
Reads results/h1b_*.csv and the trajectories in /root/outputs/order/h1b_*/.
- h1b_saddle_b4: correction time t_corr (= t1, first a_1 = 1/2) vs ln(1/sigma) per rho; fitted slope
  vs the prediction 1/|lambda_min| = 1/((1-rho) rho^(1/3)).
- h1b_natural_b4: |lambda_pred| x (t1 - t2) distribution vs sigma (predicted sigma-independent);
  measured lambda_min at the wrong-plateau midpoint vs prediction; rotation check (a_2 -> 0).
- h1b_saddle_mps4 / h1b_natural_mps4 (m = 4 contrast): lambda_min vs sigma, a_2 stays ~1.
- Example run figure: loss, overlaps, bond spectrum, Hessian spectrum at the wrong plateau.
Writes results/h1b_fits.csv and results/h1b_*.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

from common import RAW, RES  # noqa: E402
OUT = RAW


def lam_pred(rho, k_cores=3):
    return (1 - rho) * rho ** ((k_cores - 2) / k_cores)


def saddle_fits(df):
    """Per-seed fits of t_corr against sigma (the same seed = the same perturbation direction at
    every sigma), mean +- SE over seeds. Log law t = a + b ln(1/sigma) vs power law t = c sigma^-p,
    both least squares in t (same space, same weights); runs that did not rotate are excluded and
    counted."""
    rows = []
    for rho, g in df.groupby("rho"):
        rotated = g[g.a2_min_after_t1 < 0.1]
        per_seed = []
        for seed, gs in rotated.groupby("seed"):
            if gs.sigma.nunique() < 5:
                continue
            x, y = np.log(1 / gs.sigma.values), gs.t1.values
            b, a = np.polyfit(x, y, 1)
            rms_log = np.sqrt(np.mean((y - (a + b * x)) ** 2))
            p_pow, lc = np.polyfit(x, np.log(y), 1)  # start for the power law
            from scipy.optimize import curve_fit

            (c, p_pow), _ = curve_fit(lambda xx, c, p: c * np.exp(p * xx), x, y,
                                      p0=(np.exp(lc), p_pow), maxfev=20000)
            rms_pow = np.sqrt(np.mean((y - c * np.exp(p_pow * x)) ** 2))
            per_seed.append(dict(slope=b, intercept=a, rms_log=rms_log, p_power=p_pow,
                                 rms_power=rms_pow))
        ps = pd.DataFrame(per_seed)
        n = len(ps)
        rows.append(dict(rho=rho, n_runs=len(g), n_rotated=len(rotated), n_seeds_fit=n,
                         sigma_min=g.sigma.min(), sigma_max=g.sigma.max(),
                         slope=ps.slope.mean(), slope_se=ps.slope.std(ddof=1) / np.sqrt(n),
                         slope_pred=1 / lam_pred(rho), intercept=ps.intercept.mean(),
                         rms_log=ps.rms_log.mean(), rms_power=ps.rms_power.mean(),
                         p_power=ps.p_power.mean(),
                         a2_min_after_t1_max=rotated.a2_min_after_t1.max(),
                         lam_min_mean=g.lam_min.mean(), lam_min_sd=g.lam_min.std(),
                         lam_min_raw_mean=g.lam_min_raw.mean(),
                         lam_pred=-lam_pred(rho)))
    return pd.DataFrame(rows)


def plot_saddle(df, fits):
    plt = style.plt
    style.setup()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, (rho, g) in enumerate(df.groupby("rho")):
        col = style.SERIES[i]
        x = np.log(1 / g.sigma)
        ax.plot(x, g.t1, "o", color=col, ms=3.5, alpha=0.7, label=f"rho={rho}")
        f = fits[fits.rho == rho].iloc[0]
        xx = np.linspace(x.min(), x.max(), 10)
        ax.plot(xx, f.intercept + f.slope_pred * xx, color=col, lw=1)
    ax.set_xlabel("ln(1/sigma)")
    ax.set_ylabel("correction time t_corr")
    ax.set_title("binary4 from the wrong saddle: lines = slope 1/|lambda_min| (theory)", fontsize=7)
    ax.legend(fontsize=6)
    fig.savefig(RES / "h1b_correction_time.png")
    plt.close(fig)


def plot_natural(nat):
    plt = style.plt
    style.setup()
    w = nat[nat.wrong & (nat.a2_min_after_t1 < 0.1)].copy()
    w["scaled"] = (w.t1 - w.t2) * np.abs(w.lam_pred)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw=dict(wspace=0.45))
    ax = axes[0]
    for i, (rho, g) in enumerate(w.groupby("rho")):
        for j, (s, gg) in enumerate(g.groupby("sigma")):
            x = np.log10(s) + 0.08 * (i - 1)
            ax.plot(np.full(len(gg), x), gg.scaled, "o", color=style.SERIES[i], ms=2.5, alpha=0.5,
                    label=f"rho={rho}" if j == 0 else None)
            ax.plot([x - 0.05, x + 0.05], [gg.scaled.mean()] * 2, color=style.INK, lw=1.2)
    ax.set_xlabel("log10 sigma")
    ax.set_ylabel("|lambda_pred| (t1 - t2)")
    ax.set_title("binary4 natural wrong-order runs that rotate: |lambda| (t1 - t2) vs sigma",
                 fontsize=7)
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    ax.set_ylim(0, 5.5)
    ax = axes[1]
    for i, (rho, g) in enumerate(w.groupby("rho")):
        ax.plot(g.lam_pred, g.lam_min, "o", color=style.SERIES[i], ms=3, alpha=0.6, label=f"rho={rho}")
    lim = [min(w.lam_pred.min(), w.lam_min.min()) * 1.1, 0]
    ax.plot(lim, lim, color=style.MUTED, lw=0.8)
    ax.set_xlabel("predicted lambda_min = -(1-rho) rho^(1/3)")
    ax.set_ylabel("lambda_min at minimum speed")
    ax.set_title("no resolved plateau: lambda_min between 0 and the saddle value", fontsize=7)
    ax.legend(fontsize=6, loc="lower right")
    fig.savefig(RES / "h1b_natural.png")
    plt.close(fig)


def plot_example(cases):
    """cases: list of (npz path, csv row, title, log_time, predicted saddle lambda_min). One row of panels per case: loss,
    overlaps (+ second singular value of edge 0), gauge-projected Hessian spectrum at the
    wrong-plateau midpoint."""
    plt = style.plt
    style.setup()
    fig, axes = plt.subplots(len(cases), 3, figsize=(10, 2.7 * len(cases)),
                             gridspec_kw=dict(wspace=0.35, hspace=0.55), squeeze=False)
    for r, (path, row, name, logt, pred) in enumerate(cases):
        z = np.load(path)
        tr, eigs = z["traj"], z["eigs"]
        t = tr[:, 0]
        if logt:
            m = t > 0
        else:
            lo = 0.85 * row.t2
            hi = min(t[-1], max(row.t1, row.t2) + 3 * (abs(row.t1 - row.t2) + 5))
            m = (t >= lo) & (t <= hi)
        ax = axes[r, 0]
        ax.plot(t[m], tr[m, 1], color=style.INK)
        for lev, lab in [(0.5, "s1^2/2"), (0.5 * row.rho**2, "s2^2/2")]:
            ax.axhline(lev, color=style.MUTED, lw=0.6, ls="--")
            ax.text(t[m][0], lev, lab, fontsize=6, ha="left", va="bottom", color=style.MUTED)
        ax.set_ylabel("loss")
        ax.set_title(name, fontsize=7)
        ax = axes[r, 1]
        ax.plot(t[m], tr[m, 2], color=style.SERIES[0], label="a1 (larger)")
        ax.plot(t[m], tr[m, 3], color=style.SERIES[1], label="a2 (smaller)")
        ax.plot(t[m], tr[m, 6], color=style.SERIES[2], lw=1, ls="--", label="2nd sing. value, edge 0")
        if np.isfinite(row.get("t_mid", np.nan)):
            ax.axvline(row.t_mid, color=style.MUTED, lw=0.6)
        ax.set_ylabel("overlap")
        ax.legend(fontsize=5.5, loc="center right")
        for a in axes[r, :2]:
            a.set_xlabel("t")
            if logt:
                a.set_xscale("log")
        ax = axes[r, 2]
        if len(eigs):
            ax.plot(np.arange(len(eigs)), eigs, "o", color=style.SERIES[0], ms=3)
            ax.axhline(0, color=style.MUTED, lw=0.6)
            ax.axhline(pred, color=style.SERIES[1], lw=0.8, ls="--")
            ax.set_ylabel("projected Hessian eig.")
            ax.set_xlabel("index")
            ax.set_title(f"midpoint: lambda_min = {eigs[0]:.4f} (raw {row.lam_min_raw:.4f}; "
                         f"m=2 saddle value {pred:.4f})" if pred else
                         f"midpoint: lambda_min = {eigs[0]:.2e} (raw {row.lam_min_raw:.2e}; m>=3: no "
                         "strict saddle predicted)", fontsize=6.5)
    fig.savefig(RES / "h1b_example.png")
    plt.close(fig)


def main():
    out = []
    if (RES / "h1b_saddle_b4.csv").exists():
        sad = pd.read_csv(RES / "h1b_saddle_b4.csv")
        fits = saddle_fits(sad)
        fits["set"] = "saddle_b4"
        out.append(fits)
        plot_saddle(sad, fits)
        print(fits.to_string(index=False))
    if (RES / "h1b_natural_b4.csv").exists():
        nat = pd.read_csv(RES / "h1b_natural_b4.csv")
        nat["lam_pred"] = -lam_pred(nat.rho)
        w = nat[nat.wrong]
        # classify before pooling: the correction-time statistics use rotated runs only
        # (a_2 falls below 0.1 after t1); the others added the larger component in a new bond index
        w = w.assign(scaled=np.where(w.a2_min_after_t1 < 0.1, (w.t1 - w.t2) * lam_pred(w.rho), np.nan))
        g = w.groupby(["rho", "sigma"]).agg(n=("seed", "size"), n_rotated=("scaled", "count"),
                                           scaled_mean=("scaled", "mean"),
                                           scaled_se=("scaled", lambda x: x.std() / np.sqrt(x.count())),
                                           lam_min=("lam_min", "mean"),
                                           lam_min_raw=("lam_min_raw", "mean"),
                                           rotated=("a2_min_after_t1", lambda x: np.mean(x < 0.1)),
                                           loss_after=("loss_after_rotation", "mean")).reset_index()
        sep = []
        for row in w.itertuples():
            z = np.load(OUT / "h1b_natural_b4" / f"rho{row.rho}_s{row.sigma:g}_seed{row.seed}.npz")
            tr = z["traj"]
            ws = []
            t_down = row.a2_down if np.isfinite(row.a2_down) else np.inf
            for col, t_ev, t_stop in ((3, row.t2, t_down), (2, row.t1, np.inf)):
                # width of the rise through the event: last 0.1-crossing before it to the first
                # 0.9-crossing after it, within the same rise (a_2 must reach 0.9 before it falls
                # back below 1/2 during the rotation; else NaN and only a_1's width is used)
                t, a = tr[:, 0], tr[:, col]
                before = np.nonzero((t <= t_ev) & (a < 0.1))[0]
                after = np.nonzero((t >= t_ev) & (t <= t_stop) & (a >= 0.9))[0]
                ws.append(t[after[0]] - t[before[-1]] if len(before) and len(after) else np.nan)
            sep.append((row.t1 - row.t2) / np.nanmax(ws) if np.isfinite(ws).any() else np.nan)
        w = w.assign(separation=sep)
        g = g.merge(w.groupby(["rho", "sigma"]).agg(
            separation_median=("separation", "median"),
            frac_plateau=("separation", lambda v: np.mean(v > 3))).reset_index())
        g["set"] = "natural_b4"
        out.append(g)
        print(g.to_string(index=False))
        plot_natural(nat)
        if (RES / "h1b_natural_mps4.csv").exists():
            m4 = pd.read_csv(RES / "h1b_natural_mps4.csv")
            m4w = m4[m4.wrong]
            stalled = m4w.stalled.astype(bool) if "stalled" in m4w else pd.Series(False, m4w.index)
            g4 = m4w.assign(stalled=stalled, gap=m4w.t1 - m4w.t2).groupby(["rho", "sigma"]).agg(
                n=("seed", "size"), n_stalled=("stalled", "sum"),
                a2_min_after_t1=("a2_min_after_t1", "min"),
                gap_median=("gap", "median"), lam_min_median=("lam_min", "median"),
                lam_min_most_negative=("lam_min", "min"),
                lam_min_raw_median=("lam_min_raw", "median")).reset_index()
            g4["set"] = "natural_mps4"
            out.append(g4)
            print(g4.to_string(index=False))
            ex_b4 = w[(w.rho == 0.8)].sort_values("sigma").iloc[len(w[w.rho == 0.8]) // 2]
            ex_m4 = m4w[~stalled & np.isfinite(m4w.t1)].iloc[0]
            cases = []
            if (RES / "h1b_saddle_b4.csv").exists():
                sad = pd.read_csv(RES / "h1b_saddle_b4.csv")
                ex_s = sad[(sad.rho == 0.8) & (sad.sigma == 1e-4) & (sad.seed == 0)].iloc[0]
                ex_s = ex_s.copy()
                ex_s["lam_pred"] = -lam_pred(0.8)
                cases.append((OUT / "h1b_saddle_b4" / "rho0.8_s0.0001_seed0.npz", ex_s,
                              "binary4 (m=2), start at the wrong saddle + noise 1e-4, rho=0.8", True,
                              -lam_pred(0.8)))
            cases.append((OUT / "h1b_natural_b4" /
                          f"rho{ex_b4.rho}_s{ex_b4.sigma:g}_seed{ex_b4.seed}.npz", ex_b4,
                          f"binary4 (m=2), random init, rho={ex_b4.rho}, sigma={ex_b4.sigma:g}, "
                          f"seed {ex_b4.seed}", False, -lam_pred(ex_b4.rho)))
            cases.append((OUT / "h1b_natural_mps4" /
                          f"rho{ex_m4.rho}_s{ex_m4.sigma:g}_seed{ex_m4.seed}.npz", ex_m4,
                          f"mps4 (m=4), random init, rho={ex_m4.rho}, sigma={ex_m4.sigma:g}, "
                          f"seed {ex_m4.seed}", False, 0.0))
            plot_example(cases)
    for name in ("h1b_saddle_b4", "h1b_saddle_mps4"):
        if not (RES / f"{name}.csv").exists():
            continue
        d = pd.read_csv(RES / f"{name}.csv")
        g = d.groupby(["rho", "sigma"]).agg(n=("seed", "size"), t1_mean=("t1", "mean"),
                                           a2_min=("a2_min_after_t1", "min"),
                                           lam_min=("lam_min", "mean"),
                                           lam_min_sd=("lam_min", "std"),
                                           lam_min_raw=("lam_min_raw", "mean"),
                                           n_neg=("n_neg", "mean")).reset_index()
        g["set"] = name + "_by_sigma"
        out.append(g)
        print(g.to_string(index=False))
        # primary saddle-type rule: per-seed scaling of lambda_min (and of t1) with sigma
        rows = []
        for (rho, seed), gs in d.groupby(["rho", "seed"]):
            x = np.log(gs.sigma.values)
            rows.append(dict(rho=rho, seed=seed,
                             lam_exp=np.polyfit(x, np.log(np.abs(gs.lam_min.values)), 1)[0],
                             t1_exp=np.polyfit(x, np.log(gs.t1.values), 1)[0]))
        e = pd.DataFrame(rows).groupby("rho").agg(
            n_seeds=("seed", "size"), lam_exponent=("lam_exp", "mean"),
            lam_exponent_se=("lam_exp", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
            t1_exponent=("t1_exp", "mean"),
            t1_exponent_se=("t1_exp", lambda v: v.std(ddof=1) / np.sqrt(len(v)))).reset_index()
        e["set"] = name + "_exponents"
        out.append(e)
        print(e.to_string(index=False))
    if out:
        pd.concat(out).to_csv(RES / "h1b_fits.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
