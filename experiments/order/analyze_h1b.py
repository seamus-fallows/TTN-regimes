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


# saddle-start sets with a strict saddle, and the class of run the law applies to:
# "rotated" = the learned bond index rotates to the larger component (a_2 < 0.1 afterwards; m = 2,
# k >= 3), "added" = the larger component is added on a spare bond index (a_2 stays > 0.9; k = 2)
LOG_LAW_SETS = {"h1b_saddle_b4": ("binary4 (m=2, k=3)", "rotated"),
                "h1b_saddle_mps4_sep": ("MPS4 cores 0,2 (m=2, k=3)", "rotated"),
                "h1b_saddle_mps4_adj": ("MPS4 cores 1,2 (m=2, k=2)", "added")}
SCALING_SETS = ["h1b_saddle_mps4", "h1b_saddle_mps6", "h1b_saddle_b8"]  # m >= 3 (odeco)
NATURAL_SETS = {"h1b_natural_b4": "rotated", "h1b_natural_mps4_sep": "rotated",
                "h1b_natural_mps4_adj": "added", "h1b_natural_mps4": "added"}
# predicted saddle eigenvalue for every run (also the ones without a resolved midpoint Hessian):
# (number of cores K, kind)
PRED_OF = {"h1b_natural_b4": (3, "rotation"), "h1b_natural_mps4_sep": (4, "rotation"),
           "h1b_natural_mps4_adj": (4, "add"), "h1b_natural_mps4": (4, "none")}


def lam_predicted(rho, k_cores, kind):
    q = rho ** (1.0 / k_cores)
    return {"rotation": -(1 - rho) * q ** (k_cores - 2), "add": -q ** (k_cores - 2),
            "none": 0.0 * rho}[kind]


def run_class(df):
    a = df.a2_min_after_t1
    return np.where(a < 0.1, "rotated", np.where(a > 0.9, "added", np.where(a.isna(), "none",
                                                                             "other")))


def saddle_fits(df, cls):
    """Per-seed fits of t_corr = t1 against sigma (the same seed = the same perturbation direction
    at every sigma), mean +- SE over seeds, for runs of class `cls`. Log law t = a + b ln(1/sigma) vs
    power law t = c sigma^-p, both least squares in t (same space, same weights). Also the local
    slope between consecutive sigma times |lambda_pred| (1 = the asymptotic law)."""
    from scipy.optimize import curve_fit

    df = df.assign(cls=run_class(df))
    rows, local = [], []
    for rho, g in df.groupby("rho"):
        sub = g[g.cls == cls]
        lam = float(np.abs(g.lam_pred.iloc[0]))
        per_seed = []
        for seed, gs in sub.groupby("seed"):
            gs = gs.sort_values("sigma", ascending=False)
            if gs.sigma.nunique() < 5:
                continue
            x, y = np.log(1 / gs.sigma.values), gs.t1.values
            b, a = np.polyfit(x, y, 1)
            rms_log = np.sqrt(np.mean((y - (a + b * x)) ** 2))
            p_pow, lc = np.polyfit(x, np.log(y), 1)
            (c, p_pow), _ = curve_fit(lambda xx, c, p: c * np.exp(p * xx), x, y,
                                      p0=(np.exp(lc), p_pow), maxfev=20000)
            rms_pow = np.sqrt(np.mean((y - c * np.exp(p_pow * x)) ** 2))
            per_seed.append(dict(slope=b, intercept=a, rms_log=rms_log, p_power=p_pow,
                                 rms_power=rms_pow))
            for k in range(len(x) - 1):
                local.append(dict(rho=rho, seed=seed, sigma_hi=gs.sigma.values[k],
                                  sigma_lo=gs.sigma.values[k + 1],
                                  local_x_lam=(y[k + 1] - y[k]) / (x[k + 1] - x[k]) * lam))
        ps = pd.DataFrame(per_seed)
        n = len(ps)
        rows.append(dict(rho=rho, n_runs=len(g), n_class=len(sub),
                         n_other_class=int((g.cls != cls).sum()), n_seeds_fit=n,
                         sigma_min=g.sigma.min(), sigma_max=g.sigma.max(),
                         slope=ps.slope.mean(), slope_se=ps.slope.std(ddof=1) / np.sqrt(n),
                         slope_pred=1 / lam, intercept=ps.intercept.mean(),
                         rms_log=ps.rms_log.mean(), rms_power=ps.rms_power.mean(),
                         p_power=ps.p_power.mean(), a2_min_after_t1_max=sub.a2_min_after_t1.max(),
                         a2_min_after_t1_min=sub.a2_min_after_t1.min(),
                         lam_min_mean=g.lam_min.mean(), lam_min_sd=g.lam_min.std(),
                         lam_min_raw_mean=g.lam_min_raw.mean(), lam_pred=-lam,
                         n_near_add_mean=g.n_near_add.mean() if "n_near_add" in g else np.nan,
                         lam_4_mean=g.lam_4.mean() if "lam_4" in g else np.nan))
    local = pd.DataFrame(local)
    if len(local):
        local = local.groupby(["rho", "sigma_hi", "sigma_lo"]).local_x_lam.agg(
            ["mean", "std", "count"]).reset_index()
    return pd.DataFrame(rows), local


def plot_saddle(sets):
    """sets: list of (title, df, fits); t1 vs ln(1/sigma) with lines of the predicted slope."""
    plt = style.plt
    style.setup()
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 2.9), squeeze=False,
                             gridspec_kw=dict(wspace=0.35))
    for ax, (title, df, fits) in zip(axes[0], sets):
        for i, (rho, g) in enumerate(df.groupby("rho")):
            col = style.SERIES[i]
            x = np.log(1 / g.sigma)
            ax.plot(x, g.t1, "o", color=col, ms=3, alpha=0.7, label=f"rho={rho}")
            f = fits[fits.rho == rho].iloc[0]
            xx = np.linspace(x.min(), x.max(), 10)
            ax.plot(xx, f.intercept + f.slope_pred * xx, color=col, lw=1)
        ax.set_xlabel("ln(1/sigma)")
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("t1 from the wrong saddle")
    fig.suptitle("lines: slope 1/|lambda_min| predicted (rotation for k = 3, adding for k = 2)",
                 fontsize=8, y=1.03)
    fig.savefig(RES / "h1b_correction_time.png")
    plt.close(fig)


def plot_natural(nat):
    plt = style.plt
    style.setup()
    w = nat[nat.wrong & (nat.a2_min_after_t1 < 0.1)].copy()
    w["lam_pred"] = lam_predicted(w.rho, 3, "rotation")  # all runs, resolved Hessian or not
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
    wr = w[w.lam_min.notna()]  # runs with a resolved minimum-speed point (others: unresolved)
    for i, (rho, g) in enumerate(wr.groupby("rho")):
        ax.plot(g.lam_pred, g.lam_min, "o", color=style.SERIES[i], ms=3, alpha=0.6, label=f"rho={rho}")
    lim = [min(wr.lam_pred.min(), wr.lam_min.min()) * 1.1, 0]
    ax.plot(lim, lim, color=style.MUTED, lw=0.8)
    ax.set_xlabel("predicted lambda_min = -(1-rho) rho^(1/3)")
    ax.set_ylabel("lambda_min at minimum speed")
    ax.set_title("no resolved plateau (lambda_min at the minimum-speed point)", fontsize=7)
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


def natural_table(name, cls):
    """Wrong-order runs from random init: classes, unresolved Hessians, |lambda_pred| (t1 - t2)
    for the class the law applies to, separation ratios."""
    nat = pd.read_csv(RES / f"{name}.csv")
    for col in ("lam_min", "lam_min_raw", "loss_after_rotation", "a2_down"):
        if col not in nat:
            nat[col] = np.nan
    if "gap" not in nat:
        nat["gap"] = nat.t1 - nat.t2
    w = nat[nat.wrong].copy()
    w["cls"] = run_class(w)
    w["stalled"] = w.stalled.astype(bool) if "stalled" in w else False
    k_cores, kind = PRED_OF[name]
    w["lam_pred"] = lam_predicted(w.rho, k_cores, kind)
    w["scaled"] = np.where(w.cls == cls, w.gap * np.abs(w.lam_pred), np.nan)
    sep = []
    for row in w.itertuples():
        path = OUT / name / f"rho{row.rho}_s{row.sigma:g}_seed{row.seed}.npz"
        if not path.exists() or not np.isfinite(row.t1):
            sep.append(np.nan)
            continue
        tr = np.load(path)["traj"]
        ws = []
        t_down = row.a2_down if np.isfinite(row.a2_down) else np.inf
        for col, t_ev, t_stop in ((3, row.t2, t_down), (2, row.t1, np.inf)):
            # rise through the event: last 0.1-crossing before it to the first 0.9-crossing after
            # it within the same rise (NaN if a_2 turns down before reaching 0.9)
            t, a = tr[:, 0], tr[:, col]
            before = np.nonzero((t <= t_ev) & (a < 0.1))[0]
            after = np.nonzero((t >= t_ev) & (t <= t_stop) & (a >= 0.9))[0]
            ws.append(t[after[0]] - t[before[-1]] if len(before) and len(after) else np.nan)
        sep.append((row.t1 - row.t2) / np.nanmax(ws) if np.isfinite(ws).any() else np.nan)
    w["separation"] = sep
    g = w.groupby(["rho", "sigma"]).agg(
        n=("seed", "size"), n_rotated=("cls", lambda c: int((c == "rotated").sum())),
        n_added=("cls", lambda c: int((c == "added").sum())),
        n_stalled=("stalled", "sum"),
        n_unresolved_hessian=("lam_min", lambda v: int(v.isna().sum())),
        scaled_mean=("scaled", "mean"),
        scaled_se=("scaled", lambda x: x.std() / np.sqrt(max(x.count(), 1))),
        lam_min=("lam_min", "mean"), lam_min_raw=("lam_min_raw", "mean"),
        loss_after=("loss_after_rotation", "mean"), gap_median=("gap", "median"),
        separation_median=("separation", "median"),
        frac_plateau=("separation", lambda v: np.mean(v > 3))).reset_index()
    g["set"] = name
    return g, w


def main():
    out, local_all, log_sets = [], [], []
    for name, (title, cls) in LOG_LAW_SETS.items():
        if not (RES / f"{name}.csv").exists():
            continue
        sad = pd.read_csv(RES / f"{name}.csv")
        if "lam_pred" not in sad or sad.lam_pred.isna().all():
            sad["lam_pred"] = -(1 - sad.rho) * sad.rho ** (1 / 3)
        fits, local = saddle_fits(sad, cls)
        fits["set"] = name
        out.append(fits)
        if len(local):
            local["set"] = name
            local_all.append(local)
        log_sets.append((title, sad, fits))
        print(name, "\n", fits.to_string(index=False))
    if log_sets:
        plot_saddle(log_sets)
    if local_all:
        pd.concat(local_all).to_csv(RES / "h1b_local_slopes.csv", index=False, float_format="%.5g")
    nat_rows = {}
    for name, cls in NATURAL_SETS.items():
        if (RES / f"{name}.csv").exists():
            g, w = natural_table(name, cls)
            out.append(g)
            nat_rows[name] = w
            print(g.to_string(index=False))
    if (RES / "h1b_natural_b4.csv").exists():
        nat = pd.read_csv(RES / "h1b_natural_b4.csv")
        plot_natural(nat[nat.sigma <= 0.01])
    if "h1b_natural_b4" in nat_rows and "h1b_natural_mps4" in nat_rows:
        w, m4w = nat_rows["h1b_natural_b4"], nat_rows["h1b_natural_mps4"]
        ex_b4 = w[(w.rho == 0.8) & (w.cls == "rotated")].sort_values("sigma")
        ex_b4 = ex_b4.iloc[len(ex_b4) // 2]
        ex_m4 = m4w[~m4w.stalled & np.isfinite(m4w.t1)].iloc[0]
        cases = []
        sad = pd.read_csv(RES / "h1b_saddle_b4.csv")
        ex_s = sad[(sad.rho == 0.8) & (sad.sigma == 1e-4) & (sad.seed == 0)].iloc[0]
        cases.append((OUT / "h1b_saddle_b4" / "rho0.8_s0.0001_seed0.npz", ex_s,
                      "binary4 (m=2, k=3), start at the wrong saddle + noise 1e-4, rho=0.8", True,
                      -(1 - 0.8) * 0.8 ** (1 / 3)))
        if (RES / "h1b_saddle_mps4_adj.csv").exists():
            adj = pd.read_csv(RES / "h1b_saddle_mps4_adj.csv")
            ex_a = adj[(adj.rho == 0.8) & (adj.sigma == 1e-4) & (adj.seed == 0)].iloc[0]
            cases.append((OUT / "h1b_saddle_mps4_adj" / "rho0.8_s0.0001_seed0.npz", ex_a,
                          "MPS4 cores 1,2 (m=2, k=2), wrong saddle + noise 1e-4, rho=0.8", True,
                          float(ex_a.lam_pred)))
        cases.append((OUT / "h1b_natural_b4" /
                      f"rho{ex_b4.rho}_s{ex_b4.sigma:g}_seed{ex_b4.seed}.npz", ex_b4,
                      f"binary4 (m=2), random init, rho={ex_b4.rho}, sigma={ex_b4.sigma:g}, "
                      f"seed {ex_b4.seed}", False, -(1 - ex_b4.rho) * ex_b4.rho ** (1 / 3)))
        cases.append((OUT / "h1b_natural_mps4" /
                      f"rho{ex_m4.rho}_s{ex_m4.sigma:g}_seed{ex_m4.seed}.npz", ex_m4,
                      f"mps4 (m=4), random init, rho={ex_m4.rho}, sigma={ex_m4.sigma:g}, "
                      f"seed {ex_m4.seed}", False, 0.0))
        plot_example(cases)
    for name in SCALING_SETS + list(LOG_LAW_SETS):
        if not (RES / f"{name}.csv").exists():
            continue
        d = pd.read_csv(RES / f"{name}.csv")
        g = d.groupby(["rho", "sigma"]).agg(n=("seed", "size"), t1_mean=("t1", "mean"),
                                           n_t1_finite=("t1", lambda v: int(np.isfinite(v).sum())),
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
            ok = np.isfinite(gs.t1.values) & np.isfinite(gs.lam_min.values)
            if ok.sum() < 3:
                continue
            x = np.log(gs.sigma.values[ok])
            rows.append(dict(rho=rho, seed=seed,
                             lam_exp=np.polyfit(x, np.log(np.abs(gs.lam_min.values[ok])), 1)[0],
                             t1_exp=np.polyfit(x, np.log(gs.t1.values[ok]), 1)[0]))
        if not rows:
            continue
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
