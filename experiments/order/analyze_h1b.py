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

RES = Path(__file__).parent / "results"
OUT = Path("/root/outputs/order")


def lam_pred(rho, k_cores=3):
    return (1 - rho) * rho ** ((k_cores - 2) / k_cores)


def saddle_fits(df):
    rows = []
    for rho, g in df.groupby("rho"):
        x, y = np.log(1 / g.sigma.values), g.t1.values
        ok = np.isfinite(y)
        a, b = np.polyfit(x[ok], y[ok], 1, cov=False), None
        res = y[ok] - np.polyval(a, x[ok])
        n = ok.sum()
        se = np.sqrt(res @ res / (n - 2) / np.sum((x[ok] - x[ok].mean()) ** 2))
        rows.append(dict(rho=rho, n_runs=int(n), slope=a[0], slope_se=se,
                         slope_pred=1 / lam_pred(rho), intercept=a[1],
                         a2_min_after_t1_max=g.a2_min_after_t1.max(),
                         lam_min_mean=g.lam_min.mean(), lam_pred=-lam_pred(rho)))
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
    w = nat[nat.wrong].copy()
    w["scaled"] = (w.t1 - w.t2) * np.abs(w.lam_pred)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for i, (rho, g) in enumerate(w.groupby("rho")):
        for j, (s, gg) in enumerate(g.groupby("sigma")):
            x = np.log10(s) + 0.08 * (i - 1)
            ax.plot(np.full(len(gg), x), gg.scaled, "o", color=style.SERIES[i], ms=2.5, alpha=0.5,
                    label=f"rho={rho}" if j == 0 else None)
            ax.plot([x - 0.05, x + 0.05], [gg.scaled.mean()] * 2, color=style.INK, lw=1.2)
    ax.set_xlabel("log10 sigma")
    ax.set_ylabel("|lambda_min| (t1 - t2)")
    ax.set_title("natural wrong-order runs (binary4): no growth with 1/sigma", fontsize=7)
    ax.legend(fontsize=6)
    ax = axes[1]
    for i, (rho, g) in enumerate(w.groupby("rho")):
        ax.plot(g.lam_pred, g.lam_min, "o", color=style.SERIES[i], ms=3, alpha=0.6, label=f"rho={rho}")
    lim = [min(w.lam_pred.min(), w.lam_min.min()) * 1.1, 0]
    ax.plot(lim, lim, color=style.MUTED, lw=0.8)
    ax.set_xlabel("predicted lambda_min = -(1-rho) rho^(1/3)")
    ax.set_ylabel("measured lambda_min (plateau midpoint)")
    ax.legend(fontsize=6)
    fig.savefig(RES / "h1b_natural.png")
    plt.close(fig)


def plot_example(path_b4, row_b4, path_m4, row_m4):
    plt = style.plt
    style.setup()
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2))
    for r, (path, row, name) in enumerate([(path_b4, row_b4, "binary4 (m=2)"),
                                           (path_m4, row_m4, "mps4 (m=4)")]):
        z = np.load(path)
        tr, eigs = z["traj"], z["eigs"]
        t = tr[:, 0]
        lo, hi = row.t2 * 0.85, min(t[-1], max(row.t1, row.t2) + 3 * (abs(row.t1 - row.t2) + 5))
        m = (t >= lo) & (t <= hi)
        ax = axes[r, 0]
        ax.plot(t[m], tr[m, 1], color=style.INK)
        for lev, lab in [(0.5, "s1^2/2"), (0.5 * row.rho**2, "s2^2/2")]:
            ax.axhline(lev, color=style.MUTED, lw=0.6, ls="--")
            ax.text(hi, lev, lab, fontsize=6, ha="right", va="bottom", color=style.MUTED)
        ax.set_ylabel("loss")
        ax.set_title(f"{name}, rho={row.rho}, sigma={row.sigma:g}, seed {row.seed}", fontsize=7)
        ax = axes[r, 1]
        ax.plot(t[m], tr[m, 2], color=style.SERIES[0], label="a1 (larger)")
        ax.plot(t[m], tr[m, 3], color=style.SERIES[1], label="a2 (smaller)")
        ax.plot(t[m], tr[m, 6], color=style.SERIES[2], lw=1, ls="--",
                label="2nd singular value, edge 0")
        if np.isfinite(row.get("t_mid", np.nan)):
            ax.axvline(row.t_mid, color=style.MUTED, lw=0.6)
        ax.set_ylabel("overlap")
        ax.legend(fontsize=6)
        ax = axes[r, 2]
        if len(eigs):
            ax.plot(np.arange(len(eigs)), eigs, "o", color=style.SERIES[0], ms=3)
            ax.axhline(0, color=style.MUTED, lw=0.6)
            ax.axhline(row.lam_pred if r == 0 else 0, color=style.SERIES[1], lw=0.8, ls="--")
            ax.set_ylabel("gauge-projected Hessian eigenvalue")
            ax.set_title(f"lambda_min = {eigs[0]:.4f} (pred {row.lam_pred if r == 0 else 0:.4f})",
                         fontsize=7)
        for a in axes[r, :2]:
            a.set_xlabel("t")
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
        w = w.assign(scaled=(w.t1 - w.t2) * lam_pred(w.rho))
        g = w.groupby(["rho", "sigma"]).agg(n=("seed", "size"), scaled_mean=("scaled", "mean"),
                                           scaled_se=("scaled", lambda x: x.std() / np.sqrt(len(x))),
                                           lam_min=("lam_min", "mean"),
                                           rotated=("a2_min_after_t1", lambda x: np.mean(x < 0.1)),
                                           loss_after=("loss_after_rotation", "mean")).reset_index()
        g["set"] = "natural_b4"
        out.append(g)
        print(g.to_string(index=False))
        plot_natural(nat)
        if (RES / "h1b_natural_mps4.csv").exists():
            m4 = pd.read_csv(RES / "h1b_natural_mps4.csv")
            m4w = m4[m4.wrong]
            print("mps4 natural wrong-order runs:", len(m4w), "lambda_min range",
                  m4w.lam_min.min(), m4w.lam_min.max(), "min a2 after t1", m4w.a2_min_after_t1.min())
            ex_b4 = w[(w.rho == 0.8)].sort_values("sigma").iloc[len(w[w.rho == 0.8]) // 2]
            ex_m4 = m4w.iloc[0]
            plot_example(OUT / "h1b_natural_b4" / f"rho{ex_b4.rho}_s{ex_b4.sigma:g}_seed{ex_b4.seed}.npz",
                         ex_b4, OUT / "h1b_natural_mps4" /
                         f"rho{ex_m4.rho}_s{ex_m4.sigma:g}_seed{ex_m4.seed}.npz", ex_m4)
    if (RES / "h1b_saddle_mps4.csv").exists():
        m4s = pd.read_csv(RES / "h1b_saddle_mps4.csv")
        g = m4s.groupby(["rho", "sigma"]).agg(n=("seed", "size"), t1_mean=("t1", "mean"),
                                             a2_min=("a2_min_after_t1", "min"),
                                             lam_min=("lam_min", "mean")).reset_index()
        g["set"] = "saddle_mps4"
        out.append(g)
        print(g.to_string(index=False))
    if out:
        pd.concat(out).to_csv(RES / "h1b_fits.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
