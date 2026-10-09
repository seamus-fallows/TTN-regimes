"""Figures of the optimizer comparison, from saved trajectories and summary CSVs.

    uv run python -m experiments.optimizers.plots <figure> [...]   (see FIGURES at the bottom)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from experiments.optimizers.analysis import RESULTS, collect, load
from ttn import plotting as tp
from ttn.diagnostics import plateau_scaling

FIG = RESULTS / "figures"

# One colour per optimizer family, fixed across all figures (GD = dashed GF colour, DMRG = dashed
# ALS colour: each is the discretization / two-site variant of the other).
FAMILY_COLOR = dict(zip(
    ["gf", "adam", "rmsprop", "momentum", "regauge", "riemannian", "precond", "als"], tp.palette(8)))
FAMILY_COLOR.update(gd=FAMILY_COLOR["gf"], dmrg=FAMILY_COLOR["als"])
NAMES = {"gf": "GF", "gd": "GD", "momentum": "heavy ball", "adam": "Adam", "rmsprop": "RMSProp",
         "regauge": "GD + re-gauge", "riemannian": "Riemannian GD", "precond": "Gram-precond. GD",
         "als": "ALS (1-site)", "dmrg": "DMRG (2-site)"}
CLOCK = {"gf": "GF time t", "gd": "GF time lr n", "momentum": "GF time lr n/(1-beta)",
         "als": "sweeps", "dmrg": "sweeps"}


def _fmt(x) -> str:
    return f"{x:g}"


def method_key(row) -> str:
    """Optimizer plus the hyperparameters that distinguish settings within a sweep."""
    o = row["optimizer"]
    parts = [NAMES[o]]
    if o in ("gd", "momentum", "adam", "rmsprop", "regauge", "riemannian", "precond"):
        parts.append(f"lr {_fmt(row['lr'])}")
    if o == "adam" and pd.notna(row.get("beta2")) and row.get("beta2") not in (None, 0.999):
        parts.append(f"b2 {_fmt(row['beta2'])}")
    if o in ("adam", "rmsprop") and pd.notna(row.get("eps")) and row.get("eps") not in (None, 1e-8):
        parts.append(f"eps {_fmt(row['eps'])}")
    if o == "regauge":
        parts.append(f"every {int(row['period'])}")
    return ", ".join(parts)


def clock_factor(row) -> float:
    """Multiply the stored clock (steps for discrete methods) by this to get the plotted clock:
    GF time for GD and heavy ball, steps otherwise."""
    if row["optimizer"] == "gd":
        return row["lr"]
    if row["optimizer"] == "momentum":
        return row["lr"] / (1 - row.get("beta", 0.9))
    return 1.0


def with_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = [method_key(r) for _, r in df.iterrows()]
    df["clock"] = [clock_factor(r) for _, r in df.iterrows()]
    return df


# ---- scaling of event times with sigma ------------------------------------------------------------


def scaling_table(df: pd.DataFrame, quantities=("t_first", "gap_0", "gap_1", "t_loss_0.001")) -> pd.DataFrame:
    """Per method and quantity: fitted power-law exponent of T vs 1/sigma over all seeds (T in the
    method's clock), its standard error, the log-law slope, and the number of runs used. Runs that
    did not converge are excluded, and counted."""
    rows = []
    for method, g in df.groupby("method"):
        for q in quantities:
            if q not in g:
                continue
            ok = g[g["converged"] & np.isfinite(g[q])]
            fit = plateau_scaling(ok["sigma"], ok[q] * ok["clock"])
            rows.append({"method": method, "optimizer": g["optimizer"].iloc[0], "quantity": q,
                         "exponent": fit["power"]["exponent"], "exponent_se": fit["power"]["exponent_se"],
                         "log_slope": fit["log"]["slope"], "log_slope_se": fit["log"]["slope_se"],
                         "n_runs": fit["n"], "n_not_converged": int((~g["converged"]).sum()),
                         "sigma_min": ok["sigma"].min(), "sigma_max": ok["sigma"].max()})
    return pd.DataFrame(rows)


def fig_scaling(name: str, methods: list[str], quantities=("t_first", "gap_0", "gap_1"),
                titles=("first escape", "second event - first", "third event - second"), tag=""):
    """Event times vs sigma (mean +- s.e. over seeds; converged runs), each method normalized by its
    geometric mean so that slopes can be compared on one axis. Lines and legend: least-squares fit
    of log T on log(1/sigma) over all seeds (ttn.diagnostics.plateau_scaling), T ~ sigma^-p."""
    df = with_keys(collect(name))
    tp.set_style()
    fig, axes = tp.subplots(1, len(quantities), width=tp.TEXT_WIDTH, panel_aspect=0.95, sharey=True)
    handles, exps = {}, {}
    for ax, q, title in zip(np.atleast_1d(axes), quantities, titles):
        for m in methods:
            g = df[(df["method"] == m) & df["converged"]]
            if g.empty or q not in g:
                continue
            vals = g[q] * g["clock"]
            ok = np.isfinite(vals) & (vals > 0)
            # geometric means (the fit is in log space), error bars = s.e. of the mean log
            stats = np.log(vals[ok]).groupby(g["sigma"][ok]).agg(["mean", "sem", "count"])
            stats = stats[stats["count"] >= 2]
            if stats.empty:
                continue
            fit = plateau_scaling(g["sigma"][ok], vals[ok])["power"]
            norm = np.exp(stats["mean"].mean())
            geo = np.exp(stats["mean"])
            err = np.vstack([geo * (1 - np.exp(-stats["sem"])), geo * (np.exp(stats["sem"]) - 1)])
            fam = g["optimizer"].iloc[0]
            marker = {"Adam, lr 0.001, b2 0.9": "s", "Adam, lr 0.01": "^"}.get(m, "o")
            h = ax.errorbar(stats.index, geo / norm, yerr=np.nan_to_num(err) / norm, fmt=marker, ms=3.5,
                            color=FAMILY_COLOR[fam], mec="white", mew=0.6, elinewidth=0.8)
            xs = np.geomspace(stats.index.min(), stats.index.max(), 50)
            pref = np.exp(fit["log_prefactor"])
            ax.plot(xs, pref * (1 / xs) ** fit["exponent"] / norm, color=FAMILY_COLOR[fam], lw=0.9,
                    ls="--" if fam in ("gd", "dmrg") else "-")
            handles.setdefault(m, h)
            exps.setdefault(m, []).append(f"{fit['exponent']:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"init scale $\sigma$")
        ax.set_title(title)
    np.atleast_1d(axes)[0].set_ylabel("event time (geometric mean over seeds),\nrelative to its mean over sigma")
    labels = [f"{m}   (p = {' / '.join(exps[m])})" for m in handles]
    fig.legend(list(handles.values()), labels, loc="lower center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.17 - 0.035 * ((len(labels) + 1) // 2)), frameon=False,
               title=r"fitted $T\propto\sigma^{-p}$, p per panel (all seeds)", title_fontsize=6.5)
    tp.savefig(fig, FIG / f"{name}_scaling{tag}")
    return fig


# ---- side-by-side trajectories --------------------------------------------------------------------


def fig_side_by_side(name: str, sigma: float, seed: int, methods: list[str], edge_index: int = 0,
                     tag: str = "", log_t: bool = False, extra_filter: dict | None = None):
    """Rows: loss L/L0, overlaps a_r, bond spectrum of one edge; one column per method, all from
    the same initial point (same sigma, seed)."""
    df = with_keys(collect(name))
    sel = df[(df["sigma"] == sigma) & (df["seed"] == seed)]
    for k, v in (extra_filter or {}).items():
        sel = sel[sel[k] == v]
    tp.set_style()
    n = len(methods)
    fig, axes = tp.subplots(3, n, width=tp.TEXT_WIDTH, panel_aspect=0.9 * n / 3 * 3 / n * 1.0,
                            sharex="col", squeeze=False)
    fig.set_size_inches(tp.TEXT_WIDTH, 0.62 * tp.TEXT_WIDTH * 3 / max(n, 3) * 1.6)
    colors = tp.palette(4)
    for j, m in enumerate(methods):
        rows = sel[sel["method"] == m]
        if rows.empty:
            axes[0, j].set_title(m + " (missing)", fontsize=7)
            continue
        row = rows.iloc[0]
        tr = load(name, row["file"])
        d = tr.data
        c = row["clock"]
        t_f = (d["fine_t"] if "fine_t" in d else d["t"]) * c
        loss = d["fine_loss"] if "fine_loss" in d else d["loss"]
        a = d["fine_overlaps"] if "fine_overlaps" in d else d["overlaps"]
        ax = axes[0, j]
        ax.plot(t_f, loss / loss[0], color=tp.INK, lw=1.0)
        ax.set_yscale("log")
        ax.set_title(m.replace(", lr", "\nlr").replace(", every", ", every"), fontsize=7)
        ax = axes[1, j]
        for r in range(a.shape[1]):
            ax.plot(t_f, a[:, r], color=colors[r], lw=1.0, label=f"$a_{r + 1}$ (s={d['teacher_spectra'][0][r]:.0f})")
        ax.set_ylim(-0.1, 1.15)
        ax = axes[2, j]
        sp = d["spectra"][:, edge_index, :]
        for k in range(sp.shape[1]):
            ax.plot(d["t"] * c, sp[:, k], color=colors[k], lw=1.0, label=f"$s_{k + 1}$")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 10)
        ax.set_xlabel(CLOCK.get(row["optimizer"], "steps"), fontsize=7)
        if log_t:
            for i in range(3):
                axes[i, j].set_xscale("symlog", linthresh=1.0 if c != 1 else 1.0)
    axes[0, 0].set_ylabel(r"$L/L_0$")
    axes[1, 0].set_ylabel(r"overlap $a_r$")
    axes[2, 0].set_ylabel("bond spectrum")
    axes[1, -1].legend(fontsize=6, loc="lower right")
    axes[2, -1].legend(fontsize=6, loc="lower right")
    fig.align_ylabels(axes[:, 0])
    tp.savefig(fig, FIG / f"{name}_side_{tag}sigma{sigma:g}_seed{seed}")
    return fig


if __name__ == "__main__":
    print("use from scripts/make_figures.py")
