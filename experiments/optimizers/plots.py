"""Figures of the optimizer comparison, from saved trajectories and summary CSVs.

uv run python -m experiments.optimizers.plots <figure> [...]   (see FIGURES at the bottom)
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from experiments.optimizers.analysis import RESULTS, collect, load
from ttn import plotting as tp

FIG = RESULTS / "figures"

# One colour per optimizer family, fixed across all figures (GD = dashed GF colour, DMRG = dashed
# ALS colour: each is the discretization / two-site variant of the other).
FAMILY_COLOR = dict(
    zip(
        ["gf", "adam", "rmsprop", "momentum", "regauge", "riemannian", "precond", "als"],
        tp.palette(8),
    )
)
FAMILY_COLOR.update(gd=FAMILY_COLOR["gf"], dmrg=FAMILY_COLOR["als"])
NAMES = {
    "gf": "GF",
    "gd": "GD",
    "momentum": "heavy ball",
    "adam": "Adam",
    "rmsprop": "RMSProp",
    "regauge": "GD + re-gauge",
    "riemannian": "Riemannian GD",
    "precond": "Gram-precond. GD",
    "als": "ALS (1-site)",
    "dmrg": "DMRG (2-site)",
}
CLOCK = {  # x-axis labels in the report's notation (eta: step, n: step count)
    "gf": "$t$",
    "gd": r"$\eta n$",
    "momentum": r"$\eta n/(1-\beta)$",
    "als": "sweeps",
    "dmrg": "sweeps",
}


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


def _log_law_rms(x: np.ndarray, y: np.ndarray) -> float:
    """rms residual in ln T of the log law T = a + b ln(1/sigma), fitted by least squares in ln T (the
    same space and weighting as the power law; analysis standard 5), with T > 0 on the data enforced."""
    from scipy.optimize import least_squares

    cl, *_ = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)
    b0 = max(cl[1], 1e-12 * y.mean())
    a0 = max(y.min() - b0 * x.min(), 1e-6 * y.min()) if b0 > 0 else y.mean()

    def resid(ab):
        model = ab[0] + ab[1] * x
        return np.log(np.clip(model, 1e-300, None)) - np.log(y)

    best = least_squares(resid, x0=[a0 if a0 + b0 * x.min() > 0 else y.min(), b0])
    return float(np.sqrt(np.mean(best.fun**2)))


def per_seed_fits(g: pd.DataFrame, q: str, min_sigmas: int = 4) -> dict:
    """Scaling of quantity q (method clock) with sigma, fitted per seed (each seed is one init direction
    rescaled by sigma; notes/briefs/README.md, analysis standards 1, 3, 4, 5).

    Per seed: keep converged runs whose event is resolved (`<q>_resolved`) and, for event quantities, whose
    learning order is the seed's most common order (same saddle sequence; not applied to the times to reach a
    loss level, t_loss_*), need >= min_sigmas sigma values; fit
    power  ln T = c + p ln(1/sigma)  and  log  T = a + b ln(1/sigma), both scored by the rms residual of
    ln T (the log-law prediction is clipped at a tiny positive value). Returns mean +- s.e. of p over seeds,
    the mean scores and the exclusion counts."""
    res_col = (
        "t_first_resolved"
        if q == "t_first"
        else (f"{q}_resolved" if f"{q}_resolved" in g else None)
    )
    p_list, rms_pow, rms_log = [], [], []
    n_unres = n_order = n_short = 0
    for seed, gs in g.groupby("seed"):
        gs = gs[gs["converged"] & np.isfinite(gs[q] * gs["clock"])]
        if res_col is not None:
            n_unres += int((~gs[res_col].astype(bool)).sum())
            gs = gs[gs[res_col].astype(bool)]
        if gs.empty:
            continue
        if not q.startswith(
            "t_loss"
        ):  # plateau durations: one saddle sequence per fit (standard 3)
            modal = gs["order"].astype(str).value_counts().index[0]
            n_order += int((gs["order"].astype(str) != modal).sum())
            gs = gs[gs["order"].astype(str) == modal]
        y = (gs[q] * gs["clock"]).values
        x = np.log(1 / gs["sigma"].values)
        if gs["sigma"].nunique() < min_sigmas or np.any(y <= 0):
            n_short += 1
            continue
        A = np.vstack([np.ones_like(x), x]).T
        cp, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        p_list.append(cp[1])
        rms_pow.append(np.sqrt(np.mean((np.log(y) - A @ cp) ** 2)))
        rms_log.append(_log_law_rms(x, y))
    p_arr = np.array(p_list)
    return {
        "exponent": p_arr.mean() if len(p_arr) else np.nan,
        "exponent_se": p_arr.std(ddof=1) / np.sqrt(len(p_arr)) if len(p_arr) > 1 else np.nan,
        "n_seeds": len(p_arr),
        "rms_lnT_power": np.mean(rms_pow) if rms_pow else np.nan,
        "rms_lnT_log": np.mean(rms_log) if rms_log else np.nan,
        "n_unresolved": n_unres,
        "n_other_order": n_order,
        "n_seeds_too_few_sigma": n_short,
        "per_seed": " ".join(f"{v:.2f}" for v in p_arr),
    }


def scaling_table(
    df: pd.DataFrame, quantities=("t_first", "gap_0", "gap_1", "t_loss_0.001")
) -> pd.DataFrame:
    """Per method and quantity: per-seed power-law exponents (T ~ sigma^-p, mean +- s.e. over seeds), the
    log-law alternative's score, and what was excluded (unresolved, other saddle sequence, too few sigma)."""
    rows = []
    for method, g in df.groupby("method"):
        for q in quantities:
            if q not in g:
                continue
            fit = per_seed_fits(g, q)
            rows.append(
                {
                    "method": method,
                    "optimizer": g["optimizer"].iloc[0],
                    "quantity": q,
                    "n_runs": len(g),
                    "n_not_converged": int((~g["converged"]).sum()),
                    "sigma_min": g["sigma"].min(),
                    "sigma_max": g["sigma"].max(),
                    "n_sigma": g["sigma"].nunique(),
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def fig_scaling(
    name: str,
    methods: list[str],
    quantities=("t_first", "gap_0", "gap_1"),
    titles=("first escape", "second event - first", "third event - second"),
    tag="",
):
    """Event times vs sigma (mean +- s.e. over seeds; converged runs), each method normalized by its
    geometric mean so that slopes can be compared on one axis. Lines and legend: per-seed fits of
    log T on log(1/sigma) (`per_seed_fits`: resolved events, the seed's modal saddle sequence), mean +- s.e.
    of the exponent over seeds, T ~ sigma^-p."""
    df = with_keys(collect(name))
    tp.set_style()
    fig, axes = tp.subplots(1, len(quantities), width=tp.TEXT_WIDTH, panel_aspect=0.95, sharey=True)
    handles, exps = {}, {}
    for ax, q, title in zip(np.atleast_1d(axes), quantities, titles):
        for m in methods:
            g = df[(df["method"] == m) & df["converged"]]
            if g.empty or q not in g:
                continue
            res_col = "t_first_resolved" if q == "t_first" else f"{q}_resolved"
            if res_col in g:  # plot resolved events only (analysis standard 1)
                g = g[g[res_col].astype(bool)]
            vals = g[q] * g["clock"]
            ok = np.isfinite(vals) & (vals > 0)
            # geometric means (the fit is in log space), error bars = s.e. of the mean log
            stats = np.log(vals[ok]).groupby(g["sigma"][ok]).agg(["mean", "sem", "count"])
            stats = stats[stats["count"] >= 2]
            if stats.empty:
                continue
            fit = per_seed_fits(g, q)
            norm = np.exp(stats["mean"].mean())
            geo = np.exp(stats["mean"])
            err = np.vstack([geo * (1 - np.exp(-stats["sem"])), geo * (np.exp(stats["sem"]) - 1)])
            fam = g["optimizer"].iloc[0]
            marker = {"Adam, lr 0.001, b2 0.9": "s", "Adam, lr 0.01": "^"}.get(m, "o")
            h = ax.errorbar(
                stats.index,
                geo / norm,
                yerr=np.nan_to_num(err) / norm,
                fmt=marker,
                ms=3.5,
                color=FAMILY_COLOR[fam],
                mec="white",
                mew=0.6,
                elinewidth=0.8,
            )
            xs = np.geomspace(stats.index.min(), stats.index.max(), 50)
            if np.isfinite(
                fit["exponent"]
            ):  # line with the mean per-seed slope through the geometric means
                lx = np.log(1 / stats.index.values)
                c0 = np.mean(stats["mean"].values - fit["exponent"] * lx)
                ax.plot(
                    xs,
                    np.exp(c0 + fit["exponent"] * np.log(1 / xs)) / norm,
                    color=FAMILY_COLOR[fam],
                    lw=0.9,
                    ls="--" if fam in ("gd", "dmrg") else "-",
                )
            handles.setdefault(m, h)
            exps.setdefault(m, []).append(f"{fit['exponent']:.2f}+-{fit['exponent_se']:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"init scale $\sigma$")
        ax.set_title(title)
    np.atleast_1d(axes)[0].set_ylabel(
        "event time (geometric mean over seeds),\nrelative to its mean over sigma"
    )
    labels = [f"{m}   (p = {' / '.join(exps[m])})" for m in handles]
    fig.legend(
        list(handles.values()),
        labels,
        loc="lower center",
        ncol=2,
        fontsize=6.5,
        bbox_to_anchor=(0.5, -0.17 - 0.035 * ((len(labels) + 1) // 2)),
        frameon=False,
        title=r"$T\propto\sigma^{-p}$: per-seed fits, mean $\pm$ s.e. over seeds, p per panel",
        title_fontsize=6.5,
    )
    tp.savefig(fig, FIG / f"{name}_scaling{tag}")
    return fig


# ---- side-by-side trajectories --------------------------------------------------------------------


def fig_side_by_side(
    name: str,
    sigma: float,
    seed: int,
    methods: list[str],
    edge_index: int = 0,
    tag: str = "",
    log_t: bool = False,
    extra_filter: dict | None = None,
):
    """Rows: loss L/L0, overlaps a_r, bond spectrum of one edge; one column per method, all from
    the same initial point (same sigma, seed)."""
    df = with_keys(collect(name))
    sel = df[(df["sigma"] == sigma) & (df["seed"] == seed)]
    for k, v in (extra_filter or {}).items():
        sel = sel[sel[k] == v]
    tp.set_style()
    n = len(methods)
    fig, axes = plt.subplots(
        3, n, figsize=(tp.TEXT_WIDTH, 4.2), sharex="col", squeeze=False, layout="constrained"
    )
    colors = tp.palette(4)
    weights = None
    for j, m in enumerate(methods):
        rows = sel[sel["method"] == m]
        if rows.empty:
            axes[0, j].set_title(m + " (missing)", fontsize=8)
            continue
        row = rows.iloc[0]
        weights = json.loads(row["weights"])
        tr = load(name, row["file"])
        d = tr.data
        c = row["clock"]
        t_f = (d["fine_t"] if "fine_t" in d else d["t"]) * c
        loss = d["fine_loss"] if "fine_loss" in d else d["loss"]
        a = d["fine_overlaps"] if "fine_overlaps" in d else d["overlaps"]
        ax = axes[0, j]
        ax.plot(t_f, loss / loss[0], color=tp.INK, lw=1.0)
        ax.set_yscale("log")
        ax.set_title(display_title(m), fontsize=8)
        ax = axes[1, j]
        for r in range(a.shape[1]):
            ax.plot(t_f, a[:, r], color=colors[r], lw=1.0)
        ax.set_ylim(-0.1, 1.15)
        ax = axes[2, j]
        sp = d["spectra"][:, edge_index, :]
        for k in range(sp.shape[1]):
            ax.plot(d["t"] * c, sp[:, k], color=colors[k], lw=1.0)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 10)
        ax.set_xlabel(CLOCK.get(row["optimizer"], "steps"), fontsize=8)
        if log_t:
            for i in range(3):
                axes[i, j].set_xscale("symlog", linthresh=1.0 if c != 1 else 1.0)
    axes[0, 0].set_ylabel(r"$L/L_0$")
    axes[1, 0].set_ylabel(r"overlap $a_r$")
    axes[2, 0].set_ylabel(r"singular values $s^e_k$")
    # one legend below the panels: component r and singular-value index k share colour slot r = k
    handles = [
        Line2D(
            [], [], color=colors[r], lw=1.2, label=f"$a_{r + 1}$, $s^e_{r + 1}$ ($s_{r + 1}={w:g}$)"
        )
        for r, w in enumerate(weights or [])
    ]
    handles += [
        Line2D([], [], color=colors[k], lw=1.2, label=f"$s^e_{k + 1}$")
        for k in range(len(weights or []), len(colors))
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), fontsize=8)
    fig.align_ylabels(axes[:, 0])
    tp.savefig(fig, FIG / f"{name}_side_{tag}sigma{sigma:g}_seed{seed}")
    return fig


def _lr_text(x: float) -> str:
    """A step size in mathtext: 10^k, m 10^k for x < 0.01, other values as decimals."""
    k = np.log10(x)
    if abs(k - round(k)) < 1e-9:
        return f"10^{{{int(round(k))}}}"
    if x < 0.01:
        m, e = f"{x:.0e}".split("e")
        return rf"{m}\cdot10^{{{int(e)}}}"
    return f"{x:g}"


def display_title(method: str) -> str:
    """Panel title in the report's notation (eta for the step) from a method key of method_key()."""
    parts = method.split(", ")
    title, rest = parts[0], []
    for p in parts[1:]:
        if p.startswith("lr "):
            rest.append(rf"$\eta={_lr_text(float(p[3:]))}$")
        elif p.startswith("eps "):
            rest.append(rf"$\epsilon={_lr_text(float(p[4:]))}$")
        elif p.startswith("b2 "):
            rest.append(rf"$\beta_2={p[3:]}$")
        elif p == "every 1":
            rest.append("every step")
        else:
            rest.append(p)
    title = {
        "GF": "gradient flow",
        "ALS (1-site)": "ALS (one-site)",
        "DMRG (2-site)": "DMRG (two-site)",
    }.get(title, title)
    return title + "\n" + ", ".join(rest) if rest else title + "\n"


if __name__ == "__main__":
    print("use from scripts/make_figures.py")


# ---- Adam internals ---------------------------------------------------------------------------------


def fig_probe(name: str = "adam_probe"):
    """Adam's state along representative runs (adam_probe.py): overlaps, the median preconditioner
    sqrt(vhat) (with the pure-memory decay beta2^(n/2) from the end of each learning event as a guide)
    and the preconditioned sharpness relative to the adaptive-edge-of-stability threshold."""
    import glob

    from experiments.optimizers.run import OUT_ROOT

    runs = []
    for f in sorted(glob.glob(str(OUT_ROOT / name / "*_probe.npz"))):
        z = np.load(f)
        cfg = json.loads(str(z["cfg"]))
        runs.append((cfg, {k: z[k] for k in z.files}))
    order = [
        ("binary", 4, "adam", 1e-3, 0.9),
        ("binary", 4, "adam", 1e-3, 0.999),
        ("binary", 8, "adam", 1e-3, None),
        ("mps", 8, "adam", 1e-3, None),
    ]
    picked = []
    for key in order:
        for cfg, z in runs:
            if (cfg["tree"], cfg["n_sites"], cfg["optimizer"], cfg["lr"], cfg.get("beta2")) == key:
                picked.append((cfg, z))
    tp.set_style()
    fig, axes = tp.subplots(
        3, len(picked), width=tp.TEXT_WIDTH, panel_aspect=0.85, sharex="col", squeeze=False
    )
    colors = tp.palette(3)
    for j, (cfg, z) in enumerate(picked):
        b2 = cfg.get("beta2", 0.999)
        teacher = "odeco" if cfg["teacher"] == "odeco" else "pattern E8"
        axes[0, j].set_title(
            f"{cfg['tree']}{cfg['n_sites']} {teacher}\nAdam lr {cfg['lr']:g}, b2 {b2:g}", fontsize=7
        )
        t, a = z["fine_t"], z["fine_overlaps"]
        for r in range(a.shape[1]):
            axes[0, j].plot(
                np.maximum(t, 1), a[:, r], color=colors[r], lw=1.0, label=f"$a_{r + 1}$"
            )
        axes[0, j].set_ylim(-0.1, 1.15)
        n = z["step"]
        axes[1, j].plot(n, z["p_median"], color=tp.INK, lw=1.0, label=r"median $\sqrt{\hat v}$")
        i0 = int(np.argmax(z["p_median"]))
        guide = z["p_median"][i0] * b2 ** ((n[i0:] - n[i0]) / 2)
        axes[1, j].plot(
            n[i0:], guide, color=tp.MUTED, lw=0.8, ls="--", label=r"$\beta_2^{n/2}$ decay"
        )
        axes[1, j].set_yscale("log")
        axes[2, j].plot(n, z["precond_sharpness"] / z["threshold"], color=tp.INK, lw=1.0)
        axes[2, j].axhline(1.0, color=tp.MUTED, lw=0.8, ls="--")
        axes[2, j].set_yscale("log")
        axes[2, j].set_xlabel("steps")
        for i in range(3):
            axes[i, j].set_xscale("log")
    axes[0, 0].set_ylabel("overlap $a_r$")
    axes[1, 0].set_ylabel("preconditioner")
    axes[2, 0].set_ylabel(r"$\lambda_{\max}(P^{-1}H)$ / AEoS")
    axes[0, -1].legend(fontsize=6, loc="center right")
    axes[1, -1].legend(fontsize=6, loc="lower left")
    fig.align_ylabels(axes[:, 0])
    tp.savefig(fig, FIG / "adam_probe")
    return fig


def entry_count_prediction(cfg: dict) -> np.ndarray:
    """Heuristic balance index of a sign-flow solution per edge: |n_v - n_p| / (n_v + n_p), n = number
    of entries of the core (notes/optimizers.md 1.2 f)."""
    from experiments.optimizers.common import make_problem

    spec = make_problem(cfg).spec
    t = spec.tree
    return np.array(
        [
            abs(spec.sizes[v] - spec.sizes[t.parent[v]]) / (spec.sizes[v] + spec.sizes[t.parent[v]])
            for v in t.edges
        ]
    )


def fig_balance(
    examples: list[tuple[str, dict, list[str]]], scatter: list[tuple[str, dict]], tag=""
):
    """Left panels: conservation-law drift of the library recorder (max_e |Q_e(t) - Q_e(0)| relative) for
    several methods from one init, against the fraction of the run. Right: final balance index per edge
    of Adam / RMSProp runs against the entry-count heuristic."""
    tp.set_style()
    n = len(examples)
    fig, axes = tp.subplots(1, n + 1, width=tp.TEXT_WIDTH, panel_aspect=1.0)
    for ax, (name, flt, methods) in zip(axes[:n], examples):
        df = with_keys(collect(name))
        for k, v in flt.items():
            df = df[df[k] == v]
        for m in methods:
            rows = df[df["method"] == m]
            if rows.empty:
                continue
            d = load(name, rows.iloc[0]["file"]).data
            t = d["t"] / d["t"][-1]
            fam = rows.iloc[0]["optimizer"]
            ax.plot(
                np.maximum(t, 1e-6),
                np.maximum(d["conservation_drift"], 1e-17),
                color=FAMILY_COLOR[fam],
                lw=1.0,
                ls="--" if fam in ("gd", "dmrg") else "-",
                label=m,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("fraction of the run (own clock)")
        r0 = df.iloc[0]
        ax.set_title(
            f"{r0['tree']}{int(r0['n_sites'])} {r0['teacher']}, sigma {r0['sigma']:g}, seed {int(r0['seed'])}",
            fontsize=7,
        )
        ax.legend(fontsize=6)
    axes[0].set_ylabel(r"drift of $Q_e$ (relative)")
    ax = axes[-1]
    markers = ["o", "s", "^", "D"]
    for i, (name, flt) in enumerate(scatter):
        df = collect(name)
        for k, v in flt.items():
            df = df[df[k] == v]
        df = df[df["optimizer"].isin(["adam", "rmsprop"]) & df["converged"]]
        if df.empty:
            continue
        r0 = df.iloc[0]
        cfg0 = {
            "tree": str(r0["tree"]),
            "teacher": str(r0["teacher"]),
            **{k: int(r0[k]) for k in ("n_sites", "d", "chi", "teacher_seed")},
        }
        cfg0["weights"] = json.loads(df.iloc[0]["weights"])
        if cfg0["teacher"] == "pattern":
            cfg0["patterns"] = json.loads(df.iloc[0]["patterns"])
        pred = entry_count_prediction(cfg0)
        meas = np.array([load(name, f).data["balance"][-1] for f in df["file"]])
        ax.errorbar(
            pred,
            meas.mean(0),
            yerr=meas.std(0),
            fmt=markers[i % 4],
            ms=3.5,
            color=tp.palette(4)[i],
            mec="white",
            mew=0.5,
            elinewidth=0.7,
            label=f"{cfg0['tree']}{cfg0['n_sites']} {cfg0['teacher']}",
        )
    ax.plot([0, 1], [0, 1], color=tp.MUTED, lw=0.8, ls="--")
    ax.set_xlim(-0.03, 0.8)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel(r"entry-count heuristic $|n_v-n_p|/(n_v+n_p)$")
    ax.set_ylabel("balance index at the end (Adam, RMSProp)")
    ax.legend(fontsize=6)
    tp.savefig(fig, FIG / f"balance{tag}")
    return fig


def fig_adam_escape(tag=""):
    """Adam's first escape. Left: binary4, sign-flow time lr n_1 vs sigma for several lr, with GF's t_1
    (GF time) for contrast. Right: mps8, steps to the first escape vs sigma for eps = 1e-8 and ~0, with the
    prediction n_1 = (eps / lr) T_GF(sigma) + n_sign: T_GF from each seed's GF escape time at sigma = 0.3
    (deep sweep, balanced init) scaled by (0.3 / sigma)^(K-2), n_sign = the eps ~ 0 median at that lr."""
    tp.set_style()
    fig, axes = tp.subplots(1, 2, width=tp.TEXT_WIDTH, panel_aspect=0.8)
    ax = axes[0]
    df = with_keys(collect("scaling_b4"))
    gf = df[df["optimizer"] == "gf"]
    st = gf.groupby("sigma")["t_first"].agg(["median", "count"])
    ax.plot(
        st.index,
        st["median"],
        "o-",
        color=tp.INK,
        ms=3.5,
        lw=1.0,
        mec="white",
        mew=0.5,
        label=r"gradient flow: $t_1$",
    )
    ad = df[(df["optimizer"] == "adam") & (df["beta2"].isna() | (df["beta2"] == 0.999))]
    lrs = sorted(ad["lr"].unique())
    shade = dict(zip(lrs, tp.ordinal_colors(len(lrs))))  # one shade per step size, both panels
    for lr in lrs:
        g = ad[ad["lr"] == lr]
        st = (g["t_first"] * lr).groupby(g["sigma"]).median()
        ax.plot(
            st.index,
            st.values,
            "s-",
            color=shade[lr],
            ms=3.0,
            lw=1.0,
            mec="white",
            mew=0.5,
            label=rf"Adam, $\eta={_lr_text(lr)}$: $\eta n_1$",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"initialization scale $\sigma$")
    ax.set_ylabel(r"first-escape time ($t_1$ or $\eta n_1$)")
    ax.set_title(r"binary4, odeco teacher ($K=3$)", fontsize=8)
    ax.legend(fontsize=8, loc="center left", handlelength=1.8)
    ax = axes[1]
    eps = with_keys(collect("eps_mps8"))
    deep = collect("deep")
    ref = deep[
        (deep["optimizer"] == "gf")
        & (deep["tree"] == "mps")
        & (deep["teacher"] == "odeco")
        & (deep["init"] == "balanced")
        & (deep["sigma"] == 0.3)
    ].set_index("seed")["t_first"]
    k_minus_2 = 6
    for lr in sorted(eps["lr"].unique()):
        color = shade.get(lr, tp.INK)
        for e, mk, ls in ((1e-8, "o", "-"), (1e-30, "s", ":")):
            g = eps[np.isclose(eps["lr"], lr) & np.isclose(eps["eps"], e, rtol=1e-6, atol=0)]
            if g.empty:
                continue
            t1 = g["t_first"].where(np.isfinite(g["t_first"]), np.nan)
            st = t1.groupby(g["sigma"]).median()
            lower = g[~np.isfinite(g["t_first"])].groupby("sigma")["meta_n_steps"].max()
            ax.plot(
                st.index,
                st.values,
                mk + ls,
                color=color,
                ms=3.0,
                lw=1.0,
                mec="white",
                mew=0.5,
                label=rf"$\eta={_lr_text(lr)}$, $\epsilon={_lr_text(e)}$",
            )
            if len(lower):
                ax.plot(lower.index, lower.values, "^", color=color, ms=4, mfc="none")
            if e == 1e-8 and len(ref):
                n_sign = eps[
                    np.isclose(eps["lr"], lr) & np.isclose(eps["eps"], 1e-30, rtol=1e-6, atol=0)
                ]["t_first"].median()
                sig = np.geomspace(eps["sigma"].min(), eps["sigma"].max(), 60)
                pred = (e / lr) * np.median(ref.values) * (0.3 / sig) ** k_minus_2 + n_sign
                ax.plot(sig, pred, color=color, lw=0.8, ls="--")
    ax.plot([], [], "^", color=tp.INK_SECONDARY, ms=4, mfc="none", label="no escape in budget")
    ax.plot(
        [], [], color=tp.INK_SECONDARY, lw=0.8, ls="--", label=r"prediction, $\epsilon=10^{-8}$"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"initialization scale $\sigma$")
    ax.set_ylabel(r"steps $n_1$ to the first escape")
    ax.set_title(r"MPS8, odeco teacher ($K=8$), Adam", fontsize=8)
    ax.legend(fontsize=8, loc="upper right", handlelength=1.8, labelspacing=0.25)
    tp.savefig(fig, FIG / f"adam_escape{tag}")
    return fig


def fig_implicit_bias(name: str = "implicit_bias_b4"):
    """Round 1 (superseded by fig_implicit_bias2: one data draw per init seed). Finite samples (binary4,
    chi = 4 > teacher rank 3): fraction of runs recovering the teacher (population loss < 1e-2, L0 = 7) and
    fraction ending with a rank-4 edge, against n. All runs are counted, including runs that did not reach
    the training tolerance (GF: the long t_max = 1e6 runs)."""
    df = with_keys(collect(name))
    gf_long = (df["optimizer"] == "gf") & (df["t_max"] == 1e6)
    df = df[(df["optimizer"] != "gf") | gf_long].copy()
    df.loc[gf_long, "method"] = "GF"
    df = df[df["optimizer"] != "momentum"]
    df["recovered"] = df["pop_loss_end"] < 1e-2
    df["full_rank"] = df["eff_ranks_end"].astype(str).str.contains("4")
    tp.set_style()
    fig, axes = tp.subplots(1, 2, width=tp.TEXT_WIDTH, panel_aspect=0.75)
    for m in ["GF", "Adam, lr 0.001", "Adam, lr 0.01", "RMSProp, lr 0.001"]:
        g = df[df["method"] == m]
        if g.empty:
            continue
        fam = g["optimizer"].iloc[0]
        marker = "^" if m == "Adam, lr 0.01" else "o"
        st = g.groupby("n_samples").agg(
            rec=("recovered", "mean"), full=("full_rank", "mean"), n=("seed", "count")
        )
        se_r = np.sqrt(st["rec"] * (1 - st["rec"]) / st["n"])
        se_f = np.sqrt(st["full"] * (1 - st["full"]) / st["n"])
        axes[0].errorbar(
            st.index,
            st["rec"],
            yerr=se_r,
            fmt=marker + "-",
            ms=3.5,
            lw=1.0,
            color=FAMILY_COLOR[fam],
            mec="white",
            mew=0.5,
            label=m,
        )
        axes[1].errorbar(
            st.index,
            st["full"],
            yerr=se_f,
            fmt=marker + "-",
            ms=3.5,
            lw=1.0,
            color=FAMILY_COLOR[fam],
            mec="white",
            mew=0.5,
            label=m,
        )
    for ax in axes:
        ax.axvline(45, color=tp.MUTED, lw=0.8, ls=":")
        ax.set_xlabel("number of samples n")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("teacher recovered (population loss < 0.01)")
    axes[1].set_ylabel("ends with bond rank 4 > teacher rank 3")
    axes[0].legend(fontsize=6)
    axes[0].set_title("binary4, chi = 4, sigma in {0.01, 0.1}, 12 runs per point", fontsize=7)
    axes[1].set_title("dotted: dimension of the rank-(3,3) manifold (45)", fontsize=7)
    tp.savefig(fig, FIG / "implicit_bias_b4")
    return fig


def fig_implicit_bias2(name: str = "implicit_bias2_b4"):
    """Finite samples, several data draws: population loss of Adam vs GF on the same (data draw, init
    seed), runs that reached the training tolerance (empirical loss < 1e-8 L0) only. Panels n = 45, 60."""
    df = with_keys(collect(name))
    df = df[df["meta_stop_reason"].astype(str).eq("converged")]
    key = ["n_samples", "sigma", "data_seed", "seed"]
    gf = df[df["optimizer"] == "gf"].set_index(key)["pop_loss_end"]
    tp.set_style()
    fig, axes = tp.subplots(1, 2, width=tp.TEXT_WIDTH, panel_aspect=0.9)
    for ax, n in zip(axes, (45, 60)):
        for m, mk in (("Adam, lr 0.001", "o"), ("Adam, lr 0.01", "^")):
            g = df[(df["method"] == m) & (df["n_samples"] == n)].set_index(key)["pop_loss_end"]
            common = g.index.intersection(gf.index)
            ax.plot(
                np.maximum(gf.loc[common], 1e-9),
                np.maximum(g.loc[common], 1e-9),
                mk,
                ms=3.5,
                color=FAMILY_COLOR["adam"],
                mec="white",
                mew=0.5,
                alpha=0.8 if mk == "o" else 0.6,
                label=f"{m} ({len(common)} pairs)",
            )
        lim = (1e-9, 1e2)
        ax.plot(lim, lim, color=tp.MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("population loss, GF")
        ax.set_title(
            f"binary4, n = {n}: 4 data draws x 6 inits x sigma in {{0.01, 0.1}}", fontsize=7
        )
        ax.legend(fontsize=6, loc="upper left")
    axes[0].set_ylabel("population loss, Adam (same draw and init)")
    tp.savefig(fig, FIG / "implicit_bias2_b4")
    return fig
