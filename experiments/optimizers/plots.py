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
    order = [("binary", 4, "adam", 1e-3, 0.9), ("binary", 4, "adam", 1e-3, 0.999),
             ("binary", 8, "adam", 1e-3, None), ("mps", 8, "adam", 1e-3, None)]
    picked = []
    for key in order:
        for cfg, z in runs:
            if (cfg["tree"], cfg["n_sites"], cfg["optimizer"], cfg["lr"], cfg.get("beta2")) == key:
                picked.append((cfg, z))
    tp.set_style()
    fig, axes = tp.subplots(3, len(picked), width=tp.TEXT_WIDTH, panel_aspect=0.85, sharex="col",
                            squeeze=False)
    colors = tp.palette(3)
    for j, (cfg, z) in enumerate(picked):
        b2 = cfg.get("beta2", 0.999)
        teacher = "odeco" if cfg["teacher"] == "odeco" else "pattern E8"
        axes[0, j].set_title(f"{cfg['tree']}{cfg['n_sites']} {teacher}\nAdam lr {cfg['lr']:g}, b2 {b2:g}",
                             fontsize=7)
        t, a = z["fine_t"], z["fine_overlaps"]
        for r in range(a.shape[1]):
            axes[0, j].plot(np.maximum(t, 1), a[:, r], color=colors[r], lw=1.0, label=f"$a_{r + 1}$")
        axes[0, j].set_ylim(-0.1, 1.15)
        n = z["step"]
        axes[1, j].plot(n, z["p_median"], color=tp.INK, lw=1.0, label=r"median $\sqrt{\hat v}$")
        i0 = int(np.argmax(z["p_median"]))
        guide = z["p_median"][i0] * b2 ** ((n[i0:] - n[i0]) / 2)
        axes[1, j].plot(n[i0:], guide, color=tp.MUTED, lw=0.8, ls="--", label=r"$\beta_2^{n/2}$ decay")
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
    return np.array([abs(spec.sizes[v] - spec.sizes[t.parent[v]]) / (spec.sizes[v] + spec.sizes[t.parent[v]])
                     for v in t.edges])


def fig_balance(examples: list[tuple[str, dict, list[str]]], scatter: list[tuple[str, dict]], tag=""):
    """Left panels: conservation-law drift of the library recorder (max_e |Q_e(t) - Q_e(0)| relative) for
    several methods from one init, against the fraction of the run. Right: final balance index per edge
    of Adam / RMSProp runs against the entry-count heuristic."""
    tp.set_style()
    n = len(examples)
    fig, axes = tp.subplots(1, n + 1, width=tp.TEXT_WIDTH, panel_aspect=0.9)
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
            ax.plot(np.maximum(t, 1e-6), np.maximum(d["conservation_drift"], 1e-17), color=FAMILY_COLOR[fam],
                    lw=1.0, ls="--" if fam in ("gd", "dmrg") else "-", label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("fraction of the run (own clock)")
        ax.set_title(f"{flt.get('tree', '')}{rows.iloc[0]['n_sites'] if not rows.empty else ''}, "
                     f"sigma {flt.get('sigma', '')}", fontsize=7)
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
        cfg0 = {"tree": str(r0["tree"]), "teacher": str(r0["teacher"]), **{k: int(r0[k]) for k in
                ("n_sites", "d", "chi", "teacher_seed")}}
        cfg0["weights"] = json.loads(df.iloc[0]["weights"])
        if cfg0["teacher"] == "pattern":
            cfg0["patterns"] = json.loads(df.iloc[0]["patterns"])
        pred = entry_count_prediction(cfg0)
        meas = np.array([load(name, f).data["balance"][-1] for f in df["file"]])
        ax.errorbar(pred, meas.mean(0), yerr=meas.std(0), fmt=markers[i % 4], ms=3.5, color=tp.palette(4)[i],
                    mec="white", mew=0.5, elinewidth=0.7, label=f"{cfg0['tree']}{cfg0['n_sites']} {cfg0['teacher']}")
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
    ax.plot(st.index, st["median"], "o-", color=FAMILY_COLOR["gf"], ms=3.5, lw=1.0, mec="white", mew=0.5,
            label="GF: $t_1$ (GF time)")
    ad = df[(df["optimizer"] == "adam") & (df["beta2"].isna() | (df["beta2"] == 0.999))]
    shades = tp.ordinal_colors(4)
    for c, lr in zip(shades, sorted(ad["lr"].unique())):
        g = ad[ad["lr"] == lr]
        st = (g["t_first"] * lr).groupby(g["sigma"]).median()
        ax.plot(st.index, st.values, "s-", color=c, ms=3.0, lw=1.0, mec="white", mew=0.5,
                label=f"Adam lr {lr:g}: lr $n_1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"init scale $\sigma$")
    ax.set_ylabel("first-escape time")
    ax.set_title("binary4 odeco (K = 3)", fontsize=7)
    ax.legend(fontsize=6)
    ax = axes[1]
    eps = with_keys(collect("eps_mps8"))
    deep = collect("deep")
    ref = deep[(deep["optimizer"] == "gf") & (deep["tree"] == "mps") & (deep["teacher"] == "odeco")
               & (deep["init"] == "balanced") & (deep["sigma"] == 0.3)].set_index("seed")["t_first"]
    k_minus_2 = 6
    colors = {1e-3: tp.palette(2)[1], 1e-2: tp.palette(7)[6]}
    for lr in sorted(eps["lr"].unique()):
        for e, mk, ls in ((1e-8, "o", "-"), (1e-30, "s", ":")):
            g = eps[(eps["lr"] == lr) & (eps["eps"] == e)]
            if g.empty:
                continue
            t1 = g["t_first"].where(np.isfinite(g["t_first"]), np.nan)
            st = t1.groupby(g["sigma"]).median()
            lower = g[~np.isfinite(g["t_first"])].groupby("sigma")["meta_n_steps"].max()
            ax.plot(st.index, st.values, mk + ls, color=colors.get(lr, tp.INK), ms=3.0, lw=1.0, mec="white",
                    mew=0.5, label=f"Adam lr {lr:g}, eps {e:g}")
            if len(lower):
                ax.plot(lower.index, lower.values, "^", color=colors.get(lr, tp.INK), ms=4, mfc="none",
                        label=f"  (no escape within budget)")
            if e == 1e-8 and len(ref):
                n_sign = eps[(eps["lr"] == lr) & (eps["eps"] == 1e-30)]["t_first"].median()
                sig = np.geomspace(eps["sigma"].min(), eps["sigma"].max(), 60)
                pred = (e / lr) * np.median(ref.values) * (0.3 / sig) ** k_minus_2 + n_sign
                ax.plot(sig, pred, color=colors.get(lr, tp.INK), lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"init scale $\sigma$")
    ax.set_ylabel("steps to the first escape")
    ax.set_title("mps8 odeco (K = 8); dashed: eps prediction", fontsize=7)
    ax.legend(fontsize=5.5)
    tp.savefig(fig, FIG / f"adam_escape{tag}")
    return fig
