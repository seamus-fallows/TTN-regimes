"""Report figures of Sec. 5.5 (discrete steps, gradient noise, finite samples), from committed CSVs.

Three compact figures at the report's text width in the `ttn.plotting` style. They read only the
summary tables in the output directory (the committed ones with
--out-dir experiments/optim_samples/results), never raw runs:
  report_gd       GD: outcome classes against eta/eta_c on binary N=4 with the scalar map above them,
                  relative error of the first event against eta, and the learning-rate axis across
                  sigma (gd_sweep.csv, map1d_points.csv, centred_chi9_gf.csv)
  report_sgd      online SGD: first escape and second plateau against B, the P9 law on the aligned
                  route, the learned component under noise (sgd_summary.csv,
                  sgd_second_plateau.csv, p9_conditions.csv, p9_fit.csv, p9_sgd.csv)
  report_samples  finite samples: test error against n (chi = 4, 16) and effective bond rank (chi = 4)
                  (samples_summary.csv)
Differences from the step figures (plot_gd.py, plot_regime.py, plot_sgd.py, plot_p9.py,
analyze_samples.py): the end of the catapult window is drawn per sigma from the data (largest tested
step at which some run ends at gradient flow's loss: 1.5 x 2/lambda_GF at sigma = 1, 3 x at sigma >= 2),
not as 3 x 2/lambda_GF everywhere; the random-init second plateau is compared with P9 at the c_gamma
fitted on the aligned route (p9_fit.csv), not at c_gamma = 35; all five sigma of step 1 are pooled in
the class fractions.

Run: uv run python experiments/optim_samples/plot_report.py --out-dir experiments/optim_samples/results
Writes report_gd.pdf/.png, report_sgd.pdf/.png and report_samples.pdf/.png there.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS as RES
from ttn import plotting as tp

# Outcome classes of analyze_gd.py that occur in the plotted runs ("other" never does: checked below).
CLASSES = [
    ("gf_like", "GF-like"),
    ("catapult", "catapult"),
    ("oscillating", "bounded oscillation"),
    ("reentry", "thrown back to the origin"),
    ("diverged", "diverged"),
]
CLASS_COLOR = tp.regime_colors([c for c, _ in CLASSES])  # any two classes can touch
LAM_BAL = 3 * 3.0 ** (4 / 3)  # K s_1^(2-2/K): binary N=4 (K = 3), s_1 = 3
ETA_C = 2 / LAM_BAL


def step1_binary4(df):
    """Step-1 GD runs on binary N=4: plain model, chi = 3, eta relative to eta_c, seeds 0-4."""
    return df[
        (df.tree == "binary")
        & (df.n == 4)
        & ~df.centred
        & (df.chi == 3)
        & (df.ratio_ref == "teacher")
        & df.root_scale.isna()
        & (df.seed <= 4)
    ]


def fig_gd():
    df = pd.read_csv(RES / "gd_sweep.csv")
    b4 = step1_binary4(df)
    chi9 = df[(df.chi == 9) & (df.ratio_ref == "gf")]
    unknown = set(b4.cls) | set(chi9.cls)
    assert unknown <= {c for c, _ in CLASSES}, unknown
    fig = plt.figure(figsize=(tp.TEXT_WIDTH, 2.75))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.45], width_ratios=[1.0, 1.0, 1.08])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_cls = fig.add_subplot(gs[1, 0], sharex=ax_map)
    ax_err = fig.add_subplot(gs[:, 1])
    ax_lr = fig.add_subplot(gs[:, 2])

    # (a) the balanced scalar map (K = 3) above the outcome classes of the full network
    att = pd.read_csv(RES / "map1d_points.csv")
    a = att[(att.K == 3) & (att.ratio <= 3.0)]
    ax_map.plot(a.ratio, a.x, ".", ms=0.6, color=tp.INK_SECONDARY, alpha=0.5, rasterized=True)
    ax_map.set(ylim=(-0.15, 1.35), yticks=[0, 1], ylabel=r"$q/s_1^{1/K}$")
    ax_map.set_title(r"(a) scalar map ($K=3$), binary $N=4$", fontsize=8.5)
    ax_map.tick_params(labelbottom=False)
    frac = b4.groupby("ratio").cls.value_counts(normalize=True).unstack(fill_value=0)
    bottom = np.zeros(len(frac))
    for cls, lab in CLASSES:
        v = frac[cls].to_numpy() if cls in frac else np.zeros(len(frac))
        ax_cls.fill_between(
            frac.index,
            bottom,
            bottom + v,
            step="mid",
            color=CLASS_COLOR[cls],
            lw=0.4,
            edgecolor="white",
            label=lab,
        )
        bottom += v
    ax_cls.plot(
        frac.index, np.full(len(frac), -0.04), "|", color=tp.INK_SECONDARY, ms=3, clip_on=False
    )
    for ax in (ax_map, ax_cls):
        ax.axvline(1.0, color=tp.INK, lw=0.6, ls="--")
    ax_cls.set(xlim=(0, 3.05), ylim=(0, 1), xlabel=r"$\eta/\eta_c$", ylabel="fraction of runs")
    ax_cls.grid(False)

    # (b) relative error of the first event: mean +- s.e. over 5 seeds; lines: eta c_1 / t_1
    sig = sorted(b4.sigma.unique())
    colors = tp.ordinal_colors(len(sig))
    for col, s in zip(colors, sig):
        g = b4[(b4.sigma == s) & (b4.cls == "gf_like")]
        m = g.groupby("ratio").rel_err1.agg(["mean", "sem"])
        pred = g.assign(p=g.eta * g.pred_shift_over_eta1 / g.t1_gf).groupby("ratio").p.mean()
        ax_err.plot(pred.index, pred.values, "-", color=col, lw=1)
        ax_err.errorbar(
            m.index,
            m["mean"],
            yerr=m["sem"],
            fmt="o",
            ms=3,
            color=col,
            lw=0.8,
            label=rf"$\sigma={s:g}$",
        )
    ax_err.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\eta/\eta_c$",
        ylabel=r"$(t_1^{\rm GD}-t_1^{\rm GF})/t_1^{\rm GF}$",
    )
    ax_err.set_title("(b) first event, binary $N=4$", fontsize=8.5)
    ax_err.legend(loc="upper left", fontsize=8, handletextpad=0.2)

    # (c) learning-rate axis: every run in the (sigma, eta) plane, coloured by class
    rng = np.random.default_rng(0)
    plain = pd.concat([b4, chi9[~chi9.centred]])
    for sub, mk in ((plain, "o"), (chi9[chi9.centred], "s")):
        for cls, _ in CLASSES:
            g = sub[sub.cls == cls]
            ax_lr.scatter(
                g.sigma * np.exp(rng.uniform(-0.08, 0.08, len(g))),
                g.eta,
                marker=mk,
                s=5,
                color=CLASS_COLOR[cls],
                lw=0,
                alpha=0.8,
                rasterized=True,
            )
    gf = pd.read_csv(RES / "centred_chi9_gf.csv")
    lam = gf[gf.centred].groupby("sigma").lam_gf.mean()
    ax_lr.axhline(ETA_C, color=tp.INK, lw=0.8, ls=":", label=r"$\eta_c$")
    ax_lr.plot(
        lam.index, 2 / lam.values, "-", color=tp.INK, lw=1, label=r"$2/\lambda_{\rm GF}(\sigma)$"
    )
    ends = []  # end of the catapult window: largest ratio with a run at GF's loss (GF-like/catapult)
    for s, g in chi9[chi9.centred].groupby("sigma"):
        ok = g[g.cls.isin(["gf_like", "catapult"])]
        ends.append((s, ok.ratio.max() * 2 / lam.loc[s]))
    ends = np.array(ends)
    ax_lr.plot(ends[:, 0], ends[:, 1], "--", color=tp.INK, lw=0.8, label="catapults end")
    ax_lr.scatter([], [], marker="o", s=9, color=tp.INK_SECONDARY, lw=0, label="plain model")
    ax_lr.scatter([], [], marker="s", s=9, color=tp.INK_SECONDARY, lw=0, label="centred, $\\chi=9$")
    ax_lr.legend(loc="lower left", fontsize=8, handlelength=1.6, borderaxespad=0.2)
    ax_lr.set(
        xscale="log",
        yscale="log",
        xlim=(0.002, 20),
        ylim=(8e-6, 1.0),
        xlabel=r"$\sigma$ (per core)",
        ylabel=r"step $\eta$",
    )
    ax_lr.set_title("(c) learning-rate axis, binary $N=4$", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c, _ in CLASSES]
    fig.legend(
        handles,
        [lab for _, lab in CLASSES],
        loc="outside lower center",
        ncol=5,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.2,
    )
    tp.savefig(fig, RES / "report_gd")
    plt.close(fig)


def p9_law(x):
    """T_SGD / T_GF = -ln(1 - x) / x for x < 1 (theory note P9); trapped (inf) for x >= 1."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.inf)
    ok = x < 1
    out[ok] = -np.log1p(-x[ok]) / np.maximum(x[ok], 1e-12)
    return out


def fig_sgd():
    fits = pd.read_csv(RES / "p9_fit.csv").set_index("law").c_gamma
    c_p9 = fits["P9"]
    fig, (ax_b, ax_p9, ax_p10) = plt.subplots(
        1, 3, figsize=(tp.TEXT_WIDTH, 2.35), width_ratios=[1.0, 1.05, 1.0]
    )

    # (a) online SGD from fan-in init, eta = 5e-4: first escape and second plateau against B
    s = pd.read_csv(RES / "sgd_summary.csv")
    b = s[(s.block == "B_sweep") & (s.init == "fanin")]
    sig = sorted(b.sigma.unique())
    colors = dict(zip(sig, tp.ordinal_colors(len(sig))))
    for sg in sig:
        g = b[b.sigma == sg].assign(rel=lambda d: d.shift1 / d.t_esc_gf).groupby("B").rel
        ax_b.errorbar(
            g.mean().index,
            g.mean(),
            yerr=g.sem(),
            fmt="o-",
            ms=3,
            lw=1,
            color=colors[sg],
            label=rf"$\sigma={sg:g}$",
        )
    sec = pd.read_csv(RES / "sgd_second_plateau.csv")
    sec = sec[(sec.block == "B_sweep") & ~sec.near_tie & (sec.n_inits >= 3)]
    for sg, g in sec.groupby("sigma"):
        g = g.sort_values("B")
        ax_b.errorbar(
            g.B,
            g.T_ratio - 1,
            yerr=g.T_ratio_se_inits,
            fmt="s-",
            ms=3.5,
            lw=1,
            mfc="white",
            color=colors[sg],
        )
        ax_b.plot(g.B, p9_law(c_p9 * g.eta / g.B * g.T_gf) - 1, ":", color=colors[sg], lw=1.2)
    ax_b.axhline(0, color=tp.MUTED, lw=0.6)
    ax_b.plot([], [], "o", ms=3, color=tp.INK_SECONDARY, label="first escape")
    ax_b.plot([], [], "s", ms=3.5, mfc="white", color=tp.INK_SECONDARY, label="second plateau")
    ax_b.plot(
        # the report calls the collapse constant c_col (gamma is reserved for the order exponent)
        [],
        [],
        ":",
        color=tp.INK_SECONDARY,
        lw=1.2,
        label=rf"$-\ln(1-x)/x$, $c_{{\rm col}}={c_p9:.1f}$",
    )
    ax_b.set(
        xscale="log",
        xlabel=r"batch size $B$ ($\eta=5\cdot10^{-4}$)",
        ylabel="relative change vs. GF",
    )
    ax_b.set_title("(a) first escape, second plateau", fontsize=8.5)
    ax_b.legend(loc="upper right", fontsize=8, handlelength=1.5, labelspacing=0.2)

    # (b) aligned route start: T_SGD / T_GF against z = (eta/B) T_GF; laws with one fitted c each
    cond = pd.read_csv(RES / "p9_conditions.csv")
    b0s = sorted(cond.b0.unique(), reverse=True)  # T_GF grows as b0 falls: light -> dark
    bcol = dict(zip(b0s, tp.ordinal_colors(len(b0s))))
    zz = np.geomspace(1e-3, 0.08, 400)
    laws = [  # (name in p9_fit.csv, legend formula, line style, law)
        ("P9", r"$-\ln(1-x)/x$", "-", p9_law),
        (
            "pole",
            r"$1/(1-x)$",
            "--",
            lambda x: np.where(x < 1, 1 / (1 - np.minimum(x, 0.999)), np.inf),
        ),
        ("linear", r"$1+x/2$", ":", lambda x: 1 + x / 2),
    ]
    for name, formula, ls, law in laws:
        y = law(fits[name] * zz)
        ok = np.isfinite(y) & (y < 6)
        ax_p9.plot(
            zz[ok],
            y[ok],
            ls,
            color=tp.MUTED,
            lw=1,
            label=rf"{formula}, $c_{{\rm col}}={fits[name]:.1f}$",
        )
    for (b0, bb), g in cond.groupby(["b0", "B"]):
        e = g[g.escaped == 1]
        ax_p9.errorbar(
            e.z,
            e.ratio,
            yerr=e.ratio_se,
            fmt="o" if bb == 64 else "s",
            ms=3.5,
            lw=1,
            color=bcol[b0],
            mfc=bcol[b0] if bb == 64 else "white",
        )
        t = g[g.escaped == 0]  # every run trapped: drawn at the top of the panel
        ax_p9.plot(t.z, np.full(len(t), 5.55), "v", ms=4.5, color=bcol[b0], mfc="white")
    ax_p9.set(
        xscale="log",
        ylim=(0.95, 5.9),
        xlabel=r"$(\eta/B)\,T_{\rm GF}$",
        ylabel=r"$T_{\rm SGD}/T_{\rm GF}$ (escaped runs)",
    )
    ax_p9.set_title("(b) later plateau, aligned route", fontsize=8.5)
    ax_p9.legend(loc="center left", bbox_to_anchor=(0.0, 0.6), fontsize=8, handlelength=1.6)

    # (c) the learned component during those runs: minimum of a_1 per run
    p = pd.read_csv(RES / "p9_sgd.csv")
    rng = np.random.default_rng(1)
    for b0 in b0s:
        g = p[p.b0 == b0]
        ax_p10.scatter(
            g.eta_over_eta_ms * np.exp(rng.uniform(-0.05, 0.05, len(g))),
            g.min_a1,
            s=7,
            color=bcol[b0],
            lw=0,
            label=rf"$b_0={b0:g}$",
        )
    ax_p10.axhline(0.5, color=tp.INK, lw=0.7, ls="--")
    ax_p10.set_xscale("log")
    ax_p10.set_xticks([0.03, 0.06, 0.12, 0.25], ["0.03", "0.06", "0.12", "0.25"])
    ax_p10.minorticks_off()
    ax_p10.set(xlabel=r"$\eta/\eta^*_{\rm ms}(B)$", ylabel=r"$\min_t a_1(t)$")
    ax_p10.set_title("(c) learned component", fontsize=8.5)
    ax_p10.legend(loc="lower left", fontsize=8, handletextpad=0.1, labelspacing=0.2)
    tp.savefig(fig, RES / "report_sgd")
    plt.close(fig)


SAMPLE_SERIES = [  # label in samples_summary.csv, legend text; colours in this fixed order
    ("small init, σ = 0.3 (GD)", r"small init, $\sigma=0.3$"),
    ("small init, σ = 0.5 (GD)", r"small init, $\sigma=0.5$"),
    ("centred, σ = 1 (GD)", r"centred, $\sigma=1$"),
    ("lazy limit (kernel regression, exact)", "lazy limit (exact)"),
]


def fig_samples():
    s = pd.read_csv(RES / "samples_summary.csv")
    color = dict(zip([lab for lab, _ in SAMPLE_SERIES], tp.palette(len(SAMPLE_SERIES))))
    fig, axes = plt.subplots(1, 3, figsize=(tp.TEXT_WIDTH, 2.3))
    floor = 1e-6
    panels = [
        ("N=8, d=2, χ=4", "test_mean", r"(a) test error, $\chi=4$"),
        ("N=8, d=2, χ=16", "test_mean", r"(b) test error, $\chi=16$"),
        ("N=8, d=2, χ=4", "eff_rank", r"(c) effective bond rank, $\chi=4$"),
    ]
    for ax, (setup, col, title) in zip(axes, panels):
        sub = s[s.setup == setup]
        for lab, _ in SAMPLE_SERIES:
            g = sub[sub.label == lab].sort_values("n")
            if g.empty:
                continue
            if col == "test_mean":
                y = g.test_mean.clip(lower=floor)
                err = np.where(g.test_mean > floor, g.test_se, 0)
            else:
                y, err = g.eff_rank, g.eff_rank_se
            ax.errorbar(g.n, y, yerr=err, fmt="o-", ms=3, lw=1, color=color[lab])
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 32, 64, 128, 256, 512], ["16", "32", "64", "128", "256", "512"])
        ax.minorticks_off()
        ax.set(xlabel=r"training-set size $n$")
        ax.set_title(title, fontsize=8.5)
        if col == "test_mean":
            ax.axhline(1, color=tp.MUTED, lw=0.7)
            ax.set(yscale="log", ylim=(floor / 2, 300), ylabel=r"$\|W-W^*\|_F^2/\|W^*\|_F^2$")
            marks = [(28, "28"), (112, "112")] if "χ=4" in setup else [(28, "28"), (256, "256")]
            for x, txt in marks:
                ax.axvline(x, color=tp.MUTED, lw=0.7, ls="--")
                ax.text(x * 1.06, 60, txt, fontsize=8, color=tp.INK_SECONDARY)
        else:
            ax.axhline(2, color=tp.MUTED, lw=0.7)
            ax.set(ylabel="mean effective bond rank")
    axes[1].text(
        230,
        1e-6,
        r"exact ($<10^{-20}$)",
        fontsize=8,
        color=tp.INK_SECONDARY,
        ha="right",
        va="center",
    )
    handles = [
        plt.Line2D([], [], color=color[lab], marker="o", ms=3, lw=1) for lab, _ in SAMPLE_SERIES
    ]
    fig.legend(
        handles, [t for _, t in SAMPLE_SERIES], loc="outside lower center", ncol=4, fontsize=8
    )
    tp.savefig(fig, RES / "report_samples")
    plt.close(fig)


def main():
    tp.set_style()
    fig_gd()
    fig_sgd()
    fig_samples()


if __name__ == "__main__":
    main()
