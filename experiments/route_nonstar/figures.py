"""Draft figures for experiment route_nonstar (from results/*.csv and, for the example
trajectories, the raw records).

usage: uv run python experiments/route_nonstar/figures.py [--raw DIR] [--res DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import read  # noqa: E402  # noqa: E402

from ttn.plotting import MUTED, palette, savefig, set_style  # noqa: E402

HERE = Path(__file__).resolve().parent
BIG = 12


def style():
    set_style("sans")
    matplotlib.rcParams.update({"font.size": BIG, "axes.titlesize": BIG, "axes.labelsize": BIG,
                                "xtick.labelsize": BIG - 1, "ytick.labelsize": BIG - 1,
                                "legend.fontsize": BIG - 1, "figure.titlesize": BIG + 1,
                                "lines.linewidth": 2.0})


def example_trajectories(raw: Path, res: Path, out: Path):
    """One run per target: amplitudes of the target patterns and of the largest off-target string."""
    cases = [("R3", -2.0, 0, "the recombinant 101 appears and stays for a long plateau", False),
             ("NC_ht4", -2.0, 0, "plateaus 2 and 3 carry 0010 / 0011", False),
             ("TD_binary8", -2.0, 0, "both edits learned at once, with their recombinant", False),
             ("CTL1_mps6", -2.0, 0, "star control: only partial targets (log time)", True)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (name, ls, seed, msg, logx) in zip(axes.flat, cases):
        stem = str(raw / "raw" / name / f"s{ls:+.2f}_seed{seed}")
        if not Path(stem + ".json").exists():
            ax.set_visible(False)
            continue
        rec = json.loads(Path(stem + ".json").read_text())
        z = np.load(stem + ".npz")
        t, P, W, keep = z["t"], z["P"], z["W"], z["keep"]
        drops = [d for d in rec["drops"] if d]
        t0 = drops[0]["t_half"] if drops else 0.0
        x = t - t0
        ok = x > (0.1 if logx else -0.02 * x[-1])
        pats = [p for p in rec["crossings"]]
        weights = None
        tgt = read("runs", res)  # noqa: F841  (presence check)
        cfg = __import__("yaml").safe_load((HERE / "configs" / "targets.yaml").read_text())
        weights = {t_["name"]: t_["weights"] for t_ in cfg["targets"]}[name]
        cols = palette(len(pats))
        for a, p in enumerate(pats):
            ax.plot(x[ok], P[ok, a] / weights[a], color=cols[a], label=f"{p} (weight {weights[a]:g})")
        # the off-target string with the largest |W| on any plateau
        best, amp = None, 0.0
        for s in rec["segments"]:
            for o in (s or {}).get("off_strings", []) or []:
                if abs(o["amp"]) > amp:
                    best, amp = o["string"], abs(o["amp"])
        if best is not None:
            j = int(best, 2)
            xk = x[keep]
            okk = xk > (0.1 if logx else -0.02 * x[-1])
            ax.plot(xk[okk], np.abs(W[okk, j]) / max(weights), color="black", ls="--", lw=2.2,
                    label=f"{best}: not in the target (|amplitude|)")
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.25)
        ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("time after the first escape (gradient-flow time units"
                      + (", log scale)" if logx else ")"))
        ax.set_ylabel("amplitude / its target weight")
        ax.set_title(f"{name} ({rec['tree']}, sigma = 10^{ls:g}):\n{msg}", fontsize=BIG - 1)
        ax.legend(loc="center right", fontsize=BIG - 3, frameon=True, facecolor="white",
                  framealpha=0.9, edgecolor="none")
    fig.suptitle("On targets that break the star condition, training stalls at points that "
                 "contain a string outside the target (dashed)")
    savefig(fig, out / "fig1_example_trajectories")
    plt.close(fig)


def plateau_identity(res: Path, out: Path):
    tr = read("transitions", res)
    tr = tr[(tr.segment >= 1) & tr.d_near.notna() & (tr.log10_sigma <= -1.5)]
    tr["recomb"] = tr.off_recombinants.fillna("").str.len() > 0
    g = tr.groupby(["target", "family", "key"]).agg(d=("d_near", "median"), n=("seed", "size"),
                                                    recomb=("recomb", "mean")).reset_index()
    g = g[g.n >= 20]
    fam_order = ["recombinant", "nested", "two_edits", "random", "control"]
    targets = (tr[["target", "family"]].drop_duplicates()
               .assign(f=lambda x: x.family.map(fam_order.index)).sort_values(["f", "target"]))
    order = list(targets.target)
    labels = [f"{t} ({f})" for t, f in zip(targets.target, targets.family)]
    cols = palette(2)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(order) + 2.5))
    floor = 1e-8
    for _, r in g.iterrows():
        y = order.index(r.target)
        c = cols[1] if r.recomb > 0.5 else cols[0]
        ax.scatter(max(r.d, floor), y, s=90, color=c, edgecolor="black", lw=0.5, zorder=3,
                   marker="D" if r.recomb > 0.5 else "o")
    ax.axvline(0.03, color="black", ls="--", lw=1.5)
    ax.text(0.035, -0.9, "partial-target point\nif left of the dashed line (< 0.03)",
            fontsize=BIG - 2, va="top")
    ax.set_yticks(range(len(order)), labels)
    ax.set_ylim(len(order) - 0.5, -2.2)
    ax.set_xscale("log")
    ax.set_xlim(floor / 2, 1)
    ax.set_xlabel("distance from the plateau to the nearest partial target\n"
                  "(relative to |W*|; median over seeds; values below 1e-8 drawn at 1e-8)")
    ax.scatter([], [], color=cols[0], edgecolor="black", lw=0.5, s=90, marker="o",
               label="plateau without off-target strings")
    ax.scatter([], [], color=cols[1], edgecolor="black", lw=0.5, s=90, marker="D",
               label="plateau carrying a recombinant string")
    ax.legend(loc="upper left", fontsize=BIG - 1)
    ax.set_title("After the first escape, plateaus of non-star targets are often not partial "
                 "targets:\nthey carry a recombinant string (diamonds) or, for R3d / R4d, wrong "
                 "amplitudes;\nstar controls always sit on partial targets "
                 "(one marker per target and plateau type)")
    savefig(fig, out / "fig2_plateau_identity")
    plt.close(fig)


def exponents(res: Path, out: Path):
    s0 = read("fit_summary", res)
    s0 = s0[s0.key != "origin"]
    s = s0[s0.n_fitted >= 5].copy()
    # secondary (deviation 2): keys fitted only with floor-flagged sigmas included
    s2 = s0[(s0.n_fitted < 5) & (s0.n_fitted_with_flagged.fillna(0) >= 5)].copy()
    s2["median"], s2["q25"], s2["q75"] = (s2.median_with_flagged, s2.q25_with_flagged,
                                          s2.q75_with_flagged)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), sharey=True)
    names = [("paper", "paper's route depth c*_J - 2\n(applied outside its scope)"),
             ("route", "numerical route rule k* - 2\n(at the measured plateau point)"),
             ("pair", "pairwise rule: min Steiner cost - 2")]
    cols = palette(2)
    rng = np.random.default_rng(0)
    for ax, (p, label) in zip(axes, names):
        for df, hollow in [(s, False), (s2, True)]:
            star = df.target.str.startswith(("CTL", "B4")).values
            x = df[f"exp_{p}"].values + rng.uniform(-0.07, 0.07, len(df))
            y = df["median"].values
            err = np.vstack([y - df.q25.values, df.q75.values - y])
            for mask, c, lab in [(~star, cols[1], "targets breaking the star condition"),
                                 (star, cols[0], "star-satisfying controls")]:
                if not mask.any():
                    continue
                ax.errorbar(x[mask], y[mask], yerr=err[:, mask], fmt="D" if hollow else "o",
                            color=c, mfc="white" if hollow else c, ms=8, mec=c if hollow else
                            "black", mew=1.5 if hollow else 0.5, capsize=3,
                            label=None if hollow else lab)
            if p == "route" and not hollow:
                groups = {}
                for xi, yi, (_, r) in zip(x, y, df.iterrows()):
                    if abs(yi - r[f"exp_{p}"]) > 0.3:
                        groups.setdefault((round(r[f"exp_{p}"]), round(yi, 1)), []).append(
                            (xi, yi, r.target))
                for (ex, _), pts in groups.items():
                    names_ = ", ".join(sorted({t for _, _, t in pts}))
                    ax.annotate(names_, (pts[0][0], pts[0][1]), xytext=(10, -18),
                                textcoords="offset points", fontsize=BIG - 3)
        lim = [-0.5, 5.5]
        ax.plot(lim, lim, color="black", lw=1.5, label="measured = predicted")
        ax.fill_between(lim, [lim[0] - 0.3, lim[1] - 0.3], [lim[0] + 0.3, lim[1] + 0.3],
                        color=MUTED, alpha=0.2, lw=0, label="within 0.3")
        ax.set_xlim(lim)
        ax.set_ylim(-0.7, 5.5)
        ax.set_xlabel("predicted exponent: " + label)
        ax.set_title({"paper": "Paper's route depth", "route": "Numerical route rule",
                      "pair": "Pairwise rule"}[p])
    axes[0].errorbar([], [], fmt="D", color="black", mfc="white", ms=8, mew=1.5,
                     label="fit incl. floor-flagged sigmas (secondary)")
    axes[0].set_ylabel("measured exponent of plateau length vs sigma\n"
                       "(median over seeds, bars = quartiles)")
    axes[0].legend(loc="upper left", fontsize=BIG - 3)
    axes[0].text(2.3, -0.45, "predicted 0 = log-time plateau\nor no plateau at all", fontsize=BIG - 3)
    fig.suptitle("Plateau length ~ sigma^-exponent: the numerical route rule predicts the exponent "
                 "on non-star targets;\nthe paper's route depth and the pairwise rule predict no "
                 "plateau (0) where one exists (one point per target and plateau type)")
    savefig(fig, out / "fig3_exponents_vs_predictors")
    plt.close(fig)


def effective_depth(res: Path, out: Path):
    tr = read("transitions", res)
    tr = tr[(tr.segment >= 1) & tr.next_eff_depth_loss.notna() & (tr.log10_sigma <= -1.5)
            & (tr.floor_flag == False)]  # noqa: E712
    g = tr.groupby(["target", "key"]).agg(eff=("next_eff_depth_loss", "median"),
                                          q25=("next_eff_depth_loss", lambda v: v.quantile(.25)),
                                          q75=("next_eff_depth_loss", lambda v: v.quantile(.75)),
                                          k=("route_k", "median"), c=("pred_paper", "median"),
                                          n=("seed", "size")).reset_index()
    g = g[g.n >= 20]
    g["c"] = g.c.fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    cols = palette(2)
    for ax, (col, lab) in zip(axes, [("c", "paper's route depth c*_J (0 = nothing missing)"),
                                     ("k", "route rule k* at the plateau point")]):
        x = g[col].values + np.random.default_rng(1).uniform(-0.08, 0.08, len(g))
        y = g.eff.values
        star = g.target.str.startswith(("CTL", "B4")).values
        err = np.vstack([y - g.q25, g.q75 - y])
        for m, c, lab in [(~star, cols[1], "targets breaking the star condition"),
                          (star, cols[0], "star-satisfying controls")]:
            ax.errorbar(x[m], y[m], yerr=err[:, m], fmt="o", color=c, ms=7, mec="black",
                        mew=0.5, capsize=3, label=lab)
        ax.plot([0, 7], [0, 7], color="black", lw=1.5, label="effective depth = predicted")
        ax.text(7.3, 0.3, "one marker = one target and plateau type\n(median over seeds, bars = quartiles)",
                ha="right", fontsize=BIG - 3)
        ax.set_xlabel(lab)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(0, 7.5)
    axes[0].set_ylabel("effective depth of the escape\n(slope of the loss growth law)")
    axes[0].legend(loc="upper left", fontsize=BIG - 2)
    fig.suptitle("The escape from a plateau behaves like a route of depth k*, not of the "
                 "paper's depth")
    savefig(fig, out / "fig4_effective_depth")
    plt.close(fig)


def escape_type(res: Path, out: Path):
    tr = read("transitions", res)
    tr = tr[(tr.segment >= 1) & tr.lam_min.notna() & (tr.floor_flag == False)]  # noqa: E712
    et = read("escape_type", res)
    fs = read("fit_summary", res)[["target", "key", "route_k_median"]]
    et = et.merge(fs, on=["target", "key"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    cols = palette(6)
    picks = {"strict": et[et.type == "strict"].sort_values("n_points", ascending=False).head(4),
             "degenerate": pd.concat([et[(et.type == "degenerate") & (et.route_k_median == k)]
                                      .sort_values("n_points", ascending=False).head(2)
                                      for k in (3, 4, 5)])}
    for ax, kind in zip(axes, ["strict", "degenerate"]):
        for i, (_, r) in enumerate(picks[kind].iterrows()):
            g = tr[(tr.target == r.target) & (tr.key == r.key)]
            m = g.groupby("log10_sigma").lam_min.median()
            lab = f"{r.target}: {r.key.split('@')[0][:30]} (k* = {int(r.route_k_median)})"
            ax.plot(10.0**m.index.values, np.abs(m.values), "o-", color=cols[i], label=lab)
        if kind == "degenerate":
            xs = np.array([1e-4, 1e-1])
            for k, c0 in [(3, 3e-1), (4, 3e-2), (5, 3e-3)]:
                ax.plot(xs, c0 * (xs / 0.1) ** (k - 2), ls="--", color=MUTED, lw=1.2)
                ax.text(1.2e-1, c0, f"slope {k - 2}\n(k* = {k})", fontsize=BIG - 3,
                        color=MUTED, va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("initialization scale sigma")
        ax.set_title({"strict": "k* = 2: strict saddles, curvature stays negative",
                      "degenerate": "k* >= 3: flat saddles, curvature ~ sigma^(k*-2)"}[kind])
        ax.legend(fontsize=BIG - 4, loc="lower right", frameon=True, facecolor="white",
                  framealpha=0.9, edgecolor="none")
    axes[0].set_ylabel("|smallest Hessian eigenvalue| at the plateau\n(median over seeds)")
    fig.suptitle("Escape type follows the route rule's k*: order-2 plateaus keep a fixed negative "
                 "curvature, higher-order ones flatten as sigma^(k*-2)")
    savefig(fig, out / "fig5_escape_type")
    plt.close(fig)


def sequence(res: Path, out: Path):
    from analyze import sequence_scores

    seq = sequence_scores(read("transitions", res))
    seq = seq[seq.log10_sigma <= -1.5]
    order = sorted(seq.target.unique(), key=lambda n: (n.startswith(("CTL", "B4")), n))
    cols = palette(4)
    labels = ["paper's greedy rule (route depth)", "numerical route rule", "pairwise rule",
              "largest weight first"]
    hits = ["hit_paper", "hit_route_any", "hit_pair", "hit_weight"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for ax, kind, title in [
            (axes[0], "learn", "escapes in which a pattern is learned"),
            (axes[1], "cleanup", "clean-up escapes: no pattern is learned, an off-target string "
                                 "is removed or amplitudes are corrected")]:
        g = seq[seq.next_kind == kind]
        w = g.groupby("target")[hits].mean().reindex(order)
        n = g.groupby("target").size().reindex(order).fillna(0).astype(int)
        for i, (c, lab) in enumerate(zip(hits, labels)):
            ax.bar(np.arange(len(w)) + (i - 1.5) * 0.2, w[c].fillna(0).values, 0.2,
                   color=cols[i], label=lab)
        for j, k in enumerate(n.values):
            ax.text(j, 1.03, f"n={k}", ha="center", fontsize=BIG - 5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("fraction predicted\ncorrectly")
        ax.set_title(title)
    axes[1].set_xticks(range(len(order)), order, rotation=45, ha="right")
    h, l_ = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l_, ncol=2, loc="center right", fontsize=BIG - 1, frameon=True,
                   facecolor="white", edgecolor="none")
    fig.suptitle("What happens next: all rules name the next learned pattern; only the route "
                 "rule predicts the clean-ups,\nwhich are almost half of the escapes on non-star "
                 "targets (window sigmas; n = number of escapes)")
    savefig(fig, out / "fig6_sequence")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/outputs/route-nonstar")
    ap.add_argument("--res", default=str(HERE / "results"))
    a = ap.parse_args()
    style()
    res = Path(a.res)
    out = res
    for f in (lambda: example_trajectories(Path(a.raw), res, out),
              lambda: plateau_identity(res, out), lambda: exponents(res, out),
              lambda: effective_depth(res, out), lambda: escape_type(res, out),
              lambda: sequence(res, out)):
        f()


if __name__ == "__main__":
    main()
