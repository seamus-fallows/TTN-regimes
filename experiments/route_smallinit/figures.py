"""Draft figures from results/*.csv (one message per figure, stated in its title).

usage: uv run python experiments/route_smallinit/figures.py [--res DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttn.plotting import MUTED, palette, savefig, set_style, subplots  # noqa: E402

HERE = Path(__file__).resolve().parent
STR_COLS = ("order", "J", "J_before", "J_prev_events", "J_str", "first_pattern", "patterns", "forgotten",
            "transition", "region", "weights", "bonds", "order_dense", "order_check")


class EmptyDataError(Exception):
    pass


def read_csv(path):
    """read_csv keeping pattern strings such as 00201 as text."""
    try:
        head = pd.read_csv(path, nrows=0).columns
    except pd.errors.EmptyDataError as exc:
        raise EmptyDataError(str(exc)) from exc
    return pd.read_csv(path, dtype={c: str for c in head if c.endswith("_str") or c in STR_COLS})
SMALL = -3.0
ORDER = ["D4", "DB4", "B4", "S5A", "S5B", "S7", "H4", "B8", "TIE", "PS"]


def big_fonts():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})


def fig_rules(ev, out):
    """Agreement of the three rules with the pattern actually learned, on transitions where the
    rules disagree, per target (sigma <= 1e-3)."""
    e = ev[(ev.variant == "dense") & (ev.event > 0) & (ev.log10_sigma <= SMALL + 1e-9) & ~ev.floor_flag]
    dis = e[(e.pred_greedy != e.pred_pairwise) | (e.pred_greedy != e.pred_weight)]
    names = [t for t in ORDER if t in set(dis.target)]
    fig, ax = subplots(1, 1, width=7.5, panel_aspect=0.45)
    cols = palette(3)
    w = 0.26
    for k, (rule, lab) in enumerate([("agree_greedy", "greedy rule (route depth, then weight)"),
                                     ("agree_pairwise", "pairwise rule (grow from one learned pattern)"),
                                     ("agree_weight", "largest weight next")]):
        vals = [dis[dis.target == t][rule].mean() for t in names]
        ax.bar(np.arange(len(names)) + (k - 1) * w, vals, w, color=cols[k], label=lab)
        for i, v in enumerate(vals):
            ax.text(i + (k - 1) * w, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(range(len(names)), [f"{t}\nn={int((dis.target == t).sum())}" for t in names])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("fraction of transitions where\nthe rule named the learned pattern")
    ax.set_xlabel("target (only transitions where the three rules disagree; sigma <= 1e-3)")
    ax.set_title("The greedy route rule names the next pattern far more often\nthan the pairwise rule or the weight order")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=9.5)
    savefig(fig, out / "fig_rules")
    plt.close(fig)


def fig_rules_sigma(ev, out):
    """Greedy agreement vs sigma, ties and non-ties separately (all targets pooled)."""
    e = ev[(ev.variant == "dense") & (ev.event > 0) & ~ev.floor_flag]
    fig, ax = subplots(1, 1, width=6.5, panel_aspect=0.55)
    cols = palette(3)
    for k, (mask, lab) in enumerate([(~e.tie, "no tie in route depth"),
                                     (e.tie, "tie in route depth (broken by weight in theory)")]):
        g = e[mask].groupby("sigma").agree_greedy.agg(["mean", "size"])
        ax.plot(g.index, g["mean"], "o-", color=cols[k], label=lab)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("initialization scale sigma")
    ax.set_ylabel("fraction of transitions\nmatching the greedy rule")
    ax.set_title("Away from ties the greedy rule is right at every sigma;\nties in route depth are decided by the init, not by weight")
    ax.legend(loc="lower left")
    savefig(fig, out / "fig_rules_vs_sigma")
    plt.close(fig)


def fig_durations(ev, ref, out):
    """Median plateau duration vs sigma for one cell per route depth (clean events), with the
    predicted law: sigma^-(c-2), or the log law for c = 2 (slope 1/(s_r A_r(J)) per e-fold)."""
    e = ev[(ev.variant == "dense") & (ev.event > 0) & ~ev.floor_flag & (ev.run_status == "done") & ev.clean]
    picks = [("B8", "{0,1,2}->3"), ("B8", "{0}->1"), ("B8", "{0,1}->2"), ("PS", "{1,2}->0")]
    fig, axes = subplots(1, len(picks), width=13, panel_aspect=1.0)
    for k, (ax, (t, tr)) in enumerate(zip(axes, picks)):
        x = e[(e.target == t) & (e.transition == tr)]
        if not len(x):
            continue
        c = int(x.depth.iloc[0])
        g = x.groupby("sigma").duration.quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(g.index, g[0.25], g[0.75], color=palette(1)[0], alpha=0.25, lw=0)
        ax.plot(g.index, g[0.5], "o-", color=palette(1)[0], label="training (median, quartiles)")
        r = ref[(ref.target == t) & (ref.transition == tr)].sort_values("sigma")
        if len(r):
            ax.plot(r.sigma, r.duration, "s", mfc="none", color=palette(2)[1], label="reference flow on M*")
        sig = np.array(sorted(g.index))
        s0, y0 = sig[0], g[0.5].loc[sig[0]]
        if c > 2:
            ax.plot(sig, y0 * (sig / s0) ** (-(c - 2)), "--", color=MUTED, lw=2, label=f"prediction: sigma^-{c - 2}")
            ax.set_yscale("log")
        else:
            slope = 1.0 / (x.pattern_weight.iloc[0] * x.gamma.iloc[0])
            ax.plot(sig, y0 + slope * np.log(s0 / sig), "--", color=MUTED, lw=2, label="prediction: log(1/sigma)/(s A)")
        ax.set_xscale("log")
        ax.set_xlabel("initialization scale sigma")
        before = x.J_before_str.iloc[0]
        ax.set_title(f"{t}: after {before.replace(' ', ', ')}\nlearn {x.pattern_str.iloc[0]} (route depth c = {c})", fontsize=10)
        if k == 0:
            ax.set_ylabel("plateau duration (gradient-flow time)")
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Plateau durations follow the route-depth law sigma^-(c-2) (log law for c = 2)", fontsize=12.5)
    savefig(fig, out / "fig_durations")
    plt.close(fig)


def _weight(x):
    return float(x.pattern_weight.iloc[0]) if "pattern_weight" in x else 1.0


def fig_exponents(ex, out):
    """Per-seed exponents (clean events, sigma in [1e-5, 1e-3]) against the prediction c - 2."""
    e = ex[(ex.variant == "dense") & (ex.basis == "event_clean") & (ex.depth >= 3)]
    fig, ax = subplots(1, 1, width=7.0, panel_aspect=0.75)
    rng = np.random.default_rng(0)
    first = e.J_before == "{}"
    for k, (mask, lab, mk) in enumerate([(~first, "later plateau (route depth c of the learned pattern)", "o"),
                                         (first, "first plateau, from the origin (c = K cores)", "^")]):
        x = e[mask]
        ax.scatter(x.depth - 2 + rng.uniform(-0.18, 0.18, len(x)), x.exponent, s=14, alpha=0.45,
                   color=palette(2)[k], marker=mk, label=f"{lab}: {len(x)} seed fits")
    hi = e.depth.max() - 2 + 0.5
    ax.plot([0.5, hi], [0.5, hi], "--", color=MUTED, lw=2, label="prediction: exponent = c - 2")
    ax.set_xlabel("predicted exponent c - 2")
    ax.set_ylabel("fitted exponent per seed\n(duration ~ sigma^-exponent)")
    ax.set_title("Per-seed duration exponents sit on the route-depth prediction c - 2\n(clean transitions; outliers: see the forgetting and shortcut figures)")
    ax.legend(fontsize=9, loc="upper left")
    savefig(fig, out / "fig_exponents")
    plt.close(fig)


def fig_effdepth(ev, out):
    e = ev[(ev.variant == "dense") & (ev.event > 0) & (ev.log10_sigma <= SMALL + 1e-9) & ev.eff_depth.notna()]
    fig, ax = subplots(1, 1, width=6.0, panel_aspect=0.7)
    rng = np.random.default_rng(1)
    ax.scatter(e.depth + rng.uniform(-0.18, 0.18, len(e)), e.eff_depth.clip(upper=8), s=6, alpha=0.3,
               color=palette(1)[0], label="one learning event (sigma <= 1e-3)")
    for c, x in e.groupby("depth"):
        ax.plot([c - 0.25, c + 0.25], [x.eff_depth.median()] * 2, color="k", lw=2)
    ax.plot([1.5, 6.5], [1.5, 6.5], "--", color=MUTED, label="prediction: effective depth = c")
    ax.set_xlabel("route depth c of the learned pattern (number of dormant cores)")
    ax.set_ylabel("effective depth from the growth law")
    ax.set_title("Each learning event grows with the effective depth\nthat the route depth predicts")
    ax.legend(fontsize=9, loc="upper left")
    savefig(fig, out / "fig_effective_depth")
    plt.close(fig)


def fig_onestep(ev, out):
    e = ev[(ev.variant == "dense") & (ev.event > 0) & (ev.log10_sigma <= SMALL + 1e-9)]
    fig, ax = subplots(1, 1, width=7.0, panel_aspect=0.6)
    bins = np.logspace(-2, 6, 65)
    for k, (col, lab) in enumerate([("onestep_ratio", "from the amplitude P_r at plateau entry (brief)"),
                                    ("onestep_ratio_B", "from the size of the dormant route entries")]):
        x = e[col].replace([np.inf, -np.inf], np.nan).dropna()
        x = x[x > 0]
        ax.hist(x.clip(1e-2, 1e6), bins=bins, histtype="step", lw=2, color=palette(2)[k],
                label=f"{lab}: median {x.median():.2g}")
    ax.axvline(1, ls="--", color=MUTED, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_xlabel("observed / predicted plateau time (plateau entry to next event; last bin: >= 1e6 or never)")
    ax.set_ylabel("number of events")
    ax.set_title("The scalar route equation predicts the plateau length from the size\nof the dormant route entries, not from the amplitude P_r")
    ax.legend(fontsize=9)
    savefig(fig, out / "fig_onestep")
    plt.close(fig)


def fig_offtarget(ev, out):
    e = ev[(ev.variant == "dense") & (ev.event > 0)]
    fig, ax = subplots(1, 1, width=6.5, panel_aspect=0.6)
    g = e.groupby("sigma").peak_offtarget.quantile([0.5, 0.9, 1.0]).unstack()
    for k, (q, lab) in enumerate([(0.5, "median"), (0.9, "90th percentile"), (1.0, "maximum")]):
        ax.plot(g.index, g[q].clip(lower=1e-16), "o-", color=palette(3)[k], label=lab)
    ax.axhline(1e-2, ls="--", color=MUTED, label="1% of ||W||^2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("initialization scale sigma")
    ax.set_ylabel("peak off-target fraction of ||W||^2\nduring a transition")
    ax.set_title("Most transitions stay on the aligned manifold (median off-target part ~ sigma^2);\nthe takeovers and two-at-once transitions of PS, S5A, S5B do not (maximum)")
    ax.legend(fontsize=9)
    savefig(fig, out / "fig_offtarget")
    plt.close(fig)


def fig_census(runs, out):
    r = runs[runs.variant == "dense"]
    cats = ["clean", "only the first escape mixed", "two patterns at once (later)", "forgetting",
            "trapped (a route dies)", "other (non-canonical plateau, unfinished)"]

    def cat(b):
        if b == "clean":
            return cats[0]
        if "trapped" in b:
            return cats[4]
        if "forgetting" in b:
            return cats[3]
        if "two_at_once" in b:
            return cats[2]
        if b == "first_escape_mixed":
            return cats[1]
        return cats[5]

    r = r.assign(cat=r.behaviour.map(cat))
    names = [t for t in ORDER if t in set(r.target)]
    fig, ax = subplots(1, 1, width=7.5, panel_aspect=0.45)
    bottom = np.zeros(len(names))
    cols = palette(6)
    for k, c in enumerate(cats):
        vals = np.array([(r[r.target == t].cat == c).mean() for t in names])
        ax.bar(range(len(names)), vals, bottom=bottom, color=cols[k], label=c)
        bottom += vals
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("fraction of runs (all sigma, 40 seeds)")
    ax.set_title("Most runs follow a clean saddle-to-saddle path;\nthe deviations are specific to some targets")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5)
    savefig(fig, out / "fig_census")
    plt.close(fig)




def fig_shortcut(res, out):
    """PS: the plateau at J = {000000} (from its pass to the next pass) vs sigma, against Theorem 3.2
    (order 4: sigma^-2) and the route depth (5: sigma^-3); and the relearning of 000000 (depth 5)."""
    ps = read_csv(res / "passes.csv")
    runs = read_csv(res / "runs.csv")
    ps = ps[(ps.variant == "dense") & (ps.target == "PS")]
    hist = runs[(runs.variant == "dense") & (runs.target == "PS")].set_index(["log10_sigma", "seed"]).history
    ps = ps.assign(history=[hist.get((a, b)) for a, b in zip(ps.log10_sigma, ps.seed)])
    q_first = ps[ps["history"].fillna("").str.startswith("+000000 -000000")]
    plateau = q_first[q_first.n_pass == 1]  # the pass after 000000: the pair escapes from J = {000000}
    relearn = ps[(ps.pattern == 0) & (ps.J_before == "{1,2}")]
    fig, axes = subplots(1, 2, width=11, panel_aspect=0.8)
    for ax, x, preds, title in [
        (axes[0], plateau, [(2, "Theorem 3.2 (order 4, spectral shortcut): sigma^-2", "--"),
                            (3, "route depth 5: sigma^-3", ":")],
         f"PS: plateau after learning 000000 first\n({plateau.seed.nunique()} of 40 seeds take this path)"),
        (axes[1], relearn, [(3, "route depth 5: sigma^-3", "--"), (4, "origin escape K = 6: sigma^-4", ":")],
         "PS: learning 000000 after the pair 110001, 110010"),
    ]:
        g = x.groupby("sigma").duration.quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(g.index, g[0.25], g[0.75], color=palette(1)[0], alpha=0.25, lw=0)
        ax.plot(g.index, g[0.5], "o-", color=palette(1)[0], label="training (median, quartiles)")
        sig = np.array(sorted(g.index))
        s0 = sig[len(sig) // 2]
        y0 = g[0.5].loc[s0]
        for k, (ex_, lab, ls) in enumerate(preds):
            ax.plot(sig, y0 * (sig / s0) ** (-ex_), ls, color=MUTED if k == 0 else "k", lw=2, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("initialization scale sigma")
        ax.set_ylabel("plateau duration (gradient-flow time)")
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=8.5)
    fig.suptitle("Pair-star target: the first plateau follows the spectral-shortcut order, not the route depth", fontsize=12.5)
    savefig(fig, out / "fig_shortcut_PS")
    plt.close(fig)


def fig_example(out, name="B8", lsig=-4.0, seed=0, tag="fig_example_run", msg=None):
    """One run, one panel per plateau: amplitudes P_a/s_a against the time since the previous
    learning event (exact), with the learned pattern's route depth."""
    import sys

    import yaml

    sys.path.insert(0, str(HERE))
    from fractions import Fraction

    from run import learning_events, predictions, run_full
    from model import build_target

    cfg = {c["name"]: c for c in yaml.safe_load(open(HERE / "configs/targets.yaml"))["targets"]}[name]
    target = build_target(cfg)
    model, traj, info = run_full(cfg, 10.0**lsig, seed)
    P = np.array([o[1] for o in traj.obs])
    s = np.array(target.weights)
    _, _, cross = learning_events(traj, P, s)
    # every upward crossing of half the weight ("pass"), in time order (= the learning events when
    # nothing is forgotten)
    events = sorted([dict(pattern=a, t=c[0], i=c[2]) for a in range(target.n) for c in cross[(a, 0.5)] if c[1] > 0],
                    key=lambda e: e["t"])
    fig, axes = subplots(1, len(events), width=2.6 * len(events) + 1.5, panel_aspect=1.1, sharey=True)
    cols = palette(target.n)
    prev, i_prev = Fraction(0), 0
    J = []
    for k, (ax, ev) in enumerate(zip(axes, events)):
        i_end = min(ev["i"] + 40, len(P) - 1)
        t = np.array([float(traj.exact(i) - prev) for i in range(i_prev, i_end + 1)])
        if k == 0:  # first escape: linear time in units of the first event time
            t = t / float(ev["t"])
        ok = t > 0
        for a in range(target.n):
            ax.plot(t[ok], np.abs(P[i_prev:i_end + 1, a][ok]) / s[a], color=cols[a],
                    label="".join(map(str, target.patterns[a])) + f" (weight {s[a]:g})")
        r = ev["pattern"]
        J = [a for a in range(target.n) if a != r and P[ev["i"] - 1, a] > 0.5 * s[a]]
        c = target.depth(J, r) if J else target.tree.n_nodes
        pr = predictions(cfg, target, J)["greedy"] if J else None
        ax.set_xscale("linear" if k == 0 else "log")
        ax.set_yscale("log")
        ax.set_ylim(1e-22, 3)
        ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        what = "".join(map(str, target.patterns[r]))
        if pr is None:
            sub = f"from the origin (all K = {c} cores)"
        else:
            ok_ = "matches" if pr == r else "differs from"
            sub = f"route depth c = {c}; {ok_} greedy"
        lost = [b for b in range(target.n) if P[i_prev, b] > 0.5 * s[b] and P[ev["i"], b] < 0.5 * s[b]]
        if lost:
            sub += "\nloses " + ", ".join("".join(map(str, target.patterns[b])) for b in lost)
        ax.set_title(f"plateau {k} -> learns {what}\n{sub}", fontsize=9.5)
        ax.set_xlabel("time / first-event time" if k == 0 else "time since previous event")
        prev, i_prev = ev["t"], ev["i"]
    axes[0].set_ylabel("|amplitude P_a| / weight s_a")
    axes[0].legend(fontsize=8.5, loc="lower right")
    msg = msg or ("patterns are learned one at a time,\neach after a plateau whose length is set by its route depth "
                  "(dotted line: half the weight = learned)")
    fig.suptitle(f"One training run ({name}, sigma = 1e{lsig:g}, seed {seed}): {msg}", fontsize=12)
    savefig(fig, out / tag)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(HERE / "results"))
    args = ap.parse_args()
    res = Path(args.res)
    set_style("sans")
    big_fonts()
    ev = read_csv(res / "events.csv")
    runs = read_csv(res / "runs.csv")
    try:
        ex = read_csv(res / "exponents.csv")
    except EmptyDataError:
        ex = pd.DataFrame()
    targets = read_csv(res / "targets.csv")
    w = {t: [float(v) for v in s.split()] for t, s in zip(targets.target, targets.weights)}
    ev["pattern_weight"] = [w[t][int(p)] for t, p in zip(ev.target, ev.pattern)]
    ref = ev[(ev.variant == "ref") & (ev.event > 0)]
    fig_rules(ev, res)
    fig_rules_sigma(ev, res)
    fig_durations(ev, ref, res)
    if len(ex):
        fig_exponents(ex, res)
    fig_effdepth(ev, res)
    fig_onestep(ev, res)
    fig_offtarget(ev, res)
    fig_census(runs, res)
    fig_shortcut(res, res)
    fig_example(res)
    fig_example(res, name="S5A", lsig=-4.0, seed=1, tag="fig_example_forgetting",
                msg="a lighter pair is learned first;\nthe heavy 00000 takes over in log time and erases 10101 and 00201, "
                    "which are relearned later")


if __name__ == "__main__":
    main()
