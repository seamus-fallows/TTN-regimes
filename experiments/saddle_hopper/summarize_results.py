"""Collect the validation outputs into the summary tables and figures of notes/saddle_hopper.md.

Reads results/*.csv written by validate_*.py; writes results/summary_sequences.csv,
results/summary_laws.csv, results/mismatches.csv, results/fig_laws.png, results/fig_match.png.
usage: uv run python experiments/saddle_hopper/summarize_results.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"


def family_locality(r):
    if r["name"].startswith("S="):
        k = r["k_pred"]
        return (
            "2 comp, k = 1"
            if not np.isfinite(k)
            else f"2 comp, k = {int(k)}"
            if k <= 4
            else "2 comp, k >= 5"
        )
    return r["name"]


def sequences():
    rows = []
    loc = pd.read_csv(RES / "locality_runs_scored.csv")
    loc["family"] = loc.apply(family_locality, axis=1)
    loc["init"] = loc.start.map(
        {"small": "fan-in", "small_balanced": "balanced", "saddle": "saddle start"}
    )
    loc["tree_g"] = loc.tree.where(~loc.name.str.startswith("S="), "all 9 trees")
    s2s = pd.read_csv(RES / "s2s_core_runs_scored.csv")
    s2s["family"] = "odeco R=3, " + s2s.teacher
    s2s["init"] = s2s["init"].map({"fanin": "fan-in", "balanced": "balanced"})
    s2s["tree_g"] = s2s.tree
    for name, df in (("locality", loc), ("s2s_core", s2s)):
        for (fam, tree, init), g in df.groupby(["family", "tree_g", "init"]):
            sc = g[g.result != "unscored"]
            row = dict(
                dataset=name,
                family=fam,
                tree=tree,
                init=init,
                runs=len(g),
                scored=len(sc),
                main=int((sc.result == "main").sum()),
                rare=int((sc.result == "rare").sum()),
                mismatch=int((sc.result == "mismatch").sum()),
                unscored=len(g) - len(sc),
            )
            row["match"] = (row["main"] + row["rare"]) / row["scored"] if row["scored"] else np.nan
            row["main_frac"] = row["main"] / row["scored"] if row["scored"] else np.nan
            if "n_main_orders" in g:
                row["deterministic_runs"] = int((sc.n_main_orders == 1).sum())
            rows.append(row)
    st = pd.read_csv(RES / "data_teacher_star_classes.csv")
    for (tree, w), g in st.groupby(["tree", "w"]):
        main = int(g[g.predicted.str.contains("main")].runs.sum())
        rare = int(g[g.predicted.str.contains("rare")].runs.sum())
        n = int(g.runs.sum())
        rows.append(
            dict(
                dataset="data_teacher",
                family=f"star w={w}",
                tree=tree,
                init="fan-in",
                runs=n,
                scored=n,
                main=main,
                rare=rare,
                mismatch=n - main - rare,
                unscored=0,
                match=(main + rare) / n,
                main_frac=main / n,
            )
        )
    om = pd.read_csv(RES / "order_multi.csv")
    for (tree, R, rho), g in om.groupby(["tree", "R", "rho"]):
        if g.freq_gf.sum() <= 0:
            continue
        f = g.groupby("branch").freq_gf.sum()
        rows.append(
            dict(
                dataset="order",
                family=f"odeco R={R}, rho={rho}",
                tree=tree,
                init="fan-in",
                runs=np.nan,
                scored=np.nan,
                main=f.get("main", 0.0),
                rare=f.get("rare", 0.0),
                mismatch=f.get("none", 0.0),
                unscored=np.nan,
                match=f.get("main", 0.0) + f.get("rare", 0.0),
                main_frac=f.get("main", 0.0),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(RES / "summary_sequences.csv", index=False)
    two = loc[loc.name.str.startswith("S=")]
    per = two[two.result != "unscored"].groupby(["tree", "family", "init"]).result.value_counts().unstack(fill_value=0)
    per["match"] = (per.get("main", 0) + per.get("rare", 0)) / per.sum(axis=1)
    per.to_csv(RES / "summary_twocomp_per_tree.csv")
    return out


def laws():
    rows = []
    t = pd.read_csv(RES / "locality_twocomp_laws.csv")
    for r in t.itertuples():
        rows.append(
            dict(
                dataset="locality",
                case=f"{r.tree} {r.case} {r.start} chi={r.chi} w={r.w}",
                quantity=r.quantity,
                predicted=r.predicted,
                measured=r.measured,
                se=r.se,
                agree=r.agree,
            )
        )
        if isinstance(r.hess_agree, (bool, np.bool_)) or r.hess_agree in ("True", "False"):
            q = (
                "Hessian lambda_min (k=2)"
                if r.quantity == "log slope"
                else "lambda_min sigma-exponent"
            )
            rows.append(
                dict(
                    dataset="locality",
                    case=f"{r.tree} {r.case} {r.start} chi={r.chi} w={r.w}",
                    quantity=q,
                    predicted=r.hess_pred,
                    measured=r.hess_measured,
                    se=np.nan,
                    agree=r.hess_agree in (True, "True"),
                )
            )
    lv = pd.read_csv(RES / "locality_levels.csv")
    for r in lv.itertuples():
        if pd.isna(r.k_hopper):
            continue
        case = f"{r.case} {r.tree} {r.start} order {r.order} level {r.level_j} ({r.branch} branch)"
        # measured: locality's detected-plateau fraction where the hopper's branch is the one locality
        # predicted, else the plateau level implied by locality's L_mid_err (5% tolerance)
        rows.append(
            dict(
                dataset="locality",
                case=case,
                quantity="plateau level",
                predicted=r.level_hopper,
                measured=r.level_locality if r.level_basis == "detected_frac" else r.level_measured,
                se=np.nan,
                agree=bool(r.level_agree),
                basis=f"detected in {r.detected_frac:.2f} of runs"
                if r.level_basis == "detected_frac"
                else "L_mid",
            )
        )
        if r.branch != "main":
            continue
        if r.k_hopper == 2 and np.isfinite(r.logslope) and r.law == "log":
            pred = r.slope_saddle if r.start == "saddle" else r.slope_small
            rows.append(
                dict(
                    dataset="locality",
                    case=case,
                    quantity="log slope",
                    predicted=pred,
                    measured=r.logslope,
                    se=np.nan,
                    agree=abs(r.logslope / pred - 1) <= 0.1,
                )
            )
        elif r.k_hopper >= 3:
            meas = r.p_detected if np.isfinite(r.p_detected) else r.p_fit
            if np.isfinite(meas):
                rows.append(
                    dict(
                        dataset="locality",
                        case=case,
                        quantity="exponent",
                        predicted=r.k_hopper - 2,
                        measured=meas,
                        se=r.p_se,
                        agree=abs(meas - (r.k_hopper - 2)) <= 0.2,
                    )
                )
    s = pd.read_csv(RES / "s2s_core_exponents.csv")
    for r in s.itertuples():
        q = "exponent" if r.quantity.startswith("p_") else "lambda_min sigma-exponent"
        rows.append(
            dict(
                dataset="s2s_core",
                case=f"{r.sweep} {r.tree} chi={r.chi} {r.teacher} {r.quantity}",
                quantity=q,
                predicted=float(r.predicted),
                measured=r.measured,
                se=r.se,
                agree=r.agree,
            )
        )
    st = pd.read_csv(RES / "s2s_core_strict.csv")
    for r in st.itertuples():
        rows.append(
            dict(
                dataset="s2s_core",
                case=f"{r.tree} chi={r.chi} {r.teacher} seed {r.seed} learned {r.learned}",
                quantity="strict wrong-order lambda (ratio measured/predicted)",
                predicted=1.0,
                measured=r.lam_measured_median_ratio,
                se=np.nan,
                agree=abs(r.lam_measured_median_ratio - 1) <= 0.1,
            )
        )
    a = pd.read_csv(RES / "data_teacher_saddle_start.csv")
    for r in a.itertuples():
        case = f"{r.teacher} {r.tree} (saddle start)"
        rows.append(
            dict(
                dataset="data_teacher",
                case=case,
                quantity="plateau level",
                predicted=r.level_hopper,
                measured=r.level_measured,
                se=np.nan,
                agree=abs(r.level_hopper / r.level_measured - 1) < 1e-4,
            )
        )
        rows.append(
            dict(
                dataset="data_teacher",
                case=case,
                quantity="saddle type",
                predicted=r.type_hopper,
                measured=r.type_measured,
                se=np.nan,
                agree=r.type_hopper.startswith("strict") == (r.type_measured == "strict"),
            )
        )
        if r.law == "log":
            rows.append(
                dict(
                    dataset="data_teacher",
                    case=case,
                    quantity="Hessian lambda_min (k=2)",
                    predicted=r.lam_hopper,
                    measured=r.lam_measured,
                    se=np.nan,
                    agree=abs(r.lam_hopper / r.lam_measured - 1) <= 0.1,
                )
            )
            meas = (
                r.slope_measured if np.isfinite(r.slope_measured) else r.slope_measured_intermediate
            )
            rows.append(
                dict(
                    dataset="data_teacher",
                    case=case,
                    quantity="log slope",
                    predicted=r.slope_hopper,
                    measured=meas,
                    se=r.slope_measured_se,
                    agree=abs(meas / r.slope_hopper - 1) <= 0.1,
                )
            )
        elif np.isfinite(r.exponent_measured):
            rows.append(
                dict(
                    dataset="data_teacher",
                    case=case,
                    quantity="exponent",
                    predicted=r.exponent_hopper,
                    measured=r.exponent_measured,
                    se=np.nan,
                    agree=abs(r.exponent_measured - r.exponent_hopper) <= 0.2,
                )
            )
    b = pd.read_csv(RES / "data_teacher_random_init.csv")
    for r in b.itertuples():
        case = f"{r.sweep} {r.case} {r.tree} saddle {int(r.saddle)} (random init)"
        if r.law == "log":
            rows.append(
                dict(
                    dataset="data_teacher",
                    case=case,
                    quantity="log slope",
                    predicted=r.slope_hopper,
                    measured=r.slope_measured,
                    se=r.slope_measured_se,
                    agree=abs(r.slope_measured / r.slope_hopper - 1) <= 0.1,
                )
            )
        else:
            rows.append(
                dict(
                    dataset="data_teacher",
                    case=case,
                    quantity="exponent",
                    predicted=r.exponent_hopper,
                    measured=r.exponent_measured,
                    se=r.exponent_measured_se,
                    agree=abs(r.exponent_measured - r.exponent_hopper) <= 0.2,
                )
            )
    h = pd.read_csv(RES / "data_teacher_star_hessian.csv")
    for r in h.itertuples():
        case = f"star {r.tree} w={r.w}"
        rows.append(
            dict(
                dataset="data_teacher",
                case=case,
                quantity="star strict (rho(S) > s_q)",
                predicted=r.strict_hopper,
                measured=r.strict_measured,
                se=np.nan,
                agree=r.strict_hopper == r.strict_measured,
            )
        )
        if r.strict_measured:
            rows.append(
                dict(
                    dataset="data_teacher",
                    case=case,
                    quantity="Hessian lambda_min (fixed-rank)",
                    predicted=r.lam_hopper,
                    measured=r.lam_measured,
                    se=np.nan,
                    agree=abs(r.lam_hopper / r.lam_measured - 1) <= 0.1,
                )
            )
    o = pd.read_csv(RES / "order_sigma_exponents.csv")
    for r in o.itertuples():
        pred = r.gamma_hopper
        rows.append(
            dict(
                dataset="order",
                case=f"{r.sweep} {r.tree} d={r.d} chi={r.chi} rho={r.rho}",
                quantity="wrong-order exponent gamma (heuristic)",
                predicted=pred,
                measured=r.gamma_measured,
                se=r.gamma_se,
                agree=abs(r.gamma_measured - pred) <= max(0.2 * pred, 2 * r.gamma_se, 0.02),
            )
        )
    hb = pd.read_csv(RES / "order_h1b.csv")
    for r in hb.itertuples():
        case = f"binary4 rho={r.rho} (wrong-order saddle start)"
        rows.append(
            dict(
                dataset="order",
                case=case,
                quantity="Hessian lambda_min (fixed-rank)",
                predicted=r.lam_hopper,
                measured=r.lam_measured,
                se=np.nan,
                agree=abs(r.lam_hopper / r.lam_measured - 1) <= 0.1,
            )
        )
        rows.append(
            dict(
                dataset="order",
                case=case,
                quantity="rotation log slope",
                predicted=r.slope_hopper,
                measured=r.slope_measured,
                se=r.slope_se,
                agree=abs(r.slope_measured / r.slope_hopper - 1) <= 0.1,
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(RES / "summary_laws.csv", index=False)
    return out


def rare_trend():
    """Rows with >= 4 rare-branch runs: rare fraction below vs above the row's median sigma."""
    d = pd.read_csv(RES / "locality_runs_scored.csv")
    d = d[d.result != "unscored"]
    rows = []
    for key, g in d.groupby(["sweep", "name", "tree", "start", "chi", "w"]):
        if (g.result == "rare").sum() < 4:
            continue
        med = np.median(sorted(g.sigma.unique()))
        lo, hi = g[g.sigma < med], g[g.sigma > med]
        rows.append(dict(zip(["sweep", "name", "tree", "start", "chi", "w"], key), n=len(g),
                         rare=int((g.result == "rare").sum()),
                         frac_rare_small_sigma=float((lo.result == "rare").mean()),
                         frac_rare_large_sigma=float((hi.result == "rare").mean())))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "locality_rare_trend.csv", index=False)
    return out


def mismatches():
    loc = pd.read_csv(RES / "locality_runs_scored.csv")
    m = loc[loc.result == "mismatch"]
    out = (
        m.groupby(["cause", "sweep", "name", "tree", "start", "chi", "w", "cls"])
        .agg(
            runs=("seed", "size"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
            sigmas=("sigma", lambda x: f"{min(x):.3g}-{max(x):.3g}"),
            orders=("order", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    out.insert(0, "dataset", "locality")
    om = pd.read_csv(RES / "order_multi.csv")
    om = om[(om.branch == "none") & (om.freq_gf > 0)]
    for r in om.itertuples():
        out.loc[len(out)] = dict(
            dataset="order",
            cause="finite sigma: add route competes with the rotation "
            "(theory_saddles 4.4; sigma = 0.05)",
            sweep="multi2",
            name=f"odeco R={r.R} rho={r.rho}",
            tree=r.tree,
            start="small",
            chi=4,
            w="",
            cls="other order",
            runs=r.freq_gf,
            seeds="",
            sigmas="0.05",
            orders=str(r.order),
        )
    out.to_csv(RES / "mismatches.csv", index=False)
    return out


COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
]  # dataviz reference palette, slots 1-3 (validated all-pairs)
MARKERS = ["o", "s", "^"]


def fig_laws(lw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [
        ("log slopes (k = 2 routes, rotations)", lambda q: "log slope" in q, True),
        ("power-law exponents (k - 2)", lambda q: q == "exponent", False),
        ("Hessian lambda_min (strict saddles)", lambda q: q.startswith("Hessian lambda_min"), True),
    ]
    for ax, (title, sel, logscale) in zip(axes, panels):
        d = lw[lw.quantity.map(sel)].copy()
        d["predicted"] = pd.to_numeric(d.predicted, errors="coerce")
        d["measured"] = pd.to_numeric(d.measured, errors="coerce")
        d = d.dropna(subset=["predicted", "measured"])
        if logscale:
            d = d[(d.predicted.abs() > 0) & (d.measured.abs() > 0)]
            x, y = d.predicted.abs(), d.measured.abs()
        else:
            x, y = d.predicted, d.measured
        for i, ds in enumerate(sorted(d.dataset.unique())):
            sel_ds = d.dataset == ds
            ok = d.agree.astype(str) == "True"
            if not logscale:  # integer predictions: offset the datasets and show the fit errors
                x = d.predicted + 0.14 * (i - 1)
                se = pd.to_numeric(d.se, errors="coerce")
                ax.errorbar(x[sel_ds], y[sel_ds], yerr=se[sel_ds].where(se[sel_ds] < 3), fmt="none",
                            ecolor=COLORS[i % 3], elinewidth=0.8, alpha=0.6, zorder=2)
            ax.scatter(
                x[sel_ds & ok],
                y[sel_ds & ok],
                s=34,
                marker=MARKERS[i % 3],
                color=COLORS[i % 3],
                edgecolor="white",
                linewidth=0.6,
                label=f"{ds} ({int((sel_ds & ok).sum())}/{int(sel_ds.sum())} agree)",
                zorder=3,
            )
            ax.scatter(
                x[sel_ds & ~ok],
                y[sel_ds & ~ok],
                s=40,
                marker=MARKERS[i % 3],
                facecolor="none",
                edgecolor=COLORS[i % 3],
                linewidth=1.2,
                zorder=3,
            )
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        if logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo, hi = lo / 1.5, hi * 1.5
        else:
            lo, hi = lo - 0.4, hi + 0.4
        ax.plot([lo, hi], [lo, hi], color="#8a8985", lw=1, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("hopper prediction" + (" (abs)" if logscale else ""))
        ax.set_ylabel("measured" + (" (abs)" if logscale else ""))
        ax.grid(color="#e7e6e2", lw=0.6)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(
        "Saddle hopper vs committed runs (filled: within the section-2 tolerance; open: outside)",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(RES / "fig_laws.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_match(sq):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = sq[sq.dataset.isin(["locality", "s2s_core", "data_teacher"])].copy()
    d = d.groupby(["dataset", "family", "init"])[["main", "rare", "mismatch"]].sum().reset_index()
    d["n"] = d[["main", "rare", "mismatch"]].sum(axis=1)
    d = d[d.n > 0]
    d["label"] = (
        d.dataset + ": " + d.family + " (" + d["init"] + ", n=" + d.n.astype(int).astype(str) + ")"
    )
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.26 * len(d) + 1.4))
    left = np.zeros(len(d))
    for col, c, lab in (
        ("main", COLORS[0], "main branch"),
        ("rare", COLORS[1], "rare branch (P ~ sigma^gamma)"),
        ("mismatch", COLORS[2], "mismatch"),
    ):
        frac = (d[col] / d.n).values
        ax.barh(
            range(len(d)),
            frac,
            left=left,
            color=c,
            edgecolor="white",
            linewidth=1.0,
            height=0.78,
            label=lab,
        )
        left += frac
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.label, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of scored runs")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", color="#e7e6e2", lw=0.6)
    fig.tight_layout()
    fig.savefig(RES / "fig_match.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    sq = sequences()
    lw = laws()
    mm = mismatches()
    ec = exponent_causes()
    rt = rare_trend()
    fig_laws(lw)
    fig_match(sq)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    print(sq.to_string())
    print(
        lw.groupby(["dataset", "quantity"])
        .agree.agg(lambda x: f"{int((x.astype(str) == 'True').sum())}/{len(x)}")
        .to_string()
    )
    print(mm.groupby(["dataset", "cause"]).runs.sum().to_string())
    print(ec.groupby(ec.cause.str.split(":").str[0].str.split("(").str[0]).size().to_string())
    print("rare runs thinner at small sigma:", int((rt.frac_rare_small_sigma < rt.frac_rare_large_sigma).sum()),
          "equal:", int((rt.frac_rare_small_sigma == rt.frac_rare_large_sigma).sum()), "of", len(rt))





def exponent_causes():
    """Cause of every locality exponent outside the 0.2 tolerance, by explicit checks against locality's
    own diagnostics (stall dropout, sigma <= 0.03 / one-decade extensions, seeds)."""
    loc = Path(__file__).parents[1] / "locality" / "results"
    t = pd.read_csv(RES / "locality_twocomp_laws.csv")
    t = t[(t.quantity == "exponent") & ~t.agree.astype(bool)]
    sd = pd.read_csv(loc / "stall_dropout_bias.csv")
    ext = pd.read_csv(loc / "extensions_table.csv")
    rows = []
    for r in t.itertuples():
        pred, m, se = r.predicted, r.measured, r.se
        d = sd[
            (sd.tree == r.tree) & (sd.case == r.case) & (sd.start == r.start) & (sd.chi == r.chi)
        ]
        e = ext[(ext.tree == r.tree) & (ext.case == r.case)]
        pns = float(d.p_never_stalled.iloc[0]) if len(d) else np.nan
        if not np.isfinite(m):
            cause = "no escaping run with >= 3 sigmas (all stalled)"
        elif np.isfinite(se) and abs(m - pred) <= 2 * se:
            cause = "within 2 SE (few fitted seeds)"
        elif np.isfinite(pns) and abs(pns - pred) <= 0.2:
            cause = f"stall dropout: never-stalled seeds give {pns:.2f}"
        elif (
            r.table != "extensions_table"
            and len(e[e.start == r.start])
            and any(abs(x - pred) <= 0.2 for x in e[e.start == r.start].p_fit)
        ):
            cause = (
                "pre-asymptotic: the sigma <= 0.03 / one-decade extension (same start) gives "
                + ", ".join(f"{x:.2f}" for x in e[e.start == r.start].p_fit)
            )
        elif (
            r.table != "extensions_table"
            and len(e)
            and r.start not in set(e.start)
            and any(abs(x - pred) <= 0.2 for x in e.p_fit)
        ):
            cause = (
                "pre-asymptotic: the sigma <= 0.03 extension (other starts: "
                + ", ".join(f"{s} {x:.2f}" for s, x in zip(e.start, e.p_fit))
                + ")"
            )
        elif r.fit_seeds <= 2:
            cause = f"few seeds ({int(r.fit_seeds)})"
        else:
            cause = (
                "unexplained (steep MPS route, locality 7.9; theory_saddles P14(ii))"
                if r.tree.startswith("mps")
                else "unexplained"
            )
        rows.append(
            dict(
                table=r.table,
                tree=r.tree,
                case=r.case,
                start=r.start,
                chi=r.chi,
                k=r.k_pred,
                predicted=pred,
                measured=m,
                se=se,
                fit_seeds=r.fit_seeds,
                sigma_range=r.sigma_range,
                p_never_stalled=pns,
                cause=cause,
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exponent_mismatches.csv", index=False)
    return out


if (
    __name__ == "__main__"
    and len(__import__("sys").argv) > 1
    and __import__("sys").argv[1] == "causes"
):
    pd.set_option("display.width", 250)
    print(exponent_causes().to_string())


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()
