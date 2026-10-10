"""Derived numbers quoted in report Sec. 4.1-4.3 and 4.6 that are not single entries of another table.

Reads only committed tables in results/ (no raw data) and writes results/report_numbers.csv with one
row per quantity (key, value, detail). Bootstrap resamples use a fixed seed, so reruns are identical.

    uv run python experiments/s2s_core/report_numbers.py --results experiments/s2s_core/results --out experiments/s2s_core/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 1000


def boot(values, stat, seed=0):
    """Bootstrap standard error of stat(values)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    return float(np.std([stat(rng.choice(v, len(v))) for _ in range(N_BOOT)], ddof=1))


def median_ci(values, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    meds = np.median(rng.choice(v, size=(2000, len(v))), axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def chi_slope(g: pd.DataFrame, chis, col="T") -> float:
    """Log-log slope of the per-chi median of `col` against chi."""
    y = [np.log(np.median(g[g.chi == c][col])) for c in chis]
    return float(np.polyfit(np.log(chis), y, 1)[0])


def chi_slope_se(g: pd.DataFrame, chis, seed=0) -> float:
    """Bootstrap SE of chi_slope: resample the initializations within each chi."""
    rng = np.random.default_rng(seed)
    parts = {c: g[g.chi == c] for c in chis}
    out = []
    for _ in range(N_BOOT):
        gb = pd.concat([p.iloc[rng.integers(0, len(p), len(p))] for p in parts.values()])
        out.append(chi_slope(gb, chis))
    return float(np.std(out, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results"))
    ap.add_argument(
        "--out", required=True, help="output directory (e.g. experiments/s2s_core/results)"
    )
    args = ap.parse_args()
    res = Path(args.results)
    rows = []

    def put(key, value, detail=""):
        rows.append({"key": key, "value": value, "detail": detail})

    # ---- first escape (first_runs.csv) -------------------------------------------------------
    fr = pd.read_csv(res / "first_runs.csv")
    put("first.n_runs", len(fr), "gradient-flow first-escape runs (231 jobs x 8 sigmas)")
    put(
        "first.r1_agree",
        int((fr.r1 == fr.r1_corrflow).sum()),
        "first component = correlation-flow r1",
    )
    bad = fr[fr.r1 != fr.r1_corrflow]
    for _, b in bad.iterrows():
        put(
            "first.r1_exception",
            f"{b.tree} chi={b.chi} {b.teacher} {b.init} seed={b.seed} sigma={b.sigma}",
            f"learned component {b.r1} (0-based) instead of {b.r1_corrflow}",
        )
    smin = fr.groupby("tree").sigma.transform("min")
    at_min = fr[fr.sigma == smin]
    put(
        "first.r1_agree_smallest_sigma",
        f"{int((at_min.r1 == at_min.r1_corrflow).sum())}/{len(at_min)}",
    )
    for tree, g in fr.groupby("tree"):
        put(f"first.largest_first.{tree}", f"{int((g.r1 == 0).sum())}/{len(g)}")
        put(
            f"first.race_agree.{tree}",
            f"{int((g.r1 == g.r1_race).sum())}/{len(g)}",
            "first component = argmin_r of the single-component blow-up times (race)",
        )
    jobs = (
        fr.groupby(["tree", "chi", "teacher", "init", "seed"])
        .agg(T=("T", "first"), r1=("r1_corrflow", "first"))
        .reset_index()
    )
    m = (
        jobs[jobs.init == "fanin"]
        .drop(columns="init")
        .merge(
            jobs[jobs.init == "balanced"].drop(columns="init"),
            on=["tree", "chi", "teacher", "seed"],
            suffixes=("_f", "_b"),
        )
    )
    ratio = m.T_b / m.T_f
    cell_med = ratio.groupby([m.tree, m.chi, m.teacher]).median()
    put(
        "first.balanced_over_fanin_T.range",
        f"{ratio.min():.3f}-{ratio.max():.3f}",
        f"{len(m)} seeds",
    )
    put(
        "first.balanced_over_fanin_T.cell_medians",
        f"{cell_med.min():.3f}-{cell_med.max():.3f}",
        f"{len(cell_med)} cells",
    )
    put("first.balanced_keeps_r1", f"{int((m.r1_f == m.r1_b).sum())}/{len(m)}")

    # ---- leading-order sweeps -----------------------------------------------------------------
    ld = pd.read_csv(res / "lo_d.csv")
    for (tree, d), g in ld.groupby(["tree", "d"]):
        med, lo, hi = median_ci(g["T"])
        sd = float(np.log(g["T"]).std(ddof=1))
        put(
            f"lo_d.{tree}.d{d}.T_median",
            f"{med:.3f} [{lo:.3f}, {hi:.3f}]",
            f"chi={g.chi.iloc[0]}, n={len(g)}",
        )
        put(
            f"lo_d.{tree}.d{d}.sd_lnT",
            f"{sd:.3f} +- {boot(np.log(g['T']), lambda x: np.std(x, ddof=1)):.3f}",
        )
        put(f"lo_d.{tree}.d{d}.tKH_median", f"{np.median(g.t_KH):.4f}")
    for tree, g in ld.groupby("tree"):
        k = g.groupby("d").t_KH.median()
        put(f"lo_d.{tree}.tKH_fall_d2_to_d4", f"{k.loc[2] / k.loc[4]:.2f}")
    put(
        "lo_d.core_fraction_dev_max",
        f"{ld.frac_dev_max.max():.2e}",
        "max over inits of |K x (core squared-norm fraction) - 1| at the end of the correlation flow",
    )

    lc = pd.read_csv(res / "lo_chi.csv")
    lc1 = lc[lc.weights.astype(str) == "1"]
    for tree, g in lc1.groupby("tree"):
        chis_all = sorted(g.chi.unique())
        lo_chi = 4 if tree in ("binary4", "mps4") else 2
        chis = [c for c in chis_all if c >= lo_chi]
        put(
            f"lo_chi.{tree}.slope_T",
            f"{chi_slope(g, chis):.3f} +- {chi_slope_se(g, chis):.3f}",
            f"chi={chis[0]}-{chis[-1]}, single component s=1, 100 inits per chi",
        )
        put(
            f"lo_chi.{tree}.slope_tKH",
            f"{chi_slope(g, chis, 't_KH'):.2f}",
            f"chi={chis[0]}-{chis[-1]}",
        )
        if tree == "mps8":
            t2, t8 = (np.median(g[g.chi == c]["T"]) for c in (2, 8))
            k2, k8 = (np.median(g[g.chi == c].t_KH) for c in (2, 8))
            put("lo_chi.mps8.T_chi2_over_chi8", f"{t2 / t8:.2f}")
            put("lo_chi.mps8.tKH_chi2_over_chi8", f"{k2 / k8:.1f}")

    lk = pd.read_csv(res / "lo_K_table.csv")
    put("lo_K.n_trees", len(lk), f"K={lk.K.min()}-{lk.K.max()}")
    put("lo_K.T_median.range", f"{lk.T_median.min():.3f}-{lk.T_median.max():.0f}")
    put("lo_K.T_over_P3.range", f"{lk.T_over_P3.min():.2f}-{lk.T_over_P3.max():.2f}")
    put(
        "lo_K.T_over_tKH.range",
        f"{lk.T_over_tKH_median.min():.2f}-{lk.T_over_tKH_median.max():.2e}",
    )

    # ---- full runs (main_runs.csv, main_plateaus.csv) ------------------------------------------
    runs = pd.read_csv(res / "main_runs.csv")
    comp = runs[runs["class"] == "complete"]
    for (tree, teacher), g in comp.groupby(["tree", "teacher"]):
        put(
            f"main.s2s_among_complete.{tree}.{teacher}",
            f"{int(g.s2s.sum())}/{len(g)}",
            "every rho_i > 3",
        )
    b4c = comp[(comp.tree == "binary4") & (comp.teacher == "close")]
    for s, g in b4c.groupby("sigma"):
        put(f"main.s2s_among_complete.binary4.close.sigma{s:g}", f"{int(g.s2s.sum())}/{len(g)}")
    for tree in ("binary4", "mps4"):
        g = comp[(comp.tree == tree) & (comp.init == "fanin")]
        put(f"main.s2s_among_complete.{tree}.fanin", f"{int(g.s2s.sum())}/{len(g)}")
    s2s = comp[comp.s2s == True]  # noqa: E712
    for (tree, teacher), g in s2s.groupby(["tree", "teacher"]):
        put(
            f"main.plateau_spectrum_err_max.{tree}.{teacher}",
            f"{g.max_rel_plateau_spectrum_err.max():.2e}",
            "max relative deviation of the top |S| singular values of every edge from the learned weights, "
            "runs with the S2S signature",
        )
        put(
            f"main.plateau_next_sv_over_smin_max.{tree}.{teacher}",
            f"{g.max_plateau_next_sv_over_smin.max():.3f}",
        )
    sep = s2s[(s2s.teacher == "separated") & (s2s.tree == "binary4")]
    for s in (0.003, 0.1):
        put(
            f"main.plateau_spectrum_err_max.binary4.separated.sigma{s:g}",
            f"{sep[sep.sigma == s].max_rel_plateau_spectrum_err.max():.2e}",
        )
    tested = comp.dropna(subset=["edge_rank_change_all_one"])
    put(
        "main.edge_rank_one_step",
        f"{int(tested.edge_rank_change_all_one.sum())}/{len(tested)}",
        "complete runs with event-window spectra in which every event raises the rank by one on every edge",
    )
    exc = tested[tested.edge_rank_change_all_one == 0].groupby(["tree", "teacher"]).size()
    for (tree, teacher), n in exc.items():
        put(f"main.edge_rank_exceptions.{tree}.{teacher}", int(n))
    m8 = runs[runs.tree == "mps8"]
    put(
        "main.mps8.learned",
        "; ".join(f"{k}: {v}" for k, v in m8.n_learned.value_counts().sort_index().items()),
        f"{len(m8)} runs; components learned",
    )
    k7 = runs[(runs.K == 7) & (runs.chi == 3) & (runs.init == "fanin")]
    for tree, g in k7.groupby("tree"):
        put(
            f"main.K7_chi3_fanin_complete.{tree}",
            f"{int((g['class'] == 'complete').sum())}/{len(g)}",
            "sweeps main and k78",
        )

    pl = pd.read_csv(res / "main_plateaus.csv")
    q = pl[pl.rho_next > 3]  # plateaus bracketed by resolved events

    def level(label, sel):
        g = q[sel(q)]
        i = g.loss_rel_err.idxmax()
        r = g.loc[i]
        put(
            f"plateau_level_err.{label}",
            f"{r.loss_rel_err:.2e}",
            f"max over {len(g)} plateaus; at {r.tree} chi={r.chi} {r.teacher} {r.init} sigma={r.sigma:g} "
            f"seed={r.seed} plateau {r.plateau} (rho={r.rho_next:.2f})",
        )

    level("binary4.separated", lambda x: (x.tree == "binary4") & (x.teacher == "separated"))
    level("binary4.close", lambda x: (x.tree == "binary4") & (x.teacher == "close"))
    level("mps4.fanin", lambda x: (x.tree == "mps4") & (x.init == "fanin"))
    level("mps4.balanced", lambda x: (x.tree == "mps4") & (x.init == "balanced"))
    level("K7_8.chi6", lambda x: (x.K >= 7) & (x.chi == 6))
    level("K7_8.chi3", lambda x: (x.K >= 7) & (x.chi == 3))
    level(
        "sigma0.003.not_binary4_close",
        lambda x: (x.sigma == 0.003) & ~((x.tree == "binary4") & (x.teacher == "close")),
    )

    # Wrong-order plateau of binary4, close teacher, chi = 3, seed 8: does the gap grow?
    w = runs[
        (runs.sweep == "main")
        & (runs.tree == "binary4")
        & (runs.chi == 3)
        & (runs.teacher == "close")
        & (runs.seed == 8)
        & (runs.order == "1-3-2")
    ]
    p = -np.polyfit(np.log(w.sigma), np.log(w.t3 - w.t2), 1)[0]
    put(
        "wrong_order.seed8.gap2_exponent",
        f"{p:.2f}",
        f"t3 - t2 ~ sigma^-p over sigma={w.sigma.min():g}-{w.sigma.max():g} ({len(w)} runs)",
    )
    put(
        "wrong_order.seed8.order_at_sigma0.003",
        runs[
            (runs.sweep == "main")
            & (runs.tree == "binary4")
            & (runs.chi == 3)
            & (runs.teacher == "close")
            & (runs.seed == 8)
            & (runs.sigma == 0.003)
        ].order.iloc[0],
    )
    put(
        "wrong_order.rho_next.range",
        f"{pl[(pl.tree == 'binary4') & (pl.chi == 3) & (pl.teacher == 'close') & (pl.learned == '1-3')].rho_next.min():.2f}-"
        f"{pl[(pl.tree == 'binary4') & (pl.chi == 3) & (pl.teacher == 'close') & (pl.learned == '1-3')].rho_next.max():.2f}",
        "binary4 chi=3 close, plateaus with components 1 and 3 learned",
    )

    # Stalled / unresolved runs at their end (slice_trends.csv): smallest dormant route slice / sigma and
    # whether it was still decaying (log-log slope over the last decade below -0.1).
    st = pd.read_csv(res / "slice_trends.csv")
    st["decaying"] = st.log_slope_last_decade < -0.1
    for (sweep, tree, chi, init), g in st.groupby(["sweep", "tree", "chi", "init"]):
        put(
            f"slices.{sweep}.{tree}.chi{chi}.{init}",
            f"median {g.min_slice_end_over_sigma.median():.3g}; decaying {int(g.decaying.sum())}/{len(g)}",
            "smallest dormant slice / sigma at the end of stalled and unresolved runs",
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "report_numbers.csv", index=False, lineterminator="\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
