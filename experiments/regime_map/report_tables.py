"""Summary tables for the report (Sec. 3 and Sec. 5.1-5.4), from the saved runs.

Writes small CSV tables into experiments/regime_map/results/report/ (the committed copies are what
`plot_report.py` and the report read; neither needs raw data afterwards):

  runs_sigma_chi_b4.csv     one row per run of sigma_chi_b4 (binary N = 4, d = 3; centred, plain, two-copy):
                            label, min rho, kernel changes (probe and tensor-space), final loss and, for
                            centred runs, the Eckart-Young floor of W_0 + W* across the cut (l_best)
  widths_deep_s2s_b4.csv    event widths deep in the S2S regime (centred, chi = 36, sigma = 0.01) and the
                            aligned-mode widths
  floor_sigma_chi_b4.csv    deficient bonds (chi = 3, 4, 6, centred): stall onset, floor check, large-sigma
                            check against the lazy residual predicted at initialization (lazy_pred_b4.csv)
  floor_sigma_chi_b8.csv    the same stall onset for binary N = 8 (d = 3, chi = 3, 6, 24; d = 2)
  imbalance_map_b4.csv      label counts per (chi, alpha, root scale) of imbalance_b4
  k_sweep_runs.csv,         the sigma x K sweeps per run (stalled runs with their stall type) and per
  k_sweep_cells.csv         (series, tree, sigma): label counts, stalled runs split into "all components
                            learned" (a floor) and "some component never learned"
  k_sweep_floor_runs.csv    the first (floor-stalled) run above the label bracket of each unresolved MPS seed
  stall_test_runs.csv,      stall test at chi = teacher rank (K = 7, 8): outcome and stall type from the bond
  stall_test_summary.csv    spectra (`analysis.stall_type`: R-th singular value < 1e-12 dead, >= 1e-4 long
                            plateau, decaying in between)
  large_chi_runs.csv        one row per run of large_chi_b4: kernel changes, parameter and core changes
  inventory.csv             runs per sweep: labels, transients above 1/2, labels changed by the
                            last-crossing event rule of fix round 1

Labels are those of the fix round (`analysis.load`: last-crossing events; the label rule uses the
128-probe kernel `dK`); `dK_tensor` (tensor-space kernel J D J^T) comes from the runs or from the
fix-round cache.

Run from the repository root (raw outputs: the copy of the remote outputs; cache: fix round 1):
    uv run python experiments/regime_map/report_tables.py --raw data/regime_map \
        --tensor-cache data/regime_map_local/tensor_kernel [--out DIR] [--force]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
warnings.filterwarnings("ignore")

import analysis as an  # noqa: E402

OUT = os.path.join(HERE, "results", "report")
FORCE = False
W_STAR_SQ = 14.0  # ||W*||^2 of the odeco teacher (3, 2, 1)


def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUT, name)
    if os.path.exists(path) and not FORCE:
        raise FileExistsError(f"{path} exists; pass --force or --out")
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {path} ({len(df)} rows)")


def n_up_max(v) -> int:
    return int(np.max(v)) if isinstance(v, (list, np.ndarray)) and len(v) else 0


def first_component(times) -> int:
    """1-based index (components in decreasing weight) of the first component learned; 0 if none."""
    t = np.asarray(times, float)
    return int(np.nanargmin(t)) + 1 if np.isfinite(t).any() else 0


# ---- binary N = 4: sigma x chi ---------------------------------------------------------------


def sigma_chi_b4() -> pd.DataFrame:
    df = an.load("sigma_chi_b4")
    runs = pd.DataFrame(
        dict(
            model=df.model,
            chi=df.chi,
            sigma=df.sigma,
            seed=df.seed,
            label=df.label,
            min_rho=df.min_rho,
            n_events=df.n_events,
            dK=df.dK,
            dK_tensor=df.dK_tensor,
            dK_probe=df.dK_probe if "dK_probe" in df else df.dK,
            kernel=an.LABEL_KERNEL,  # kernel of dK and of the labels (REGIME_MAP_LABEL_KERNEL)
            l_end=df.l_end,
            l_best=df.l_best,
            transient=[n_up_max(v) > 1 for v in df.n_up],
        )
    )
    for kt in an.K_THRESHOLDS:  # labels at every (kernel, separation) threshold pair
        for rt in an.RHO_THRESHOLDS:
            runs[f"label_{kt}_{rt:g}"] = df[f"label_{kt}_{rt}"]
    runs = runs.sort_values(["model", "chi", "sigma", "seed"])
    save(runs, "runs_sigma_chi_b4.csv")
    # event widths deep in the S2S regime (centred, chi = 36, smallest sigma) against the aligned-mode
    # widths w_r = s_r^-(2-2/K) omega_K, omega_3 = I_3(0.1, 0.9) (theory_regimes R8)
    deep = df[(df.model == "centred") & (df.chi == 36) & np.isclose(df.sigma, df.sigma.min())]
    omega3 = an._I(3, 0.1, 0.9)
    rows = []
    for _, r in deep.iterrows():
        for comp, (w, s_r) in enumerate(zip(r.event_widths, (3.0, 2.0, 1.0))):
            rows.append(
                dict(
                    seed=r.seed,
                    sigma=r.sigma,
                    component=comp + 1,
                    weight=s_r,
                    width=w,
                    width_aligned=s_r ** (-(2 - 2 / 3)) * omega3,
                )
            )
    save(pd.DataFrame(rows), "widths_deep_s2s_b4.csv")
    return runs


def p4_sigma_floor(chi: int, d: int = 3, K: int = 3) -> float:
    """theory_regimes P4: sigma where the mean centred floor (1 - dim M / D) sigma^(2K) / ||W*||^2
    reaches the convergence threshold 1e-3 (binary N = 4: dim M = 2 d^2 chi - chi^2, D = d^4)."""
    dim_m, D = 2 * d * d * chi - chi * chi, d**4
    return (1e-3 * W_STAR_SQ / (1 - dim_m / D)) ** (1 / (2 * K))


def floor_b4(runs: pd.DataFrame) -> None:
    lazy = pd.read_csv(os.path.join(HERE, "results", "lazy_pred_b4.csv"))
    rows = []
    for chi in (3, 4, 6):
        g = runs[(runs.model == "centred") & (runs.chi == chi)]
        per = g.groupby("sigma").label.apply(lambda s: (s == "stalled").sum())
        n_seeds = g.groupby("sigma").size()
        st = g[g.label == "stalled"]
        top = g[g.sigma == g.sigma.max()].merge(
            lazy[(lazy.chi == chi)][["seed", "residual"]], on="seed"
        )
        rows.append(
            dict(
                chi=chi,
                n_runs=len(g),
                n_stalled=len(st),
                sigma_first_stalled=per[per > 0].index.min(),
                sigma_all_stalled=per[per == n_seeds].index.min(),
                floor_rel_dev_max=float((st.l_end / st.l_best - 1).abs().max()),
                sigma_floor_P4=p4_sigma_floor(chi),
                lazy_residual_mean_theory=(1 - chi / 9) ** 2,
                sigma_top=g.sigma.max(),
                l_end_top_median=float(top.l_end.median()),
                l_end_top_min=float(top.l_end.min()),
                l_end_top_max=float(top.l_end.max()),
                residual_rel_dev_max=float((top.l_end / top.residual - 1).abs().max()),
                dK_top_max=float(top.dK.max()),
            )
        )
    save(pd.DataFrame(rows), "floor_sigma_chi_b4.csv")


def floor_b8() -> None:
    rows, runs = [], []
    for exp in ("sigma_chi_b8", "sigma_chi_b8_d2"):
        df = an.load(exp)
        bonds = pd.Series("uniform", index=df.index)
        if "bonds" in df.columns:  # explicit bond dims by subtree size; NaN = uniform chi
            bonds = pd.Series(
                [
                    str(b) if isinstance(b, dict) or (isinstance(b, str) and b) else "uniform"
                    for b in df.bonds
                ],
                index=df.index,
            )
        runs.append(
            pd.DataFrame(
                dict(
                    exp=exp,
                    model=df.model,
                    d=df.d,
                    chi=df.chi,
                    bonds=bonds,
                    sigma=df.sigma,
                    seed=df.seed,
                    label=df.label,
                    min_rho=df.min_rho,
                    n_events=df.n_events,
                    dK=df.dK,
                    l_end=df.l_end,
                    stop=df.stop,
                )
            )
        )
        df, bonds = df[df.model == "centred"], bonds[df.model == "centred"]
        for (chi, b), g in df.groupby([df.chi, bonds]):
            per = g.groupby("sigma").label.apply(lambda s: (s == "stalled").sum())
            n = g.groupby("sigma").size()
            lazy = g.groupby("sigma").label.apply(lambda s: (s == "lazy").sum())
            rows.append(
                dict(
                    exp=exp,
                    d=int(g.d.iloc[0]),
                    chi=chi,
                    bonds=b,
                    n_runs=len(g),
                    sigma_min=g.sigma.min(),
                    sigma_max=g.sigma.max(),
                    sigma_first_stalled_above_0p8=per[(per > 0) & (per.index > 0.8)].index.min(),
                    sigma_all_stalled=per[(per == n) & (per.index > 0.8)].index.min(),
                    sigma_first_lazy=lazy[lazy > 0].index.min(),
                    sigma_lazy_majority=lazy[lazy > n / 2].index.min(),
                    n_lazy=int(lazy.sum()),
                )
            )
    save(pd.DataFrame(rows), "floor_sigma_chi_b8.csv")
    save(
        pd.concat(runs).sort_values(["exp", "model", "chi", "bonds", "sigma", "seed"]),
        "runs_sigma_chi_b8.csv",
    )


# ---- imbalance ---------------------------------------------------------------------------------


def label_counts(g: pd.DataFrame) -> dict:
    c = g.label.value_counts()
    return dict(
        n=len(g),
        **{f"n_{r}": int(c.get(r, 0)) for r in an.REGIMES},
        majority=c.index[0],
        n_majority=int(c.iloc[0]),
    )


def imbalance() -> None:
    df = an.load("imbalance_b4")
    df["alpha"] = [float(f"{a:.4g}") for a in df.alpha]
    rows = []
    for (chi, alpha, lam), g in df.groupby(["chi", "alpha", "root_scale"]):
        rows.append(
            dict(
                chi=chi,
                alpha=alpha,
                root_scale=lam,
                lam_rel=lam / np.sqrt(chi),
                **label_counts(g),
                n_stalled_kernel_like=int(((g.label == "stalled") & (g.dK < 0.1)).sum()),
                dK_median=g.dK.median(),
                dK_tensor_median=g.dK_tensor.median() if "dK_tensor" in g else np.nan,
            )
        )
    save(pd.DataFrame(rows), "imbalance_map_b4.csv")


# ---- sigma x K -----------------------------------------------------------------------------------

K_SERIES = (  # (series, exp, row filter)
    ("centred, chi = 4", "sigma_K_centred_d2", lambda d: d.chi == 4),
    (
        "centred, full bonds",
        "sigma_K_centred_d2",
        lambda d: (d.chi == 16) | ((d.kind == "mps") & (d.n == 6) & (d.chi == 8)),
    ),
    ("plain, chi = 2", "sigma_K_d2", lambda d: d.chi == 2),
    ("plain, chi = 4", "sigma_K_d2", lambda d: d.chi == 4),
    ("plain, full bonds", "sigma_K_full_d2", lambda d: d.chi > 0),
    ("plain d = 3, chi = 3", "sigma_K", lambda d: d.chi == 3),
    ("plain d = 3, chi = 6", "sigma_K", lambda d: d.chi == 6),
)


def k_sweep() -> None:
    """Per run (with the stall type of stalled runs from their bond spectra, R = teacher rank) and
    per (series, tree, sigma) cell."""
    runs, cells = [], []
    for series, exp, sel in K_SERIES:
        df = an.load(exp)
        df = df[sel(df)]
        R = 2 if int(df.d.iloc[0]) == 2 else 3
        for _, r in df.iterrows():
            st = an.stall_type(exp, r.to_dict(), R=R) if r.label == "stalled" else {}
            runs.append(
                dict(
                    series=series,
                    exp=exp,
                    d=int(r.d),
                    chi=r.chi,
                    tree=f"{r.kind}{r.n}",
                    K=r.K,
                    sigma=r.sigma,
                    seed=r.seed,
                    label=r.label,
                    n_events=r.n_events,
                    all_learned=bool(r.n_events >= R),
                    first_learned=first_component(r.event_times),
                    min_rho=r.min_rho,
                    l_end=r.l_end,
                    dK=r.dK,
                    stop=r.stop,
                    sR_end=st.get("sR_end", np.nan),
                    stall_kind=st.get("stall_kind", "").replace("n/a", "unclassified"),
                )
            )
        for (kind, n, K, sigma), g in df.groupby(["kind", "n", "K", "sigma"]):
            st = g[g.label == "stalled"]
            cells.append(
                dict(
                    series=series,
                    exp=exp,
                    d=int(g.d.iloc[0]),
                    tree=f"{kind}{n}",
                    kind=kind,
                    n_sites=n,
                    K=K,
                    sigma=sigma,
                    **label_counts(g),
                    n_stalled_all_learned=int((st.n_events >= R).sum()),
                    n_stalled_not_learned=int((st.n_events < R).sum()),
                )
            )
    runs = pd.DataFrame(runs).sort_values(["series", "K", "tree", "sigma", "seed"])
    save(runs, "k_sweep_runs.csv")
    save(pd.DataFrame(cells), "k_sweep_cells.csv")
    # the first run above the label bracket of every unresolved MPS seed (centred, chi = 4; fix round 1,
    # results/fix1/k_sweep_per_seed.csv): floor-stalled; S2S-like when min rho > 3 with both learned
    ps = pd.read_csv(os.path.join(HERE, "results", "fix1", "k_sweep_per_seed.csv"))
    unres = ps[(ps.kind == "mps") & ps.committed_status.str.startswith("unresolved")]
    c = runs[runs.series == "centred, chi = 4"]
    rows = []
    for _, u in unres.iterrows():
        g = c[(c.tree == u.tree) & (c.seed == u.seed) & (c.sigma > u.committed_lo * 1.0001)]
        if g.empty:
            continue
        r = g.sort_values("sigma").iloc[0]
        rows.append(
            dict(
                tree=u.tree,
                K=r.K,
                seed=u.seed,
                bracket_lo=u.committed_lo,
                sigma=r.sigma,
                label=r.label,
                min_rho=r.min_rho,
                n_events=r.n_events,
                first_learned=r.first_learned,
                l_end=r.l_end,
                s2s_like=bool(r.min_rho > 3 and r.n_events >= 2),
            )
        )
    save(pd.DataFrame(rows), "k_sweep_floor_runs.csv")
    k_sweep_laws_crossing(ps)


def k_sweep_laws_crossing(ps: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> None:
    """The two laws sigma_c^K = const (A) and sigma_c^K / sqrt(D) = const (B) on per-seed boundaries
    whose bracket ends at a crossing (status 'label' or 'rho' in fix1/k_sweep_per_seed.csv), not at a
    stall that learned nothing (round-2 verification, E1-E2); same fit as fix_round1.law_comparison:
    log u = const against the per-K medians, weights from bootstrap errors (seeds resampled per tree)."""
    rng = np.random.default_rng(seed)
    t = ps[ps.status.isin(("label", "rho")) & np.isfinite(ps.sigma_c)]
    rows = []
    for kmax in (7, 10, 12):
        sel = t[(t.K >= 4) & (t.K <= kmax)]
        for law in ("A: sigma^K", "B: sigma^K / sqrt(D)"):
            meds, errs, Ks, ns = [], [], [], []
            for (kind, n), g in sel.groupby(["kind", "n"]):
                lu = int(g.K.iloc[0]) * np.log(g.sigma_c.values) - (
                    n * np.log(2) / 2 if law[0] == "B" else 0
                )
                if lu.size < 2:
                    continue
                boot = [np.median(rng.choice(lu, lu.size)) for _ in range(n_boot)]
                meds.append(np.median(lu))
                errs.append(max(np.std(boot), 1e-3))
                Ks.append(int(g.K.iloc[0]))
                ns.append(lu.size)
            meds, errs = np.array(meds), np.array(errs)
            w = 1 / errs**2
            c = (w * meds).sum() / w.sum()
            rows.append(
                dict(
                    K_range=f"4-{kmax}",
                    law=law,
                    n_trees=len(meds),
                    u0=float(np.exp(c)),
                    chi2=float((((meds - c) / errs) ** 2).sum()),
                    dof=len(meds) - 1,
                    spread=float(np.exp(meds.max() - meds.min())),
                    medians=";".join(
                        f"K{k}:{np.exp(m):.3g}(n={s})" for k, m, s in zip(Ks, meds, ns)
                    ),
                )
            )
    save(pd.DataFrame(rows), "k_sweep_laws_crossing.csv")


# ---- stall test (chi = teacher rank, K = 7, 8) ---------------------------------------------------


def stall_test() -> None:
    df = an.load("stall_test")
    balanced = (
        [b is True or str(b) == "True" for b in df.balance]
        if "balance" in df
        else [False] * len(df)
    )
    init = ["balanced" if b else "fan-in" for b in balanced]
    rows = []
    for (_, r), ini in zip(df.iterrows(), init):
        st = an.stall_type("stall_test", r.to_dict(), R=3) if r.label == "stalled" else {}
        rows.append(
            dict(
                model=r.model,
                tree=f"{r.kind}{r.n}",
                K=r.K,
                init=ini,
                sigma=r.sigma,
                seed=r.seed,
                label=r.label,
                n_events=r.n_events,
                l_end=r.l_end,
                t_end=r.t_end,
                t_max=r.t_max,
                stop=r.stop,
                sR_end=st.get("sR_end", np.nan),
                stall_kind=st.get("stall_kind", "").replace("n/a", "unclassified"),
            )
        )
    runs = pd.DataFrame(rows).sort_values(["model", "tree", "init", "sigma", "seed"])
    save(runs, "stall_test_runs.csv")
    summ = []
    for (model, tree, ini), g in runs.groupby(["model", "tree", "init"]):
        kinds = g.stall_kind.value_counts()
        summ.append(
            dict(
                model=model,
                tree=tree,
                init=ini,
                n=len(g),
                converged=int((g.label != "stalled").sum()),
                labels=";".join(f"{k}:{v}" for k, v in g.label.value_counts().items()),
                dead_route=int(kinds.get("dead route", 0)),
                decaying=int(kinds.get("decaying", 0)),
                long_plateau=int(kinds.get("long plateau", 0)),
                not_classified=int(kinds.get("unclassified", 0)),
                sR_end_median_stalled=float(g[g.label == "stalled"].sR_end.median()),
                sigmas=";".join(f"{s:g}" for s in sorted(g.sigma.unique())),
            )
        )
    save(pd.DataFrame(summ), "stall_test_summary.csv")


# ---- large chi -------------------------------------------------------------------------------------


def large_chi() -> None:
    df = an.load("large_chi_b4")
    root = 2  # binary_tree(4): cores 0, 1 bottom, 2 root
    runs = pd.DataFrame(
        dict(
            model=df.model,
            sigma=df.sigma,
            lr=df.lr,
            chi=df.chi,
            seed=df.seed,
            label=df.label,
            dK=df.dK,
            dK_tensor=df.dK_tensor,
            dth=df.dth,
            root_change=[d[root] for d in df.dth_core],
            leaf_change=[np.mean([v for i, v in enumerate(d) if i != root]) for d in df.dth_core],
        )
    ).sort_values(["model", "sigma", "lr", "chi", "seed"])
    save(runs, "large_chi_runs.csv")


# ---- inventory of the library sweeps ------------------------------------------------------------

LIBRARY_SWEEPS = (
    "sigma_chi_b4",
    "chi_b4",
    "sigma_chi_b8",
    "sigma_chi_b8_d2",
    "sigma_K",
    "sigma_K_d2",
    "sigma_K_full_d2",
    "sigma_K_centred_d2",
    "imbalance_b4",
    "large_chi_b4",
    "large_chi_b8",
    "stall_test",
)


def inventory() -> None:
    """Runs per sweep (after de-duplication), labels, transients above 1/2 before a component is
    learned (n_up > 1) and labels changed by the last-crossing rule of fix round 1."""
    rows = []
    for exp in LIBRARY_SWEEPS:
        df = an.load(exp)
        lab = df.label.value_counts()
        rows.append(
            dict(
                exp=exp,
                n_runs=len(df),
                models=";".join(sorted(df.model.unique())),
                trees=";".join(sorted({f"{k}{n}" for k, n in zip(df.kind, df.n)})),
                d=";".join(str(x) for x in sorted(df.d.unique())),
                chi=";".join(str(x) for x in sorted(df.chi.unique())),
                seeds=df.seed.nunique(),
                sigma_min=df.sigma.min(),
                sigma_max=df.sigma.max(),
                **{f"n_{r}": int(lab.get(r, 0)) for r in an.REGIMES},
                n_transient=int(sum(n_up_max(v) > 1 for v in df.n_up)) if "n_up" in df else 0,
                # repeated upward crossings of 1/2 in a component that is then learned (it reaches 0.9:
                # finite final-rise width); the rest cross repeatedly only in a component never learned
                n_transient_learned=int(
                    sum(
                        any(n > 1 and np.isfinite(w) for n, w in zip(nu, ws))
                        for nu, ws in zip(df.n_up, df.event_widths)
                    )
                )
                if "n_up" in df
                else 0,
                n_label_changed=int((df.label != df.label_old).sum()) if "label_old" in df else 0,
                integrators=";".join(sorted(df.integrator.dropna().unique()))
                if "integrator" in df
                else "",
                drift_max=float(df.q_drift.max()) if "q_drift" in df else np.nan,
                drift_median=float(df.q_drift.median()) if "q_drift" in df else np.nan,
                stop_max_steps=int((df.stop == "max_steps").sum()) if "stop" in df else 0,
            )
        )
    save(pd.DataFrame(rows), "inventory.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", required=True, help="raw-output root (e.g. data/regime_map)")
    ap.add_argument(
        "--tensor-cache", default=None, help="fix-round cache of dK_tensor (dir of <exp>.csv)"
    )
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    an.OUT = os.path.abspath(a.raw)
    if a.tensor_cache:
        an.TENSOR_CACHE = os.path.abspath(a.tensor_cache)
    OUT, FORCE = a.out, a.force
    steps = {
        "sigma_chi_b4": lambda: floor_b4(sigma_chi_b4()),
        "sigma_chi_b8": floor_b8,
        "imbalance": imbalance,
        "k_sweep": k_sweep,
        "stall_test": stall_test,
        "large_chi": large_chi,
        "inventory": inventory,
    }
    for name in a.only or list(steps):
        print("==", name, flush=True)
        steps[name]()
