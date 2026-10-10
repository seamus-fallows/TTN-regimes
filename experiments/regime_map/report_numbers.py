"""Derived numbers quoted in the report (Sec. 3 and Sec. 5.1-5.4), from committed tables only.

Every number of those sections is either an entry of a committed table (results/*.csv, results/fix1/*.csv,
results/report/*.csv) or one of the aggregates computed here. Writes results/report/numbers.csv with one row
per number: key, value, and the table it comes from.

Run from the repository root:  uv run python experiments/regime_map/report_numbers.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROWS: list[dict] = []


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RES, name), keep_default_na=False, na_values=[""])


def put(key: str, value, source: str) -> None:
    ROWS.append(dict(key=key, value=value, source=source))


def gmean(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.log(v).mean()))


def sec3_sigma_chi() -> None:
    # per-seed rho exponents on the S2S side, both binary4 sweeps
    a = read("fix1/s2s_per_seed_chi_b4.csv").rho_slope.dropna()
    b = read("fix1/s2s_per_seed_sigma_chi_b4.csv").rho_slope.dropna()
    p = -pd.concat([a, b])
    main = p[(p >= 1.0145) & (p <= 1.0485)]  # 1.015-1.048 after rounding
    put("rho exponent: fits", len(p), "fix1/s2s_per_seed_*.csv")
    put("rho exponent: fits in 1.015-1.048 (rounded)", len(main), "fix1/s2s_per_seed_*.csv")
    put(
        "rho exponent: exceptions min-max",
        f"{p[p > 1.0485].min():.3f}-{p[p > 1.0485].max():.3f}",
        "fix1/s2s_per_seed_*.csv",
    )
    r = read("fix1/s2s_per_seed_chi_b4.csv").pred_over_measured
    put(
        "deepest-run ratio: range of 53 seeds",
        f"{r[r < 2].min():.3f}-{r[r < 2].max():.3f}",
        "fix1/s2s_per_seed_chi_b4.csv",
    )
    put("deepest-run ratio: outlier", f"{r.max():.3f}", "fix1/s2s_per_seed_chi_b4.csv")
    w = read("report/widths_deep_s2s_b4.csv")
    for comp, g in w.groupby("component"):
        put(
            f"deep S2S width, component {comp}: range (aligned)",
            f"{g.width.min():.4f}-{g.width.max():.4f} ({g.width_aligned.iloc[0]:.4f})",
            "report/widths_deep_s2s_b4.csv",
        )
    # lazy boundary against P5, R4 check, first-order estimate
    ls = read("fix1/lazy_summary_chi_b4.csv")
    c = ls[ls.model == "centred"]
    above = c[c.chi <= 16].apply(lambda r: r.tensor_med / r.p5_hi - 1, axis=1)
    put(
        "lazy boundary above the P5 bracket at chi = 9-16 (%)",
        ";".join(f"{100 * x:.1f}" for x in above),
        "fix1/lazy_summary_chi_b4.csv",
    )
    put(
        "lazy boundary below P5 lower edge at chi = 128 (%)",
        f"{100 * (1 - c[c.chi == 128].tensor_med.iloc[0] / c[c.chi == 128].p5_lo.iloc[0]):.3f}",
        "fix1/lazy_summary_chi_b4.csv",
    )
    r4 = read("fix1/lazy_r4_check_chi_b4.csv")
    r4 = r4[(r4.model == "centred") & (r4.chi >= 32) & (r4.dK_tensor > 0.03) & (r4.dK_tensor < 0.3)]
    put("near-boundary runs (chi >= 32, 0.03 < dK < 0.3)", len(r4), "fix1/lazy_r4_check_chi_b4.csv")
    put(
        "probe/tensor dK: min, median, max",
        f"{r4.probe_over_tensor.min():.2f}, {r4.probe_over_tensor.median():.2f}, {r4.probe_over_tensor.max():.2f}",
        "fix1/lazy_r4_check_chi_b4.csv",
    )
    put(
        "dK / R4 law: min, median, max",
        f"{r4.tensor_over_xi0.min():.2f}, {r4.tensor_over_xi0.median():.2f}, {r4.tensor_over_xi0.max():.2f}",
        "fix1/lazy_r4_check_chi_b4.csv",
    )
    lp = read("lazy_pred_b4.csv")
    fo = lp.groupby("chi").kappa.median().apply(lambda k: (k / 0.1) ** (1 / 3))
    put(
        "first-order lazy estimate (kappa/0.1)^(1/3) at chi = 12, 36",
        f"{fo[12]:.2f}, {fo[36]:.2f}",
        "lazy_pred_b4.csv",
    )
    sc = read("sensitivity_per_seed_sigma_chi_b4_centred.csv")
    put(
        "lazy boundary, tensor kernel, chi = 36 (geometric mean)",
        f"{gmean(sc[sc.chi == 36]['lazy_tensor_0.1']):.3f}",
        "sensitivity_per_seed_sigma_chi_b4_centred.csv",
    )
    # floor below the cut dimension
    fl = read("report/floor_sigma_chi_b4.csv")
    put(
        "stalled centred runs at chi = 3, 4, 6",
        int(fl.n_stalled.sum()),
        "report/floor_sigma_chi_b4.csv",
    )
    put(
        "max |L_end / Eckart-Young - 1| (%)",
        f"{100 * fl.floor_rel_dev_max.max():.3f}",
        "report/floor_sigma_chi_b4.csv",
    )
    put(
        "stall onset / P4 estimate",
        ";".join(f"{x:.2f}" for x in fl.sigma_all_stalled / fl.sigma_floor_P4),
        "report/floor_sigma_chi_b4.csv",
    )
    put(
        "sigma = 40: max |L_end / residual - 1| (%)",
        f"{100 * fl.residual_rel_dev_max.max():.2f}",
        "report/floor_sigma_chi_b4.csv",
    )
    # collapse
    sl = read("fix1/slopes_vs_chi_chi_b4.csv")
    for chi_min in (16, 32):
        s = sl[(sl.quantity == "sigma_c") & (sl.chi_min == chi_min)].iloc[0]
        put(
            f"-1/3 from the S2S slope, chi >= {chi_min}: pooled SE, per-seed SE",
            f"{abs(s.se_from_minus_third):.2f}, {abs(s.per_seed_mean + 1 / 3) / s.per_seed_se:.2f}",
            "fix1/slopes_vs_chi_chi_b4.csv",
        )
    co = read("fix1/collapse_chi_b4.csv")
    co = co[co.chi >= 32]
    put(
        "rich window, chi >= 32: probe, tensor",
        f"{co.window_probe.min():.1f}-{co.window_probe.max():.1f}, {co.window_tensor.min():.1f}-{co.window_tensor.max():.1f}",
        "fix1/collapse_chi_b4.csv",
    )
    put(
        "Gram scale p = chi sigma^2 / d^2 at sigma sqrt(chi) = 0.30, 6.50",
        f"{0.30**2 / 9:.3f}, {6.50**2 / 9:.2f}",
        "(arithmetic)",
    )
    # seven cores
    b8 = read("boundaries_per_seed_sigma_chi_b8_d2.csv")
    u = b8[b8.row.isin(["centred chi=4", "centred chi=8", "centred chi=16"])]
    put(
        "K = 7, d = 2, chi = 4/8/16: median boundary [lo, hi]",
        f"{u.sigma_c.median():.3f} [{u.s2s_lo.median():.3f}, {u.s2s_hi.median():.3f}]",
        "boundaries_per_seed_sigma_chi_b8_d2.csv",
    )
    o = b8[b8.row == "centred chi=8/32"]
    put(
        "K = 7, d = 2, bonds 8/32: median boundary [lo, hi]",
        f"{o.sigma_c.median():.3f} [{o.s2s_lo.median():.3f}, {o.s2s_hi.median():.3f}]",
        "boundaries_per_seed_sigma_chi_b8_d2.csv",
    )
    put(
        "K = 7, bonds 8/32: kernel boundary (median, seeds)",
        f"{o.sigma_K01.median():.3f}, {o.sigma_K01.notna().sum()}",
        "boundaries_per_seed_sigma_chi_b8_d2.csv",
    )
    rb = read("report/runs_sigma_chi_b8.csv")
    big = rb[(rb.exp == "sigma_chi_b8_d2") & (rb.chi == 32)]
    lz = big.groupby("sigma").apply(lambda g: f"{(g.label == 'lazy').sum()}/{len(g)}")
    put(
        "K = 7, bonds 8/32: lazy runs per sigma",
        "; ".join(f"{s:g}:{v}" for s, v in lz.items() if s > 1.3),
        "report/runs_sigma_chi_b8.csv",
    )
    st = big[(big.label == "stalled") & (big.sigma > 1.3)]
    put(
        "K = 7, bonds 8/32: stalled lazy-side runs, L/L0 range, max dK",
        f"{st.l_end.min():.4f}-{st.l_end.max():.4f}, {st.dK.max():.3f}",
        "report/runs_sigma_chi_b8.csv",
    )
    sq = rb[(rb.exp == "sigma_chi_b8_d2") & (rb.chi == 16) & (rb.sigma > 2)]
    put(
        "K = 7, chi = 16 (square), sigma > 2: L/L0 range, stops",
        f"{sq.l_end.min():.3f}-{sq.l_end.max():.3f}, {','.join(sorted(set(sq.stop)))}",
        "report/runs_sigma_chi_b8.csv",
    )
    put(
        "K = 7 rich window (8/32)",
        f"{o.sigma_K01.median() / o.sigma_c.median():.2f}",
        "boundaries_per_seed_sigma_chi_b8_d2.csv",
    )


def sec3_K_and_rest() -> None:
    fr = read("report/k_sweep_floor_runs.csv")
    like = fr[fr.s2s_like.astype(str) == "True"]
    put(
        "unresolved MPS seeds; S2S-like first floor run",
        f"{len(fr)}; {len(like)}",
        "report/k_sweep_floor_runs.csv",
    )
    put(
        "S2S-like: rho range; L/L0 range; smaller first",
        f"{like.min_rho.min():.1f}-{like.min_rho.max():.0f}; {like.l_end.min():.4f}-{like.l_end.max():.3f}; {(like.first_learned == 2).sum()}",
        "report/k_sweep_floor_runs.csv",
    )
    ps = read("fix1/k_sweep_per_seed.csv")
    for K in (8, 10, 12):
        g = ps[ps.K == K]
        cross = g[g.status.isin(["label", "rho"])]
        put(
            f"K = {K}: seeds ending at a crossing (values)",
            f"{len(cross)} ({', '.join(f'{x:.2f}' for x in cross.sigma_c)})",
            "fix1/k_sweep_per_seed.csv",
        )
    cells = read("report/k_sweep_cells.csv")
    c = cells[cells.series == "centred, chi = 4"]
    first = c.loc[c.groupby("tree").sigma.idxmin()].sort_values("K")
    put(
        "centred chi = 4, smallest sigma: stalled / runs",
        "; ".join(f"{r.tree}:{r.n_stalled}/{r.n}" for r in first.itertuples()),
        "report/k_sweep_cells.csv",
    )
    runs = read("report/k_sweep_runs.csv")
    st = runs[(runs.label == "stalled") & runs.series.str.startswith("plain")]
    put(
        "plain K sweeps, stalled: dead, decaying, long plateau, total",
        f"{(st.stall_kind == 'dead route').sum()}, {(st.stall_kind == 'decaying').sum()}, {(st.stall_kind == 'long plateau').sum()}, {len(st)}",
        "report/k_sweep_runs.csv",
    )
    d3 = cells[(cells.series == "plain d = 3, chi = 6")]
    b8 = d3[(d3.tree == "binary8") & (d3.sigma <= 0.64)]
    m8 = d3[d3.tree == "mps8"]
    put(
        "d = 3, chi = 6: binary8 stalled at sigma <= 0.63 (runs); MPS8 stalled / runs",
        f"{b8.n_stalled.sum()} ({b8.n.sum()}); {m8.n_stalled.sum()}/{m8.n.sum()}",
        "report/k_sweep_cells.csv",
    )
    fb = cells[(cells.series == "plain, full bonds") & cells.tree.str.startswith("mps")]
    put(
        "plain full bonds, MPS: max stalled seeds per cell",
        int(fb.n_stalled.max()),
        "report/k_sweep_cells.csv",
    )
    # thresholds and labels
    t21 = read("fix1/table_2_1_s2s.csv")
    put(
        "rho 2 -> 5 ratio range; max s.e.",
        f"{t21.ratio_rho2_rho5.min():.2f}-{t21.ratio_rho2_rho5.max():.2f}; {t21.ratio_se.max():.2f}",
        "fix1/table_2_1_s2s.csv",
    )
    ls = read("fix1/lazy_summary_chi_b4.csv")
    c32 = ls[(ls.model == "centred") & (ls.chi >= 32)]
    put(
        "dK 0.05 -> 0.2 ratio, chi >= 32: probe, tensor",
        f"{c32.ratio_005_02_probe.min():.3f}-{c32.ratio_005_02_probe.max():.3f}, {c32.ratio_005_02_tensor.min():.3f}-{c32.ratio_005_02_tensor.max():.3f}",
        "fix1/lazy_summary_chi_b4.csv",
    )
    r = read("report/runs_sigma_chi_b4.csv")
    cen = r[r.model == "centred"]
    changed = max(int((cen[k] != cen.label).sum()) for k in cen.columns if k.startswith("label_"))
    put(
        "centred binary4 labels; max changed over the 9 threshold pairs",
        f"{len(cen)}; {changed}",
        "report/runs_sigma_chi_b4.csv",
    )
    rich = cen[cen.label == "rich"]
    put(
        "rich runs: dK probe min, median, max",
        f"{rich.dK.min():.3f}, {rich.dK.median():.0f}, {rich.dK.max():.2g}",
        "report/runs_sigma_chi_b4.csv",
    )
    # spread of per-seed boundaries at one chi
    sp1 = sc_spread(read("sensitivity_per_seed_sigma_chi_b4_centred.csv"), "s2s_3")
    b = read("boundaries_per_seed_chi_b4.csv")
    sp2 = sc_spread(b[b.model == "centred"], "sigma_c")
    allsp = pd.concat([sp1, sp2])
    put(
        "per-seed boundary spread (max/min) at chi <= 12; chi >= 16",
        f"{allsp[allsp.index <= 12].min():.1f}-{allsp[allsp.index <= 12].max():.1f}; {allsp[allsp.index >= 16].min():.1f}-{allsp[allsp.index >= 16].max():.1f}",
        "sensitivity_per_seed_sigma_chi_b4_centred.csv, boundaries_per_seed_chi_b4.csv",
    )
    # two-copy model
    tc = r[r.model == "twocopy"]
    t3 = tc[(tc.chi == 3) & (tc.sigma >= 1)].groupby("sigma")
    put(
        "two-copy chi = 3: labels and L/L0 range per sigma",
        "; ".join(
            f"{s:g}:{dict(g.label.value_counts())}:{g.l_end.min():.3g}-{g.l_end.max():.3g}"
            for s, g in t3
        ),
        "report/runs_sigma_chi_b4.csv",
    )
    sm = tc[tc.chi.isin([18, 24, 36]) & (tc.sigma < 0.02)].groupby(["chi", "sigma"])
    put(
        "two-copy, smallest sigma: S2S seeds; max min_rho",
        "; ".join(
            f"chi {c}, {s:g}: {(g.label == 's2s').sum()}, {g.min_rho.max():.1f}" for (c, s), g in sm
        ),
        "report/runs_sigma_chi_b4.csv",
    )
    inv = read("report/inventory.csv")
    put(
        "runs; transients before learning; repeated crossings; labels changed",
        f"{inv.n_runs.sum()}; {inv.n_transient_learned.sum()}; {inv.n_transient.sum()}; {inv.n_label_changed.sum()}",
        "report/inventory.csv",
    )
    # large chi levels
    lc = read("report/large_chi_runs.csv")
    sp = lc[(lc.model == "centred") & (lc.sigma == 1.5) & (lc.lr == "sp") & (lc.chi == 128)]
    put(
        "SP, chi = 128, sigma = 1.5: median root, leaf change",
        f"{sp.root_change.median():.3f}, {sp.leaf_change.median():.2g}",
        "report/large_chi_runs.csv",
    )
    st = read("report/stall_test_runs.csv")
    put(
        "stall test budgets (time units)",
        f"{st.t_max.min():.3g}-{st.t_max.max():.3g}",
        "report/stall_test_runs.csv",
    )


def sc_spread(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("chi")[col].apply(lambda v: v.max() / v.min())


if __name__ == "__main__":
    sec3_sigma_chi()
    sec3_K_and_rest()
    out = os.path.join(RES, "report", "numbers.csv")
    pd.DataFrame(ROWS).to_csv(out, index=False)
    print(pd.DataFrame(ROWS).to_string())
    print("wrote", out)
