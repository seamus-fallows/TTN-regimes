"""Fix round 1 of the regime map, run locally (notes/regime_map.md, section "Fix round 1 (local)").

The remote raw outputs (/root/outputs/regime_map) were not available for this round. The runs that
the fixes need are therefore regenerated with the fixed driver (`run_regime.py`: last-crossing
events, tensor-space kernel) from the committed configs (same seeds, teacher and init streams,
probes), and the fix-round tables and figures are computed from them and the committed CSVs:

  python experiments/regime_map/fix_round1.py run      [--raw DIR] [--n-proc 12] [--only GROUP ...]
  python experiments/regime_map/fix_round1.py analyse  [--raw DIR] [--out DIR] [--force]

--raw is the raw-output root (default $REGIME_MAP_OUT). `analyse` works unchanged on the complete
remote folder: pre-fix rows are relabelled from their trajectories and dK_tensor is computed from
theta_end where it is missing (`analysis.relabel_last_crossing`, `analysis.add_tensor_kernel`).
Outputs go to results/fix1/ (existing files are kept unless --force).

Run groups (the cutoffs come from the committed per-seed CSVs, so the selection is reproducible):
  s2s        chi_b4 and sigma_chi_b4, centred: every run with sigma up to 2 x the largest committed
             S2S bracket end, deepest-run prediction or rho = 2 crossing of its chi (chi < 9:
             sigma <= 1.003, below the floor-stalled runs)
  lazy       chi_b4, centred and two-copy (chi <= 64): the runs within a factor 2 of each seed's
             committed (probe-kernel) lazy boundary; the three largest sigma where it is off-grid
  large_chi  large_chi_b4, centred, sigma = 1.5 (tensor-space kernel; theory P16)
  k_sweep    sigma_K_centred_d2, chi = 4: for every seed whose committed boundary was unresolved,
             the runs above its last S2S run, one at a time, until one has min_i rho_i <= 3
Not rerun locally (cost): the rich middle and far-lazy end of the chi_b4 grid, two-copy chi = 128
(P ~ 7e4), and the K = 15 chains (binary16: > 8 min per run); FIX1_K_TREES selects the K-sweep trees.
"""

from __future__ import annotations

import argparse
import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CONFIGS = os.path.join(HERE, "configs")
RESULTS = os.path.join(HERE, "results")
GROUPS = ("s2s", "lazy", "large_chi", "k_sweep")
S2S_FACTOR = 2.0
K_TREES = os.environ.get("FIX1_K_TREES", "mps6,mps8,mps10,mps12,binary16").split(",")


def configs(name: str) -> list[dict]:
    from sweep import expand

    with open(os.path.join(CONFIGS, name + ".yaml")) as f:
        return expand(yaml.safe_load(f))


def committed(name: str) -> pd.DataFrame:
    """A per-seed table as committed before fix round 1 (frozen copy in results/fix1/prefix/)."""
    return pd.read_csv(os.path.join(RESULTS, "fix1", "prefix", name))


# ---- run selection --------------------------------------------------------------------------


def s2s_cutoffs() -> dict:
    """Largest sigma to rerun per (exp, chi) for the S2S relabel: S2S_FACTOR x the largest committed
    S2S bracket end, deepest-run prediction or rho = 2 crossing over the seeds of that chi."""
    cb = committed("boundaries_per_seed_chi_b4.csv")
    cb = cb[cb.model == "centred"]
    out = {}
    for chi, g in cb.groupby("chi"):
        out[("chi_b4", chi)] = S2S_FACTOR * np.nanmax([g.s2s_hi.max(), g.sigma_c_pred.max()])
    sc = committed("sensitivity_per_seed_sigma_chi_b4_centred.csv")
    for chi, g in sc.groupby("chi"):
        top = S2S_FACTOR * np.nanmax(g.s2s_2.values)
        if ("chi_b4", chi) in out:
            top = max(top, out[("chi_b4", chi)])
        out[("sigma_chi_b4", chi)] = min(top, 1.003) if chi < 9 else top
    return out


def select_s2s() -> list[dict]:
    cut = s2s_cutoffs()
    runs = []
    for exp in ("chi_b4", "sigma_chi_b4"):
        for c in configs(exp):
            if c["model"] == "centred" and c["sigma"] <= cut[(exp, c["chi"])] * (1 + 1e-9):
                runs.append(c)
    return runs


def select_lazy() -> list[dict]:
    cb = committed("boundaries_per_seed_chi_b4.csv")
    kb = {(r.model, r.chi, r.seed): r.sigma_K01 for r in cb.itertuples()}
    cfgs = configs("chi_b4")
    runs = []
    for c in cfgs:
        if c["model"] == "twocopy" and c["chi"] > 64:
            continue
        s = kb[(c["model"], c["chi"], c["seed"])]
        if np.isfinite(s):
            if s / 2 <= c["sigma"] <= 2 * s:
                runs.append({**c, "refine": "auto"})
        else:  # boundary above the grid (chi = 9, seed 0): its three largest sigma
            grid = sorted(
                x["sigma"] for x in cfgs if x["model"] == c["model"] and x["chi"] == c["chi"]
            )
            if c["sigma"] >= grid[-3]:
                runs.append({**c, "refine": "auto"})
    return runs


def select_large_chi() -> list[dict]:
    return [
        {**c, "refine": False}
        for c in configs("large_chi_b4")
        if c["model"] == "centred" and c["sigma"] == 1.5
    ]


def k_sweep_chains() -> list[tuple[dict, list[float], float]]:
    """(base cfg, sigma grid above the last S2S run, last S2S sigma) for every unresolved seed of
    the committed centred K sweep (chi = 4); trees in K_TREES (env FIX1_K_TREES)."""
    ps = committed("boundaries_per_seed_sigma_K_centred_d2_chi4.csv")
    cfgs = [c for c in configs("sigma_K_centred_d2") if c["chi"] == 4]
    chains = []
    for r in ps.itertuples():
        if r.s2s_status.startswith("ok") or r.tree.split(" ")[0] not in K_TREES:
            continue
        kind, n = (
            r.tree.split(" ")[0].rstrip("0123456789"),
            int(r.tree.split(" ")[0].lstrip("abcdefghijklmnopqrstuvwxyz")),
        )
        mine = [c for c in cfgs if c["kind"] == kind and c["n"] == n and c["seed"] == r.seed]
        grid = sorted(c["sigma"] for c in mine)
        lo = r.s2s_lo
        if not np.isfinite(lo):
            continue
        above = [s for s in grid if s > lo * (1 + 1e-9)]
        base = {k: v for k, v in mine[0].items() if k != "sigma"}
        chains.append((base, above, float(lo)))
    return chains


def _results(raw: str, exp: str) -> pd.DataFrame:
    import json

    path = os.path.join(raw, exp, "results.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return pd.DataFrame([r for r in rows if "error" not in r])


def k_sweep_next(raw: str) -> list[dict]:
    """Next run of every unfinished chain: the first sigma above the last S2S run that has not been
    run; a chain ends at its first run with a ratio <= 3 or at the end of the grid."""
    done = _results(raw, "sigma_K_centred_d2")
    todo = []
    for base, above, _ in k_sweep_chains():
        for s in above:
            sel = (
                done[
                    (done.kind == base["kind"])
                    & (done.n == base["n"])
                    & (done.seed == base["seed"])
                    & (done.chi == base["chi"])
                    & np.isclose(done.sigma, s)
                ]
                if len(done)
                else done
            )
            if sel.empty:
                todo.append({**base, "sigma": s})
                break
            rhos = [x for x in sel.rhos.iloc[-1] if x is not None]
            if rhos and np.nanmin(rhos) <= 3.0:
                break
    return todo


# ---- run --------------------------------------------------------------------------------------


def cost(c: dict) -> tuple:
    """Longest runs first: big trees, then large chi and the lazy side (many stiff steps)."""
    return (-c["n"], -(c.get("chi") or 0), -c["sigma"])


def run(groups, raw: str, n_proc: int) -> None:
    import run_regime

    if "k_sweep" in groups:
        while True:
            todo = k_sweep_next(raw)
            print(f"k_sweep: {len(todo)} chains to advance", flush=True)
            if not todo:
                break
            run_regime.run_sweep(todo, n_proc=n_proc, order=cost, out=raw)
    cfgs = []
    for g, sel in (
        ("s2s", select_s2s),
        ("lazy", select_lazy),
        ("large_chi", select_large_chi),
    ):
        if g in groups:
            cfgs += sel()
    if cfgs:
        # events are needed on the S2S side only: no refinement for the large-chi runs, and none
        # for runs that end lazy (the first selection that lists a run wins)
        uniq = {}
        for c in cfgs:
            uniq.setdefault((c["exp"], run_regime.run_id(c)), c)
        run_regime.run_sweep(list(uniq.values()), n_proc=n_proc, order=cost, out=raw)


# ---- analysis helpers -------------------------------------------------------------------------


def as_old(df: pd.DataFrame) -> pd.DataFrame:
    """The same runs labelled with the pre-fix event times (first crossing of 1/2)."""
    import analysis as an

    d = df.copy()
    d["rhos"] = d.rhos_old
    d["events"] = [list(zip(t, w)) for t, w in zip(d.event_times_old, d.event_widths)]
    d["min_rho"] = d.min_rho_old
    for kt in an.K_THRESHOLDS:
        for rt in an.RHO_THRESHOLDS:
            d[f"label_{kt}_{rt}"] = [
                an.label(le, dk, rh, kt, rt) for le, dk, rh in zip(d.l_end, d.dK, d.rhos)
            ]
    d["label"] = d["label_0.1_3.0"]
    return d


def geo(v):
    """(geometric mean, standard-error factor, n) of the finite positive values."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.nan, np.nan, 0
    lv = np.log(v)
    se = lv.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
    return float(np.exp(lv.mean())), float(np.exp(se)), int(v.size)


def mse(v):
    """(mean, standard error, n) of the finite values."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan
    return float(v.mean()), se, int(v.size)


def slope_fit(x, y):
    """(slope of log y vs log x over all points, SE from the residual scatter, n, decades in x)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2:
        return np.nan, np.nan, int(ok.sum()), np.nan
    X = np.stack([np.log(x[ok]), np.ones(ok.sum())], 1)
    coef, *_ = np.linalg.lstsq(X, np.log(y[ok]), rcond=None)
    r = np.log(y[ok]) - X @ coef
    cov = (r @ r) / max(ok.sum() - 2, 1) * np.linalg.inv(X.T @ X)
    dec = float(np.log10(x[ok].max() / x[ok].min()))
    return float(coef[0]), float(np.sqrt(cov[0, 0])), int(ok.sum()), dec


def per_seed_slopes(tab, col, chi_min):
    """Slope fitted per seed across chi (a seed shares its teacher across chi): mean, SE, n."""
    s = []
    for _, g in tab[tab.chi >= chi_min].groupby("seed"):
        b, _, n, _ = slope_fit(g.chi, g[col])
        if n >= 3:
            s.append(b)
    return mse(s)


def save(df: pd.DataFrame, out: str, name: str, force: bool) -> None:
    path = os.path.join(out, name)
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists; pass --force")
    df.to_csv(path, index=False)
    print("wrote", path, flush=True)


# ---- A. S2S side: last-crossing relabel (N = 4) -----------------------------------------------


def s2s_analysis(out: str, force: bool) -> dict:
    """Run-level relabel, reproduction check against the committed per-seed CSVs, corrected
    per-seed tables and the corrected S2S columns of table 2.1."""
    import analysis as an

    res = {}
    cb = an.load("chi_b4")
    cb = cb[cb.model == "centred"]
    sc = an.load("sigma_chi_b4")
    sc = sc[sc.model == "centred"]
    rows = []
    for exp, df in (("chi_b4", cb), ("sigma_chi_b4", sc)):
        res[f"{exp}_runs"] = len(df)
        res[f"{exp}_bump_runs"] = int(df.bump.sum())
        res[f"{exp}_label_changed"] = int((df.label != df.label_old).sum())
        for r in df[df.bump | (df.label != df.label_old)].itertuples():
            rows.append(
                dict(
                    exp=exp,
                    chi=r.chi,
                    seed=r.seed,
                    sigma=r.sigma,
                    label_old=r.label_old,
                    label=r.label,
                    min_rho_old=r.min_rho_old,
                    min_rho=r.min_rho,
                    n_up=str(r.n_up),
                )
            )
    save(pd.DataFrame(rows), out, "relabel_runs.csv", force)
    # reproduction check: the pre-fix rule applied to the reruns vs the committed per-seed values
    com = committed("boundaries_per_seed_chi_b4.csv")
    com = com[com.model == "centred"]
    old = an.per_seed_table(as_old(cb), ["model", "chi"], 3)
    new = an.per_seed_table(cb, ["model", "chi"], 3)
    m = com.merge(old, on=["model", "chi", "seed"], suffixes=("_com", "_old"))
    for col in ("sigma_c", "sigma_c_pred", "sigma_K01", "rho_slope"):
        rel = np.abs(m[f"{col}_old"] / m[f"{col}_com"] - 1)
        res[f"repro_chi_b4_{col}_maxrel"] = float(np.nanmax(rel))
    res["repro_chi_b4_status_equal"] = int((m.s2s_status_old == m.s2s_status_com).sum())
    cs = committed("sensitivity_per_seed_sigma_chi_b4_centred.csv")
    old_s = an.sensitivity_table(as_old(sc), ["chi"])
    ms = cs.merge(old_s, on=["chi", "seed"], suffixes=("_com", "_old"))
    for col in ("s2s_2", "s2s_3", "s2s_5"):
        rel = np.abs(ms[f"{col}_old"] / ms[f"{col}_com"] - 1)
        res[f"repro_sigma_chi_{col}_maxrel"] = float(np.nanmax(rel))
        res[f"repro_sigma_chi_{col}_nan_equal"] = int(
            (ms[f"{col}_old"].isna() == ms[f"{col}_com"].isna()).sum()
        )
    # corrected per-seed tables
    new["pred_over_measured"] = new.sigma_c_pred / new.sigma_c
    keep = [
        "model",
        "chi",
        "seed",
        "sigma_c",
        "s2s_status",
        "s2s_lo",
        "s2s_hi",
        "sigma_c_pred",
        "pred_over_measured",
        "rho_slope",
        "n_s2s",
        "n_runs",
        "n_bump",
        "n_relabelled",
    ]
    save(new[keep], out, "s2s_per_seed_chi_b4.csv", force)
    new_s = an.sensitivity_table(sc, ["chi"])
    ps = an.per_seed_table(sc, ["chi"], 3)
    new_s = new_s.merge(
        ps[["chi", "seed", "sigma_c_pred", "rho_slope", "n_s2s"]], on=["chi", "seed"]
    )
    new_s["pred_over_measured"] = new_s.sigma_c_pred / new_s.s2s_3
    save(
        new_s[
            [
                "chi",
                "seed",
                "s2s_2",
                "s2s_3",
                "s2s_5",
                "sigma_c_pred",
                "pred_over_measured",
                "rho_slope",
                "n_s2s",
                "n_runs",
                "n_bump",
            ]
        ],
        out,
        "s2s_per_seed_sigma_chi_b4.csv",
        force,
    )
    # table 2.1 (sigma_chi_b4, centred): S2S columns, new and committed
    t21 = []
    for chi, g in new_s.groupby("chi"):
        c = cs[cs.chi == chi]
        gm, f, n = geo(g.s2s_3)
        gm0, f0, n0 = geo(c.s2s_3)
        pm, pf, _ = geo(g.sigma_c_pred)
        r, rse, _ = mse(g.pred_over_measured)
        e, ese, _ = mse(g.rho_slope)
        q, qse, _ = mse(g.s2s_2 / g.s2s_5)
        t21.append(
            dict(
                chi=chi,
                s2s_boundary=gm,
                se_factor=f,
                n_seeds=n,
                committed=gm0,
                committed_se_factor=f0,
                committed_n=n0,
                pred_deepest=pm,
                pred_se_factor=pf,
                pred_over_meas=r,
                pred_over_meas_se=rse,
                pred_over_meas_min=g.pred_over_measured.min(),
                pred_over_meas_max=g.pred_over_measured.max(),
                rho_exponent=e,
                rho_exponent_se=ese,
                ratio_rho2_rho5=q,
                ratio_se=qse,
                n_bump_runs=int(g.n_bump.sum()),
            )
        )
    save(pd.DataFrame(t21), out, "table_2_1_s2s.csv", force)
    res["table_2_1"] = pd.DataFrame(t21)
    res["chi_b4_new"] = new
    return res


# ---- B. lazy side: tensor-space kernel (N = 4, theory P5-P7) -----------------------------------

NORM_T = np.sqrt(14.0)  # ||W*|| for the odeco teacher (3, 2, 1)


def r4_constants(chi):
    """(c1, c2) of theory_regimes R4 for binary N = 4, d = 3, weights (3, 2, 1), SP."""
    return 341.5 / chi**3, 537.0 / chi**3


def p5_bracket(chi, thr=0.1):
    """sigma_lazy of P5 for xi = +1 and -1 (theory_regimes 2.3) and the large-chi asymptote."""
    c1, c2 = r4_constants(chi)
    root = np.sqrt(c1**2 + 4 * c2 * thr)
    xs = [(-c1 + root) / (2 * c2), (c1 + root) / (2 * c2)]
    sig = sorted((NORM_T / x) ** (1 / 3) for x in xs)
    return sig[0], sig[1], (NORM_T**2 * c2 / thr) ** (1 / 6)


def p6_value(chi, thr=0.1):
    """Two-copy lazy boundary of P6: x_c = 2 (thr / c2)^(1/2)."""
    _, c2 = r4_constants(chi)
    return (NORM_T / (2 * np.sqrt(thr / c2))) ** (1 / 3)


def lazy_analysis(out: str, force: bool) -> dict:
    """Probe vs tensor-space kernel at the lazy boundary (chi_b4): per-seed boundaries at the three
    thresholds, comparison with P5 / P6 / P7, delta_K(x) against c2 x^2, slopes vs chi."""
    import analysis as an

    res = {}
    df = an.load("chi_b4")
    rows, runs = [], []
    for model in ("centred", "twocopy"):
        d = df[df.model == model]
        for (chi, seed), g in d.groupby(["chi", "seed"]):
            row = dict(model=model, chi=chi, seed=seed)
            for kt in an.K_THRESHOLDS:
                row[f"probe_{kt:g}"] = an.seed_kernel_boundary(g, kt)
                row[f"tensor_{kt:g}"] = an.seed_kernel_boundary(g, kt, col="dK_tensor")
            rows.append(row)
        for r in d[d.label != "stalled"].itertuples():
            if np.isfinite(r.dK_tensor):
                runs.append(
                    dict(
                        model=model,
                        chi=r.chi,
                        seed=r.seed,
                        sigma=r.sigma,
                        x=NORM_T / r.sigma**3,
                        dK_probe=r.dK,
                        dK_tensor=r.dK_tensor,
                        probe_over_tensor=r.dK / r.dK_tensor,
                        label=r.label,
                    )
                )
    ps = pd.DataFrame(rows)
    runs = pd.DataFrame(runs)
    save(ps, out, "lazy_per_seed_chi_b4.csv", force)
    save(runs[runs.dK_tensor < 1.0], out, "lazy_runs_chi_b4.csv", force)
    summ = []
    for (model, chi), g in ps.groupby(["model", "chi"]):
        lo, hi, asym = p5_bracket(chi)
        s = dict(
            model=model,
            chi=chi,
            n_probe=int(g["probe_0.1"].notna().sum()),
            n_tensor=int(g["tensor_0.1"].notna().sum()),
            probe_med=g["probe_0.1"].median(),
            tensor_med=g["tensor_0.1"].median(),
            probe_med_x_sqrtchi=g["probe_0.1"].median() * np.sqrt(chi),
            tensor_med_x_sqrtchi=g["tensor_0.1"].median() * np.sqrt(chi),
            tensor_min=g["tensor_0.1"].min(),
            tensor_max=g["tensor_0.1"].max(),
            ratio_005_02_probe=(g["probe_0.05"] / g["probe_0.2"]).median(),
            ratio_005_02_tensor=(g["tensor_0.05"] / g["tensor_0.2"]).median(),
        )
        if model == "centred":
            s.update(p5_lo=lo, p5_hi=hi, p5_asymptote=asym)
        else:
            s.update(p6=p6_value(chi))
        near = runs[(runs.model == model) & (runs.chi == chi) & (runs.dK_tensor.between(0.03, 0.3))]
        s["probe_over_tensor_med"] = near.probe_over_tensor.median() if len(near) else np.nan
        s["probe_over_tensor_min"] = near.probe_over_tensor.min() if len(near) else np.nan
        s["probe_over_tensor_max"] = near.probe_over_tensor.max() if len(near) else np.nan
        summ.append(s)
    summ = pd.DataFrame(summ)
    # two-copy / centred ratio of the tensor boundaries per (chi, seed): P6 predicts 2^(-1/3)
    piv = ps.pivot_table(index=["chi", "seed"], columns="model", values="tensor_0.1")
    if {"centred", "twocopy"} <= set(piv.columns):
        rr = (piv.twocopy / piv.centred).groupby("chi").median()
        summ["twocopy_over_centred_tensor"] = summ.chi.map(rr)
    save(summ, out, "lazy_summary_chi_b4.csv", force)
    # slopes vs chi (pooled over seeds, and per seed)
    fits = []
    for model in ("centred", "twocopy"):
        t = ps[ps.model == model]
        for col in ("probe_0.1", "tensor_0.1"):
            for cmin in (16, 32):
                sel = t[t.chi >= cmin]
                b, se, n, dec = slope_fit(sel.chi, sel[col])
                pm, pse, pn = per_seed_slopes(t, col, cmin)
                fits.append(
                    dict(
                        model=model,
                        quantity=col,
                        chi_min=cmin,
                        slope=b,
                        se=se,
                        n=n,
                        decades=dec,
                        per_seed_mean=pm,
                        per_seed_se=pse,
                        n_seeds=pn,
                    )
                )
    save(pd.DataFrame(fits), out, "lazy_slopes_chi_b4.csv", force)
    # delta_K(x) against the R4 form: ratio to c2 x^2 and the xi = 0 value, per run (centred)
    c = runs[runs.model == "centred"].copy()
    c1, c2 = r4_constants(c.chi.values)
    c["c2x2"] = c2 * c.x**2
    c["r4_xi0"] = np.sqrt((c1 * c.x) ** 2 + (c2 * c.x**2) ** 2)
    c["r4_lo"] = np.sqrt(np.maximum((c1 * c.x) ** 2 - 2 * c1 * c2 * c.x**3 + (c2 * c.x**2) ** 2, 0))
    c["r4_hi"] = np.sqrt((c1 * c.x) ** 2 + 2 * c1 * c2 * c.x**3 + (c2 * c.x**2) ** 2)
    c["in_bracket"] = (c.dK_tensor >= c.r4_lo) & (c.dK_tensor <= c.r4_hi)
    c["tensor_over_xi0"] = c.dK_tensor / c.r4_xi0
    save(c[c.dK_tensor < 1.0], out, "lazy_r4_check_chi_b4.csv", force)
    res.update(summary=summ, fits=pd.DataFrame(fits), r4=c, per_seed=ps)
    return res


# ---- C. S2S boundary vs chi and the collapse variable (claim 2) --------------------------------


def collapse_analysis(s2s: dict, lazy: dict, out: str, force: bool) -> dict:
    """Slopes of the per-seed S2S boundary (last-crossing rule) vs chi, measured and predicted
    sigma_c sqrt(chi), rich window lazy / S2S per seed."""
    new = s2s["chi_b4_new"]
    lz = lazy["per_seed"]
    lz = lz[lz.model == "centred"][["chi", "seed", "probe_0.1", "tensor_0.1"]]
    t = new.merge(lz, on=["chi", "seed"], how="left")
    fits = []
    for col in ("sigma_c", "probe_0.1", "tensor_0.1"):
        for cmin in (9, 16, 32):
            sel = t[t.chi >= cmin]
            b, se, n, dec = slope_fit(sel.chi, sel[col])
            pm, pse, pn = per_seed_slopes(t, col, cmin)
            fits.append(
                dict(
                    quantity=col,
                    chi_min=cmin,
                    slope=b,
                    se=se,
                    n=n,
                    decades=dec,
                    se_from_minus_third=(b + 1 / 3) / se,
                    se_from_minus_half=(b + 0.5) / se,
                    per_seed_mean=pm,
                    per_seed_se=pse,
                    n_seeds=pn,
                )
            )
    fits = pd.DataFrame(fits)
    save(fits, out, "slopes_vs_chi_chi_b4.csv", force)
    rows = []
    for chi, g in t.groupby("chi"):
        rows.append(
            dict(
                chi=chi,
                n_seeds=int(g.sigma_c.notna().sum()),
                sigma_c_sqrtchi_measured=g.sigma_c.median() * np.sqrt(chi),
                sigma_c_sqrtchi_pred=g.sigma_c_pred.median() * np.sqrt(chi),
                pred_over_measured_med=(g.sigma_c_pred / g.sigma_c).median(),
                pred_over_measured_min=(g.sigma_c_pred / g.sigma_c).min(),
                pred_over_measured_max=(g.sigma_c_pred / g.sigma_c).max(),
                lazy_probe_sqrtchi=g["probe_0.1"].median() * np.sqrt(chi),
                lazy_tensor_sqrtchi=g["tensor_0.1"].median() * np.sqrt(chi),
                window_probe=(g["probe_0.1"] / g.sigma_c).median(),
                window_tensor=(g["tensor_0.1"] / g.sigma_c).median(),
            )
        )
    rows = pd.DataFrame(rows)
    save(rows, out, "collapse_chi_b4.csv", force)
    return dict(fits=fits, collapse=rows, table=t)


# ---- D. large chi with the tensor-space kernel (claim 5, P16) ----------------------------------


def large_chi_analysis(out: str, force: bool) -> dict:
    import analysis as an

    df = an.load("large_chi_b4")
    df = df[(df.model == "centred") & np.isclose(df.sigma, 1.5)]
    root = 2  # binary_tree(4): cores 0, 1 bottom, 2 root
    df = df.assign(
        root_change=[d[root] for d in df.dth_core],
        leaf_change=[np.mean([v for i, v in enumerate(d) if i != root]) for d in df.dth_core],
    )
    rows, levels = [], []
    theory = {
        "sp": (-3.0, -1.5, -1.0, -2.5),
        "ntk": (-1.0, -0.5, -1.0, -0.5),
        "mup": (0.0, 0.0, -0.5, 0.0),
    }
    for lr, g in df.groupby("lr"):
        fit = g[g.chi >= 16]
        for q, th in zip(
            ("dK", "dK_tensor", "dth", "root_change", "leaf_change"),
            (theory[lr][0], theory[lr][0], theory[lr][1], theory[lr][2], theory[lr][3]),
        ):
            b, se, n, dec = slope_fit(fit.chi, fit[q])
            rows.append(
                dict(
                    lr=lr,
                    quantity=q,
                    slope=b,
                    se=se,
                    n_runs=n,
                    n_chi=fit.chi.nunique(),
                    decades=dec,
                    theory=th,
                    se_from_theory=(b - th) / se,
                )
            )
        for chi, h in g.groupby("chi"):
            levels.append(
                dict(
                    lr=lr,
                    chi=chi,
                    n=len(h),
                    dK_probe_med=h.dK.median(),
                    dK_tensor_med=h.dK_tensor.median(),
                    dK_tensor_min=h.dK_tensor.min(),
                    dK_tensor_max=h.dK_tensor.max(),
                    probe_over_tensor=(h.dK / h.dK_tensor).median(),
                )
            )
    save(pd.DataFrame(rows), out, "large_chi_slopes_tensor.csv", force)
    save(pd.DataFrame(levels), out, "large_chi_levels.csv", force)
    return dict(slopes=pd.DataFrame(rows), levels=pd.DataFrame(levels))


# ---- E. imbalance check point P20 (claim 6) -----------------------------------------------------


def imbalance_analysis(review_raw: str, out: str, force: bool) -> dict:
    """P20 point (chi = 16, alpha = 3.375) from the reviewers' reruns of the committed code
    (data/review_regime_map/imbalance_b4): delta_K (probe and tensor) vs lambda, per-seed local
    slopes (mean +- SE), saturation level."""
    import analysis as an

    keep = an.OUT
    an.OUT = review_raw
    try:
        df = an.load("imbalance_b4")
        df = an.add_tensor_kernel(df, "imbalance_b4")
    finally:
        an.OUT = keep
    df = df[(df.chi == 16) & np.isclose(df.alpha.astype(float), 3.375)]
    lam = np.sort(df.root_scale.unique())
    rows = []
    for col in ("dK", "dK_tensor"):
        piv = df.pivot_table(index="seed", columns="root_scale", values=col)
        for i, l in enumerate(lam):
            r = dict(
                kernel=col,
                lam=l,
                median=piv[l].median(),
                lo=piv[l].min(),
                hi=piv[l].max(),
                n=int(piv[l].notna().sum()),
                labels=str(dict(df[df.root_scale == l].label.value_counts())),
            )
            if i + 1 < len(lam):
                s = np.log(piv[lam[i + 1]] / piv[l]) / np.log(lam[i + 1] / l)
                r.update(
                    slope_to_next=float(
                        np.log(piv[lam[i + 1]].median() / piv[l].median()) / np.log(lam[i + 1] / l)
                    ),
                    slope_per_seed_mean=s.mean(),
                    slope_per_seed_se=s.std(ddof=1) / np.sqrt(s.size),
                )
            rows.append(r)
    rows = pd.DataFrame(rows)
    save(rows, out, "imbalance_P20.csv", force)
    return dict(table=rows, runs=df)


# ---- F. K sweep: counts, rho classification of floor-stalled runs, law comparison (claim 4) ----


def tree_parts(tree: str):
    name = tree.split(" ")[0]
    kind = name.rstrip("0123456789")
    return kind, int(name[len(kind) :])


def k_sweep_analysis(out: str, force: bool, n_boot: int = 2000) -> dict:
    """Corrected counts of the committed K sweep (centred, chi = 4), brackets of the unresolved
    seeds reclassified by rho from the local reruns (floor-stalled runs count by their min_i rho_i),
    and a comparison of the laws sigma_c^K = const and sigma_c^K / sqrt(D) = const."""
    import analysis as an

    ps = committed("boundaries_per_seed_sigma_K_centred_d2_chi4.csv")
    ps["kind"], ps["n"] = zip(*[tree_parts(t) for t in ps.tree])
    try:
        runs = an.load("sigma_K_centred_d2")
        runs = runs[runs.chi == 4]
    except FileNotFoundError:
        runs = pd.DataFrame(columns=["kind", "n", "seed", "sigma", "min_rho", "label", "l_end"])
    rows = []
    for r in ps.itertuples():
        g = runs[(runs.kind == r.kind) & (runs.n == r.n) & (runs.seed == r.seed)]
        row = dict(
            tree=f"{r.kind}{r.n}",
            kind=r.kind,
            n=r.n,
            K=r.K,
            seed=r.seed,
            committed_sigma_c=r.sigma_c,
            committed_status=r.s2s_status,
            committed_lo=r.s2s_lo,
            committed_hi=r.s2s_hi,
        )
        lo, hi, status = r.s2s_lo, r.s2s_hi, "label" if r.s2s_status.startswith("ok") else "bracket"
        reruns = []
        if not r.s2s_status.startswith("ok") and np.isfinite(r.s2s_lo):
            above = g[g.sigma > r.s2s_lo * (1 + 1e-9)].sort_values("sigma")
            hi, status = np.nan, "bracket (open)"
            for q in above.itertuples():
                reruns.append(f"{q.sigma:g}:{q.label}:rho={q.min_rho:.3g}:l={q.l_end:.2g}")
                if np.isfinite(q.min_rho) and q.min_rho <= 3.0:
                    hi, status = q.sigma, "rho"
                    break
                if np.isfinite(q.min_rho):  # S2S-like (also when floor-stalled): the block goes on
                    lo = q.sigma
            if status != "rho" and len(above) == 0:
                hi, status = r.s2s_hi, "bracket (not rerun)"
        row.update(
            lo=lo,
            hi=hi,
            status=status,
            sigma_c=float(np.sqrt(lo * hi)) if np.isfinite(hi) else np.nan,
            reruns="; ".join(reruns),
        )
        rows.append(row)
    tab = pd.DataFrame(rows)
    save(tab, out, "k_sweep_per_seed.csv", force)
    # per-K summary: counts by status; medians over seeds with a resolved value; lower bounds
    summ = []
    for (kind, n), g in tab.groupby(["kind", "n"]):
        K = int(g.K.iloc[0])
        res_v = g.sigma_c[g.status.isin(["label", "rho"])]
        summ.append(
            dict(
                tree=f"{kind}{n}",
                K=K,
                N=n,
                n_seeds=len(g),
                n_resolved_label=int((g.status == "label").sum()),
                n_resolved_rho=int((g.status == "rho").sum()),
                n_open=int(g.status.str.startswith("bracket").sum()),
                committed_n_unresolved=int(g.committed_status.str.startswith("unresolved").sum()),
                sigma_c_median_resolved=res_v.median() if len(res_v) else np.nan,
                lo_median=g.lo.median(),
                hi_median=g.hi.median(),
                committed_median=g.committed_sigma_c.median(),
            )
        )
    summ = pd.DataFrame(summ).sort_values("K")
    save(summ, out, "k_sweep_summary.csv", force)
    laws = law_comparison(tab, n_boot)
    save(laws, out, "k_sweep_laws.csv", force)
    return dict(per_seed=tab, summary=summ, laws=laws)


def law_comparison(tab: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Fit log u = const to the per-K medians of u = sigma_c^K (law A) and u = sigma_c^K / sqrt(D)
    (law B, D = 2^N), weights from bootstrap errors of the medians (seeds resampled within each
    tree); chi^2, dof and the spread max/min of the medians, for several K ranges."""
    rng = np.random.default_rng(seed)
    out = []
    for name, ok_status in (
        ("resolved seeds", ("label", "rho")),
        (
            "all seeds, bracket midpoints",
            ("label", "rho", "bracket", "bracket (open)", "bracket (not rerun)"),
        ),
    ):
        t = tab[tab.status.isin(ok_status) & np.isfinite(tab.sigma_c)]
        for kmax in (7, 10, 12):
            sel = t[(t.K >= 4) & (t.K <= kmax)]
            if sel.empty:
                continue
            for law in ("A: sigma^K", "B: sigma^K / sqrt(D)"):
                meds, errs, Ks = [], [], []
                for (kind, n), g in sel.groupby(["kind", "n"]):
                    K = int(g.K.iloc[0])
                    lu = K * np.log(g.sigma_c.values) - (
                        n * np.log(2) / 2 if law.startswith("B") else 0
                    )
                    if lu.size < 2:
                        continue
                    boot = [np.median(rng.choice(lu, lu.size)) for _ in range(n_boot)]
                    meds.append(np.median(lu))
                    errs.append(max(np.std(boot), 1e-3))
                    Ks.append(K)
                meds, errs = np.array(meds), np.array(errs)
                w = 1 / errs**2
                c = (w * meds).sum() / w.sum()
                chi2 = float((((meds - c) / errs) ** 2).sum())
                out.append(
                    dict(
                        seeds=name,
                        K_range=f"4-{kmax}",
                        law=law,
                        n_trees=len(meds),
                        u0=float(np.exp(c)),
                        chi2=chi2,
                        dof=len(meds) - 1,
                        spread=float(np.exp(meds.max() - meds.min())),
                        medians=";".join(f"K{k}:{np.exp(m):.3g}" for k, m in zip(Ks, meds)),
                    )
                )
    return pd.DataFrame(out)


# ---- G. committed-CSV checks (two-copy S2S side, table 2.1 kernel columns) --------------------


def csv_checks(out: str, force: bool) -> dict:
    tc = committed("sensitivity_per_seed_sigma_chi_b4_twocopy.csv")
    cc = committed("sensitivity_per_seed_sigma_chi_b4_centred.csv")
    rows = []
    for chi in sorted(cc.chi.unique()):
        a, b = cc[cc.chi == chi], tc[tc.chi == chi]
        gk, fk, nk = geo(a["lazy_0.1"])
        g2, f2, n2 = geo(b["lazy_0.1"])
        rows.append(
            dict(
                chi=chi,
                centred_lazy_probe=gk,
                centred_se_factor=fk,
                centred_n=nk,
                centred_ratio_005_02=mse(a["lazy_0.05"] / a["lazy_0.2"])[0],
                centred_ratio_se=mse(a["lazy_0.05"] / a["lazy_0.2"])[1],
                twocopy_lazy_probe=g2,
                twocopy_n=n2,
                twocopy_s2s3_finite=int(b.s2s_3.notna().sum()),
                twocopy_s2s5_finite=int(b.s2s_5.notna().sum()),
                twocopy_s2s3_max=b.s2s_3.max(),
            )
        )
    rows = pd.DataFrame(rows)
    save(rows, out, "csv_checks_sigma_chi_b4.csv", force)
    return dict(table=rows)


# ---- figures ----------------------------------------------------------------------------------


def fig_s2s(s2s: dict, out: str, force: bool) -> None:
    """Per-seed S2S boundary vs chi (chi_b4): committed (first-crossing events) vs last-crossing
    events, and the deepest-run prediction over the measured boundary per seed."""
    import plots

    new = s2s["chi_b4_new"]
    com = committed("boundaries_per_seed_chi_b4.csv")
    com = com[com.model == "centred"]
    m = new.merge(com[["chi", "seed", "sigma_c"]], on=["chi", "seed"], suffixes=("", "_com"))
    fig, axes = plots.plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    jit = np.exp(0.03 * (m.seed - 2.5))
    ax.scatter(
        m.chi * jit,
        m.sigma_c_com,
        s=22,
        facecolors="none",
        edgecolors=plots.MUTED,
        label="committed (first crossing of 1/2)",
    )
    ax.scatter(
        m.chi * jit, m.sigma_c, s=9, color=plots.COLORS["s2s"], label="fix round 1 (last crossing)"
    )
    ch = m[~np.isclose(m.sigma_c, m.sigma_c_com, equal_nan=True)]
    for r in ch.itertuples():
        ax.annotate(
            "",
            (r.chi, r.sigma_c),
            (r.chi, r.sigma_c_com),
            arrowprops=dict(arrowstyle="->", color=plots.INK, lw=0.8),
        )
    med = m.groupby("chi").sigma_c_pred.median()
    ax.plot(med.index, med.values, "k--", lw=1, label="deepest-run prediction (median of seeds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"S2S boundary $\sigma_c$ (per seed)")
    ax.set_title("chi_b4, centred, 6 seeds: label boundary per seed", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    ax.scatter(m.chi * jit, m.sigma_c_pred / m.sigma_c, s=12, color=plots.COLORS["s2s"])
    ax.axhline(1.0, color=plots.MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel("deepest-run prediction / measured")
    ax.set_title(
        "calibrated on each seed's deepest run (tests rho ~ 1/sigma)", fontsize=9, loc="left"
    )
    fig.tight_layout()
    path = os.path.join(out, "fix1_s2s_chi_b4.png")
    if os.path.exists(path) and not force:
        raise FileExistsError(path)
    fig.savefig(path, bbox_inches="tight")
    plots.plt.close(fig)


def fig_lazy(lazy: dict, out: str, force: bool) -> None:
    """(left) lazy boundary x sqrt(chi) per seed, probe vs tensor-space kernel, with P5 (centred:
    bracket xi = +-1 and asymptote) and P6 (two-copy); (right) delta_K(tensor) / (c2 x^2) vs x
    for the centred runs at chi >= 32, with the R4 bracket."""
    import plots

    ps, summ, r4 = lazy["per_seed"], lazy["summary"], lazy["r4"]
    fig, axes = plots.plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for model, cp, ct in (("centred", "#86b6ef", "#104281"), ("twocopy", "#f5b394", "#a3401a")):
        t = ps[ps.model == model]
        if t.empty:
            continue
        jit = np.exp(0.03 * (t.seed - 2.5))
        ax.scatter(
            t.chi * jit,
            t["probe_0.1"] * np.sqrt(t.chi),
            s=10,
            color=cp,
            marker="s",
            label=f"{model}: probe kernel (label rule)",
        )
        ax.scatter(
            t.chi * jit,
            t["tensor_0.1"] * np.sqrt(t.chi),
            s=12,
            color=ct,
            marker="o",
            label=f"{model}: tensor-space kernel",
        )
    s = summ[summ.model == "centred"].sort_values("chi")
    ax.fill_between(
        s.chi,
        s.p5_lo * np.sqrt(s.chi),
        s.p5_hi * np.sqrt(s.chi),
        color="#2a78d6",
        alpha=0.15,
        lw=0,
        label=r"P5 bracket ($\xi=\pm1$, isotropic $c_1, c_2$)",
    )
    ax.axhline(6.50, color="#104281", ls=":", lw=1, label="P5 asymptote 6.50")
    s2 = summ[summ.model == "twocopy"].sort_values("chi")
    if len(s2):
        ax.plot(s2.chi, s2.p6 * np.sqrt(s2.chi), "--", color="#a3401a", lw=1, label="P6 (two-copy)")
    ax.set_xscale("log")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"lazy boundary $\sigma_{\rm lazy}\sqrt{\chi}$ ($\delta_K$ = 0.1)")
    ax.set_title("chi_b4: lazy boundary per seed", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False, ncol=1)
    ax = axes[1]
    sel = r4[(r4.chi >= 32) & (r4.dK_tensor < 1.0)]
    cmap = plots.plt.get_cmap("viridis")
    chis = sorted(sel.chi.unique())
    for j, chi in enumerate(chis):
        g = sel[sel.chi == chi].sort_values("x")
        ax.scatter(
            g.x,
            g.dK_tensor / g.c2x2,
            s=8,
            color=cmap(j / max(len(chis) - 1, 1)),
            label=f"chi = {chi}",
        )
        xx = np.geomspace(g.x.min(), g.x.max(), 50)
        c1, c2 = r4_constants(chi)
        lo = np.sqrt(np.maximum((c1 * xx) ** 2 - 2 * c1 * c2 * xx**3 + (c2 * xx**2) ** 2, 0)) / (
            c2 * xx**2
        )
        hi = np.sqrt((c1 * xx) ** 2 + 2 * c1 * c2 * xx**3 + (c2 * xx**2) ** 2) / (c2 * xx**2)
        ax.fill_between(xx, lo, hi, color=cmap(j / max(len(chis) - 1, 1)), alpha=0.12, lw=0)
    ax.axhline(1.0, color=plots.INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x=\Vert W^*\Vert/\sigma^3$")
    ax.set_ylabel(r"$\delta_K$ (tensor) / $c_2x^2$")
    ax.set_title("centred: measured vs R4 (bands: xi in [-1, 1])", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    path = os.path.join(out, "fix1_lazy_tensor_chi_b4.png")
    if os.path.exists(path) and not force:
        raise FileExistsError(path)
    fig.savefig(path, bbox_inches="tight")
    plots.plt.close(fig)


def fig_large_imbalance(large: dict, imb: dict, out: str, force: bool) -> None:
    """(left) delta_K vs chi at sigma = 1.5 (centred) for SP / NTK / muP-TTN, probe vs tensor-space
    kernel; (right) delta_K vs root scale lambda at the P20 point, probe vs tensor."""
    import plots

    fig, axes = plots.plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    lv = large["levels"]
    for lr, g in lv.groupby("lr"):
        c = plots.PARAM_COLORS[lr]
        g = g.sort_values("chi")
        ax.loglog(
            g.chi,
            g.dK_tensor_med,
            "o-",
            color=c,
            ms=4,
            label=f"{plots.PARAM_NAMES[lr]}: tensor kernel",
        )
        ax.loglog(g.chi, g.dK_probe_med, "s:", color=c, ms=3, mfc="white", label="probe kernel")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"$\delta_K(T)$ (median of 6 seeds)")
    ax.set_title("large_chi_b4, centred, sigma = 1.5", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    t = imb["table"]
    for col, c, m, lab in (
        ("dK", "#86b6ef", "s", "probe kernel"),
        ("dK_tensor", "#104281", "o", "tensor-space kernel"),
    ):
        g = t[t.kernel == col].sort_values("lam")
        ax.loglog(g.lam, g["median"], m + "-", color=c, ms=4, label=f"median, {lab}")
        ax.fill_between(g.lam, g.lo, g.hi, color=c, alpha=0.15, lw=0)
    g = t[t.kernel == "dK_tensor"].sort_values("lam")
    lam = g.lam.values[:4]
    ax.loglog(lam, g["median"].values[0] * (lam / lam[0]) ** 2, "k--", lw=1, label="slope 2 (P20)")
    ax.axhspan(0.6, 0.7, color=plots.MUTED, alpha=0.2, lw=0, label="P20 saturation 0.6-0.7")
    ax.axvline(4.0, color=plots.MUTED, ls=":", lw=0.8)
    ax.set_xlabel(r"root scale $\lambda$ ($\chi=16$, $\alpha=3.375$; dotted: $\sqrt\chi$)")
    ax.set_ylabel(r"$\delta_K(T)$")
    ax.set_title("imbalance check point P20 (6 seeds; band: range)", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    path = os.path.join(out, "fix1_large_chi_imbalance.png")
    if os.path.exists(path) and not force:
        raise FileExistsError(path)
    fig.savefig(path, bbox_inches="tight")
    plots.plt.close(fig)


def fig_k_sweep(k: dict, out: str, force: bool) -> None:
    """Per-seed brackets of the centred K sweep (d = 2, chi = 4): committed label brackets and the
    brackets after classifying floor-stalled runs by min_i rho_i; laws A and B fitted to the
    resolved medians."""
    import plots

    t = k["per_seed"]
    fig, ax = plots.plt.subplots(figsize=(6.5, 4.0))
    for r in t.itertuples():
        x = r.K + 0.09 * (r.seed - 2.5)
        c = "#2a78d6" if r.kind == "binary" else "#eb6834"
        if np.isfinite(r.committed_lo) and np.isfinite(r.committed_hi):
            ax.plot(
                [x - 0.03, x - 0.03],
                [r.committed_lo, r.committed_hi],
                color=plots.MUTED,
                lw=2,
                alpha=0.5,
            )
        if np.isfinite(r.hi):
            ax.plot([x, x], [r.lo, r.hi], color=c, lw=1.2)
            ax.plot(
                x,
                np.sqrt(r.lo * r.hi),
                "o",
                color=c,
                ms=3,
                mfc=c if r.status in ("label", "rho") else "white",
            )
        elif np.isfinite(r.lo):
            ax.annotate(
                "", (x, r.lo * 1.3), (x, r.lo), arrowprops=dict(arrowstyle="->", color=c, lw=1)
            )
    ax.axhline(np.sqrt(2), color=plots.MUTED, ls=":", lw=1)
    ax.axhline(1.0, color=plots.MUTED, ls="--", lw=0.8)
    ax.plot([], [], color=plots.MUTED, lw=2, alpha=0.5, label="committed label bracket")
    ax.plot([], [], "o-", color="#eb6834", ms=3, label="fix round 1 (rho classification), MPS")
    ax.plot([], [], "o-", color="#2a78d6", ms=3, label="binary tree")
    ax.plot([], [], marker=r"$\uparrow$", ls="none", color=plots.INK, label="open: lower bound")
    ax.set_yscale("log")
    ax.set_xlabel("number of cores K")
    ax.set_ylabel(r"S2S boundary $\sigma_c$ (per seed)")
    ax.set_title(
        "centred, d = 2, chi = 4: brackets per seed (dotted: sqrt(2))", fontsize=9, loc="left"
    )
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out, "fix1_k_sweep.png")
    if os.path.exists(path) and not force:
        raise FileExistsError(path)
    fig.savefig(path, bbox_inches="tight")
    plots.plt.close(fig)


# ---- driver -----------------------------------------------------------------------------------


def analyse(raw: str, out: str, force: bool, review_raw: str | None = None) -> None:
    """Every fix-round table and figure, plus the standard figures that the local data support
    (summary.png, sigma_c_vs_K.png/.csv, boundaries_vs_chi_chi_b4.png, the large-chi sigma = 1.5
    centred figure) regenerated into results/ with the fixed code."""
    import json

    import analysis as an
    import plots

    an.OUT = raw
    os.makedirs(out, exist_ok=True)
    review_raw = review_raw or os.path.join(
        os.path.dirname(os.path.normpath(raw)), "review_regime_map"
    )
    summary = {}
    s2s = s2s_analysis(out, force)
    summary["s2s"] = {k: v for k, v in s2s.items() if not isinstance(v, pd.DataFrame)}
    lazy = lazy_analysis(out, force)
    col = collapse_analysis(s2s, lazy, out, force)
    large = large_chi_analysis(out, force)
    imb = imbalance_analysis(review_raw, out, force)
    k = k_sweep_analysis(out, force)
    csv_checks(out, force)
    fig_s2s(s2s, out, force)
    fig_lazy(lazy, out, force)
    fig_large_imbalance(large, imb, out, force)
    fig_k_sweep(k, out, force)
    # standard figures and tables regenerated with the fixed code (results/)
    plots.RES, plots.FORCE = RESULTS, force
    chi_tab = lazy["per_seed"]
    chi_tab = chi_tab[chi_tab.model == "centred"].rename(
        columns={"probe_0.1": "sigma_K01", "tensor_0.1": "sigma_K01_tensor"}
    )
    chi_tab = s2s["chi_b4_new"][["chi", "seed", "sigma_c"]].merge(
        chi_tab[["chi", "seed", "sigma_K01", "sigma_K01_tensor"]], on=["chi", "seed"]
    )
    plots.summary_figure(chi_tab, k["per_seed"], "summary")
    tables = {}
    for label, name, w in (
        (
            "centred, chi = 4 (2x rank)",
            "boundaries_per_seed_sigma_K_centred_d2_chi4.csv",
            (3.0, 2.0),
        ),
        ("plain, chi = 4 (2x rank)", "boundaries_per_seed_sigma_K_d2_chi4.csv", (3.0, 2.0)),
        ("plain, chi = 2 (= rank)", "boundaries_per_seed_sigma_K_d2_chi2.csv", (3.0, 2.0)),
    ):
        t = committed(name)
        t["kind"], t["n"] = zip(*[tree_parts(x) for x in t.tree])
        tables[label] = (t, w)
    plots.fig_sigma_c_vs_K(
        [
            ("centred, chi = 4 (2x rank)", "sigma_K_centred_d2", 4),
            ("plain, chi = 4 (2x rank)", "sigma_K_d2", 4),
            ("plain, full bonds", "sigma_K_full_d2", "full"),
            ("plain, chi = 2 (= rank)", "sigma_K_d2", 2),
        ],
        tables=tables,
    )
    plots.fig_chi_b4("chi_b4", os.path.join(RESULTS, "lazy_pred_b4.csv"))
    plots.fig_large_chi("large_chi_b4", "centred", 1.5)
    for key, val in (
        ("lazy_summary", lazy["summary"]),
        ("collapse", col["collapse"]),
        ("slopes", col["fits"]),
        ("large_chi", large["slopes"]),
        ("k_summary", k["summary"]),
        ("k_laws", k["laws"]),
    ):
        summary[key] = val.to_dict(orient="records")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("wrote", os.path.join(out, "summary.json"))
# ---- main -------------------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["run", "analyse", "list"])
    ap.add_argument("--raw", default=os.environ.get("REGIME_MAP_OUT", "/root/outputs/regime_map"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "fix1"))
    ap.add_argument("--n-proc", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=list(GROUPS), choices=GROUPS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--review-raw", default=None,
                    help="reviewers' reruns (imbalance P20 point); default <raw>/../review_regime_map")
    a = ap.parse_args()
    os.environ["REGIME_MAP_OUT"] = a.raw
    if a.command == "list":
        for g, sel in (
            ("s2s", select_s2s),
            ("lazy", select_lazy),
            ("large_chi", select_large_chi),
        ):
            print(g, len(sel()))
        print("k_sweep chains", len(k_sweep_chains()), "next:", len(k_sweep_next(a.raw)))
    elif a.command == "run":
        run(a.only, a.raw, a.n_proc)
    else:
        analyse(a.raw, a.out, a.force, a.review_raw)
