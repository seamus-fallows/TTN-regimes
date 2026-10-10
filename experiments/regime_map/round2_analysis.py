"""Round 2 analysis (notes/regime_map.md sections 4.3-4.4): per-seed sigma_c, cell statistics with
bootstrap CIs, the law fits (H_A: beta = 0, H_B: beta = 1) and the d-ratio.

usage: uv run python experiments/regime_map/round2_analysis.py [--raw DIR] [--out DIR] [--force]
Writes per_seed_*.csv, cells_*.csv, fits.csv, ratio_*.csv and round2_*.png to --out
(default results/round2); refuses to overwrite without --force.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import aligned_boundary_a0  # noqa: E402
from round2 import ARMS, OUT, WEIGHTS, run_class  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "round2")
THRS = (2.0, 3.0, 5.0)
RNG = np.random.default_rng(0)
N_BOOT = 10_000


def load(raw: str, arm: str) -> pd.DataFrame:
    path = os.path.join(raw, ARMS[arm]["exp"], "results.jsonl")
    with open(path) as f:
        rows = [r for r in map(json.loads, f) if "error" not in r]
    df = pd.DataFrame(rows)
    df["K"] = df["n"] - 1
    return df.drop_duplicates(subset=["n", "d", "seed", "sigma"], keep="last")


def seed_boundary(rows: pd.DataFrame, K: int, thr: float):
    """(sigma_c, status, lo, hi) for one seed (section 4.1): the first S2S-like -> rich-like
    transition coming from small sigma; log-log interpolation of rho between the bracketing runs
    when both ratios are measured, else the geometric midpoint. status: 'ok'; 'ok_lb' (the S2S-like
    end is a lower bound on rho); '+gap' if an uninformative run lies inside the bracket;
    'never_s2s' / 'all_s2s' (censored: sigma_c below lo / above hi); 'domain' (plain model: S2S-like up
    to lo, and the next run up, hi, starts with an initial overlap >= 0.1, outside the event rule's
    domain: sigma_c > lo, undefined above hi)."""
    r = rows.sort_values("sigma")
    sig = r.sigma.values
    cls = [run_class(x, thr, domain=(x["model"] == "plain")) for x in r.to_dict("records")]
    dom = [sig[i] for i, c in enumerate(cls) if c[0] == "D"]
    if dom:  # runs at or above the smallest out-of-domain sigma carry no event information
        keep = sig < min(dom)
        sig, cls = sig[keep], [c for c, k in zip(cls, keep) if k]
    S = [i for i, c in enumerate(cls) if c[0] == "S"]
    if not S:
        return np.nan, "never_s2s", np.nan, float(sig[0]) if len(sig) else np.nan
    after = [j for j in range(S[0] + 1, len(cls)) if cls[j][0] == "R"]
    if not after:
        if dom:
            return np.nan, "domain", float(sig[S[-1]]), float(min(dom))
        return np.nan, "all_s2s", float(sig[S[-1]]), np.nan
    j = after[0]
    i = max(k for k in S if k < j)
    lo, hi = float(sig[i]), float(sig[j])
    ri, rj = cls[i][1], cls[j][1]
    if np.isfinite(ri) and ri > 0 and np.isfinite(rj) and rj > 0:
        w = (math.log(thr) - math.log(ri)) / (math.log(rj) - math.log(ri))
        sc, status = math.exp(math.log(lo) + w * (math.log(hi) - math.log(lo))), "ok"
    else:
        sc, status = math.sqrt(lo * hi), "ok" if np.isfinite(ri) and ri > 0 else "ok_lb"
    if any(cls[k][0] is None for k in range(i + 1, j)):
        status += "+gap"
    return sc, status, lo, hi


def per_seed(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    out = []
    u0 = {}
    for (n, d, seed), g in df.groupby(["n", "d", "seed"]):
        K = n - 1
        rec = dict(arm=arm, n=n, K=K, d=d, seed=seed, n_runs=len(g), sigma_max_run=g.sigma.max())
        for thr in THRS:
            sc, st, lo, hi = seed_boundary(g, K, thr)
            key = (K, thr)
            if key not in u0:
                u0[key] = aligned_boundary_a0(K, WEIGHTS, rho_thr=thr) ** K
            tag = "" if thr == 3.0 else f"_{thr:g}"
            rec.update({f"sigma_c{tag}": sc, f"status{tag}": st, f"lo{tag}": lo, f"hi{tag}": hi,
                        f"y{tag}": math.log(sc**K / u0[key]) if np.isfinite(sc) else np.nan})
        # local exponent of rho vs sigma over the three largest-sigma S2S-like runs with a measured
        # ratio below the boundary (or below the domain limit), to compare with -(K - 2) (R9)
        recs = sorted(g.to_dict("records"), key=lambda x: x["sigma"])
        top = [x for x in recs if np.isfinite(rec["hi"]) and x["sigma"] < rec["hi"]] if np.isfinite(rec["hi"]) else recs
        top = [x for x in top if run_class(x, 3.0, x["model"] == "plain")[0] == "S"
               and run_class(x, 3.0, x["model"] == "plain")[1] > 0][-3:]
        if len(top) >= 2:
            ls = np.log([x["sigma"] for x in top])
            lr = np.log([x["rhos"][0] for x in top])
            rec["rho_slope_top"] = float(np.polyfit(ls, lr, 1)[0])
            rec["rho_slope_range"] = f"{top[0]['sigma']:.3g}-{top[-1]['sigma']:.3g}"
        # domain limit of this seed: a(0) scales exactly as sigma^K (same init draw), so the sigma at
        # which max_r |a_r(0)| = 0.1 follows from any run
        if "a0" in g and g.a0.notna().all() and (g.model == "plain").all():
            x = g.iloc[0]
            amax = float(np.max(np.abs(x["a0"])))
            rec["sigma_dom"] = float(x["sigma"] * (0.1 / amax) ** (1 / K)) if amax > 0 else np.nan
        # cost of the walker: total wall time and the slowest run
        rec["wall_total"] = g.wall.sum()
        rec["wall_max"] = g.wall.max()
        out.append(rec)
    return pd.DataFrame(out)


def boot(values: np.ndarray, fn, n=N_BOOT):
    v = np.asarray(values, float)
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    return np.array([fn(v[i]) for i in idx])


def cells(ps: pd.DataFrame, thr: float = 3.0) -> pd.DataFrame:
    tag = "" if thr == 3.0 else f"_{thr:g}"
    out = []
    for (K, d), g in ps.groupby(["K", "d"]):
        sc = g[f"sigma_c{tag}"].values
        ok = np.isfinite(sc)
        rec = dict(K=K, d=d, n=len(g), n_res=int(ok.sum()),
                   censored_low=int((g[f"status{tag}"] == "never_s2s").sum()),
                   censored_high=int((g[f"status{tag}"] == "all_s2s").sum()),
                   censored_domain=int((g[f"status{tag}"] == "domain").sum()))
        if ok.sum() >= 1:
            med = float(np.median(sc[ok]))
            rec.update(sigma_c_med=med, sigma_c_min=float(sc[ok].min()), sigma_c_max=float(sc[ok].max()))
            y = g[f"y{tag}"].values[ok]
            rec["y_mean"] = float(y.mean())
            if ok.sum() >= 2:
                b = boot(sc[ok], np.median)
                rec.update(sigma_c_lo=float(np.percentile(b, 2.5)), sigma_c_hi=float(np.percentile(b, 97.5)))
                rec["y_se"] = float(boot(y, np.mean).std())
                # robust variant (not pre-registered): median of y, bootstrap SE of the median
                rec["y_med"] = float(np.median(y))
                rec["y_med_se"] = max(float(boot(y, np.median).std()), 1e-3)
            # random-part shrinkage parameter of H_W at the median boundary
            rec["shrink"] = med**K * d ** ((K + 1) / 2) / (min(WEIGHTS) * d)
        if "sigma_dom" in g:
            rec["sigma_dom_med"] = float(g.sigma_dom.median())
        hi_lb = g.loc[g[f"status{tag}"].isin(["all_s2s", "domain"]), f"lo{tag}"]
        if len(hi_lb):
            rec["censored_high_min_lb"] = float(hi_lb.min())
        out.append(rec)
    return pd.DataFrame(out)


def wls(x, y, se, fixed_beta=None):
    """y = -gamma + beta x, weights 1/se^2. Returns dict(gamma, beta, se_gamma, se_beta, chi2,
    dof, p, birge); SEs scaled by sqrt(chi2/dof) when that exceeds 1."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), 1 / np.asarray(se, float) ** 2
    if fixed_beta is None:
        X = np.column_stack([-np.ones_like(x), x])
        cov = np.linalg.inv(X.T @ (w[:, None] * X))
        coef = cov @ X.T @ (w * y)
        res = y - X @ coef
        dof = len(x) - 2
        gamma, beta = coef
    else:
        beta = fixed_beta
        gamma = -np.sum(w * (y - beta * x)) / np.sum(w)
        res = y - (-gamma + beta * x)
        cov = np.array([[1 / np.sum(w), 0.0], [0.0, 0.0]])
        dof = len(x) - 1
    chi2 = float(np.sum(w * res**2))
    birge = math.sqrt(chi2 / dof) if dof > 0 and chi2 > dof else 1.0
    return dict(gamma=float(gamma), beta=float(beta), se_gamma=float(math.sqrt(cov[0, 0]) * birge),
                se_beta=float(math.sqrt(cov[1, 1]) * birge), chi2=chi2, dof=dof,
                p=float(stats.chi2.sf(chi2, dof)) if dof > 0 else np.nan, birge=birge)


def contrast(cl: pd.DataFrame):
    """beta from Delta y_K = y(3) - y(2) = beta (N/2) ln 1.5 (one parameter, no gamma)."""
    a = cl[cl.d == 2].set_index("K")
    b = cl[cl.d == 3].set_index("K")
    Ks = [K for K in a.index if K in b.index and np.isfinite(a.y_se.get(K, np.nan))
          and np.isfinite(b.y_se.get(K, np.nan))]
    if len(Ks) < 1:
        return None, pd.DataFrame()
    dy = np.array([b.y_mean[K] - a.y_mean[K] for K in Ks])
    se = np.array([math.hypot(b.y_se[K], a.y_se[K]) for K in Ks])
    x = np.array([(K + 1) / 2 * math.log(1.5) for K in Ks])
    w = 1 / se**2
    beta = float(np.sum(w * dy * x) / np.sum(w * x**2))
    chi2 = float(np.sum(w * (dy - beta * x) ** 2))
    dof = len(Ks) - 1
    birge = math.sqrt(chi2 / dof) if dof > 0 and chi2 > dof else 1.0
    fit = dict(gamma=np.nan, beta=beta, se_gamma=np.nan, se_beta=float(1 / math.sqrt(np.sum(w * x**2)) * birge),
               chi2=chi2, dof=dof, p=float(stats.chi2.sf(chi2, dof)) if dof > 0 else np.nan, birge=birge)
    tab = pd.DataFrame(dict(K=Ks, dy=dy, se=se, x=x, beta_K=dy / x))
    return fit, tab


def d_ratio(ps: pd.DataFrame) -> pd.DataFrame:
    out = []
    for K, g in ps.groupby("K"):
        a = g[(g.d == 2) & np.isfinite(g.sigma_c)].sigma_c.values
        b = g[(g.d == 3) & np.isfinite(g.sigma_c)].sigma_c.values
        if len(a) < 2 or len(b) < 2:
            continue
        r = float(np.median(b) / np.median(a))
        ia = RNG.integers(0, len(a), size=(N_BOOT, len(a)))
        ib = RNG.integers(0, len(b), size=(N_BOOT, len(b)))
        bs = np.median(b[ib], axis=1) / np.median(a[ia], axis=1)
        out.append(dict(K=K, r=r, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                        r_HA=1.0, r_HB=1.5 ** ((K + 1) / (2 * K)), n2=len(a), n3=len(b)))
    return pd.DataFrame(out)


def all_fits(arm: str, ps: pd.DataFrame) -> list[dict]:
    rows = []
    for thr in THRS:
        cl = cells(ps, thr)
        use = cl[(cl.n_res >= 4) & np.isfinite(cl.get("y_se", np.nan))]
        subsets = {"all": use}
        if arm == "P":
            subsets["no_shrink"] = use[use.shrink < 1]
        # not pre-registered: the same cells with medians of y (one outlying seed, y ~ -10, moves a
        # cell mean by ~2 for 6 seeds)
        subsets["all_robust"] = use.assign(y_mean=use.y_med, y_se=use.y_med_se)
        for name, u in subsets.items():
            x = (u.K + 1) / 2 * np.log(u.d)
            if len(u) >= 3:
                fits = {"joint": wls(x, u.y_mean, u.y_se), "joint_A": wls(x, u.y_mean, u.y_se, 0.0),
                        "joint_B": wls(x, u.y_mean, u.y_se, 1.0)}
                for d in (2, 3):
                    m = u.d == d
                    if m.sum() >= 3:
                        fits[f"d{d}"] = wls(x[m], u.y_mean[m], u.y_se[m])
                        fits[f"d{d}_A"] = wls(x[m], u.y_mean[m], u.y_se[m], 0.0)
                        fits[f"d{d}_B"] = wls(x[m], u.y_mean[m], u.y_se[m], 1.0)
                c, _ = contrast(u)
                if c is not None:
                    fits["contrast"] = c
                for k, f in fits.items():
                    rows.append(dict(arm=arm, thr=thr, subset=name, fit=k, n_cells=len(u),
                                     K_range=f"{u.K.min()}-{u.K.max()}", **f))
    return rows


def censored_table(ps: pd.DataFrame, fits: list[dict]) -> pd.DataFrame:
    """Seeds censored above (S2S-like up to the domain limit or to sigma_max), with their lower
    bound lo, compared with the H_A and H_B predictions whose gamma is fitted on the resolved cells
    (rho* = 3, subset 'all')."""
    g = {f["fit"]: f for f in fits if f["thr"] == 3.0 and f["subset"] == "all"}
    if not {"joint_A", "joint_B"} <= set(g):
        return pd.DataFrame()
    gA, gB = g["joint_A"]["gamma"], g["joint_B"]["gamma"]
    cen = ps[ps.status.isin(["domain", "all_s2s"])].copy()
    out = []
    for _, r in cen.iterrows():
        K, d = int(r.K), int(r.d)
        u0 = aligned_boundary_a0(K, WEIGHTS) ** K
        sA = (u0 * math.exp(-gA)) ** (1 / K)
        sB = (u0 * math.exp(-gB) * d ** ((K + 1) / 2)) ** (1 / K)
        out.append(dict(K=K, d=d, seed=int(r.seed), status=r.status, lo=r.lo, hi=r.hi,
                        sigma_dom=r.get("sigma_dom", np.nan), y_lb=math.log(r.lo**K / u0),
                        sigma_HA=sA, sigma_HB=sB, above_HA=bool(r.lo > sA), above_HB=bool(r.lo > sB)))
    return pd.DataFrame(out)


def censored_y(ps: pd.DataFrame, thr: float = 3.0) -> pd.DataFrame:
    """Per seed: x = ln sqrt(D), and y exact ('ok*'), a lower bound (S2S-like up to the domain limit
    or sigma_max: y > ln(lo^K / u_0c)) or an upper bound ('never_s2s': y < ln(hi^K / u_0c))."""
    tag = "" if thr == 3.0 else f"_{thr:g}"
    out = []
    u0 = {}
    for _, r in ps.iterrows():
        K, d, st = int(r.K), int(r.d), r[f"status{tag}"]
        if K not in u0:
            u0[K] = aligned_boundary_a0(K, WEIGHTS, rho_thr=thr) ** K
        x = (K + 1) / 2 * math.log(d)
        if st.startswith("ok"):
            out.append(dict(K=K, d=d, x=x, y=r[f"y{tag}"], kind="exact"))
        elif st in ("domain", "all_s2s") and np.isfinite(r[f"lo{tag}"]):
            out.append(dict(K=K, d=d, x=x, y=math.log(r[f"lo{tag}"] ** K / u0[K]), kind="lower"))
        elif st == "never_s2s" and np.isfinite(r[f"hi{tag}"]):
            out.append(dict(K=K, d=d, x=x, y=math.log(r[f"hi{tag}"] ** K / u0[K]), kind="upper"))
    return pd.DataFrame(out)


def tobit(cy: pd.DataFrame, fixed_beta=None):
    """Censored-normal ML fit y = -gamma + beta x + s eps over seeds (not pre-registered; uses the
    censored seeds that the pre-registered fits leave out). Returns (gamma, beta, s, loglik)."""
    from scipy.optimize import minimize

    x, y, k = cy.x.values, cy.y.values, cy.kind.values
    ex, lo, up = k == "exact", k == "lower", k == "upper"

    def nll(p):
        g, b, ls = (p[0], fixed_beta, p[1]) if fixed_beta is not None else p
        sd = math.exp(ls)
        z = (y - (-g + b * x)) / sd
        return -(stats.norm.logpdf(z[ex]).sum() - ex.sum() * ls + stats.norm.logsf(z[lo]).sum()
                 + stats.norm.logcdf(z[up]).sum())

    p0 = [-float(np.mean(y)), 0.5, math.log(np.std(y) + 1e-3)]
    if fixed_beta is not None:
        p0 = [p0[0] + fixed_beta * float(np.mean(x)), p0[2]]
    res = minimize(nll, p0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
    p = res.x
    if fixed_beta is not None:
        return float(p[0]), fixed_beta, float(math.exp(p[1])), -float(res.fun)
    return float(p[0]), float(p[1]), float(math.exp(p[2])), -float(res.fun)


def tobit_summary(ps: pd.DataFrame, arm: str, thr: float = 3.0, n_boot: int = 300) -> list[dict]:
    cy = censored_y(ps, thr)
    if len(cy) < 5:
        return []
    g, b, sd, ll = tobit(cy)
    llA, llB = tobit(cy, 0.0)[3], tobit(cy, 1.0)[3]
    bs = []
    cells_ = [grp for _, grp in cy.groupby(["K", "d"])]
    for _ in range(n_boot):  # resample seeds within each (K, d) cell
        smp = pd.concat([c.iloc[RNG.integers(0, len(c), len(c))] for c in cells_])
        bs.append(tobit(smp)[1])
    se = float(np.std(bs))
    return [dict(arm=arm, thr=thr, subset="tobit_all_seeds", fit="tobit", n_cells=len(cells_),
                 K_range=f"{cy.K.min()}-{cy.K.max()}", gamma=g, beta=b, se_gamma=np.nan, se_beta=se,
                 chi2=np.nan, dof=len(cy) - 3, p=np.nan, birge=np.nan, sd=sd,
                 n_exact=int((cy.kind == "exact").sum()), n_lower=int((cy.kind == "lower").sum()),
                 n_upper=int((cy.kind == "upper").sum()),
                 LR_A=2 * (ll - llA), p_A=float(stats.chi2.sf(2 * (ll - llA), 1)),
                 LR_B=2 * (ll - llB), p_B=float(stats.chi2.sf(2 * (ll - llB), 1)))]


def decision(f_joint: dict, f_A: dict, f_B: dict) -> str:
    b, se = f_joint["beta"], f_joint["se_beta"]
    zA, zB = abs(b) / se, abs(b - 1) / se
    okA = f_A["chi2"] / max(f_A["dof"], 1) < 2
    okB = f_B["chi2"] / max(f_B["dof"], 1) < 2
    if zA <= 2 and zB > 3 and okA:
        return "H_A favoured"
    if zB <= 2 and zA > 3 and okB:
        return "H_B favoured"
    if zA > 3 and zB > 3:
        return "neither"
    return "unresolved"


COL = {2: "#2a78d6", 3: "#eb6834"}


def fig_round2(arm: str, ps: pd.DataFrame, path: str, title: str):
    """(a) sigma_c vs K per seed and cell medians (bootstrap 95% CI) with the fitted laws; (b) y vs
    ln sqrt(D) with the joint, beta = 0 and beta = 1 fits; (c) d-ratio vs K."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cl = cells(ps)
    use = cl[(cl.n_res >= 4) & np.isfinite(cl.y_se)]
    x = (use.K + 1) / 2 * np.log(use.d)
    fits = {b: wls(x, use.y_mean, use.y_se, b) for b in (None, 0.0, 1.0)} if len(use) >= 3 else {}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    Ks = np.arange(3, 16)
    u0 = np.array([aligned_boundary_a0(int(K), WEIGHTS) ** K for K in Ks])
    for d in (2, 3):
        g = ps[ps.d == d]
        off = -0.12 if d == 2 else 0.12
        ok = np.isfinite(g.sigma_c)
        ax.plot(g.K[ok] + off, g.sigma_c[ok], ".", color=COL[d], alpha=0.35, ms=5)
        cens = g[g.status.isin(["all_s2s", "domain"])]
        ax.plot(cens.K + off, cens.lo, "^", color=COL[d], alpha=0.6, ms=5, mfc="none")
        c = cl[(cl.d == d) & np.isfinite(cl.sigma_c_med)]
        lo = c.sigma_c_med - c.get("sigma_c_lo", c.sigma_c_med)
        hi = c.get("sigma_c_hi", c.sigma_c_med) - c.sigma_c_med
        ax.errorbar(c.K + off, c.sigma_c_med, yerr=[lo.fillna(0), hi.fillna(0)], fmt="o", color=COL[d],
                    capsize=3, label=f"d = {d}: median, 95% CI")
        if fits:
            gB = fits[1.0]["gamma"]
            ax.plot(Ks, (u0 * np.exp(-gB) * d ** ((Ks + 1) / 2)) ** (1 / Ks), "--", color=COL[d], lw=1,
                    label=f"H_B fit (beta = 1), d = {d}")
        ax.axhline(math.sqrt(d), color=COL[d], lw=0.6, ls=":")
        if "sigma_dom_med" in c:
            cd = cl[(cl.d == d) & np.isfinite(cl.get("sigma_dom_med", np.nan))]
            ax.plot(cd.K + off, cd.sigma_dom_med, "_", color=COL[d], ms=14, mew=2,
                    label=f"d = {d}: domain limit (median)")
    if fits:
        gA = fits[0.0]["gamma"]
        ax.plot(Ks, (u0 * np.exp(-gA)) ** (1 / Ks), "-", color="0.4", lw=1, label="H_A fit (beta = 0)")
    ax.axhline(1, color="0.4", lw=0.6, ls=":")
    ax.set_xlabel("K (binary tree, N = K + 1)")
    ax.set_ylabel("sigma_c (rho* = 3)")
    ax.set_title(f"(a) {title}: per-seed sigma_c (^ = lower bound)")
    ax.legend(fontsize=7)
    ax = axs[1]
    for d in (2, 3):
        c = use[use.d == d]
        ax.errorbar((c.K + 1) / 2 * math.log(d), c.y_mean, yerr=c.y_se, fmt="o", color=COL[d], capsize=3,
                    label=f"d = {d} (cell mean +- bootstrap SE)")
    if fits:
        xx = np.linspace(0, float(x.max()) * 1.05, 50)
        f = fits[None]
        ax.plot(xx, -f["gamma"] + f["beta"] * xx, "k-", lw=1,
                label=f"joint fit: beta = {f['beta']:.2f} +- {f['se_beta']:.2f}")
        for b, ls in ((0.0, ":"), (1.0, "--")):
            g = fits[b]["gamma"]
            ax.plot(xx, -g + b * xx, "k" + ls, lw=0.8,
                    label=f"beta = {b:g} (H_{'A' if b == 0 else 'B'}): chi2/dof = {fits[b]['chi2']:.0f}/{fits[b]['dof']}")
    ax.set_xlabel("ln sqrt(D)")
    ax.set_ylabel("y = ln(sigma_c^K / u_0c(K))")
    ax.set_title("(b) law test")
    ax.legend(fontsize=7)
    ax = axs[2]
    r = d_ratio(ps)
    if len(r):
        ax.errorbar(r.K, r.r, yerr=[r.r - r.lo, r.hi - r.r], fmt="ko", capsize=3, label="median(d=3) / median(d=2)")
    ax.axhline(1, color="0.4", ls=":", label="H_A: 1")
    ax.plot(Ks, 1.5 ** ((Ks + 1) / (2 * Ks)), "k--", lw=0.8, label="H_B: 1.5^(N/2K)")
    ax.set_xlabel("K")
    ax.set_ylabel("r_K = sigma_c(d = 3) / sigma_c(d = 2)")
    ax.set_title("(c) d-ratio (bootstrap 95% CI)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=OUT)
    ap.add_argument("--out", default=RES)
    ap.add_argument("--arms", default="P,C")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def save(df, name):
        path = os.path.join(a.out, name)
        if os.path.exists(path) and not a.force:
            raise FileExistsError(path)
        df.to_csv(path, index=False, float_format="%.6g")

    fit_rows = []
    for arm in a.arms.split(","):
        try:
            df = load(a.raw, arm)
        except FileNotFoundError:
            continue
        ps = per_seed(df, arm)
        save(ps, f"per_seed_{arm}.csv")
        for thr in THRS:
            save(cells(ps, thr), f"cells_{arm}_rho{thr:g}.csv")
        save(d_ratio(ps), f"ratio_{arm}.csv")
        tim = df.groupby(["K", "d"]).agg(P=("P", "first"), n_runs=("wall", "size"), wall_med=("wall", "median"),
                                         wall_max=("wall", "max"), steps_med=("n_steps", "median"),
                                         stopped_max_steps=("stop", lambda x: int((x == "max_steps").sum())))
        save(tim.reset_index(), f"timing_{arm}.csv")
        fpath = os.path.join(a.out, f"round2_sigma_c_{arm}.png")
        if os.path.exists(fpath) and not a.force:
            raise FileExistsError(fpath)
        fig_round2(arm, ps, fpath, {"P": "plain, chi = 4", "C": "centred, full bonds"}[arm])
        fr = all_fits(arm, ps)
        fit_rows += fr
        for thr in THRS:
            tb = tobit_summary(ps, arm, thr)
            fit_rows += tb
            for t in tb:
                print(arm, f"rho*={thr:g} tobit (all seeds, not pre-registered): beta = {t['beta']:.3f} +- {t['se_beta']:.3f}",
                      f"(exact {t['n_exact']}, lower {t['n_lower']}, upper {t['n_upper']}); LR vs H_A p = {t['p_A']:.2g}, vs H_B p = {t['p_B']:.2g}")
        cen = censored_table(ps, fr)
        if len(cen):
            save(cen, f"censored_{arm}.csv")
            print(arm, "censored seeds:", len(cen), "above H_A:", int(cen.above_HA.sum()),
                  "above H_B:", int(cen.above_HB.sum()))
        for thr in THRS:
            for sub in ("all", "no_shrink", "all_robust"):
                get = {r["fit"]: r for r in fr if r["thr"] == thr and r["subset"] == sub}
                if {"joint", "joint_A", "joint_B"} <= set(get):
                    print(arm, f"rho*={thr:g}", sub, "beta = %.3f +- %.3f" % (get["joint"]["beta"], get["joint"]["se_beta"]),
                          "->", decision(get["joint"], get["joint_A"], get["joint_B"]))
    if fit_rows:
        fits = pd.DataFrame(fit_rows)
        save(fits, "fits.csv")


if __name__ == "__main__":
    main()
