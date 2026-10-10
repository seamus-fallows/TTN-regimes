"""Verifier: H1b recomputation from the committed per-run CSVs (+ npz eigenvalues), own code."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

RES = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/order/results"
RAW = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/order"


def lam_pred(rho, K, kind):
    q = rho ** (1 / K)
    return -(1 - rho) * q ** (K - 2) if kind == "rot" else -(q ** (K - 2))


def cls(df):
    a = df.a2_min_after_t1
    return np.where(
        a.isna(),
        "none",
        np.where(a < 0.1, "rotated", np.where(df.a2_down.isna(), "added", "other")),
    )


for name, K, kind in (
    ("h1b_saddle_b4", 3, "rot"),
    ("h1b_saddle_mps4_sep", 4, "rot"),
    ("h1b_saddle_mps4_adj", 4, "add"),
):
    d = pd.read_csv(f"{RES}/{name}.csv")
    d["cls"] = cls(d)
    print(
        f"=== {name}: runs {len(d)}, classes {d.cls.value_counts().to_dict()}, sigma {sorted(d.sigma.unique())}"
    )
    for rho, g in d.groupby("rho"):
        lp = lam_pred(rho, K, kind)
        sl_all, sl_small, rl, rp = [], [], [], []
        loc = {}
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("sigma", ascending=False)
            x, y = np.log(1 / gs.sigma.values), gs.t1.values
            sl_all.append(np.polyfit(x, y, 1)[0])
            m = gs.sigma.values <= 1e-4
            sl_small.append(np.polyfit(x[m], y[m], 1)[0])
            b, a = np.polyfit(x, y, 1)
            rl.append(np.sqrt(np.mean((y - a - b * x) ** 2)))
            p, lc = np.polyfit(x, np.log(y), 1)
            (c, p), _ = curve_fit(
                lambda xx, c, p: c * np.exp(p * xx), x, y, p0=(np.exp(lc), p), maxfev=20000
            )
            rp.append(np.sqrt(np.mean((y - c * np.exp(p * x)) ** 2)))
            for k in range(len(x) - 1):
                loc.setdefault((gs.sigma.values[k], gs.sigma.values[k + 1]), []).append(
                    (y[k + 1] - y[k]) / (x[k + 1] - x[k]) * abs(lp)
                )
        se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
        print(
            f" rho {rho}: pred lam {lp:.5f} 1/|lam| {1 / abs(lp):.4f} | slope all {np.mean(sl_all):.4f}+-{se(sl_all):.4f}"
            f" | slope sigma<=1e-4 {np.mean(sl_small):.4f}+-{se(sl_small):.4f} | rms log {np.mean(rl):.3f} power {np.mean(rp):.3f}"
        )
        print(
            "   local slope x |lam| (1e-2->1e-7):",
            " ".join(f"{np.mean(v):.4f}" for k, v in sorted(loc.items(), reverse=True)),
        )
        print(
            f"   lam_min raw mean {g.lam_min_raw.mean():.5f} (sd {g.lam_min_raw.std():.1e}); projected mean {g.lam_min.mean():.5f};"
            f" projected at sigma<=1e-3 {g[g.sigma <= 1e-3].lam_min.mean():.5f}; a2_min_after_t1 range [{g.a2_min_after_t1.min():.2e}, {g.a2_min_after_t1.max():.2e}]"
        )
        if kind == "add":
            # multiplicity of the adding eigenvalue from the stored projected spectra
            mult = []
            for row in g[g.sigma <= 1e-3].itertuples():
                z = np.load(f"{RAW}/{name}/rho{rho}_s{row.sigma:g}_seed{row.seed}.npz")
                e = z["eigs"]
                mult.append(
                    (
                        int(np.sum(np.abs(e - lp) < 0.01 * abs(lp))),
                        int(np.sum(e < -1e-3)),
                        float(np.sort(e)[3]),
                    )
                )
            mult = np.array(mult)
            print(
                f"   [npz] sigma<=1e-3: #eig within 1% of -s1 s2^((K-2)/K): {np.unique(mult[:, 0], return_counts=True)};"
                f" #neg {np.unique(mult[:, 1], return_counts=True)}; 4th eig mean {mult[:, 2].mean():.4f} vs piggyback {lam_pred(rho, K, 'rot'):.4f}"
            )

print("\n=== natural wrong-order runs ===")
for name, K, kind, law in (
    ("h1b_natural_b4", 3, "rot", "rotated"),
    ("h1b_natural_mps4_sep", 4, "rot", "rotated"),
    ("h1b_natural_mps4_adj", 4, "add", "added"),
    ("h1b_natural_mps4", 4, None, "added"),
):
    d = pd.read_csv(f"{RES}/{name}.csv")
    w = d[d.wrong].copy()
    w["cls"] = cls(w)
    print(f"--- {name}: wrong runs {len(w)} of {len(d)}; stalled {int(w.stalled.sum())}")
    for sig_lo, sig_hi in ((0, 0.011), (0.05, 0.2)):
        ww = w[(w.sigma > sig_lo) & (w.sigma < sig_hi)]
        if not len(ww):
            continue
        print(
            f"   sigma in ({sig_lo},{sig_hi}): n {len(ww)} classes {ww.cls.value_counts().to_dict()},"
            f" unresolved midpoint Hessian {int((ww.lam_min.isna() & np.isfinite(ww.t1)).sum())}"
        )
    if kind is None:
        print("   median gap", w.gap.median(), "lam_min median", w.lam_min.median())
        continue
    w["lp"] = [lam_pred(r, K, kind) for r in w.rho]
    w["scaled"] = w.gap * np.abs(w.lp)
    s = w[(w.cls == law) & (w.sigma <= 0.011)]
    tab = s.groupby(["rho", "sigma"]).scaled.agg(["mean", "count", "median"])
    print(
        "   |lam|(t1-t2) by (rho, sigma):",
        "; ".join(
            f"{k[0]},{k[1]:g}: {v['mean']:.2f} (n{int(v['count'])})" for k, v in tab.iterrows()
        ),
    )
    # slope of scaled vs ln(1/sigma), pooled over rho with rho fixed effects
    X = np.column_stack(
        [np.log(1 / s.sigma)] + [(s.rho == r).astype(float) for r in sorted(s.rho.unique())]
    )
    coef, res, *_ = np.linalg.lstsq(X, s.scaled.values, rcond=None)
    resid = s.scaled.values - X @ coef
    cov = np.linalg.inv(X.T @ X) * resid.var(ddof=X.shape[1])
    print(
        f"   slope of |lam|(t1-t2) vs ln(1/sigma): {coef[0]:+.3f} +- {np.sqrt(cov[0, 0]):.3f} (log law would give +1)"
    )
    if law == "rotated":
        r = s[s.lam_min.notna()]
        frac = np.mean((r.lam_min >= r.lp - 1e-9) & (r.lam_min <= 0))
        print(
            f"   midpoint lam_min between saddle value and 0 (sigma<=1e-2): {frac:.3f} of {len(r)}; median gap {s.gap.median():.2f}"
        )
        r1 = w[(w.sigma > 0.05) & w.lam_min.notna()]
        if len(r1):
            print(
                f"   sigma=0.1: lam_min mean {r1.lam_min.mean():.3f}, frac below saddle value {np.mean(r1.lam_min < r1.lp):.2f}"
            )
    else:
        print(
            f"   a2_min_after_t1 median by rho: {s.groupby('rho').a2_min_after_t1.median().round(3).to_dict()}; gap median {s.gap.median():.3f}"
        )

print("\n=== m >= 3 saddle starts (scaling rule) ===")
for name in ("h1b_saddle_mps4", "h1b_saddle_mps6", "h1b_saddle_b8"):
    d = pd.read_csv(f"{RES}/{name}.csv")
    st = d.stalled.sum() if "stalled" in d else np.nan
    print(
        f"--- {name}: runs {len(d)}, sigma {sorted(d.sigma.unique())}, seeds {sorted(d.seed.unique())}, stalled {st},"
        f" t1 inf {int((~np.isfinite(d.t1)).sum())}, a2 at/after t1 min {d.a2_min_after_t1.min():.4f}"
    )
    for rho, g in d.groupby("rho"):
        le, te = [], []
        for seed, gs in g.groupby("seed"):
            ok = np.isfinite(gs.t1.values) & np.isfinite(gs.lam_min.values)
            if ok.sum() < 3:
                print(f"   rho {rho} seed {seed}: only {ok.sum()} finite points (excluded)")
                continue
            x = np.log(gs.sigma.values[ok])
            le.append(np.polyfit(x, np.log(np.abs(gs.lam_min.values[ok])), 1)[0])
            te.append(np.polyfit(x, np.log(gs.t1.values[ok]), 1)[0])
        se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
        print(
            f"   rho {rho}: n_seeds {len(le)} lam exponent {np.mean(le):.3f}+-{se(le):.3f}  t1 exponent {np.mean(te):.3f}+-{se(te):.3f}"
        )
