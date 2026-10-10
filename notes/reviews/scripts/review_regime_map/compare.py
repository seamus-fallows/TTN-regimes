"""Compare reviewer reruns (committed code, data/review_regime_map) with the committed summary CSVs."""

import os
import sys

import numpy as np
import pandas as pd

REPO = "c:/Users/seamu/Iliad-ML/TTN_regimes"
sys.path.insert(0, REPO + "/experiments/regime_map")
import analysis  # noqa: E402
import plots  # noqa: E402

analysis.OUT = REPO + "/data/review_regime_map"
RES = REPO + "/experiments/regime_map/results/"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
COLS = [
    "sigma_c",
    "s2s_status",
    "s2s_lo",
    "s2s_hi",
    "sigma_c_pred",
    "sigma_K01",
    "rho_slope",
    "n_s2s",
]


def cmp_rows(mine, theirs, keys):
    m = mine.merge(theirs, on=keys, suffixes=("_rerun", "_commit"))
    out = m[
        keys + [f"{c}_{s}" for c in COLS for s in ("rerun", "commit") if f"{c}_{s}" in m.columns]
    ]
    print(out.T.to_string())


def chi_b4():
    df = analysis.load("chi_b4")
    tab = analysis.per_seed_table(df, ["model", "chi"], 3)
    full = tab.merge(
        df.groupby(["chi", "seed"]).size().rename("n_runs").reset_index(), on=["chi", "seed"]
    )
    full = full[full.n_runs == 22]
    theirs = pd.read_csv(RES + "boundaries_per_seed_chi_b4.csv")
    print("== chi_b4 per-seed (full 22-sigma grids only)")
    cmp_rows(full.drop(columns="n_runs"), theirs, ["model", "chi", "seed"])


def sigma_chi_b4():
    df = analysis.load("sigma_chi_b4")
    df = df[df.model == "centred"]
    n = df.groupby(["chi", "seed"]).size()
    print("== sigma_chi_b4 runs per (chi, seed):", n.to_dict())
    full = [k for k, v in n.items() if v == 19]
    st = analysis.sensitivity_table(df, ["chi"])
    st = st[[(c, s) in full for c, s in zip(st.chi, st.seed)]]
    theirs = pd.read_csv(RES + "sensitivity_per_seed_sigma_chi_b4_centred.csv")
    m = st.merge(theirs, on=["chi", "seed"], suffixes=("_rerun", "_commit"))
    print(m.T.to_string())
    # chi = 12, seed 4 deep runs (seed dropped as NaN in the committed table) and floor runs
    sub = df[((df.chi == 12) & (df.seed == 4)) | (df.chi == 3)]
    print(
        sub[
            ["chi", "seed", "sigma", "label", "l_end", "l_best", "dK", "min_rho", "n_events"]
        ].to_string()
    )
    for _, r in sub.iterrows():
        print(
            "  events",
            r.chi,
            r.seed,
            r.sigma,
            np.round(np.array(r.events, float), 3).tolist(),
            "rhos",
            np.round(r.rhos, 3),
        )


def imbalance():
    df = analysis.load("imbalance_b4")
    df = df[df.chi == 16]
    g = df.groupby("root_scale")
    out = pd.DataFrame(
        {
            "n": g.size(),
            "median_dK": g.dK.median(),
            "labels": g.label.apply(lambda s: s.value_counts().to_dict()),
        }
    )
    print("== imbalance chi = 16, alpha = 3.375")
    print(out.to_string())
    med = out.median_dK.values
    lam = out.index.values
    print("local slopes", np.round(np.diff(np.log(med)) / np.diff(np.log(lam)), 2)[:4])


def large_chi():
    df = analysis.load("large_chi_b4")
    df = df[(df.model == "centred") & np.isclose(df.sigma, 1.5) & (df.chi >= 16)]
    print("== large_chi_b4 centred sigma 1.5: seeds", sorted(df.seed.unique()), "n =", len(df))
    root = 2
    df = df.assign(
        root_change=[d[root] for d in df.dth_core],
        leaf_change=[np.mean([v for i, v in enumerate(d) if i != root]) for d in df.dth_core],
    )
    theirs = pd.read_csv(RES + "large_chi_exponents_large_chi_b4_centred_sigma1.5.csv")
    rows = []
    for lr, s in df.groupby("lr"):
        for q in ("dK", "dth", "root_change", "leaf_change"):
            b, e = plots.pooled_slope(s.chi, s[q])
            t = theirs[(theirs.lr == lr) & (theirs.quantity == q)]
            rows.append(
                dict(
                    lr=lr,
                    q=q,
                    rerun=f"{b:.3f} +- {e:.3f} (n={len(s)})",
                    commit=f"{t.slope.iloc[0]:.3f} +- {t.se.iloc[0]:.3f} (n={t.n_runs.iloc[0]})",
                )
            )
        print(lr, "median dK by chi", s.groupby("chi").dK.median().round(4).to_dict())
    print(pd.DataFrame(rows).to_string())


def sigma_K():
    df = analysis.load("sigma_K_centred_d2")
    n = df.groupby(["kind", "n", "chi", "seed"]).size()
    print("== sigma_K_centred_d2 runs:", n.to_dict())
    tab = analysis.per_seed_table(df, ["kind", "n"])
    theirs = pd.read_csv(RES + "boundaries_per_seed_sigma_K_centred_d2_chi4.csv")
    theirs["kind"] = theirs.tree.str.extract(r"^([a-z]+)")
    theirs["n"] = theirs.tree.str.extract(r"^[a-z]+(\d+)").astype(int)
    full = tab[[n.get((k, nn, 4, s), 0) >= 12 for k, nn, s in zip(tab.kind, tab.n, tab.seed)]]
    cmp_rows(full, theirs.drop(columns=["tree"]), ["kind", "n", "seed"])
    print(
        df[["kind", "n", "seed", "sigma", "label", "l_end", "dK", "min_rho"]]
        .sort_values(["kind", "n", "seed", "sigma"])
        .to_string()
    )


if __name__ == "__main__":
    for name in sys.argv[1:]:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            print("FAILED", name, repr(e))
