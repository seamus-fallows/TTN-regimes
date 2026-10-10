"""Validate the hopper on experiments/order (round-1 results on main; re-run when the fix round lands).

Inputs (committed, experiments/order/results/): sigma_fits.csv (P(smaller first) ~ sigma^gamma: fitted
gamma vs n (1 - rho)/rho), h1b_fits.csv (wrong-order saddle: lambda_min and correction slope),
multi2_perms.csv (R = 3, 4 components: frequency of every order).
usage: uv run python experiments/saddle_hopper/validate_order.py [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ORD = ROOT / "experiments" / "order" / "results"
sys.path.insert(0, str(HERE))
import predict  # noqa: E402

RES = HERE / "results"
SWEEPS = {  # sweep -> (tree kind, n, sites where component 2 differs; None = odeco)
    "b4_sigma": ("binary", 4, None),
    "m4_sigma": ("mps", 4, None),
    "b8_sigma": ("binary", 8, None),
    "pat_mps4_adj": ("mps", 4, [1, 2]),
    "pat_mps4_sep": ("mps", 4, [0, 2]),
    "pat_mps6_m2adj": ("mps", 6, [2, 3]),
    "pat_mps6_m2end": ("mps", 6, [0, 5]),
    "pat_mps6_m3": ("mps", 6, [0, 1, 2]),
    "pat_mps6_m4": ("mps", 6, [0, 1, 2, 3]),
}


def two_case(kind, n, sites, rho, d, chi, init="fanin", start=None):
    pats = (
        [[0] * n, [1] * n]
        if sites is None
        else [[0] * n, [1 if i in sites else 0 for i in range(n)]]
    )
    return dict(
        teacher=dict(kind="pattern", patterns=pats, weights=[1.0, rho]),
        tree=kind,
        n=n,
        d=d,
        chi=chi,
        init=init,
        start_component=start,
        max_events=4,
        max_sequences=16,
    )


def smaller_candidate(pred):
    """First-escape candidate whose saddle is the smaller component (overlap 1 on component 1)."""
    for c in pred["candidates"]:
        if c["overlaps"] and c["overlaps"][1] > 0.99 and c["overlaps"][0] < 0.01:
            return c
    return None


SIGMA_MULTI = {"binary4": 0.05, "mps4": 0.1, "mps6": 0.2, "binary8": 0.25}  # order configs/multi.yaml


def competition(pred, order, sigma):
    """For an order no branch produces: after the wrong first escape (order[0] first), compare the rotation
    (O(1/lambda) in natural runs) with the direct add route (aligned time sigma^-(k-2) / ((k-2) mu), no head
    start): theory_saddles 4.4 says the rotation wins as sigma -> 0."""
    for s in pred["sequences"]:
        evs = s["events"]
        if evs[0]["learned"] == [order[0]] and len(evs) > 1 and evs[1]["kind"] == "rotation":
            ev = evs[1]
            k, mu = ev.get("alt_k"), ev.get("alt_mu")
            t_add = sigma ** (-(k - 2)) / ((k - 2) * mu) if k and k >= 3 else np.nan
            return dict(t_rotation=1.0 / ev["rate"], t_add_route=t_add, add_k=k)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=predict.workers_default())
    args = ap.parse_args()
    fits = pd.read_csv(ORD / "sigma_fits.csv")
    cases = {
        (r.sweep, r.rho, r.d, r.chi): two_case(*SWEEPS[r.sweep], r.rho, int(r.d), int(r.chi))
        for r in fits.itertuples()
    }
    h1b = pd.read_csv(ORD / "h1b_fits.csv")
    h1b = h1b[h1b.set == "saddle_b4"]
    for r in h1b.itertuples():  # binary4, d = 2, chi = 4, smaller component learned (saddle start)
        cases[("h1b", r.rho)] = two_case("binary", 4, None, r.rho, 2, 4, init="saddle", start=1)
    perms = pd.read_csv(ORD / "multi2_perms.csv")
    trees = {
        "binary4": ("binary", 4),
        "mps4": ("mps", 4),
        "mps6": ("mps", 6),
        "binary8": ("binary", 8),
    }
    for tree, R, rho in perms[["tree", "R", "rho"]].drop_duplicates().itertuples(index=False):
        kind, n = trees[tree]
        cases[("multi", tree, R, rho)] = dict(
            teacher=dict(kind="odeco", weights=[rho**i for i in range(R)]),
            tree=kind,
            n=n,
            d=4,
            chi=4,
            init="fanin",
            max_events=6,
            max_sequences=64,
        )
    preds = predict.run_many(list(cases.values()), RES / "predictions_order.json", args.workers)
    rows = []
    for r in fits.itertuples():
        p = preds[predict.key_of(cases[(r.sweep, r.rho, r.d, r.chi)])]
        c = smaller_candidate(p)
        rows.append(
            dict(
                sweep=r.sweep,
                tree=r.tree,
                d=r.d,
                chi=r.chi,
                rho=r.rho,
                m=r.m,
                smaller_is_attractor=bool(c and c["n_unstable"] == 0),
                n_unstable_hopper=c["n_unstable"] if c else None,
                gamma_hopper=c["gamma"] if c else None,
                gamma_order_pred=r.gamma_pred,
                gamma_measured=r.gamma,
                gamma_se=r.gamma_se,
                gamma_small_sigma=r.gamma_small_sigma,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "order_sigma_exponents.csv", index=False)
    print(df.to_string())
    hrows = []
    for r in h1b.itertuples():
        p = preds[predict.key_of(cases[("h1b", r.rho)])]
        ev = p["sequences"][0]["events"][1]
        hrows.append(
            dict(
                rho=r.rho,
                event=ev["kind"],
                lam_hopper=-ev["rate"] if ev["kind"] == "rotation" else None,
                lam_measured=r.lam_min_mean,
                slope_hopper=ev.get("slope_saddle"),
                slope_measured=r.slope,
                slope_se=r.slope_se,
                forgotten=str(ev["forgotten"]),
            )
        )
    hdf = pd.DataFrame(hrows)
    hdf.to_csv(RES / "order_h1b.csv", index=False)
    print(hdf.to_string())
    prows = []
    for (tree, R, rho), g in perms.groupby(["tree", "R", "rho"]):
        p = preds[predict.key_of(cases[("multi", tree, R, rho)])]
        allowed = predict.allowed(p, "first")
        for o in g.itertuples():
            order = tuple(int(x) - 1 for x in str(o.order))
            got = {pc for pc, _ in allowed.get(order, set())}
            prows.append(
                dict(
                    tree=tree,
                    R=R,
                    rho=rho,
                    order=o.order,
                    freq_gf=o.gf,
                    freq_race_model=o.race_model,
                    branch="main" if "main" in got else ("rare" if got else "none"),
                    **(competition(p, order, SIGMA_MULTI.get(tree, np.nan)) if not got else {}),
                )
            )
    pdf = pd.DataFrame(prows)
    pdf.to_csv(RES / "order_multi.csv", index=False)
    print(
        pdf.groupby(["tree", "R", "rho", "branch"]).freq_gf.sum().unstack(fill_value=0).to_string()
    )


if __name__ == "__main__":
    main()
