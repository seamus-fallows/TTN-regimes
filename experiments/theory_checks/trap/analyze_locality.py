"""Confusion tables: predicted (alive / pinned / dead, from the init) vs observed classes.

Observed: locality runs 'stalled' (5,000-evaluation stall detector; includes slow escapes) vs escaped
(greedy, other order, rotation); stall-diagnosis runs dead / pinned / escaped (results/r12.csv of the
locality experiment, 60x patience); pilot runs 'collapsed' (route ends at roundoff) vs 'converged'
(t_max / timeout = unresolved).

usage: python experiments/theory_checks/trap/analyze_locality.py [results dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
PILOT = Path("C:/Users/seamu/Iliad-ML/TTN_regimes/data/pilot_h2_locality")

OBS = {
    "stalled": "stall",
    "greedy": "escaped",
    "other_order": "escaped",
    "rotation": "escaped",
    "collapsed": "dead",
    "converged": "escaped",
    "t_max": "unresolved",
    "timeout": "unresolved",
}


def tree_name(r):
    return r["kind"] if pd.isna(r["n"]) else f"{r['kind']}{int(r['n'])}"


def load():
    df = pd.read_csv(RES / "locality_pilot_predictions.csv.gz", dtype={"S": str})
    df["tree"] = df.apply(tree_name, axis=1)
    df["obs"] = df["obs"].map(OBS).fillna("other")
    r12 = pd.read_csv(ROOT / "experiments" / "locality" / "results" / "r12.csv", dtype={"S": str})
    r12 = r12.rename(columns={"outcome": "obs_diag"})[
        ["tree", "S", "start", "chi", "sigma", "seed", "obs_diag"]
    ]
    df = df.merge(r12, how="left", on=["tree", "S", "start", "chi", "sigma", "seed"])
    df.loc[(df.sweep == "stalldiag") & df.obs_diag.notna(), "obs"] = df.obs_diag
    # pilot runs that hit t_max / the wall clock with the second component at roundoff are dead
    # (the pilot's collapse detector does not run on saddle starts)
    pil = []
    for f in ["runs.csv", "saddle_runs.csv", "extend_runs.csv"]:
        x = pd.read_csv(PILOT / f)
        x["start"] = x.start.map({"init": "small", "saddle": "saddle"})
        x["tree"] = x.tree + x.n.astype(str)
        x["sweep"] = f[:-4]
        pil.append(x[["sweep", "tree", "kind", "start", "sigma", "seed", "a2_end"]])
    pil = pd.concat(pil)
    pil["S"] = [pilot_S(k, int(t[-1]) if t[-1].isdigit() else 0, t) for k, t in zip(pil.kind, pil.tree)]
    df = df.merge(pil.drop(columns="kind"), how="left", on=["sweep", "tree", "S", "start", "sigma", "seed"])
    dead = (df.source == "pilot") & (df.obs == "unresolved") & (df.a2_end.abs() < 1e-20)
    df.loc[dead, "obs"] = "dead"
    return df


def pilot_S(kind, _n, tree):
    n = int("".join(c for c in tree if c.isdigit()))
    if kind == "local":
        sites = [0, 1]
    elif kind == "far":
        sites = [0, n - 1]
    elif kind == "global":
        sites = list(range(n))
    else:
        sites = [int(c) for c in kind[len("sites"):].split("-")]
    return "".join(map(str, sites))


def rates(df, by, col):
    t = df.groupby(by + [col]).bad.agg(["sum", "count"]).rename(columns={"sum": "stalled", "count": "runs"})
    t["rate"] = (t.stalled / t.runs).round(3)
    return t


def main():
    df = load()
    df.to_csv(RES / "locality_pilot_joined.csv.gz", index=False)
    sub = df[df.obs.isin(["stall", "escaped", "dead", "pinned"]) & (df.k >= 4)].copy()
    sub["bad"] = sub.obs.isin(["stall", "dead", "pinned"])
    # the criterion is derived for the plateau after the larger component (q = 0); runs whose
    # correlation flow escapes along the smaller one first (wrong order) are reported separately
    rates(sub[sub.q == 1], ["source", "start"], "verdict").to_csv(RES / "wrong_order_runs.csv")
    sub = sub[sub.q == 0]
    # 1. per criterion, per protocol
    parts = []
    for col in ["verdict", "reduced_sigma", "verdict_r12", "verdict_q"]:
        t = rates(sub, ["source", "start"], col).reset_index().rename(columns={col: "predicted"})
        t.insert(0, "criterion", col)
        parts.append(t)
    summ = pd.concat(parts)
    summ.to_csv(RES / "summary_by_criterion.csv", index=False)
    # 2. per case (tree, teacher S, chi, protocol): stalled / runs per predicted class
    case = rates(sub, ["source", "tree", "S", "k", "start", "chi"], "verdict").reset_index()
    wide = case.pivot_table(index=["source", "tree", "S", "k", "start", "chi"], columns="verdict",
                            values=["stalled", "runs"], fill_value=0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide.reset_index().to_csv(RES / "confusion_by_case.csv", index=False)
    # 3. stall diagnosis (dead / pinned / escaped after 60x patience) vs prediction
    diag = df[df.obs_diag.notna() & (df.sweep == "stalldiag")]
    t3 = diag.groupby(["verdict", "obs_diag"]).size().unstack(fill_value=0)
    t3.to_csv(RES / "stall_diagnosis_confusion.csv")
    # 4. per seed: is the observed class constant across sigma, and does it match?
    g = sub.groupby(["source", "tree", "S", "start", "chi", "seed"])
    per_seed = g.agg(pred=("verdict", "first"), n_sigma=("bad", "size"), n_bad=("bad", "sum")).reset_index()
    per_seed.to_csv(RES / "per_seed.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summ.to_string(index=False))
        print(t3)
        print(per_seed.groupby(["source", "pred"]).apply(
            lambda x: pd.Series(dict(seeds=len(x), all_bad=(x.n_bad == x.n_sigma).sum(),
                                     none_bad=(x.n_bad == 0).sum()))))


if __name__ == "__main__":
    main()
