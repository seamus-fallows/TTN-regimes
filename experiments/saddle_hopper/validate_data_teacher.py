"""Validate the hopper on experiments/data_teacher (generic teachers, covariance, stars, cross-tree).

All inputs are committed tables in experiments/data_teacher/results/:
  A. r2_saddle_hessian.csv, r2_saddle_fits.csv: generic teachers from the exact rank-one saddle (saddle
     type, lambda_min, plateau level, escape law on mps / binary / ht, N = 4, 8);
  B. r2_random_v2_summary.csv, e4a_sigma_v2_summary.csv, e2_sigma_v2_summary.csv,
     e4b_ttn_v2_summary.csv: the same from random init (log slope with head start, or power law);
  C. r2_star_hessian.csv, r2_star_random_classes.csv: star teachers;
  D. e1_order_summary.csv: aligned anisotropic covariance (order, swap);
  E. e4c_theory.csv, r2_cross_v2_events.csv, e4c_cross_v2_events.csv, p11_counts.csv: MPS-8 teachers on a
     binary-8 student (bottom-up order P11, first edges).
usage: uv run python experiments/saddle_hopper/validate_data_teacher.py [--workers 8] [--parts ABCDE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
DT_RES = ROOT / "experiments" / "data_teacher" / "results"
sys.path.insert(0, str(HERE))
import predict  # noqa: E402

RES = HERE / "results"

R2 = {  # configs/r2_saddle.yaml, r2_saddle_b.yaml (d = 2, chi = 4)
    "nonorth_c0.2": (4, dict(kind="nonorth", c=0.2, weights=[1.0, 0.7])),
    "nonorth_c0.45": (4, dict(kind="nonorth", c=0.45, weights=[1.0, 0.7])),
    "misaligned_20": (4, dict(kind="misaligned", theta=20, weights=[1.0, 0.8], alpha=1.0)),
    "randbin_302": (4, dict(kind="r2_random", tree="binary", teacher_seed=302, decay=1.0)),
    "randbin_303": (4, dict(kind="r2_random", tree="binary", teacher_seed=303, decay=1.0)),
    "randmps_301": (4, dict(kind="r2_random", tree="mps", teacher_seed=301, decay=1.5)),
    "nonorth8_c0.45": (8, dict(kind="nonorth", c=0.45, weights=[1.0, 0.7])),
    "nonorth8_c0.7": (8, dict(kind="nonorth", c=0.7, weights=[1.0, 0.7])),
    "misaligned8_20": (8, dict(kind="misaligned", theta=20, weights=[1.0, 0.8], alpha=2.0)),
    "misaligned8_45": (8, dict(kind="misaligned", theta=45, weights=[1.0, 0.8], alpha=2.0)),
    "randbin8_302": (8, dict(kind="r2_random", tree="binary", teacher_seed=302, decay=1.0)),
    "randbin8_303": (8, dict(kind="r2_random", tree="binary", teacher_seed=303, decay=1.0)),
    "randmps8_303": (8, dict(kind="r2_random", tree="mps", teacher_seed=303, decay=1.5)),
}


def r2_case(name, tree, init, max_events):
    n, teacher = R2[name]
    return dict(
        teacher=teacher,
        tree=tree,
        n=n,
        d=2,
        chi=4,
        init=init,
        max_events=max_events,
        max_sequences=8,
    )


def first_route(pred, start_level=None):
    """The event that leaves the first saddle (main sequences; optionally the one starting at the
    rank-one saddle with the given level)."""
    seqs = [s for s in pred["sequences"] if len(s["events"]) > 1]
    if start_level is not None:
        seqs = [
            s for s in seqs if abs(s["events"][0]["to_level"] - start_level) <= 1e-3 * start_level
        ]
    seqs.sort(key=lambda s: (s["pclass"] != "main", s["events"][0]["to_level"]))
    return (seqs[0]["events"][0], seqs[0]["events"][1]) if seqs else (None, None)


def part_a(workers):
    """Saddle start on generic teachers: type, lambda_min, level, escape law."""
    hess = pd.read_csv(DT_RES / "r2_saddle_hessian.csv")
    fits = pd.read_csv(DT_RES / "r2_saddle_fits.csv")
    cases = {
        (r.teacher, r.tree): r2_case(r.teacher, r.tree, "saddle", 1)
        for r in hess.itertuples()
        if r.teacher in R2
    }
    preds = predict.run_many(list(cases.values()), RES / "predictions_data_teacher.json", workers)
    rows = []
    for r in hess.itertuples():
        if (r.teacher, r.tree) not in cases:
            continue
        p = preds[predict.key_of(cases[(r.teacher, r.tree)])]
        start, ev = first_route(p)
        if ev is None:
            continue
        lam_pred = (
            min(ev["lam_old"], -ev["gamma_k2_max"]) if ev["gamma_k2_max"] > 0 else ev["lam_old"]
        )
        strict = ev["type"].startswith("strict")
        f = fits[(fits.teacher == r.teacher) & (fits.tree == r.tree)]
        f5 = f[f.event == "onset f=0.05"]
        fi = f[f.event.str.startswith("intermediate")]
        row = dict(
            teacher=r.teacher,
            tree=r.tree,
            level_measured=r.l_saddle,
            level_hopper=start["to_level"],
            type_measured=r.type,
            type_hopper=ev["type"],
            lam_measured=r.lambda_min,
            lam_hopper=lam_pred if strict else 0.0,
            k_hopper=ev.get("k"),
            route=str(ev.get("C")),
        )
        if ev.get("k") == 2:
            row.update(
                law="log",
                slope_hopper=ev["slope_saddle"],
                slope_measured=float(f5.A_mean_loc.iloc[0]) if len(f5) else np.nan,
                slope_measured_se=float(f5.A_sem_loc.iloc[0]) if len(f5) else np.nan,
                slope_data_teacher_pred=float(f5.pred_A.iloc[0]) if len(f5) else np.nan,
            )
            if len(fi):
                row.update(slope_measured_intermediate=float(fi.A_mean_loc.iloc[0]))
        else:
            row.update(
                law=f"sigma^-{ev['k'] - 2}",
                exponent_hopper=ev["k"] - 2,
                exponent_measured=float(f5.p_mean_loc.iloc[0]) if len(f5) else np.nan,
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "data_teacher_saddle_start.csv", index=False)
    return df


RANDOM = {  # random-init sweeps: (summary file, case name -> (n, d, teacher))
    "r2_random": ("r2_random_v2_summary.csv", {k: (v[0], 2, v[1]) for k, v in R2.items() if v[0] == 4}),
    "e4a_sigma": ("e4a_sigma_v2_summary.csv", {f"overlap{c}": (4, 4, dict(kind="nonorth", c=c, weights=[1.0, 0.7]))
                                               for c in (0.0, 0.05, 0.2, 0.45)}),
    "e2_sigma": ("e2_sigma_v2_summary.csv", {f"theta_deg{t}": (4, 4, dict(kind="misaligned", theta=t, weights=[1.0, 0.8],
                                                                        alpha=1.0)) for t in (0, 5, 20)}),
    "e4b_ttn": ("e4b_ttn_v2_summary.csv", {f"teacher_seed{s}": (4, 3, dict(kind="ttn", tree="mps", ranks=[2, 3, 2],
                                                                         teacher_seed=s, decay=0.5))
                                          for s in (101, 102, 103, 104, 105)}),
}


def part_b(workers):
    """Random init: slope of the escape from the rank-one saddle with the head start (or its exponent)."""
    rows, cases = [], {}
    for sweep, (fname, teachers) in RANDOM.items():
        t = pd.read_csv(DT_RES / fname)
        if "f" in t:
            t = t[np.isclose(t.f, 0.05)] if np.isclose(t.f, 0.05).any() else t[np.isclose(t.f, 0.03)]
        for r in t.itertuples():
            if r.case not in teachers:
                continue
            n, d, teacher = teachers[r.case]
            case = dict(teacher=teacher, tree=r.tree, n=n, d=d, chi=4, init="fanin", max_events=2, max_sequences=8)
            cases[(sweep, r.case, r.tree)] = case
            rows.append((sweep, r))
    preds = predict.run_many(list(cases.values()), RES / "predictions_data_teacher.json", workers)
    out = []
    for sweep, r in rows:
        p = preds[predict.key_of(cases[(sweep, r.case, r.tree)])]
        starts = sorted({round(s["events"][0]["to_level"], 9) for s in p["sequences"] if s["pclass"] == "main"})
        j = int(r.saddle) if np.isfinite(r.saddle) else 0
        level = starts[j] if j < len(starts) else np.nan
        start, ev = first_route(p, level if np.isfinite(level) else None)
        if ev is None:
            continue
        row = dict(sweep=sweep, case=r.case, tree=r.tree, saddle=r.saddle, level_hopper=level,
                   n_seeds=r.n_seeds_loc, k_hopper=ev.get("k"), route=str(ev.get("C")))
        if ev.get("k") == 2:
            row.update(law="log", slope_hopper=ev["slope_small"], slope_measured=r.A_loc_mean,
                       slope_measured_se=r.A_loc_sem, slope_data_teacher_pred=getattr(r, "A_pred", np.nan),
                       lam_hopper=-ev["gamma_k2_max"], lam_measured=r.lam_small_sigma)
        else:
            row.update(law=f"sigma^-{ev['k'] - 2}", exponent_hopper=ev["k"] - 2, exponent_measured=r.p_loc_mean,
                       exponent_measured_se=r.p_loc_sem, lam_measured=r.lam_small_sigma)
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(RES / "data_teacher_random_init.csv", index=False)
    return df


def star_case(tree, n, w, init, max_events):
    pats = ([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]] if n == 4 else
            [[0] * 8, [1, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0]])
    return dict(teacher=dict(kind="pattern", patterns=pats, weights=[2.0, w, w]), tree=tree, n=n, d=2, chi=3,
                init=init, start_component=0 if init == "saddle" else None, max_events=max_events, max_sequences=32)


def star_class(seq):
    """data_teacher's random-init classes for a predicted sequence: 'q last' (q never lost), 'near q'
    (q learned, a_q > 0.72, then forgotten), 'blended' (q forgotten after a first saddle with a_q < 0.72)."""
    evs = seq["events"]
    lost = any(0 in ev["forgotten"] for ev in evs)
    if not lost:
        return "q last"
    aq = evs[0]["overlaps"][0] if evs[0]["overlaps"] else np.nan
    return "near q" if aq > 0.72 else "blended"


def part_c(workers):
    hess = pd.read_csv(DT_RES / "r2_star_hessian.csv")
    cls = pd.read_csv(DT_RES / "r2_star_random_classes.csv")
    cases = {}
    for r in hess.itertuples():
        kind = r.tree.rstrip("0123456789")
        cases[("saddle", r.tree, r.w)] = star_case(kind, int(r.tree[len(kind):]), r.w, "saddle", 1)
    for (tree, w) in cls[["tree", "w"]].drop_duplicates().itertuples(index=False):
        kind = tree.rstrip("0123456789")
        cases[("fanin", tree, w)] = star_case(kind, int(tree[len(kind):]), w, "fanin", 6)
    preds = predict.run_many(list(cases.values()), RES / "predictions_data_teacher.json", workers)
    hrows = []
    for r in hess.itertuples():
        p = preds[predict.key_of(cases[("saddle", r.tree, r.w)])]
        start, ev = first_route(p)
        hrows.append(dict(tree=r.tree, w=r.w, rho=r.rho, strict_measured=bool(r.n_negative > 0),
                          lam_measured=r.lambda_min, strict_hopper=ev["type"].startswith("strict"),
                          lam_hopper=ev["lam_old"], theory=r.pred_if_strict, next_event=ev["kind"],
                          forgotten=str(ev["forgotten"])))
    hdf = pd.DataFrame(hrows)
    hdf.to_csv(RES / "data_teacher_star_hessian.csv", index=False)
    crows = []
    for (tree, w), g in cls.groupby(["tree", "w"]):
        p = preds[predict.key_of(cases[("fanin", tree, w)])]
        pred = {}
        for s in p["sequences"]:
            if s["end"] != "converged":
                continue
            c = star_class(s)
            pred.setdefault(c, set()).add(s["pclass"] if s["pclass"] == "main" else
                                          s["branches"][0].split("(")[0].strip())
            gam = [ev.get("gamma_wrong", 0.0) for ev in s["events"][:1]]
            pred[c].add("gamma=%.3g" % gam[0])
        for c, gc in g.groupby("cls"):
            crows.append(dict(tree=tree, w=w, cls=c, runs=len(gc), predicted=" / ".join(sorted(pred.get(c, {"none"}))),
                              sigmas=f"{gc.sigma.min():g}-{gc.sigma.max():g}"))
        for c in pred:
            if c not in set(g.cls):
                crows.append(dict(tree=tree, w=w, cls=c, runs=0, predicted=" / ".join(sorted(pred[c]))))
    cdf = pd.DataFrame(crows)
    cdf.to_csv(RES / "data_teacher_star_classes.csv", index=False)
    return hdf, cdf


def part_d(workers):
    """Aligned covariance (configs/e1_order.yaml, variant 'sigma'): first component and swaps."""
    import torch

    from ttn.losses import power_law_cov

    ev = torch.diagonal(power_law_cov(4, 0.25)).tolist()
    teachers = {"T1_clean_reversal": [1.0, 0.8], "T2_race_vs_greedy": [1.0, 0.6]}
    eig_index = [1, 0]
    rest = [ev[i] for i in range(4) if i not in eig_index]
    diag = [ev[eig_index[0]], ev[eig_index[1]]] + rest  # component r uses basis vector e_r
    obs = pd.read_csv(DT_RES / "e1_order_summary.csv")
    obs = obs[obs.variant == "sigma"]
    cases = {(t, tree): dict(teacher=dict(kind="aligned_cov", weights=w, site_eigs=[diag] * 4), tree=tree, n=4,
                             d=4, chi=4, init="fanin", max_events=4, max_sequences=16)
             for t, w in teachers.items() for tree in ("binary", "mps")}
    preds = predict.run_many(list(cases.values()), RES / "predictions_data_teacher.json", workers)
    rows = []
    for (t, tree), g in obs.groupby(["teacher", "tree"]):
        p = preds[predict.key_of(cases[(t, tree)])]
        main = [s for s in p["sequences"] if s["pclass"] == "main"]
        firsts = sorted({s["events"][0]["learned"][0] for s in main if s["events"][0]["learned"]})
        swap_main = sorted({any(e["kind"] == "rotation" for e in s["events"]) for s in main})
        rare = [s for s in p["sequences"] if s["pclass"] == "rare"]
        for sig, gs in g.groupby("sigma"):
            rows.append(dict(teacher=t, tree=tree, sigma=sig, runs=len(gs), first_hopper=str(firsts),
                             first_det=len(firsts) == 1, P_first_pred=float((gs["first"] == firsts[0]).mean())
                             if len(firsts) == 1 else np.nan,
                             swap_hopper_main=str(swap_main), P_swap=float((gs.n_unlearn > 0).mean()),
                             rare_gamma=str(sorted({round(s["events"][0].get("gamma_wrong", 0), 3) for s in rare}))))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "data_teacher_aligned_cov.csv", index=False)
    return df


BIN8_ROOT = (2, 5)  # binary8 edges: 0, 1 | 3, 4 bottom (parents 2, 5); 2, 5 = the root bipartition


def edge_order(seq):
    """Edges in the order in which they first reach rank 2 (ties within one event as a set)."""
    seen, out = set(), []
    for ev in seq["events"][1:]:
        new = {e for e, (a, b) in enumerate(zip(ev["from_profile"], ev["to_profile"])) if b >= 2 and a < 2}
        new -= seen
        if new:
            out.append(sorted(new))
            seen |= new
    return out


def p11(order):
    """theory_teachers P11: the root bipartition gains rank only after a bottom bond on each half."""
    seen = set()
    for grp in order:
        if set(grp) & set(BIN8_ROOT):
            return bool(seen & {0, 1}) and bool(seen & {3, 4})
        seen |= set(grp)
    return None


def part_e(workers):
    sweeps = {"e4c_cross": [201, 202, 203], "r2_cross": [203, 204, 209, 212, 218],
              "r2_cross_b": [491, 548, 643, 694, 846]}
    cases = {s: dict(teacher=dict(kind="ttn", tree="mps", ranks=[2] * 7, teacher_seed=s, decay=0.6), tree="binary",
                     n=8, d=2, chi=4, init="fanin", max_events=5, max_sequences=24)
             for seeds in sweeps.values() for s in seeds}
    preds = predict.run_many(list(cases.values()), RES / "predictions_data_teacher.json", workers)
    counts = pd.read_csv(DT_RES / "p11_counts.csv")
    ev_files = {"e4c_cross": "e4c_cross_v2_events.csv", "r2_cross": "r2_cross_v2_events.csv",
                "r2_cross_b": "r2_cross_b_v2_events.csv"}
    rows = []
    for sweep, seeds in sweeps.items():
        evs = pd.read_csv(DT_RES / ev_files[sweep])
        evs = evs[(evs.tree == "binary") & np.isclose(evs.f, 0.05) & (evs.k == 2)]
        for s in seeds:
            p = preds[predict.key_of(cases[s])]
            main = [q for q in p["sequences"] if q["pclass"] == "main"]
            orders = [edge_order(q) for q in main]
            p11s = [p11(o) for o in orders]
            first = sorted({tuple(o[0]) for o in orders if o})
            e = evs[evs.case.str.startswith(f"teacher_seed{s}")]
            obs_first = []
            for _, g in e.groupby(["sigma", "seed"]):
                t0 = g.t.min()
                obs_first.append(tuple(sorted(g[g.t <= t0 * (1 + 1e-9)].edge)))
            hit = [any(set(of) & set(pf) for pf in first) for of in obs_first]
            # stricter: the first two edges to cross form one of the predicted sibling pairs
            pair_hit = []
            for _, g in e.groupby(["sigma", "seed"]):
                two = tuple(sorted(g.sort_values("t").edge.iloc[:2]))
                pair_hit.append(any(two == tuple(pf) for pf in first))
            c = counts[(counts.sweep == sweep) & (counts.teacher_seed == s)]
            rows.append(dict(sweep=sweep, teacher_seed=s, n_main=len(main), p11_hopper=str(sorted(set(map(str, p11s)))),
                             first_edges_hopper=str(first), runs=len(obs_first),
                             first_edge_match=float(np.mean(hit)) if hit else np.nan,
                             first_pair_match=float(np.mean(pair_hit)) if pair_hit else np.nan,
                             p11_measured=f"{int(c.p11.sum())}/{int(c.n.sum())}" if len(c) else "",
                             discriminating=bool(c.discriminating.any()) if len(c) else None))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "data_teacher_cross_tree.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=predict.workers_default())
    ap.add_argument("--parts", default="ABCDE")
    args = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for part, fn in (("A", part_a), ("B", part_b), ("C", part_c), ("D", part_d), ("E", part_e)):
        if part in args.parts:
            out = fn(args.workers)
            for o in (out if isinstance(out, tuple) else (out,)):
                print(o.to_string(), flush=True)


if __name__ == "__main__":
    main()
