"""Validate the hopper on experiments/locality (two- and multi-component pattern teachers, 9 trees).

Inputs: the committed tables in experiments/locality/results/ (exponents, Hessians, levels) and, for the
per-run sequences, the raw run records written by experiments/locality/run.py (JSON lines; locally in
data/locality/, override with LOCALITY_OUT), restricted to the committed configs exactly as
locality's own tables are (`analysis.load_records(config=...)`).

usage: uv run python experiments/saddle_hopper/validate_locality.py [--workers 8]
Writes experiments/saddle_hopper/results/locality_*.csv and predictions_locality.json (cache).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
LOC = ROOT / "experiments" / "locality"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(LOC))
import predict  # noqa: E402

RAW = Path(os.environ.get("LOCALITY_OUT", ROOT / "data" / "locality"))
RES = HERE / "results"
SWEEPS = [
    ("twocomp", "twocomp"),
    ("twocomp_balanced", "twocomp_balanced"),
    ("twocomp_chi4", "twocomp_chi4"),
    ("robust", "robust"),
    ("k5ext", "k5_extension"),
    ("khigh", "khigh_extension"),
    ("lambda2K", "lambda2_K"),
    ("multicomp", "multicomp"),
    ("multicomp_wide", "multicomp_wide"),
]
INIT = {"small": "fanin", "small_balanced": "balanced", "saddle": "saddle"}


def case_of(rec):
    """Hopper case for a locality run record (sigma and seed do not enter the prediction)."""
    return dict(
        teacher=dict(kind="pattern", patterns=rec["patterns"], weights=rec["weights"]),
        tree=rec["kind"],
        n=rec.get("n"),
        d=rec.get("d", 2),
        chi=rec.get("chi", 2),
        init=INIT[rec.get("start", "small")],
        start_component=0,
    )


def observe(rec, classify):
    cross = rec.get("crossings", {})
    R = len(rec["weights"])
    n05 = [len(cross.get(f"a{r}_0.5", [])) for r in range(R)]
    first = [(cross.get(f"a{r}_0.5") or [np.inf])[0] for r in range(R)]
    if "events" in rec:
        learned = sorted({r for _, r in rec["events"]})
    else:  # stalled runs keep only the overlaps where they stalled
        learned = [r for r, a in enumerate(rec.get("a_stall", [])) if a > 0.5]
    ev = rec.get("events", [])
    widths = [final_width(cross, r) for _, r in ev]
    rho = [(tb - ta) / np.nanmax([wa, wb, 1e-300]) for (ta, _), (tb, _), wa, wb in
           zip(ev[:-1], ev[1:], widths[:-1], widths[1:])]
    return dict(cls=classify(rec), status=rec.get("status"), sigma=rec["sigma"], seed=rec["seed"],
                order=tuple(r for _, r in ev), n05=n05,
                first_order=tuple(int(r) for r in np.argsort(first) if np.isfinite(first[r])),
                learned=learned, L_stall=rec.get("L_stall"), rho_min=min(rho) if rho else np.nan,
                transient_dip=transient_dip(cross, R))


def final_width(cross, r):
    """t(a = 0.9) - t(a = 0.1) around the final 1/2 crossing (locality analysis.final_rise_width)."""
    c05 = cross.get(f"a{r}_0.5", [])
    if not c05:
        return np.nan
    t_half = c05[-1]
    lo = [t for t in cross.get(f"a{r}_0.1", []) if t <= t_half]
    hi = [t for t in cross.get(f"a{r}_0.9", []) if t >= t_half]
    return (hi[0] - lo[-1]) if lo and hi else np.nan


def transient_dip(cross, R):
    """Every re-crossing of 1/2 is a dip shorter than the time the component had been learned before it,
    a touch (above 1/2 for less than half a final-rise width) or a wiggle (all its crossings within 3
    final-rise widths): no plateau with the component forgotten, unlike a rotation through a strict saddle."""
    dips = []
    for r in range(R):
        c = cross.get(f"a{r}_0.5", [])
        if len(c) < 3:
            continue
        w = final_width(cross, r)
        if np.isfinite(w) and c[-1] - c[0] < 3 * w:
            dips.append(True)
            continue
        for i in range(1, len(c) - 1, 2):  # c[i] down, c[i+1] up, c[i-1] the previous up
            touch = np.isfinite(w) and c[i] - c[i - 1] < 0.5 * w  # above 1/2 for less than half a width
            dips.append(touch or c[i + 1] - c[i] < c[i] - c[i - 1])
    return bool(dips) and all(dips)


def next_events_after(pred, learned):
    """Predicted events that leave a saddle whose learned set equals `learned` (for stalled runs)."""
    out = []
    for s in pred["sequences"]:
        evs = s["events"]
        for a, b in zip(evs[:-1], evs[1:]):
            if set(a["learned"]) == set(learned):
                out.append((s["pclass"], b))
        if s["end"].startswith("stall") and set(evs[-1]["learned"]) == set(learned):
            out.append((s["pclass"], dict(flags=["blocked: " + s["end"]], k=None, kind="stall")))
    return out


def score(pred, obs):
    """Scoring of notes/saddle_hopper.md section 2 for one locality run."""
    if obs["cls"] in ("timeout", "error", "incomplete"):
        return "unscored", obs["cls"]
    if obs["cls"] == "stalled":
        nxt = next_events_after(pred, obs["learned"])
        flagged = [
            (pc, ev)
            for pc, ev in nxt
            if any(f.startswith(("trap", "blocked")) for f in ev["flags"])
        ]
        if any(pc == "main" for pc, _ in flagged):
            return "main", "stall with trap flag"
        if flagged:
            return "rare", "stall with trap flag (rare branch)"
        ks = sorted({ev.get("k") for _, ev in nxt if ev.get("k")})
        return "mismatch", f"stall, route k={ks} has no trap flag"
    allowed = predict.allowed(pred, "final")
    o, rot = obs["order"], obs["cls"] == "rotation"
    got = allowed.get(o, set())
    if ("main", rot) in got:
        return "main", ""
    if ("rare", rot) in got:
        return "rare", ""
    if got:
        return (
            "mismatch",
            "rotation observed but not predicted" if rot else "rotation predicted, none observed",
        )
    return "mismatch", "order not predicted"


def finite_sigma_add(pred, obs):
    """Wrong first escape without rotation: theory_saddles 4.4 needs the rotation (O(1/|lambda_rot|) in
    natural runs) to beat the direct add route (aligned time sigma^-(k-2) / ((k-2) mu)); within a
    factor 3 the two compete at this sigma."""
    for s in pred["sequences"]:
        for ev in s["events"]:
            if ev["kind"] == "rotation" and ev.get("alt_k") and ev["alt_k"] >= 3:
                k, mu = ev["alt_k"], ev["alt_mu"]
                t_add = obs["sigma"] ** (-(k - 2)) / ((k - 2) * mu)
                if t_add < 3.0 / ev["rate"]:
                    return True, t_add, 1.0 / ev["rate"]
                return False, t_add, 1.0 / ev["rate"]
    return None, np.nan, np.nan


def mismatch_cause(res, why, obs, rec, finite_add):
    """Cause labels of notes/saddle_hopper.md section 2, assigned by explicit checks."""
    if obs["cls"] == "stalled":
        if rec.get("start") == "small_balanced":
            return "stall: pinned or very slow route under balanced init (no dead trap predicted)"
        return "stall without trap flag (theory wrong?)"
    if np.isfinite(obs["rho_min"]) and obs["rho_min"] < 3:
        return "threshold artifact: events not separated (rho < 3), order not resolved"
    if why.startswith("rotation observed") and obs["transient_dip"]:
        return "threshold artifact: transient dip of a learned component during the next escape"
    if why.startswith("rotation predicted") and finite_add:
        return "finite sigma: direct add route competes with the rotation (theory_saddles 4.4)"
    return "unexplained (theory wrong?)"


def compare_two_component(preds):
    """Exponent, log-slope and Hessian rows of locality's two-component tables against the hopper."""
    rows = []
    for tab in ("twocomp_table", "twocomp_balanced_table", "twocomp_chi4_table", "robust_table", "extensions_table"):
        t = pd.read_csv(LOC / "results" / f"{tab}.csv")
        for _, r in t.iterrows():
            if r.tree in ("star3", "midroot6"):
                kind, n = r.tree, None
            else:
                kind = r.tree.rstrip("0123456789")
                n = int(r.tree[len(kind):])
            S = [int(c) for c in r.case.split("=")[1]]
            w = [float(x) for x in r.w.strip("()").split(",")]
            nn = n or 6
            pats = [[0] * nn, [1 if i in S else 0 for i in range(nn)]]
            case = dict(teacher=dict(kind="pattern", patterns=pats, weights=w), tree=kind, n=n, d=int(r.d),
                        chi=int(r.chi), init=INIT[r.start], start_component=0)
            pred = preds.get(predict.key_of(case))
            if pred is None:
                continue
            seqs = [s for s in pred["sequences"] if s["pclass"] == "main" and s["events"][0]["learned"][:1] == [0]
                    and len(s["events"]) > 1]
            if not seqs:
                continue
            ev = seqs[0]["events"][1]
            k = ev.get("k")
            out = dict(table=tab, tree=r.tree, case=r.case, start=r.start, d=r.d, chi=r.chi, w=r.w, k_pred=k,
                       k_locality=r.k_rank, fit_seeds=r.fit_seeds, sigma_range=r.sigma_range)
            if k == 2:
                slope = ev["slope_saddle"] if r.start == "saddle" else ev["slope_small"]
                out.update(quantity="log slope", predicted=slope, measured=r.logslope, se=r.logslope_se,
                           agree=bool(abs(r.logslope / slope - 1) <= 0.1), law=r.law,
                           hess_pred=-ev["rate"], hess_measured=r.hess_min_small,
                           hess_agree=bool(abs(r.hess_min_small / -ev["rate"] - 1) <= 0.1))
            elif k is None or k <= 1 or ev["kind"] != "route":
                out.update(quantity="no plateau (k = 1: joint event)", predicted=0.0, measured=r.p_fit, se=r.p_se,
                           agree=bool(abs(r.p_fit) <= 0.2), law=r.law)
            else:
                out.update(quantity="exponent", predicted=k - 2, measured=r.p_fit, se=r.p_se,
                           agree=bool(abs(r.p_fit - (k - 2)) <= 0.2), law=r.law, hess_pred=k - 2,
                           hess_measured=r.hess_exp, hess_agree=bool(abs(r.hess_exp - (k - 2)) <= 0.3)
                           if np.isfinite(r.hess_exp) else None)
            rows.append(out)
    return pd.DataFrame(rows)


def compare_levels(preds):
    """locality levels_summary.csv (multi-component teachers, per order and level) against the hopper:
    plateau level, escape degree k*, exponent / log slope of the plateau that ends at the next event."""
    t = pd.read_csv(LOC / "results" / "levels_summary.csv")
    cfg = {}
    import yaml

    for name in ("multicomp", "multicomp_wide"):
        for c in yaml.safe_load(open(LOC / "configs" / f"{name}.yaml"))["cases"]:
            cfg[(c["name"], c["kind"] + str(c["n"]), c["start"])] = c
    rows = []
    for _, r in t.iterrows():
        c = cfg.get((r.case, r.tree, r.start))
        if c is None:
            continue
        case = dict(teacher=dict(kind="pattern", patterns=c["patterns"], weights=c["weights"]), tree=c["kind"],
                    n=c["n"], d=c.get("d", 2), chi=c["chi"], init=INIT[r.start], start_component=0)
        pred = preds.get(predict.key_of(case))
        if pred is None:
            continue
        order = tuple(int(x) for x in r.order.split(">"))
        match = [s for s in pred["sequences"] if order in predict.orders(s, "final")[0]]
        j = int(r.level_j)
        e = r.L_mid_err_median
        # measured plateau level: locality stores |L_mid - level| / level against its own route sequence
        meas = [r.level * (1 + e), r.level * (1 - e)] if np.isfinite(e) else []
        best = None
        same = [s for s in match if s["pclass"] == "main" and j + 1 < len(s["events"])
                and abs(s["events"][j]["to_level"] / r.level - 1) < 1e-3]
        if same:  # the branch locality's route sequence describes: the measurement is detected_frac
            s = same[0]
            best = (0.0, s, np.nan, np.nan)
            match = []
        for s in match:
            evs = s["events"]
            if j + 1 >= len(evs):
                continue
            mine = evs[j]["to_level"]
            dev = min((abs(m / mine - 1) for m in meas), default=np.nan)
            key = (dev if np.isfinite(dev) else 9.0) + (0.0 if s["pclass"] == "main" else 0.05)
            if best is None or key < best[0]:
                best = (key, s, dev, min(meas, key=lambda m: abs(m / mine - 1)) if meas else np.nan)
        out = dict(case=r.case, tree=r.tree, start=r.start, order=r.order, level_j=j,
                   level_locality=r.level, k_locality=r.k_star, runs=r.runs, detected_frac=r.detected_frac,
                   L_mid_err_median=e, p_fit=r.p_fit, p_se=r.p_se, p_detected=r.p_detected,
                   logslope=r.logslope, law=r.law, branch=best[1]["pclass"] if best else "none")
        if best:
            evs = best[1]["events"]
            ev = evs[j + 1]
            agree = (r.detected_frac >= 0.5) if same else (best[2] < 0.05)
            out.update(level_hopper=evs[j]["to_level"], level_measured=best[3], level_dev=best[2],
                       level_basis="detected_frac" if same else "L_mid", level_agree=bool(agree),
                       k_hopper=ev.get("k"), kind=ev["kind"], slope_small=ev.get("slope_small"),
                       slope_saddle=ev.get("slope_saddle"), exponent=ev.get("exponent"))
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=predict.workers_default())
    args = ap.parse_args()
    from analysis import classify, load_records  # experiments/locality (read-only)

    RES.mkdir(exist_ok=True)
    recs = []
    for pref, cfg in SWEEPS:
        for r in load_records([str(RAW / pref)], config=[str(LOC / "configs" / f"{cfg}.yaml")]):
            if r.get("patterns") is None or "weights" not in r:
                continue
            r["_sweep"] = pref
            recs.append(r)
    print(f"{len(recs)} locality runs with records", flush=True)
    cases = [case_of(r) for r in recs]
    preds = predict.run_many(cases, RES / "predictions_locality.json", workers=args.workers)
    rows = []
    for rec, case in zip(recs, cases):
        pred = preds[predict.key_of(case)]
        obs = observe(rec, classify)
        res, why = score(pred, obs)
        fs, t_add, t_rot = (None, np.nan, np.nan)
        if res == "mismatch" and why.startswith("rotation predicted"):
            fs, t_add, t_rot = finite_sigma_add(pred, obs)
        cause = mismatch_cause(res, why, obs, rec, fs) if res == "mismatch" else ""
        main_orders = {o for o, v in predict.allowed(pred, "final").items() if any(pc == "main" for pc, _ in v)}
        greedy = [s for s in pred["sequences"] if s["pclass"] == "main" and len(s["events"]) > 1
                  and s["events"][0]["learned"][:1] == [0]]
        k_pred = greedy[0]["events"][1].get("k") if greedy and len(rec["weights"]) == 2 else None
        rows.append(
            dict(
                sweep=rec["_sweep"],
                name=rec.get("name") or "S=" + "".join(map(str, rec["S"])),
                tree=rec["kind"] + (str(rec["n"]) if rec.get("n") else ""),
                start=rec.get("start", "small"),
                d=rec.get("d", 2),
                chi=rec.get("chi", 2),
                w="/".join(f"{x:g}" for x in rec["weights"]),
                sigma=obs["sigma"],
                seed=obs["seed"],
                cls=obs["cls"],
                order=">".join(map(str, obs["order"])),
                first_order=">".join(map(str, obs["first_order"])),
                k_pred=k_pred,
                n_main_orders=len(main_orders),
                result=res,
                why=why,
                cause=cause,
                rho_min=obs["rho_min"],
                finite_sigma_add=fs,
                t_add=t_add,
                t_rot=t_rot,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "locality_runs_scored.csv", index=False)
    lev = compare_levels(preds)
    lev.to_csv(RES / "locality_levels.csv", index=False)
    cmp2 = compare_two_component(preds)
    cmp2.to_csv(RES / "locality_twocomp_laws.csv", index=False)
    print(cmp2.groupby(["table", "quantity"]).agree.agg(["sum", "count"]).to_string())
    print(df.groupby(["sweep", "result"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
