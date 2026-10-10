"""Validate the hopper on experiments/s2s_core (odeco teachers, 3 components, d = 4, five trees).

Inputs (committed): experiments/s2s_core/results/main_runs.csv (per run: class, order of first full
rises, 1-based), main_exponents.csv (per-seed exponent fits), main_strict_saddles.csv.
usage: uv run python experiments/saddle_hopper/validate_s2s_core.py [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
S2S = ROOT / "experiments" / "s2s_core" / "results"
sys.path.insert(0, str(HERE))
import predict  # noqa: E402
from validate_locality import next_events_after  # noqa: E402

RES = HERE / "results"
WEIGHTS = {"separated": [3.0, 2.0, 1.0], "close": [1.0, 0.85, 0.7]}


def split_tree(name):
    kind = name.rstrip("0123456789")
    return kind, int(name[len(kind) :])


def case_of(tree, chi, teacher, init):
    kind, n = split_tree(tree)
    return dict(
        teacher=dict(kind="odeco", weights=WEIGHTS[teacher]),
        tree=kind,
        n=n,
        d=4,
        chi=int(chi),
        init=init,
    )


def score(pred, row):
    cls = row["class"]
    order = (
        tuple(int(x) - 1 for x in str(row["order"]).split("-"))
        if isinstance(row["order"], str)
        else ()
    )
    if cls in ("stalled", "unresolved"):
        return "unscored", cls
    if cls == "trapped":
        nxt = next_events_after(pred, sorted(order))
        flagged = [pc for pc, ev in nxt if any(f.startswith("trap") for f in ev["flags"])]
        if "main" in flagged:
            return "main", "trap flagged"
        if flagged:
            return "rare", "trap flagged (rare branch)"
        return "mismatch", "trap without flag (init %s)" % row["init"]
    allowed = predict.allowed(pred, "first")
    got = {pc for pc, _ in allowed.get(order, set())}
    if "main" in got:
        return "main", ""
    if "rare" in got:
        return "rare", ""
    return "mismatch", "order not predicted"


def event_exponent(pred, j):
    """Exponents of event j over the main sequences (set; None for log/rotation events)."""
    out = set()
    for s in pred["sequences"]:
        if s["pclass"] == "main" and len(s["events"]) > j:
            out.add(s["events"][j].get("exponent"))
    return sorted(x for x in out if x is not None) or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=predict.workers_default())
    args = ap.parse_args()
    runs = pd.read_csv(S2S / "main_runs.csv")
    keys = runs[["tree", "chi", "teacher", "init"]].drop_duplicates().values.tolist()
    cases = {tuple(k): case_of(*k) for k in keys}
    preds = predict.run_many(
        list(cases.values()), RES / "predictions_s2s_core.json", workers=args.workers
    )
    out = []
    for _, row in runs.iterrows():
        pred = preds[predict.key_of(cases[(row.tree, row.chi, row.teacher, row.init)])]
        res, why = score(pred, row)
        out.append(
            dict(
                sweep=row.sweep,
                tree=row.tree,
                chi=row.chi,
                teacher=row.teacher,
                init=row.init,
                sigma=row.sigma,
                seed=row.seed,
                cls=row["class"],
                order=row.order,
                result=res,
                cause=why,
            )
        )
    df = pd.DataFrame(out)
    df.to_csv(RES / "s2s_core_runs_scored.csv", index=False)
    summ = (
        df.groupby(["tree", "chi", "teacher", "init"]).result.value_counts().unstack(fill_value=0)
    )
    summ.to_csv(RES / "s2s_core_match_summary.csv")
    print(summ.to_string())
    # exponents: first escape K - 2; gaps: route degree of events 1, 2 of the main sequences
    ex = pd.read_csv(S2S / "main_exponents.csv")
    rows = []
    for _, r in ex.iterrows():
        pred = preds.get(predict.key_of(case_of(r.tree, r.chi, r.teacher, r.init)))
        if pred is None:
            continue
        for col, j in (
            ("p_t1", 0),
            ("p_gap1", 1),
            ("p_gap2", 2),
            ("lam1_slope", 1),
            ("lam2_slope", 2),
        ):
            if not np.isfinite(r[col]):
                continue
            e = event_exponent(pred, j)
            if e is None:
                continue
            se = r.get(col + "_se", np.nan)
            rows.append(
                dict(
                    sweep=r.sweep,
                    tree=r.tree,
                    chi=r.chi,
                    teacher=r.teacher,
                    init=r.init,
                    quantity=col,
                    measured=r[col],
                    se=se,
                    predicted=e[0] if len(e) == 1 else str(e),
                    agree=bool(
                        len(e) == 1 and abs(r[col] - e[0]) <= (0.3 if "lam" in col else 0.2)
                    ),
                )
            )
    pd.DataFrame(rows).to_csv(RES / "s2s_core_exponents.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    # strict wrong-order saddle (binary4, close teacher: {1, 3} learned, 2 not): rotation rate
    st = pd.read_csv(S2S / "main_strict_saddles.csv")
    srows = []
    for _, r in st.iterrows():
        pred = preds[predict.key_of(case_of(r.tree, r.chi, r.teacher, r.init))]
        learned = sorted(int(x) - 1 for x in str(r.learned).split("-"))
        rates = sorted(
            {
                round(ev["rate"], 6)
                for s in pred["sequences"]
                for a, ev in zip(s["events"][:-1], s["events"][1:])
                if ev["kind"] == "rotation" and sorted(a["learned"]) == learned
            }
        )
        srows.append(
            dict(
                tree=r.tree,
                chi=r.chi,
                teacher=r.teacher,
                seed=r.seed,
                learned=r.learned,
                lam_theory=r.lam_pred,
                lam_hopper=-rates[0] if rates else np.nan,
                lam_measured_median_ratio=r.lam_over_pred_median,
                slope=r.slope,
            )
        )
    pd.DataFrame(srows).to_csv(RES / "s2s_core_strict.csv", index=False)
    print(pd.DataFrame(srows).to_string())


if __name__ == "__main__":
    main()
