"""Independent relabel of the copied regime-map runs with the last-crossing rule.

For each library row (pre-fix: no 'event_rule'): merge log + fine rows of its trajectory, and for every
component that crossed 1/2 upward more than once before first reaching 0.9, replace its event time by the
last upward crossing of 1/2 before that point. Widths: stored (refined final-rise widths). Label rule of
notes/diagnostics.md. Output: one CSV per experiment.
Usage: uv run python relabel.py EXP [EXP ...]
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "c:/Users/seamu/Iliad-ML/TTN_regimes"
sys.path.insert(0, os.path.join(ROOT, "experiments/regime_map"))
from run_regime import row_cfg, run_id  # noqa: E402  (file names only)

RAW = os.path.join(ROOT, "data/regime_map")
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def merged(z):
    t, A = z["t"], z["overlaps"]
    if "fine_t" not in z.files:
        return t, A
    ft, fA = z["fine_t"], z["fine_overlaps"]
    new = ~np.isin(ft, t)
    tt = np.concatenate([t, ft[new]])
    AA = np.concatenate([A, fA[new]])
    o = np.argsort(tt, kind="stable")
    return tt[o], AA[o]


def comp_info(t, a, mid=0.5, hi=0.9, interp="log"):
    """(n_up, t_last) for one component; n_up counts upward crossings of mid before first reach of hi."""
    reach = np.flatnonzero(a >= hi)
    if reach.size == 0:
        return -1, np.nan
    ih = reach[0]
    above = a[: ih + 1] >= mid
    n_up = int(above[0]) + int(np.sum(above[1:] & ~above[:-1]))
    below = np.flatnonzero(~above)
    if below.size == 0:
        return n_up, float(t[0])
    j = below[-1]
    f = (mid - a[j]) / (a[j + 1] - a[j])
    if interp == "log" and t[j] > 0:
        return n_up, float(np.exp(np.log(t[j]) + f * (np.log(t[j + 1]) - np.log(t[j]))))
    return n_up, float(t[j] + f * (t[j + 1] - t[j]))


def ratios(times, widths):
    times, widths = np.asarray(times, float), np.asarray(widths, float)
    k = np.isfinite(times)
    o = np.argsort(times[k], kind="stable")
    t, w = times[k][o], widths[k][o]
    if len(t) < 2:
        return np.array([])
    sc = np.maximum(w[:-1], w[1:])
    return np.diff(t) / sc


def label(l_end, dK, rh):
    if l_end is None or not np.isfinite(l_end) or l_end >= 1e-3:
        return "stalled"
    if dK < 0.1:
        return "lazy"
    if rh.size >= 1 and np.all(rh > 3):  # NaN fails
        return "s2s"
    return "rich"


MAP = {"saddle-to-saddle": "s2s"}


def run(exp):
    rows = []
    with open(os.path.join(RAW, exp, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r or "event_times" not in r:
                continue
            rows.append(r)
    # de-duplicate on the file name (keep the latest row)
    seen = {}
    for i, r in enumerate(rows):
        cfg = {k: v for k, v in row_cfg(r).items() if v is not None}
        seen[run_id(cfg)] = i
    out = []
    for rid, i in seen.items():
        r = rows[i]
        path = os.path.join(RAW, exp, "traj", rid + ".npz")
        times = np.asarray(
            r["event_times"] if r.get("event_rule") != "last" else r["event_times_old"], float
        )
        widths = np.asarray(r["event_widths"], float)
        rec = dict(
            exp=exp,
            rid=rid,
            model=r.get("model"),
            kind=r.get("kind"),
            n=r.get("n"),
            d=r.get("d"),
            chi=r.get("chi"),
            sigma=r.get("sigma"),
            seed=r.get("seed"),
            K=r.get("K"),
            root_scale=r.get("root_scale"),
            alpha=r.get("alpha"),
            lr=r.get("lr"),
            balance=r.get("balance"),
            l_end=r.get("l_end"),
            dK=r.get("dK"),
            stored_label=MAP.get(r.get("label"), r.get("label")),
            event_rule=r.get("event_rule"),
        )
        rh_old = ratios(times, widths)
        rec["label_old_mine"] = label(rec["l_end"], rec["dK"], rh_old)
        rec["min_rho_old"] = float(np.min(rh_old)) if rh_old.size else np.nan
        if not os.path.exists(path):
            rec.update(
                found=False,
                n_up="",
                label_new=rec["label_old_mine"],
                min_rho_new=rec["min_rho_old"],
                min_rho_new_lin=np.nan,
                changed_t=False,
            )
            out.append(rec)
            continue
        with np.load(path, allow_pickle=False) as z:
            t, A = merged(z)
        t_new, t_lin, nups = times.copy(), times.copy(), []
        for c in range(A.shape[1]):
            n_up, tl = comp_info(t, A[:, c])
            n_up2, tl2 = comp_info(t, A[:, c], interp="lin")
            nups.append(n_up)
            if n_up > 1:
                t_new[c], t_lin[c] = tl, tl2
        rh = ratios(t_new, widths)
        rh_lin = ratios(t_lin, widths)
        rec.update(
            found=True,
            n_up=json.dumps(nups),
            label_new=label(rec["l_end"], rec["dK"], rh),
            label_new_lin=label(rec["l_end"], rec["dK"], rh_lin),
            min_rho_new=float(np.min(rh)) if rh.size else np.nan,
            min_rho_new_lin=float(np.min(rh_lin)) if rh_lin.size else np.nan,
            changed_t=bool(np.any(~np.isclose(t_new, times, equal_nan=True))),
            times_new=json.dumps([float(x) for x in t_new]),
            widths=json.dumps([float(x) for x in widths]),
        )
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUTD, f"relabel_{exp}.csv"), index=False)
    n_bump = int(df.n_up.astype(str).str.contains(r"\b[2-9]\b").sum())
    ch = df[df.label_new != df.stored_label]
    print(
        exp,
        "rows",
        len(rows),
        "unique",
        len(df),
        "found",
        int(df.found.sum()),
        "bump",
        n_bump,
        "stored!=mine_old",
        int((df.stored_label != df.label_old_mine).sum()),
        "label changes",
        len(ch),
        dict(zip(*np.unique((ch.stored_label + "->" + ch.label_new).values, return_counts=True)))
        if len(ch)
        else {},
        "lin-vs-log label diffs",
        int((df.get("label_new_lin", df.label_new) != df.label_new).sum()),
        flush=True,
    )


if __name__ == "__main__":
    for e in sys.argv[1:]:
        run(e)
