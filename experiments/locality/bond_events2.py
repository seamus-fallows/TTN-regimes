"""Step 4, redone (review issue 2): do the predicted bonds, and only they, gain rank at each event, and
do they rise together?

For every learning event j >= 1 of a run (component r learned; t_lo, t_hi = its a_r = 0.1 / 0.9 crossings
around the final 1/2 crossing; width w = t_hi - t_lo):
  - predicted edges: two-component teachers: the Steiner edges of S; multi-component teachers: the internal
    edges of the minimal route found by the route rule at the actual plateau-midpoint state before the event
    (`rules.route_rule` on the dense W there, residual tolerance 5%);
  - ranks before (plateau midpoint) and after (t_hi, i.e. right after the event, not the end of the run):
    effective rank = #{s_k^e > f x teacher s_k^{e,*}} counting only indices the teacher has (f = 0.05,
    notes/diagnostics.md; sensitivity f = 0.03, 0.1);
  - crossing times: for every predicted edge, the first time its new singular value s_{m+1}^e (m = rank
    before) crosses f x teacher s_{m+1}^{e,*} (log-linear interpolation between stored integrator steps);
    spread = (max - min) / w over the predicted edges, offset = (mean - t_half) / w;
  - leakage: on the non-predicted edges, max over [plateau midpoint, t_hi] of s_{m+1}^e / s_r, and (two
    components) the max over the whole run of s_2^e / s2.

usage: uv run python experiments/locality/bond_events2.py <prefix> [...] --out results/bond_events2.csv --workers 16
"""

from __future__ import annotations

import argparse
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from analysis import load_records  # noqa: E402
from rules import get_tree, route_rule  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402
from ttn.network import TTNSpec, bond_spectra, to_dense  # noqa: E402

FS = (0.03, 0.05, 0.1)
PROFILE = False


def teacher_spectra_np(tree, w_star):
    out = {}
    for v in tree.edges:
        below = sorted(tree.subtree_sites[v])
        rest = [i for i in range(w_star.ndim) if i not in below]
        m = np.transpose(w_star, below + rest).reshape(math.prod(w_star.shape[i] for i in below), -1)
        s = np.linalg.svd(m, compute_uv=False)
        out[v] = s[s > 1e-9 * s[0]]
    return out


def crossing(t, y, level):
    """First time y(t) crosses `level` upward (log-linear interpolation)."""
    above = np.where(y >= level)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return t[0]
    y0, y1 = max(y[i - 1], 1e-300), y[i]
    f = (np.log(level) - np.log(y0)) / (np.log(y1) - np.log(y0)) if y1 > y0 else 1.0
    return t[i - 1] + f * (t[i] - t[i - 1])


def run_rows(rec):
    tree = get_tree(rec["kind"], rec.get("n"))
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    # the actual teacher (random orthonormal site bases, teacher seed), not the standard-basis pattern
    w_star = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec["teacher_seed"]).dense().numpy()
    tsp = teacher_spectra_np(tree, w_star)
    data = np.load(rec["traj"])
    t, Y = data["t"], data["y"]
    two = len(rec["weights"]) == 2
    events, cross = rec["events"], rec["crossings"]
    cache = {}

    def spectra_at(i):
        if i not in cache:
            bs = bond_spectra(spec, spec.unflatten(torch.from_numpy(Y[:, i])))
            cache[i] = {v: bs[v].numpy() for v in tree.edges}
        return cache[i]

    rows = []
    for j in range(1, len(events)):
        t_half, r = events[j]
        p = rec["plateaus"][j - 1]
        if p.get("t_mid") is None:
            continue
        lo = [x for x in cross[f"a{r}_0.1"] if x <= t_half]
        hi = [x for x in cross[f"a{r}_0.9"] if x >= t_half]
        if not lo or not hi:
            continue
        t_lo, t_hi = lo[-1], hi[0]
        w = t_hi - t_lo
        i_mid = int(np.argmin(np.abs(t - p["t_mid"])))
        i_hi = int(np.searchsorted(t, t_hi))
        i_hi = min(i_hi, len(t) - 1)
        if two:
            nodes = tree.steiner_nodes(sorted(rec["S"]))
            predicted = sorted(v for v in tree.edges if v in nodes and tree.parent[v] in nodes)
            k_star = len(nodes)
        else:
            cores = spec.unflatten(torch.from_numpy(Y[:, i_mid]))
            w_mid = to_dense(spec, cores).numpy()
            k_star, hits = route_rule(tree, w_mid, w_star, tol=0.05, sub_tol=1e-2)
            C = set(max(hits, key=lambda h: h[1])[0]) if hits else set()
            predicted = sorted(v for v in tree.edges if v in C and tree.parent[v] in C)
        s_mid, s_hi = spectra_at(i_mid), spectra_at(i_hi)
        t_prev = events[j - 1][0]
        row = dict(tree=tree.name, case=rec.get("name") or "S=" + "".join(map(str, rec["S"])),
                   start=rec["start"], chi=rec["chi"], sigma=rec["sigma"], seed=rec["seed"], event=j,
                   component=r, k_star=k_star, width=w, predicted=str(predicted), key=rec.get("_key"),
                   # separation ratio of the plateau before the event (< 3: no resolved plateau, the
                   # "before" state is already part of the escape)
                   rho=(t_half - t_prev) / w)
        for f in FS:
            def rank(s, v):
                ts = tsp[v]
                return int(sum(s[i] > f * ts[i] for i in range(min(len(ts), len(s)))))
            rising = sorted(v for v in tree.edges if rank(s_hi[v], v) > rank(s_mid[v], v))
            row[f"observed_f{f}"] = str(rising)
            row[f"match_f{f}"] = set(rising) == set(predicted)
        # crossing times of the predicted edges' new singular value (f = 0.05)
        idx = np.arange(i_mid, i_hi + 1)
        if len(idx) > 1500:
            idx = np.unique(np.linspace(i_mid, i_hi, 1500).astype(int))
        tt = t[idx]
        times = []
        for v in predicted:
            m = int(sum(s_mid[v][i] > 0.05 * tsp[v][i] for i in range(min(len(tsp[v]), len(s_mid[v])))))
            if m >= len(tsp[v]):
                continue
            series = np.array([spectra_at(i)[v][m] for i in idx])
            times.append(crossing(tt, series, 0.05 * tsp[v][m]))
        times = np.array([x for x in times if np.isfinite(x)])
        row["n_cross"] = len(times)
        row["spread_over_width"] = float((times.max() - times.min()) / w) if len(times) > 1 else np.nan
        row["offset_over_width"] = float((times.mean() - t_half) / w) if len(times) else np.nan
        # leakage on the other edges during the plateau and the event
        s_r = rec["weights"][r]
        leak = 0.0
        for v in tree.edges:
            if v in predicted:
                continue
            m = int(sum(s_mid[v][i] > 0.05 * tsp[v][i] for i in range(min(len(tsp[v]), len(s_mid[v])))))
            if m < len(s_mid[v]):
                leak = max(leak, max(spectra_at(i)[v][m] for i in idx[:: max(1, len(idx) // 200)]) / s_r)
        row["leak_window"] = leak
        if two:  # rest of the run from the plateau midpoint (after the initial noise has relaxed)
            sub = np.unique(np.linspace(i_mid, len(t) - 1, min(len(t) - i_mid, 400)).astype(int))
            row["leak_run"] = max((spectra_at(i)[v][1] / s_r for i in sub for v in tree.edges if v not in predicted),
                                  default=0.0)
        rows.append(row)
    return rows


SEQS: dict = {}


def profile_rows(rec):
    """Multi-component teachers: at every predicted saddle (route rule at rank-constrained optima,
    results/route_sequences.json) take the middle of the time the loss spends within 2% of the predicted
    level, and compare the observed effective bond ranks (f = 0.05) with the predicted rank profile."""
    import json

    if not SEQS:
        SEQS.update(json.loads((Path(__file__).parent / "results" / "route_sequences.json").read_text()))
    tree = get_tree(rec["kind"], rec.get("n"))
    seq = SEQS.get(f"{rec.get('name')}_{tree.name}")
    if seq is None:
        return []
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"], seed=rec["teacher_seed"])
    w_star = teacher.dense().numpy()
    tsp = teacher_spectra_np(tree, w_star)
    t2 = teacher.norm2()
    data = np.load(rec["traj"])
    t, Y = data["t"], data["y"]
    from ttn.losses import population_loss

    sub = np.unique(np.linspace(0, len(t) - 1, min(len(t), 3000)).astype(int))
    L = np.array([population_loss(spec, spec.unflatten(torch.from_numpy(Y[:, i])), teacher, teacher_norm2=t2).item()
                  for i in sub])
    rows = []
    for j, step in enumerate(seq):
        if step["k_star"] == 0:
            continue
        inside = np.where(np.abs(L / step["loss"] - 1) < 0.02)[0]
        if len(inside) < 3:
            rows.append(dict(tree=tree.name, case=rec["name"], start=rec["start"], sigma=rec["sigma"],
                             seed=rec["seed"], level_j=j, resolved=False))
            continue
        i = sub[inside[len(inside) // 2]]
        bs = bond_spectra(spec, spec.unflatten(torch.from_numpy(Y[:, i])))
        observed = {v: int(sum(bs[v][q] > 0.05 * tsp[v][q] for q in range(min(len(tsp[v]), len(bs[v])))))
                    for v in tree.edges}
        predicted = {int(k): v for k, v in step["profile"].items()}
        rows.append(dict(tree=tree.name, case=rec["name"], start=rec["start"], sigma=rec["sigma"], seed=rec["seed"],
                         level_j=j, resolved=True, match=observed == predicted,
                         observed=str(observed), predicted=str(predicted)))
    return rows


def _set_profile(flag):
    global PROFILE
    PROFILE = flag


def safe_rows(rec):
    try:
        if PROFILE:
            return profile_rows(rec)
        return run_rows(rec)
    except Exception as exc:  # noqa: BLE001
        return [dict(error=repr(exc)[:200], key=rec.get("_key"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--profiles", action="store_true", help="multi-component: rank profile at each predicted saddle")
    args = ap.parse_args()
    global PROFILE
    PROFILE = args.profiles
    recs = [r for r in load_records(args.inputs) if r.get("status") not in ("timeout", "error", "stalled")
            and r.get("traj") and len(r.get("events", [])) >= 2]
    if args.max_runs and len(recs) > args.max_runs:
        recs = [recs[i] for i in np.random.default_rng(0).choice(len(recs), args.max_runs, replace=False)]
    with Pool(args.workers, initializer=_set_profile, initargs=(PROFILE,)) as pool:
        rows = [x for xs in pool.imap_unordered(safe_rows, recs, chunksize=4) for x in xs]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    if PROFILE:
        ok = df[df["resolved"] == True]  # noqa: E712
        print("levels", len(df), "resolved", len(ok), "profile match", ok["match"].mean())
        print(ok.groupby(["tree", "case", "level_j"])["match"].agg(["mean", "size"]).to_string())
        return
    ok = df[df.get("error").isna()] if "error" in df else df
    print("events", len(ok), "errors", len(df) - len(ok))
    for f in FS:
        print(f"f = {f}: match {ok[f'match_f{f}'].mean():.4f}")
    print("spread/width median", ok["spread_over_width"].median(), "95%", ok["spread_over_width"].quantile(0.95))
    print("leak_window max", ok["leak_window"].max(), "leak_run max", ok.get("leak_run", pd.Series([np.nan])).max())


if __name__ == "__main__":
    main()
