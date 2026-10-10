"""Collect the per-run summaries of a sweep (run.py) into one table and summary tables.

    uv run python experiments/s2s_core/analyze_main.py [--raw /root/outputs/s2s_core/main] [--out results]

Writes <out>/main_runs.csv (one row per run) and <out>/main_summary.csv (per tree, chi, teacher,
sigma: run counts by class, S2S fraction, plateau-level errors, Hessian summary). Classes:
  complete    all components learned, every event resolved
  trapped     not complete; a never-learned component's slice on some core fell below 1e-8 sigma
              after the first escape (dead route end, the conservation-law trap; any later escape
              would be seeded by float64 roundoff, so it is not reported as physics)
  stalled     not complete, all slices alive, and the run reached t_max (1e3 x the RMS-overlap
              escape time of the weakest component)
  unresolved  not complete, all slices alive, stopped by the step / wall budget or by the
              amplitude-based roundoff stop of main.yaml (that stop is too strict at K >= 7)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FLOOR = 1e-14


# Cores carrying physical legs: the piggyback block is a strict (Hessian) direction only when the
# differing sites sit on m = 2 such cores (theory_saddles 4.1); for odeco teachers m = this count.
N_SITE_CORES = {"binary4": 2, "binary8": 4, "mps4": 4, "mps8": 8, "ht4": 4}


def load(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, prow = [], []
    for f in sorted(raw.glob("*.json")):
        if f.name == "summary.json":
            continue
        s = json.loads(f.read_text())
        R = len(s["weights"])
        ev = np.array([np.nan if x is None else x for x in s["events"]], dtype=float)
        order = s["order"]
        t_sorted = [ev[r] for r in order] + [np.nan] * (R - len(order))
        w = np.array([np.nan if x is None else x for x in s["widths"]], dtype=float)
        complete = len(order) == R and np.all(np.isfinite(ev))
        # Smallest dormant amplitude after the first escape (intervals 1..), over components that are
        # still dormant then.
        after = [v for row in s["dormant_min_between_events"][1:] for v in row.values()]
        min_after = min(after) if after else np.nan
        # Dead route end: a never-learned component whose slice on some core (cores with sites,
        # which include every route end) fell below 1e-8 sigma after the first escape: the
        # conservation-law trap; any later escape of it is seeded by float64 roundoff.
        min_slice = np.nan
        if not complete and order:
            d = np.load(f.with_suffix(".npz"))
            after = d["t"] > 1.2 * ev[order[0]]
            never = [r for r in range(R) if r not in order]
            if after.any() and never:
                min_slice = float(np.nanmin(d["slice_norms"][after][:, :, never])) / s["sigma"]
        if complete:
            cls = "complete"
        elif np.isfinite(min_slice) and min_slice < 1e-8:
            cls = "trapped"
        elif s["stop_reason"] == "completed":
            cls = "stalled"
        else:
            cls = "unresolved"  # stopped by the step / wall budget or the amplitude stop while alive
        plateaus = s["plateaus"]
        rel_level = [abs(p["loss"] - p["partial_fit_level"]) / p["partial_fit_level"] for p in plateaus]
        # Bond-rank-constrained critical point (lead): at the plateau midpoint after learning S, every
        # edge's top |S| singular values equal the learned weights (truncated teacher spectrum) and the
        # rest are small. Max relative deviation over edges, and max next singular value / min weight.
        spec_dev, spec_next = [], []
        for i, p in enumerate(plateaus):
            learned = sorted((s["weights"][q] for q in order[: i + 1]), reverse=True)
            k = len(learned)
            for sv in p["spectra"]:
                sv = sorted(sv, reverse=True)
                spec_dev.append(max(abs(a - b) / b for a, b in zip(sv[:k], learned)))
                if len(sv) > k:
                    spec_next.append(sv[k] / min(s["weights"]))
        lam_min = [p["saddle"]["lambda_min"] for p in plateaus if "saddle" in p]
        lam_max = [p["saddle"]["lambda_max"] for p in plateaus if "saddle" in p]
        n_neg = [p["saddle"]["n_negative"] for p in plateaus if "saddle" in p]
        # Edge check recomputed from the saved event-window spectra with a threshold f s_r that
        # avoids the weights already learned (s_r / 2 can coincide with one, e.g. 2 / 2 = 1).
        edge_ok, edge_spread = [], []
        if complete:
            dz = np.load(f.with_suffix(".npz"))
            for j, r in enumerate(order):
                key = f"fine_spectra_{r}"
                if key not in dz:
                    continue
                sr = s["weights"][r]
                prev = [s["weights"][q] for q in order[:j]]
                fsel = next((fr for fr in (0.5, 0.75, 0.3, 0.85, 0.2)
                             if all(abs(fr * sr - x) >= 0.15 * sr for x in prev)), 0.5)
                tz, sz = dz[f"fine_t_{r}"], np.nan_to_num(dz[key])
                rk = (sz > fsel * sr).sum(axis=-1)
                steps = [tz[np.flatnonzero(rk[:, e] >= rk[-1, e])[0]] for e in range(rk.shape[1])]
                edge_ok.append(bool(np.all(rk[-1] - rk[0] == 1)))
                if np.isfinite(w[r]) and w[r] > 0:
                    edge_spread.append((max(steps) - min(steps)) / w[r])
        # theory_saddles R10 (k = K for odeco, A = 1): at a plateau midpoint the most negative
        # eigenvalue is ~ -(K-1) s_r b^(K-2), b = (s_r |a_r|)^(1/K), for the dormant component r with
        # the largest such value.
        lam_pred = []
        for i, p in enumerate(plateaus):
            dormant = [r for r in range(R) if r not in order[: i + 1]]
            vals = [-(s["K"] - 1) * s["weights"][r] * (s["weights"][r] * abs(p["overlaps"][r])) ** ((s["K"] - 2) / s["K"])
                    for r in dormant]
            lam_pred.append(min(vals) if vals else np.nan)
        transient = [r for r in range(R) if np.isfinite(ev[r]) and s["first_crossings"][r] is not None
                     and s["first_crossings"][r] < ev[r] * (1 - 1e-6)]
        row = {
            "sweep": raw.name,
            "tree": f"{s['tree']}{s['N']}", "K": s["K"], "chi": s["chi"], "teacher": s["teacher"],
            "sigma": s["sigma"], "seed": s["seed"], "init": s["init"], "class": cls,
            "stop": s["stop_reason"], "n_learned": len(order), "order": "-".join(str(r + 1) for r in order),  # 1-based, survives CSV round trips
            "greedy": order == sorted(order) and len(order) == R,
            "t1": t_sorted[0], "t2": t_sorted[1], "t3": t_sorted[2] if R > 2 else np.nan,
            "w_max": np.nanmax(w) if np.isfinite(w).any() else np.nan,
            "rho_min": min(s["separation_ratios"]) if s["separation_ratios"] else np.nan,
            "s2s": s["s2s"], "relearned": ",".join(map(str, transient)),
            "max_rel_plateau_level_err": max(rel_level) if rel_level else np.nan,
            "max_rel_plateau_spectrum_err": max(spec_dev) if spec_dev else np.nan,
            "max_plateau_next_sv_over_smin": max(spec_next) if spec_next else np.nan,
            "plateau1_lam_min": lam_min[0] if lam_min else np.nan,
            "plateau2_lam_min": lam_min[1] if len(lam_min) > 1 else np.nan,
            "plateau1_lam_max": lam_max[0] if lam_max else np.nan,
            "plateau1_lam_pred": lam_pred[0] if lam_pred else np.nan,
            "plateau2_lam_pred": lam_pred[1] if len(lam_pred) > 1 else np.nan,
            "plateau1_n_neg": n_neg[0] if n_neg else np.nan,
            "plateau2_n_neg": n_neg[1] if len(n_neg) > 1 else np.nan,
            "min_dormant_after_first": min_after,
            "min_dormant_slice_over_sigma": min_slice,
            # Spread over edges of the time each edge's rank steps up, relative to the event width
            # (runs with event-window spectra only), and whether every edge gained exactly one value.
            "edge_spread_over_width_max": max(
                [(max(v) - min(v)) / w[int(r)] for r, v in s.get("edge_rank_step_times", {}).items()
                 if np.isfinite(w[int(r)]) and w[int(r)] > 0] or [np.nan]),
            "edge_rank_change_all_one": all(edge_ok) if edge_ok else np.nan,
            "edge_spread_over_width_max2": max(edge_spread) if edge_spread else np.nan,
            "drift": s["max_conservation_drift"], "q0_max": s["q0_max"], "wall": s["wall_time"],
        }
        rho = s["separation_ratios"] + [np.nan, np.nan]
        row["rho1"], row["rho2"] = rho[0], rho[1]
        rows.append(row)
        # One row per plateau midpoint: which events precede it (`after`), the learned set, the
        # route prediction (R10) and, on trees with m = 2, the piggyback prediction of a strict
        # saddle when a larger component is unlearned: s_q^((K-2)/K) (s_q - max_r s_r).
        ev_sorted = sorted(t for t in t_sorted if np.isfinite(t))
        for pos, p in enumerate(plateaus, 1):
            after = int(sum(t < p["t_mid"] for t in ev_sorted))
            learned = order[:after]
            unlearned = [r for r in range(R) if r not in learned]
            pig = np.nan
            if N_SITE_CORES.get(row["tree"]) == 2 and learned and unlearned:
                smax = max(s["weights"][r] for r in unlearned)
                cands = [s["weights"][q] ** ((s["K"] - 2) / s["K"]) * (s["weights"][q] - smax)
                         for q in learned if s["weights"][q] < smax]
                pig = min(cands) if cands else np.nan
            prow.append({k: row[k] for k in ("sweep", "tree", "K", "chi", "teacher", "init", "sigma", "seed",
                                             "class", "order")} | {
                "plateau": pos, "after": after, "learned": "-".join(str(q + 1) for q in learned),
                "t_mid": p["t_mid"], "loss_rel_err": abs(p["loss"] - p["partial_fit_level"]) / p["partial_fit_level"],
                "rho_next": rho[after - 1] if 1 <= after <= len(s["separation_ratios"]) else np.nan,
                "lam_min_stored": p.get("saddle", {}).get("lambda_min", np.nan),
                "lam_pred_route": lam_pred[pos - 1] if pos - 1 < len(lam_pred) else np.nan,
                "lam_pred_piggyback": pig})
    return pd.DataFrame(rows), pd.DataFrame(prow)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["sweep", "tree", "K", "chi", "teacher", "init", "sigma"])
    return g.agg(
        n=("seed", "size"),
        complete=("class", lambda c: int((c == "complete").sum())),
        trapped=("class", lambda c: int((c == "trapped").sum())),
        stalled=("class", lambda c: int((c == "stalled").sum())),
        unresolved=("class", lambda c: int((c == "unresolved").sum())),
        s2s=("s2s", "sum"),
        greedy=("greedy", "sum"),
        rho_min_median=("rho_min", "median"),
        level_err_max=("max_rel_plateau_level_err", "max"),
        drift_max=("drift", "max"),
    ).reset_index()


def gap_fits(df: pd.DataFrame, plat: pd.DataFrame, min_sigmas: int = 4) -> pd.DataFrame:
    """Per-seed power-law fits against sigma, each on the runs where that step is a resolved S2S
    step of the seed's modal saddle sequence (review s2s_core round 1, issue 1):
      t_1    runs whose first component is the seed's modal first component;
      gap1   t_2 - t_1 on runs with >= 2 events, first two components = the seed's modal pair, rho_1 > 3;
      gap2   t_3 - t_2 on complete runs in the seed's modal order with rho_1, rho_2 > 3.
    Plateau 1 / 2 Hessian slopes (log(-lambda_min) vs log sigma, projected and raw) use the same run
    sets. Exponent p in t ~ sigma^-p; at least `min_sigmas` sigmas per fit.
    """
    def fit(x, y):
        ok = np.isfinite(y) & (y > 0)
        return -np.polyfit(x[ok], np.log(y[ok]), 1)[0] if ok.sum() >= min_sigmas else np.nan

    def slope(x, lam):
        neg = np.isfinite(lam) & (lam < 0)
        return np.polyfit(x[neg], np.log(-lam[neg]), 1)[0] if neg.sum() >= min_sigmas else np.nan

    rows = []
    keys = ["sweep", "tree", "K", "chi", "teacher", "init", "seed"]
    pl = plat.set_index(keys[:-1] + ["sigma", "seed", "after"]) if len(plat) else None
    for key, g in df[df.n_learned >= 1].groupby(keys):
        row = dict(zip(keys, key))
        for i in (1, 2):
            row.update({f"lam{i}_slope": np.nan, f"lam{i}_slope_raw": np.nan,
                        f"lam{i}_over_pred_median": np.nan, f"sigma_range_gap{i}": ""})
        first = g.order.str.split("-").str[0]
        g1 = g[first == first.mode().iloc[0]].sort_values("sigma")
        row["p_t1"] = fit(np.log(g1.sigma.values), g1.t1.values)
        row["n_t1"] = len(g1)
        two = g[g.n_learned >= 2]
        pair = two.order.str.split("-").str[:2].str.join("-")
        sets = {}
        if len(two):
            modal2 = pair.mode().iloc[0]
            sets[1] = two[(pair == modal2) & (two.rho1 > 3)].sort_values("sigma")
            row["order2"] = modal2
        comp = g[g["class"] == "complete"]
        if len(comp):
            modal = comp.order.mode().iloc[0]
            sets[2] = comp[(comp.order == modal) & (comp.rho1 > 3) & (comp.rho2 > 3)].sort_values("sigma")
            row["order"] = modal
        for i, col in ((1, "gap1"), (2, "gap2")):
            d = sets.get(i)
            if d is None or not len(d):
                row[f"p_{col}"], row[f"n_{col}"] = np.nan, 0
                continue
            x = np.log(d.sigma.values)
            y = (d.t2 - d.t1).values if i == 1 else (d.t3 - d.t2).values
            row[f"p_{col}"], row[f"n_{col}"] = fit(x, y), len(d)
            row[f"sigma_range_{col}"] = f"{d.sigma.min():g}-{d.sigma.max():g}"
            # Hessian at the plateau that precedes this step (after = i events).
            lam, raw, pred = [], [], []
            for _, r in d.iterrows():
                k = (*key[:-1], r.sigma, r.seed, i)
                if pl is not None and k in pl.index:
                    e = pl.loc[[k]].iloc[0]
                    lam.append(e.get("lam_min", np.nan)), raw.append(e.get("lam_min_raw", np.nan))
                    pred.append(e.lam_pred_route)
                else:
                    lam.append(np.nan), raw.append(np.nan), pred.append(np.nan)
            lam, raw, pred = map(np.asarray, (lam, raw, pred))
            row[f"lam{i}_slope"], row[f"lam{i}_slope_raw"] = slope(x, lam), slope(x, raw)
            ok = np.isfinite(lam) & (lam < 0) & np.isfinite(pred) & (pred < 0)
            row[f"lam{i}_over_pred_median"] = float(np.median(lam[ok] / pred[ok])) if ok.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def strict_saddles(plat: pd.DataFrame) -> pd.DataFrame:
    """Plateaus where the piggyback block predicts a strict saddle (m = 2 trees with a larger component
    unlearned): measured lambda_min (projected, raw) vs the prediction, per (seed, learned set), with
    the slope of log(-lambda_min) vs log(sigma) (~0 for a strict saddle)."""
    x = plat[np.isfinite(plat.lam_pred_piggyback)]
    rows = []
    for key, g in x.groupby(["sweep", "tree", "K", "chi", "teacher", "init", "seed", "learned"]):
        g = g.sort_values("sigma")
        lam = g.lam_min.values
        neg = np.isfinite(lam) & (lam < 0)
        rows.append({**dict(zip(["sweep", "tree", "K", "chi", "teacher", "init", "seed", "learned"], key)),
                     "n_sigma": len(g), "sigma_range": f"{g.sigma.min():g}-{g.sigma.max():g}",
                     "orders": ",".join(sorted(set(g.order))),
                     "lam_pred": g.lam_pred_piggyback.iloc[0],
                     "lam_min_proj_range": f"{np.nanmin(lam):.3f}..{np.nanmax(lam):.3f}",
                     "lam_min_raw_range": f"{np.nanmin(g.lam_min_raw):.3f}..{np.nanmax(g.lam_min_raw):.3f}",
                     "lam_over_pred_median": float(np.nanmedian(lam / g.lam_pred_piggyback.values)),
                     "slope": np.polyfit(np.log(g.sigma[neg]), np.log(-lam[neg]), 1)[0] if neg.sum() >= 3 else np.nan,
                     "rho_next_median": float(np.nanmedian(g.rho_next))})
    return pd.DataFrame(rows)


def class_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run classes per (sweep, tree, chi, teacher, init), pooled over sigma."""
    g = df.groupby(["sweep", "tree", "K", "chi", "teacher", "init"])
    out = g["class"].value_counts().unstack(fill_value=0).reset_index()
    out["n"] = g.size().values
    out["sigmas"] = g.sigma.apply(lambda x: f"{x.min():g}-{x.max():g}").values
    return out


def order_table(df: pd.DataFrame) -> pd.DataFrame:
    """Learning order: per (tree, chi, teacher), the fraction of runs with each first component, the
    greedy fraction, and the fraction of seeds whose first component is the same at every sigma."""
    rows = []
    for key, g in df[df.n_learned >= 1].groupby(["sweep", "tree", "K", "chi", "teacher", "init"]):
        first = g.order.astype(str).str.split("-").str[0]
        same = g.assign(first=first).groupby("seed")["first"].nunique()
        row = dict(zip(["sweep", "tree", "K", "chi", "teacher", "init"], key))
        row.update({"n_runs": len(g), **{f"P_first_{c}": float((first == c).mean()) for c in "123"},
                    "P_greedy_complete": float(g[g["class"] == "complete"].greedy.mean()) if (g["class"] == "complete").any() else np.nan,
                    "seeds_same_first_all_sigma": f"{int((same == 1).sum())}/{len(same)}"})
        rows.append(row)
    return pd.DataFrame(rows)


def fit_summary(fits: pd.DataFrame) -> pd.DataFrame:
    """Mean +- SE over seeds of the per-seed exponents, vs the prediction K - 2."""
    def se(x):
        x = x.dropna()
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

    g = fits.groupby(["sweep", "tree", "K", "chi", "teacher", "init"])
    out = g.agg(n_seeds=("seed", "size"),
                p_t1=("p_t1", "mean"), p_t1_se=("p_t1", se),
                p_gap1=("p_gap1", "mean"), p_gap1_se=("p_gap1", se), n_gap1=("p_gap1", "count"),
                p_gap1_min=("p_gap1", "min"), p_gap1_max=("p_gap1", "max"),
                p_gap2=("p_gap2", "mean"), p_gap2_se=("p_gap2", se), n_gap2=("p_gap2", "count"),
                p_gap2_min=("p_gap2", "min"), p_gap2_max=("p_gap2", "max"),
                lam1_slope=("lam1_slope", "mean"), lam1_slope_se=("lam1_slope", se),
                lam2_slope=("lam2_slope", "mean"), lam2_slope_se=("lam2_slope", se),
                lam1_slope_raw=("lam1_slope_raw", "mean"), lam2_slope_raw=("lam2_slope_raw", "mean"),
                lam1_over_pred=("lam1_over_pred_median", "median"),
                lam2_over_pred=("lam2_over_pred_median", "median")).reset_index()
    out["predicted"] = out.K - 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["/root/outputs/s2s_core/main"])
    ap.add_argument("--out", required=True, help="output directory (never defaults to results/)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--hess", nargs="*", default=[],
                    help="recomputed Hessian tables (recompute_hessians.py) for THESE raw folders; "
                         "without them the lambda_min stored by the runs is used")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loaded = [load(Path(r)) for r in args.raw if Path(r).exists()]
    df = pd.concat([x[0] for x in loaded], ignore_index=True)
    plat = pd.concat([x[1] for x in loaded], ignore_index=True)
    plat["lam_min"], plat["lam_min_raw"] = plat.lam_min_stored, np.nan
    hfiles = [pd.read_csv(h) for h in args.hess if Path(h).exists()]
    if hfiles:
        hs = pd.concat(hfiles, ignore_index=True)
        keys = ["sweep", "tree", "chi", "teacher", "init", "sigma", "seed", "plateau"]
        plat = plat.drop(columns=["lam_min", "lam_min_raw"]).merge(
            hs[keys + ["lam_min_proj", "lam_min_raw"]].rename(columns={"lam_min_proj": "lam_min"}),
            on=keys, how="left")
        plat["lam_min"] = plat.lam_min.fillna(plat.lam_min_stored)
    # Back-compatible per-run columns: Hessian at the plateaus after 1 and 2 events.
    for i in (1, 2):
        pi = plat[plat.after == i][["sweep", "tree", "chi", "teacher", "init", "sigma", "seed",
                                     "lam_min", "lam_min_raw"]].drop_duplicates(
            ["sweep", "tree", "chi", "teacher", "init", "sigma", "seed"])
        pi = pi.rename(columns={"lam_min": f"plateau{i}_lam", "lam_min_raw": f"plateau{i}_lam_min_raw"})
        df = df.merge(pi, on=["sweep", "tree", "chi", "teacher", "init", "sigma", "seed"], how="left")
        df[f"plateau{i}_lam_min"] = df.pop(f"plateau{i}_lam")
    df.to_csv(out / f"{args.tag}_runs.csv", index=False)
    plat.to_csv(out / f"{args.tag}_plateaus.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / f"{args.tag}_summary.csv", index=False)
    class_table(df).to_csv(out / f"{args.tag}_classes.csv", index=False)
    fits = gap_fits(df, plat)
    fits.to_csv(out / f"{args.tag}_gap_fits.csv", index=False)
    fs = fit_summary(fits) if len(fits) else fits
    fs.to_csv(out / f"{args.tag}_exponents.csv", index=False)
    strict_saddles(plat).to_csv(out / f"{args.tag}_strict_saddles.csv", index=False)
    order_table(df).to_csv(out / f"{args.tag}_order.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(fs.round(3).to_string())


if __name__ == "__main__":
    main()
