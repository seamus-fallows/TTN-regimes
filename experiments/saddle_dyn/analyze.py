"""Analysis of the saddle_dyn runs against the registered rules (notes/saddle_dyn.md, sections 4-5).

Usage: uv run python experiments/saddle_dyn/analyze.py [--raw /root/outputs/saddle_dyn] [--out DIR]
Writes CSVs to experiments/saddle_dyn/results/ (or --out) and prints the decision tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from runs import CELLS  # noqa: E402
from sweep import HB_GRID, grid, window_exps  # noqa: E402

FLOOR = 0.01
TEST = ["PS08", "W6", "HT4W", "HT4L", "HT4S", "B8W", "B8W05"]
CONTROL = ["PS06", "C1", "C2", "C3"]
HB_WINDOW = {"PS08": (-3.5, -2.0), "PS06": (-3.5, -2.0), "B8W": (-2.5, -1.0)}


def k_star(cell):
    return {"PS08": 5, "PS06": 5, "W6": 6, "C1": 6, "C2": 5}.get(cell, 7)


def gam(cell):
    return CELLS[cell]["order"] - 2, k_star(cell) - 2


def load(raw, kind):
    rows = []
    for f in sorted((raw / kind).glob("*/*.json")):
        r = json.loads(f.read_text())
        stem = f.stem
        r["tag"] = stem.split("_", 2)[2] if stem.count("_") >= 2 else ""
        rows.append(r)
    return rows


def ha_table(recs):
    out = []
    for r in recs:
        a = r.get("amps_esc") or {}
        aft = r.get("after") or {}
        out.append(dict(
            cell=r["cell"], seed=r["seed"], eps=r["eps"], lx=round(math.log10(r["eps"]), 4) if r["eps"] != 0.15 else math.log10(0.15),
            tag=r["tag"], status=r["status"], escaped=bool(r.get("escaped")),
            t_esc=r.get("t_esc_0.001"), t_esc_1e4=r.get("t_esc_0.0001"), t_esc_1e2=r.get("t_esc_0.01"),
            esc_type=r.get("esc_type"), esc_ratio=r.get("esc_ratio"), floor_phi=r.get("floor_phi"),
            flagged=(r.get("floor_phi") is not None and r["floor_phi"] > FLOOR),
            cons_drift=r.get("cons_drift"), q_after=(aft.get("amps") or {}).get("q"), q_forgotten=aft.get("q_forgotten"),
            L_after=aft.get("L"), W_q_esc=a.get("q"), nfev=r.get("nfev"), secs=r.get("secs"),
            t_last=r.get("t_last"), dL_last=r.get("dL_last"),
            delta_r_last=json.dumps(r.get("delta_r_last")) if r.get("delta_r_last") is not None else None,
        ))
    return pd.DataFrame(out)


def ols_gamma(le, lt):
    """t ~ eps^-gamma: slope of log t on log eps is -gamma."""
    b, c = np.polyfit(le * math.log(10), lt, 1)
    return -b, c


def fixed_rss(le, lt, g):
    x = le * math.log(10)
    c = np.mean(lt + g * x)
    return float(np.sum((lt - c + g * x) ** 2))


def two_mech(le, lt, go, gr):
    """log t = -log(eps^go / A + eps^gr / B); returns log10 eps_c (rates equal), rss."""
    x = le * math.log(10)

    def res(p):
        a, b = p
        return lt + np.logaddexp(go * x - a, gr * x - b)

    p0 = [np.mean(lt + go * x), np.mean(lt + gr * x)]
    sol = least_squares(res, p0)
    a, b = sol.x
    return (b - a) / (gr - go) / math.log(10), float(np.sum(sol.fun**2))


def seed_fits(df, cell, x_col, t_col, win, ok_col):
    """Per-seed OLS in the window; a seed enters only if every window point is ok."""
    rows = []
    for seed, g in df.groupby("seed"):
        g = g.set_index(x_col)
        pts = [e for e in win]
        have = [e for e in pts if e in g.index]
        complete = len(have) == len(pts) and all(bool(g.loc[e, ok_col]) for e in pts)
        row = dict(cell=cell, seed=seed, complete=complete, n_points=len(have))
        if complete:
            le = np.array(pts)
            lt = np.log(np.array([g.loc[e, t_col] for e in pts], dtype=float))
            row["gamma"], row["intercept"] = ols_gamma(le, lt)
            go, gr = gam(cell)
            row["rss_free"] = float(np.sum((lt - (row["intercept"] - row["gamma"] * le * math.log(10))) ** 2))
            for gname, gv in [("ord", go), ("route", gr), ("ord_minus1", go - 1), ("four", 4.0)]:
                row[f"rss_{gname}"] = fixed_rss(le, lt, gv)
        rows.append(row)
    return pd.DataFrame(rows)


def local_exponents(df, cell, x_col, t_col, ok_col):
    rows = []
    for seed, g in df.groupby("seed"):
        g = g[g[ok_col]].sort_values(x_col, ascending=False)
        xs, ts = g[x_col].to_numpy(), g[t_col].to_numpy(dtype=float)
        for i in range(len(xs) - 1):
            rows.append(dict(cell=cell, seed=seed, x_hi=xs[i], x_lo=xs[i + 1],
                             gamma_loc=-(math.log(ts[i + 1]) - math.log(ts[i])) / ((xs[i + 1] - xs[i]) * math.log(10))))
    return pd.DataFrame(rows, columns=["cell", "seed", "x_hi", "x_lo", "gamma_loc"])


def decide(cell, fits, kind="test"):
    go, gr = gam(cell)
    f = fits[fits.complete]
    n = len(f)
    out = dict(cell=cell, n_fitted=n)
    if n == 0:
        out["decision"] = "INCONCLUSIVE (no complete seed)"
        return out
    g = f.gamma.to_numpy()
    mean, se = float(g.mean()), float(g.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    pred = go if kind == "test" else gr
    out.update(gamma_mean=mean, gamma_se=se, gamma_median=float(np.median(g)), gamma_min=float(g.min()),
               gamma_max=float(g.max()), pred=pred, frac_within_0p3=float(np.mean(np.abs(g - pred) <= 0.3)),
               rss_free=float(f.rss_free.sum()), rss_ord=float(f.rss_ord.sum()), rss_route=float(f.rss_route.sum()),
               rss_ord_minus1=float(f.rss_ord_minus1.sum()), rss_four=float(f.rss_four.sum()))
    a = abs(mean - pred) <= max(0.15, 2 * se if np.isfinite(se) else 0.15)
    b = out["frac_within_0p3"] >= 0.8
    d = n >= 10
    if kind == "test":
        c = out["rss_ord"] < out["rss_route"] / 10
        alt_ok = c
    else:
        alts = [out["rss_ord_minus1"]] + ([out["rss_four"]] if cell == "PS06" else [])
        alt_ok = all(out["rss_route"] < x / 10 for x in alts)
    out.update(rule_a=a, rule_b=b, rule_c=alt_ok, rule_d=d)
    if a and b and alt_ok and d:
        out["decision"] = "SUPPORTED"
    elif abs(mean - pred) > 0.5 and np.isfinite(se) and se <= 0.15 and d:
        out["decision"] = "REFUTED"
    else:
        why = [k for k, v in (("mean", a), ("seeds within 0.3", b), ("RSS vs alternative", alt_ok), (">=10 seeds", d)) if not v]
        out["decision"] = "INCONCLUSIVE (fails: " + ", ".join(why) + ")"
    return out


def crossover(df, cell, x_col, t_col, ok_col, min_t=100.0):
    go, gr = gam(cell)
    if go == gr:
        return None
    rows = []
    for seed, g in df.groupby("seed"):
        g = g[g[ok_col] & (g[t_col] > min_t)]
        if len(g) < 4:
            continue
        le, lt = g[x_col].to_numpy(), np.log(g[t_col].to_numpy(dtype=float))
        lec, rss2 = two_mech(le, lt, go, gr)
        _, c = ols_gamma(le, lt)
        gfree, _ = ols_gamma(le, lt)
        rss1 = float(np.sum((lt - (c - gfree * le * math.log(10))) ** 2))
        rows.append(dict(cell=cell, seed=seed, log10_eps_c=lec, rss_two_mech=rss2, rss_power=rss1, n=len(g)))
    return pd.DataFrame(rows, columns=["cell", "seed", "log10_eps_c", "rss_two_mech", "rss_power", "n"])


def summarize_ha(raw, out):
    recs = load(raw, "HA")
    df = ha_table(recs)
    df.to_csv(out / "ha_runs.csv", index=False)
    main = df[df.tag == ""].copy()
    main["ok"] = main.escaped & ~main.flagged
    fits_all, loc_all, dec, cross_all, counts = [], [], [], [], []
    for cell in TEST + CONTROL:
        d = main[main.cell == cell]
        if d.empty:
            continue
        win = window_exps(cell)
        fits = seed_fits(d, cell, "lx", "t_esc", win, "ok")
        fits_all.append(fits)
        loc = local_exponents(d, cell, "lx", "t_esc", "ok")
        loc_all.append(loc)
        res = decide(cell, fits, "test" if cell in TEST else "control")
        # local exponents at the two smallest pairs inside the window
        lw = loc[(loc.x_lo >= min(win) - 1e-9) & (loc.x_hi <= max(win) + 1e-9)]
        pairs = sorted(lw.x_lo.unique())[:2]
        for j, p in enumerate(pairs):
            v = lw[lw.x_lo == p].gamma_loc
            res[f"loc{j}_x_lo"] = p
            res[f"loc{j}_mean"] = float(v.mean())
            res[f"loc{j}_se"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
        cr = crossover(d, cell, "lx", "t_esc", "ok") if cell in TEST else None
        if cr is not None and len(cr):
            cross_all.append(cr)
            res["log10_eps_c_median"] = float(cr.log10_eps_c.median())
            res["log10_eps_c_mean"] = float(cr.log10_eps_c.mean())
        wd = d[d.lx.between(min(win) - 1e-9, max(win) + 1e-9)]
        res.update(n_runs=len(d), n_window_runs=len(wd), n_escaped=int(d.escaped.sum()),
                   n_trapped=int((d.status == "trapped").sum()), n_wall=int((d.status == "wall").sum()),
                   n_flagged=int(d.flagged.sum()), n_window_flagged=int(wd.flagged.sum()),
                   n_split=int((wd.esc_type == "split").sum()), n_mixed=int((wd.esc_type == "mixed").sum()),
                   n_route=int((wd.esc_type == "route").sum()),
                   frac_q_forgotten=float(wd.q_forgotten.dropna().astype(bool).mean()) if wd.q_forgotten.notna().any() else float("nan"),
                   max_cons_drift=float(d.cons_drift.max()), max_phi=float(d.floor_phi.max()))
        res["trapped_seeds"] = ",".join(str(s) for s in sorted(d[d.status == "trapped"].seed.unique()))
        dec.append(res)
    fits_all = pd.concat(fits_all) if fits_all else pd.DataFrame()
    fits_all.to_csv(out / "ha_seed_fits.csv", index=False)
    pd.concat(loc_all).to_csv(out / "ha_local_exponents.csv", index=False) if loc_all else None
    if cross_all:
        pd.concat(cross_all).to_csv(out / "ha_crossover.csv", index=False)
    dec = pd.DataFrame(dec)
    dec.to_csv(out / "ha_decisions.csv", index=False)
    # checks: tolerance and plain-loss reruns against the main runs
    chk = []
    key = main.set_index(["cell", "seed", "lx"])
    for _, r in df[df.tag.isin(["tol", "plain"])].iterrows():
        k = (r.cell, r.seed, r.lx)
        if k in key.index and r.escaped and bool(key.loc[k, "escaped"]):
            t0 = float(key.loc[k, "t_esc"])
            chk.append(dict(cell=r.cell, seed=r.seed, lx=r.lx, tag=r.tag, t_main=t0, t_check=r.t_esc,
                            rel_diff=(r.t_esc - t0) / t0))
        else:
            chk.append(dict(cell=r.cell, seed=r.seed, lx=r.lx, tag=r.tag, t_main=None, t_check=r.t_esc, rel_diff=None,
                            note=f"main escaped={bool(key.loc[k, 'escaped']) if k in key.index else None}, check status={r.status}"))
    pd.DataFrame(chk).to_csv(out / "ha_checks.csv", index=False)
    return df, dec, fits_all, pd.DataFrame(chk)


def hb_table(recs):
    out = []
    for r in recs:
        aft = r.get("after") or {}
        out.append(dict(cell=r["cell"], seed=r["seed"], sigma=r["sigma"], ls=round(math.log10(r["sigma"]), 4),
                        status=r["status"], t_q=r.get("t_q"), t_v=r.get("t_v"), t_esc=r.get("t_esc_0.001"),
                        dt=r.get("dt"), esc_type=r.get("esc_type"), esc_ratio=r.get("esc_ratio"),
                        floor_phi=r.get("floor_phi"), flagged=(r.get("floor_phi") is not None and r["floor_phi"] > FLOOR),
                        q_at_v=(r.get("amps_v") or {}).get("q"), regauge_err=r.get("regauge_err"),
                        cons_drift=r.get("cons_drift"), secs=(r.get("phase1") or {}).get("secs", 0) + (r.get("phase2") or {}).get("secs", 0)))
    return pd.DataFrame(out)


def summarize_hb(raw, out):
    recs = load(raw, "HB")
    if not recs:
        return None, None
    df = hb_table(recs)
    df.to_csv(out / "hb_runs.csv", index=False)
    df["ok"] = (df.status == "escaped") & ~df.flagged & (df.dt.fillna(0) > 0)
    dec, fits_all, loc_all = [], [], []
    for cell in ["PS08", "PS06", "B8W"]:
        d = df[df.cell == cell]
        if d.empty:
            continue
        lo, hi = HB_WINDOW[cell]
        win = [e for e in grid(HB_GRID[cell]) if lo - 1e-9 <= e <= hi + 1e-9]
        fits = seed_fits(d, cell, "ls", "dt", win, "ok")
        fits_all.append(fits)
        loc = local_exponents(d, cell, "ls", "dt", "ok")
        loc_all.append(loc)
        res = decide(cell, fits, "control" if cell == "PS06" else "test")
        res["kind"] = "HB"
        st = d.status.value_counts().to_dict()
        res.update({f"n_{k}": v for k, v in st.items()})
        visit_seeds = d[d.status.isin(["escaped", "no_escape", "trapped", "wall"])].seed.unique()
        res["n_seeds"] = int(d.seed.nunique())
        res["n_seeds_visiting_any"] = int(len(visit_seeds))
        res["n_unresolved_dt_lt10"] = int(((d.status == "escaped") & (d.dt < 10)).sum())
        res["n_flagged"] = int(d.flagged.sum())
        wd = d[d.ls.between(lo - 1e-9, hi + 1e-9) & (d.status == "escaped")]
        res.update(n_split=int((wd.esc_type == "split").sum()), n_mixed=int((wd.esc_type == "mixed").sum()),
                   n_route=int((wd.esc_type == "route").sum()))
        lw = loc[(loc.x_lo >= lo - 1e-9) & (loc.x_hi <= hi + 1e-9)]
        for j, p in enumerate(sorted(lw.x_lo.unique())[:2]):
            v = lw[lw.x_lo == p].gamma_loc
            res[f"loc{j}_x_lo"], res[f"loc{j}_mean"] = p, float(v.mean())
            res[f"loc{j}_se"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
        if cell == "B8W":  # route law above the crossover: pairs with both sigma >= 0.1
            v = loc[(loc.x_lo >= -1.0 - 1e-9)].gamma_loc
            res["above_mean"] = float(v.mean()) if len(v) else float("nan")
            res["above_n"] = int(len(v))
        dec.append(res)
    pd.concat(fits_all).to_csv(out / "hb_seed_fits.csv", index=False)
    pd.concat(loc_all).to_csv(out / "hb_local_exponents.csv", index=False)
    dec = pd.DataFrame(dec)
    dec.to_csv(out / "hb_decisions.csv", index=False)
    return df, dec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/root/outputs/saddle_dyn")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    df, dec, fits, chk = summarize_ha(raw, out)
    cols = ["cell", "decision", "n_fitted", "gamma_mean", "gamma_se", "pred", "frac_within_0p3", "rss_ord", "rss_route",
            "loc0_mean", "loc1_mean", "log10_eps_c_median", "n_runs", "n_escaped", "n_trapped", "n_wall", "n_flagged",
            "n_split", "n_mixed", "n_route", "frac_q_forgotten"]
    print(dec[[c for c in cols if c in dec]].to_string(index=False))
    if len(chk):
        print("\nchecks:\n", chk.groupby("tag").rel_diff.describe().to_string())
    hb, hdec = summarize_hb(raw, out)
    if hdec is not None:
        print("\nH-B:\n", hdec.to_string(index=False))
