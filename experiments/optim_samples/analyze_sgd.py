"""Step-2 analysis: SGD event times vs batch size B and step size eta, compared with GF and theory.

Events are persistent (last upward crossing of a_r = 1/2; runs continue past the second GF event).
For each init the components are ordered by their GF events (c1 = first, c2 = second); "event 1"
and "event 2" of an SGD run are the persistent events of c1 and c2, and a run whose persistent order
differs from GF's is classified "flipped" (reported, kept out of the plateau statistics).

Per (block, sigma, init, B, eta): mean and std over noise seeds of the events, the shift of the mean
relative to GF (fan-in init: the GF reference of gd_sweep.py; aligned init: exact 1D integral), the
conservation drift. Pooled over inits: mean of the per-init values with the standard error ACROSS
inits (fix round 1: the round-1 report quoted a single init's s.e.).

Theory (aligned balanced mode, notes/optim_samples.md section 4): with f(a) = a^{K-1} m(a),
m = s - a^K, per-sample factor variance V(a) = m^2 (3^N - 1) + R^2 (R^2 = squared residuals of the
components still unlearned when component r escapes), the event a^K = s/2 occurs at
    E t^SGD - t^GF = (eta/2) int (f'/f) (1 + V/(B m^2)) da,   Var t^SGD = (eta/B) int V/(m^2 f) da.

Run: uv run python experiments/optim_samples/analyze_sgd.py [--out-dir DIR]
Writes sgd_runs.csv, sgd_summary.csv (per init), sgd_pooled.csv (over inits), sgd_fits.csv,
sgd_second_plateau.csv.
"""

import json

import numpy as np
import pandas as pd
from scipy.integrate import quad

from common import RAW
from common import RESULTS as OUT

WEIGHTS = [3.0, 2.0, 1.0]


def aligned_theory(r, a0, eta, batch, k=3, n=4):
    """GF event time, SGD mean shift and std for component r of the aligned balanced init.
    Components with larger weights are learned earlier (residual ~0); smaller ones are unlearned."""
    s = WEIGHTS[r]
    others = sum(w**2 for i, w in enumerate(WEIGHTS) if i > r)
    a_e = (s / 2) ** (1 / k)

    def f(a):
        return a ** (k - 1) * (s - a**k)

    def fprime(a):
        return (k - 1) * a ** (k - 2) * s - (2 * k - 1) * a ** (2 * k - 2)

    def v(a):
        m = s - a**k
        return m**2 * (3**n - 1) + others

    t_gf = quad(lambda a: 1 / f(a), a0, a_e, limit=200)[0]
    gd = 0.5 * quad(lambda a: fprime(a) / f(a), a0, a_e, limit=200)[0]
    noise = 0.5 * quad(lambda a: fprime(a) / f(a) * v(a) / (s - a**k) ** 2, a0, a_e,
                       limit=200)[0] / batch
    var = quad(lambda a: v(a) / ((s - a**k) ** 2 * f(a)), a0, a_e, limit=200)[0] / batch
    return t_gf, eta * gd, eta * noise, np.sqrt(eta * var)


def gf_reference():
    """Per (sigma, seed): GF persistent events, final-rise widths, the GF order of the first two
    events and their separation ratio rho = gap / max(width_1, width_2) (notes/diagnostics.md)."""
    ref = {}
    for p in (RAW / "gd").glob("binary4_sig*_gf.json"):
        d = json.loads(p.read_text())
        tk = d["task"]
        if tk.get("root_scale") or tk.get("centred") or tk.get("chi", 3) != 3:
            continue
        nan = float("nan")
        t50 = np.array([nan if v is None else v for v in d["t50"]], float)
        t10 = np.array([nan if v is None else v for v in d["t10"]], float)
        t90 = np.array([nan if v is None else v for v in d["t90"]], float)
        order = [int(i) for i in np.argsort(np.where(np.isfinite(t50), t50, np.inf))]
        c1, c2 = order[0], order[1]
        width = t90 - t10
        rho = (t50[c2] - t50[c1]) / np.nanmax(width[[c1, c2]])
        ref[(tk["sigma"], tk["seed"])] = dict(t50=t50, c1=c1, c2=c2, rho=rho)
    return ref


def load_runs(ref):
    rows = []
    for f in (RAW / "sgd").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            tk = d["task"]
            t50 = np.array([np.nan if v is None else v for v in d["t50"]], float)
            if tk["init"] == "aligned":
                c1, c2 = 0, 1
                gf_t = [aligned_theory(r, tk["sigma"], 1.0, 1.0)[0] for r in range(3)]
                rho = np.inf
            else:
                g = ref.get((tk["sigma"], tk["seed"]))
                if g is None:
                    continue
                c1, c2, gf_t, rho = g["c1"], g["c2"], g["t50"], g["rho"]
            row = dict(block=tk["block"], init=tk["init"], sigma=tk["sigma"], seed=tk["seed"],
                       noise_seed=tk["noise_seed"], B=tk["B"], eta=tk["eta"], diverged=d["diverged"],
                       t1=t50[c1], t2=t50[c2], t1_gf=gf_t[c1], t2_gf=gf_t[c2], rho_gf=rho,
                       t1_first=d["t50_first"][c1] if d["t50_first"][c1] is not None else np.nan,
                       n_cross=str(d["n_cross"]), t_end=d["steps"] * tk["eta"])
            if np.isfinite(row["t1"]) and np.isfinite(row["t2"]):
                row["run_class"] = "flipped" if row["t2"] < row["t1"] else "same_order"
            else:
                row["run_class"] = "unresolved"
            if not d["diverged"]:
                row.update(loss_end=d["loss_end"], q_drift=d["q_drift"],
                           trq_change=float(np.mean(np.abs(d["trq_change"]))),
                           a1_min_after=d["a_min_after_learned"][c1])
            rows.append(row)
    return pd.DataFrame(rows)


def fit_power(x, y):
    """Least-squares fit log y = c + p log x; returns p and its standard error."""
    x, y = np.log(np.asarray(x)), np.log(np.asarray(y))
    a = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    dof = max(1, len(x) - 2)
    s2 = np.sum((y - a @ coef) ** 2) / dof
    return coef[1], np.sqrt(s2 * np.linalg.inv(a.T @ a)[1, 1])


def p7_estimate(row, gd_coef):
    """P7: (eta/2)(K-1)(1 + kappa/B) ln(a_e/a0) ~ (1 + kappa/B) x (GD's own shift eta c_1)."""
    kappa = 3**4 - 1 + sum(w**2 for w in WEIGHTS[1:]) / WEIGHTS[0] ** 2
    return row.eta * gd_coef * (1 + kappa / row.B)


def main():
    ref = gf_reference()
    df = load_runs(ref)
    df.to_csv(OUT / "sgd_runs.csv", index=False)
    keys = ["block", "init", "sigma", "seed", "B", "eta"]
    ok = df[df.run_class != "unresolved"]
    per_init = df.groupby(keys).agg(
        n_runs=("noise_seed", "size"), frac_div=("diverged", "mean"),
        frac_flipped=("run_class", lambda c: np.mean(c == "flipped")),
        frac_unresolved=("run_class", lambda c: np.mean(c == "unresolved")),
        t1_gf=("t1_gf", "first"), t2_gf=("t2_gf", "first"), rho_gf=("rho_gf", "first"),
        trq_change=("trq_change", "mean"), q_drift=("q_drift", "mean"),
        a1_min_after=("a1_min_after", "min"))
    stats = ok.groupby(keys).agg(t1_mean=("t1", "mean"), t1_std=("t1", "std"), t1_n=("t1", "count"))
    seq = ok[ok.run_class == "same_order"].assign(T=lambda d: d.t2 - d.t1)
    stats2 = seq.groupby(keys).agg(t2_mean=("t2", "mean"), t2_std=("t2", "std"),
                                   t2_n=("t2", "count"), T_mean=("T", "mean"), T_std=("T", "std"))
    per_init = per_init.join(stats).join(stats2).reset_index()
    per_init["shift1"] = per_init.t1_mean - per_init.t1_gf
    per_init["shift1_se_noise"] = per_init.t1_std / np.sqrt(per_init.t1_n)
    per_init["shift2"] = per_init.t2_mean - per_init.t2_gf
    per_init["T_gf"] = per_init.t2_gf - per_init.t1_gf
    per_init["T_ratio"] = per_init.T_mean / per_init.T_gf
    # Theory: aligned exact 1D formulas; fan-in: 1D small-a spread and P7's delay estimated with
    # GD's own first-order coefficient c_1 of that init (gd_shift_prediction.csv).
    shifts = pd.read_csv(OUT / "gd_shift_prediction.csv")
    shifts = shifts[shifts.tree.eq("binary") & shifts.n.eq(4) & shifts.root_scale.isna()]
    th = []
    for _, row in per_init.iterrows():
        out = {}
        if row.init == "aligned":
            for r in (1, 2):
                t_gf, gd, noise, std = aligned_theory(r - 1, row.sigma, row.eta, row.B)
                out[f"pred_shift{r}"], out[f"pred_std{r}"] = gd + noise, std
        else:
            c1 = ref[(row.sigma, row.seed)]["c1"]
            s = WEIGHTS[c1]
            v = s**2 * (3**4 - 1) + sum(w**2 for w in WEIGHTS if w != s)
            out["pred_std1"] = np.sqrt(row.t1_gf * row.eta * v / row.B) / s
            sel = shifts[(shifts.sigma == row.sigma) & (shifts.seed == row.seed) & (shifts.comp == c1)]
            if len(sel):
                out["pred_shift1"] = p7_estimate(row, sel.pred_shift_over_eta.iloc[0])
        th.append(out)
    per_init = pd.concat([per_init, pd.DataFrame(th)], axis=1)
    per_init["std1_over_pred"] = per_init.t1_std / per_init.pred_std1
    per_init.to_csv(OUT / "sgd_summary.csv", index=False)
    # Pooled over inits: mean of per-init values, s.e. ACROSS inits.
    pooled = per_init.groupby(["block", "init", "sigma", "B", "eta"]).agg(
        n_inits=("seed", "nunique"), shift1=("shift1", "mean"), shift1_se=("shift1", "sem"),
        pred_shift1=("pred_shift1", "mean"), t1_gf=("t1_gf", "mean"),
        frac_flipped=("frac_flipped", "mean"), frac_unresolved=("frac_unresolved", "mean"),
        std1_over_pred_min=("std1_over_pred", "min"), std1_over_pred_max=("std1_over_pred", "max"),
        trq_change=("trq_change", "mean"), trq_change_se=("trq_change", "sem"),
        a1_min_after=("a1_min_after", "min")).reset_index()
    pooled.to_csv(OUT / "sgd_pooled.csv", index=False)
    # Scaling fits per init seed, then mean +- s.e. over seeds (analysis standards).
    sweeps = {"B": per_init[(per_init.block == "B_sweep")],
              "eta|B=4": per_init[((per_init.block == "eta_sweep_B4") | (per_init.block == "B_sweep"))
                                  & (per_init.B == 4)],
              "eta|B=64": per_init[((per_init.block == "eta_sweep_B64") | (per_init.block == "B_sweep"))
                                   & (per_init.B == 64)]}
    fits = []
    for name, g0 in sweeps.items():
        for (init, sigma), g in g0.groupby(["init", "sigma"]):
            for quantity, col, pred in (("std(t1)", "t1_std", 0.5),
                                        ("|tr Q_e change|", "trq_change", 1.0)):
                exps = []
                for _, gs in g.groupby("seed"):
                    gs = gs[np.isfinite(gs[col]) & (gs[col] > 0)]
                    xs = gs.eta / gs.B
                    if xs.nunique() >= 5 and xs.max() / xs.min() >= 10:
                        exps.append(fit_power(xs, gs[col])[0])
                if exps:
                    fits.append(dict(sweep=name, init=init, sigma=sigma, quantity=quantity,
                                     exponent_mean=np.mean(exps),
                                     exponent_se=np.std(exps, ddof=1) / np.sqrt(len(exps))
                                     if len(exps) > 1 else np.nan, n_seeds=len(exps),
                                     predicted=pred))
    fits = pd.DataFrame(fits)
    fits.to_csv(OUT / "sgd_fits.csv", index=False)
    # Second plateau (P9): inits with a resolved GF plateau (rho > 3) and no near-tie, same-order runs;
    # standard error across inits.
    ties = pd.read_csv(OUT / "order_flips.csv").groupby(["sigma", "seed"]).rival_peak.first()
    r = per_init[(per_init.init == "fanin") & (per_init.rho_gf > 3)].copy()
    r["near_tie"] = [0.15 <= ties.get((s, sd), 0.0) < 0.6 for s, sd in zip(r.sigma, r.seed)]
    sec = (r.groupby(["near_tie", "block", "sigma", "B", "eta"])
           .agg(n_inits=("seed", "nunique"), T_gf=("T_gf", "mean"), T_ratio=("T_ratio", "mean"),
                T_ratio_se_inits=("T_ratio", "sem"),
                T_ratios=("T_ratio", lambda v: " ".join(f"{x:.3f}" for x in v.values)))
           .reset_index())
    z = sec.eta / sec.B * sec.T_gf
    for c in (35.0, 45.0):
        x = c * z
        sec[f"pred_P9_c{int(c)}"] = np.where(x < 1, -np.log1p(-np.minimum(x, 0.999)) / x, np.inf)
    sec.to_csv(OUT / "sgd_second_plateau.csv", index=False)
    print("inits (seeds 0-4) without a resolved GF second plateau (rho <= 3):",
          sorted({k for k, g in ref.items() if g["rho"] <= 3 and k[1] <= 4}))
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(pooled.to_string(float_format=lambda v: f"{v:.4g}"))
        print(fits.to_string(float_format=lambda v: f"{v:.3g}"))
        print(sec[~sec.near_tie].to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
