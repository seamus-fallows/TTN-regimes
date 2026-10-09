"""Step-2 analysis: SGD event times vs batch size B and step size eta, compared with GF and theory.

For every (block, sigma, init, B, eta): mean and std of the event times over noise seeds, the shift
of the mean relative to gradient flow (fan-in init: the GF reference of gd_sweep.py for the same
init; aligned init: exact 1D integral), conservation-law drift and effective ranks at the end.

Theory (aligned balanced mode, notes/optim_samples.md section 4): with f(a) = a^{K-1} m(a),
m = s - a^K, per-sample factor variance V(a) = m^2 (3^N - 1) + R^2 (R^2 = sum of the other
components' squared residuals), the event a^K = s/2 occurs at
    E t^SGD - t^GF = (eta/2) int (f'/f) (1 + V/(B m^2)) da,   Var t^SGD = (eta/B) int V/(m^2 f) da.

Run: uv run python experiments/optim_samples/analyze_sgd.py
Writes results/sgd_runs.csv, sgd_summary.csv, sgd_fits.csv and prints the fits.
"""

import json

import numpy as np
import pandas as pd
from scipy.integrate import quad

from common import RAW
from common import RESULTS as OUT
WEIGHTS = [3.0, 2.0, 1.0]


def aligned_theory(r, a0, eta, batch, k=3, n=4):
    """GF event time, SGD mean shift and std for component r of the aligned balanced init."""
    s = WEIGHTS[r]
    others = sum(w**2 for i, w in enumerate(WEIGHTS) if i != r)  # unlearned: residual = s_i
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


def load_runs():
    rows = []
    for f in (RAW / "sgd").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            tk = d["task"]
            row = dict(block=tk["block"], init=tk["init"], sigma=tk["sigma"], seed=tk["seed"],
                       noise_seed=tk["noise_seed"], B=tk["B"], eta=tk["eta"],
                       diverged=d["diverged"], steps=d["steps"])
            for r in range(3):
                row[f"t{r + 1}"] = d["t50"][r] if d["t50"][r] is not None else np.nan
            if not d["diverged"]:
                row.update(loss_end=d["loss_end"], q_drift=d["q_drift"],
                           trq_change=float(np.mean(np.abs(d["trq_change"]))),
                           eff_ranks=str(d["eff_ranks"]))
            rows.append(row)
    return pd.DataFrame(rows)


def gf_reference():
    ref = {}
    for p in (RAW / "gd").glob("binary4_*_gf.json"):
        d = json.loads(p.read_text())
        tk = d["task"]
        if tk.get("root_scale"):
            continue
        ref[(tk["sigma"], tk["seed"])] = [np.nan if v is None else v for v in d["t50"]]
    return ref


def fit_power(x, y, yerr=None):
    """Least-squares fit log y = c + p log x; returns p and its standard error."""
    x, y = np.log(np.asarray(x)), np.log(np.asarray(y))
    a = np.vstack([np.ones_like(x), x]).T
    coef, res, *_ = np.linalg.lstsq(a, y, rcond=None)
    dof = max(1, len(x) - 2)
    s2 = np.sum((y - a @ coef) ** 2) / dof
    cov = s2 * np.linalg.inv(a.T @ a)
    return coef[1], np.sqrt(cov[1, 1])


def main():
    df = load_runs()
    ref = gf_reference()
    for r in range(3):
        df[f"t{r + 1}_gf"] = [
            ref.get((s, sd), [np.nan] * 3)[r] if i == "fanin" else
            aligned_theory(r, s, 1.0, 1.0)[0]
            for s, sd, i in zip(df.sigma, df.seed, df.init)]
    df.to_csv(OUT / "sgd_runs.csv", index=False)
    # Classify runs before pooling event-2 statistics: "joint" = component 2 learned together with
    # component 1 (a near-tied race flipped by the noise, no second plateau), "sequential" = a second
    # plateau as in GF, "unresolved" = no event 2 within the run.
    gap, gap_gf = df.t2 - df.t1, df.t2_gf - df.t1_gf
    df["run_class"] = np.where(df.t2.isna(), "unresolved",
                               np.where(gap < 0.5 * gap_gf, "joint", "sequential"))
    df.to_csv(OUT / "sgd_runs.csv", index=False)
    keys = ["block", "init", "sigma", "seed", "B", "eta"]
    seq = df[df.run_class == "sequential"]
    per_init = df.groupby(keys).agg(
        n_runs=("noise_seed", "size"), frac_div=("diverged", "mean"),
        frac_joint=("run_class", lambda c: np.mean(c == "joint")),
        frac_unresolved=("run_class", lambda c: np.mean(c == "unresolved")),
        t1_mean=("t1", "mean"), t1_std=("t1", "std"), t1_n=("t1", "count"), t1_gf=("t1_gf", "first"),
        t2_gf=("t2_gf", "first"), trq_change=("trq_change", "mean"), q_drift=("q_drift", "mean"))
    per_init = per_init.join(seq.groupby(keys).agg(t2_mean=("t2", "mean"), t2_std=("t2", "std"),
                                                    t2_n=("t2", "count"))).reset_index()
    for r in (1, 2):
        per_init[f"shift{r}"] = per_init[f"t{r}_mean"] - per_init[f"t{r}_gf"]
        per_init[f"shift{r}_se"] = per_init[f"t{r}_std"] / np.sqrt(per_init[f"t{r}_n"])
        per_init[f"cv{r}"] = per_init[f"t{r}_std"] / per_init[f"t{r}_gf"]
    # Theory for the aligned block (and the 1D small-a std estimate for fan-in inits).
    th = []
    for _, row in per_init.iterrows():
        out = {}
        for r in (1, 2):
            if row.init == "aligned":
                t_gf, gd, noise, std = aligned_theory(r - 1, row.sigma, row.eta, row.B)
                out[f"pred_shift{r}"] = gd + noise
                out[f"pred_std{r}"] = std
            else:
                s = WEIGHTS[r - 1]
                v = s**2 * (3**4 - 1) + sum(w**2 for w in WEIGHTS if w != s)
                out[f"pred_std{r}"] = np.sqrt(row[f"t{r}_gf"] * row.eta * v / row.B) / s
        th.append(out)
    per_init = pd.concat([per_init, pd.DataFrame(th)], axis=1)
    per_init.to_csv(OUT / "sgd_summary.csv", index=False)
    # Scaling fits per init seed (the same init direction is reused at every eta/B), then
    # mean +- s.e. over seeds. Sweeps: B at eta = 5e-4; eta at B = 4 and at B = 64 (each including
    # the B-sweep point at eta = 5e-4).
    sweeps = {"B": per_init[(per_init.block == "B_sweep")],
              "eta|B=4": per_init[((per_init.block == "eta_sweep_B4") | (per_init.block == "B_sweep"))
                                  & (per_init.B == 4)],
              "eta|B=64": per_init[((per_init.block == "eta_sweep_B64") | (per_init.block == "B_sweep"))
                                   & (per_init.B == 64)]}
    fits = []
    for name, g0 in sweeps.items():
        for (init, sigma), g in g0.groupby(["init", "sigma"]):
            for quantity, col, pred, min_n in (("std(t1)", "t1_std", 0.5, 5),
                                               ("std(t2), sequential runs", "t2_std", 0.5, 5),
                                               ("|tr Q_e change|", "trq_change", 1.0, 1)):
                exps = []
                for seed, gs in g.groupby("seed"):
                    nn = gs["t1_n"] if col != "t2_std" else gs["t2_n"]
                    gs = gs[np.isfinite(gs[col]) & (gs[col] > 0) & (nn >= min_n)]
                    xs = gs.eta / gs.B
                    if xs.nunique() >= 5 and xs.max() / xs.min() >= 10:
                        exps.append(fit_power(xs, gs[col])[0])
                if exps:
                    fits.append(dict(sweep=name, init=init, sigma=sigma, quantity=quantity,
                                     exponent_mean=np.mean(exps),
                                     exponent_se=np.std(exps, ddof=1) / np.sqrt(len(exps))
                                     if len(exps) > 1 else np.nan,
                                     n_seeds=len(exps), predicted=pred))
    fits = pd.DataFrame(fits)
    # Second plateau (P9 of notes/theory_optimizer.md): T = t2 - t1 per run vs GF, excluding
    # near-tie inits (GF: the eventual second component peaks at 0.15-0.6 during the first event).
    ties = pd.read_csv(OUT / "order_flips.csv").groupby(["sigma", "seed"]).rival_peak.first()
    r = df[(df.init == "fanin") & ~df.diverged & (df.run_class != "joint")].copy()
    r["near_tie"] = [0.15 <= ties.get((s, sd), 0.0) < 0.6 for s, sd in zip(r.sigma, r.seed)]
    r["T"] = r.t2 - r.t1
    r["T_gf"] = r.t2_gf - r.t1_gf
    r["T_ratio"] = r["T"] / r["T_gf"]
    sec = (r.groupby(["near_tie", "block", "sigma", "B", "eta"])
           .agg(n=("T_ratio", "count"), T_gf=("T_gf", "mean"), T_ratio=("T_ratio", "mean"),
                T_ratio_se=("T_ratio", "sem"), unresolved=("t2", lambda v: int(v.isna().sum())))
           .reset_index())
    x = 35.0 * sec.eta / sec.B * sec.T_gf
    sec["pred_P9_c35"] = np.where(x < 1, -np.log1p(-np.minimum(x, 0.999)) / x, np.inf)
    sec.to_csv(OUT / "sgd_second_plateau.csv", index=False)
    fits.to_csv(OUT / "sgd_fits.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        cols = ["block", "init", "sigma", "B", "eta", "n_runs", "frac_div", "frac_joint", "t1_gf", "shift1",
                "shift1_se", "t1_std", "pred_std1", "t2_gf", "shift2", "shift2_se", "t2_std",
                "pred_std2", "trq_change"]
        cols += [c for c in ["pred_shift1", "pred_shift2"] if c in per_init]
        summary = per_init.groupby(["block", "init", "sigma", "B", "eta"])[cols[5:]].mean()
        print(summary.to_string(float_format=lambda v: f"{v:.4g}"))
        print(fits.to_string(float_format=lambda v: f"{v:.3g}"))
        print(sec.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
