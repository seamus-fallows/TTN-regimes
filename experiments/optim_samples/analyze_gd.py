"""Summarize the step-1 sweep (gd_sweep.py): one row per GD run, joined with its GF reference.

Per run: relative event-time errors (t_r^GD - t_r^GF)/t_r^GF, whether the learning order matches GF,
the outcome class, final sharpness / lambda_bal, conservation drift, tail loss, and the backward-error
prediction of the event shift (from predict_shift.py, if present).

Outcome classes (checked in this order):
  diverged     loss non-finite or > 1e6 L(0)
  reentry      after the first event the loss returns above 0.9 L(0) (kicked back to the origin)
  oscillating  over the last 20% of steps (<= 2000) the total variation of the loss exceeds 10 x its net change
               (+1e-9 L(0)), or the final point is linearly unstable (sharpness * eta / 2 > 1.001)
  gf_like      otherwise (monotone; compared with GF through the event times and the final loss)

Run: uv run python experiments/optim_samples/analyze_gd.py  -> results/gd_sweep.csv, gd_summary.csv
"""

import json
import math

import numpy as np
import pandas as pd

from common import RAW as RAW_ROOT
from common import RESULTS as OUT

RAW = RAW_ROOT / "gd"


def load(path):
    return json.loads(path.read_text())


def tail_class(npz_path, loss0):
    z = np.load(npz_path)
    t, loss = z["t"], z["loss"]
    # The npz keeps the last 2000 steps at full resolution; use the uniform-spacing tail.
    dt = np.diff(t)
    step = np.min(dt[dt > 0]) if len(dt) else 1.0
    n_steps = int(round(t[-1] / step))
    n_tail = max(20, min(2000, n_steps // 5))  # last 20% of the run, at most 2000 steps
    tail_idx = np.nonzero(t >= t[-1] - n_tail * step * 1.0001)[0]
    tail = loss[tail_idx]
    tv = np.sum(np.abs(np.diff(tail)))
    net = abs(tail[-1] - tail[0])
    return tv > 10 * net + 1e-9 * loss0, float(tv), float(net)


EOS_RUNS = [("binary4_sig0.1_seed0", r) for r in (0.9, 1.2, 2.0)] + [
    ("binary4_sig0.1_seed0_root4.0", 0.9), ("binary8_sig0.9_seed0", 0.95)]


def export_eos():
    """Small copies of a few trajectories (for the EoS figure) in results/eos_example*.csv."""
    traj, probes = [], []
    for base, ratio in EOS_RUNS:
        path = RAW / f"{base}_r{ratio}.json"
        if not path.exists():
            continue
        d = load(path)
        z = np.load(path.with_suffix(".npz"))
        idx = np.unique(np.concatenate([np.linspace(0, len(z["t"]) - 1, 3000).astype(int),
                                        np.arange(max(0, len(z["t"]) - 400), len(z["t"]))]))
        for i in idx:
            traj.append(dict(run=base, ratio=ratio, t=z["t"][i], loss=z["loss"][i],
                             **{f"a{r + 1}": z["a"][i, r] for r in range(z["a"].shape[1])}))
        for pr in d["probe"]:
            probes.append(dict(run=base, ratio=ratio, t=pr["t"], loss=pr["loss"],
                               stab=pr["sharp"] * d["eta"] / 2, q_drift=pr["q_drift"],
                               trq=np.mean(np.abs(pr["q_traces"]))))
    pd.DataFrame(traj).to_csv(OUT / "eos_example_traj.csv", index=False)
    pd.DataFrame(probes).to_csv(OUT / "eos_example_probe.csv", index=False)


def main():
    gf = {}
    for p in RAW.glob("*_gf.json"):
        d = load(p)
        tk = d["task"]
        gf[(tk["tree"], tk["n"], tk["sigma"], tk["seed"], tk.get("root_scale"),
            bool(tk.get("centred", False)))] = d
    shifts = {}
    shift_file = OUT / "gd_shift_prediction.csv"
    if shift_file.exists():
        for _, r in pd.read_csv(shift_file).iterrows():
            rs = None if pd.isna(r.root_scale) else r.root_scale
            centred = bool(r.get("centred", False))
            shifts[(r.tree, r.n, r.sigma, r.seed, rs, centred, int(r.comp))] = r.pred_shift_over_eta
    rows = []
    for p in sorted(q for q in RAW.glob("*.json") if not q.name.endswith("_gf.json")):
        d = load(p)
        tk = d["task"]
        key = (tk["tree"], tk["n"], tk["sigma"], tk["seed"], tk.get("root_scale"),
               bool(tk.get("centred", False)))
        ref = gf.get(key)
        if ref is None:
            continue
        t_gd = np.array([np.nan if v is None else v for v in d["t50"]], float)
        t_gf = np.array([np.nan if v is None else v for v in ref["t50"]], float)
        order_gd = [int(i) for i in np.argsort(np.where(np.isnan(t_gd), np.inf, t_gd))
                    if np.isfinite(t_gd[i])]
        order_gf = [int(i) for i in np.argsort(np.where(np.isnan(t_gf), np.inf, t_gf))
                    if np.isfinite(t_gf[i])]
        if d["diverged"]:
            cls = "diverged"
        elif d["reentry"]:
            cls = "reentry"
        else:
            osc, tv, net = tail_class(p.with_suffix(".npz"), d["loss0"])
            # A converged GD run must sit at a linearly stable point: sharpness * eta / 2 <= 1.
            unstable = d.get("sharp_end", 0.0) * d["eta"] / 2 > 1.001
            cls = "oscillating" if (osc or unstable) else "gf_like"
        k = ref["K"]
        lam_bal = k * 3.0 ** (2 - 2 / k)
        row = dict(tree=tk["tree"], n=tk["n"], K=k, sigma=tk["sigma"], seed=tk["seed"],
                   root_scale=tk.get("root_scale"), centred=bool(tk.get("centred", False)),
                   ratio=tk["ratio"], eta=d["eta"], sharp_init=d["probe"][0]["sharp"],
                   cls=cls, loss_end=d["loss_end"], loss_gf_end=ref["loss_end"],
                   loss0=d["loss0"], loss_tail_mean=d["loss_tail_mean"],
                   same_order=order_gd[: len(order_gf)] == order_gf[: len(order_gd)],
                   n_events_gd=len(order_gd), n_events_gf=len(order_gf),
                   max_cross=max(d["n_cross"]),
                   sharp_end=d.get("sharp_end", math.nan),
                   sharp_end_rel=d.get("sharp_end", math.nan) / lam_bal,
                   stab_end=d.get("sharp_end", math.nan) * d["eta"] / 2,
                   sharp_gf_rel=ref["sharp_end"] / lam_bal,
                   q_drift_end=d.get("q_drift_end", math.nan),
                   trq_end=np.mean(np.abs(d["q_traces_end"])) if "q_traces_end" in d else math.nan,
                   trq_gf=np.mean(np.abs(ref["q_traces_end"])),
                   eff_ranks_end=str(d.get("eff_ranks_end")), eff_ranks_gf=str(ref["eff_ranks_end"]),
                   gf_q_drift=ref["q_drift"])
        for r in range(len(t_gf)):
            row[f"t{r + 1}_gf"] = t_gf[r]
            row[f"t{r + 1}_gd"] = t_gd[r]
            row[f"rel_err{r + 1}"] = (t_gd[r] - t_gf[r]) / t_gf[r]
            row[f"shift_over_eta{r + 1}"] = (t_gd[r] - t_gf[r]) / d["eta"]
            row[f"pred_shift_over_eta{r + 1}"] = shifts.get(key + (r,), math.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "gd_sweep.csv", index=False)
    g = df.groupby(["tree", "n", "centred", "sigma", "root_scale", "ratio"], dropna=False)
    summ = g.agg(
        n_runs=("seed", "size"),
        frac_gf_like=("cls", lambda c: np.mean(c == "gf_like")),
        frac_osc=("cls", lambda c: np.mean(c == "oscillating")),
        frac_reentry=("cls", lambda c: np.mean(c == "reentry")),
        frac_div=("cls", lambda c: np.mean(c == "diverged")),
        frac_same_order=("same_order", "mean"),
        rel_err1_mean=("rel_err1", "mean"), rel_err1_se=("rel_err1", "sem"),
        rel_err2_mean=("rel_err2", "mean"), rel_err2_se=("rel_err2", "sem"),
        sharp_end_rel=("sharp_end_rel", "mean"), sharp_end_rel_se=("sharp_end_rel", "sem"),
        sharp_gf_rel=("sharp_gf_rel", "mean"),
        trq_end=("trq_end", "mean"), trq_gf=("trq_gf", "mean"),
        q_drift_end=("q_drift_end", "mean"),
        loss_tail_rel=("loss_tail_mean", lambda s: np.nan),
    ).reset_index()
    tail_rel = g.apply(lambda x: np.mean(x.loss_tail_mean / x.loss0), include_groups=False)
    summ["loss_tail_rel"] = tail_rel.values
    summ.to_csv(OUT / "gd_summary.csv", index=False)
    export_eos()
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        cols = ["tree", "n", "centred", "sigma", "root_scale", "ratio", "frac_gf_like", "frac_osc",
                "frac_reentry", "frac_div", "frac_same_order", "rel_err1_mean", "rel_err2_mean",
                "sharp_end_rel", "sharp_gf_rel", "trq_end", "trq_gf", "loss_tail_rel"]
        print(summ[cols].to_string(float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
