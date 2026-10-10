"""Independent checks (own numpy implementation, indep_b4.py) of the regime-map claims for binary N = 4.
Usage: python indep_run.py OUT_DIR TASK [TASK ...]; tasks: lazy s2s rho floor large imb
Writes one JSON line per run to OUT_DIR/indep_<task>.jsonl and a summary JSON per task."""

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import indep_b4 as ib  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SEEDS = range(6)


def log(task, row):
    with open(os.path.join(OUT, f"indep_{task}.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def summary(task, obj):
    with open(os.path.join(OUT, f"indep_{task}_summary.json"), "w") as f:
        json.dump(obj, f, indent=1)
    print(task, json.dumps(obj)[:2000], flush=True)


def task_lazy():
    res = {}
    for chi in (16, 32, 64, 128):
        vals = []
        for s in SEEDS:

            def f(sig, chi=chi, s=s):
                r = ib.run(chi, sig, s)
                log("lazy", r)
                return r["dK"] if r["label"] != "stalled" else np.nan

            x, _ = ib.bisect_log(f, 0.25, 3.0, 0.1, n_iter=7)
            vals.append(x)
        vals = np.array(vals)
        res[chi] = dict(
            per_seed=vals.tolist(),
            median=float(np.nanmedian(vals)),
            median_x_sqrtchi=float(np.nanmedian(vals) * math.sqrt(chi)),
            theory_P5=6.50 / math.sqrt(chi),
        )
    summary("lazy", res)
    return res


def task_s2s():
    res = {}
    for chi in (16, 32, 64, 128):
        vals = []
        for s in SEEDS:

            def f(sig, chi=chi, s=s):
                r = ib.run(chi, sig, s)
                log("s2s", r)
                return r["min_rho"] if r["l_end"] < 1e-3 else np.nan

            x, _ = ib.bisect_log(f, 0.004, 0.4, 3.0, n_iter=7)
            vals.append(x)
        vals = np.array(vals)
        res[chi] = dict(
            per_seed=vals.tolist(),
            median=float(np.nanmedian(vals)),
            median_x_sqrtchi=float(np.nanmedian(vals) * math.sqrt(chi)),
            geo_mean_x_sqrtchi=float(np.exp(np.nanmean(np.log(vals))) * math.sqrt(chi)),
        )
    summary("s2s", res)
    return res


def task_rho():
    res = {}
    for chi in (9, 64):
        for s in (0, 1, 2):
            sig = np.geomspace(0.004, 0.02 if chi == 64 else 0.05, 6)
            rho = []
            for x in sig:
                r = ib.run(chi, float(x), s)
                log("rho", r)
                rho.append(r["min_rho"])
            rho = np.array(rho)
            ok = np.isfinite(rho) & (rho > 3)
            b = np.polyfit(np.log(sig[ok]), np.log(rho[ok]), 1)[0] if ok.sum() >= 3 else np.nan
            res[f"chi{chi}_seed{s}"] = dict(slope=float(b), rho=rho.tolist(), sigma=sig.tolist())
    summary("rho", res)
    return res


def task_floor():
    res = []
    for chi in (3, 4, 6):
        for s in (0, 1, 2):
            for sig in (1.003, 2.52):
                r = ib.run(chi, sig, s)
                log("floor", r)
                res.append(
                    dict(
                        chi=chi,
                        seed=s,
                        sigma=sig,
                        label=r["label"],
                        l_end=r["l_end"],
                        l_best=r["l_best"],
                        ratio=r["l_end"] / r["l_best"] if r["l_best"] > 0 else np.nan,
                        status=r["status"],
                    )
                )
    summary("floor", res)
    return res


def task_large():
    res = {}
    for lr in ("sp", "ntk", "mup"):
        rows = []
        for chi in (16, 32, 64, 128):
            for s in SEEDS:
                r = ib.run(chi, 1.5, s, lr=lr)
                log("large", r)
                rows.append((chi, r["dK"], r["dth"], r["rel_root"], r["rel_leaf"], r["l_end"]))
        a = np.array(rows)
        out = {}
        for j, q in enumerate(("dK", "dth", "root", "leaf"), start=1):
            x, y = np.log(a[:, 0]), np.log(a[:, j])
            X = np.stack([x, np.ones_like(x)], 1)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            rr = y - X @ coef
            se = math.sqrt((rr @ rr) / (len(x) - 2) * np.linalg.inv(X.T @ X)[0, 0])
            out[q] = (float(coef[0]), float(se))
        out["median_dK_by_chi"] = {
            int(c): float(np.median(a[a[:, 0] == c, 1])) for c in (16, 32, 64, 128)
        }
        out["max_l_end"] = float(a[:, 5].max())
        res[lr] = out
    summary("large", res)
    return res


def task_imb():
    res = {}
    alpha, chi = 3.375, 16
    for lam in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        sig = (alpha / lam) ** (1 / 3)
        dks, labs = [], []
        for s in SEEDS:
            r = ib.run(chi, sig, s, root_scale=lam)
            log("imb", r)
            dks.append(r["dK"])
            labs.append(r["label"])
        res[lam] = dict(median_dK=float(np.median(dks)), dK=dks, labels=labs)
    lams = sorted(res)
    med = [res[l]["median_dK"] for l in lams]
    res["local_slopes"] = [
        float(np.log(med[i + 1] / med[i]) / np.log(lams[i + 1] / lams[i])) for i in range(4)
    ]
    summary("imb", res)
    return res


def task_imbmap():
    """chi = 24 imbalance map: label counts over seeds at alpha x lambda/sqrt(chi) (claim: S2S widest near
    lambda = sqrt(chi); light root -> lazy at smaller alpha)."""
    chi = 24
    res = {}
    for lr_rel in (0.01, 0.1, 0.2, 1.0, 10.0, 100.0):
        lam = lr_rel * math.sqrt(chi)
        for alpha in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0):
            sig = (alpha / lam) ** (1 / 3)
            labs, rhos, dks = [], [], []
            for s in SEEDS:
                r = ib.run(chi, sig, s, root_scale=lam)
                log("imbmap", r)
                labs.append(r["label"]); rhos.append(r["min_rho"]); dks.append(r["dK"])
            res[f"lamrel{lr_rel}_alpha{alpha:g}"] = dict(labels={k: labs.count(k) for k in set(labs)},
                                                         med_rho=float(np.nanmedian(rhos)), med_dK=float(np.median(dks)))
    summary("imbmap", res)
    return res


if __name__ == "__main__":
    for task in sys.argv[2:]:
        t0 = time.time()
        globals()[f"task_{task}"]()
        print(f"== {task} done in {time.time() - t0:.0f}s", flush=True)
