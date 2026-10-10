"""Verifier: m = 2 law, fast binomial-ML fits (analytic gradient) for one set: argv = name [rho ...]."""

import json
import sys
import numpy as np
from scipy.optimize import minimize

RAW = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/order"
N_MULT = {"pat_mps6_m2adj": 4, "pat_mps4_adj": 4}


def fit(sig, k, n, quad=False):
    ls = np.log(sig)
    Z = np.column_stack([np.ones_like(ls), ls] + ([ls**2] if quad else []))

    def f(par):
        p = np.clip(np.exp(Z @ par), 1e-15, 1 - 1e-12)
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p)), -Z.T @ ((k - n * p) / (1 - p))

    x0 = np.array(
        list(np.polyfit(ls, np.log(np.maximum(k, 0.5) / n), 1)[::-1]) + ([0.0] if quad else [])
    )
    x0[0] = min(x0[0], -1e-3 - np.max(Z[:, 1:] @ x0[1:]))
    return minimize(f, x0, jac=True, method="BFGS", options=dict(gtol=1e-9, maxiter=2000)).x


name = sys.argv[1]
only = [float(v) for v in sys.argv[2:]]
data = {}
for line in open(f"{RAW}/{name}/runs.jsonl"):
    r = json.loads(line)
    for run in r["runs"]:
        data.setdefault(run["rho"], {}).setdefault(run["sigma"], {})[r["seed"]] = int(
            run["winner"] != 0
        )
rng = np.random.default_rng(12345)
for rho, bysig in sorted(data.items()):
    if only and rho not in only:
        continue
    sig = np.array(sorted(bysig, reverse=True))
    seeds = sorted(set.intersection(*[set(bysig[s]) for s in sig]))
    W = np.array([[bysig[s][sd] for s in sig] for sd in seeds])
    k, n = W.sum(0), np.full(len(sig), len(seeds))
    g = fit(sig, k, n)[1]
    q = fit(sig, k, n, quad=True)
    pos = np.nonzero(k > 0)[0]
    tail = pos[-3:]
    gt = fit(sig[tail], k[tail], n[tail])[1]
    bg, bq, bt = [], [], []
    for _ in range(300):
        kb = W[rng.integers(0, len(W), len(W))].sum(0)
        bg.append(fit(sig, kb, n)[1])
        bq.append(fit(sig, kb, n, quad=True)[2])
        if np.all(kb[tail] > 0):
            bt.append(fit(sig[tail], kb[tail], n[tail])[1])
    pred = N_MULT.get(name, 1) * (1 - rho) / rho
    p = k / n
    loc = [
        np.log(p[j] / p[j + 1]) / np.log(sig[j] / sig[j + 1])
        if p[j] > 0 and p[j + 1] > 0
        else np.nan
        for j in range(len(sig) - 1)
    ]
    nested = all(np.all(W[W[:, j + 1] == 1, j] == 1) for j in range(len(sig) - 1))
    print(
        f"{name} rho{rho}: seeds {len(seeds)} sigma {sig.min():g}..{sig.max():g} ({len(sig)} pts) k={k.tolist()}"
        f" | gamma {g:.3f}+-{np.std(bg, ddof=1):.3f} vs n(1-rho)/rho={pred:.3f} ((1-rho)/rho={(1 - rho) / rho:.3f})"
        f" | curvature c={q[2]:+.4f}+-{np.std(bq, ddof=1):.4f} | local {np.round(loc, 3).tolist()}"
        f" | tail sigma {sig[tail].tolist()}: {gt:.3f}+-{np.std(bt, ddof=1):.3f} | nested {nested}",
        flush=True,
    )
