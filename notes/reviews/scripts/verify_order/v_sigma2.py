"""Verifier: m = 2 wrong-order law, own binomial-ML fits from the raw first-event sweeps."""

import json
import numpy as np
from scipy.optimize import minimize

RAW = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/order"
import sys
SETS_ALL = [
    ("b4_sigma", 2, 1),
    ("b4_sigma", 4, 1),
    ("pat_mps6_m2end", 4, 1),
    ("pat_mps4_sep", 4, 1),
    ("pat_mps6_m2adj", 4, 4),
    ("pat_mps4_adj", 4, 4),
]


def load(name):
    out = {}
    for line in open(f"{RAW}/{name}/runs.jsonl"):
        r = json.loads(line)
        for run in r["runs"]:
            out.setdefault((r["chi"], run["rho"]), {}).setdefault(run["sigma"], {})[r["seed"]] = (
                int(run["winner"] != 0)
            )
    return out


def nll(par, ls, k, n, quad=False):
    lp = par[0] + par[1] * ls + (par[2] * ls**2 if quad else 0)
    p = np.clip(np.exp(lp), 1e-15, 1 - 1e-12)
    return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))


def fit(sig, k, n, quad=False):
    ls = np.log(sig)
    x0 = list(np.polyfit(ls, np.log(np.maximum(k, 0.5) / n), 1)[::-1]) + ([0.0] if quad else [])
    best = None
    for start in (x0, [x0[0] + 0.3, x0[1]] + ([0.0] if quad else [])):
        r = minimize(
            nll,
            start,
            args=(ls, k, n, quad),
            method="Nelder-Mead",
            options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000),
        )
        if best is None or r.fun < best.fun:
            best = r
    return best.x


rng = np.random.default_rng(12345)
for name, chi, n_mult in SETS:
    data = load(name)
    for (c, rho), bysig in sorted(data.items()):
        if (ONLY_RHO and rho not in ONLY_RHO) or c != chi or rho in (0.5,) or (name == "b4_sigma" and rho not in (0.7, 0.8, 0.9, 0.95)):
            continue
        sig = np.array(sorted(bysig, reverse=True))
        seeds = sorted(set.intersection(*[set(bysig[s]) for s in sig]))
        W = np.array([[bysig[s][sd] for s in sig] for sd in seeds])  # seeds x sigma
        k, n = W.sum(0), np.full(len(sig), len(seeds))
        g = fit(sig, k, n)[1]
        boots, bq, bsmall = [], [], []
        for _ in range(150):
            Wb = W[rng.integers(0, len(W), len(W))]
            kb = Wb.sum(0)
            boots.append(fit(sig, kb, n)[1])
            bq.append(fit(sig, kb, n, quad=True)[2])
        q = fit(sig, k, n, quad=True)
        # curvature in units of the exponent change per decade: d(gamma)/d(log10 sigma) = 2 c ln 10
        pred = n_mult * (1 - rho) / rho
        nested = all(np.all(W[W[:, j + 1] == 1, j] == 1) for j in range(len(sig) - 1))
        p = k / n
        loc = [
            np.log(p[j] / p[j + 1]) / np.log(sig[j] / sig[j + 1])
            if p[j] > 0 and p[j + 1] > 0
            else np.nan
            for j in range(len(sig) - 1)
        ]
        # fit over the three smallest sigma with P > 0 (asymptotic end)
        pos = np.nonzero(k > 0)[0]
        tail = pos[-3:] if len(pos) >= 3 else pos
        g_tail = fit(sig[tail], k[tail], n[tail])[1] if len(tail) >= 2 else np.nan
        bt = []
        for _ in range(150):
            Wb = W[rng.integers(0, len(W), len(W))]
            kb = Wb.sum(0)
            if np.all(kb[tail] > 0):
                bt.append(fit(sig[tail], kb[tail], n[tail])[1])
        print(
            f"{name} chi{chi} rho{rho}: seeds {len(seeds)} sigma {sig.min():g}..{sig.max():g} ({len(sig)} pts)"
            f" P={np.round(p, 3).tolist()} | gamma {g:.3f}+-{np.std(boots, ddof=1):.3f} vs n(1-rho)/rho={pred:.3f}"
            f" ({(g - pred) / np.std(boots, ddof=1):+.1f} SE); curvature c={q[2]:+.4f}+-{np.std(bq, ddof=1):.4f}"
            f" (d gamma/d decade {2 * q[2] * np.log(10):+.3f}); local {np.round(loc, 3).tolist()};"
            f" tail({len(tail)} smallest sigma with P>0) {g_tail:.3f}+-{np.std(bt, ddof=1) if len(bt) > 2 else np.nan:.3f}; nested {nested}"
        )
