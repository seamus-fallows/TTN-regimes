"""Round 2 (notes/regime_map.md section 4): per-seed S2S boundary sigma_c(K) in realizable problems.

Arm P: plain model, chi = 4 (= 2 x teacher rank), teacher weights (3, 2), binary_tree(N), N = 4..16,
d = 2 and 3. Arm C: centred model with full bonds (chi = cut dimension), same teacher, small N.

Each seed of each (arm, N, d) is a "walker" that adds sigma values until the first S2S -> rich
transition coming from small sigma is bracketed within a factor 2^(1/(K-2)) in sigma (a factor ~2 in
rho, since rho ~ sigma^-(K-2)). Runs are classified by `run_class`. The walkers run asynchronously in
one spawned process pool; rows go to <out>/<exp>/results.jsonl, and a restart rebuilds every walker
from the rows already there (run ids as in run_regime).

usage: uv run python experiments/regime_map/round2.py run [--arms P,C] [--n-proc 370] [--out DIR]
           [--stop-at HH:MM (UTC; no new runs after it)]
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import datetime as dt
import json
import math
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_regime as rr  # noqa: E402
from analysis import aligned_boundary_a0  # noqa: E402

OUT = "/root/outputs/regime_map_r2"
WEIGHTS = [3.0, 2.0]
SEEDS = range(6)
ARMS = {
    "P": dict(
        exp="r2_plain",
        base=dict(model="plain", kind="binary", chi=4, cap=True, weights=WEIGHTS, method="lsoda",
                  l_floor=1e-6, max_steps=30000, tensor_kernel=False),
        cells=[(n, d) for d in (2, 3) for n in range(4, 17)],
    ),
    "C": dict(
        exp="r2_centred_full",
        # LSODA: the dopri5 pilots (binary 8/10 at d = 2, 6/7 at d = 3; P = 830-3900) had not
        # finished after 20 min under full load; P <= 1152 keeps the exact Hessian affordable
        base=dict(model="centred", kind="binary", chi=10**6, cap=True, weights=WEIGHTS,
                  method="lsoda", direct_alpha=1e6, l_floor=1e-6, max_steps=30000),
        cells=[(n, 2) for n in range(4, 10)] + [(n, 3) for n in range(4, 6)],
    ),
}


def sigma_guess(K: int, d: int) -> float:
    """H_B prediction calibrated on theory R9 (bin8, d = 2, sigma_c = 0.71): sigma^K = 0.86 sqrt(D) u_0c."""
    u0c = aligned_boundary_a0(K, WEIGHTS) ** K
    return (0.86 * d ** ((K + 1) / 2) * u0c) ** (1 / K)


def step_factor(K: int) -> float:
    return min(4 ** (1 / max(K - 2, 1)), 2.5)


def run_class(r: dict, thr: float = 3.0, domain: bool = True):
    """('S', rho) S2S-like, ('R', rho) rich-like, ('D', nan) outside the event rule's domain, or
    (None, nan) no information on the separation.

    rho = the single separation ratio (R = 2). Domain: the final rise starts at a_r = 0.1, so a run
    whose initial overlap max_r |a_r(0)| >= 0.1 (plain model at large sigma, where W_0 itself
    projects on the components with amplitude ~ sigma^K / (sqrt(D) s_r)) has no defined events
    ('D'; checked only when `domain`, i.e. for the plain model). Without a ratio: a run that did not
    converge and in which exactly one component reached 0.9 is S2S-like if (t_end - t_1) / w_1 > thr
    (a lower bound on rho, returned as -value); a converged run without a ratio is rich-like (events
    merged); anything else carries no information."""
    if domain and "a0" in r and np.max(np.abs(r["a0"])) >= 0.1:
        return "D", np.nan
    rho = r["rhos"][0] if r.get("rhos") else np.nan
    if rho is not None and np.isfinite(rho):
        return ("S" if rho > thr else "R"), float(rho)
    w = np.asarray(r["event_widths"], float)
    t = np.asarray(r["event_times"], float)
    converged = r["l_end"] < 1e-3
    if not converged and np.isfinite(w).sum() == 1:
        i = int(np.flatnonzero(np.isfinite(w))[0])
        lb = (r["t_end"] - t[i]) / w[i] if np.isfinite(t[i]) and w[i] > 0 else np.nan
        if np.isfinite(lb) and lb > thr:
            return "S", -float(lb)
    if converged:
        return "R", np.nan
    return None, np.nan


def sigma_limits(K: int, d: int):
    """sigma_min: initial component amplitude sigma^K / sqrt(D) >= 1e-9 (float64 floor);
    sigma_max: sigma^K <= 1e6 and sigma <= 3."""
    smin = (1e-9 * d ** ((K + 1) / 2)) ** (1 / K)
    return smin, min(3.0, 1e6 ** (1 / K))


def walker_next(rows: list[dict], K: int, d: int, thr: float = 3.0, domain: bool = True):
    """(new sigmas, status) for one seed. status: 'ok' (bracket <= target), 'never_s2s' (no S2S-like
    run down to sigma_min), 'all_s2s' (S2S-like up to sigma_max), 'domain' (S2S-like up to the
    smallest sigma whose initial overlaps leave the event rule's domain, bracketed within the
    target), or 'open'."""
    smin, smax = sigma_limits(K, d)
    f = step_factor(K)
    target = min(2 ** (1 / max(K - 2, 1)), 1.5)
    pts = sorted((r["sigma"], *run_class(r, thr, domain)) for r in rows)
    dom = [p[0] for p in pts if p[1] == "D"]
    if dom:  # nothing above the smallest out-of-domain sigma is informative
        smax = min(smax, min(dom))
        pts = [p for p in pts if p[0] < smax]
        if not pts:
            return [smax / f, smax / f**2], "open"
    sig = np.array([p[0] for p in pts])
    cls = [p[1] for p in pts]
    rho = np.array([p[2] for p in pts])
    S = [i for i, c in enumerate(cls) if c == "S"]
    if not S:
        lo = sig[0]
        if lo <= smin * 1.0001:
            return [], "never_s2s"
        step = f
        R = [i for i, c in enumerate(cls) if c == "R" and np.isfinite(rho[i]) and rho[i] > 0]
        if R:  # rho ~ sigma^-(K-2): jump to where the lowest rich run would reach thr, and beyond
            step = max(step, min((thr / rho[R[0]]) ** (1 / max(K - 2, 1)) * 1.1, f**3))
        return [max(lo / step, smin), max(lo / (step * f), smin)], "open"
    first = S[0]
    after = [j for j in range(first + 1, len(pts)) if cls[j] == "R"]
    if not after:
        hi = sig[-1]
        if dom:
            if smax / hi <= target * 1.0001:
                return [], "domain"
            if hi < smax / f**2:  # far below the domain limit: keep stepping up
                pass
            else:
                ratio = smax / hi
                return [hi * ratio ** (1 / 3), hi * ratio ** (2 / 3)], "open"
        if hi >= smax * 0.9999:
            return [], "all_s2s"
        step = f
        last = S[-1]
        if rho[last] > 0 and np.isfinite(rho[last]):
            step = max(f**0.5, min((rho[last] / thr) ** (1 / max(K - 2, 1)) * 1.05, f**3))
        new = sorted({min(hi * step, smax), min(hi * step * f, smax)})
        if dom:
            new = [x for x in new if x < smax * 0.9999]
            if not new:  # the step would leave the domain: close in on the domain limit instead
                ratio = smax / hi
                new = [hi * ratio ** (1 / 3), hi * ratio ** (2 / 3)]
        return new, "open"
    j = after[0]
    i = max(k for k in S if k < j)
    lo, hi = sig[i], sig[j]
    if hi / lo <= target * 1.0001:
        return [], "ok"
    ratio = hi / lo
    return [lo * ratio ** (1 / 3), lo * ratio ** (2 / 3)], "open"


def make_cfg(arm: str, n: int, d: int, seed: int, sigma: float) -> dict:
    """Time budget: run_regime's default T = 50 t_esc + 1e4 / max(lam_bar, 1) is short at large
    sigma, where the plain model first shrinks W_0 and then escapes from a small residual scale
    (pilot: binary 16, d = 3, sigma = 2 stopped at T = 1.5e3 with nothing learned); here
    T = max(50 t_esc + 1e4, 1e6), and max_steps bounds the cost."""
    a = ARMS[arm]
    sigma = float(f"{sigma:.5g}")
    K = n - 1
    t_esc = (0.3 * sigma) ** (-(K - 2)) / ((K - 2) * min(WEIGHTS))
    t_max = float(f"{max(50 * t_esc + 1e4, 1e6):.4g}")
    return dict(a["base"], exp=a["exp"], n=n, d=d, seed=seed, sigma=sigma, t_max=t_max)


def load_rows(out: str, exp: str) -> list[dict]:
    path = os.path.join(out, exp, "results.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in map(json.loads, f) if "error" not in r]


def run(arms, n_proc: int, out: str, stop_at: float | None):
    walkers = {}  # (arm, n, d, seed) -> dict(rows, pending, status)
    for arm in arms:
        a = ARMS[arm]
        rows = load_rows(out, a["exp"])
        for n, d in a["cells"]:
            for seed in SEEDS:
                mine = [r for r in rows if r["n"] == n and r["d"] == d and r["seed"] == seed]
                walkers[(arm, n, d, seed)] = dict(rows=mine, pending=0, status="new")
    files = {}
    for arm in arms:
        os.makedirs(os.path.join(out, ARMS[arm]["exp"]), exist_ok=True)
        files[arm] = open(os.path.join(out, ARMS[arm]["exp"], "results.jsonl"), "a")
    futures = {}
    log = lambda *x: print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"), *x, flush=True)  # noqa: E731

    # fork on Linux: a spawned worker holds its own copy of torch (~440 MB RSS), so 330 spawned
    # workers exceed the 125 GB of this machine (the first launch died with BrokenProcessPool)
    ctx = get_context("fork" if sys.platform == "linux" else "spawn")
    with ProcessPoolExecutor(n_proc, mp_context=ctx, initializer=rr._init_worker,
                             initargs=(out,)) as pool:

        def advance(key):
            w = walkers[key]
            arm, n, d, seed = key
            K = n - 1
            if w["pending"] or w["status"] in ("ok", "never_s2s", "all_s2s", "domain"):
                return
            if stop_at is not None and time.time() > stop_at:
                w["status"] = "stopped"
                return
            if not w["rows"]:
                g, f = sigma_guess(K, d), step_factor(K)
                smin, smax = sigma_limits(K, d)
                new = sorted({min(max(g * f**j, smin), smax) for j in (-1, 0, 1, 2, 3)})
                status = "open"
            else:
                new, status = walker_next(w["rows"], K, d, domain=(arm == "P"))
            w["status"] = status
            done = {round(r["sigma"], 4) for r in w["rows"]}
            for s in new:
                cfg = make_cfg(arm, n, d, seed, s)
                if round(cfg["sigma"], 4) in done:
                    continue
                futures[pool.submit(rr.safe_run, cfg)] = key
                w["pending"] += 1
            if status == "open" and not w["pending"]:  # nothing new to run: stuck
                w["status"] = "stuck"

        for key in walkers:
            advance(key)
        log(f"{len(futures)} runs submitted for {len(walkers)} walkers")
        n_done = 0
        while futures:
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                key = futures.pop(fut)
                row = fut.result()
                files[key[0]].write(json.dumps(row, default=float) + "\n")
                files[key[0]].flush()
                w = walkers[key]
                w["pending"] -= 1
                n_done += 1
                if "error" in row:
                    log("ERROR", key, row["sigma"], row["error"][-300:])
                else:
                    w["rows"].append(row)
                advance(key)
            if n_done % 25 == 0:
                st = {}
                for w in walkers.values():
                    st[w["status"]] = st.get(w["status"], 0) + 1
                log(f"{n_done} done, {len(futures)} running, walkers {st}")
    for f in files.values():
        f.close()
    st = {}
    for key, w in walkers.items():
        st.setdefault(w["status"], []).append(key)
    log("final:", {k: len(v) for k, v in st.items()})
    for k in ("stuck", "stopped", "never_s2s", "all_s2s"):
        if k in st:
            log(k, st[k])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--arms", default="P")
    ap.add_argument("--n-proc", type=int, default=370)
    ap.add_argument("--out", default=os.environ.get("REGIME_MAP_R2_OUT", OUT))
    ap.add_argument("--stop-at", default=None, help="UTC HH:MM today: no new runs after it")
    a = ap.parse_args()
    stop = None
    if a.stop_at:
        h, m = map(int, a.stop_at.split(":"))
        now = dt.datetime.now(dt.timezone.utc)
        stop = now.replace(hour=h, minute=m, second=0, microsecond=0).timestamp()
    from concurrent.futures.process import BrokenProcessPool

    for attempt in range(5):  # a worker killed from outside breaks the pool: rebuild from the rows
        try:
            run(a.arms.split(","), a.n_proc, a.out, stop)
            break
        except BrokenProcessPool as e:
            print("BrokenProcessPool, restarting from the saved rows:", e, flush=True)
