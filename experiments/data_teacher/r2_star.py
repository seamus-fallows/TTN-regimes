"""Round 2, item 2: star teachers (theory_saddles.md v2, section 4.6, prediction P8).

Teacher (d = 2, pattern_teacher, standard basis): q = 0000 (s_q), r = 1010, r' = 1001 (weights w, w'),
so r and r' share the piggyback node (core of site 0, e1) and rho(S) = sqrt(w^2 + w'^2).
Prediction: "q learned" is a strict saddle iff rho(S) > s_q, with lambda_min = -a^(K-2)(rho - s_q),
a = s_q^(1/K): -0.3716 (mps4), -0.3310 (binary4), -0.4311 (ht4) for (2; 1.6, 1.6). binary8 analogue:
q = 0^8, r, r' = 1 on {0,2} and {0,4}.

Modes:
  hessian   gauge-projected Hessian at the exact balanced theta_q (aligned_init) for several weights
  random    random fan-in init, library LSODA, two-stage dense grid: forgetting depth vs sigma
  saddle    start at theta_q + fan-in noise sigma: does q get forgotten? (threshold scan in w)
Usage: uv run python experiments/data_teacher/r2_star.py {hessian|random|saddle} [--procs 28]
Outputs: results/r2_star_hessian.csv, results/r2_star_random.csv, results/r2_star_saddle.csv;
raw runs in /root/outputs/data_teacher/r2_star_{random,saddle}/.
"""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import RESULTS, OUT_ROOT
from run_dense import two_stage

from ttn.diagnostics import gauge_directions, hessian, project_out_gauge, saddle_type
from ttn.dynamics import Problem, Recorder, integrate_gf, merged_rows
from ttn.network import TTNSpec, aligned_init
from ttn.teachers import pattern_teacher
from ttn.tree import make_tree

HERE = Path(__file__).parent
D = 2


def star(n, s_q, w1, w2):
    if n == 4:
        pats = [[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]
    else:  # binary8 analogue: r, r' differ from q on {0,2} and {0,4}
        pats = [[0] * 8, [1, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0]]
    return pattern_teacher([s_q, w1, w2], pats, D, basis="standard")


def spec_for(tree, n, chi=3):
    return TTNSpec.uniform(make_tree(tree, n), D, chi)


def theta_q(spec, teacher):
    """Exact balanced critical point with only q learned (amplitude s_q^(1/K) on every core); built
    from q alone, since with d = 2 some bonds are smaller than the number of components."""
    k = spec.n_cores
    amps = torch.full((k, 1), float(teacher.weights[0]) ** (1.0 / k), dtype=torch.float64)
    return spec.flatten(aligned_init(spec, teacher.factors[:1], amps))


def hessian_mode():
    rows = []
    cases = [("mps", 4), ("binary", 4), ("ht", 4), ("binary", 8)]
    for w in (1.2, 1.3, 1.4, 1.45, 1.6):
        for tree, n in cases:
            torch.set_num_threads(1)
            spec = spec_for(tree, n)
            teacher = star(n, 2.0, w, w)
            prob = Problem(spec, teacher)
            th = theta_q(spec, teacher)
            h = hessian(prob.loss, th)
            _, eigs = project_out_gauge(h, gauge_directions(spec, th))
            st = saddle_type(eigs)
            lam_raw = float(torch.linalg.eigvalsh(h)[0])
            k = spec.n_cores
            rho = math.sqrt(2) * w
            pred = -(2.0 ** ((k - 2) / k)) * (rho - 2.0)
            rows.append(dict(tree=f"{tree}{n}", w=w, rho=rho, grad=float(prob.grad(th).norm()),
                             lambda_min=st["lambda_min"], lambda_min_raw=lam_raw, pred_if_strict=pred,
                             n_negative=st["n_negative"]))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "r2_star_hessian.csv", index=False)
    print(df.to_string())


def summarize(t, ov, loss):
    """a_q dynamics: first peak, minimum after the peak (forgetting depth), final; plateau times."""
    aq = ov[:, 0]
    k_peak = int(np.argmax(aq > 0.5)) if (aq > 0.5).any() else int(np.argmax(aq))
    # first local maximum of a_q after it passes 0.5 (or its global max if it never does)
    k = k_peak
    while k + 1 < len(aq) and aq[k + 1] >= aq[k]:
        k += 1
    peak = float(aq[k])
    after = aq[k:]
    k_min = k + int(np.argmin(after))
    forgotten = bool(aq[k_min] < 0.5)
    t_r = [t[np.argmax(ov[:, j] > 0.5)] if (ov[:, j] > 0.5).any() else np.nan for j in range(3)]
    relearn = np.nonzero((np.arange(len(aq)) > k_min) & (aq > 0.5))[0]
    # time spent near the "q learned" level L = (w^2 + w'^2)/2 and the "q forgotten" level s_q^2/2
    return dict(aq_peak=peak, t_peak=float(t[k]), aq_min_after=float(aq[k_min]),
                t_min=float(t[k_min]), forgotten=forgotten,
                t_relearn=float(t[relearn[0]]) if len(relearn) else np.nan,
                t_q=t_r[0], t_r=t_r[1], t_rp=t_r[2], loss_end=float(loss[-1]))


def level_time(t, loss, level, tol=0.01):
    """Total time with |L - level| < tol (a plateau at that level)."""
    m = np.abs(loss - level) < tol
    dt = np.diff(t, prepend=t[0])
    return float(dt[m].sum())


def random_job(args):
    tree, n, w, sigma, seed = args
    torch.set_num_threads(1)
    path = OUT_ROOT / "r2_star_random" / f"{tree}{n}_w{w}_s{sigma}_seed{seed}.npz"
    if path.exists():
        return json.loads(str(np.load(path)["summary"]))
    spec = spec_for(tree, n)
    teacher = star(n, 2.0, w, w)
    prob = Problem(spec, teacher)
    tr = two_stage(prob, spec.flatten(spec.init(sigma, seed=seed)), 1e8, 3000, 1e-6)
    m = merged_rows(tr.data)  # log + fine rows (event resolution, final library)
    t, loss, ov = m["t"], m["loss"], m["overlaps"]
    s = dict(tree=f"{tree}{n}", w=w, sigma=sigma, seed=seed, **summarize(t, ov, loss),
             time_at_q_level=level_time(t, loss, w * w), time_at_forgot_level=level_time(t, loss, 2.0),
             max_drift=float(np.nanmax(tr.data["conservation_drift"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=t, loss=loss, overlaps=ov, summary=json.dumps(s))
    print(s["tree"], w, sigma, seed, "forgot", s["forgotten"], f"min aq {s['aq_min_after']:.3g}",
          flush=True)
    return s


def saddle_job(args):
    """stop_aq (passed with the job, not as a module global, which spawned workers would not see): stop
    once a_q < stop_aq. 0.1 (sweep), 0.6 (threshold scan; on ht4 the strict escape
    ends at a blended point with a_q ~ 0.67 that is left only in ~1/eps time), 0.99 (ht4 completion runs)."""
    tree, n, w, sigma, seed, stop_aq = args if len(args) == 6 else (*args, 0.1)
    torch.set_num_threads(1)
    path = OUT_ROOT / "r2_star_saddle" / f"{tree}{n}_w{w}_s{sigma}_seed{seed}.npz"
    if path.exists():
        return json.loads(str(np.load(path)["summary"]))
    spec = spec_for(tree, n)
    teacher = star(n, 2.0, w, w)
    prob = Problem(spec, teacher)
    noise = spec.flatten(spec.init(sigma, seed=seed))
    th0 = theta_q(spec, teacher) + noise
    l0 = float(prob.loss(th0))
    # chunks of doubling length until q is forgotten (a_q < 0.1) or r is learned (a_r > 0.5)
    t_off, chunk, parts, theta = 0.0, 20.0, [], th0
    while True:
        tr = integrate_gf(prob, theta, t_max=chunk, n_log=400, rtol=1e-9, atol=1e-14,
                          method="lsoda", recorder=Recorder(spectra=False, overlaps=True))
        m = merged_rows(tr.data)
        parts.append((m["t"] + t_off, m["loss"], m["overlaps"]))
        ov = m["overlaps"]
        theta = torch.as_tensor(tr.data["theta_end"])
        t_off += chunk
        # stop once q is forgotten (a_q < stop_aq) or r is learned
        if (ov[:, 0] < stop_aq).any() or (ov[:, 1:] > 0.5).any() or t_off > 1e9:
            break
        chunk *= 2
    t = np.concatenate([p[0] for p in parts])
    ov = np.concatenate([p[2] for p in parts])
    aq = ov[:, 0]
    forgot = bool((aq < 0.5).any())
    first = "forget_q" if forgot and (not (ov[:, 1:] > 0.5).any() or
                                      np.argmax(aq < 0.5) <= np.argmax((ov[:, 1:] > 0.5).any(axis=1))) \
        else ("learn_r" if (ov[:, 1:] > 0.5).any() else "none")
    s = dict(tree=f"{tree}{n}", w=w, rho=math.sqrt(2) * w, sigma=sigma, seed=seed, outcome=first,
             t_forget=float(t[np.argmax(aq < 0.5)]) if forgot else np.nan,
             t_end=float(t[-1]), loss_start=l0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=t, overlaps=ov, summary=json.dumps(s))
    print(s["tree"], w, sigma, seed, s["outcome"], f"t={s['t_end']:.3g}", flush=True)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hessian", "random", "saddle", "threshold", "complete_ht"])
    ap.add_argument("--procs", type=int, default=28)
    a = ap.parse_args()
    if a.mode == "hessian":
        hessian_mode()
    elif a.mode == "random":
        jobs = []
        for tree, sig in (("mps", [0.01, 0.0158, 0.0251, 0.0398, 0.0631, 0.1]),
                          ("binary", [0.005, 0.0079, 0.0126, 0.02, 0.0316, 0.05])):
            for w in (1.6, 1.2):
                jobs += [(tree, 4, w, s, seed) for s in sig for seed in range(10)]
        jobs.sort(key=lambda j: j[3])
        with Pool(a.procs) as pool:
            rows = pool.map(random_job, jobs, chunksize=1)
        pd.DataFrame(rows).to_csv(RESULTS / "r2_star_random.csv", index=False)
    elif a.mode == "complete_ht":
        # the 5 ht4 runs (w = 1.42, eps = 1e-6) missing from the sweep: on ht4 the strict escape from
        # theta_q ends at the blended rank-1 point, which at w = 1.42 has a_q = 0.984 (star_blended.py), and
        # going below a_q = 0.9 takes the polynomial ~1/eps stage; stop at departure (a_q < 0.99) instead
        jobs = [("ht", 4, 1.42, 1e-6, seed, 0.99) for seed in range(5)]
        with Pool(5) as pool:
            pool.map(saddle_job, jobs, chunksize=1)
    elif a.mode == "threshold":
        # fine scan around w* = s_q / sqrt(2) = 1.41421 (rho(S) = s_q) at small noise
        jobs = [(tree, n, w, s, seed, 0.6)
                for tree, n in (("mps", 4), ("binary", 4))
                for w in (1.410, 1.412, 1.413, 1.4138, 1.4146, 1.4155, 1.417, 1.419)
                for s in (1e-6,) for seed in range(3)]
        with Pool(a.procs) as pool:
            rows = pool.map(saddle_job, jobs, chunksize=1)
        pd.DataFrame(rows).to_csv(RESULTS / "r2_star_threshold_scan.csv", index=False)
    else:
        jobs = [(tree, n, w, s, seed)
                for tree, n in (("mps", 4), ("binary", 4), ("ht", 4))
                for w in (1.30, 1.36, 1.40, 1.42, 1.44, 1.48, 1.60)
                for s in (1e-2, 1e-3, 1e-4, 1e-6) for seed in range(5)]
        with Pool(a.procs) as pool:
            rows = pool.map(saddle_job, jobs, chunksize=1)
        pd.DataFrame(rows).to_csv(RESULTS / "r2_star_saddle.csv", index=False)
