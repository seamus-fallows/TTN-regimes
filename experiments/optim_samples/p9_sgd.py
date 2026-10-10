"""Test of theory_optimizer P9/P10: SGD on a later degenerate plateau (lengthening and trapping).

Binary N=4, d = chi = 3, odeco [3, 2, 1] (teacher seed 101). Start on the aligned route: component 1
learned (per-core amplitude 3^{1/3}), components 2 and 3 at per-core amplitude b0, so gradient flow
stays aligned and the second plateau has the exact length T_GF (event a_2 = 1/2; component 2 needs a
k = 3 route). Online SGD (`ttn.dynamics.run_sgd`) at eta/B in {3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4} (<= 0.25-0.3
eta*_ms(B)) with B = 64, and B = 16 at b0 in {0.01, 0.003} (independent noise seeds) to test that only
eta/B matters; b0 in {0.01, 0.005, 0.003, 0.002}; 6 noise seeds per point. Events: persistent last
crossing of a_2 = 1/2.
P9: T_SGD / T_GF = -ln(1-x)/x, x = c_gamma (eta/B) T_GF, trapped for x >= 1; heuristic c_gamma ~ 26 here.
Recorded: overlaps (events, min a_1 for P10), bond spectra (second Schmidt value sigma_2 of the
root edges = gauge-invariant route amplitude ~ b^3).

Run: uv run python experiments/optim_samples/p9_sgd.py [--workers 15]
Raw: /root/outputs/optim_samples/p9/*.npz/json; summary: results/p9_sgd.csv
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy.integrate import quad

from common import RAW, setup
from ttn.diagnostics import crossing_times
from ttn.dynamics import Problem, Recorder, integrate_gf, run_sgd
from ttn.network import aligned_init

OUT = RAW / "p9_r2"  # fix round 1: independent noise seeds (round-1 runs, B-coupled, in RAW / "p9")
from common import RESULTS as RES  # noqa: E402
B0S = [0.01, 0.005, 0.003, 0.002]
ETA_OVER_B = [3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4]
N_SEEDS = 6
ETA_MS = {16: 0.0160597, 64: 0.0527863}  # exact mean-square SGD limits (sgd_stability.py)
S_NEW, K = 2.0, 3


def t_gf_exact(b0):
    """Aligned balanced k=3 route: db/dt = b^2 (s - b^3), event b^3 = s/2."""
    return quad(lambda b: 1 / (b**2 * (S_NEW - b**3)), b0, (S_NEW / 2) ** (1 / 3), limit=200)[0]


def init(spec, teacher, b0):
    amps = torch.tensor([[3.0 ** (1 / 3), b0, b0]] * spec.n_cores)
    return spec.flatten(aligned_init(spec, teacher.factors, amps))


def run(task):
    torch.set_num_threads(1)
    b0, batch, eta, seed = task
    tag = f"b0{b0}_B{batch}_eta{eta}_seed{seed}"
    path = OUT / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    spec, teacher = setup("binary", 4)
    prob = Problem(spec, teacher)
    theta0 = init(spec, teacher, b0)
    t_gf = t_gf_exact(b0)
    t_max = min(4 * t_gf, 1600.0)
    if seed < 0:
        traj = integrate_gf(prob, theta0, t_max, method="lsoda", loss_floor=0.75,
                            log_times=np.linspace(t_max / 2000, t_max, 2000))
    else:
        n_steps = int(t_max / eta)
        # Independent noise for every (b0, B, eta, seed): no shared sample streams across B.
        mixed = np.random.SeedSequence([int(round(b0 * 1e6)), batch, int(round(eta * 1e9)),
                                        seed]).generate_state(1)[0]
        gen = torch.Generator().manual_seed(int(mixed))
        traj = run_sgd(prob, theta0, eta, n_steps, batch, gen, loss_floor=0.75,
                       log_steps=np.linspace(0, n_steps, 2000), recorder=Recorder(conservation=False))
    t, a = traj["t"], traj["overlaps"]
    spectra = traj["spectra"]  # (n, E, chi): edges of binary4 are the two bottom cores
    sigma2 = np.nanmean(spectra[:, :, 1], axis=1)
    t2 = float(crossing_times(t, a[:, 1], 0.5, last=True))  # persistent event (diagnostics.md)
    out = dict(b0=b0, B=batch, eta=eta, eta_over_B=eta / batch, seed=seed, t_gf=t_gf, t_max=t_max,
               t2=t2, t2_first=float(crossing_times(t, a[:, 1], 0.5)),
               escaped=bool(np.isfinite(t2)), min_a1=float(np.nanmin(a[:, 0])),
               frac_time_a1_below_half=float(np.mean(a[:, 0] < 0.5)),
               eta_over_eta_ms=eta / ETA_MS[batch] if batch in ETA_MS else None,
               a_end=a[-1].tolist(), stop=traj.meta.get("stop_reason"), wall=traj.meta["wall_time"],
               route_amp_start=float(sigma2[0] ** (1 / 3)), route_amp_end=float(sigma2[-1] ** (1 / 3)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{tag}.npz", t=t, a=a, sigma2=sigma2, loss=traj["pop_loss"])
    path.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(15, os.cpu_count() or 1))
    args = ap.parse_known_args()[0]  # --raw-dir / --out-dir: see common.py
    tasks = [(b0, 64, 0.0, -1) for b0 in B0S]
    for b0 in B0S:
        for r in ETA_OVER_B:
            tasks += [(b0, 64, r * 64, s) for s in range(N_SEEDS)]
            if b0 in (0.01, 0.003):  # same eta/B at B = 16, independent noise
                tasks += [(b0, 16, r * 16, s) for s in range(N_SEEDS)]
    tasks.sort(key=lambda tk: -t_gf_exact(tk[0]) / max(tk[2], 1e-3))
    with Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(run, tasks))
    df = pd.DataFrame(rows)
    gf = df[df.seed < 0].set_index("b0").t2
    df = df[df.seed >= 0].copy()
    df["t2_gf_run"] = df.b0.map(gf)
    df["ratio"] = df.t2 / df.t2_gf_run
    df["z"] = df.eta_over_B * df.t2_gf_run  # x = c_gamma z
    RES.mkdir(exist_ok=True)
    df.sort_values(["b0", "B", "eta", "seed"]).to_csv(RES / "p9_sgd.csv", index=False)
    print(gf)
    print(df.groupby(["b0", "B", "eta_over_B"]).agg(
        ratio=("ratio", "mean"), ratio_sd=("ratio", "std"), escaped=("escaped", "mean"),
        min_a1=("min_a1", "min")).to_string())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
