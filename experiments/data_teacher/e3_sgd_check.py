"""Step 3 check with sampled data: online SGD (fresh batch every step) on correlated inputs vs the
exact population gradient flow of configs/e3_correlated.yaml (same init).

Minimal SGD loop (the library's run_sgd had not landed): theta <- theta - eta grad L_batch, with
L_batch the empirical square loss (ttn.losses.empirical_loss) on B fresh samples of the
shared-latent-factor inputs; time t = eta * steps. The exact population loss (moment matrix) is
recorded along the way. Usage: uv run python experiments/data_teacher/e3_sgd_check.py
"""

from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS, OUT_ROOT  # noqa: E402
from correlated import dense_metric_loss_fn, moment_matrix, sample_correlated, site_cross_cov  # noqa: E402
from run_generic import expand  # noqa: E402
from setups import build  # noqa: E402

from ttn.losses import empirical_loss  # noqa: E402

HERE = Path(__file__).parent
RHO, TREE, SIGMA, SEEDS = 0.3, "binary", 0.05, [0, 1, 2]
SETTINGS = [(0.01, 256), (0.01, 4096), (0.002, 4096)]  # (eta, batch)
T_MAX = 400.0


def run(args):
    eta, batch, seed = args
    path = OUT_ROOT / "e3_sgd" / f"{TREE}_rho{RHO}_eta{eta}_B{batch}_seed{seed}.npz"
    if path.exists():
        return dict(np.load(path))
    torch.set_num_threads(1)
    cfg = yaml.safe_load(open(HERE / "configs/e3_correlated.yaml"))
    case = next(c for c in expand(cfg) if c["rho"] == RHO)
    spec, teacher, _, _, _ = build(dict(case, tree=TREE), seed)
    metric = moment_matrix(4, 4, site_cross_cov(4, RHO))
    pop = dense_metric_loss_fn(spec, teacher.dense(), metric)
    flat = spec.flatten(spec.init(SIGMA, seed=seed)).clone().requires_grad_(True)
    gen = torch.Generator().manual_seed(1000 + seed)  # data stream, independent of init and teacher
    n_steps = int(T_MAX / eta)
    rec_every = max(1, n_steps // 2000)
    ts, losses = [], []
    for step in range(n_steps + 1):
        if step % rec_every == 0:
            with torch.no_grad():
                ts.append(step * eta)
                losses.append(pop(flat).item())
        x = sample_correlated(batch, 4, 4, RHO, gen)
        loss = empirical_loss(spec, spec.unflatten(flat), x, teacher(x))
        (g,) = torch.autograd.grad(loss, flat)
        with torch.no_grad():
            flat -= eta * g
    out = dict(t=np.array(ts), loss=np.array(losses))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def main():
    jobs = [(eta, b, s) for eta, b in SETTINGS for s in SEEDS]
    with Pool(len(jobs)) as pool:
        res = pool.map(run, jobs)
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(5 * len(SEEDS), 4))
    rows = []
    for j, seed in enumerate(SEEDS):
        gf = np.load(OUT_ROOT / "e3_correlated" / f"rho{RHO}_{TREE}_s{SIGMA}_seed{seed}.npz",
                     allow_pickle=True)
        ax = axes[j]
        ax.plot(gf["t"], gf["loss"] / gf["loss"][0], "k-", lw=2, label="exact population GF")
        for (eta, b, s), r in zip(jobs, res):
            if s != seed:
                continue
            ax.plot(r["t"], r["loss"] / r["loss"][0], lw=1, label=f"online SGD eta={eta}, B={b}")
            for level in (0.9, 0.5, 0.1, 0.01):
                t_sgd = r["t"][np.argmax(r["loss"] / r["loss"][0] < level)]
                t_gf = gf["t"][np.argmax(gf["loss"] / gf["loss"][0] < level)]
                rows.append(dict(seed=seed, eta=eta, batch=b, level=level, t_sgd=t_sgd, t_gf=t_gf))
        ax.set(xlabel="t = eta x steps", ylabel="population loss / L(0)", yscale="log",
               title=f"{TREE}-4, rho={RHO}, sigma={SIGMA}, seed {seed}", xlim=(0, T_MAX))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "e3_sgd_check.png", dpi=110)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e3_sgd_check.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
