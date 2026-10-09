"""Pilot: covariance-induced order reversal on binary-4 / MPS-4 (throwaway, pre-library).

Teacher: comp 1 (s=1) on eigenvector e2, comp 2 (s=s2) on e1 at every site; power-law site spectra.
Runs (Sigma, s) and paired whitened baselines (I, Lambda s) and (I, sqrt(Lambda) s) with the same init.
"""
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pilot_gf import crossing_time, run_gf  # noqa: E402

from ttn.losses import power_law_cov  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

D, N = 4, 4


def job(args):
    tree_kind, s2, alpha, sigma, seed, variant, t_max = args
    torch.set_num_threads(1)
    spec = TTNSpec.uniform(make_tree(tree_kind, N), D, 4)
    cov = power_law_cov(D, alpha)
    lam = torch.diagonal(cov)
    big_lam = torch.stack([lam[1] ** N, lam[0] ** N])  # comp 1 on e2, comp 2 on e1
    s = torch.tensor([1.0, s2], dtype=torch.float64)
    if variant == "sigma":
        w, metrics = s, [cov] * N
    elif variant == "white_Ls":
        w, metrics = big_lam * s, None
    else:  # white_sqrtLs
        w, metrics = big_lam.sqrt() * s, None
    teacher = pattern_teacher(w, [[1] * N, [0] * N], D, basis="standard")
    cores = spec.init(sigma, seed=seed)
    rec = run_gf(spec, cores, teacher, metrics, t_max=t_max, n_rec=600, w_star=teacher.dense())
    ov = rec["overlaps"]
    t_half = [crossing_time(rec["t"], ov[:, r]) for r in range(2)]
    out = dict(args=args, t=rec["t"], loss=rec["loss"], overlaps=ov, t_half=t_half,
               big_lam=big_lam.numpy(), q_drift=rec["q_drift"], nfev=rec["nfev"], wall=rec["wall"],
               spectra=rec["spectra"])
    print(args, "t_half", np.round(t_half, 2), "nfev", rec["nfev"], "wall %.0f" % rec["wall"],
          flush=True)
    return out


if __name__ == "__main__":
    jobs = []
    for tree_kind, t_max in [("binary", 3000.0), ("mps", 3e4)]:
        for seed in range(8):
            for variant in ["sigma", "white_Ls", "white_sqrtLs"]:
                jobs.append((tree_kind, 0.6, 0.25, 0.05 if tree_kind == "binary" else 0.1,
                             seed, variant, t_max))
    with Pool(30) as pool:
        results = pool.map(job, jobs, chunksize=1)
    with open("/root/outputs/data_teacher/pilots/pilot_reversal.pkl", "wb") as f:
        pickle.dump(results, f)
