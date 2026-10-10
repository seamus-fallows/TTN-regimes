"""(1) run_regime.tensor_kernel vs ttn.diagnostics.tensor_tangent_kernel on small cases.
(2) Independent dK_tensor (autograd.functional.jacobian of to_dense) for sampled saved runs vs the cache.
Usage: uv run python tensor_check.py small | sample EXP N_PER_GROUP
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
ROOT = "c:/Users/seamu/Iliad-ML/TTN_regimes"
sys.path.insert(0, os.path.join(ROOT, "experiments/regime_map"))
from run_regime import build, row_cfg, run_id, tensor_kernel  # noqa: E402

from ttn.diagnostics import tensor_tangent_kernel  # noqa: E402
from ttn.network import to_dense  # noqa: E402

RAW = os.path.join(ROOT, "data/regime_map")
CACHE = os.path.join(ROOT, "data/regime_map_local/tensor_kernel")
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def my_kernel(spec, theta, w):
    f = lambda th: to_dense(spec, spec.unflatten(th)).reshape(-1)  # noqa: E731
    J = torch.autograd.functional.jacobian(f, theta.detach().clone(), vectorize=True)
    Jw = J if w is None else J * w
    return Jw @ J.T


def rel(a, b):
    return float((a - b).norm() / b.norm())


def small():
    cases = [
        dict(kind="binary", n=4, d=3, chi=3, sigma=0.7, seed=0, model="centred"),
        dict(kind="binary", n=4, d=3, chi=6, sigma=1.3, seed=1, model="centred", lr="ntk"),
        dict(kind="binary", n=4, d=3, chi=5, sigma=1.0, seed=2, model="centred", lr="mup"),
        dict(kind="binary", n=4, d=3, chi=4, sigma=1.0, seed=3, model="twocopy"),
        dict(kind="mps", n=5, d=2, chi=4, sigma=0.9, seed=4, model="centred", weights=[3.0, 2.0]),
        dict(kind="binary", n=8, d=2, chi=3, sigma=0.8, seed=5, model="plain", weights=[3.0, 2.0]),
        dict(kind="binary", n=4, d=3, chi=16, sigma=1.5, seed=0, model="centred", root_scale=3.0),
    ]
    for c in cases:
        c["exp"] = "x"
        problem, theta0, _, info = build(c)
        spec = problem.spec
        w = problem.precond
        g = torch.Generator().manual_seed(7)
        theta1 = theta0 + 0.3 * theta0.norm() / theta0.numel() ** 0.5 * torch.randn(
            theta0.shape, generator=g, dtype=theta0.dtype
        )
        if w is not None:
            theta1 = theta0 + (theta1 - theta0) * (w != 0)
        A0, A1 = tensor_kernel(spec, theta0, w), tensor_kernel(spec, theta1, w)
        L0 = tensor_tangent_kernel(spec, theta0, w)
        L1 = tensor_tangent_kernel(spec, theta1, w)
        M0, M1 = my_kernel(spec, theta0, w), my_kernel(spec, theta1, w)
        dA = float((A1 - A0).norm() / A0.norm())
        dL = float((L1 - L0).norm() / L0.norm())
        dM = float((M1 - M0).norm() / M0.norm())
        print(
            c["kind"],
            c["n"],
            c.get("lr", "sp"),
            c["model"],
            c["chi"],
            "| K rel diff run_regime vs lib %.1e, vs mine %.1e" % (rel(A0, L0), rel(A0, M0)),
            "| dK: %.15f %.15f %.15f  rel %.1e %.1e"
            % (dA, dL, dM, abs(dA - dL) / dL, abs(dA - dM) / dM),
            flush=True,
        )


def sample(exp, n_per):
    rows = []
    with open(os.path.join(RAW, exp, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    byid = {}
    for r in rows:
        cfg = {k: v for k, v in row_cfg(r).items() if v is not None}
        byid[run_id(cfg)] = (r, cfg)
    cache = pd.read_csv(os.path.join(CACHE, f"{exp}.csv")).drop_duplicates("run_id", keep="last")
    cache = cache[cache.run_id.isin(byid)]
    cache["model"] = [byid[i][0].get("model") for i in cache.run_id]
    cache["chi"] = [byid[i][0].get("chi") for i in cache.run_id]
    cache["lr"] = [byid[i][0].get("lr", "sp") for i in cache.run_id]
    rng = np.random.default_rng(11)
    pick = []
    for _, g in cache.groupby(["model", "chi", "lr"]):
        pick += list(rng.choice(g.run_id.values, size=min(n_per, len(g)), replace=False))
    out = []
    for rid in pick:
        r, cfg = byid[rid]
        cfg = {k: v for k, v in cfg.items() if k not in ("refine",)}
        problem, theta0, _, info = build(cfg)
        spec = problem.spec
        with np.load(os.path.join(RAW, exp, "traj", rid + ".npz"), allow_pickle=False) as z:
            th1 = torch.from_numpy(z["theta_end"])
            rdist = float(z["rel_dist_from_init"][-1])
        w = problem.precond
        K0, K1 = my_kernel(spec, theta0, w), my_kernel(spec, th1, w)
        dk = float((K1 - K0).norm() / K0.norm())
        rd = float((th1 - theta0).norm() / theta0.norm())
        c = float(cache[cache.run_id == rid].dK_tensor.iloc[0])
        out.append(
            dict(
                exp=exp,
                rid=rid,
                mine=dk,
                cache=c,
                rel=abs(dk - c) / abs(c),
                rdist_check=abs(rd - rdist) / rdist,
            )
        )
        print(
            rid,
            "%.6g %.6g rel %.1e theta0 check %.1e"
            % (dk, c, out[-1]["rel"], out[-1]["rdist_check"]),
            flush=True,
        )
    pd.DataFrame(out).to_csv(os.path.join(OUTD, f"tensor_sample_{exp}.csv"), index=False)
    d = pd.DataFrame(out)
    print(exp, "n", len(d), "max rel", d.rel.max(), "max theta0 check", d.rdist_check.max())


if __name__ == "__main__":
    if sys.argv[1] == "small":
        small()
    else:
        sample(sys.argv[2], int(sys.argv[3]))
