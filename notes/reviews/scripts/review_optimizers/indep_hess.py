"""Independent sharpness / balance / bond-rank check at the end of reproduced runs (no experiments/ code).

Hessian of my own dense population loss 1/2||W - W*||^2 at theta_end (torch.autograd.functional.hessian),
lambda_max vs the pattern rule K s_1^(2-2/K); balance index per edge from library conserved_quantities;
for finite-sample runs: population loss and bond singular values (dense SVD of the matricization).

    uv run python indep_hess.py RAWDIR OUTCSV
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
sys.path.insert(0, str(Path(__file__).parent))
from indep_check import Prob  # noqa: E402  (my own dense problem)
from ttn.network import conserved_quantities  # noqa: E402

torch.set_num_threads(1)


def balance(p, theta):
    spec, t = p.spec, p.spec.tree
    cores = spec.unflatten(theta)
    q = conserved_quantities(spec, cores)
    out = []
    for v in t.edges:
        par = t.parent[v]
        gv = cores[v].reshape(-1, cores[v].shape[-1])
        leg = len(t.sites[par]) + t.children[par].index(v)
        gp = torch.movedim(cores[par], leg, 0).reshape(cores[par].shape[leg], -1)
        out.append((q[v].norm() / ((gv.T @ gv).norm() + (gp @ gp.T).norm())).item())
    return out


def bond_svals(p, theta):
    w = p.W(theta).detach()
    out = []
    for v in p.spec.tree.edges:
        rows = sorted(p.spec.tree.subtree_sites[v])
        cols = [s for s in range(p.n) if s not in rows]
        m = w.permute(rows + cols).reshape(3 ** len(rows), -1)
        out.append(torch.linalg.svdvals(m).numpy())
    return out


def main(raw, out):
    import pandas as pd

    rows = []
    probs = {}
    for f in sorted(Path(raw).glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            theta = torch.from_numpy(z["theta_end"])
        cfg = meta["cfg"]
        if cfg["optimizer"] not in ("gf", "gd", "momentum", "adam", "rmsprop"):
            continue
        key = (cfg["tree"], cfg["n_sites"])
        if key not in probs:
            probs[key] = Prob(cfg["tree"], cfg["n_sites"], cfg["d"], cfg["chi"], cfg["weights"])
        p = probs[key]
        L = p.loss(theta).item()
        L0 = 0.5 * float(np.sum(np.square(cfg["weights"])))
        r = dict(
            file=f.name,
            optimizer=cfg["optimizer"],
            lr=cfg.get("lr"),
            sigma=cfg["sigma"],
            seed=cfg["seed"],
            n_samples=cfg.get("n_samples"),
            pop_loss_end=L,
            stop=meta.get("stop_reason"),
        )
        if L < 1e-3 * L0 and len(theta) <= 400:
            h = torch.autograd.functional.hessian(p.loss, theta)
            lam = torch.linalg.eigvalsh(h)[-1].item()
            K = p.K
            r.update(lambda_max=lam, pattern_rule=K * max(cfg["weights"]) ** (2 - 2 / K))
            r["ratio"] = lam / r["pattern_rule"]
        r["balance_max"] = max(balance(p, theta))
        sv = bond_svals(p, theta)
        r["rank_gt_0.1"] = "".join(str(int((s > 0.1).sum())) for s in sv)
        r["s4_max"] = max(float(s[3]) if len(s) > 3 else 0.0 for s in sv)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(
        pd.DataFrame(rows)
        .groupby(["optimizer", "lr", "n_samples"], dropna=False)[["ratio", "balance_max"]]
        .agg(["count", "mean", "min", "max"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
