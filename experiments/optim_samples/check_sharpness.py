"""Check the sharpness formula lambda_max(H) = K s_1^{2-2/K} at the balanced global minimum.

Builds the balanced aligned minimum (every core amplitude s_r^{1/K}) for an odeco teacher, applies
random gauge transformations (orthogonal: stays balanced; diagonal scalings: imbalanced), and
compares the top Hessian eigenvalue of the population loss with the prediction. Also checks the
AM-GM bound: an imbalanced rank-one minimum has sharpness s^2 sum_v 1/a_v^2 >= K s^{2-2/K}.

Run: uv run python experiments/optim_samples/check_sharpness.py
Writes results/sharpness_check.csv.
"""


import pandas as pd
import torch

from ttn.losses import population_loss
from ttn.network import TTNSpec, aligned_init, apply_on_leg
from ttn.teachers import odeco_teacher
from ttn.tree import make_tree

torch.set_default_dtype(torch.float64)
from common import RESULTS as OUT  # noqa: E402


def hessian_top(spec, cores, teacher):
    flat = spec.flatten(cores).detach()
    t2 = teacher.norm2()

    def loss(p):
        return population_loss(spec, spec.unflatten(p), teacher, teacher_norm2=t2)

    h = torch.autograd.functional.hessian(loss, flat)
    ev = torch.linalg.eigvalsh(h)
    return ev[-1].item(), ev[0].item(), loss(flat).item()


def gauge(spec, cores, mats):
    """Apply B on edge v below (child core's parent leg) and B^{-T} on the parent's v-leg."""
    t = spec.tree
    out = [c.clone() for c in cores]
    for v, b in mats.items():
        p = t.parent[v]
        out[v] = apply_on_leg(out[v], b, out[v].ndim - 1)
        leg = len(t.sites[p]) + t.children[p].index(v)
        out[p] = apply_on_leg(out[p], torch.linalg.inv(b).T, leg)
    return out


def balanced_min(spec, teacher, amps_scale=None):
    k, r = spec.n_cores, teacher.n_components
    a = teacher.weights.unsqueeze(0).expand(k, r) ** (1.0 / k)
    if amps_scale is not None:
        a = a * amps_scale
    return aligned_init(spec, teacher.factors, a.clone())


def main():
    rows = []
    g = torch.Generator().manual_seed(0)
    weights = [3.0, 2.0, 1.0]
    for kind, n in [("binary", 4), ("binary", 8), ("mps", 4), ("mps", 8), ("ht", 4)]:
        for chi in (3, 5):
            spec = TTNSpec.uniform(make_tree(kind, n), 3, chi)
            teacher = odeco_teacher(n, 3, weights, seed=101)
            k = spec.n_cores
            pred = k * weights[0] ** (2 - 2 / k)
            cores = balanced_min(spec, teacher)
            lam, lmin, loss = hessian_top(spec, cores, teacher)
            rows.append(dict(tree=kind, N=n, K=k, chi=chi, case="balanced aligned", lam_max=lam,
                             lam_min=lmin, pred=pred, loss=loss))
            # Random orthogonal gauge: still balanced (Q = 0), same sharpness predicted.
            mats = {}
            for v in spec.tree.edges:
                q, _ = torch.linalg.qr(torch.randn(spec.bond_dims[v], spec.bond_dims[v], generator=g))
                mats[v] = q
            lam, lmin, loss = hessian_top(spec, gauge(spec, cores, mats), teacher)
            rows.append(dict(tree=kind, N=n, K=k, chi=chi, case="orthogonal gauge", lam_max=lam,
                             lam_min=lmin, pred=pred, loss=loss))
            # Imbalanced: scale core amplitudes by c_v with prod c_v = 1 (still a global minimum).
            c = torch.exp(0.4 * torch.randn(k, generator=g))
            c = c / c.prod() ** (1.0 / k)
            cores_i = balanced_min(spec, teacher, c.unsqueeze(1))
            lam, lmin, loss = hessian_top(spec, cores_i, teacher)
            amgm = sum(weights[0] ** 2 / (weights[0] ** (1 / k) * cv) ** 2 for cv in c.tolist())
            rows.append(dict(tree=kind, N=n, K=k, chi=chi, case="imbalanced", lam_max=lam,
                             lam_min=lmin, pred=amgm, loss=loss))
    df = pd.DataFrame(rows)
    df["rel_err"] = (df.lam_max / df.pred - 1).abs()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "sharpness_check.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.to_string(float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
