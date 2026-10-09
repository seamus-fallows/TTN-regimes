"""Throwaway: lazy-limit predictions at init (tensor-space Jacobian, small D only).
residual fraction ||P_perp W*||^2/||W*||^2 and kappa = sigma^K * deltaK_lin (sigma-independent)."""
import sys
import numpy as np
import torch
from ttn.network import TTNSpec, to_dense, forward
from ttn.teachers import odeco_teacher
from ttn.tree import make_tree
torch.set_num_threads(4)

def lazy_pred(kind, n, d, chi, seed, cap=True, weights=(3., 2., 1.), n_probe=128):
    spec = TTNSpec.uniform(make_tree(kind, n), d, chi, cap=cap)
    K = spec.n_cores
    th0 = spec.flatten(spec.init(1.0, seed=seed))
    base = odeco_teacher(n, d, list(weights), seed=1000 + seed)
    wstar = base.dense().reshape(-1)
    J = torch.func.jacrev(lambda p: to_dense(spec, spec.unflatten(p)).reshape(-1))(th0)  # D x P
    dth = torch.linalg.lstsq(J, wstar.unsqueeze(1), driver="gelsd").solution[:, 0]
    res = (wstar - J @ dth).norm() ** 2 / wstar.norm() ** 2
    g = torch.Generator().manual_seed(12345)
    probes = torch.randn(n_probe, n, d, generator=g, dtype=torch.float64)
    jp = lambda p: torch.func.jacrev(lambda q: forward(spec, spec.unflatten(q), probes))(p)
    J0 = jp(th0); K0 = J0 @ J0.T
    eps = 1e-6
    Jp = jp(th0 + eps * dth); Jm = jp(th0 - eps * dth)
    dK = (Jp @ Jp.T - Jm @ Jm.T) / (2 * eps)
    kappa = (dK.norm() / K0.norm()).item()
    rank = torch.linalg.matrix_rank(J).item()
    return dict(K=K, P=spec.n_params, D=J.shape[0], rank=rank, res=res.item(), kappa=kappa,
                rel_dth=(dth.norm() / th0.norm()).item())

if __name__ == "__main__":
    kind, n, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for chi in [int(x) for x in sys.argv[4].split(",")]:
        rows = [lazy_pred(kind, n, d, chi, s) for s in range(5)]
        r0 = rows[0]
        res = np.array([r["res"] for r in rows]); kap = np.array([r["kappa"] for r in rows])
        dthr = np.array([r["rel_dth"] for r in rows])
        print(f"{kind}{n} d={d} chi={chi} K={r0['K']} P={r0['P']} D={r0['D']} rankJ={r0['rank']} "
              f"residual={res.mean():.3f}+-{res.std():.3f} kappa={kap.mean():.2f}+-{kap.std():.2f} "
              f"rel_dth(sigma=1)={dthr.mean():.2f}  sigma_lazy(0.1)={((kap/0.1)**(1/r0['K'])).mean():.2f}")
