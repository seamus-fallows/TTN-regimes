"""Sanity checks of vttn against the shared library (contraction) and against invariances."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, torch
from vttn import Net, PatternSaddle, gauge_basis
from ttn.network import TTNSpec, to_dense

for kind, n in [("mps", 6), ("binary", 6), ("ht", 4), ("binary", 4)]:
    net = Net(kind, n)
    spec = TTNSpec.uniform(net.tree, 2, 2)
    assert tuple(map(tuple, spec.shapes)) == tuple(net.shapes), (spec.shapes, net.shapes)
    th = torch.randn(net.P, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    err = (net.dense(th) - to_dense(spec, spec.unflatten(th))).abs().max().item()
    G = gauge_basis(net, th)
    # gauge invariance: W changes at second order along gauge directions
    dv = torch.tensor(G @ np.random.default_rng(0).standard_normal(G.shape[1]))
    d1 = (net.dense(th + 1e-4 * dv) - net.dense(th)).abs().max().item()
    print(kind, n, "K", net.K, "P", net.P, "max|dense diff| vs library", err, "gauge dim", G.shape[1], "first-order W change", d1)
ps = PatternSaddle(Net("mps", 6), [((1, 1, 0, 0, 0, 1), 0.8), ((1, 1, 0, 0, 1, 0), 0.8)])
g = torch.autograd.functional.jacobian(ps.dL, ps.theta_S)
print("pair-star: k_r", ps.k_r, "k*", ps.k_star, "L_S", ps.L_S, "|grad L| at theta_S", g.abs().max().item())
