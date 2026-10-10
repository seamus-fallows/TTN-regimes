"""Checks: cancellation-free gradient = plain gradient; regauge is an exact gauge map; Hessian = d grad."""
import time
import numpy as np
import torch
from runs import make_model, CELLS, fanin

for cell in CELLS:
    m = make_model(cell)
    rng = np.random.default_rng(0)
    th = m.theta_S + 1e-3 * torch.from_numpy(rng.standard_normal(m.P))
    m.set_ref(None)
    g0, dL0, _ = m.grad(th)
    m.set_ref(m.theta_S)
    g1, dL1, _ = m.grad(th)
    t0 = time.time()
    for _ in range(20):
        m.grad(th)
    tg = (time.time() - t0) / 20
    t0 = time.time()
    H = m.hessian(th)
    th_ = time.time() - t0
    # Hessian vs finite difference of the gradient
    v = torch.from_numpy(rng.standard_normal(m.P))
    h = 1e-6
    fd = (m.grad(th + h * v)[0] - m.grad(th - h * v)[0]) / (2 * h)
    # regauge on a random point
    y = fanin(m, 3, 1.0)
    yg = m.regauge(y)
    werr = (m.W(yg) - m.W(y)).abs().max().item()
    Qd = (m.conserved(yg) - m.conserved(y)).abs().max().item()  # orthogonal map: Q -> O^T Q O, not equal
    print(f"{cell:6s} P={m.P:3d} k*={m.k_star} splits={[''.join(map(str,p)) for p in m.split_patterns]} "
          f"|g1-g0|={float((g1-g0).abs().max()):.1e} dL {dL0:.3e} {dL1:.3e} grad {tg*1e3:.2f} ms hess {th_*1e3:.0f} ms "
          f"|H v - fd|={float((H@v-fd).abs().max()):.1e} regauge W err {werr:.1e}")
