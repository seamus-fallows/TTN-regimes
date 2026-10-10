"""Gauge-projected Hessian at the EXACT balanced k=2 saddle (MPS N=4, pattern teacher [3, 0.3]).
Component 1 learned with per-core amplitude 3^(1/4), component 2 zero. Prediction: -s2 s1^((K-2)/K)."""
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from ttn.network import TTNSpec, aligned_init
from ttn.teachers import pattern_teacher
from ttn.tree import make_tree
from ttn.losses import population_loss
from ttn.diagnostics import hessian, gauge_directions, project_out_gauge

spec = TTNSpec.uniform(make_tree("mps", 4), 3, 3)
teacher = pattern_teacher([3.0, 0.3], [[0, 0, 0, 0], [1, 1, 0, 0]], 3, seed=101)
K = spec.n_cores
for imb in [0.0, 0.05]:
    amps = torch.zeros(K, 2)
    amps[:, 0] = 3.0 ** (1 / K)
    if imb:
        amps[0, 0] *= (1 + imb); amps[1, 0] /= (1 + imb)
    theta = spec.flatten(aligned_init(spec, teacher.factors, amps))
    t2 = teacher.norm2()
    f = lambda p: population_loss(spec, spec.unflatten(p), teacher, teacher_norm2=t2)
    H = hessian(f, theta)
    _, ev = project_out_gauge(H, gauge_directions(spec, theta))
    print(f"imbalance {imb}: loss {f(theta).item():.6f}, lowest eigs {ev[:4].tolist()}, pred {-0.3*3**0.5:.5f}")
