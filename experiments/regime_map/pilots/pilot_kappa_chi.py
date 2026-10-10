"""Throwaway: kappa (centred lazy kernel-change coefficient) vs chi for binary N=4 under SP and
NTK-type learning rates (eta_v = 1/F_v, normalized to mean 1 over cores)."""
import sys
import numpy as np
import torch
sys.path.insert(0, "experiments/regime_map")
from lazy_theory import lazy_predictions
from ttn.network import TTNSpec
from ttn.teachers import odeco_teacher
from ttn.tree import make_tree
torch.set_num_threads(16)
n = int(sys.argv[1]); chis = [int(x) for x in sys.argv[2].split(",")]
probes = torch.randn(128, n, 3, generator=torch.Generator().manual_seed(12345), dtype=torch.float64)
for param in ["sp", "ntk"]:
    for chi in chis:
        spec = TTNSpec.uniform(make_tree("binary", n), 3, chi, cap=False)
        eta = None
        if param == "ntk":
            per_core = [1.0 / f for f in spec.fan_in]
            eta = torch.cat([torch.full((sz,), e, dtype=torch.float64) for sz, e in zip(spec.sizes, per_core)])
        ks = []
        for seed in range(5):
            p = lazy_predictions(spec, spec.init(1.0, seed=seed), odeco_teacher(n, 3, [3., 2., 1.], seed=1000 + seed), probes, mask=eta)
            ks.append(p["kappa"])
        ks = np.array(ks)
        print(f"{param} N={n} chi={chi} kappa={ks.mean():.4g} +- {ks.std()/np.sqrt(5):.2g}", flush=True)
