"""Throwaway: large-sigma kernel change of the PLAIN (uncentred) model, kappa_plain = delta_K of the
linearized fit of W_0 away (sigma-independent), and its relative parameter change, vs chi."""
import sys
import numpy as np
import torch
sys.path.insert(0, "experiments/regime_map")
from lazy_theory import lazy_predictions
from ttn.network import TTNSpec
from ttn.teachers import TTNTeacher
from ttn.tree import make_tree
torch.set_num_threads(16)
n = int(sys.argv[1]); chis = [int(x) for x in sys.argv[2].split(",")]
probes = torch.randn(128, n, 3, generator=torch.Generator().manual_seed(12345), dtype=torch.float64)
for param in ["sp", "ntk"]:
    for chi in chis:
        spec = TTNSpec.uniform(make_tree("binary", n), 3, chi, cap=False)
        eta = None
        if param == "ntk":
            eta = torch.cat([torch.full((sz,), 1.0 / f, dtype=torch.float64) for sz, f in zip(spec.sizes, spec.fan_in)])
        ks, dth = [], []
        for seed in range(5):
            c1 = spec.init(1.0, seed=seed)
            p = lazy_predictions(spec, c1, TTNTeacher(spec, c1), probes, mask=eta)
            ks.append(p["kappa"]); dth.append(p["rel_dtheta1"])
        ks, dth = np.array(ks), np.array(dth)
        pred = np.sqrt(3**n / (sum(spec.fan_in) * sum(spec.bond_dims[v] for v in spec.tree.edges)))
        print(f"{param} N={n} chi={chi} kappa_plain={ks.mean():.4g}+-{ks.std()/np.sqrt(5):.2g} "
              f"rel_dtheta={dth.mean():.3g} (isotropic estimate {pred:.3g})", flush=True)
