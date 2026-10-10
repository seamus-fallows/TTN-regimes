"""Review check: H1b scope for m = 2 pattern pairs on MPS (K >= 4): adjacent (k = 2) vs separated
(k >= 3) differing cores. Gauge-projected Hessian at the exact wrong saddle (agent's saddle_init with
sigma = 0) and GF from the saddle + perturbation: rotation (a_2 -> 0) or addition (a_2 stays ~1)?"""
import os, sys, json
os.environ.setdefault("ORDER_RES", "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/claims_audit")
os.environ.setdefault("ORDER_RAW", "c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/claims_audit")
sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/order")
import numpy as np, torch
torch.set_num_threads(1)
from common import make_spec, make_teacher, projected_hessian, gf_trajectory, overlaps, single_thread
from h1b import saddle_init

single_thread()
cases = [("mps", 4, [[0, 0, 0, 0], [0, 1, 1, 0]], "adjacent cores 1,2 (k=2)"),
         ("mps", 4, [[0, 0, 0, 0], [1, 0, 1, 0]], "separated cores 0,2 (k=3)"),
         ("mps", 6, [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1]], "ends 0,5 (k=6)"),
         ("mps", 6, [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], "adjacent cores 2,3 (k=2)"),
         ("binary", 4, None, "binary4 odeco (k=3)")]
out = []
rho = 0.8
for kind, n, pat, label in cases:
    spec = make_spec(kind, n, 2, 4)
    K = spec.n_cores
    teacher = make_teacher(n, 2, [1.0, rho], pat)
    y_b = saddle_init(spec, teacher, rho, 0.0, 0)
    ev, ng, raw = projected_hessian(spec, teacher, y_b)
    q = rho ** (1 / K)
    rec = dict(case=label, K=K, lam_min=float(ev[0]), lam_2=float(ev[1]), lam_3=float(ev[2]), lam_4=float(ev[3]),
               lam_min_raw=float(raw[0]), pred_rotation=-(1 - rho) * q ** (K - 2), pred_add_k2=-1.0 * q ** (K - 2),
               a_at_B=overlaps(spec, teacher, y_b).tolist())
    # GF from the saddle + perturbation 1e-4 (seed 0..2): rotation or addition?
    gf = []
    for seed in range(3):
        y0 = saddle_init(spec, teacher, rho, 1e-4, seed)
        sol = gf_trajectory(spec, teacher, y0, t_max=1e7)
        a = np.array([overlaps(spec, teacher, y) for y in sol.y.T])
        i1 = np.argmax(a[:, 0] >= 0.5) if np.any(a[:, 0] >= 0.5) else None
        gf.append(dict(seed=seed, t1=float(sol.t[i1]) if i1 is not None else None,
                       a2_min_after_t1=float(a[i1:, 1].min()) if i1 is not None else None,
                       final_a=a[-1].tolist(), t_end=float(sol.t[-1])))
    rec["gf"] = gf
    out.append(rec)
    print(json.dumps(rec), flush=True)
with open("c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/claims_audit_h1b_scope.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
