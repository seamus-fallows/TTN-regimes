import sys, time
sys.argv += ["--raw", "data/review_order/raw", "--res", "data/review_order/res"]
sys.path.insert(0, "experiments/order")
import torch
torch.set_num_threads(1)
from common import single_thread, make_spec, make_teacher, unit_init, gf_first_event, lo_blowup, single_component
single_thread()
for kind, n, d, chi, sigma in [("mps",4,2,2,0.01),("mps",4,2,4,0.01),("mps",4,2,4,0.003),("binary",8,2,2,0.01),("binary",8,2,4,0.01),("binary",8,2,4,0.1),("binary",4,2,2,1e-4),("binary",4,2,2,1e-2),("mps",6,2,2,0.01)]:
    spec = make_spec(kind, n, d, chi)
    t0 = time.time()
    for seed in range(3):
        u0 = unit_init(spec, seed)
        teacher = make_teacher(n, d, [1.0, 0.8])
        t1, a, nfev, st = gf_first_event(spec, teacher, sigma*u0)
    t_gf = (time.time()-t0)/3
    t0 = time.time()
    base = make_teacher(n, d, [1.0, 1.0])
    for seed in range(3):
        u0 = unit_init(spec, seed)
        taus = [lo_blowup(spec, single_component(base, r), u0)[0] for r in range(2)]
    t_lo = (time.time()-t0)/3
    print(kind, n, d, chi, sigma, f"GF {t_gf:.2f}s/run nfev {nfev} t1 {t1:.4g}; LO race {t_lo:.2f}s/seed", flush=True)
