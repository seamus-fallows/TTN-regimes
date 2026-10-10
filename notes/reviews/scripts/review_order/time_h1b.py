import sys, time
sys.argv += ["--raw", "data/review_order/raw", "--res", "data/review_order/res"]
sys.path.insert(0, "experiments/order")
import torch
torch.set_num_threads(1)
import numpy as np
from common import single_thread, make_spec, make_teacher, unit_init, gf_trajectory, gf_events, lo_blowup, single_component
single_thread()
import h1b
spec = make_spec("binary", 4, 2, 4)
for rho, sigma in [(0.8, 1e-6), (0.8, 1e-2)]:
    teacher = make_teacher(4, 2, [1.0, rho])
    t0 = time.time()
    y0 = h1b.saddle_init(spec, teacher, rho, sigma, 0)
    sol = gf_trajectory(spec, teacher, y0, t_max=1e9)
    t_int = time.time() - t0
    rec, traj = h1b.analyse(spec, teacher, sol, rho)
    print(rho, sigma, f"integrate {t_int:.1f}s total {time.time()-t0:.1f}s", {k: rec.get(k) for k in ["t1","t2","lam_min","lam_pred","a2_reup","final_loss","t_end"]}, flush=True)
# multi timing: mps4 R=3 d=4 chi=4 sigma 0.1
spec = make_spec("mps", 4, 4, 4)
for R in [3, 4]:
    teacher = make_teacher(4, 4, [0.8**r for r in range(R)])
    u0 = unit_init(spec, 0)
    t0 = time.time()
    out = gf_events(spec, teacher, 0.1*u0, 1e9)
    print("multi mps4 R", R, f"{time.time()-t0:.1f}s", out[0], out[4], out[5], flush=True)
