"""One saddle-start run of the 'degenerate' misaligned8_20 teacher on mps8 (agent's run()), then a look at
what limits the escape: per-edge s_2 (Sigma-weighted) vs the f-rule thresholds, and the loss."""
import os, sys, json, time
os.environ["DATA_TEACHER_RESULTS"] = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/results"
os.environ["DATA_TEACHER_OUT"] = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/raw"
sys.path.insert(0, r"C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher")
import numpy as np, torch, yaml
torch.set_num_threads(1)
import r2_saddle_runs as R
from analyze_r2_saddle import teacher_s2
from common import OUT_ROOT

cfg = yaml.safe_load(open(r"C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher/configs/r2_saddle.yaml"))
tname, n = "misaligned8_20", 8
eps, seed = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01, 0
R.T_CAP = 2e6
t0 = time.time()
summ = R.run(("audit_r2_saddle", tname, cfg["teachers"][n][tname], n, "mps", eps, seed))
print("summary", summ, "wall", round(time.time() - t0, 1), flush=True)
d = np.load(OUT_ROOT / "audit_r2_saddle" / f"{tname}_mps_s{eps}_seed{seed}.npz")
t, loss, sp = d["t"], d["loss"], d["spectra"]
s2t = teacher_s2(tname, n, "mps")
print("teacher s2 per edge", np.round(s2t, 5))
print("f=0.05 thresholds", np.round(0.05 * s2t, 6), " start s2", np.round(sp[0, :, 1], 9))
idx = np.unique(np.r_[np.searchsorted(t, np.geomspace(max(t[1], 1e-2), t[-1], 40)), len(t) - 1].clip(0, len(t) - 1))
for k in idx:
    print(f"t={t[k]:10.4g} L={loss[k]:.6g} s2=" + " ".join(f"{x:.2e}" for x in sp[k, :, 1]) + "  s3max=%.1e" % sp[k, :, 2].max())
