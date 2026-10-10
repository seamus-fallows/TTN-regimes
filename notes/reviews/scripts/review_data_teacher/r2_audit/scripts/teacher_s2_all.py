"""Teacher s_2 (function-space) per student edge for the misaligned8_20 / misaligned8_45 teachers on mps8 and the
edges kept by the f-rule (s2t > 1e-2)."""
import os, sys
os.environ["DATA_TEACHER_RESULTS"] = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/results"
os.environ["DATA_TEACHER_OUT"] = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/raw"
sys.path.insert(0, r"C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher")
import numpy as np, torch
torch.set_num_threads(1)
from analyze_r2_saddle import teacher_s2
for t in ["misaligned8_20", "misaligned8_45", "nonorth8_c0.45", "misaligned_20"]:
    n = 4 if t == "misaligned_20" else 8
    s = teacher_s2(t, n, "mps")
    print(t, np.round(s, 5), "kept edges:", np.nonzero(s > 1e-2)[0].tolist())
