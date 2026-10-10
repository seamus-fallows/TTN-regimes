import sys
from pathlib import Path
sys.path.insert(0, "experiments/data_teacher")
import numpy as np, torch
torch.set_num_threads(1)
from setups import build
base = dict(kind="ttn_teacher", n_sites=8, d=2, chi=4, decay=0.6, pair="mps_on_binary", tree="binary")
def s2_cut(T, left):
    n = T.ndim
    right = [i for i in range(n) if i not in left]
    M = np.transpose(T, left + right).reshape(2 ** len(left), -1)
    s = np.linalg.svd(M, compute_uv=False)
    return s[1] / np.linalg.norm(s)  # normalised teacher (||W*|| = 1 in the config anyway)
for ts in (201, 202, 203, 204, 209, 212, 218, 491, 548, 643, 694, 846):
    spec, teacher = build(dict(base, teacher_seed=ts), 0)[:2]
    T = teacher.dense().detach().numpy() if hasattr(teacher.dense(), "detach") else np.asarray(teacher.dense())
    b = {c: s2_cut(T, list(c)) for c in ((0, 1), (2, 3), (4, 5), (6, 7))}
    r = s2_cut(T, [0, 1, 2, 3])
    halfL, halfR = max(b[(0, 1)], b[(2, 3)]), max(b[(4, 5)], b[(6, 7)])
    disc = r > min(halfL, halfR)
    above_all = r > max(halfL, halfR)
    print(f"{ts}: root {r:.4f} bottoms {[round(v,4) for v in b.values()]} disc={disc} above_all={above_all} "
          f"root/max_all={r/max(halfL,halfR):.4f} root/weaker-half-max={r/min(halfL,halfR):.4f} norm={np.linalg.norm(T):.4f}")
