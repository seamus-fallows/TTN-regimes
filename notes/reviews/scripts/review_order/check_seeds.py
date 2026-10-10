import torch, numpy as np, sys
sys.path.insert(0, 'experiments/order')
torch.set_num_threads(1)
from ttn.network import TTNSpec
from ttn.tree import make_tree
for kind, n in [("mps", 4), ("mps", 6)]:
    s2 = TTNSpec.uniform(make_tree(kind, n), 2, 2, cap=True)
    s4 = TTNSpec.uniform(make_tree(kind, n), 4, 2, cap=True)
    print(kind, n, "shapes d2", s2.shapes, "d4", s4.shapes)
    print(" sites", make_tree(kind, n).sites)
    for seed in range(3):
        c2 = s2.init(1.0, seed=seed); c4 = s4.init(1.0, seed=seed)
        for v,(a,b) in enumerate(zip(c2,c4)):
            # physical leg first? compare slice 0,1 normalized
            fa = a.flatten()[:4]; fb = b.flatten()[:4]
            ratio = (fb/fa).numpy()
            print(f"  seed {seed} core {v}: first4 d2 {np.round(fa.numpy(),3)} d4 {np.round(fb.numpy(),3)}")
        break
