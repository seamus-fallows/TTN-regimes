import time, torch
torch.set_num_threads(1)
import sys
if len(sys.argv) > 1: torch.backends.opt_einsum.enabled = False
from ttn.tree import make_tree
from ttn.network import TTNSpec
from ttn.teachers import pattern_teacher
from ttn.losses import population_loss

for kind, n in [("mps", 4), ("binary", 4), ("ht", 4), ("mps", 8), ("binary", 8), ("ht", 8)]:
    tree = make_tree(kind, n)
    spec = TTNSpec.uniform(tree, 2, 2)
    p2 = [0] * n; p2[1] = 1; p2[2] = 1
    teacher = pattern_teacher([1.0, 0.5], [[0] * n, p2], d=2, seed=0)
    t2 = teacher.norm2()
    flat = spec.flatten(spec.init(0.1, seed=0)).clone().requires_grad_(True)
    def grad(x):
        x = x.detach().requires_grad_(True)
        L = population_loss(spec, spec.unflatten(x), teacher, teacher_norm2=t2)
        return torch.autograd.grad(L, x)[0]
    for _ in range(20): grad(flat)
    t0 = time.perf_counter(); m = 300
    for _ in range(m): grad(flat)
    print(kind, n, "K", spec.n_cores, "params", spec.n_params, f"{(time.perf_counter()-t0)/m*1e3:.3f} ms/grad")
