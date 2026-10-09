import time, torch
torch.set_num_threads(1)
torch.backends.opt_einsum.enabled = False
from ttn.tree import make_tree
from ttn.network import TTNSpec
from ttn.teachers import pattern_teacher
from ttn.losses import population_loss

for kind, n in [("mps", 4), ("ht", 4), ("mps", 8), ("ht", 8)]:
    tree = make_tree(kind, n)
    spec = TTNSpec.uniform(tree, 2, 2)
    p2 = [0] * n; p2[1] = 1; p2[2] = 1
    teacher = pattern_teacher([1.0, 0.5], [[0] * n, p2], d=2, seed=0)
    t2 = teacher.norm2()
    flat = spec.flatten(spec.init(0.1, seed=0))
    def loss(x):
        return population_loss(spec, spec.unflatten(x), teacher, teacher_norm2=t2)
    g = torch.func.grad(loss)
    t0 = time.perf_counter()
    gc = torch.compile(g, fullgraph=True, dynamic=False)
    gc(flat); tc = time.perf_counter() - t0
    for _ in range(20): gc(flat)
    t0 = time.perf_counter(); m = 1000
    for _ in range(m): gc(flat)
    tg = (time.perf_counter() - t0) / m
    print(kind, n, f"compile {tc:.1f}s, {tg*1e6:.0f} us/grad", float((gc(flat) - g(flat)).abs().max()))
