"""Independent recount of negative Hessian eigenvalues at the stored exact rank-1 saddles."""
import sys, json
sys.path.insert(0, "experiments/data_teacher")
import numpy as np, torch, yaml
torch.set_num_threads(1)
from r2_common import teacher_for, student
from ttn.network import to_dense
stored = torch.load("experiments/data_teacher/results/r2_saddle_cores.pt", weights_only=False)
cfg = {}
for f in ("r2_saddle.yaml", "r2_saddle_b.yaml"):
    c = yaml.safe_load(open("experiments/data_teacher/configs/" + f))
    for n, block in c["teachers"].items():
        for t, tc in block.items():
            cfg[t] = (int(n), tc)
combos = sys.argv[1:]
for combo in combos:
    tname, tree = combo.split("|")
    n, tc = cfg[tname]
    teacher, metrics = teacher_for(tc["kind"], n, **tc.get("args", {}))
    spec = student(tree, n)
    cores = [c.clone() for c in stored[combo]["cores"]]
    Wt = teacher.dense().detach()
    shapes = [c.shape for c in cores]
    sizes = [c.numel() for c in cores]
    def unflat(th):
        out, i = [], 0
        for s, k in zip(shapes, sizes):
            out.append(th[i:i + k].reshape(s)); i += k
        return out
    def loss(th):
        D = to_dense(spec, unflat(th)) - Wt
        if metrics is not None:
            X = D
            for i, S in enumerate(metrics):  # apply Sigma_i on site i
                X = torch.movedim(torch.tensordot(S, torch.movedim(X, i, 0), dims=([1], [0])), 0, i)
            return 0.5 * (D * X).sum()
        return 0.5 * (D * D).sum()
    th = torch.cat([c.reshape(-1) for c in cores]).double()
    g = torch.autograd.functional.jacobian(loss, th)
    H = torch.autograd.functional.hessian(loss, th)
    ev = torch.linalg.eigvalsh(H)
    nneg = int((ev < -1e-6).sum())
    # per-edge blocks
    offs = np.cumsum([0] + sizes)
    gam = []
    for v in spec.tree.edges:
        u = spec.tree.parent[v]
        idx = torch.cat([torch.arange(offs[v], offs[v + 1]), torch.arange(offs[u], offs[u + 1])])
        gam.append(-float(torch.linalg.eigvalsh(H[idx][:, idx])[0]))
    print(f"{combo}: loss {float(loss(th)):.10f} (stored {stored[combo]['loss']:.10f}) |grad| {float(g.norm()):.1e} "
          f"lambda_min {float(ev[0]):.10f} n_neg(<-1e-6) {nneg} n(<-1e-8) {int((ev < -1e-8).sum())} "
          f"next eig above negatives {float(ev[nneg]):.2e} edge 1/gamma {[round(1/x,3) if x>1e-6 else None for x in gam]}")
