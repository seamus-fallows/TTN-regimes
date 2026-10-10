"""Own raw Hessian at saved plateau midpoints of binary4 close-teacher runs (wrong-order plateaus)."""
import json, sys
import numpy as np, torch, pandas as pd
torch.set_num_threads(1)
sys.path.insert(0, 'experiments/s2s_core'); sys.path.insert(0, '.')
import s2s
p = pd.read_csv('experiments/s2s_core/results/main_plateaus.csv')
for chi, seed, sigma in [(3, 8, 0.005), (3, 8, 0.01), (3, 8, 0.1), (3, 5, 0.07), (3, 5, 0.1), (6, 4, 0.03)]:
    f = f'data/s2s_core/main/binary4_chi{chi}_w1-0.85-0.7_fanin_s{sigma:g}_seed{seed}'
    s = json.load(open(f + '.json')); d = np.load(f + '.npz')
    spec, tch = s2s.setup('binary', 4, 4, chi, [1.0, 0.85, 0.7], 1000)
    U, w = tch.factors, tch.weights
    Wstar = sum(w[r] * torch.einsum('i,j,k,l->ijkl', U[r, 0], U[r, 1], U[r, 2], U[r, 3]) for r in range(3))
    n0 = 16 * chi
    def L(x):
        A = x[:n0].reshape(4, 4, chi); B = x[n0:2 * n0].reshape(4, 4, chi); M = x[2 * n0:].reshape(chi, chi)
        W = torch.einsum('ija,ab,klb->ijkl', A, M, B)
        return 0.5 * ((W - Wstar) ** 2).sum()
    mids = d['mid_params']
    rows = p[(p.tree == 'binary4') & (p.chi == chi) & (p.teacher == 'close') & (p.seed == seed) & (p.sigma == sigma) & (p.sweep == 'main')]
    for i in range(mids.shape[0]):
        # check layout: library flattens cores in order; core shapes (4,4,chi),(4,4,chi),(chi,chi,1)
        x = torch.tensor(mids[i])
        H = torch.autograd.functional.hessian(L, x)
        ev = torch.linalg.eigvalsh(H).numpy()
        r = rows[rows.plateau == i + 1]
        print(chi, seed, sigma, 'plateau', i + 1, 'learned', r.learned.values, 'own raw lam_min', round(ev[0], 6),
              'repo raw', np.round(r.lam_min_raw.values, 6), 'repo proj', np.round(r.lam_min.values, 6), 'loss err', r.loss_rel_err.values)
