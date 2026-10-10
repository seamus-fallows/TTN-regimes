"""Mechanism check (light compute): s2s_core MPS4, chi = 3, odeco (3, 2, 1). Gradient flow from the
committed init up to the second escape (a_q2 = 0.99), then the phase-2 GN quantity C' at both chain
ends from the actual state (learned bond directions = top-2 right singular vectors of each child core,
learned physical patterns u_q1, u_q2). Prediction: the last component dies iff C' < 0 at both ends.

usage: PYTHONPATH=. python experiments/theory_checks/trap/gf_check_s2s.py [sigma] [teacher] > out.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from predict_s2s import block2  # noqa: E402
from trapcrit import contract, neighbours  # noqa: E402
from ttn.dynamics import Problem  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import odeco_teacher  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

torch.set_num_threads(1)
WEIGHTS = {"separated": [3.0, 2.0, 1.0], "close": [1.0, 0.85, 0.7]}


def learned_bases(spec, cores, n_learned=2):
    out = {}
    for c in spec.tree.edges:
        g = cores[c].reshape(-1, cores[c].shape[-1])
        _, _, vh = torch.linalg.svd(g, full_matrices=True)
        out[c] = (vh[:n_learned].T, vh[n_learned:].T)
    return out


def run(seed, sigma, teacher_name):
    spec = TTNSpec.uniform(make_tree("mps", 4), 4, 3)
    teacher = odeco_teacher(4, 4, WEIGHTS[teacher_name], seed=1000)
    prob = Problem(spec, teacher)
    th0 = spec.flatten(spec.init(sigma, seed=seed)).numpy()

    def f(_, y):
        return prob.velocity(torch.from_numpy(y)).numpy()

    def jac(_, y):
        return -prob.hessian(torch.from_numpy(y)).numpy()

    def ov(y):
        return teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y))).numpy()

    def second(_, y):
        a = np.sort(ov(y))
        return a[-2] - 0.99

    second.terminal = True
    sol = solve_ivp(
        f, (0, 1e12), th0, method="LSODA", jac=jac, rtol=1e-9, atol=1e-12 * sigma, events=second
    )
    y = sol.y[:, -1]
    a = ov(y)
    wts = WEIGHTS[teacher_name]
    # SVD order = learned amplitude order = weight order (balanced learned cores, a_r = s_r^(1/K))
    learned = sorted([int(i) for i in np.argsort(-a)[:2]], key=lambda i: -wts[i])
    cores = spec.unflatten(torch.from_numpy(y.copy()))
    bs = learned_bases(spec, cores)
    tree = spec.tree
    C = []
    for u in (0, 3):
        ((w, e),) = neighbours(tree, u)
        X = block2(spec, teacher, cores, bs, learned, u, e)
        Y = block2(spec, teacher, cores, bs, learned, w, e)
        C.append(float(torch.linalg.eigvalsh(X.T @ X - Y.T @ Y).max()) / sigma**2)
    r = [i for i in range(3) if i not in learned][0]
    red = reduced_last(spec, teacher, cores, bs, learned, r, wts)
    return sol.t[-1], learned, a, C, red


def reduced_last(spec, teacher, cores, bs, learned, r, wts):
    """Reduced route model (theory_trap.md 2.5) for the last component on MPS4 at the actual state:
    route = all cores, ends 0 and 3; decay of an end weighted by w_alpha = s_alpha^(2(K-2)/K) for the
    partner's normal entries with learned index alpha on its far leg; drive c = s_r."""
    tree, K = spec.tree, spec.n_cores
    b0 = []
    for v in range(K):
        ops = {("site", v): teacher.factors[r, v]}
        for _, f in neighbours(tree, v):
            ops[("edge", f)] = bs[f][1][:, 0]
        b0.append(float(contract(cores[v], tree, v, ops)))
    wa = [wts[g] ** (2 * (K - 2) / K) for g in learned]
    rest, ys = [], []
    for u in (0, 3):
        ((w, e),) = neighbours(tree, u)
        X = block2(spec, teacher, cores, bs, learned, u, e)
        Y = block2(spec, teacher, cores, bs, learned, w, e)
        rest.append(max(float(X.pow(2).sum()) - b0[u] ** 2, 0.0))
        y2 = Y.pow(2).sum(dim=1).reshape(-1, 2).sum(dim=0)  # rows ordered phys-major, learned index last
        ys += [float(y2[0]), float(y2[1])]
    z0 = np.array(b0 + rest + ys)
    s_r = wts[r]

    def rhs(_, z):
        b, re, y = z[:4], z[4:6], z[6:].reshape(2, 2)
        db = np.array([s_r * np.prod(np.delete(b, i)) for i in range(4)])
        dre, dy = np.zeros(2), np.zeros((2, 2))
        for j, u in enumerate((0, 3)):
            gam = wa[0] * y[j, 0] + wa[1] * y[j, 1]
            db[u] -= gam * b[u]
            dre[j] = -2 * gam * re[j]
            dy[j] = -2 * np.array(wa) * (b[u] ** 2 + re[j]) * y[j]
        return np.concatenate([db, dre, dy.ravel()])

    def big(_, z):
        return np.abs(z[:4]).max() - 100 * np.abs(z0[:4]).max()

    def dead(_, z):
        return max(abs(z[0]) / abs(z0[0]), abs(z[3]) / abs(z0[3])) - 1e-12

    big.terminal = dead.terminal = True
    scale = np.abs(z0[:4]).max() ** 2
    sol = solve_ivp(rhs, (0, 1e5 / scale), z0, method="LSODA", rtol=1e-9, atol=1e-18, events=[big, dead])
    if sol.status == 1:
        return "escape" if sol.t_events[0].size else "dead"
    return "pinned"


def run_tree(kind, n, seed, sigma, teacher_name, chi=3):
    """Any tree: gradient flow to the second escape, then the phase-2 sign test at every leaf end
    (the last component's route spans the tree). Returns (t2, learned, C' per leaf / sigma^2)."""
    spec = TTNSpec.uniform(make_tree(kind, n), 4, chi)
    teacher = odeco_teacher(n, 4, WEIGHTS[teacher_name], seed=1000)
    prob = Problem(spec, teacher)
    th0 = spec.flatten(spec.init(sigma, seed=seed)).numpy()

    def ov(y):
        return teacher.component_overlaps(spec, spec.unflatten(torch.from_numpy(y))).numpy()

    def second(_, y):
        return np.sort(ov(y))[-2] - 0.99

    second.terminal = True
    sol = solve_ivp(lambda t, y: prob.velocity(torch.from_numpy(y)).numpy(), (0, 1e14), th0,
                    method="LSODA", jac=lambda t, y: -prob.hessian(torch.from_numpy(y)).numpy(),
                    rtol=1e-8, atol=1e-11 * sigma, events=second)
    y = sol.y[:, -1]
    a = ov(y)
    wts = WEIGHTS[teacher_name]
    learned = sorted([int(i) for i in np.argsort(-a)[:2]], key=lambda i: -wts[i])
    cores = spec.unflatten(torch.from_numpy(y.copy()))
    bs = learned_bases(spec, cores)
    tree = spec.tree
    C = {}
    for u in range(tree.n_nodes):
        nb = neighbours(tree, u)
        if len(nb) != 1:
            continue
        ((w, e),) = nb
        X = block2(spec, teacher, cores, bs, learned, u, e)
        Y = block2(spec, teacher, cores, bs, learned, w, e)
        C[u] = round(float(torch.linalg.eigvalsh(X.T @ X - Y.T @ Y).max()) / sigma**2, 4)
    return sol.t[-1], learned, C


if __name__ == "__main__":
    # gf_check_s2s.py [sigma] [teacher]              MPS4 chi = 3, seeds 0-9, sign test + reduced model
    # gf_check_s2s.py tree KIND N SIGMA SEEDS        any tree (separated teacher), sign test at the leaves
    if len(sys.argv) > 1 and sys.argv[1] == "tree":
        kind, n, sigma = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
        for seed in [int(x) for x in sys.argv[5].split(",")]:
            t2, learned, C = run_tree(kind, n, seed, sigma, "separated")
            nd = sum(c <= 0 for c in C.values())
            verdict = "dead" if nd >= 2 else "pinned" if nd else "alive"
            print(f"{kind}{n} seed {seed} sigma {sigma} t2={t2:.4g} learned={learned} C'={C} "
                  f"dying={nd} pred={verdict}", flush=True)
    else:
        sigma = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
        tname = sys.argv[2] if len(sys.argv) > 2 else "separated"
        for seed in range(10):
            t2, learned, a, C, red = run(seed, sigma, tname)
            dead = all(c < 0 for c in C)
            print(f"seed {seed} t2={t2:.4g} learned={learned} C'/sigma^2={[round(c, 4) for c in C]} "
                  f"sign={'dead' if dead else 'alive'} reduced={red}", flush=True)
