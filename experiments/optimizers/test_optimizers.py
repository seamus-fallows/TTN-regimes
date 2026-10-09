"""Checks of the experiment-specific gauge code (run: uv run pytest experiments/optimizers)."""

import torch

from experiments.optimizers.gauge import canonicalize, env_grams, euler_tour, hsvd, move_center
from ttn.network import TTNSpec, to_dense
from ttn.tree import binary_tree, mps_tree


def _specs():
    return [TTNSpec.uniform(binary_tree(4), 3, 4), TTNSpec.uniform(binary_tree(8), 2, 3),
            TTNSpec.uniform(mps_tree(5), 3, 4)]


def test_canonicalize_keeps_w_and_isometries():
    for spec in _specs():
        cores = spec.init(0.7, seed=3)
        w = to_dense(spec, cores)
        for center in range(spec.n_cores):
            can = canonicalize(spec, cores, center)
            assert torch.allclose(to_dense(spec, can), w, atol=1e-13)
            t = spec.tree
            for v in range(spec.n_cores):
                if v == center:
                    continue
                u = t.path(v, center)[1]
                from experiments.optimizers.gauge import leg_to

                g = torch.movedim(can[v], leg_to(spec, v, u), -1)
                g = g.reshape(-1, g.shape[-1])
                assert torch.allclose(g.T @ g, torch.eye(g.shape[1], dtype=g.dtype), atol=1e-12)
            assert torch.allclose(can[center].norm(), w.norm(), atol=1e-12)


def test_env_grams_match_dense_hessian():
    spec = TTNSpec.uniform(binary_tree(4), 2, 3)
    cores = spec.init(1.0, seed=1)
    below, above = env_grams(spec, cores)
    for v in range(spec.n_cores):
        x = cores[v].clone().requires_grad_(True)

        def w_of(g):
            cs = list(cores)
            cs[v] = g
            return to_dense(spec, cs).reshape(-1)

        jac = torch.autograd.functional.jacobian(w_of, x).reshape(-1, x.numel())
        h = jac.T @ jac
        t = spec.tree
        mats = [torch.eye(spec.phys_dims[s], dtype=torch.float64) for s in t.sites[v]]
        mats += [below[c] for c in t.children[v]]
        mats += [above[v] if v != t.root else torch.ones(1, 1, dtype=torch.float64)]
        k = mats[0]
        for m in mats[1:]:
            k = torch.kron(k, m)
        assert torch.allclose(h, k, atol=1e-12)


def test_hsvd_exact_on_format_and_tour():
    for spec in _specs():
        cores = spec.init(1.0, seed=2)
        w = to_dense(spec, cores)
        assert torch.allclose(to_dense(spec, hsvd(spec, w)), w, atol=1e-12)
        tour = euler_tour(spec)
        assert len(tour) == 2 * spec.n_cores - 1
        c2 = canonicalize(spec, cores, tour[0])
        for a, b in zip(tour[:-1], tour[1:]):
            c2 = move_center(spec, c2, a, b)
        assert torch.allclose(to_dense(spec, c2), w, atol=1e-12)


def test_balance_keeps_w_and_zeroes_q():
    from experiments.optimizers.gauge import balance
    from ttn.network import conserved_quantities

    for spec in _specs():
        cores = spec.init(0.3, seed=4)
        w = to_dense(spec, cores)
        bal = balance(spec, cores)
        assert torch.allclose(to_dense(spec, bal), w, atol=1e-12 * w.norm())
        q = conserved_quantities(spec, bal)
        scale = max(c.norm().item() ** 2 for c in bal)
        assert max(qq.abs().max().item() for qq in q.values()) < 1e-10 * scale
