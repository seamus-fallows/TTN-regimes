import math

import pytest
import torch

from ttn.general import inner_general
from ttn.losses import population_loss, sample_inputs
from ttn.network import (
    TTNSpec,
    aligned_init,
    antisymmetric_double,
    bond_spectra,
    conserved_quantities,
    forward,
    inner,
    to_dense,
)
from ttn.teachers import odeco_teacher, pattern_teacher, random_cp_teacher, random_ttn_teacher
from ttn.tree import binary_tree, ht_tree, make_tree, mps_tree

TREES = [("binary", 4), ("binary", 6), ("binary", 8), ("mps", 3), ("mps", 5), ("ht", 4), ("ht", 5)]


def make(kind, n, d=2, chi=3, sigma=1.0, seed=0):
    spec = TTNSpec.uniform(make_tree(kind, n), d, chi)
    return spec, spec.init(sigma, seed=seed)


def matricize(w, below):
    rest = [i for i in range(w.ndim) if i not in below]
    perm = list(below) + rest
    rows = math.prod(w.shape[i] for i in below)
    return w.permute(perm).reshape(rows, -1)


def test_core_counts():
    assert binary_tree(8).n_nodes == 7
    assert mps_tree(8).n_nodes == 8
    assert ht_tree(8).n_nodes == 15
    t = binary_tree(8)
    assert sorted(t.subtree_sites[t.root]) == list(range(8))
    assert t.postorder[-1] == t.root


@pytest.mark.parametrize("kind,n", TREES)
def test_forward_matches_dense(kind, n):
    spec, cores = make(kind, n)
    w = to_dense(spec, cores)
    x = torch.randn(7, n, 2, dtype=torch.float64)
    expected = torch.stack([_contract_dense(w, x[b]) for b in range(7)])
    assert torch.allclose(forward(spec, cores, x), expected, atol=1e-12)


def _contract_dense(w, xb):
    out = w
    for i in range(xb.shape[0]):
        out = torch.tensordot(out, xb[i], dims=([0], [0]))
    return out


@pytest.mark.parametrize("kind,n", TREES)
def test_inner_matches_dense(kind, n):
    spec, a = make(kind, n, seed=1)
    _, b = make(kind, n, seed=2)
    wa, wb = to_dense(spec, a), to_dense(spec, b)
    assert torch.allclose(inner(spec, a, b), (wa * wb).sum(), atol=1e-12)
    metrics = [torch.randn(2, 2, dtype=torch.float64) for _ in range(n)]
    metrics = [m @ m.T + 0.1 * torch.eye(2, dtype=torch.float64) for m in metrics]
    mw = wb
    for i, m in enumerate(metrics):
        mw = torch.movedim(torch.tensordot(mw, m, dims=([i], [1])), -1, i)
    assert torch.allclose(inner(spec, a, b, metrics), (wa * mw).sum(), atol=1e-12)


def test_inner_general_across_trees():
    n = 6
    spec_a, a = make("binary", n, seed=3)
    spec_b, b = make("mps", n, seed=4)
    expected = (to_dense(spec_a, a) * to_dense(spec_b, b)).sum()
    assert torch.allclose(inner_general(spec_a, a, spec_b, b), expected, atol=1e-12)


@pytest.mark.parametrize("kind,n", TREES)
def test_bond_spectra_match_dense_svd(kind, n):
    spec, cores = make(kind, n, chi=3, seed=5)
    w = to_dense(spec, cores)
    spectra = bond_spectra(spec, cores)
    for v in spec.tree.edges:
        sv = torch.linalg.svdvals(matricize(w, spec.tree.subtree_sites[v]))
        k = spectra[v].numel()
        assert torch.allclose(spectra[v], sv[:k], atol=1e-12)
        assert torch.all(sv[k:] < 1e-10)


def test_bond_spectra_with_metrics():
    n, d = 4, 2
    spec, cores = make("binary", n, seed=6)
    metrics = [torch.diag(torch.tensor([2.0, 0.5], dtype=torch.float64)) for _ in range(n)]
    w = to_dense(spec, cores)
    for i, m in enumerate(metrics):
        w = torch.movedim(torch.tensordot(w, m.sqrt(), dims=([i], [1])), -1, i)
    spectra = bond_spectra(spec, cores, metrics)
    for v in spec.tree.edges:
        sv = torch.linalg.svdvals(matricize(w, spec.tree.subtree_sites[v]))
        assert torch.allclose(spectra[v], sv[: spectra[v].numel()], atol=1e-12)


def test_init_norm_in_expectation():
    spec = TTNSpec.uniform(binary_tree(8), d=3, chi=4)
    sigma = 0.7
    vals = [inner(spec, c, c).item() for c in (spec.init(sigma, seed=s) for s in range(400))]
    mean = sum(vals) / len(vals)
    assert abs(mean / sigma ** (2 * spec.n_cores) - 1) < 0.2


def test_odeco_teacher_spectra_are_weights():
    n, d = 6, 3
    weights = [3.0, 2.0, 1.0]
    teacher = odeco_teacher(n, d, weights, seed=0)
    for kind in ("binary", "mps", "ht"):
        spec = TTNSpec.uniform(make_tree(kind, n), d, 3)
        cores = aligned_init(
            spec, teacher.factors, torch.ones(spec.n_cores, 3, dtype=torch.float64)
        )
        # Put the weights on the root so W = W*.
        amps = torch.ones(spec.n_cores, 3, dtype=torch.float64)
        amps[spec.tree.root] = torch.tensor(weights, dtype=torch.float64)
        cores = aligned_init(spec, teacher.factors, amps)
        assert torch.allclose(to_dense(spec, cores), teacher.dense(), atol=1e-12)
        for v, s in bond_spectra(spec, cores).items():
            assert torch.allclose(s[:3], torch.tensor(weights, dtype=torch.float64), atol=1e-12)
        assert population_loss(spec, cores, teacher).abs() < 1e-12


@pytest.mark.parametrize("kind,n", [("binary", 4), ("mps", 4), ("ht", 4)])
def test_population_loss_matches_dense_and_monte_carlo(kind, n):
    d = 2
    spec, cores = make(kind, n, d=d, seed=7)
    teacher = random_cp_teacher(n, d, [1.0, 0.5], seed=1)
    loss = population_loss(spec, cores, teacher)
    dense = 0.5 * (to_dense(spec, cores) - teacher.dense()).pow(2).sum()
    assert torch.allclose(loss, dense, atol=1e-12)
    g = torch.Generator().manual_seed(0)
    x = sample_inputs(200_000, n, d, g)
    mc = 0.5 * (forward(spec, cores, x) - teacher(x)).pow(2).mean()
    assert abs(mc.item() / loss.item() - 1) < 0.05


def test_ttn_teacher_cross_tree_loss():
    n, d = 4, 2
    t_spec = TTNSpec.uniform(mps_tree(n), d, 2)
    teacher = random_ttn_teacher(t_spec, seed=2)
    spec, cores = make("binary", n, d=d, seed=8)
    loss = population_loss(spec, cores, teacher)
    dense = 0.5 * (to_dense(spec, cores) - teacher.dense()).pow(2).sum()
    assert torch.allclose(loss, dense, atol=1e-12)
    assert abs(teacher.norm2().item() - 1.0) < 1e-12


def test_pattern_teacher_bond_ranks():
    # Second component differs from the first only on sites 0, 1.
    n, d = 4, 2
    teacher = pattern_teacher([1.0, 0.5], [[0, 0, 0, 0], [1, 1, 0, 0]], d, seed=0)
    w = teacher.dense()
    rank_01 = torch.linalg.matrix_rank(matricize(w, [0, 1])).item()
    rank_0 = torch.linalg.matrix_rank(matricize(w, [0])).item()
    assert rank_01 == 1 and rank_0 == 2


@pytest.mark.parametrize("kind,n", [("binary", 4), ("mps", 4), ("ht", 4), ("binary", 8)])
def test_gauge_conservation_law(kind, n):
    """dQ_v/dt = 0 under gradient flow: Q(theta - eps grad) - Q(theta) = O(eps^2)."""
    d = 2
    spec, cores = make(kind, n, d=d, sigma=1.0, seed=9)
    teacher = random_cp_teacher(n, d, [1.0, 0.7], seed=3)
    flat = spec.flatten(cores).requires_grad_(True)
    loss = population_loss(spec, spec.unflatten(flat), teacher)
    (grad,) = torch.autograd.grad(loss, flat)
    q0 = conserved_quantities(spec, spec.unflatten(flat.detach()))
    for eps in (1e-4, 1e-5):
        q1 = conserved_quantities(spec, spec.unflatten(flat.detach() - eps * grad))
        drift = max((q1[v] - q0[v]).abs().max().item() for v in q0)
        assert drift < 50 * eps**2 * (1 + grad.norm().item() ** 2)


def test_antisymmetric_double_is_two_copies():
    """With the mask, the doubled TTN's gradient equals the two-copy model's gradient."""
    n, d = 4, 2
    spec, cores = make("binary", n, d=d, chi=2, sigma=1.5, seed=10)
    spec2, cores2, mask = antisymmetric_double(spec, cores)
    assert to_dense(spec2, cores2).abs().max() < 1e-12
    teacher = random_cp_teacher(n, d, [1.0], seed=4)
    flat = spec2.flatten(cores2).requires_grad_(True)
    loss = population_loss(spec2, spec2.unflatten(flat), teacher)
    (grad,) = torch.autograd.grad(loss, flat)
    # Unmasked: the root's mixed blocks do receive gradient (block structure is not preserved).
    g_root = spec2.unflatten(grad)[spec.tree.root]
    assert g_root[:2, 2:].abs().max() > 1e-3
    # Masked gradient on the "+" copy equals the gradient of 1/2||W(p) - W(m) - T||^2 wrt p.
    p = spec.flatten(cores).clone().requires_grad_(True)
    m = spec.flatten(cores).clone()
    from ttn.network import to_dense as dense
    resid = dense(spec, spec.unflatten(p)) - dense(spec, spec.unflatten(m)) - teacher.dense()
    (gp,) = torch.autograd.grad(0.5 * resid.pow(2).sum(), p)
    g_masked = spec2.unflatten(grad * mask)
    for v in range(spec.n_cores):
        idx = [slice(None)] * len(spec.tree.sites[v])
        idx += [slice(0, spec.bond_dims[c]) for c in spec.tree.children[v]]
        idx += [slice(None) if v == spec.tree.root else slice(0, spec.bond_dims[v])]
        assert torch.allclose(g_masked[v][tuple(idx)], spec.unflatten(gp)[v], atol=1e-12)
