import pytest
import torch

from ttn.losses import centred_population_loss, population_loss
from ttn.network import (
    TTNSpec,
    aligned_init,
    antisymmetric_double,
    conserved_quantities,
    inner,
    sum_difference_cores,
    to_dense,
)
from ttn.teachers import ShiftedTeacher, odeco_teacher, random_cp_teacher, random_ttn_teacher
from ttn.tree import Tree, binary_tree, mps_tree


def test_binary_tree_has_n_minus_1_cores():
    for n in range(2, 17):
        t = binary_tree(n)
        assert t.n_nodes == n - 1
        assert all(len(t.sites[v]) + len(t.children[v]) == 2 for v in range(t.n_nodes))


def test_tree_equality_is_structural_and_accepts_lists():
    t = mps_tree(4)
    t2 = Tree(list(t.parent), [list(s) for s in t.sites])
    assert t2 == t and hash(t2) == hash(t)
    s, s2 = TTNSpec.uniform(t, 2, 2), TTNSpec.uniform(t2, 2, 2)
    inner(s, s.init(1.0, seed=1), s2.init(1.0, seed=2), spec_b=s2)  # no ValueError


def test_teacher_and_init_streams_are_independent():
    spec = TTNSpec.uniform(mps_tree(6), 2, 4)
    w0, ws = to_dense(spec, spec.init(1e-2)), random_ttn_teacher(spec).dense()
    assert abs((w0 * ws).sum() / (w0.norm() * ws.norm())) < 0.5
    u = odeco_teacher(4, 2, [1.0, 0.5], seed=0).factors[:, 0, :]
    g0 = TTNSpec.uniform(mps_tree(4), 2, 2).init(1.0, seed=0)[0]
    assert (u @ g0)[1, 0].abs() > 1e-8  # first core not triangular in the teacher basis


@pytest.mark.parametrize("tree", [binary_tree(4), mps_tree(4), binary_tree(5)])
def test_two_copy_sum_difference_form(tree):
    spec = TTNSpec.uniform(tree, 2, 2)
    teacher = random_cp_teacher(tree.n_sites, 2, [1.0, 0.5], seed=1)
    spec2, cores2, mask = antisymmetric_double(spec, spec.init(1.0, seed=2))
    g = torch.Generator().manual_seed(0)
    cores2 = [
        c + m * 0.3 * torch.randn(c.shape, generator=g, dtype=torch.float64)
        for c, m in zip(cores2, spec2.unflatten(mask))
    ]
    flat = spec2.flatten(cores2).requires_grad_(True)
    (g1,) = torch.autograd.grad(population_loss(spec2, spec2.unflatten(flat), teacher), flat)
    flat2 = spec2.flatten(cores2).requires_grad_(True)
    loss2 = population_loss(spec2, sum_difference_cores(spec2, spec2.unflatten(flat2)), teacher)
    (g2,) = torch.autograd.grad(loss2, flat2)
    assert torch.allclose(g1 * mask, g2, atol=1e-12)


def test_two_copy_loss_accurate_at_large_sigma():
    tree = binary_tree(8)
    spec = TTNSpec.uniform(tree, 2, 2)
    teacher = random_cp_teacher(8, 2, [1.0, 0.5], seed=1)
    g = torch.Generator().manual_seed(0)
    plus = spec.init(10.0, seed=2)
    minus = [c + 1e-8 * torch.randn(c.shape, generator=g, dtype=torch.float64) for c in plus]
    spec2, cores2, _ = antisymmetric_double(spec, plus)
    t = tree
    for v in range(spec.n_cores):  # overwrite the "-" blocks with the minus copy
        idx = [slice(None)] * len(t.sites[v])
        idx += [slice(spec.bond_dims[c], 2 * spec.bond_dims[c]) for c in t.children[v]]
        idx += [slice(None)] if v == t.root else [slice(spec.bond_dims[v], 2 * spec.bond_dims[v])]
        cores2[v][tuple(idx)] = -minus[v] if v == t.root else minus[v]
    resid = to_dense(spec, plus) - to_dense(spec, minus) - teacher.dense()  # error ~1e-9
    exact = 0.5 * resid.pow(2).sum()
    stable = population_loss(spec2, sum_difference_cores(spec2, cores2), teacher)
    assert abs(stable - exact) < 1e-7


def test_conservation_with_lr_multipliers():
    spec = TTNSpec.uniform(binary_tree(4), 2, 2)
    teacher = random_cp_teacher(4, 2, [1.0, 0.7], seed=3)
    eta = [0.5, 2.0, 1.5]
    cores = [c.requires_grad_(True) for c in spec.init(1.0, seed=9)]
    grads = torch.autograd.grad(population_loss(spec, cores, teacher), cores)
    h = 1e-4
    with torch.no_grad():
        qp = conserved_quantities(spec, [c - h * e * g for c, e, g in zip(cores, eta, grads)], eta)
        qm = conserved_quantities(spec, [c + h * e * g for c, e, g in zip(cores, eta, grads)], eta)
    assert max((qp[v] - qm[v]).abs().max().item() for v in qp) < 1e-10


def test_argument_validation():
    spec = TTNSpec.uniform(binary_tree(4), 2, 2)
    with pytest.raises(ValueError):
        spec.init(1.0, core_scales=[1.0, 1.0])
    with pytest.raises(ValueError):
        conserved_quantities(spec, spec.init(1.0), [1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        antisymmetric_double(TTNSpec.uniform(binary_tree(2), 2, 2), [torch.ones(2, 2, 1)])
    with pytest.raises(ValueError):
        aligned_init(
            spec,
            odeco_teacher(4, 3, [1.0, 0.5, 0.2]).factors[:, :, :2],
            torch.ones(3, 3, dtype=torch.float64),
        )


def test_centred_loss_matches_shifted_teacher_and_is_stable():
    spec = TTNSpec.uniform(binary_tree(8), 2, 2)
    base = random_cp_teacher(8, 2, [1.0, 0.5], seed=1)
    g = torch.Generator().manual_seed(0)
    for sigma in (1.0, 10.0):
        cores0 = spec.init(sigma, seed=4)
        cores = [c + 1e-8 * torch.randn(c.shape, generator=g, dtype=torch.float64) for c in cores0]
        resid = to_dense(spec, cores) - to_dense(spec, cores0) - base.dense()  # error ~1e-9
        exact = 0.5 * resid.pow(2).sum()
        assert abs(centred_population_loss(spec, cores, cores0, base) - exact) < 1e-7
        if sigma == 1.0:
            shifted = population_loss(spec, cores, ShiftedTeacher(base, spec, cores0))
            assert abs(shifted - exact) < 1e-12
