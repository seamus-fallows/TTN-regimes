"""Checks of ttn.routes against the statements of the saddle-order / aligned-dynamics paper.

Each test names the statement it checks. They double as numerical sanity checks of Sections 7-8
(which, unlike Sections 3-4, were not independently reviewed)."""

import itertools
import math
import random

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from ttn.dynamics import Problem
from ttn.network import to_dense
from ttn.routes import (
    AlignedManifold,
    RouteTarget,
    amplitudes,
    crossing_time,
    effective_depth,
    off_target_mass,
)
from ttn.teachers import random_orthogonal
from ttn.tree import make_tree

torch.set_num_threads(1)

B4 = ([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], [1.0, 0.3, 0.6])
PAIR_STAR = ([[0] * 6, [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0]], [1.0, 0.8, 0.8])
EX87 = ([[0] * 7, [1] * 7, [0, 0, 2, 2, 1, 1, 1]], [1.0, 0.9, 0.5])
NESTED = ([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], [1.0, 0.6, 0.35])


def random_star_targets(n_targets=30, seed=0):
    """Random pattern targets that satisfy the star condition, on several trees."""
    rng = random.Random(seed)
    out = []
    while len(out) < n_targets:
        kind, n = rng.choice([("mps", 4), ("mps", 5), ("binary", 4), ("binary", 6), ("ht", 4)])
        d = rng.choice([2, 3])
        pats = set()
        while len(pats) < rng.choice([2, 3, 4]):
            pats.add(tuple(rng.randrange(d) for _ in range(n)))
        target = RouteTarget(
            make_tree(kind, n), sorted(pats), [rng.uniform(0.3, 1.0) for _ in pats]
        )
        if target.is_star:
            out.append((target, d))
    return out


def test_star_condition_on_named_teachers():
    assert RouteTarget(make_tree("mps", 4), *B4).is_star
    assert RouteTarget(make_tree("mps", 6), *PAIR_STAR).is_star
    assert RouteTarget(make_tree("mps", 7), *EX87).is_star
    # nested teacher: 1100 takes one side of 1110 and the other side of 0000 across the middle cut
    assert not RouteTarget(make_tree("mps", 4), *NESTED).is_star


def test_rank_across_edge_equals_number_of_blocks():
    """Proposition 7.2."""
    for target, d in random_star_targets():
        tree = target.tree
        dense = target.teacher(d).dense()
        for e in tree.edges:
            below = list(tree.subtree_sites[e])
            rest = [s for s in range(tree.n_sites) if s not in below]
            mat = dense.permute(below + rest).reshape(d ** len(below), -1)
            assert torch.linalg.matrix_rank(mat).item() == target.n_blocks[e]


def test_local_types_agree_exactly_outside_the_steiner_subtree():
    """Lemma 7.3, and Corollary 7.4 (routing regions are nonempty and connected)."""
    for target, _ in random_star_targets():
        for a, b in itertools.combinations(range(target.n), 2):
            shared = {
                v
                for v in range(target.tree.n_nodes)
                if target.local_type(v, a) == target.local_type(v, b)
            }
            assert shared == set(range(target.tree.n_nodes)) - target.steiner(target.diff(a, b))
        for r in range(target.n):
            for size in range(1, target.n):
                for J in itertools.combinations([a for a in range(target.n) if a != r], size):
                    region = target.region(J, r)
                    assert region and target.steiner(region) == region


def test_manifold_output_is_exactly_the_target_span():
    """Theorem 7.6: W = sum_a P_a U_a on M*, with P_a the product of the route entries."""
    rng = np.random.default_rng(0)
    for target, d in random_star_targets():
        man = AlignedManifold(target, d, chi=target.n + 1)
        g = rng.uniform(0.3, 1.5, man.n_entries) * rng.choice([-1.0, 1.0], man.n_entries)
        cores = man.spec.unflatten(man.theta(g))
        expect = torch.zeros((d,) * target.tree.n_sites, dtype=torch.float64)
        for p, amp in zip(target.patterns, man.amplitudes(g)):
            expect[p] = amp
        assert torch.allclose(to_dense(man.spec, cores), expect, atol=1e-13)
        assert off_target_mass(man.spec, cores, man.teacher) < 1e-13
        assert np.allclose(
            amplitudes(man.spec, cores, man.teacher).numpy(), man.amplitudes(g), atol=1e-13
        )


def test_block_construction_recombines_without_the_star_condition():
    """The star condition is what rules out unintended strings: the nested teacher produces one."""
    target = RouteTarget(make_tree("mps", 4), *NESTED)
    man = AlignedManifold(target, 2, require_star=False)
    cores = man.spec.unflatten(man.theta(np.ones(man.n_entries)))
    assert off_target_mass(man.spec, cores, man.teacher) > 0.1


def test_manifold_is_invariant_and_flow_matches_the_reduced_equations():
    """Theorem 7.7: forbidden entries feel no force on M*, and the reduced flow is the full flow."""
    rng = np.random.default_rng(1)
    for target, d in random_star_targets(8, seed=1):
        man = AlignedManifold(target, d, chi=target.n + 1)
        problem = Problem(man.spec, man.teacher)
        g = rng.uniform(0.2, 1.2, man.n_entries)
        grad = problem.grad(man.theta(g))
        assert grad[~man.mask].abs().max() < 1e-13
        assert np.allclose(
            -grad[man.mask.nonzero()[:, 0]].numpy()[np.argsort(np.argsort(man.flat))],
            man.velocity(g),
            atol=1e-12,
        )
        # finite-difference check of the reduced Jacobian
        eps, jac = 1e-6, man.jacobian(g)
        for k in range(man.n_entries):
            dg = np.zeros(man.n_entries)
            dg[k] = eps
            assert np.allclose(
                (man.velocity(g + dg) - man.velocity(g - dg)) / (2 * eps), jac[:, k], atol=1e-7
            )


def test_full_gradient_flow_from_the_manifold_follows_the_reduced_flow():
    """Theorem 7.7 as a trajectory: full-space LSODA from an aligned start vs the reduced system."""
    from scipy.integrate import solve_ivp

    target = RouteTarget(make_tree("mps", 4), *B4)
    man = AlignedManifold(target, 2, chi=4)
    problem = Problem(man.spec, man.teacher)
    g0 = np.full(man.n_entries, 0.3)
    t_end = 60.0
    with threadpool_limits(limits=1, user_api="blas"):
        full = solve_ivp(
            lambda t, y: problem.velocity(torch.from_numpy(y)).numpy(),
            (0, t_end),
            man.theta(g0).numpy(),
            method="LSODA",
            rtol=1e-11,
            atol=1e-14,
            jac=lambda t, y: problem.jacobian(torch.from_numpy(y)).numpy(),
        )
    reduced = man.flow(g0, t_end, rtol=1e-11)
    theta_end = torch.from_numpy(full.y[:, -1])
    assert theta_end[~man.mask].abs().max() < 1e-12
    assert np.allclose(man.g(theta_end), reduced["g"][-1], atol=1e-7)
    assert np.all(reduced["P"][-1] > 0.2 * np.array(target.weights))  # all three are being learned


def test_route_depths_of_the_paper_examples():
    """Example 8.7 (splice of depth 2), the pair-star, and Corollary 8.6 (monotone in J)."""
    ex = RouteTarget(make_tree("mps", 7), *EX87)
    assert ex.depth([0, 1], 2) == 2 and sorted(ex.region([0, 1], 2)) == [2, 3]
    assert ex.pair_cost(0, 2) == 5 and ex.pair_cost(1, 2) == 4
    ps = RouteTarget(make_tree("mps", 6), *PAIR_STAR)
    assert [s["depth"] for s in ps.greedy_sequence()] == [5, 2]
    for target, _ in random_star_targets():
        for r in range(target.n):
            others = [a for a in range(target.n) if a != r]
            for size in range(1, len(others)):
                for J in itertools.combinations(others, size):
                    for extra in set(others) - set(J):
                        assert target.region([*J, extra], r) <= target.region(J, r)


def test_partial_target_points_are_critical_iff_every_depth_is_at_least_two():
    """Lemma 8.1 and Theorem 8.4(1), in the full parameter space with spare bond dimension."""
    seen = set()
    for target, d in random_star_targets(40, seed=2):
        man = AlignedManifold(target, d, chi=target.n + 1)
        problem = Problem(man.spec, man.teacher)
        for size in range(1, target.n):
            for J in itertools.combinations(range(target.n), size):
                theta = man.theta(man.partial_target(J))
                amps = amplitudes(man.spec, man.spec.unflatten(theta), man.teacher).numpy()
                expect = np.where(np.isin(np.arange(target.n), J), target.weights, 0.0)
                assert np.allclose(amps, expect, atol=1e-13)
                critical = problem.grad(theta).abs().max().item() < 1e-12
                assert critical == (target.min_depth(J) >= 2)
                seen.add(critical)
    assert seen == {True, False}


def test_balanced_partial_target_point():
    target = RouteTarget(make_tree("mps", 6), *PAIR_STAR)
    man = AlignedManifold(target, 2, chi=3)
    g = man.partial_target([0], relax_from=1e-2)
    assert np.allclose(man.amplitudes(g), [1.0, 0.0, 0.0], atol=1e-14)
    active = g[man.used_by([0])]
    assert np.allclose(active, active[0], atol=1e-3)  # balanced: all active entries equal


def test_one_component_order():
    """Example 4.5 (order 4 with k* = 5), its control, and Corollary 4.4."""
    tree = make_tree("mps", 6)
    res = RouteTarget(tree, *PAIR_STAR).one_component_order(0)
    assert res["k_star"] == 5 and res["l_star"] == 2 and res["order"] == 4
    assert res["mu"][2] == pytest.approx(0.8 * math.sqrt(2))
    control = RouteTarget(tree, PAIR_STAR[0], [1.0, 0.6, 0.6]).one_component_order(0)
    assert control["order"] == 5 and control["mechanism"] == "direct"
    # one weaker residual: order = Steiner size; one stronger residual with a low split: 2 l*
    assert (
        RouteTarget(tree, [[0] * 6, [1, 1, 0, 0, 0, 1]], [1.0, 0.7]).one_component_order(0)["order"]
        == 6
    )
    assert (
        RouteTarget(tree, [[0] * 6, [1, 1, 0, 0, 0, 1]], [1.0, 1.5]).one_component_order(0)["order"]
        == 4
    )
    ht = RouteTarget(make_tree("ht", 4), [[0] * 4, [1] * 4], [1.0, 1.5]).one_component_order(0)
    assert ht["k_star"] == 7 and ht["order"] == 6


def test_scalar_route_equation_and_effective_depth():
    """Eqs. 6.15-6.16 from theta_J with balanced dormant entries: Example 8.7 (depth 2) and a
    depth-4 route; the effective-depth estimator recovers the depth."""
    for (pats, w), tree, J, r in [
        (EX87, make_tree("mps", 7), [0, 1], 2),
        (B4, make_tree("mps", 4), [0, 1], 2),
    ]:
        target = RouteTarget(tree, pats, w)
        man = AlignedManifold(target, 3 if pats is EX87[0] else 2)
        depth, s = target.depth(J, r), target.weights[r]
        times = []
        for delta in (1e-3, 1e-4):
            g0 = man.partial_target(J, dormant=delta)
            gamma = man.amplitudes(np.where(man.used_by(J), g0, 1.0))[
                r
            ]  # A_r(J): active entries of r
            res = man.flow(g0, 1e30, stop=lambda t, g, p: p[r] > 0.9 * s)
            times.append(crossing_time(res["t"], res["P"][:, r], 0.5 * s))
            est = effective_depth(res["t"], res["P"][:, r], s)
            assert est["depth"] == pytest.approx(depth, abs=0.05)
        if depth == 2:  # log law: t = log(1/delta) / (gamma s) + const
            assert (times[1] - times[0]) == pytest.approx(math.log(10) / (gamma * s), rel=1e-2)
        else:  # t = delta^-(d-2) / ((d-2) gamma s)
            assert times[1] == pytest.approx(
                1e-4 ** -(depth - 2) / ((depth - 2) * gamma * s), rel=2e-2
            )


def test_align_recovers_route_entries_under_random_gauges():
    """A point near theta_J on M*, hidden by random orthogonal bond rotations: `align` brings it
    back and `route_entry_norms` returns the dormant entries."""
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(3)
    for (pats, w), tree, d, J, r in [
        (EX87, make_tree("mps", 7), 3, [0, 1], 2),
        (B4, make_tree("mps", 4), 2, [0, 1], 2),
        (B4, make_tree("ht", 4), 2, [0], 2),
    ]:
        target = RouteTarget(tree, pats, w)
        man = AlignedManifold(target, d, chi=4)
        g = man.partial_target(J, relax_from=1e-2)
        dormant = ~man.used_by(J)
        g[dormant] = rng.uniform(1e-4, 1e-3, dormant.sum())
        cores = [c.clone() for c in man.spec.unflatten(man.theta(g))]
        for e in tree.edges:  # random orthogonal gauge on every bond
            q = random_orthogonal(man.spec.bond_dims[e], gen)
            p = tree.parent[e]
            leg = len(tree.sites[p]) + tree.children[p].index(e)
            cores[e] = torch.tensordot(cores[e], q, dims=([cores[e].ndim - 1], [0]))
            cores[p] = torch.movedim(torch.tensordot(cores[p], q, dims=([leg], [0])), -1, leg)
        aligned, quality = man.align(cores, J)
        assert min(quality.values()) > 0.99
        norms = man.route_entry_norms(aligned, J, r)
        assert sorted(norms) == sorted(target.region(J, r))
        for v, value in norms.items():
            # the norm over the dormant subspace contains the entry of r (and possibly entries of
            # other missing components that share its physical index there)
            assert value >= abs(g[man.route[r, v]]) * (1 - 1e-6)
            assert value <= 1.01 * np.sqrt((g[dormant] ** 2).sum())
        # learned amplitudes are untouched by the gauge maps
        assert np.allclose(
            amplitudes(man.spec, aligned, man.teacher).numpy()[J], np.array(w)[J], atol=1e-9
        )
