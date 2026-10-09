import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1

from ttn.losses import population_loss
from ttn.network import TTNSpec, aligned_init
from ttn.teachers import odeco_teacher
from ttn.theory import (
    balanced_mode_solution,
    balanced_mode_solution_k2,
    escape_time,
    escape_time_asymptotic,
    odeco_aligned_prediction,
    product_gaussian_sampler,
    unbalanced_escape_time,
    unbalanced_mode_solution,
    winner_probability,
)
from ttn.tree import make_tree


def closed_form_escape_time(K, s, a0, level=0.5):
    """Independent closed form via G(a) = int_0^a x dx/(s - x^K) = a^2/(2s) F(a^K/s),
    F(z) = 2F1(1, 2/K; 1 + 2/K; z)."""
    a_star = (level * s) ** (1 / K)

    def g(a):
        return a**2 / (2 * s) * hyp2f1(1, 2 / K, 1 + 2 / K, a**K / s)

    if K == 2:
        singular = np.log(a_star / a0) / s
    else:
        singular = (a0 ** (2 - K) - a_star ** (2 - K)) / ((K - 2) * s)
    return singular + (g(a_star) - g(a0)) / s


def full_mode_ode(s, a0, t):
    """All K amplitudes of one mode, a_v' = (s - P) prod_{w != v} a_w, without any reduction."""

    def rhs(_, a):
        return np.array([(s - np.prod(a)) * np.prod(np.delete(a, v)) for v in range(len(a))])

    sol = solve_ivp(rhs, (0, t[-1]), a0, t_eval=t, method="DOP853", rtol=1e-12, atol=1e-15)
    return sol.y.T


# ---- single mode -----------------------------------------------------------------------------


def test_k2_closed_form_matches_numerical():
    s, a0 = 1.5, 1e-3
    t = np.linspace(0, 12, 60)
    num = balanced_mode_solution(2, s, a0, t)
    np.testing.assert_allclose(num, balanced_mode_solution_k2(s, a0, t), rtol=1e-10)
    assert num[-1] ** 2 == pytest.approx(s, rel=1e-8)


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_escape_time_is_level_crossing_of_ode(K):
    s, a0, level = 0.8, 0.05, 0.5
    T = escape_time(K, s, a0, level)
    assert T == pytest.approx(closed_form_escape_time(K, s, a0, level), rel=1e-12)
    a = balanced_mode_solution(K, s, a0, [0.0, 0.5 * T, T])
    assert a[1] ** K < level * s
    assert a[2] ** K / (level * s) == pytest.approx(1, abs=1e-8)


def test_escape_time_k2_logistic():
    s, a0, level = 0.7, 1e-3, 0.3
    p0, p1 = a0**2, level * s
    expected = np.log(p1 * (s - p0) / (p0 * (s - p1))) / (2 * s)
    assert escape_time(2, s, a0, level) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("K", [2, 3, 4, 6])
def test_escape_time_asymptotic_small_init(K):
    s = 1.3
    rel_err_leading = []
    for a0 in [0.3, 0.1, 0.03, 0.01, 1e-3]:
        exact = escape_time(K, s, a0)
        leading = escape_time_asymptotic(K, s, a0, correction=False)
        corrected = escape_time_asymptotic(K, s, a0)
        rel_err_leading.append(abs(exact / leading - 1))
        # The remainder of the corrected formula is -G(a0)/s ~ -a0^2 / (2 s^2).
        assert abs(exact - corrected) <= a0**2 / s**2 + 1e-12 * exact
    assert rel_err_leading == sorted(rel_err_leading, reverse=True)
    assert rel_err_leading[-1] < 1e-3


def test_escape_time_asymptotic_vectorized():
    s, a0 = np.array([0.5, 1.0, 2.0]), np.array([[1e-2], [1e-3]])
    out = escape_time_asymptotic(4, s, a0)
    assert out.shape == (2, 3)
    assert out[1, 2] == pytest.approx(escape_time(4, 2.0, 1e-3), rel=1e-9)


def test_unbalanced_with_zero_imbalance_equals_balanced():
    K, s, a0 = 4, 1.2, 0.05
    t = np.linspace(0, 1.5 * escape_time(K, s, a0), 50)
    balanced = balanced_mode_solution(K, s, a0, t)
    unbalanced = unbalanced_mode_solution(s, np.full(K, a0), t)
    assert unbalanced.shape == (len(t), K)
    np.testing.assert_allclose(unbalanced, np.repeat(balanced[:, None], K, axis=1), rtol=1e-9)
    assert unbalanced_escape_time(s, np.full(K, a0)) == pytest.approx(
        escape_time(K, s, a0), rel=1e-10
    )


@pytest.mark.parametrize(
    "a0",
    [
        [0.05, 0.08, 0.12],
        [-0.05, 0.08, -0.12],  # an even number of sign flips is a symmetry
        [0.02, 0.02, 0.3, 0.5],
        [0.01, 0.2, 0.3, 0.4, 0.6],
        [1e-6, 0.3, 0.4],  # one tiny core: no plateau, it just grows
    ],
)
def test_unbalanced_matches_full_ode(a0):
    s, a0 = 1.1, np.array(a0)
    T = unbalanced_escape_time(s, a0)
    t = np.linspace(0, 2 * T, 60)
    reduced = unbalanced_mode_solution(s, a0, t)
    np.testing.assert_allclose(reduced, full_mode_ode(s, a0, t), rtol=1e-9)
    imbalance = reduced**2 - reduced[:, :1] ** 2
    np.testing.assert_allclose(imbalance, imbalance[:1].repeat(len(t), 0), atol=1e-12)
    p_at_escape = np.prod(unbalanced_mode_solution(s, a0, [0.0, T])[-1])
    assert p_at_escape / (0.5 * s) == pytest.approx(1, abs=1e-9)


# ---- whole teacher ---------------------------------------------------------------------------


def test_plateau_levels_and_learning_order():
    weights = [1.0, 4.0, 2.0]
    K, a0 = 3, 1e-3
    probe = odeco_aligned_prediction(weights, K, a0, [0.0, 1.0])
    esc = probe["escape_times"]
    assert list(probe["order"]) == [1, 2, 0]
    np.testing.assert_allclose(probe["plateau_levels"], [10.5, 2.5, 0.5, 0.0])
    e1, e2, e3 = np.sort(esc)
    t = np.array([0.0, 0.5 * e1, 0.5 * (e1 + e2), 0.5 * (e2 + e3), 2 * e3])
    loss = odeco_aligned_prediction(weights, K, a0, t)["loss"]
    expected = probe["plateau_levels"][[0, 0, 1, 2, 3]]
    np.testing.assert_allclose(loss, expected, atol=1e-6)


def test_per_component_init_can_reverse_the_order():
    # Larger init on the weaker component: it escapes first (K = 3, T ~ 1 / (s a0)).
    out = odeco_aligned_prediction([1.0, 2.0], 3, [0.1, 0.01], [0.0, 1.0])
    assert list(out["order"]) == [0, 1]
    assert out["escape_times"][0] == pytest.approx(escape_time(3, 1.0, 0.1))


def ttn_gradient_flow(spec, teacher, cores0, t):
    """Reference: d theta/dt = -grad L on the flat parameter vector, autograd gradients."""
    t2 = teacher.norm2()

    def rhs(_, theta):
        flat = torch.from_numpy(theta).requires_grad_(True)
        loss = population_loss(spec, spec.unflatten(flat), teacher, teacher_norm2=t2)
        return -torch.autograd.grad(loss, flat)[0].numpy()

    theta0 = spec.flatten(cores0).numpy()
    sol = solve_ivp(rhs, (0, t[-1]), theta0, t_eval=t, method="DOP853", rtol=1e-12, atol=1e-14)
    return [spec.unflatten(torch.from_numpy(theta)) for theta in sol.y.T]


@pytest.mark.parametrize(
    "kind,n,balanced",
    [
        ("binary", 4, True),  # K = 3
        ("mps", 4, True),  # K = 4
        ("ht", 3, True),  # K = 5
        ("binary", 4, False),
        ("mps", 4, False),
    ],
)
def test_prediction_matches_ttn_gradient_flow(kind, n, balanced):
    d, weights = 3, [2.0, 1.0, 0.6]
    teacher = odeco_teacher(n, d, weights, seed=1)
    spec = TTNSpec.uniform(make_tree(kind, n), d, 3)
    K = spec.n_cores
    # Scale chosen so that the slowest component escapes at t ~ 20.
    a0 = np.full((K, 3), (12.0 * (K - 2)) ** (-1 / (K - 2)))
    if not balanced:
        a0 = a0 * np.random.default_rng(0).uniform(0.7, 1.4, size=(K, 3))
    cores0 = aligned_init(spec, teacher.factors, torch.from_numpy(a0))
    t_end = 1.3 * odeco_aligned_prediction(weights, K, a0, [0.0, 1.0])["escape_times"].max()
    t = np.linspace(0, t_end, 40)
    pred = odeco_aligned_prediction(weights, K, a0, t)
    states = ttn_gradient_flow(spec, teacher, cores0, t)
    overlaps = np.stack([teacher.component_overlaps(spec, c).numpy() for c in states])
    losses = np.array([population_loss(spec, c, teacher).item() for c in states])
    np.testing.assert_allclose(overlaps, pred["overlaps"], atol=1e-6)
    np.testing.assert_allclose(losses, pred["loss"], atol=1e-6)
    assert overlaps[-1].min() > 0.99  # all three components were learned during the run


# ---- heuristic race --------------------------------------------------------------------------


def test_winner_probability_heuristic():
    # Equal weights, iid effective amplitudes: every component wins equally often.
    p = winner_probability([1.0, 1.0, 1.0], 4, product_gaussian_sampler(4, 1e-2), 30_000)
    assert p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p, 1 / 3, atol=0.02)
    # Equal effective amplitudes: the largest weight always wins.
    p = winner_probability([1.0, 3.0, 2.0], 4, lambda rng, shape: np.full(shape, 1e-2), 10)
    np.testing.assert_array_equal(p, [0.0, 1.0, 0.0])
    # K = 3, c_r = eps U(0, 1): component 0 wins iff 2 c_0 > c_1, probability 3/4.
    p = winner_probability([2.0, 1.0], 3, lambda rng, shape: 1e-4 * rng.uniform(size=shape), 40_000)
    assert p[0] == pytest.approx(0.75, abs=0.01)
