import numpy as np
import pytest
import torch
from scipy.special import expit
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from ttn.diagnostics import (
    Plateau,
    centered_kernel_alignment,
    classify_regime,
    crossing_times,
    effective_rank,
    effective_rank_relative,
    entropy_rank,
    event_widths,
    find_plateaus,
    gauge_complement,
    gauge_directions,
    hessian,
    hessian_eigs,
    hvp,
    kernel_alignment,
    kernel_change,
    lanczos_extremes,
    merge_events,
    plateau_midpoints,
    plateau_scaling,
    product_features,
    project_out_gauge,
    projected_fit_loss,
    ratio_lower_bounds,
    relative_change,
    rise_times,
    saddle_scaling,
    saddle_type,
    sample_spacing,
    separation_ratios,
    smallest_nonzero,
    spectral_events,
    tangent_kernel,
    tensor_jacobian,
    tensor_tangent_kernel,
    upward_crossings,
)
from ttn.losses import population_loss, power_law_cov, sample_inputs
from ttn.network import (
    TTNSpec,
    aligned_init,
    antisymmetric_double,
    forward,
    to_dense,
    two_copy_cores,
)
from ttn.teachers import odeco_teacher, pattern_teacher, random_cp_teacher
from ttn.tree import Tree, make_tree

# ---- ranks -----------------------------------------------------------------------------------


def test_ranks_on_known_spectra():
    nan = np.nan
    sv = np.array([[3.0, 2.0, 1.0, nan], [1.0, 1.0, 1.0, 1.0], [1e-14, 1e-15, nan, nan]])
    assert effective_rank(sv, 0.5).tolist() == [3, 4, 0]
    assert effective_rank(sv, np.array([1.5, 0.5, 1e-16])).tolist() == [2, 4, 2]
    assert np.allclose(smallest_nonzero(sv), [1.0, 1.0, 1e-15])
    assert np.isnan(smallest_nonzero(np.array([0.0, 0.0, nan])))
    # Notes' convention: threshold 0.1 x smallest nonzero teacher Schmidt value on each edge.
    teacher = np.array([[3.0, 2.0, 0.0, nan], [1.0, 1.0, 1.0, 1.0], [2.0, nan, nan, nan]])
    assert effective_rank_relative(sv, 0.1, smallest_nonzero(teacher)).tolist() == [3, 4, 0]
    assert effective_rank_relative(sv, 0.4).tolist() == [2, 4, 1]  # relative to s_1

    p = np.array([9.0, 4.0, 1.0]) / 14
    assert entropy_rank(sv[0]) == pytest.approx(np.exp(-(p * np.log(p)).sum()))
    assert entropy_rank(sv[1]) == pytest.approx(4.0)
    assert entropy_rank(np.array([2.0, 0.0, nan])) == pytest.approx(1.0)
    assert entropy_rank(np.array([1e-170, 1e-170])) == pytest.approx(2.0)  # no underflow
    assert entropy_rank(np.zeros(3)) == 0.0 and entropy_rank(np.full(3, nan)) == 0.0


def test_ranks_vectorized_over_trajectories():
    rng = np.random.default_rng(0)
    spectra = np.sort(rng.random((7, 3, 4)), axis=-1)[..., ::-1].copy()
    spectra[:, 1, 2:] = np.nan  # edge 1 has bond dimension 2
    thr = np.array([0.3, 0.5, 0.7])
    ranks, ent = effective_rank(spectra, thr), entropy_rank(spectra)
    assert ranks.shape == ent.shape == (7, 3)
    for i in range(7):
        for e in range(3):
            s = spectra[i, e][np.isfinite(spectra[i, e])]
            assert ranks[i, e] == np.sum(s > thr[e])
            p = s**2 / np.sum(s**2)
            assert ent[i, e] == pytest.approx(np.exp(-(p * np.log(p)).sum()))


# ---- events ----------------------------------------------------------------------------------


def test_crossing_times_and_widths_on_sigmoids():
    t = np.logspace(-2, 3, 2001)
    centers, scales = np.array([3.0, 40.0, 500.0]), np.array([0.5, 4.0, 20.0])
    curves = np.column_stack([expit((t[:, None] - centers) / scales), np.zeros_like(t)])
    times = crossing_times(t, curves, 0.5)
    assert np.allclose(times[:3], centers, rtol=1e-3) and np.isnan(times[3])
    widths = event_widths(t, curves)
    assert np.allclose(widths[:3], 2 * np.log(9) * scales, rtol=1e-2) and np.isnan(widths[3])
    # Per-curve levels (e.g. half of each curve's final value), and a 1-D curve.
    levels = 0.25 * np.ones(4)
    expected = centers + scales * np.log(1 / 3)
    assert np.allclose(crossing_times(t, curves, levels)[:3], expected, rtol=1e-3)
    assert crossing_times(t, curves[:, 1], 0.5) == pytest.approx(40.0, rel=1e-3)


def test_crossing_times_interpolation_conventions():
    assert crossing_times([1.0, 100.0], [0.0, 1.0], 0.5) == pytest.approx(10.0)  # log t
    assert crossing_times([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], 0.25) == pytest.approx(0.25)  # t = 0
    assert crossing_times([1.0, 2.0, 3.0], [0.7, 0.8, 0.9], 0.5) == 1.0  # starts above
    assert crossing_times([1.0, 2.0, 4.0], [np.nan, 0.0, 1.0], 0.5) == pytest.approx(np.sqrt(8))
    assert crossing_times([1.0, 2.0, 4.0], [0.0, np.nan, 1.0], 0.5) == 4.0


def test_event_widths_ignore_transient_bumps():
    # Like a_2 in GF on binary4: a transient bump while component 1 is learned, real rise later.
    t = np.linspace(0, 200, 4001)
    rise = expit((t - 100) / 2.0)
    for height in (0.4, 0.6):
        curve = rise + height * np.exp(-((t - 10) ** 2) / 2)
        assert event_widths(t, curve) == pytest.approx(2 * np.log(9) * 2.0, rel=1e-2)
        assert crossing_times(t, curve, 0.5, last=True) == pytest.approx(100, abs=0.01)
    assert crossing_times(t, curve, 0.5) < 20  # the first crossing is the transient
    assert np.isnan(crossing_times(t, 1 - rise, 0.5, last=True))  # ends below the level
    assert crossing_times(t, rise + 1, 0.5, last=True) == 0.0


def test_separation_ratios_and_merging():
    rho = separation_ratios([100.0, 10.0, np.nan, 130.0], [5.0, 1.0, 3.0, 10.0])
    assert np.allclose(rho, [90 / 5, 30 / 10])  # sorted: 10 (w 1), 100 (w 5), 130 (w 10)
    assert separation_ratios([1.0], [1.0]).size == 0
    assert np.isinf(separation_ratios([1.0, 2.0], [0.0, 0.0])[0])
    assert np.isnan(separation_ratios([1.0, 2.0], [np.nan, 0.5])[0])
    times, widths = merge_events([10.0, 10.5, 50.0, 10.2], [1.0, 1.0, 2.0, 0.5])
    assert np.allclose(times, [np.mean([10.0, 10.2, 10.5]), 50.0])
    assert np.allclose(widths, [1.0, 2.0])


def test_plateau_midpoints():
    t = np.linspace(0, 10, 101)
    speed = np.minimum(np.abs(t - 4.0), np.abs(t - 8.5)) + 0.1
    assert np.allclose(plateau_midpoints(t, speed, [9.5, np.nan, 1.0, 6.0]), [4.0, 8.5])


def test_resolution_of_events_on_a_log_grid():
    # A rise of width 2 ln 9 * 0.2 = 0.88 at t = 40 on a log grid (spacing 6.5 there): its width
    # comes out ~0.8 intervals whatever its true value, and is flagged.
    t = np.geomspace(1, 100, 30)
    curves = np.column_stack([expit((t - 40) / 0.2), expit((np.log(t) - np.log(5)) / 0.02)])
    lo, hi = rise_times(t, curves)
    widths, spacing = hi - lo, sample_spacing(t, lo, hi)
    assert widths[0] == pytest.approx(event_widths(t, curves)[0])
    assert widths[0] > 3 * 0.88 and widths[0] < 3 * spacing[0]  # unresolved
    fine = np.union1d(t, np.linspace(30, 50, 2001))
    lo, hi = rise_times(fine, expit((fine - 40) / 0.2))
    assert hi - lo == pytest.approx(0.88, rel=0.01) and hi - lo > 3 * sample_spacing(fine, lo, hi)
    grid = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    assert sample_spacing(grid, 1.5, 1.7) == 1.0 and sample_spacing(grid, 0.5, 5.0) == 4.0
    np.testing.assert_array_equal(sample_spacing(grid, [1.0, np.nan], [2.0, 3.0]), [1.0, np.nan])
    # Pair flags follow the event order of separation_ratios (sorted, NaN times dropped).
    flags = ratio_lower_bounds([10.0, np.nan, 5.0, 20.0], [False, True, True, False])
    np.testing.assert_array_equal(flags, [True, False])
    bump = np.array([0.0, 0.6, 0.2, 0.1, 0.7, 1.0])
    assert upward_crossings(bump, 0.5) == 2 and upward_crossings(np.ones(3), 0.5) == 1
    np.testing.assert_array_equal(upward_crossings(np.column_stack([bump, 1 - bump]), 0.5), [2, 2])


def test_spectral_events_and_merging():
    # Two edges; component A raises s_1 on both edges at t = 10, component B raises s_2 on edge 0
    # only at t = 100 (edge 1 has bond 1, padded); a decaying extra value on edge 0 is no event.
    t = np.linspace(0, 200, 4001)
    s = np.full((len(t), 2, 3), np.nan)
    s[:, 0, 0] = s[:, 1, 0] = 3 * expit((t - 10) / 0.5) + 1e-6
    s[:, 0, 1] = 1 * expit((t - 100) / 1.0) + 1e-6
    s[:, 0, 2] = 0.3 * np.exp(-t / 20) + 1e-6
    ref = [[3.0, 1.0], [3.0]]
    ev = spectral_events(t, s, reference=ref)
    assert list(zip(ev["edge"], ev["k"])) == [(0, 0), (0, 1), (1, 0)]
    assert np.allclose(ev["time"], [10, 100, 10], atol=0.01) and np.all(ev["n_cross"] == 1)
    assert np.allclose(ev["width"], 2 * np.log(9) * np.array([0.5, 1.0, 0.5]), rtol=0.01)
    times, widths, groups = merge_events(ev["time"], ev["width"], return_groups=True)
    assert np.allclose(times, [10, 100], atol=0.01) and [list(g) for g in groups] == [[0, 2], [1]]
    assert separation_ratios(times, widths)[0] == pytest.approx(90 / (2 * np.log(9)), rel=0.01)
    # Without a reference every value present at the end counts, the decaying one included.
    assert len(spectral_events(t, s)["time"]) == 4


# ---- regime assignment -----------------------------------------------------------------------


def test_classify_regime():
    assert classify_regime(0.5, [10.0, 5.0], 1e-6, 1.0) == "saddle-to-saddle"
    assert classify_regime(0.5, [10.0, 2.0], 1e-6, 1.0) == "rich"
    assert classify_regime(0.5, [], 1e-6, 1.0) == "rich"  # a single event is not S2S
    assert classify_regime(0.5, [10.0, np.nan], 1e-6, 1.0) == "rich"
    assert classify_regime(0.05, [10.0, 5.0], 1e-6, 1.0) == "lazy"
    assert classify_regime(0.5, [10.0], 2e-3, 1.0) == "stalled"
    assert classify_regime(0.05, [], 0.5, 1.0) == "stalled"
    assert classify_regime(0.0, [], np.nan, 1.0) == "diverged"
    assert classify_regime(0.0, [], 2e6, 1.0) == "diverged"
    # Threshold sensitivity (notes: delta_K in {0.05, 0.1, 0.2}, separation in {2, 3, 5}).
    assert classify_regime(0.15, [4.0], 1e-6, 1.0, {"delta_K": 0.2}) == "lazy"
    assert classify_regime(0.5, [4.0], 1e-6, 1.0, {"separation": 5.0}) == "rich"
    assert classify_regime(0.5, [4.0], 1e-6, 1.0) == "saddle-to-saddle"
    # Overlapping events whose widths are only grid-limited bounds: undetermined, not rich.
    assert classify_regime(0.5, [10.0, 2.0], 1e-6, 1.0, lower_bounds=[False, True]) == (
        "undetermined"
    )
    assert classify_regime(0.5, [2.5, 2.0], 1e-6, 1.0, lower_bounds=[False, True]) == "rich"
    assert classify_regime(0.5, [10.0, np.nan], 1e-6, 1.0, lower_bounds=[True, True]) == "rich"
    # Deficient bonds: a lazy run ends at the projected fit (floor), not below 1e-3 L(0).
    assert classify_regime(0.05, [], 0.31, 1.0, lazy_floor=0.30) == "lazy (projected fit)"
    assert classify_regime(0.05, [], 0.40, 1.0, lazy_floor=0.30) == "stalled"
    assert classify_regime(0.5, [], 0.31, 1.0, lazy_floor=0.30) == "stalled"
    assert classify_regime(0.05, [], 1e-6, 1.0, lazy_floor=0.0) == "lazy"


def test_plateau_scaling():
    rng = np.random.default_rng(0)
    sigma = np.repeat(np.logspace(-4, -1, 6), 3)  # 6 scales x 3 seeds
    log_growth = 5 + 2 * np.log(1 / sigma) + 0.1 * rng.standard_normal(sigma.size)
    fit = plateau_scaling(sigma, log_growth)
    assert fit["grows"] and fit["n"] == 18 and fit["log"]["slope"] == pytest.approx(2, abs=0.05)
    power_growth = 3 * sigma**-1.5 * np.exp(0.05 * rng.standard_normal(sigma.size))
    fit = plateau_scaling(sigma, power_growth)
    assert fit["grows"] and fit["power"]["exponent"] == pytest.approx(1.5, abs=0.03)
    assert fit["power"]["r2"] > 0.99
    bounded = 7 + 0.1 * rng.standard_normal(sigma.size)
    bounded[0] = np.nan  # runs without a plateau are skipped
    fit = plateau_scaling(sigma, bounded)
    assert not fit["grows"] and fit["n"] == 17


# ---- plateaus --------------------------------------------------------------------------------


def _steps(x, centers, sizes, scale, final):
    """final + sum of logistic drops: drop i lowers the loss by sizes[i] around x = centers[i]."""
    return final + sum(s * expit(-(x - c) / scale) for c, s in zip(centers, sizes))


def test_find_plateaus_linear_time_separation():
    # Drops at t = 200 and 210 are 0.05 apart in log t but 10 apart in linear time, with
    # widths 2 ln 9 * 0.3 = 1.3: a plateau at the partial-fit level 0.1 lies between them.
    t = np.union1d(np.logspace(-2, 3, 400), np.linspace(0, 1000, 20001))
    plateaus, drops = find_plateaus(t, _steps(t, [10, 200, 210], [0.6, 0.3, 0.09], 0.3, 0.01))
    assert np.allclose([d.t_half for d in drops], [10, 200, 210], atol=0.02)
    assert np.allclose([d.t_fastest for d in drops], [10, 200, 210], atol=0.5)
    assert np.allclose([d.width for d in drops], 2 * np.log(9) * 0.3, rtol=0.02)
    assert np.allclose([d.loss_before for d in drops], [1.0, 0.4, 0.1], rtol=1e-3)
    assert np.allclose([d.loss_after for d in drops], [0.4, 0.1, 0.01], rtol=1e-3)
    assert np.allclose([p.level for p in plateaus], [1.0, 0.4, 0.1, 0.01], rtol=1e-3)
    for p, (a, b) in zip(plateaus, [(0, 10), (10, 200), (200, 210), (210, 1000)]):
        assert a <= p.t_start < p.t_end <= b and p.t_end - p.t_start > 0.6 * (b - a)
    # A log-only grid (dt ~ 6 at t = 200) cannot resolve that plateau.
    coarse = np.logspace(-2, 3, 400)
    plateaus, _ = find_plateaus(coarse, _steps(coarse, [10, 200, 210], [0.6, 0.3, 0.09], 0.3, 0.01))
    assert not any(abs(p.level - 0.1) < 0.02 for p in plateaus)
    assert any(abs(p.level - 0.4) < 0.01 for p in plateaus)


def test_find_plateaus_log_time_staircase():
    t = np.logspace(-2, 4, 3000)
    centers, sizes = [1.0, 30.0, 1000.0], [0.6, 0.3, 0.09]
    plateaus, drops = find_plateaus(t, _steps(np.log(t), np.log(centers), sizes, 0.03, 0.01))
    assert np.allclose([d.t_half for d in drops], centers, rtol=0.01)
    assert np.allclose([p.level for p in plateaus], [1.0, 0.4, 0.1, 0.01], rtol=1e-3)
    # Steps of log-width 0.15 are wide in linear time (rho ~ 1.4): drops, but no plateaus between
    # them, as the notes judge plateaus in linear time.
    plateaus, drops = find_plateaus(t, _steps(np.log(t), np.log(centers), sizes, 0.15, 0.01))
    assert np.allclose([d.t_half for d in drops], centers, rtol=0.02)
    assert np.all(separation_ratios([d.t_half for d in drops], [d.width for d in drops]) < 3)
    assert not any(p.t_start > 1.0 and p.t_end < 1000.0 for p in plateaus)


def test_find_plateaus_lazy_converged_flat_and_stalled():
    rng = np.random.default_rng(0)
    t = np.concatenate([[0.0], np.logspace(-3, 2, 300)])
    for loss in (np.exp(-2 * t), np.exp(-2 * t) + 1e-17 * rng.standard_normal(t.size)):
        plateaus, drops = find_plateaus(t, loss)  # converges below the floor: no final plateau
        assert plateaus == [] and len(drops) == 1
        assert drops[0].t_half == pytest.approx(np.log(2) / 2, rel=1e-3)
    assert find_plateaus(t, np.ones_like(t)) == ([Plateau(0.0, 100.0, 1.0)], [])
    t = np.linspace(0, 1000, 10001)
    plateaus, drops = find_plateaus(t, _steps(t, [10], [0.5], 0.3, 0.5))  # stuck after one drop
    assert len(drops) == 1 and np.allclose([p.level for p in plateaus], [1.0, 0.5])
    assert plateaus[-1].t_end == 1000.0


def test_find_plateaus_overlapping_drops_with_rounding_noise():
    # Loss values rounded to 1e-16 plus noise, as from the expanded population loss. The drops at
    # 2000 and 2004 (widths 4.4) overlap and count as one; the run converges to 0.
    rng = np.random.default_rng(1)
    t = np.union1d(np.logspace(-2, 4, 2000), np.linspace(0, 3000, 30001))
    clean = _steps(t, [1000, 2000, 2004], [0.6, 0.3, 0.1], 1.0, 0.0)
    noisy = np.round(clean / 1e-16) * 1e-16 + 1e-16 * rng.standard_normal(t.size)
    plateaus, drops = find_plateaus(t, noisy)
    assert len(drops) == 2 and drops[0].t_half == pytest.approx(1000, abs=0.1)
    assert 2000 < drops[1].t_half < 2004 and drops[1].loss_after < 1e-11
    assert np.allclose([p.level for p in plateaus], [1.0, 0.4], rtol=1e-6)


# ---- kernels ---------------------------------------------------------------------------------


def _model(kind="binary", n=4, d=2, chi=2, seed=0):
    spec = TTNSpec.uniform(make_tree(kind, n), d, chi)
    return spec, spec.flatten(spec.init(1.0, seed=seed))


def _fd_jacobian(fn, theta, eps=1e-6):
    cols = [(fn(theta + eps * e) - fn(theta - eps * e)) / (2 * eps) for e in torch.eye(len(theta))]
    return torch.stack(cols, dim=-1)


@pytest.mark.parametrize("kind", ["binary", "mps", "ht"])
def test_tangent_kernel_matches_finite_differences(kind):
    spec, theta = _model(kind, chi=3)
    x = sample_inputs(6, 4, 2, torch.Generator().manual_seed(1))
    jac = _fd_jacobian(lambda th: forward(spec, spec.unflatten(th), x), theta)
    assert torch.allclose(tangent_kernel(spec, theta, x), jac @ jac.T, atol=1e-8)
    assert torch.allclose(tangent_kernel(spec, spec.unflatten(theta), x), jac @ jac.T, atol=1e-8)
    mask = (torch.rand(spec.n_params, generator=torch.Generator().manual_seed(2)) > 0.3).double()
    assert torch.allclose(tangent_kernel(spec, theta, x, mask), (jac * mask) @ jac.T, atol=1e-8)


@pytest.mark.parametrize("kind", ["binary", "mps", "ht"])
def test_tensor_tangent_kernel_consistent_with_probe_kernel(kind):
    spec, theta = _model(kind, chi=3, seed=3)
    x = sample_inputs(10, 4, 2, torch.Generator().manual_seed(4))
    phi = product_features(x)
    w = to_dense(spec, spec.unflatten(theta)).reshape(-1)
    assert torch.allclose(phi @ w, forward(spec, spec.unflatten(theta), x), atol=1e-12)
    k_tensor = tensor_tangent_kernel(spec, theta)
    assert k_tensor.shape == (16, 16)
    assert torch.allclose(phi @ k_tensor @ phi.T, tangent_kernel(spec, theta, x), atol=1e-10)
    mask = torch.zeros(spec.n_params, dtype=torch.float64)
    mask[spec.core_slices()[spec.tree.root]] = 1.0  # only the root trains
    expected = tangent_kernel(spec, theta, x, mask)
    assert torch.allclose(phi @ tensor_tangent_kernel(spec, theta, mask) @ phi.T, expected)


def _unit_tensor_jacobian(spec, theta):
    """Reference: the previous `tensor_jacobian`, one dense contraction per unit core tensor. It
    allocates eye(core size), so small cores only."""
    cores = spec.unflatten(theta)
    cols = []
    for v, shape in enumerate(spec.shapes):
        for unit in torch.eye(spec.sizes[v], dtype=theta.dtype):
            cols.append(to_dense(spec, cores[:v] + [unit.view(shape)] + cores[v + 1 :]).reshape(-1))
    return torch.stack(cols, dim=1)


def _rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def _per_core(spec, values):
    """Flat vector equal to values[v] on core v (per-core learning-rate multipliers)."""
    out = torch.empty(spec.n_params, dtype=torch.float64)
    for block, value in zip(spec.core_slices(), values):
        out[block] = value
    return out


_TREES = {
    **{f"{kind}{n}": make_tree(kind, n) for kind in ("binary", "mps", "ht") for n in (4, 5)},
    # a root with its own site and three children (one with two sites); a single core
    "star": Tree((3, 3, 3, -1), ((0,), (2,), (1, 4), (3,))),
    "one-core": Tree((-1,), ((0, 1, 2),)),
}


@pytest.mark.parametrize("name", list(_TREES))
@pytest.mark.parametrize("d,chi", [(2, 3), (3, 2)])
def test_tensor_kernel_matches_unit_tensor_construction(name, d, chi):
    spec = TTNSpec.uniform(_TREES[name], d, chi)
    theta = spec.flatten(spec.init(1.0, seed=5))
    jac = _unit_tensor_jacobian(spec, theta)
    assert _rel_err(tensor_jacobian(spec, theta), jac) < 1e-13
    g = torch.Generator().manual_seed(6)
    masks = {
        "none": None,
        "lr": _per_core(spec, [1.0 / f for f in spec.fan_in]),  # constant on each core
        "random": 0.1 + torch.rand(spec.n_params, generator=g, dtype=torch.float64),
        "0/1": (torch.rand(spec.n_params, generator=g) > 0.4).double(),
    }
    if spec.n_cores > 1:
        masks["frozen root"] = _per_core(spec, [v != spec.tree.root for v in range(spec.n_cores)])
    for key, m in masks.items():
        expected = jac @ jac.T if m is None else (jac * m) @ jac.T
        assert _rel_err(tensor_tangent_kernel(spec, theta, m), expected) < 1e-13, key
    m = masks["random"]
    k = tensor_tangent_kernel(spec, theta, m)
    assert _rel_err(tensor_tangent_kernel(spec, spec.unflatten(theta), m.numpy()), k) < 1e-14
    with pytest.raises(ValueError):
        tensor_tangent_kernel(spec, theta, torch.ones(spec.n_params + 1, dtype=torch.float64))


@pytest.mark.parametrize("name", [n for n in _TREES if n != "one-core"])
def test_tensor_kernel_two_copy_and_centred_match_unit_tensor_construction(name):
    spec = TTNSpec.uniform(_TREES[name], 2, 2)
    cores, cores0 = spec.init(1.0, seed=7), spec.init(1.0, seed=8)
    # Two-copy model: the gradient mask of antisymmetric_double, alone and times learning rates
    # (a problem's precond).
    spec2, cores2, mask = antisymmetric_double(spec, cores)
    theta2 = spec2.flatten(cores2)
    jac2 = _unit_tensor_jacobian(spec2, theta2)
    assert _rel_err(tensor_jacobian(spec2, theta2), jac2) < 1e-13
    for w in (mask, mask * _per_core(spec2, [1.0 / f for f in spec.fan_in])):
        assert _rel_err(tensor_tangent_kernel(spec2, theta2, w), (jac2 * w) @ jac2.T) < 1e-13
    # Centred model W(theta) - W(theta0) on doubled bonds (two_copy_cores, as in Problem.effective)
    # with theta0 fixed: only the "+" blocks train, and the kernel is the plain one at theta.
    spec_c, cores_c = two_copy_cores(spec, cores, cores0)
    theta_c = spec_c.flatten(cores_c)
    ones, zeros = [torch.ones_like(c) for c in cores], [torch.zeros_like(c) for c in cores]
    plus = spec_c.flatten(two_copy_cores(spec, ones, zeros)[1])
    jac_c = _unit_tensor_jacobian(spec_c, theta_c)
    k_c = tensor_tangent_kernel(spec_c, theta_c, plus)
    assert _rel_err(k_c, (jac_c * plus) @ jac_c.T) < 1e-13
    assert _rel_err(k_c, tensor_tangent_kernel(spec, spec.flatten(cores))) < 1e-13


class _LargestAllocation(TorchDispatchMode):
    """Largest storage (bytes) produced by any aten op inside the context (tracemalloc does not see
    torch's CPU allocator)."""

    def __init__(self):
        super().__init__()
        self.nbytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if isinstance(t, torch.Tensor):
                self.nbytes = max(self.nbytes, t.untyped_storage().nbytes())
        return out


def test_tensor_kernel_and_jacobian_need_no_quadratic_memory():
    # Regime-map chi_b4 point: two-copy binary4, d = 3, uncapped chi = 128 (doubled bonds 256). The
    # unit-tensor construction allocated eye(core size) for the 256 x 256 root: 34 GB.
    spec = TTNSpec.uniform(make_tree("binary", 4), 3, 128, cap=False)
    spec2, cores2, mask = antisymmetric_double(spec, spec.init(1.0, seed=0))
    theta = spec2.flatten(cores2)
    w = mask * _per_core(spec2, [1.0 / f for f in spec.fan_in])
    assert 8 * max(spec2.sizes) ** 2 > 3e10
    with _LargestAllocation() as alloc_k:
        k = tensor_tangent_kernel(spec2, theta, w)
    with _LargestAllocation() as alloc_j:
        jac = tensor_jacobian(spec2, theta)
    assert alloc_k.nbytes <= 8 * max(spec2.n_params, k.numel())  # nothing larger than theta or K
    assert alloc_j.nbytes <= 8 * jac.numel()  # nothing larger than J itself
    assert _rel_err(k, (jac * w) @ jac.T) < 1e-13
    x = sample_inputs(12, 4, 3, torch.Generator().manual_seed(1))
    phi = product_features(x)
    assert _rel_err(phi @ k @ phi.T, tangent_kernel(spec2, theta, x, w)) < 1e-12


@pytest.mark.parametrize("metrics", [False, True], ids=["whitened", "power-law"])
def test_projected_fit_loss_is_the_least_squares_residual(metrics):
    # chi = 1 on binary4 (d = 2): the tangent space misses part of a generic residual.
    spec = TTNSpec.uniform(make_tree("binary", 4), 2, 1)
    theta = spec.flatten(spec.init(1.0, seed=3))
    r = torch.randn(2, 2, 2, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    mask = torch.ones(spec.n_params, dtype=torch.float64)
    mask[:2] = 0.0  # two frozen coordinates
    site = [power_law_cov(2, 1.0)] * 4 if metrics else None
    jac = tensor_jacobian(spec, theta)[:, mask != 0].numpy()
    m_half = np.eye(1)
    for m in site or [torch.eye(2, dtype=torch.float64)] * 4:
        vals, vecs = np.linalg.eigh(m.numpy())
        m_half = np.kron(m_half, vecs @ np.diag(np.sqrt(vals)) @ vecs.T)
    b = m_half @ r.numpy().ravel()
    x = np.linalg.lstsq(m_half @ jac, b, rcond=None)[0]
    expected = 0.5 * np.sum((m_half @ jac @ x - b) ** 2)
    floor = projected_fit_loss(spec, theta, r, mask, site)
    assert floor == pytest.approx(expected, rel=1e-9) and 0.05 * np.sum(b**2) < floor
    full = TTNSpec.uniform(make_tree("binary", 4), 2, 4)  # full bonds: fits anything
    theta = full.flatten(full.init(1.0, seed=3))
    assert projected_fit_loss(full, theta, r, site_metrics=site) < 1e-20


def test_kernel_alignment_and_change():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(10)
    assert kernel_alignment(np.outer(y, y), y) == pytest.approx(1.0)
    assert kernel_alignment(np.eye(10), y) == pytest.approx(1 / np.sqrt(10))
    k = rng.standard_normal((10, 10))
    k = k @ k.T
    h = np.eye(10) - 1 / 10
    kc, yc = h @ k @ h, h @ np.outer(y, y) @ h
    cortes = np.sum(kc * yc) / (np.linalg.norm(kc) * np.linalg.norm(yc))
    assert centered_kernel_alignment(k, y) == pytest.approx(cortes)
    assert kernel_alignment(k + 5.0, y + 3.0, centered=True) == pytest.approx(cortes)
    assert kernel_change(torch.tensor(2 * k), torch.tensor(k)) == pytest.approx(1.0)
    assert relative_change([3.0, 4.0], [0.0, 1.0]) == pytest.approx(np.sqrt(18))


# ---- Hessian ---------------------------------------------------------------------------------


def _loss(spec, teacher):
    return lambda th: population_loss(spec, spec.unflatten(th), teacher)


def _grad(loss_fn):
    def grad(th):
        th = th.detach().requires_grad_(True)
        return torch.autograd.grad(loss_fn(th), th)[0]

    return grad


def test_hessian_and_hvp_match_finite_differences():
    spec, theta = _model("mps", chi=2, seed=5)
    loss_fn = _loss(spec, random_cp_teacher(4, 2, [1.0, 0.6], seed=1))
    fd = _fd_jacobian(_grad(loss_fn), theta)
    h = hessian(loss_fn, theta)
    assert torch.allclose(h, fd, atol=1e-7) and torch.equal(h, h.T)
    v = torch.randn(spec.n_params, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    assert torch.allclose(hvp(loss_fn, theta, v), h @ v, atol=1e-12)
    assert torch.allclose(hessian_eigs(loss_fn, theta), torch.linalg.eigvalsh(fd), atol=1e-6)


def test_lanczos_extremes_match_full_spectrum():
    spec, theta = _model("binary", chi=3, seed=7)
    loss_fn = _loss(spec, random_cp_teacher(4, 2, [1.0, 0.6], seed=2))
    eigs = hessian_eigs(loss_fn, theta).numpy()
    both = lanczos_extremes(loss_fn, theta, k=4)
    assert np.allclose(both, np.concatenate([eigs[:4], eigs[-4:]]), atol=1e-8)
    vals, vecs = lanczos_extremes(loss_fn, theta, k=3, which="smallest", vectors=True)
    assert np.allclose(vals, eigs[:3], atol=1e-8)
    h = hessian(loss_fn, theta).numpy()
    assert np.allclose(h @ vecs, vecs * vals, atol=1e-7)
    assert np.allclose(lanczos_extremes(loss_fn, theta, k=2, which="largest"), eigs[-2:])


def test_saddle_type():
    s = saddle_type(np.array([-0.5, 0.0, 1e-6, 2.0]))
    assert s["type"] == "strict" and s["n_negative"] == 1
    assert s["eps"] == pytest.approx(2e-3) and s["ratio"] == pytest.approx(-0.25)
    s = saddle_type(torch.tensor([-1e-5, 0.0, 2.0], dtype=torch.float64))
    assert s["type"] == "non-strict" and s["n_negative"] == 0 and s["lambda_min"] == -1e-5
    # NaN padding (a recorder's hessian_eigs_gauge row) is ignored, not propagated.
    s = saddle_type(np.array([-0.5, 0.0, 2.0, np.nan, np.nan]))
    assert s["type"] == "strict" and s["n_negative"] == 1 and s["lambda_max"] == 2.0


def test_saddle_scaling():
    rng = np.random.default_rng(0)
    sigma = np.repeat([0.08, 0.04, 0.02, 0.01], 3)
    noise = np.exp(0.05 * rng.standard_normal(sigma.size))
    strict = saddle_scaling(sigma, -0.5 * noise)  # lambda_min -> constant
    assert strict["type"] == "strict" and abs(strict["exponent"]) < 0.1 and strict["n"] == 12
    order3 = saddle_scaling(sigma, -2.0 * sigma * noise)  # k = 3 route: ~sigma^1
    assert order3["type"] == "higher-order" and order3["order"] == pytest.approx(3, abs=0.1)
    lam = -2.0 * sigma * noise
    lam[0] = 0.3  # non-negative values are skipped
    assert saddle_scaling(sigma, lam)["n"] == 11
    assert saddle_scaling([0.1, 0.1], [-1.0, -2.0])["type"] == "undetermined"


# ---- gauge -----------------------------------------------------------------------------------


@pytest.mark.parametrize("kind,n", [("binary", 4), ("mps", 4), ("ht", 4), ("binary", 6)])
def test_gauge_directions_keep_w_fixed_and_are_independent(kind, n):
    spec, theta = _model(kind, n=n, chi=3, seed=8)
    u = gauge_directions(spec, theta)
    n_gauge = sum(spec.bond_dims[v] ** 2 for v in spec.tree.edges)
    assert u.shape == (spec.n_params, n_gauge)
    assert torch.linalg.matrix_rank(u).item() == n_gauge
    u = u / u.norm(dim=0)

    def w_of(th):
        return to_dense(spec, spec.unflatten(th)).reshape(-1)

    # dW along each direction vanishes: a finite step changes W only at second order.
    w0 = w_of(theta)
    for eps in (1e-3, 1e-4):
        change = max((w_of(theta + eps * u[:, j]) - w0).abs().max().item() for j in range(n_gauge))
        assert change < 10 * eps**2
    generic = torch.randn(spec.n_params, dtype=torch.float64, generator=torch.Generator())
    assert (w_of(theta + 1e-4 * generic) - w0).abs().max() > 1e-6


@pytest.mark.parametrize("kind", ["binary", "mps", "ht"])
def test_gauge_directions_in_hessian_null_space_at_global_minimum(kind):
    n, d, weights = 4, 3, [2.0, 1.0, 0.5]
    teacher = odeco_teacher(n, d, weights, seed=0)
    spec = TTNSpec.uniform(make_tree(kind, n), d, 3)
    amps = torch.ones(spec.n_cores, 3, dtype=torch.float64)
    amps[spec.tree.root] = torch.tensor(weights, dtype=torch.float64)
    theta = spec.flatten(aligned_init(spec, teacher.factors, amps))
    loss_fn = _loss(spec, teacher)
    assert abs(loss_fn(theta).item()) < 1e-12  # W = W*
    h = hessian(loss_fn, theta)
    u = gauge_directions(spec, theta)
    assert (h @ u).abs().max() < 1e-10 * h.abs().max()
    h_perp, eigs = project_out_gauge(h, u)
    rank = torch.linalg.matrix_rank(u).item()
    assert h_perp.shape == (spec.n_params - rank,) * 2 and eigs.min() > -1e-10
    # Away from a critical point the gauge directions are not Hessian zero modes.
    off = theta + 0.1 * torch.randn(spec.n_params, dtype=torch.float64)
    assert (hessian(loss_fn, off) @ gauge_directions(spec, off)).abs().max() > 1e-3


def test_project_out_gauge_on_known_matrix():
    q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
    lam = torch.tensor([-3.0, -1.0, 0.5, 2.0, 4.0, 7.0], dtype=torch.float64)
    h = q @ torch.diag(lam) @ q.T
    u = torch.cat([q[:, :2], q[:, :1] + q[:, 1:2]], dim=1)  # rank 2: spans the first two eigvecs
    c = gauge_complement(u)
    assert c.shape == (6, 4) and torch.allclose(c.T @ c, torch.eye(4, dtype=torch.float64))
    assert (c.T @ u).abs().max() < 1e-12
    h_perp, eigs = project_out_gauge(h, u)
    assert torch.allclose(eigs, lam[2:], atol=1e-12)
    assert torch.allclose(c @ h_perp @ c.T, h - q[:, :2] @ torch.diag(lam[:2]) @ q[:, :2].T)


def _k2_learned_saddle(s1, s2):
    """MPS4 (d = chi = 2), pattern teacher whose second component differs from the first on sites
    0 and 1 (a k = 2 route through cores 0, 1), and its learned saddle s1 U1 in balanced gauge."""
    spec = TTNSpec.uniform(make_tree("mps", 4), 2, 2)
    teacher = pattern_teacher([s1, s2], [[0, 0, 0, 0], [1, 1, 0, 0]], 2, seed=1000)
    amps = torch.full((spec.n_cores, 1), s1 ** (1 / spec.n_cores), dtype=torch.float64)
    return spec, teacher, spec.flatten(aligned_init(spec, teacher.factors[:1], amps))


def _dormant_noise(theta, eta, seed):
    """theta plus N(0, eta^2) noise on its zero entries (here: the unused bond index)."""
    z = torch.randn(theta.shape, generator=torch.Generator().manual_seed(seed), dtype=theta.dtype)
    return theta + eta * z * (theta == 0)


@pytest.mark.parametrize("s1,s2", [(1.0, 0.5), (2.0, 0.25)])
def test_gauge_projection_unbiased_near_k2_learned_saddle(s1, s2):
    # Each bond's unused index has a gauge generator of size ~eta. The old cut (rtol 1e-10) projected
    # it out, removing part of the escape direction (notes/reviews/locality_round1.md, B3).
    spec, teacher, saddle = _k2_learned_saddle(s1, s2)
    k = spec.n_cores
    expected = -s2 * s1 ** ((k - 2) / k)
    loss_fn = _loss(spec, teacher)
    biases = []
    for eta in (1e-6, 1e-5, 1e-4):
        for seed in range(4):
            theta = _dormant_noise(saddle, eta, seed)
            h, u = hessian(loss_fn, theta), gauge_directions(spec, theta)
            raw = torch.linalg.eigvalsh(h)[0].item()
            assert raw == pytest.approx(expected, abs=1e-6)
            assert project_out_gauge(h, u)[1][0].item() == pytest.approx(raw, abs=1e-6)
            # Only the three tiny generators (one per bond) are dropped.
            assert gauge_complement(u).shape[1] == spec.n_params - u.shape[1] + len(spec.tree.edges)
            biases.append(project_out_gauge(h, u, rtol=1e-10)[1][0].item() - raw)
    assert max(biases) > 0.1 * abs(expected)


def test_runner_gauge_projected_eigs_unbiased_with_frozen_core():
    # The runner's recorded spectra use project_out_gauge's default cut, applied after its selection
    # of the gauge directions that keep the untrained (here: frozen root) coordinates fixed.
    from ttn.dynamics import Problem
    from ttn.runner import gauge_projected_eigs

    spec, teacher, saddle = _k2_learned_saddle(1.0, 0.5)
    theta = _dormant_noise(saddle, 1e-4, seed=0)
    frozen = torch.ones(spec.n_params, dtype=torch.float64)
    frozen[spec.core_slices()[spec.tree.root]] = 0.0  # the route (cores 0, 1) stays trained
    for mask, n_removed in ((None, 12 - 3), (frozen, 8 - 2)):
        raw, gauge = gauge_projected_eigs(Problem(spec, teacher, mask=mask), theta)
        assert raw[0] == pytest.approx(-0.5, abs=1e-6) and gauge[0] == pytest.approx(-0.5, abs=1e-6)
        assert np.isnan(gauge).sum() == n_removed


@pytest.mark.parametrize("chi", [3, 4])
@pytest.mark.parametrize("kind", ["binary", "mps", "ht"])
def test_projected_spectrum_at_global_minimum_has_only_genuine_zero_modes(kind, chi):
    # W = W* (odeco teacher, aligned init), so H = J^T J and its zero modes are null(J): the gauge
    # directions plus, with chi = 4 > R = 3, the flat directions of the unused bond index.
    n, d, weights = 4, 3, [2.0, 1.0, 0.5]
    teacher = odeco_teacher(n, d, weights, seed=0)
    spec = TTNSpec.uniform(make_tree(kind, n), d, chi)
    amps = torch.ones(spec.n_cores, 3, dtype=torch.float64)
    amps[spec.tree.root] = torch.tensor(weights, dtype=torch.float64)
    theta = spec.flatten(aligned_init(spec, teacher.factors, amps))
    h, u = hessian(_loss(spec, teacher), theta), gauge_directions(spec, theta)
    c = gauge_complement(u)
    removed = torch.eye(spec.n_params, dtype=torch.float64) - c @ c.T  # projector on the cut span
    assert (h @ removed).abs().max() < 1e-12 * h.abs().max()  # well-conditioned directions are null
    rank_u = torch.linalg.matrix_rank(u).item()
    assert c.shape[1] == spec.n_params - rank_u  # the cut drops only exact dependencies here
    n_flat = spec.n_params - torch.linalg.matrix_rank(tensor_jacobian(spec, theta)).item()
    _, eigs = project_out_gauge(h, u)
    zero = eigs.abs() < 1e-10 * eigs.abs().max()
    assert int(zero.sum()) == n_flat - rank_u and (n_flat > rank_u) == (chi > 3)
    assert eigs[~zero].min() > 0.2
