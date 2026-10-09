import numpy as np
import pytest
import torch
from scipy.special import expit

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
    relative_change,
    saddle_type,
    separation_ratios,
    smallest_nonzero,
    tangent_kernel,
    tensor_tangent_kernel,
)
from ttn.losses import population_loss, sample_inputs
from ttn.network import TTNSpec, aligned_init, forward, to_dense
from ttn.teachers import odeco_teacher, random_cp_teacher
from ttn.tree import make_tree

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
