import dataclasses
import itertools
import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from ttn.dynamics import (
    Problem,
    Recorder,
    Trajectory,
    integrate_gf,
    make_dataset,
    run_gd,
    run_sgd,
)
from ttn.losses import power_law_cov
from ttn.network import (
    TTNSpec,
    aligned_init,
    antisymmetric_double,
    conserved_quantities,
    to_dense,
)
from ttn.teachers import ShiftedTeacher, odeco_teacher, random_cp_teacher
from ttn.tree import binary_tree, mps_tree

# Every tensor here is tiny: multithreaded LAPACK calls (QR in bond_spectra, Cholesky) cost
# milliseconds instead of microseconds on a loaded machine.
torch.set_num_threads(1)


def random_problem(tree, d=2, chi=2, sigma=0.8, seed=0, weights=(1.0, 0.6), **kwargs):
    spec = TTNSpec.uniform(tree, d, chi)
    teacher = random_cp_teacher(tree.n_sites, d, list(weights), seed=seed)
    problem = Problem(spec, teacher, **kwargs)
    return problem, spec.flatten(spec.init(sigma, seed=seed + 100))


def aligned_problem(tree, weights, a0s, d=3):
    """Odeco teacher with an aligned balanced init: amplitude a0s[r] on every core for component r."""
    teacher = odeco_teacher(tree.n_sites, d, weights, seed=1)
    spec = TTNSpec.uniform(tree, d, 3)
    amps = torch.tensor(a0s, dtype=torch.float64).expand(spec.n_cores, len(weights))
    return Problem(spec, teacher), spec.flatten(aligned_init(spec, teacher.factors, amps))


def scalar_ode_overlaps(t, k, weights, a0s):
    """Every core amplitude obeys a' = a^{K-1}(s - a^K); the component overlaps are a^K / s."""
    out = []
    for s, a0 in zip(weights, a0s):
        sol = solve_ivp(
            lambda _, a, s=s: a ** (k - 1) * (s - a**k),
            (0.0, t[-1]),
            [a0],
            t_eval=t,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        out.append(sol.y[0] ** k / s)
    return np.stack(out, axis=1)


# ---- (1) exactness against the scalar ODE of the aligned init ---------------------------------


@pytest.mark.parametrize("tree", [binary_tree(4), mps_tree(4)], ids=["binary4", "mps4"])
def test_aligned_init_follows_scalar_ode(tree):
    weights, a0s = [2.0, 1.0], [0.3, 0.3]
    problem, theta0 = aligned_problem(tree, weights, a0s)
    traj = integrate_gf(problem, theta0, t_max=30.0, n_log=80)
    assert traj.meta["stop_reason"] == "completed"
    expected = scalar_ode_overlaps(traj["t"], problem.spec.n_cores, weights, a0s)
    assert np.abs(traj["overlaps"] - expected).max() < 1e-6
    assert np.allclose(traj["overlaps"][-1], 1.0, atol=1e-6)


# ---- (2) conservation laws ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "tree,lr_multipliers",
    [(binary_tree(4), None), (mps_tree(4), [1.0, 0.5, 2.0, 1.5])],
    ids=["binary4", "mps4-lr-multipliers"],
)
def test_conservation_laws(tree, lr_multipliers):
    problem, theta0 = random_problem(tree, chi=3, sigma=0.5, seed=1, lr_multipliers=lr_multipliers)
    traj = integrate_gf(problem, theta0, t_max=200.0, n_log=60)
    loss = traj["loss"]
    assert loss[-1] < 0.05 * loss[0]
    assert traj["conservation_drift"].max() < 1e-7
    # The recorded drift is the documented quantity, recomputed from the final parameters.
    spec, eta = problem.spec, lr_multipliers or [1.0] * problem.spec.n_cores
    q0 = conserved_quantities(spec, spec.unflatten(theta0), lr_multipliers)
    cores = spec.unflatten(torch.from_numpy(traj["theta_end"]))
    q1 = conserved_quantities(spec, cores, lr_multipliers)
    change = max((q1[v] - q0[v]).abs().max().item() for v in q0)
    scale = max(q.abs().max().item() for q in q0.values())
    scale += max(c.pow(2).sum().item() / e for c, e in zip(cores, eta))
    assert math.isclose(traj["conservation_drift"][-1], change / scale, rel_tol=1e-9)


# ---- (3) agreement with a fixed-step RK4 reference -----------------------------------------------


def rk4(problem, theta0, h, n_steps, every):
    y, out = theta0.clone(), [theta0.clone()]
    for step in range(1, n_steps + 1):
        k1 = problem.velocity(y)
        k2 = problem.velocity(y + 0.5 * h * k1)
        k3 = problem.velocity(y + 0.5 * h * k2)
        k4 = problem.velocity(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if step % every == 0:
            out.append(y.clone())
    return torch.stack(out).numpy()


def test_matches_rk4_reference():
    """Anisotropic inputs, lr multipliers and a non-orthogonal teacher, short horizon."""
    metrics = [power_law_cov(2, 1.0)] * 4
    problem, theta0 = random_problem(
        mps_tree(4), sigma=0.9, seed=2, site_metrics=metrics, lr_multipliers=[1.0, 0.5, 2.0, 1.0]
    )
    h, n_steps, every = 2.0**-8, 2**8, 2**5  # times 1/8, 2/8, ..., 1 are exact in binary
    reference = rk4(problem, theta0, h, n_steps, every)
    log_times = np.arange(1, 9) / 8
    traj = integrate_gf(
        problem, theta0, t_max=1.0, log_times=log_times, recorder=Recorder(save_params=True)
    )
    assert np.array_equal(traj["t"], np.concatenate([[0.0], log_times]))
    assert np.abs(traj["params"] - reference).max() < 1e-8 * np.abs(reference).max()
    assert traj["rel_dist_from_init"][-1] > 0.05  # the parameters did move


# ---- (4) gradient descent converges to gradient flow ---------------------------------------------


def test_gd_tracks_gf_with_first_order_error():
    problem, theta0 = random_problem(binary_tree(4), sigma=0.9, seed=3)
    t_end = 1.0
    gf = integrate_gf(problem, theta0, t_max=t_end, log_times=[t_end], rtol=1e-11, atol=1e-14)
    errors = []
    for lr in (0.02, 0.01, 0.005):
        gd = run_gd(problem, theta0, lr, n_steps=round(t_end / lr), n_log=20)
        assert gd.meta["stop_reason"] == "completed"
        assert math.isclose(gd["t"][-1], t_end)
        errors.append(np.abs(gd["theta_end"] - gf["theta_end"]).max())
    assert errors[0] < 0.05 * np.abs(gf["theta_end"]).max()
    for coarse, fine in itertools.pairwise(errors):
        assert 1.7 < coarse / fine < 2.3


# ---- (5) monotone loss ------------------------------------------------------------------------


def test_population_loss_non_increasing():
    problem, theta0 = random_problem(mps_tree(4), chi=2, sigma=0.4, seed=4, weights=(1.0, 0.7))
    traj = integrate_gf(problem, theta0, t_max=150.0, n_log=100)
    loss = traj["loss"]
    assert loss[-1] < 0.01 * loss[0]
    # Roundoff of the expanded loss |W|^2 - 2<W,W*> + |W*|^2 is ~1e-16 |W*|^2.
    assert np.all(np.diff(loss) <= 1e-13 * loss[0])


# ---- (6) antisymmetric doubling with the gradient mask -------------------------------------------


def split_copies(spec, cores2):
    """(theta+, theta-) from the block-diagonal cores of `antisymmetric_double`."""
    t = spec.tree
    plus, minus = [], []
    for v in range(t.n_nodes):
        blocks = []
        for block in (0, 1):
            idx = [slice(None)] * len(t.sites[v])
            for c in t.children[v]:
                chi = spec.bond_dims[c]
                idx.append(slice(block * chi, (block + 1) * chi))
            chi = spec.bond_dims[v]
            idx.append(slice(None) if v == t.root else slice(block * chi, (block + 1) * chi))
            blocks.append(cores2[v][tuple(idx)])
        plus.append(blocks[0])
        minus.append(-blocks[1] if v == t.root else blocks[1])
    return plus, minus


def test_antisymmetric_double_with_mask():
    spec = TTNSpec.uniform(binary_tree(4), 2, 2)
    teacher = random_cp_teacher(4, 2, [1.0, 0.5], seed=5)
    spec2, cores2, mask = antisymmetric_double(spec, spec.init(1.0, seed=5))
    problem = Problem(spec2, teacher, mask=mask)
    assert problem.two_copy  # detected from the mask
    traj = integrate_gf(
        problem, spec2.flatten(cores2), t_max=50.0, n_log=40, recorder=Recorder(save_params=True)
    )
    params = traj["params"]
    assert np.all(params[:, mask.numpy() == 0] == 0.0)
    assert traj["loss"][0] == pytest.approx(0.5 * teacher.norm2().item(), rel=1e-12)  # W_0 = 0
    assert traj["loss"][-1] < 0.05 * traj["loss"][0]
    assert traj["conservation_drift"].max() < 1e-7  # per-copy Q
    for theta in params[::5]:
        cores = spec2.unflatten(torch.from_numpy(theta))
        plus, minus = split_copies(spec, cores)
        two_copy = to_dense(spec, plus) - to_dense(spec, minus)
        assert torch.allclose(to_dense(spec2, cores), two_copy, atol=1e-12)
    plus, minus = split_copies(spec, spec2.unflatten(torch.from_numpy(params[-1])))
    assert (spec.flatten(plus) - spec.flatten(minus)).abs().max() > 0.1  # the copies separated
    # Same flow with the plain (cancelling) loss and the mask alone.
    plain = Problem(spec2, teacher, mask=mask, two_copy=False)
    traj2 = integrate_gf(plain, spec2.flatten(cores2), t_max=50.0, log_times=traj["t"][1:])
    assert np.abs(traj2["theta_end"] - traj["theta_end"]).max() < 1e-6


def test_cancellation_free_losses_at_large_init():
    """At sigma = 10, K = 7, ||W(theta0)||^2 ~ 1e14: the plain expanded loss has absolute error
    ~1e-2, while the two-copy and centred problems evaluate 1/2||W*||^2 at theta0 exactly."""
    spec = TTNSpec.uniform(binary_tree(8), 2, 2)
    base = random_cp_teacher(8, 2, [1.0, 0.5], seed=1)
    cores0 = spec.init(10.0, seed=2)
    target = 0.5 * base.norm2().item()
    spec2, cores2, mask = antisymmetric_double(spec, cores0)
    two_copy = Problem(spec2, base, mask=mask)
    assert two_copy.loss(spec2.flatten(cores2)).item() == pytest.approx(target, rel=1e-12)
    centred = Problem(spec, ShiftedTeacher(base, spec, cores0))
    theta0 = spec.flatten(cores0)
    assert centred.loss(theta0).item() == pytest.approx(target, rel=1e-12)
    rec = Recorder()
    rec.start(centred, theta0)
    rec.record(centred, theta0, 0.0)
    data = rec.arrays()
    assert data["loss"][0] == pytest.approx(target, rel=1e-12)
    assert np.abs(data["overlaps"]).max() < 1e-12  # overlaps and spectra of W - W_0 = 0
    assert np.nanmax(data["spectra"]) < 1e-12


# ---- (7) saddle-to-saddle at small init ------------------------------------------------------------


def test_small_init_three_step_loss_curve():
    weights = [3.0, 2.0, 1.0]
    spec = TTNSpec.uniform(binary_tree(4), 3, 3)
    teacher = odeco_teacher(4, 3, weights, seed=0)
    theta0 = spec.flatten(spec.init(0.05, seed=0))
    traj = integrate_gf(Problem(spec, teacher), theta0, t_max=1e4, loss_floor=1e-10)
    meta = traj.meta
    print(
        f"\nsmall init: {meta['n_steps']} steps ({meta['n_rejected']} rejected, "
        f"{meta['n_evals']} gradient evaluations), {meta['wall_time']:.2f} s, "
        f"stopped at t = {meta['t_end']:.1f} ({meta['stop_reason']})"
    )
    assert meta["stop_reason"] == "converged"
    t, overlaps, loss = traj["t"], traj["overlaps"], traj["loss"]
    crossings = []
    for r in range(len(weights)):
        above = overlaps[:, r] >= 0.5
        assert above.any()
        crossings.append(t[np.argmax(above)])
    order = np.argsort(crossings)
    assert len(set(crossings)) == 3
    # Between consecutive crossings the loss sits on the saddle where the learned components are fit.
    for i in range(2):
        learned = order[: i + 1]
        mid = math.sqrt(crossings[order[i]] * crossings[order[i + 1]])
        saddle = 0.5 * sum(weights[r] ** 2 for r in range(3) if r not in learned)
        assert np.interp(mid, t, loss) == pytest.approx(saddle, rel=0.05)
    # Runtime: the step count is machine-independent (wall time is printed above).
    assert meta["n_steps"] < 1000


# ---- implicit solvers for stiff later plateaus ------------------------------------------------------


def test_implicit_methods_on_a_stiff_later_plateau():
    """Component 0 (s = 4) is learned at t ~ 1, after which the Hessian has an eigenvalue ~19;
    component 1 (s = 0.05) escapes only at t ~ 1 / (s a0) = 200. DP5 is then stability-limited,
    while LSODA (switching to BDF) and Radau take large steps and stay exact."""
    weights, a0s = [4.0, 0.05], [0.3, 0.1]
    problem, theta0 = aligned_problem(binary_tree(4), weights, a0s)
    k = problem.spec.n_cores
    t_max, log_times = 400.0, np.geomspace(0.1, 400.0, 15)
    runs = {}
    for method, tol in (("lsoda", 1e-6), ("radau", 1e-8)):
        traj = runs[method] = integrate_gf(
            problem, theta0, t_max, log_times=log_times, method=method
        )
        assert traj.meta["stop_reason"] == "completed" and traj.meta["n_steps"] < 1000
        assert np.array_equal(traj["t"], np.concatenate([[0.0], log_times]))  # hit exactly
        expected = scalar_ode_overlaps(traj["t"], k, weights, a0s)
        assert np.abs(traj["overlaps"] - expected).max() < tol
        assert np.allclose(traj["overlaps"][-1], 1.0, atol=1e-5)
        assert traj["conservation_drift"].max() < 1e-9
    assert 0 < runs["lsoda"].meta["stiff_at"] < 10  # switched to BDF after the first component
    assert runs["radau"].meta["stiff_at"] is None
    dp5 = integrate_gf(problem, theta0, t_max, log_times=log_times, max_steps=150)
    assert dp5.meta["stop_reason"] == "max_steps"
    assert dp5.meta["stiff_at"] is not None and dp5.meta["t_end"] < 50


# ---- further checks: stopping rules, SGD, saving -------------------------------------------------


def test_loss_floor_and_divergence():
    problem, theta0 = aligned_problem(binary_tree(4), [2.0, 1.0], [0.3, 0.3])
    traj = integrate_gf(problem, theta0, t_max=1e3, loss_floor=1e-6)
    assert traj.meta["stop_reason"] == "converged"
    assert traj["loss"][-1] < 1e-6 <= traj["loss"][-2]
    assert traj["t"][-1] == traj.meta["t_end"] < 1e3
    gd = run_gd(problem, theta0, lr=0.1, n_steps=10_000, loss_floor=1e-6)
    assert gd.meta["stop_reason"] == "converged" and gd["loss"][-1] < 1e-6
    blowup = run_gd(problem, theta0, lr=5.0, n_steps=1000)
    assert blowup.meta["stop_reason"] == "diverged"
    assert np.all(np.isfinite(blowup["loss"]))


def test_full_batch_sgd_is_gd_on_the_empirical_loss():
    problem, theta0 = random_problem(binary_tree(4), sigma=0.9, seed=6)
    g = torch.Generator().manual_seed(0)
    data = make_dataset(problem, 64, g, label_noise=0.1)
    train = dataclasses.replace(problem, dataset=data)
    gd = run_gd(train, theta0, lr=0.05, n_steps=40, n_log=10)
    sgd = run_sgd(
        problem, theta0, lr=0.05, n_steps=40, batch_size=64, generator=g, dataset=data, n_log=10
    )
    assert np.allclose(sgd["theta_end"], gd["theta_end"], atol=1e-12)
    assert np.allclose(sgd["loss"], gd["loss"], atol=1e-12)  # training (empirical) loss
    assert np.allclose(sgd["batch_loss"], gd["loss"], atol=1e-12)
    assert not np.allclose(sgd["pop_loss"], sgd["loss"])


def test_online_sgd_reduces_population_loss():
    metrics = [power_law_cov(2, 0.5)] * 4
    problem, theta0 = random_problem(binary_tree(4), sigma=0.9, seed=7, site_metrics=metrics)
    g = torch.Generator().manual_seed(1)
    traj = run_sgd(problem, theta0, lr=0.02, n_steps=600, batch_size=16, generator=g, n_log=30)
    assert traj.meta["stop_reason"] == "completed"
    assert np.array_equal(traj["loss"], traj["pop_loss"])  # online: training loss = population
    assert traj["pop_loss"][-1] < 0.3 * traj["pop_loss"][0]
    assert np.all(np.isfinite(traj["batch_loss"])) and np.all(np.isfinite(traj["batch_loss_avg"]))


def test_trajectory_save_load_and_extras(tmp_path):
    problem, theta0 = random_problem(binary_tree(4), sigma=0.9, seed=8)
    extra = [("grad_norm", lambda prob, theta, t: prob.grad(theta).norm().item())]
    rec = Recorder(save_params=True, extra=extra)
    traj = integrate_gf(problem, theta0, t_max=2.0, n_log=10, recorder=rec)
    n = len(traj["t"])
    assert traj["spectra"].shape == (n, len(problem.spec.tree.edges), 2)
    assert traj["core_norms"].shape == (n, problem.spec.n_cores)
    assert traj["params"].shape == (n, problem.spec.n_params) and traj["grad_norm"].shape == (n,)
    path = traj.save(tmp_path / "run")
    loaded = Trajectory.load(path)
    assert loaded.meta == traj.meta
    assert set(loaded.data) == set(traj.data)
    for key in traj.data:
        assert np.array_equal(loaded[key], traj[key], equal_nan=True)
