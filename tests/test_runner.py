import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit

from ttn.config import RunConfig, stream_seed
from ttn.diagnostics import event_widths, gauge_directions, hessian, project_out_gauge
from ttn.dynamics import Problem, Trajectory
from ttn.network import TTNSpec
from ttn.runner import (
    build,
    expand_grid,
    gauge_projected_eigs,
    learning_events,
    load_run,
    load_summaries,
    main,
    parse_seeds,
    plateau_hessians,
    run,
    run_sweep,
    summary,
)
from ttn.teachers import random_cp_teacher
from ttn.theory import escape_time, odeco_aligned_prediction
from ttn.tree import mps_tree

# Tiny tensors: multithreaded LAPACK calls cost milliseconds instead of microseconds on a busy machine.
torch.set_num_threads(1)


def tiny(**overrides) -> RunConfig:
    """Binary tree, N = 4 (K = 3), d = chi = 2, two odeco components, short gradient flow."""
    base = RunConfig(name="tiny", d=2, chi=2).with_overrides(
        {
            "teacher.weights": [1.0, 0.5],
            "init.sigma": 0.5,
            "optimizer.t_max": 5.0,
            "logging.n_log": 10,
        }
    )
    return base.with_overrides(overrides)


# ---- config --------------------------------------------------------------------------------------


def test_config_round_trip_and_overrides(tmp_path):
    cfg = tiny(**{"chi": [2, 2], "lr_multipliers": [1.0, 0.5, 2.0], "logging.hessian_at": [1.0]})
    assert RunConfig.load(cfg.save(tmp_path / "cfg.yaml")) == cfg
    assert RunConfig.from_dict(cfg.to_dict()) == cfg

    # Partial file; YAML 1.1 reads 1e-3 and 1e5 as strings, the declared types convert them.
    (tmp_path / "partial.yaml").write_text(
        "name: p\ninit: {sigma: 1e-3}\noptimizer: {t_max: 1e5, n_steps: 1e4}\nchi: [2, 3]\n"
    )
    cfg = RunConfig.load(tmp_path / "partial.yaml")
    assert cfg.init.sigma == 1e-3 and isinstance(cfg.init.sigma, float)
    assert cfg.optimizer.t_max == 1e5 and cfg.optimizer.n_steps == 10_000
    assert isinstance(cfg.optimizer.n_steps, int) and cfg.chi == [2, 3]
    assert cfg.teacher == RunConfig().teacher  # untouched sections keep their defaults

    cfg = RunConfig.load(
        tmp_path / "partial.yaml",
        [
            "init.sigma=0.01",
            "teacher.weights=[1, 0.5]",
            "optimizer.loss_floor=null",
            "tree.kind=mps",
            "init.aligned_a0=[[0.1, 0.2]]",
            "chi=4",
        ],
    )
    assert cfg.init.sigma == 0.01 and cfg.teacher.weights == [1.0, 0.5]
    assert cfg.optimizer.loss_floor is None and cfg.tree.kind == "mps" and cfg.chi == 4
    assert cfg.init.aligned_a0 == [[0.1, 0.2]] and cfg.get("tree.kind") == "mps"
    assert cfg.with_overrides({"seed": 3, "loss.kind": "empirical"}).loss.kind == "empirical"

    with pytest.raises(KeyError, match="init.sigam"):
        cfg.with_overrides(["init.sigam=1"])
    with pytest.raises(KeyError, match="optimizer.tmax"):
        RunConfig.from_dict({"optimizer": {"tmax": 1.0}})
    with pytest.raises(ValueError, match="integer"):
        cfg.with_overrides({"optimizer.n_steps": 10.5})
    # Choice fields are checked when the config is built, before anything runs.
    with pytest.raises(ValueError, match="optimizer.method"):
        cfg.with_overrides(["optimizer.method=LSODA"])
    with pytest.raises(ValueError, match="tree.kind"):
        cfg.with_overrides({"tree.kind": "chain"})
    assert cfg.with_overrides(["init.scheme=centred"]).init.scheme == "centered"

    # UTF-8 on every platform (the locale encoding of Windows would garble the name).
    (tmp_path / "utf8.yaml").write_text("name: σ-sweep\n", encoding="utf-8")
    assert RunConfig.load(tmp_path / "utf8.yaml").name == "σ-sweep"


def test_seed_streams_are_distinct_and_fixed():
    seeds = [stream_seed(s, k) for s in range(3) for k in ("teacher", "init", "data", "probe")]
    assert len(set(seeds)) == len(seeds)
    assert stream_seed(0, "init") == 3141116543  # numpy SeedSequence: fixed across runs/machines


# ---- grids ---------------------------------------------------------------------------------------


def test_expand_grid_names():
    base = tiny(name="b")
    grid = {"init.sigma": [0.1, 0.05], "chi": [2, 3]}
    configs = expand_grid(base, grid, seeds=[0, 1])
    names = [c.name for c in configs]
    assert len(configs) == 8 and len(set(names)) == 8
    assert names[:3] == [
        "b__sigma=0.1__chi=2__seed=0",
        "b__sigma=0.1__chi=2__seed=1",
        "b__sigma=0.1__chi=3__seed=0",
    ]
    assert [c.name for c in expand_grid(base, grid, seeds=[0, 1])] == names  # deterministic
    last = configs[-1]
    assert (last.init.sigma, last.chi, last.seed) == (0.05, 3, 1)
    assert last.teacher == base.teacher and base.name == "b"  # base unchanged

    # An ambiguous short label keeps the full key; lists and strings get readable names.
    names = [c.name for c in expand_grid(base, {"teacher.seed": [1, 2]}, seeds=[0])]
    assert names == ["b__teacher.seed=1__seed=0", "b__teacher.seed=2__seed=0"]
    cfg = expand_grid(base, {"teacher.weights": [[1.0, 0.5]], "tree.kind": ["mps"]})[0]
    assert cfg.name == "b__weights=1.0-0.5__kind=mps"
    assert expand_grid(base)[0].name == "b"
    with pytest.raises(ValueError, match="unique"):
        expand_grid(base, {"init.sigma": [0.1, 0.1]})
    with pytest.raises(TypeError, match="lists"):  # a bare string would be iterated by letter
        expand_grid(base, {"tree.kind": "mps"})
    assert parse_seeds("0-3,7") == [0, 1, 2, 3, 7] and parse_seeds("5") == [5]


# ---- one run -------------------------------------------------------------------------------------


def test_run_tiny_config():
    cfg = tiny()
    traj = run(cfg)
    meta = traj.meta
    assert meta["config"] == cfg.to_dict() and meta["name"] == "tiny"
    assert set(meta["git"]) == {"commit", "ttn_dirty", "ttn_sha256"}
    assert len(meta["git"]["ttn_sha256"]) == 64
    assert meta["timing"]["total"] >= meta["timing"]["train"] > 0
    assert meta["seeds"]["init"] == stream_seed(0, "init")
    assert {"scipy", "opt_einsum", "platform"} <= set(meta["versions"])
    assert meta["threads"]["blas"] and all(n == 1 for _, n in meta["threads"]["blas"])
    np.testing.assert_allclose(meta["teacher_spectra"], [[1.0, 0.5], [1.0, 0.5]], rtol=1e-12)
    n_log = len(traj["t"])
    assert traj["spectra"].shape == (n_log, 2, 2) and traj["overlaps"].shape == (n_log, 2)
    n_fine = len(traj["fine_t"])
    assert n_fine > n_log and traj["fine_overlaps"].shape == (n_fine, 2)
    s = summary(traj)
    assert s["stop_reason"] == "completed" and s["t_end"] == 5.0
    assert s["loss_end"] < s["loss_0"] and s["conservation_drift"] < 1e-8
    assert {"t_cross_0", "width_1", "order", "n_plateaus", "max_rel_change"} <= set(s)
    assert s["n_fine_rows"] == n_fine and s["event_kind"] == "overlap"
    assert {"n_cross_1", "transient_1", "min_signal", "float64_floor", "n_unresolved"} <= set(s)
    crossings = np.sort([s["t_cross_0"], s["t_cross_1"]])  # events: sorted, NaN (never) last
    np.testing.assert_array_equal([s["t_event_1"], s["t_event_2"]], crossings)
    json.dumps(s)  # JSON-ready

    # Seeding: reproducible; the run seed changes the init, a fixed teacher seed keeps the teacher.
    assert np.array_equal(run(cfg)["theta_end"], traj["theta_end"])
    other = build(cfg.with_overrides({"seed": 1}))
    assert not torch.equal(other.theta0, build(cfg).theta0)
    assert not torch.equal(other.teacher.factors, build(cfg).teacher.factors)
    fixed = [build(cfg.with_overrides({"seed": s, "teacher.seed": 7})) for s in (0, 1)]
    assert torch.equal(fixed[0].teacher.factors, fixed[1].teacher.factors)


def test_run_aligned_init_matches_theory():
    weights, a0 = [2.0, 1.0], [0.3, 0.2]
    cfg = tiny(
        **{
            "init.scheme": "aligned",
            "init.aligned_a0": a0,
            "teacher.weights": weights,
            "optimizer.t_max": 30.0,
            "logging.n_log": 30,
        }
    )
    traj = run(cfg)
    expected = odeco_aligned_prediction(weights, 3, a0, traj["t"])
    assert np.abs(traj["overlaps"] - expected["overlaps"]).max() < 1e-6
    fine = odeco_aligned_prediction(weights, 3, a0, traj["fine_t"])
    assert np.abs(traj["fine_overlaps"] - fine["overlaps"]).max() < 1e-6
    # Events come from the fine rows, not from the 30 log times over 4 decades (6-8% of t apart,
    # wider than the escapes): times and widths as on a dense grid.
    s = summary(traj)
    for r in range(2):
        assert s[f"t_cross_{r}"] == pytest.approx(expected["escape_times"][r], rel=2e-3)
    dense_t = np.linspace(0.0, 30.0, 30001)
    dense = odeco_aligned_prediction(weights, 3, a0, dense_t)["overlaps"]
    for r, width in enumerate(event_widths(dense_t, dense)):
        assert s[f"width_{r}"] == pytest.approx(width, rel=0.01)
    assert s["n_unresolved"] == 0


@pytest.mark.parametrize("kind", ["gf", "gd"])
def test_run_kernel_probe_and_hessians(kind):
    times = [0.0, 1.5]
    cfg = tiny(
        **{
            "optimizer.kind": kind,
            "optimizer.lr": 0.05,
            "optimizer.n_steps": 60,
            "optimizer.t_max": 3.0,
            "logging.ntk_probe": 8,
            "logging.hessian_at": times,
        }
    )
    traj = run(cfg)
    assert traj["ntk_change"][0] == 0.0 and traj["ntk_change"][-1] > 0
    assert np.all(np.isfinite(traj["ntk_alignment"]))
    assert np.allclose(traj["hessian_t"], times) and np.isin(times, traj["t"]).all()
    problem = build(cfg).problem
    n = problem.spec.n_params
    assert traj["hessian_eigs"].shape == (2, n) == traj["hessian_eigs_gauge"].shape
    # Gauge directions are exact zero modes of H: removing them leaves fewer eigenvalues.
    assert np.isnan(traj["hessian_eigs_gauge"][0]).sum() > 0
    theta0 = build(cfg).theta0
    _, direct = project_out_gauge(
        hessian(problem.loss, theta0), gauge_directions(problem.spec, theta0)
    )
    gauge = traj["hessian_eigs_gauge"][0]
    np.testing.assert_allclose(gauge[np.isfinite(gauge)], direct.numpy(), atol=1e-12)
    s = summary(traj)
    assert s["regime"] in ("stalled", "rich", "saddle-to-saddle") and s["delta_K_end"] > 0
    assert traj.meta["hessian_missing"] == []
    with pytest.warns(UserWarning, match="no Hessian"):  # asked for after the end of the run
        late = run(cfg.with_overrides({"logging.hessian_at": [50.0], "logging.ntk_probe": 0}))
    assert late.meta["hessian_missing"] == [50.0] and "hessian_t" not in late.data


def test_gauge_projected_eigs_with_masks():
    """Only gauge directions that keep the untrained coordinates fixed are removed: with a frozen
    core (here the root of MPS4) those of the bonds away from it, in the two-copy model the
    per-copy gauges (two copies x sum_v chi_v^2 directions). A frozen root used to crash."""
    spec = TTNSpec.uniform(mps_tree(4), 2, 2)
    teacher = random_cp_teacher(4, 2, [1.0, 0.5], seed=3)
    theta = spec.flatten(spec.init(0.8, seed=4))
    mask = torch.ones(spec.n_params, dtype=torch.float64)
    mask[spec.core_slices()[spec.tree.root]] = 0.0
    eigs, gauge = gauge_projected_eigs(Problem(spec, teacher, mask=mask), theta)
    n_trained = int(mask.sum())
    assert eigs.shape == gauge.shape == (n_trained,)
    kept = [v for v in spec.tree.edges if spec.tree.parent[v] != spec.tree.root]
    assert np.isnan(gauge).sum() == sum(spec.bond_dims[v] ** 2 for v in kept) == 8

    cfg = tiny(**{"init.scheme": "antisymmetric", "optimizer.t_max": 1.0})
    traj, setup = run(cfg), build(cfg)
    theta2 = torch.from_numpy(traj["theta_end"])
    eigs, gauge = gauge_projected_eigs(setup.problem, theta2)
    per_copy = 2 * sum(setup.spec.bond_dims[v] ** 2 for v in setup.spec.tree.edges)
    assert len(eigs) == int(setup.problem.mask.sum()) and np.isnan(gauge).sum() == per_copy


@pytest.mark.parametrize(
    "overrides",
    [
        {"init.scheme": "antisymmetric", "logging.hessian_at": [1.0]},
        {"init.scheme": "centered", "init.sigma": 1.0},
        {
            "optimizer.kind": "sgd",
            "loss.kind": "empirical",
            "loss.n_train": 32,
            "optimizer.n_steps": 50,
            "optimizer.batch_size": 8,
        },
        {"optimizer.kind": "sgd", "optimizer.n_steps": 50, "loss.label_noise": 0.1},
        {"teacher.kind": "random_ttn", "teacher.ttn_tree_kind": "mps", "tree.kind": "mps"},
        {
            "teacher.kind": "pattern",
            "teacher.patterns": [[0, 0, 0, 0], [1, 1, 0, 0]],
            "inputs.cov": "power_law",
            "lr_multipliers": [1.0, 2.0, 0.5],
        },
        {
            "inputs.cov": "custom",
            "inputs.matrices": [[2.0, 0.0], [0.0, 0.5]],
            "optimizer.method": "lsoda",
        },
    ],
    ids=["two-copy", "centred", "sgd-empirical", "sgd-online", "ttn-teacher", "pattern", "custom"],
)
def test_run_variants(overrides):
    traj = run(tiny(**overrides))
    s = summary(traj)
    assert s["stop_reason"] in ("completed", "converged")
    assert math.isfinite(s["loss_end"]) and s["loss_end"] < s["loss_0"]
    if "hessian_eigs_gauge" in traj.data:  # two-copy: per-copy gauges removed, the rest finite
        assert 0 < np.isfinite(traj["hessian_eigs_gauge"]).sum() < traj["hessian_eigs"].size


def test_spectral_events_for_non_orthogonal_teacher():
    """Correlated CP components (overlap 0.61) escape together: both a_r cross 1/2 at t ~ 38, and
    the second stage (t ~ 560, after a plateau at L = 0.0138) raises no overlap event. Spectral
    events see both stages (the overlap events gave rho = 0.005, 'rich')."""
    cfg = tiny(
        **{
            "teacher.kind": "random_cp",
            "teacher.weights": [1.0, 0.8],
            "teacher.seed": 55,
            "init.sigma": 0.05,
            "optimizer.method": "lsoda",
            "optimizer.t_max": 1e5,
            "optimizer.loss_floor": 1e-10,
            "logging.n_log": 200,
            "logging.ntk_probe": 8,
        }
    )
    traj = run(cfg)
    s = summary(traj)
    assert not traj.meta["problem"]["orthogonal_components"]
    assert abs(s["t_cross_0"] - s["t_cross_1"]) < 1.0  # the overlaps rise together
    events = learning_events(traj)
    assert events["kind"] == "spectral" and np.all(events["resolved"])
    assert events["time"] == pytest.approx([38.0, 559.5], rel=0.01)
    assert s["regime"] == "saddle-to-saddle" and s["rho_min"] > 15 and s["n_plateaus"] == 2


def test_summary_events_with_a_transient():
    # a_1 bumps above 1/2 (to 0.6) while a_0 is learned, falls back, and is learned at t = 60:
    # its event is the last upward crossing, its width that of the final rise.
    t = np.linspace(0.0, 100.0, 2001)
    a0 = expit((t - 20) / 0.5)
    a1 = 0.6 * np.exp(-((t - 21) ** 2) / 2) + expit((t - 60) / 1.0)
    loss = 0.5 * ((3 * (1 - a0)) ** 2 + (2 * (1 - a1)) ** 2)
    data = {"t": t, "loss": loss, "pop_loss": loss, "rel_dist_from_init": t / 100.0}
    data["overlaps"] = np.column_stack([a0, a1])
    problem = {"target_norm2": 13.0, "component_targets": [3.0, 2.0], "orthogonal_components": True}
    meta = {"stop_reason": "completed", "t_end": 100.0, "n_steps": 2000, "wall_time": 0.0}
    s = summary(Trajectory(data, meta | {"problem": problem}))
    assert s["n_cross_1"] == 2 and s["t_cross_1"] == pytest.approx(60.0, abs=0.01)
    assert s["width_1"] == pytest.approx(2 * np.log(9), rel=0.01)
    assert s["transient_1"] == pytest.approx(0.6, abs=0.01) and s["transient_0"] < 1e-6
    assert s["rho_min"] == pytest.approx(40.0 / (2 * np.log(9)), rel=0.01)
    # a_0(0) = expit(-40) = 4e-18: its escape signal starts below float64 roundoff of |W*|.
    assert s["n_unresolved"] == 0 and s["float64_floor"] and s["min_signal"] < 1e-17


def test_summary_labels_lazy_floor_and_divergence():
    # Deficient bond (chi = 2 < cut dimension 4): the centred model at large sigma ends at the
    # projected fit, not below 1e-3 L(0); 'stalled' before the lazy floor was known.
    cfg = tiny(
        **{
            "init.scheme": "centered",
            "init.sigma": 4.0,
            "optimizer.t_max": 0.3,
            "logging.n_log": 30,
            "logging.ntk_probe": 16,
        }
    )
    s = summary(run(cfg))
    assert 0.1 < s["lazy_floor"] / s["loss_0"] < 0.9 and s["delta_K_end"] < 0.1
    assert 1.0 < s["loss_end_over_floor"] < 1.1 and s["regime"] == "lazy (projected fit)"

    diverged = tiny(**{"optimizer.kind": "gd", "optimizer.lr": 50.0, "logging.ntk_probe": 4})
    s = summary(run(diverged))
    assert s["stop_reason"] == "diverged" and s["regime"] == "diverged"


def test_plateau_hessians():
    # Aligned init, weights (2, 1): one plateau, where component 0 is learned and component 1's
    # mode (amplitudes ~0.05 on each core) still has a descent direction.
    cfg = tiny(
        **{
            "init.scheme": "aligned",
            "init.aligned_a0": [0.05, 0.05],
            "teacher.weights": [2.0, 1.0],
            "optimizer.t_max": 200.0,
            "logging.save_params": True,
            "logging.n_log": 100,
        }
    )
    traj = run(cfg)
    (plateau,) = plateau_hessians(cfg, traj)
    assert escape_time(3, 2.0, 0.05) < plateau["t"] < escape_time(3, 1.0, 0.05)
    assert plateau["lambda_min"] < 0 and plateau["n_negative"] >= 1
    assert np.isfinite(plateau["eigs_gauge"]).sum() > 0


# ---- sweeps --------------------------------------------------------------------------------------


def two_extra(problem, theta, t):
    return 2.0


def test_run_sweep_parallel_and_resumable(tmp_path, capsys):
    good = expand_grid(tiny(name="sw"), seeds=[0, 1])
    bad = tiny(name="sw__bad", chi=[2])  # valid config, fails when run
    with pytest.raises(RuntimeError, match="sw__bad"):  # the other runs still complete
        run_sweep([*good, bad], tmp_path, workers=2, extra=[("two", two_extra)])
    assert "one entry per edge" in (tmp_path / "sw__bad.error.txt").read_text(encoding="utf-8")
    assert not (tmp_path / ".sweep.lock").exists()
    table = pd.read_csv(tmp_path / "summary.csv")
    assert sorted(table["name"]) == ["sw__seed=0", "sw__seed=1"]
    assert set(table["init.sigma"]) == {0.5} and "t_cross_0" in table
    stamps = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("sw__seed=*")}
    assert len(stamps) == 4  # .npz and .json per run
    traj = load_run(tmp_path, "sw__seed=1")
    theta_end = traj["theta_end"]
    assert np.all(traj["two"] == 2.0)  # extra diagnostics reach the workers
    meta = json.loads((tmp_path / "sw__seed=1.json").read_text(encoding="utf-8"))["meta"]
    assert meta["threads"]["torch"] == 1 and meta["threads"]["OPENBLAS_NUM_THREADS"] == "1"

    capsys.readouterr()
    assert len(run_sweep(good, tmp_path, workers=2)) == 2
    assert "2 done, 0 to run" in capsys.readouterr().out
    assert stamps == {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("sw__seed=*")}

    # Same names, other config: refused before anything runs (no silent reuse of old results).
    changed = [c.with_overrides({"optimizer.t_max": 2.0}) for c in good]
    with pytest.raises(ValueError, match="another config"):
        run_sweep(changed, tmp_path, workers=2)
    # One sweep per directory at a time.
    (tmp_path / ".sweep.lock").write_text("pid 1", encoding="utf-8")
    with pytest.raises(RuntimeError, match="another sweep"):
        run_sweep(good, tmp_path, workers=2)
    (tmp_path / ".sweep.lock").unlink()

    # A run without its .json (interrupted) is redone, and gives the same result in a new process.
    (tmp_path / "sw__seed=1.json").unlink()
    run_sweep(good, tmp_path, workers=2)
    assert "1 done, 1 to run" in capsys.readouterr().out
    assert np.array_equal(load_run(tmp_path, "sw__seed=1")["theta_end"], theta_end)
    assert (tmp_path / "sw__seed=0.npz").stat().st_mtime_ns == stamps["sw__seed=0.npz"]
    stored = load_summaries(tmp_path, names=["sw__seed=0"])
    fresh = load_summaries(tmp_path, names=["sw__seed=0"], recompute=True)
    assert len(stored) == 1 and stored["t_cross_0"][0] == fresh["t_cross_0"][0]


def test_command_line(tmp_path, capsys):
    tiny(name="cli").save(tmp_path / "base.yaml")
    (tmp_path / "grid.yaml").write_text("init.sigma: [0.5, 0.4]\nseeds: [3]\n")
    args = [
        str(tmp_path / "base.yaml"),
        "--grid",
        str(tmp_path / "grid.yaml"),
        "--out",
        str(tmp_path / "out"),
        "optimizer.t_max=2.0",
    ]
    main([*args, "--dry-run"])
    assert "cli__sigma=0.5__seed=3\ncli__sigma=0.4__seed=3\n2 runs" in capsys.readouterr().out
    main(args)
    table = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert sorted(table["name"]) == ["cli__sigma=0.4__seed=3", "cli__sigma=0.5__seed=3"]
    assert set(table["optimizer.t_max"]) == {2.0} and set(table["seed"]) == {3}
