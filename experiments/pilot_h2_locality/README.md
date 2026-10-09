# Pilot: does the time to learn the next component depend on locality? (hypothesis H2)

Status: **pilot** (quick test). Results below are **empirical**.

## Question

Hypothesis H2 (`notes/prior_knowledge.md`, section 8): after gradient flow from small init has
learned one teacher component, the time to learn the next one scales like

| k = cores that must acquire a new direction | predicted extra plateau t2 − t1 |
| --- | --- |
| k ≥ 3 | ∝ σ^−(k−2) |
| k = 2 | ∝ log(1/σ) |
| k = 1 | none (single-core, convex update) |

where k is the number of cores in the minimal subtree (Steiner tree) spanning the sites where the
new component differs from the learned one; its edges are exactly the bonds whose teacher rank is 2.
The first escape should scale like t1 ∝ σ^−(K−2) (K = number of cores). Same teacher, different
tree, different k: e.g. a component that differs on sites {0,1} needs k = 2 on an MPS but k = 1 on a
binary tree whose bottom core carries sites 0 and 1.

## Setup

- Model as in `notes/model.md`: f(x) = <W, x_1 ⊗ … ⊗ x_N>, whitened Gaussian inputs, exact
  population loss L = ½‖W − W*‖², d = 2, bond dimension χ = 2 on every bond (`TTNSpec.uniform`),
  fan-in init with per-core scale σ (`TTNSpec.init`), float64.
- Trees: MPS N = 4 (K = 4), binary N = 4 (K = 3); larger: MPS N = 6 (K = 6), binary N = 8 (K = 7).
- Teacher (`pattern_teacher`, random per-site orthonormal bases, **teacher seed 1000 for every run**,
  independent of the init seeds): two orthogonal
  rank-one components with weights s1 = 1.0, s2 = 0.5. Component 1 uses basis vector 0 at every site;
  component 2 uses basis vector 1 on a set of sites S and vector 0 elsewhere:

  | kind | S | k on MPS4 | k on binary4 | k on MPS6 | k on binary8 |
  | --- | --- | --- | --- | --- | --- |
  | local | {0,1} | 2 | 1 | 2 | 1 |
  | far | {0,N−1} | 4 | 3 | 6 | 5 |
  | global (odeco) | all sites | 4 | 3 | 6 | 7 |
  | sites0-2 (extra) | {0,2} | 3 | – | 3 | 3 |
  | sites0-3 (extra) | {0,3} | – | – | 4 | – |
  | sites0-4 (extra) | {0,4} | – | – | 5 | – |

  The extra kinds scan k at fixed tree (MPS6: k = 2…6; binary8: k = 1, 3, 5, 7).
- Seeds: init seeds 0–7 (N = 4), 0–5 (MPS6, binary8), 0–3 (saddle start). Since the Gaussian init is
  invariant under rotations of each physical leg, a fixed teacher with random inits samples the
  relevant relative orientations.
- Dynamics: gradient flow dθ/dt = −∇L. Integrated with scipy's LSODA (Adams/BDF switching) with the
  exact autograd Hessian as Jacobian, rtol = 1e−10, atol = 1e−13·σ. Explicit Runge–Kutta is
  accurate but too slow: once a component is learned the flow is stiff (Hessian eigenvalues ~K),
  so explicit steps are stability-limited (~1) on second plateaus that last up to ~1e9 time units.
- Measurements: t_r = first time a_r = `component_overlaps(...)[r]` crosses 0.5, located by
  root-finding on the integrator's dense output inside the step where the crossing happens (not by
  interpolating a sampling grid). t_first = min(t1, t2), gap = |t2 − t1| = time from the first to the
  second learning event (the order can be reversed: component 2 learned first).
- Runs in which the second component is never learned end in one of three ways, reported separately:
  **collapsed**: the **collapse rule** (`h2sim.collapsed`) fired, i.e. at t ≥ 30·t_first the largest
  second singular value of W over all bonds was below 1e−9 of its value just after the first escape
  and still falling steeply (or at round-off ~1e−14), or the run ended with that value at round-off;
  **decaying**: it reached the horizon (t_max = 1e5·σ^−(K−2) for N = 4, 1e4·σ^−(K−2) for the larger
  trees, or the wall-clock limit) with the rank-2 content still falling; **growing**: it reached the
  horizon with the rank-2 content rising, i.e. a slow escape that was cut off (censored). The growing
  MPS6 runs were rerun with a 100× longer horizon (`configs/extend.yaml`) and those reruns are used.
  The collapse rule was checked offline on 483 stored trajectories of the discarded first sweep (see
  below; the dynamics are valid for this purpose): it never fired on a run that later learned the
  second component, including runs whose rank-2 content dipped by 1e6 before recovering, and the runs
  it fires on kept decaying to round-off when followed to 1e3–1e4·t_first. `configs/accuracy.yaml`
  re-checks it on fresh runs.
- Exponents: slope of log t vs log σ, fitted per seed (the init direction is fixed per seed and only
  scaled by σ), then mean ± std over seeds; "asymptotic" = fit over the 4 smallest σ only. Gap fits
  for k ≥ 3 use only runs with a distinct intermediate plateau: the time from the first component
  being learned (a = 0.9) to the second being half learned (a = 0.5) exceeds twice the width of the
  first transition (a: 0.1 → 0.9). At large σ the two escapes can overlap and their time difference is
  then not a second-escape time; the number of excluded runs is given. For k = 2 we
  also fit gap = A·ln(1/σ) + B. A balanced-mode estimate (DLN mode dynamics, the mode-2 amplitude grows
  from σ to σ^(1−s2/s1) during the first plateau) gives A = (1 − s2/s1)/s2 = 1 for these weights.

## How to run

From the repo root (each sweep uses ≤ 8 processes, one thread each; existing runs are skipped, so an
interrupted sweep is resumed by rerunning the same command):

```bash
uv run python experiments/pilot_h2_locality/run_sweep.py experiments/pilot_h2_locality/configs/n4.yaml
uv run python experiments/pilot_h2_locality/run_sweep.py experiments/pilot_h2_locality/configs/large.yaml
uv run python experiments/pilot_h2_locality/run_sweep.py experiments/pilot_h2_locality/configs/extend.yaml
uv run python experiments/pilot_h2_locality/run_sweep.py experiments/pilot_h2_locality/configs/saddle.yaml
uv run python experiments/pilot_h2_locality/analyze.py          # tables + figures in results/
# integration checks
uv run python experiments/pilot_h2_locality/run_sweep.py experiments/pilot_h2_locality/configs/accuracy.yaml
uv run python experiments/pilot_h2_locality/check_accuracy.py   # results/accuracy.csv
```

Raw per-run outputs (`<tag>.json` scalars, `<tag>.npz` trajectories: t, loss, overlaps, top-2 bond
singular values at every kept step, initial/final parameters) go to `data/pilot_h2_locality/<sweep>/`
(gitignored), and `run_sweep.py` collects the scalars of each sweep into
`data/pilot_h2_locality/<csv>`. `analyze.py` needs these; the committed `results/` hold the compact
per-run table, the fits and the figures. Code: `h2sim.py` (one run), `run_sweep.py` (parallel
sweeps), `analyze.py`, `check_accuracy.py`.

## Variants

- More σ values or seeds: edit `sigmas` / `seeds` in a config and rerun; only new runs are computed.
- Other teachers: any set of differing sites is a kind `sitesA-B-...` (e.g. `sites0-3`).
- Other integrators/tolerances: `method` (LSODA, Radau, BDF, DOP853, RK45), `rtol`, `atol_factor`,
  `max_step` in a config or group `overrides` (see `configs/accuracy.yaml`).
- `overrides: {start: saddle}`: start exactly at the rank-one saddle (component 1 learned, aligned
  balanced cores) plus a fan-in perturbation of scale σ, which times the second escape alone
  (`configs/saddle.yaml`; the collapse rule does not apply there, so runs without an escape end at
  t_max or at the wall-clock limit).
- `overrides: {chi: 4}`: larger bond dimension (`configs/chi4.yaml`, prepared but **not run** in this
  pilot; it would test whether the collapse depends on χ = 2 being the minimal bond dimension).

## Data validity: rerun after a library fix

A first sweep (about 480 runs, 2026-10-09) was made before library commit 57d5262, in which
teachers and student inits drew from the same random streams: with teacher seed 0 and init seed 0
the student's site-0 core was exactly triangular in the teacher basis, so the seed-0 runs started
correlated with the teacher. That sweep was discarded (moved to
`data/pilot_h2_locality_PREFIX_INVALID/`, not used in any result). Everything below was rerun from
scratch with the fixed library and a fixed teacher seed (1000) unrelated to the init seeds.

At about 20:32 another process on the shared machine killed all python processes. This interrupted
the accuracy sweep: its last run (the Radau variant of MPS6 `far`, σ = 0.1, seed 0) was lost and
was rerun afterwards. The N = 4 sweep had finished at 20:13 and the large-tree sweep started at
20:33; all 784 main runs, all accuracy runs and all saddle-start and extension runs were checked to
be present and readable (no truncated files).

## Results

From small init: 784 runs (N = 4: 8 seeds × 8 σ per teacher kind; MPS6 / binary8: 6 seeds × 6 / 5 σ),
19 of them rerun with a longer horizon. Compact per-run table: `results/runs_derived.csv`; per-seed
exponents: `results/per_seed.csv`; fits: `results/fits.csv` (the table below is `results/fits.md`).

### Integration accuracy (`results/accuracy.csv`)

- 26 reference runs (all trees and teacher kinds, escaping and stuck, σ down to 5e−4) were rerun
  with tighter rtol (1e−12), tighter atol (×1e−6), Radau, BDF, explicit DOP853 (rtol 1e−10 and 1e−12)
  and RK45 (1e−12) where the second plateau is short enough for stability-limited explicit steps,
  and with a capped step size (max_step 2–10, i.e. ~1e3–1e5 steps per plateau). Over 46 completed
  comparisons,
  t1 and t2 agree with the reference to ≤ 1.8e−7 relative (median ~1e−9): LSODA rtol 1e−12 ≤ 1.2e−7,
  Radau ≤ 1.0e−7, BDF 8e−9, DOP853 ≤ 1.8e−8, RK45 1.8e−8, capped steps ≤ 1.8e−7, tighter atol ≤ 2e−8.
  The capped-step runs show that plateaus are integrated, not stepped over; crossing times are
  root-found inside the integration step, not interpolated on a grid.
- Gauge conservation laws (`conserved_quantities`): drift median 6e−11, max 2e−7 (the longest
  runs: ~1e5 steps up to t ≈ 5e9).
- Collapse rule: a collapsed case (MPS4 `far`, σ = 0.02, seed 1) followed without the rule to t_max
  (≈ 6500·t_first) stays collapsed (largest second bond singular value 2e−15); two escaping cases
  are identical with and without it. A stuck MPS6 state at t = 3000 is the same with LSODA and with
  explicit DOP853 (second singular values 2.70e−4 in both).
- The Radau variant of MPS6 `far` (σ = 0.1, seed 0), lost to the external process kill, was rerun:
  it reproduces t1 to 6e−10 but did not reach t2 (≈ 9.6e6) within its 1500 s wall-clock limit (Radau
  is slow on long stiff plateaus). The three other variants of that case agree on t2 to ≤ 2.1e−8.
  In total there are 47 comparisons.

### First escape: t_first ∝ σ^−(K−2) (confirmed exactly)

Per-seed slopes are −2.00 (MPS4), −1.00 (binary4), −4.00 (MPS6), −5.00 (binary8), within 0.003, for
every teacher kind (`results/fig_t_first.png`). σ spans 1e−3–0.1 (MPS4), 2e−4–0.05 (binary4),
0.05–0.3 (MPS6), 0.07–0.3 (binary8); t_first spans ~1e2–1e7.

### Second escape vs k (`results/fig_gap.png`, `results/fig_kscan.png`)

| tree | K | kind | k | runs (no 2nd escape: collapsed/decaying/growing) | t_first: pred | t_first: fit (all σ) | t_first: fit (smallest 4 σ) | gap: pred | gap: fit (all σ) | gap: fit (smallest 4 σ) | runs excluded from gap fit | gap vs ln(1/σ) | median gap / t_first |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mps4 | 4 | local | 2 | 64 (0) | -2 | -2.00 ± 0.00 | -2.00 ± 0.00 | log(1/sigma) | -0.23 ± 0.10 | -0.17 ± 0.05 | 0 | 1.01 ± 0.01 (smallest 4 σ: 1.00 ± 0.00) | 0.00053 |
| mps4 | 4 | far | 4 | 64 (40/0/0) | -2 | -2.00 ± 0.00 | -2.00 ± 0.00 | -2 | -2.00 ± 0.01 | -2.00 ± 0.00 | 0 |  | 10.2 |
| mps4 | 4 | global | 4 | 64 (0) | -2 | -2.00 ± 0.00 | -2.00 ± 0.00 | -2 | -2.00 ± 0.00 | -2.00 ± 0.00 | 0 |  | 4.82 |
| mps4 | 4 | sites0-2 | 3 | 64 (0) | -2 | -2.00 ± 0.00 | -2.00 ± 0.00 | -1 | -1.00 ± 0.02 | -1.00 ± 0.00 | 0 |  | 0.034 |
| binary4 | 3 | local | 1 | 64 (0) | -1 | -1.00 ± 0.00 | -1.00 ± 0.00 | no plateau | max abs(gap) = 0.059 | max abs(gap) = 0.059 | 0 | -0.00 ± 0.00 (smallest 4 σ: -0.00 ± 0.00) | 2.99e-06 |
| binary4 | 3 | far | 3 | 64 (0) | -1 | -1.00 ± 0.00 | -1.00 ± 0.00 | -1 | -1.00 ± 0.00 | -1.00 ± 0.00 | 0 |  | 2 |
| binary4 | 3 | global | 3 | 64 (0) | -1 | -1.00 ± 0.00 | -1.00 ± 0.00 | -1 | -1.00 ± 0.00 | -1.00 ± 0.00 | 3 |  | 2.73 |
| mps6 | 6 | local | 2 | 36 (0) | -4 | -4.00 ± 0.00 | -4.00 ± 0.00 | log(1/sigma) | -0.35 ± 0.41 | -0.28 ± 0.24 | 0 | 0.83 ± 0.41 (smallest 4 σ: 0.95 ± 0.15) | 0.000256 |
| mps6 | 6 | far | 6 | 36 (6/0/3) | -4 | -4.00 ± 0.00 | -4.00 ± 0.00 | -4 | -4.49 ± 0.91 | -4.56 ± 0.87 | 0 |  | 252 |
| mps6 | 6 | global | 6 | 36 (6/0/1) | -4 | -4.00 ± 0.00 | -4.00 ± 0.00 | -4 | -4.96 ± 0.54 | -5.22 ± 0.70 | 0 |  | 50.8 |
| mps6 | 6 | sites0-2 | 3 | 36 (0) | -4 | -4.00 ± 0.00 | -4.00 ± 0.00 | -1 | -1.01 ± 0.06 | -1.01 ± 0.04 | 3 |  | 0.00212 |
| mps6 | 6 | sites0-3 | 4 | 36 (12/4/0) | -4 | -4.00 ± 0.00 | -4.00 ± 0.00 | -2 | -2.12 ± 0.17 | -2.05 ± 0.06 | 0 |  | 0.244 |
| mps6 | 6 | sites0-4 | 5 | 36 (6/12/0) | -4 | -4.00 ± 0.00 | -4.00 ± 0.00 | -3 | -3.61 ± 0.37 | -3.75 ± 0.24 | 0 |  | 14.4 |
| binary8 | 7 | local | 1 | 30 (0) | -5 | -5.00 ± 0.00 | -5.00 ± 0.00 | no plateau | max abs(gap) = 0.28 | max abs(gap) = 0.28 | 0 | -0.06 ± 0.06 (smallest 4 σ: -0.04 ± 0.04) | 3.12e-07 |
| binary8 | 7 | far | 5 | 30 (0) | -5 | -5.00 ± 0.00 | -5.00 ± 0.00 | -3 | -2.99 ± 0.11 | -3.00 ± 0.11 | 0 |  | 0.599 |
| binary8 | 7 | global | 7 | 30 (0/5/0) | -5 | -5.00 ± 0.00 | -5.00 ± 0.00 | -5 | -4.92 ± 0.12 | -4.97 ± 0.05 | 0 |  | 40.8 |
| binary8 | 7 | sites0-2 | 3 | 30 (0) | -5 | -5.00 ± 0.00 | -5.00 ± 0.00 | -1 | -1.05 ± 0.08 | -1.01 ± 0.06 | 0 |  | 0.00221 |

Reading the table (exponents are means ± std over seeds of per-seed fits, over escaping runs):

- **k = 1** (binary trees, `local`): no second plateau. |t2 − t1| ≤ 0.06 (binary4) and ≤ 0.28
  (binary8), i.e. ≤ 2.4e−3 and ≤ 2.3e−4 of t1, shrinking with σ; no run has a distinct intermediate
  plateau; the loss curves for all σ collapse onto one curve when aligned at t_first
  (`results/fig_loss_local_aligned.png`). The binary tree sees this teacher as rank one on every bond.
- **k = 2** (MPS, `local`): gap = A·ln(1/σ) + B with A = 1.01 ± 0.01 (MPS4, σ = 1e−3–0.1, per seed
  0.99–1.03; `results/fig_local.png`) and 0.95 ± 0.15 for MPS6 over its 4 smallest σ (0.05–0.15; the
  all-σ value 0.83 ± 0.41 is pulled down by σ ≥ 0.15, where two seeds have nearly constant gaps of
  8–10). A = 1 is the balanced-mode value (1 − s2/s1)/s2 (it includes the head start the second mode
  gets during the first plateau). Log-log slopes are only −0.2 to −0.35, so this is not a power law; B
  depends on the seed (−1.5 to 10). The plateau at L = 0.125 is short (≲ 10 time units) but present.
- **k = 3**: −1.00 ± 0.00 (MPS4 sites {0,2}, binary4 `far`, binary4 `global`), −1.01 ± 0.04 (MPS6
  sites {0,2}), −1.01 ± 0.06 (binary8 sites {0,2}). Every seed escapes.
- **k = 4**: −2.00 ± 0.00 (MPS4 `far`, 3 escaping seeds; MPS4 `global`, all 8 seeds), −2.05 ± 0.06
  (MPS6 sites {0,3}, 3 escaping seeds). Many seeds never escape (next section).
- **k = 5**: binary8 `far` −3.00 ± 0.11 (all 6 seeds escape), as predicted. MPS6 sites {0,4}:
  −3.75 ± 0.24 (per seed −3.47, −3.83, −3.94; only 3 of 6 seeds escape), steeper than predicted.
- **k = 6** (MPS6): `far`: seeds 0, 4, 5 give −4.06, −3.88, −3.97 (as predicted), seeds 1, 3 give
  −5.05, −5.87; `global`: seed 1 gives −4.01, seeds 0, 3, 4, 5 give −5.24, −5.58, −5.52, −5.76.
  The exponents are bimodal: −(k−2) for some seeds and about one to two units steeper for others.
- **k = 7** (binary8 `global`): −4.97 ± 0.05 (5 of 6 seeds), as predicted.

The tree dependence predicted by H2 shows up directly for identical teachers: `local` gives a
log-like plateau on the MPS (k = 2) and none on the binary tree (k = 1); `far` gives σ^−2 on MPS4
(k = 4) and σ^−1 on binary4 (k = 3); on binary8 the `far` gap grows like σ^−3 while the first escape
grows like σ^−5. Prefactors are large and seed-dependent: median gap/t_first ≈ 10 (MPS4 `far`),
5 (MPS4 `global`), 2–3 (binary4), 40 (binary8 `global`), 50–250 (MPS6 `global`, `far`), and up to
4e4 in single runs.

### Surprise 1: no second escape; the flow converges to the rank-one saddle

In 95 of the 332 runs with k ≥ 4 (and in none with k ≤ 3) the second component is never learned and
the loss stays at 0.125. By seed (the same seeds fail at every σ unless noted):

| tree | kind | k | seeds without a second escape | how the runs end |
| --- | --- | --- | --- | --- |
| mps4 | far | 4 | 1, 3, 4, 5, 7 (of 8) | collapsed (40 runs) |
| mps4 | global | 4 | none | |
| mps6 | sites {0,3} | 4 | 2, 4; 5 only at σ ≥ 0.1 | collapsed 12, decaying 4 |
| mps6 | sites {0,4} | 5 | 2, 4, 5 (of 6) | collapsed 6, decaying 12 |
| mps6 | far | 6 | 2; 1 at σ ≤ 0.07; 3 at σ = 0.05 | collapsed 6, growing 3 |
| mps6 | global | 6 | 2; 3 at σ = 0.05 | collapsed 6, growing 1 |
| binary8 | far | 5 | none | |
| binary8 | global | 7 | 1 (of 6) | decaying 5 |

*Collapsed*: W's rank-2 content (largest second singular value over the bonds) falls by ≥ 1e9, to
2e−11–3e−16, and the overlap with the missing component to 1e−23–1e−36: W converges to the rank-one
saddle where only component 1 is represented (`results/fig_bond_s2.png`). *Decaying*: still falling
when the horizon was reached (after 160–7300·t_first). *Growing*: still rising at the 900 s
wall-clock limit even after the 100× extension (`configs/extend.yaml`), i.e. probably a very slow
escape; other σ of the same seeds do escape. The 15 extended runs that escaped took 2e3–4e4·t_first.

In the collapsed end states, decomposed in the learned bond basis (MPS4 `far` seeds 1, 3; MPS6 `far`
and `global` seed 2; MPS6 sites {0,4} seed 4): the entries of the end cores of the new chain, which
map the new site direction into the unused bond channel, are 1e−12–1e−14 (both ends in 4 of 5
states), while the interior chain entries stay at 1e−3–1e−1 (in one state an interior entry is also
4e−9). The new component's chain is cut, so its drive, a product over the chain, vanishes.

Mechanism (**conjecture**): near the rank-one saddle each of the k chain entries (size ~σ) is driven
at rate ∝ s2·σ^(k−1). An end entry also forms, with a "cross" entry of the neighbouring core (unused
channel → learned channel), an off-teacher term of W at second order. The loss damps that pair like a
two-layer linear net fitting zero, at a rate ∝ σ³ (end² − cross² is conserved and the smaller factor
dies). For k ≤ 3 the drive is faster. For k = 4 both scale alike, so the outcome is set by the init
direction and does not depend on σ for a given seed (MPS4 `far`). For k ≥ 5 the damping wins at small
σ. If both ends die, W collapses. If one end is pinned at a small quasi-static value, the escape slows
to ∝ σ^−(2k−6), i.e. σ^−4 for k = 5 and σ^−6 for k = 6. That would explain the steeper MPS6 exponents
(−3.5 to −3.9 for k = 5, −5.0 to −5.9 for k = 6) as a crossover between −(k−2) and −(2k−6). This was
not tested beyond the end-state decompositions above. Binary8 (`far`, k = 5; `global`, k = 7) shows
no steepening.

### Surprise 2: order of learning

- MPS4 `global` seed 1 and MPS6 `global` seed 1 learn component 2 (weight 0.5) first at every σ; the
  intermediate plateau then sits at L = 0.5. This is expected for an odeco teacher (H1: both
  components are local maxima of the spherical flow), and never happens for the matrix-like
  teachers (`local`, `far`, sites) once the two escapes are separated.
- At the largest σ the two escapes can overlap (MPS6 sites {0,2} seed 1 at σ ≥ 0.15, binary4
  `global` seed 7 at σ ≥ 0.01): both components are learned within a few time units, in either order.
  These 6 runs have no distinct intermediate plateau and are excluded from the gap fits.
- In the k = 1 runs the two overlaps cross 0.5 within ≤ 0.3 time units, in either order (binary4
  seeds 2, 3, 6, 7; binary8 seeds 2, 3); this is not a real reversal.

### Plateau levels and loss-curve shapes (`results/fig_loss_curves.png`)

Every distinct intermediate plateau lies at L = 0.125 (= s2²/2, component 1 learned) or 0.5
(= s1²/2, component 2 learned first), and all runs without a second escape end at L = 0.125
(component 1 is always learned). No other plateau levels occur. Loss curves are staircases
0.625 → 0.125 → 0 for k ≥ 3, a single drop for k = 1, and a single drop with a short shoulder at
0.125 for k = 2 whose length grows by the same amount for each factor of σ.

### Second escape alone: start at the rank-one saddle (`results/saddle_fits.md`, `results/fig_saddle_escape.png`)

340 runs (4 seeds × 5 σ per kind) start exactly at the saddle where component 1 is learned
(balanced aligned cores, `aligned_init`) plus a fan-in perturbation of per-core scale σ, so the
second escape is timed without the first plateau. The collapse rule cannot be used here (there is no
first escape), so runs without an escape stop at t_max = 1e4·σ^−(K−2) or at a 150 s wall-clock limit;
at small σ and k ≥ 5 the latter may cut off slow escapes.

- k = 1: escape after 0.66–0.68 time units at every σ (≈ ln 2: a single core relaxes linearly).
- k = 2: t_esc = A·ln(1/σ) + B with A = 2.3 ± 0.4 (MPS4) and 2.5 ± 0.8 (MPS6), against the
  balanced-mode value 1/s2 = 2 without a head start; from small init it is 1. That the coefficient
  halves when the first plateau precedes the escape supports the head-start picture.
- k = 3: −0.98, −1.00, −1.01, −0.96, −1.06 (MPS4, binary4 `far`, binary4 `global`, MPS6, binary8),
  all seeds escape.
- k = 4: −2.00 ± 0.01 (MPS4 `far`) and −1.95 ± 0.06 (MPS4 `global`) for 3 of 4 seeds; seed 3
  collapses at every σ for both teachers (second singular values ~1e−16). MPS6 sites {0,3}: 6 of 20
  runs without an escape, exponents scattered (−2.60 ± 1.26).
- k = 5: binary8 `far` −2.92 ± 0.06, all 20 runs escape; MPS6 sites {0,4}: 11 of 20 without an escape.
- k = 6, 7 (MPS6 `far`/`global`, binary8 `global`): 12–16 of 20 without an escape; exponents rest
  on one or two seeds (−3.1 to −4.4).

So the second escape from an isotropically perturbed saddle shows the same pattern as the full
dynamics: H2's exponents for k ≤ 4 and for binary8 `far`, extinction for some seeds already at
k = 4, and mostly no escape for k ≥ 5 on MPS6 and k = 7 on binary8. The perturbations at σ = 0.2–0.3
are not small compared with the O(1) learned cores, so the all-σ fits for MPS6 and binary8 include a
pre-asymptotic range.

## Verdict on H2

**Partly supported.**

- **Supported:** the first escape (t1 ∝ σ^−(K−2), exact) and the locality claim. The same teacher
  gives no second plateau (k = 1), a log(1/σ) plateau with the predicted coefficient (k = 2), or a
  σ^−(k−2) plateau (k = 3, 4; binary8 k = 5, 7; some MPS6 seeds at k = 6), depending on how many
  cores of the given tree must change.
- **Not supported as stated for k ≥ 4:**
  1. The second escape often never happens: 95 of 332 runs with k ≥ 4 converge to (or stay at) the
     rank-one saddle, from 0 of 8 seeds (MPS4 `global`) to 5 of 8 (MPS4 `far`) and 3 of 6
     (MPS6 sites {0,4}).
  2. On MPS6 with k = 5 and 6 the escape-time exponent is seed-dependent and often one to two units
     steeper than −(k−2).

  H2 needs an extinction/pinning amendment: k sets the exponent only when the new component's
  chain survives intact.

Runs started exactly at the rank-one saddle (second escape alone) show the same pattern.

## Caveats

- Small model: d = 2, χ = 2 (the minimal bond dimension for a rank-2 teacher, so the new component
  has exactly one free channel per bond), teacher weights (1, 0.5), one teacher seed. Whether the
  collapse survives larger χ is untested (`configs/chi4.yaml` is prepared but was not run).
- σ ranges: two decades for N = 4, under one decade for the larger trees (MPS6 0.05–0.3, binary8
  0.07–0.3), where the largest σ are pre-asymptotic (e.g. MPS6 `local`); the "smallest 4 σ" fits are
  the more asymptotic ones. t_first exponents are precise to ±0.003 per seed; gap exponents for
  k ≥ 5 differ between seeds by up to 2.
- "No second escape" means not learned within the horizon: until the collapse rule fired
  (≥ 30·t_first, validated as described in Setup) or after 160–7300·t_first. 4 runs were still
  rising at the horizon.
- 6 seeds for the larger trees and 8 for N = 4, so the collapse fractions are rough.
- The interpretation of the collapse and of the steeper exponents is a conjecture.

## Runtime

Median wall time per run (one core of a heavily shared machine): 3 s (binary4), 5 s (MPS4), 10 s
(binary8), 15 s (MPS6); 90th percentiles 4–50 s. Runs without a second escape at small σ use the
wall-clock limit (400–900 s). LSODA needs ~1e3–1e4 steps per run regardless of how long the plateaus
are. With 8 processes: N = 4 sweep (448 runs) 20 min, larger trees (336 runs) 21 min, extension
(19 runs) 15 min, saddle start (340 runs, median 2–8 s per run) about 20 min.

## Files

- `h2sim.py`: one run (teacher, init, LSODA gradient flow, crossing detection, collapse rule).
- `run_sweep.py`: parallel sweeps from a YAML config (≤ 8 processes, resumable).
- `analyze.py`: fits, tables, figures; `check_accuracy.py`: integration cross-checks.
- `configs/`: `n4.yaml`, `large.yaml`, `extend.yaml`, `saddle.yaml`, `accuracy.yaml`, `chi4.yaml`
  (not run).
- `results/`: `fits.csv`, `fits.md`, `per_seed.csv`, `runs_derived.csv`, `saddle_fits.csv`,
  `saddle_fits.md`, `saddle_per_seed.csv`, `accuracy.csv`, and the figures `fig_t_first.png`,
  `fig_gap.png`, `fig_kscan.png`, `fig_local.png`, `fig_loss_curves.png`,
  `fig_loss_local_aligned.png`, `fig_bond_s2.png`, `fig_saddle_escape.png`.
- Raw data (gitignored): `data/pilot_h2_locality/{runs,extend,saddle,accuracy}/` and the full
  per-run tables `data/pilot_h2_locality/*.csv`.
