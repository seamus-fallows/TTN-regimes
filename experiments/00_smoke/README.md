# 00_smoke: does the pipeline reproduce stage-like learning?

Status: **complete** (smoke test of the library and the runner; also the compute benchmark for the
plan). All results are **empirical** unless marked otherwise.

## Question

Does the full pipeline (YAML config -> `ttn.runner` -> gradient flow -> recorder -> diagnostics ->
figures) reproduce the known stage-like (saddle-to-saddle) learning of a small TTN from small
initialization? Concretely: plateaus at the partial-fit levels L_k, bond singular values that grow
one at a time to the teacher's, components learned in order of weight, a first escape time that
scales as sigma^-(K-2), and how that time compares with the exact prediction for an aligned init.
This is a known result (Saxe et al. 2014 for deep linear networks; aligned TTNs reduce to the same
mode equations), not a new claim.

## Setup

- Model as in `notes/model.md`: binary tree with N = 4 sites, K = 3 cores (two bottom cores with
  sites {0,1} and {2,3}, one root), d = 3, chi = 3; whitened Gaussian inputs; exact population loss
  L = 1/2 ||W - W*||^2; float64.
- Teacher: odeco, weights (3, 2, 1), Haar-random site bases, **teacher seed 0 in every run** (the run
  seed changes only the init). Partial-fit (saddle) levels L_k = 1/2 sum_{r>k} s_r^2 = 7, 2.5, 0.5, 0.
- Init: fan-in (`TTNSpec.init`), per-core scale sigma, drawn from the run seed's init stream
  (`ttn.config.stream_seed`).
- Dynamics: gradient flow, Dormand-Prince 5(4) (`integrate_gf`, rtol 1e-8, atol 1e-12), stopped when
  L < 1e-10 (t_max = 1e5). Recorded at 200 log-spaced log times (`default_log_times`) plus "fine
  rows" (loss, overlaps, spectra) at the integrator steps where theta or the loss changed by 1%
  (`ttn.dynamics.Recorder`): the log grid alone is 8% of t apart, wider than the escapes, while the
  integrator takes 20-50 steps per escape.
- Diagnostics (`notes/diagnostics.md`), on log and fine rows together (`ttn.runner.learning_events`):
  component overlaps a_r, bond spectra, tangent kernel on 128 probe inputs (delta_K),
  `find_plateaus`, events (last upward crossing of a_r = 1/2; width = final rise 0.1 -> 0.9) and
  separation ratios, with a resolution check (width >= 3 sample intervals across the rise), regime
  label (`classify_regime`).
- Runs: `configs/base.yaml` (sigma = 0.05, seed 0) and the sweep `configs/grid_sigma.yaml`
  (sigma in {0.1, 0.05, 0.02, 0.01, 0.005}, seeds 0-7, 40 runs).
- Theory comparison: `ttn.theory.escape_time(K=3, s=3, a0=sigma)`, the exact time for a balanced
  **aligned** init (cores diagonal in the teacher basis, amplitude a0 = sigma on every core) to reach
  a_1 = 1/2. A random fan-in init is not aligned: its effective amplitude along the teacher's first
  component is not sigma, and it must first align with it. The two therefore differ by an O(1)
  factor; only the exponent -(K-2) = -1 is a parameter-free prediction (**derived** for aligned init;
  **known** for the first escape from generic small init). The factor is measured below.

## How to run

From the repository root:

```
uv run python -m ttn.runner experiments/00_smoke/configs/base.yaml --out data/00_smoke/base
uv run python -m ttn.runner experiments/00_smoke/configs/base.yaml \
    --grid experiments/00_smoke/configs/grid_sigma.yaml --out data/00_smoke/grid_sigma --workers 8
uv run python experiments/00_smoke/analyze.py
```

Runtime on a 24-core machine shared with other agents' jobs (load average ~16): 2.5 min in total
(base run 7.5 s, of which 3.2 s training; sweep 2 min 13 s with 8 workers, 3-99 s per run;
analysis 6 s). Reruns reproduce every trajectory bitwise (checked: all 40 sweep runs have the same
theta_end, log times and losses as the runs made before the recorder and summary changes); only
the numpy-computed kernel diagnostics can differ in the last bit when the BLAS thread count
differs.

Raw outputs (gitignored) go to `data/00_smoke/base/` and `data/00_smoke/grid_sigma/`: per run
`<name>.npz` (the recorded trajectory and final parameters) and `<name>.json` (summary, full
config, git commit, hash of the library code, seeds, timing, thread counts, versions), and
`summary.csv` (one row per run). `analyze.py` writes the figures and `results/escape_times.csv`
(per sigma), `results/seeds.csv` (per seed: fits across sigma) and `results/runs.csv` (per run).
The outputs of the first version of this experiment (log rows only) are kept in
`data/00_smoke/pre_fix/`.

## How to run variants

- Change any field with dotted-key overrides, e.g. an MPS with LSODA:
  `uv run python -m ttn.runner experiments/00_smoke/configs/base.yaml tree.kind=mps optimizer.method=lsoda --out data/00_smoke/mps4`.
  All fields and defaults are in `ttn/config.py` (teachers, init schemes aligned / antisymmetric /
  centered, input covariances, empirical loss, GD / SGD, Hessians at chosen times, ...).
- More seeds: `--seeds 0-19` overrides the grid file's `seeds`; finished runs are skipped, so a sweep
  can be extended or resumed after an interruption. A finished run is reused only if its stored
  config is the requested one (otherwise the sweep stops before running anything: use another
  `--out`); `--recompute` recomputes the stored summaries after a change to `ttn.runner.summary`.
  One sweep per output directory at a time (`<out>/.sweep.lock`; delete it if a killed sweep left
  it behind).
- Other sweeps: a grid file maps dotted keys to value lists (outer product), e.g.
  `{init.sigma: [0.1, 0.01], chi: [3, 6]}`; `--dry-run` lists the run names (and validates the
  configs: unknown keys and unknown choices such as `optimizer.method=LSODA` are errors).
- Runs that are averaged over seeds need one time grid: `logging.log_times=[...]` replaces the
  default log grid. Lazy-regime runs need full bonds (or report the lazy floor, `summary`'s
  `lazy_floor` and `loss_end_over_floor`): with a deficient bond they end at the projected fit.
- Later plateaus are stiff (after the first escape the Hessian has O(1) eigenvalues while the next
  plateau can be long): for K >= 4 or long later plateaus use `optimizer.method=lsoda`.

## Results

Data: runs of 2026-10-09 (second version, after the library review) at commit 1c57061 plus the
uncommitted library fixes; every run records `git.ttn_dirty = true` and the hash of the library
code `git.ttn_sha256 = b2e44e52...` (`ttn.runner.code_hash`). Statistics: sigma only rescales one
fixed init direction per seed, so the 40 runs are 8 independent samples (seeds) at 5 scales; every
mean +- SE pooled over sigma is over the 8 seeds of a per-seed quantity, and exponents are fitted
per seed.

1. **Stage-like learning in the base run** (`results/loss`, `results/overlaps`,
   `results/bond_spectra`). The loss sits at L = 7, 2.5 and 0.5 and drops between them;
   components are learned in order of weight (a_r = 1/2 at t = 15.36, 31.31, 77.73); on each edge
   the bond singular values rise one at a time, from initial values of 1e-5 to 1e-4, to the
   teacher weights 3, 2, 1. Both edges induce the same bipartition {0,1} | {2,3}, so their spectra
   coincide, as they must. The escapes are short: widths (0.1 -> 0.9) 0.49, 0.80 and 2.02 time
   units, so the separation ratios are rho = 19.9 and 23.0. `find_plateaus` finds three plateaus;
   `classify_regime` says saddle-to-saddle (delta_K = 1.5e6, far from lazy). Conservation-law drift
   1e-10.
   Resolution: up to the stop at t = 86 the log grid has 109 points, 8% of t apart (1.2-5.8 time
   units at the events), wider than the escapes; the 233 fine rows sample the rises every 0.04-0.17
   time units. A dense linear grid (8001 points) gives the same widths to 0.2% and rho = 19.94 for
   the smallest ratio (19.92 here). The first version of this README quoted widths 1.0, 1.9, 4.7
   and a smallest ratio of 8.0 from the log grid alone: those measured the grid, not the dynamics.
2. **The sweep.** All 40 runs converge (L < 1e-10), all learn the components in the order 3, 2, 1,
   all are labelled saddle-to-saddle, and `find_plateaus` finds three plateaus in all 40. Every event
   is resolved (width >= 3 sample intervals) and no dormant escape signal comes near the float64
   floor (smallest 3e-8 of |W| in the base run). The smallest separation ratio is 4.91 (sigma = 0.1,
   seed 4, events 2-3; its events 1-2 are 14.1 widths apart): S2S at the threshold 3, but 'rich'
   at the threshold 5 of the notes' sensitivity check. The first version reported this run as
   two plateaus with drops 2.7 widths apart; that was the grid too.
3. **Separation ratios vs sigma** (`results/separation`). Both ratios grow as 1/sigma: per-seed fits
   give rho_1 ~ sigma^p with p = -1.017 +- 0.005 and rho_2 with p = -1.006 +- 0.003 (mean +- SE over
   8 seeds), against -(K-2) = -1 (theory_regimes R9, heuristic: the gaps between events scale like
   the first escape time if the dormant amplitudes entering each plateau stay proportional to
   sigma; widths are O(1)). The per-seed median of rho_1 sigma is 1.36 +- 0.19 (range 0.76-2.15;
   theory_regimes measured a median of 0.9 over 30 seeds at sigma = 0.06-0.24). Extrapolating
   rho ~ 1/sigma from sigma = 0.1 puts each seed's S2S boundary (rho_min = 3) at
   sigma_c = sigma rho_min / 3 = 0.16-0.38, median 0.32, as in theory_regimes R9 (median 0.31,
   quartiles 0.22-0.39). This is condition (iii) of the S2S signature for the gaps themselves;
   `plateau_scaling` on (sigma, t_1) confirms it for the first plateau.
4. **First escape time vs sigma** (`results/escape_time`, `results/escape_times.csv`,
   `results/seeds.csv`). Per seed, t_1 (a_1 = 1/2) ~ sigma^p with p = -1.0071 +- 0.0011 (mean +- SE
   over 8 seeds; range -1.011 to -1.002), against -(K-2) = -1; the aligned-init theory has
   -1.019 over the same range. The small offset is the O(1) term: per seed,
   t_1 = A sigma^-1 + B fits with B = -0.17 +- 0.01.
5. **The O(1) factor.** Random init escapes later than the aligned init with a0 = sigma: the
   per-seed coefficient A is 2.40 +- 0.26 times the aligned one, 1 / ((K-2) s_1) = 1/3 (range
   1.52-3.56), and the per-seed mean of t_1(random) / t_1(aligned) over sigma is 2.43 +- 0.26
   (range 1.53-3.61). The factor does not depend on sigma (2.50, 2.44, 2.41, 2.41, 2.40 from
   sigma = 0.1 to 0.005, each +- 0.26). It reflects the init's effective amplitude along the first
   component, so it is a property of the init distribution, not of the dynamics. (The first version
   gave +- 0.11, an SE over the 40 runs as if they were independent.)
6. Side observation (not followed up here). Later components are not independent modes under
   random init. Relative to independent aligned modes, the second escape comes 3.8-4.0 times later
   (per sigma, SE 0.4 over seeds) and the third 7.2-7.7 times later (SE 2.3), with a large seed
   spread: in seed 1 the third component escapes at t = 451 for sigma = 0.05, against 58-146 in the
   other seeds. Later escapes depend on how earlier phases rescaled the dormant directions
   (PLAN.md, Q2).

Conclusion: the pipeline reproduces the known saddle-to-saddle phenomenology with the predicted
exponents, now for the separation ratios as well as for the first escape, and it labels the runs
with the diagnostics of `notes/diagnostics.md` on events resolved in linear time. On the log grid
alone the event widths, separation ratios and plateau counts were grid artifacts (they hid the
1/sigma growth of the ratios: per-seed exponents of rho_1 between -0.07 and -0.33, median -0.20),
while the escape times and their exponent were right to ~1-3%.

## Benchmark (compute plan)

`benchmark.py` measures the cost of gradients and of whole gradient-flow runs; see its docstring for
the exact protocol.

```
uv run python experiments/00_smoke/benchmark.py --part grad              # ~1 min
uv run python experiments/00_smoke/benchmark.py --part gf --workers 8    # ~35 min, resumable
```

Measured on 2026-10-09 on a 24-core Windows desktop that other jobs kept at 95-100% load, one
thread per process (8 worker processes for part 2, 34 min wall in total). CPU times are the
reliable numbers; wall times were 1.2-1.7x larger. Columns are (d, chi); bonds are capped at the
rank their bipartition allows (so chi = 8 is 4 on some MPS edges and on binary4 with d = 2).
Teacher: odeco with R = min(3, d, chi) components (weights 3, 2, 1).

Changes since these measurements (library review, same day; measured on binary4, binary8, mps8
and binary16 with d = 3): inside the integrators a log row costs 0.7-0.8 gradients (the
integrator passes its loss and velocity; 2.2-2.5 when recomputed, as in part 1 below); a fine row
costs 0.6 gradients (233 fine rows in the base run, ~6% of its cost); the tangent kernel on 128
probes is one batched backward pass, 1.5-10 gradients (it was one pass per probe point, ~30
gradients on binary4: 44% of the base run); Hessians are vectorized (0.1 s instead of 2.3 s at
P = 1376); `integrate_gf` runs LSODA / Radau with one OpenBLAS thread by itself (threadpoolctl), so
the oversubscription of item 1 below no longer needs OPENBLAS_NUM_THREADS=1. The gradient-flow
runs of part 2 are unaffected (same steps and gradient counts).

**Cost per gradient** (CPU ms; one loss + gradient, 6 per DP5 step). It grows with the number of
cores, about 0.27 ms per core, and does not depend on d or chi in this range (20 to 4160
parameters). The cost is per-operation overhead of the tree contraction and autograd, not
arithmetic. One Recorder log entry costs about 2.3 gradients.

| tree (K) | 2, 2 | 2, 4 | 2, 8 | 3, 2 | 3, 4 | 3, 8 | 4, 2 | 4, 4 | 4, 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| binary4 (3) | 0.8 | 0.9 | 0.9 | 0.9 | 0.9 | 0.7 | 0.9 | 0.9 | 0.8 |
| binary8 (7) | 1.8 | 1.8 | 1.9 | 1.9 | 1.7 | 2.1 | 1.9 | 1.6 | 2.0 |
| binary16 (15) | 4.0 | 3.1 | 3.6 | 3.7 | 3.5 | 3.8 | 3.9 | 3.5 | 3.7 |
| mps8 (8) | 2.0 | 2.0 | 2.2 | 2.0 | 2.2 | 2.2 | 2.1 | 2.1 | 2.0 |
| mps16 (16) | 4.0 | 4.0 | 3.8 | 4.2 | 4.9 | 5.2 | 4.7 | 4.6 | 4.1 |

**Parameters**:

| tree (K) | 2, 2 | 2, 4 | 2, 8 | 3, 2 | 3, 4 | 3, 8 | 4, 2 | 4, 4 | 4, 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| binary4 (3) | 20 | 48 | 48 | 40 | 88 | 208 | 68 | 144 | 320 |
| binary8 (7) | 52 | 208 | 384 | 92 | 288 | 1376 | 148 | 400 | 1600 |
| binary16 (15) | 116 | 528 | 1728 | 196 | 688 | 3712 | 308 | 912 | 4160 |
| mps8 (8) | 56 | 168 | 424 | 84 | 282 | 930 | 112 | 416 | 1312 |
| mps16 (16) | 120 | 424 | 1448 | 180 | 666 | 2466 | 240 | 928 | 3360 |

**One exact Hessian** (CPU; LSODA's Jacobian, vectorized autograd). It grows roughly as P^2 above
P ~ 1000, and LSODA also factorizes a P x P matrix at each Jacobian:

| tree (K) | 2, 2 | 2, 4 | 2, 8 | 3, 2 | 3, 4 | 3, 8 | 4, 2 | 4, 4 | 4, 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| binary4 (3) | 2.5 ms | 2.8 ms | 2.7 ms | 2.5 ms | 2.7 ms | 3.7 ms | 3.2 ms | 3.2 ms | 4.5 ms |
| binary8 (7) | 6.4 ms | 9.5 ms | 17 ms | 6.3 ms | 12 ms | 0.18 s | 7.5 ms | 15 ms | 0.22 s |
| binary16 (15) | 16 ms | 38 ms | 0.33 s | 16 ms | 55 ms | 1.9 s | 20 ms | 94 ms | 2 s |
| mps8 (8) | 7.1 ms | 9.2 ms | 17 ms | 7.4 ms | 13 ms | 57 ms | 7.7 ms | 17 ms | 0.12 s |
| mps16 (16) | 13 ms | 29 ms | 0.2 s | 20 ms | 59 ms | 0.66 s | 21 ms | 86 ms | 1.2 s |

**One gradient-flow run with LSODA** at a fixed function-space init scale sigma^K = 1e-3, to
t = 1e6 (or until L < 1e-10): CPU s, with the number of components learned (a_r >= 1/2 at the end)
in brackets. Each run took 900-8600 gradient evaluations and 0-563 Hessians:

| tree (K) | 2, 2 | 2, 4 | 2, 8 | 3, 2 | 3, 4 | 3, 8 | 4, 2 | 4, 4 | 4, 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| binary4 (3) | 3 (2) | 3 (2) | 3 (2) | 3 (2) | 3 (3) | 6 (3) | 3 (2) | 5 (3) | 6 (3) |
| binary8 (7) | 10 (2) | 16 (2) | 17 (2) | 14 (2) | 22 (3) | 176 (3) | 18 (2) | 22 (2) | 189 (3) |
| binary16 (15) | 24 (1) | 42 (1) | 363 (2) | 19 (1) | 50 (1) | 1705 (2) | 30 (1) | 42 (1) | 836 (1) |
| mps8 (8) | 8 (1) | 15 (1) | 23 (2) | 11 (1) | 17 (1) | 72 (2) | 15 (1) | 24 (2) | 146 (2) |
| mps16 (16) | 29 (1) | 36 (1) | 168 (1) | 10 (0) | 50 (1) | 460 (1) | 7 (0) | 11 (0) | 642 (1) |

**DP5 for d = 3**, capped at 5000 steps (same init scale). "First escape" is the first log time
at which some a_r >= 1/2 (logs about 3-10% apart), "steps before it" the step count at the log
time before; "after it" counts the remaining steps and the time they advanced (from the first
log time after the escape, so for the runs that stopped right after escaping it is ~0):

| tree (K) | chi | first escape t | steps before it | after it: steps, time advanced | stop |
|---|---:|---:|---:|---|---|
| binary4 (3) | 2 | 15.7 | 126 | 304, 44.4 | converged at t = 60.1 |
| binary4 (3) | 4 | 5.78 | 113 | 284, 27.3 | converged at t = 33.1 |
| binary4 (3) | 8 | 7.16 | 115 | 4885, 1.19e+03 | max_steps at t = 1.2e+03 |
| binary8 (7) | 2 | 989 | 105 | 4895, 317 | max_steps at t = 1.31e+03 |
| binary8 (7) | 4 | 151 | 93 | 4907, 336 | max_steps at t = 487 |
| binary8 (7) | 8 | 218 | 91 | 4909, 644 | max_steps at t = 862 |
| binary16 (15) | 2 | 1.05e+04 | 101 | 4899, 60 | max_steps at t = 1.06e+04 |
| binary16 (15) | 4 | 8.02e+04 | 155 | 4845, 0 | max_steps at t = 8.02e+04 |
| binary16 (15) | 8 | 3.8e+04 | 126 | 4874, 0 | max_steps at t = 3.8e+04 |
| mps8 (8) | 2 | 1.46e+03 | 104 | 4896, 244 | max_steps at t = 1.7e+03 |
| mps8 (8) | 4 | 497 | 98 | 4902, 266 | max_steps at t = 763 |
| mps8 (8) | 8 | 347 | 108 | 4892, 562 | max_steps at t = 909 |
| mps16 (16) | 2 | none by 1e6 | 206 (whole run) | | completed |
| mps16 (16) | 4 | 8.94e+04 | 152 | 4848, 0 | max_steps at t = 8.94e+04 |
| mps16 (16) | 8 | 1.5e+04 | 102 | 4898, 0 | max_steps at t = 1.5e+04 |

**What this means for the compute plan**

1. Parallelize over runs, one process per core, with one torch thread and one BLAS thread each
   (the runner's sweep workers do this). A gradient takes about 0.27 ms per core, and
   multithreading cannot speed up operations this small. Multithreaded BLAS is actively harmful
   on a busy machine: with numpy/scipy's default 24 OpenBLAS threads, the binary16 (d = 3,
   chi = 4) LSODA run above took 175 s of wall time and 2314 s of CPU. With one thread it took
   68 s and 50 s. A 688 x 688 LU took 0.8 s instead of 5 ms.
2. LSODA, which is needed beyond the first escape for K >= 4, costs per run:
   - K = 3: 3-6 s CPU.
   - K = 7-8: 10-25 s at chi <= 4, and 1-3 min at chi = 8 with d >= 3 (P ~ 900-1600).
   - K = 15-16: 7-50 s at chi <= 4, and 3-30 min at chi = 8, where Hessians and LU
     factorizations dominate once P is 1500-4000.

   LSODA's cost grows only logarithmically with t_max. A sweep of 100 runs therefore costs about
   1 CPU hour for K <= 8 at chi <= 4, and 5-50 CPU hours for K = 15-16 at chi = 8.
3. DP5 crosses a first plateau in 90-155 steps (600-1000 gradients, about 5 s CPU at most, at
   K = 16), whatever its length (t = 6 to 9e4). After the first escape it is stability-limited, at
   0.01-0.15 time units per step: the remaining ~4900 steps advanced binary8 and mps8 by only
   240-650 time units, and binary16 and mps16 by less than one log interval (60 to < 1e3). Use it
   for K = 3, for first escapes, or with a loss floor; otherwise use LSODA.
4. At a fixed function-space scale, plateaus lengthen steeply with K. At sigma^K = 1e-3 the
   first escape comes at:
   - t = 4-16 for K = 3;
   - t = 60-8000 for K = 7-8;
   - t = 1.6e3-9e5 or not at all by t = 1e6 for K = 15-16.

   By t = 1e6, the trees with K >= 8 had learned only 0-2 of their components. Complete
   saddle-to-saddle runs on large trees need a larger sigma, or must be restricted to the first
   escapes.
5. An overparameterized student (chi > R) approaches W* algebraically at the end. L < 1e-10 is
   then reached only at t ~ 1e5-1e6 (binary4 with chi = 8: t = 4.6e5) or not at all. Use a
   relative floor, or judge runs by their events.
6. Memory: about 0.3 GB per worker process (torch), and up to about 1 GB at P ~ 4000 (dense
   Hessian).
