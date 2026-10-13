# route_generic: does "depth = number of cores that change" hold outside pattern targets?

Agent `route-generic`, branch `agent/route-generic`, 2026-10-13. Task: `notes/briefs/route_theory.md`
sections 1-3 and 4.4. Experiment folder: `experiments/route_generic/`. Raw outputs:
`/root/outputs/route-generic/` (not in git). Status tags: **proven**, **derived** (under stated
assumptions), **empirical**, **conjecture**.

Part I (sections 1-7) is the pre-registration, written and committed before any sweep run. After the
sweeps only Part II (results) and Part III (deviations, with reasons) are added.

# Part I: pre-registration

## 1. Question and the picture under test

The paper (`report/ttn_saddle_order_and_aligned_training_dynamics.pdf`) proves its results for pattern
targets W* = sum_a s_a U_a (orthogonal one-hot product tensors). Its qualitative picture of training from
small init sigma is:

- learning happens in steps (plateaus separated by fast learning events);
- each step has an **effective depth d** equal to the number of cores that must change for it (the route
  depth c_J(r) = |R_J(r)| for pattern targets), visible in the growth law of the emerging amplitude,
  dy/dt = d gamma^(2/d) y^(2-2/d) (s - y) (eq. 1.4 / 6.15);
- the plateau before it lasts ~ sigma^-(d-2) (log(1/sigma) for d = 2) (eq. C.6).

Question: outside pattern targets (random CP teachers with non-orthogonal components, random TTN
teachers, MNIST), do the three numbers below agree event by event, and where do they stop agreeing?

1. **k_region**: the bonds that gain rank at the event and the number of cores in the smallest subtree
   containing those bonds' end nodes ("cores that change");
2. **effective depth d_eff**: from the growth law of the emerging bond singular value
   (`ttn.routes.effective_depth`, the estimator of `experiments/locality/growth.py`);
3. **duration exponent p**: plateau duration ~ sigma^-p, per-seed fits over >= 1.5 decades of sigma.

Prediction of the picture: d_eff = k_region and p = k_region - 2 (p = 0, a log(1/sigma) law, when
k_region = 2).

What is already known and is cited, not redone (sources: `notes/data_teacher.md` R2.0/R2.1 and the "safe
to cite" list of `notes/reviews/data_teacher_round3_verification.md`; `notes/real_data.md` section 8 and
`notes/reviews/real_data_round2_verification.md`): (i) for non-orthogonal CP, misaligned, random TTN
teachers the escape from the bond-rank-one saddle on MPS students (N = 4, 8) is a strict saddle with
a log(1/sigma) time and slope 1/gamma_e of the edge that grows (within 0.4 % from a saddle start,
1.3 % from random init), while binary and HT students escape in ~ sigma^-1 (a 3-core route through an
internal core); (ii) random-TTN-teacher bond ranks grow one bond at a time on MPS-4; (iii) on MNIST
(4 x 4, binary16 and mps16) the first escape is exactly sigma^-(K-2) (an identity of small-init flow),
the first binary route is a sibling pair of bottom cores (k = 3) with gap A/sigma + B (log law
rejected), and 6 of 9 later MPS edges rise in log time while edges 1 and 2 rise with a power-law-like
sigma dependence (unexplained). New here: the per-event comparison of k_region, d_eff and p, for every
event (not only the first rank increase), with the growth-law depth, on mps6 / binary8 and on MNIST.

## 2. Cells, grids, seeds

Model (brief section 3): whitened inputs, exact population loss L = 1/2 ||W - W*||^2 evaluated on the
dense tensor, fan-in Gaussian init `spec.init(sigma, seed)` on all parameters (the same Gaussians at
every sigma for a seed), float64, `torch.set_num_threads(1)` and BLAS pinned to one thread per worker.
Trees: **mps6** (K = 6 cores, 5 bonds) and **binary8** (K = 7 cores: 4 bottom cores with 2 sites each,
2 internal cores, a root; 6 bonds).

| target (CSV `target`) | definition | d | chi (student) | teachers x init seeds |
|---|---|---|---|---|
| `pattern` (control, inside the theory) | mps6: 000000 (1), 001100 (0.7), 101000 (0.5), 111110 (0.35); predicted route depths after the first pattern: 2, 3, 4 (regions {2,3}, {0,1,2}, {1,2,3,4}). binary8: 00000000 (1), 00000111 (0.7), 01000100 (0.5), 10100100 (0.35); depths 3, 4 (a splice: pairwise cost 5), 3. Both satisfy the star condition (`RouteTarget.is_star`) | 2 | max blocks + 2 = 5 | 1 x 30 |
| `cp03`, `cp06` | 3 rank-one components, per-site overlap c = 0.3 / 0.6 between every pair at every site (Gram (1 - c) I + c 11^T, Cholesky rows, Haar rotation per site), weights uniform in [0.3, 1], sorted, largest = 1 | 3 | 5 | 5 x 6 |
| `ttn` | random TTN on the student's tree, Gaussian cores, bond dimension 2, ||W*|| = 1 (`random_ttn_teacher`) | 3 | 4 | 5 x 6 |
| MNIST | existing pipeline `experiments/real_data/run.py` unchanged: MNIST 3 vs 5, 4 x 4 pixels (16 sites), raw cos/sin features, binary16 quadtree (K = 15) and mps16 snake (K = 16), chi = 4, n_train = 4000, phase 2 horizon tau = 1e4 | 2 | 4 | 10 init seeds (10 more if time allows) |

Teacher seeds 1-5 (`teacher_generator`, independent of init seeds), init seeds 0-29 (pattern) / 0-5.
The trees mps6 and binary8 cannot be used for MNIST without changing the pipeline's image size; the brief
asks for MNIST "as far as the existing real-data pipeline allows", so MNIST uses its 16-site trees.

Sigma grids: synthetic **sigma = 10^-3.5 ... 10^-1 on a quarter-decade grid (11 values)**; MNIST
**sigma = 0.005 ... 0.5, quarter decades (9 values)**.

Integration (synthetic, `run.py`): scipy's `LSODA` class stepped directly (dense output never used,
`notes/saddle_dyn.md` section 8, item 3), exact Hessian as Jacobian, rtol 1e-10, atol 1e-15; every
accepted step recorded (time, loss, top bond singular values of every edge, target amplitudes
<U_a, W> for CP targets). Clocks: absolute t, and tau = t - t_a with the solver re-zeroed exactly at the
first escape (first step with L < L0/2), so that later gaps are float64-resolved; the solver is also
re-zeroed when its step falls below 1e-11 of its local time (item 5), failed steps are retried from the
last good step with first step 1e-2 (at most 5). Stops: L < 1e-7 L0 ("converged"), tau > 1e9
("unfinished"), 3600 s wall ("wall"), 4e5 steps ("stalled"). MNIST: the real_data runner (DP5 to the
first escape, then LSODA with the Gauss-Newton Newton matrix on the tau clock), rtol 1e-8, atol 1e-14.

Timing pilot (before this registration; not part of the results): one seed per synthetic target and
tree at sigma = 1e-3 and 0.1 (`configs/pilot.yaml`, 16 runs: 8-150 s each, all converged), and 4 MNIST
runs at sigma = 0.01, 0.05 with tau = 2000 (150-250 s). Disclosed: I looked at the event tables of the
pilot runs to check the event definitions below (section 3). What I saw: on the pattern targets the
bond-based events reproduce the route depths (binary8, sigma = 1e-3: k_region 3, 4, 3 with d_eff 3.0,
4.2, 3.2); on the CP targets on mps6 all five bonds gain rank in each later event (k_region = 6) while
d_eff was 2.0 (c = 0.6) and 1.4-1.8 (c = 0.3) with durations that changed by less than 3x over two
decades of sigma; on the random TTN teacher on mps6 single bonds gained rank (k_region = 2, d_eff
1.5-2.0). The definitions were not changed after seeing this, except adding the component count of an
event's bond set (section 3, `n_components`) because two non-adjacent bonds rose together once.

## 3. Definitions (estimators)

All on the recorded solver steps; code `experiments/route_generic/events.py` (synthetic) and
`mnist_events.py` (MNIST).

- **Target values.** Synthetic: the singular values s_k^{e,*} of W* across every edge e (dense SVD);
  the rank of e is the number with s_k^{e,*} > 1e-10 s_1^{e,*}. Only these k are tracked.
- **Rise** of (e, k): rise time = last upward crossing of s_k^e(t) = 0.5 s_k^{e,*} (log-interpolated
  between steps; the brief's "last upward crossing of half the target"), onset = last upward crossing of
  0.05 s_k^{e,*} before it. A pair (e, k) whose value at the end is below 0.5 s_k^{e,*} is "not learned".
  MNIST (no target): the ratio r_k^e = s_k^e / s_1^e of the data-weighted spectra (`data.function_spectra`)
  for k >= 1; rise time = last upward crossing of r = 0.05 (the real_data convention), onset = last
  upward crossing of 0.005 before it.
- **Learning event**: rises whose [onset, rise time] intervals overlap (transitively) form one event.
  Event time T_j = the earliest rise time in the group. Event 0 is the group that contains the first
  singular values (escape from the origin); on MNIST event 0 is the first escape (phase 1 of the runner,
  t1 = t_a), and any rise whose onset precedes the rate peak of the first loss drop is merged into it.
- **Bonds that change** = the edges of the event's rises. **k_region** = number of cores in the smallest
  subtree containing their end nodes (edge e joins core e and its parent). Also recorded:
  `n_components` (connected components of the set of end-node pairs; > 1 means separate routes rose at
  the same time) and `k_max_component` (cores of the largest component).
- **Duration** of the plateau before event j >= 1: T_j - T_{j-1} on the tau clock; for event 0 the
  absolute time of its first rise (MNIST: t_a, the runner's end of phase 1).
- **Effective depth** of a rise: `ttn.routes.effective_depth(t, s_k^e(t), s_k^{e,*})` on the steps of the
  plateau before the event (between T_{j-1} and the rise time; event 0: from t = 0), i.e. the slope
  2 - 2/d of log[(dy/dt)/(s - y)] against log y for y/s in [1e-4, 1e-2], >= 6 steps. MNIST: on r_k^e(t)
  with s = r_k^e at the end of the run. The event's d_eff = median over its rises with an estimate.
  Secondary (reported, not used in the decisions): the same with the window [1e-3, 1e-1]. Pattern
  control: also the brief's amplitude version (`effective_depth` on P_a = <U_a, W> for the pattern whose
  amplitude crosses s_a/2 at the event).
- **Floor flag** (brief section 3): an event is flagged if any of its rises has min s_k^e / s_1^e < 1e-14
  (100x the ~1e-16 accuracy of the SVD) on the plateau before its onset (the escape signal is then
  within 100x of roundoff). Flagged events are excluded from fits and counted. MNIST: the data-weighted
  spectra are accurate only to ~1e-8 s_1 (square root of eigenvalues); rises whose ratio stays below
  1e-6 on the plateau are counted as "below the measurement floor" (onset not resolved; rise time still
  valid).
- **Run outcome**: converged / unfinished / wall / stalled / failed (solver) / error, as recorded.
  **Trapped** (post hoc): not converged and some target singular value never rose, with its final value
  below 1e-10 of its target (a dead route). **Forgotten / relearned**: a rise with more than one upward
  crossing of the half level (`relearn` count).

## 4. Series and fits

- A **series** = (target, tree, teacher seed, init seed, event signature), the signature being the set
  of rises already completed before the event plus the event's own rise set ("conditional on the state
  reached", brief section 3). The same seed at different sigma is the same init direction rescaled.
- **Fit window**: the points of the series with the event present, not floor-flagged and duration > 0,
  sorted by sigma; the window is the first 7 of them (the small-sigma end). A fit needs >= 5 points that
  span >= 1.5 decades. Series failing this are counted, not fitted.
- **Duration exponent p**: least squares of log(duration) on log(sigma) in the window, p = -slope (with
  its OLS standard error). Prediction p_pred = k_region - 2 (p_pred = 0 for k_region = 2).
- **Log law vs power law** (for every fitted series): duration = A ln(1/sigma) + B (weights 1/duration,
  i.e. relative residuals) against the power law; each scored by the rms of log(fit/data); "log
  preferred" if its rms is smaller.
- **Series d_eff**: median of the event's d_eff over the window points where it is measured (>= 1).
- Event 0 (escape from the origin) is reported as a check of the estimators (its p = K - 2 is an exact
  identity of small-init flow) and is never pooled with the later events.

## 5. Decision rules (per target x tree, later events pooled; also broken down by k_region)

Over fitted series: Dd = d_eff - k_region, Dp = p - (k_region - 2). Reported: median, quartiles,
fraction with |D| <= 0.3, for both.
- **HOLDS**: median |Dd| <= 0.3 and median |Dp| <= 0.3 and >= 70 % of series within 0.3 for both,
  with >= 10 fitted series.
- **FAILS**: median |Dd| > 0.5 or median |Dp| > 0.5, with >= 10 fitted series.
- **PARTIAL** otherwise (stating which number disagrees), **INCONCLUSIVE** with < 10 fitted series.
- Secondary (does the growth law predict the sigma law even where k_region does not?): fraction of
  series with |p - (d_eff - 2)| <= 0.3, and for d_eff within 0.3 of 2 the fraction preferring the log law.
- Pattern control, additionally: bond-based k_region against the theory's c_J(r) for the pattern learned
  (exact agreement expected, since under the star condition the bonds gaining rank are exactly the edges
  of R_J(r)); this validates the pipeline.
- "Where it stops holding" is read off the breakdown by target, tree, k_region, n_components and sigma.

## 6. Outputs

`experiments/route_generic/results/`: `runs.csv` (one row per run), `events.csv` (one row per event and
run), `rises.csv` (one row per singular-value rise), `exponents.csv` (one row per series),
`summary.csv` (one row per target x tree x k_region class), MNIST rows in the same files with
`target = mnist`. Columns documented in `experiments/route_generic/README.md`. Draft figures from these
CSVs only.

## 7. Budget and cuts

~4 h wall clock. Synthetic sweep: 2640 runs (pilot: 8-150 s each) on ~64 workers; MNIST: 180 runs on
~56 workers. If too slow, cut from the expensive end: first the MNIST extra seeds, then the smallest
MNIST sigma, then synthetic sigma = 10^-3.5; anything cut is listed in Part III.
