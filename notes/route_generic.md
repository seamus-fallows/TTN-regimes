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

# Part II: results

Sweeps: 2640 synthetic runs (19:41-20:34 UTC), all converged (L < 1e-7 L0), no solver failure, no stall,
no trapped run, max 411 s per run. MNIST: see section 12. Tables: `experiments/route_generic/results/`
(`summary.md` has the tables below, generated by `tables.py` from `summary.csv`, `runs.csv`, `events.csv`,
`exponents.csv`). Words used below: a **route** is the set of cores whose entries must change together for
a new component to appear; **k_region** counts the cores touched by the bonds that gain rank;
**d_eff** is the depth read off the growth law of the new singular value; **p** is the exponent of the
plateau duration, duration ~ sigma^-p (p ~ 0.1-0.2 is what a log(1/sigma) law looks like over these
sigma, so p within 0.3 of 0 counts as the log law).

## 8. Checks of the pipeline

- **Event 0** (escape from the origin, a proven identity): d_eff = 7.00 on binary8 (K = 7) and 5.99-6.00
  on mps6 (K = 6), medians over 21-30 series per target; p = 5.000 and 4.000 = K - 2 exactly.
  *Check of the estimators, not evidence.*
- **Pattern control, bonds** (*empirical, confirming the derived Prop. 7.2 / 8.2*): in all 1889 learning
  events of the 660 pattern runs (every sigma), the bonds that gain rank are exactly the edges of the
  routing region R_J(r) of the pattern learned, and k_region = c_J(r); the learned pattern is the one the
  greedy rule predicts in every event (the three rules agree on these two targets, so this is not a test
  of the greedy rule).

## 9. Results per target (later events: everything after the escape from the origin)

Medians [quartiles] over fitted series (one series = one seed and one event followed across sigma, 7
smallest usable sigma, >= 1.5 decades). Verdicts by the registered rules (section 5).

| target | tree | series | k_region | d_eff | p | d_eff within 0.3 of k_region | p within 0.3 of k_region - 2 | p within 0.3 of d_eff - 2 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pattern (control) | binary8 | 90 | 3 [3, 4] | 3.10 [3.04, 4.18] | 1.01 [1.00, 2.00] | 96% | 100% | 96% | HOLDS |
| pattern (control) | mps6 | 81 | 3 [2, 4] | 3.30 [1.73, 4.09] | 1.01 [0.18, 2.00] | 62% | 100% | 60% | PARTIAL |
| CP, overlap 0.3 | binary8 | 55 | 7 [7, 7] | 3.09 [2.34, 4.08] | 1.00 [0.93, 1.08] | 12% | 9% | 29% | FAILS |
| CP, overlap 0.3 | mps6 | 60 | 6 [6, 6] | 1.53 [1.28, 1.67] | 0.12 [0.10, 0.14] | 0% | 0% | 2% | FAILS |
| CP, overlap 0.6 | binary8 | 77 | 7 [3, 7] | 3.06 [3.04, 3.09] | 1.00 [1.00, 1.01] | 43% | 42% | 97% | FAILS |
| CP, overlap 0.6 | mps6 | 45 | 6 [6, 6] | 1.97 [1.90, 2.00] | 0.13 [0.12, 0.15] | 0% | 0% | 89% | FAILS |
| random TTN | binary8 | 71 | 3 [3, 3] | 3.04 [3.02, 3.07] | 1.00 [1.00, 1.00] | 83% | 83% | 100% | HOLDS |
| random TTN | mps6 | 64 | 2 [2, 3] | 2.03 [1.83, 2.09] | 0.15 [0.11, 0.18] | 38% | 55% | 72% | FAILS |

Breakdown by k_region (all classes in `summary.md`): pattern binary8 k = 3: d_eff 3.04, p 1.00 (60
series), k = 4 (the splice): d_eff 4.18, p 2.00 (30); pattern mps6 k = 2: d_eff 1.51, p 0.14 (21),
k = 3: 3.24, 1.00 (30), k = 4: 4.09, 2.00 (30); CP 0.6 binary8 k = 3: 3.06, 1.00 (34, HOLDS), k = 7:
3.05, 1.00 (43); CP 0.3 binary8 k = 7: 3.02 [2.29, 4.16], 0.99 (49); random TTN binary8 k = 3: 3.02,
1.00 (59, 100 % within 0.3 for both), k = 5 and 7: 3.10 and 3.19, p 1.00 and 1.02 (5 and 7 series);
random TTN mps6 k = 2: 2.00, 0.12 (35), k = 3: 2.04, 0.18 (14), k = 4-6: 2.03-2.12, 0.14-0.16 (15).

### 9.1 Pattern control (inside the theory) — *empirical, confirming the derived picture*

On binary8 the picture holds event by event: the depth-3 sibling routes give d_eff 3.04 and p 1.00, the
depth-4 splice (route depth 4 although every pairwise cost is >= 5) gives d_eff 4.18 and p 2.00. On mps6
the duration exponents agree in 100 % of the series (p = 0.14, 1.00, 2.00 for c = 2, 3, 4; the depth-2
event follows the log law in 21/21 series) and the depth-4 event gives d_eff 4.09. Two deviations of the
growth-law depth, both inside the theory's own assumptions:
- the depth-2 event (pattern 001100 right after 000000) gives d_eff 1.51 [1.50, 1.54]: its plateau lasts
  only ~2-3 time units at sigma = 10^-3.5, so the new amplitude grows while the first pattern is still
  being learned and its active factors are not yet constant (Assumption C.1 (2) fails there); the
  amplitude-based estimate is 1.46. The exponent is unaffected (log law).
- the depth-3 event on mps6 gives d_eff 3.24 (63 % within 0.3) on the bond singular values but 3.03 on the
  amplitude P_r (the brief's definition); the bond value mixes in the dormant depth-4 pattern, which uses
  the same bonds.
So the bond-based pipeline reads the theory's numbers off correctly when the target is a pattern target.

### 9.2 Random TTN teachers — *empirical*

- **binary8: holds.** Every later event is a sibling pair of bonds through their common parent (bottom
  pairs (0,1), (3,4), then the root pair (2,5); 98.6 % of the events with k_region >= 3 start with such a
  3-core pair): d_eff 3.02, p 1.00 (100 % within 0.3). This extends to all later events the
  data_teacher result (sigma^-1 for the first rank increase on binary students; P11, bottom-up growth).
  When two sibling pairs rise in overlapping windows they form one event with k_region 5 or 7 (12 series):
  d_eff and p stay at 3 and 1, so k_region over-counts there.
- **mps6: fails as soon as more than one bond rises.** Single-bond events (k_region 2) follow the picture
  (d_eff 2.00, p 0.12, log law preferred in 74 %). But 38 % of the fitted series are events in which 2-5
  bonds rise in overlapping windows (k_region 3-6); they too grow with d_eff ~ 2.0 and have log-law
  plateaus (p 0.14-0.18). In 76 % of these events a single bond starts first and the others follow later
  on the same plateau (median onset spread 0.29 of the plateau; 49 % of them are not even connected): the
  plateau is set by the first 2-core move, not by the size of the region.

### 9.3 Random CP teachers with overlapping components — *empirical*

- **mps6 (both overlaps): fails.** Every later event raises a new singular value on all five bonds
  (k_region = 6), and the bonds start nearly together (median onset spread 0.03 of the plateau for
  c = 0.3, 0.29 for c = 0.6). Yet the plateau grows only like log(1/sigma): p = 0.12 (c = 0.3) and 0.13
  (c = 0.6), log law preferred in 87 % and 100 % of the series, where the picture predicts sigma^-4. For
  c = 0.6 the new singular values grow exponentially (d_eff 1.97, a strict saddle), consistent with the
  data_teacher result that the rank-one saddle of a generic teacher on an MPS student is strict, with
  negative directions on every edge. So all six cores change, but the event is a first-order (depth-2)
  instability.
- **binary8: fails on k_region, not on the law.** The first event after the escape raises the second
  (sometimes also the third) singular value on all six bonds (k_region 7), but d_eff 3.05 (c = 0.6) and
  p 1.00: a depth-3 event. The later events of c = 0.6 that raise only the root pair (2,5) hold (k_region
  3, d_eff 3.06, p 1.00). 57 % of the c = 0.6 all-bond events start with a single 3-core pair (the others
  start on all bonds at once); for c = 0.3 all bonds start together (0 % start with a pair).
- **c = 0.3: the growth law is not a power law.** The new singular values grow exponentially (local
  depth 2.0) at amplitudes 1e-6-1e-4 of their target, slow down around 1e-3, then take off abruptly and
  together on all bonds (local depth > 8 above 1e-2); the registered window [1e-4, 1e-2] straddles the
  phases and returns d_eff 1.53 on mps6 and a wide 3.02 [2.29, 4.16] on binary8. Here d_eff does not
  predict p (2 % and 21 % within 0.3), although p itself is still log-like on mps6 (0.12) and ~1 on
  binary8 (0.99). About a quarter of the c = 0.3 / 0.6 binary8 runs (90 and 86 of 330) have a singular
  value that crosses half its target, falls back and rises again (relearned).

### 9.4 What replaces "depth = cores that change" for these targets — *empirical; mechanism derived for the first rank increase in theory_teachers R9 / theory_saddles R13, conjecture for later events*

Across all generic synthetic targets the plateau exponent is that of the **cheapest local route that is
open at the current state**: depth 2 on the chain (every adjacent core pair has free modes, so the saddle
is strict and the plateau lasts ~ log(1/sigma)), depth 3 on the binary tree (an internal core at rank one
has no free mode, so the cheapest move is a sibling pair through its parent; plateau ~ sigma^-1). No
generic event here needed more: none of the 372 fitted later-event series of the CP and random-TTN
targets has p above 1.5 (95th percentiles 0.19 on mps6, 1.13 on binary8), however many cores the event
changed in the end. The growth-law depth tracks the duration exponent (p within 0.3 of
d_eff - 2) in 72-100 % of the series for CP 0.6 and random TTN teachers, but not for CP 0.3, whose growth
curves are multi-phase. The picture "number of cores that change" holds only when an event is a single
route whose bonds start together and no cheaper sub-route exists: pattern targets, and sibling-pair events
on the binary tree.

## 11. Exploratory: the Hessian at the plateau before an event (added after the registration)

`plateau_hessian.py`: from the stored state at the first escape the flow is re-integrated to the middle of
the plateau before each event (seeds 0-1 of every teacher, sigma = 1e-3 and 1e-2), and the exact Hessian is
diagonalized. A strict saddle has a sigma-independent negative eigenvalue; at a saddle of order c the
negative curvature seen at a distance ~sigma from it scales like sigma^(c-2). Slope of log|lambda_min|
against log sigma between 1e-3 and 1e-2 (median over events; `results/plateau_hessians_scaling.csv`):

| target | tree | k_region | events | slope of abs(lambda_min) vs sigma | lambda_min at sigma = 1e-3 | cores carrying the most negative direction |
|---|---|---|---|---|---|---|
| pattern | mps6 | 3 | 2 | 0.97 | -7.6e-4 | 3 |
| pattern | mps6 | 4 | 2 | 1.97 | -4.5e-7 | 4 |
| random TTN | mps6 | 2-6 | 23 | -0.03 to 0.01 | -0.10 to -0.22 | 2 (54 of 58 events) |
| CP 0.6 | mps6 | 6 | 17 | -0.01 | -0.10 | 2 (37 of 37) |
| CP 0.3 | mps6 | 6 | 20 | 0.02 | -0.009 | 2-6 |
| random TTN | binary8 | 3-7 | 23 | 0.99-1.00 | -2.5e-4 to -4.9e-4 | 3 (46 of 46) |
| CP 0.6 | binary8 | 3 | 12 | 1.00 | -2.9e-4 | 3 |
| CP 0.6 | binary8 | 7 | 15 | 0.00 (teachers 4, 5) / 0.98 (teachers 1-3) | -0.04 | 3 |

*Empirical (exploratory).* On the pattern control the sigma-scaling of the plateau curvature gives the
route order exactly (slopes 0.97, 1.97 for c = 3, 4). Before every generic event on the chain the plateau
is a strict saddle (sigma-independent lambda_min) whose most negative direction sits on two adjacent
cores, even when the event ends up changing all six cores: the event is started by a 2-core instability.
On the binary tree the plateaus are order-3 saddles (lambda_min ∝ sigma) whose most negative direction
sits on 3 cores (a sibling pair and its parent). One exception not followed up: for 2 of the 5 CP 0.6
teachers on binary8 the plateau before the all-bond event also has a sigma-independent negative
eigenvalue (-0.11 to -0.38, on a sibling triple), yet that event grows like depth 3 and lasts ~sigma^-1
(p = 1.00, d_eff 3.02 for these teachers), so this negative direction is not the one that starts the
event; these are also the cells with relearned singular values.

## 10. Floor flags, cuts, counts

- Floor-flagged events (a rise below 1e-14 s_1 on its plateau; excluded from fits): only on binary8 and
  only at sigma <= 1.8e-3: CP 0.3 110 events (110 runs), CP 0.6 68, pattern 45 (the depth-4 splice at
  sigma = 10^-3.5, 10^-3.25), random TTN 37 (merged events). mps6: none. The fit window then slides to the
  next 7 usable sigma (rule of section 4).
- Series not fitted (fewer than 5 points or < 1.5 decades): mostly events whose signature changes with
  sigma (two events merge at larger sigma), e.g. 121 of 210 random-TTN mps6 series; counts per target in
  `summary.md`.
- Nothing was cut from the synthetic grid.

# Part III: deviations from the pre-registration

1. **Jacobian implementation (numerics only, 19:44, before any result was looked at).** The registration
   says "exact Hessian as Jacobian"; the first implementation differentiated the dense tensor twice
   (2.5 s per call on binary8, d = 3), which made the binary8 runs ~5x slower than the budget allowed.
   It was replaced by the library's contraction-based exact Hessian (`ttn.dynamics.Problem.hessian`,
   0.12 s; equal to the dense one to 1e-21 on a test point). The Jacobian only enters LSODA's Newton
   matrix, not the vector field, so trajectories agree to the integration tolerance. The 30 runs
   finished with the first version (pattern, binary8, sigma = 10^-3.5) were kept; the sweep was
   restarted for the rest.
2. **Clock of the event-0 growth-law fit (numerics only).** The registration fits the escape from the
   origin on the absolute clock. At sigma = 10^-3.5 the escape happens at t ~ 1e17-1e18, where the
   float64 spacing of t is 16-128 time units, longer than the growth window (~100-3000 units), so the
   fit returned nothing on binary8. Event 0 (and the amplitude version) now uses the relative clock,
   whose differences inside the escape's solver segment are exact (`events.relative_clock`). With it
   event 0 gives d_eff = 7.00 on binary8 (K = 7) and 6.00 on mps6 (K = 6), the exact identity.
3. **Exploratory columns added after the registration** (not used by the registered decisions):
   `first_bond`, `onset_spread_rel` (spread of the onsets of an event's rises over the plateau duration)
   and `k_lead` (cores of the rises whose onset lies within 10 % of the plateau duration of the earliest
   onset: the bonds that start the event), with `dd_lead`, `dp_lead` in exponents.csv. Added at 20:10
   after seeing, in a partial table of the smallest sigma, that events of generic targets often contain
   bonds that start growing well after the first ones.
4. **Exploratory plateau Hessians** (section 11, `plateau_hessian.py`), added after the synthetic results
   were known, to see what starts a multi-core event. Not part of the registered decisions.
