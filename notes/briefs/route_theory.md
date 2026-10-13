# Brief: testing the route theory against training (four remote agents, 2026-10-13)

Lead: the local session on the user's machine. It coordinates, runs the cheap local parts, makes the
final figures and writes the results document. You are one of four remote agents working in parallel:
`route-smallinit`, `route-escape`, `route-nonstar`, `route-generic`. Read sections 1-3 (common), then
your own section in 4.

## 1. What is being tested

Paper: `report/ttn_saddle_order_and_aligned_training_dynamics.pdf` (31 pages; read the abstract,
Sections 1, 2, 7, 8, 10 and Appendix C; your section says what else). It studies gradient-flow training
of a tree tensor network (TTN) on a target that is a sum of a few one-hot patterns
W* = sum_a s_a U_a ("pattern teacher"). From small init the network learns the patterns one at a time,
with plateaus in between. Predictions:

1. With one pattern learned, the order p of the saddle (how hard the plateau is to leave) is
   min(k*, 2 l*) (Theorem 3.2). Already tested in this repo (`notes/saddle_dyn.md`,
   `notes/reviews/saddle_order_verification_*.md`); not your job unless your section says so.
2. A plateau of order p lasts ~ sigma^-(p-2) (log(1/sigma) for p = 2), sigma the init scale.
3. With a set J of patterns learned, a missing pattern r can reuse cores from any learned pattern. The
   cores it cannot reuse are the routing region R_J(r) (the intersection over a in J of the Steiner
   subtrees of the nodes where r and a differ); its size c_J(r) is the route depth. The order of the
   plateau is at most c*_J = min_r c_J(r), and exactly that on the aligned manifold (Theorem 8.4).
4. Greedy rule (Appendix C, not proven, needs Assumption C.1): the next pattern learned is the one with
   the smallest c_J(r); ties go to the larger s_r A_r(J).
5. For targets satisfying the edgewise star condition (Assumption 7.1), the "block-aligned manifold"
   M* is invariant and the dynamics on it reduce to scalar equations (Theorem 7.7), the TTN analogue of
   the Saxe equations: an isolated route of depth d obeys dy/dt = d gamma^(2/d) y^(2-2/d) (s - y).

The user's question: **how well do these predictions describe TTNs trained from small random init, and
where do they fail?** Proven statements do not need re-proving numerically; what matters is whether
real training behaves as predicted, and any place it does not. Report failures as prominently as
successes. The user is not an expert in this theory: write plainly, define every term.

## 2. Shared code (already written and tested; do not modify `ttn/`)

`ttn/routes.py` (usage examples: `tests/test_routes.py`, all passing):

- `RouteTarget(tree, patterns, weights)`: `.is_star`, `.star_violations`, `.blocks`, `.diff(a, b)`,
  `.steiner(nodes)`, `.pair_cost(a, b)`, `.region(J, r)`, `.depth(J, r)`, `.min_depth(J)`,
  `.partial_loss(J)`, `.greedy_next(J)`, `.greedy_sequence(first)`, `.one_component_order(j)`
  (Theorem 3.2: k*, mu_l, l*, order), `.teacher(d)` (a `CPTeacher` in the standard site basis).
- `AlignedManifold(target, d, chi)`: `.spec` (TTNSpec; bond e has dimension max(chi, blocks on e)),
  `.theta(g)` / `.g(theta)` / `.mask` (the allowed entries), `.amplitudes(g)`,
  `.partial_target(J, dormant=, relax_from=)` (theta_J; `relax_from=1e-2` gives the balanced
  representative that small-init flow approaches), `.flow(g0, t_max, stop=)` (the exact reduced flow
  on M*), `.align(cores, J)` (orthogonal bond rotations that put a generic network near theta_J into
  the block basis) and `.route_entry_norms(aligned, J, r)` (size of the dormant entries pattern r
  needs at every node of R_J(r)).
- `amplitudes(spec, cores, teacher)`: P_a = <U_a, W>, gauge invariant, valid for any network.
- `off_target_mass(spec, cores, teacher)`: fraction of ||W||^2 outside the span of the target patterns
  (exactly 0 on M*).
- `crossing_time`, `effective_depth(t, P, s)` (depth d from the slope 2 - 2/d of
  log[(dP/dt)/(s - P)] against log P for P/s in [1e-4, 1e-2]).

If you find a bug in `ttn/routes.py`, do not edit it: work around it in your folder and say so at the
top of your report.

## 3. Conventions for all four agents

- Repo conventions in `CLAUDE.md` (uv, float64, `torch.set_num_threads(1)`, config-driven, seeded).
  Additionally pin BLAS to one thread per worker process (`threadpoolctl.threadpool_limits(1)` or
  `OPENBLAS_NUM_THREADS=1`): an earlier sweep was oversubscribed by inherited BLAS pools.
- Model: whitened inputs, exact population loss L = 1/2 ||W - W*||^2, pattern targets in the standard
  site basis, fan-in Gaussian init `spec.init(sigma, seed)` on **all** parameters, bond dimensions with
  spare channels (blocks + 2). The same seed gives the same Gaussians at every sigma.
- Integration: gradient flow, LSODA with the exact Hessian as Jacobian, rtol 1e-9 to 1e-10, atol 1e-14
  to 1e-15. **Read `notes/saddle_dyn.md` section 8 (deviations 1-7) before writing a driver**: events
  must be detected on solver steps (scipy's LSODA dense output is wrong by percents), the clock must be
  re-zeroed at late times, a fresh LSODA started on a plateau can stall in nonstiff mode, dead routes
  need a trap rule. Reuse `experiments/saddle_dyn/sdyn.py` (step-driven driver, cancellation-free
  residual about a reference point) or `experiments/locality/run.py` (library model, stall detection).
- float64 floor: the plain loss is safe for routes of depth <= 4 down to sigma ~ 1e-5; for deeper
  routes or starts at a plateau use the cancellation-free residual about the reference point. Flag
  every run whose escape signal is within 100x of roundoff, and exclude flagged runs from fits.
- Definitions (use exactly these, so the four agents' tables can be combined):
  - amplitude P_a(t) = <U_a, W(t)>; learned set J(t) = {a : P_a > s_a / 2};
  - learning event of a: the last upward crossing of P_a = s_a / 2 (record every crossing; a pattern
    whose amplitude later falls below s_a / 4 is "forgotten" at that time);
  - plateau k: between learning events k and k + 1; duration = difference of the event times;
  - escape time from a plateau start: first time L < L_J - 1e-3;
  - effective depth of an event: `effective_depth` on the amplitude of the pattern being learned,
    over the plateau before its event;
  - off-target mass: `off_target_mass`; its peak over a transition = max between the midpoints of the
    plateaus before and after.
- Score each transition **conditional on the learned set J actually reached** (the theory does not
  predict the first pattern from the origin, and which pattern is first is random at close weights).
- Pre-register before any sweep: write the cells, grids, fit windows (>= 1.5 decades at the small end),
  estimators and decision thresholds into your note and commit it; afterwards only add a list of
  deviations with reasons. Per-seed exponents by least squares of log(duration) on log(sigma) inside
  the window; report the distribution over seeds (median, quartiles, fraction within 0.3 of the
  prediction), not only the mean. Count and report every run that is trapped, stalled, unfinished or
  floor-flagged; never drop runs silently.
- Outputs: experiment folder `experiments/<your name with _ for ->/` (README in the repo's format,
  `configs/`, code, `results/` with tidy long-format CSVs and figures); raw outputs in
  `/root/outputs/<your name>/` (not in git); note `notes/<your name with _>.md` (pre-registration, then
  results with status tags proven / derived / empirical / conjecture); report
  `agent-runs/<your name>/REPORT.md`, ending with a line `Status: done` when you are finished.
  Commit and push early and often (the lead watches your branch); push the first results table as
  soon as it exists rather than at the end.
- CSVs are the main deliverable: the lead makes the final figures from them. Every CSV needs its
  columns documented in the README (meaning and units), one row per observation (run, event, plateau
  or fit), with the identifying columns `target, tree, d, chi, sigma (or eps), seed`.
- Draft figures are still required, and they must be easy to read for a non-expert: one message per
  figure stated in the title, axes labelled in words with units, the prediction drawn as a labelled
  reference line or marker, at most six colours, large fonts, no clutter. Use `ttn/plotting.py`.
- Aim to finish within about 4 hours of wall-clock time. If something is too slow, cut the grid from
  the expensive end and say what was cut.

## 4. Tasks

### 4.1 `route-smallinit`: training from small random init on targets that satisfy the star condition

This is the main experiment. Question: when a TTN is trained by gradient flow from a small random init
on a star-satisfying target, (a) are its plateaus the canonical partial-target points
W = sum over J of s_a U_a, (b) is the next pattern the one the greedy rule predicts, (c) do plateau
durations scale as sigma^-(c - 2) with c the route depth, (d) does each event show the predicted
effective depth, (e) does training stay close to the aligned manifold, and (f) do the four items of
Assumption C.1 hold?

Targets: about 10, with K <= 7 cores and every later route depth <= 4 (deeper routes do not leave a
usable sigma window). Choose them with `ttn.routes` so that the greedy rule, the pairwise rule
(next = smallest min over a in J of `pair_cost(a, r)`, the Prim rule of eq. C.8) and plain weight order
predict different sequences wherever possible. Cover: (i) decoupled patterns that share nothing
(e.g. mps5, d = 3: 00000, 11111, 22222 with weights 1, 0.7, 0.45; Corollary 7.8 predicts every
plateau ~ sigma^-(K-2)); (ii) shared entries where a smaller-weight pattern has the shorter route
(B4: mps4 0000 (1), 0110 (0.3), 1001 (0.6)); (iii) splicing on the greedy path (route depth below
every pairwise cost), e.g. mps5 d = 3 {00000, 00201, 10101, 11200} (last step depth 2, pairwise 3) and
{00000, 02201, 20111, 20200} (depth 3, pairwise 4), mps7 d = 2 {0000000, 0001011, 0100011, 0110100}
(depth 2, pairwise 3), ht4 d = 3 {0000, 0122, 1100, 2022} and binary8 d = 2
{00000000, 00001111, 01011100, 10000011} (here the splice also changes which pattern is predicted
next); choose weights so that the first pattern is the one that puts the splice on the path;
(iv) a tie in depth broken by weight; (v) one target with a predicted spectral shortcut at its first
plateau (mps6 pair-star 000000 (1), 110001 (0.8), 110010 (0.8); see `notes/saddle_dyn.md` 9.6).
Trees: mps, binary and ht, not only mps.

Runs: sigma on a quarter-decade grid over at least two decades at the small end allowed by the float64
floor and by cost; 40 seeds per (target, sigma). Also, per target and sigma, the reference run on the
manifold (`AlignedManifold.flow` from uniform entries sigma): its sequence and event times.

Per run, record (CSV names are suggestions; keep the content):
- `runs.csv`: status, the order in which patterns were learned, forgetting events, final loss, flags.
- `events.csv`, one row per learning event: pattern, event time, J before the event, duration of the
  plateau before it, route depth c_J(r), the pattern each rule predicts from that J (greedy, pairwise,
  weight), effective depth, peak off-target mass over the transition, and for the plateau before the
  event: its loss against L_J and the relative distance ||W - sum over J of s_a U_a|| / ||W*|| at the
  minimum-gradient point (is the plateau a canonical partial-target point?).
- `plateaus.csv`, one row per (plateau, missing pattern): route depth, signed amplitude P_r at the
  minimum-gradient point, the route-entry norms from `align` + `route_entry_norms` (minimum, maximum
  and geometric mean over the nodes, each divided by sigma), and the alignment quality.

Analysis:
- sequence agreement per transition against the three rules, as a function of sigma;
- per-seed exponents of plateau duration against sigma, grouped by (target, J before, pattern), against
  c - 2 and against pairwise - 2;
- effective depth against route depth;
- Assumption C.1: (1) are the route-entry norms of order sigma at plateau entry (slope of log norm
  against log sigma; spread across nodes), (2) are the active factors independent of sigma, (3) and (4)
  are consecutive events separated (fraction of events where another pattern passes 10% of its weight
  before the current one passes 90%);
- one-step-ahead test of the scalar route equation: from the amplitude P_r at plateau entry, the depth
  c and the prefactor gamma = A_r(J) (the product of the active entries of r at the balanced theta_J,
  from `partial_target(J, relax_from=...)`), the balanced route has entries b = (P_r / gamma)^(1/c) and
  escapes after 1 / ((c - 2) gamma s_r b^(c-2)) (log(b*/b) / (gamma s_r) for c = 2). Compare predicted
  and observed durations and predicted and observed winners; negative P_r at plateau entry is a
  result to report (the route must change sign first), not an error;
- runs that do something else (two patterns at once, forgetting, traps): count them and say what
  they did.

### 4.2 `route-escape`: leaving a known plateau

Question: start the network exactly at a plateau theta_J with several patterns learned, add a small
perturbation of size eps, and measure (a) the order of the saddle, (b) the escape time against eps, and
(c) which pattern is learned first. Predictions: Theorem 8.4 (order exactly c*_J inside M*, at most
c*_J in the full parameter space), eqs. 6.15-6.16 and C.6 (escape time ~ eps^-(c-2) with prefactor
1 / ((c - 2) s_r A_r(J)); log law for c = 2), the greedy rule, Corollary 8.6.

Also read Sections 5, 6 and 9 of the paper and `notes/saddle_dyn.md` (the same protocol at
one-pattern saddles: 5 of 7 test cells supported, but the control cells with a pure route law were
inconclusive because many random directions were trapped or had the exponent shifted by +1; your
experiment revisits that question at multi-pattern saddles).

1. Cells: about 10 (target, J) pairs chosen with `ttn.routes`, with spare bond channels: Example 8.7
   (mps7, d = 3, J = {0000000, 1111111}, missing 0022111: a depth-2 splice although budding from one
   parent would cost 4 or 5; predicted log law); unique shortest routes of depth 3, 4 and 5; cells
   with two missing patterns of different depth where the shorter route has the smaller weight; a tie
   in depth with different weights; one missing pattern seen from J and from a larger J'
   (Corollary 8.6); branched trees (binary, ht) as well as mps. Use the balanced theta_J
   (`partial_target(J, relax_from=1e-2)`).
2. Static order at every cell: m(t) = min of L - L_J over the sphere ||theta - theta_J|| = t (method
   and code: `experiments/theory_checks/saddle_order/saddle_order.py`; generalise it to the reference
   theta_J with the cancellation-free excess 1/2 ||D||^2 - <R, D>, D = W(theta) - W(theta_J)), once
   restricted to M* (`AlignedManifold.mask`) and once in the full space. The slope of log|m| against
   log t is the order. List every cell whose full-space order is below c*_J (a "shortcut": Remark 8.5)
   and describe its descent direction (off-target mass, which entries). Also the smallest Hessian
   eigenvalue at theta_J.
3. Dynamics from theta_J + eps xi with three kinds of perturbation: (A) inside M* with positive dormant
   entries (the setting of the theorems), (B) inside M* with random signs, (C) Gaussian on all
   parameters (the realistic case). 16 directions per cell and kind, eps on a quarter-decade grid over
   at least two decades. Record the escape time, the first pattern learned, the off-target mass at
   escape, the effective depth, and the outcome (escaped / trapped / unfinished / floor-flagged). For
   (A) and (B) also integrate the reduced flow (`AlignedManifold.flow`) from the same start: the full
   network must reproduce it (Theorem 7.7), and its escape time must match eq. 6.16 including the
   prefactor. For (C) report the distribution of per-direction exponents: the fraction at c - 2, the
   fraction shifted, the fraction trapped. If shifted or trapped directions appear, find out why
   (follow the route entries with `route_entry_norms`, look at the sign of P_r at the start).
4. Theorem 9.2 (static only). Its Assumption 9.1 is not constructive. Test the formula
   order = min over parents a of min(k*_a, 2 l*_a) on the natural family: two learned patterns that
   differ at every node (they share no entry), and missing patterns that each differ from exactly one
   parent on a small region. Compute the prediction per parent with
   `RouteTarget.one_component_order(a, residuals=[...])` and measure the full-space order by the
   sphere minimum. Include a case where both parents are "direct" and a case where one parent has a
   spectral shortcut (a pair-star family, order 4 with k* = 5). State plainly whether the formula
   holds on this family.

CSVs: `cells.csv`, `static_orders.csv`, `escape_runs.csv` (one row per run), `escape_fits.csv` (one
row per direction), `thm92.csv`.

### 4.3 `route-nonstar`: targets that break the star condition

Question: what does training from small random init do on pattern targets that violate the star
condition, and which predictions survive? The paper proves nothing here; Appendix B conjectures
order = min(direct degree, 2 x spectral level).

Read first: `notes/locality.md` sections 5 and 9 and `experiments/locality/README.md`. That experiment
already trained the nested teachers TA, TB, N4, TC and TD (all violate the star condition) and found
that the flow stops at "rank-constrained optima" and follows a numerical "route rule"
(`experiments/locality/rules.py`: `route_sequence`, `route_rule`). Do not redo it. What is missing
there, and is your job:

1. For each plateau: is it a canonical partial-target point (W = sum over J of s_a U_a) or something
   else? Measure the relative distance to the nearest partial target, the off-target mass and which
   off-target strings carry it (a "recombinant" such as 101 = left of 100 + right of 001).
2. The sequence of learning events, plateau durations against sigma (per-seed exponents), effective
   depths, against three predictors: the paper's route depth applied anyway (the intersection can be
   empty: depth 0 means the pattern is produced for free as a recombinant), the repo's numerical route
   rule, and the pairwise rule.
3. At each plateau, whether the escape is strict (negative Hessian eigenvalue, log law) or higher
   order, and whether the exponent is consistent with min(direct, 2 x spectral) where you can work
   those out by hand for the small cases.

Targets, K <= 7 cores: the smallest recombinant family (path of 3 or 4 nodes: q = 000, plus 100 and
001, with and without 101 in the target, with 101 at the product weight and at a different weight);
nested chains (0000, 1100, 1110, 1111 on mps4 and on binary4 / ht4); TD-like targets with two separate
local edits; and about six random d = 2 targets on mps6 and binary8 that fail the star condition
(report `RouteTarget.star_violations` for each). For contrast run two star-satisfying controls of the
same size through the same code. Sigma grid and seeds as in 4.1 (quarter decades over at least two
decades, 40 seeds); CSVs as in 4.1 (`runs.csv`, `events.csv`, `plateaus.csv`) plus a column that names
the nearest partial target and the off-target strings.

State the answer plainly: for targets outside the star condition, which of the paper's predictions
still describe training, which fail, and what replaces them.

### 4.4 `route-generic`: targets outside the theory (random teachers, MNIST)

Question: every result of the paper assumes a pattern target. Does the qualitative picture survive for
targets it does not cover: (a) random CP teachers with non-orthogonal components, (b) random TTN
teachers, (c) MNIST? The picture to test: learning happens in steps; each step has an effective depth
equal to the number of cores that must change for it; and the plateau before it lasts
~ sigma^-(depth - 2).

Read first: `experiments/data_teacher/README.md`, `notes/data_teacher.md`,
`notes/reviews/` for data_teacher (the "safe to cite" list), `experiments/real_data/README.md`,
`notes/real_data.md`. Much is already measured there; cite it instead of redoing it. The raw outputs
of those experiments are not on your machine, so rerun a modest set with their configs and runners.

For each learning event (a bond singular value rising from the noise floor), measure:
1. which bonds gain rank at the event, and k_region = the number of cores in the smallest subtree
   containing those bonds' end nodes;
2. the effective depth from the growth law of the emerging singular value or overlap
   (`ttn.routes.effective_depth`; same estimator as `experiments/locality/growth.py`);
3. the exponent of the plateau duration against sigma (per-seed fits, at least 1.5 decades).

Compare the three numbers event by event, for: a pattern-target control (inside the theory), random CP
teachers with overlaps of 0.3 and 0.6 between components, random TTN teachers, and MNIST as far as the
existing real-data pipeline allows (at least the first few events; say clearly what could not be
measured). Trees: mps6 and binary8; 20 or more seeds; sigma over two decades.

CSV: `events.csv` with one row per event (teacher family, tree, sigma, seed, event index, bonds that
change, k_region, effective depth, duration) and `exponents.csv` (per seed and event). State the
answer plainly: outside pattern targets, does "depth = number of cores that change" still hold, and
where does it stop holding?
