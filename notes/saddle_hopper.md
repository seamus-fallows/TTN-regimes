# Saddle hopper: the S2S sequence predicted from the theory, compared with the committed runs

Status: **v2, 2026-10-10.** v1 (agent `saddle-hopper`) audited in
`notes/reviews/saddle_hopper_verification.md`; v2 applies that audit (errata E1-E11), re-runs the order
comparison on the order fix round, and adds a test of `notes/theory_saddle_order.md` on the scored
teachers (section 9). Code: `experiments/saddle_hopper/` (README there).

Tags: **proven**, **derived**, **empirical**, **conjecture**, and **bookkeeping** (an operational
combination of results already in the theory notes and experiments, with nothing new in it).

**What to cite.** As a whole-sequence validation the hopper is **bookkeeping of known theory**: its
predictions re-derive the experiments' own predictions, and where it commits to a single order that
order is the trivial "weight order, no rotation" in all but 107 runs (section 4). Safe to cite:
- the implementation itself (deterministic, byte-identical re-run, reproduces the theory notes' worked
  examples; section 3);
- the multi-saddle accumulated-utility slopes of the TC teacher on mps8 from the saddle (5 plateaus
  within 3.5%; section 5), the one new numerical check of the v1 comparisons;
- the test of the saddle-order theorem on the scored teachers (section 9): new, with its own status tags.

Not safe to cite: "99.2%" or "all 1,151 follow a predicted sequence" as predictive power; "174/174" as
independent law tests; "no mismatch needs theory wrong"; the 13 s2s_core 1-3-2 runs as support for rule
R4c; hopper exponents for wrong-order or multi-residual saddles with k* >= 5 (section 9 gives the
correct order there).

## 0. What this is

`notes/theory_saddles.md` section 11 asks for "a numerical saddle hopper": at a plateau state compute
the fixed-rank (old) block and the local residuals R_C of the connected core sets C, follow the most
unstable direction, go to the next plateau, repeat. `experiments/saddle_hopper/hopper.py` does this for
any tree with N <= 8 sites, any dense teacher (pattern, odeco, CP, TTN) and any product input
covariance, and adds the first escape (theory_escape), the accumulated-utility race of the k = 2
routes (theory_teachers 3.4), the time laws and the failure flags. Section 1 lists the rules, section 2
the scoring; sections 3-6 compare with the committed runs; section 9 tests the newer saddle-order
theorem against the same data.

Whether the rules were fixed before the comparisons cannot be shown either way (the hopper landed as one
commit; no pre-registration; audit section 2). The predictions do not use sigma or seed, and no rule or
threshold is tuned per case. Data-dependent elements: rule R4c (made explicit after reading the
s2s_core notes), the saddle start from the lowest rank-one attractor (data_teacher's protocol), and the
mismatch-cause checks of section 6 (written after seeing the mismatches).

## 1. Rules

Notation as in theory_saddles.md section 1: K cores, chi_e bond dimensions (library caps), profile
r = (r_e) bond ranks of a saddle, R = Sigma (W* - W_S) the (metric-weighted) residual, C a connected
core set, R_C the local residual (R contracted with the learned hanging subtrees T_f of every boundary
edge f of C; legs grouped per core of C: its sites and its boundary bond legs).

**R0 Inputs.** Tree, d, chi; dense teacher W*; optional product covariance Sigma = (x)_i Sigma_i;
init protocol `fanin`, `balanced` (fan-in made exactly balanced) or `saddle` (exact first saddle plus
noise); optional labelled components U_r (only used to name events: overlap
a_r = <W, Sigma U_r>/<W*, Sigma U_r>, learned iff a_r > 1/2).

**R1 Saddles** (theory_saddles 5.1 items 3-4; theory_teachers R1, R3). A plateau is a balanced
(Q_e = 0) critical point theta_S of the loss restricted to a bond-rank profile, zero-padded to chi.
The hopper obtains it by descending the loss at fixed profile (L-BFGS) from the point where the
previous escape left, then balancing. Plateau level = loss at theta_S. If the optimum leaves a bond
direction unused, the profile is lowered to the effective ranks.

**R2 First escape** (theory_escape E1, E3, E4, E8, E9 and 5.3; theory_saddles 4.5; theory_covariance
C4, C7). The escape tensor is T = Sigma W*. Candidates: fixed points of the grouped rank-one problem
max <T, (x)_v y_v> (one unit vector per site-carrying core; alternating power iteration from 32 random
starts and from every labelled component). Each candidate is classified by the Riemannian Hessian of
g(theta) = <W(theta), T> at its balanced bond-rank-one point on the unit sphere, in the student's full
parameterization (so adjacent pairs contribute chi_e directions, theory_escape 3.3):
- no positive eigenvalue: an **attractor**. One attractor: the first escape is deterministic as
  sigma -> 0. Several: the order is **random** with sigma-independent probabilities (tensor-like
  race; the hopper does not compute the probabilities, which need the init).
- n_u positive eigenvalues h_u: a **rare** branch, reached first with probability ~ C sigma^gamma,
  gamma = sum_u h_u / (K g) (heuristic exponent n (1 - rho)/rho; prefactor unknown).
- First plateau: duration ~ sigma^-(K-2) (exponent exact, prefactor random).
- The first saddle is the rank-one critical point of the true loss in the candidate's basin.

**R3 Saddle type** (theory_saddles R3, R6, Prop. 1, 4.3, 5.1 item 2). At theta_S: (i) the
fixed-rank block = the Hessian of the loss on the profile-r TTN, its lowest eigenvalue lambda_old
(gauge zero modes ignored by a tolerance 1e-7 x lambda_max); (ii) every connected C (with room: chi_e >
r_e on its internal edges) and its coupling mu_C = injective norm of R_C (sigma_max for |C| = 2, which
is exactly the k = 2 Hessian rate gamma_C). k* = min |C| with mu_C > 1e-6 ||R||. Saddle type: **strict**
if lambda_old < 0 or k* = 2; **higher-order of order k*** if the old block is PSD and k* >= 3;
**stall** if R != 0 but every route is blocked by chi.

**Caveat on R3 and R5 (order = k*): false in general.** `notes/theory_saddle_order.md` (Theorems 1,
2; the level version Theorem 1' is in its verification, `notes/reviews/saddle_order_verification_proof.md`,
which calls all three proven) shows that at a saddle with one learned component the order is k* only
when the generalized piggyback operator S (couplings between core sets of degree < k*/2) has
lambda_max(S) < s_q; if lambda_max(S|L_l) > s_q at a level l < k*/2 the order is 2 l* < k*. The
hopper's old block catches l = 1 (the classical piggyback, a strict saddle) but not l >= 2. Section 9
computes S at every rank-one saddle of the scored teachers: the order drops at 30 of them (all
wrong-first saddles on ht4, ht8, binary8, order 6 instead of 7), and the runs through them and through
"tie" saddles explain 19 of the 88 locality mismatches. Until the hopper implements the operator,
its exponents on wrong-order or multi-residual saddles with k* >= 5 should not be cited.

**R4 Next event.**
- (a) lambda_old < 0 and |lambda_old| >= the largest k = 2 rate: a **rotation** of the learned
  structure (log time); the next saddle is the fixed-rank critical point reached from theta_S +- the
  unstable eigenvector (both signs; two branches if they differ).
- (b) else, if k = 2 routes exist: the **accumulated-utility (AGF) race** (theory_teachers 3.4,
  theory_saddles 3.2): each k = 2 route C carries a log-amplitude ell_C in units of log(1/sigma),
  -1 + kappa_C at the first saddle, -1 for routes that appear later; on a plateau of length
  tau log(1/sigma) it grows by gamma_C tau; the route with the smallest -ell_C/gamma_C escapes; exact
  ties escape together; near ties (within 10%) are flagged.
- (c) else (k* >= 3): every fixed point of <R_C, (x) Z> over the routes of size k* is a candidate,
  classified like the first escape by the Riemannian Hessian of the route sub-TTN (Prop. 1: the escape
  is that sub-TTN's first-escape problem): one attractor: deterministic; several: **random**
  (seed-dependent tie, theory_saddles 3.4); fixed points that are saddles there: **rare** branches
  (rule **R4c**, made explicit after reading the s2s_core notes on 1-3-2 orders; untested beyond the
  13 runs that motivated it).
- The escape raises the rank by one on the internal edges of C; the next saddle is the fixed-rank
  critical point reached from theta_S plus the route direction (R1).

**R5 Time laws.** First plateau ~ sigma^-(K-2). k = 2: Delta t = tau log(1/sigma) + O(1) with
tau = -ell_C/gamma_C; small init, first later saddle: tau = (1 - kappa_C)/gamma_C with the head start
kappa_C = mu_C / (lambda_T a1^(K-k)) (lambda_T = <T, (x) y> of the first candidate, a1 = per-core
amplitude of the first saddle; pattern teachers: (1 - s_r/s_q)/(s_r s_q^((K-2)/K))); saddle start:
1/gamma_C. k >= 3: Delta t ~ sigma^-(k-2) (but see the caveat under R3); aligned-init prefactor
(1 - kappa_C)/((k-2) mu_C) at the first later saddle; a route through every core (k = K) escapes at the
absolute time sigma^-(K-2)/((K-2) mu) (theory_saddles 3.3); random-init prefactors are not predicted
(theory P5). Rotation: 1/|lambda_old| per unit log(1/eps) from an exact-saddle start, O(1/|lambda_old|)
(no resolved plateau) after a rare first escape (theory_escape 5.3, order note 5).

**R6 Hessian at plateau points.** Strict: lambda_min -> -gamma (k = 2) or lambda_old (rotation),
sigma-independent. Higher-order: lambda_min -> 0 like sigma^(k*-2) (theory_saddles 6.2-6.3).

**R7 Failure flags.**
- *Trap* (theory_saddles R12, section 8): unbalanced init (fan-in, or saddle start with fan-in
  noise): route degree k >= 5 -> "dead route possible (stable dead state)"; k = 4 -> "undecided";
  k <= 3 -> no trap. Balanced init: no dead trap. The flag allows both stalling and escaping, so a
  stall "matched" by it is admissible, not predicted.
- *Forgetting*: a learned component falls below 1/2 at the next saddle (stars, wrong order).
- *Wrong order*: a rare first-escape branch; with m = 2 and k >= 3 it is followed by a rotation and
  the forgotten component is relearned; with k = 2 the larger component is added directly; with
  m >= 3 the wrong order is kept. The last clause fails at the order-6 and "tie" saddles of section 9,
  where the escape goes along mixed patterns and the learned component is forgotten.
- *Blocked routes*: R != 0 but chi_e = r_e on an internal edge of every route.
- *Near tie*: in the k = 2 race (finite-sigma offsets O(1)/gamma can reorder, theory_teachers 3.4).

**R8 Covariance** (theory_covariance C1, C4, C5). The loss is 1/2 ||W - W*||_Sigma^2 in the original
coordinates, so the Hessian there gives the gradient-flow rates (no separate preconditioner); T =
Sigma W* drives the first escape and the head starts.

What the hopper does not contain because the theory does not: the race probabilities among
attractors (they need the init; theory_escape 2.5 gives a per-init recipe), random-init prefactors,
the sigma -> 0 behaviour of traps (conjecture in R12), the depth and length of star forgetting
(theory_saddles 11), and the generalized piggyback operator of theory_saddle_order (section 9).

## 2. Scoring

- **Sequence match per run.** A run matches if its observed order of learned components (with the
  experiment's own event definition: final 1/2 crossing for `locality`, first full rise for
  `s2s_core`, first 1/2 crossing for `order`) is produced by a main branch of the prediction; "rare"
  if only a rare branch produces it; "mismatch" otherwise. Within a random choice (race, tie, joint
  event) every order counts. Locality: a rotation run (a component crossing 1/2 again) matches only a
  branch with a rotation. s2s_core: the rotation flag is **not** scored. A stalled run counts as
  matched if the route it waits for carries a trap or blocked-route flag; these are reported separately
  below (the flag allows both outcomes). Runs stopped by a wall-clock or step budget are not scored.
- **Strict rates.** "Main" counts rare-branch runs as misses. Every rate is shown next to trivial
  baselines on the same runs (`summarize_results.baselines`, `results/baselines.csv`): **B1** weight
  order (largest first) and no rotation (locality's class "greedy" is exactly B1), **B1r** weight order
  with rotation ignored, **B2** any order without rotation.
- **Exponents.** Predicted k - 2 against the experiment's per-seed fits; agreement if within 0.2 (the
  refutation threshold of theory_saddles P14(ii)). Log slopes: agreement if within 10%.
- **Hessian.** Strict values: agreement if within 10%; higher-order: measured sigma-exponent of
  lambda_min within 0.3 of k* - 2.
- **Mismatch causes** (section 6) are labels from explicit checks written after seeing the mismatches,
  with free thresholds (dip test 0.5 and 3 final-rise widths; add-vs-rotation factor 3). They relabel
  mismatches; they do not change the counts.

## 3. The hopper reproduces the theory notes' worked examples (bookkeeping)

`results/theory_examples.csv`, `theory_sequences.txt` (hopper vs the number stated in the note):

| example | quantity | note | hopper |
|---|---|---|---|
| B4 on mps4 (theory_saddles P10) | q_A log slope; q_B degree | 2.333; 4 | 2.3333; 4 |
| B4 on binary4 / ht4 | degrees; q_A, q_B order on ht4 | 3, 3; tie (random) | 3, 3; tie of 2 attractors (random) |
| E8 on mps8 / binary8 | q_A slope; q_B coupling; q_A after q_B | 1.5; 0.7616; 0.3348 | 1.5; 0.76143; 0.33465 |
| N4 on mps4 (partial learning) | levels; partial-step rate | 0.26, 0.08, 0.0569; 0.1897 | 0.26, 0.08, 0.05689; 0.18974 |
| star (2; 1.6, 1.6) at q, mps4 / ht4 | lambda_min | -0.371573 / -0.431073 | -0.371573 / -0.431073 |
| star on mps4, random init | sequence (theory_saddles 4.6) | blended rank-one point, then "r, r' learned, q forgotten" (L = 2), then q relearned | the same (main branch); passing near q is a rare branch, gamma = 0.131 |
| odeco (1, 0.5) on binary4, smaller learned | piggyback lambda | -0.39685 | -0.39685 |
| "mixed" CP teacher on mps4 (theory_teachers 4.3) | edge order and cumulative slopes | 0 -> 2 -> 1; 6.823, 12.178, 14.467 | 0 -> 2 -> 1; 6.821, 12.178, 14.467 |
| sym(0.5) CP teacher on mps4 | second-escape slope A | 5.853 | 5.8534 |
| two-component tables (theory_saddles Table A) | route degree k, all (tree, S) of locality | Steiner size | equal on every row (226 cases) |

The rare-branch exponents are those of theory_escape E4 and 5.3: gamma = n (1 - rho)/rho with
n = chi_e for adjacent cores (e.g. 3 for B4's q_A on mps4 with chi = 3, giving 7.0; 4 for the adjacent
MPS4 pair at chi = 4), equal to the order agent's `gamma_pred` in all 32 rows of its sigma fits.

Reproducibility (audit section 1): a full re-run from empty caches gives byte-identical CSV and PNG
outputs (5.4 min on 8 processes) and identical predicted sequences in all 418 cases.

## 4. Sequences: do the runs follow a predicted sequence?

Scoring of section 2. "Main" = produced by a branch that survives as sigma -> 0 (deterministic or
random); "rare" = only by a branch with P ~ sigma^gamma; "trap-flag stall" = a stalled run admitted by
the k >= 4 trap flag (not a prediction); unscored = stopped by a budget.

**locality** (`results/locality_runs_scored.csv`; 11,446 runs of the committed configs, 11,432 scored):

| outcome | runs | share |
|---|---|---|
| main branch, not stalled | 10,437 | 91.3% |
| rare branch | 262 | 2.3% |
| stalled, admitted by the trap flag (fan-in or saddle start, route k >= 4) | 645 | 5.6% |
| mismatch (41 balanced-init stalls, 47 sequences) | 88 | 0.8% |

**Against trivial baselines** (`results/baselines.csv`, same runs; B1 = weight order and no rotation,
B1r = weight order with rotation ignored, B2 = any order without rotation):

| subset | runs | hopper main + rare | hopper main | B1 | B1r | B2 |
|---|---|---|---|---|---|---|
| all scored | 11,432 | 99.2% | 96.9% (645 via the trap flag) | 85.8% | 87.8% | 91.6% |
| not stalled | 10,746 | 99.6% | 97.1% | 91.3% | 93.4% | 97.5% |
| one main order, not stalled | 7,965 | 99.7% | 97.0% | 95.8% | 97.9% | 97.5% |
| - saddle start (first component fixed by the protocol) | 2,919 | 99.97% | 99.97% | 99.97% | 99.97% | 100% |
| - fan-in / balanced, two components | 4,291 | 99.95% | 96.1% | **96.1%** | 99.5% | 96.6% |
| - fan-in / balanced, multi-component | 755 | 97.4% | 90.6% | 78.0% | 80.5% | 93.5% |
| - main order differs from the weight order (B4 mps4, E8 mps8) | 107 | 100% | 88.8% | 0% | 0% | 88.8% |
| several main orders (race / joint event), not stalled | 2,781 | 99.1% | 97.5% | 78.4% | 80.5% | 97.2% |

Reading: where the hopper commits to one order, that order is B1 in all but 107 runs. On the
informative runs (one main order, not stalled) it beats B1 by 1.2 points (97.0% vs 95.8%); on fan-in /
balanced two-component runs the two predictions are identical (96.1% each). **The hopper beats the
baseline in one place: the 107 multi-component runs whose main order is not the weight order (B4 on
mps4, E8 on mps8), 88.8% vs 0%, through the route rule** (fewest cores first; theory_saddles and
locality). On race runs its content is "no rotation", the same as B2. For two-component teachers its
extra content over B1 is only in the branches: rotation after a wrong first escape on matrix-like pairs
(m = 2, k >= 3: all 147 rotation-class runs there), direct addition on adjacent pairs (k = 2: 19 runs),
no rotation on tensor-like pairs (m >= 3) - which fails: 19 of the 20 rotation-class runs on
tensor-like pairs are genuine forgetting events at order-6 or tie saddles (section 9). These are
theory_saddles 4.4 and H1b, confirmed by the order agent before the hopper existed.

The 2,869 runs with several main orders include 735 k = 1 joint-event runs and 48 R = 4 runs with 4
main orders; the 8,563 runs with a single main order include 3,176 saddle starts and 598 stalls.

**Stalls.** The trap flag is set on every route with k >= 4 under unbalanced init, so "every stall sits
on a flagged route" says only **no stall at k <= 3: 0 of 4,531 scored two-component runs with k = 2, 3**
(the citable content). Stall fractions under fan-in are not predicted: k = 4 127/1056, k = 5 20/768,
k = 6 117/360, k = 7 20/60, k = 8 33/57. The 41 balanced-init stalls are mismatches (section 6).

**Rare branches** are fixed by the algorithm (every non-maximal fixed point of the escape problem), not
chosen per run, but they cannot fail run by run, and at the sigma used they are not rare: in 10 of the
28 locality rows with at least 4 rare runs the rare branch takes >= 20% of the runs (up to 46%, mps8
S = {0,7} balanced); stars 48/120; order binary4 R = 3, 4: 30-65%. They can be tested only through
their sigma trend. Locality reuses each seed's init direction at every sigma, so a seed-level test is
the right one (audit, 47 rows with a single rare gamma): 64 seeds are on the rare branch at the row's
largest sigma, 7 still at the smallest (P ~ sigma^gamma predicts 7.7, a sigma-independent race 64),
and 2 seeds join. So the first-escape rare branches behave as sub-dominant branches as sigma -> 0
(theory_escape E9; consistent with the order agent's gamma fits). All 262 locality rare runs come from
first-escape rare branches; none depends on R4c.

**s2s_core** (`results/s2s_core_runs_scored.csv`; 1,576 runs): 1,151 scored (863 complete, 288
trapped), 425 unscored (215 stalled at the time budget, 210 unresolved). Of the 863 complete runs, 850
are main and 13 rare; the 288 trapped runs are fan-in runs on routes with k = K >= 4, admitted by the
trap flag; no balanced run traps. Against the baseline:
- **binary4 is the only informative cell** (one main order): 367/380 complete runs main (96.6%),
  **equal to the weight order** (B1 96.6%); the binary4 greedy first escape is predicted
  deterministically and observed in 380/380 runs.
- mps4, ht4, binary8, mps8: all 6 orders are main (random race), so the 483 complete runs there match
  trivially (B1 47.8%).
- **The 13 rare runs (binary4 close teacher, order 1-3-2) depend on rule R4c alone, which was added
  after reading the s2s_core notes on these orders**: without R4c they are 13 mismatches, and they
  cannot count as support for R4c (gamma = (0.85 - 0.7)/0.7 = 0.21; the following strict saddle has
  the predicted -0.133186, measured 0.93 and 0.98 of it in two runs, 3.7 in a third taken
  mid-transition).
- The rotation flag is not scored for s2s_core.

**data_teacher.** Stars (w = 1.6, random init, `results/data_teacher_star_classes.csv`): every observed
class is a predicted branch, but the strict main-branch rate is 72/120 (60%): binary4 "q last" 42/60
(main: the top DLN mode r + r' first), "near q" 18/60 (rare, gamma = 0.131); mps4 "blended" 30/60
(main: the first escape ends at the blended rank-one point, a_q = 0.67), "near q" 24/60 (rare,
gamma = 0.131), "q last" 6/60 (rare, gamma = 0.60). With gamma = 0.131, sigma^gamma is 0.5-0.75 at the
measured sigma, so "rare" is not rare there. Aligned covariance (`data_teacher_aligned_cov.csv`):
binary4: first component deterministic (the larger Lambda s) for both teachers, measured 0.90-0.975 (T1)
and 0.75-0.80 (T2), the rest being the predicted rare branch (gamma 0.6 and 0.2, decreasing with sigma);
swap (rotation) predicted on the main branch for T2 only (Lambda_0 s_0^2 > Lambda_1 s_1^2), measured
0.25 / 0.53 / 0.75 at sigma = 0.1 / 0.05 / 0.02 (T2) against 0.025-0.075 (T1, all on wrong-first runs);
mps4: random first (measured sigma-independent 0.775 and 0.55), no swap (0/240 runs). MPS-8 teachers on
a binary-8 student (`data_teacher_cross_tree.csv`, 12 teachers, 264 runs): P11 holds on every predicted
main sequence (249/256, data_teacher's own count); the first edge to cross is a bottom edge in 264/264
runs. All of this re-tallies data_teacher's own predictions (bookkeeping).

**order** (fix round, merged on main; `results/order_multi.csv`): re-run on current main after fixing
the H1b filter of `validate_order.py` (the fix round renamed the set `saddle_b4` to `h1b_saddle_b4`;
without the fix the H1b table came back empty). The GF order frequencies and the 32 gamma rows are
unchanged; only the order agent's race-model column changed. R = 3 and 4 odeco teachers: all observed
orders are main on mps4, mps6, binary8 (random race); on binary4 the greedy order is the main branch
(0.30-0.71 of the runs at sigma = 0.05) and other orders are rare branches (0.30-0.65), except 3% (R = 3)
and 5.5% (R = 4) of the runs at rho = 0.9 (orders 2-3-1, 3-2-1 and five R = 4 orders) that need the
add route to beat a rotation (section 6).

## 5. Laws: degrees, exponents, slopes, Hessians, levels

`results/summary_laws.csv`, `fig_laws.png`. Agreement counts (section 2 tolerances), with what each
check is:

| dataset | quantity | agree | what it is |
|---|---|---|---|
| data_teacher | rank-one plateau level (12 teachers x 3 trees) | 36/36 (to 6 digits) | static recomputation |
| | saddle type strict / higher-order | 36/36 | static recomputation |
| | lambda_min at the rank-one saddle (k = 2 rate) | 12/12 (to 4 digits) | static recomputation |
| | stars: strict iff rho(S) > s_q; lambda_min | 20/20; 8/8 (to 1e-15) | static recomputation |
| | log slopes (from the saddle 12, from random init 19) | 31/31 | 27 reproduce data_teacher's own pred_A (to 8e-8); 4 new |
| | exponents k* - 2 (binary / HT, MPS4) | 31/31 | reproduce data_teacher's pred_p |
| locality | route degree k of the second event | equal to locality's k_rank on every row (197/197) | re-tally |
| | exponent (two-component tables and multi-component levels) | 179/217 | locality's fits vs locality's H2 predictions |
| | lambda_min sigma-exponent (k >= 3) | 144/171 | re-tally |
| | k = 2 log slope | 34/38 | rows whose fit chose a log law only (see below) |
| | k = 2 lambda_min | 23/25 | re-tally |
| | plateau level | 55/89 | 21 misses are levels detected in < half the runs, 13 are L_mid deviations > 5% on other-branch orders |
| s2s_core | t_1 and gap exponents K - 2 | 45/47 | s2s_core's own comparison |
| | lambda_min sigma-exponent | 29/30 | re-tally |
| | strict wrong-order lambda (measured / predicted) | 2/3 | equals s2s_core's lam_theory; the third run is mid-transition |
| order | wrong-order exponent gamma (heuristic) | 29/32 | the order agent's gamma_pred |
| | wrong-order lambda_min; rotation log slope | 4/4 (5 digits); 4/4 (within 1.5%) | the order agent's lam_pred, slope_pred |

**data_teacher, "174/174", reclassified:** 112 rows (levels, saddle types, k = 2 lambda_min, star
strictness, star lambda) are **static recomputations** at the same saddles data_teacher computed itself
(two codes, one object; agreement to 6 digits is expected and does not test the dynamics); 58 rows (27
log slopes and 31 exponents) **reproduce data_teacher's own predictions** against its own
measurements; **4 rows are new** (misaligned covariance: the e2 sweep at theta = 20 and 5 deg,
misaligned_20, misaligned8_20), measured within 0.2-0.8%, but the head start is under 1% of the slope
there, so they test 1/gamma (known), not the head start. Cite the 174 rows as
implementation agreement. The saddle-start slopes are 1/|lambda_min| at the point the runs started
from, close to a tautology for an S2S test.

**Locality log slopes:** the 34/38 keeps only rows where locality's fit chose a log law (selection on
the outcome): 10 main-branch k = 2 rows with power-law fits are dropped, mostly plateaus locality rarely
detected, but one is a clear miss: TC mps8 small init, level 2, predicted 7.74, measured 5.62.

What the comparisons add to the theory notes' own checks:

- **Multi-saddle accumulated-utility slopes (the one new numerical check, safe to cite).** The hopper
  applies the AGF bookkeeping of theory_teachers 3.4 across several saddles of nested pattern teachers
  (rule R4b). **TC on mps8 from the saddle: predicted log slopes 1.667, 5.356, 7.737, then a k = 4
  plateau (exponent 2), then 80.26; measured (locality's level table) 1.718, 5.540, 7.581, 2.018,
  78.92 (5 plateaus within 3.5%).** N4 on mps4 (small init): 0.667, 5.270, 7.271 against 0.517, 5.143,
  7.038 (levels 1-2 within 3.3%; the first plateau is short and hardly resolved). TA and TB level 1 are
  10.5% low (3.48, 3.46 vs 3.89, 3.86: partial carry-over between saddles, A-carry, not modelled); TC
  from small init misses at level 2 (5.62 vs 7.74).
- **Plateau levels on the branch actually taken** (a small correction to locality's level table): B4
  on ht4, order 0>1>2: 0.18 (measured 0.1799; table 0.045); E8 on binary8, order 0>1>2: 0.245
  (measured 0.2450; table 0.08).
- **Covariance** (weak): the head start with T = Sigma W* gives the misaligned-covariance random-init
  slope, which data_teacher left unpredicted: theta = 20 deg 12.22 vs 12.14 +- 0.03, theta = 5 deg 193.4
  vs 194.9 +- 1.2, misaligned_20 at d = 2: 52.57 vs 52.46 +- 0.10; the no-head-start value 1/gamma is
  12.31, 193.5, 52.94, so this tests 1/gamma, not the head start.
- Everything else (locality's route sequences, data_teacher's rank-one saddles and slopes, order's
  gamma_pred and H1b values, s2s_core's exponents) is a check of the hopper's implementation against
  the experiments' own predictions, not an independent test of the theory.

## 6. Mismatches and their causes

Every mismatch, with the run lists, is in `results/mismatches.csv` (sequences) and
`results/exponent_mismatches.csv` (exponents). The cause labels come from explicit checks
(`validate_locality.mismatch_cause`, `summarize_results.exponent_causes`) **written after seeing the
mismatches**, with free thresholds; they are post-hoc labels, not tests. The predictions and the
scoring were not changed by them.

**Sequences (locality, 88 runs):**

| cause label (v1) | runs | where | v2 reading |
|---|---|---|---|
| stall: pinned or very slow route under balanced init | 41 | k >= 4 MPS routes (mps8 {1,5}, {1,6}, {0,7}, {2,5}; mps6 {1,3,5}) | failures of the time law: the trap rule (R7) says balanced init has no dead trap and locality confirms these routes are alive, but nothing predicts escapes 1e5 times later than normal (locality 7.5). All sit at correct-order saddles, where the order is k* (section 9). |
| threshold artifact: events not separated (separation ratio < 3) | 34 | TD on mps8 (10), TA, TB, N4 on mps4 / ht4 / binary4 (14), two-component runs at large sigma (10) | the same defect affects 2,711 matched runs (2,745 scored runs have rho_min < 3), so the excuse is not applied symmetrically; resolved non-stall runs only: main 97.6%, main + rare 99.8%. **7 of the 10 two-component runs are generalized-piggyback escapes with forgetting** (section 9: ht4 {0,1,2,3} chi = 4 at sigma = 0.15; ht4 {0,1,2} balanced, binary8 {0,2,4} chi = 4 and the mps6 whole chain, at large sigma). |
| threshold artifact: transient dip | 13 | after a (race-allowed) wrong first escape on a tensor-like pair: ht4 {0,1,2} balanced (5), ht4 {0,1,2,3} (3), binary8 {0,2,4} at chi = 4 (4), mps6 {2,3,4} (1) | **12 of the 13 are not threshold artifacts**: the first-learned component is forgotten (overlap down to 2e-6 ... 0.03) during an escape along mixed patterns at the generalized-piggyback amplitude, at order-6 (ht4 {0,1,2,3}) or tie saddles (ht4 {0,1,2}, binary8 {0,2,4}); section 9. Only the mps6 {2,3,4} run (k* = 3) is a touch without forgetting (overlap >= 0.905 after learning). |

So the v1 statement "no sequence mismatch needs the theory to be wrong" is withdrawn. The causes were
assigned post hoc; the 41 balanced stalls are slow or pinned escapes the time law does not predict; and
19 of the 88 mismatches are the generalized-piggyback escapes of theory_saddle_order (Theorem 2 and the
tie case), which the hopper's rules (R3, R7: "m >= 3: wrong order kept, no rotation") get wrong.

The order experiment's binary4 runs at rho = 0.9 that no branch produces (7 orders, 0.5-2% of runs each,
8.5% in total over R = 3 and 4; `results/order_multi.csv`) are runs where, after a wrong escape, a direct
add route beat the rotation. At sigma = 0.05 the rotation takes 1/|lambda| = 5.6-10.4 time units and the
aligned add route sigma^-1/mu = 20 (less with its head start): comparable, so the sigma -> 0 rule of
theory_saddles 4.4 (the rotation wins) is not yet in force (finite sigma; post-hoc label).

**Exponents (locality, 38 of 217 outside 0.2):** 23 within 2 SE of k - 2 (rows with 2-8 fitted seeds,
mostly k >= 5); 2 pre-asymptotic (mps6 {1,3,5} small init 3.76, but 3.05 / 3.15 at sigma <= 0.03 from the
saddle / balanced; ht4 {0,1,2} fan-in 4.23, but 4.06 balanced over a decade); 2 with no escaping run
(mps8 {0,7} saddle, midroot6 {0,5} chi = 4); 2 with 1-2 seeds (mps8 {1,6} small, mps8 {0,7} balanced);
4 multi-component levels (B4 ht4 fan-in 4.73 +- 1.52, a stall-prone row; three rarely detected levels);
**5 unexplained: steep MPS routes** (mps8 {1,5} from the saddle, 3.39 and 3.67 at sigma <= 0.03; mps6
whole chain small init 4.84 and chi = 4 4.99; mps8 {1,6} balanced over one decade 4.51), the open
problem of locality 7.9 (route ends that shrink during the plateau; theory_saddles P14(ii) met on thin
evidence). The saddle-order theorem does not explain them (section 9: they sit at correct-order
saddles of order exactly k*, and an order drop would make exponents smaller, not steeper).

**Other laws:** the four k = 2 slopes outside 10% are TA and TB level 1 (measured 10.5% below the
prediction; a route that appears only at the second saddle starts at log-amplitude -1 in rule R4b; if
part of the previous route's amplitude carries over, the theory's assumption (A-carry), the slope is
lower, as measured, but the hopper has no rule for how much) and two short plateaus (mps4 {1,2} at
w = 0.75, head start 0.75; TC mps8 small init level 0). The s2s_core misses are ht4 gap 2 (4.75 +- 0.14
and 4.27 with few seeds; s2s_core notes the same) and one strict-saddle eigenvalue taken mid-transition
(chi = 6, no resolved plateau). The three order rows outside the tolerance are adjacent pairs whose
measured gamma is still rising toward n (1 - rho)/rho at sigma = 1e-5 (the order agent's finding).

## 7. Status of the claims

| claim | status | new or bookkeeping |
|---|---|---|
| The hopper reproduces every worked example of the four theory notes and locality's route degree on every row (197/197); a re-run from empty caches is byte-identical | checked | bookkeeping (implementation) |
| Main-branch sequence agreement, always next to the baseline: locality 96.9% of 11,432 scored runs (645 of them stalls admitted by the trap flag; 97.1% of the 10,746 runs that did not stall vs 91.3% for "weight order, no rotation"); informative locality runs 97.0% vs 95.8%; s2s_core binary4 96.6% vs 96.6% | empirical | bookkeeping: the main order is the weight order except in 107 runs |
| Route-rule orders where the main order is not the weight order (B4 mps4, E8 mps8): 95/107 main, 107/107 with the predicted wrong-first branch | empirical | bookkeeping of theory_saddles + locality (the one place the hopper beats the baseline) |
| Rotation after a wrong first escape on matrix-like pairs (all 147 rotation runs on m = 2, k >= 3 pairs); direct addition on adjacent pairs | empirical | bookkeeping of theory_saddles 4.4 / H1b (order agent) |
| No stall at k <= 3 (0 of 4,531 scored two-component runs) | empirical | bookkeeping of R12 + locality |
| First-escape rare branches thin out as P ~ sigma^gamma in aggregate (7 of 64 seeds still rare at the smallest sigma vs 7.7 predicted) | empirical (seed-level, audit) | new aggregate evidence, consistent with the order agent's gamma fits |
| **Multi-saddle AGF slopes of nested pattern teachers: TC on mps8 from the saddle, 5 plateaus within 3.5%**; N4 on mps4 levels 1-2 within 3.3% | derived (theory_teachers 3.4 + theory_saddles 3.2 combined), empirical | **new numbers, safe to cite**; TA/TB level 1 10.5% low, TC small init level 2 a miss |
| data_teacher law checks: 112 static recomputations + 58 reproductions of data_teacher's predictions + 4 new (weak) rows | checked | bookkeeping (implementation agreement) |
| Rare branches at later escapes (R4c), gamma from the route problem's Riemannian Hessian | derived (Prop. 1 + E9), heuristic exponent | made explicit after reading the s2s_core notes; untested beyond the 13 runs that motivated it |
| Head start with T = Sigma W* gives the misaligned-covariance random-init slope to 0.8% | empirical (weak: tests 1/gamma) | new, weak |
| Levels of the branch actually taken (B4 ht4, E8 binary8 tie branches) | empirical | small correction to locality's level table |
| "No mismatch needs theory wrong" (v1) | **withdrawn** | causes post hoc; 41 balanced stalls are time-law failures; 19 mismatches are generalized-piggyback escapes the hopper's rules miss (section 9) |
| Saddle-order test on the scored teachers (section 9) | see section 9 | new |

## 8. Limitations and open issues

- **Saddle order.** The hopper uses order = k* (R3, R5) and the classical piggyback only. It should
  implement the generalized piggyback operator (theory_saddle_order; `gp_check.py` computes it at
  rank-one saddles) and a tie rule (when a polymer curve has the route's order, the larger constant
  decides the escape direction, section 9). Saddles with several learned components are outside the
  theorem (108 hopper events with k >= 5 leave such saddles, `results/gp_later_saddles.csv`).
- **Race probabilities are not computed.** Where the theory says "random" (tensor-like first escapes,
  k* >= 3 ties) every order is allowed, so 2,869 locality runs and the tensor-like s2s_core runs test only
  the absence of rotations (which fails at order-6 and tie saddles) and the laws. theory_escape 2.5 gives
  a per-init recipe (the order agent verified it on 40,800 MPS runs); adding it would turn these into
  per-run predictions.
- **Rare-branch probabilities** have only an exponent (prefactor unknown), and some exponents are small
  (0.13 for the star): "rare" can mean 20-65% at accessible sigma.
- **Traps:** the rule flags k >= 4 under unbalanced init but does not give a rate (fan-in stall fractions
  0 at k <= 3, 12% at k = 4, 3-58% at k >= 5). Under balanced init 41 runs stall (pinned or extremely slow
  escapes in locality's longer re-runs, none dead).
- **Carry-over between saddles (A-carry)** is assumed zero for routes that appear at a later saddle;
  TA and TB suggest a partial carry-over for nested teachers (slopes 10% lower than predicted).
- **Finite sigma:** the hopper's order is the sigma -> 0 order. It flags near ties in the k = 2 race but
  does not model O(1) offsets, the add-route vs rotation competition, or onset thresholds.
- **Scoring:** the s2s_core rotation flag is not scored; unresolved events (rho_min < 3) are excused for
  mismatches but not for matches; both should be treated symmetrically (audit section 9).
- **Scope:** dense tensors, N <= 8; product covariances only (cross-site correlations, data_teacher step
  3, are not covered); gradient flow only.

## 9. Test of the saddle-order theorem on the scored teachers (new)

**Question.** theory_saddle_order (Theorems 1, 2; level version Theorem 1' in its verification): at a
saddle with one learned component q, the order is k* if lambda_max(S|L_l) < s_q for every l < k*/2, and
2 l* < k* if lambda_max(S|L_l*) > s_q (S = the generalized piggyback operator on core sets of degree
<= l). Do the teachers of the scored experiments reach saddles in the second regime, and are the runs
through them among the unexplained cases (the 5 steep MPS exponents, the mismatches, the stalls)? This
tests the theorem on data taken before it existed; it is reported as it came out.

**Method.** `gp_check.py` (results `gp_saddle_order.csv`, `gp_later_saddles.csv`): all 288 (tree, d,
chi, teacher) cases of the four prediction caches; every first-escape candidate of the hopper (main and
rare branches) gives a rank-one saddle, 587 in all. At each: the rank-one critical point in whitened
coordinates (power iteration; criticality ||R_{v}|| / ||R|| <= 3.3e-15), core bases rotated so that
q_v = e_0, k*, and lambda_max(S|L_l)/s_q for l < k*/2 with the operator of
`experiments/theory_checks/saddle_order/gp_operator.py` (sets restricted to site-carrying cores; it
reproduces the theory note's pair-star 1.131 -> order 4, its control 0.849 -> order 5, HT4 1111 (1.5)
-> order 6, MPS6 110001 (1.5) -> order 4). `gp_constants.py` (`gp_constants.csv`) computes the leading
constant m(t)/t^p of min over ||delta|| = t of L - L_S at the saddles below, with the sphere
minimization of the theory check. `gp_escape_direction.py` (`gp_escape_direction.csv`) reads the
escape direction from stored trajectories (no new runs).

**Landscape: where the order drops** (proven: Theorems 1', 2 applied by exact computation; pattern
teachers, s_r > 0, index condition satisfied):
- 160 of the 587 rank-one saddles have k* >= 5. 147 saddles (all k*) have lambda_max(S_1) > s_q:
  classical piggyback, strict, already the hopper's rotation rule (wrong-order m = 2 saddles, stars).
- **Level l = 2: none. Level l = 3: 30 saddles, order 6 instead of k* = 7**, all with a component
  learned before a larger one whose pattern differs on two sibling pairs of leaf cores (split
  {l0, l1} | {l2, l3}, two vertex-disjoint degree-3 polymers, lambda_max(S|L_3) = the larger weight):
  locality ht4 S = {0,1,2,3} (chi = 2, 4) and ht8 S = {0,1,2,3}; s2s_core odeco on ht4 (6) and binary8
  (8) with the second or third component learned; order binary8 two-component (rho = 0.5, 0.8, 0.95)
  and odeco R = 3, 4 (10). The first escape reaches them through the race (main branch).
- No correct-order saddle drops (Cor. 2a covers two components; the multi-component teachers too), none
  on an MPS, none borderline (largest lambda/s_q below 1 at k* >= 5: 0.99, a star at level 1).
- 108 hopper events with k >= 5 leave saddles with two or more learned components; the theorem does
  not cover them.
- **Ties.** Where a split has degree exactly k*/2 above s_q, a polymer curve has the route's order k*:
  the theorem still gives order k*, but the constant decides the descent direction. ht4 S = {0,1,2}
  and binary8 S = {0,2,4} (wrong first, k* = 6; a polymer on the sibling pair plus an old rotation of
  the third core): m(t)/t^6 -> -0.0252, against -0.0042 for the single route (derived leading
  constant -(s_r^2 - s_q^2)/(54 a^6) = -0.02516); mps4 S = {0,1,2,3} and the mps6 whole chain: the
  minimum equals the route value (-0.0625, -0.00463), the polymer curves being 2x weaker. At ht4
  S = {0,1,2,3} (wrong first) the sphere minimum gives slopes 6.02 -> 6.0006 and m/t^6 -> -0.002097 =
  -s_q (s_r - s_q)/(216 a^6) (two symmetric polymers): the order-6 drop holds numerically for this
  teacher.

**Dynamics on the old data: escape direction.** Two signatures distinguish the polymer escape from the
route at small overlap a of the escaping component: x = (L_S - L)/(s_r^2 a) is ~1 for the route and
1 - s_q/s_r for two symmetric polymers (at most that for any two-polymer configuration); in full
trajectories the mixed patterns (larger component on some cores of its support, learned one on the
others) reach sqrt(s_q s_r a) for polymers (mixed ratio ~1) and stay near 0 on the route.

| saddle (wrong first unless stated) | escapes | x | mixed ratio | learned component afterwards |
|---|---|---|---|---|
| locality ht4 S = {0,1,2,3}, order 6, chi = 2 (sigma 0.40-0.60) | 5 | 0.50-0.62 (polymer 0.50) | 0.84-1.01 | forgotten in 4 (overlap 2e-5 ... 0.03), 0.63 in 1 |
| same, chi = 4 (sigma 0.15-0.60) | 6 | 0.60 at sigma 0.15, 0.72-0.91 at 0.2-0.45, 0.35 at 0.6 | 0.87 -> 0.32 | 0.31 at sigma 0.15, >= 0.69 above |
| locality ties, polymer-favoured: ht4 S = {0,1,2} balanced, binary8 S = {0,2,4} chi = 4 (sigma 0.05-0.5) | 14 | 0.41-0.49 at sigma <= 0.24, further off above (two symmetric polymers 0.50; the landscape-optimal polymer + rotation curve 0.375) | 0.79-1.03 (1.0 and 1.41 for the same two curves) | forgotten in 13 (overlap 2e-6 ... 0.38) |
| locality ties, route-favoured: mps4 S = {0,1,2,3} (sigma 0.01-0.2) | 24 | 0.79-1.00 | <= 0.38 | kept (>= 0.90) |
| same, mps6 whole chain (sigma 0.08-0.5) | 5 | 0.09-0.71 | 0.58-0.91 | 0.32-0.85 |
| locality order k*, no tie (mps4, mps6, mps8, star3 tensor-like pairs) | 67 | >= 0.97 at sigma <= 0.02 (lower at large sigma) | <= 0.12 at sigma <= 0.02 | kept (>= 0.90) in all 67 |
| locality greedy-first controls (2 per setting) | 32 | >= 0.99 | <= 0.04 | kept |
| s2s_core ht4, order 6 (close teacher, seed 4) | 5 sigmas | **0.2998, 0.2999, 0.289** at sigma 0.07, 0.15, 0.2 (polymer 0.300); 0.27, 0.12 at 0.3, 0.4 | - | - |
| s2s_core ht4, order 6 (separated: chi = 6 seed 2; chi = 3 seed 9) | 2 + 2 | 0.348 at sigma 0.1 (polymer 0.333), 0.93 at 0.3; 0.55, 0.71 at 0.3, 0.4 | - | - |
| s2s_core, every other escape from an order-6 saddle (ht4 and binary8; 21 toward a larger, 10 toward a smaller component) | 31 | 0.94-1.01 at sigma <= 0.3 (20 escapes, route); at sigma = 0.4 8 of 11 in 0.93-1.07, 3 off (-6.1, -0.24, 0.32: plateau not settled) | - | - |
| order h1b saddle starts, binary8 rho = 0.8, order 6 (sigma 0.15-0.35) | 29 | median 0.978 (0.70-0.999; polymer 0.20) | - | - |
| controls at correct-order saddles: s2s_core greedy first; order h1b MPS4, MPS6 | 70 + 80 + 19 | median 0.975, 0.991, 0.989 | - | - |

**Time law.** By S9 of theory_saddle_order (derived) a polymer escape takes ~ sigma^-(2l-2) =
sigma^-4 instead of the route's sigma^-5. s2s_core per-seed gap-1 exponents (sigma 0.07-0.4): the one
ht4 seed that escapes along the polymers at every sigma with data (close, seed 4) has **4.03**; the
seed with intermediate x (separated, seed 9) 4.38; the seed that switches from polymer to route inside
the range (separated chi = 6, seed 2) 4.83; the ht4 wrong-first seeds whose escapes are route-like 4.97-5.47;
binary8 wrong-first seeds (route) 4.89-5.04; order h1b binary8 (route) t ~ sigma^-5.3+-0.3. The mean
over wrong-first ht4 seeds (4.96, 13 seeds) hides this split.

**Where these runs sit among the unexplained cases.**
- *5 steep MPS exponents:* not explained. They sit at correct-order MPS saddles (order exactly k*), and
  an order drop would lower the exponent, not raise it.
- *Mismatches:* **19 of the 88 locality sequence mismatches** are these escapes: 12 of the 13
  "transient dips" (ht4 S = {0,1,2,3}: 3, order 6; ht4 S = {0,1,2} balanced: 5 and binary8 S = {0,2,4}:
  4, polymer-favoured ties) and 7 of the 34 "events not separated" (ht4 S = {0,1,2,3} chi = 4 at sigma
  0.15; ht4 S = {0,1,2} at 0.26, 0.36; binary8 S = {0,2,4} at 0.35, 0.5; mps6 whole chain at 0.24,
  0.5). Each is a wrong-first run on a tensor-like pair where the first-learned component is forgotten
  (overlap 2e-6 ... 0.38) while mixed patterns grow at the polymer amplitude: a rotation through a
  generalized-piggyback saddle, which the hopper's rule "m >= 3: wrong order kept" misses and its
  post-hoc labels called threshold artifacts. The 13th dip (mps6 S = {2,3,4}, k* = 3) keeps its
  component (>= 0.905).
- *Stalls:* none sits at an order-6 saddle. The ht4 S = {0,1,2,3} stalls have the larger component
  learned (a_stall = (1, 0): the correct-order saddle, k* = 7, dead routes under fan-in); every
  wrong-first ht4 / binary8 run of s2s_core escapes its order-6 saddle, and its traps and budget stalls
  sit at saddles with two learned components (outside the theorem).

**Verdict.** The regime of Theorem 2 occurs in the scored teachers (30 of 587 rank-one saddles, all
wrong-first, ht4 / ht8 / binary8, order 6 < 7), and the old data show its mechanism: escapes along mixed
patterns with x at the polymer value to 3 digits (s2s_core ht4: 0.2998 vs 0.3000) and mixed amplitudes
at 0.84-1.03 of the polymer prediction (locality), with the sigma^-4 law in the one seed that takes
this escape throughout (4.03). At the sigma used it coexists with the route (seed-dependent; binary8 and
most ht4 seeds take the route). It explains 19 locality mismatches together with the tie case (same
order, larger constant), none of the steep MPS exponents and none of the stalls. Status: landscape
classification **proven** (theorems applied by exact computation); ht4 S = {0,1,2,3} order 6 and the
tie constants **empirical** (numerical sphere minima; the tie constant's formula derived here); escape
signatures and the forgetting that follows a generalized-piggyback escape **empirical, new**
(theory_saddle_order leaves the trajectory after the escape open); the sigma^-4 law: one seed,
consistent with S9 (derived).
