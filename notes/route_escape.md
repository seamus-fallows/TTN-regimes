# route_escape: leaving a known multi-pattern plateau

Agent `route-escape`, branch `agent/route-escape`, brief `notes/briefs/route_theory.md` sections 1-3 and 4.2.
Experiment folder `experiments/route_escape/`; raw outputs `/root/outputs/route-escape/` (not in git).
Paper under test: `report/ttn_saddle_order_and_aligned_training_dynamics.pdf` ("the paper").

**Part I (sections 1-6) is the pre-registration, committed and pushed before any sweep.** It is not
edited afterwards; deviations go to section 7 with the reason. Results are in Part II.

## 1. Question and terms

A tree tensor network (TTN) is trained by gradient flow on a "pattern target" W* = sum_a s_a U_a
(U_a a one-hot product tensor, the string a; s_a its weight). A *plateau* is a long stretch where the
loss barely moves; the theory says plateaus sit at *partial-target points* theta_J, where the network
represents exactly the patterns of a learned set J (W = sum over J of s_a U_a). Here the network is
started exactly at theta_J, nudged by a small perturbation of size eps, and we watch how it leaves.

- *Route depth* c_J(r) of a missing pattern r: the number of cores that r cannot borrow from any
  learned pattern (the routing region R_J(r), paper eq. 8.4). c*_J = min over missing r.
- *Saddle order* p: the power with which the loss can first decrease, L - L_J ~ -t^p at distance t
  from theta_J (paper Def. 2.1). p = 2 means an ordinary negative curvature direction.
- *M** (block-aligned manifold): the subspace of parameters that only uses the target's own
  "local entries" (paper Sec. 7.2); it is invariant under the flow (Thm 7.7), so a start inside it
  stays inside it.
- *Shortcut*: a full-space order below c*_J (paper Remark 8.5): perturbations outside M* that
  cooperate ("spectral" mechanism of Thm 3.2).

Predictions under test:
- **P1 (Thm 8.4(3), proven).** Order inside M* = c*_J.
- **P2 (Thm 8.4(2), proven).** Full-space order <= c*_J. Where it is lower: which perturbation does it.
- **P3 (Thm 7.7, proven).** A start inside M* follows the reduced flow exactly.
- **P4 (eq. 6.16, derived; eq. C.6).** From balanced dormant entries b(0) = eps, a route of depth c
  escapes after eps^-(c-2) / ((c-2) gamma s_r) (c >= 3), log(1/eps) / (gamma s_r) (c = 2),
  gamma = A_r(J) the product of the active entries of r at theta_J.
- **P5 (escape-time law, derived).** From a generic start the escape time scales as eps^-(p-2)
  (log for p = 2), p the full-space order (= c*_J unless there is a shortcut).
- **P6 (greedy rule, App. C, conjecture-level).** The first pattern learned is the missing r with the
  smallest c_J(r); ties go to the larger s_r A_r(J).
- **P7 (Cor. 8.6, proven).** Seen from a larger learned set J' ⊃ J, the route of r is no longer:
  c_J'(r) <= c_J(r) (and so, on M*, the order and the escape exponent).
- **P8 (Thm 9.2, proven under Assumption 9.1, which is not constructive).** On the family "two
  learned patterns that share no entry, each missing pattern differing from one parent on a small
  region", the full-space order is min over parents a of min(k*_a, 2 l*_a).

## 2. Cells

Bond dimension: max over edges of the number of star blocks, + 2 spare channels (chi below); d =
physical dimension; P = number of parameters. theta_J is the balanced representative
`AlignedManifold.partial_target(J, relax_from=1e-2)`. c = route depth, R = routing region (node
indices), pair = smallest pairwise Steiner cost min over a in J of |St(V_ar)|, sA = s_r A_r(J) at the
balanced theta_J. "Thm 9.2" = the formula min over parents of min(k*_a, 2 l*_a) with the parents of
the config (`configs/cells.yaml`; only meaningful for the Thm 9.2 family PS2, T92a-d; the other cells
splice, i.e. c < pair, which Assumption 9.1 excludes).

| cell | tree, size | learned J (weight) | missing: pattern (weight, depth c, region R, pair, sA) | c*_J | greedy | Thm 9.2 |
|---|---|---|---|---|---|---|
| EX87_J01 | mps7, d=3, chi=5, P=405 | 0000000 (1.0), 1111111 (0.9) | 0022111 (s=0.5, c=2, R=[2, 3], pair=4, sA=0.478) | 2 | 0022111 | - |
| EX87_J0 | mps7, d=3, chi=5, P=405 | 0000000 (1.0) | 1111111 (s=0.9, c=7, R=all, pair=7, sA=0.900); 0022111 (s=0.5, c=5, R=[2..6], pair=5, sA=0.500) | 5 | 0022111 | (5) |
| S3 | mps7, d=3, chi=6, P=576 | 0000000 (1.0), 1111111 (0.9) | 0222111 (s=0.6, c=3, R=[1, 2, 3], pair=4, sA=0.574); 2222211 (s=0.8, c=5, R=[0..4], pair=5, sA=0.776) | 3 | 0222111 | - |
| S4 | mps7, d=3, chi=6, P=576 | 0000000 (1.0), 1111111 (0.9) | 0222211 (s=0.6, c=4, R=[1..4], pair=5, sA=0.582); 2222221 (s=0.8, c=6, R=[0..5], pair=6, sA=0.788) | 4 | 0222211 | - |
| U3HT | ht4, d=2, chi=6, P=516 | 0010 (1.0), 0100 (0.9), 1001 (0.8) | 1111 (s=0.6, c=3, R=[2, 5, 6], pair=5, sA=0.339) | 3 | 1111 | - |
| U4B8 | binary8, d=2, chi=5, P=355 | 00110100 (1.0), 01001101 (0.8), 10101101 (0.7) | 10110111 (s=0.6, c=4, R=[2, 4, 5, 6], pair=5, sA=0.462) | 4 | 10110111 | - |
| U5B8 | binary8, d=2, chi=5, P=355 | 01001101 (1.0), 10101101 (0.9), 10110111 (0.8) | 00110100 (s=0.6, c=5, R=[0, 2, 4, 5, 6], pair=5, sA=0.528) | 5 | 00110100 | (5) |
| TWO23 | mps7, d=3, chi=5, P=405 | 0000000 (1.0), 1111111 (0.9) | 0022111 (s=0.2, c=2, R=[2, 3], pair=4, sA=0.191); 0000222 (s=0.9, c=3, R=[4, 5, 6], pair=3, sA=0.900) | 2 | 0022111 | - |
| TWO34 | binary8, d=2, chi=5, P=355 | 00110100 (1.0), 10101101 (0.8) | 01001101 (s=0.3, c=3, R=[0, 1, 2], pair=3, sA=0.264); 10110111 (s=0.9, c=4, R=[2, 4, 5, 6], pair=5, sA=0.872) | 3 | 01001101 | - |
| TIE3 | ht4, d=2, chi=5, P=315 | 0100 (1.0), 1011 (0.9) | 0001 (s=0.8, c=3, R=[2, 5, 6], pair=5, sA=0.776); 0010 (s=0.4, c=3, R=[2, 5, 6], pair=5, sA=0.388) | 3 | 0001 | - |
| PS2 | mps6, d=3, chi=5, P=330 | 000000 (1.0), 222222 (0.9) | 110001 (s=0.8, c=6, R=all, pair=6, sA=0.800); 110010 (s=0.8, c=5, R=[0..4], pair=5, sA=0.800) | 5 | 110010 | **4** |
| T92a | mps6, d=3, chi=6, P=468 | 000000 (1.0), 111111 (0.8) | 002020 (s=0.5, c=3, parent 000000); 211211 (s=0.5, c=4, parent 111111) | 3 | (static only) | 3 |
| T92b | mps6, d=3, chi=6, P=468 | 000000 (1.0), 111111 (0.9) | 220002 (0.8, c=6), 220020 (0.8, c=5) (parent 000000: pair-star, mu_2 = 1.13 > 1, l* = 2); 211121 (0.5, c=5, parent 111111) | 5 | (static only) | **4** |
| T92c | mps6, d=3, chi=6, P=468 | 000000 (1.0), 111111 (0.9) | as T92b with pair weights 0.6 (mu_2 = 0.85 < 1: no shortcut) | 5 | (static only) | 5 |
| T92d | mps6, d=3, chi=6, P=468 | 000000 (1.0), 111111 (0.5) | 200020 (0.5, c=5, parent 000000); 221112, 221121 (0.4 each, c=6, 5; parent 111111: pair-star, mu_2 = 0.57 > s_b = 0.5) | 5 | (static only) | **4** |

Coverage of the brief's list: Example 8.7 (EX87_J01, log law); unique shortest routes of depth 3 (S3
mps, U3HT ht), 4 (S4 mps, U4B8 binary) and 5 (EX87_J0 mps, U5B8 binary); two missing patterns of
different depth with the shorter route at the smaller weight (TWO23: 2 vs 3, TWO34: 3 vs 4); a tie in
depth with different weights (TIE3); Corollary 8.6 (0022111 seen from J = {0000000}, depth 5, and from
J' = {0000000, 1111111}, depth 2: EX87_J0 vs EX87_J01); branched trees (U3HT, TIE3 on ht4; U4B8,
U5B8, TWO34 on binary8); a predicted shortcut (PS2). Most cells splice (c < pair): the route depth is
not a distance to any single learned pattern. Theorem 9.2 family: T92a (both parents direct), T92b
(one parent with a pair-star spectral shortcut, order 4 with k* = 5), T92c (its control), T92d (the
shortcut at the weaker parent), and PS2 (one parent with residuals).

## 3. Protocols

**Model.** Whitened inputs, exact population loss L = 1/2 ||W - W*||^2, float64,
`torch.set_num_threads(1)` and BLAS pinned to one thread (threadpoolctl) in every worker.
**Cancellation-free loss** about a reference point S on M* with only active entries:
L - L_J = 1/2 ||Delta||^2 - <R', Delta> + 1/2 sum over J of (s_a - P_a(S))^2,
Delta = W(theta) - W(S) by the exact telescoping sum over cores, R' = W* - W(S), and the gradient
d<E, W(theta)>/d theta with E = Delta - R' held fixed (as `experiments/saddle_dyn/sdyn.py`).

**Static order** (`static.py`). m(t) = min of L - L_J over ||theta - theta_J|| = t, by L-BFGS on the
sphere (12 random starts per t: full Gaussian, non-active only, mostly active; plus continuation from
the previous t), t = 0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025, 0.0175, 0.0125; once with the
perturbation restricted to M* (`AlignedManifold.mask`, "Mstar") and once on all parameters ("full").
Recorded per t: m, the minimiser's weight on active / dormant-M* / off-M* entries, the off-target
fraction of D = W - W(theta_J) and its largest entries. Hessian at theta_J (plain loss, exact):
smallest eigenvalue, number of negative and zero eigenvalues, full and restricted to M*.

**Dynamics** (`escape.py`). Start theta_J + eps xi, four kinds of xi (z iid N(0, 1) from
`numpy.random.default_rng(1000 * kind_id + i)`, the same z at every eps):
- **A**: dormant entries of M* = eps |z| (positive; active entries and everything else at theta_J);
- **A_bal**: dormant entries of M* = eps exactly (one direction; the literal setting of eq. 6.16);
- **B**: dormant entries of M* = eps z (random signs);
- **C**: eps z on every parameter.
Directions i = 0..15 for A, B, C (16 each). Grid: log10 eps = -1, -1.25, ..., -4.5 (15 points, 3.5
decades). Integration: gradient flow, scipy LSODA stepped by hand, rtol 1e-10, atol 1e-15, exact
Hessian as Jacobian; events on solver steps; the escape time refined by bisection with exact
re-integration; local clock re-zeroed when the step falls below 1e-11 of the local time; failed steps
restarted (at most 5); at t = 200 the reference S switches, inside the running solver, to the active
part of the current state (sdyn deviations 3, 5, 6, 7). Trap rule (sdyn deviation 1): stop as
**trapped** when, after t > 1e6 and before any escape, 0 < L - L_J < 1e-30 or every missing amplitude
|P_r| < 1e-45 at 10 consecutive checks a factor 10^0.1 apart. Wall cap 1800 s per run (**unfinished**).
A run stops 200 time units after the first upward crossing P_r = s_r/2 of a missing pattern, or at
10 t_esc + 1000 if none.
For A, A_bal and B the reduced flow on M* (`AlignedManifold.velocity/jacobian`, Theorem 7.7) is
integrated from the same entries with the same driver (atol 1e-30), and the **isolated-route
prediction** t_iso(r) is computed for every missing r: the escape time of the single route
y = gamma prod(b_v), db_v/dt = gamma (s_r - y) prod_{u != v} b_u (eq. 6.14 with the start's own
dormant factors b_v(0), unequal and possibly negative; Prop. 6.2 makes it a 1-D quadrature) up to the
same threshold -s_r y + y^2/2 = -1e-3. For balanced starts t_iso is eq. 6.16 including the
finite-threshold correction; t_616 = the leading term of eq. 6.16 itself is recorded too.

**Recorded per run** (`escape_runs.csv`): outcome (escaped / trapped / unfinished / failed;
floor-flagged), t_esc (first time L < L_J - 1e-3; also -1e-4, -1e-2), the first missing pattern to
cross s_r/2 ("first learned") and its time, the pattern carrying the escape (largest s_r P_r at
t_esc), off-target mass at t_esc (`||W off the target strings||^2 / ||W||^2`), effective depth
(`ttn.routes.effective_depth` on P of the first learned pattern before its event, window P/s in
[1e-4, 1e-2]), forgetting (P_a < s_a/4 for a learned a), the learned set at the end, P_r(0), the
floor index; for A/A_bal/B also t_red, the reduced flow's first learned pattern, t_iso, t_616. For C
(and all kinds) diagnostics at log-spaced checkpoints: `align` + `route_entry_norms` of every missing
pattern, P_r.

**Floor index** (sdyn 5.3): Phi = max over t < t_esc of [int_0^t ||g_err,new|| ds] / ||theta_new(t)||,
g_err = 10 u x (gradient contraction with absolute values), "new" = all entries not active at theta_J,
sampled every 0.02 in log10 t. **Flag if Phi > 0.01** (accumulated roundoff could have moved the new
entries by 1 %, i.e. the signal is within 100x of roundoff); flagged runs are excluded from fits and
counted.

## 4. Estimators and fit windows

- **Static order** at a cell and space: the local slope of log|m| against log t at the smallest pair
  (t = 0.0175 -> 0.0125); also all local slopes. The order is "resolved" if the last two local slopes
  differ by <= 0.1.
- **Per-direction exponent** (c >= 3 or p >= 3): OLS slope gamma_i of log t_esc on log(1/eps) over the
  window **log10 eps in [-4.5, -3.0]** (7 points, 1.5 decades); a direction enters only if it escaped
  at every window point with no floor flag (otherwise: trapped if any window run is trapped, else
  incomplete). Also local exponents between neighbouring eps.
- **Log law** (c = 2 or p = 2): per direction, OLS slope of t_esc against ln(1/eps) over the same
  window, compared with 1/(gamma s_r) (A, A_bal, B) or reported (C).
- **Prefactor**: per run, ratio t_esc / t_iso(r) for the predicted route r (A, A_bal, B), and for A_bal
  t_esc / t_616; per direction the ratio at the smallest window eps.
- **Distribution over directions**: median, quartiles, fraction within 0.3 of the prediction; for C,
  fractions "at prediction" (|gamma_i - (p - 2)| <= 0.3), "shifted" (complete but |gamma_i - (p - 2)| >
  0.3; the shift reported), "trapped", "incomplete".
- **First learned**: per cell, kind and eps, the fraction of escaped runs whose first learned pattern
  is the greedy prediction; for A, A_bal and B also the fraction equal to the per-direction
  isolated-route winner argmin_r t_iso(r).

## 5. Decision rules

- **P1** holds at a cell if the Mstar order = c*_J within 0.2.
- **P2**: full-space order "= c*_J" if within 0.2; **shortcut** if below c*_J - 0.5 (then the
  minimiser is described: weight off M*, off-target fraction, largest entries); else "unresolved".
- **Hessian**: strict saddle (order 2) iff the smallest eigenvalue < -1e-8.
- **P3** holds for a run if |t_esc / t_red - 1| < 1e-4 and the first learned patterns agree; for the
  cell if it holds in every A/A_bal/B run where both escape.
- **P4** (A_bal): supported if |t_esc / t_iso - 1| <= 0.05 and, for c >= 3, |t_esc / t_616 - 1| <= 0.05 at
  the smallest window eps; c = 2: the log-law slope within 5 % of 1/(gamma s_r). For random A and B
  starts the same 5 % on t_esc / t_iso at every window point (the isolated-route generalization).
- **P5**: per cell and kind, SUPPORTED if >= 80 % of the complete directions have |gamma_i - (p - 2)| <=
  0.3, the median is within 0.15, and at least 10 directions are complete; REFUTED if the median is
  off by > 0.5 with >= 10 complete directions; INCONCLUSIVE otherwise (reason stated). p = c*_J for A,
  A_bal, B (on M*, Theorem 8.4(3)); p = the measured static full-space order for C (if the static
  order is unresolved, both c*_J and the measured value are reported).
- **P6**: SUPPORTED at a cell and kind if >= 90 % of the escaped window runs learn the greedy pattern
  first; the eps dependence is reported over the whole grid (TWO23 and TWO34 are expected to switch to
  the larger-weight pattern near the top of the grid, eps ~ 0.1).
- **P7**: the static orders and escape exponents of EX87_J0 and EX87_J01 are compared (5 and 2
  expected).
- **P8**: the formula holds at an instance if the full-space static order equals it within 0.2;
  "holds on the family" if it holds at all five instances.

## 6. Budget

About 4 h of wall clock on this machine (122 usable cores; one single-threaded process per core).
Pilot (10 runs): 8-73 s per run, floor index <= 1.5e-7. Expected 11 cells x 49 directions x 15 eps =
8085 dynamics runs, ~1 h with 118 workers; static 30 tasks (cells x 2 spaces), minutes each. If too
slow, the grid is cut from the expensive end (smallest eps of the deepest cells, then kind C
directions 12-15), and the windows are not moved.

## 7. Deviations from the pre-registration

1. **Solver stall: restarted solvers are BDF; runs without outcome rerun (20:13, numerics only).**
   Three runs of the first sweep hit the 1800 s wall cap (EX87_J0 A seed 13 and U5B8 C seeds 12, 13,
   all at eps = 10^-4.5): 231k-245k steps at a fixed t ~ 1e13-4e14 with L - L_J already negative.
   Cause: after a clock restart (or a failed first step) a fresh scipy LSODA starts in nonstiff
   (Adams) mode and stays pinned at the stability limit of the learned directions (the stall of
   sdyn deviation 7). Fix: every restarted solver is scipy's BDF (always stiff, exact Jacobian, same
   tolerances); the first solver stays LSODA. Check: EX87_J0 A 13, eps = 10^-4.5: escapes at
   3.73327039e14 in 40 s, reduced flow 3.73327076e14 (relative difference 1e-7). Every run that
   ends without an outcome (wall / failed / t_max) is rerun with this driver after the sweep, and
   nothing else (not outcome-selective).
   *Amended 20:40:* the stall kept hitting S4 A/B runs (13 wall-capped runs by 20:39, each costing a
   worker 30 min), so the first sweep was stopped at 3990 finished runs and restarted with the fixed
   driver for the remaining 4161 runs (files record `driver = "bdf_restart"`); the 13 wall-capped
   runs were moved to `unfinished_try1/` and rerun at the same time. Runs that had finished with the
   old driver are kept (a stall costs time, not accuracy; the fixed driver reproduces a run finished by
   the old one, EX87_J0 A seed 13 at eps = 1e-4, whose solver had been restarted: t_esc = 1.18056255e13
   against 1.18056262e13, relative difference 5e-8; reduced flow 1.18056262e13).
   *Amended 21:18 (1b, stall rule):* the restarted sweep showed the same stall without any restart:
   LSODA started on an M* plateau (S4 and PS2, kinds A/B) can stay in nonstiff mode from the first
   step (PS2 A seed 14, eps = 1e-2: 255k steps of ~0.1 at t < 2.5e4, plateau 1.5e5 long; S4 A/B runs
   took 2000 s or hit the cap). Rule added: if 2000 consecutive steps advance t by less than 50 %
   before the escape, the integration continues from the current state with BDF (same tolerances,
   exact Jacobian). Healthy runs never meet the condition, so their results are unchanged. Check: PS2 A
   seed 14, eps = 1e-2: t_esc = 152312.615 in 32 s, reduced flow 152312.621 (4e-8). The restarted
   sweep was stopped with 31 runs still in progress; these and every run without an outcome (wall
   cap) were rerun with this rule (`driver = "bdf_restart+stall"`, column `n_stall_switch`).
2. **Added kind Cp (causal test; 20:05, after the first shifted C direction was seen).** U5B8 C seed 9
   escapes with exponent 4 instead of 3, and the diagnostics show its route entry at leaf node 0
   decaying to zero on the plateau. To test whether the entries outside M* cause this, kind Cp
   uses the same Gaussian direction as C (same seed) projected onto M* (entries outside the mask left
   at theta_J). Cp runs cover the fit window (log10 eps = -4.5 ... -3, 7 points) for all 16 seeds of
   every dynamics cell. Cp is not part of any registered decision; it is reported as a diagnostic.

# Part II: results

Data: 30 static tasks, 8085 escape runs (kinds A, A_bal, B, C) and the Cp diagnostic runs; raw JSON in
`/root/outputs/route-escape/`, tidy CSVs in `experiments/route_escape/results/` (columns:
`experiments/route_escape/README.md`). Tables below are printed by `experiments/route_escape/tables.py`.
Status tags: **proven** (a theorem of the paper; numerics only illustrate it), **empirical**
(measured here), **derived**, **conjecture**.

## 8. Static order (P1, P2, Hessian)

| cell | tree | c_star | order_Mstar | order_full | decision | hess_lam_min | hess_n_neg | minimiser_off_M |
|---|---|---|---|---|---|---|---|---|
| EX87_J01 | mps7 | 2 | 2.000 | 2.000 | = c*_J | -0.478 | 3 | 0.086 |
| EX87_J0 | mps7 | 5 | 5.000 | 5.000 | = c*_J | -1.3e-15 | 0 | 0.836 |
| S3 | mps7 | 3 | 3.000 | 3.000 | = c*_J | -1.3e-15 | 0 | 0.921 |
| S4 | mps7 | 4 | 4.000 | 4.000 | = c*_J | -1.6e-15 | 0 | 0.878 |
| U3HT | ht4 | 3 | 3.000 | 3.000 | = c*_J | -2.1e-15 | 0 | 0.775 |
| U4B8 | binary8 | 4 | 4.000 | 4.000 | = c*_J | -1.4e-15 | 0 | 0.601 |
| U5B8 | binary8 | 5 | 5.000 | 5.000 | = c*_J | -1.7e-15 | 0 | 0.772 |
| TWO23 | mps7 | 2 | 2.000 | 2.000 | = c*_J | -0.191 | 3 | 0.710 |
| TWO34 | binary8 | 3 | 3.000 | 3.000 | = c*_J | -1.4e-15 | 0 | 0.996 |
| TIE3 | ht4 | 3 | 3.000 | 3.000 | = c*_J | -1.6e-15 | 0 | 0.215 |
| PS2 | mps6 | 5 | 5.000 | 4.000 | shortcut | -9.8e-16 | 0 | 0.972 |
| T92a | mps6 | 3 | 3.000 | 3.000 | = c*_J | -1.1e-15 | 0 | 0.827 |
| T92b | mps6 | 5 | 5.000 | 4.000 | shortcut | -1.6e-15 | 0 | 0.997 |
| T92c | mps6 | 5 | 5.000 | 5.000 | = c*_J | -1.6e-15 | 0 | 0.661 |
| T92d | mps6 | 5 | 5.000 | 4.000 | shortcut | -1.2e-15 | 0 | 0.972 |

- **P1 holds in all 15 cells** (empirical check of a proven statement): inside M* the measured order
  is c*_J to within 0.0004 (local slope at t = 0.0175 -> 0.0125; the last two local slopes agree to
  < 0.001 everywhere, so every order is resolved).
- **P2: the full-space order equals c*_J in 12 cells and is lower in 3 (PS2, T92b, T92d: 4.000 against
  c*_J = 5)** (empirical). These are exactly the cells where a parent has a pair-star spectral
  shortcut (Thm 3.2 / Example 4.5 applied to one learned parent). The full-space minimiser there lies
  almost entirely outside M* (97-100 % of its squared norm), and its output change is off-target:
  e.g. PS2 at t = 0.0125: W(min) - W(theta_J) has 110000 (7.8e-5) and 000010, 000001 (6.3e-5 each),
  i.e. a two-core piece "11" on nodes 0-1 plus single-core rotations at nodes 4 and 5, which pair up
  into both residuals 110001 and 110010 (+4.9e-9 each) while the learned 000000 shrinks (-6.7e-9).
  This is the mechanism of the paper's Example 4.5, now at a later (two-pattern) saddle. In T92d the
  same structure sits at the weaker parent 111111 (221111, 111121, 111112). No other shortcut was
  found: at every other cell the full-space minimiser reaches the same order and value by putting the
  route partly or wholly in other spare channels (9-100 % of its squared norm outside the M*
  coordinates, which fix one channel per block); the loss decrease is carried by the missing
  pattern itself (e.g. EX87_J0 at t = 0.0125: m = -2.73e-12 = -s_r x Delta P_r with Delta P_r = 5.46e-12;
  off-target pieces of similar size appear in the minimiser, but they cost only their square).
- **Hessian.** Strict saddle (negative eigenvalue) only where c*_J = 2: EX87_J01 (lambda_min =
  -0.478 = -s_r A_r(J)) and TWO23 (-0.191 = -s_r A_r(J)), 3 negative eigenvalues in the full space (one
  per spare channel on the route's internal edge), 1 inside M*. Everywhere else lambda_min = 0 to
  roundoff (|lambda_min| < 3e-15; gauge and flat directions), including the three shortcut cells:
  their order-4 descent is invisible to the Hessian, as Thm 3.2 allows for l* = 2.

## 9. Theorem 9.2 on its natural family (P8)

| cell | parent | parent_weight | residuals | k_star | l_star | parent_order | formula | c_star | order_full | holds |
|---|---|---|---|---|---|---|---|---|---|---|
| PS2 | 000000 | 1.000 | 110001:0.8 110010:0.8 | 5 | 2.000 | 4 | 4 | 5 | 4.000 | True |
| T92a | 000000 | 1.000 | 002020:0.5 | 3 |  | 3 | 3 | 3 | 3.000 | True |
| T92a | 111111 | 0.800 | 211211:0.5 | 4 |  | 4 | 3 | 3 | 3.000 | True |
| T92b | 000000 | 1.000 | 220002:0.8 220020:0.8 | 5 | 2.000 | 4 | 4 | 5 | 4.000 | True |
| T92b | 111111 | 0.900 | 211121:0.5 | 5 |  | 5 | 4 | 5 | 4.000 | True |
| T92c | 000000 | 1.000 | 220002:0.6 220020:0.6 | 5 |  | 5 | 5 | 5 | 5.000 | True |
| T92c | 111111 | 0.900 | 211121:0.5 | 5 |  | 5 | 5 | 5 | 5.000 | True |
| T92d | 000000 | 1.000 | 200020:0.5 | 5 |  | 5 | 4 | 5 | 4.000 | True |
| T92d | 111111 | 0.500 | 221112:0.4 221121:0.4 | 5 | 2.000 | 4 | 4 | 5 | 4.000 | True |

**The formula holds at all five instances** (empirical): both parents direct (T92a: 3), a pair-star
spectral shortcut at one parent (T92b, PS2: 4 < k* = 5), its control without the shortcut (T92c: 5),
and the shortcut at the weaker parent (T92d: mu_2 = 0.57 > s_b = 0.5, order 4). The shortcut depends on
the parent's own weight, not on the largest one, as the formula says. Caveat: this family is the one
the theorem was built for (parents share no entry); splicing cells (c_J(r) < pair cost) are outside
Assumption 9.1, and there the formula with the nearest parent over-predicts (e.g. EX87_J01: 4 against
the measured 2), as the paper's remark after Thm 9.2 says.

## 10. Starts inside M*: full network = reduced flow = isolated-route formula (P3, P4)

Kinds A (positive dormant entries), A_bal (all dormant entries = eps), B (random signs) and the
diagnostic Cp (Gaussian projected onto M*), every eps, every cell.

| cell | kind | P3_runs | P3_max_rel | P3_first_agree | P3 | iso_frac_within_5pct | iso_ratio_median | iso_winner_frac_window |
|---|---|---|---|---|---|---|---|---|
| EX87_J01 | A | 220.000 | 1.3e-09 | 1.000 | holds | 1.000 | 0.999 | 1.000 |
| EX87_J01 | A_bal | 13.000 | 6.8e-11 | 1.000 | holds | 1.000 | 0.999 | 1.000 |
| EX87_J01 | B | 234.000 | 1.2e-09 | 1.000 | holds | 1.000 | 0.999 | 1.000 |
| EX87_J01 | Cp | 112.000 | 1.2e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| TWO23 | A | 224.000 | 7.1e-10 | 1.000 | holds | 1.000 | 0.999 | 1.000 |
| TWO23 | A_bal | 14.000 | 2.6e-10 | 1.000 | holds | 1.000 | 0.999 | 1.000 |
| TWO23 | B | 237.000 | 9.1e-10 | 1.000 | holds | 1.000 | 0.999 | 1.000 |
| TWO23 | Cp | 112.000 | 1.1e-09 | 1.000 | holds | 1.000 | 0.999 | 1.000 |
| S3 | A | 239.000 | 3.6e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| S3 | A_bal | 15.000 | 2.3e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| S3 | B | 240.000 | 3.4e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| S3 | Cp | 112.000 | 3.6e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U3HT | A | 237.000 | 4.1e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U3HT | A_bal | 15.000 | 1.8e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U3HT | B | 240.000 | 5.8e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U3HT | Cp | 112.000 | 5.8e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| TWO34 | A | 238.000 | 4.8e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| TWO34 | A_bal | 15.000 | 9.3e-10 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| TWO34 | B | 240.000 | 6.3e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| TWO34 | Cp | 112.000 | 2.6e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| TIE3 | A | 238.000 | 3.9e-09 | 0.992 | fails | 0.000 | 0.796 | 0.938 |
| TIE3 | A_bal | 14.000 | 1.6e-09 | 1.000 | holds | 0.000 | 0.804 | 1.000 |
| TIE3 | B | 240.000 | 5.7e-09 | 1.000 | holds | 0.125 | 0.984 | 0.875 |
| TIE3 | Cp | 112.000 | 8.5e-09 | 1.000 | holds | 0.188 | 0.949 | 0.812 |
| S4 | A | 230.000 | 5.2e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| S4 | A_bal | 15.000 | 9.2e-07 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| S4 | B | 232.000 | 5.1e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| S4 | Cp | 112.000 | 5.2e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U4B8 | A | 238.000 | 2.7e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U4B8 | A_bal | 15.000 | 1.2e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U4B8 | B | 238.000 | 2.8e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U4B8 | Cp | 112.000 | 9.9e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| EX87_J0 | A | 240.000 | 9.8e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| EX87_J0 | A_bal | 15.000 | 2.2e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| EX87_J0 | B | 240.000 | 5.0e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| EX87_J0 | Cp | 112.000 | 2.1e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U5B8 | A | 239.000 | 4.5e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U5B8 | A_bal | 15.000 | 4.6e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U5B8 | B | 240.000 | 2.9e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| U5B8 | Cp | 112.000 | 6.0e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| PS2 | A | 240.000 | 4.3e-08 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| PS2 | A_bal | 15.000 | 3.4e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| PS2 | B | 240.000 | 9.4e-09 | 1.000 | holds | 1.000 | 1.000 | 1.000 |
| PS2 | Cp | 112.000 | 1.5e-07 | 1.000 | holds | 1.000 | 1.000 | 1.000 |

- **P3 holds** (empirical illustration of the proven Thm 7.7): in all 4 kinds x 11 cells the full
  network's escape time equals the reduced flow's to at most 9.2e-7 relative (median 3e-10), 6597
  runs; the full network stays exactly on M* (off-target mass 0 at every escape). The single "fails"
  entry (TIE3 A) is one run at eps = 10^-1.25 where the two tied patterns cross s/2 within one
  solver step of the reduced flow, which then reports the lower pattern index (bookkeeping, not
  dynamics: the escape times agree to 2e-10 and the full run's two crossings are 0.008 apart).
- **P4 (eq. 6.16 with its prefactor) holds wherever the route is isolated** (empirical): in 10 of 11
  cells every window run of A, A_bal, B and Cp has |t_esc / t_iso - 1| <= 0.05 (median ratio 0.9986-
  1.0000; the largest deviation is 1.0 %, PS2 Cp at eps = 1e-3, where the second missing pattern's
  route shares entries; depth-2 cells up to 0.6 %). For balanced starts the
  literal eq. 6.16 holds:

| cell | depth | sA | smallest_eps | t_esc | t_iso | ratio_iso | t_616 | ratio_616 | log_slope | log_slope_pred |
|---|---|---|---|---|---|---|---|---|---|---|
| EX87_J01 | 2 | 0.478 | 3.2e-05 | 15.225 | 15.232 | 1.000 | 21.680 | 0.702 | 2.092 | 2.092 |
| TWO23 | 2 | 0.191 | 3.2e-05 | 40.517 | 40.562 | 0.999 | 54.201 | 0.748 | 5.231 | 5.231 |
| S3 | 3 | 0.574 | 3.2e-05 | 55124.201 | 55124.568 | 1.000 | 55139.028 | 1.000 |  |  |
| U3HT | 3 | 0.339 | 3.2e-05 | 93138.677 | 93139.825 | 1.000 | 93160.338 | 1.000 |  |  |
| TWO34 | 3 | 0.264 | 3.2e-05 | 1.2e+05 | 1.2e+05 | 1.000 | 1.2e+05 | 1.000 |  |  |
| TIE3 | 3 | 0.776 | 3.2e-05 | 32770.261 | 40724.661 | 0.805 | 40736.488 | 0.804 |  |  |
| S4 | 4 | 0.582 | 3.2e-05 | 8.6e+08 | 8.6e+08 | 1.000 | 8.6e+08 | 1.000 |  |  |
| U4B8 | 4 | 0.462 | 3.2e-05 | 1.1e+09 | 1.1e+09 | 1.000 | 1.1e+09 | 1.000 |  |  |
| EX87_J0 | 5 | 0.500 | 3.2e-05 | 2.1e+13 | 2.1e+13 | 1.000 | 2.1e+13 | 1.000 |  |  |
| U5B8 | 5 | 0.528 | 3.2e-05 | 2.0e+13 | 2.0e+13 | 1.000 | 2.0e+13 | 1.000 |  |  |
| PS2 | 5 | 0.800 | 3.2e-05 | 1.3e+13 | 1.3e+13 | 1.000 | 1.3e+13 | 1.000 |  |  |

  For depth >= 3 the leading term of eq. 6.16, eps^-(c-2) / ((c-2) s_r A_r(J)), is within 0.03 % of
  the measured time at eps = 10^-4.5 (ratio_616 = 1.000). For depth 2 the slope of t_esc against
  ln(1/eps) equals 1/(s_r A_r(J)) to 6 digits (EX87_J01: 2.092 vs 2.092; TWO23: 5.231 vs 5.231); the
  leading term of eq. 6.16 lacks the additive O(1) constant (ratio 0.70-0.75), which the isolated-
  route integral includes (ratio 1.000). Random signs (B) do not change the exponent: a negative
  starting product only lengthens the escape (the smallest factor has to cross zero), exactly as the
  isolated-route integral with the start's own factors predicts (B: ratio within 5 % in every window
  run of 10 cells).
- **P4 fails in TIE3** (empirical): the two tied missing patterns 0001 and 0010 share 2 of their 3
  dormant entries, so their routes are not isolated (Assumption C.1(3) fails). With positive entries
  (A, A_bal) the shared entries are pushed by both routes and the escape comes earlier than the
  isolated-route formula (window ratio 0.54-0.91, median 0.80; balanced 0.80); with random signs (B)
  the two routes can help or fight (ratio 0.51-3.2, median 0.98). The reduced flow of Thm 7.7 still predicts every run exactly.

## 11. Escape exponents (P5)

Per direction: OLS slope gamma of log t_esc on log(1/eps) over eps = 10^-4.5 ... 10^-3.

| cell | kind | gamma_pred | n_complete | n_at_pred | n_shifted | n_trapped | n_incomplete | gamma [q1, q3] | frac_within_03 | P5 | greedy_frac_window | P6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EX87_J01 | A | 0.000 | 16 | 16 | 0 | 0 | 0 | 0.176 [0.163, 0.193] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| EX87_J01 | A_bal | 0.000 | 1 | 1 | 0 | 0 | 0 | 0.185 [0.185, 0.185] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| EX87_J01 | B | 0.000 | 16 | 16 | 0 | 0 | 0 | 0.163 [0.150, 0.170] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| EX87_J01 | C | 0.000 | 16 | 16 | 0 | 0 | 0 | 0.190 [0.183, 0.201] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| EX87_J01 | Cp | 0.000 | 16 | 16 | 0 | 0 | 0 | 0.163 [0.148, 0.191] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| TWO23 | A | 0.000 | 16 | 16 | 0 | 0 | 0 | 0.162 [0.151, 0.176] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| TWO23 | A_bal | 0.000 | 1 | 1 | 0 | 0 | 0 | 0.170 [0.170, 0.170] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| TWO23 | B | 0.000 | 16 | 16 | 0 | 0 | 0 | 0.151 [0.140, 0.157] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| TWO23 | C | 0.000 | 16 | 16 | 0 | 0 | 0 | 0.174 [0.168, 0.183] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| TWO23 | Cp | 0.000 | 16 | 16 | 0 | 0 | 0 | 0.151 [0.138, 0.175] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| S3 | A | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.002 [1.001, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S3 | A_bal | 1.000 | 1 | 1 | 0 | 0 | 0 | 1.002 [1.002, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S3 | B | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.001 [1.001, 1.001] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S3 | C | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.003 [1.003, 1.004] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S3 | Cp | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.002 [1.001, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U3HT | A | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.002 [1.001, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U3HT | A_bal | 1.000 | 1 | 1 | 0 | 0 | 0 | 1.002 [1.002, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U3HT | B | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.001 [1.001, 1.001] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U3HT | C | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.003 [1.002, 1.004] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U3HT | Cp | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.001 [1.000, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TWO34 | A | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.001 [1.001, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TWO34 | A_bal | 1.000 | 1 | 1 | 0 | 0 | 0 | 1.002 [1.002, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TWO34 | B | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.001 [1.001, 1.001] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TWO34 | C | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.002 [1.001, 1.003] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TWO34 | Cp | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.001 [1.000, 1.001] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TIE3 | A | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.002 [1.002, 1.003] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TIE3 | A_bal | 1.000 | 1 | 1 | 0 | 0 | 0 | 1.003 [1.003, 1.003] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TIE3 | B | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.001 [1.001, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TIE3 | C | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.003 [1.003, 1.004] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| TIE3 | Cp | 1.000 | 16 | 16 | 0 | 0 | 0 | 1.001 [1.001, 1.002] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S4 | A | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S4 | A_bal | 2.000 | 1 | 1 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S4 | B | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S4 | C | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [2.000, 2.001] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| S4 | Cp | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U4B8 | A | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U4B8 | A_bal | 2.000 | 1 | 1 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U4B8 | B | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U4B8 | C | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U4B8 | Cp | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [2.000, 2.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| EX87_J0 | A | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| EX87_J0 | A_bal | 3.000 | 1 | 1 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| EX87_J0 | B | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| EX87_J0 | C | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [2.999, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| EX87_J0 | Cp | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U5B8 | A | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U5B8 | A_bal | 3.000 | 1 | 1 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U5B8 | B | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| U5B8 | C | 3.000 | 7 | 7 | 0 | 0 | 9 | 3.001 [3.000, 3.002] | 1.000 | INCONCLUSIVE | 1.000 | SUPPORTED |
| U5B8 | Cp | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| PS2 | A | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| PS2 | A_bal | 3.000 | 1 | 1 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| PS2 | B | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |
| PS2 | C | 2.000 | 16 | 16 | 0 | 0 | 0 | 2.000 [1.999, 2.001] | 1.000 | SUPPORTED | 0.384 | NOT SUPPORTED |
| PS2 | Cp | 3.000 | 16 | 16 | 0 | 0 | 0 | 3.000 [3.000, 3.000] | 1.000 | SUPPORTED | 1.000 | SUPPORTED |

- **Inside M* (A, A_bal, B, Cp): every one of the 441 directions at cells with c*_J >= 3 has gamma
  within 0.005 of c*_J - 2** (the depth-2 log law is in section 10). Status: empirical, consistent with the proven
  order (Thm 8.4(3)) and the derived eq. 6.16.
- **Gaussian starts (C): 167 of 176 directions are complete; at the 9 cells with p >= 3 all complete
  ones are within 0.05 of p - 2** (135 of 135; 131 within 0.01), p the static full-space order. The 9 incomplete ones are all in U5B8 (depth 5); each has at least one
  floor-flagged window run (eps <= 10^-4) and is excluded by the registered rule, so U5B8 C is
  INCONCLUSIVE by rule. Not registered, for information: fitting all 7 window runs of these 9
  directions gives 2.996-3.034 for 8 of them and **4.004 for direction 9**, which is at 4.0 already on
  its unflagged points (t_esc = 3.0e10, 3.0e12 at eps = 1e-3, 10^-3.5; section 12).
- **PS2 C: gamma = 2.000 for all 16 directions**, the shortcut order 4 measured statically, not the
  route depth 5 (gamma = 3). The Gaussian start finds the spectral shortcut; starts inside M* (A, B,
  Cp) cannot reach it and escape with gamma = 3.000. This is the dynamical counterpart of Remark 8.5.
- **Depth-2 cells: P5 is INCONCLUSIVE by my own registered rule, which was mis-specified for a log
  law**: t = a ln(1/eps) + b has a log-log slope a/t ~ 0.16 in this window, never within 0.15 of 0.
  The log law itself is confirmed by the registered P4 test (slope of t against ln(1/eps) = 1/(s_r
  A_r(J)) to 6 digits for A, A_bal, B; Gaussian starts: ratio 0.996-1.005 to the same prediction).
- No direction of any kind was trapped and none had a shifted exponent apart from U5B8 C direction 9.

## 12. Why a Gaussian direction is shifted: route entries drained by off-target clusters

U5B8 C direction 9 escapes with exponent 4 at every eps from 1e-3 to 10^-4.5 (t_esc = 3.0e10, 3.0e12,
3.05e14, 3.05e16; the last two floor-flagged), one power more than c*_J - 2 = 3 (`fig7_shift_mechanism`).
Following the route entries with `align` + `route_entry_norms` (`escape_runs.csv`, columns
`route_*`; raw checkpoints in the JSON):

1. In every Gaussian (C) run of the deep cells the route entries at the two *ends* of the routing
   region are drained during the plateau, and only there (EX87_J0: nodes 2 and 6; S4: 1 and 4; U4B8:
   2 and 4; U5B8: 0 and 4; PS2: 4 and 0). Median lowest/initial ratio at eps = 1e-4: U4B8 0.94, S4
   0.78, EX87_J0 0.73, PS2 0.68, U5B8 0.43 (column `route_depletion`). Starts inside M* show no
   drain.
2. The drain runs on the time scale eps^-2: the time at which the U5B8 leaf entry halves times eps^2
   is constant per direction over 1.5 decades of eps (direction 0: 1.75, 1.35, 1.15, 1.41 at eps =
   1e-3 ... 10^-4.5; direction 2: 0.33, 0.24, 0.27, 0.24; direction 9: 1.74, 1.41, 1.20, 1.27).
   For a depth-c route the escape takes eps^-(c-2), so the drain finishes first when c >= 5,
   competes when c = 4, and is too slow to matter when c = 3.
3. In most directions the drain stops at a finite level (direction 2 at 0.007 of its start at eps =
   10^-4.5) and only the prefactor changes (direction 2 keeps exponent 3.000). In direction 9 it goes
   to zero (below 1e-4 of its start at eps = 10^-4.5, numerically 0 at eps = 1e-4): the route through leaf node 0 is dead, P_r stays ~eps^6, and the escape needs one
   more small factor: exponent 4.
4. **Causal tests.** (a) Cp, the same Gaussian direction projected onto M*: all 16 U5B8 directions
   (including 9) escape with gamma = 3.000, no drain. So the entries outside M* cause the shift.
   (b) `diag_partner.py`: in direction 9, set to zero only the 30 "partner" entries (of 355
   parameters): the entries of the region node next to each region end whose leg toward the end is a
   spare channel and whose other legs are active channels. The drain disappears (the leaf entry stays
   at 1.33 eps throughout) and the escape returns to eps^-3: t_esc = 2.56e8, 8.10e9, 2.56e11 at eps =
   1e-3, 10^-3.5, 1e-4 (with the partners: 3.0e10, 3.0e12, 3.05e14).

Reading (derived from the tests above; the selection rule is a conjecture): a partner entry and the
route entry at the region end together form a two-core off-target piece (a learned pattern with two
cores changed). It costs loss at fourth order in eps, so both entries shrink on the time scale
eps^-2, coupled by the bond Gram conservation law of their shared edge (Prop. 6.3). Whether the route
entry or the partner is the one driven to zero depends on the direction (which side of that
conserved balance is larger); when the route entry loses, the route is cut and the escape exponent
rises by one. This is the mechanism that `notes/saddle_dyn.md` D7 conjectured for its control
cells ("route entry slaved by a decaying two-core cluster"); here it is observed directly and
removed by intervention. It does not contradict the theory: the order is a statement about the
fastest possible descent, and a shifted direction descends more slowly. It does limit the escape-time
law to typical directions when c >= 5. In this experiment 1 of 48 Gaussian directions at depth-5
cells is shifted (U5B8 9; EX87_J0 and PS2: none).

## 13. Which pattern is learned first (P6, greedy rule)

| cell | kind | -1.0 | -1.25 | -1.5 | -1.75 | -2.0 | -2.25 | -2.5 | -2.75 | -3.0 | -3.25 | -3.5 | -3.75 | -4.0 | -4.25 | -4.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PS2 | A | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PS2 | B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PS2 | C | 0.800 | 0.733 | 0.500 | 0.500 | 0.438 | 0.438 | 0.438 | 0.438 | 0.438 | 0.375 | 0.375 | 0.375 | 0.375 | 0.375 | 0.375 |
| TIE3 | A | 0.750 | 0.812 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TIE3 | B | 0.812 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TIE3 | C | 0.688 | 0.812 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TWO23 | A | 0.688 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TWO23 | B | 0.750 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TWO23 | C | 0.625 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TWO34 | A | 0.812 | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TWO34 | B | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TWO34 | C | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

- **In the fit window (eps <= 1e-3) the greedy pattern is learned first in 100 % of the escaped runs of
  every cell and kind, except PS2 C (38 %)** (empirical). This includes the cells built to separate
  the rules: TWO23 (depth-2 route at weight 0.2 beats a depth-3 route at weight 0.9), TWO34 (depth 3 at
  0.3 beats depth 4 at 0.9; the depth-4 route is a splice that the pairwise rule would cost 5), and
  the tie TIE3 (equal depth 3: the larger s_r A_r(J) = 0.78 wins over 0.39).
- **Large eps: the rule fails, as it should asymptotically only** (empirical). At eps = 0.1 the
  larger-weight deeper route wins in 12-38 % of the runs of TWO23 and 6-19 % of TWO34, and the smaller
  tie partner in 19-31 % of TIE3; by eps = 10^-1.5 (TWO23) and 10^-1.75 (TWO34, TIE3) the greedy pattern
  wins every run.
- With unequal random dormant entries the per-direction winner is predicted by the isolated-route
  times (argmin_r t_iso): 100 % of the window runs in 10 cells, 81-94 % in TIE3 (shared entries,
  section 10), column `iso_winner`.
- **PS2 C: the greedy rule fails, because the escape is the spectral shortcut, not a route**
  (empirical). The Gaussian start builds the pair-star pieces of parent 000000 (an off-target two-core
  piece 110000 and single-core rotations at nodes 4 and 5; off-target mass 0.0087 at escape, against
  < 1e-7 in every other cell), which produce both residuals 110001 (depth 6) and 110010 (depth 5)
  together: they cross s/2 within 1e-9 t_esc of each other (median), so "first" is a coin flip (38 %
  greedy). The parent 000000 is forgotten in 228 of 238 runs and the network ends at {222222, 110001,
  110010}: a swap of one learned pattern for two, which no route rule describes. Inside M* (A, B, Cp)
  PS2 learns 110010 first in every run, as predicted.

## 14. Corollary 8.6 (P7)

0022111 seen from J = {0000000} (EX87_J0, depth 5) and from J' = {0000000, 1111111} (EX87_J01, depth
2): static order 5.000 -> 2.000 in both spaces; strict saddle only at J' (Hessian eigenvalue -0.478);
escape time from balanced starts at eps = 1e-4: 6.6667e11 (= eps^-3 / (3 x 0.5), eq. 6.16) -> 12.8
(log law with slope 2.092 = 1/(0.5 x 0.956)); exponents 3.000 (all kinds) -> log law (all kinds).
**Holds** (empirical illustration of a proven statement; `fig5_cor86`).

## 15. Effective depth and off-target mass

- Effective depth (`ttn.routes.effective_depth` on the first learned amplitude, window P/s in
  [1e-4, 1e-2]; window runs, kinds A, A_bal, B, C) has median c + 0.002 (depth 2), c + 0.02-0.10
  (depth 3), c + 0.13-0.16 (depth 4), c + 0.10-0.17 (depth 5). The estimator reads slightly high and its
  bias grows with c (2 - 2/d is flat in d), but it identifies the route depth in every cell, including
  the splices (EX87_J01: 2.006, pairwise cost 4; S3: 3.048, pairwise 4; U3HT 3.105, pairwise 5).
  Exceptions: PS2 C (3.79-3.80, the shortcut: order 4) and the shifted U5B8 direction (up to 5.6).
- Off-target mass at escape: exactly 0 inside M*; for Gaussian starts median 2e-11-1.4e-9, max 1e-7,
  except PS2 C (0.0087, the shortcut). Escapes from Gaussian starts are on-target route escapes
  wherever there is no shortcut.

## 16. Run inventory, numerics, float64 floor

| cell | kind | runs_escaped | runs_floor_flagged | runs_trapped | runs_unfinished | runs_failed |
|---|---|---|---|---|---|---|
| EX87_J01 | A | 220 | 0 | 0 | 0 | 0 |
| EX87_J01 | A_bal | 13 | 0 | 0 | 0 | 0 |
| EX87_J01 | B | 234 | 0 | 0 | 0 | 0 |
| EX87_J01 | C | 240 | 0 | 0 | 0 | 0 |
| EX87_J01 | Cp | 112 | 0 | 0 | 0 | 0 |
| TWO23 | A | 224 | 0 | 0 | 0 | 0 |
| TWO23 | A_bal | 14 | 0 | 0 | 0 | 0 |
| TWO23 | B | 237 | 0 | 0 | 0 | 0 |
| TWO23 | C | 240 | 0 | 0 | 0 | 0 |
| TWO23 | Cp | 112 | 0 | 0 | 0 | 0 |
| S3 | A | 239 | 0 | 0 | 0 | 0 |
| S3 | A_bal | 15 | 0 | 0 | 0 | 0 |
| S3 | B | 240 | 0 | 0 | 0 | 0 |
| S3 | C | 240 | 0 | 0 | 0 | 0 |
| S3 | Cp | 112 | 0 | 0 | 0 | 0 |
| U3HT | A | 237 | 0 | 0 | 0 | 0 |
| U3HT | A_bal | 15 | 0 | 0 | 0 | 0 |
| U3HT | B | 240 | 0 | 0 | 0 | 0 |
| U3HT | C | 240 | 0 | 0 | 0 | 0 |
| U3HT | Cp | 112 | 0 | 0 | 0 | 0 |
| TWO34 | A | 238 | 0 | 0 | 0 | 0 |
| TWO34 | A_bal | 15 | 0 | 0 | 0 | 0 |
| TWO34 | B | 240 | 0 | 0 | 0 | 0 |
| TWO34 | C | 240 | 0 | 0 | 0 | 0 |
| TWO34 | Cp | 112 | 0 | 0 | 0 | 0 |
| TIE3 | A | 238 | 0 | 0 | 0 | 0 |
| TIE3 | A_bal | 14 | 0 | 0 | 0 | 0 |
| TIE3 | B | 240 | 0 | 0 | 0 | 0 |
| TIE3 | C | 240 | 0 | 0 | 0 | 0 |
| TIE3 | Cp | 112 | 0 | 0 | 0 | 0 |
| S4 | A | 240 | 0 | 0 | 0 | 0 |
| S4 | A_bal | 15 | 0 | 0 | 0 | 0 |
| S4 | B | 240 | 0 | 0 | 0 | 0 |
| S4 | C | 240 | 0 | 0 | 0 | 0 |
| S4 | Cp | 112 | 0 | 0 | 0 | 0 |
| U4B8 | A | 240 | 0 | 0 | 0 | 0 |
| U4B8 | A_bal | 15 | 0 | 0 | 0 | 0 |
| U4B8 | B | 240 | 0 | 0 | 0 | 0 |
| U4B8 | C | 240 | 0 | 0 | 0 | 0 |
| U4B8 | Cp | 112 | 0 | 0 | 0 | 0 |
| EX87_J0 | A | 240 | 0 | 0 | 0 | 0 |
| EX87_J0 | A_bal | 15 | 0 | 0 | 0 | 0 |
| EX87_J0 | B | 240 | 0 | 0 | 0 | 0 |
| EX87_J0 | C | 240 | 0 | 0 | 0 | 0 |
| EX87_J0 | Cp | 112 | 0 | 0 | 0 | 0 |
| U5B8 | A | 240 | 0 | 0 | 0 | 0 |
| U5B8 | A_bal | 15 | 0 | 0 | 0 | 0 |
| U5B8 | B | 240 | 0 | 0 | 0 | 0 |
| U5B8 | C | 227 | 13 | 0 | 0 | 0 |
| U5B8 | Cp | 112 | 0 | 0 | 0 | 0 |
| PS2 | A | 240 | 0 | 0 | 0 | 0 |
| PS2 | A_bal | 15 | 0 | 0 | 0 | 0 |
| PS2 | B | 240 | 0 | 0 | 0 | 0 |
| PS2 | C | 238 | 0 | 0 | 0 | 0 |
| PS2 | Cp | 112 | 0 | 0 | 0 | 0 |

- 9315 escape runs: 9245 escaped, 13 floor-flagged (all U5B8 C: 9 at eps = 10^-4.5, 2 at 10^-4.25,
  and direction 9 also at 1e-4 and 10^-3.75; Phi up to 35 in direction 9, 0.012-0.06 otherwise), 57 "below_at_start" (eps >= 10^-1.5: L - L_J < -1e-3
  already at t = 0, no escape time; outside the window), 0 trapped, 0 failed. Unfinished after the
  reruns: PS2 C direction 1 at eps = 0.1 and 0.056 (see section 7 / the report; outside the window).
- Deviations 1 / 1b (solver stalls) cost time only; reruns reproduce finished runs to 5e-8.
- Floor index: at most 2.8e-3 in every run outside U5B8 C. In U5B8 C it grows as eps^-3 (largest
  value per eps: 3.4e-5 at 1e-3, 3.3e-3 at 10^-3.5, 0.34 at 1e-4, 35 at 10^-4.5), dominated by the slow
  direction 9. The bound is pessimistic (it accumulates the roundoff of damped directions as if it
  were coherent), and the flagged runs of the other 8 directions lie on their eps^-3 lines (fits
  including them: 2.996-3.034), but they are excluded as registered.

## 17. Claims and status

| # | claim | status |
|---|---|---|
| E1 | At all 15 cells the order inside M* equals c*_J (2, 3, 4 or 5) to 4e-4 (Thm 8.4(3)) | empirical illustration of a proven result |
| E2 | In the full space the order equals c*_J at 12 cells and is 4 < c*_J = 5 at PS2, T92b, T92d, exactly where a learned parent has a pair-star spectral shortcut; the minimiser is the Example 4.5 construction at that parent (off-target two-core piece + single-core rotations, parent amplitude decreasing) | empirical |
| E3 | Theorem 9.2's formula min over parents of min(k*_a, 2 l*_a) gives the full-space order at all 5 instances of its natural family, including a shortcut at the weaker parent | empirical |
| E4 | theta_J is a strict saddle (negative Hessian eigenvalue -s_r A_r(J)) only when c*_J = 2; the order-4 shortcuts have a positive-semidefinite Hessian | empirical |
| E5 | A start inside M* follows the reduced flow of Thm 7.7: escape times equal to <= 9.2e-7 relative in 6597 runs, output exactly on target | empirical illustration of a proven result |
| E6 | For an isolated route, eq. 6.16 holds with its prefactor: balanced starts match eps^-(c-2) / ((c-2) s_r A_r(J)) to 0.03 % at eps = 10^-4.5 (c = 3, 4, 5) and the log-law slope 1/(s_r A_r(J)) to 6 digits (c = 2); random positive or signed starts match the isolated-route integral with their own factors to <= 1 % | empirical (eq. 6.16 is derived) |
| E7 | When two missing routes share dormant entries (TIE3), the isolated-route law misses by up to a factor 2-3 (positive starts: earlier, ratio 0.54-0.91), while the reduced flow stays exact; the exponent is unchanged | empirical |
| E8 | Escape exponents: all 441 directions inside M* at c*_J >= 3 within 0.005 of c*_J - 2; all 135 complete Gaussian directions at p >= 3 within 0.05 of p - 2 (p = static full-space order) | empirical |
| E9 | At the shortcut cell PS2, Gaussian starts escape as eps^-2 (order 4), starts inside M* as eps^-3 (order 5 = c*_J): the shortcut is found by generic training | empirical |
| E10 | Greedy rule: the shortest route (ties: larger s_r A_r(J)) is learned first in 100 % of window runs at every cell and kind except the shortcut cell PS2 with Gaussian starts (38 %); at eps ~ 0.1 the larger-weight deeper route or tie partner wins in 6-38 % of runs | empirical |
| E11 | At the shortcut, both pair-star residuals are learned together and the parent is forgotten (228/238 runs): the network swaps one learned pattern for two | empirical |
| E12 | Corollary 8.6 at EX87: depth 5 -> 2, order 5 -> 2, escape eps^-3 -> log(1/eps) | empirical illustration of a proven result |
| E13 | In Gaussian starts at depth >= 4 the route entries at the ends of the routing region are drained on the time scale eps^-2 by off-target two-core pieces formed with "partner" entries outside M*; when a route entry is drained to zero the escape exponent rises by one (U5B8 direction 9: 4 instead of 3); projecting onto M* or zeroing the 30 partner entries removes the drain and restores eps^-3 | empirical (causal tests) |
| E14 | Which side of the drain loses (route entry or partner) is set by the conserved bond Gram balance of their edge | conjecture |
| E15 | Effective depth from the amplitude growth identifies c_J(r) in every cell (median within 0.2, reads high by up to 0.17 at c = 5), including splices whose pairwise cost is larger | empirical |

## 18. Open and next

- The selection rule of E14 is not tested: compute the conserved Gram difference on the region-end
  edge at the start of each Gaussian direction and check that it predicts which directions are cut.
  A theory of the drain would also say whether the fraction of cut directions grows with depth.
- Shared dormant entries (TIE3) need the coupled reduced flow, not the isolated-route law; a
  two-route closed form (shared factor + two private factors) would give the tie-break correctly.
- The registered P5 rule cannot be met by a log law (section 11); for depth 2 the log-law slope test
  is the right one.
- PS2 C after the swap: what the next plateau is (000000 must be relearned from {222222, 110001,
  110010}) was not followed.
'''
