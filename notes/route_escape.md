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

(none yet)
