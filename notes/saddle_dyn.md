# saddle_dyn: does the escape time from a learned rank-one saddle follow the saddle order?

Agent `saddle-dyn`, branch `agent/saddle-dyn`. Experiment folder: `experiments/saddle_dyn/`.
Raw outputs: `/root/outputs/saddle_dyn/`.

**Part I (sections 1-7) is the pre-registration, written and pushed before any run.** It is not
edited after the runs start; deviations go to section 8 with the reason. Results against the rules
are in Part II.

## 1. Background and what is tested

`notes/theory_saddle_order.md` v2 proves the saddle order at a learned rank-one saddle theta_S
(order = k* if lambda_max(S) < s_q; order = 2 l* < k* when the generalized piggyback operator S
exceeds s_q at a level l* < k*/2; Theorems 1, 1', 2, Corollary 0). Its section 6 derives (reduced
model, not proven) the dynamical consequence S9: the escape from theta_S takes
t_esc ~ eps^-(p-2) from a perturbation of size eps, p the saddle order, and ~ sigma^-(p-2) from small
init, with a crossover to the route law sigma^-(k*-2) above a scale b (b ~ 0.155 for the binary8
wrong-order saddle, ~0.27 for HT4 1111 at weight 1.5, aligned amplitudes, leading order). So far
S9 rests on 3-5 seeds of one teacher (MPS6 pair-star).

Notation: gamma = escape-time exponent, t_esc ∝ eps^-gamma. gamma_ord = p - 2 (theorem's order),
gamma_route = k* - 2 (route law).

- **H-A (saddle start).** From theta_S + eps xi, t_esc ∝ eps^-gamma_ord asymptotically (eps -> 0).
  On control teachers (lambda_max(S) < s_q, p = k*) the route law gamma_route = k* - 2 holds.
- **H-B (small init).** Runs from small fan-in init that visit theta_S have plateau lengths
  dt ∝ sigma^-gamma_ord below the crossover, and the route law above it (where a crossover is
  predicted inside the accessible sigma range: binary8).

## 2. Cells

Pattern teachers, standard site basis, d = 2, q = 0...0 learned at theta_S, residuals as listed
(weight in brackets). theta_S balanced: G_v = s_q^(1/K) e_q ⊗ e_0 ⊗ ... ⊗ e_0. Orders from
`experiments/theory_checks/saddle_order/gp_operator.py` (computed for this note; lambda_max of S
on the level-l low sets in brackets).

| cell | tree, chi | s_q | residuals | k* | lambda_max(S \| L_l) | order p | gamma_ord | gamma_route | role |
|---|---|---|---|---|---|---|---|---|---|
| PS08 | mps6, 2 | 1 | 110001 (0.8), 110010 (0.8) | 5 | l=2: 1.131 | 4 | 2 | 3 | test (pair-star) |
| PS06 | mps6, 2 | 1 | 110001 (0.6), 110010 (0.6) | 5 | l=2: 0.849 | 5 | 3 | 3 | control of PS08 |
| W6 | mps6, 2 | 1 | 110001 (1.5) | 6 | l=2: 1.5 | 4 | 2 | 4 | test (wrong order) |
| C1 | mps6, 2 | 1 | 110001 (0.7) | 6 | l=2: 0.7 | 6 | 4 | 4 | control of W6 |
| C2 | mps5, 2 | 1 | 10101 (1.5) | 5 | l=2: 0 | 5 | 3 | 3 | control: wrong order, no low split (Cor. 2b) |
| HT4W | ht4, 2 | 1 | 1111 (1.5) | 7 | l=3: 1.5 | 6 | 4 | 5 | test (wrong order, two polymers) |
| C3 | ht4, 2 | 1 | 1111 (0.5) | 7 | l=3: 0.5 | 7 | 5 | 5 | control of HT4W |
| B8W | binary8, 4 | 0.8 | 11111111 (1.0) | 7 | l=3: 1.0 | 6 | 4 | 5 | test (the `order` agent's wrong-order odeco saddle, relabelled 0<->1) |
| HT4L | ht4, 2 | 0.5 | 1111 (1.0) | 7 | l=3: 1.0 | 6 | 4 | 5 | test (locality teacher 0000 (1), 1111 (0.5), smaller learned first) |
| HT4S | ht4, 3, d = 4 | 0.85 | 1111 (1.0), 2222 (0.7) | 7 | l=3: 1.0 | 6 | 4 | 5 | test (s2s_core "close" odeco teacher (1, 0.85, 0.7), second component learned first) |
| B8W05 | binary8, 4 | 0.5 | 11111111 (1.0) | 7 | l=3: 1.0 | 6 | 4 | 5 | test (`order` binary8 at rho = 0.5) |

B8W is the `order` agent's teacher W* = e_{0^8} + 0.8 e_{1^8} (odeco, standard basis, chi = 4,
`experiments/order/configs/h1b_saddle_b8.yaml`) with the site basis relabelled so that the learned
(smaller) component is q = 0^8: physically identical. Crossover estimates (note section 7 item 5,
b = 4 kappa / (5 s_r), kappa = (s_r - s_q) s_q^((K-6)/K)): B8W 0.155, HT4W 0.267, HT4L 0.362,
HT4S 0.117 (ignoring its second residual), B8W05 0.362.

HT4L, HT4S and B8W05 were added after the lead's message of 2026-10-10 ~15:05 UTC (the
`saddle_hopper` agent found 30 order-6 saddles among the team's past runs, all wrong-first on
ht4 / ht8 / binary8; `notes/saddle_hopper.md` section 9), **before this registration was committed
and before any run**. HT4S is the teacher of the one s2s_core seed with a polymer escape at every
sigma (gap exponent 4.03). ht8 (K = 15) is not included: its order-6 saddle has the same local
structure as HT4L.

## 3. Protocols

**Relabelling.** HT4S and B8W/B8W05 come from odeco teachers in the standard basis; the site basis is
relabelled so that the learned component is e_0 at every site (an exact symmetry of the whitened
loss).

**Dynamics (all runs).** Gradient flow on the whitened population loss, float64,
`torch.set_num_threads(1)` per process, scipy LSODA (`solve_ivp(method="LSODA")`), rtol = 1e-10,
atol = 1e-15, events located by root finding on the dense output (durations resolved by
construction). Contraction and topology from the verifier's independent `vttn.Net`
(`experiments/theory_checks/saddle_order_verify/vttn.py`). Near theta_S the gradient is computed
**cancellation-free**: E = W - W* = Delta - R' with Delta = W(G) - W(S) by the exact telescoping sum
sum_j W(S_1..S_{j-1}, G_j - S_j, G_{j+1}..G_K) (no O(1) terms cancel), R' = W* - W(S) computed once.
(I do not use `ttn.dynamics.integrate_gf`: its loss is not cancellation-free at a learned saddle,
which costs a factor ~eps in the float64 floor; see 5.3. Its LSODA backend is the same scipy routine.)

**H-A (saddle start).** theta_0 = theta_S + eps xi_i, xi_i with iid N(0, 1) entries on every
parameter (old and new), generator `numpy.random.default_rng(7000 + i)`, the same xi_i at every eps
(per-seed scaling, as the note's `gf_escape.py`). B8W and B8W05 instead use the `order` agent's protocol: a
fan-in perturbation sigma Z_v / sqrt(F_v) (Z iid N(0,1), `torch.Generator` seed 7000 + i), so that
sigma matches their sigma and the crossover estimate. Seeds i = 0..11 (12 directions per cell).
- **Escape:** t_esc = first t with L - L_S < -1e-3 (the note's threshold; secondary thresholds
  -1e-4 and -1e-2 recorded). The run continues to 3 t_esc + 100 to record what happens next
  (W[q] there: "q forgotten" if W[q] < 0.1 s_q).
- **Grids:** eps = 10^(-1, -1.25, ..., -5) (17 values), truncated at the small end where the float64
  floor or cost bites (estimated in 5.3): C1, HT4W, HT4L, HT4S to 1e-4, C3 to 10^-3.5. B8W, B8W05:
  sigma = 10^(-0.5, -0.75, ..., -2.75) (0.316 -> 0.00178, 10 values) plus sigma = 0.15 (overlap with
  the `order` runs).
- **Stop:** t_max = 1e30 or 3600 s wall time per run ("unfinished", reported with the range reached).

**H-B (small init).** theta_0 = fan-in init sigma Z_v / sqrt(F_v), the same Z per seed at every sigma.
PS08/PS06: the verifier's `gf_origin.py` init (`torch.Generator().manual_seed(12345 + seed)`, vttn
shapes), so seeds 0-11 reproduce their runs. B8W: `torch.Generator` seed 5000 + seed. The flow is
integrated with the plain loss until the visit time t_v (below); there all bonds are rotated by an
exact orthogonal gauge transformation (Householder, mapping each bond's q-message to e_0: gradient
flow commutes with orthogonal gauge maps) and the run continues cancellation-free with reference
S = the rank-one old part of the rotated cores.
- **Visit:** t_v = first time L <= L_S + 1e-4 (L is nonincreasing under gradient flow). The run
  visits theta_S if at t_v W[q] > s_q/2 and every residual has |W[r]| < s_r/2 ("q-first"); otherwise
  it is classified by what is learned (e.g. "residual-first") and stopped.
- **Plateau length:** dt = t_esc - t_q, t_q = time W[q] crosses s_q/2 (the verifier's definition),
  t_esc = first time after t_v with L < L_S - 1e-3.
- **Grids and seeds:** PS08, PS06: sigma = 10^(-1, -1.25, ..., -3.5) (11 values); PS08 seeds 0-39,
  PS06 seeds 0-11. B8W: sigma = 10^(-0.5, ..., -2.5) (9 values), seeds 0-29 (the `order` agent found
  41% wrong order at sigma = 0.01, chi = 4). Rule: if fewer than 10 seeds of PS08 or B8W visit
  theta_S, add seeds in blocks of 20 until 10 visit or 100 seeds have run.
- **Optional (only if the budget allows after everything above):** W6 and HT4W from small init.

## 4. Fit windows (fixed here) and estimators

Windows are closed intervals of the grid, at the small-eps end, each spanning >= 1.5 decades:

| cell | H-A window (eps; B8W: sigma) | points | H-B window (sigma) |
|---|---|---|---|
| PS08, PS06, W6, C2 | [1e-5, 1e-3] | 9 | PS08, PS06: [10^-3.5, 1e-2] (7 points) |
| C1, HT4W, HT4L, HT4S | [1e-4, 10^-2.5] | 7 | - |
| C3 | [10^-3.5, 1e-2] | 7 | - |
| B8W | [10^-2.75, 10^-1.25] = [0.00178, 0.0562] | 7 | [10^-2.5, 10^-1] = [0.00316, 0.1] (7 points) |
| B8W05 | as B8W | 7 | - |

- **Per-seed exponent** gamma_i: OLS of log t_esc (H-B: log dt) on log eps over the window. A seed
  enters only if it escapes (H-B: visits and escapes) at every window point with no floor flag
  (5.3); others are reported separately, never dropped silently.
- **Cell estimate:** mean gamma ± SE over seeds (and the median, and the spread). Also the local
  exponents log(t(eps_j+1)/t(eps_j)) / log(eps_j/eps_j+1) at the two smallest-eps pairs (mean ± SE).
- **Competing laws, fairly:** for gamma_fix in {gamma_ord, gamma_route} (and gamma = 4 for PS06,
  the verifier's anomalous value), fit log t = c_i - gamma_fix log eps per seed (c_i free) over the
  same window points; report the summed RSS (log space, equal weights) of each law. For cells where
  gamma_ord = gamma_route only the free fit is reported.
- **Crossover (all test cells; descriptive):** per seed fit
  t = 1 / (eps^gamma_ord / A + eps^gamma_route / B) on the whole grid (rates of the two mechanisms
  add), crossover eps_c = B / A, mean ± SE; compared with b for B8W/HT4W (no conversion between
  sigma (fan-in, random direction) and b (aligned amplitudes) is derived, so only agreement within a
  factor 3 is called "consistent").
- **Escape type** (at t_esc): split-type if max over the low-split partial patterns p of |W[p]| >=
  5 max_r |W[r]| (e.g. W[110000] for PS08 / W6; W[1100], W[0011] for HT4W; the two leaf-pair halves
  of 1^8 for B8W/B8W05; the leaf-pair halves of 1111 for HT4L/HT4S), route-type if <= 1, mixed otherwise. Counts per cell; fits are on all seeds, and
  repeated by type if both types occur.

## 5. Decision rules

### 5.1 Test cells (p < k*: PS08, W6, HT4W, B8W, HT4L, HT4S, B8W05)

- **SUPPORTED** if all of: (a) |mean gamma - gamma_ord| <= max(0.15, 2 SE); (b) >= 80% of the
  fitted seeds have |gamma_i - gamma_ord| <= 0.3; (c) RSS(gamma_ord) < RSS(gamma_route) / 10;
  (d) >= 10 fitted seeds.
- **REFUTED** if |mean gamma - gamma_ord| > 0.5 with SE <= 0.15 and >= 10 fitted seeds.
- **INCONCLUSIVE** otherwise (stated why: too few seeds, scatter, window not asymptotic as shown by
  the local exponents still drifting by > 0.1 between the two smallest pairs, floor).

### 5.2 Control cells (p = k*: PS06, C1, C2, C3)

Same rules with gamma_pred = k* - 2 and, in (c), the alternative gamma_ord - 1 = k* - 3 (a
spurious reduced exponent) and, for PS06, also 4 (the anomalous value); the control is SUPPORTED if
(a), (b), (d) hold and RSS(k* - 2) < RSS(alternative) / 10 for each alternative. Seeds that never
escape (dead routes / traps, as seen before for PS06) are counted and reported, not fitted; a
control with fewer than 10 escaping seeds is INCONCLUSIVE on the exponent.

### 5.3 Float64 floor and accuracy

- **Floor index** (per run, after the switch to the cancellation-free form): an upper bound on the
  roundoff in the velocity of the new (dormant) entries, g_err = 10 u C(|E| + |Delta|_abs + |R'|, |G|)
  (u = 2^-53, C(A, |G|) = the gradient contraction with absolute values, |Delta|_abs the telescoping
  sum with absolute values), accumulated: Phi = max_t [int_0^t ||g_err,new|| ds] / ||theta_new(t)||,
  sampled at the solver's output times up to t_esc. **Flag if Phi > 0.01** (accumulated roundoff could
  have moved the dormant entries by 1%); flagged runs are excluded from fits and counted.
  A priori (worst-case coherent roundoff) the cancellation-free form has Phi ~ u eps^(3-k) for a
  k-core route (the plain loss: u eps^(2-k)), which motivated the grid truncations in section 3.
- **Tolerance check:** seeds 0-2 of every H-A cell at the largest, middle and smallest window points
  rerun at rtol 1e-12, atol 1e-17; required |delta t_esc| / t_esc < 1e-3, otherwise the cell's
  exponents are reported as "tolerance-limited".
- **Plain-loss cross-check:** seeds 0-2 at the two smallest eps of each H-A cell rerun with the plain
  loss (E = W - W*); the difference in t_esc is reported (informational: where the plain form fails).
- **Conservation:** max drift of the bond conserved quantities Q_e reported per run.

### 5.4 H-B

Rules 5.1/5.2 on the visiting seeds' dt in the H-B window (PS08: gamma_ord = 2, route 3; PS06:
control, 3, alternatives 2 and 4; B8W: 4, route 5). In addition, for B8W only ("route law above the
crossover"): the mean local exponent over the pairs with both sigma >= 0.1 (from the H-B grid,
0.316 -> 0.178 -> 0.1) is compared with 5 and 4: "route law above crossover: SUPPORTED" if within 0.4
of 5 and > 0.4 away from 4; "not seen" if within 0.4 of 4; else INCONCLUSIVE. The same comparison is
made on the B8W and B8W05 H-A grids (pairs among sigma = 0.316, 0.178, 0.15, 0.1). Every run is reported:
counts of q-first (visiting), residual-first, unresolved plateaus (dt < 10), trapped, unfinished,
floor-flagged.

## 6. Status tags

Claims in Part II carry: proven (the orders, from the note), derived (the reduced-model exponent),
empirical (everything measured here), conjecture.

## 7. Budget

<= ~4 h wall clock of compute on this machine (192 cores; one thread per process). If a cell is too
slow, the grid is cut from the small end and the range reached is reported; windows are then
**not** moved (a cell whose window cannot be completed is INCONCLUSIVE for that reason).

## 8. Deviations from the pre-registration

1. **Trap stop rule (added before the sweep, after a timing test).** A test run (PS06, seed 0,
   eps = 1e-2) never escapes: its route entries die (W[r] - 0 down to 1e-49 ... 1e-57) and
   0 < L - L_S ~ 1e-32, and LSODA then crawls (t ~ 1e22 after 40 s, far from t_max = 1e30), so every
   such run would have cost the full 3600 s wall cap. Rule: a run is stopped as **trapped** when
   0 < L - L_S < 1e-30 at 10 consecutive checks spaced by a factor 10^0.1 in t (one decade) with
   t > 1e6. Rationale: an escape needs L - L_S < 0, and a dormant structure of size ~1e-15 in W at
   such times is below the float64 floor (library rule: signal < 1e-15 |W|); before an escape
   L - L_S passes through (0, 1e-30) in one step, not for a decade. Trapped runs are counted and
   reported with their final residual amplitudes, never fitted. Not applied in H-B phase 1.
   *Extended (15:55, same reason):* a run is also stopped as trapped when every residual amplitude
   |Delta[r]| = |W[r] - W_S[r]| stays below 1e-45 at those 10 checks (routes dead by 30+ orders of
   magnitude; observed: 1e-49 ... 1e-304); such an escape could only be seeded by roundoff.
2. **Event root finding.** One H-B run crashed in scipy's event solver (brentq got equal signs at
   the bracket ends because LSODA's dense output does not exactly reproduce the step values). The
   solver now falls back to the first sign change on a 200-point grid of the step (refined by
   brentq). *Superseded by item 3.*
3. **Integrator driver replaced; first sweep discarded (15:40-15:55).** The crash in item 2 had a
   deeper cause: scipy's LSODA dense output (used by `solve_ivp` for event location and by my state
   sampling) is not reliable here: at step times it differed from the actual step values by 2-4 %
   (PS06 H-B, seed 1, during the first escape), and it can miss a crossing that the step values
   show. The first sweep (15:21-15:38, ~200 H-A and ~350 H-B runs) used it for event times, escape
   amplitudes and the H-B hand-off state, so it was stopped and set aside (not deleted:
   `/root/outputs/saddle_dyn_try1_badevents/`, logs in `agent-runs/saddle-dyn/logs/try1/`). The new
   driver steps scipy's `LSODA` class directly, detects events on the step values, and refines each
   crossing by bisection with exact re-integrations from the last step (time resolved to 1e-9
   relative); every state used (escape amplitudes, trajectories, the H-B hand-off) is an integration
   state, never interpolated. Check: PS08 seed 0, eps = 1e-2: t_esc = 37086.4189 (rtol 1e-12 rerun:
   37086.4189, relative difference 8e-10); H-B PS08 seed 7, sigma = 0.01: dt = 211.885 (verifier
   211.9). Same integrator settings (LSODA, rtol 1e-10, atol 1e-15, exact Hessian as Jacobian).
4. **H-B re-gauge after relaxation (15:55).** The registration re-gauges at the visit time t_v
   (L = L_S + 1e-4). There the learned amplitude is not yet converged (W[q] = 0.989 in a test run),
   so the reference S was 1 % off the saddle, leaving a constant ~6e-5 in L - L_S and an absolute
   roundoff 2^-66 = 1.4e-20 in it (the cancellation the reference is meant to remove). Now the run
   continues with the plain loss for 50 time units after t_v (old directions relax at O(1) rates),
   and the re-gauge and the reference are taken there; escapes inside those 50 units are recorded
   (they are unresolved plateaus, dt < 60).
5. **Local clock (15:55; numerics only).** Escapes are fast transitions at late absolute times
   (H-B first escapes at t ~ 1e13-1e15, where the float64 spacing of t is 0.004-0.1, comparable to
   the step sizes), and LSODA failed there ("tout too close to t") in ~5 % of the runs. The driver now
   re-zeroes the solver clock (the ODE is autonomous; absolute t = offset + local t) whenever the
   step falls below 1e-7 of the local time. Runs that had finished with the previous version are
   kept (a run that hit the resolution limit failed outright, it did not return a wrong number;
   C2 seed 7, eps = 1e-5: t_esc identical to 1e-14 in both versions).
6. **H-A re-reference after relaxation; restart first step (16:15; numerics only; H-A rerun).**
   (a) A perturbed run settles on a nearby point of the saddle manifold (old amplitudes and bond
   directions shifted by O(eps)), not on theta_S, so with S = theta_S the telescoping terms at the
   q entry stay O(eps) (4.5e-4 in HT4W, eps = 1e-4) and dominate the roundoff of the dormant
   velocities; the floor index then flagged most small-eps runs of the HT4 cells. As in H-B, the
   run now relaxes for 200 time units (cancellation-free about theta_S), then is re-gauged
   (orthogonal, exact) and re-referenced to the rank-one old part reached; the roundoff bound on the
   dormant velocities drops ~3000x (HT4W seed 0, eps = 1e-4: 1.7e-22 -> 5.9e-26), and the
   escape time from that state is unchanged (5.295052e17 with either reference). The floor index
   is computed from the re-reference on. (b) Solver restarts (item 5) used LSODA's automatic first
   step, which is huge on a flat plateau, so the restarted (nonstiff) solver failed with istate -5
   (repeated corrector failures) against the O(1) eigenvalues of the learned directions; restarts
   now start with a step of 0.01 and failed steps are retried (at most 5 per run, recorded).
   The H-A runs of the second attempt (15:55-16:13, ~700 runs, about theta_S) are set aside in
   `/root/outputs/saddle_dyn_HA_try2_thetaref/` and the whole H-A grid is rerun with this code.
   (c) The floor index used the trapezoid rule on steps log-spaced in t only, so its last interval
   spanned most of the plateau and multiplied the roundoff bound at the escape point (O(1)
   amplitudes, ~1e10 times the plateau value) by it (HT4W seed 0, eps = 1e-4: Phi = 3.9, against
   ~1e-5 with the escape resolved). The steps are now log-spaced in both t and t_esc - t.
   For one code version, H-B is rerun as well (its second attempt is in
   `/root/outputs/saddle_dyn_HB_try2_quadrature/`). Final sweeps started 16:21.
7. **Runs without outcome rerun with an in-solver reference switch (18:04; numerics only).** In the
   final sweeps 88 runs (H-A, H-B and checks) ended at the 3600 s wall cap or with a solver failure.
   Two causes: the machine was oversubscribed (each worker inherited a 64-thread BLAS pool; load
   ~250 on 192 cores), and, more importantly, a stall: the fresh LSODA started on the plateau after the
   re-reference of item 6 can stay in nonstiff (Adams) mode with its step pinned at the stability
   limit of the learned directions (PS08 seed 7, eps = 1e-4: h = 0.093 constant, t = 1.4e3 after
   15000 steps), because nothing on the plateau triggers its stiffness switch. Fix: no new solver at
   the plateau. H-A: one solver from theta_S + eps xi; at t = 200 the reference becomes the rank-one
   old part of the current state (index-0 entries; no re-gauge is needed, the O(eps) gauge shift sits
   in new entries). H-B: the orthogonal re-gauge is applied at t_v (fast dynamics), and the reference
   switches to the old part of the converged state 50 units later, in the same solver. The 88 runs
   (moved to `/root/outputs/saddle_dyn_unfinished_try3/`) are rerun with this driver, same tolerances
   and cap, BLAS pinned to one thread; the ~2700 runs that had finished keep their results (a stall
   costs time, not accuracy; the rerun of previously stalled runs reproduces finished values: PS08 seed
   0, eps = 1e-2, t_esc to 3e-10; H-B PS08 seed 7, dt = 211.8845). Files record `driver = "switch"`.
   Not outcome-selective: every run without an outcome was rerun, and nothing else.
   *Addendum (18:36):* the same stall also hit runs whose solver had been restarted by the local-clock
   rule of item 5 (threshold: step < 1e-7 of the local time, which fired on transient step drops on
   plateaus, e.g. right after the reference switch; a restarted LSODA starts nonstiff). The threshold
   is now 1e-11 (float64 spacing is 2e-16 t, so resolution is still protected; the H-B first escapes
   need it at ~1e-14). The ~45 runs still without outcome at 18:36 were rerun with this (PS08 seed 9,
   eps = 10^-4.75: escape at 2.6331e10 in 85 s, after a 1-hour stall before).

# Part II: results

Final data: H-A 1836 runs, H-B 842 runs, checks 165 runs (`experiments/saddle_dyn/results/*.csv`; raw JSON in
`/root/outputs/saddle_dyn/`). Tables below are printed by `experiments/saddle_dyn/tables.py` from the CSVs
of `analyze.py`. Figures: `results/ha_escape.png` (t_esc vs eps, every run, coloured by escape type),
`results/ha_local.png` (local exponents), `results/hb_dt.png`, `results/hb_local.png`. Dynamics: gradient
flow, LSODA rtol 1e-10, atol 1e-15 (section 3, with deviations 1-7 of section 8, all numerical).

## 9. Summary against the registered rules

**H-A (saddle start).** The theorem's order sets the escape exponent, to three digits, in every test cell
whose window lies below the crossover: PS08 (order 4: gamma = 2.001 ± 0.001, 12 seeds), W6 (order 4 with
k* = 6: 2.001 ± 0.001), HT4W (order 6: 3.996 ± 0.003), HT4L (3.997 ± 0.003) and binary8 at rho = 0.5
(B8W05: 3.998 ± 0.007): all **SUPPORTED** (every seed within 0.05 of p - 2; the route law is worse by an
RSS factor of 1e3-2e5). Two test cells do not show the reduced law in their window: **HT4S is REFUTED by
the registered rule** (gamma = 4.72 ± 0.14 over 11 complete seeds: 3 at 4.0, 8 at 5.0, one incomplete seed at 6; the route law fits better, RSS 29 against 75) and **B8W (rho = 0.8)
is INCONCLUSIVE** (4.48 ± 0.15; 7 seeds at 4.0-4.2, 5 at 5.0-5.2). The registered rule cannot tell the
reason; the per-seed data (9.2-9.4) say that in both cells the window is not below the crossover for
every seed (the route law, or the reduced law shifted by one power, competes there), not that the order
is wrong. **Controls (p = k*):** all four are **INCONCLUSIVE** for the registered reason "fewer than 10
seeds escape at every window point": many seeds are trapped (dead routes: PS06 31 of 107 window runs,
C1 42/74, C2 39/108, C3 49/81). The seeds that escape show the route law k* - 2 exactly (PS06: 4 of 5 complete
seeds at 3.000; C1: 3 at 4.0; C2: 3 of 5 at 3.00) or the route law **plus exactly one** (PS06, C2: 4.00;
C3: 6.0) or plus two (C1: ~6), never a smaller exponent.

**H-B (small init).** Pair-star: **SUPPORTED** (16 of 40 seeds visit theta_S; all 16 complete,
gamma = 2.002 ± 0.001, route law sigma^-3 worse by RSS 148 vs 0.0027). Binary8 wrong order (rho = 0.8):
**INCONCLUSIVE** (13 visiting seeds of 30, gamma = 4.31 ± 0.12; 9 seeds converge to 4.00-4.16, seed 1 is at 4.55 and falling, 3 stay at
5.0); its registered side test "route law above the crossover" is **SUPPORTED** (local exponent 4.87 for
sigma >= 0.1). PS06 control: **INCONCLUSIVE** (only 5 of 12 seeds visit theta_S; 7 learn the residual pair
first, which corrects the verifier's "12/12 q-first", see 9.6).

**Answer to the brief.** Where the theorem's order is below k* and the perturbation is small enough, the
escape time follows eps^-(p-2) (and sigma^-(p-2) from small init) with no measurable deviation: 5 of 7
test cells from saddle starts and the pair-star from small init, 76 seed fits in total, all within 0.05
of p - 2. The route law k* - 2 is the exponent at larger eps (binary8, sigma >= 0.1) and remains it, for some
seeds, down to the smallest eps reached in two cells (HT4S, binary8 at rho = 0.8): the crossover is seed-
and teacher-dependent, and the leading-order estimate b does not locate it (HT4S: b = 0.12, but
the route law persists to eps = 1e-4). On the controls the route law holds where the route survives, and
the seed-to-seed spread is not scatter but integer shifts of the exponent (+1, in C1 +2) and traps.

## 9.1 H-A decision table

| cell | p, k* | pred. gamma (alt.) | fitted seeds | gamma mean ± SE | median [min, max] | within 0.3 | RSS pred / alt | local exp. (2 smallest pairs) | decision |
|---|---|---|---|---|---|---|---|---|---|
| PS08 | 4, 5 | 2 (3) | 12 | 2.001 ± 0.001 | 2.000 [2.00, 2.01] | 1.00 | 0.00575 / 238 | 2.000 ± 0.000, 2.000 ± 0.000 | SUPPORTED |
| W6 | 4, 6 | 2 (4) | 12 | 2.001 ± 0.001 | 2.000 [2.00, 2.01] | 1.00 | 0.0044 / 954 | 2.000 ± 0.000, 2.000 ± 0.000 | SUPPORTED |
| HT4W | 6, 7 | 4 (5) | 12 | 3.996 ± 0.003 | 3.999 [3.97, 4.01] | 1.00 | 0.0136 / 112 | 3.999 ± 0.001, 3.999 ± 0.000 | SUPPORTED |
| HT4L | 6, 7 | 4 (5) | 12 | 3.997 ± 0.003 | 3.998 [3.98, 4.01] | 1.00 | 0.0114 / 112 | 3.999 ± 0.001, 4.000 ± 0.000 | SUPPORTED |
| HT4S | 6, 7 | 4 (5) | 11 | 4.723 ± 0.141 | 4.995 [3.97, 5.02] | 0.27 | 74.6 / 29.2 | 4.703 ± 0.149, 4.713 ± 0.143 | REFUTED |
| B8W | 6, 7 | 4 (5) | 12 | 4.479 ± 0.147 | 4.187 [3.97, 5.22] | 0.58 | 52.2 / 57 | 4.428 ± 0.147, 4.417 ± 0.145 | INCONCLUSIVE (fails: mean, seeds within 0.3, RSS vs alternative) |
| B8W05 | 6, 7 | 4 (5) | 12 | 3.998 ± 0.007 | 3.990 [3.97, 4.05] | 1.00 | 0.0931 / 112 | 3.997 ± 0.003, 3.998 ± 0.002 | SUPPORTED |
| PS06 | 5, 5 | 3 (2, 4) | 5 | 3.200 ± 0.200 | 3.000 [3.00, 4.00] | 0.80 | 19.9 / 159, 79.6 | 3.333 ± 0.211, 3.200 ± 0.200 | INCONCLUSIVE (fails: RSS vs alternative, >=10 seeds) |
| C1 | 6, 6 | 4 (3) | 3 | 3.967 ± 0.038 | 4.001 [3.89, 4.01] | 1.00 | 0.378 / 26.4 | 3.986 ± 0.016, 3.992 ± 0.009 | INCONCLUSIVE (fails: >=10 seeds) |
| C2 | 5, 5 | 3 (2) | 5 | 3.396 ± 0.242 | 3.002 [3.00, 4.00] | 0.60 | 38.9 / 217 | 3.624 ± 0.183, 3.400 ± 0.245 | INCONCLUSIVE (fails: seeds within 0.3, RSS vs alternative, >=10 seeds) |
| C3 | 7, 7 | 5 (4) | 1 | 6.000 | 6.000 [6.00, 6.00] | 0.00 | 9.28 / 37.1 | 6.645 ± 0.645, 6.000 | INCONCLUSIVE (fails: mean, seeds within 0.3, RSS vs alternative, >=10 seeds) |

Run inventory (every run; "trapped" = deviation 1; "failed" = LSODA failure after 5 retries, all at
t > 1e17 in effectively dead states; runs missing from a cell's grid total (156 or 204 per cell, 132 for
binary8) were still unfinished when the compute budget ended, see 9.7):

| cell | runs | escaped | trapped | failed | wall | floor-flagged | window runs: escaped / trapped / failed | window escapes: split / mixed / route | q forgotten after escape |
|---|---|---|---|---|---|---|---|---|---|
| PS08 | 204 | 204 | 0 | 0 | 0 | 0 | 108 / 0 / 0 | 108 / 0 / 0 | 1.00 |
| W6 | 204 | 204 | 0 | 0 | 0 | 0 | 108 / 0 / 0 | 108 / 0 / 0 | 1.00 |
| HT4W | 156 | 153 | 3 | 0 | 0 | 0 | 84 / 0 / 0 | 84 / 0 / 0 | 0.71 |
| HT4L | 156 | 154 | 2 | 0 | 0 | 0 | 84 / 0 / 0 | 84 / 0 / 0 | 0.79 |
| HT4S | 154 | 154 | 0 | 0 | 0 | 0 | 82 / 0 / 0 | 57 / 1 / 24 | 0.00 |
| B8W | 132 | 132 | 0 | 0 | 0 | 0 | 84 / 0 / 0 | 73 / 9 / 2 | 0.00 |
| B8W05 | 132 | 132 | 0 | 0 | 0 | 0 | 84 / 0 / 0 | 84 / 0 / 0 | 0.00 |
| PS06 | 198 | 154 | 44 | 0 | 0 | 3 | 76 / 31 / 0 | 0 / 0 / 76 | 0.00 |
| C1 | 133 | 74 | 58 | 1 | 0 | 14 | 32 / 42 / 0 | 0 / 0 / 32 | 0.00 |
| C2 | 204 | 133 | 71 | 0 | 0 | 0 | 69 / 39 / 0 | 0 / 0 / 69 | 0.00 |
| C3 | 129 | 53 | 75 | 1 | 0 | 10 | 31 / 49 / 1 | 0 / 0 / 31 | 0.00 |

Notes: "within 0.3" = fraction of fitted seeds with |gamma_i - prediction| <= 0.3; RSS = summed squared
log residuals of the fixed-exponent fits (prediction / alternative: route law for test cells; k* - 3 (and
4 for PS06) for controls); local exponents = mean ± SE over seeds at the second-smallest and smallest pair
of the window.

## 9.2 Test cells below the crossover: PS08, W6, HT4W, HT4L, B8W05

- Every window run escapes, none is floor-flagged (largest Phi 2e-5), and every window escape is split-type
  (W[p] >= 5 max |W[r]|; PS08/W6: W[110000] = 0.085 at the threshold; HT4/binary8: both leaf-pair halves).
- The local exponents are flat at p - 2 over the whole window (`results/ha_local.png`): 2.000 ± 0.000
  (PS08, W6), 3.999 (HT4W, HT4L), 3.997 (B8W05) at the smallest pairs, and already within ~0.1 of p - 2 at
  eps = 1e-2 (MPS6) or sigma = 0.06 (B8W05). The two-mechanism fit finds no crossover inside the grid
  (fitted eps_c above 1 in 3 of 4 cells; B8W05: 10^-0.2).
- W6 is the sharpest test: order 4 against k* = 6 (route law eps^-4 vs eps^-2): the route law is worse by an
  RSS factor 2e5.
- After the escape q is forgotten: W[q] < 0.1 s_q at 3 t_esc + 100 in 100 % of the window runs of PS08 and
  W6 and in 71-79 % of HT4W/HT4L (the rest still on the way down); never in B8W05 (q kept).

## 9.3 HT4S: REFUTED by the rule, above the crossover in fact

HT4S (the s2s_core "close" odeco teacher: s_q = 0.85, residuals 1111 (1.0) and 2222 (0.7), ht4, d = 4,
chi = 3) has the smallest gap s_r - s_q = 0.15 of all cells. Per seed (complete fits): seeds 3, 8 at
3.97-3.98 (order-6 law, split escapes); seed 10 at 4.04, crossing from 5 to 4 inside the window (route-type
escapes at eps >= 10^-3, split-type below); seeds 1, 2, 11 at 5.00 with route-type escapes (the route law of
1111); seeds 0, 4, 5, 7, 9 at 4.95-5.02 with split-type escapes; seed 6 (incomplete) at 6.0 (split). The
local exponents of the other seeds are flat over the window (1.5 decades); the two-mechanism fit puts the
crossover at eps_c ~ 1e-9 (median), far below the grid. So in this teacher the order-6 law
is seen in 3 of 12 seeds at eps <= 3e-3; the others follow the route law or the split mechanism shifted by
one power (as in 9.5). The registered verdict stands (REFUTED: |4.72 - 4| > 0.5, SE 0.14, 11 seeds); its
reading is that the prediction is asymptotic and the asymptotic range of this teacher is not reached at
eps >= 1e-4, and that the leading-order crossover b = 0.117 is wrong here by orders of magnitude.

## 9.4 Binary8 wrong order (rho = 0.8): the crossover, seed by seed

Saddle starts (fan-in perturbation as in the `order` experiment, sigma = 0.316 ... 0.00178, plus 0.15): the
median local exponent falls from 5.2 at sigma ~ 0.3 to 4.05 at the smallest sigma (`results/ha_local.png`).
Seeds 1, 2, 4, 5, 6, 9, 11 converge to 4.00-4.05 (order 6); seeds 0, 3, 7, 8, 10 stay at 4.97-5.01 down to
sigma = 0.00178 (route law or shifted split law; 73 split, 9 mixed, 2 route escapes in the window).
Registered side test "route law above the crossover" (pairs with sigma >= 0.1): local exponent 5.10 ±
0.05 (36 pairs): **SUPPORTED**; the `order` agent's 5.3 ± 0.3 at sigma = 0.15-0.35 is reproduced
(the runs at sigma = 0.15: median t_esc 5.5e4 over 12 seeds, theirs 4.5e4 over 6 other directions, with the event a_1 = 1/2 instead of L - L_S = -1e-3).
For rho = 0.5 (B8W05, larger gap) every seed is at 4.00 below sigma ~ 0.06, and the side test is
INCONCLUSIVE (4.48 ± 0.11: already between the laws at sigma >= 0.1). The crossover moves to larger sigma
when the gap s_r - s_q grows, as the leading-order b does (0.155 -> 0.362), but b is not a quantitative
predictor of where a given seed crosses.

## 9.5 Controls: route law, integer shifts and traps

Per seed (`results/ha_runs.csv`, `results/ha_local_exponents.csv`), the controls split into three groups
with sharp boundaries:

| cell (k* - 2) | route law exactly | route law + 1 | + 2 or more | trapped at every eps |
|---|---|---|---|---|
| PS06 (3) | seeds 4, 6, 7, 9 (3.00) | 1, 2, 8, 10, 11 (4.00; trapped at the smallest eps) | 5 (5.8, then trapped) | 0, 3 |
| C2 (3) | 4, 6, 7 (3.00) | 1, 2, 8, 9, 10 (4.00) | - | 0, 3, 5, 11 |
| C1 (4) | 1, 6, 8 (4.0) | - | 2, 7, 10, 11 (6.0-6.6; then trapped; escapes floor-flagged) | 0, 3, 4, 5, 9 |
| C3 (5) | none | 2, 5, 7, 8 (6.0) | 10 (6.6-7.3) | 0, 3, 6, 9, 11 (1, 4: escape only at eps >= 0.06) |

All control escapes are route-type (the residual itself grows; no split pattern). The shifted seeds are
not in transit (flat local exponents over 1-2 decades) and their escape direction is the same route as in
the unshifted seeds. The shift is a property of the direction xi: the same seed is shifted at every eps,
and shifted seeds tend to become trapped at smaller eps. In the H-B control runs the dormant state at
theta_S is exactly theta_S + sigma xi' (every route entry scales as sigma^1.00, `diag/ps06_route_entries_seed*.log`),
so the shift is not in the initial amplitudes but in the dynamics along xi'. Conjecture (not tested): a
route entry that also belongs to a decaying two-core cluster is driven towards zero on the faster cluster
time scale (eps^-2) and slaved at O(eps^2); the route product then carries one more factor eps, and the
escape one more power; if the entry is driven to zero, the route is dead (trap). The shifted escapes are
slower than the route law, never faster, so they do not contradict the order (a lower bound on how fast
L can fall), but they show that "t_esc ~ eps^-(p-2)" (S9) holds for typical, not all, directions.
Some shifted escapes in C1, C3 and PS06 (t_esc 1e18-1e21) are floor-flagged (Phi up to 27) and excluded.

## 9.6 H-B: from small initialization

Every run is counted (`results/hb_runs.csv`). Classification at the visit time t_v (L = L_S + 1e-4):
"escaped" = q learned first, plateau, escape; "residual_first" = a residual crossed s_r/2 first;
"no_visit" (B8W) = the larger component was learned first (the correct order: it never visits the
wrong saddle); "wall" = 3600 s cap (the machine was oversubscribed, see 9.7).

| cell | seeds | q-first (visit theta_S) | never visit | fitted (complete window) | gamma mean ± SE | median [min, max] | prediction | decision |
|---|---|---|---|---|---|---|---|---|
| PS08 | 40 | 16 (at every sigma) | 24 (residual pair first, at every sigma) | 16 | 2.002 ± 0.001 | 2.000 [2.00, 2.01] | 2 (route 3) | **SUPPORTED** |
| PS06 (control) | 12 | 5 (seed 1 trapped) | 7 (residual pair first) | 4 | 3.70 ± 0.24 | 3.90 [3.00, 4.00] | 3 | INCONCLUSIVE (4 seeds) |
| B8W | 30 | 13 | 17 (larger component first) | 13 | 4.31 ± 0.12 | 4.07 [4.00, 5.03] | 4 (route 5) | INCONCLUSIVE (mean 0.31 off; 9/13 within 0.3) |

- **PS08 (pair-star).** The sigma^-2 law holds in every visiting seed: 16 complete seeds, gamma = 2.002 ±
  0.001 (range 2.00-2.01), RSS 0.0027 against 148 for the route law sigma^-3; local exponents at the two smallest pairs
  2.000 ± 0.000. 173 of 176 escapes are split-type (W[110000] ≈ 0.085 at the threshold), 3 mixed (at
  sigma = 0.1). The learning order is a property of the seed, not of sigma: 16/40 seeds learn q first at
  every sigma, 24/40 learn the residual pair first at every sigma (the verifier had 3/10 q-first).
  Seed 7 reproduces the verifier's plateau (dt = 211.88 vs 211.9 at sigma = 0.01).
- **PS06 (control).** 7 of 12 seeds learn the residual pair first, at every sigma from 0.1 to 3e-4.
  The verifier's analysis (`analyze_gf.py`) reported 12/12 "q-first": its test (q-first if
  t_esc > t_q) cannot see this case for the 0.6 teacher, because the pair-first state has L ≈ 0.5 >
  L_S = 0.36, so the loss crosses L_S - 1e-3 only after q is learned later; for the 0.8 teacher the
  pair state has L = 0.5 < L_S = 0.64 and the test works. So 6 of the verifier's 10 fitted control
  exponents (seeds 3, 5, 8, 9, 10, 11) are not q-plateaus. Among the genuine q-first seeds: seed 0
  has 3.00 (route law), seeds 2 and 7 4.00, seed 4 3.81 (local exponents rising 3.1 -> 3.93), seed 1
  trapped;
  every escape goes through the k* = 5 route of r2 (W[110010] = 0.0017 at the threshold, no split
  pattern). Diagnostic (`diag/ps06_route_entries.py`): at the q saddle reached from small init every
  dormant entry scales exactly as sigma^1 (slopes 1.00 for all route entries in seeds 0 and 2), so
  the exponent 4 is not in the initial amplitudes: the saddle state is theta_S + sigma xi' with a
  seed-dependent xi', i.e. an H-A start along a special direction; see 9.5 for the same integer shift
  in saddle starts.
- **B8W (binary8 wrong order, rho = 0.8).** 13 of 30 seeds learn the smaller component first (43 %;
  the `order` agent: 41 %). Of the 13 complete seeds, 9 converge to 4.00-4.16 and seed 1 is at 4.55 (still falling; local
  exponents fall from ~5 at sigma = 0.3 to 4.00-4.02 at sigma = 0.003: the predicted crossover from the route law
  to the order-6 law), and 3 (seeds 6, 24, 28) stay at 5.00 although their escapes at small sigma are
  split-type (polymer direction, W[p]/W[r] = 12.6, the same ratio as in the other seeds). The
  registered rule gives INCONCLUSIVE (mean 4.31 ± 0.12, 9 of 13 seeds within 0.3).
  **Route law above the crossover: SUPPORTED** by the registered comparison: local exponent 4.87
  (26 pairs with both sigma >= 0.1) against 5 (route) and 4 (order 6).

## 9.7 Accuracy, float64 floor, budget

- **Integrator.** LSODA (rtol 1e-10, atol 1e-15, exact Hessian), stepped by hand, events refined by exact
  re-integration (deviation 3). Tolerance reruns (rtol 1e-12, atol 1e-17; seeds 0-2 at the largest,
  middle and smallest window point of every cell): 77 comparable pairs, max |relative change of t_esc|
  = 1.6e-7. Plain-loss reruns (seeds 0-2 at the two smallest eps): 48 comparable pairs, max 8.3e-8: at
  these eps the plain loss gives the same escape times when both escape. But in 4 cases (PS06 seeds 1, 2
  at the two smallest eps; C2 seed 1 at 1e-5; C3 seed 2 at 10^-3.5) the cancellation-free run finds the
  route dead (residual amplitude 1e-30 ... 1e-35, trapped or failed in a dead state) while the plain-loss
  run "escapes" at t = 5e19 ... 8e21: roundoff-seeded escapes, the float64-floor failure that the
  cancellation-free form avoids. Non-comparable checks (34) are runs where the main run is trapped and
  the check is trapped, failed in a dead state, or reached t_max.
- **Conservation.** Max drift of the bond conserved quantities over a run: <= 2e-9 in every escaped run of
  the test cells and H-B; up to 7e-6 in the slowest escaped control runs (t_esc ~ 1e21) and up to 1.75 in
  one failed and one trapped run at t ~ 1e23-1e27 (integration in dead states; not used).
- **Floor index** (after the reference switch): Phi <= 2e-5 in every escaped run of the test cells
  (largest: B8W 1.3e-5), <= 2e-6 in H-B; 16 escaped control runs are flagged (C1 7, C3 7, PS06 2: the
  slowest, shifted escapes with t_esc = 1e18-1e21) and excluded from the fits, as registered.
- **Unfinished runs.** All runs ended with an outcome (escaped, trapped, or LSODA failure in a dead state)
  except 34 H-A runs and 6 checks stopped at the end of the compute budget (19:12; list in
  `results/unfinished.json`): PS06 6 (4 at eps >= 0.018, outside the window), C1 23 (14 outside the window;
  the others are seeds 2, 5, 7, 10, 11 at eps = 10^-2.75 ... 10^-3.5, all of them seeds whose route dies),
  C3 3 (seeds 2, 8, 10 at 10^-3.25 ... 10^-3.5) and HT4S seed 6 at eps = 10^-3.75 and 1e-4 (the seed at 6.0).
  None of them can change a decision: the controls stay below 10 complete seeds, and HT4S stays REFUTED
  with or without seed 6.
- **Budget.** Compute 15:21-19:12 (3.85 h of wall clock). The first two attempts (15:21-16:13)
  were discarded (deviations 3, 6). The final sweeps ran 16:21-18:04 with each worker inheriting a 64-thread
  BLAS pool (load ~250 on 192 cores); 88 runs without outcome (wall cap, solver stalls) were rerun 18:04-19:12
  with the in-solver reference switch (deviation 7).

## 9.8 Claims and status

| # | claim | status |
|---|---|---|
| D1 | From theta_S + eps xi (12 Gaussian directions per cell, eps <= 1e-3 resp. 3e-3), the escape time scales as eps^-(p-2) with p the theorem's order < k*: PS08 (p = 4, k* = 5: gamma = 2.001 ± 0.001), W6 (4 vs 6: 2.001 ± 0.001), HT4W (6 vs 7: 3.996 ± 0.003), HT4L (3.997 ± 0.003), binary8 rho = 0.5 (3.998 ± 0.007); every seed within 0.05 | empirical |
| D2 | In those cells every window escape is along the split (generalized-piggyback) direction, and q is then forgotten in all PS08/W6 runs and most HT4W/HT4L runs (not in binary8) | empirical |
| D3 | From small init, the pair-star seeds that visit theta_S (16/40; whether a seed visits is independent of sigma) have plateaus dt ∝ sigma^-(2.002 ± 0.001) for sigma = 3e-4 ... 1e-2 (16 seeds) | empirical |
| D4 | binary8 wrong order (rho = 0.8): the local escape exponent falls from the route law (5.10 ± 0.05 at sigma >= 0.1, saddle starts; 4.87 from small init) towards the order-6 law; at the smallest sigma 7/12 saddle-start seeds and 9/13 visiting small-init seeds are at 4.0-4.2 (one more at 4.55, falling), the others at 5.0 | empirical |
| D5 | HT4S (gap s_r - s_q = 0.15): only 3 of 11 complete seeds show the order-6 law at eps in [1e-4, 3e-3]; the others follow 5 (route law, or the split law shifted by one power); the leading-order crossover estimate b = 4 kappa / (5 s_r) does not predict where seeds cross | empirical |
| D6 | Controls (p = k*): seeds escape with exactly the route law k* - 2, or with k* - 1 (or k*), or are trapped (dead routes); never with a smaller exponent; the shift is a property of the direction | empirical |
| D7 | The shift comes from route entries slaved at O(eps^2) by the faster two-core cluster dynamics (one more factor eps in the route product); traps when they are slaved to zero | conjecture |
| D8 | The verifier's "exponent ~4 in half of the control seeds" (pair-star 0.6 from small init) is in part a classification artefact (7 of 12 seeds learn the residual pair first; its test cannot see this for the 0.6 teacher) and otherwise the shift of D6 | empirical |
| D9 | S9 ("t_esc ~ eps^-(p-2)", note section 6) holds for typical directions below a teacher- and seed-dependent crossover; above it, and for a minority of directions, the escape is slower (route law or shifted law), never faster, in line with the order being a lower bound on the descent | empirical (consistent with the proven order) |
| D10 | scipy's LSODA dense output (and so `solve_ivp` events with LSODA) can be off by percents at step times; a fresh LSODA started on a plateau can stay nonstiff at its stability limit; plain (non-cancellation-free) losses produce roundoff-seeded escapes from dead routes at eps <= 1e-4.5 | empirical (numerics) |

## 9.9 Open and next

- The integer shifts (9.5) need an explanation (D6 is a conjecture): a direct test is to follow the route
  entries and the cluster amplitudes along a shifted and an unshifted seed and check the slaving
  b ~ eps^2.
- HT4S and binary8 (rho = 0.8): the reduced law should appear at smaller eps; the two-mechanism fits put the
  crossover at eps ~ 1e-9 (HT4S) and sigma ~ 0.03 (binary8, median; seed-dependent), so HT4S is out of reach
  in float64 (floor estimate u eps^-3 ~ 1e-16 eps^-3), binary8 is reachable with sigma down to ~1e-4.
- The controls need many more seeds (or a different perturbation family) to get 10 complete seeds through
  the windows: 3-4 of 12 seeds keep their route alive at all eps.
- After the escape: q is forgotten in PS08/W6 (all runs) and in most HT4W/HT4L runs, never in the binary8
  cells and HT4S within 3 t_esc + 100 (not analysed further; the note's section 6 describes the PS08 case).
