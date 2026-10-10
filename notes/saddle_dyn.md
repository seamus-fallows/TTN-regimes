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
2. **Event root finding.** One H-B run crashed in scipy's event solver (brentq got equal signs at
   the bracket ends because LSODA's dense output does not exactly reproduce the step values). The
   solver now falls back to the first sign change on a 200-point grid of the step (refined by
   brentq). No result changes; the 5 H-B runs written before the fix are kept (resumable sweep).

# Part II: results

(to be filled in after the runs)
