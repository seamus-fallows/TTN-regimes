# Research plan

Status: **v2** (2026-10-10): the status of every question (Q1–Q8) and backlog item (B1–B5) is updated to
the state after the reviews, fix rounds and verifications; "Beyond the plan" (end of section 1) and
section 6 (remaining work) are new. v1 (2026-10-09) was revised after the literature review
(`notes/literature.md` v1). Updated as results come in; superseded decisions go to `notes/decisions.md`.

Model and diagnostics are fixed in `notes/model.md` and `notes/diagnostics.md`. Hypotheses H1–H6 refer to
`notes/prior_knowledge.md` section 8; their literature status is in `notes/literature.md` section 3.
Tags: **known** (cite and show in our setup), **partly known**, **open**.

## 0. Where the literature leaves us

No paper characterizes saddle-to-saddle dynamics of gradient flow on dense-core TTNs: neither the sequence of
saddles nor how plateaus scale with init or tree shape. The closest works are Furman et al. 2026 (exactly our
model and loss, landscape only: no spurious minima for realizable targets along balanced trajectories;
non-global critical points are rank-deficient; descent order bounded by the size of the rank-deficient core)
and Razin, Maman & Cohen 2022 (CP-structured nodes; per-node growth exponent). The near-origin mechanism
(early directional convergence, escape after ~sigma^-(K-2)) is a corollary of Kumar & Haupt 2025. Our main
contribution is therefore the **dynamical, topology-dependent picture** (Q1–Q3), with Q4 (learning order) and
Q5 (regime map) as the other substantial open items.

## 1. Research questions, ranked

### Q1. Saddle sequence of gradient flow on TTNs from small init — *open for K >= 3* (top open problem in the literature)

- Known around it: the full sequence is proven at depth 2 (Jiang, Chen & Ding 2023; Jin et al. 2023), which
  covers two-core TTNs; at depth >= 3 only the first escape at generic init (Jacot et al. 2021) and decoupled
  DLNs / deep diagonal nets; Furman 2026 gives the landscape.
- **Hypothesis**: plateaus sit at bond-rank-constrained critical points (bond spectra = truncated teacher
  spectra; for odeco teachers loss levels L_k = 1/2 sum_{r>k} s_r^2); bond spectra grow one value at a time.
- **Theory**: Saxe-type exact solution for aligned odeco teachers (explicit for TTNs here; listed as future
  work by Furman 2026); perturbation theory away from alignment.
- **Experiments** (`s2s-core`): binary/MPS/HT, N = 4, 8, odeco and close-weight teachers, random init,
  sigma over >= 1.5 decades, 10 seeds; check plateau locations against constrained critical points.
- **Status (2026-10-10): done.** `s2s-core` reviewed, fixed and verified
  (`notes/reviews/s2s_core_round2_verification.md`; errata E1–E21 applied in `notes/s2s_core.md`); report
  Sec. 4.1–4.3 and 4.6. The hypothesis holds where tested: from fan-in init with odeco teachers the plateaus
  sit at the partial-fit levels L_S (bond-rank-constrained critical points with truncated teacher spectra
  on every bond), one rank step per event on every edge (497/531 runs), on binary4, mps4, binary8 and ht4;
  mps8 learns at most two components. At chi = R and K = 7, 8 most fan-in runs trap or stall
  (conservation-law trap; report Sec. 4.6). Generic teachers: plateaus at rank-constrained critical points
  (Q7). Aligned init = exact Saxe-type modes (known for DLNs; explicit for TTNs).

### Q2. Plateau-duration scaling with sigma, K, chi — *partly known*

- **Known**: first escape after ~||theta_0||^-(K-2) (Kumar & Haupt 2025a); escape time
  1/(K(K-2) g(w*) delta^(K-2)) from a nondegenerate KKT point w* (Kumar & Haupt 2025b); exponent K-2 in all
  DLN-type analogues.
- **Open for TTNs**: gauge orbits make every KKT point degenerate for chi >= 2 (the 2025b theorem needs a
  quotient extension); prefactor distribution under random fan-in init; dependence on chi given the
  conservation laws (fan-in init has root bonds child-heavy: E[Q_v] = sigma^2 (1 - chi_p/chi_v) I);
  later gaps (dormant entries are rescaled by earlier phases).
- **Experiments** (`s2s-core`): t_1 vs sigma, chi, K against the Kumar-Haupt formula with g(w*) =
  s_1 K^{-K/2}; dormant-block norms at each event; later gaps.
- **Theory**: `notes/theory_escape.md` (v2, verified).
- **Status (2026-10-10): done** (`s2s-core`, as Q1; theory_escape v2; report Sec. 4.3). Per seed
  t_1 = sigma^-(K-2) T(theta_hat) + c_K with no free parameter (exponents 0.9996, 2.0000, 4.998, 5.9995 at
  K = 3, 4, 7, 8); the Kumar-Haupt time is a proven lower bound, 3-2700x below the median; later gaps
  ~ sigma^-(K-2) per seed for odeco teachers (K = 3, 4 over 1.5 decades; K = 7, 8 over <= 1 decade); weak
  chi-dependence of the prefactor; the balanced KKT escape direction is proven (known in substance:
  Kumar & Haupt 2025a, Furman et al. 2026). Dormant components are suppressed by earlier escapes, and
  fan-in init can kill their routes (dead routes; Q3, report Sec. 4.6).

### Q3. Topology-dependent escape exponents (locality) — *partly known (landscape), open (dynamics)*

- **Known**: descent order <= N, the number of nodes of a connected rank-deficient core (Furman 2026,
  upper bound); Tucker examples (Frandsen & Ge 2022); general principle "plateau ~ delta^-(L-2), L = lowest
  homogeneity degree of the dormant block" (Kumar & Haupt 2025c) and the r = 1/2/>=3 trichotomy for DLNs with
  r small layers (Rawal & DeWeese 2026).
- **Hypothesis H2 (refined)**: at a saddle, the next escape's exponent is set by the lowest degree with
  nonzero residual correlation in the dormant block; for pattern teachers this is the number k of cores in
  the Steiner subtree of the sites where the new component differs (k = 1: no plateau; k = 2: log; k >= 3:
  power law). Compare k with Furman's N. Gaps scale with the dormant-block norms entering the plateau, which
  equal ~sigma only if earlier phases did not rescale them.
- **Hypothesis H3 (corrected)**: a saddle is strict iff some direction has a nonzero degree-2 residual term
  (k = 2 escapes, wrong-component points); every saddle needs >= 2 cores (single-core blocks are PSD). A
  tree version of the DLN "tightened" criterion (Achour et al. 2024) is the target.
- **H1b**: learned-the-smaller-first points are strict saddles (DLN analogue known; TTN 2-core route open).
- **Experiments** (`locality`): exponent table over trees x teachers; Hessians at plateaus; same teacher on
  MPS / binary / HT. Local pilot `experiments/pilot_h2_locality` (done).
- **Theory**: `notes/theory_saddles.md` (v2 verified; v3 after the saddle-order theorem).
- **Status (2026-10-10): done.** `locality` reviewed twice and verified
  (`notes/reviews/locality_round2_verification.md`); report Sec. 4.4. H2 holds: the plateau exponent is
  k - 2 with k = cores in the Steiner subtree of the differing sites, for k = 1-5 on 9 trees (k = 6-7 on
  binary8/ht4 with balanced init only; k = 8 not measured; named exceptions); k = 2 log law with head
  start, lambda_2 = s_2 s_1^((K-2)/K); same teacher on MPS / binary / HT gives log / sigma^-3 / ~sigma^-5.
  H3 holds by the lambda_min scaling rule. Theory: Steiner identity proven; the saddle order is k* when
  the generalized piggyback operator stays below s_q and drops to 2l* < k* when it exceeds s_q at a level
  l* < k*/2 (saddle-order theorem, proven; "Beyond the plan"). Dead routes at k >= 4 under fan-in init
  (balanced init reduces them ~10x). H1b: see Q4.

### Q4. Learning order and the first-escape selection rule — *partly known (analogues), open for TTNs*

- **Known in analogues**: order set by (weight) x (init overlap)^(p-2) for degree p >= 3 (tensor PCA,
  two-layer nets, AGF, deep diagonal nets, tensor power method); largest-first above a gap factor; over-
  parameterization restores largest-first (Ge et al. 2021; Ren 2025).
- **Open**: which KKT points of g the TTN's random init selects; whether local maxima are bond-rank-one;
  P(greedy) vs gap, K, chi, d; the 2-core correction route (H1b).
- **Experiments** (`order`).
- **Status (2026-10-10): done.** `order` reviewed, fixed and verified
  (`notes/reviews/order_round2_verification.md`, safe-to-cite list C1-C17; errata applied, `notes/order.md`
  section 12); report Sec. 4.5. The order depends on m, the number of cores carrying the differing sites:
  m >= 3 gives a random, sigma-independent order (an exact race of independent blow-up times on MPS,
  40800/40800 runs); m = 2 is greedy as sigma -> 0 with P(wrong) consistent with sigma^((1-rho)/rho) for
  non-adjacent pairs (adjacent pairs steeper; n = chi_e conjecture). H1b holds for non-adjacent m = 2
  pairs (strict saddle, rotation, forget and relearn); adjacent pairs add the larger component instead;
  m >= 3 keeps the wrong order (binary8 only above the saddle-order crossover). R = 3: consistent with
  independent clocks; R = 4 = chi: most runs do not learn all four components.

### Q5. Map of regimes — *partly known in analogues, open for TTNs*

- **Corrections from the literature**: (1) at large chi the expected tangent kernel is isotropic, so lazy
  training needs no centring there (W_0 is fitted away); centring matters at finite chi / finite samples;
  laziness needs chi >> K on every bond. (2) The init-to-target norm ratio alone does not decide the regime
  (CP needs two parameters): test collapse in sigma^K, per-bond Gram scale, sigma^K chi^(K-1). (3) No muP
  exists for TTNs; a per-core muP from signal propagation would be new.
- **Hypothesis H4 (revised)**: centred-model sigma moves the TTN from S2S through rich to lazy at finite
  chi; the S2S boundary in per-core sigma moves toward 1 as K grows; imbalance is a second knob.
- **Experiments** (`regime-map`); **theory**: `notes/theory_regimes.md` (v2, verified; corrections after
  experiments 2026-10-10).
- **Status (2026-10-10): round 1 done, round 2 running.** `regime-map` reviewed; fix round 1 done locally
  on the copied raw outputs and verified (`notes/reviews/regime_map_round2_verification.md`); report Sec. 3
  and 5.1-5.4. At N = 4: S2S, rich and lazy regions; a lazy regime only when every bond reaches the cut
  dimension (below it centred runs stall at the Eckart-Young floor); both boundaries scale ~ chi^-1/2
  (collapse in sigma sqrt(chi); known idea, TTN measurement new); theory_regimes P5-P7, P16, P20 confirmed
  with the tensor kernel; imbalance: a light root lowers the lazy boundary, a heavy root does not make
  training lazier. H4's large-K part is open: the round-1 K >= 8 values were brackets, and sigma_c^K =
  const vs sigma_c^K / sqrt(D) = const cannot be distinguished; regime-map round 2 tests it ("Beyond the
  plan").

### Q6. Input covariance — *partly known*

- **Known/derived**: product covariance is an invertible reparameterization of the cores with physical
  legs, so the landscape is that of a white problem with target ((x)_i Sigma_i^{1/2}) W*; the gradient-flow
  metric and conservation laws change, so time scales and order can change. Misaligned covariance in linear
  nets: realignment, non-monotone active sets.
- **Open**: separating the landscape and metric effects quantitatively; cross-site correlations.
- **Experiments** (`data-teacher`); **theory**: `notes/theory_covariance.md` (v2, verified).
- **Status (2026-10-10): done.** `data-teacher` ran two rounds; reviewed, fixed and verified
  (`notes/reviews/data_teacher_round3_verification.md`; errata E1-E16 applied); report Sec. 6. Aligned
  covariance is a reweighting (race with weights Lambda s, landscape with sqrt(Lambda) s; exact clocks);
  misaligned covariance and cross-site correlations make the later escapes of MPS students strict and
  log-time, while binary/HT students keep sigma^-1 there (higher-order).

### Q7. Teacher structure — *open*

- Non-orthogonal CP teachers, random TTN teachers (are plateaus at HSVD-truncation-like constrained critical
  points?), teachers on a different tree than the student. **Experiments** (`data-teacher`);
  **theory**: `notes/theory_teachers.md` (v2, verified).
- **Status (2026-10-10): done** (`data-teacher`, as Q6; report Sec. 6). Any non-orthogonality (CP overlap,
  generic TTN teachers) turns the escape of MPS students from the rank-one saddle into a strict, log-time
  escape; binary/HT keep sigma^-1: the tree decides the saddle type. Plateaus are rank-constrained critical
  points (clearly distinct from every SVD truncation for one teacher only). Star teachers: a larger learned
  component can be forgotten and relearned (strict iff rho(S) > s_q). Teacher and student on different
  trees: report Sec. 6.7.

### Q8. Learning rate, SGD noise, finite samples — *partly known (DLN/diagonal/two-layer)*

- **Corrected H6**: small steps track GF and the DLN sharpness formula S = K s^{2-2/K} should carry over; but
  large steps and SGD noise can change the late saddle sequence (re-balancing, stochastic collapse, noise
  equilibrium along gauge orbits). **Experiments** (`optim-samples`); **theory**:
  `notes/theory_optimizer.md` (v2, verified).
- **Status (2026-10-10): done.** `optim-samples` reviewed, fixed and verified
  (`notes/reviews/optim_samples_round2_verification.md`, errata for citing); report Sec. 5.5.
  eta_max = 2/(K s_1^(2-2/K)) is exact at small init; below it GD keeps gradient flow's saddle sequence
  (765/765 runs) with an O(eta) event shift; above it a 2-cycle, then divergence or a return to the origin;
  SGD never shortens the first plateau and lengthens or traps later ones; with finite samples, small init
  recovers W* with its bond ranks from n ~ 256, while the lazy limit cannot at finite chi.

### Backlog (launch when an agent slot frees up)

- **B1 Optimizer dependence**: GD vs DMRG/ALS sweeps, Riemannian / dynamical low-rank (BUG), natural
  gradient and Adam on the same teacher: are plateaus removed, is the order changed? (Practically relevant:
  TN practitioners train with sweeps, not GD.)
  **Status (2026-10-10): done** (agent `optimizers`: reviewed, `notes/reviews/optimizers_round1.md`, and
  fixed; report Sec. 5.6). GD and heavy ball follow gradient flow; Adam/RMSProp keep weight-ordered
  sequential learning but remove the sigma^-(K-2) scaling of the first escape (sigma-independent number of
  steps; eps crossover law known: Sabanayagam & Lucey 2026); re-gauged GD, Riemannian GD, ALS and DMRG
  remove the plateaus; Gram preconditioning is not sigma-free.
- **B2 Exact theory**: completeness of the per-bond conservation laws (Marcotte et al. 2023 algorithm on
  small trees); explicit metric (J J^T)^{-1} for balanced TTNs; escape-path theorem modulo gauge.
  **Status (2026-10-10): partly covered by the theory programme; not run as a separate item.** Proven and
  in report App. A: equal norm shift of all cores, invariance of unused bond channels, balanced KKT points
  of the escape, the saddle-order theorem. Open (to report Sec. 7): completeness of the conservation laws,
  the explicit metric, an escape-path theorem modulo gauge.
- **B3 Imbalance**: per-core sigma_v and the fan-in root imbalance; stalls near non-min-norm critical points
  (Furman 2026 Thm G.1).
  **Status (2026-10-10): covered inside other items.** Root scale at fixed alpha mapped by `regime-map`
  (theory_regimes R13; P20 confirmed); the fan-in root imbalance and the dead routes and traps it causes
  studied as failure modes (`s2s-core`, `locality`, theory_saddles; overlaps Furman Thm G.1, see
  `notes/reviews/citation_check.md`); gauge-balanced init removes frozen routes but not the long plateaus
  at large K (report Sec. 4.6). Per-core sigma_v beyond the root was not swept.
- **B4 Reproductions**: independent re-runs of every surprising result before it enters the report.
  **Status (2026-10-10): done as the review gate.** Every agent round got a review (A: partial reproduction
  plus an independent check; B: adversarial audit of claims vs data), and every fix round an independent
  verification; 20 reviews and verifications in `notes/reviews/` (scripts kept for provenance).
- **B5 Real data** (launched 2026-10-09 as `real-data`): MNIST/Fashion-MNIST with cos/sin local features on
  quadtree-binary, snake-MPS and scrambled trees: do plateaus, stepwise bond growth, the locality rule and the regime
  map survive outside the teacher-student setting?
  **Status (2026-10-10): done** (reviewed; local fix round; round 2 with a pre-registered test; verified,
  `notes/reviews/real_data_round2_verification.md`, errata E1-E9 applied; report Sec. 6.9). On MNIST 3 vs 5
  (4x4, chi = 4) bond ranks grow one route at a time and the next route is the lowest-degree one (22/26
  units); plateaus creep instead of sitting flat; k = 3 gaps fit A/sigma + B; the "locality" effect is a
  pixel-order (orientation) effect measured by route amplitudes; regimes S2S -> rich -> lazy only partly.

### Beyond the plan (added 2026-10-10)

- **Saddle-order theorem** (local theory task; `notes/theory_saddle_order.md` v2, verified twice:
  `notes/reviews/saddle_order_verification_proof.md`, `..._numerics.md`; theory_saddles v3; report Sec. 4.4
  and App. A.4). "Saddle order = k*" is false in general. Proven: the order is k* when the generalized
  piggyback operator stays below s_q (always for k* = 2; for k* = 3, 4 with a positive definite old block;
  for two-component teachers in the correct order), and 2l* < k* when it exceeds s_q at a level l* < k*/2
  (pattern teachers with a spare bond index; first case k* = 5, the MPS6 pair-star; HT4 and binary8
  wrong-order saddles have order 6, not 7). Its dynamical prediction is being tested by the remote agent
  `saddle-dyn` (pre-registered; brief `notes/briefs/saddle-dyn.md`; running).
- **Saddle hopper** (local agent; `notes/saddle_hopper.md` v2, `experiments/saddle_hopper/`): predicts the
  whole saddle sequence from the theory notes' rules and scores it on the committed runs. The adversarial
  audit (`notes/reviews/saddle_hopper_verification.md`) found it reproducible but permissive: as a
  whole-sequence validation it is **bookkeeping** of known theory, not new evidence (the trivial
  weight-order rule scores the same). Citable: the implementation, the multi-saddle slopes of the TC
  teacher on mps8, and the check of the saddle-order condition at 587 saddles of past runs (30 have order 6
  instead of k* = 7).
- **Real-data pre-registered order test** (`real-data` round 2; `notes/real_data.md` section 9, registered
  before any run and verified intact): across 27 pixel orders of binary16, the data-only share of route
  amplitude on k <= 5 routes predicts the final loss (Spearman -0.60 [-0.82, -0.22], p = 0.001; within 20
  new random orders -0.55, p = 0.015) but not the time to L <= 0.25 (-0.30, p = 0.12). Mechanism:
  conjecture.
- **Regime-map round 2** (remote, running; pre-registered in `notes/regime_map.md` section 4 on branch
  `agent/regime-map`): realizable K sweep (`binary_tree(N)`, K = 3-15, teacher (3, 2), d = 2 and 3, spare
  channels chi = 2R in the plain model, full bonds in the centred model) to decide whether sigma_c(K) tends
  to 1 (sigma_c^K = const) or to sqrt(d) (sigma_c^K / sqrt(D) = const). Report Sec. 3 keeps a placeholder
  for it.
- **Prior-work audit** (`notes/reviews/citation_check.md`): five results first tagged new had prior or
  concurrent work and were re-tagged known (decision log, 2026-10-10 midday).

## 2. Theory programme

Local theory agents (derive -> adversarial verify -> revise) write `notes/theory_*.md`, each ending with
quantitative predictions that are forwarded to the experiment agents: escape (Q2, Q4), saddles (Q3),
regimes (Q5), covariance (Q6), optimizer (Q8), teachers (Q7). Status (2026-10-10): all six reached v2
(derive -> adversarial verify -> revise) on 2026-10-09; theory_saddles is at v3 after the saddle-order
theorem (`notes/theory_saddle_order.md` v2, verified twice); theory_regimes carries dated corrections after
the regime-map experiments.

## 3. Division of work

Remote agents (crab-student, autonomous mode, branch `agent/<name>`, briefs in `notes/briefs/`), launched
2026-10-09 ~18:25 UTC, before the library was complete (decision log):

| agent | questions | machine | $/h |
|---|---|---|---|
| `s2s-core` | Q1, Q2 | 64 vCPU | 0.74 |
| `locality` | Q3 | 48 vCPU | 0.35 |
| `order` | Q4 | 96 vCPU | 0.84 |
| `regime-map` | Q5 | 384 vCPU | 1.04 |
| `data-teacher` | Q6, Q7 | 32 vCPU | 0.47 |
| `optim-samples` | Q8 | 32 vCPU | 0.37 |

Added later (remote): `optimizers` (B1) and `real-data` (B5) on 2026-10-09, `saddle-dyn` on 2026-10-10
(running). Local agents also ran the regime-map and real-data fix rounds (their machines could not
restart), the saddle-order theorem, the saddle hopper, every review and verification, and the report
sections.

Local agents: library build and review, theory programme, pilots, report infrastructure, reviews of agent
results, reproductions. At least 4–8 agents are kept doing useful work at all times; when one finishes, the
next item from the backlog or a follow-up is launched.

## 4. Management

- **Check-ins** every ~2 h: `crab-student list/status/report`; read REPORT.md and figures; answer and
  redirect by `crab-student send`. A monitor flags state changes (needs input, errors, stops) immediately.
- **Review before merge**: rerun at least one key figure from the agent's code (local agent); check seeds and
  error bars; check conclusions follow from the data; reproduce surprising results before building on them.
- **Merge**: `crab-student pull`, review the diff, merge `agent/<name>` into `main`; move generally useful
  code into `ttn/` with tests; update `CLAUDE.md` and this plan.
- **Stop / redirect** a line of work when a pilot refutes its hypothesis with no clear alternative, two
  check-ins pass without new evidence, or it exceeds its budget share without a deliverable.
- **Shutdown**: before destroying an instance, confirm code and summary data are pushed and merged, notes
  written, raw outputs copied to `data/<experiment>/`; record spend in `notes/spend.md`.

## 5. Compute and cost

- Workload: CPU-bound float64 integration of small networks; many independent runs; no GPU needed.
- Burn rate with six agents: $3.81/h (~$91/day). Budget $300 total.
- Plan: round 1 (six agents) ~24–36 h -> ~$90–140; round 2 (follow-ups, backlog B1/B3, reproductions) <=
  $100; reserve >= $50. Agents that finish early are stopped (machines stop themselves when idle) and
  destroyed after their work is merged.
- Status (2026-10-10 ~15:00 UTC): about $38 spent (`notes/spend.md`). Running: regime-map round 2 ($1.04/h)
  and saddle-dyn ($1.02/h, expected $8-12). Every other machine was deleted after its raw outputs were copied
  to `data/<experiment>/` and verified.

## 6. Remaining work (2026-10-10)

- **Running work, then review and merge:** regime-map round 2 (fills the large-K placeholder of report
  Sec. 3 and settles or leaves open H4's large-K part; theory_regimes R9/P15) and saddle-dyn (dynamics of
  the saddle-order theorem; report Sec. 4.4 / App. A.4). Each gets an independent verification before its
  numbers enter the report; then copy and verify the raw outputs, delete the machines, record the spend.
- **Report Sec. 1** (introduction and summary of findings): one row per question Q1–Q8 (finding, status
  tag, new/known, section and figure), written last from the verified notes and the safe-to-cite lists.
- **Report Sec. 4.7** (dependence on each parameter): one summary table (sigma, K and tree shape, chi, d,
  teacher spectrum, imbalance, step size) with the effect on the saddle sequence, plateau durations and
  order, the evidence and the status tags, pulled from Sec. 4.1–4.6.
- **Report Sec. 7** (open questions), each with what would settle it. Candidates: saddle order at the
  borderline lambda_max = s_q and without the index condition; sigma_c(K) at large K (if round 2 leaves it
  open); n = chi_e for adjacent m = 2 pairs; k = 8 and k >= 6 on MPS; random-init prefactors; Adam's later
  plateaus; the muP-TTN limit; trap probability as sigma -> 0; an escape-path theorem modulo gauge and the
  completeness of the conservation laws (B2); the real-data route-amplitude mechanism.
- **Final consistency pass:** notes, report, CLAUDE.md and this plan against the verified claims
  (statuses, withdrawn claims, event and kernel rules: `notes/diagnostics.md` vs report Sec. 2, including
  its known dip-and-relearn inconsistency); stale skeleton placeholders (Sec. 4 opening, App. B common
  settings, App. C software and figures); figures regenerated by their scripts; App. C commands.
- **Final review:** an independent end-to-end review of the compiled report (every claim against its note
  and the safe-to-cite lists, status and known/new tags, citations), then the last PDF build.
