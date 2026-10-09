# Research plan

Status: **v1** (2026-10-09), revised after the literature review (`notes/literature.md` v1). Updated as
results come in; superseded decisions go to `notes/decisions.md`.

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

### Q1. Saddle sequence of gradient flow on TTNs from small init — *open* (top open problem in the literature)

- Known around it: full saddle-to-saddle theory in DLNs is proven only for the first escape at generic
  init (Jacot et al. 2021) and for decoupled DLNs / deep diagonal nets; Furman 2026 gives the landscape.
- **Hypothesis**: plateaus sit at bond-rank-constrained critical points (bond spectra = truncated teacher
  spectra; for odeco teachers loss levels L_k = 1/2 sum_{r>k} s_r^2); bond spectra grow one value at a time.
- **Theory**: Saxe-type exact solution for aligned odeco teachers (explicit for TTNs here; listed as future
  work by Furman 2026); perturbation theory away from alignment.
- **Experiments** (`s2s-core`): binary/MPS/HT, N = 4, 8, odeco and close-weight teachers, random init,
  sigma over >= 1.5 decades, 10 seeds; check plateau locations against constrained critical points.

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
- **Theory**: `notes/theory_escape.md` (in progress).

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
  MPS / binary / HT. Local pilot `experiments/pilot_h2_locality` running.
- **Theory**: `notes/theory_saddles.md` (in progress).

### Q4. Learning order and the first-escape selection rule — *partly known (analogues), open for TTNs*

- **Known in analogues**: order set by (weight) x (init overlap)^(p-2) for degree p >= 3 (tensor PCA,
  two-layer nets, AGF, deep diagonal nets, tensor power method); largest-first above a gap factor; over-
  parameterization restores largest-first (Ge et al. 2021; Ren 2025).
- **Open**: which KKT points of g the TTN's random init selects; whether local maxima are bond-rank-one;
  P(greedy) vs gap, K, chi, d; the 2-core correction route (H1b).
- **Experiments** (`order`).

### Q5. Map of regimes — *partly known in analogues, open for TTNs*

- **Corrections from the literature**: (1) at large chi the expected tangent kernel is isotropic, so lazy
  training needs no centring there (W_0 is fitted away); centring matters at finite chi / finite samples;
  laziness needs chi >> K on every bond. (2) The init-to-target norm ratio alone does not decide the regime
  (CP needs two parameters): test collapse in sigma^K, per-bond Gram scale, sigma^K chi^(K-1). (3) No muP
  exists for TTNs; a per-core muP from signal propagation would be new.
- **Hypothesis H4 (revised)**: centred-model sigma moves the TTN from S2S through rich to lazy at finite
  chi; the S2S boundary in per-core sigma moves toward 1 as K grows; imbalance is a second knob.
- **Experiments** (`regime-map`); **theory**: `notes/theory_regimes.md` (in progress).

### Q6. Input covariance — *partly known*

- **Known/derived**: product covariance is an invertible reparameterization of the cores with physical
  legs, so the landscape is that of a white problem with target ((x)_i Sigma_i^{1/2}) W*; the gradient-flow
  metric and conservation laws change, so time scales and order can change. Misaligned covariance in linear
  nets: realignment, non-monotone active sets.
- **Open**: separating the landscape and metric effects quantitatively; cross-site correlations.
- **Experiments** (`data-teacher`); **theory**: `notes/theory_covariance.md` (in progress).

### Q7. Teacher structure — *open*

- Non-orthogonal CP teachers, random TTN teachers (are plateaus at HSVD-truncation-like constrained critical
  points?), teachers on a different tree than the student. **Experiments** (`data-teacher`);
  **theory**: `notes/theory_teachers.md` (in progress).

### Q8. Learning rate, SGD noise, finite samples — *partly known (DLN/diagonal/two-layer)*

- **Corrected H6**: small steps track GF and the DLN sharpness formula S = K s^{2-2/K} should carry over; but
  large steps and SGD noise can change the late saddle sequence (re-balancing, stochastic collapse, noise
  equilibrium along gauge orbits). **Experiments** (`optim-samples`); **theory**:
  `notes/theory_optimizer.md` (in progress).

### Backlog (launch when an agent slot frees up)

- **B1 Optimizer dependence**: GD vs DMRG/ALS sweeps, Riemannian / dynamical low-rank (BUG), natural
  gradient and Adam on the same teacher: are plateaus removed, is the order changed? (Practically relevant:
  TN practitioners train with sweeps, not GD.)
- **B2 Exact theory**: completeness of the per-bond conservation laws (Marcotte et al. 2023 algorithm on
  small trees); explicit metric (J J^T)^{-1} for balanced TTNs; escape-path theorem modulo gauge.
- **B3 Imbalance**: per-core sigma_v and the fan-in root imbalance; stalls near non-min-norm critical points
  (Furman 2026 Thm G.1).
- **B4 Reproductions**: independent re-runs of every surprising result before it enters the report.

## 2. Theory programme

Local theory agents (derive -> adversarial verify -> revise) write `notes/theory_*.md`, each ending with
quantitative predictions that are forwarded to the experiment agents: escape (Q2, Q4), saddles (Q3),
regimes (Q5), covariance (Q6), optimizer (Q8), teachers (Q7). All six are in progress.

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
