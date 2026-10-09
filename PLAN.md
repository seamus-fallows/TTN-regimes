# Research plan

Status: **draft v0** (2026-10-09), written before the literature review returned; the
known/partly-known/open tags are provisional until `notes/literature.md` is final. Updated as
results come in; superseded decisions go to `notes/decisions.md`.

Model and diagnostics are fixed in `notes/model.md` and `notes/diagnostics.md`. Hypotheses H1–H6
refer to `notes/prior_knowledge.md` section 8.

## 1. Research questions, ranked

Ranking = importance to the main question (saddle-to-saddle in TTNs) x expected payoff x
tractability.

### Q1. Does gradient flow on a TTN from small initialization show saddle-to-saddle dynamics, and which saddles does it visit? — *partly known*

- **Known**: implicit bias toward low hierarchical tensor rank and incremental learning in
  hierarchical tensor factorization (Razin, Maman & Cohen 2022); full saddle-to-saddle theory in
  deep linear networks (Jacot et al. 2021) and diagonal networks (Pesme & Flammarion 2023).
- **Hypothesis**: from small fan-in init, the loss plateaus at the levels of partial fits
  (odeco teacher: L_k = 1/2 sum_{r>k} s_r^2), bond spectra grow one singular value at a time, and each
  plateau sits near a critical point of the bond-rank-constrained problem.
- **Theory**: reduction to Saxe-type ODEs for odeco teachers with aligned init (exact); local
  analysis of the saddles (Q3).
- **Experiments**: binary/MPS/HT trees, N = 4, 8 (16 where affordable), odeco teachers, sigma over
  2 decades, 10 seeds; show plateaus, bond spectra, overlaps; aligned-init runs vs exact solution.
- **Plan changes if**: plateaus do not appear for generic random init (then characterize when they
  do), or the visited saddles are not rank-constrained critical points.

### Q2. How do plateau durations scale with init scale, tree size and bond dimension? — *partly known (DLN analogue), open for TTNs*

- **Hypothesis**: first escape t_1 ~ rho_0^{-(K-2)} [tau_align + K^{K/2-1} / ((K-2) s_1)], where
  rho_0 = ||theta_0|| is the total initial parameter norm and K the number of cores (derivation:
  spherical-flow + radial growth of the degree-K homogeneous function g = <W, W*>; reduces to the
  exact Saxe-type result a_0^{-(K-2)}/((K-2)s) for aligned init). Consequences: t_1 ~ sigma^{-(K-2)};
  and at fixed sigma, t_1 decreases with bond dimension through rho_0^2 ~ sigma^2 sum_v chi_v.
- **Theory**: Kumar & Haupt-style directional convergence; compute tau_align and its distribution.
- **Experiments**: t_1 vs sigma (exponent), vs chi at fixed sigma, vs K (N and tree type); prefactor
  distribution over seeds vs theory.
- **Plan changes if**: exponents deviate from K-2 (would indicate the effective degree is not the
  number of cores, e.g. due to imbalance) or chi-dependence is absent (rho_0 picture wrong).

### Q3. Do later escape times depend on where the new component lives in the tree (locality)? — *believed open (H2, H3)*

- **Hypothesis (H2)**: after some components are learned, the next escape needs k cores to acquire
  new directions, k = number of cores in the Steiner subtree of the sites where the new component
  differs from what is learned (k = 1 if they are all on one core). Escape time ~ sigma^{-(k-2)} for
  k >= 3, ~ log(1/sigma)/lambda for k = 2 (lambda = s_new times the learned amplitudes on the
  rest of the path), no plateau for k = 1. **The same teacher therefore has different saddle
  sequences on an MPS and on a binary tree.**
- **Hypothesis (H3)**: the saddle is strict (gauge-projected Hessian has a negative eigenvalue) iff
  k = 2; for k >= 3 it is a higher-order saddle.
- **Hypothesis (H1b)**: if the smaller of two components is learned first (possible by H1), the
  "piggyback" direction (reuse the learned bond index, rotate the physical directions toward the
  larger component) has Hessian eigenvalue q(s_learned - s_new) < 0, so the order is corrected
  through a strict saddle in time O(log 1/sigma) before the second rank is added. (Hand derivation
  for the N=4 binary tree with odeco teacher; balanced learned amplitudes q = s^{1/K}.)
- **Theory**: second-order expansion of L around the learned-component saddle in the directions
  that add the new component; identify the lowest-degree escape route; reduced ODE for its
  amplitude.
- **Experiments**: "pattern" teachers (components differing on chosen site sets) on MPS, binary,
  HT and custom trees with N = 4–8; exponent of t_2 - t_1 vs sigma vs predicted k; Hessian spectra at
  plateau midpoints; same teacher on different trees.
- **Pilot running** (`experiments/pilot_h2_locality`). **Plan changes if** the pilot refutes H2:
  then Q3 becomes "map empirically which escapes are fast" and the agent is redirected.

### Q4. Which component is learned first, and how random is the order? — *partly known (H1)*

- **Known (general)**: small-init gradient flow of homogeneous models first converges in direction
  to a KKT point of the correlation function, then grows (Kumar & Haupt); for odeco tensors every
  component is an attracting fixed point of the tensor power method (Anandkumar et al. 2014).
- **Hypothesis (H1)**: for genuinely tensorial teachers the first escape direction is decided by the
  basin of the initialization, so the order is random; P(largest first) increases with s_1/s_2 and
  decreases with K (the init overlap enters with power K-2), and may concentrate as chi grows.
- **Experiments**: odeco teachers with s_2/s_1 in [0.5, 0.99]; P(greedy order) vs ratio, K, chi, d;
  matrix-like teachers (unique local max) as control; time-resolved check of H1b.

### Q5. Map of regimes — *known in DLNs/diagonal nets; open for TTNs*

- **Hypothesis (H4)**: one knob, the per-core init scale sigma in the centred model (target
  W* + W_0, `ShiftedTeacher`), moves the TTN from saddle-to-saddle (small sigma) through rich
  non-incremental to lazy (large sigma). The function-space init scale is sigma^K, so the S2S
  boundary sigma_c(K) moves toward 1 as K grows. Imbalance between cores (root vs rest) is a second
  knob (Kunin et al. 2024). At large bond dimension, an NTK-type parameterization is lazy and a
  muP-type one keeps feature learning.
- **Theory**: lazy boundary from Chizat et al.'s criterion (relative weight change vs function
  change); S2S boundary from the escape-time formulas (separation ratio rho = 3); muP-type scalings
  from the spectral condition (Yang, Simon & Bernstein 2023) per core.
- **Experiments**: phase diagrams sigma x chi, sigma x K, sigma x imbalance with the regime rules of
  `notes/diagnostics.md` and their threshold sensitivity; large-chi sequences under two
  parameterizations; antisymmetric (two-copy) construction as a cross-check of the centred one.

### Q6. Does input covariance change the picture? — *partly known (Saxe-type reweighting)*

- **Hypothesis (H5)**: with product covariance aligned to the teacher, component r behaves as if its
  weight were Lambda_r s_r with Lambda_r = prod_i lambda_{r,i} (escape times and order follow
  Lambda_r s_r); misalignment and cross-site correlations are the cases that could matter.
- **Experiments**: power-law site spectra, aligned and misaligned teachers; one cross-site-correlated
  input model with sampled data. Report whether anything qualitatively new happens.

### Q7. Does teacher structure change the picture? — *open*

- Non-orthogonal CP teachers; random TTN teachers (are the visited saddles the hierarchical-SVD
  truncations? in which order do bond ranks grow?); teachers whose correlation structure mismatches
  the student's tree (connects to Q3).

### Q8. Learning rate, SGD noise and finite samples — *partly known (EoS, implicit bias)*

- GD stability limit eta_max ~ 2 / (K s_1^{2-2/K}) from the sharpness at the balanced minimum (to
  verify); edge-of-stability behaviour; whether large steps or SGD noise shorten plateaus at small
  sigma. Finite n: sample complexity of S2S (low hierarchical rank bias) vs lazy (min-norm in
  tangent space).

## 2. Theory programme (runs in parallel with experiments)

Local theory agents (no compute) write derivations to `notes/theory_*.md`, each ending with
testable predictions, which are passed to the experiment agents:
1. **Escape from the origin** (Q2, Q4): spherical-flow/radial decomposition for the TTN's g; tau_align;
   rho_0 dependence; race model for the order.
2. **Saddle analysis** (Q3): expansion around learned-component saddles; the k-rule; strict vs
   higher-order; piggyback correction (H1b); reduced ODEs for escape amplitudes.
3. **Regime boundaries** (Q5): lazy criterion, S2S separation criterion, large-chi parameterizations.

## 3. Division of work

One remote agent per well-scoped task (briefs in `notes/briefs/<name>.md`). Each works on branch
`agent/<name>` in its own experiment folder(s), starts from the shared library and
`experiments/00_smoke`, and delivers: a notes file, figures, summary data, code with an updated README.

| agent | questions | first deliverable |
|---|---|---|
| `s2s-core` | Q1, Q2 | plateau sequences + escape-time scaling vs sigma, K, chi with theory overlay |
| `locality` | Q3 | exponent table (measured vs predicted k) across trees/teachers; Hessian at saddles |
| `order` | Q4 | P(greedy order) vs ratio, K, chi; H1b check |
| `regime-map` | Q5 | phase diagrams with diagnostics and threshold sensitivity |
| `data-teacher` | Q6, Q7 | covariance and teacher-structure checks |
| `optim-samples` | Q8 | LR/SGD/finite-sample effects |

Launch order: `s2s-core`, `locality`, `order`, `regime-map` first (main focus); `data-teacher` and
`optim-samples` when the first four are producing results (or immediately if budget allows).

## 4. Management

- **Check-ins**: every ~2 h while agents run (`crab-student list/status/report`); read their
  REPORT.md and figures; answer questions within the check-in.
- **Review before merge**: rerun at least one key figure from the agent's code; check seeds and
  error bars; check conclusions follow from data; reproduce surprising results (another seed range,
  or a local rerun) before building on them.
- **Merge**: `crab-student pull`, review the diff, merge `agent/<name>` into `main`; move generally
  useful code into `ttn/` with tests; update `CLAUDE.md` (status, findings) and this plan.
- **Stop / redirect** a line of work when: a pilot refutes its hypothesis and no clear alternative
  question remains; two check-ins pass without new evidence; or its cost exceeds its budget share
  without a deliverable.
- **Shutdown**: before destroying an instance, confirm code + summary data are pushed and merged,
  notes written, raw outputs copied to `data/<experiment>/`. Spend recorded in `notes/spend.md`.

## 5. Compute and cost

To be filled with the runner benchmark. Working assumptions: CPU-bound float64 runs of small
networks; 32–64 vCPU machines at $0.15–0.8/h; GPU unnecessary. Budget $300 total; target <= $200 for
two rounds of six agents, keeping >= $100 for follow-ups and reproductions.
