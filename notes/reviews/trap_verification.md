# Verification: `notes/theory_trap.md` v1 and `experiments/theory_checks/trap/`

Verifier (theory + numerics), 2026-10-10. Scope: the derivation (sections 2-6, 8), the per-run tests
(section 7), the stated failures. Everything below was checked on the merged commit 046292f.

**Bottom line.** The core mechanism holds: C_u is conserved, and the end's normal block goes to
[C_u]_+. I confirmed this in full gradient flow, not just the reduced model. The T3 identity and the
c_e = 1 probability law are right. The per-run tables reproduce exactly, and my own code gets the
same answers. Problems: (i) "dead iff >= 2 ends die" and the pinned escape law miss the inner
three-core cluster channel, which matters from k = 6 (my GF shows a pinned k = 6 route decaying, not
escaping); (ii) the k = 4 reduced ODE is basis-dependent when an inner edge has chi >= 3; (iii) the
d >= 3 probabilities at a capped end bond are wrong; (iv) several failure attributions ("large
sigma") are not supported; (v) "20/20" and "6/6" (s2s) need qualifiers. The s2s leakage is stated
honestly.

## What was done

- **Re-run.** `predict_locality.py` (4 workers, 8 min) and `analyze_locality.py` into a scratch dir:
  `verdict`, `verdict_r12`, `verdict_q`, `q`, `k`, `C_ends` identical on all 11,200 rows;
  `summary_by_criterion.csv` identical. Headline counts reproduced (408/457, 141/934, 38/2,101;
  R12 417/1,143; L0 462/1,514; pilot 102/109, 20/182, 2/146; MPS4 chi = 2: 150/150 vs 0/344).
- **Independent recomputation** (own code: own Steiner subtree, own leaf-to-root contraction for the
  overlaps, own correlation flow, learned directions from the environment of the learned component
  instead of the code's top singular vector, own block extraction and own k = 4 reduced route ODE;
  only `get_tree`, `TTNSpec.init(1, seed)`, `pattern_teacher(..., seed=1000)` and `gauge_balance`
  reused). Stratified sample of 96 (case, seed) keys with k >= 4 (3 per tree x start x d x chi):
  learned component q 96/96, verdict 94/96, C_u median abs. difference 6e-5. The two flips are MPS8
  (K = 8) near-boundary ends (|C| < 0.003): at K = 8 the correlation-flow stop M = 1e10 leaves the
  learned/small split ambiguous at O(M^(-1/K)) = 6% (C_u differs by up to 0.29 in one excluded q = 1
  balanced case). M cannot be raised much: DOP853 fails at M = 1e16 (blow-up in finite time).
- **Seeding checked**: locality `initial_point` = `spec.init(sigma, seed)` (+ `aligned_init` of
  component 0 with bond index 0 for saddle starts); pilot `simulate` the same; teacher seed 1000 in
  both; the predictor's `spec.init(1, seed)` with w = e_0 is the same init direction. Balanced runs:
  locality used `run.balance_gauge`, the predictor `s2s.gauge_balance`; C_u is invariant under the
  residual orthogonal gauge, and my sample agrees on the balanced keys.
- **Own gradient-flow checks** (library loss, own LSODA driver, MPS6 {0,5}, saddle start, chi = 2,
  sigma = 0.03, to t = 3e6): see T2 and T6 below.
- **Out-of-sample run of their HT4 check** (`gf_check_s2s.py tree ht 4 0.2 6,7,8,9`; the note used
  seeds 0-5): see section "s2s-core".
- **Monte Carlo** of the end-death probability (own, 2e5 draws): see T11.

## Verdict per claim

| claim | verdict | notes |
|---|---|---|
| T1 normal-part lemma | **correct** (proven) | Checked: x_par (x) y and x (x) y_par are single-core changes; x_perp (x) y_perp is orthogonal to T; siteless core -> tangent; only the order-k route term overlaps R = s_r U_r for k >= 3. The adiabatic elimination is a derivation (fine as "derived"). |
| T2 C_u conserved, X^T X -> [C_u]_+ | **correct** (proven in the reduced flow) | Standard two-layer zero-target balancedness. Confirmed in the FULL gradient flow (my run, MPS6 {0,5}, sigma = 0.03): seed 1 (both ends C < 0): C = -0.6311 / -0.6869 constant to 4 digits after the O(1) transient, both normal blocks -> 1e-30, partner Grams -> 0.6311 / 0.6869 = [C]_-; seed 3: dying end C = -0.0704 -> -0.0696 (1% drift, O(sigma)), partner Gram -> 0.0696; live end X^T X -> 0.2087 = C, partner -> 0. |
| T3 identity C_u = Q_nunu + ||Y_nv||^2 - ||X_nv||^2 + (y_par^2 - x_par^2) | **correct** (exact bookkeeping); O(sigma^3) size **correct** for small init | Q_e[w, nu] = a(x_par - y_par) + O(sigma^2) is conserved and O(sigma^2) from a small init, so the last term is O(sigma^3). Saddle-start caveat (O(sigma^2), compute from blocks) is stated and implemented. |
| R12 kept only the route entry of Y_nv | **correct** | Code implements R12 as q_nn + |y_route|^2 at the plateau start. |
| T4 siteless partners die only via three-core clusters (rate sigma^4), active only for k >= 6 | **correct** (derived) for a rank-one saddle | Cluster conservation (all three squares lose at the same rate; lightest dies) checked; code rule c1 < 0 and c1 < c2 matches. Same three-core mechanism exists through INNER route cores; see T6. |
| T5 k >= 2j; k <= 3 never traps | **correct** (derived) | Empirical: 0 stalls in all 5,744 committed k <= 3 runs (locality + pilot). Note: k <= 3 verdicts are "alive" by construction of the rule, so this is a test of the rule, not of C_u. |
| T6a "dead iff >= 2 ends die" | **gap** | Correct as sigma -> 0 if the end channels are the only decay channels (linear stability gamma gamma' > kappa^2 rederived: kappa ~ sigma^(k-2), gamma ~ sigma^2). Missing channel: three-core clusters through inner route cores (end or inner core, siteful inner core, next core) decay at rate sigma^4 and can remove an inner route entry (for a non-differing inner site the inner route entry IS the cluster's middle factor). For an alive route this is marginal at k = 6 and dominant for k >= 7 (the note says so only in failure 3 / open issues). For a route with ONE dying end it is dominant already for k >= 6 (sigma^-4 << sigma^-(2k-6)) and marginal at k = 5. So "iff" holds for k <= 5 only. |
| T6b pinned end x* ~ sigma^(k-3), escape after sigma^-(2k-6) | **correct as a model computation, untested; pre-empted for k >= 6** | Algebra rederived (2k-3 factors, T = gamma b^-(2k-4)/((2k-4)c^2)). My GF (MPS6 {0,5} seed 3, k = 6, one dying end, sigma = 0.03): the dying end settles at |x|/sigma ~ 2e-7 (pinned, as predicted), but then the dormant overlap DECREASES (1.2e-20 at t = 2.5e5, 9.1e-21 at 1e6, 1.0e-21 at 3e8; algebraic decline, no growth) instead of growing; the pilot labels this seed dead at sigma = 0.1-0.3 (s2 at roundoff). Consistent with the inner-cluster channel, not with a pinned escape. |
| T6c k = 4: sigma-free competition decided by the reduced ODE | **correct** (derived) for one new direction per route edge; **wrong in code at chi >= 3 inner edges** | `reduced_route` picks `complement(w)[:, 0]` on inner route edges: an arbitrary basis vector. My test, MPS4 {0,3}, chi = 4, small init, 8 seeds x 4 bases: seed 0 "escape" in the code's basis, "dead" in 3 of 3 random rotations of the inner new space. Verdicts at chi >= 3 inner edges are not gauge invariant. |
| T7 Prop. 1 (Gram bound), corollary (C8) | **correct** (proven) | "A tree-leaf end with lambda_2(Q(0)) > 0 cannot die" uses sigma_2(G_u)^2 ~ ||X_perp||^2 (the learned column absorbs x_par); fine at leading order. |
| T7 Prop. 2 (spare directions protect if c_e > n_w) | **correct** (derived, generic) | |
| T8 outcome depends only on the init direction | **correct** (derived, sigma -> 0) with O(sigma) and O(M^(-1/K)) corrections | Empirical per seed (their per_seed.csv): predicted dead 45/65 seeds stall at every sigma, 16 mixed, 4 never; alive 447/462 never. |
| T9 balanced init: tree-leaf ends cannot die | **correct** (derived, leading order) | "Removes the odeco trap (0 trapped)" is right for the label "trapped", but 69 of 150 balanced odeco runs are "stalled" (binary8 27/30, HT4 12/30, MPS8 30/30); say so. |
| T10 per-run tests | **correct, reproduced**; two labelling caveats (below) | |
| T11 p_u = I_{F_u/(F_u+F_w)}(n_u/2, n_w/2) | **correct** for c_e = 1 (rederived: A/(A+B) ~ Beta(n_u/2, n_w/2)); MC 0.391 / 0.499 / 0.391 / 0.335 / 0.291 vs 0.392 / 0.5 / 0.392 / 0.333 / 0.292 | **wrong** for "the end bond at its cap chi_e = d" with d >= 3: there c_e = d - 1 > 1 and death needs C_u <= 0 on all c_e directions (the note's own T2). MC: 0.023 (d = 3), 1.5e-4 (d = 4), not 0.25 and 0.142. |
| MC: 0.1-0.3 dead routes per run | **plausible, not re-run** | Closed forms for k >= 5 two-ended routes are products of the (correct) single-end law; the HT4 four-ended route is 0.33-0.42, outside "0.1-0.3". |

## Per-run tests (section 7)

**Reproducibility: correct.** See "What was done". The tables of section 7 match the CSVs.

**Are the classes fixed in advance?**
- Observed classes come from detectors defined before this work (locality stall detector; stall
  diagnosis `r12.csv`; pilot collapse detector) with ONE exception: pilot runs that hit t_max or the
  wall clock with |a2_end| < 1e-20 are relabelled "dead" by `analyze_locality.py`. This affects 46 of
  the 102 "dead" predicted-dead pilot runs; 10 of the 102 end with the second singular value at
  1e-9..1e-4, i.e. not at roundoff. Using only the pilot's own detector: predicted dead 56/63 (89%)
  collapse, predicted alive 0/150. The conclusion stands; "102 of 109 die" should read "die or end
  with the second component below 1e-20".
- Predicted classes: the rules (C <= 0, k >= 2j with the marginal case counted, >= 2 dying ends,
  k = 4 reduced ODE) are already in the first commit (3888622), before the per-run tables. Changes
  after it: the q = 1 (wrong-order) exclusion and the reduced-route event logic. Neither helped the
  headline (predicted-alive stall rate 1.2% -> 1.8%); the excluded wrong-order runs behave like the
  rest (predicted dead 10/12 stall, alive 1/36). The reduced-model thresholds (1e3 growth, 1e-12 decay,
  1e4 decay times) are ad hoc but generic. No evidence of tuning on outcomes; I cannot see work done
  before the first commit.
- Caption erratum: R12 and L0 verdicts in the code use "dead iff >= 2 ends negative" (`classify`),
  not "dead iff all ends negative" as section 7 says; identical for two-ended routes.

**Independent C_u**: see "What was done" (94/96 verdicts; the two flips are K = 8 boundary cases).

## s2s-core and leakage (section 7.5-7.6)

- **Order from the init: correct.** First learned component 1,196/1,196 (order column is 1-based);
  not trivial: 352 runs learn a non-largest component first (baseline "largest first": 844/1,196).
  First two 872/1,029 (baseline 464). It is 200 init directions, each counted at several sigma.
- **Leakage: stated honestly.** Section 7.5 says the init-only phase-2 verdict fails (table shows
  it) and that the 20/20 and 6/6 use gradient flow up to the second escape. Keep that wording in
  the report; do not call the s2s trap "predicted from the init".
- **"MPS4 20/20" overstates by one.** Separated: reduced ODE 10/10 (trap seeds 3, 5, 6, 7 at every
  sigma; my check of `main_runs.csv`), sign test 8/10. Close: seeds 1, 7 dead (observed trapped; seed 7
  completes at sigma >= 0.07), seed 5 is "escape, one end at C' = +0.0097" vs observed "stalled" at
  every sigma. That is 19/20 plus one consistent-but-not-predicted stall.
- **HT4 6/6 does not extend cleanly.** Seeds 6-9 (my run of their script, sigma = 0.2): seed 7 dead
  -> trapped (right), seed 9 alive -> stalled (right, not trapped), seeds 6 and 8 pinned (one leaf with
  C' < 0) -> trapped (wrong under "dead iff two ends die"). 8/10 over all seeds. At k = 7 and sigma =
  0.2 a pinned leaf is ~sigma^(k-4) = 8e-3 relative times the product of six O(0.3) route amplitudes
  over |C|, roughly 1e-5, still far above the "trapped" threshold 1e-8 sigma, so "trapped = deeply
  pinned" (the binary8 argument) is unlikely to rescue these two; a second death channel is more
  likely (see T6a).

## Stated failures: complete? honest?

I listed every false alarm (56, predicted dead but escaped) and miss (40, predicted alive but stalled)
of the fan-in, k >= 4, q = 0 table by (tree, D, start, chi, seed, sigma). All of them fall in the five
listed classes, so the list is **complete as an inventory**. Several **attributions are wrong**:

1. **"Large sigma" (failure 3) is not shown for the k = 6 misses.** midroot6 {0,5} seed 4 stalls at
   all six sigma of its grid (small init) and at 0.08-0.167 (saddle); MPS6 {0,5} and MPS6 {0,...,5}
   seed 4 (saddle) stall at 0.08-0.24 and ESCAPE at 0.347 and 0.5. The grid starts at 0.08, so "all at
   sigma >= 0.08" is the grid, not a finding, and the failures sit at the SMALL end. These ends are
   barely alive (e.g. MPS6 {0,5} seed 4: C = +0.007): slow escapes that exceed the stall patience, or the
   marginal (k = 6) inner three-core cluster. Seed-specific, not finite-sigma.
2. **HT4 {1,2} seed 1** stalls at all six sigma (0.03-0.3); the note says slow escapes (fine), but only
   the two dead runs at 0.19 and 0.30 belong to "large sigma".
3. "Two predicted-dead MPS8 {1,6}/HT8 runs at sigma >= 0.45": three runs (HT8 {0,1,2,3} at 0.455 and
   0.6, MPS8 {1,6} at 0.5).
4. **Pinned -> dead is under-reported.** Failure 5 lists the 5 stall-diagnosis runs (all sigma >= 0.19).
   The pilot has 20 predicted-pinned runs labelled dead; MPS6 {0,5} saddle seed 3 is dead at roundoff
   at sigma = 0.2, 0.3, near it at 0.1 (s2 = 3e-10), and declining at 0.02-0.05 (s2 ~ 5e-8 at timeout).
   My GF at sigma = 0.03 shows the same decline after pinning (T6b). So pinned -> dead is not a
   large-sigma artefact; it is the missing inner-cluster channel (T6a).
5. **Failure 1 (k = 4 with spare inner channels) is worse than described**: besides the scalar-route
   approximation, the code's scalar is a basis-dependent choice (T6c); the 26/46 row is partly arbitrary.
6. Not listed: the K = 8 correlation-flow state at M = 1e10 is ambiguous at ~6% (2/96 verdict flips in my
   sample, all with |C| < 0.003).
7. "The stall rate of pinned runs grows with k": k = 6 31% (87/284), k = 7 13% (8/60), k = 8 81%: not
   monotone.
8. Multi-stage odeco traps (section 7.5): honestly described as not predictable from the init.

## Errata (to apply in `notes/theory_trap.md`)

1. Section 8, "Dependence on ... d": the values "0.392, 0.25, 0.142 with the end bond at its cap
   chi_e = d" use the scalar law where c_e = d - 1 > 1. Correct (MC): 0.392, ~0.023, ~1.5e-4. The
   formula I_x(n_u/2, n_w/2) is for c_e = 1 only; say so where it is stated (also in the summary).
2. T6 / section 2.5: "dead iff at least two ends die" -> "for k <= 5, as sigma -> 0"; add that
   three-core clusters through inner route cores (rate sigma^4) are a further death channel, marginal
   at k = 6 and dominant for k >= 7, and that they pre-empt the pinned escape for k >= 6.
3. T6 pinned law sigma^-(2k-6): tag conjecture for k >= 6 (pre-empted, see 2); untested at k = 5.
4. `trapcrit.reduced_route`: inner route edges with c > 1 use an arbitrary basis vector; restrict the
   k = 4 ODE verdict to routes whose every edge has c = 1 (as `predict_locality` already does for the
   finite-sigma version) or implement vector amplitudes; re-tabulate MPS4 chi = 4.
5. Section 7 caption: R12 and L0 use ">= 2 ends", not "all ends".
6. Section 7: "102 of 109 die" -> "die or end with |a2| < 1e-20 (56 of 63 by the pilot's own collapse
   detector)"; say that the relabelling rule is new and that 10 of the 102 are not at roundoff.
7. Failure 3: drop "large sigma" for the k = 6 midroot6/MPS6 seed-4 misses (they fail at the small end
   of a grid that starts at 0.08); "two runs" -> three runs (HT8 x2, MPS8 {1,6}).
8. Failure 5: add the 20 pilot predicted-pinned runs labelled dead (MPS6 {0,5}-family seed 3 at all
   sigma); attribute to the inner-cluster channel, not to large sigma. Theory_saddles P14(v) change
   ("at small sigma") is therefore NOT supported.
9. Section 7.5 / section 10 report item 3: "20/20" -> "19/20 + 1 stalled with a barely-alive end";
   HT4 "6/6" -> "6/6 on seeds 0-5, 8/10 on all ten seeds".
10. T9 / section 6: "removes the odeco trap" -> "no balanced odeco run traps (0/150), but 69/150 stall".
11. "The stall rate of pinned runs grows with k" -> not monotone (k = 7: 8/60).
12. Section 7 / T10: note the K = 8 numerical ambiguity of the correlation-flow state at M = 1e10
    (~6%; flips verdicts with |C_u| < ~0.003); integrate the blow-up in a log-amplitude variable if
    larger M is needed.

## Recommended status tags

| claim | tag |
|---|---|
| T1 lemma | proven |
| adiabatic elimination / L_eff | derived |
| T2 (conservation, limit [C_u]_+) | proven (reduced flow); checked in full GF (this review) |
| T3 identity | proven (exact); O(sigma^3) remainder derived; checked |
| T4 siteless partners, three-core clusters | derived |
| T5 k >= 2j; k <= 3 never | derived; empirical (0 / 5,744) |
| T6 "dead iff >= 2 ends die" | derived for k <= 5; incomplete for k >= 6 |
| T6 pinned amplitude sigma^(k-3) | derived; checked qualitatively (one GF run) |
| T6 pinned escape sigma^-(2k-6) | conjecture (untested; pre-empted for k >= 6) |
| T6 k = 4 sigma-free competition | derived (c = 1 edges only) |
| T7 Prop. 1 / corollary | proven |
| T7 Prop. 2 (spare directions) | derived |
| T8 init direction decides | derived (sigma -> 0); empirical (per seed, with exceptions) |
| T9 balanced leaf ends | derived |
| T10 per-run predictor | empirical, reproduced and independently recomputed |
| T11 single-end law, c_e = 1 | proven (exact for the stated Gaussian saddle-start model); checked |
| T11 d >= 3 at the cap | wrong (see erratum 1) |
| s2s order from init | empirical (1,196/1,196) |
| s2s trap from phase-2 state | empirical (MPS4 19/20 + 1; HT4 8/10), not an init-only prediction |

## Safe to cite

- On a rank-one plateau, the decisive conserved quantity of a route end is C_u = X_perp^T X_perp -
  Y_perp^T Y_perp (normal-visible blocks of end and partner); the end's normal block converges to
  [C_u]_+ (proven in the reduced flow; confirmed in full gradient flow).
- C_u = Q_nunu + ||Y_nv||^2 - ||X_nv||^2 + O(sigma^3) after a small-init escape; R12 kept only the
  route entry of Y_nv.
- Routes with k <= 3 cores never trap (derived; 0 stalls in 5,744 runs).
- Fan-in locality runs, k >= 4: predicted dead 408/457 (89%) stall vs predicted alive 38/2,101 (1.8%);
  R12: 417/1,143 (36%) vs 51/1,233 (4.1%) (reproduced; init + sigma-free correlation flow only).
- MPS4 chi = 2: 150/150 predicted-dead runs stall or die, 0/344 others (reproduced).
- Pilot: predicted dead 56/63 collapse by the pilot's own detector; alive 0/150.
- P(one end dies) = I_{F_u/(F_u+F_w)}(n_u/2, n_w/2) for one new direction, saddle start: 0.392 at an
  MPS tree-leaf end, 0.5 at the MPS root end (d = chi = 2).
- Balanced init: tree-leaf ends cannot die at leading order; no balanced odeco run traps (0/150).
- s2s-core: the first learned component is predicted from the init in 1,196/1,196 runs; the odeco
  trap is NOT predicted from the init by this machinery (only from the state after the second escape).

**Not safe to cite**: "dead iff two ends die" for k >= 6; the pinned escape law; any d >= 3 trap
probability at a capped end bond; the MPS4 chi = 4 (k = 4) per-run verdicts; "large sigma" as the cause
of the k = 6 misses and of pinned -> dead runs; "20/20" and "6/6" without the qualifiers above.
