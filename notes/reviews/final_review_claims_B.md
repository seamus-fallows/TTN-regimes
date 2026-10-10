# Final review B: claims of Sections 5-7 and Appendix A (adversarial review, 2026-10-10)

**Scope.** Sec. 5 (other regimes, learning rate, SGD, finite samples, optimizers), Sec. 6 (covariance,
teacher structure, real data), Sec. 7 (open questions), Appendix A (derivations; A.4 against
`notes/theory_saddle_order.md` v2 and its two verifications). Every substantive claim was checked against
the verification reviews (errata and safe-to-cite lists first), the round-1/2 reviews, the experiment and
theory notes, `notes/reviews/citation_check.md` part b, and the committed CSV/JSON in
`experiments/*/results/` (numbers recomputed with pandas; throwaway scripts in the session scratchpad,
`final_review_B/`). Six sub-reviews (Secs. 5.1-5.4 + A.8; 5.5 + A.7; 5.6; 6.1-6.9 + A.5-A.6; 6.10-6.11;
A.1-A.3) were merged here; every critical item and most major items were re-checked by the lead reviewer.
Not reported: the placeholders (regime-map round 2, saddle-dyn, trap theory) and `notes/report_todo.md`.
Line numbers: `report/sections/` at the time of review (05 = `05_other_regimes.tex`, 06 =
`06_covariance_teacher.tex`, 07 = `07_open_questions.tex`, A = `A_derivations.tex`, 01, 04 likewise).

**Bottom line.** 4 critical, 28 major, 90 minor items. No theorem of App. A is wrong; one proven
sub-statement, prop:A-hessian(d), lacks two hypotheses (M23), and a few tags are stronger than their sources
(M25, M26, M28). The algebra of A.1-A.8 was spot-checked throughout (constants recomputed: delta_K, omega_K, a_0c, u_0c = 1.306e-2,
the EoS amplitude, c_1, c_2 and 6.50 chi^-1/2, xi_c brackets, sigma_floor, P(Q>0) = (2/3)^3, the route-degree
table, the piggyback and star eigenvalues, the A.6 covariance identities), and A.4 matches
`theory_saddle_order.md` v2 including Theorem 1' and the scope fixes of both verifications. The problems
are claims stated beyond their evidence, a few restated narrowed claims, a misapplied formula in a figure,
and inconsistencies between sections. Major items are grouped by section; the ones to fix first are M22
(saddle hopper absent, Sec. 7 item 1 asks for it), M2 (spare channels), M10 (Adam's order), M14 (App. A star
example vs Sec. 6.9), M19 (quadtree "lowest"), M23 (prop:A-hessian(d)) and M24 (ln(1/sigma) vs O(1)).

## Critical

**C1. SGD "never faster" is false run by run.**

- Location: 05:415 "No SGD run escaped a plateau faster than gradient flow."; same message in 05:269-270
  ("gradient noise never shortens a plateau"), 01:124 (table, "SGD never shortened a plateau") and 01:290
  ("Gradient noise never shortened a plateau").
- Problem: true only for the aligned-route k = 3 test (114 of 114 escaped runs slower). From random
  initialization single runs scatter to both sides of gradient flow. Related to, but stronger than,
  report_todo item 2 (which concerns means of the first plateau only).
- Evidence: `optim_samples/results/sgd_runs.csv` (re-counted): 787 of 1600 random-init runs have
  `t_esc < t_esc_gf` (141 by more than 5%; median ratio -0.2% at sigma = 0.1); second plateau shorter than
  gradient flow's in 657 of 1529 runs; `collapse_test.csv` k = 2 test: one run 4.26 vs 4.73.
  `theory_optimizer.md` scopes "no run escaped faster" to the route test (P9 iv).
- Fix (05:415): `No escaped run on the aligned route (114 of 114) was faster than gradient flow; from random
  initialization single runs scatter to both sides of gradient flow's times (first escape earlier in 787 of
  1600 runs, by more than $5\%$ in 141), and only the means are never shortened beyond $\sim0.3\%$.`
  Apply with todo item 2 to 05:269-270, 01:124 and 01:290 ("on average, gradient noise never shortened a
  plateau by more than $\sim0.3\%$").

**C2. The small-sigma floor formula is used outside its hypothesis and drawn over data it misses by 2-23x.**

- Location: 05:33-35 "at small $\isc$ this floor rises like $\isc^{2\Kc}$ (mean
  $(1-\dim\mathcal M_\chi/d^\Ns)\,\isc^{2\Kc}/\normF{\Wt}^2$, the seed median about $0.7$ times it,
  \eqref{eq:A-centred-floor})", for chi = 3, 4, 6; caption of fig:other-regimes "dotted: $0.7$ times the
  small-$\isc$ mean"; `experiments/regime_map/plot_report.py:292-297` draws it for chi = 3, 4, 6.
- Problem: eq:A-centred-floor is derived at a smooth point, chi_e = rank of W* across e (A:1358), i.e. chi = 3
  only. At chi = 4, 6 the spare bond rank absorbs part of W_0 and the floor is far lower.
- Evidence (`regime_map/results/report/runs_sigma_chi_b4.csv`, centred, l_best, sigma <= 0.63; ratio to the
  formula, constant in sigma): six-seed median 0.48, 0.18, 0.044 and mean 0.57, 0.20, 0.072 at chi = 3, 4, 6.
  The dotted lines therefore sit 1.5x, 3.9x and 16x above the plotted medians. An independent Monte Carlo
  of the first-order Eckart-Young floor (sub-review) gives median/formula 0.75, 0.20, 0.032.
  Knock-on, outside this scope: 03:94-95 ("1.7-2.2 times the small-sigma estimate") and the open triangles of
  fig:regime-map(a) use the same estimate at chi = 4, 6.
- Fix (05:33-35): `at small $\isc$ this floor rises like $\isc^{2\Kc}$; with $\chi$ equal to the teacher's bond
  rank ($\chi=3$) its mean is $(1-\dim\mathcal M_\chi/d^\Ns)\,\isc^{2\Kc}/\normF{\Wt}^2$
  \eqref{eq:A-centred-floor} (median of the six seeds $0.48$ times it), and at $\chi=4$, $6$, where the spare
  bond rank absorbs part of $\Wz$, it is $0.18$ and $0.044$ times that expression`. Caption: `dotted
  ($\chi=3$ only): the small-$\isc$ mean \eqref{eq:A-centred-floor}`; restrict the plot line to chi == 3 (and
  draw it at 1x, not 0.7x, or say which constant is used). Re-derive the Sec. 3 stall-onset comparison.

**C3. Star summary restates the E16-narrowed claim without its scope.**

- Location: 06:382-383 (tab:cov-summary) "star teachers & larger learned component left iff
  $\lambda_{\max}(\Gamma)>\tw q$; ..."; milder: 01:120 (table) and 01:310 "star teachers make the flow
  forget a larger learned component", 04:1216 (table cell).
- Problem: the WITHDRAWN list at 06:19-20 forbids exactly this ("stars lose a learned component exactly when
  rho(S) > s_q" without the saddle-start scope, E16).
- Evidence: `data_teacher_round3_verification.md` E16: the iff holds started at theta_q with eps <= 1e-3; at
  eps = 1e-2 the apparent threshold moves to w ~ 1.43-1.44 (`r2_star_threshold.csv`: [1.42, 1.44] on MPS4 and
  binary4); "random-init runs never sit at theta_q" (Sec. 6.9 says so too).
- Fix (table): `star teachers & started at the larger learned component ($\varepsilon\le10^{-3}$) the flow
  leaves it iff $\lambda_{\max}(\Gamma)>\tw q$; from random initialization it never sits there; log time on
  MPS4, polynomial on HT4 & piggyback graph; rates within $0.21\%$ \\`. In 01:310 and 04:1216 append
  `(started at it; from random initialization the flow never sits there)`.

**C4. Section 6 intro and summary restate the E14-narrowed "binary and HT" claim for "anything".**

- Location: 06:32-35 "Anything that makes the components non-orthogonal in the metric of the loss changes the
  law of the escape from the rank-one saddle, and the tree decides how: strict and logarithmic in $1/\isc$ on
  an MPS, of higher order and $\propto\isc^{-1}$ on binary and HT trees"; 06:656-659 "Every other complication
  ... the student's tree decides whether the escape ... is logarithmic (MPS) or polynomial (binary, HT)".
- Problem: (i) correlated inputs, one of the complications, were never run on HT (E14; WITHDRAWN list
  06:14-15); (ii) sigma^-1 was never measured on HT: HT was run only from the saddle start (eps^-1);
  random initialization only on binary4 and MPS4 (`r2_random_v2_*` has no HT rows); (iii) "anything" goes
  beyond the four mechanisms tested, at N = 4, 8, d = 2-4.
- Evidence: E14 ("for cross-site correlations only binary-4 was run, never HT"); Sec. 6.2 scope paragraph
  (06:156-159) and 06:282 already state both limits.
- Fix (06:32-35): `Each of the four ways we tried of making the components non-orthogonal in the metric of
  the loss (overlapping CP components, a misaligned covariance, random TTN teachers, correlated sites) changes
  the law of the escape from the rank-one saddle ($\Ns=4$, $8$), and the tree decides how: strict and
  logarithmic on an MPS, of higher order on binary and HT trees, with escapes $\propto\varepsilon^{-1}$ from the
  saddle and $\propto\isc^{-1}$ from random initialization (binary4 only; correlated sites were not run on
  HT)`. Same qualifiers in 06:656-659 ("Every other complication tested ... polynomial (binary; HT from the
  saddle start, never with correlated inputs)").

## Major

### Sections 5.1-5.4 and App. A.8 (regime map)

**M1. Large bond dimension: "confirming the predictions, except ..." lists only the SP deviations.**

- Location: 05:197-198 "\empirical{} confirming the predictions, except for the SP leaf cores ($6.5$ standard
  errors steeper) and, at $3.3$ standard errors, the SP parameter change."
- Evidence (`regime_map/results/fix1/large_chi_slopes_tensor.csv`, centred, sigma = 1.5, column
  `se_from_theory`): NTK kernel slope -0.79 is 5.5 s.e. from -1; muP-TTN kernel +0.085, parameters -0.064 and
  leaf cores -0.065 are 3.6, 3.5 and 3.6 s.e. from 0; NTK root 2.9 s.e. The muP parameter and leaf slopes come
  from chi = 16 (medians 0.69, then 0.58, 0.59, 0.57). Every cell of tab:large-chi itself is correct.
- Fix: `\empirical{} confirming the predicted exponents in sign and size: SP kernel and root, NTK parameters and
  leaf cores within $1.1$ standard errors; the NTK kernel fit ($5.5$ standard errors from $-1$) is pulled by
  the pre-asymptotic $\chi=16$--$32$ (local slope $-1.01$ between $64$ and $128$); the $\mu$P-TTN kernel,
  parameters and leaf cores drift slightly ($+0.085$, $-0.064$, $-0.065$; $3.5$--$3.6$ standard errors from
  $0$, set by $\chi=16$); the SP leaf cores ($6.5$) and parameter change ($3.3$ standard errors) are steeper
  than predicted.`

**M2. "Spare channels help where every route end has one" is contradicted by Sec. 5.3 and Sec. 4.6 data.**

- Location: 05:158 "Spare channels help where every route end has one"; A:1408-1409 "spare channels
  ($\chi\ge2\ncomp$) avoided stalls in the runs made (one at $d=3$; 26 at $d=2$)"; same wording at 04:982.
- Evidence: at d = 3, chi = 6 = 2R the MPS8 stalls in 50 of 50 runs (05:159-160 itself;
  `results/report/k_sweep_runs.csv`), its end bonds capped at d = R; in s2s_core at d = 4, chi = 6 every MPS8 and
  HT4 route end has exactly one spare channel (cap 4 = R + 1), yet 7 of 50 MPS8 runs trap and 34 stall, and
  5 of 50 HT4 runs trap (04:996-998). Only trees with uncapped ends are stall-free (binary8: 0 of 30 at
  sigma <= 0.63; 0 of 78 with full bonds at d = 2). The A.8 counts (one run at d = 3) are the pilot's, now stale.
- Fix (05:158): `Spare channels remove the stalls only where no route end is capped: on the binary tree with
  $\Ns=8$ at $d=3$, $\chi=6$ no run stalls at $\isc\le0.63$ (30 runs), but on the MPS with $\Ns=8$ every run
  stalls (50 of 50), its end bonds being capped at $d=3$, the teacher rank, and one spare channel per end is
  not enough (MPS8 and HT4 at $d=4$, \cref{sec:s2s-failure});`. A:1408-1409: `spare channels removed the
  stalls only where no route end is capped (binary8: 30 runs at $d=3$, $\chi=6$; 26 at $d=2$), not where end
  bonds are capped at $\ncomp$ or $\ncomp+1$ (MPS8 at $d=3$, $\chi=6$: 50 of 50 stall;
  \cref{sec:other-stalled,sec:s2s-failure})`. Same change at 04:982.

**M3. "Lazy runs need uncapped chi >= 2d^2" contradicts the binary4 lazy runs at chi = d^2.**

- Location: A:1327-1328 "so lazy runs need uncapped $\chi\ge2d^2$ (\cref{sec:other-lazy})", against 05:40
  "A lazy regime therefore needs every bond at its cut dimension (binary4: $\chi\ge9$)" and Fig. 5(a).
- Evidence: binary4, centred, chi = 9 (square bottom cores): 34 of 36 runs at sigma >= 4 are labelled lazy
  (`runs_sigma_chi_b4.csv`, recounted). `theory_regimes.md` R3 says only that runs "should use" chi >= 2d^2;
  the failure to converge within the step limit is shown for binary8 at chi = 16.
- Fix (A:1327-1328): `so lazy fits at bonds equal to their cut dimension converge slowly (binary4, $\chi=9$:
  lazy, some runs ending between $10^{-6}$ and $10^{-3}$; binary8, $\chi=16$: not within the step limit,
  \cref{sec:other-lazy}); uncapped $\chi\ge2d^2$ avoids this`.

**M4. "Merged first escapes" of the two-copy model: no run merges its first two events (uncertain).**

- Location: 05:96-101 "each copy learns part of both of the first two components at once, and the first two
  events merge ... so the sweep fails condition (iii) of the saddle-to-saddle signature".
- Evidence (`runs_sigma_chi_b4.csv`, two-copy, chi = 18, 24, 36, sigma <= 0.04, re-counted): every run has
  three events; min rho at sigma = 0.010 is 2.7-14.4 (centred: 11.8-29); per seed min rho grows like
  sigma^-0.1 to sigma^-1.0 between 0.04 and 0.01. Condition (iii) asks for growth like a power of 1/sigma, which
  counting rho = 5 crossings does not test. Literal merging appears only in the pilot and the superseded
  scipy runs. (The regime-map verification accepted the claim; flagged here as uncertain.)
- Fix: `their near-origin escapes start together and the first two events crowd. On binary4 at $\chi=18$,
  $24$, $36$ every run at $\isc\le0.04$ still has three events, but $\min_i\sep i$ at $\isc=0.010$ is only
  $2.7$--$14$ (centred: $12$--$29$) and grows per seed like $\isc^{-0.1}$ to $\isc^{-1.0}$ (centred: about
  $\isc^{-1}$); the labels (5, 4, 5 seeds saddle-to-saddle at $0.010$; 4, 3, 2 at $0.016$) do not follow the
  $\isc^{-(\Kc-2)}$ law, so the two-copy sweep is no quantitative check of the boundary.`

**M5. Sec. 7 item 4 asks what Sec. 5.3 seems to answer: two meanings of "stalled".**

- Location: 07:61-62 "whether most stalled odeco runs at $\Kc=7,8$ are long plateaus or dead is a
  \conjecture"; against 05:123-124 ("From fan-in initialization the stalled route is dead ... in 28 of 32")
  and 05:156-157 (546 of 795 dead).
- Problem: item 4 means the s2s-core class "stalled" (route-end slices still above 1e-8 sigma, 04:973-975,
  1034-1037); Sec. 5.3 means the label of tab:regimes (loss above 1e-3 L(0) at the budget). A reader sees an
  open question answered two sections earlier.
- Fix (07:61-62): `whether the odeco runs at $\Kc=7,8$ that end with live route ends (the stalled class of
  \cref{sec:s2s-failure}; the balanced runs of \cref{tab:stall-test}) face long plateaus or slowly dying routes
  is a \conjecture{}; from fan-in initialization most regime-map stalls are dead routes
  (\cref{sec:other-stalled})`.

### Section 5.5 and App. A.7 (discrete steps, SGD, finite samples)

**M6. "Gradient flow's order in all 765 runs" includes deep-tree runs whose references learn one component.**

- Location: 05:296-297 "GD learned the components in gradient flow's order in all 765 runs, including 40 more
  seeds on binary $\Ns=4$ and both deep trees at $\isc=0.9$"; also 01:288-289 ("765 of 765 runs").
- Evidence (`optim_samples/results/gd_sweep.csv`, eta <= 0.95 eta_c, re-counted): the gradient-flow references
  of the deep trees learn 1 or 2 of the three components within the run (binary N = 8: 2 events in 36 runs,
  1 in 9; MPS N = 8: 1 event in 36 runs, 2 in 9), so on the MPS the order check is the first component only.
  Round-1 review item 7 noted the same.
- Fix: append `(on the deep trees within a budget in which gradient flow learns one or two of the three
  components: on the MPS only the first component is compared in 36 of 45 runs)`.

**M7. Trapping of the aligned route: one run diverged and two thirds of the "trapped" runs also forgot.**

- Location: 05:399-402 "all 30 runs with $z=(\lr/\batch)T_{\rm GF}\ge0.025$ were trapped" and "$x=1$
  separates escaped from trapped conditions in 23 of 24 cases"; 05:410-412 "all runs are trapped at $0.002$".
- Evidence (`p9_sgd.csv`, re-counted): of the 30 runs with z >= 0.0249, 29 trapped and 1 diverged (b0 = 0.003,
  B = 16); 20 of the 30 dipped below a_1 = 1/2 (19 of the 24 at eta/B = 2.5e-4, i.e. 0.25-0.30 eta*_ms), so
  trapping is confounded with forgetting there; the b0 = 0.005 condition has z = 0.0249. `optim_samples.md`
  (376-379): "the clean trapping evidence is b0 = 0.002 at 1.25e-4 (6/6 trapped, 1/6 dips)".
- Fix: `none of the 30 runs at $z\gtrsim0.025$ escaped (29 trapped, one diverged); in 19 of the 24 at
  $\lr/\batch=2.5\cdot10^{-4}$ ($0.25$--$0.30\lr^*_{\rm ms}$) the learned component also fell below
  $\frac12$, so the clean trap is $b_0=0.002$ at $\lr/\batch=1.25\cdot10^{-4}$ (6 of 6, one dip)`.

**M8. Heuristic SGD law: its consequences are tagged derived, and the synthesis states more than was tested.**

- Location: 05:409 "Consequence \derived:"; 05:413 "escapes if $\abs{\lambda_-}>c_\gamma\lr/\batch$
  \derived"; 05:495-503 (the "When does gradient flow describe practical training?" paragraph, tagged
  "\derived{} (from the results above; one teacher, trees with $\Ns\le8$)"), in particular "falls like
  $\isc^{-2(\Kc-1)}$ on the lazy side", "(about $10^{-3}\batch$ here)", "($0.1\lr^*_{\rm ms}$ was safe over 1000
  time units)".
- Problems and evidence: (i) both consequences follow from the drift law, which A.7 tags \conjecture
  (prop:A-sgd-collapse, A:1272, 1280-1282; `theory_optimizer.md` 5.4b: heuristic); (ii) the synthesis mixes
  measured thresholds (catapult window, step fractions) with derivations, and SGD was tested on binary N = 4
  only; (iii) the sigma^-2(K-1) fall was measured at K = 3 only (sigma^-3.965; the chi = 3 version is on the
  WITHDRAWN list at 05:258-259); (iv) `sgd_stability.csv`: eta*_ms = 1.07e-3 B, 1.0e-3 B for B = 1, 16 but
  0.82e-3 B and 0.44e-3 B at B = 64, 256; (v) 0.1 eta*_ms was never run: no erasure in 72 runs at
  <= 0.076 eta*_ms, which all escaped by t <= 405 (not 1000), and one erasure in six runs of about 1000 time
  units at 0.15 eta*_ms (05:430-432; `p9_sgd.csv`).
- Fix: `Consequence of the law (\conjecture):`; `(\conjecture, same drift model)`; synthesis tag
  `\empirical{} \derived{} (a summary of the results above; one teacher; GD on trees with $\Ns\le8$, SGD on
  binary $\Ns=4$)`; `falls like $\isc^{-2(\Kc-1)}$ on the lazy side (measured at $\Kc=3$: $\isc^{-3.97}$)`;
  `(about $10^{-3}\batch$ here for $\batch\le16$)`; `(no erasure in 72 runs of up to 400 time units at
  $\lr\le0.076\lr^*_{\rm ms}$; one in six runs of 1000 time units at $0.15\lr^*_{\rm ms}$)`.

**M9. App. A.7 restates the P3 claim that erratum 4 narrowed.**

- Location: A:1186-1189 "and GD converges to a flatter, more balanced minimum, with sharpness just below
  $2/\lr$ on binary $\Ns=4$ (\empirical; ...".
- Evidence: `optim_samples_round2_verification.md` erratum 4: the chi = 9 catapult runs (binary N = 4, eta between
  2/lambda_GF and eta_c) end at 0.58-1.00 x 2/eta per run, "not slightly below 2/eta; P3 is not simply
  confirmed"; Sec. 5.5 (05:364-365) has it right. Only the imbalanced small initializations end 0.05-3% below
  min(lambda_GF, 2/eta).
- Fix: `with sharpness at most $3\%$ below $2/\lr$ for imbalanced small initializations of binary $\Ns=4$, but
  $0.58$--$1.00$ times $2/\lr$ after a catapult from larger initialization (\cref{sec:other-optim})
  (\empirical;`.

### Section 5.6 (optimizers)

**M10. "Adam keeps the order" goes beyond the data, which support the first component.**

- Location: 05:544 heading "Adam: same order, but ..."; 01:129 "Adam, RMSProp: same order"; 01:291-292 "Adam and
  RMSProp keep the order"; 01:302 "the order kept by Adam" (listed as new); 04:1237 "Adam keeps the order";
  07:107 "Adam keeps the order"; table cell 05:641-642.
- Evidence (`optimizers/results/order_agreement_with_gf.csv`): first component as gradient flow 0.94-0.96 on the
  deep trees, 0.87-0.98 on binary4; full order 0.67-0.75 (Adam, deep), 0.69 (RMSProp), 0.83 (later_b8);
  `optimizers.md`: "full order 67-83%". Counting only runs where both gradient flow and the optimizer learn all
  three components, 80-88% (sub-review), the rest real swaps of the later components.
- Fix: heading `Adam: same first component, but the first escape forgets $\isc$`; elsewhere `keep(s) gradient
  flow's first component (94--100\% of resolved deep-tree runs) and mostly its order`; 01:302 `the first
  component kept by Adam`; table `first component as gradient flow: 94--100\% (deep), 87--98\% ($\Kc=3$); full
  order 87--98\% ($\Kc=3$), 67--83\% (deep)`.

**M11. Sec. 5.2 says Adam is rich "in the same sense" as the two-copy model; Sec. 5.6 says otherwise.**

- Location: 05:101-102 "Adam's runs from small initialization are rich, non-incremental in the same sense".
- Evidence: tab:optimizers (05:640) "rich ($\Kc=3$), S2S ($\Kc=7$, $\isc\le0.4$)"; `later_b8_summary.csv`: 24 of 24
  (eta = 1e-3) and 22 of 24 (1e-2) runs at sigma <= 0.4 are S2S; on binary4 Adam fails rho > 3 because the first
  escape is as wide as its own time (05:560-562), not because events crowd as in the two-copy model.
- Fix: `On binary4 Adam's runs are rich, non-incremental for another reason: the first escape is as wide as its
  own time, so 79--100\% fail $\sep i>3$; on binary $\Ns=8$ most are saddle-to-saddle
  (\cref{sec:other-optimizers}).`

**M12. "Within 0.2% of the balanced sharpness" holds only from balanced initializations.**

- Location: 05:665 "gradient flow, GD and heavy ball end within $0.2\%$ of it".
- Evidence (`theory_optimizer_P1P2.csv`): deep binary from fan-in initialization: ratios 1.001-1.074 (GF),
  1.053-1.106 (GD), 1.065-1.131 (heavy ball), balance index up to 0.35; within 0.2% only from balanced
  initializations and on binary4. `optimizers.md` (314-316) calls the fan-in part "only qualitative".
- Fix: `from balanced initializations (and on binary4) gradient flow, GD and heavy ball end within $0.2\%$ of
  it; from fan-in initialization on binary $\Ns=8$ they keep the initial imbalance (balance index up to
  $0.35$) and end up to $1.07$, $1.11$ and $1.13$ times sharper.`

**M13. Sec. 5.6 cites fix-round results that no verification has checked.**

- Location: Sec. 5.6 throughout; App. B 919-922; 04:999-1001 ("dead route ends in 11").
- Problem: every other experiment of Secs. 5-6 is cited from a verified version (`*_verification.md`);
  optimizers has only the round-1 review. Unverified fix-round outputs in the report: the GD delay slope
  0.0199, heavy ball's factor 19, route ends dead in 11 of 12 runs, the four-draw finite samples, the Gram
  re-runs (p = 0.65, 9/24, 147-632 vs 26-35 steps), 60/60 budget calls, 59/60 ties. CLAUDE.md still says the fix
  round is "running" and the note's header says "v0, in progress" (both stale). The sub-review recomputed all of
  these from the committed CSVs and found them consistent, but that is not an independent re-run.
- Fix: in App. B (919-922) add `Review: \path{notes/reviews/optimizers_round1.md}; the fix round (exact
  gradient-flow events, Gram re-runs, four-draw finite samples, route-end check, GD delay slope) was not
  re-verified independently.` Update the stale status lines.

### Sections 6.1-6.9 and App. A.5-A.6 (covariance and teachers)

**M14. App. A's star example contradicts Sec. 6.9 (and E7): "in all 5 forgets it".**

- Location: A:582-584 "gradient flow passes near $j$ in 4 of 5 seeds and in all 5 forgets it, learns $r$,
  $r'$ and relearns $j$".
- Evidence: the source (`theory_saddles.md` 4.6; sigma = 0.05 only) has peaks a_q = 0.95, 0.94, 0.68, 0.99 and a
  fifth seed with a_q <= 0.43 that never learns j; by Sec. 6.9's classes 0.68 is "blended" (peak <= 0.68;
  `r2_star_random_classes.csv`: MPS4 1 q-last, 4 near-q, 5 blended of 10); E7: near-q MPS4 seeds 3, 5, 8 keep q at
  sigma = 0.1 and seed 6 for sigma >= 0.025; "random-init runs never sit at theta_q" (E16, Sec. 6.9).
- Fix: `Dynamics \empirical{} (MPS $\Ns=4$, $\isc=0.05$, five seeds): $\ov j$ peaks at $0.68$--$0.99$ in four
  seeds and stays below $0.43$ in the fifth; every seed then learns $r$, $r'$ and relearns $j$, and none sits at
  $\params_j$; in the ten-seed runs of \cref{sec:teacher-star} some near-$j$ seeds keep $j$ (one for
  $\isc\ge0.025$, three more at $\isc=0.1$); with weights $1.2$ ($\lambda_{\max}(\Gamma)=1.70$) it plateaus at
  $j$.`

**M15. Correlated inputs, online SGD: "within 4%" is 6.7%.**

- Location: 06:279-280 "Online SGD (binary4, $\rho=0.3$, $\batch\ge256$, three seeds) crosses the same loss
  levels within $4\%$ of gradient flow."
- Evidence (`data_teacher/results/e3_sgd_check.csv`, re-computed): seed 1, B = 256, eta = 0.01: 8.6 vs 8.06
  (+6.7%) at level 0.9 and 9.0 vs 8.46 (+6.4%) at 0.5; seed 2: -3.8%; at B = 4096 within 1.9%. The 4% is in no
  note. Event times are on a grid of 0.2 (2.5% at t = 8).
- Fix: `crosses the same loss levels within $2\%$ of gradient flow at $\batch=4096$ and within $7\%$ at
  $\batch=256$ (three seeds, $\isc=0.05$; event times on a grid of $0.2$).`

**M16. "sigma^-1" and "lambda_min proportional to sigma" are claimed for HT, where only the saddle start was run.**

- Location: 06:371-372 (tab:cov-summary) "$\propto\varepsilon^{-1}$, $\isc^{-1}$ on binary and HT"; caption of
  fig:dt-contrast 06:205-206 "a higher-order saddle ($\lambda_{\min}\propto\isc$, escape
  $\propto\varepsilon^{-1}$ or $\isc^{-1}$) on binary and HT trees".
- Evidence: random initialization was run on MPS4 and binary4 only (06:124-126, 06:156-157; `r2_random_v2_*`
  has no HT rows); HT has saddle-start eps^-1 fits only.
- Fix (table): `$\propto\varepsilon^{-1}$ on binary and HT, $\propto\isc^{-1}$ on binary4 (random
  initialization not run on HT)`; (caption): `a higher-order saddle on binary and HT trees (escape
  $\propto\varepsilon^{-1}$; from random initialization, binary4: $\lambda_{\min}\propto\isc$, escape
  $\propto\isc^{-1}$)`.

**M17. "Plateaus ... blends rather than partial sums" is stated for all plateaus and all four settings.**

- Location: 06:35-37 "Plateaus sit at critical points of the bond-rank-constrained loss, blends rather than
  partial sums of components (\cref{sec:cov-misaligned,sec:cov-correlated,sec:teacher-cp,sec:teacher-ttn})".
- Evidence: shown for rank-one plateaus only; at higher ranks the critical points of the random TTN teachers
  coincide with SVD truncations (E15; 06:326-327); under correlated inputs the MPS4 rank-one saddle stays the
  pure component (`data_teacher.md` R1 step 3: |cos| 1.000 with e1^(x)4).
- Fix: `Rank-one plateaus sit at critical points of the bond-rank-constrained loss, blends rather than partial
  sums of components (on binary4 and MPS4; under correlated inputs only on binary4, the MPS4 rank-one point
  keeping the pure component)`.

**M18. Sec. 1 and a Sec. 6 caption extend the 0.4% / 1.3% agreement to every non-orthogonal teacher.**

- Location: 01:222-226 "Non-orthogonal teachers (overlapping components, misaligned covariance, random TTN
  teachers) ... with slopes within $0.4\%$ (from the saddle) and $1.3\%$ (from random initialization, MPS4) of
  the predicted rates"; caption fig:dt-mps-laws 06:249-250 "any non-orthogonality in the metric of the loss
  turns the $\isc^{-2}$ second plateau into a logarithmic one, with the predicted slope where there is a
  prediction".
- Evidence: the 1.3% covers the five N = 4, d = 2 teachers of Sec. 6.2; the misaligned teacher has no
  random-init prediction (only a bound, 06:144-145); Sec. 6.5 at c = 0.05: 595 vs 570 (+4.4%); Sec. 6.6 random
  MPS4 teachers (d = 3): up to 1.6%; E16 restricts the README's "<= 0.4% / <= 1.3%" accordingly.
- Fix (01): `... and $1.3\%$ (from random initialization: MPS4, $d=2$; $1.6\%$ for random MPS4 teachers with
  $d=3$; $+4\%$ at a CP overlap of $0.05$, not yet asymptotic; misaligned covariance: bound only)`; caption:
  `each of the three non-orthogonalities tested turns ... with the predicted slope where there is a prediction
  ($c=0.05$: $+4\%$, not yet asymptotic)`.

### Sections 6.10-6.11 (real data)

**M19. "In the main sweep the quadtree ends lowest" is false: the snake MPS of the same sweep ends lower.**

- Location: 06:559 "In the main sweep the quadtree ends lowest: at $\isc\le0.14$ a scrambled order ends below
  the quadtree of the same seed in 1 of 50 runs".
- Evidence: `real_data/results/fix1_analysis.md` ("Final train loss per seed, main sweep"): the MPS ends below
  the quadtree in 5 of 5 seeds at every sigma <= 0.3 (3 of 5 at 0.5); scrambled orders end below it in 14 of
  15 runs at sigma = 0.5; `notes/real_data.md` 8.4 (504-511) says the same.
- Fix: `Among the binary orders of the main sweep the quadtree ends lowest at small $\isc$ (the snake MPS ends
  lower still at every seed for $\isc\le0.3$; at $\isc=0.5$ scrambled orders end below the quadtree in 14 of
  15 runs):`.

**M20. "Route amplitude" means two different things, and the cited remark then says the opposite.**

- Location: 06:504-513 (definition of the route amplitude mu_uw, "The sibling pair with the largest mu_uw
  rises first ...", "Degree comes before amplitude ... (\cref{rem:A-sequence})"); also 01:134, 01:312, 07:136,
  B:1122.
- Problem: in Sec. 6.10 the route amplitude is the residual coupling (the c of rem:A-sequence); everywhere else
  it is the size of the dormant route entries (04:408 "The prefactor $A$ is set by the route amplitudes when
  the plateau starts"; A:658; A:696-697). rem:A-sequence(2) says ties of equal k are decided by "the coupling
  and head start $c\,b_{\rm eff}^{k-2}$, dominated for $k\ge3$ by the random route amplitudes"; read with the
  Sec. 6.10 definition this contradicts the Sec. 6.10 finding that mu decides.
- Evidence: `routes_raw.json` (96 rank-one states, sub-review): the largest sibling mu exceeds the second by a
  median factor 60 (quadtree, minimum 19) and 5 (scrambled), so on MNIST the coupling, not the dormant size,
  decides.
- Fix: rename mu_uw "route coupling" in the real-data text (and in 01, 07, B), or at least at 06:505:
  `\emph{route coupling} $\mu_{uw}$ (the coupling $c$ of \cref{rem:A-sequence}, not the size of the dormant
  route entries called route amplitudes elsewhere)`, and after "chance $\frac14$)": `Unlike in the teacher
  runs, the couplings themselves decide these ties: the largest sibling coupling exceeds the second by a median
  factor $60$ (quadtree) and $5$ (scrambled orders).`

**M21. The summaries claim more than Sec. 6.10 shows.**

- Location: 06:659-661 "The baseline theory predicts each change after the substitution in the table, except
  the slope under a misaligned covariance from random initialization, which is only bounded."; 06:385-386
  (tab:cov-summary, real data) "the pixel order sets the final loss, not through locality".
- Evidence: the same table row lists "plateaus creep", which nothing predicts (`real_data.md` section 5 row
  "flat plateaus ... | partly", 8.6); 06:537-538: MPS edges 1 and 2 prefer a power law, edge 0 is mixed, "which
  rank-one counting ... does not explain" (also Sec. 7 item 12); the order evidence is "neither necessary nor
  sufficient" (06:592), the three horizontal-pair local orders rank 1, 2 and 4, and the mechanism is a
  conjecture; the effect reverses at sigma = 0.5 (M19).
- Fix (06:659-661): `... which is only bounded, and, on MNIST, the creep of the plateaus and the power-law rises
  of MPS edges 0--2, which it does not explain.`; table: `the pixel order sets the final loss at small $\isc$
  (locality in the image neither necessary nor sufficient)`.

### Section 7 and Appendix A (A.1-A.4)

**M22. Sec. 7 item 1 asks for a saddle hopper that exists; the report never mentions it.**

- Location: 07:22-28 "\emph{What would settle it:} ... an implementation of the rule that hops from saddle to
  saddle (at each plateau, compute the fixed-rank block and the local residuals ..., follow the most unstable
  direction, re-optimize at the new rank profile), compared run by run with gradient flow ...".
- Evidence: `notes/saddle_hopper.md` v2 and `experiments/saddle_hopper/` implement exactly this and score it
  on the committed runs; the audit `notes/reviews/saddle_hopper_verification.md` (sections 6, 8) finds it
  reproducible but permissive: main-branch agreement 96.9% of 11,432 locality runs, 97.0% vs 95.8% for the
  trivial weight-order rule on the informative runs, 96.6% vs 96.6% on s2s_core; citable: the implementation, the
  multi-saddle accumulated-utility slopes of the TC teacher on MPS8 (five plateaus within 3.5%). PLAN.md lists
  it as citable. `grep -i hopper report/sections/*.tex`: no hit. Its section 9 (v2, after the audit, not
  verified) also bears on A.4 and Sec. 7 item 2: the order-6 regime of thm:A-order-sharp occurs at 30 of 587
  rank-one saddles of past runs (locality HT4 and HT8 with S = {0,1,2,3}, s2s_core odeco on HT4 and binary8, order
  binary8 two-component), with the polymer escape signature (x = 0.2998 vs 0.3000) and sigma^-4.03 in the one
  seed that takes it throughout; it accounts for 19 of the 88 locality sequence mismatches. Sec. 4.4 (04:437-446)
  names only the HT4 runs as failing the condition.
- Fix: in Sec. 7 item 1 replace the hopper half of "What would settle it" by `a hopper that implements the
  rule exists (\path{notes/saddle_hopper.md}; audited): it reproduces the theory's worked examples and the
  measured sequences, but no better than the weight order where it commits to one sequence (97.0\% against
  95.8\% of informative runs), so what is missing is predictive power: probabilities for its branches, and a
  saddle-order criterion at saddles with two or more learned components (the hopper applies
  \cref{thm:A-order-lower,thm:A-order-sharp} only at the 587 rank-one saddles)`; add one sentence with the safe-to-cite results in Sec. 4.4 or
  App. C; once its section 9 is verified, add the 30 order-6 saddles to the list at 04:437-446 and to item 2.

**M23. prop:A-hessian(d), tagged proven, is false as stated.**

- Location: A:542-544 "the Hessian kernel has dimension $\sum_er_e^2+\dim\params_{\rm new}-2\sum_{k^*=2\text{
  routes}}c_e$"; A:535 "with one learned component the negative one is $-\lambda$".
- Problem: it assumes (i) rank one for every two-core route block B = R_{u,v} (part (b) itself allows several
  sigma_k(B)) and (ii) a fixed-rank block nondegenerate modulo gauge.
- Evidence (exact Hessian, MPS4, d = chi = 3, j = 0000 with weight 1; sub-review script, re-run here): one residual
  1100 (0.5): kernel 59 = formula; residuals 1100 (0.5) and 2200 (0.4) on the same route: kernel 55, formula 59,
  negative eigenvalues -0.5, -0.5, -0.4, -0.4; tie 1010 with weight 1: kernel 64, formula 63.
  `theory_saddles.md` (637-645) has the same unstated hypotheses; its check C1 covers one-residual cases only.
- Fix: A:535 `with one learned component and one residual on the route the negative one is $-\lambda$`; (d):
  `If the fixed-rank block is nondegenerate modulo gauge (one learned component: $\tw j$ is not an eigenvalue of
  $\Gamma$), the Hessian kernel has dimension $\sum_er_e^2+\dim\params_{\rm new}-2\sum_{\{u,v\}}c_e\rank
  \mathcal R_{\{u,v\}}$, summed over the two-core routes`. Fix the note the same way.

**M24. rem:A-sequence(2) puts the wrong-order correction at ln(1/sigma); Sec. 4 measured O(1/|lambda|).**

- Location: A:692-694 "(wrong order, stars) the next event is a rotation of the learned structure or a $k=2$
  route, whichever is more unstable, in time $\propto\ln(1/\isc)$".
- Evidence: 04:778-780 "the correction in (iii) takes a time $O(1/\abs{\lambda_{\min}})$, not
  $\ln(1/\isc)/\abs{\lambda_{\min}}$"; `order_round2_verification.md` C13: |lambda|(t1 - t2) = 0.98-1.38, flat in
  sigma from 1e-4 to 1e-2; stars leave q within about 10 time units (`theory_saddles.md` 4.6).
- Fix: `in time $\ln(1/\varepsilon)/\abs{\lambda_{\min}}$, $\varepsilon$ the unstable deviation when the
  plateau starts ($O(1)$ after a wrong first escape from small initialization, so $O(1/\abs{\lambda_{\min}})$,
  \cref{sec:s2s-order})`.

**M25. rem:A-recipe is tagged proven but its last sentence is derived, and Sec. 1/4 tag it differently.**

- Location: A:173-178 "\proven{} at LO. ... and the dominant component of the final $u$ is the first one
  learned as $\isc\to0$: a parameter-free prediction of $\tev{1}$".
- Problem: the t* formula is proven at LO; turning it into t_1 needs prop:A-first-event (\derived), and naming
  the first component needs the KKT point reached to be a component configuration, which fails for stars
  (rem:A-sequence(1)). The appendix's own rule (A:30-31) asks for both tags. 01:158-159 tags the component
  identification "\derived{} \empirical"; 04:270-271 tags only the scaling and the identity proven.
- Fix: `\proven{} at LO for $t^*$; \derived{} for $\tev1$ (\cref{prop:A-first-event}) and for the first
  component, which needs the KKT point reached to be a component configuration (\cref{prop:A-component-hessian};
  not for stars, \cref{rem:A-sequence})`.

**M26. prop:A-head-start omits its hypothesis s_r < s_j.**

- Location: A:632-634 "Let the initialization be aligned: every core of the learned component $j$ and of the
  route starts at amplitude $\isc$ ...".
- Problem: both laws need s_r < s_j: for s_r > s_j eq. A-dt-k2 gives a negative Delta t_2 and for k >= 3
  b_eff^-(k-2) < 0 (the route blows up first). `theory_saddles.md` Table A assumes s_r < s_q; the
  tab:A-route-degree caption says "With the larger component learned first".
- Fix: `Let $\tw r<\tw j$ (the larger component is learned first) and let the initialization be aligned: ...`.

**M27. The binary4 closed form is compared with the data the wrong way round.**

- Location: A:315-316 "the median is $\tev1\approx\pi d/(2\tw1\isc\sqrt{\bd b})$ ... ($0.80$--$1.00$ of measured
  medians \empirical)".
- Evidence: `theory_escape.md` 4.4 (380-382): "Measured median / (pi d/(2 sqrt chi_b)): 0.84-0.95 ... The closed
  form overestimates"; prediction 3: c in [0.80, 0.95] and [0.92, 1.00].
- Fix: `(the measured medians are $0.80$--$1.00$ times this value \empirical: the closed form overestimates)`.

**M28. "New" tags where prior work is listed (citation_check part b) or the novelty was withdrawn.**

- 05:25 "\proven{} \newresult[on every tree]" for rank(Theta) = dimension of the fixed-rank manifold: B20 calls
  this "same in geometric form" (Uschmajew & Vandereycken 2013; Holtz et al. 2012). A:1303-1305 is right. Fix:
  `\proven{} \newresult[the mean on general trees and the full-bond condition; the rank is the dimension of the
  fixed-rank manifold, \citealp{uschmajew2013geometry,holtz2012manifolds}; for MPS and TT the mean kernel is
  implicit in \citealp{guo2021neural,wesel2025tensor}]`.
- 06:69-71 "a learned component $a$ of binary4 is unstable toward $b$ ... iff ... \proven{} \newresult{}" and
  A:1097 (prop:A-piggyback-cov, "\newresult{}"): binary4 is a deep linear network; with prop:A-precond this is
  Achour et al. 2024, Thm 7 (round-1 review; B4). Fix: `\newresult[for TTNs; on binary4, a deep linear network,
  it follows from \citealp[Thm~7]{achour2024loss} with \cref{prop:A-precond}]`.
- A:599 rem:A-wrong-order "\derived{} \newresult{}": its case (1) covers binary4, whose strict wrong-order saddle
  and rotation are known; Sec. 4 withdrew "the binary4 results (greedy limit, strict wrong-order saddle,
  rotation) as new" (04:40-41, 749); `order_round2_verification.md` C11. Fix: `\newresult[beyond binary $\Ns=4$;
  there, a deep linear network, the strict wrong-order saddle is known: \citealp[Thm~7]{achour2024loss}]`.
- A:59 lem:A-unused "\newresult[elementary]": B25 lists the embedding principle (Zhang, Zhang, Luo & Xu,
  NeurIPS 2021; Simsek et al. 2021) as a generalization; neither is in references.bib. Fix:
  `\newresult[elementary; the TTN case of the embedding principle, \citealp{zhang2021embedding,simsek2021geometry}]`
  and add both entries (metadata: `notes/literature.md`).
- Lower weight: A:321 prop:A-race bare "\newresult{}" (B9: Ben Arous et al., same tensor-vs-matrix dichotomy;
  cited in 01:245 and Sec. 4 but not here): `\newresult[the race on trees; the dichotomy is known in multi-spike
  tensor PCA, \citealp{benarous2025stochastic}]`; A:926 "The rank-one plateau. \derived{} \newresult{}" (B12:
  first half a special case of Razin et al. 2021, Thm 2, which 06:328 cites): add `(in CP factorization:
  \citealp{razin2021implicit})`.

## Minor

Format: location "quoted text" | problem / evidence | fix. "(sub)" marks items taken from a sub-review and
not re-run by the lead reviewer.

### Section 5.1-5.4, A.8

- m1. 05:119, A:1404 "for $\Kc\ge5$ the route can die first" | traps already at K = 4 (MPS4: 32 of 80, 04:986-994);
  `theory_regimes.md` calls K >= 5 not sharp | add `(already at $\Kc=4$ for some initializations, \cref{sec:s2s-failure})`.
- m2. 05:122 "never learn the last component" | 9 of 18 fan-in MPS8 runs learn only one component
  (`stall_test_runs.csv`, sub) | `never learn the last component within the budget (9 of 18 fan-in MPS8 runs
  learn only the first)`.
- m3. 05:156-157 stall classes "dead / decaying / long plateau" | thresholds on the end value, not trends; 182 of
  the 795 runs stopped at the step limit; some "long plateau" runs at sigma >= 0.77 learned nothing (sub) |
  `... end with the $\ncomp$-th singular value at roundoff (546), between $10^{-12}$ and $10^{-4}$ (212) or above
  (37); 182 of them stopped at the step limit`.
- m4. 05:51-52 "lazy in 2 of 5 at $\isc=1.42$ and in 4 of 5 or 6 at each $\isc$ from $2.47$ to $7.6$" | every
  chi = 32 run at sigma >= 1.42 hit the step limit (lazy ones below 1e-3); 2 of 3 at 1.87 and 2 of 5 at 10 not
  quoted (sub) | add both facts.
- m5. 05:64-65 "($\dTK^{\rm probe}$ of the binary4 rich runs: $0.10$ to $10^7$, median 68)" | centred runs only
  (all binary4 rich runs: median 47, max 9.4e7; sub) | `of the centred binary4 rich runs`.
- m6. 05:57-59 "the root moving by more than $100\%$ while $\dTK<0.1$" under \derived | one theory-note run
  (spectral norm +214%, P19) | tag `\empirical{} (one run)`.
- m7. 05:92 "at the lazy residual from $\isc=16$ on" | at sigma = 15.9 the two-copy runs end 1.3-3.4% below
  their residuals; within 0.3% from sigma = 25 (`lazy_pred_b4.csv`, sub) | `from $\isc=25$ on (within $3.4\%$
  at $16$)`.
- m8. 05:93 "A light root with a deficient bond behaves the same way, lazy and then rich
  (\cref{sec:regime-imbalance,app:boundaries})" | one theory-note seed; Sec. 3 (03:213) shows only floor stalls at
  chi = 3 (sub) | cite the seed (`theory_regimes.md` P20) and drop the Sec. 3 pointer, or tag \conjecture.
- m9. 05:187, 199, 206, A:1417 "fixed $\xi$" | muP-TTN rescales the root by (2 chi)^-1/2, so xi grows with chi
  (6.3-17.8 at sigma = 1.5); R12 is stated at fixed sigma (sub, uncertain) | `fixed $\isc$ (fixed $\xi$ under SP
  and NTK)`.
- m10. 05:209-211, A:1418-1419 factor-wise rule "makes the initial kernel ... vanish like $\chi^{-2}$" | from
  two-seed theory checks (P17), not from the binary8 runs called "not interpreted" one sentence earlier | add
  `(theory checks, two seeds, $\chi=4$--$16$)`.
- m11. tab:large-chi caption "in brackets the prediction of \cref{app:boundaries}" | A.8 gives only the kernel
  exponents | add the parameter, root and leaf exponents to A.8 or cite `theory_regimes.md` 4.4.
- m12. A:1352 "against $2.25$--$2.82$, $6.73$--$7.53$ in gradient flow" | two seeds; 7.53 is 3% outside the
  predicted [6.67, 7.31] | say so and point to the six-seed tensor-kernel test of Sec. 3.
- m13. A:1366-1367 "runs above it are stalled" | at sigma = 0.63 (above 0.56) 4 of 6 runs converge | `runs well
  above it are stalled (median seed from $\isc\approx0.64$, every seed from $1.0$)`.
- m14. A:1370 "$\vartheta=0.37$--$0.70$" | definition and sample missing | `(time to $\Loss/\Loss(0)=0.3$;
  two seeds)`.
- m15. A:1398-1400 "median per-seed $\isc_c=0.27$--$0.31$ ... and $0.71$" | plain-model values (centred map:
  0.237; 0.83 [0.72, 0.97] at K = 7; sub) | add `(plain model)`.
- m16. A:1401 and 07:90 "all large-$\Kc$ runs have $d=2$" | plain d = 3 sweeps run up to K = 15 but stall
  throughout | `all large-$\Kc$ boundary estimates are at $d=2$`.

### Section 5.5 and A.7

- m17. 05:401 "the commuting-pair estimate of \cref{app:optimizer} gives 26 for this teacher" and A:1286-1287
  "($1.08$ from a commuting-pair estimate; about 33 for weights $(3,2,1)$ ...: $c_\gamma$ depends on teacher and
  tree)" | A.7 contains neither the estimate's formula nor 26 and reads as if the estimate were 33;
  `theory_optimizer.md` 660-663 gives the formula, 1.08 and about 26; both tests are on binary N = 4 (tree not
  varied) | A.7: `($1.08$ from the commuting-pair estimate of \path{notes/theory_optimizer.md}, which gives $26$
  for weights $(3,2,1)$, $d=\chi=3$, where the fit gives $32.6$, \cref{sec:other-optim}; $c_\gamma$ depends on the
  teacher, other trees untested)`; add ", $\chi=2$" after "(weights $(1,0.6)$".
- m18. A:1215-1216 "Checked (binary $\Ns=4$, $\lr=1.05\lr_c$): late loss $1.45\cdot10^{-2}$ ... $1.53\cdot10^{-2}$
  ...; divergence at $1.5$--$1.65\lr_c$" | teacher (1, 0.6, 0.3), sigma = 0.05 (theory note V2), not named; with
  s_1 = 3 (Sec. 5.5) eq. A-eos-loss gives 0.138 and Sec. 5.5 reports divergence at 1.5-1.75 (`p5_excess_loss.csv`:
  0.1306 vs 0.1306 map) | name the teacher and add `with the teacher $(3,2,1)$ of \cref{sec:other-optim}: $0.1306$
  against $0.1306$ (map) and $0.138$ \eqref{eq:A-eos-loss}`.
- m19. 05:309-310 "Since $c_1$ grows like $\ln(1/\isc)$" (in a \derived paragraph) and 05:539-541 "the asymptotic
  slope $\lr(\Kc-1)/2=0.020$ that follows from \cref{prop:A-gd-shift}" | A:1232-1233 tags this asymptotic form
  \conjecture{} (heuristic) | add `(asymptotically, \conjecture{} in \cref{app:optimizer}; confirmed here)`.
- m20. 05:540 "Heavy ball's delay grows $19.0=(1+\beta)/(1-\beta)$ times faster" | measured 0.37642/0.019912 =
  18.9 (sub) | `$18.9\approx(1+\beta)/(1-\beta)=19$`.
- m21. 05:311 "$0.0099$" | `gd_sweep.csv` mean 0.009845 (sub) | `$0.0098$`.
- m22. 05:346-347 "already oscillate or catapult at $0.9\lr_c$" | 2 of 5 runs on binary N = 8, 3 of 5 on the MPS
  (sub) | `some runs (2 of 5 on binary $\Ns=8$, 3 of 5 on the MPS) already ...`.
- m23. 05:385-386 "$p=0.43$--$0.52$ ... against $\frac12$" | sigma = 0.1: 0.434 +- 0.020, three s.e. low
  (`sgd_fits.csv`) | add `(three s.e.\ low at $\isc=0.1$)`.
- m24. 05:422-423 "because Gaussian product features have kurtosis $3^\Ns$" | 81 explains part of the factor 144;
  `optim_samples.md` 425: effective kurtosis about 145 = 1.8 x 3^4 | add `and the coupling to other patterns a
  factor $1.8$`.
- m25. 05:429-430 "these are also the only runs whose order differs ... (10) or is unresolved (19)" | 21 unresolved
  runs, 19 at B = 1 and two at B = 4 (sub) | `(19 at $\batch=1$)`.
- m26. 05:457-459, 05:479-480, 07:147-148 "from $\nsamp=384$ at $\chi=16$", "200--250 samples or 5--9 times the
  teacher-manifold dimension in both settings" | chi = 16 run at sigma = 0.5 only; 200-250/48 = 4.2-5.2, so 4-9
  times, and the threshold is for chi = 4 students | `... at $\chi=16$ ($\isc=0.5$ only)`; `200--250 samples or 4--9
  times the teacher-manifold dimension with $\chi=4$ students`.
- m27. fig:optim-sgd caption (a) "second plateau (open; three initializations ...)" | symbols exist only for
  sigma = 0.03, 0.1 (one initialization at 0.3; `plot_report.py` needs three) | `(open; $\isc=0.03$, $0.1$)`.
- m28. fig:optim-gd caption "$x=q/\tw1^{1/\Kc}$" | A.7 uses q_r = s_r^(1/K) and x = a/s_1^(1/K) | `$x=a/\tw1^{1/\Kc}$, $a$
  the per-core amplitude`.
- m29. A:1269 "SGD near the minimum diverges at $\lr_{\rm as}$" | checked on N = 4 only | add `(checked on $\Ns=4$)`.
- m30. A:1288-1290 "no learned component was erased in 13 runs at $\lr\le0.25\lr^*_{\rm ms}$ ... 20 of 36 longer
  runs" | the (1, 0.6) runs reached 0.25-0.32 of the learned-block threshold, not of eq. A-ms-threshold; "longer"
  is unsupported (about 460 vs 480 time units per run; sub) | `in 13 runs at up to $0.25$--$0.32$ of the
  learned-block threshold ... in 20 of 36 runs of up to 1000 time units at $0.25$--$0.30\lr^*_{\rm ms}$`.
- m31. A:1237-1240 | the second half of citation_check row 42 (Corlouer et al.'s printed Ito drift, eq. 6, is not
  covariant under SGD's rescaling symmetry) is not applied | add the row's suggested sentence after
  eq. A-sgd-first.

### Section 5.6

- m32. 05:686-687 "its own least-squares solution, $\sim\isc^{-(\Kc-1)}$ times larger than the core, so its first
  step overshoots the loss by $\sim\isc^{-2\Kc(\Kc-1)}$ (\derived; up to $4\cdot10^{34}$)" | in the aligned scalar
  model (`optimizers.md` 1.7) the least-squares core has size s sigma^-(K-1), i.e. sigma^-K times the core; with
  the printed ratio the overshoot would be sigma^-2K(K-2) (the corrected v1 error). The final exponent is right
  (fitted slopes -11.9, -12.0). 4e34 is a per-sigma median; single runs reach 4.9e36 (binary4) and 7.5e52 (deep)
  (sub) | `of size $\sim\isc^{-(\Kc-1)}$ ($\isc^{-\Kc}$ times the core), so its first step overshoots the loss by
  $\sim\isc^{-2\Kc(\Kc-1)}$ (\derived{} in the aligned scalar model; medians up to $4\cdot10^{34}\Loss_0$ on binary4,
  single runs up to $5\cdot10^{36}$, deep trees $10^{52}$)`; same at B:975-976.
- m33. 05:557-558 and tab:optimizers "gradient flow has $5.00\pm0.02$ ($\isc\le0.4$ ...; fit computed from the
  committed runs in the review of this experiment)" | from superseded runs; `later_b8_summary.csv`: 4.991 +- 0.002;
  Adam's 0.049 and 0.145 are fitted over sigma in [0.25, 0.8] (on sigma <= 0.4: 0.09 and 0.17; sub) | `$4.991\pm0.002$`
  and state both ranges.
- m34. 05:659 "up to $8\cdot10^4$ steps at $\isc=0.25$" | median 8.1e4, range up to 1.32e5 (sub) | `median
  $8\cdot10^4$, up to $1.3\cdot10^5$ steps`.
- m35. 05:717-718 "relative drift is $10^{-11}$, $10^{-5}$ and $3\cdot10^{-4}$ under gradient flow, GD and heavy
  ball" | one run (binary4, sigma = 0.008, seed 0); over all binary4 runs GD 4e-5 to 2.2e-3 and heavy ball 1e-3 to
  4.9e-2, growing with sigma (sub) | `$\sim10^{-11}$, and, growing with $\isc$, $4\cdot10^{-5}$--$2\cdot10^{-3}$ and
  $10^{-3}$--$5\cdot10^{-2}$`.
- m36. 05:551-552 "Adam sees $\isc$ only through $\isc/\lr$, and the number $n_1$ ... stays bounded" | the bound
  needs one more step (`optimizers.md` 1.2(b)) | insert `; its first step is $-\lr\,\mathrm{sign}\,\nabla\Loss(\params_0)$,
  after which $\abs{\params}\sim\lr$ and the initial gradient weighs $\sim(\isc/\lr)^{\Kc-1}$ in the averages,
  so the trajectory has a limit as $\isc/\lr\to0$`.
- m37. 05:591-594 epsilon test "measured/predicted is $0.9989$--$1.0003$ ... and $1.0009$--$1.0062$ ...; near
  $\isc_\times\approx0.05$--$0.1$ the deviations are systematic" | at eta = 1e-2, sigma = 0.03 the deviations are
  +1.0% to +3.5%, covered by neither range; below the crossover the crawl term is > 98% of n_1, so the test
  checks "Adam = gradient flow at t = eta n/eps", not the additive form (sub) | give the sigma of each range and
  add that sentence.
- m38. tab:optimizers cells | Gram "first escape" holds the exponent of the time to 1e-3 L_0 and omits the flat time
  at eta = 0.1-0.5; heavy-ball cost is T_GF(1 - beta)/eta; GF "evaluations" are counted to 1e-6 L_0, not to the
  caption's 1e-3 L_0; re-gauged "p = 0.00" vs the text's |p| <= 0.02, "sigma-free (K = 8)" not measured (sigma = 0.3
  and 0.8 differ by 27-44%), "6/6 (K = 3)" only when re-gauging every step; Riemannian "one step (K = 3)" in 5 of 6
  seeds (sub) | correct the cells accordingly.
- m39. 05:677 "it fits all components in one step on binary $\Ns=4$" | 5 of 6 seeds; reaching 1e-3 L_0 takes 2
  steps (sub) | `within two steps (all three cross $\frac12$ at step 1 in 5 of 6 seeds)`.
- m40. 05:692-694 "gradient flow and Adam reach the same population loss" | at n = 45 paired differences -5.1 to
  +4.0 (L_0 = 7) with no consistent sign (sub) | `show no consistent difference in population loss`.
- m41. 05:555-556 "RMSProp alike (\cref{fig:optimizers-adam-escape})" | the figure shows Adam only | cite the
  RMSProp exponents (0.012 +- 0.004, 0.001 +- 0.019).
- m42. Captions: fig:optimizers-side shows an Adam run labelled S2S (rho = 7.1, 45), atypical (most binary4 Adam
  runs are rich); "60 ALS and DMRG runs" are 60 (problem, method, seed) combinations; fig:optimizers-adam-escape
  "121--139 ... 23.5--29.3 steps" are medians per sigma (runs 111-140 and 21.5-30.9) (sub) | say so.
- m43. 05:709-711 "These methods are equivariant under the whole gauge group ... and see $\W_0$ only through its
  frames, so they keep neither plateaus (\derived) nor $\isc$ (\proven)" | re-gauged GD is equivariant and sees the
  frames too, yet is sequential on deep trees; the missing plateaus come from the O(1) rate; "nor sigma" is exact
  for Riemannian GD only at eta = 1 | `so they keep no $\isc$ (\proven{} for ALS and DMRG, exact for Riemannian GD
  at $\lr=1$); because they move $\W$ at a rate of order one they keep no plateaus (\derived); re-gauged GD shares
  the equivariance and the $\isc$-independence but not the rate`. Consider citing B19 (Sabanayagam & Lucey,
  Prop. 9) in the Gram paragraph.

### Sections 6.1-6.9, A.5-A.6

- m44. 06:408 "$w=1.4138$ ($\lambda_{\max}(\Gamma)/\tw q=0.9998$)" | 1.4138/sqrt 2 = 0.99971; each bracket point has
  three seeds (`r2_star_threshold_scan.csv`; the verification table has the same slip) | `0.9997`, `(three seeds
  each)`.
- m45. 06:409 "At $\varepsilon=10^{-2}$ it moves to $1.43$--$1.44$" | MPS4 and binary4 only; HT4 stays in
  [1.40, 1.42] (`r2_star_threshold.csv`) | add `on MPS4 and binary4`.
- m46. 06:383 "rates within $0.2\%$" | the body says 0.21% (MPS4, w = 1.6: -0.21%) | `$0.21\%$`.
- m47. 06:377 "MPS4 rank-one maxima merge at $c=1/2$" | equal weights only; for (1, 0.7) the fold is near
  c = 0.36 (sub; consistent with the measured bracket 0.35-0.45) | `merge at $c=1/2$ (equal weights; fold at
  $c\approx0.36$ for $(1,0.7)$)`.
- m48. 06:202 caption "geometric means over four or five seeds" | some points rest on 1-3 seeds
  (`r2_random_v2_runs.csv`: randbin_303 MPS one seed at sigma >= 0.03; sub) | `over one to five seeds (four or five
  at the three smallest $\isc$)`.
- m49. 06:222-224 "median $\abs{\cos}\ge0.998$ ... for $\theta\le30^\circ$, over the resolved plateaus (20--50\%
  of the MPS4 runs per $\theta$ have none at $\isc=0.1$)" | the theta <= 30 deg scope should cover the level errors
  too; at 45 deg (MPS4, beta = 1) 2 of 20 resolved, level error 5.8%, |cos| 0.53, and 65% and 90% unresolved (sub) |
  extend the scope and add the 45 deg numbers.
- m50. 06:95-96 "median relative error $10^{-8}$--$10^{-3}$" | regular plateaus; binary4 T2 runs on their way to a
  swap sit at 1-3% (`e1_plateaus.csv`, sub) | add `(runs that later swap: 1--3\%)`.
- m51. 06:304 "an $\isc$-independent plateau curvature" (all c > 0) | at c = 0.05 lambda_min is -0.0018 at
  sigma <= 3e-3, -0.0029 at 0.03, -0.022 at 0.12 (`e4a_sigma_v2_runs.csv`, sub) | `a plateau curvature that tends
  to its exact-saddle value as $\isc$ decreases`.
- m52. 06:53-54 "standard error over 5--10 seeds" | several fits use four seeds (c = 0.2 random init, correlated
  runs, teacher 102; verification M2) | `4--10`.
- m53. 06:322 "$\alpha=0.72$, $0.83$, $0.41$, ..." | 0.41 rests on one seed (`e4b_p1_exponents.csv`) | `(0.41: one
  seed)`.
- m54. 06:347 "$\isc=0.3$, $0.4$, $0.5$, eight seeds" | teachers 201 and 202 at sigma = 0.4 only (`p11_counts.csv`) |
  add `(two teachers at $0.4$ only)`.
- m55. 06:353 "grows the predicted bond first in 2 of 24 runs" | 2/24 is the onset re-analysis (f = 0.03, 0.05; 1/24
  at 0.1); the note's spectral rule gives 6/24 (13-38%; `e4c_theory.csv`) | name the event rule and add
  `e4c_cross_v2_summary.csv` to the provenance.
- m56. 06:277-278 "$14$--$19\to139$--$185$ on binary4" | pools rho = 0-0.4; like for like at rho = 0.4: 19 -> 185
  (`r2_corr_runs.csv`) | `$19\to185$ on binary4 at $\rho=0.4$`.
- m57. 06:270 "Second escape \newresult{}" | the verification's safe-to-cite row 11 classes the correlated-input
  second escape as a replication (06:275-276 says so) | keep \newresult on the prediction (06:264) only.
- m58. 06:420-426 random-initialization stars "$q$ is relearned last" | E7: near-q MPS4 seeds 3, 5, 8 keep q at
  sigma = 0.1 (min a_q 0.53-0.99), seed 6 for sigma >= 0.025 | `$q$ is relearned last in the runs that lose it; at
  $\isc=0.1$ the near-$q$ seeds keep it`.
- m59. 06:291 "on binary4 the rank-one point is the top singular pair of the $(01|23)$ matricization ...
  (\cref{prop:A-two-components})" | that proposition does not contain the binary4 statement | cite
  \cref{rem:A-binary4-dln}.
- m60. Related work not cited (citation_check part b): 06:98-100 (aligned covariance) B13 also lists Rawal &
  DeWeese (Thm 4.2, Prop. 4.5); 06:407 (star forgetting) B4 lists deactivation results (Pesme & Flammarion 2023;
  AGF Lemma B.1) | add as related.
- m61. A:932 "In the cases computed it lies $6$--$12\%$ below the HSVD truncation" | `theory_teachers.md` 2.3 gives
  6.6% of L_0 and 6-10% of the plateau value; 12% not found (sub, uncertain) | check and correct.

### Sections 6.10-6.11

- m62. 06:488-490 "so counts are per (order, seed) or per \emph{unit} ..., never per run" | the section then gives
  per-state counts across sigma (96 of 124 states; all 96; 91 of 96; 95 of 96; 1 of 50 runs), the kind the round-1
  review flagged; E4 allows 91/96 only next to the unit counts | `so independent counts are per (order, seed) or
  per unit; per-state counts (one state per run, repeating a critical point over $\isc$) are given only next to
  them`.
- m63. 06:494-497 "$t_a\isc^{\Kc-2}$ is constant per seed ... This holds for any input law, since as $\isc\to0$ the
  flow from $\isc Z$ is the flow from $Z$ on a clock slowed by $\isc^{-(\Kc-2)}$" | the identity is asymptotic, and
  it is the flow of <T, W> (the quadratic term keeps a factor sigma^K); the \proven tag is right for the limit |
  `As $\isc\to0$, $t_a\isc^{\Kc-2}$ tends to a per-seed constant for any input law: with $\params=\isc\tilde\params$
  and $s=\isc^{\Kc-2}t$, $\tilde\params$ follows the flow of $\ip{T}{\W}$ from $Z$ up to terms of relative order
  $\isc^{\Kc}$`.
- m64. 06:520-522 "The verdicts hold for all 124 binary runs, each point attributed ... (91 points kept" | 33 runs
  excluded (`real_data.md` 8.1) | `on the full sweep of 124 binary runs (91 points kept, 33 excluded with their
  reason; ten seed series)`.
- m65. 06:525 "Binary16 at the rank-one plateau" | quadtree runs only (`hessians_raw.json`) | `Binary16 (quadtree)`.
- m66. 06:530 "has two or three negative directions" | eigenvalues below -1e-3 lambda_max; 2-3 only at sigma <= 0.1
  (4-6 at 0.3, 10 at 0.5; sub) | `two or three eigenvalues below $-10^{-3}\lambda_{\max}$ at $\isc\le0.1$`.
- m67. 06:540-542 "the S2S signature holds on binary trees (quadtree: median smallest separation ratio $2.8$ at
  $\isc=0.5$, $24$ at $0.05$)" | 2.8 is below the threshold 3; 40% of quadtree runs are S2S at 0.5 (`tables.md`) |
  `the separation ratios on binary trees grow as $\isc\to0$ (quadtree: median smallest ratio $2.8$ at $\isc=0.5$,
  $24$ at $0.05$; all quadtree runs saddle-to-saddle at $\isc\le0.07$), not on the MPS`.
- m68. 06:625 "Only snake chains were run" and 07:141 "(snake order only)" | App. B (1156) lists three MPS chains in
  other orders (review runs; `real_data.md` 8.5: 0.083, 0.084, 0.128) | `Only snake chains were swept (three chains
  in other orders, one seed, are review runs)`.
- m69. 06:485-486 "three scrambled binary orders, seven $\isc$" | scrambled2 has four sigma; whitened runs are part
  of the sweep (tab:exp-real-data) | add both.
- m70. 06:490 "$\pm$ is the standard error over seeds" | not for 1.22 +- 0.03 (over 11 series) nor for the Fashion
  exponent 0.03 +- 0.09 (pooled fit) | append `(over series for the gap exponent; of the pooled fit for the
  Fashion-MNIST exponent)`.
- m71. 06:633-634 "confounded by 23 runs stopped at the wall-clock limit" | the sigma = 0.5 confound is that all
  five quadtree runs there stopped early (`chi8_tables.md`) | `confounded, since all five quadtree runs at
  $\isc=0.5$ stopped at the wall-clock limit (23 of the 45 runs did)`.
- m72. 06:642 "at $\isc\ge2$ the runs follow the linearized model" | at sigma = 2 up to 1.5% from the linear model
  and one run labelled rich; <= 5e-5 from sigma = 2.8 (`regime_analysis.json`, sub) | `at $\isc\ge2.8$`.
- m73. 06:590-591 "ranks the orders by that loss" (also caption 06:613, 01:312, 07:137) | Spearman -0.60 / -0.55 is a
  moderate rank correlation | `predicts the ranking of the orders by that loss (Spearman $-0.60$; $-0.55$ within
  random orders)`.
- m74. 06:596 "a random order with $S=0.994$ ranks third" (offered against the necessity of locality) |
  scrambled110 has 4 of its 8 bottom pairs adjacent in the image (chance about 1.6); scrambled112 (S = 0.990, no
  adjacent pair) ranks sixth and is the clean example (`extra_orders_predictor.json`, sub) | cite both.
- m75. 06:595-596 "(the quadtree's outcome, the lowest, was seen before registration)" | two of the three
  transposed-quadtree seeds were seen too (`real_data.md` 9.3) | add.
- m76. 06:474 "with $T$ for $\Wt$" | T replaces W* only in the first escape; later saddles are critical points of
  the C-weighted loss, and only the route counting carries over | `with $T$ for $\Wt$ in the first escape; the
  route counting at the rank-one point does not depend on the target`.
- m77. 06:585 "Deviations. None in the design or the analysis." | the experiment's own list (`real_data.md` 9.6)
  includes the 8-process budget exceeded for a few minutes | say so.

### Appendix A.1-A.4 and Section 7

- m78. A:169 "Checked per initialization (within $0.8\%$ of gradient flow for $\Kc=3,4$, $\isc\le0.03$)" | 0.8% is
  binary4 at sigma = 0.01 (MPS4: 0.1% at 0.03), for t* without the offset (`theory_escape.md` table 2.4) | `without
  the offset, gradient flow deviates from $t^*$ by at most $0.8\%$ (binary $\Ns=4$, $\isc=0.01$) and $0.1\%$ (MPS
  $\Ns=4$, $\isc=0.03$), and the deviation equals the $\delta_\Kc$ term to two digits`.
- m79. A:344 "the race predicts the first component on $94$--$100\%$ of initializations" | compared with the
  two-component leading-order flow, not with gradient flow (`theory_escape.md` 5.2; gradient flow: 328/328
  binary8, 152/160 HT4, 04:356-358) | `names the winner of the two-component leading-order flow on ...`.
- m80. A:467 "for $k^*\in\{3,4\}$ with a positive definite fixed-rank block" | the block always has gauge zero
  modes; A:839 and the note say "modulo gauge" | add `modulo gauge`.
- m81. A:234 "partners sharing a terminal core: \cref{prop:A-piggyback-graph}" | that proposition is about the loss
  Hessian at theta_S, not the spherical Hessian of g (`theory_escape.md` 3.3, `theory_saddles.md` 4.5) | `partners
  sharing a terminal core give $\Kc^{1-\Kc/2}(\Gamma-\tw j\Id)$ on the rotations, so $\params_j$ is a local maximum
  if $\lambda_{\max}(\Gamma)<\tw j$ \derived`.
- m82. A:611-612 "Checked on binary and MPS $\Ns=4$, before the random streams ... were separated" | the order
  experiments re-checked the branches with separated streams (`order_round2_verification.md` C11-C14) | add
  `and with separated streams in \cref{sec:s2s-order}`.
- m83. A:711-713, 732 (prop:A-trap) | B5 lists related work not cited: Kumar & Haupt's shrinkage branch (TMLR,
  Lemma 2), AGF Lemma B.1, and Jacot et al. App. C.1 (balanced initialization excludes convergence of the LO flow
  to a saddle in deep linear networks, which supports the conjecture at A:732) | add to the bracket.
- m84. A:397 lem:A-routes bare "\newresult{}" (uncertain) | Furman et al.'s straight-line computation
  (T(t) = T + t^N U, no lower-order Gauss-Newton term; Lemmas D.7-D.8) is the whole-component case of "a route pays
  no Gauss-Newton cost at the order of its gain" | `\newresult[for sub-routes; for a whole connected rank-deficient
  component: \citealp{furman2026benign}, Lemmas~D.7--D.8]`.
- m85. A:870-872 (A.4) "measured on 32 instances ..., 12 of them with independent code, with slopes within $0.003$"
  | 2 of the 32 are the borderline group D (lambda = s_j), which has no prediction; the 12 independent instances
  were chosen independently, not re-runs of a subset, and agree within 5e-4 (`theory_saddle_order.md` 5;
  numerics verification) | `on 32 instances (30 with a prediction; 2 at the boundary $\lambda=\tw j$ have order
  $k^*$), ... and on 12 further instances computed with independent code (slopes within $5\cdot10^{-4}$)`.
- m86. A:846-848 cor:A-order(iv) "for a pattern teacher with the index condition" | thm:A-order-sharp also needs
  all s_r > 0 (proof verification, section 7) | `for a pattern teacher with all $\tw r>0$ and the index condition`.
- m87. A:872-874 "the cluster obeys $\dot b=c\,b^{2\ell-1}$" | `theory_saddle_order.md` 6: with two multi-core nodes
  (HT4 1111, binary8) the reduced model is a coupled polymer system that needs the polymers to align in sign (not
  checked) | add `(one multi-core node; with several, e.g.\ HT4 with $D$ all sites, a coupled polymer system,
  not checked)`.
- m88. A:875-878 "gradient flow from $\params_S$ plus a perturbation of size $\isc$ escapes ... then forgets $j$ and
  relearns it later" | the forgetting was observed in the verifier's runs from small initialization (`gf_origin`),
  not in the runs from theta_S | attribute: `... along $110000$; in the runs from small initialization that visit
  $\params_S$ the flow then forgets $j$ and relearns it later`.
- m89. Sec. 7 item 12 (07:141-143) "two edges rise like power laws where rank-one counting predicts logarithmic
  routes" | Sec. 6.10 (06:537-538) says edges 1 and 2 prefer a power law and edge 0 is mixed | `two edges rise
  like power laws and a third is mixed`.
- m90. 01:249-252 (outside the assigned sections, uncertain) "the ends are decided long before the route could
  escape once $\Kc\ge5$ \derived{} (reduced model)" | `theory_saddles.md` files this time-scale comparison under
  "Timescales and sigma -> 0 (conjecture)"; A:724-725 tags the related stability claim \conjecture | tag
  \conjecture{} or say that only the reduced-model rates are derived.

## Verified as correct (selected; recomputed unless stated)

- Sec. 5.1-5.4: lazy residual fractions 0.44/0.31/0.11 = (1 - chi/9)^2 and 0.13% at sigma = 40; stall onsets
  1.0/1.0/1.6; every cell of tab:stall-test (28 of 32 dead, median 1.44e-5, 12 of 18); 546/212/37 of 795; 30 runs,
  50 of 50; every cell of tab:large-chi, the sigma = 1 and plain-model slopes, xi = 1.11, 3.74; transients 116/14/15
  of 9,165; two-copy labels 5,4,5 and 4,3,2; App. A.8 constants (c_1 = 341.5/chi^3, c_2 = 537.3/chi^3, 6.50
  chi^-1/2, xi_c brackets, 2^-1/3, sigma_floor 0.56/0.74, omega_K, a_0c, 13.283, u_0c), root share
  chi^2/(2d^2 + chi^2).
- Sec. 5.5, A.7: 195/80/38-of-40/25-of-25 GD outcome counts; 765/765; 526 comparisons 65.8/89.5/97.1%; Richardson
  1.18e-5; lambda_GF ~ sigma^3.965; 86/90 catapults, 0.58-1.00 x 2/eta; 104/150; SGD delays and spread exponents;
  c_gamma = 32.62, chi^2 378.6/534.5/929.2; 19.8% vs 17.7%; eta*_ms values; finite-sample counts; eta_c =
  0.1541/0.0435/0.0366; the EoS amplitude 3(hK - 2)/(4(K - 1)(5K - 4)) against the map (K = 3, 5, 7) and
  L_EoS; prop:A-gd-shift and prop:A-sgd-collapse algebra.
- Sec. 5.6: binary4 exponents, P6 slope 0.0199, eta n_1 = 0.30/0.18/0.16/0.17, binary8 Adam exponents, epsilon-law
  ratios, 60/60, balance index 0.21-0.96 and sharpness 1.01-6.1, 59/60 ties, Riemannian and re-gauge counts, Gram
  p = 0.65 and 9/24, cost cells; ALS/DMRG sigma-independence (Q factors), Riemannian GD at eta = 1 maps W_0 to
  HSVD(P_T W*) exactly, re-gauge fractions (4/9)^2 = 0.20 and (4/81)^2 = 2.4e-3.
- Sec. 6.1-6.9, A.5-A.6: every cell of tab:dt-saddle; random-init slopes vs h_e (max 1.3%); 52.46 vs 52.94;
  aligned-covariance percentages, exponents and swap probabilities; misaligned 12.14 and 194.9; correlated 2.133,
  14.67, 113.9, 4.616; CP 5.686/34.40/595; random TTN alphas and offsets 0.07-0.15; P11 163/168, 249/256; star
  Hessians -0.3716/-0.3310/-0.4311, 130.0 and 29.39 vs 129.9 and 29.3; A.5-A.6 algebra (two-component Hessian and
  threshold (N - 2)/N, fold formula, sigma_2(M_e) = det M_e/s_bar, Lambda_eff(c, N), lambda_2 at small c,
  piggyback-covariance block and lambda_min > -b, aligned clocks, Lambda^esc >= Lambda^loss, prop:A-precond(e)).
- Sec. 6.10: 95/96, 96/124, 91/96, 22/26 units, 11/11 (10 new), the 11 clean series (rms <= 0.060, B < 0, free
  exponent 1.22 +- 0.03), binary lambda_min exponents 0.87-0.89 and 1.15, MPS -0.054 to -0.079 and 0.047 +- 0.016,
  28/29 edge fits, creep 0.76, every number of the pre-registered test (rho, intervals, p, 24 distinct S, 74 new
  runs, 4 of 81 censored, ranks 1, 2, 4 and 13-21, scrambled110 third), rank 304 of 528.
- App. A.1-A.4: lem:A-gd-identity (Q+ = Q(1 - eta^2 Delta)); E Q_e(0) = sigma^2(1 - chi_p/chi_v); cor:A-blowup;
  delta_3..delta_8; prop:A-gmax examples; prop:A-kkt; prop:A-component-hessian blocks; prop:A-product-mode limits
  and the binary4 closed form; prop:A-race; prop:A-matrix-like exponent and local slopes 0.38/0.43; all 26 entries
  of tab:A-route-degree; prop:A-steiner; piggyback, star (sqrt 2 w) and triangle (2s) values; prop:A-head-start
  formulas (k = 2, k >= 3, k = K); prop:A-signatures growth exponents and lambda_min; trap Q = sigma^2(X_2/2 - X_6/4),
  P(Q > 0) = (2/3)^3 = 0.30. A.4: definitions, Theorems 1 and 1', Theorem 2 with s_r > 0 and the index condition,
  the corollary (i)-(iii), the HT4 examples (order 6 for three-site D and for D = all sites), the pair-star
  counterexample (Hessian 0/1/6, -0.0350 t^4, control order 5), MPS6 110001 and HT4 1111 (-t^6/432), and the
  dynamics statements all match `theory_saddle_order.md` v2 and both verifications (the v1 gap, Theorem 1', is
  closed; the v1 "N <= 4: nothing changes" error is not repeated).
- Withdrawn lists: apart from C3, C4, M9 and the novelty items of M28, no claim on the WITHDRAWN lists at the
  heads of Secs. 5.1, 5.5, 5.6, 6 and 6.10 is restated in Secs. 5-7 or App. A.
