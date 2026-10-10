# Final review of claims, part A: abstract and Sections 1-4 (2026-10-10)

Adversarial reviewer, read-only (this file is the only one written). Scope: abstract (`report/main.tex`),
`report/sections/01_introduction.tex` (incl. Table 1 and the main findings), `02_setup.tex`, `03_regime_map.tex`,
`04_saddle_to_saddle.tex` (4.1-4.7, incl. Table `tab:s2s-parameters`). Line numbers refer to the current files.

**Sources checked.** Verification reviews (`s2s_core_round2`, `locality_round2`, `order_round2`,
`regime_map_round2`, `data_teacher_round3`, `optim_samples_round2`, `real_data_round2`, `saddle_order_verification_proof`,
`saddle_order_verification_numerics`), `citation_check.md` part b, `notes/theory_saddle_order.md` v2, `notes/locality.md`
sections 7-10, `report_todo.md`, Appendix A (theorem statements), and the committed CSVs: `s2s_core/results/`
(`first_summary`, `main_exponents`, `report_numbers`, `aligned`, `lo_K_table`, `lo_chi_slopes`, `dormant_decay`,
`main_runs`), `locality/results/` (`twocomp_table`, `twocomp_chi4_table`, `stall_dropout_bias`, `run_classes_by_sweep`),
`order/results/` (`main_summary`, `sigma_table`, `h1b_fits`, `multi3_summary`), `regime_map/results/`
(`fix1/table_2_1_s2s`, `fix1/slopes_vs_chi_chi_b4`, `fix1/lazy_summary_chi_b4`, `fix1/imbalance_P20`,
`report/k_sweep_laws_crossing`, `sigma_c_vs_K`, `report/floor_*`, `report/imbalance_map_b4`, `report/numbers`),
`pilot_h2_locality/README.md`.

**Bottom line.** No critical problem: every headline number I recomputed matches the CSVs or the verification notes
(list at the end), and no withdrawn claim is restated verbatim. Six major items: the abstract states the saddle-order
theorem without its scope; "new parameters lower the loss only along routes" is stated as a general fact although the
report's own theorem gives other descents; one sentence says the spectral condition "fails" from k* = 5 on; two status
tags are stronger than their sources; the Sec. 2 reporting rule is not met by several fits that are not marked. The
minor items are scope qualifiers, a few internal inconsistencies, and missing error bars.

Not reported (known pending): the placeholders for regime-map round 2 and saddle-dyn, and items 1-10 of
`notes/report_todo.md`. Item m10 below only extends todo item 2 to Sec. 1.

## Critical

None found.

## Major

**M1. Abstract: the saddle-order theorem without its scope.**
- Location: `main.tex` l.146-148, "We prove that $k^*$ is the saddle order under a spectral condition, which every
  two-component teacher learned in weight order satisfies, and that the order can be lower from $k^*=5$ on."
- Problem: the sentence follows "Later, the tree decides", so it reads as a statement about every later saddle. The
  theorem covers only saddles with one learned component (and needs the route, $\chi_e\ge2$ on one minimal Steiner
  tree). Saddles with two or more learned components are not covered (Sec. 4.4 l.443-444; intro item 4 l.208-209).
- Evidence: `thm:A-order-lower` setting "One learned component $j$" (A_derivations l.752); `theory_saddle_order.md`
  S3, open list ("several learned components"); proof verification section 13.
- Fix: "At a saddle with one learned component we prove that $k^*$ is the saddle order under a spectral condition,
  which every two-component teacher learned in weight order satisfies, and that the order can be lower from $k^*=5$
  on."

**M2. "New parameters lower the loss only along routes" is stated as a general fact.**
- Location: 01 l.171-173 ("Once a component is learned, new parameters lower the loss only along routes, ..."); 04
  l.12-15 ("near a saddle, new parameters can lower the loss only through connected sets of cores whose new entries
  combine into a term of the residual, and the number of cores of the cheapest route sets the type of the saddle and
  the length of the plateau"); 04 l.390-392 (same with "only through routes").
- Problem: false in general, by the report's own results. `thm:A-order-sharp` builds descents from new-entry clusters
  aimed at non-residual patterns (alone, or with rotations of learned cores) at order $2\ell<k^*$. The HT4 $1111$
  wrong-order example uses two polymers of new entries only (order 6 < 7). App. A after `lem:A-routes` says that
  disconnected sets "also contribute, possibly below the route degree". The statement holds only with the
  qualifier "at no Gauss-Newton cost", or for two-component teachers in the correct order (`cor:A-order`(iii)).
- Evidence: A_derivations l.413-418 and l.790-797; `theory_saddle_order.md` section 0 and S8;
  `saddle_order_verification_numerics.md` section 1 (the order-4 descent needs the curved path with old rotations).
- Fix: 01 l.172: "new parameters lower the loss at no Gauss--Newton cost only along routes, ...", and add at the end of
  the sentence: "(from $k^*=5$ on, clusters of new entries aimed at non-residual patterns can descend faster;
  item~4)". 04 l.13: "new parameters lower the loss at no Gauss--Newton cost only through connected sets ..., and,
  when the spectral condition of \cref{app:saddle-order} holds, the number of cores of the cheapest route sets ...".
  04 l.391: "new parameters can lower the loss at no Gauss--Newton cost only through \emph{routes}, ..."; after
  "(\cref{lem:A-routes})" add "; for two-component teachers in the correct order no other curve descends faster
  (\cref{cor:A-order})".

**M3. Sec. 4.4 says that the spectral condition "fails" from $k^*=5$ on.**
- Location: 04 l.434, "It fails from $k^*=5$ on."
- Problem: the condition can fail from $k^*=5$ on; it does not always fail. The control teacher (0.6, 0.6) has
  $k^*=5$, satisfies the condition and has order 5. The intro (l.202) and App. A (l.468) correctly say "can fail".
- Evidence: proof verification section 9 (control: order $k^*=5$, "as Theorem 1 says"); numerics verification
  section 1, table row "mps6 pair-star (0.6, 0.6)".
- Fix: "It can fail from $k^*=5$ on."

**M4. Sec. 4.4 tags the forgetting after the order-4 escape \derived; the source and the intro say empirical.**
- Location: 04 l.437-439, "gradient flow escapes in $\propto\isc^{-2}$ instead of $\isc^{-3}$ and then forgets the
  first component \derived{} (checked on this teacher; ...)". Related: 01 l.206-208, "(both checked on this teacher,
  from the saddle and on the seeds that visit it from small initialization)".
- Problem: (a) one \derived tag covers two claims. The escape law is derived (reduced model). The forgetting is an
  observation (`theory_saddle_order.md` S11: "empirical"). The intro tags the forgetting \empirical, so 1 and 4.4
  disagree. (b) The forgetting was observed only from small initialization: the numerics verifier's $q$-first seeds,
  plus 6 of 7 wrong-order HT4 locality runs. The saddle-start runs (check 1, 5 seeds) timed only the escape. So
  "both checked ... from the saddle" overstates.
- Evidence: `theory_saddle_order.md` section 6 ("What comes next (empirical, numerics verifier)"), S9, S11;
  numerics verification section 2 ("Where it goes next (new; open in the note)"). The proof verifier's GF check
  records only the state at escape.
- Fix: 04 l.437-439: "gradient flow escapes in $\propto\isc^{-2}$ instead of $\isc^{-3}$ \derived{} (checked on this
  teacher: 5 seeds from the saddle, and the seeds that visit it from small initialization) and then forgets the first
  component \empirical{} (seen from small initialization)". 01 l.207-208: "(the escape law checked from the saddle and
  from small initialization, the forgetting from small initialization only)".

**M5. Table `tab:s2s-parameters`: the SGD trap law tagged \derived; its source is a conjecture.**
- Location: 04 l.1246-1247, "at fixed $\lr/\batch$ SGD stalls where gradient flow is only slow \derived{} (from the
  heuristic law of \cref{prop:A-sgd-collapse})".
- Problem: `prop:A-sgd-collapse` is tagged "\conjecture{} (heuristic law)" (A_derivations l.1272; Sec. 5, l.389). The
  intro (l.291) and Table 1 (Q8, "\conjecture{} (SGD law)") agree with that. A consequence of a conjecture cannot be
  \derived.
- Evidence: A_derivations l.1271-1284; 05 l.388-392; optim-samples verification (P9 law fitted, $c_\gamma$ one free
  constant).
- Fix: "at fixed $\lr/\batch$ SGD traps where gradient flow is only slow \empirical{} (law: \conjecture,
  \cref{prop:A-sgd-collapse})".

**M6. The Sec. 2 reporting rule is stated as universal, but several fits in Secs. 3-4 do not meet it and are not marked.**
- Location: 02 l.562-567, "Every scaling claim uses at least five initialization scales over at least one decade and
  five seeds per point ...".
- Problem: the $\dagger$ marks in Tables `tab:s2s-gaps` and `tab:locality-exponents` cover only "less than one
  decade". Unmarked fits that miss the rule:
  - locality per-seed fits use "the (at least three) $\isc$ at which it escapes" (04 l.488-490);
  - the kept wrong-order branch (04 l.909-911) uses 5 and 6 seeds over factors 4 and 2.3 in $\isc$, and is reported as
    $\isc^{4.22\pm0.09}$ and $\isc^{5.2\pm0.3}$;
  - the binary4 close-teacher gap fits keep 4-5 $\isc$ (about 0.8 decade) for some seeds after the $\rho>3$ cut, but
    are reported "over 1.5 decades" (04 l.124-125);
  - the $\dagger$ rows with fewer than five fitted seeds (MPS8 $\{1,5\}$ saddle, 4 seeds; MPS8 $\chi=3$ gaps, 3 seeds)
    are marked only for their $\isc$ range, not for the seed count.
- Evidence: s2s-core verification E9; `locality/results/twocomp_table.md` (fitted seeds and $n_\sigma$ per row);
  `order/results/h1b_fits.csv` (`h1b_saddle_b8_exponents`: 6 seeds, $\isc$ 0.15-0.35).
- Fix: 02 l.563: "Scaling claims aim at five initialization scales over at least one decade and five seeds per point
  ...; fits that fall short are marked $\dagger$ or state their range and seed count." Then add $\dagger$ to the
  kept-branch MPS6 and binary8 exponents, and add "(as few as 4 $\isc$ for some close-teacher seeds)" at 04 l.125.

## Minor

Abstract and Section 1

**m1. Abstract: "not on binary trees" is broader than the mechanism and the data.**
- Location: `main.tex` l.148-149, "the saddle after the first escape is strict on matrix product states, not on binary
  trees."
- Problem: the mechanism is a two-core route between adjacent cores with sites. Binary trees with $\Ns\ne2^m$ have an
  internal core that carries a site (Table `tab:trees` caption: "a single site joins the core above"), so the
  mechanism predicts a strict saddle there. The tested trees are binary4/8 and HT4/8 (N = 4, 8, d = 2). The intro
  states the condition ("whose internal cores carry no site", l.226); the abstract does not.
- Evidence: data-teacher verification, safe-to-cite #1-#3 (13 generic teachers; N = 4, 8); 02 l.96-98.
- Fix: "... is strict on matrix product states, not on binary or hierarchical Tucker trees whose internal cores carry
  no site."

**m2. Abstract and the intro paragraph: the regime sequence without "centred model".**
- Location: `main.tex` l.150-151 ("Larger $\isc$ leads through a rich regime to a lazy one ..."); 01 l.64-66.
- Problem: the map is measured on the centred model. Item 8 (l.265) and Table 1 Q5 say so; the abstract and the
  one-paragraph answer do not.
- Evidence: 03 l.40-42 ("The main family is the centred model").
- Fix: abstract: "With the initial function removed, larger $\isc$ leads through ..."; 01 l.64: "Raising $\isc$ (centred
  model) moves training ...".

**m3. Intro, one-paragraph answer: the Steiner identity tagged \proven without its teacher class.**
- Location: 01 l.55-57, "the cheapest route has $k^*$ cores, the Steiner subtree of the cores that carry the sites
  where the next component differs \proven".
- Problem: `prop:A-steiner` assumes (A2), a pattern teacher with orthonormal components (odeco included). Two
  sentences later the paragraph says that non-orthogonal teachers behave differently.
- Evidence: A_derivations l.376-377 (A2) and the proof of `prop:A-steiner` (pattern overlaps 0 or 1).
- Fix: "... where the next component differs \proven{} (orthogonal pattern teachers)".

**m4. Intro item 6: "locality beats weight" stated generally, without a tag.**
- Location: 01 l.241-242, "A smaller component with a shorter route is learned first (\emph{locality beats weight},
  ...)".
- Problem: no tag. The evidence is one teacher (B4) on one tree (MPS4). The rule is a $\isc\to0$ statement: a log
  route beats a power-law route (`rem:A-sequence`(2)); at finite $\isc$ weights and prefactors compete.
- Evidence: 04 l.625-629.
- Fix: "As $\isc\to0$, a smaller component with a shorter route is learned first \derived{} (\emph{locality beats
  weight}; teacher B4 on MPS4, every escaping run \empirical)".

**m5. Several components: "predicts the plateau levels and their laws" / "predicts every plateau level".**
- Location: 01 l.182-184; 04 l.421-422 ("it predicts every plateau level and its $k^*$").
- Problem: the test matched detected plateaus to predicted levels. The converse (detected plateaus at unpredicted
  levels) is not reported. There is one exception (B4 on HT4, fan-in: $4.73\pm0.90$, 23 of 48 runs stalled), and
  several $k^*=2$ laws are undetermined (TD level 1, the MPS8 first levels from small initialization).
- Evidence: `notes/locality.md` 7.4 (`detected_frac`; "Every $k^*\ge3$ level but one"); locality verification (f).
- Fix: 04 l.422: "every predicted level with $k^*\ge3$ is found as a plateau (88--100\% of runs), with one exception
  below, and $k^*=k$ for two components". 01 l.183: "predicts the plateau levels and, where they are long enough to
  measure, their laws, with one exception (B4 on HT4, fan-in) ...".

**m6. Table 1 Q3 and intro item 1: omitted qualifiers.**
- Location: 01 l.98 (Q3: "confirmed for $k\le5$ on trees with $\Kc=3$--$15$"); 01 l.147-148 (item 1: "each event
  raises every bond rank by one").
- Problem: Q3 drops the named exception (MPS8 $\{1,5\}$ from the saddle, $3.67\pm0.20$; 04 l.501-502). Item 1 drops
  the count that Table 1 Q1 gives (497 of 531).
- Fix: Q3: "confirmed for $k\le5$ on trees with $\Kc=3$--$15$, with named exceptions"; item 1: "each event raises
  every bond rank by one (497 of 531 runs)".

**m7. Intro item 5: "within ... 1.3% (from random initialization)" covers five teachers, not every non-orthogonal one.**
- Location: 01 l.224-225.
- Problem: the item names three classes (overlapping components, misaligned covariance, random TTN teachers). The
  1.3% holds for the five teachers of the random-init test. The overlapping-CP teacher with $c=0.05$ gives
  $595\pm11$ against 570 (+4%, not yet asymptotic), and misaligned\_20 sits below a bound, not at a prediction.
- Evidence: data-teacher verification, safe-to-cite #5 and #7.
- Fix: "... and $1.3\%$ (from random initialization, MPS4, five teachers; at small overlap $c=0.05$ not yet
  asymptotic) of the predicted rates".

**m8. Intro item 10 and Table 1 B5: "ranks pixel orders by their final loss".**
- Location: 01 l.312-313; Table 1 B5 l.134 ("ranked before training by a route-amplitude share").
- Problem: the pre-registered result is a Spearman correlation of $-0.60$ $[-0.82,-0.22]$ over 27 orders ($-0.55$
  within 20 random orders). "Ranks" suggests a near-perfect ordering.
- Evidence: real-data verification, section 1 and safe-to-cite row 1.
- Fix: "... correlates with the final loss across pixel orders (Spearman $-0.60$, 27 orders; pre-registered), but not
  with the early speed"; B5: "final loss correlated before training with a route-amplitude share".

**m9. "Bottom-up" growth generalized beyond binary8.**
- Location: Table 1 Q7 l.120 ("bottom-up growth on binary trees"); 01 l.309 ("binary students learn bottom-up"); 04
  l.1185 ("higher-order on binary and HT trees, which grow bottom-up").
- Problem: tested only for binary8 students of MPS8 teachers at $\isc=0.3$--$0.5$ (163 of 168 discriminating runs);
  Sec. 6 states "Scope: $\Ns=8$". HT students were never tested for bottom-up order.
- Evidence: 06 l.343-356; data-teacher verification, safe-to-cite #10.
- Fix: Q7: "bottom-up growth (binary8 student)"; 01 l.309: "binary8 students learn bottom-up"; 04 l.1185: "...
  higher-order on binary and HT trees; binary8 students grow bottom-up".

**m10. Sec. 1 restates the SGD claim of todo item 2.**
- Location: Table 1 Q8 l.124 ("SGD never shortened a plateau"); 01 l.290 ("Gradient noise never shortened a
  plateau").
- Problem: todo item 2 lists only Sec. 5.5. The same overclaim sits in Table 1 and item 9: 9 of 23 cells of Fig. 14a
  are negative (at $\isc=0.1$, $\batch=128$: $-0.32\%\pm0.13\%$).
- Fix: apply the todo-2 wording in both places: "SGD never shortened a plateau by more than about $0.3\%$".

Section 3

**m11. Sec. 3 opening: asymptotic constants given without their $\chi$ range; "100 times" without "up to".**
- Location: 03 l.28-31.
- Problem: $\isc_{\rm lazy}\approx6.5/\sqrt\chi$ and the rich window "of about 20" hold at $\chi\ge32$
  ($\isc_{\rm lazy}\sqrt\chi=8.96$, 7.99, 7.43, 6.75 at $\chi=9$, 12, 16, 24). The body says the light-root effect is
  "up to a factor 100" (l.209).
- Evidence: `fix1/lazy_summary_chi_b4.csv` (`tensor_med_x_sqrtchi`); `report/numbers.csv` (rich window 18.4-22.8 /
  19.5-24.2 at $\chi\ge32$).
- Fix: "... and lazy above $\isc_{\rm lazy}\approx6.5/\sqrt\chi$ at large $\chi$ ($\chi\ge32$), ..."; "a light root
  makes training lazy at an up to 100 times smaller function-space scale".

**m12. Sec. 3.2: "The saddle-to-saddle boundary falls from 0.237 ... to 0.043" hides non-monotone, noisy small-$\chi$
values.**
- Location: 03 l.69-70.
- Problem: at $\chi=4$ the boundary is $0.099\times/1.51$ and at $\chi=12$ it is $0.075\times/1.33$, both below
  $\chi=6$ (0.137) and $\chi=9$ (0.136). Per-seed boundaries spread by factors 1.9-15.9 at $\chi\le12$ (1.4-2.7 at
  $\chi\ge16$).
- Evidence: `fix1/table_2_1_s2s.csv`; `report/numbers.csv` ("per-seed boundary spread").
- Fix: add after "at $\chi=36$": "(not monotone at small $\chi$: $0.099$ at $\chi=4$, $0.075$ at $\chi=12$, where
  per-seed values spread by up to a factor 16)".

**m13. Sec. 3.2 "Seven cores": $\chi=16$ fails to be lazy only within the step limit.**
- Location: 03 l.116-121 ("... and only the latter reach a lazy regime, from $\isc\approx1.4$").
- Problem: at $d=2$, $\chi=16$ is exactly the cut dimension (full bonds). Its runs at $\isc>2$ stop at
  `max_steps` with $\Loss/\Loss(0)=0.05$--$0.44$. Sec. 5.1 (05 l.48-49) says "not lazy within the step limit". Sec. 3
  presents this as a result, which reads against "lazy only when every bond reaches its cut dimension".
- Evidence: `report/numbers.csv` ("K = 7, chi = 16 (square), sigma > 2: ... max_steps"); `report/floor_sigma_chi_b8.csv`.
- Fix: "... and only the latter reach a lazy regime within the step budget, from $\isc\approx1.4$ (at $\chi=16$, where
  the bonds are exactly full, runs above $\isc=2$ stop at the step limit unconverged; \cref{sec:other-lazy})".

**m14. Table `tab:regime-summary`: four entries disagree with the text or the caption.**
- Location: 03 l.240-253.
- Problems and fixes:
  1. Row "scale": "$\isc_{\rm lazy}=3.0$ to $0.57$ for $\chi=3$--$128$". No lazy regime exists below $\chi=9$; 3.0 is
     the $\chi=9$ value. Fix: "$\isc_c=0.24$ to $0.023$ ($\chi=3$--$128$), $\isc_{\rm lazy}=3.0$ to $0.57$
     ($\chi=9$--$128$)".
  2. Row "bond dimension": "rich window $18$--$23$" is the probe-kernel range, but the caption says "Lazy boundaries
     with the tensor kernel". The tensor-kernel range is 19-24 (03 l.105; `report/numbers.csv`). Fix: "rich window
     $19$--$24$".
  3. Same row, status "\empirical{} \derived". The saddle-to-saddle $\chi^{-1/2}$ law is not derived (03 l.112-113:
     heuristic, \conjecture). Fix: "\empirical; \derived{} (lazy constants)".
  4. Row "parameterization": "$\chi^{-1.01}$ for $\chi\ge64$ ($-1$)" is a two-point local slope ($\chi=64$, 128).
     The full-range NTK fit is $-0.79\pm0.04$, and "approaches $-1$" was withdrawn (regime-map verification S1).
     Fix: "NTK: local slope $-1.01$ between $\chi=64$ and $128$ ($-1$; full-range fit $-0.79\pm0.04$)".

**m15. Sec. 3.4: inconsistent treatment of 3-of-6 ties.**
- Location: 03 l.208-210, "at $0.1$ a tie, 3 of 6" versus "A heavy root ($\lambda/\sqrt\chi\ge3$) is lazy from
  $\fsc=10$ (3 of 6; ...)".
- Problem: at $\fsc=10$ every heavy-root cell has 3 lazy and 3 rich runs, the same tie the text names one line
  earlier. Balance has 4 of 6.
- Evidence: `report/imbalance_map_b4.csv` ($\chi=24$, $\fsc=10$, $\lambda/\sqrt\chi=3.16$--$100$: 3/3 each).
- Fix: "A heavy root ($\lambda/\sqrt\chi\ge3$) ties at $\fsc=10$ (3 of 6 lazy) and is lazy at $\fsc=100$ (6 of 6),
  close to balance (4 of 6 at $\fsc=10$)."

Section 4

**m16. Sec. 4.1: the strict binary4 saddle is "reached only after a wrong first escape".**
- Location: 04 l.70-73.
- Problem: in the data the strict saddle (components 1 and 3 learned, 2 missing) follows a correct first escape and a
  wrong second one (order 1-3-2, seeds 8 and 5). The first component is the largest in 639 of 640 binary4 runs.
- Evidence: s2s-core verification #11 and #19; 04 l.170-172.
- Fix: "..., reached only after an escape out of weight order (here the second: order 1-3-2)."

**m17. Sec. 4.1: scope of two numbers.**
- Location: 04 l.168-169 ("In cell medians the magnitude is $0.87$--$1.15$ times the balanced-route value ... at
  $\Kc=3,4$ and $1.36$--$3.15$ times it at $\Kc=7,8$"); 04 l.124-125 ("over 1.5 decades"; see also M6).
- Problem: 0.87-1.15 holds for the main fan-in cells (balanced binary4 $\chi=3$: 1.19). At $\Kc=7,8$ the k78 cells reach
  3.9 (`main_exponents.csv`, `lam1_over_pred` 3.87 for binary8 balanced).
- Evidence: s2s-core verification E11.
- Fix: "In the medians of the fan-in cells the magnitude is $0.87$--$1.15$ times ... at $\Kc=3,4$ ($1.19$ for balanced
  binary4) and $1.36$--$3.9$ times it at $\Kc=7,8$".

**m18. Sec. 4.3: "the median grows from 0.42 (HT3) to 814 (binary16)".**
- Location: 04 l.319-321.
- Problem: the dependence on $\Kc$ is not monotone. HT3 ($\Kc=5$) has the smallest median. binary4 ($\Kc=3$) has 0.86
  and MPS4 ($\Kc=4$) 0.67. At $\Kc=7$, HT4 has 0.58 and binary8 3.04. The family matters as much as $\Kc$ at small
  $\Kc$.
- Evidence: `s2s_core/results/lo_K_table.csv`.
- Fix: "the median ranges from $0.42$ (HT3, $\Kc=5$) to $814$ (binary16, $\Kc=15$), growing with $\Kc$ within each
  family".

**m19. Sec. 4.4: "wrong-order and star saddles ... were not tested", and a figure takeaway that ignores them.**
- Location: 04 l.598-600; caption of `fig:locality-hessian`, 04 l.613-614 ("negative curvature that survives
  $\isc\to0$ exists only for two-core routes").
- Problem: wrong-order strict saddles with $k\ge3$ were measured: binary4 (Sec. 4.1) and the rotation branch of
  Sec. 4.5 ($\lambda_{\min}$ to 4 digits). Star saddles were measured in Sec. 6. The takeaway contradicts both.
- Evidence: 04 l.170-176, l.903-906; data-teacher verification #12-#14.
- Fix: l.600: "were not tested in this sweep (wrong order: \cref{sec:s2s-phenomenology,sec:s2s-order}; stars:
  \cref{sec:teacher-star})". Caption: "Takeaway: with the larger component learned first, negative curvature that
  survives $\isc\to0$ exists only for two-core routes."

**m20. Novelty of the trap is wider than the citation audit allows; Luo et al. missing in Sec. 4.6.**
- Location: 04 l.473-474 ("New for TTNs: ... (v)~the conservation-law trap (\cref{prop:A-trap})"); 04 l.1111-1112
  ("exactly balanced initialization is expected to remove the trap").
- Problem: citation check B5 calls the mechanism related to known shrinkage results: Jacot et al. App. C.1; Kumar &
  Haupt; AGF Lemma B.1. It keeps "new" only for the TTN route-end condition, and asks that Luo et al. 2025 be cited
  as a known failure of exactly balanced initialization. `prop:A-trap` and intro item 7 already narrow the novelty;
  Sec. 4.4 (v) and 4.6 do not.
- Fix: (v): "the route-end condition of the conservation-law trap (\cref{prop:A-trap})". At l.1112 add "(in four-layer
  matrix factorization exactly balanced initialization can fail, \citealp{luo2025global})".

**m21. Sec. 4.5: "a selection rule" listed as new without its known analogue.**
- Location: 04 l.807 (list of new results).
- Problem: the rule is the TTN form of the (weight)$\times$(overlap)$^{p-2}$ rule of multi-spike tensor PCA, as the
  body says (l.854-855). The order verification C7 calls it "related: Ben Arous et al.".
- Fix: replace "a selection rule" by "the TTN form of the selection rule of tensor PCA".

**m22. Sec. 4.5: the natural-run statistics silently exclude $\isc=0.1$.**
- Location: 04 l.912-916 ("296 of 301 binary4 runs ... rotate ... $0.98$--$1.38$ ... a wrong plateau is resolved in at
  most $2.5\%$ of runs").
- Problem: all three numbers hold for $\isc=10^{-4}$--$10^{-2}$. At $\isc=0.1$ only 30 of the 120 natural binary4
  wrong-order runs rotate (17, 12, 1 of 40 at $\varrho=0.7$, 0.8, 0.9; 52 in another class), the scaled times are
  0.85, 1.16, 0.38, and 5-7.5% show a wrong plateau.
- Evidence: `order/results/h1b_fits.csv` (`h1b_natural_b4`, $\isc=0.1$ rows); order verification C13 ("flat in
  sigma 1e-4..1e-2").
- Fix: "Natural runs at $\isc=10^{-4}$--$10^{-2}$ (...) show no $\ln(1/\isc)$ part ...; at $\isc=0.1$, where the
  wrong saddle is barely reached, only 30 of 120 binary4 runs rotate."

**m23. Sec. 4.6 opening: route death attributed to the trap mechanism without a hedge.**
- Location: 04 l.970-971, "a route dies, its end cores decaying to roundoff under the conservation laws (the trap of
  \cref{prop:A-trap})".
- Problem: the section shows that the mechanism's criterion is neither necessary nor sufficient. Seven dead runs had a
  live end, and the theory's own refutation criterion counts these against the mechanism as stated (l.1105-1111). The
  signature has been measured only for the odeco $\Kc=4$ runs.
- Fix: "a route dies, its end cores decaying to roundoff (the conservation-law trap of \cref{prop:A-trap} is the
  proposed mechanism, \derived{} in a reduced model; its criterion is tested below)".

**m24. Sec. 4.4 / intro item 4: the control teacher's anomalous exponent is not mentioned.**
- Location: 04 l.437-439 ("escapes in $\propto\isc^{-2}$ instead of $\isc^{-3}$"); 01 l.205-206.
- Problem: "instead of $\isc^{-3}$" implies that teachers meeting the condition follow the route law. For the matched
  control (0.6, 0.6; order $k^*=5$), small-initialization exponents tend to about 3 in 4 seeds and about 4 in 3-6
  seeds; 2 of 12 seeds are trapped. The theory note calls this unexplained and links it to the steep MPS $k=5$ rows
  of Sec. 4.6. This may belong to the pending saddle-dyn sentence; if so, include it there.
- Evidence: numerics verification section 2 (control table); `theory_saddle_order.md` section 6, caveat 4.
- Fix: after "instead of $\isc^{-3}$" add "(for the matched control with order $5$ the small-initialization exponent is
  about 3 in some seeds and about 4 in others, unexplained; \cref{sec:s2s-failure})".

**m25. Table `tab:s2s-parameters`: the star row tags forgetting \proven.**
- Location: 04 l.1216-1217, "a star with $\lambda_{\max}(\Gamma)>\tw j$ makes a larger learned component be forgotten
  \proven{} \empirical".
- Problem: what is proven is the strictness of the saddle (Hessian). That the flow then forgets the component is
  measured from saddle starts with $\varepsilon\le10^{-3}$. From random initialization the flow never sits at
  $\theta_q$.
- Evidence: data-teacher verification #12 (proven Hessian), #13-#15 (empirical dynamics; random init).
- Fix: "a star with $\lambda_{\max}(\Gamma)>\tw j$ makes the learned saddle strict \proven{} and the larger component
  is forgotten \empirical{} (from the saddle)".

**m26. Missing error bars or seed counts.**
- 04 l.628-629, teacher B4: "$p=2.02$" and "($1.05$, $1.03$)"; 04 l.642, TA: "($1.02$)", "($2.08$)". These have no s.e.
  and no seed count. `locality/results/levels_summary.csv` has them: B4 MPS4 $2.015\pm0.013$ (5 seeds); B4 binary4
  $1.051\pm0.008$, $1.028\pm0.005$ (8); TA binary4 $1.023\pm0.005$ (8); TA HT4 $1.093\pm0.022$, $2.084\pm0.040$ (7).
  Fix: quote them as mean $\pm$ s.e. with the seed count, as for TD and TC (l.646-648).
- 04 l.314, MPS4 $d$ sweep: "$1.39$, $1.41$, $1.50$" without the 95% intervals that binary4 gets one line earlier.
  `report_numbers.csv`: [1.27, 1.58], [1.30, 1.52], [1.43, 1.60]; these overlap the theory note's 1.30, 1.35, 1.46.
  Fix: add the intervals.
- 03 l.157, $\Kc$ sweep: quartiles only for $\Kc=3$. `sigma_c_vs_K.csv` has them for every $\Kc$. Fix: add them, or
  "(medians; quartiles in \cref{app:exp-regimes})".

**m27. Abstract: no hint of the failure modes.**
- Location: `main.tex` l.140-142, "From small initialization ..., gradient flow is saddle-to-saddle: ...".
- Problem: Sec. 4.6 shows that sequences often end early: most $\Kc=7,8$ runs at $\chi=R$, 398 of 5,682 pattern-teacher
  runs at $k\ge4$. The withdrawn s2s-core claim was "S2S on every tree". The abstract is not wrong, but it states the
  picture without its main limitation.
- Fix (optional, a few words): "... gradient flow is saddle-to-saddle (until a route dies, which happens often on large
  trees): ...".

**m28. Nits.**
- 02 l.419: the conservation drift "must stay below $10^{-6}$". App. B l.180-181 reports 11 of 1,576 s2s-core runs above
  it (at most $3.3\cdot10^{-6}$). Fix: "is required to stay below $10^{-6}$ (exceptions in \cref{app:experiments})".
- Site numbering changes between sections: from 1 in Sec. 2 (l.134-135, $D=\{1,2\}$), Sec. 3 (cut $\{1,2\}|\{3,4\}$)
  and `fig:s2s-runs`; from 0 in Secs. 4.4-4.5. Each place declares its convention, but the Sec. 2 example $D=\{1,2\}$
  is Sec. 4.4's $D=\{0,1\}$. Fix: number from 0 in Sec. 2 as well, or add "(sites numbered from 1 here)" at 02 l.134.
- 04 l.951 ("\derived{} at LO, exact on MPS") versus `prop:A-race` "\proven{} (at LO)". Fix: "\proven{} at LO on MPS,
  \derived{} elsewhere".

## Checked and consistent (no action)

- Abstract, intro and Table 1 numbers: 1,847/1,848; residuals $4\cdot10^{-8}$ to $2.2\cdot10^{-3}$ (231 pairs);
  $T/t_{\rm KH}$ from 3 to $3\cdot10^7$; 497/531; 11,600 runs (`run_classes_by_sweep.csv` gives 11,447, plus 170
  diagnosis runs); 0.60-0.88 over 23 settings; 398/5,682; 6.40-6.56; 0.17-0.75; 765/765; 0.4% and 1.3%.
- `tab:s2s-gaps`: every cell, bracket and seed count, against `main_exponents.csv`. Sec. 4.1 plateau and spectrum
  errors, against `report_numbers.csv`. Example event times in `fig:s2s-runs`, against `main_runs.csv`. Aligned
  deviations, against `aligned.csv`. First-escape residuals and offset-removed exponents (4.996-4.999 includes the
  close/balanced binary8 cells), against `first_summary.csv`. Prefactor slopes and the $d$ sweep. Decay rates
  0.26-1.06 $\isc^2$ (mean of the two ends), against `dormant_decay.csv`.
- `tab:locality-exponents`: every row, against `twocomp_table`, `twocomp_chi4_table` and `stall_dropout_bias`
  ($\chi=4$ MPS rows 3.19/2.97 follow the CSV, not the note's 3.00). `tab:locality-stalls` totals. Pilot numbers
  (mean $\pm$ s.d.), against its README.
- Sec. 4.5: `main_summary` (ranges, $\Kc$ and $\chi$ dependence, 0.87 vs 0.73), `sigma_table` ($\isc$-independence,
  MPS6 $m=2,3,4,6$), `tab:order-gamma`, `h1b_fits` (slopes, 11-88 and 95-106, kept-branch exponents),
  `multi3_summary` (0.705/0.490 vs 0.518/0.319, 70-89%).
- Sec. 3: `table_2_1_s2s`, `slopes_vs_chi`, `lazy_summary` (0.07% below the bracket at $\chi=128$), `imbalance_P20`,
  `k_sweep_laws_crossing` ($\chi^2$ 7.5/7.3, which differ from the verifier's 7.7/6.9 because the code differs; the
  conclusion is the same), `sigma_c_vs_K` (stalls 0/1/0/3/0/5/3/6, 5/5), floors (159 runs, 0.04%), `numbers.csv`.
- Withdrawn lists of the s2s-core, locality, order, regime-map, data-teacher (E12-E17) and real-data reviews:
  none restated in the abstract, Sec. 1, Tables 1 and `tab:s2s-parameters`, or the captions of Secs. 3-4, apart from
  the items above.
