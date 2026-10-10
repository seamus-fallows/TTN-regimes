# Final review, round 2 (multi-lens workflow + adversarial verification), 2026-10-10

Six reviewers (intro/summary, setup+regimes, s2s, other+covariance, appendix math, global); every critical or
major finding was checked by an adversarial verifier. Minor findings are unverified. Raw JSON: final_review_round2.json.

- intro-summary: 17 findings (15 minor, unverified); serious: 2 confirmed, 0 refuted, 0 unsure
- setup-regimes: 13 findings (9 minor, unverified); serious: 4 confirmed, 0 refuted, 0 unsure
- s2s: 11 findings (8 minor, unverified); serious: 3 confirmed, 0 refuted, 0 unsure
- other-cov: 10 findings (7 minor, unverified); serious: 2 confirmed, 1 refuted, 0 unsure
- appendix-math: 3 findings (2 minor, unverified); serious: 1 confirmed, 0 refuted, 0 unsure
- global: 22 findings (17 minor, unverified); serious: 5 confirmed, 0 refuted, 0 unsure

## intro-summary

### intro-summary-1 (major) [verifier: confirmed]
- Location: report/sections/01_introduction.tex l.217-218 (finding 10) and l.309-310 (Sec. 1.3, Tensor factorization)
- Quote: Random TTN teachers plateau at fixed-rank critical points, not at truncated singular value decompositions / new are that generic teachers plateau there and not at truncated SVDs
- Problem: Restates the random-TTN plateau claim that erratum E15 narrowed, which is on the Sec. 6 WITHDRAWN list. It also claims more novelty than citation_check B12 allows.
- Evidence: Sec. 6.4: 'for three teachers that point is separated from every SVD-type truncation, while for the other two, and at the higher-rank plateaus, it lies within 0.0032 of one, so the plateaus do not discriminate'. tab:cov-summary: '3 of 5 teachers separated'. data_teacher_round3_verification E15. citation_check B12: the rank-one plateau is a special case of Razin et al. 2021 Thm 2 (CP); only 'non-criticality and the eps^4 gap' are new.
- Fix: Finding 10 (01:217-218): 'Rank-one plateaus of random MPS4 teachers sit at the bond-rank-one critical point, separated from every SVD-type truncation for 3 of 5 teachers (for the other two, and at higher-rank plateaus, the two lie within 0.0032 and are not distinguished)'. Sec. 1.3 (01:309-310): 'new are that the hierarchical SVD truncation is generically not critical (an $\varepsilon^4$ gap), the measured rank-one plateaus at that critical point (separated from SVD truncations for 3 of 5 random teachers; in CP factorization a rank-one trajectory is known, \citealp{razin2021implicit}), the tree contrast and the bottom-up order'.
- Verifier: 01_introduction.tex l.217-218 states the claim with no scope: 'Random TTN teachers plateau at fixed-rank critical points, not at truncated singular value decompositions'. Sec. 6.4 (06:284-287), tab:cov-summary (06:505-506) and E15 (data_teacher_round3_verification l.251-254) all narrow it: only teachers 102, 104 and 105 are separated, and for the other two teachers and at every higher-rank plateau the critical point lies within 0.0032 of a truncation. The 06 header (l.22) lists exactly this unscoped claim as WITHDRAWN. I rechecked e4b_ttn_plateaus.csv. The distance from critical point to nearest truncation is 0.00137 for teacher 101 and 0.0032 for 103 at rank one, and at most 4e-4 at higher ranks. At sigma=1e-3, seeds 0 and 4 of teacher 101 sit closer to a truncation than to the critical point. The novelty half also holds in substance, though the finding reads B12 backwards: in the citation_check legend, 'special case' means Razin's CP result is a special case of ours. Still, B12's wording column names as new only 'non-criticality and the eps^4 gap'. A:1009-1011 and 06:290-291 both cite Razin for the CP rank-one trajectory, but 01:309-310 claims 'generic teachers plateau there' as new without that precedent and without the E15 scope. The proposed fix is essentially right. Two refinements: only random MPS4 teachers (d=3) were run, and the Sec. 1.3 sentence should keep the scope too.

### intro-summary-2 (major) [verifier: confirmed]
- Location: 01_introduction.tex l.194-195 (finding 7); 07_open_questions.tex l.225-226 (open question 4)
- Quote: it separates the runs that stall ($89\%$ of those it predicts dead, $1.8\%$ of the others)
- Problem: The 1.8% is the stall rate of the predicted-alive class only. The criterion has three classes (dead, pinned, alive), and pinned runs stall at 15%, so 'the others' stall at 5.9%.
- Evidence: Sec. 4.8: '89% of the predicted-dead runs stall, against 1.8% of the predicted-alive ones'. D_s2s.tex 'Per-run tests': dead 408/457 (89%), pinned 141/934 (15%), alive 38/2,101 (1.8%). Runs not predicted dead: 179/3,035 = 5.9%.
- Fix: 01:194-195: '(89% of those it predicts dead, 15% of those with one dying end, 1.8% of those it predicts alive)', or the verified wording '(89% of those it predicts dead, 1.8% of those it predicts alive)'. 07:225-226: '($89\%$ of the predicted-dead runs stall, $15\%$ of the pinned ones (one dying end), $1.8\%$ of the predicted-alive ones)'.
- Verifier: The criterion has three predicted classes. D_s2s.tex l.576 gives dead 408/457 (89%), pinned 141/934 (15%) and alive 38/2,101 (1.8%), for 3,492 runs and 587 stalls. Sec. 4.8 (04:830-831) and trap_verification 'Safe to cite' (l.196) attach 1.8% to the predicted-alive class only. '1.8% of the others' in 01:194-195 and 07:225-226 therefore misstates the denominator: runs not predicted dead stall at (141+38)/(934+2,101) = 179/3,035 = 5.9%. This is also an inconsistency with Sec. 4.8. Under the rubric it is arguably a wrong number in a headline. The proposed numbers are correct (89.3%, 15.1%, 1.81%), but the term 'pinned' appears nowhere in 01_introduction.tex, so the intro needs a gloss or the verified two-class wording.

### intro-summary-3 (minor) [minor, unverified]
- Location: 07_open_questions.tex l.290-291 (open question 8)
- Quote: from about $200$--$250$ samples, $5$--$9$ times the dimension of the teacher manifold
- Problem: The ratio disagrees with App. D: review B fix m26 was applied in App. D but not here.
- Evidence: D_optim.tex l.125-126: '200--250 samples or 4--9 times the teacher-manifold dimension with chi=4 students'. 200/48 = 4.2 (N=6, d=3, manifold dimension 48). final_review_claims_B m26.
- Fix: '4--9 times the dimension of the teacher manifold (students with chi=4)'.

### intro-summary-4 (minor) [minor, unverified]
- Location: 07_open_questions.tex l.186-187; 01_introduction.tex l.161-162 (finding 3 header), l.168 (finding 4 header), l.111 (Table 1 Q3)
- Quote: with one exception (\cref{sec:s2s-locality})
- Problem: Cross-references point to the wrong subsection.
- Evidence: The three- and four-component levels and their one exception (B4 on HT4, fan-in) are in Sec. 4.6 (sec:s2s-several), not Sec. 4.4. Finding 3 also cites only 4.4 for 'with three or four components the rule predicts the plateau levels'. Finding 4 points to App. A.4 and Table 1 Q3 to 4.4 for the saddle-order theorem, which the main text now states in Sec. 4.5 (sec:s2s-saddle-order).
- Fix: Cite sec:s2s-several in open question 1 and finding 3, and sec:s2s-saddle-order in finding 4 and Table 1 Q3.

### intro-summary-5 (minor) [minor, unverified]
- Location: main.tex l.148; 01_introduction.tex l.70-71 (bullet 3), l.172 (finding 4), l.279 (Sec. 1.3)
- Quote: From $k^*=5$ on it can: on MPS6 a teacher with $k^*=5$ has order $4$
- Problem: These sentences imply the order equals k* below k*=5. In fact, at k*=3 or 4 the order is 2 whenever the saddle is strict (stars, wrong order); only a lower order at a non-strict saddle starts at k*=5. In finding 4, 'it' follows 'never happens for two-component teachers learned in weight order', yet the counterexample has three components.
- Evidence: cor:A-order(ii). Sec. 4.5: 'For k*=3 or 4 it happens exactly when the saddle is strict. From k*=5 on it can happen while the Hessian is positive semidefinite'. App. A: 'only for k*>=5 (given a fixed-rank block positive definite modulo gauge)'. Sec. 6.6: stars are strict with every route k*>=3.
- Fix: Abstract, bullet and Sec. 1.3: 'from k*=5 on the order can be lower even when the saddle is not strict'. Finding 4: 'From k*=5 on it can happen with a positive semidefinite Hessian: on MPS6 a three-component teacher with k*=5 has order 4'.

### intro-summary-6 (minor) [minor, unverified]
- Location: main.tex l.148-149 (abstract)
- Quote: Non-orthogonal teachers make the next saddle strict on matrix product states, not on binary or hierarchical Tucker trees.
- Problem: The contrast is stated with no scope. Review A m1 asked for 'whose internal cores carry no site'; that qualifier was dropped.
- Evidence: The mechanism needs internal cores without sites (finding 5; Sec. 6.2). On binary trees with N != 2^m an internal core carries a site (tab:trees caption). Sec. 6.2 scope: N=4, 8, d=2, correlated inputs never run on HT. Bullet 4 says 'In every case tested'.
- Fix: '... not on the binary and hierarchical Tucker trees tested, whose internal cores carry no site.'

### intro-summary-7 (minor) [minor, unverified]
- Location: main.tex l.151-152 (abstract)
- Quote: a lazy one, which needs every bond at the rank of the matricization it cuts
- Problem: 'The rank of the matricization' reads as the rank of a given tensor, such as the teacher's. The condition in the body is the cut dimension, the largest possible rank.
- Evidence: Sec. 5.1: 'A lazy regime therefore needs every bond at its cut dimension (binary4: chi>=9)', while the teacher's bond rank there is 3. Sec. 1.1 defines the cut dimension as 'the largest rank that the matricization of W across that bond can have'.
- Fix: 'which needs every bond at the largest rank a matricization across it can have'.

### intro-summary-8 (minor) [minor, unverified]
- Location: 01_introduction.tex l.150-154 (finding 1)
- Quote: on binary4, MPS4, binary8 and HT4 ... and the plateaus grow as $\isc^{-(\Kc-2)}$ at higher-order saddles
- Problem: The scaling is claimed for HT4, which Sec. 4.1 does not count as confirmed.
- Evidence: Sec. 4.1: 'HT4 is not a clean confirmation (per seed 3.93--5.78)'. tab:s2s-gaps, HT4 second gap: 4.75+-0.14.
- Fix: Append '(HT4 not cleanly: per seed 3.93--5.78)'.

### intro-summary-9 (minor) [minor, unverified]
- Location: 01_introduction.tex l.81-82 (six points, bullet 6)
- Quote: Adam keeps the first component but leaves the origin in a number of steps that does not depend on $\isc$
- Problem: Drops the epsilon caveat that Table 1 Q9 and finding 9 carry: with the default epsilon, the sigma law returns at small sigma.
- Evidence: Sec. 5.6: with epsilon=1e-8 the law sigma^-(K-2) returns below sigma_x of about 0.05-0.1 (MPS8).
- Fix: Add 'until its regularizer epsilon brings the gradient-flow law back'.

### intro-summary-10 (minor) [minor, unverified]
- Location: 01_introduction.tex l.285-286 and l.272-273 (Sec. 1.3)
- Quote: Locality beats weight has only analogues: the smallest component escaping first, and the lowest leap first
- Problem: The analogue is uncited and ambiguous: here 'component' otherwise means a teacher component. The quote attributed to Lemmas D.8-D.9 and Remark D.10 is from Furman's Fig. 4 caption.
- Evidence: citation_check B10: analogue 'Furman Fig. 4 (smaller core escapes first)'. B1: 'Cite Furman Fig. 4'. Summary item 2 places 'the smallest core delivers the least suppressed escape' in the Fig. 4 caption.
- Fix: 'the smallest connected set of rank-deficient cores escaping first (\citealp[Fig.~4, Remark~D.10]{furman2026benign})'. Cite Fig. 4 at l.272 as well.

### intro-summary-11 (minor) [minor, unverified]
- Location: 01_introduction.tex l.306 (Sec. 1.3)
- Quote: the exact $\isc$-independent race on MPS (the TTN form of the selection rule)
- Problem: Treats the exact race and the approximate selection rule as one thing; Sec. 4.7 keeps them apart.
- Evidence: Sec. 4.7: the leading-order race names the winner in all 40,800 MPS runs, while the TTN form of the selection rule 'names 76--92% of the winners'.
- Fix: 'the exact sigma-independent race on MPS, the TTN form of the selection rule (76--92% of winners)'.

### intro-summary-12 (minor) [minor, unverified]
- Location: 01_introduction.tex l.240-241 (Sec. 1.3)
- Quote: With three or more matrices the saddles visited from an exactly balanced initialization are non-strict
- Problem: The scope qualifier required by the citation audit is missing.
- Evidence: citation_check summary item 4 and a1 row 9: Achour et al. (Thm 7 and the remark after Prop. 8) cover saddles with S=[1,r] (top singular values) from exactly 0-balanced initialization, for full-rank targets with distinct singular values.
- Fix: '... every saddle that selects the top singular values, visited from an exactly balanced initialization, is non-strict (full-rank targets with distinct singular values)'.

### intro-summary-13 (minor) [minor, unverified]
- Location: 01_introduction.tex l.117 (Table 1 Q4); l.187 (finding 6)
- Quote: \conjecture{} (rate)
- Problem: The tag for the power-law rate disagrees with Sec. 4.7.
- Evidence: Sec. 4.7 (ii) and 'Matrix-like pairs': for n=1 the rate is '\empirical{} (heuristic mechanism; the data below support it)'; only n=chi_e is a conjecture (lead decision, 04 header). Finding 6 tags the same claim '\conjecture{} \empirical'.
- Fix: '\empirical{} (rate, heuristic mechanism; adjacent cores: \conjecture)'.

### intro-summary-14 (minor) [minor, unverified]
- Location: 01_introduction.tex l.320-321 (Sec. 1.3, Lazy and rich training)
- Quote: Known in our setting: maps of regimes as such, the effect of a light layer
- Problem: Sec. 1.3 calls the light-layer effect known, while Sec. 3.4 marks the light-root result new and citation_check B23 keeps it new. The imbalance result is missing from the list of new items.
- Evidence: Sec. 3.4: 'A light root makes training lazy; a heavy one does not. \empirical{} \newresult[for TTNs]'. citation_check B23: Kunin 2024 and Domine 2025 are 'related'; suggested wording 'keep'.
- Fix: 'the effect of a light layer in two-layer networks (...)', and add 'the root-imbalance map' to the new items for finding 8.

### intro-summary-15 (minor) [minor, unverified]
- Location: 01_introduction.tex l.176-221 (main findings 5-10)
- Problem: The main findings exceed the binding limit of five lines each.
- Evidence: notes/report_style.md: 'Main findings: at most five lines each ... numbers live in the sections'. In main.pdf pp. 7-8, findings 5, 6 and 10 take 6 lines, 7 and 9 take 7, and 8 takes 8 (it carries 1.07-2.03, 85 of 96, 76 of 96 and sqrt d).
- Fix: Move per-number detail (for example the finding 8 bounds and seed counts) to the sections cited.

### intro-summary-16 (minor) [minor, unverified]
- Location: 07_open_questions.tex l.199-203 (open question 2)
- Quote: on one teacher only, the escape law was checked from the saddle and from small initialization, the forgetting from small initialization
- Problem: Says forgetting at an order-below-k* saddle was seen on one teacher only; Sec. 4.5 reports it on a second.
- Evidence: Sec. 4.5: at the order-6 HT4 saddles (D={0,1,2,3}, wrong order), 'in 6 of the 7 such runs ... the smaller component is forgotten ... and then relearned'. Sec. 4.6: one HT4 seed shows the predicted cluster signature (0.2998 vs 0.300).
- Fix: 'the escape law on one teacher (MPS6); the forgetting there from small initialization, and in 6 of 7 HT4 wrong-order runs at an order-6 saddle'.

### intro-summary-17 (minor) [minor, unverified]
- Location: 01_introduction.tex l.191-195 (finding 7)
- Quote: most runs with $\Kc=7,8$ and the bond dimension equal to the teacher rank never learn the last component
- Problem: 'Never' overstates: many of these runs are slow (out of budget), not dead. The initialization-only criterion that follows works only for pattern teachers.
- Evidence: Sec. 4.8: dead routes in at least 9/30 (binary8), 24/30 (HT4) and 14/30 (MPS8) runs, the rest slow. Regime-map stall test: 'do not learn the last component within the budget'. D_s2s.tex: for odeco teachers 'the trap is predicted from the state after the second escape ..., not from the initialization'.
- Fix: 'do not learn the last component within the budget (many with a dead route)'; and 'for pattern teachers, computed from the initialization alone, ...'.

## setup-regimes

### setup-regimes-01 (major) [verifier: confirmed]
- Location: report/sections/03_regime_map.tex l.158-165 (Sec. 3.3, 'The limit at large K')
- Quote: so in the plain model $\isc_c$ does not tend to 1
- Problem: The sentence states a K→∞ limit from data at K ≤ 15. It also reads as the outcome of the 'pre-registered second sweep' introduced two sentences earlier, but the decisive comparison (censored lower bounds against laws extrapolated from the resolved cells) was not pre-registered. The rejection also depends on the domain-censoring amendment.
- Evidence: round3 verification verdict #8: 'confirmed within scope (binary, chi = 4, teacher (3, 2), K <= 15). Caveats: the rejection uses the domain censoring (amendment 4.6) and a comparison that was not registered'. Safe-to-cite #2 says 'rejected for the plain model at K <= 15'. notes/regime_map.md 4.8 gives the same status. D_regimes.tex l.98-99 already states the scoped version.
- Fix: 03 l.164-165: '... so σ_c → 1 is rejected for the plain model at K ≤ 15 (binary trees, χ = 4, teacher (3,2), d = 2, 3); this comparison of the bounds with the laws, and the domain censoring it relies on, were added after the pre-registration.' Add the same clause at D l.98-99, and qualify the intro bullet (01 l.201-205) and Sec. 7 (l.249-251) to match.
- Verifier: 03 l.159-165 opens with 'A pre-registered second sweep ... tests them' and ends 'so in the plain model σ_c does not tend to 1', with no caveat. Two sources state caveats the report leaves out. regime_map_round3_verification #8 confirms the claim only within scope (binary, χ=4, teacher (3,2), K≤15) and adds: 'the rejection uses the domain censoring (amendment 4.6) and a comparison that was not registered'. Its #2 lists the censored-vs-law comparison as a post-hoc addition. notes/regime_map.md 4.8 has the same caveat in its status line. The pre-registration was commit f57a28c at 12:26; amendment 4.6 item 2, the domain rule, came after the first arm-P rows. No report section mentions an amendment or a non-registered comparison: grep finds nothing, and B l.766 and D l.90-94 present the domain rule as part of the pre-registered sweep. Safe-to-cite #2 words it 'rejected ... at K≤15'; Sec. 3 states it as a limit. The scope part is the weaker half, since N=4-16 appears in the same paragraph. The missing pre-registration caveat is the real overclaim. The same unqualified wording is in the intro (01 l.201-205) and Sec. 7 (l.249-255), and D l.98-99 also lacks the caveat.

### setup-regimes-02 (major) [verifier: confirmed]
- Location: report/sections/D_regimes.tex l.121-122 vs 03_regime_map.tex l.147-148
- Quote: The plain model agrees with the centred one up to $\Kc=4$ and lies above it from $\Kc=5$.
- Problem: This contradicts Sec. 3 ('the same boundaries within one grid step up to K=7'). It is also wrong at K=7 and at d=3. The round-2 arms differ in χ as well as in W_0: the plain arm has χ=4, the centred arm full bonds, and Sec. 3 itself shows that extra channels lower σ_c. The sentence therefore cannot isolate the unlearning effect it is used to support.
- Evidence: round2/cells_P_rho3.csv vs cells_C_rho3.csv, d=2: K=5 0.586 vs 0.501; K=6 0.850 vs 0.652; K=7 0.777 vs 0.807 (plain below); K=8 1.116 (2/6 resolved) vs 0.833. d=3: K=3 0.202 vs 0.148; K=4 0.436 vs 0.280. Round 1 at equal χ=4 (sigma_c_vs_K.csv): plain = centred at K=3, 5, 6, 7.
- Fix: Replace D l.121-122 with: 'At d = 2, K = 3, 4 the two arms have the same bonds (χ = 4 is full there) and agree seed by seed. From K = 5, where the centred bonds exceed 4, the plain medians are higher at K = 5, 6, 8 (0.59 vs 0.50, 0.85 vs 0.65, every seed ≥ 1.07 vs 0.83) and about equal at K = 7 (0.78 vs 0.81). At d = 3 (centred bonds 9) they are higher already at K = 3, 4 (0.20 vs 0.15, 0.44 vs 0.28). The arms differ in χ as well as in W_0, so this comparison does not isolate the unlearning of W_0; at equal χ = 4 the first sweep gives the same boundaries up to K = 7 (\cref{sec:regime-K}).'
- Verifier: D l.121-122 compares round-2 arm P (plain, χ=4) with arm C (centred, full bonds). It does not say the two arms have different χ. I computed the full-bond cut dimensions with ttn.tree.binary_tree. At d=2 they are 4 at K=3 and 4, so the two arms have identical bonds there, and per_seed_P/C.csv agree almost seed by seed (e.g. 0.265/0.265, 0.446/0.447). From K=5 the centred bonds are 8 (16 from K=7). The divergence therefore starts exactly where χ starts to differ, so the sentence cannot isolate the unlearning of W_0. At equal χ=4, round 1 (sigma_c_vs_K.csv) gives identical plain and centred values at K=3, 5, 6, 7 (0.170, 0.615, 0.682, 0.752), which is what Sec. 3 l.147-148 says. Read side by side, the two sections contradict each other unless the χ difference is stated. The sentence is also wrong at d=3, where the centred bonds are 9: plain 0.20 vs 0.15 at K=3 and 0.44 vs 0.28 at K=4, with two plain seeds censored above 0.996. One detail of the finding is fragile. 'Below at K=7' (0.777 vs 0.807) uses resolved seeds only; if the censored plain seed (bound 1.36) is counted, the plain median is about 0.86. So K=7 is a tie within noise, not clearly below. Also, Sec. 3 does not cleanly show that extra channels lower σ_c up to the cut dimension: at K=7, χ=4, 8, 16 all give 0.83. The confound stands either way.

### setup-regimes-03 (major) [verifier: confirmed]
- Location: report/sections/03_regime_map.tex l.211-213 (tab:regime-summary, thresholds row)
- Quote: $\times1.30$--$1.35$ for $\delta_*$: $0.05\to0.2$ ($1.27$--$1.36$)
- Problem: The scope is missing. The factor 1.30-1.35 holds only at χ ≥ 32; over most of panel (a) (χ = 9-24) the lazy boundary moves much more.
- Evidence: Round-2 verification, safe-to-cite #6: '1.30-1.35 at chi >= 32 (P7) and by 1.97 at chi = 9'. fix1/lazy_summary_chi_b4.csv tensor ratios: 1.82, 1.72, 1.47, 1.39 at χ = 9, 12, 16, 24. D_regimes l.80-81 gives the scoped statement.
- Fix: '×1.30–1.35 for δ_*: 0.05→0.2 at χ ≥ 32 (1.27–1.36), up to ×1.97 at χ = 9'
- Verifier: In fix1/lazy_summary_chi_b4.csv, ratio_005_02_tensor is 1.342, 1.313, 1.309, 1.296 and 1.300 at χ = 32-128. It is 1.819, 1.720, 1.474 and 1.389 at χ = 9, 12, 16 and 24. sensitivity_per_seed_sigma_chi_b4_centred.csv gives 1.965±0.077 at χ=9, 1.71 at 12, 1.49 at 18, 1.40 at 24 and 1.33 at 36. Round-2 verification safe-to-cite #6 scopes the factor to 'χ ≥ 32 (P7) and 1.97 at χ = 9'. D l.80-83 gives the scoped statement, and the bracketed prediction 1.27-1.36 comes from slopes at χ ≥ 32 only. The table row gives neither scope, while its ρ entry (2.14-2.34) does cover the whole χ = 3-36 sweep. The row therefore understates the lazy-boundary sensitivity over most of the map. The proposed fix is correct; at χ=9, 1.97 is the σ×χ-sweep mean that D also uses.

### setup-regimes-04 (major) [verifier: confirmed]
- Location: report/sections/02_setup.tex l.548-549 vs Sec. 3 (03 l.143-153, 101-108, 229-236)
- Quote: Every regime boundary comes with the threshold sensitivity check of \cref{tab:regimes}
- Problem: Sec. 3 reports threshold sensitivity only for the binary4 σ×χ map (912 centred labels) and the round-2 fits (ρ* = 2, 5). The round-1 K-sweep boundaries (0.17-0.75 and the brackets near 1), the K=7 boundaries (0.83, 0.53, kernel 1.45) and the imbalance boundaries ('factor 100 in α') have none, and nothing marks them as exceptions.
- Evidence: D_regimes l.76-87, the only sensitivity paragraph, covers binary4 only. App. B l.712-713 says the labels were computed at all nine threshold pairs, so the data exist. Final review A, M6, rated the same kind of universal-rule gap major.
- Fix: Either report the boundary shifts for the K sweeps, binary8 and the imbalance map from the stored nine-pair labels, or change 02 l.548-549 to: 'Threshold sensitivity (the variations of tab:regimes) is reported for the binary4 boundaries and the large-K fits (App. D); the other boundaries (K sweeps, binary8, imbalance) use the default thresholds only.'
- Verifier: 02 l.548-549 states a universal rule: 'Every regime boundary comes with the threshold sensitivity check of tab:regimes'. The only sensitivity results in the report are these: D l.76-87 (binary4 only: ρ 2→5 over χ=3-36, δ_* 0.05→0.2, 70 of 912 centred binary4 labels), the tab:regime-summary row derived from it, and the round-2 β fits at ρ* = 2, 5 (D l.104-108). A grep for threshold variations finds nothing else. Several Sec. 3 boundaries have no check: the round-1 K sweep (0.17-0.75 and the brackets near 1), the K=7 boundaries (0.83, 0.53, kernel 1.45), and the imbalance boundary (factor 100 in α). notes/regime_map.md 2.3, 2.4 and 2.6 report none either, and the committed CSVs for these sweeps (boundaries_per_seed_sigma_K_*, *_b8*, report/imbalance_map_b4.csv) have no per-threshold columns. App. B l.716-717 says labels were computed at all nine threshold pairs, so the shifts could be extracted from the raw data. Nothing marks these boundaries as exceptions. This is the same kind of gap as final review A M6, whose fix restricted only the scaling-fit half of this paragraph. The proposed fix is correct.

### setup-regimes-05 (minor) [minor, unverified]
- Location: report/sections/03_regime_map.tex l.87-94 and table l.195
- Quote: slope $-0.53\pm0.05$ ... ($-\frac13$, $2.7$--$2.9$ standard errors from the saddle-to-saddle slope)
- Problem: The paragraph quotes the pooled SE (0.053) but the distance computed with the per-seed SE; with ±0.05 the distance is 3.7-3.9 SE. Separately, the table's 'σ_c = 0.24 to 0.023' uses the geometric mean at χ=128 while the text uses medians (σ_c√χ = 0.27-0.33): 0.023·√128 = 0.26 falls outside that range.
- Evidence: numbers.csv: '-1/3 from the S2S slope, chi >= 16: pooled SE, per-seed SE: 3.67, 2.67'; per-seed fit −0.528 ± 0.073. fix1/s2s_per_seed_chi_b4.csv at χ=128: median 0.0244, geometric mean 0.0231.
- Fix: '2.7–2.9 standard errors in per-seed fits (3.7 pooled)'. In the table, use the median 0.024 or say 'geometric means'.

### setup-regimes-06 (minor) [minor, unverified]
- Location: report/sections/03_regime_map.tex l.74-76; table l.196
- Quote: it lies $1.5$--$6\%$ above the bracket ($\isc_{\rm lazy}\sqrt\chi=8.96$ at $\chi=9$)
- Problem: The χ=9 value (and the table's 3.0) rests on 3 of 6 seeds of the χ sweep. The six-seed σ×χ sweep gives 3.16 (σ√χ = 9.47), about 12% above the bracket edge 2.81.
- Evidence: fix1/lazy_summary_chi_b4.csv: n_tensor = 3 at χ=9. tensor_relabel/lazy_boundary_summary_sigma_chi_b4.csv: centred χ=9, tensor_med 3.156 (n=6). Round-2 verification E6: 'chi = 9 has only 3 seeds with a tensor boundary'.
- Fix: Add '(3 seeds; 3.16, 12% above, in the six-seed σ×χ sweep)'.

### setup-regimes-07 (minor) [minor, unverified]
- Location: report/sections/02_setup.tex l.497 (tab:regimes, lazy row)
- Quote: the loss converges and $\dTK(T)<0.1$ (so $\dth$ is small as well)
- Problem: This implication is unsupported, and it conflicts with Sec. 5.1, 'Lazy by the kernel is not small motion': under SP the root's own kernel block does not depend on the root.
- Evidence: 05_other_regimes.tex l.68-73: 'the root moved by more than 100% while δΘ<0.1'. A_derivations l.1452 says the same.
- Fix: Drop the parenthesis, or write '(δθ and per-core changes reported alongside, Sec. 5.1)'.

### setup-regimes-08 (minor) [minor, unverified]
- Location: report/sections/02_setup.tex l.402-404
- Quote: in general a best approximation of $\Wt$ with the current bond ranks
- Problem: Sec. 2 defines learned saddles as best approximations, which is stronger than later sections support. Sec. 6 finds plateaus at critical points of the bond-rank-constrained loss and says they approach a local best approximation only as a conjecture. Sec. 4's best-approximation route rule is a conjecture as an algorithm.
- Evidence: 06 l.41-42; l.279-284 ('should approach a local best rank-one approximation ... \conjecture'; three teachers separated from every SVD-type truncation); 04 l.527-534.
- Fix: 'in general a critical point of the loss restricted to the current bond ranks (Sec. 6)'.

### setup-regimes-09 (minor) [minor, unverified]
- Location: tab:notation, 02_setup.tex l.657, 701, 708, 712 vs App. A
- Quote: $\Delta t_2$, $\lambda_2$; $A$ & time between the first two events; $k=2$ escape rate
- Problem: λ, λ_2 and r_1 have second meanings that the table does not name, against the one-symbol rule.
- Evidence: App. A.4 l.932, 936: λ_2 = λ_max(Γ^(2)) ('a star with λ_2=0.8√2=1.131', 'λ_2=0.85'). App. A.3 eq:A-lambda2 uses bare λ for the k=2 rate, while the table reserves bare λ for the root scale. A.7 l.1295 and D_optim l.19 use λ_2(H). The A.4 example names residuals r_1=110001, r_2=110010, while the table defines r_1 as the component learned first.
- Fix: Write λ_ℓ with ℓ=2 explicitly in A.4 and λ_2 in eq:A-lambda2, rename the example residuals, and name any remaining second uses in the table.

### setup-regimes-10 (minor) [minor, unverified]
- Location: tab:notation l.614-616, 688-689, 710-711, 719-722, 737; 02_setup.tex l.59, l.110
- Quote: $C_u$; $Q_{\nu\nu}$; $X_{\rm nv}$, $Y_{\rm nv}$ & conserved quantity of a route end $u$
- Problem: Several letters have two meanings that the table does not name:
- C with a core subscript means the environment contraction (l.59, prop:A-kernel) and also the route-end conserved quantity (prop:A-trap, Sec. 4).
- I is listed only as the regularized incomplete beta function, but it is the identity throughout Secs. 2-3 (Σ_i=I, E Q_e=σ²(1−λ²/χ)I).
- c_1, c_2 (kernel constants) collide with c_r (route coupling) at r=1, 2.
- c_e uses the bond letter e as an event index.
- The tab:trees caption uses 'N=2^m', while m=|V_D|.
- Evidence: Table rows cited above; 02 l.16, 384; 03 l.224; 04 l.811-827.
- Fix: Rename one member of each pair (e.g. an environment E_v, a tree-depth exponent with another letter), add the identity I to the table, and name the rest.

### setup-regimes-11 (minor) [minor, unverified]
- Location: 03_regime_map.tex l.49, l.98; D_regimes.tex l.65, 67, 106; B l.159; 05 l.424; tab:notation l.641, 677-678
- Quote: the width-dependent scales $\isc^2w$ of \citealp{tu2024mixed} and $\fsc\sqrt h$
- Problem: Several symbols are used outside their table meanings or are untranslated:
- f is the geometric SE factor (Sec. 3) and the onset fraction (App. B), but a bond in the table.
- The cited widths w and h are not translated (w is the weight ratio).
- χ² is a chi-square statistic in App. D but the squared bond dimension in Sec. 5 (l.69, 125).
- Sec. 5 l.424 'L_1/L_0, L_2/L_0' uses the dropped L_k notation, which clashes with L_S.
- The table lists ℓ, which is never used, and points δ_* to tab:regimes, which never introduces it.
- Evidence: grep: \lossn appears only in the table; tab:regimes uses '0.1', not δ_*; style spec: 'Drop the L_k notation; use L_S', 'Translate symbols of cited papers into ours'.
- Fix: Rename or define each in place (σ²×width; 'χ² statistic' as a word); write L_{1}/L(0); remove ℓ; define δ_* in Sec. 2.8.

### setup-regimes-12 (minor) [minor, unverified]
- Location: 07_open_questions.tex l.74; 05_other_regimes.tex l.280-288; 07 l.123, 129
- Quote: (29 trapped, one diverged)
- Problem: Terminology drifts from the Sec. 2 definitions:
- Sec. 2 defines 'route size k', but Sec. 7 ('route degrees') and App. A ('route degree', 'escape degree') use an undefined synonym.
- Sec. 2 defines 'dead route (trap)' by the conservation-law mechanism, and the style spec drops 'trapped' as a label. Sec. 5/7 nevertheless call SGD stochastic-collapse runs 'trapped' or say SGD 'traps'.
- Evidence: 02 l.420, 519-524; style spec 'Drop "trapped" as a separate label'.
- Fix: Define 'degree' as a synonym in Sec. 2.7, or use 'size' throughout. Name the SGD outcome as a different mechanism (e.g. 'dead route by stochastic collapse').

### setup-regimes-13 (minor) [minor, unverified]
- Location: 02_setup.tex l.413-414; 03_regime_map.tex l.72, l.208
- Quote: when such moves descend faster than the cheapest route is decided in \cref{app:saddle-order}
- Problem: Three wording slips:
- Garbled sentence (should read 'whether').
- l.72 '\derived{} \cref{prop:A-lazy} With the tensor kernel' lacks parentheses and a full stop, so it renders as one run-on sentence.
- The table row gives 'δΘ∝χ^{−3.02±0.06}' without the 'SP:' label, although NTK and μP-TTN are labelled.
- Evidence: Source lines quoted.
- Fix: 'whether such moves descend faster than the cheapest route is decided in ...'; '(\cref{prop:A-lazy}). With ...'; 'SP: δΘ∝χ^{−3.02±0.06}'.

## s2s

### s2s-1 (major) [verifier: confirmed]
- Location: report/sections/04_saddle_to_saddle.tex l.62-66 (box, 'Why routes')
- Quote: The next plateau is therefore set by $k^*$: it is absent ($k^*=1$), lasts $\propto\ln(1/\isc)$ ($k^*=2$), or lasts $\propto\isc^{-(k^*-2)}$ ($k^*\ge3$).
- Problem: The box states the k* plateau law without the spectral condition. Claims-A M2 removed this same overclaim at l.16-18 and l.336-338, and Sec. 4.4 contradicts it. The Steiner-route sentence is tagged \proven without its teacher class.
- Evidence: Sec. 4.4 (l.477-483) says the order equals k* only when no piggyback coupling exceeds s_j; from k*=5 on it can drop (the MPS6 teacher escapes in sigma^-2, not sigma^-3). Sec. 4.1's binary4 1-3-2 plateaus have k=3 but are strict (gap exponent -0.08, not sigma^-1). experiments/saddle_hopper/results/gp_saddle_order.csv: the order is below k* at 177 of 587 reachable rank-one saddles. prop:A-steiner assumes orthogonal pattern teachers (claims-A m3).
- Fix: l.63: 'If nothing descends faster than the route (the spectral condition of \cref{sec:s2s-saddle-order}, proven for one learned component: always for $k^*=2$ and for two-component teachers learned in weight order; it fails at wrong-order and star saddles, \cref{sec:s2s-several}), the next plateau is set by $k^*$: ...'. l.66: '... \proven{} (orthogonal pattern teachers, one learned component; \cref{prop:A-steiner})'.
- Verifier: Box l.62-64 says 'The next plateau is therefore set by k*: ... lasts ∝σ^-(k*-2) (k*≥3)' with no condition. The section's opening (l.16-18, after claims-A M2) and Sec. 2 setup-routes ('when the saddle order is k* (below)') both carry the qualifier. The box cites prop:A-escape-degree, but that proposition states it 'fails at the strict saddles of prop:A-piggyback-graph' and needs the generalized piggyback operator below s_j. The report contradicts the box in two places: Sec. 4.1 l.153-156, where binary4 (k=K=3) has a strict 1-3-2 saddle with gap ∝σ^{0.08} (D_s2s l.78), and Sec. 4.4 l.477-507 (order 4 < k*=5 on MPS6). gp_saddle_order.csv has order_drop=True at 177 of 587 rows (147 at level 1 with order 2, 30 at level 3 with order 6). The box's Steiner sentence is tagged \proven with no teacher class, yet prop:A-steiner assumes (A2), orthonormal pattern teachers, and one learned component. Sec. 2 ('For an orthonormal pattern teacher with one learned component') and intro l.163-164 (claims-A m3, applied) both state this. The box text came from the readability review HI-3 proposal, which was not a claims check. The proposed fix is essentially right; the version below also covers saddles with several learned components and k*=2.

### s2s-2 (major) [verifier: confirmed]
- Location: 04 l.656-657 (order prediction (iii) 'Kept'); l.730-735 (test); l.771 (Status); fig:order-wrong caption l.751-752; l.511-518; l.150-151
- Quote: \emph{Kept} for $m\ge3$: no negative Hessian eigenvalue; $\U1$ is added after a plateau $\propto\isc^{-(k-2)}$ and the wrong order stays.
- Problem: Sec. 4.6 says that tensor-like wrong orders are always kept with a sigma^-(k-2) plateau, and it cites binary8 sigma^{5.2+-0.3} as confirmation. Sec. 4.4, App. A rem:A-wrong-order and the hopper analysis say otherwise when V_D splits into two low-degree sets. The same unstated exception affects 4.1 prediction (iii), 'every plateau ... lasts sigma^-(K-2)', at wrong-first saddles. 'Which experiments satisfy the condition' (l.511-518) leaves out these binary8 runs.
- Evidence: theory_saddle_order.md S12 and section 7 item 5: binary8 odeco from the wrong saddle has order 6 < k=7. Its asymptotic exponent is 4; the measured 5.25 at sigma=0.15-0.35 lies at or above the crossover b~0.155. notes/saddle_hopper.md section 9 lists 'order binary8 two-component (rho=0.5,0.8,0.95)' among the 30 order-6 saddles. 12 of 13 locality 'transient dips' on tensor-like pairs are forgetting events at order-6 or tie saddles. Sec. 4.4 l.516: 6 of 7 HT4 wrong-order runs forget and relearn, against Status 'tensor-like pairs keep the wrong order' and the caption 'for m>=3 nothing corrects the wrong order'. Also 'every run follows the predicted branch' fails for one of six MPS6 seeds, which stalls at every sigma (order_round2 M5).
- Fix: (iii) 'Kept' for m≥3: append 'unless the differing cores split into two clusters whose Steiner subtrees have fewer than k/2 cores (binary8 odeco; HT4 with D = all sites): there the order is 6 < k = 7 \proven, below a crossover (σ≈0.155 for binary8 at w=0.8) the escape is expected ∝σ^-4 \derived, and the smaller component is forgotten and relearned (\conjecture; 6 of 7 HT4 runs \empirical; \cref{sec:s2s-saddle-order})'. 4.1 (iii)/(iv): add 'except at saddles where a smaller component was learned first on binary8 and HT4 (order 6: asymptotically σ^-4; the HT4 seed at 4.03)'. Test l.730: 'every run follows the predicted branch (one MPS6 seed of six stalls)'. l.734: 'binary8 σ^{5.2±0.3} (σ=0.15-0.35, at or above the crossover below which the order-6 escape, ∝σ^-4, is expected to win)'. l.514-518: name the binary8 order-experiment and the s2s_core HT4/binary8 wrong-first runs among the 30. Status l.771: 'tensor-like pairs keep the wrong order where the differing cores do not split into two small clusters (HT4, D = all sites: forgotten and relearned in 6 of 7 runs)'. Caption: 'for m≥3 no strict saddle corrects the wrong order'.
- Verifier: (iii) at l.656-657 ('Kept for m≥3: ... plateau ∝σ^-(k-2) and the wrong order stays') leaves out the caveat that its own source rem:A-wrong-order(3) gives: with a split into two sets of degree below k/2, 'the order is 2ℓ*<k ... j forgotten and relearned (\conjecture; 6 of 7 HT4 runs \empirical)'. gp_saddle_order.csv rows 490/492/494 (order binary8 00000000/11111111, w=0.5/0.8/0.95, wrong-first) have order 6, k*=7. theory_saddle_order S12 and §7 item 5 say order 6 is proven, the exponent is 4 below the crossover b≈0.155 (derived), and the binary8 runs (σ≥0.15, t1 exponent 5.25±0.27) sit 'at or above the crossover'. So citing binary8 σ^{5.2±0.3} as confirming σ^{k-2} omits that it is pre-asymptotic. Sec. 4.4 l.514-516 (6 of 7 HT4 D={0,1,2,3} wrong-order runs forget and relearn) contradicts both Status l.771 ('tensor-like pairs keep the wrong order') and the caption l.751-752 ('for m≥3 nothing corrects the wrong order'). saddle_hopper.md §9 adds 12 of 13 'transient dips' at order-6 or tie saddles, where the learned component is forgotten. The same gap affects 4.1 (iii) and (iv): the s2s_core odeco wrong-first saddles on HT4/binary8 (CSV rows 288-331) have order 6, and the HT4 seed with exponent 4.03 is the polymer escape (D_s2s l.296-299). 4.1 notes only the binary4 exception and calls HT4 'not a clean confirmation' without saying why. 'Every run follows the predicted branch' (l.730) also clashes with D_s2s l.374 ('one MPS6 seed of six stalls') and order verification C14/M5. The proposed fix drops the tags: the forgetting at small σ is a \conjecture (empirical for HT4), the σ^-4 law is \derived below the crossover, and the order is \proven.

### s2s-3 (major) [verifier: confirmed]
- Location: 04 l.597-599 (hopper paragraph); l.512-514
- Quote: Checked at the 587 rank-one saddles that the scored runs can reach, the condition of \cref{sec:s2s-saddle-order} fails at 30, where the order drops from $7$ to $6$
- Problem: The count is wrong. Under the definition in Sec. 4.4, which counts strict saddles at k*=3,4 as failures, the condition fails at 177 of the 587 saddles, not 30. Separately, 'every saddle with one learned component of the multi-component teachers ... (there lambda_max(Gamma)<=0.7<s_j)' holds only for correct-order saddles.
- Evidence: gp_saddle_order.csv: order_drop=True for 177 rows: 147 at level 1 (k*=3-8, order 2) and 30 at level 3 (order 6). saddle_hopper.md section 9: '147 saddles (all k*) have lambda_max(S_1) > s_q: classical piggyback, strict'; 'No correct-order saddle drops'. Wrong-first saddles of TA, TB and N4 on HT4, B4 on binary4, MPS4 and HT4, and E8 and TC drop to order 2. Example: TA with 1100 learned has lambda_max(Gamma)=1.0 > 0.6. Runs visit them: levels_summary.csv lists TA HT4 1>0>2 in 4+1 runs, TB in 2+1, N4 in 1+1.
- Fix: l.597-602: 'Checked at the 587 rank-one saddles that the scored runs can reach, the condition of \cref{sec:s2s-saddle-order} fails at 177: at 147 the saddle is strict (wrong order or stars; order 2), and at 30 the Hessian is positive semidefinite and the order drops from 7 to 6 \proven{} (...); these 30 are all reached by a wrong first escape on binary and HT trees ...'. l.512-513: 'and every correct-order saddle with one learned component of the multi-component teachers of \cref{sec:s2s-several} (there λmax(Γ)≤0.7<s_j; their wrong-first saddles, visited by a few HT4 runs of TA, TB and N4, are strict)'.
- Verifier: Sec. 4.4 l.478-483 defines the condition as 'no piggyback coupling exceeds s_j', and 'for k*=3 or 4 it happens exactly when the saddle is strict'. Recomputed from gp_saddle_order.csv: 587 rows, order_drop=True in 177. Of these, 147 are at level 1 (order 2, k*=3:92, 4:16, 5:26, 6:10, 7:1, 8:2) and 30 at level 3 (order 6, k*=7). So 'the condition fails at 30' (l.598) undercounts. It also conflicts with the report's own appendix D_s2s l.291-293 ('At 147 of them λmax(Γ)>s_q: strict saddles, wrong order or stars ... at level ℓ=3 ... 30') and with saddle_hopper.md §9. The l.512-514 claim ('every saddle with one learned component of the multi-component teachers ... λmax(Γ)≤0.7<s_j') holds only on the main (correct-order) branch, where λmax ≤ 0.7. The rare wrong-first saddles are strict: TA ht4 λ/s_q=1.24, TB ht4 1.19, N4 ht4 1.67/1.5, B4 binary4/mps4/ht4, E8 binary8/mps8, TC binary8. Runs do visit them: levels_summary.csv has TA ht4 1>0>2 in 4+1 runs, TB ht4 in 2+1, N4 ht4 in 1+1. One detail of the evidence is imprecise: for TA the reachable saddle is the rank-one combination of 1100 and 1110 (1110 is k=1 from 1100; s_q=0.695, λmax≈0.86), not 1100 alone with λ=1.0. The conclusion is unchanged. The proposed fix is correct.

### s2s-4 (minor) [minor, unverified]
- Location: 04 l.286-287 (first-escape test)
- Quote: With the offset removed, the per-seed exponents are $\Kc-2$ to within $0.004$, except in that cell ($0.990$--$0.996$).
- Problem: These are cell means, not per-seed values. The per-seed deviations are larger.
- Evidence: experiments/s2s_core/results/first_fits.csv: per seed, the largest deviation outside binary4-close is 0.0071 (HT4 chi=6: 4.9929; binary8 close: 4.9931). Binary4 close per seed ranges 0.966-0.9997 (chi=6 fan-in minimum 0.9659). 0.990-0.996 is the range of the four binary4-close cell means (first_summary.csv; s2s verification E17: 'per cell').
- Fix: 'the cell means are K-2 to within 0.004 (per seed 0.007), except binary4 with the close teacher (cell means 0.990-0.996, per seed down to 0.966)'.

### s2s-5 (minor) [minor, unverified]
- Location: 04 l.71-72 (box example); l.553-554 (Locality beats weight)
- Quote: $0110$ before $1001$ in all 60 runs
- Problem: 'All 60 runs' leaves out 36 of the 96 B4 runs on MPS4, all of which stall with a dead k=4 route, so the sigma^-2 fit uses 5 of 8 seeds. The box also calls the k=2 level 'a logarithmic plateau', but the body says it is too short to be detected.
- Evidence: data/locality multicomp + multicomp_wide: B4 MPS4 has 96 runs (12 sigma x 8 seeds). 36 of them, seeds 1, 2 and 6 at every sigma, stall with overlaps (1,1,0) and L=0.18: 0110 is learned and 1001 never is. levels_runs.csv keeps 60 runs (seeds 0, 3, 4, 5, 7). levels_summary.csv: the k*=2 level is detected in 0 of 60 runs, and the fair comparison is 0.222 vs 0.250, nearly tied.
- Fix: '0110 before 1001 in every run (60 complete; in the other 36, three seeds at every sigma, the k=4 route dies before 1001)'. In the box: 'k=2: a short logarithmic level'.

### s2s-6 (minor) [minor, unverified]
- Location: 04 l.822-824 and l.830-831 (trap criterion)
- Quote: the criterion separates the 3,492 fan-in locality runs with $k\ge4$ ($89\%$ of the predicted-dead runs stall, against $1.8\%$ of the predicted-alive ones)
- Problem: The 3,492 runs include a third class, predicted pinned, that the sentence does not report. Also, 'whether a run traps depends on its direction, not on sigma' is stated without 'as sigma->0', although the data contradict it at the sigma used. The withdrawn list forbids 'a stalling seed stalls at every sigma'.
- Evidence: theory_trap.md section 7: 3,492 = 457 predicted dead (408 stall) + 934 predicted pinned (141 stall, 15%) + 2,101 predicted alive (38 stall). locality.md 7.5: 49 stalling seeds stall at every sigma, 34 at some sigmas only. Trap verification T8: 45 of 65 predicted-dead seeds stall at every sigma, 16 are mixed.
- Fix: Add '(predicted pinned: 141 of 934 stall)'. Write 'as sigma->0, whether a run traps depends on its direction (at the sigma used, 34 of 83 stalling seeds stall at some sigma only)'.

### s2s-7 (minor) [minor, unverified]
- Location: 04 l.911-912
- Quote: the criterion above replaces it (on the same runs, 30 of the 38 predicted dead are dead, 7 of the 126 others)
- Problem: The two criteria were not evaluated on the same runs.
- Evidence: The R12 counts (30/55, 7/67) cover 122 fan-in and saddle-start runs (locality.md 7.5). The new counts come from the stall diagnosis: dead 30/6/2, pinned 5/63/0, alive 2/56/0, i.e. 164 runs including balanced ones (theory_trap.md section 7).
- Fix: '(on the 164 stall-diagnosis runs, 30 of the 38 ...)'.

### s2s-8 (minor) [minor, unverified]
- Location: 04 l.873-874
- Quote: 398 of 5,682 two-component runs stall, all with $k\ge4$ (none of 5,112 with $k\le3$)
- Problem: The 5,112 are not a subset of the 5,682, so the denominator for k>=4 reads as 570 when it is 2,122.
- Evidence: locality/results/twocomp_table.csv: the 5,682 runs include 3,560 with k<=3 (0 stalls) and 2,122 with k>=4. The 5,112 counts k<=3 runs of the two-component and robustness sweeps together (locality.md 7.5).
- Fix: '(none of the 3,560 with k<=3; none of 5,112 with the robustness sweep)'.

### s2s-9 (minor) [minor, unverified]
- Location: 04 l.591-594 (hopper)
- Quote: scored on all committed runs without tuning ... it matches $97.0\%$ of the event sequences, against $95.8\%$
- Problem: The 97.0% vs 95.8% comparison applies to the locality runs only, which the sentence does not say. 'Without tuning' is not supported by the verification.
- Evidence: saddle_hopper_verification sections 6 and 8: 97.0% vs 95.8% is for the informative locality runs; on s2s-core binary4 both give 96.6%. Section 2: 'No evidence either way'; rule R4c was added after reading the s2s-core notes.
- Fix: 'on the informative locality runs it matches 97.0% ... (s2s-core binary4: 96.6% for both)'. Replace 'without tuning' with 'with no per-case tuning'.

### s2s-10 (minor) [minor, unverified]
- Location: 04 l.202-203 (binary4 strict saddle)
- Quote: the runs that learn in the order 1-3-2 rest at $\lambda_{\min}$ between $-0.11$ and $-0.15$, flat in $\isc$
- Problem: The scope is missing: the range comes from the chi=3 plateaus only.
- Evidence: s2s_core.md 2.5 and verification E6: -0.106..-0.146 comes from the 9 chi=3 plateaus. At chi=6, seed 4 is 1-3-2 at 3 sigma with no resolved plateau (rho~0.3) and midpoint lambda_min -0.36..-0.53.
- Fix: 'at chi=3 the nine 1-3-2 plateaus rest at ...; at chi=6 these runs pass without a resolved plateau'.

### s2s-11 (minor) [minor, unverified]
- Location: 04 l.707-708
- Quote: $d$ enters only through the bond caps (\derived; no effect at equal bonds, $\abs{\Delta P}\le0.021$)
- Problem: The \derived tag covers binary8, where the claim is only empirical, and 'no effect' is stronger than the test supports.
- Evidence: order_round2 C5: 'derived (MPS); empirical (binary8, one chi)'. E3: 'differences larger than about 0.06 are excluded at 95%' ('no d effect at the 0.04 level' was withdrawn).
- Fix: '(\derived{} on MPS; at equal bonds |Delta P|<=0.021 on MPS4 and binary8, differences above 0.06 excluded at 95%)'.

## other-cov

### other-cov-1 (major) [verifier: confirmed]
- Location: 06_covariance_teacher.tex:555-561 (paragraph after the order-test bullets); also 06:513 (tab:cov-summary row 'real data') and 06:579 (fig:real-order caption)
- Quote: Locality in the image is neither necessary nor sufficient: the three orders with horizontal pixel pairs (quadtree, snake, row-major; S=0.989) rank 1, 2 and 4 of 27 by O1, but ... rank 13--21; ... (4 of its 8 bottom pairs happen to be adjacent in the image, against 1.6 expected by chance) ...
- Problem: Post-hoc analyses of the registered outcomes are given without an exploratory label, directly after the registered result. These are the rank comparisons of the structured orders, the adjacency count of scrambled110 and the conclusion 'neither necessary nor sufficient'. They read as part of the pre-registered finding. Only the plateau-state share is labelled 'Exploratory, not registered'.
- Evidence: notes/real_data.md section 9.5 (the registration) says: 'Anything else is labelled exploratory.' The registered hypotheses concern only Spearman(S, O1/O2) plus secondary analyses (a)-(c). The order ranks (recomputed from order_test.md: quadtree 1, snake 2, rowmajor 4, hilbert 13, quadtreeT 18, colmajor 21) and the 1.6-by-chance adjacency count (computed in a final-review sub-review) are outside that plan. The round-2 verification calls these items 'descriptive' (safe-to-cite row 2).
- Fix: 06:554-555: '... but not by their early speed. \emph{Descriptive, not registered} \empirical: locality in the image is neither necessary nor sufficient: the three orders ...'. 06:513 (tab:cov-summary): '(locality in the image neither necessary nor sufficient; descriptive, not registered)'. 06:578-579 (fig:real-order takeaway): '... and local orders with vertical or mixed pairs rank only 13--21 of 27 (descriptive, not registered)'.
- Verifier: The problem is real, but it is closer to minor than major. notes/real_data.md section 9.5 registers only Spearman (i)/(ii) and secondaries (a)-(c), and adds 'Anything else is labelled exploratory.' At 06:554-561 the post-hoc conclusion 'Locality in the image is neither necessary nor sufficient' rests on order ranks (1, 2, 4 vs 13-21) and an adjacency count (4 of 8 against 1.6 by chance; 24 of 120 grid pairs are adjacent, x 8 = 1.6, first computed in the final_review_claims_B m74 sub-review). That sentence carries no label and not even a status tag, although notes/report_style.md requires a tag on every claim. The next sentence labels only the plateau share 'Exploratory, not registered', which invites the reading that the rest of the paragraph is registered. Mitigations: all the numbers are correct (claims_B verified ranks 1, 2, 4, 13-21 and scrambled110 third). The round-2 verification lists the ranks as safe to cite with status 'empirical', and its exploratory-vs-confirmatory check asked only for the E2 wording. The finding misreads that row: 'descriptive' there refers only to the quadtree's first place (its outcome was seen before registration), not to every item. One part of the proposed fix cannot be applied: the fig:real-order caption does not contain 'neither necessary nor sufficient'.

### other-cov-2 (major) [verifier: refuted]
- Location: 05_other_regimes.tex:50-51 (Sec. 5.1, 'Lazy only with full bonds')
- Quote: This floor rises like $\isc^{2\Kc}$ at small $\isc$ (at $\chi=3$ as \eqref{eq:A-centred-floor} predicts, lower at $\chi=4$, $6$, ...)
- Problem: The text says the χ=3 floor follows eq:A-centred-floor, with 'lower' reserved for χ=4, 6. In fact the measured χ=3 level is about half the predicted mean. This contradicts App. D and the figure's own dotted line.
- Evidence: D_regimes.tex:132-134: 'the median of the six seeds is 0.48 times it'. final_review_claims_B C2: the mean is 0.57 times the formula at χ=3, recomputed from runs_sigma_chi_b4.csv. The fig:other-regimes caption draws eq:A-centred-floor at 1x for χ=3 ('dotted (χ=3 only): the small-σ mean'), so the line sits about 2x above the plotted medians.
- Fix: 'This floor rises like σ^{2K} at small σ. At χ=3 its level is 0.48 times the mean (eq:A-centred-floor) (median of six seeds). At χ=4, 6 the spare bond rank absorbs part of W_0 and the floor is lower still (App. D).'
- Verifier: eq:A-centred-floor is the mean over initializations, and the chi=3 data agree with it within the seed scatter. I ran a first-order Monte Carlo of the binary4 init (W0 = A R B^T with Gaussian A (9x3), R (3x3), B (9x3); P_perp = the complement of the 3-dim row and column spaces of W*). It gives E||P_perp W0||^2 / E||W0||^2 = 0.4435 against 1 - 45/81 = 0.4444, so the formula is exact for the mean. The distribution is broad: median/mean 0.73, 10-90% quantiles 0.24-2.06. The claims_B C2 sub-review's own Monte Carlo found median/formula 0.75. From runs_sigma_chi_b4.csv (centred, l_best/formula, constant to 3 digits over sigma = 0.01-0.4, so the sigma^(2K) law holds exactly), the six chi=3 seeds are 0.27, 0.38, 0.47, 0.49, 0.79 and 1.01: mean 0.57, median 0.48. If the formula is right, a six-seed mean <= 0.57 occurs with probability about 0.09 and a median <= 0.48 about 0.13, so nothing contradicts 'as eq:A-centred-floor predicts'. App. D says the same thing: the floor 'has mean (eq) ... the median of the six seeds is 0.48 times it ... [the expression] applies only at chi=3', which is the framing C2 asked for. So Sec. 5 does not contradict App. D. The dotted line in the figure is labelled as the mean and should sit about 1.4x above the medians; the remaining gap is six-seed scatter. At chi=4 and 6 the floor really is lower (medians 0.18 and 0.044; Monte Carlo 0.20 and 0.032). The proposed fix would also be wrong in two ways: it compares a median with the formula's mean, and it drops the 'saturates at the lazy residual' clause. An optional wording improvement only: '(six-seed mean 0.57 times it, within the seed scatter)'.

### other-cov-3 (major) [verifier: confirmed]
- Location: 01_introduction.tex:217-218 and 309-310 (outside the assigned sections; restates the Sec. 6.4 result)
- Quote: Random TTN teachers plateau at fixed-rank critical points, not at truncated singular value decompositions" / "new are that generic teachers plateau there and not at truncated SVDs
- Problem: Sec. 1 states the narrowed claim without its scope. The header of Sec. 6 lists this exact form as withdrawn (E15). Sec. 6.4 separates the rank-one plateau from SVD truncations for only 3 of 5 teachers.
- Evidence: 06:342-344: 'for three teachers that point is separated from every SVD-type truncation, while for the other two, and at the higher-rank plateaus, it lies within 0.0032 of one, so the plateaus do not discriminate'. The 06 header WITHDRAWN list includes '"plateaus are always critical points, not HSVD truncations" without the scope of E15'. data_teacher_round3_verification E15: separated for teachers 102, 104, 105 only; at higher ranks the two coincide.
- Fix: 01:217-218: 'Random TTN teachers plateau at fixed-rank critical points; at rank one these are separated from every SVD-type truncation for 3 of 5 teachers (at higher ranks the two coincide)'. 01:309-310: 'new are that rank-one plateaus of generic teachers sit at these critical points (separated from truncated SVDs for 3 of 5 random TTN teachers), the tree contrast and the bottom-up order'.
- Verifier: 01:217-218 ('Random TTN teachers plateau at fixed-rank critical points, not at truncated singular value decompositions', tagged \empirical) and 01:309-310 ('generic teachers plateau there and not at truncated SVDs') state, without its scope, the claim that the 06 header (line 22) lists as narrowed: '"plateaus are always critical points, not HSVD truncations" without the scope of E15'. data_teacher_round3_verification E15 and safe-to-cite row 9 limit it: the rank-one plateau is separated from every SVD-type truncation only for teachers 102, 104 and 105, and at higher ranks the two coincide. notes/data_teacher.md 4(b) gives the higher-rank detail: at (2,1,1), (2,2,2) and (2,3,1) HSVD and the critical point coincide to 1e-9, so 'not at truncated SVDs' is unsupported there, and in fact false. For teachers 101 and 103 the nearest truncation lies within 0.0014 and 0.0032, which the plateau does not resolve. Sec. 6.4 (06:285-287), tab:cov-summary (06:505-506) and claims_B M17 all keep this scope, so Sec. 1 is inconsistent with Sec. 6. The proposed fix is correct in substance; I only sharpen the wording ('random TTN teachers' in place of 'random teachers', and the higher-rank caveat).

### other-cov-4 (minor) [minor, unverified]
- Location: 05_other_regimes.tex:103-105 (Sec. 5.2, 'Crowded first escapes')
- Quote: $\min_i\sep i$ at $\isc=0.010$ is only $2.7$--$14$ (centred: $12$--$29$) and grows far more slowly than $\isc^{-(\Kc-2)}$
- Problem: The main text says the separation grows 'far more slowly' than σ^{-(K-2)} for all runs. App. D gives a per-seed range whose upper end is the predicted law itself.
- Evidence: D_regimes.tex:149-151: 'grows per seed like σ^{-0.1} to σ^{-1.0} between σ=0.04 and 0.01' (K-2 = 1). My per-seed fits on runs_sigma_chi_b4.csv (two-copy, χ=18, 24, 36, σ=0.01-0.04) give exponents 0.08-1.03, median about 0.5. χ=24 seed 5 rises from 3.5 to 14.4, i.e. about σ^{-1.03}.
- Fix: '... and grows per seed like σ^{-0.1} to σ^{-1.0} (median about σ^{-0.5}), in most seeds far more slowly than σ^{-(K-2)}, ...'

### other-cov-5 (minor) [minor, unverified]
- Location: 05_other_regimes.tex:249-251 (Sec. 5.5, 'Above the limit')
- Quote: a two-cycle, then chaos, whose time-averaged excess loss the network matches to $0.9\%$ while it learns the other components
- Problem: The sentence suggests the network matches the map's excess loss in the chaotic regime as well. The 0.9% agreement was measured only at 1.05-1.3 η_c, where the map is in its two-cycle and period-four range. The network does not follow the map's chaotic window.
- Evidence: D_optim.tex:20-26: 'two-cycle from η_c, period four from 1.245η_c, chaos from 1.32η_c'; 'within 0.9% at 1.05-1.3η_c'; 'The map's bounded chaotic window does not survive: the network diverges at 1.5-1.75η_c'. optim_samples.md 3.3 agrees (p5_excess_loss.csv at 1.05, 1.1, 1.2, 1.3 only).
- Fix: '... a two-cycle (period four from 1.245η_c), whose time-averaged excess loss the network matches to 0.9% at 1.05-1.3η_c while it learns the other components. The map's chaotic window (from 1.32η_c) does not survive: from 1.5η_c GD diverges or is thrown back ...'

### other-cov-6 (minor) [minor, unverified]
- Location: 06_covariance_teacher.tex:37-41 (Sec. 6 introduction)
- Quote: (overlapping CP components, a misaligned covariance, random TTN teachers, correlated sites) changes the law of the escape from the rank-one saddle ($\Ns=4$, $8$) ... correlated sites were not run on HT
- Problem: The scope '(N=4, 8)' is attached to all four non-orthogonalities and is qualified only for HT. Correlated sites were run at N=4 only.
- Evidence: 06 (Sec. 6.3, 'Correlated sites'): 'HT students and N>4 were not run with correlated inputs.' Sec. 7 item: 'correlated inputs were run neither on HT trees nor on MPS with N>4'. data_teacher_round3 safe-to-cite row 11: 'Cross-site correlated inputs (N = 4, d = 3)'.
- Fix: '... (N=4, 8; correlated sites only at N=4 and not on HT) ...'

### other-cov-7 (minor) [minor, unverified]
- Location: 06_covariance_teacher.tex:602-603 (Sec. 6.8 summary)
- Quote: the creep of the plateaus and the power-law rises of MPS edges 0--2, which it does not explain
- Problem: Edge 0 is listed among the power-law edges. Sec. 6.7 and the sources say edge 0 is mixed.
- Evidence: 06 (Sec. 6.7, MPS bullet): 'Edges 1 and 2 prefer a power law and edge 0 is mixed'. real_data.md 8.6(3): edges 1 (0/5) and 2 (1/5) prefer a power law, edge 0 is mixed (2/5). final_review_claims_B m89 made the same correction for Sec. 7.
- Fix: '... the power-law rises of MPS edges 1 and 2 (edge 0 mixed), which it does not explain.'

### other-cov-8 (minor) [minor, unverified]
- Location: 05_other_regimes.tex:230 (fig:optim-gd caption, panel a)
- Quote: $x=q/\tw1^{1/\Kc}$ with $q$ the per-core amplitude of component 1
- Problem: Here q is the running per-core amplitude. App. A.7 uses q_r for the fixed value s_r^{1/K} and a for the running amplitude, and Sec. 6.6 uses q for the star's pattern and component. This breaks the style spec's 'one symbol, one meaning'. final_review_claims_B m28 asked for this change and it was not applied.
- Evidence: A_derivations.tex:1243 'q_r = s_r^{1/K}'; A_derivations.tex:1297 'x = a/s_1^{1/K} (a the per-core amplitude)'; 06 Sec. 6.6 'q=0000', 'θ_q'.
- Fix: '$x=a/\tw1^{1/\Kc}$ with $a$ the per-core amplitude of component 1'.

### other-cov-9 (minor) [minor, unverified]
- Location: 07_open_questions.tex:290-291 (Sec. 7; restates the Sec. 5.5 / App. D finite-sample result)
- Quote: from about $200$--$250$ samples, $5$--$9$ times the dimension of the teacher manifold
- Problem: The multiple of the manifold dimension disagrees with App. D. final_review_claims_B m26 corrected it there but not in Sec. 7.
- Evidence: D_optim.tex: '200--250 samples or 4--9 times the teacher-manifold dimension with χ=4 students'. The manifold dimensions are 28 (N=8, d=2) and 48 (N=6, d=3). 200/48 = 4.2 and 250/28 = 8.9.
- Fix: '..., 4--9 times the dimension of the teacher manifold (χ=4 students), ...'

### other-cov-10 (minor) [minor, unverified]
- Location: 06_covariance_teacher.tex:483-484 (tab:cov-summary caption)
- Quote: \caption{Input covariance and teacher structure: what changes, and what predicts it. ``Escape'': the escape from the rank-one saddle.}
- Problem: The caption has no source pointer. The binding style spec (notes/report_style.md) says: 'Captions end "(source: App. B, Table N)"'. Every other table and figure in Sec. 5-6, including the summary table tab:optimizers, ends with one.
- Evidence: Compare tab:optimizers '(source: \cref{tab:exp-optimizers})' and tab:cov-slopes '(source: \cref{tab:exp-data-teacher})'.
- Fix: Append '(sources: \cref{tab:exp-data-teacher,tab:exp-real-data})'.

## appendix-math

### appendix-math-1 (major) [verifier: confirmed]
- Location: report/sections/A_derivations.tex:783-784, prop:A-trap (ii); echoed without orientation in 02_setup.tex:719 (tab:notation) and 04_saddle_to_saddle.tex:816
- Quote: $C_u=Q_{\nu\nu}+\norm{Y_{\rm nv}}^2-\norm{X_{\rm nv}}^2+O(\isc^3)$, with $Q_{\nu\nu}=\nu\T\Q e\nu$ on the route's new direction $\nu$
- Problem: The identity, tagged proven, uses the unoriented bond quantity Q_e. The report defines Q_e as child minus parent (eq:conserved). The identity only holds with the end-minus-partner orientation. As written, its sign is wrong at every route end that is the parent of its route edge, for example the MPS root end that the same proposition discusses ('0.5 at the root end').
- Evidence: Report definition: 02_setup.tex:366-367 defines Q_e = G_v G_v^T - G_p G_p^T, with v the child; lem:A-gd-identity uses the same convention. Source: notes/theory_trap.md section 1 says 'Q^{(u)}_e = +-Q_e is oriented end minus partner (+ if u is the child of e)', and T3 uses nu^T Q^{(u)}_e nu. Other appendix: D_s2s.tex:551-552 states it correctly ('nu^T Q^{(u)}_e nu ... the conserved bond imbalance (end minus partner)'). Contradiction inside App. A: A:113-114 says edges into the root are child-heavy (E Q_e > 0). Under the formula as written, a root end would then have C_u > 0 on average and survive. But A:1515-1517 says the 'child-heavy root bonds of fan-in initialization ... can kill' the route, and theory_trap.md section 10 item 2 says the MPS root end is unfavourable on average.
- Fix: Write: 'with $Q_{\nu\nu}=\nu\T Q^{(u)}_e\nu$, the conserved imbalance of $e$ oriented end minus partner ($Q^{(u)}_e=\Q e$ if $u$ is the child of $e$, $-\Q e$ if it is the parent)'. Say 'end minus partner' in the tab:notation row for $Q_{\nu\nu}$ and at 04:816 ('the imbalance of their bond, end minus partner').
- Verifier: The problem is real. (1) The report defines Q_e as child minus parent: 02_setup.tex:366-369, eq:conserved, Q_e = G_v G_v^T/eta_v - G_p G_p^T/eta_p with p = pa(v). The macro \Q{#1} = Q_{#1} (macros.tex:99) is used with this meaning everywhere else, for example A:113-115 and D_s2s:500. (2) prop:A-trap (ii) at A_derivations.tex:783-784 writes Q_{nu nu} = nu^T \Q e nu. Nothing in the report sets an orientation: grep finds 'end minus partner' only at D_s2s.tex:552, and 'orient' nowhere. (3) Bookkeeping shows the identity needs the oriented form. Split the nu-slices into the parallel part, X^perp and the non-visible entries. Then C_u = nu^T(G_u G_u^T - G_w G_w^T)nu + ||Y_nv||^2 - ||X_nv||^2 + (y_par^2 - x_par^2). The bracket is +Q_e only if u is the child of e. The source agrees: notes/theory_trap.md:55 ('Q^{(u)}_e = +-Q_e is oriented end minus partner (+ if u is the child of e)') and :106 (T3 uses Q^{(u)}_{nu nu}). So does the tested code: experiments/theory_checks/trap/trapcrit.py:231-233, 'sign = 1.0 if u == e else -1.0', with edges named by their child. trap_verification.md:48 verified exactly that identity. (4) The error is broader than the root end. In ttn/tree.py:178-181 MPS node k has parent k+1, so on every MPS route the higher-index end is the parent of its route edge (for example the root end of MPS4 {0,3}, and core 5 of MPS8 {1,5}). As written, the identity has the wrong sign at half of all MPS route ends. (5) The internal contradiction holds. A:113-114 says root edges are child-heavy (E Q_e > 0). Read with the unoriented formula, the MPS root end would on average be favourable. That contradicts A:1515-1517 (child-heavy root bonds 'can kill' the route), D_s2s:500 and theory_trap.md:746 ('MPS root end unfavourable on average'). D_s2s.tex:551-552 states the identity correctly, so App. A and App. D also disagree. tab:notation (02_setup.tex:719-720, 'imbalance of its bond') and 04:816 ('the imbalance of their bond') give no orientation either. In Sec. 4, 'imbalance' means Q_e (04:800, 805). No numbers in the report are affected: the code uses the right sign, and the beta-function probabilities are written in end and partner fan-ins, which carry no orientation. Severity major is appropriate: a statement tagged proven is false as written at parent ends, and two appendices disagree. The proposed fix is correct: Q^{(u)}_e = Q_e if u is the child of e and -Q_e if it is the parent, with 'end minus partner' added at tab:notation and 04:816. D_s2s.tex:470 (Q_{nu nu} in the earlier criterion) takes the corrected definition from prop:A-trap, so it needs no separate edit.

### appendix-math-2 (minor) [minor, unverified]
- Location: report/sections/A_derivations.tex:785-787, prop:A-trap (iii) (same usage in D_s2s.tex:555)
- Quote: (iii) An end with visible degree $j$ ($j=2$ if the partner carries sites, $3$ at an HT or binary leaf of a rank-one saddle) decays at a rate $\propto\isc^{2(j-1)}$, so it can kill its route only if $k\ge2j$
- Problem: Notation clash. In this appendix j is the learned component, but prop:A-trap also uses j for the visible degree. Example: the proposition sits under 'One learned component' while (iii)-(iv) use 'k >= 2j' and '2j < k <= 5'.
- Evidence: A:40-41 (Conventions): 'At a learned saddle the learned component is $j$'. tab:notation (02_setup.tex:630-631): '$j$, $q$: learned'. Its singular-value row even says 'the singular-value index is $n$, since $j$ is the learned component'. notes/report_style.md: 'one symbol, one meaning'. The visible degree has no entry in tab:notation.
- Fix: Rename the visible degree, for example $\deg_{\rm vis}(u)$ or $j_{\rm vis}$, in prop:A-trap and D_s2s.tex:555, and add it to tab:notation.

### appendix-math-3 (minor) [minor, unverified]
- Location: report/sections/A_derivations.tex:916-918, cor:A-order (iv); informal versions at 01_introduction.tex:169-172 and 04_saddle_to_saddle.tex:478-480
- Quote: (iv) In general, for a pattern teacher with all $\tw r>0$ and the index condition, the order is $\min(k^*,2\ell^*)$, $\ell^*$ the first level below $k^*/2$ with $\lambda_{\ell^*}>\tw j$, provided $\lambda_{\ell^*-1}<\tw j$.
- Problem: The statement is tagged proven, but the case where no level below k*/2 exceeds s_j is left implicit. There the formula gives k*, which needs lambda_{ceil(k*/2)-1} < s_j strictly. At equality the theorems are silent, except at k* = 3. Sections 1 and 4 also say the order is 'exactly k* unless a piggyback coupling ... exceeds the learned weight'. That asserts order k* at equality for k* >= 5. Sec. 4 flags the open boundary only for k* = 4.
- Evidence: thm:A-order-lower(a) needs strict lambda_{ceil(k*/2)-1} < s_j. notes/theory_saddle_order.md, Corollary 0: 'If no level below k*/2 exceeds s_q and lambda_max(S) < s_q, the order is k*'. Corollary 1: equality is open for k* = 4. The appendix itself (A:942) says that at the boundary lambda_max(Gamma) = s_j 'the theorems are silent' (group D of the note, k* = 4 and 5).
- Fix: In (iv) add: '; if no level below $k^*/2$ exceeds $\tw j$, the order is $k^*$ provided $\lambda_{\lceil k^*/2\rceil-1}<\tw j$ (at equality only $k^*=3$ is settled)'. In Secs. 1 and 4 write 'is exactly $k^*$ when the piggyback coupling ... stays below the learned weight' (or 'unless it reaches or exceeds').

## global

### global-01 (critical) [verifier: confirmed]
- Location: 01_introduction.tex:194-195 (finding 7); 07_open_questions.tex:225-226 (open question 4)
- Quote: it separates the runs that stall ($89\%$ of those it predicts dead, $1.8\%$ of the others)
- Problem: The criterion has three predicted classes, but the intro and Sec. 7 report only two. 1.8% is the stall rate of the predicted-alive runs. The predicted-pinned runs (934) stall at 15%, so 'the others' stall at about 5.9%. As written, the headline overstates how well the criterion separates the runs.
- Evidence: trap_verification.md, safe to cite: 'predicted dead 408/457 (89%) stall vs predicted alive 38/2,101 (1.8%)'. D_s2s.tex:576 gives 'pinned 141/934 (15%)'. Sec. 4.8 (04:830) states it correctly: '1.8% of the predicted-alive ones'. (141+38)/(934+2101) = 5.9%.
- Fix: Intro 01:194-195 and Sec. 7 07:225-226: '(89% of the runs it predicts dead stall, 15% of those it predicts pinned, 1.8% of those it predicts alive)'. The pending k=4 update (theory_trap.md v1.2; trap_k4_verification.md) changes 89% to 93% (408/437) and alive to 38/2,121, still 1.8%. That update's change list covers Sec. 4 l.830 and Sec. 7 l.225 but not the intro, so apply 93% at 01:194 as well when it lands.
- Verifier: Both quoted places are as the reviewer says. 01:194-195 reads '(89% of those it predicts dead, 1.8% of the others)' and 07:225-226 reads '(89% of the predicted-dead runs stall, 1.8% of the others)'. Neither place mentions the pinned class. D_s2s:576 gives three classes: dead 408/457, pinned 141/934 (15%), alive 38/2,101 (1.8%). The safe-to-cite line in trap_verification.md assigns the 1.8% to the predicted-alive runs only. I recounted from experiments/theory_checks/trap/results/locality_pilot_joined.csv.gz (locality, fan-in saddle + small, k>=4, q=0; outcome 'stall' or 'dead'). Dead 408/437, pinned 141/934, alive 38/2,121. The runs not predicted dead therefore stall at 179/3,055 = 5.9% (179/3,035 = 5.9% with the report's current v1 counts), not 1.8%. Sec. 4.8 (04:830-831) gets it right ('1.8% of the predicted-alive ones'). The intro is a headline, so this is a wrong number in a headline. The proposed fix is correct for the current numbers.

### global-02 (major) [verifier: confirmed]
- Location: 07_open_questions.tex:226; D_s2s.tex:627-628
- Quote: HT routes with $k=5$ never stall (0 of 144) because HT leaves need $k\ge6$
- Problem: The claim generalizes a count that covers small-initialization HT4 runs only. The report's own data show stalls on HT routes with k=5.
- Evidence: D_s2s.tex:477-478 defines the 144 as 'small-initialization HT4 runs with k=5'. Sec. 4.4 (04:383) and tab:locality-exponents (D_s2s:179) note that HT8 {1,2} (k=5) has seeds that stall. D_s2s:596-597: one seed of HT4 {1,2} stalls at all six σ from the saddle, and two of those runs are dead. Sec. 4.8 (04:875): 7 of 624 binary and HT runs with k=5 stall.
- Fix: Sec. 7: 'why HT routes with k=5 do not stall from small initialization (0 of 144 HT4 runs)'. D_s2s:627-628: 'HT routes with k=5 cannot die at leading order because HT leaves need k>=6: none of 144 small-initialization HT4 runs with k=5 stalls; from the saddle 7 of 192 do (one HT4 {1,2} seed at all six sigma, two of these runs dead at sigma>=0.19 and listed among the known failures; one HT8 {1,2} run).'
- Verifier: The source lines say 'HT routes with $k=5$ never stall' (07:226) and 'HT routes with $k=5$ never stall (0 of 144) because HT leaves need $k\ge6$' (D_s2s:627-628). The 144 comes from D_s2s:477-478, which counts small-initialization HT4 runs only. Sec. 4.4 (04:383) and tab:locality-exponents (D_s2s:179) say HT8 {1,2} (k=5) has seeds that stall. Sec. 4.8 (04:875) counts 7 of 624 binary and HT runs with k=5 that stall. The joined trap table shows all 7 are HT runs started from the saddle: HT4 {1,2} seed 1 at all six sigma (0.03-0.3), and the stall diagnosis finds the runs at 0.19 and 0.30 dead (D_s2s:596-597 lists them as known failures, predicted alive); plus HT8 {1,2} seed 1 at sigma=0.075. So HT routes with k=5 stall in 7 of 192 saddle-start runs. The unscoped 'never' is false. The proposed fix is mostly right, but 'slow escapes' is only the note's attribution for the HT4 runs. The HT8 stall was never diagnosed.

### global-03 (major) [verifier: confirmed]
- Location: 04_saddle_to_saddle.tex:762 (also D_s2s.tex:387-388)
- Quote: With four components most runs stop before the last one (MPS4 289 of 400; binary4 none): the stalls of \cref{sec:s2s-failure}
- Problem: This restates a withdrawn claim. The section's own WITHDRAWN list (04:614) contains 'R = 4: gradient flow mostly stalls before the last component'. The verification found that MPS6 often learns only 2 of 4, so 'before the last one' is wrong there. The multi2 runs were stopped by a criterion, so 'stop' does not show that they would never learn the last component.
- Evidence: order_round2_verification.md C16/E10: 'most runs do not learn all four components' (MPS4 289/400, MPS6 396/399, binary8 389/393; MPS6 learns only 2 of 4 in 7/11 no-stop runs). order.md 7.6: "'before the last component' was wrong for MPS6".
- Fix: 04:762: 'With four components most runs do not learn all four (MPS4 289 of 400, MPS6 396 of 399, binary8 389 of 393; binary4 none; seen directly in the runs without a stall stop, none of 22 complete by t=10^9, and inferred for the others); MPS6 often learns only two: the stalls of sec:s2s-failure, here also at K=4.' D_s2s:387-388: 'MPS6 does not learn all four in 396 of 399 runs (only two at the stop in 178 of them) and binary8 in 389 of 393, and none of 22 runs without any stall stop learns all four by t=10^9 (MPS6 learns two of four in 7 of 11).'
- Verifier: Sec. 4's WITHDRAWN comment (04:614) bans '"R = 4: gradient flow mostly stalls before the last component"', and the style spec says withdrawn claims are not restated anywhere. 04:762 nonetheless reads 'most runs stop before the last one ... the stalls of sec:s2s-failure', which is the same claim. The main text names only MPS4 (288 of its 289 incomplete runs have 3 of 4 at the stop) and binary4, so the numbers there are not wrong, but the wording is the withdrawn one. The factual error is at D_s2s:387: 'MPS6 stops before the fourth in 396 of 399 runs'. In multi2_summary.csv, MPS6 has learned_at_stop 2:88/3:111 and 2:90/3:107, so 178 of its 396 incomplete runs had only two components. The no-stop runs (multi_nostop_summary.csv) have 2 of 4 in 7 of 11. order.md 7.6 says '"before the last component" was wrong for MPS6'. The safe-to-cite wording (C16) is 'most runs do not learn all four components'. The reviewer's side point that 'stop' implies 'never learn' is weak, because the text says only 'stop'. However, E10 does require a qualifier on 'not learned by t=1e9', and the proposed fix leaves it out.

### global-04 (major) [verifier: confirmed]
- Location: 01_introduction.tex:217-218 (finding 10); 01_introduction.tex:309-310 (Sec. 1.3)
- Quote: Random TTN teachers plateau at fixed-rank critical points, not at truncated singular value decompositions
- Problem: This restates a withdrawn claim without the scope that E15 requires. The Sec. 6 WITHDRAWN list (06:22-23) forbids "plateaus are always critical points, not HSVD truncations" without E15's scope. Sec. 1.3 repeats it: 'new are that generic teachers plateau there and not at truncated SVDs'.
- Evidence: data_teacher_round3_verification.md E15: only rank-one plateaus are distinguishable, for 3 of 5 teachers; at higher ranks the two coincide. Sec. 6.4 (06:285-287) states this correctly, and Table 8 has '3 of 5 teachers separated'.
- Fix: Intro: 'for random MPS4 teachers the rank-one plateau sits at the bond-rank-one critical point, distinguishable from every SVD-type truncation for 3 of 5 teachers (the two coincide for the others and at higher ranks)'. Sec. 1.3: 'new are that the rank-one plateaus of random MPS4 teachers sit at that critical point, separated from SVD truncations for 3 of 5 teachers'.
- Verifier: 01:217-218 says 'Random TTN teachers plateau at fixed-rank critical points, not at truncated singular value decompositions'. 01:309-310 says 'new are that generic teachers plateau there and not at truncated SVDs'. Neither has a scope. Erratum E15 in data_teacher_round3_verification.md: rank-one plateaus converge to the bond-rank-1 critical point, but they can be told apart from every SVD-type truncation only for 3 of 5 random MPS4 teachers (102, 104, 105), and at higher ranks the two coincide. The withdrawn list (06:22) forbids 'plateaus are always critical points, not HSVD truncations' without E15's scope. Sec. 6.4 (06:285-287: 'for the other two, and at the higher-rank plateaus, it lies within 0.0032 of one, so the plateaus do not discriminate') and Table 8 (06:505-506, '3 of 5 teachers separated') are correctly scoped. So the intro contradicts Sec. 6. The fix is right but should also keep the MPS4 scope: all five teachers were random MPS4 teachers on an MPS4 student.

### global-05 (major) [verifier: confirmed]
- Location: 04_saddle_to_saddle.tex:516-518 and 597-602; D_s2s.tex:261-262, 289-303
- Quote: the condition ... fails at 30, where the order drops from $7$ to $6$ \proven{} (the theorems applied by exact computation)
- Problem: The 30-of-587 order-6 saddles, the HT4 escape signature x=0.2998 and the 19-of-88 mismatch attribution come from section 9 of the hopper note. App. D itself says that section 'was added after the audit and is not covered by it'. The main text states these results with \proven / \newresult and no caveat.
- Evidence: saddle_hopper_verification.md, section 8 (safe to cite) does not include them. final_review_claims_B.md M22: 'once its section 9 is verified, add the 30 order-6 saddles'. No later verification exists in notes/reviews/.
- Fix: Preferred, as M22 requires: have section 9 of the hopper note verified before citing it. Until then, keep \proven only for the HT4 D={0,1,2,3} wrong-order saddle (verified). In Sec. 4.5 and 4.6, present the census of 30 out of 587 saddles, the x=0.2998 signature and the attribution of 19 of 88 mismatches as computations on committed data, with no \proven or \newresult on them. Add a pointer to \cref{app:supp-s2s-hopper}, where their audit status is already stated.
- Verifier: Every factual point holds. These results come from section 9 of notes/saddle_hopper.md, added after the audit: the 30 order-6 saddles out of 587 (04:516-518, 04:597-599, tagged \proven), x=0.2998 (\empirical \newresult) and 19 of the 88 mismatches (D_s2s:303). D_s2s:261-262 says section 9 'was added after the audit and is not covered by it'. The safe-to-cite list in saddle_hopper_verification.md section 8 predates it. final_review_claims_B.md M22 says to add the 30 order-6 saddles 'once its section 9 is verified'. No later review covers it (grep of notes/reviews/ for 587 or 0.2998 finds only M22). So that condition in M22 did not stick. Two mitigating points. First, the numbers reproduce from committed CSVs: gp_saddle_order.csv has exactly 30 level-3 drops among 587 rows, and gp_escape_direction.csv gives x=0.29976 for HT4 close seed 4 at sigma 0.07. This is a problem of verification status, not a wrong number, so major at most. Second, the HT4 D={0,1,2,3} wrong-order case (order 6 < 7) is itself verified (saddle_order_verification_proof.md, row 7.5 and S8). Under the style spec, review history goes in App. C/D, not the main text, so the main-text fix should be a pointer, not an audit remark.

### global-06 (minor) [minor, unverified]
- Location: A_derivations.tex:1277, 1307, 1386
- Quote: divergence at $1.5$--$1.65\lr_c$ ($1.5$--$1.75\lr_c$ in \cref{sec:other-optim})
- Problem: These \cref targets became stale when material moved to App. D. The cited numbers are not in Sec. 5.4: '1.5–1.75 η_c', '0.58–1.00 times 2/η after a catapult' and '20 of 36 runs ... at 0.25–0.30 η*_ms'.
- Evidence: All three numbers appear only in D_optim.tex (lines 25, 41, 84). Sec. 5.4 gives only 'from 1.5η_c on' and 'one in six runs ... at 0.15η*_ms'.
- Fix: Point these three references to \cref{app:D-optim}.

### global-07 (minor) [minor, unverified]
- Location: 04_saddle_to_saddle.tex:874; A_derivations.tex:787
- Quote: 398 of 5,682 two-component runs stall, all with $k\ge4$ (none of 5,112 with $k\le3$)
- Problem: Read in place, the parenthetical makes 5,112 a subset of the 5,682 twocomp runs. That would leave only 570 runs with k≥4, but the next sentence already counts 816 runs at k=5. The 5,112 also counts the robustness sweep. App. A gives a third number for the same claim, '0 stalls in 5,744 runs'.
- Evidence: D_s2s:420-421: the k≥4 twocomp runs number 2,122 (768+816+360+118+60), so twocomp has 3,560 runs with k≤3. locality.md: '5,112 runs with k<=3 in the two-component and robustness sweeps'. trap_verification: 5,744 = locality + pilot.
- Fix: Name each population: '(none of the 3,560 with k≤3; none of 5,112 across the two-component and robustness sweeps)'. Keep 5,744 in App. A, labelled 'locality and pilot'.

### global-08 (minor) [minor, unverified]
- Location: 02_setup.tex:710 vs 755; 04:352-353; 06:111-112; A:703, A:1057
- Quote: head start; $c_r$ & factor $1-\tw2/\tw1$ ... ($s_2(M_e)/\bar s$: the head start)
- Problem: 'Head start' names different quantities. Sec. 4.4, App. A:703 and Table 9 use the factor 1−s_r/s_j. Sec. 6.2 and App. A.5 use the ratio s_2(M_e)/s̄. cor:A-second-escape uses the growth factor σ^{-s_2/s̄}. The notation table lists both meanings.
- Evidence: tab:notation rows at 02:710 and 02:755; 04:352 'The factor $1-\tw2/\tw1$ is a \emph{head start}'; 06:111 'where $s_2(M_e)/\bar s$ is the head start'.
- Fix: Settle one meaning, e.g. head start = s_2/s_1 (resp. s_2(M_e)/s̄), and call 1−s_2/s_1 'the head-start factor'. Update tab:notation, Sec. 4.4, Table 9 and A.3.

### global-09 (minor) [minor, unverified]
- Location: 07_open_questions.tex:290-291 vs D_optim.tex:125-126
- Quote: from about $200$--$250$ samples, $5$--$9$ times the dimension of the teacher manifold
- Problem: The same threshold is quoted as '5–9 times' in Sec. 7 and '4–9 times' in App. D.
- Evidence: Manifold dimensions are 28 (binary8) and 48 (binary6); 200/48 = 4.2. D_optim says '4--9 times'.
- Fix: Use one range ('4–9 times') in both places.

### global-10 (minor) [minor, unverified]
- Location: 04_saddle_to_saddle.tex:846
- Quote: seed-dependent ($0.26$ to $1.06\,\isc^2$)
- Problem: The upper end of the decay-rate range differs from the verified value.
- Evidence: s2s_core_round2_verification.md safe-to-cite #12: 'Levels are seed-dependent (0.26-1.04)'. The per-core values in dormant_decay.csv carry a ±4% window bias (s2s_core.md: asymptotic 1.00-1.04 for seed 7).
- Fix: Write '0.26 to 1.04 σ²'.

### global-11 (minor) [minor, unverified]
- Location: A_derivations.tex:1116; B_experimental_details.tex:34, 38
- Quote: Let $\Sig=\Sigi1\otimes\dots\otimes\Sigi\Ns$ ... f_\params\big(\Sigi{1}\fac{r}{1},\dots,\Sigi{\Ns}\fac{r}{\Ns}\big)
- Problem: These appendix formulas number sites from 1 (Σ_1…Σ_N, u_r^{(1)}…u_r^{(N)}, z_1…z_N). The report convention numbers sites from 0.
- Evidence: main.tex Conventions and tab:notation: 'Sites are numbered from 0'. Sec. 2 writes Σ_0⊗…⊗Σ_{N−1}.
- Fix: Index these sites 0,…,N−1.

### global-12 (minor) [minor, unverified]
- Location: 01_introduction.tex:89-143 (tab:intro-summary)
- Problem: Table 1 is never referenced in the text.
- Evidence: The cross-reference scan over all .tex files finds no reference to tab:intro-summary. Every other table and figure is referenced.
- Fix: Add a reference, e.g. after 'The answer in six points': '\cref{tab:intro-summary} gives the findings by research question'.

### global-13 (minor) [minor, unverified]
- Location: D_optim.tex:199-201 vs 05_other_regimes.tex:471-478
- Quote: is given once, in \cref{sec:summary-parameters} (``Landscape and clock'')
- Problem: Sec. 5.5 still has a full 'Interpretation' paragraph with the same two-property reading of tab:optimizers. App. D's 'given once' is therefore false, and the statement now lives in two places, against the style spec.
- Evidence: 05:471-478 ('Two properties of an update decide what survives...') and 07:151-163 make the same argument. report_todo item 16.
- Fix: Cut Sec. 5.5 'Interpretation' to a one-line pointer to Sec. 7.1, or drop 'given once' from D_optim.

### global-14 (minor) [minor, unverified]
- Location: B_experimental_details.tex:722-724
- Quote: the $\Kc$ sweeps, the maps of the binary tree with $\Ns=8$ and the large-$\chi$ tables keep $\dTK^{\rm probe}$
- Problem: This contradicts the rest of the report: the large-χ exponents use the tensor kernel. Only the large-χ regime labels use the probe kernel.
- Evidence: B.1 (B:143-145): 'the large-χ sweep, whose exponents use the tensor kernel'. D_regimes:11-13 says the same. The tab:large-chi caption reads '$\dTK$: tensor kernel', and Sec. 5.3 says 'tensor kernel'.
- Fix: Write 'the large-χ labels keep δΘ^probe (its exponents use the tensor kernel)'.

### global-15 (minor) [minor, unverified]
- Location: 05_other_regimes.tex:280-283
- Quote: (29 trapped, one diverged) ... separates escaped from trapped settings
- Problem: Sec. 5.4 uses 'trapped' as a run class. The spec drops 'trapped' as a label, and App. B.7 calls this class 'stalled' ('called trapped in the code').
- Evidence: report_style.md: "Drop 'trapped' as a separate label; say 'dead route (trap)'". B:1053-1054.
- Fix: Write 'stalled (the route dies)' or 'dead route', e.g. '(29 with a dead route, one diverged)'.

### global-16 (minor) [minor, unverified]
- Location: D_s2s.tex:470; 02_setup.tex:741-742
- Quote: an end is unfavourable if $c=Q_{\nu\nu}+\norm{y_{\rm route}}^2<0$
- Problem: The bare symbol c is reused, although the spec reserves c for the input correlation. In the same vein, tab:notation calls x=c_col(η/B)T_GF 'the only use of x other than an input'. But x is also the scalar-map variable (Fig. 13 caption, A.7), the Perron vector (A.4 proof) and the hopper escape signature (D_s2s:297).
- Evidence: report_style.md: 'Input correlation: c'. tab:notation row c: 'cross-site input correlation'.
- Fix: Rename the R12 quantity (e.g. C'_u) and drop the 'only use of x' remark, or list the other uses of x.

### global-17 (minor) [minor, unverified]
- Location: 06_covariance_teacher.tex:603
- Quote: the power-law rises of MPS edges 0--2, which it does not explain
- Problem: The summary says edges 0–2 rise as power laws, but Sec. 6.7 says edge 0 is mixed.
- Evidence: 06:460-461: 'Edges 1 and 2 prefer a power law and edge 0 is mixed'.
- Fix: 'the power-law rises of MPS edges 1 and 2 (edge 0 mixed)'.

### global-18 (minor) [minor, unverified]
- Location: D_s2s.tex:645-651
- Quote: The same review reproduced $\mathcal{T}(Z)$ for the 160 binary4 and MPS4 first-escape jobs
- Problem: Wrong attribution in the provenance paragraph. 'The same review' refers to the order recomputation, but the independent T(Z) integration comes from the s2s-core round-2 verification.
- Evidence: s2s_core_round2_verification.md:23-27, 58 (independent sphere-flow T on all 160 jobs). App. B.2 (B:323-325) attributes it to the s2s-core verification.
- Fix: Write 'The s2s-core verification reproduced T(Z)...'.

### global-19 (minor) [minor, unverified]
- Location: 05_other_regimes.tex:424 (caption of fig:optimizers-side)
- Quote: partial-fit levels $\Lpart1/\Lpart0=0.36$ and $\Lpart2/\Lpart0=0.07$
- Problem: The caption uses the L_k notation (k = number of components learned), which the spec dropped in favour of L_S. Under the L_S convention, L_2 would mean only component 2 learned, which is 5/7, not 0.07.
- Evidence: report_style.md: 'Drop the L_k notation (eq. 10); use L_S'. tab:notation defines only L_S and L_0 = L(0).
- Fix: Write 'L_{\{1\}}/L(0)=0.36 and L_{\{1,2\}}/L(0)=0.07'.

### global-20 (minor) [minor, unverified]
- Location: 01_introduction.tex:278-279 (Sec. 1.3)
- Quote: the matching lower bound on the saddle order with one learned component, and its failure from $k^*=5$ on
- Problem: 'Its failure from k*=5 on' says the bound fails, but it only can fail. This is the wording that claims review A M3 corrected in Sec. 4.
- Evidence: final_review_claims_A.md M3. The k*=5 control teacher meets the condition and has order 5 (A.4).
- Fix: '... and that it can fail from k*=5 on'.

### global-21 (minor) [minor, unverified]
- Location: 04_saddle_to_saddle.tex:705-706 vs 683
- Quote: $P(\text{larger first})=0.79$, $0.69$, $0.63$ at $w=0.5$ on MPS with $\Kc=4$, $6$, $8$
- Problem: The setting is not stated. A few lines earlier, MPS4 at χ=4 gives P(larger first)=0.815 at w=0.5, so the two values for MPS4 look contradictory.
- Evidence: order/results/main_summary.csv: MPS4 d=χ=2: 0.7925; MPS4 d=2, χ=4: 0.815; MPS6 and MPS8 at d=χ=2: 0.685 and 0.63.
- Fix: Add '(d=χ=2)'.

### global-22 (minor) [minor, unverified]
- Location: B_experimental_details.tex:1353
- Quote: the 28 rank-one states that are not near rank one
- Problem: Self-contradictory wording.
- Evidence: B:1286-1287: 'States farther than 0.05 (relative) from a rank-one product are dropped (28 of 124)'.
- Fix: 'the 28 states that are not near rank one'.
