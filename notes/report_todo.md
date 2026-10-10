# Report: items for the final consistency pass (lead's list)

Collected from agents' hand-backs; each item names where it came from. Tick off in the final pass.

1. Sec. 4.6: "trap decay rate constant per seed to a few percent" holds for the separated teacher only; in Fig. 11
   the close-teacher rates change by factors 1.3 and 3 (figure QA).
2. Sec. 5.5: "no setting shortened the first plateau" is slightly too strong: 9 of 23 cells of Fig. 14a are negative;
   at sigma = 0.1, B = 128 the change is -0.32% +- 0.13% (figure QA). Say "by more than ~0.3%".
3. App. A prop:A-matrix-like(b) is tagged conjecture; the order verification (C8) calls it derived (heuristic) and
   Sec. 4.5 follows C8. Reconcile with the rule: heuristic -> \conjecture, or \empirical "heuristic mechanism" when
   the data support it.
4. Sec. 4.5 quotes three numbers that exist only in the order verification (small-sigma fits, McNemar test, stall-rule
   margin): point to notes/reviews/scripts/verify_order/ as their provenance (App. B/C).
5. notes/diagnostics.md: the "Open: ... fix pending" line on the probe vs tensor kernel is stale (notes cleanup).
6. Sec. 2.7 says delta_K uses the tensor kernel where feasible; Sec. 3's phase-map labels still use the probe kernel
   until regime-map round 2 relabels them (requested). Make text, captions and tables agree after round 2 lands.
7. Sec. 3 large-K placeholder: fill from regime-map round 2.
8. Sec. 1, 4.4, App. A.4 placeholders for the dynamics of the saddle-order theorem: fill from saddle-dyn.
9. Sec. 4.6: integrate the trap criterion / probability results (trap-theory agent) if they hold up.
10. Check that no withdrawn claim is restated anywhere (abstract, intro, summary tables, captions).
11. Sec. 2 and tab:notation: event widths w_r -> omega_r and T(Z) -> \mathcal{T}(Z) (spec additions after the Sec. 2
    edit); add the other editors' renamed symbols to tab:notation; settle remaining clashes.
12. Sec. 2 cuts X5, X6, X7, X9 (readability cut list) were not done because their labels are cited from 04, A, B:
    move them once the other files are final (~1 page).
13. tab:notation additions: \mathcal{T}(Z); Adam's eps vs the perturbation \varepsilon; omega_r; Sec. 3/5 symbols
    c_col, x_W, b_0, w, xi, lambda (root scale), sigma_c, sigma_lazy, eta_c, eta*_ms, lambda_GF, c_e, sigma_x.
14. Sec. 2.7 still says the Sec. 3 labels use the probe kernel; the binary4 maps now use the tensor kernel.
15. App. C: a row for the teaser figure (experiments/report_figures/make_teaser.py + report/tikz/teaser.tex);
    provenance lines for the literature search and the hopper audit (cited from 1.3 and 7.2).
16. D_optim: the "Interpretation" (invariance) paragraph now lives in Sec. 7.1; reduce the D_optim copy to a pointer.
17. Sec. 7: remove the label guard once Sec. 4.7 is deleted (after the ed-s2s merge).
18. Sec. 1: the SGD statement must match Sec. 5.5 ("no setting shortened the first plateau beyond noise: 11 of 23
    cell means negative, >= -1.0%, all within 2.5 s.e.").
19. Sec. 1 finding 7 (dead routes) and Sec. 4 failure modes: update from notes/theory_trap.md after its verification.
20. Abstract is ~235 words; trim to <= 200.
21. Sec. 7 table: the input-covariance references print "6.1, 6.3, 6.3"; drop the duplicate \ref.
22. tab:notation additions from Sec. 6: cos phi, lambda_e, M_e, s-bar, h_e, lambda_q/lambda_b, mu_uw, \mathcal{S}, Y, C,
    t_a, tau, t_0, beta; from App. A: see the header comment of A_derivations.tex; from Sec. 4/App. D: zeta, x.
23. Sec. 4 still uses gamma for the growth exponent and (1-kappa_e)/gamma_e in a table; D_regimes uses kappa_K (A writes
    a_0c/sigma_c). Align with App. A (beta(a_r) growth exponent, lambda_e, head start s_2(M_e)/s-bar).
24. Captions: settle one source form, "(source: Table N)".
25. Fig. 3 (s2s_core/plot_report.py): check the panel titles use 0-based sites (Sec. 6 editor's note).
26. references.bib: add zhang2021embedding and simsek2021geometry (lem:A-unused TODO-CITE); verify on arXiv.
27. A.8 large-K text vs Sec. 3: check they agree (both from regime-map round 2 + its verification).
28. Trap: Sec. 4 failure-modes paragraph and Sec. 1 finding 7 from notes/reviews/trap_verification.md (safe to cite);
    apply its errata to notes/theory_trap.md too.
29. saddle-dyn: fill the placeholders in Sec. 1, Sec. 4 (k* subsection) and App. A.4 when it reports and is verified.
30. Trap k = 4 fix (notes/theory_trap.md v1.2, Prop. 3) VERIFIED (notes/reviews/trap_k4_verification.md: use ITS full list of safe report changes and its errata: sigma-independence only as sigma -> 0 away from the boundary; stall_diagnosis CSV unfiltered).
    If verified: Sec. 4 89% -> 93%, drop "k = 4 with spare inner channels" from the known failures; D_s2s 408/457 ->
    408/437, alive 38/2,101 -> 38/2,121, k = 4 203/232 -> 203/212 and 5/796 -> 5/816, stall diagnosis 30/6/2 -> 30/4/2
    and 2/56/0 -> 2/58/0, "56 false alarms" -> 36, delete "MPS4 at chi = 4: 20 of 46", "30 of 38 ... 7 of 126" ->
    "30 of 36 ... 7 of 128"; prop:A-trap (iv) wording (one new direction on each END edge; spare inner directions enter
    only through <beta1,beta2> and |beta1|^2+|beta2|^2); MC MPS4 {0,3} chi = 4 P(dead) 0.09/0.08 -> 0.05/0.05.
31. notes/theory_saddles.md lines ~46, 815, 1071-72, 1084-85 still carry the old k = 4 restriction or 408/457.
