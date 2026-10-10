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
