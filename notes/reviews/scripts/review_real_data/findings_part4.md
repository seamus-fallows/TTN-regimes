# part 4: robustness reruns and provenance

## Fashion sneaker vs boot, quadtree s0.14 seed0, t2 = 120 (agent code)
drops 0.495 -> 0.333 (tau 8.6) -> 0.130 (tau 39.1); no edge reaches s2/s1 = 0.05 by tau 120 (rank-1 double drop, as claimed);
tau(L<=0.15) 45.2 (ref median 45.2). plateau_hessians: proj lambda_min -0.1246 at tau 21.75, L 0.3330 (ref -0.1266, 22.49, 0.3328).
## chi = 8, quadtree s0.14 seed0, t2 = 220
first rising edges [3,4] at 82.05/82.75 (same pair as chi = 4, as claimed); gap drop1->drop2 75.4 (chi = 4: 100.2).
## Provenance gaps
- chi8 first-route exponents 1.17+-0.06 / 0.97+-0.11 / 0.79+-0.28 and Fashion sigma^-0.92..-0.97: not in chi8_scaling.json /
  fashion_scaling.json (scaling.py needs >= 4 sigma; these sweeps have 3, all fits are NaN) and no committed script makes them.
- notes/REPORT scrambled2 first-route 0.91+-0.23 (2 seeds) vs committed scaling.json/scaling.txt 0.74+-0.21 (3 seeds).
- fashion_tables.md: binary "first rise = single edge" 0.80 (12/15 runs), not mentioned in notes section 7.
## Other code notes
- tables.sibling_pair() checks only that both edges are bottom edges (no common-parent check); first_two_siblings() does check.
- tables.py labels L(1e4) also for img8 (horizon 3000).
- phase 1 stops at the first DP5 step below the 1% floor: tau origin overshoots the crossing (~0.9 units, MPS s0.1); gaps/t1 unaffected.
- route-amplitude lambda is gauge dependent (proportional to 1/(|g_u||g_w|), g from core SVD) but bottom cores are balanced
  (top core singular values 1.09-1.32), so shares/argmax are unaffected (independent unit-norm dense-tensor version agrees).
