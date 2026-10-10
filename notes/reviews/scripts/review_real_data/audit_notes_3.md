# real-data claims audit: working notes, part 3

## first_route gap attribution (scaling.first_route)
- Rule: gap = (first drop after drop 0 whose widened window END is after the route's rank crossing) - drop 0.
  No check that the crossing lies inside that drop's window. If the route rose before drop 1 (merged into
  drop 0 at large sigma), the gap is assigned to a later event.
- Evidence of merged first routes: quadtree seed 3 sigma 0.3 / 0.5 plateau-midpoint states (between drops
  0 and 1) have L = 0.220 / 0.226 < 0.27 and rank1_rel_err 0.81 / 0.94 (routes_raw, hessians_raw), i.e. the
  (3,4) route had already risen; both sigma are inside seed 3's fitted range (0.05-0.5, 7 points).
  scrambled2 seeds 0, 4 at sigma 0.5 (rank1_rel_err 0.995, 0.69) are inside their fits too.
- Clean series (events_table: same signature on drop 1 at every sigma; 11 series): free exponents
  1.06-1.36, mean 1.216 +- 0.031; A/sigma + B with relative weights fits each (rms 0.003-0.06) with
  B < 0 (-5 ... -41). Repo headline 1.01 +- 0.14 (quadtree) is lower than the clean values for the same
  seeds (1.06, 1.36, 1.06, 1.24).

## absolute clock (claim 5)
- t_a sigma^13 per quadtree seed: 110, 166.8, 18.8, 206, 37.8 (constant to 4 digits across sigma).
- seed 0 at sigma 0.5: MPS t_a sigma^14 = 83.6 (other reviewer's calib run) vs quadtree 110 sigma^13:
  t1(MPS)/t1(quadtree) = 0.76/sigma = 1.5 (0.5), 7.6 (0.1), 15 (0.05). On the absolute clock the quadtree
  reaches every loss level first; "MPS fastest" holds only on tau = t - t_a.
