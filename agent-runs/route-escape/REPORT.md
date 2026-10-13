# route-escape: leaving a known multi-pattern plateau (route theory, brief section 4.2)

Status: done - Branch: agent/route-escape - Updated: 2026-10-13 21:52 UTC

**Bug in `ttn/routes.py`: none found.** All library functions I used (RouteTarget, AlignedManifold with
partial_target / velocity / jacobian / mask / align / route_entry_norms, one_component_order,
effective_depth) behaved as documented; the full network reproduces `AlignedManifold`'s reduced flow
to 1e-6 in 6597 runs. One usage note: `RouteTarget.greedy_next` breaks ties with the weight only (unit
active entries), while the paper's tie-break is s_r A_r(J); I computed A_r(J) at the balanced theta_J
(`common.Cell.greedy`). Both give the same answer in all my cells.

## Summary

The task: start a tree tensor network exactly at a plateau theta_J (a set J of learned patterns),
nudge it by eps, and check three predictions of the paper: the saddle order (how steeply the loss can
start to fall), the escape-time law with its prefactor, and which missing pattern is learned first.
I used 11 cells (route depths 2-5; mps, ht and binary trees; two missing patterns of different depth;
a tie; Corollary 8.6; a predicted shortcut) plus 4 static-only Theorem 9.2 instances: 30 static order
measurements and 9317 gradient-flow runs. **Almost everything matches, to 3-6 digits**: inside the
aligned manifold M* the order is the route depth c*_J in every cell, the full network follows the
paper's reduced equations exactly, and the escape time follows eq. 6.16 including its prefactor;
Gaussian starts escape with the exponent of the full-space order; the greedy rule picks the first
pattern at small eps; the Theorem 9.2 formula holds on its family. **Failures and limits:** (1) where
a learned pattern has a "pair-star" spectral shortcut, the full-space order is 4 instead of c*_J = 5,
and generic (Gaussian) starts take that shortcut: they learn two patterns at once, forget a learned
one, and break the greedy rule; (2) when two missing routes share entries (a tie), the isolated-route
escape formula is off by up to a factor 2-3 (the reduced flow is still exact); (3) in a depth-5 cell
one of 16 Gaussian directions escapes with exponent 4 instead of 3, because a route entry is drained
to zero by off-target clusters (shown by two interventions); (4) at eps ~ 0.1 the greedy rule fails
in 6-38 % of runs.

## Results

Terms: *route depth* c_J(r) = the number of cores the missing pattern r cannot borrow from the learned
ones; c*_J = the smallest over missing r. *Saddle order* p: the loss can fall as -t^p at distance t
from theta_J. *M** = the aligned manifold (only the target's own local entries). *Shortcut* = a
full-space order below c*_J. Kinds of start: A = positive dormant entries inside M*; A_bal = all
dormant entries equal (the literal setting of eq. 6.16); B = random signs inside M*; C = Gaussian on
all parameters; Cp = C projected onto M* (diagnostic). eps = size of the perturbation per entry; time
is gradient-flow time. Every number below is in `experiments/route_escape/results/*.csv` (columns
documented in `experiments/route_escape/README.md`), produced by
`uv run python experiments/route_escape/analyze.py` from the raw JSON in `/root/outputs/route-escape/`;
detailed tables: `notes/route_escape.md` Part II.

**1. Static order** (`static_orders.csv`, `fig1_static_orders.png`):

| cell | tree | c*_J | order inside M* | order in the full space | verdict |
|---|---|---|---|---|---|
| EX87_J01 (Example 8.7) | mps7 | 2 | 2.000 | 2.000 | = c*_J (strict saddle, Hessian -0.478 = -s_r A_r) |
| TWO23 | mps7 | 2 | 2.000 | 2.000 | = c*_J (Hessian -0.191 = -s_r A_r) |
| S3, U3HT, TWO34, TIE3 | mps7, ht4, binary8, ht4 | 3 | 3.000 | 3.000 | = c*_J |
| S4, U4B8 | mps7, binary8 | 4 | 4.000 | 4.000 | = c*_J |
| EX87_J0, U5B8 | mps7, binary8 | 5 | 5.000 | 5.000 | = c*_J |
| **PS2** | mps6 | 5 | 5.000 | **4.000** | **shortcut** |
| T92a / T92c | mps6 | 3 / 5 | 3.000 / 5.000 | 3.000 / 5.000 | = c*_J |
| **T92b, T92d** | mps6 | 5 | 5.000 | **4.000** | **shortcut** |

Inside M* the order equals c*_J at all 15 cells (to 4e-4): Thm 8.4(3) confirmed. In the full space the
only shortcuts are the three cells where a learned parent has pair-star residuals; the minimiser there
is the construction of the paper's Example 4.5 at that parent (an off-target two-core piece plus
single-core rotations; the parent's amplitude drops). The shortcut cells have no negative Hessian
eigenvalue (order 4 with a flat Hessian). List of shortcut cells: PS2, T92b, T92d; no other cell.

**2. Theorem 9.2** (`thm92.csv`, `fig6_thm92.png`): the formula min over parents of min(k*, 2 l*)
equals the measured full-space order at **all 5 instances**: T92a 3 (both parents direct), T92b 4
(shortcut at one parent), T92c 5 (its control), T92d 4 (shortcut at the *weaker* parent, s = 0.5), PS2 4.
**The formula holds on this family.** It does not apply to the splicing cells (outside its assumption).

**3. Starts inside M* (A, A_bal, B, Cp)** (`decisions.csv`, `fig2_inside_Mstar.png`):
- Full network vs the reduced flow of Thm 7.7: escape times agree to at most 9.2e-7 relative (median
  3e-10) in 6597 runs; the output stays exactly on target.
- Eq. 6.16 with its prefactor (balanced starts, eps = 10^-4.5): measured t_esc / [eps^-(c-2) / ((c-2)
  s_r A_r(J))] = 1.000 (within 0.03 %) in all depth-3, 4 and 5 cells; e.g. EX87_J0 at eps = 1e-4:
  6.6667e11 measured, 6.6667e11 predicted. Depth 2: slope of t_esc against ln(1/eps) = 2.092 vs
  1/(s A) = 2.092 (EX87_J01), 5.231 vs 5.231 (TWO23).
- Random positive and signed starts: within 1 % of the isolated-route formula evaluated with the
  start's own dormant entries, in every fit-window run of 10 cells (a negative starting product just
  takes longer). **Fails in TIE3** (two missing patterns sharing 2 of their 3 dormant entries): ratio
  0.54-0.91 (positive starts), 0.51-3.2 (random signs).
- Exponents: all 441 directions at depth >= 3 within 0.005 of c*_J - 2 (16 directions per kind and
  cell, fit over eps = 10^-4.5 ... 10^-3). No trapped direction.

**4. Gaussian starts (C)** (`escape_fits.csv`, `fig3_exponents.png`), exponent per direction against
p - 2, p = the static full-space order:

| | complete directions | at prediction (within 0.3) | shifted | trapped | incomplete (floor-flagged run in window) |
|---|---|---|---|---|---|
| 9 cells with p >= 3 | 135 | 135 (all within 0.05; 131 within 0.01) | 0 | 0 | 9 (all U5B8) |
| of which PS2 (p = 4 < c*_J = 5) | 16 | 16 at 2.000 (route depth would give 3) | 0 | 0 | 0 |
| 2 depth-2 cells | 32 | log law: slope / prediction 0.996-1.005 | 0 | 0 | 0 |

Of the 9 incomplete U5B8 directions, fitting all their runs (not a registered fit) gives 2.996-3.034
for 8 and **4.004 for direction 9**, which is at 4 already on its unflagged points.

**Why direction 9 is shifted** (`fig7_shift_mechanism.png`, note section 12): in every Gaussian run at
depth >= 4 the route entries at the two ends of the routing region shrink during the plateau, on the
time scale eps^-2 (half-time x eps^2 constant over 1.5 decades of eps), shorter than the route's own
eps^-3 at depth 5. Usually the shrinking stops at a finite level and only the prefactor changes; in
direction 9 the leaf entry goes to zero and the route needs one more small factor. Causal tests: the
same directions projected onto M* (Cp) escape as eps^-3 (all 16, including 9); and zeroing only the 30
"partner" entries (of 355 parameters) that form off-target two-core pieces with the leaf entry removes
the drain and restores eps^-3 (t_esc = 2.56e8 / 8.10e9 / 2.56e11 instead of 3.0e10 / 3.0e12 / 3.05e14
at eps = 1e-3, 10^-3.5, 1e-4). This is the mechanism `notes/saddle_dyn.md` conjectured (its D7), now
observed and removed by intervention. Which directions get cut (I suspect the conserved bond Gram
balance of the edge decides it) is a conjecture, not tested.

**5. Which pattern is learned first** (`fig4_first_learned.png`): the greedy pattern (shortest route;
tie -> larger s_r A_r(J)) is first in **100 % of the fit-window runs of every cell and kind, except PS2
with Gaussian starts**. This includes TWO23 (depth 2 at weight 0.2 beats depth 3 at weight 0.9),
TWO34 (depth 3 at 0.3 beats a depth-4 splice at 0.9) and the tie TIE3 (s A = 0.78 beats 0.39).
**Failures:** at eps = 0.1 the other pattern wins in 12-38 % (TWO23), 6-19 % (TWO34) and 19-31 % (TIE3)
of the runs, from eps <= 10^-1.75 never. **PS2 C: greedy only 38 %**: the shortcut produces both
pair-star residuals together (their learning times differ by ~1e-9 t_esc), and the parent 000000 is
forgotten in 228 of 238 runs, which end at {222222, 110001, 110010}: one learned pattern swapped for
two, which no route rule describes.

**6. Corollary 8.6** (`fig5_cor86.png`): 0022111 seen from J = {0000000} (depth 5) and from
J' = {0000000, 1111111} (depth 2): order 5 -> 2; escape eps^-3 (6.7e11 at eps = 1e-4) -> log law (12.8
at eps = 1e-4). Holds.

**7. Effective depth** read off the amplitude growth = route depth in every cell (median c + 0.002 ...
c + 0.17; splices included, e.g. EX87_J01 2.006 where the pairwise cost is 4); exceptions: PS2 C (3.8,
the order-4 shortcut) and the shifted U5B8 direction. **Off-target mass at escape**: 0 inside M*,
<= 1e-7 for Gaussian starts, except PS2 C (0.0087).

**Run counts** (9317 escape runs, all kinds and eps): 9245 escaped, 13 floor-flagged (all U5B8 C,
eps <= 10^-3.75; excluded from fits), 57 below the escape threshold already at t = 0 (eps >= 10^-1.5,
so no escape time), **0 trapped by the registered rule, 0 failed, 2 unfinished** (PS2 C direction 1 at
eps = 0.1 and 0.056, effectively trapped: see Caveats). Nothing was cut from the grid.

## What was done

1. Read the brief, the paper, the saddle_dyn note and code; chose cells with `ttn.routes`.
2. Code in `experiments/route_escape/`: common.py (cells, cancellation-free model), driver.py (step-driven
   LSODA after sdyn.py), escape.py, static.py, sweep.py, analyze.py, tables.py, figures.py, diag_partner.py.
3. Pre-registration `notes/route_escape.md` Part I, pushed before any sweep (d487b50).
4. Sweep on 118 cores: static orders (first table pushed 08d22ad, complete de9cc03), escape runs; a
   solver stall was found and fixed twice (3486c4e, 2db1ca7) and the sweep restarted (7e31646).
5. Diagnostics: kind Cp for all cells, the partner intervention on U5B8 direction 9.
6. Analysis, tables, figures, note Part II, README, this report (final commits on the branch).

## Decisions

- **Cells.** Mostly mps7 d=3 with J = {0000000, 1111111} (a clean family where the route is a splice
  of length 2-5), plus binary8 and ht4 cells from a random search with `ttn.routes`. The unique shortest
  routes got a deeper, heavier competitor so that "unique" is actually tested. Bond dimension = the
  largest number of star blocks + 2 on every edge (AlignedManifold takes one chi).
- **Perturbations.** eps = size per entry (iid N(0, eps^2) for C), comparable with a small-init sigma.
  A/B perturb only the dormant M* entries; one extra balanced direction (A_bal) tests eq. 6.16 literally.
- **Prefactor test for unequal starts.** Eq. 6.16 is for balanced factors; for random starts I compare
  with the exact isolated-route integral using the start's own factors (eq. 6.14 plus the balance
  invariants of Prop. 6.2), which reduces to eq. 6.16 when the factors are equal.
- **Grid** 10^-1 ... 10^-4.5 (3.5 decades, not just 2), because the pilots were cheap; fit window
  [10^-4.5, 10^-3] as registered.
- **Solver stalls**: switched to BDF on restarts and on detected stalls rather than raising the wall
  cap (note section 7, deviations 1 and 1b); every run without an outcome was rerun, nothing else.
- **Added Cp and the partner intervention** after seeing the shifted direction (deviation 2); they are
  diagnostics, not registered decisions.
- **Kept the registered rules even where they misfire**: P5 reads INCONCLUSIVE for the depth-2 cells
  (the rule tests a log-log slope against 0, which a log law cannot meet: my mistake in the
  pre-registration; the log law itself is confirmed by the registered P4 test), and U5B8 C reads
  INCONCLUSIVE because of floor flags.

## Files left on the machine

| path | size | what | needed to reproduce |
|---|---|---|---|
| /root/outputs/route-escape/escape/ | 1.3 GB | one JSON per escape run (trajectory, checkpoints, route-entry norms); `*/unfinished_try1/` holds 22 runs that hit the wall cap before the stall fixes | yes, for analyze.py (or rerun sweep.py, ~1.5 h on 118 cores) |
| /root/outputs/route-escape/static/ | 0.2 MB | 30 static-order JSONs | yes (or rerun, ~15 min) |
| /root/outputs/route-escape/diag/ | 0.9 MB | partner-intervention runs | for fig7 |
| /root/outputs/route-escape/logs/ | 0.8 MB | sweep and diagnostic logs | no |
| /root/outputs/route-escape/pilot/, test/ | < 0.1 MB | pilot and driver-check runs | no |

## Caveats and next steps

- **Unfinished (effectively trapped):** PS2 C direction 1 at eps = 0.1 and 0.056 (the two largest
  eps, outside the fit window) hit the 1800 s cap at t ~ 1e19 without escaping: the large Gaussian
  kick (L - L_J = +0.85 and +0.18 at t = 0) relaxed back onto the plateau with both missing amplitudes
  dead. The registered trap rule did not fire because L - L_J sits at ~1e-18 (roundoff), above its
  1e-30 threshold. They are the only non-escaping runs; nothing in the fits depends on them.
- The escape-time law for generic starts holds for typical directions only at depth >= 5 (1 of 48
  Gaussian directions at the depth-5 cells shifted). Next: test the conserved Gram balance of the
  region-end edge as a predictor of cut directions, and run more directions at depth 5-6.
- U5B8 C is INCONCLUSIVE by the registered floor rule; the floor index is a pessimistic bound and the
  flagged runs lie on their lines, but they stay excluded.
- Ties with shared dormant entries need the coupled reduced flow, not the isolated-route law.
- All results are for saddle starts; whether small-init training reaches these theta_J is the job of
  `route-smallinit`.

## Log
- 19:20 Read BRIEF.md, notes/briefs/route_theory.md (1-3, 4.2), the paper (all sections), notes/saddle_dyn.md, sdyn.py, saddle_order.py. Branch check OK: the four route files exist, `uv run pytest tests/test_routes.py` passes (13 tests). Machine: 128 cores, cgroup quota 122.9 CPUs.
- 19:25 Chose 11 dynamics cells and 4 static-only Theorem 9.2 instances with ttn.routes (all star).
- 19:33 Wrote common.py (cancellation-free model), driver.py (step-driven LSODA after sdyn.py), escape.py, static.py. Pilot: full net = reduced flow to 2e-9 (S3, A); 10 pilot runs 8-73 s, floor index <= 1.5e-7.
- 19:36 Pre-registration notes/route_escape.md committed and pushed (d487b50) before any sweep.
- 19:37 Sweep started (118 workers): 30 static tasks + 8085 escape runs.
- 19:49 Static orders (27/30): inside M* = c*_J; full space = c*_J except PS2, T92b, T92d (order 4). First results table pushed (08d22ad).
- 19:50 First failure seen: U5B8 kind C seed 9 escapes with exponent 4 (prediction 3); its route entry at leaf node 0 decays to 0 on the eps^-2 time scale.
- 20:12 Static orders complete (30/30); the Thm 9.2 formula holds on all 5 instances. Pushed (de9cc03).
- 20:13 Solver stall found (restarted LSODA stuck in nonstiff mode); restarted solvers become BDF (deviation 1).
- 20:19 Added kind Cp (C projected onto M*) as a causal test (deviation 2).
- 20:40 Main sweep stopped at 3990/8085 runs (13 more stalls) and restarted with the fixed driver.
- 20:45 Partner intervention: zeroing 30 partner entries removes the drain and restores eps^-3 in U5B8 direction 9.
- 21:18 Second stall pattern (LSODA nonstiff from the first step on M* plateaus): stall rule switches to BDF (deviation 1b); 50 remaining or capped runs rerun; Cp sweep completed (1232 runs).
- 21:30 Analysis, tables, figures, note Part II, README results.
- 21:50 Last two runs (PS2 C direction 1, eps 0.1 and 0.056) ended at the wall cap, effectively trapped; final commit and push.

Status: done
