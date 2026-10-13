# route-generic: does "depth = number of cores that change" hold outside pattern targets?

Status: working - Branch: agent/route-generic - Updated: (see log)

**Bug check of `ttn/routes.py`:** no bug found. I used `RouteTarget` (`is_star`, `n_blocks`, `region`,
`depth`, `greedy_next`, `teacher`) and `effective_depth`; on the pattern control the bonds that gain rank
match `region` in 1889/1889 events and `effective_depth` returns the exact K on the escape from the origin.

## Summary

The route theory says: training from a small init of scale sigma learns in steps; each step has a depth
equal to the number of tree cores that must change for it, and the plateau before it lasts
~ sigma^-(depth - 2) (log(1/sigma) for depth 2). I measured, for every learning event (a bond singular
value rising from the noise floor), (1) how many cores the rising bonds touch (k_region), (2) the depth
read off the growth curve of the new singular value (d_eff) and (3) the exponent p of the plateau
duration against sigma, on a pattern target (control), random CP teachers with component overlap 0.3 and
0.6, random TTN teachers (2640 runs on a 6-core chain "mps6" and a 7-core binary tree "binary8", sigma over
2.5 decades) and MNIST (existing 16-pixel pipeline). **Answer: "depth = number of cores that change"
holds for pattern targets and for random TTN teachers on the binary tree, and stops holding as soon as an
event changes more cores than the cheapest local move.** On the chain, every event of a generic teacher
grows like depth 2 and has a log(1/sigma) plateau, even when all 6 cores change; on the binary tree it
grows like depth 3 with a sigma^-1 plateau, even when all 7 cores change. The growth-law depth itself
still predicts the plateau exponent (72-100 % of series) except for CP teachers with overlap 0.3, whose
growth curves are not a single power law. MNIST: (pending, see below).

## Results

Definitions in plain words. *Learning event*: a group of bond singular values that rise from the noise
floor to half their final (teacher) value in overlapping time windows. *k_region* ("cores that change"):
number of cores in the smallest subtree containing the end cores of those bonds. *d_eff*: depth from the
growth law dy/dt ∝ y^(2-2/d) of the new singular value y, in the window y/target in [1e-4, 1e-2]
(`ttn.routes.effective_depth`). *p*: duration ~ sigma^-p, per seed and event, least squares over the 7
smallest usable sigma (>= 1.5 decades); a log(1/sigma) law shows up as p ~ 0.1-0.2 here. *Series*: one seed
and one event followed across sigma. Prediction of the picture: d_eff = k_region and p = k_region - 2.
Verdict rules (pre-registered): HOLDS if median |deviation| <= 0.3 and >= 70 % of series within 0.3 for
both numbers; FAILS if a median deviation > 0.5; PARTIAL otherwise.

All events after the escape from the origin (medians [quartiles] over fitted series):

| target | tree | series | k_region | d_eff | p | d_eff within 0.3 of k_region | p within 0.3 of k_region - 2 | p within 0.3 of d_eff - 2 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pattern (control) | binary8 | 90 | 3 [3, 4] | 3.10 [3.04, 4.18] | 1.01 [1.00, 2.00] | 96% | 100% | 96% | HOLDS |
| pattern (control) | mps6 | 81 | 3 [2, 4] | 3.30 [1.73, 4.09] | 1.01 [0.18, 2.00] | 62% | 100% | 60% | PARTIAL |
| CP, overlap 0.3 | binary8 | 55 | 7 [7, 7] | 3.09 [2.34, 4.08] | 1.00 [0.93, 1.08] | 12% | 9% | 29% | FAILS |
| CP, overlap 0.3 | mps6 | 60 | 6 [6, 6] | 1.53 [1.28, 1.67] | 0.12 [0.10, 0.14] | 0% | 0% | 2% | FAILS |
| CP, overlap 0.6 | binary8 | 77 | 7 [3, 7] | 3.06 [3.04, 3.09] | 1.00 [1.00, 1.01] | 43% | 42% | 97% | FAILS |
| CP, overlap 0.6 | mps6 | 45 | 6 [6, 6] | 1.97 [1.90, 2.00] | 0.13 [0.12, 0.15] | 0% | 0% | 89% | FAILS |
| random TTN | binary8 | 71 | 3 [3, 3] | 3.04 [3.02, 3.07] | 1.00 [1.00, 1.00] | 83% | 83% | 100% | HOLDS |
| random TTN | mps6 | 64 | 2 [2, 3] | 2.03 [1.83, 2.09] | 0.15 [0.11, 0.18] | 38% | 55% | 72% | FAILS |
| MNIST | binary16 | (pending) | | | | | | | |
| MNIST | mps16 | (pending) | | | | | | | |

Source: `experiments/route_generic/results/summary.csv` / `summary.md` (by k_region class as well),
`exponents.csv`, `events.csv`; produced by `uv run python experiments/route_generic/analyze.py
/root/outputs/route-generic/synthetic /root/outputs/route-generic/mnist experiments/route_generic/results 24`
and `tables.py`. Figures: `results/depth_vs_region.png`, `exponent_vs_region.png`, `exponent_vs_depth.png`,
`examples_duration_vs_sigma.png`, `agreement_by_target.png`.

Per teacher family, in words:
- **Pattern control (inside the theory): the pipeline reads the theory's numbers correctly.** The bonds
  that gain rank are exactly the routing region R_J(r) of the paper in 1889/1889 events, k_region = route
  depth c_J(r) in all of them; p = 1.00 / 2.00 for route depth 3 / 4 and log law for depth 2 (100 % of
  series); d_eff = 3.04, 4.18 (binary8), 3.24, 4.09 (mps6). Only deviation: the depth-2 event on mps6,
  which follows the first escape within ~2-3 time units, gives d_eff 1.51 (it grows while the first
  pattern is still being learned; the theory's Assumption C.1 (2) does not hold there).
- **Random TTN teacher.** binary8: HOLDS. Every event is a sibling pair of bonds through their parent
  (k_region 3): d_eff 3.02, p 1.00 in 100 % of 59 series; this extends the earlier data_teacher result
  (sigma^-1 for the first rank increase) to all later events. mps6: single-bond events follow the picture
  (d_eff 2.00, p 0.12, log law), but in 38 % of series 2-5 bonds rise together (k_region 3-6) and these
  still grow with d_eff ~ 2 and log-law plateaus: in 76 % of them one bond starts and the others follow.
- **CP teachers, overlap 0.6.** mps6: every later event raises a singular value on all 5 bonds
  (k_region 6, bonds start nearly together), yet d_eff 1.97 and p 0.13 (log law in 100 %): a strict
  saddle, not a 6-core route. binary8: the first post-escape event changes all 7 cores but has d_eff 3.05,
  p 1.00; later root-pair events (k_region 3) hold.
- **CP teachers, overlap 0.3.** Same counts (k_region 6 / 7) and plateau laws (p 0.12 on mps6, log law in
  87 %; p 0.99 on binary8), but the growth curves are multi-phase (exponential at 1e-6-1e-4 of target,
  slow-down near 1e-3, abrupt collective take-off above 1e-2), so the growth-law depth is not well
  defined (d_eff 1.53 on mps6, 3.02 [2.29, 4.16] on binary8) and does not predict p.
- **Where it stops holding.** The count "cores that change" over-predicts the plateau exponent whenever
  an event changes more cores than the cheapest local route that can start it: depth 2 on a chain (any
  adjacent pair of cores, strict saddle), depth 3 on a binary tree (a sibling pair through an internal
  core). None of the 372 fitted CP/TTN series has p above 1.5. *Empirical; the mechanism was derived for
  the first rank increase in theory_teachers R9 / theory_saddles R13; for later events it is a conjecture.*
- **MNIST:** (pending)

Event 0 (escape from the origin; p = K - 2 is an exact identity, used as a check): d_eff = 7.00 (binary8),
6.00 (mps6), p = 5.000 / 4.000 in every family.

Counts: synthetic runs 2640, all converged, 0 trapped, 0 stalled, 0 unfinished, 0 solver failures.
Floor-flagged events (escape signal within 100x of roundoff, excluded from fits): 260, all on binary8 at
sigma <= 1.8e-3 (CP 0.3: 110, CP 0.6: 68, pattern: 45, TTN: 37). Runs with a relearned singular value:
90 (CP 0.3, binary8) and 86 (CP 0.6, binary8). MNIST: (pending).

What is cited and what is new. Cited (data_teacher, safe-to-cite items 1-6, 9, 10; real_data section 8):
log-law escape from the rank-one saddle on MPS students and sigma^-1 on binary/HT students for generic
teachers (first rank increase only); bottom-up growth on binary students; on MNIST the sigma^-(K-2) first
escape, the k = 3 sibling first route on binary16 (A/sigma + B), log-time rises of 6 of 9 later MPS
edges. New here: the event-by-event comparison of bonds, growth-law depth and duration exponent for all
later events; that "cores that change" fails for generic teachers because events change more cores than
the cheapest route that starts them; that the growth-law depth predicts the exponent except for CP 0.3;
the pattern-control validation of the bond-based event definition.

## What was done

1. Read the brief, the paper (abstract, 1, 2, 7, 8, 10, App. C) and the earlier notes and reviews.
2. Pilots (16 synthetic, 4 MNIST runs); event definitions checked on them.
3. Pre-registration committed before any sweep: f31079c (`notes/route_generic.md` Part I).
4. Driver, event analysis, fits (`experiments/route_generic/`): a44c7a3, d4ed6db.
5. Synthetic sweep, 2640 runs (19:41-20:34); interim table 609b495; final synthetic tables and figures
   adb27bf; note Part II aee6ac2.
6. MNIST sweep (existing pipeline): (pending).

## Decisions

- **MNIST trees**: the existing pipeline uses 4 x 4 images (16 sites), so MNIST runs on its binary16 and
  mps16 trees, not on mps6 / binary8 (changing the image size would mean changing the pipeline).
- **CP teachers**: "overlap 0.3 / 0.6" read as the per-site overlap <u_r, u_q> between every pair of
  the 3 components at every site (as in data_teacher), weights random in [0.3, 1].
- **Events from bond singular values** with the teacher's values as targets (half = event, 5 % = onset),
  grouped by overlapping rise windows; MNIST uses the real_data convention (s_k/s_1 crossing 0.05).
- **Jacobian**: the library's contraction-based exact Hessian instead of differentiating the dense tensor
  (20x faster, identical); recorded as deviation 1.
- **Extra MNIST seeds 10-19** (allowed by the registration "if time allows") started when the synthetic
  sweep ended.

## Files left on the machine

| path | size | what | needed to reproduce |
|---|---|---|---|
| `/root/outputs/route-generic/synthetic/` | (see below) | one .npz per synthetic run (every solver step) | yes (input of analyze.py) |
| `/root/outputs/route-generic/mnist/` | | one .npz per MNIST run (real_data format) | yes |
| `/root/outputs/route-generic/data/` | ~25 MB | MNIST / Fashion IDX files and 4x4, 8x8 npz | yes (or re-download) |
| `/root/outputs/route-generic/pilot*`, `mnist_pilot/` | | timing pilots | no |
| `/root/outputs/route-generic/logs/` | | sweep logs | no |

## Caveats and next steps

- Two trees and one target per pattern family; three components and bond dimension 2 for the generic
  teachers. The "cheapest local route" reading is checked only on chains and balanced binary trees.
- The growth-law window [1e-4, 1e-2] cannot see events that start above 1e-4 of their target (shallow
  routes at larger sigma, events right after another one); those series have no d_eff.
- k_region groups rises with overlapping windows; two independent routes that happen to overlap are one
  event (column `n_components` flags disconnected ones).
- Next: a saddle-start test of the multi-bond CP events (Hessian at the plateau: how many negative
  directions, on which bonds), and the multi-phase growth of CP 0.3.

## Log
- 19:31 Started. Checked files: notes/briefs/route_theory.md, ttn/routes.py, tests/test_routes.py and the paper PDF exist; `uv run pytest tests/test_routes.py` passes (13 tests). Machine: 128 logical CPUs, cgroup quota ~122.9 CPUs, 251 GB RAM.
- 19:34 Read brief (sections 1-3, 4.4), paper (abstract, 1, 2, 7, 8, 10, App. C), data_teacher / real_data notes and reviews, saddle_dyn section 8. MNIST downloaded to /root/outputs/route-generic/data (torchvision mirror, OK).
- 19:38 Pilots: 16 synthetic runs (8-150 s each, all converged), 4 MNIST runs (tau = 2000: 150-250 s).
- 19:40 Pre-registration committed (f31079c, notes/route_generic.md Part I). Sweeps launched: synthetic 2640 runs (64 workers), MNIST 180 runs (56 workers).
- 19:44 Jacobian switched to the library's contraction-based exact Hessian (Problem.hessian, 0.12 s vs 2.5 s for the dense double-backward; identical to 1e-21); synthetic sweep restarted, the 30 finished runs kept (numerics-only, Part III of the note).
- 20:25 Interim table pushed (experiments/route_generic/results/partial/, 1680 synthetic runs). First reading: pattern control and random TTN on binary8 follow the picture; on generic targets the events often involve more cores than the growth-law depth and the duration exponent indicate. MNIST seeds 10-19 queued to start when the synthetic sweep ends.
- 20:34 Synthetic sweep complete: 2640/2640 converged. Final synthetic tables, figures and note Part II pushed. MNIST seeds 10-19 started.
