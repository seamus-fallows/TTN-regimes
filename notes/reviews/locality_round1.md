# Review of locality round 1 (2026-10-09)

Two independent local reviewers: (A) partial reproduction from the committed code plus an independent reimplementation (dense contraction, Radau, own Hessian and gauge generators, own Steiner counts), (B) adversarial audit of claims vs data. Re-run outputs: `data/review_locality/` (local, gitignored).

## A. Reproduction

I reran the requested subset using the committed code and configs, with three small patches (below). Everything I reran matches the committed numbers to every printed digit, and my own implementation agrees with the agent's runner. The physics claims hold. The problems are in bookkeeping and in some summary numbers: the balanced-init stall count, the χ = 4 stall count, the k = 2 saddle-start λ_min, and what the 98.7% bond statistic actually measures.

## What I reran
- **Config:** `data/review_locality/configs/repro_main.yaml` copies 11 rows verbatim from `twocomp.yaml`, `twocomp_balanced.yaml` and `multicomp_wide.yaml` (same σ grids, seeds and protocols).
  - k = 1: binary4 {0,1}
  - k = 2: mps4 {1,2} small and saddle start; mps8 {3,4} saddle start
  - k = 3: binary4 {1,2}
  - k = 4: mps4 {0,1,2,3}; mps4 {0,3} fan-in and gauge-balanced
  - k = 5: binary8 {3,4}
  - k = 7: ht8 {3,4} saddle start
  - TD on mps8, saddle start
- **Scale:** 556 of 558 runs, 6 workers, about 24 minutes. I stopped the last two (ht8 seed 4 at σ = 0.198 and 0.15); the committed table lists those two as stalled.
- **Patches:** none of these edit repo files.
  - The runner copy uses eager `torch.func` instead of `torch.compile`, and takes the trajectory directory from an environment variable.
  - The analysis runs with `PYTHONUTF8=1`.
- **Analyses on the rerun:** the agent's `make_figures.py`, `growth.py` and `levels.py`, plus my own checks.
- **Library caveat:** the working tree has someone else's uncommitted changes in `ttn/diagnostics.py`, `ttn/dynamics.py` and others. I read the diff: it does not touch the functions the runner uses (velocity, Hessian, loss, gauge projection, teacher, init).

## Rerun vs committed

| row | k | quantity (prediction) | committed | rerun |
|---|---|---|---|---|
| binary4 {0,1} small | 1 | gap exponent; run order | −1.002 ± 0.001 (gap ∝ σ, no plateau); 28 / 28 | identical |
| mps4 {1,2} small | 2 | log slope (1.000); λ_min at σ = 1e-4 | 1.0085 ± 0.0024; −0.4987 | identical |
| mps4 {1,2} saddle | 2 | log slope (2.000); λ_min | 2.018 ± 0.024; −0.444 | identical (my Hessian: **−0.5000**) |
| mps8 {3,4} saddle | 2 | log slope (2.000); λ_min | 2.002 ± 0.010; −0.460 | identical (my Hessian: **−0.5000**) |
| binary4 {1,2} small | 3 | p (1); λ_min σ-exponent | 1.011 ± 0.005; 1.042 | 1.011 ± 0.005; 1.042 |
| mps4 {0,1,2,3} small | 4 | p (2) | 1.990 ± 0.006 | identical |
| mps4 {0,3} fan-in | 4 | p; greedy / rotation / stalled | 2.015 ± 0.034; 17 / 1 / 30 (seeds 1,3,4,5,7) | identical, same stalled seeds |
| mps4 {0,3} balanced | 4 | p; stalled | 1.990 ± 0.004; 0 / 48 | identical |
| binary8 {3,4} small | 5 | p (3); 3-smallest-σ p; Hessian exponent | 3.012 ± 0.082; 2.986; 3.046 | 3.012 ± 0.082; 2.986; 3.047 |
| ht8 {3,4} saddle | 7 | p (5) | 5.74 ± 1.06 (median 5.01) | 5.74 ± 1.06 (28 escaping runs identical) |
| TD mps8 saddle | k* = 2, 2, 3 | exponent at each predicted plateau level | 0.267, 0.649, 1.012 ± 0.010 | identical (agent's script and my own) |
| growth estimate k = 2/(2−γ) | 2–7 | single-run k | 1.993 … 5.120, 7.733 | same to 3 decimals |

Hessian numbers agree with the committed ones to within 0.1–0.5% for k ≥ 3, because the plateau midpoint is picked from the stored integrator steps. All event-time quantities agree to 5 significant figures.

## Independent checks (not using the agent's runner or analysis)
- **Steiner counts:** my own leaf-pruning implementation, `ttn.tree.Tree.steiner_nodes`, the agent's rank rule and the table's k column agree on all 60 configured (tree, S) pairs. They also agree on all 999 nonempty S across the 9 trees.
- **Gradient flow:** I wrote my own dense contraction, autograd gradient and Radau integrator (rtol 1e-10). The initial cores match `TTNSpec.init` exactly. All 11 runs agree with the runner on the gap t2 − t1:
  - 3e-8 for binary8 {3,4} at σ = 0.1 (k = 5);
  - 2e-8 for mps4 {0,1,2,3} (k = 4);
  - 4e-7 / 4e-6 for binary4 {1,2} (k = 3);
  - about 1e-9 for the k = 2 runs.
  - My seed-0 log slope for mps4 {1,2} small init is 1.0076 (predicted 1.0).
- **Gauge-projected Hessian (claim 2):** I computed my own Hessian and my own gauge generators at the recorded midpoints, 53 states in total.
  - My projected λ_min equals the runner's to between 1e-15 and 7e-11.
  - For k ≥ 3, seed 0, λ_min scales as σ^1.035, σ^2.003 and σ^3.064 for k = 3, 4, 5.
  - **Saddle-start k = 2:** projecting out only well-conditioned gauge directions (singular value > 1e-3 × max), or not projecting at all, gives −0.5000 ± 5e-5 in 16/16 states, which is exactly −λ2.
- **Bond spectra (claim 4):** I computed the second singular value of every bond by SVD of the dense tensor along the saved trajectories.
  - In all 131 N = 8 runs (plus 111 mps4 {1,2} runs) the non-Steiner edges never reach rank 2.
  - All Steiner edges rise within 0.01 event widths of each other and of t2.
- **Dead routes (claim 7):** I integrated two runs the runner had flagged as stalled, with the stall detector off.
  - mps4 {0,3}, seed 1 at σ = 0.2 and seed 5 at σ = 0.11: the route's second singular values fall to about 1e-16 by t ≈ 1e4–1e6. They stay there and the second component is still unlearned at t = 1e10.
  - Control (seed 0, same σ): it learns both components by t = 1e3.
  - So "stalled" really means a dead route, not a slow one.

## Discrepancies and likely causes
1. **Claim 2, k = 2 saddle start.** The REPORT says λ_min = −0.49 to −0.50; the committed table and my rerun give −0.444 ± 0.024 (mps4) and −0.460 ± 0.013 (mps8). The cause is the gauge projection: gauge directions on the unused bond index have size of order σ, still pass the 1e-10 cut, and point partly along the escape route. Removing them cuts away part of the negative direction. The physics claim (−λ2) is correct; the reported number is biased.
2. **Claim 7, "2/1008" balanced-init stalls.** `twocomp_balanced_table.csv` actually shows **16 of 1008** stalled. On the same cases with fan-in init it is 169 of 1042, so balancing cuts stalls about 10×, not about 85×. Balanced init still stalls on mps8 {2,5}, a k = 4 route (2/48).
3. **Claim 7, χ = 4 "75 of 1,002".** The case midroot6 {0,5} (30 of 30 runs stalled) is missing from `twocomp_chi4_table`, because `make_figures` drops any case in which no run escapes. The notes' own per-case list adds up to 103.
4. **Claim 4, "98.7%".** `bond_events.py` compares bond ranks at the plateau midpoint with ranks at the end of the run. At the end W ≈ W*, which has rank 2 exactly on the Steiner edges by construction, so the statistic is close to tautological. The 65 misses are k = 2 small-init runs at σ = 0.03–0.1 whose "midpoint" is already half-escaped. My own test (above) supports what claim 4 actually says.
5. **Claim 5, single-run k.** For small-init k = 2, the committed `growth_summary` gives k ≈ 2.36, 2.89 and 3.33 on mps6 {2,3}, mps8 {0,1} and mps8 {3,4}, so the estimate fails there. It is biased high for larger k: 5.06–5.32 at k = 5 and 7.7 at k = 7.
6. **Claim 1, "within ±0.05 for k ≤ 4".** 5 of the 16 k = 4 rows fall outside, with saddle-start values from 1.49 to 2.39. The "±0.15 for most k = 5 rows" statement holds (13 of 17).
7. **Claim 3, ht8 p = 5.7.** The fit spans σ = 0.15–0.6.
   - At σ ≥ 0.45 the saddle-start overlap begins below ½, so those points are not near the saddle.
   - One seed with a pinned route (seed 4, slope about 9.6) drives the mean; the median is 5.01.
   - This is consistent with 5 but does not measure it.
8. **Claim 6, TD.**
   - Reproduced: the last level's exponent is 1.012 ± 0.010, which picks the route rule (k* = 3) over the literal H2 rule (k = 2) and the bond-rank rule (k = 4).
   - Minor: the notes write "1.02 ± 0.01 (7 seeds)" while the CSV has 1.012 with 8 seeds.
   - Not supported: level 1 is called "log", but the committed fair comparison prefers a power law (residual 0.162 vs 0.389 for log, p = 0.65, 12 of 55 durations unresolved).
9. **Provenance.** The table rows ht4 {0,1,2,3}, ht8 {0,1,2,3} and mps8 {0,7} include runs on σ grids and seeds that are not in `configs/twocomp.yaml` (11 σ values from 0.08 to 0.6 instead of 6 from 0.15). Those k = 7 and 8 numbers cannot be regenerated from the committed configs.

## Bugs and portability (README check)
- **The README commands do not run on Windows without a C++ compiler.** `torch.compile` raises InductorError, and `run_task` catches all exceptions, so every run is silently recorded as "error". Eager mode gives identical numbers.
- **Output and input locations are hard-coded:** `/root/outputs/locality` in `run.py` (trajectory dir), `finalize.sh` and `bond_events.py`. This breaks analysis standard 7.
- **`make_figures.py` crashes on Windows** after writing the CSV: `write_text` without an encoding fails on "σ" under cp1252.
- **`make_figures` silently drops cases with no escaping run** (cause of discrepancy 3).
- **Rotation runs are pooled with greedy runs.** The effect is small: binary8 {3,4} goes from 3.012 to 2.998 without them.
- **`run.py` checks the wall-clock limit inside `fun()`**, which post-processing also calls. A run finishing near the limit would be recorded as "error".
- **`balance_gauge` returns non-finite cores** on midpoint states with ill-conditioned bond Gram matrices. This is harmless at t = 0, where it is actually used.
- **The README omits the `twocomp_chi4.yaml` run**, and `make_configs.py` does not write `twocomp_chi4.yaml` or `k5_extension.yaml`.

Everything is in two folders. Longer notes are in the scratchpad folder; start with `final_issues.md` and `final_table.md`.

Files are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_locality/`:
- `out/` (run records)
- `traj/` (saved trajectories)
- `figs/` (`compare_committed.txt`, `main_table.csv`, `levels_summary.csv`, `growth_summary.csv`, `my_bonds_*.json`)
- `indep_hess/`, `indep_flow/` (`compare.txt`)
- `steiner_check.txt`
- `code/` (patched copy of the agent's scripts)
- `configs/`

Notes and scripts are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality/`:
- `notes_0`–`notes_6`, `final_table.md`, `final_issues.md`
- `indep.py`, `my_fits.py`, `my_bonds.py`, `my_levels.py`, `stall_check.py`, `steiner_check.py`


## B. Claims audit

**Locality round-1 claims audit (claims auditor)**

Two of the seven claims hold as stated: the k = 2 head-start law and the k = 3 exponent on all nine trees. Two headline numbers are not real tests or are misquoted: the 98.7% bond-event figure and "2 of 1,008" stalls under gauge-balanced init. The k = 2 Hessian values are biased by a library projection bug that I confirmed numerically.

**What I checked myself**
- **Steiner counts:** all 60 distinct (tree, S) rows match `k_rank`. I used my own leaf-pruning code, `Tree.steiner_nodes`, and `rules.route_rule` at s1·U1; there were 0 mismatches.
- **Hessian on mps4:** at the exact saddle, gauge-projected λ_min = −0.5000 with one negative eigenvalue for k = 2, and none for k = 3 or 4.
- **Saddle-start reruns:** I recomputed λ_min at the minimum-speed point for k = 2 and k = 3. For k = 3 it scales as σ^1.0.

Outputs are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_locality/claims_audit/`. Supporting numbers are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality/claims_audit_evidence_{1,2,3}.txt`.

## Issues, most severe first

**1. HIGH: claim (1) is broader than the data ("k = 1..8 on 9 trees, ±0.05 for k ≤ 4").**
- **k = 4 from a saddle start:** midroot6 {0,3} gives 1.49 ± 0.35, mps6 {0,3} 2.39 ± 0.47, mps6 {1,4} 2.19 ± 0.18. The ±0.05 only holds for small-init rows (1.95–2.06).
- **k ≥ 6 rows break the project's own reporting rule:** σ spans 0.5–0.9 decades, and 10 of 15 rows have 0–4 fitted seeds.
- **k = 7:**
  - Exponents are 5.80, 6.86 ± 0.64, 5.74 and 5.91 ± 0.43 against a prediction of 5; two rows are 2–3 SE high.
  - The λ_min exponents are 5.99–6.89, not k − 2.
- **k = 8:** one case only, mps8 {0,7}, over σ 0.2–0.6.
  - Fan-in init gives 5.16 ± 1.20 with 3 seeds.
  - Balanced init gives 5.42 ± 0.17, which is 3.4 SE below the prediction of 6.
  - χ = 4 gives 7.29 ± 0.68 with 2 seeds.
- **Fan-in and balanced init disagree for k ≥ 6:** ht4 k = 7 gives 5.91 vs 4.98, and mps6 all-sites gives 4.84 vs 3.81. Selection or init imbalance biases these rows.
- **ht8 k = 5:** the REPORT says "3.1–3.8", but the committed CSV has 2.74 ± 0.37.
- **Coverage per k is sparse:** k = 2 and 4 occur only on chains and star3, k = 7 only on HT trees, k = 8 only on mps8 {0,7}.
- **Fix:** state H2 as confirmed for k = 1–5, except the MPS k = 5 rows at χ = 2. For k ≥ 6 say the exponent grows with k but is not pinned to k − 2. Extend those rows with saddle-start or balanced init over ≥ 1 decade with ≥ 5 seeds, and finish the 11 dropped slow runs, which are the longest plateaus.

**2. HIGH: claim (4), "rank rises on the Steiner edges in 98.7% of 5,132 events", is close to a tautology.**
- All 5,132 events are two-component; no multi-component event is in the CSV.
- For two components, "after" is the end of the run, where every overlap is ≥ 0.97 and W ≈ W*. The test therefore only re-checks the proven teacher-rank identity, plus rank 1 at the plateau midpoint.
- All 65 mismatches are "missing" edges: the dormant second singular value was already above threshold at the midpoint, mostly in small-init k = 2 runs.
- It does not test that the rise happens at the event, that the Steiner edges rise together, or that other edges stay rank 1.
- **Fix:** compare ranks just before and after the event (at the a = 0.1 and 0.9 crossings). Measure the spread of crossing times across Steiner edges relative to the event width. Add multi-component events, using the route rule's predicted edges rather than `rank_rule`.

**3. MEDIUM-HIGH: claim (2) λ_min values are biased by a library bug in `ttn.diagnostics.project_out_gauge`.**
- **What goes wrong:** near a rank-deficient saddle, the normalised gauge generator of a nearly unused bond index points along (ε_u, −ε_v). Unless the route is balanced, this overlaps the route's negative mode, so projecting it out removes part of that mode.
- **My check** (mps4, saddle start, σ = 1e-4 and 1e-2):

  | s2 | raw λ_min | projected λ_min (4 seeds) |
  |---|---|---|
  | 0.25 | −0.2500 | −0.196 to −0.248 |
  | 0.50 | −0.5000 | −0.416 to −0.496 |
  | 0.75 | −0.7500 | −0.636 to −0.744 |

  The projected value depends on the seed but not on σ.
- **Effect on the claims:** the committed table has −0.41 to −0.50 (claimed −0.49 to −0.50). The robust-table values −0.221 and −0.655 were reported as agreeing with −s2.
- Strict versus higher-order classification is unaffected.
- **Fix:** report raw λ_min, or project out only gauge generators that are near-null directions. The lead should warn the other agents that use gauge-projected Hessians.

**4. MEDIUM-HIGH: claim (7) stall counts are wrong and the "dead route" label is not verified.**
- **Counts:**

  | sweep | REPORT/notes | committed tables |
  |---|---|---|
  | gauge-balanced init | 2/1,008 (notes: 2 of 580) | 16/1,008 |
  | χ = 4 | 75/1,002 | 103/1,002 (the notes' own per-case list sums to 103) |
  | fan-in | 376/5,732 | 397/5,761 |

  The k5 extension adds 15 of 48 balanced-init stalls at σ ≤ 0.03.
- **"A stalling seed stalls at every σ" is false:** in 21 of 33 fan-in rows with stalls, the count is not a multiple of the number of σ values. That means σ-dependent dropout, which biases the fits.
- **Detector:** a run is flagged when |v| < 1e-12 for 5,000 evaluations. This can also fire on a slow route whose ends have shrunk to about 1e-7·σ, not only on a truly dead one.
- **Conflict with theory:** stalls under balanced init (Q = 0) contradict R12 ("exactly balanced init avoids the trap"). The theory note's P12 heuristics (about 0.5 dead per two-ended HT route; MPS stalls less often) are contradicted too: ht4 k = 5 had 0/48 stalls versus 30/48 on mps4 {0,3}. Neither conflict is discussed.
- **Fix:** correct the numbers. Record route-end amplitudes and the R12 criterion (Q + |y|² < 0 at both ends) at each stall. Run a subset longer to separate dead from slow.

**5. MEDIUM: "steep MPS k = 5 exponents come from shrinking route ends" is stated causally without support.**
- **Counterexamples in the agent's own CSV:** route ends scale like σ^α with α > 1 in rows whose exponents are on target. binary8 {0,7} with balanced init has α = 1.34 ± 0.14; the notes' formula (k−2)(2α+k−2)/k then gives 3.41, but the measured exponent is 2.99 ± 0.01. The same happens for binary8 saddle (1.28), ht4 {1,2} saddle (1.31) and mps4 k = 3 saddle (1.29). The notes claim α = 0.95–1.14 for normal rows.
- **Derivation:** the formula assumes a balanced route. In the route ODE, ε_u² − ε_v² is conserved, so small ends catch up exponentially. This gives T ≈ σ^−(k−2)·[(α−1)ln(1/σ) + 1/(k−2)]/λ, a log correction of about +0.3 rather than a shift in the power.
- **χ = 4 does not fix every case:** the mps6 k = 6 rows stay at 5.0–5.1.
- **Theory threshold:** at χ = 2 the theory note's refutation test P14(ii) is met (mps8 {1,5}: 3.72 ± 0.16 at σ ≤ 0.03). The notes should say so.
- The shrink ratio from first event to midpoint (0.53–0.78 in anomalous rows vs 0.85–1.03 in normal rows) is a real marker, so "associated with" is fair; "come from" is not.

**6. MEDIUM: claim (6), multi-component teachers. TC and TD are solid; the rest is weaker than stated.**
- **"Visited fraction" means nothing:** the loss is monotone, so every level is crossed, and `visited_frac` = 1.000 in all 111 rows.
- **"log" labels:** these come from "power < 1 and positive log slope", not from a fair comparison. The power law fits better in 9 of 23 main-order k* = 2 rows; for example E8 mps8 first level is 0.024 vs 0.112, and TD saddle second level 0.16 vs 0.39.
  - The TD small-init second level has a log slope of −0.41 ± 0.61.
  - Up to 36 of 115 durations are unresolved.
- **k* = 3 levels run high:** values are 1.05–1.68 (e.g. TB ht4 1.59 ± 0.14), and the "band catches the approach" explanation was not tested.
- **Which teachers discriminate:** TA separates literal H2 only. TD and TC are the only cases separating the route rule from the rank rule, and both are clean (see the solid list).
- **Fix:**
  - Put the fair-comparison scores in the notes table.
  - Detect all plateaus in L(t) and match them to the predicted levels.
  - Compare the run records' plateau-midpoint loss with the predicted levels.
  - Use saddle starts at each predicted saddle.

**7. MEDIUM: claim (5), single-run growth exponent.**
- For k = 2 only saddle-start rows are quoted. Small-init k = 2 gives γ medians of 1.15–1.40 (k estimate 2.4–3.3 on mps6/mps8).
- There is an upward bias that grows with tree size: ht8 gives 5.32 for k = 5 and 7.73 for k = 7.
- Resolution is too coarse at large k (dk/dγ = k²/2): the k = 7 estimate (7.73) rounds to 8, and k = 8 gives 7.66–7.89.
- Per-run 5–95% ranges are wide, e.g. 5.2–7.8 for k = 7. The summary rests on medians over the two smallest σ.

**8. MEDIUM: rotation runs are pooled into every exponent fit.**
In every table row, runs = greedy + rotation, which goes against analysis standard 3. Examples with balanced init: mps8 {0,7} has 12 of 26 runs as rotations, mps8 {1,5} 10 of 44. Refit without them.

**9. LOW-MEDIUM: dormant-amplitude test (notes 7.3).**
- The quoted k = 2 range (0.98–1.01) covers saddle-start rows only. Small-init slopes are 1.04–1.11 with SE about 0.01.
- k = 3 slopes are systematically low: −0.25 to −0.31 against −1/3.
- The regression pools seeds instead of fitting each seed.
- The test is partly circular: remaining time from a_mid follows from the growth law alone.

**10. LOW-MEDIUM: claim (3), the same-teacher contrast, mixes protocols.**
- binary8's 3.01 is small init, while mps8 and ht8 are saddle start.
- ht8's 5.7 ± 1.1 over 0.6 decades cannot tell k = 6, 7 or 8 apart.
- For mps8 small init, the log and power laws are not distinguished (RMS 0.0040 vs 0.0053).
- The qualitative ordering (log, then 3, then about 5.7) holds.

**11. LOW: smaller inconsistencies and gaps.**
- **Stale notes:** `notes/locality.md` 7.1 still says "pending", gives k = 4 as 1.81–2.39, and trims outliers from the Hessian-exponent ranges.
- **TC numbers:** section 6 and the verdict pair TC's N and k values differently.
- **"Iff" in section 6:** the H3 "iff" there lacks the "no stars" caveat that the verdict adds.
- **k = 1 separation ratio:** "ρ < 3" is a hard-coded string, not computed.
- **λ2 untested in K:** s1 = 1 everywhere, so the K-dependence of λ2 is never exercised.
- **Conservation check:** it is stated against |Q| ~ 1e-2 to 1, but for small init |Q| ~ σ².
- **Headline scope:** it should say orthogonal pattern teachers with d = χ = 2.
- **midroot6:** matching mps6 holds for exponents only; trap rates differ (30/30 vs 18/30 at χ = 4).

## Claims I consider solid
- **Proven or derived pieces checked:** the Steiner bond-rank identity, the section 2–3 derivations, the head-start factors, and the exact-saddle Hessian. My algebra and numerics agree.
- **k = 3 exponent ≈ 1** on all nine trees, both protocols: 42 rows, 0.98–1.03, three-smallest-σ estimates 0.97–1.01.
- **k = 4 small init:** 1.95–2.06; balanced init 1.94–2.01; χ = 4 1.99–2.03.
- **k = 5 on binary8 and ht4:** 2.88–3.13 across all protocols.
- **k = 2 log law with the head-start slope:** 0.98–1.01 from small init vs 2.00–2.07 from a saddle start. The fair comparison favours the log law (except mps8 S = {3,4} with small init). It holds for s2 = 0.25 (3.02 vs 3.00; saddle 4.04 vs 4.00).
- **k = 1 has no plateau.**
- **H3 classification:** k = 2 has a σ-independent negative eigenvalue (raw value exactly −λ2); for k ≥ 3, λ_min → 0 with exponent about k − 2 for k = 3–5.
- **Route rule over literal H2 and the rank rule** for TD (last plateau 1.012 ± 0.010; power RMS 0.027 vs log 1.30) and TC (k* = 4 level at 2.018 ± 0.017), saddle start over three decades.
- **B4 on mps4:** the 0.3 component is learned before the 0.6 one in every escaping run.
- **Dead routes:** none in 3,560 runs with k ≤ 3; frequent for k ≥ 4 with fan-in init at χ = 2; greatly reduced by balanced init or χ = 4. The counts need correcting (issue 4).
- **Integrator agreement:** solvers agree to 3e-8.