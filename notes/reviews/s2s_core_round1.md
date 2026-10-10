# Review of s2s-core (round1) — 2026-10-10

Two independent local reviewers: (A) partial reproduction from the committed code plus an independent reimplementation, (B) adversarial audit of claims vs data. Re-run outputs in `data/review_<folder>/` (local, gitignored).

## A. Reproduction

**s2s-core reproduction review: all four claims reproduce, with one analysis bug and one small numerical bias**

Re-running the committed code gives the same results to 7–9 digits. My own from-scratch implementation also reproduces the key measurements on the 4-site trees. The analysis bug is in the trap decay-rate fit; once corrected, the trap theory fits better than the agent reported.

In plain terms: starting from a tiny random network, gradient flow learns the target one component at a time, with long flat stretches (plateaus) between components. A per-seed formula with nothing fitted predicts when the first component is learned. The "trap", where a not-yet-learned component dies off, is real, and balancing the initial tensors removes it.

### What I reran
Committed code was copied to `data/review_s2s_core/code` with three patches: output path, the aligned-run pool size, and the figure-(a) data path. About 26 minutes wall time, at most 4 processes. This machine ran each job 2–4.6x faster than the remote one.

- **Aligned init (claim 3):** `aligned.py`, all 5 trees, both integrators. I also regenerated figure (a).
- **Full saddle-to-saddle runs (`run.py`), 45 runs:**
  - binary4 and mps4 at χ = 3, seeds 0, 1, 3, σ = 0.1, 0.03, 0.01, 0.003.
  - mps4 at χ = 6, seeds 3 and 5, σ = 0.01.
  - Gauge-balanced mps4 at χ = 3, seeds 3 and 5, σ = 0.03 and 0.01.
  - K = 7: binary8 χ = 6 seed 0 at all 5 σ; ht4 χ = 6 seed 0 at 4 σ, plus seed 9.
  - `k78` sweep: ht4 χ = 3 seed 1 with fan-in and balanced init, two committed "stalled" runs, and one stalled balanced mps8 run.
- **First escape (`first_escape.py`):** binary4 and mps4 seeds 0–2, binary8 seed 0, 8 σ each (56 rows).
- **Correlation flow only (`lo_sweep.py`):** 10 inits each for binary4, mps4, binary8 (χ = 3 and 6), ht4, mps8.
- **Continuation:** `continue_stalled.py` on the two re-created stalled runs.
- **Analysis:** `analyze_main`, `analyze_first`, `analyze_dormant`, compared run by run with the committed CSVs.

**Independent check.** I wrote my own code for binary4 and mps4. It takes only the random initial tensors and teacher vectors from `ttn`. Everything else is mine: dense contraction, loss, gradient flow (LSODA, tighter tolerance), event location, dense SVD bond spectra, gauge balancing, and the correlation-flow blow-up time by a different method (projected flow on the unit sphere). I also computed the c_K offset and the exact aligned-mode escape times by quadrature.

### My numbers vs theirs

| Claim | Theirs | Mine |
|---|---|---|
| Aligned init vs exact solution, max \|a_GF − a_theory\| | 2.0e-10, 3.1e-10, 2.2e-9, 2.5e-9, 2.8e-9 (binary4, mps4, binary8, ht4, mps8) | 1.97e-10, 3.25e-10, 2.26e-9, 2.58e-9, 2.68e-9; event times identical; my quadrature = their theory times to 8 digits; figure (a) identical |
| Rerun runs: class and learning order | — | 45/45 and 45/45 match |
| Rerun runs: event times, relative difference | — | t1 ≤ 4.5e-8, t2 ≤ 1.4e-6, t3 ≤ 2.3e-6 |
| Plateau loss and top bond singular values; Hessian λ_min | — | identical; λ_min ratio 1.00 |
| One rank step per event on every edge at once | — | true in every complete rerun |
| Per-seed gap exponents on the same σ | e.g. binary8 seed 0: 4.986 / 4.924 | identical to 4 decimals (binary4 1.00 ± 0.01, mps4 1.96–2.00, binary8 4.92–5.00) |
| First escape t1 | — | 56 rows agree to ≤ 7.4e-10; first component 56/56 |
| First-escape exponents per seed | binary4 0.9995 / 0.9990 / 0.9998; mps4 2.0000; binary8 4.9983 | identical |
| Correlation-flow T vs my sphere method | — | mps4 agrees to 6.5e-11; binary4 committed T too high by up to 2.4e-5 (bug B2) |
| t1 from my own gradient flow | — | agrees to 1–2e-8 (binary4 seed 0; mps4 seed 2, where component 2 escapes first); prediction residual at σ = 0.002 is +2.8e-6 / −4e-10 with the converged T |
| T / Kumar–Haupt time, median of 10 inits | 4.0, 10.3, 95, 489 / 2655, 1461 | identical; smallest per-seed ratio 3.2, so the lower bound holds |
| Trap at K = 4 (mps4 χ = 3) | seeds 3, 5, 6, 7 trapped at every σ | seed 3 trapped at all 4 σ (agent code and my code: dead slice at 1e-15 until t_max = 2e7); seeds 0, 1 complete |
| χ = 6 or balanced init on trapped seeds 3, 5 | complete | complete; my own balancing gives t = 8170.8, 8778.1, 316608 vs 8170.8, 8778.1, 316607.8 |
| K = 7 trap checks | ht4 seed 1: trapped (fan-in) / complete (balanced); ht4 χ = 6 seed 9 trapped; mps8 balanced stalls after 2 components | all the same |
| Continued stalled runs | 0.8244, 0.8535 × route estimate | 0.824385, 0.853504 |
| Trap decay rate / σ², mps4 seed 3 | 0.0095 / 0.0113 at σ = 0.003 / 0.01; 0.25–0.27 at σ ≥ 0.03 | 0.259–0.260 at σ = 0.003 **and** 0.01 (both end cores); 0.26 at 0.03 and 0.1; my code 0.255 at 0.01 |

### Bugs and discrepancies (most important first)

**B1 – the trap decay-rate fit includes the float64 floor (`analyze_dormant.py`).**
- **Cause:** the fit uses every point with slice > 1e-13. A dead slice then sits at about 1e-12 to 1e-11 for the rest of the run, so the fitted "rate" becomes roughly ln(total drop) / window length.
- **Which runs:** the main-sweep runs at σ ≤ 0.02 were among the first ~310 runs, which had no dead-end stop and ran to t_max. So their rates are biased low by 20–27x.
- **Effect on the claim:** in `notes/s2s_core.md` §2.7(c) and the right panel of figure (g), two seed/teacher pairs "switch between two levels by 20–100x at σ ≈ 0.02–0.03". Those switches (separated teacher, seeds 3 and 7) happen exactly where the stopping rule changes.
- **Corrected picture:** the true decay rate per seed is σ-independent. The σ² law holds more cleanly than reported.
- **Not checked:** the "3–15x below the single-end estimate" statement may be affected the same way.

**B2 – small bias in the first-escape prediction T (`s2s.correlation_flow_blowup`).**
- **Cause:** the flow stops at amplitude 1e6 and adds a balanced-mode remainder. Committed binary4 T values are too high by 1e-6 to 2.4e-5.
- **Proof:** the agent's own function with a stop at 1e12 converges to my value (0.8437321106 vs 0.8437321067).
- **Effect:** irrelevant to the ≤ 2e-3 claim. But the reported binary4 residual (1.7e-5) is mostly this bias; the true residual is 2.8e-6. mps4 (components independent) and K = 7 are unaffected.

**B3 – continuation escape times are not resolved.**
- `continue_stalled.py` reads the escape off its coarse log grid. All 8 k78 ratios are midpoints of grid cells (match to ≤ 4e-9; one cell is a factor of 1.0353), so the resolution is ±1.8%.
- This is why 0.8535043 appears for binary8 seed 2 *and* for ht4 seed 9 at two σ.
- The 0.49–0.85x range stands; "the same ratio per seed across σ" is only established to ±1 grid cell.

**B4 – figure (a) may plot the wrong integrator's curves.** `aligned.py` saves both integrators' runs to the same `{tree}.npz`, so the figure, captioned as the DP5 runs, may show LSODA curves. Visually identical; harmless.

**B5 – re-run trap in `analyze_main.py`.** By default `--hess` reads the committed `hess_main.csv`. On any re-run whose raw folder is named main / large / k78 / balanced, it silently replaces the new λ_min with the committed values. I avoided it with an empty `--hess`.

**Bookkeeping and wording**
- **Edge rank-step count:** the CSV gives 497/531 runs with one step per edge; the notes and REPORT say 517/551. The 34 exceptions are consistent with the CSV.
- **mps4 plateau-level error:** maximum is 2.8e-3 in the main sweep and 8.1e-3 in the balanced sweep, versus "≤ 2e-3".
- **"S2S on every tree" overstates mps8:** no mps8 run completes. Under the project's regime rules (`notes/diagnostics.md`) every mps8 run is "stalled"; only two separated escapes are seen.
- **HT4 per-seed gap exponents range from 3.9 to 5.8** over 0.6 decade of σ. The cell means are 5.11 ± 0.09 and 4.75 ± 0.14, so "every gap ~ σ^-(K-2) per seed" is too strong for HT4; binary8 (4.88–5.09 per seed) is fine.
- **"Remaining stalls are ever-longer plateaus"** rests on 10 of 11 continued runs, 8 of them chosen as the shortest predicted escapes.
- **K = 8 trap fraction is a lower bound:** mps8 χ = 3 fan-in runs classed "stalled" have dormant slices down to 1.1e-8σ (median 3e-7σ), i.e. they are probably still dying.

### Do the README run instructions work as written?
They work after adding `--raw` / `--workers`, but:
- **Missing from the commands:** `large.yaml`, `k78.yaml`, `first_k78.yaml`, `lo_d` / `lo_K`, `recompute_hessians.py`, `continue_stalled.py` and the plot scripts. Without them the K = 7/8 results and the trap table cannot be regenerated.
- **Undocumented combined table:** the committed `main_runs.csv` combines four sweeps; the `analyze_main.py` command that produces it is not documented.
- **`aligned.py`:**
  - spawns 10 processes and has no `--workers`;
  - hard-codes `/root/outputs` (on Windows `C:\root`);
  - its default `--out` overwrites the committed `results/aligned.csv`.
- **Other hard-coded settings:** `plot_aligned.py` hard-codes its raw-data path; `continue_stalled.py` has no limit on its process count.
- **Machine-dependent stops:** runs also stop on a wall-clock limit, so the counts of "unresolved" runs depend on CPU speed.

Files are in `C:\Users\seamu\Iliad-ML\TTN_regimes\data\review_s2s_core\`:
- `analysis\compare_runs.csv`
- `analysis\compare_first.csv`
- `analysis\compare_exponents.csv`
- `analysis\compare_main.txt`
- `indep\*.json`
- `configs\`

Scripts and working notes are in `C:\Users\seamu\AppData\Local\Temp\claude\c--Users-seamu-Iliad-ML-TTN-regimes\ccfcd9a3-98b9-48c2-8218-49a3ab02b926\scratchpad\review_s2s_core\`:
- `indep.py`
- `compare_main.py`
- `compare_first.py`
- `compare_indep.py`
- `notes.md`


## B. Claims audit

**Claims audit: s2s-core.** I checked everything against the committed CSVs; the raw outputs are not available locally. Two claims fail on the agent's own data: the "no strict saddles" claim (issue 1) and the "stalls are only longer plateaus" claim (issue 3). The first-escape law, the Kumar–Haupt lower bound, the aligned-init reproduction and the sequential learning on binary4, mps4 and binary8 hold up (list at the end).

Working notes are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_s2s_core/notes_part1.md` and `notes_part2.md`. My one recomputation is in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_s2s_core/lo_r1.json`.

## Issues, most severe first

**1. MAJOR — "plateaus are higher-order saddles; no strict saddles seen with odeco teachers" and "every gap ~ σ^-(K-2) per seed".** The agent's own data contain a counterexample.
- binary4, close teacher [1, 0.85, 0.7], χ=3: seed 8 learns in order 1-3-2 at 7 of 8 σ, and seed 5 does so at 2 more σ.
- On the plateau where the 0.85 component is unlearned but the 0.7 one is learned, the gauge-projected λ_min is −0.106 to −0.146, flat over σ = 0.005–0.1. That is a strict saddle.
- The agent's own piggyback formula (notes §1.4) predicts −0.7^{1/3}(0.85−0.7) = −0.133, so the theory is confirmed to about 10%.
- For seed 8 the per-seed λ slope is 0.03 and the gap-2 exponent is −0.08: the gap does not grow at all.
- These runs fail the S2S test (separation ratio ρ≈1) but are pooled into the cell mean (gap 2 = 0.92 ± 0.11). The notes attribute that mean to "scatter from overlapping transitions".
- χ=6, seed 4 behaves the same way at σ ≥ 0.03 (λ_min −0.36 to −0.53).
- Fitting only S2S runs gives gap1/gap2 = 1.004 ± 0.010 / 1.016 ± 0.011 (χ=3) and 1.004 / 1.014 (χ=6).
- **Fix:** classify runs by learning order and S2S status before fitting. Report the strict saddle as a confirmed prediction. Restrict "higher-order" to greedy-order binary4 plateaus and tensor-like trees.

**2. MAJOR — scope of "S2S on every tree" and "every gap ~ σ^-(K-2)".**
- mps8: 0 of 284 runs ever learns the third component, so there is no gap exponent at K=8. The two-event runs do show a plateau at the partial-fit level, but t2−t1 was never fitted.
- At χ=R with K=7, fan-in runs rarely complete (binary8 14/180, ht4 12/150), so the K=7 claims rest on χ=6.
- K=7 gaps cover only σ = 0.07–0.3, which is 0.6 decade (the project rule requires ≥1 decade).
- binary8 is robust: local slopes are stable at 4.94–5.04 (per seed 4.67–5.14).
- ht4 is weak: per-seed local slopes range 2.3–6.6, and gap 2 = 4.75 ± 0.14.
- **Fix:** say exactly what was measured: complete S2S sequences on binary4, mps4, and binary8/ht4 at χ=6; on mps8 only the first two events. Fit t2−t1 for the mps8 two-event runs.

**3. MODERATE-MAJOR — "remaining stalls are ever-longer plateaus (10 of 11 continued runs escape at 0.5–1.8× the route estimate)".**
- The 8 k78 runs were chosen as the shortest predicted remaining plateaus: predicted/elapsed 0.2–1.0, against a median of 10–60× for stalled runs.
- They are only 3 independent seeds (binary8 fan-in seed 2, mps8 balanced seed 6, ht4 balanced seed 9). Runs of one seed at different σ are rescaled copies, because every run stops at the same multiple of σ^-(K-2).
- The measured times are not resolved. With the 200-point log output grid (factor 4), all five k78 ratios sit at position 0.496 inside one 3.5%-wide grid cell. That is why binary8 seed 2 and ht4 seed 9 report the identical ratio 0.8535043128.
- mps8 at χ=6 (spread sample): 2 runs escape above the derived upper bound (1.37×, 1.81×), 1 does not escape, and 5 give no result.
- mps8 "stalled" runs have route slices at a median of about 1e-6σ (1e-6.5σ in k78 fan-in), approaching the 1e-8σ "trapped" threshold. They look like routes still decaying, not live plateaus.
- **Fix:** present this as a derived prediction checked on 3 seeds. Resolve continuation times with `zoom_rise`, continue a random sample, and report slice trends for stalled runs.

**4. MODERATE — trap fractions.**
- At K=7–8 the fractions are lower bounds limited by the time budget. In the main and large sweeps the same seed changes class with σ, and in main about half of binary8/mps8 χ=3 runs end unresolved (49–53%).
- "σ-independent" is shown only over σ = 0.15–0.3 (k78), and mps8 seeds 1, 5 and 7 switch class even there.
- "40% at K=4" holds only for the separated teacher. The close teacher gives 14/80 trapped plus 8/80 stalled, and seed 7 traps only at σ ≤ 0.05.
- In k78/large the 1e-8σ threshold is one decade above the integrator's absolute tolerance (atol = 1e-9σ). The threshold and tolerance sensitivity was checked only in main, where atol = 1e-11σ.
- **Fix:** state the fractions as "at least", and add a tolerance check on a few K=7–8 trapped and stalled runs.

**5. MODERATE — "gauge-balanced init removes the trap (0/210)" and "χ=2R removes it".**
- 120 of the 210 balanced runs are in cells where fan-in never traps either. The informative count is 0/120: mps4 χ=3 0/30 vs 12/30 for fan-in at the same σ, and K=7–8 0/90 vs 47/90.
- Balanced K=7–8 runs still mostly stall (69/90).
- χ=6 is capped at d=4 on MPS end bonds and HT leaf bonds, so "2R" is literally true only on binary trees.
- HT4 at χ=6 traps 5/50 runs (seed 9 at every σ). The notes say "2/50" and the README says χ=2R avoids it on HT4; both are wrong. mps8 at χ=6 traps 7/50.

**6. MODERATE — first component predicted by the correlation flow (r1).**
- The 80–90% agreement on binary4 with the close teacher is a solver artifact. `correlation_flow_blowup` stops at amplitude 1e6, before the direction has left the smaller component's saddle.
- I reran it with amp_stop = 1e10 and 1e14: r1 becomes the largest component in all 5 mismatched seed/init cases (`lo_r1.json`).
- The cited cause (theory_escape E9, "wrong order at finite σ") is backwards: gradient flow learns the largest component first at every σ.
- T itself shifts by up to 1.7e-3, about the size of the claimed ≤2e-3 agreement on binary4 close.
- **Fix:** stop the solver on direction convergence and recompute.

**7. MINOR — numbers that do not match the CSVs.**
- Edge-step count: claimed 517/551, CSV gives 497/531. 332 complete runs have no event-window spectra and are untested.
- ">100× drops 0% at χ=2R": actually 0.7–2.5% at K=7–8 with χ=6, and 0.6% on mps4 χ=3.
- "Later escapes: same picture": at K=7–8, χ=3 the medians are 0.07–0.50, with up to 11% >100× drops.
- Kumar–Haupt ratio range: "3–500×" (notes §3) vs "3–2700×" (REPORT).
- R10 magnitude: claimed 0.8–1.7, CSV cell medians 0.78–2.9 (per seed up to 6.7).
- Plateau level "≤2e-2" omits binary4 close (up to 0.50 in non-S2S runs). The mps4 level is 2.8e-3 (8e-3 balanced), not ≤2e-3.
- "The rest are trapped" is wrong for mps4 close: 8 runs are stalled.
- "binary4 always learns the largest first" rests on 20 seeds.
- sd ln T at d=2 is 0.218, against the predicted 0.25 ± 0.02.

**8. MINOR — reproducibility.**
- README "How to run" leaves out large, k78, first_k78, lo_d, lo_K, `continue_stalled` (with its --n/--spread/--factor values), `recompute_hessians` and the plot scripts.
- It lists `results/balanced_*.csv`, which does not exist, and still describes the old amplitude round-off stop.
- Conservation drift exceeds the project's 1e-6 limit in 11 runs (mostly mps8), and drift is measured per chunk rather than from t=0.
- The rtol 1e-6 setting was validated on the event times of one run only.

**9. MINOR — status tags.** The first-escape law, the Kumar–Haupt bound, d-flatness at full bond, the χ slopes and the race order all replicate theory_escape (E1, E2, E5, E7, E8). They should be marked as replications, not new results. "Inside the predicted Kumar–Haupt range" is a weak test, because those ranges span 1–2 decades.

## Solid claims
- **Aligned init:** follows the exact Saxe modes on all 5 trees, max deviation ≤3e-9 (χ=3, d=4). A known result, reproduced.
- **First escape, per seed:** t1 = σ^-(K-2)·T(θ̂) + c_K with no free parameter.
  - Residual at the smallest σ: ≤2e-3 (binary4), ≤6e-5 (mps4), ≤2e-7 (K=7–8).
  - Exponents with the predicted offset removed: 0.9996, 2.0000, 4.998–4.999, 5.9995, over 1.6–1.7 decades with 6–10 seeds per cell.
  - The first component matches the correlation flow in 100% of runs on tensor-like trees.
- **Kumar–Haupt / ρ0 time:** a proven lower bound on T (I checked the derivation), low by 3–2700× (medians, K=3–8). The d-flatness at full bond (1.544, 1.569, 1.563) and the χ slopes (−0.335 ± 0.028, −0.264 ± 0.037, −0.69 to −0.76) reproduce from the CSVs.
- **S2S behaviour** on binary4 and mps4 (both teachers, χ=3 and 6) and on binary8 and ht4 at χ=6:
  - Plateaus sit at the partial-fit loss levels: separated teacher ≤2e-2, falling to 6e-7 as σ shrinks.
  - The top-|S| bond spectra (the |S| largest singular values on each bond, S = components learned so far) equal the learned weights.
  - Each event raises the bond rank by one on every edge (497/531 runs with data).
  - Separation ratios grow like σ^-(K-2).
- **Gap exponents:**
  - binary4 separated: 1.003, 1.004.
  - mps4: 1.99–2.004 in all 4 cells.
  - binary8 at χ=6: 5.02, 4.97, with stable local slopes.
  - binary4 close, restricted to S2S runs: 1.00–1.02.
- **Hessian at greedy and tensor-like plateaus:** λ_min scales like σ^(K-2).
  - Slopes: 1.02–1.08 (binary4), 1.99–2.02 (mps4), 5.07–5.10 (binary8).
  - Magnitude is 0.88–1.15× the route formula at K=3–4 (separated teacher).
- **Trap at K=4, χ=3, separated teacher:**
  - The same 4/10 seeds trap at all 8 σ.
  - The decay rate scales as σ² per seed for 5 of 7 seed/teacher pairs.
  - Balanced init removes it: 0/30 vs 12/30, and at K=7–8 0/90 vs 47/90.
- **Escapes:** they shrink dormant components only mildly (median ×0.5–0.9 at χ=6; on binary4 they grow ×1.5–2). Gauge-balancing changes T by 0.88–1.92×.

## Work left on s2s-core
About half a day, mostly re-analysis and editing rather than new runs:
- refit and reclassify (issues 1, 2, 4);
- fix the r1 solver stop (issue 6);
- correct the numbers and README (issues 5, 7, 8, 9).

Optional new runs:
- a tolerance check of the K=7–8 run classes;
- a random continuation sample with resolved escape times;
- t2−t1 fits for mps8.

The edge counts and any trap reclassification need the raw files on the remote machine (`/root/outputs/s2s_core/`).