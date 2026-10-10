# Review of data-teacher (round2) — 2026-10-10

Two independent local reviewers: (A) partial reproduction from the committed code plus an independent reimplementation, (B) adversarial audit of claims vs data. Re-run outputs in `data/review_<folder>/` (local, gitignored).

## A. Reproduction

**Data-teacher round 2: reproduction review**

**Verdict:** All four groups of key claims reproduce from the committed code. Rerun outputs match the committed CSVs to between 1e-9 and 1e-3, with one explained exception (randbin_302, below). My independent reimplementation, which uses none of the agent's code and not the `ttn` library, gives the same saddle Hessians and the same escape laws. I found no errors in the results. I did find one cross-platform reproducibility bug (Windows), one per-run non-reproducibility (randbin_302), an overstated interpretation (star "forgetting"), and a loose statistic (correlated-input increments).

Terms used below:
- **Saddle-start:** gradient flow started at the exact bond-rank-1 critical point plus noise of size ε.
- **Strict saddle:** the Hessian has a negative eigenvalue.
- **Log law:** escape time t = A ln(1/ε) + B, with predicted slope A = 1/|λ_min|.
- **Power law:** t ∝ ε^(−p).

**What I reran** (4 or fewer busy processes, 09:54–10:34; the machine was at 100% CPU from other reviewers)

I copied the code unchanged (identical under `diff -rq`) to `data/review_data_teacher/r2_repro/code`. I ran it with `DATA_TEACHER_OUT` and `DATA_TEACHER_RESULTS` set and analysed with the agent's own analysis scripts. 185 raw runs in total:
1. **Saddle-start** (`r2_saddle_runs.py`, reduced configs, 86 runs):
   - mps4: 4 teachers, ε from 1e-8 to 5e-2;
   - binary4 and ht4: 2 teachers, ε from 1e-5 to 5e-2;
   - mps8: 2 teachers;
   - Hessian rows for every combination.
2. **Correlated inputs** (`r2_corr.job`): mps ρ = 0.4 and 0.2 with seeds 0, 1, 3, 4 at all 6 σ; binary ρ = 0.2 with seed 0 (54 runs).
3. **Stars** (`r2_star.py`):
   - Hessian mode;
   - threshold scan at w = 1.4138 and 1.4146 (mps4 and binary4, ε = 1e-6);
   - rotation law;
   - ht4 at ε = 1e-3 and 1e-4;
   - random-init forgetting depth (5 runs).
4. **4a:** `run_dense.py` + `analyze_law_v2.py`, c = 0.45, seeds 2, 4, 9, σ from 3e-4 to 0.03.
5. **4b:** teachers 104 and 102 at σ = 1e-3, then `analyze_ttn.py`.
6. **Random init (item 1):** `nonorth_c0.45` with seed 2 on mps4 and binary4 (8 runs).
7. **Independent checks** (own einsum TTN, HOPM or SVD saddle, scipy LSODA, own event rule): `indep_saddle.py`, `indep_star.py`.

| claim | committed | rerun (agent code) | independent |
|---|---|---|---|
| Exact-saddle λ_min | nonorth_c0.45 mps4 −0.150890082 | identical (≤1e-15; randbin 2e-11) | −0.1508900825 |
| | nonorth_c0.2 mps4 −0.02845795 | identical | −0.0284579495 |
| | binary/HT: no negative eigenvalue | same | same, binary4 and ht4 |
| MPS escape, log-law slope A (predicted 1/\|λ\|) | nonorth_c0.45: 6.648 ± 0.010 (6.627) | 6.636 ± 0.005; per run ≤ 2e-3 | 6.627 (to 1e-4) |
| | nonorth_c0.2: 35.46 ± 0.32 (35.14) | per run ≤ 6e-4 | 35.14 |
| | randbin_302: 13.04 ± 0.11 (13.09) | 12.93 ± 0.07 | – |
| | mps8 seed 0: 17.61 / 2.710 | 17.72 / 2.715 (pred 17.41 / 2.698) | – |
| Binary/HT escape, power-law exponent p | 1.005–1.011, log law never preferred | binary 1.002–1.003, HT 1.015–1.022 (shorter ε ranges); log 0/8 | binary local p 1.007 → 1.001 |
| Random init: plateau λ_min | mps4 σ-independent; binary4 ∝ σ | mps4 −0.151 at every σ; binary4 −1.7e-4 … −0.16 for σ from 1e-4 to 0.063; gaps ≤ 0.8% | – |
| Star Hessian = −a^(K−2)(ρ(S) − s_q) | to 1e-15 | all 20 rows identical | to 1e-15; sign flips between w = 1.4138 and 1.4146 (w* = √2) |
| Star threshold | 3/3 vs 0/3 | 8/8 outcomes identical (stop times identical) | – |
| Star rotation slope (mps4, w = 1.6) | 2.686 ± 0.005 (2.691) | 2.679 (seed 0) | – |
| ht4 stall | a_q ≈ 0.67; t ∝ 1/ε | a_q = 0.6732; t = 2806 / 27963 at ε = 1e-3 / 1e-4 (committed 2806 / 27963) | blended rank-1 maximum a_q = 0.6732; no negative eigenvalue on ht4, strict on mps4 |
| Star depth (random init) | σ^(5.7 ± 0.3) | min a_q identical to ≤ 3e-4 | – |
| Correlated increments per halving of σ | 2.131 ± 0.013 (2.128); 14.89 ± 0.22 (14.39) | per-run gaps ≤ 2e-8; statistic identical: 2.1307 ± 0.0129, 14.887 ± 0.224 | – |
| 4a, c = 0.45 | 5.76 ± 0.04 (5.68) | t₁ ≤ 7e-6, gaps ≤ 0.75%; A = 5.72 ± 0.03 (3 seeds); plateau λ_min −0.1509 | – |
| 4b: distance to critical point vs to nearest SVD truncation | 104: 0.0002 vs 0.0015; 102: 0.0017 vs 0.0047 | 0.00021–0.00024 vs 0.0015; 0.0016 vs 0.0047–0.0060 | – |

**Discrepancies and likely causes**

1. **randbin_302 per-run escape times do not reproduce.**
   - On mps4 the difference is a constant offset per seed: +3, −29 and −16 time units, up to 50%. On ht4 it is up to a factor of 2.9.
   - Slopes agree, and all other teachers agree to ≤ 7e-3.
   - Cause (verified): `constrained_minima` sorts the 12 L-BFGS minimizer results by loss. Gauge-equivalent copies of the saddle (same tensor W, different core sign patterns) have losses that differ by 1 ulp (0.048540957081886119 vs …126). So roundoff decides which copy is used, and with it how a given noise seed projects onto the unstable directions.
   - Only intercepts and prefactors are affected, not the laws. nonorth8_c0.7 shows a smaller version of the same thing (offsets ≤ 1 time unit).
2. **Star "forgetting depth" claim is overstated.** In random-init super-critical runs q is never learned.
   - a_q peaks at only 0.55–0.99 (mps4 median about 0.73; rerun seed 2: 0.58–0.65).
   - The time spent at the "q learned" loss level is 0.01–0.06, against 24–538 at the "q forgotten" level.
   - r is learned about 5 time units after a_q crosses 1/2.
   - So the flow passes near the blended rank-1 point (a_q = 0.67, a strict saddle on mps4) and goes straight to "r and r′ learned". It never visits the θ_q saddle.
   - "Super-critical stars forget q to σ^5.7" therefore measures the decay of a partially aligned direction, not the forgetting of a learned component through the predicted strict saddle (P8).
   - The 5.7 ± 0.3 is a standard error over per-seed exponents that range from 4.5 to 6.8 (one decade of σ, 4–6 points each).
3. **Correlated-input increments are not normalized for σ spacing.**
   - The σ grid is not exactly factor-2 spaced (ratios 1.94, 2.03, 1.98, 2, 2). `analyze_r2_corr.py` pools raw differences and quotes the standard error of the pooled values (n = 20), not per-seed.
   - Normalized per ln 2: ρ = 0.4 gives 2.131 for the smallest-σ pair and 2.145 ± 0.007 for all pairs; ρ = 0.2 gives 14.62 and 14.99 ± 0.33.
   - The increments are still drifting downward with σ, as theory_covariance 7.3c also says.
   - The conclusion stands (about +0.2% and +1.6% at the smallest σ), but the quoted 0.1% agreement partly reflects the grid spacing. Seed 1 at ρ = 0.2 drifts the opposite way.
4. **Saddle-type labels in the committed table don't match the notes.**
   - The committed `r2_saddle_hessian.csv` labels nonorth8_c0.45 (λ = −0.0048) and misaligned8_20 MPS as "non-strict", because the library's relative threshold is 1e-3·max|λ|.
   - The notes say "strict on every MPS student". By sign they are strict.
   - "Log law in 65/65" counts misaligned8_20, whose slope is 24× the prediction (flagged as degenerate in the notes).

**Bugs and reproducibility issues**

1. **Windows / spawn bug.**
   - `r2_saddle_runs.py` sets `T_CAP`, and `r2_star.py` sets `STOP_ROTATE`, via `globals()` in `__main__`. Spawned workers don't see these values.
   - Verified with `spawn_test.py`: the parent sets T_CAP = 1e11, the worker sees 1e7.
   - On Windows, `r2_saddle_ext.yaml` would silently stop the slow binary/HT runs at t ≈ 1e7 (committed escape times go up to 6e8) and record them as not escaped.
   - The committed Linux (fork) results are unaffected.
2. **Event time is grid-limited.** It is the first recorded row above threshold, with no interpolation, giving 0.3–0.8% scatter. This is consistent with the agent's ≤ 0.5% resolution check and harmless for the slopes.
3. **Thread count:** `r2_star.py hessian` calls `torch.set_num_threads(8)`, against the 1-thread rule in CLAUDE.md.

**README instructions:** They work as written when the environment variables are set. Nothing was written into `experiments/` or into a `results/` folder in the code copy. Gaps:
- The run order omits `r2_star.py threshold` (the source of the "threshold to 3e-4" result) and `complete_ht`.
- `run_dense_queue.sh` logs into `agent-runs/` unless the undocumented `DATA_TEACHER_LOGS` is set.
- `r2_chain*.sh` hard-code `--procs 28` and the results path.
- `recompute_saddle_hessians.py` hard-codes `Pool(4)`.
- The README doesn't warn about issue 1 (spawn platforms).

**Not rerun:** 4a at c = 0.2 (36.5 ± 0.8), the ε = 1e-6 ht4 star runs, misaligned8_45 and nonorth8_c0.45, and theory_covariance's predicted increment values (2.128, 14.39) themselves.

**Files:**
- Reruns, comparisons and scripts: `C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_repro/`
  - comparisons: `results/compare_*.csv`
  - independent checks: `indep_saddle.py`, `indep_star.py`
  - `spawn_test.py`
  - reduced configs: `configs/`, `code/configs/*_rev.yaml`
- Detailed notes (parts 1–7): `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_data_teacher/r2_repro/`

No repo files were edited and nothing was committed.


## B. Claims audit

**Claims audit: data-teacher round 2.** Raw outputs weren't available, so every number below is recomputed from the committed CSVs. I also ran three small checks with the agent's own code: Hessian spectra at the exact MPS-8 saddles, the teacher bond spectra for the 4c teachers, and two saddle-start reruns of `misaligned8_20` on mps8 (ε = 1e-2 and 1e-3). No repo files edited, nothing committed, all processes finished.

Terms used below:
- **MPS:** a chain-shaped tree tensor network in which every core carries an input leg.
- **binary / HT:** trees whose internal cores have no input leg.
- **Strict saddle:** the Hessian has a negative eigenvalue −γ, so the escape takes about (1/γ)·ln(1/σ).
- **Higher-order saddle:** no negative curvature, so the escape takes a power of 1/σ (σ⁻¹ here).
- **Head start (κ):** the escape direction already grows during the first escape, so with random init theory R6 predicts a slope (1−κ)/γ, which is always ≤ 1/γ.

## Ranked issues

**HIGH 1. The 4c / P11 evidence is overcounted.**
- **Claim:** "P11 holds 125/128 runs over 6 discriminating teachers" (R2.0 and REPORT).
- **Evidence:**
  - `analyze_cross_v2.py` marks a teacher as discriminating if the root cut's second singular value s₂ beats the weakest cut. P11 only disagrees with a weight-order rule if the root's s₂ exceeds both bottom bonds of one half.
  - That holds only for teachers 209 and 212, where the root is the largest cut.
  - Teachers 203, 204 and 218 each have, on every half, a bottom bond above the root, by margins of 0.3–1%.
  - Teacher 203 at σ = 0.4 is the same 8 runs in both sweeps (event times agree to 4 digits).
  - Correct counts: 46/48 runs on 2 discriminating teachers, and 117/120 distinct runs on 5 teachers.
- **Fix:** restate the counts, correct the flag, and add teachers whose root s₂ exceeds every bottom bond.

**HIGH 2. The reported slope deviations come from the fit range, and one global fit is above a hard upper bound.**
- **Cause:** the global two-parameter log-law fits over 3 decades include pre-asymptotic large-σ points.
- **Check used:** per-seed slopes from the 3 smallest σ, which is the protocol in theory_teachers §9.
- **Random init, mps4:** all 6 teachers are within 1.3% of R6, with 5 seeds each.
  - c = 0.2: 33.95 ± 0.18 against the prediction 34.14 (the agent reports 36.3, +6%).
  - randbin_303 and randmps_301 keep 5 seeds instead of 1 and 3.
  - misaligned_20: the global value 58.85 ± 0.86 is above the bound 1/|λ_min| = 52.94, which no head start allows. The local slope is 52.46 ± 0.10.
- **4a:** c = 0.2 gives 34.40 ± 0.20 (+0.8%, not +7%); c = 0.45 gives 5.686 ± 0.016.
- **Step 2, θ = 5°:** local 194.9 ± 1.2 against the global 251 (the bound is 193).
- **Saddle start, ε ≤ 1e-4:** nonorth8_c0.45 is −0.05% off. The REPORT statement "still +11% at ε = 1e-8, slow crossover" is false; the agent's own `r2_saddle_collapse.png` shows slope 1 for ε < 1e-3.
- **Binary/HT:** the local exponent p over ε 1e-5 to 1e-3 is 0.9998–1.0009 in all 24 cases. The global values 1.003–1.013 sit 5–15 standard errors above 1.
- **Fix:** report the asymptotic slopes, drop the crossover caveat, and state agreement as ≤ 1–2%. The conclusions get stronger, not weaker.

**MEDIUM 3. `misaligned8_20` is not "degenerate". The event rule fails for it, and it is still counted in "65/65".**
- **Rerun evidence:**
  - The strict escape grows at the predicted rate: 4431–4692 time units per decade of ε, against 4547 predicted.
  - It ends at an intermediate saddle where s₂ is about 3.7e-4 on every bond (0.55% of the teacher's value) and the loss does not change.
  - That level is below the f = 0.05 onset threshold (about 0.0031), so the agent's t_esc (about 2e5) times the next, slower escape.
- **Consequences:**
  - The "log law" counted for this teacher is a fit to the wrong event, with a slope 24× the prediction. The clean tally is 60/60 seeds over 12 teachers.
  - "p = 0.999–1.013 in 26 cases" is really 24 cases. For this teacher binary8 gives p = 1.079 from 2 seeds with 12/29 runs that never escaped, and ht8 escaped in only 3/30 runs.
- **Fix:** re-time the escape against the intermediate level, and note that the f-rule fails whenever an intermediate saddle's s₂ is below f·s₂*.

**MEDIUM 4. The star forgetting-depth law (min a_q ~ σ^(5.7 ± 0.3)) pools two kinds of trajectories and selects on the outcome.**
- **Two classes:** in 5 of the 8 fitted mps4 seeds, a_q (overlap with q) peaks at only 0.55–0.68, meaning the flow goes to the blended best rank-1 point and q is never learned.
  - Those seeds give an exponent of 5.35 ± 0.45.
  - Seeds that pass near q (peaks 0.78–0.91) give 6.38 ± 0.29.
- **Selection on the outcome:** the window 1e-10 < min a_q < 0.5 drops seeds 2 and 7 at σ ≤ 0.016, where min a_q is negative (a_q crosses zero). The quoted "deepest 3e-11" is one of these negative values.
- **No single exponent:** the local exponent drifts within a seed (seed 0 goes from 5.3 to 3.7). Fits use 1 decade of σ and 4–6 points.
- **Relearning time:** the "~σ^-1.4" relearning time is not in any CSV. Per seed it is 1.78 ± 0.15, or 1.5 vs 2.25 by class.
- **Fix:** classify runs by peak a_q, and describe the depth as "steep" rather than as a power law.

**MEDIUM 5. Some claims are broader than the evidence.**
- "Binary and HT students keep σ⁻¹" and "later plateaus on MPS become log-time" rest on timing only the escape from the rank-one saddle (N = 4 and 8, d = 2).
- theory_saddles §5.6 says strict two-core routes can appear on binary/HT at higher ranks.
- HT was never run with correlated inputs, and random-init runs are N = 4 only.
- **Fix:** say "escape from the rank-one saddle".

**MEDIUM 6. Correlated inputs: the numbers hold to about 2%, but the evidence is weaker than presented.**
- R2.3 repeats theory_covariance check N14: same σ grid (0.05·2^-j), d = 3, same events. That note already printed increments of 2.12–2.19 and 17.4 falling to 14.7.
- Only 4 seeds remain after seed 2 is excluded.
- The quoted ±0.013 and ±0.22 pool 20 increments as if independent.
- The σ values are rounded (ratios 1.94, 2.03, 1.98), which biases individual increments by −4.6% / +2.3%.
- Per-seed regression on log₂(1/σ):
  - ρ = 0.4: 2.144 ± 0.007 with all σ, 2.133 ± 0.003 with the 3 smallest (predicted 2.128).
  - ρ = 0.2: 14.97 ± 0.30 and 14.67 ± 0.06 (predicted 14.39; +2%, not yet asymptotic).
- Predictions for ρ = 0.1 and 0.3 can be computed from the theory but weren't.

**LOW**

7. **The committed Hessian CSV contradicts the text.**
   - `r2_saddle_hessian.csv` labels the mps8 saddles of nonorth8_c0.45 and misaligned8_20 "non-strict" with n_negative = 0, and gives n_negative = 2 for randbin_302 on mps4.
   - The cause is the library's fallback threshold of 1e-3·λ_max.
   - I recomputed 5 / 17 / 17 negative eigenvalues, as the notes say.

8. **Two deviations come from comparing against the wrong edge.**
   - randbin8_303 (+3.6%) and randmps8_303 (+4.2%): λ_min sits on edge (6,7), which appears once.
   - The event actually fires on edge (4,5) (γ = 0.18267, three times, 4/5 seeds) or edge (1,2) (γ = 0.16178, three times, 5/5 seeds).
   - 1/γ for the realized edge is 5.474 / 6.181, against measured 5.46 / 6.20.

9. **4b fits mix escape routes, and the P1 comparison is selective.**
   - The first edge switches with σ inside seeds 3–4 of teacher 102 and seeds 0–3 of teacher 104, so those per-seed fits mix routes.
   - Seeds that stay on one edge give 3.04–3.09 against 3.01, and 5.22–5.27 against 5.175.
   - The P1 exponent depends on the teacher, and only teacher 104 is compared.

10. **Star rotation law: only w = 1.6 compared, and the ht4 threshold bracket is incomplete.**
    - The slope was compared to 1/|λ| only at w = 1.6, with 4 ε values.
    - From the CSV it also matches to ≤ 0.2% at w = 1.48 on all trees and at every super-critical w on binary4.
    - On mps4 at w = 1.42 and 1.44 it is +50%. On ht4 at those w the event follows ε⁻¹, so it times the polynomial second stage.
    - The ht4 threshold at ε = 1e-6 is missing w = 1.42.

11. **Stale or loose statements.**
    - Summary row 2 quotes "55.5 ± 1.1", which is the base-range fit; the R2.1 value is 53.90 ± 0.41.
    - The f-sensitivity of MPS slopes reaches 1.8%, not "< 1.5%".
    - The escape degree is called "fixed" but is still 9–29 standard errors below 2 − 2/K.
    - The float-floor scan does not cover the star runs or the correlated-input runs.
    - The Hessian at θ_q repeats theory_saddles check C11, and the binary4 star is the known deep-linear-network case.

## Claims that hold
- **Exact rank-1 saddle type:** negative curvature on all 13 MPS cases. On all 26 binary/HT cases there is nothing below the residual-gradient noise of about 1e-9.
- **Saddle start, MPS:** log law with slope 1/γ of the realized edge to ≤ 0.4% (12 teachers, ε ≤ 1e-4, 5 seeds).
- **Saddle start, binary/HT:** ε⁻¹ with p = 1.000 ± 0.001 (24 cases).
- **Random init, mps4:** the second escape follows R6 including the head start (6 teachers, ≤ 1.3%; e.g. 1.04 against 2.78 without head start).
- **Random init, binary4:** σ⁻¹ (local p 1.001–1.005).
- **Plateau λ_min:** tends to the exact-saddle value on mps4, and scales as σ^1.0 on binary4.
- **First escape:** σ^-2.00 on mps4 and σ^-1.01 on binary4.
- **Star Hessian:** matches −a^(K−2)(ρ − s_q) to 3e-15.
- **Star threshold:** bracketed to ±3e-4 on mps4 and binary4.
- **Star rotation slope:** within 0.2% for w ≥ 1.48.
- **ht4 star:** the second stage scales as 1/ε (5 seeds).
- **Sub-critical stars:** q is never forgotten.
- **Correlated inputs:** MPS-4 log law with the two-stage increments (0.2% at ρ = 0.4, 2% at ρ = 0.2); binary-4 stays σ⁻¹.
- **4a:** asymptotic slopes match R6 within 1%; the odeco case c = 0 gives σ⁻².
- **4b:** the plateau is separated from all three SVD-type truncations for teachers 102, 104, 105.
- **4c:** P11 holds in 46/48 discriminating runs; "sibling pair first" does not generalize.
- **Float-floor scan:** flags only the odeco baselines at σ ≤ 3e-3.

## Files
Everything is in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_data_teacher/r2_audit/`:
- `00_README.md`
- `01_fit_range.md`
- `02_saddle_start_counts.md`
- `03_p11_4b.md`
- `04_stars.md`
- `05_corr.md`
- `06_misaligned8_20.md`
- `scripts/`
- the two rerun logs: `misaligned8_20_eps0.01.log` and `misaligned8_20_eps0.001.log`

Rerun outputs are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/`.