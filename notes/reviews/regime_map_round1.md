# Review of regime-map (round1) — 2026-10-10

Two independent local reviewers: (A) partial reproduction from the committed code plus an independent reimplementation, (B) adversarial audit of claims vs data. Re-run outputs in `data/review_<folder>/` (local, gitignored).

## A. Reproduction

# Reproduction review: regime-map (Q5, map of training regimes)

**Bottom line:** claims 1, 2, 3, 5 and 6 reproduce. With the committed code the per-seed values match exactly, and my own separate implementation matches them quantitatively. Claim 4 holds up to K = 7. At K ≥ 8 the reported boundaries are brackets limited by a realizability floor, not measured crossings, and the √d limit is an extrapolation. I found two medium-severity bugs.

**Terms used below.** A tree tensor network (TTN) builds the weight tensor W (one index per input site) from small "core" tensors joined by internal "bond" indices of size χ.
- **K** is the number of cores and **σ** is the starting scale of each core.
- **Saddle-to-saddle (S2S):** the loss falls in separate steps, one teacher component at a time. The separation ratio ρ (gap between steps divided by step width) is above 3.
- **Rich:** the model's features change, but the steps overlap.
- **Lazy:** training behaves like a linear model around the starting point. The tangent kernel changes by less than 10% (δK < 0.1).
- **Cut dimension:** the largest bond size that is useful on an edge.
- **Centred:** the model is trained on W − W₀, so the starting function is removed.

## What I ran (about 30 min wall time, at most 4 processes)

- **Committed code:** 177 gradient-flow runs (`run_regime.run_point` on the committed YAML configs) plus `lazy_predict` for 6 (χ, seed) pairs.
  - `chi_b4`: the full 22-σ grids for (χ = 64, seed 0) and (χ = 128, seed 1).
  - `sigma_chi_b4`: χ = 36 seed 0 (all 19 σ values), plus targeted points.
  - The imbalance check point: all 54 runs.
  - Large χ: 20 runs.
  - K sweep: MPS chain with N = 5 sites (K = 5), seed 0 (14 runs), and MPS N = 8 (K = 8) at 6 points.
- **Independent check:** my own numpy/scipy implementation of the binary N = 4 network, W = A·R·Bᵀ. It uses its own random number streams, its own integrator with exact event detection, and a hand-derived tangent kernel (checked against finite differences to 5e-11). Total 1,860 runs on my own 6–12 seeds, with boundaries found by bisection.

| claim | committed | rerun (their code) | independent |
|---|---|---|---|
| (1) `chi_b4` per-seed row (64, s0): σ_c, predicted σ_c, lazy boundary, ρ slope, number of S2S runs | 0.03393 / 0.03898 / 0.75092 / −1.02586 / 7 | identical | — |
| (1) same row for (128, s1) | 0.01758 / 0.02054 / 0.54448 / −1.02347 / 5 | identical | — |
| (1) `sigma_chi_b4` χ = 36, s0: S2S boundary at ρ = 2/3/5; lazy boundary at δK = 0.05/0.1/0.2 | 0.073302 / 0.050614 / 0.031192; 1.06123 / 0.92237 / 0.79461 | agrees to about 1e-5 relative | — |
| (1) ρ exponent per seed | −1.02 to −1.03 | −1.026, −1.023 | −1.011 to −1.022 |
| (1) step widths deep in S2S (s = 3, 2, 1) | 0.466 / 0.798 / 2.010 | — | 0.464–0.473 / 0.798 / 2.009 |
| (1) lazy boundary × √χ at χ = 32 / 64 / 128 (theory P5: 6.50) | 6.29 / 6.18 / 6.10 | — | 6.41 / 6.37 / 6.09 |
| (2) S2S boundary × √χ | 0.29 / 0.31 / 0.27 | — | 0.32 / 0.29 / 0.32 |
| (2) slope vs χ: lazy, S2S | −0.522, −0.524 | — | −0.54, −0.50 to −0.55 |
| (2) rich window (lazy ÷ S2S boundary) | about 21 | — | 19–22 |
| (3) final loss ÷ Eckart–Young floor (χ < 9) | 1 to 1e-6 | 1.0000000000 | 1 to 1e-15 (12/12) |
| (4) K = 5 seed 0: σ_c, prediction, slope | 0.861 ok / 1.879 / −2.5320 | identical (−2.5319) | — |
| (5) δK slope vs χ (σ = 1.5): SP / NTK / µP-TTN | −3.09 / −0.84 / +0.03 | −3.23 / −0.82 / +0.07 (partial) | −3.06±0.07 / −0.78±0.05 / +0.07±0.04 |
| (5) SP slopes: parameter change / root core / leaf cores | −1.63 / −1.02 / −2.96 | — | −1.66 / −1.09 / −2.98 |
| (6) median δK, λ = 0.01 … 100 (χ = 16, α = 3.375) | 1e-4, 7e-4, .0069, .036, .167, .45, .53, .50, .50 | 8.35e-5, 7.35e-4, .00689, .0363, .167, .452, .533, .502, .497; labels identical | 6.1e-5, 5.5e-4, .0057, .036, .158, .41, .54, .50, .49 |
| (6) χ = 24 map | S2S widest at λ = √χ; light root becomes lazy about 100× earlier in α | — | median ρ peaks at λ/√χ = 1; lazy boundary moves about 50–80× in α |
| lazy predictions (residual, κ) | `lazy_pred_b4.csv` | identical | — |

## Bugs and discrepancies

1. **Brief overlap bumps are taken as the learning time (medium).** In `run_regime.py:192` (`refine_events`), a component's overlap can briefly exceed 1/2 while a different component is being learned. The code then keeps that first, temporary crossing as the event time.
   - This contradicts `notes/diagnostics.md`, which says to use the last upward crossing.
   - The bump is only detected on the coarse log grid (line 236), so the label flips unpredictably with σ.
   - Example, χ = 12 seed 4: a₂ peaks at 0.556 at t ≈ 54.5. The run is labelled rich (ρ = 0.61) at σ = 0.010, 0.0149 and 0.0158, but S2S (ρ = 13–41) at σ = 0.004–0.0107. Under the documented rule every one of them is S2S (ρ = 13.8 at σ = 0.010).
   - Effects:
     - It causes the 5.9× outlier in `chi_b4` (boundary 0.0126 against a prediction of 0.075).
     - In `sigma_chi_b4` the same seed gets NaN and is silently dropped, so the χ = 12 value of 0.088 averages only 5 seeds.
     - χ = 4 seed 3 moves from 0.029 to about 0.05.
   - It is rare: 0 of 60 independent (χ, seed) sweeps had a bump above 1/2, and the medians and slopes are essentially unaffected.

2. **Unresolved K-sweep boundaries are counted as resolved (medium, claim 4).**
   - `plots.py:665` counts only the status `"unresolved"` and misses `"unresolved+stall_below"`.
   - `summary.png` (`plots.py:737`) labels these points "6/6 resolved".
   - The per-seed CSV shows the true unresolved counts are 5/6 at K = 8, 5/6 at K = 10, 6/6 at K = 12 and 5/6 at K = 15. `sigma_c_vs_K.csv` reports 0, 2, 0 and 0. The REPORT flags only K = 12 and 15.
   - My K = 8 seed 0 reruns:
     - S2S at σ = 0.67 and 0.88.
     - Stalled at 0.51: component 2 not learned by T = 3.3e5.
     - Stalled at 1.16: both components learned, but the loss sticks at 0.018, which is the centred floor at χ = 4.
   - So the reported "1.01" is the midpoint between the last S2S run and the first floor-stalled run. It is a bracket, not an S2S/rich crossing.

3. **Claim 4 is overstated.**
   - From the committed medians, the per-component amplitude u₀c = σ_c^K/√D is 0.015, 0.016, 0.034, 0.0085, 0.069, 0.010, 0.006 for K = 4–12. That is an 11× spread, against 19× for σ^K.
   - The K = 15 bracket allows anything from 4e-6 to 0.4.
   - The "within 20% in σ" test is weak because taking the 1/K root compresses differences.
   - The data cannot tell σ_c → 1 apart from σ_c → √d. "√d, not 1" should be marked as a conjecture.

4. **Arithmetic slip.** The first imbalance slope in the notes, 2.18, is the log-ratio without dividing by ln 3. The correct value is 1.98; the other three slopes are right.

5. **Minor stale numbers.** Refitting the committed CSV gives an S2S slope of −0.528 ± 0.053 (the notes say −0.524 ± 0.045). The notes' σ_c√χ values also differ slightly from the CSV. The conclusions are unchanged.

6. **Labelling nuance.** Below the cut dimension, a run that ends exactly on the floor still counts as "converged" when the floor is below 1e-3. My χ = 6, σ = 1.0 runs end with final loss = floor = 1.6e-4 to 8.8e-4 and are labelled rich. This should be stated in the notes.

## README run instructions

- **Step 1 fails on Windows:** `run_regime.py:317` uses the Unix-only `fork` start method, and the default raw-output path is `/root/outputs`. A small driver using `spawn` that calls `run_point` reproduces the committed results.
- **Step 2 as written raises FileExistsError** because the results already exist. An output folder can only be given as positional arguments 8–10; with that it works and gives identical numbers. `pilots/lazy_pred_b8.py` has no output option and silently overwrites a committed CSV.
- **Step 3:** `--out` works, but the raw-data folder is hard-coded (`analysis.py:16`).

## Work left for regime-map

None of these fixes need new compute sweeps:
- Fix bug 1 and recompute the labels from the saved trajectory data.
- Fix the unresolved counting, and mark K = 8 and 10 as brackets in the REPORT and notes.
- Restate the u₀c / √d claim as a conjecture.
- Correct the 2.18 slope.
- Add a `spawn`-based runner and an option for the raw-data folder.

A resolved boundary at K ≥ 8 would need full or spare bond channels and longer time budgets.

Files are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_regime_map/` (`indep/`, `lazy/`, one subfolder per experiment). Working notes are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_regime_map/` (`notes_part1.md`; `repro_rev/notes_part2-4.md`).


## B. Claims audit

**Regime-map claims audit (adversarial)**

**Verdict:** The N = 4 lazy-side results and the large-χ results hold up, and the lazy-boundary theory fits better than the report says. On the S2S side, the per-seed numbers are affected by an event-rule bug that I confirmed by rerunning. Claim 4 (σ_c → √d, "H4 refined") is not supported and should be treated as a conjecture.

**Ranked issues**

1. **HIGH: Claim 4, the K-dependence and σ_c → √d.**
   - **Unfair law test.** I fitted both one-parameter laws, "σ^K constant" and "σ^K/√D constant", to the per-K medians with bootstrap errors. They fit equally well: χ² 8.2 vs 9.3 for K = 4–12 (6 dof), and 0.9 vs 0.7 for the resolved K = 4–7. Because σ_c = const^(1/K), a 20% error in σ is a factor 1.2^K ≈ 4–9 in the variable. The measured u0c actually ranges from 0.006 to 0.069.
   - **K ≥ 8 is unresolved by the agent's own rule.** 5/6 seeds at K = 8, 5/6 at K = 10, 6/6 at K = 12 and 5/6 at K = 15 are unresolved, mostly with stalled runs below the S2S block. The S2S window is only 1–2 runs wide (median n_s2s = 2 at K = 8 and 12).
   - **Reporting bugs.** The REPORT stars only K = 12 and 15. The n_unresolved column of `sigma_c_vs_K.csv` counts only the exact string "unresolved", so it shows 0 where MPS8 has 5 and MPS12 has 6. `summary.png` labels these points "6/6 resolved" while drawing them as open (unresolved) symbols.
   - **One d only.** Only d = 2 was run, so √d cannot be distinguished from 1.
   - **Status tag.** "Derived from R9" is wrong: R9 is marked heuristic/conjecture in the theory note.
   - **Reruns (MPS8, seed 0):**
     - The stalled run just above the S2S window (σ = 1.16) is rich-like (ρ₁ = 2.3). It ends at l_end = 0.018, inside the floor bracket [0.012, 0.021]. So the floor hides a genuine S2S/rich transition there.
     - The stalled run at σ = 0.1 stopped at max_steps = 2e4 (t = 1.4e9, below t_max = 5.7e9). Its second route was at 1.6e-11.
   - **Fix:** use the ρ-based per-seed boundary (`seed_rho_boundary`) so floor-stalled runs are classified by ρ. Report n and status per K, compare the two laws with errors, and add d = 3 with spare channels before claiming √d. Downgrade the √d limit to conjecture and fix the status-counting bug.

2. **MEDIUM-HIGH: The event rule breaks `notes/diagnostics.md` when a transient exceeds 1/2.**
   - **The bug.** `run_regime.refine_events` keeps the first ½-crossing when it comes before the final rise. The diagnostics require the last upward crossing.
   - **Confirmed on the chi_b4 outlier (χ = 12, seed 4).** Its reported boundary is 0.0126 against a prediction of 0.075.
     - At σ = 0.0149, 0.0207 and 0.0288, a₂ bumps to 0.68–0.83 and falls back.
     - The pipeline gives ρ₁ = 0.75, 0.75, 0.71 (labelled rich).
     - The last-crossing rule gives 8.6, 5.9, 3.5 (S2S), so this seed's real boundary is about 0.03.
   - **Silent drop.** The same seed is excluded from table 2.1 at χ = 12 (n = 5, not 6). Including it gives about 0.074 instead of 0.088.
   - **χ = 4, seed 3:** at σ = 0.0398, ρ₁ goes from 0.30 to 4.6 under the correct rule, moving the boundary from 0.029 to about 0.055.
   - **Checked controls.** Three other low seeds have no bump and their low boundaries are genuine: χ = 4 seed 2, χ = 3 seed 3, and d = 2 binary4 seed 0.
   - **Unsupported explanation.** The note blames pred/measured = 1.23 and 1.50 at χ = 3, 4 on the realizability floor, but the floor sits at least 3× higher in σ.
   - **Fix:** apply the last-crossing rule and flag bumps, recompute the per-seed tables, and report n per cell.

3. **MEDIUM: The kernel change δ_K is defined differently from the theory (claims 1, 5, 6).**
   - **The mismatch.** δ_K is computed on 128 probe inputs. The constants in P5/P6/P7/P16/P20 refer to the tensor-space kernel; theory §7 asks for that kernel when N ≤ 8.
   - **Rerun at each seed's measured lazy boundary (probe δ_K = 0.10):**

     | χ | tensor δ_K (seeds 0/1) | tensor-kernel boundary | P5 bracket |
     |---|---|---|---|
     | 32 | 0.18 / 0.12 | 1.09 / 1.12 | [1.10, 1.19] |
     | 64 | 0.16 / 0.13 | 0.82 / 0.77 | [0.81, 0.83] |
     | 128 | 0.19 / 0.13 | 0.56 / 0.57 | 0.57–0.58 |

   - **Effect.** The probe/tensor ratio is 0.54–0.82 for χ ≥ 32 (0.41 and 1.13 at χ = 16). With the tensor kernel, P5 holds to 0–5%. The stated explanation, "5–8% low, as in the theory note's GF checks", is wrong: those checks used the tensor kernel.
   - **Likely the same cause, unreported:** muP δ_K of 4–5 vs P16's 6.2–6.3, and P20 saturation of 0.50 vs 0.6–0.7.
   - **Weak test.** σ_lazy scales as c₂^(1/6), so a few percent in σ hides 20–30% in the kernel coefficient. P5 also fails at χ ≤ 12 (+39% at χ = 9, +15% at χ = 12).
   - **Fix:** recompute δ_K from the saved θ_end with the tensor kernel, and compare δ_K(x) with c₂x² directly.

4. **MEDIUM: Claim 1, "every boundary predicted quantitatively."**
   - The S2S "prediction" calibrates each seed on its own deepest run. That tests ρ ∝ σ⁻¹ rather than being an a-priori prediction, and the χ-dependence is not derived.
   - The 0.97–1.08 figures are means over seeds. Individual seeds range 0.86–1.23, and the 6× failure in issue 2 was excluded.
   - The stall onset at deficient bonds is 1.6–2.2× above P4's estimate.

5. **MEDIUM: Claim 2 (collapse in σ√χ): provenance, statistics and scope.**
   - **Mislabelled numbers.** "σ_c√χ = 0.29, 0.29, 0.31, 0.30, 0.27" are medians of the aligned-mode predictions (0.293, 0.292, 0.306, 0.302, 0.267). The measured medians are 0.27, 0.33, 0.27, 0.33, 0.28, quantized by the 1.39 sigma grid.
   - **Mixed fit ranges.** "−1/3 rejected at 4.2 SE, χ ≥ 32" uses the χ ≥ 16 fit. At χ ≥ 32 it is 2.8 SE.
   - **Seed-level fits.** Seeds share the same teacher across χ. Fitting per seed gives a slope of −0.53 ± 0.07, which is 2.7–2.9 SE from −1/3.
   - **Lazy slope.** −0.543 ± 0.012 for χ ≥ 16 (3.6 SE from −1/2). σ_lazy√χ drifts from 6.83 down to 6.10 across the χ range.
   - **Range and scope.** The fits span 0.6–0.9 decades, below the one-decade standard. Results are for N = 4 with fan-in init only, and the REPORT summary drops that scope.
   - **Novelty.** σ²χ is the TTN analogue of known width scales (σ²w in Tu et al. 2024; α√h in Geiger et al. 2020). It should be cited, not tagged "new".

6. **MEDIUM-LOW: Claim 6 (imbalance).**
   - The P20 panel of `phase_imbalance_b4.png` is empty: alpha is rounded to "3.38" and then compared with 3.375. The quoted δ_K(λ) values therefore do not come from committed code.
   - The local slopes are two-point differences of medians with no errors.
   - "S2S widest at λ = √χ" rests on one cell with 3/6 seeds, on a grid with one-decade steps in α.
   - A heavy root at χ = 24 is lazy for α ≥ 10, the same as at balance. It does not make training lazier, but it does not prevent laziness either.

7. **LOW-MEDIUM: Claim 5 (large χ).**
   - Fits use only 4 values of χ (16–128).
   - The NTK slope is −0.84 ± 0.06 (2.5 SE from −1), and −0.68 at σ = 1.0 (5.5 SE). The rationale "approaching −1 as c₂x² fades" contradicts R12, where both orders scale as χ⁻¹. The bias comes from δ_K being O(1) at χ = 16–32.
   - SP: δθ is 3.3 SE off and the leaf exponent 6.5 SE off. muP: δθ slope −0.064 ± 0.018. N = 4 only.

8. **LOW: The two-copy section (2.x) quotes superseded scipy numbers.**
   - The library CSVs show S2S labels at σ ≤ 0.016 for 4–5/6 seeds at χ = 18–36, with ρ never above 5. So the sweep-level scaling check fails; the runs are not "rich at every σ".
   - The lazy boundaries are 0.82 vs 1.05 at χ = 36, not 0.80 vs 1.13.

9. **LOW: Miscellaneous.**
   - The K = 7, d = 3 "S2S boundary 0.54" rests on one S2S run per seed for 5/6 seeds.
   - §1.3 should say "C grows sub-exponentially", not "at most exponentially"; as written it also conflicts with §2.3.
   - "Stalled" also covers runs cut off at max_steps (the σ = 0.1 rerun in issue 1), which the REPORT does not list as a cause.
   - Some numbers in the note differ from the committed CSVs: S2S slope −0.524 ± 0.045 vs −0.528 ± 0.053; kernel boundary at χ = 9 is 3.01 vs 3.06; threshold ratio 1.89 vs 1.97; plain K = 3 boundary "0.20" vs 0.17.
   - Dead-route singular values of 1e-16 to 1e-18 are below the SVD's resolution.

**Solid claims (within the stated scope)**
- N = 4: per-seed ρ ∝ σ^(−1.02 to −1.03) on the S2S side. Deep-S2S event widths match the aligned theory.
- The lazy boundary scales as χ^(−1/2) for χ ≥ 32, and its P5 prefactor holds to 0–5% once the tensor kernel is used.
- The two-copy/centred lazy-boundary ratio is about 0.78 (P6). The two-copy kernel change is second order (ratio 1362 vs 1372 predicted).
- Threshold sensitivity: changing ρ from 2 to 5 moves the S2S boundary by 2.14–2.34. Changing δ_K from 0.05 to 0.2 moves the lazy boundary by 1.30–1.33 for χ ≥ 48 (P7).
- Below the cut dimension the centred target cannot be represented, and centred runs end at the Eckart–Young floor (N = 4). A lazy label requires full bonds. K = 7 is lazy only with 2× full bonds, and the evidence there is weak.
- The S2S boundary falls roughly as χ^(−1/2) at N = 4, and the rich window spans about 20× in σ.
- At χ = teacher rank with K = 7–8, gauge balancing removes dead routes but routes still decay (P14). This depends on the time budget.
- Large χ: only muP-TTN keeps an O(1) kernel change. The SP and muP exponents match P16.
- A light root lowers the lazy boundary by about 100× in α (χ = 24), and δ_K ∝ λ² at small λ.

**Work left on this experiment** (mostly re-analysis, few new runs):
- Relabel with the transient rule.
- Recompute δ_K with the tensor kernel from the saved θ_end.
- Redo the K-sweep boundaries from ρ with a fair law comparison, plus d = 3 runs if √d is to be claimed.
- Fix the three table/plot bugs.
- Restate claims 1, 2, 4 and 6 with scope and status.

This needs the 1.2 GB of raw outputs, which are still on the stopped machine (per `stopped.json`). Copy them before deleting the machine.

**For the plain-language overview:** a TTN stores a large tensor as a tree of small tensors (cores) contracted along bonds of size χ. Started from a tiny random scale σ, it learns the target's components one at a time, with long pauses in between (saddle-to-saddle). Started large, its features barely move and it behaves like a fixed kernel method (lazy). In between, it learns everything at once (rich). This experiment maps those three regimes against σ, χ, the number of cores K, and root/rest imbalance.

**Files**
- Analysis and rerun scripts, with `audit1.out`–`audit3.out`: `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_regime_map/`
- Rerun outputs: `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_regime_map/` — `r1_kernel.jsonl`, `r2_seed.jsonl`, `r3_stall.jsonl`, `r4_transients.jsonl`, and the folders `r2_chi_b4_check/`, `r3_sigma_K_centred_d2/`, `r4_b4/`, `r4_k3/`