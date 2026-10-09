# Review of optim-samples round 1 (2026-10-09)

Two independent local reviewers: (A) partial reproduction from the committed code plus independent numpy/torch reimplementations, (B) adversarial audit of claims vs data. Re-run outputs: `data/review_optim_samples/` (local, gitignored).

## A. Reproduction

## Reproduction review: optim_samples (Q8), round 1

**Verdict:** every claim I checked reproduces. The agent's code reruns to roundoff on this machine, and my own independent code confirms the key numbers. Several statements are worded more strongly than the data support, and I found four analysis or portability bugs. None of them changes a headline conclusion.

**Where things are**
- Outputs: `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optim_samples/repro/` holds the code copy and reduced configs (`code/`), raw runs (`raw*/`), CSVs (`results*/`) and logs (`logs/`).
- Notes and my independent scripts (`indep_binary4.py`, `indep_sgd_ms.py`, `indep_samples.py`): `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optim_samples/` (`notes_1.md` … `notes_6.md`).
- About 20 minutes of wall time, at most 6 processes. I edited no repo files. `git status` shows edits to `ttn/diagnostics.py` and tests, but another session made those.

### What I reran (agent code, copied, reduced configs)
- **GD stability and EoS** (`gd_sweep.py`, `analyze_gd.py`, binary N=4): 116 GD runs and 20 GF references.
  - σ = 0.01–0.3, seeds 0–2, η/η_max ∈ {0.1, 0.9, 0.95, 1.05, 1.1, 1.2, 1.5, 2}.
  - New seeds 50–52 at σ = 0.03 and 0.1, η/η_max ∈ {0.1, 0.95, 1.05}.
  - σ = 0.003, seeds 0–1, at 0.1 η_max.
- **Event shift:** `predict_shift.py` on 13 GF references.
- **Centred model:** σ ∈ {1, 1.4, 2}, 2 seeds, 7 step sizes.
- **SGD first plateau:** `sgd_sweep.py` and `analyze_sgd.py` at σ = 0.1, inits 0–2, B ∈ {1, 4, 16, 64, 256}, η = 5e-4, with **new noise seeds** (120 runs).
- **Finite samples:** `samples_sweep.py` and `analyze_samples.py` on **new data/init seeds 5–7**.
  - χ=4 small init at n ∈ {128, 256, 384}; χ=16 at n ∈ {192, 256}.
  - The lazy limits, and the centred model at σ=1, χ=4, n=384.
- **P9/P10:** `p9_sgd.run` on 8 committed tasks with the same seeds, plus 10 tasks with new noise seeds.
- **SGD stability:** the `sgd_stability` functions at B = 1 and 16.

### Independent checks (my own numpy/torch TTN code, nothing from `experiments/`)
- **A.** Binary N=4 with my own random teacher basis and init RNG: sharpness, a GD scan from 0.9 to 2.5 η_max (2 σ × 5 seeds), and excess loss against my own 1D map.
- **B.** My own GD and GF, started from the agent's init and teacher arrays. Shift/η at 0.1, 0.05 and 0.025 η_max, extrapolated to η → 0 (Richardson), for 10 inits and 30 events.
- **C.** My own SGD second-moment operator: Θ from an exact multilinear Jacobian, my own fourth-moment operator F, symmetric basis.
- **D.** My own binary-8 TTN with its own teacher and data: lazy limit and small-init GD.

### Numbers: committed vs rerun vs independent
| claim | committed | rerun (agent code) | independent |
|---|---|---|---|
| 1 λ_max at balanced min | 12.98025 | 12.980246132766668 | 12.980246132766668 (JᵀJ); finite-difference Hessian within 5e-13; gauged minimum identical |
| 1 GD boundary, binary N=4, small init | all GF-like at ≤0.95; 25/25 oscillate at 1.05 | 50/50 GF-like at ≤0.95 (6 on new seeds); 18/18 oscillate at 1.05; event times and sharpness equal committed to ≤1e-12 | 30/30 converge at ≤0.99; 10/10 two-cycle at 1.01 |
| 2 shift = η c_e | e.g. 6.183 measured vs 6.196 predicted | c_e equals committed to 9e-11; new seeds: events 1–2 within 0.03–0.4%, event 3 within 0.6–5.5% at 0.1 η_max | Richardson vs c_e over 30 events: median 4e-5, max 2.6e-3. Includes σ=0.03 seed 1 event 2: −0.1471 vs −0.1467 (measured at 0.1 η_max: −0.050) |
| 3 excess loss above η_max (vs s₁² × 1D-map loss) | 0.1306 / 0.2472 / 0.4496 vs 0.1306 / 0.2479 / 0.4496 | 0.1307 / 0.2483 / 0.4495 | 0.1305 / 0.2482 / 0.4515 vs own map 0.1306 / 0.2479 / 0.4496; at 1.01: 0.0273 vs 0.0273 |
| 3 larger η | diverge 1.5–1.75; back to origin 25/25 at 2.0; diverge ≥2.5 | 1.5: 11 diverge, 1 back to origin; 2.0: 12/12 back to origin | 1.5: 8 + 2; 1.75: 10/10 diverge; 2.0: 10/10 back to origin; 2.5: 10/10 diverge |
| 4 centred-model GD limit = 2/λ_GF | predicted 0.88 / 0.57 / 0.19 | all 22 runs below the limit (η λ_GF/2 < 1) are GF-like; observed edges 0.7–0.9 / 0.5–0.7 / 0.1–0.2 vs 2/λ_GF = 0.85 / 0.53 / 0.18 (σ = 1 / 1.4 / 2) | – |
| 5 SGD first plateau | mean shift +0.19 ± 0.24 at B=1; spread exponent 0.497 ± 0.018 | per-init shifts from −0.13 ± 0.18 to +0.29 ± 0.15, none significantly negative; exponent 0.514 ± 0.036 | – |
| 5 later plateau (P9) | T_SGD/T_GF = 1.386 / 1.380 (B = 64 / 16); 3.06 at b0 = 0.003; c_γ = 34–45 | same seeds: identical. New noise seeds: 1.350 / 1.333; 2.29 and 2.36; c_γ ≈ 38–42 | – |
| 6 SGD mean-square limit η*_ms (B = 1, 2, 4, 8, 16, 64) | 0.00107 … 0.0161, 0.0528 | B = 1 and 16 identical to 1e-15 | 0.00107017, 0.00213113, 0.00422571, 0.00830772, 0.0160597, 0.0527863 |
| 7 finite samples, N=8, d=2 | χ=4 small init recovers 5/5 from n=256 at bond rank 2; lazy χ=4 never; lazy χ=16 only at n ≥ 256 | new data: small init 0/3, 3/3, 3/3 at n = 128 / 256 / 384 (n=256 test error 1e-4–4e-4, rank 2.0); lazy χ=4 1.4–40 (Jacobian rank 112); lazy χ=16 exact at 256, 0.9–11 at 192; centred σ=1: 0.003, 0.011, 0.17 | Jacobian rank 112 (χ=4) / 256 (χ=16); lazy χ=4 1.2–46; lazy χ=16 3–10 for n ≤ 192, exact for n ≥ 256; small-init GD at n = 128 / 256 / 384: 0.72 and 3.2 / 0.0022 and 0.0065 / 5e-4 and 1e-4, rank 2 from n=256 |
| 8 P10 fails | min a₁ = 0.44–0.46 | same seeds identical (0.463, 0.438); new noise: 6/6 runs stay ≥ 0.69 | – |

### Discrepancies and likely causes
1. **"Event shift matches to 3 digits per seed at η ≤ 0.1 η_max, all events" is overstated.**
   - In the committed CSV, 84% of 199 event comparisons agree within 0.3%.
   - Second events after near-tied races deviate by 3–66% at 0.1 η_max, and still by 6% at 0.01 η_max (σ=0.03 seed 1).
   - Cause: a large O(η²) term where c_e ≈ 0. The c_e formula itself is the correct η → 0 limit, as the Richardson check shows.
2. **"170/170":** the CSV has 195 such runs for seeds 0–4 (675 including seeds 5–44). All are GF-like, so the conclusion stands.
3. **"η*_ms = 0.00107 B for B ≤ 16":** the per-B value falls to 0.00100 at B = 16, which is 6% low.
4. **Centred model:**
   - The GF references at σ ≥ 1 have not converged by T (loss 0.39–1.66 out of 7), so λ_GF is the sharpness at time T, not at a minimum.
   - Some runs above the limit are labelled GF-like because GD moved to a flatter point (edge-of-stability selection).
   - "No bounded oscillation on the lazy side" is contradicted: committed runs above the limit oscillate in 8/21 cases at σ=1.4 and 2/33 at σ=2, and in 1 of my runs.
5. **χ=16 lazy limit "worse than zero below d^N"** holds on average, not per seed: 0.64 and 0.79 committed, 0.87 in my rerun.
6. **P10 rests on few runs.**
   - The learned component dips below 1/2 in 2/8 committed runs and 0/6 new-noise runs.
   - The quoted "erased at 0.15 η*_ms (0.04, −0.07)" mixes 0.15 η*_ms (1/4 seeds) with 0.30 η*_ms.
   - All of these runs end with a₁ ≈ 1, so the component is relearned.
   - P10's "none at ≤ 0.25 η*_ms" is still violated once (B=16 seed 1: 0.438, reproduced exactly).
7. **P9 at b0 = 0.003 is seed-sensitive:** the new seeds give 2.29 and 2.36 against a committed mean of 3.06 ± 0.26. Pooling all 6 seeds gives 2.82 ± 0.23. The c_γ range 34–45 survives.
8. **`fig_stability`, MPS N=8 panel:** the 1/5 "converges above η_max" band is seed 3. Its GF learns only component 2 within T = 60 (loss 5.0 of 7.06), so this is not convergence.
9. **Chaotic regime (1.5 η_max):** 2 of 95 runs changed class relative to the committed CSV (diverged ↔ thrown back to origin, swapped between two seeds). This is roundoff amplification; the proportions are unchanged.
10. **The P5 excess-loss numbers have no committed script.** I recomputed them from the committed CSVs and they match.

### Bugs in the code I used
1. **README run instructions fail as written.** `gd_sweep.py configs/gd_sweep.yaml` run from the repo root raises FileNotFoundError; the path must be `experiments/optim_samples/configs/...`. `common.py` creates the default `/root/outputs/...` directories at import, which lands in `C:\root\...` on Windows. `--workers` defaults to 30.
2. **Event-width rule (affects the SGD second-plateau statistics).**
   - `common.gradient_flow` stores t10 as the *first* upward crossing of 0.1.
   - `analyze_sgd` then uses width = t90 − t10 to compute the separation ratio rho. That contradicts the final-rise rule in `diagnostics.md`.
   - σ=0.1 seed 2 (rival peak 0.118, not a near-tie) gets rho = 0.94 instead of about 6.9, and is wrongly excluded from the P9 second-plateau statistics.
   - The same pattern appears in 6 other GF references.
3. **`analyze_samples` triple-counts the χ=4 lazy limit.** The σ = 1, 1.5 and 2 blocks share one init direction, so the lazy rows are identical but pooled. Result: n_seeds = 15 and a standard error √3 too small at n ∈ {32, 64, 128, 256, 512}.
4. **`analyze_samples` crashes on Windows** (cp1252 console) when printing χ/σ, before it writes `samples_curves.csv` and the figure. Setting `PYTHONIOENCODING=utf-8` fixes it.
5. **Minor:**
   - The committed `gd_shift_prediction.csv` lacks the current script's `centred` column, so it came from an older version.
   - `aligned_theory` treats component 1 as unlearned for event 2.
   - Importing the scripts writes `__pycache__` into `experiments/`; I ran a copy with `PYTHONDONTWRITEBYTECODE=1`.

Several claude.ai connectors (Atlassian, Gmail, Google Calendar, Google Drive, QuickBooks, Slack) need authorizing in the claude.ai connector settings before they can be used.


## B. Claims audit

**Claims audit: optim-samples, round 1**

I recomputed every number below from the committed CSVs with pandas. I also ran four small independent checks; their outputs are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optim_samples/` (`edge.jsonl`, `centred.jsonl`, `flip_check.json`, `sgd_flip_check.jsonl`). The longer audit notes are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optim_samples/audit_part1.md` to `audit_part4.md`, and the scripts are in `scripts/` next to them.

## Ranked issues

**1. MAJOR: "Order flips of near-tied races; a flip can remove a whole plateau"** (REPORT finding 5, notes §3.2 and the §7 table; also the SGD "race flips in 10-40% of runs, no second plateau", 225 'joint' runs)
- **Cause:** events are the *first* upward crossing of a_r = 1/2 (`common.crossing_times`). `diagnostics.md` requires the last crossing when a transient goes above 1/2.
- **Evidence in the CSVs:** every flipped GD run has max_cross = 2. The one exception, σ=0.3 seed 37, has a_3 crossing but L_end = 0.5, the same as GF.
- **GD re-run with per-step overlaps:**
  - σ=0.1 seed 1: at 0.01 η_max a_2 peaks at 0.487. At 0.3 and 0.9 η_max it peaks at 0.525 and 0.591 near t≈8, falls back, and is learned at 23.34 and 23.57 (GF 23.24).
  - σ=0.1 seed 3 at 0.9 η_max: learned at 17.06 (GF 16.61).
  - σ=0.03 seed 12 at 0.3 and 0.9 η_max: learned at 55.88 and 56.16 (GF 55.77).
- **SGD re-run** (σ=0.1 seed 1, B=256 and B=16, 12 new noise seeds, T=30): 4/12 runs have a transient a_2 crossing at t=7.4-7.7. All 12 learn component 2 at t=22.6-24.5.
- **Conclusion:** no plateau is ever removed. The sweep stops 0.05 time units after the second crossing (`stop_after: 2`), so it cannot see this. The event-2 shift outliers (e.g. −110η at 0.9 η_max) are the same artifact.
- **Fix:** use last-crossing or persistence-based events and run SGD past the GF second event. Withdraw the plateau-removal claim and the σ→0 flip conjecture. In every checked case, GD below η_max kept the GF saddle sequence.

**2. MAJOR: Claim 4 (centred model).** The claim is that the GD limit is 2/λ_GF across the lazy transition, that above it lazy-side runs only re-enter or diverge, and that the limit falls like σ^{-2(K-1)}.
- The GF references are not converged: L_GF(T)/L0 = 0.06-0.17 at σ=1.4 and 0.10-0.55 at σ=2.
- With χ=3, the target W*+W0 has bond rank up to 6. So the σ≥1.4 runs are "stalled" by the `diagnostics.md` definition, not lazy. A 10× longer GF (T=20) still stalls at L/L0 = 4-17%.
- The boundary is only resolved on the 0.5/0.7/0.9/1.05 grid, with two lazy-side σ values and no exponent fit.
- The 'gf_like' class ignores the loss: σ=0.5 seeds 1 and 4 at 1.05 η_max end at 25-100× GF's loss.
- In the agent's own CSV, 9/33 σ=2 runs above the per-seed 2/λ_GF end at or below GF's loss; 6 of these are labelled 'reentry'.
- My re-run at T=20: 15/22 runs at 1.0-2.2× the per-seed 2/λ_GF (σ=1.4 and 2) converge to exactly GF's loss, with sharpness·η/2 = 0.71-1.0. That is catapult / edge-of-stability selection (Lewkowycz et al. 2020). Divergence is general only from about 2.5×.
- **Fix:** describe 2/λ_GF as the end of monotone, GF-like GD and map the catapult window above it. For a real lazy side, use χ ≥ 9 (the cut dimension) and converged GF, and fit the σ-dependence over at least one decade.

**3. MAJOR: Claim 2 accuracy.** The claim is "3 digits per seed at η ≤ 0.1 η_max, all events" and "within 2% up to 0.9 η_max".
- First events are 0.05-0.6% off at 0.1 η_max (3+ digits only at ≤ 0.03 η_max) and 2-5% off at 0.9 η_max.
- Second events at 0.1 η_max, seeds 1 and 3: 3%, 5%, 5.5% and 13% off, and one is off by a factor of 3 (−0.050 vs −0.147). Third events are up to 8% off at 0.9 η_max.
- Fig. `gd_vs_gf` (b) plots medians with ylim 0.5-1.5, which hides these. The quoted MPS8 value "−32.40" is not in the CSV.
- The formula itself is right: extrapolating the 0.01 and 0.03 η_max values to η→0 recovers c_e.
- **Fix:** report accuracy per event and the per-seed scatter; note that the O(η²) term is amplified near ties.

**4. MODERATE: Claim 5a, P7 "confirmed" (mean shift "≥ 0 within errors").**
- The quoted ±0.41, ±0.24, ±0.21 are single-init standard errors labelled "s.e. over 5 inits". The standard error of the 5-init mean is at most 0.18, 0.11 and 0.09 (0.04-0.05 measured across inits).
- The B=1 delays are therefore resolved as positive. They are 1.6-3.4× P7's estimates (+0.17, +0.12, +0.08), so the magnitude is not confirmed.
- "t1" is component 1's crossing, which is the second (co-learned) event for 2 of the 5 inits at σ=0.3.
- Per-init spread is 0.6-6× the 1D formula (2-6× for near-tie inits), not "1.2-1.4×".
- "Never shortens the first plateau" survives once transients are handled (issue 1).

**5. MODERATE: P9 (later plateaus).**
- "B=16 and B=64 agree to 0.5%": both runs use generator seed 30000+seed and consume identical samples per unit time. They are pathwise-coupled, not independent: per-seed t2 agree within 1.5 while the spread across seeds is 0.45-2.6.
- c_γ comes from inverting −ln(1−x)/x for each run (28-47 at fixed x), so the shape of the law is not tested.
- Random-init "+28 ± 5%": the standard error over the 3 inits is ±9% (13%, 27%, 44%, ordered by T_GF as P9 predicts).
- "Within errors of 0 for η/B ≤ 1.25e-4" is wrong: σ=0.03, B=4 gives +4.2 ± 0.4%, equal to P9's +4.0%.

**6. MODERATE: Claim 7, "generalization degrades monotonically with init scale".**
- At n=512 the centred σ=1.5 and σ=2 runs stop with train loss at 0.3-5% and 4-16% of initial (the note says 1e-3 to 1e-2), so fit quality is mixed in with implicit bias.
- Plain and centred models are mixed. Within small init the trend is not monotone (σ=0.3 vs 0.5 at n=512: 4.5e-5 vs 2.5e-5).
- The χ=4 lazy-limit rows are triplicated at five values of n: n_seeds=15, so the standard error is √3 too small.
- Bond rank is 2.1, not exactly 2, at n=256 for σ=0.3.
- The lazy-limit failures follow from dimension counting (tangent dimension 112 < 256), and should be presented as expected.
- Small-init GD caps the relative step at 1% near the origin, no run met the stopping tolerance, and there is only one teacher.

**7. MINOR-MODERATE: N=8 trees (claim 3 and §3.3).**
- MPS8 excess-loss means are 6%, 17% and 20% below the map at 1.05, 1.1 and 1.2 η_max, because seed 3 never learns component 1.
- Most N=8 GF references learn only 1-2 of the 3 components by T.
- σ=0.9 is not small init: the GF state is 10-16% (binary8) and 12-35% (MPS8) sharper than balanced, not "8-12%".

**8. MINOR: Claim 1 wording.**
- "Converge" means the monotone-tail class; 39/195 runs end on the last plateau (L/L0 = 0.071).
- "170/170" matches no subset of the data: there are 195 runs with seeds 0-4, and 675 including the order seeds, all GF-like.
- At 1.05 η_max the 'oscillating' label can come from sharpness·η/2 > 1.001 alone.

**9. MINOR: miscellaneous.**
- With root×6 imbalance, failures start at 0.5 η_max, not above 0.75.
- The 1D-map regimes interleave: above the first divergence, 30 grid points (K=3) and 45 (K=7) are still re-entry.
- "0.00107 B" for the SGD limit is 6% high at B=16.
- The P10 text mixes step-size fractions (min a1 = 0.04 is at 0.15 η*_ms, −0.07 at 0.30), with 4 B-coupled seeds.
- The k=2 plateau-midpoint λ_min is −0.494 to −0.511, against the exact saddle value −0.5196 that I computed.
- Derivation typos: the small-a SGD shift drops R²/s² from κ, and the collapse condition should read η q 3^n s_new ≈ 2B.
- `gd_shift_prediction.csv` predates the final `predict_shift.py` (it has no 'centred' column), and no script produces the P5 numbers.

## Claims I consider solid

- **Propositions 1 and 2.** I checked the proof, including that Q_e=0 is equivalent to an orthogonal gauge when χ=R. The numerical sharpness check agrees to 3e-15.
- **Small-init binary4 GD boundary at η_max, exact to ±1%.** My re-run at 0.97, 0.99, 1.01 and 1.03 η_max over 10 inits: everything at or below 0.99 converges like GF, everything at or above 1.01 ends on the 2-cycle.
- **P5 on binary4:** excess loss = s1² × the 1D-map loss. Means agree to 3-4 digits; my re-run gives 0.0273 and 0.0801 in all 10 inits.
- **Backward-error formula for c_e** is correctly derived, and its η→0 limit holds. The relative event error goes to 0 as σ→0 (0.111 down to 0.0036).
- **Binary4 above η_max:** divergence at 1.5-1.75 η_max (38/40 runs) and re-entry at 2.0 η_max (25/25).
- **Edge-of-stability selection from imbalanced init:** final sharpness = min(λ_GF, 2/η) to within 1-2% for converged runs. GD-created imbalance grows ∝ η.
- **SGD:** first-escape spread ∝ (η/B)^{1/2} (mean exponents 0.46-0.55); gauge drift ∝ η/B; no first-plateau shortening once events are robust.
- **P9, qualitatively:** later plateaus are lengthened or trapped, the route amplitude decays, and c_γ ≈ 30-45 here.
- **SGD mean-square stability limit:** it matches the theory note's independent computation, and the empirical limit is about 2 η*_ms.
- **k=2 route:** it is a strict saddle. There is no collapse without label noise.
- **P10 fails for teacher [3, 2, 1].** This holds even against the learned-block threshold P10 recommends, since that threshold can only be larger (interlacing).
- **Finite samples:**
  - Small-init recovery thresholds: 5/5 from n=256 (χ=4) and n=384 (χ=16).
  - Lazy-limit failures: the lazy limit never recovers W* at χ=4, and at χ=16 only from n=d^N.
  - Bond-rank ordering: small init < centred σ=1 < lazy limit.