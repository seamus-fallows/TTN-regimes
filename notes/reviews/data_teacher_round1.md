# Review of data-teacher round 1 (2026-10-09)

Two independent local reviewers: (A) partial reproduction from the committed code, (B) adversarial audit of claims vs data. Raw re-run outputs: `data/review_data_teacher/` (local, gitignored).

## A. Reproduction

**Verdict.** Every rerun reproduces the committed numbers. Pilot-integrator runs match bitwise and the library LSODA runs match to about 1e-4. The qualitative claims hold. Two quantitative claims are weaker than stated: the c = 0.2 slope (A = 31.9 ± 1.6 against the predicted 34.1) and "plateaus are critical points, not HSVD truncations". I found no correctness bugs in the runners or integrators. The problems are in the analysis.

## What I reran
I copied `experiments/data_teacher/` unchanged (diff-identical) to `data/review_data_teacher/repo_copy/`. Every script writes into `HERE/results/`, so running in place would have overwritten the committed CSVs. I used smaller versions of the committed configs, set `DATA_TEACHER_OUT=.../data/review_data_teacher/raw`, and ran at most 6 processes. Total wall time was about 30 minutes. `ttn/` is identical between the agent branch and HEAD.

1. **Step 1:**
   - `e1_exact_aligned.py`, unchanged.
   - `run_queue.sh e1_order` with `--procs 6`: binary σ = 0.1, MPS σ = 0.2, teachers T1 and T2, all 4 variants, seeds 0–9 (160 runs).
   - `analyze_e1_order.py`.
   - An independent check of the exact clock using only the library LSODA integrator, none of the agent's code.
2. **Step 4a:** `run_dense_queue.sh e4a_sigma` with c ∈ {0, 0.2}, all 6 σ (0.03–0.3), seeds 0–9 (120 dense runs, all converged, drift ≤ 3.9e-10). Then `theory_log_slope.py`, `analyze_sigma_law.py`, a per-seed re-analysis, and an independent recomputation of the predicted slopes (rank-1 power iteration, i.e. HOPM).
3. **Rank-constrained critical point:**
   - Random TTN teacher 103, σ ∈ {0.02, 0.0356, 0.0632}, seeds 0–5 (18 dense runs), then `analyze_ttn.py`, which uses `constrained.py`.
   - Misaligned covariance (α = 1, θ = 20°, MPS-4, σ = 0.1, seeds 0–5), then `predict_rank1.py` and `analyze_plateaus.py`.
   - Independent checks: gradient descent from each plateau state, HOPM, and TT-SVD truncations.

## My numbers vs theirs
| claim | committed | rerun |
|---|---|---|
| Exact clock: aligned-init TTN gradient flow vs scalar ODE | half-times agree to 1e-9 | Identical CSV. Independent LSODA check: max\|a_r^Σ(t) − a_r^white(Λ_r t)\| ≤ 4.7e-12 (binary), ≤ 4.1e-9 (MPS); half-times match the closed-form quadrature to ≤ 1.2e-5 (grid interpolation) |
| Paired (Σ,s) vs (I,Λs): same first component | 100% at these σ (40 seeds) | 10/10 in every setting; per-seed times equal to the committed ones to 1.7e-14 |
| Paired first-escape time, rel. diff., binary σ = 0.1 (T1/T2) | mean 0.68/0.65%, max 1.13/1.29% | mean 0.57/0.46%, max 1.02/1.13% (seeds 0–9 only; same per-seed values) |
| Same, MPS σ = 0.2 | mean 0.10/0.10%, max 0.35/0.39% | mean 0.08/0.11%, max 0.35/0.39% |
| 4a, c = 0: exponent of Δt₂ (medians) | 1.967 ± 0.011 | 1.967 ± 0.011; clean seeds per-seed 1.981 ± 0.007 |
| 4a, c = 0.2: log slope A | 31.87 ± 1.60 (rms 5.0%) | 31.85 ± 1.60 (rms 5.0%); per-run t1/t2/Δt₂ within 7e-4/4e-3/6e-3 |
| Predicted A (R6), c = 0.2 / 0.45 | 34.14 / 5.68 | 34.141 / 5.682 (s = 0.7 saddle: 25.72) |
| 4b teacher 103, rank-1 plateau: distance to critical point / to HSVD | 0.0254, 0.0322, 0.0415 / 0.097 | 0.0253, 0.0324, 0.0414 / 0.097; 17/17 plateaus with the same ranks |
| `constrained.py` critical points | loss 0.158769 | Confirmed: descent from the plateau states and HOPM give the same loss and distances (1e-6) |
| Misaligned θ = 20°, MPS: plateau level error, \|cos\| with predicted critical tensor | 2e-4–9e-4, ≥ 0.9999 | identical (4/6 runs have a resolved plateau) |

The only difference is the LSODA timing scatter of 1e-5–1e-4. It most likely comes from the compiled LSODA library differing between Windows and Linux; the roundoff is then amplified along the unstable escape.

## Analysis problems
1. **Plateau detection fails silently (`analyze_sigma_law.events`).**
   - The plateau is taken as the speed minimum among points with L > 1e-2 L(0), although all plateau levels are ≥ 0.2 L(0).
   - When there is no resolved rank-1 plateau, it picks the convergence tail (plateau_rel ≈ 0.011). Δt₂ then measures the time to reach about 0.5% of L(0), not a plateau duration.
   - At c = 0.2 this hits 8 of 60 runs (seed 1 at every σ; seeds 0 and 6 at σ = 0.3), and they stay in the fit.
2. **The c = 0.2 fit mixes two saddles.**
   - Seed 3 sits on the s = 0.7 saddle (predicted A = 25.7) but is compared with the s = 1 prediction of 34.1. So 31.9 is a mixture.
   - Per seed, the log law clearly beats a power law: residuals 0.6–2% against 6–10% for the 4 seeds with a clean plateau at all σ.
   - The slope itself depends on the bookkeeping: 37.4 ± 1.9 for those 4 clean seeds, 32–41 under other reasonable choices.
   - So agreement with 34.1 is at the 10–20% level, not "within the statistical error".
   - The committed c = 0.45 data have the same problem. Six seeds sit at the saddle level and give A ≈ 6.0 (prediction 5.68). Four seeds have their speed minimum 8–14% below it and give A = 2.0–3.7. The reported 4.7 averages the two groups.
3. **Error bars are too small.** `fit()` takes the standard error from the residuals of 6 per-σ means, but the same 10 seeds (same normalized init) are used at every σ. Computed per seed, the error at c = 0.2 is ±4.2, not ±1.6.
4. **HSVD is a weak point of comparison for the critical-point claim.**
   - The agent's HSVD truncation corresponds to one TT-SVD sweep direction. The other direction lands 0.0014 / 0.005 / 0.003 / 0.0015 from the rank-1 critical point for teachers 101–104; HSVD is 0.012–0.099 away.
   - That is as close as, or closer than, the plateau states themselves. For teacher 103 the plateaus are 0.028 / 0.035 / 0.044 from the TT-SVD truncation against 0.025 / 0.032 / 0.041 from the critical point.
   - Only teacher 105 separates the critical point from every truncation. What the runs show is "not this HSVD"; "a critical point rather than an SVD-type truncation" is shown clearly only for teacher 105.
   - The "distance → 0 with σ" median trend (0.042 / 0.005 / 0.004) is a composition effect: at σ = 0.063 the median is a single teacher-103 run. Within teacher 103 the distance falls slowly: 0.041 → 0.032 → 0.025.
5. **Minor overstatements.**
   - "P(first) identical under (Σ,s) and (I,Λs) in every setting": binary T2 at σ = 0.02 has 0.800 vs 0.825.
   - "\|cos\| ≥ 0.998" for misaligned MPS holds for medians only; 20–50% of MPS runs per θ have no resolved plateau (minimum \|cos\| 0.89).
   - The half-times in `e1_exact_aligned.csv` carry a common log-grid interpolation error (about 3e-4 relative). Their 1e-9 agreement is a comparison on the same grid, which doesn't affect the claim.
   - `crosscheck_library.csv` contains a `BadZipFile` row: one raw file on the remote machine was corrupt.

## Do the README instructions work?
They work on Windows/Git Bash once `DATA_TEACHER_OUT` is set; Windows process spawning causes no problems. Practical problems:
- **Overwrites committed results:** every runner and analysis script writes into `experiments/data_teacher/results/`. Reproducing overwrites the committed CSVs and figures, and a reduced run silently replaces them with partial ones. There is no option to choose the output folder.
- **Default output path:** `/root/outputs/...`, which becomes `C:\root\...` on Windows. CLAUDE.md says raw outputs go in `data/<experiment>/`.
- **Process count:** `--procs 28` is hard-coded in both queue scripts.
- **Undocumented run order:**
  - `analyze_e1_order.py` needs `summary_rows.pkl` and all four variants.
  - `analyze_sigma_law.py e4a_sigma` needs `e4a_theory_slopes.csv`.
  - `analyze_plateaus.py` needs the `.pt` file from `predict_rank1.py`.

## Files
All in `C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/`:
- `repo_copy/` – the copied code, reduced configs and rerun results
- `raw/` – rerun outputs, 530 MB
- Comparisons: `compare_e1.txt`, `compare_e4a.txt` (with `.runs.csv`), `compare_e4b_e2.txt`, `refit_cut.txt`
- Independent checks: `indep_exact_clock.csv`, `indep_constrained.csv`, `indep_theory_slope.txt`, `ttsvd_all_teachers.txt`

Longer notes (`notes_1_setup_step1.md` to `notes_4_reruns.md`) and the check scripts are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_data_teacher/`.

No repo files were edited and nothing was committed.


## B. Claims audit

# Claims audit: data-teacher round 1 (Q6/Q7)

Auditor: adversarial claims review. Inputs: notes/data_teacher.md, agent-runs/data-teacher/REPORT.md,
experiments/data_teacher/*.py, results/*.csv (raw npz not available). Re-analysis scripts:
`scratchpad/review_data_teacher/scripts/` (refit_law.py, perseed.py, perseed_shape.py, e4b_check.py).
All numbers below are recomputed from the committed CSVs.

## HIGH

### H1. Single-sigma "plateau durations" (steps 2, 3, 4a) are grid artefacts / event misfires
Claims hit: summary rows 2 ("shrinks from ~540 to < 1 time unit as theta 0 -> 45"), 3 ("MPS-4 later
plateau collapses 137 -> 0.14"), 4a ("137 -> 8.2 -> 2.2 -> ~1.7"); REPORT "MPS-4 later plateau shrinks
137 -> 0.14".
Evidence:
- analyze_plateaus.py: t_2 = first time max_e s_2^e > 1e-2 (fixed absolute threshold, the one
  theory_teachers 3.3 says not to use); pilot output grid logspace(-2, log10 t_max, 2000) ->
  one grid interval = 0.0093 t.
- Durations / (0.0093 t_1) per run: MEDIAN = 1.000 grid step for e3 rho >= 0.2, e4a c >= 0.35,
  e2 theta = 45. The quoted 0.14 / 0.28 / 0.61 / 1.37 (e3) and ~1.5-2.2 (e4a) are one output interval
  at t_1 (t_1 = 13.7 ... 107), not measurements.
- The same rule misfires on BASELINE runs: duration = 1 grid step in 25% (e2 theta = 0), 40%
  (e4a c = 0, s2 = 0.7; e3 rho = 0), 15% (e4a c = 0, s2 = 1) of runs. Per-seed check against the dense
  library runs (same seeds): e2 theta = 0 seeds 0, 1: pilot 0.15, 0.12 vs dense 310-775 (sigma 0.12/0.075);
  e4a c = 0 seeds 0, 1, 3, 4: pilot 1.5-2.9 vs dense 5-145.
- Under complications the threshold measures "s_2 reached 1e-2", not the end of the plateau: e2
  theta = 20 seeds 0-9: pilot 0.13-11 vs dense loss-based 26-46; e4a c = 0.2: pilot 1.4-37 vs dense 7-89.
  Medians: theta = 20 pilot 4.2 vs dense ~34-41; c = 0.2 pilot 8.2 vs dense 32-40.
- Plateau-level statistics silently drop unresolved runs (NaN level): MPS e4a s2 = 1, c = 0.55/0.65:
  0/20 resolved; c >= 0.45 (s2 = 0.7): 1-2/20; e3 rho >= 0.3: 3-5/20; e2 theta = 45 (alpha 1): 2/20.
- REPORT "Decisions" says every short-plateau measurement was redone with the library; the quoted
  single-sigma durations were not (the integrator cross-check does not address grid or event rule).
Fix: withdraw the single-sigma duration numbers or label them "unresolved (< 1 grid step ~ 1% of t_1)";
quote the dense numbers (theta = 20: ~26-39 vs ~200-1800 at theta = 0, sigma ~0.1: a per-seed 8-50x reduction,
not "540 -> <1"); rerun e2/e4a single-sigma cases on run_dense.py; for step 3 add a dense pilot grid;
use loss-based or f*sigma_2^e(T)-relative events (f in 0.03-0.1) and report f-sensitivity.
For step 3 also normalise by the overall time scale (t_1 falls 14x while L(0) grows 12x).

### H2. Log-vs-power discrimination and slope errors (4a, 4b) are not shown by the analysis as done
Claims hit: 4a "later plateau logarithmic, A = 31.9 +- 1.6 (pred 34.1) at c = 0.2, 4.7-5.6
(pred 5.68) at c = 0.45 ... quantitatively"; 4(a).2 verdict; REPORT "measured slopes match".
Evidence:
- Fit range: 6 sigma in one decade (log(1/sigma) = 1.2-3.5); c = 0.45 only 0.8 decade after the
  post-hoc exclusion of sigma = 0.3.
- analyze_sigma_law.py compares unlike quantities: log law fitted to arithmetic means, residual
  rms / grand mean; power law fitted to means of log, rms in log units. The figure overlays the
  mean-of-log power line on arithmetic means (c = 0 line visibly misses its own points).
- Same-space comparison on per-sigma means with SEM weights: c = 0.2 chi2 0.97 (log) vs 1.19 (power),
  c = 0.45 0.50 vs 0.08 (dof 4/3): neither rejected; log-space rms favours the power law (0.071 vs
  0.111; 0.009 vs 0.021). Only theta = 20 rejects the power law (chi2 1.2 vs 14.3), and e4b teachers
  101, 104 (chi2_pow 10.7, 35.7); teachers 102, 103, 105 not discriminated.
- Mixture: the c = 0.2 sample mixes runs at the major saddle (46/60; A_pred 34.1), the minor saddle
  (6/60; A_pred 25.7, plateau 0.67 L0) and runs with no detected plateau (8/60; speed minimum at the
  window edge L = 1e-2 L0, so t_2 is a different event). c = 0.45: 12/50 undetected (2/10 at sigma
  0.12, 4/10 at 0.19).
- SEs are residual SEs of 5-6 correlated per-sigma means (same seeds, same init direction at every
  sigma). Per-seed slope SEM: c = 0.2: 31.9 +- 4.2 (not 1.6); c = 0.45: 4.66 +- 0.57 (not 0.31);
  e4b 101: 2.76 +- 0.24 (not 0.07); 103: 0.65 +- 0.14 (not 0.004); 104: 5.95 +- 0.35 (not 0.15).
- Done properly the theory comes out reasonably well: per-seed fits restricted to runs at the
  major saddle give c = 0.45: A = 5.73 +- 0.25 (pred 5.68); c = 0.2: 39.1 +- 3.1 (pred 34.1, +15%);
  the log law beats the power law per seed in 7/8 seeds at both c (log-space rms 0.06 vs 0.10;
  0.003 vs 0.015), theta = 20 in 10/10. e4b with per-seed SEs: within 0.5-2.2 SE of A_e*.
- Theory-side fairness: R6 is asymptotic (theory_teachers 3.4: A reached at sigma ~ 1e-3-1e-2);
  c = 0.2 is near the crossover c_x ~ 0.2 at sigma = 0.3 (theory_teachers 4.4 formula), so the large-
  sigma points should not enter the slope fit; local slopes rise 20 -> 37 from sigma 0.25 to 0.04.
Fix: classify runs by visited saddle and plateau detection; per-seed (paired) slopes and errors;
compare both laws in the same space; extend sigma to 1e-3-1e-4 or start phase 2 at the exact rank-1
critical point with an eps-scaled perturbation (eps 1e-2 ... 1e-8, as theory_saddles C5); restate 4a as
"sigma-dependence drops from p = 1.97 to effective p ~ 0.7 (c = 0.2) / 0.26 (c = 0.45); log law with
R6 slope favoured by a per-seed analysis over one decade".

## MEDIUM

### M1. Event rules for "which bond first" contradict the theory note's own prescription (4b, 4c)
- 4b (analyze_e4b_order.py): t_2 and the first edge use max_e s_2^e > 1e-2 (fixed). Teacher 103:
  s_2 at t_1 is already ~0.011 at sigma = 0.2, so dt2 = 0.017 (one dense step) in half the runs;
  mean-based A = 0.65 (quoted, -25%) vs median-based 0.87 (= prediction 0.868). No f-sensitivity check.
- 4c (analyze_cross.py): events = s_k^e reaching 50% of the teacher's value (diagnostics rule), which
  theory_teachers 3.3 says fires only after later escapes. It produces artefacts: teacher 203 seeds
  0-2, 6 show non-sibling first pairs because edge {01} (target 0.154) reaches 50% before its sibling
  {23} (target 0.323); 202 seed 2's first event is a 3rd singular value (3:3).
- Teacher 104 (first edge = e* in 43%): the measured slope 5.95 is closer to A_0 = 6.05 (the edge that
  fires first in 57% of runs) than to A_e* = 5.17; the agreement claim compares with the wrong edge.
- Positive point not reported: first-edge accuracy improves as sigma decreases (teacher 104: 0.17 at
  sigma = 0.2 -> 0.67 at 0.02; teacher 101: 0.67 -> 1.0), as the asymptotic rule predicts.
Fix: events at s_k^e >= f sigma_k^e(T), f in {0.03, 0.1}; report f-dependence; per-sigma accuracy.

### M2. Step 4c statements overstated
- "First two rank increments are always a sibling pair": false in 4/24 runs (teacher 203 seeds 0, 1,
  2, 6: first two k = 2 events on {01} and {67}, 0.2-14 time units apart).
- "Largest teacher Schmidt value first: 0/24 (binary student)": teacher 201's naive edge {01} ties its
  sibling {23} to 5 digits (0.570754 vs 0.570721) and the pair crosses within 0.1 time units; 202's
  naive {45} vs sibling {67}: 0.342 vs 0.333. At the level the theory predicts (the sibling route raises
  both bonds), the pair containing the naive edge goes first in 20/24 runs.
- "Root bipartition last (24/24), confirming P11": for teachers 201 and 202 the root cut has the
  smallest teacher s_2 (0.046, 0.100 vs >= 0.10, 0.22), so a weight-order rule predicts the same;
  only teacher 203's 8 runs discriminate. Single sigma (0.4 / 0.45, log(1/sigma) < 1), N = 8 only.
- "3rd/4th singular values appear last": violated by the 50%-event artefacts (202 seed 2; 203 has
  "1:3" before "1:2").
Fix: report 20/24; add teachers where the root cut is not the weakest; event rule as in M1; 2-3 sigma.

### M3. "S2S survives" / plateau-level agreement rest on the resolved subset
- Step 3 says "every run has the plateau-escape-plateau structure"; for MPS-4 at rho >= 0.2 the rank-1
  plateau is unresolved in 55-90% of runs (one grid step; level NaN in 11-17/20). By diagnostics.md
  these runs are not shown to be S2S (no separation ratio computed). Similarly e4a MPS c >= 0.35 and
  e2 MPS theta >= 30 (the note itself calls c > 0 runs at sigma = 0.1 "rich, non-incremental").
- Level errors are medians over resolved runs only; binary-4, s2 = 1 runs at the second singular mode
  (a strict-saddle critical point not in the predicted set of local minima) give level errors up to
  63% and |cos| down to 0.002, hidden by the medians.
Fix: report n_resolved/n per cell; include strict-saddle critical points in the predicted set;
limit "S2S survives" to binary-4 (all rho) and MPS-4 rho <= 0.1 until a dense run exists.

### M4. Scope of the one-line answer and REPORT summary broader than the evidence
- All results N = 4 (MPS-4, binary-4; d = 3-4, chi = 4, two-component CP or rank-(2,3,2) MPS teachers);
  4c is N = 8 at one sigma. "Students where a 2-core route exists (MPS)" = MPS-4 only.
- sigma in [0.02, 0.3] (one decade), pre-asymptotic for R6 and for log(1/sigma) races.
- "Binary-4 ... stays polynomial": no binary-4 sigma sweep under complications 2-4; it is a DLN
  derivation (theory_teachers R10), should be tagged derived, or tested (cheap, K = 3).
- "First escape always keeps sigma^-(K-2)": measured for MPS-4 in e2/e4a/e4b (slopes -2.02 to -2.04,
  recomputed) and step 1; not for step 3 or binary-4 under 2-4.
- "Later plateaus become strict saddles": no Hessian at any non-odeco plateau; strictness is inferred
  from timing (which is H2-limited). The library has the gauge-projected Hessian; compute
  lambda_min at plateau midpoints vs sigma (theory R11 / P7).
- Correlated inputs: the log law is a conjecture in the body (3.1 item 3) but listed as established in
  the one-line answer and REPORT summary.
- "Plateaus are always critical points ... blends rather than partial sums": rank-1 only (4b higher
  ranks have HSVD = critical point); on MPS-4 with correlated inputs the saddle is the pure component.
- 4b "every later plateau is a strict saddle of duration A log(1/sigma)": only the first later plateau
  was timed vs sigma; later ones not measured.

### M5. "Early alignment with Sigma W*, then realignment" holds on binary-4, not MPS-4
Fraction of runs with |cos| to the race tensor > |cos| to the saddle at 20% norm: binary-4 75-100%
(theta >= 10); MPS-4 25-55% at theta = 10-30 (median cos with the saddle is the higher one), 70-100%
only at theta = 45. Quoted "|cos| down to 0.44-0.88" does not match the CSV medians (0.51-0.95).
Fix: qualify by tree.

### M6. Escape degree (1.4 item 10) does not match 2 - 2/K
Measured 1.26-1.28 (binary; pred 1.333) and 1.32-1.35 (MPS; pred 1.5): k_est 2.9 and 3.3 vs 3 and 4,
>10 SE off. The stated cause ("output grid") is untested; more likely the fixed window
a in [1e-4, 1e-2] overlaps the init/unbalanced phase (MPS sigma = 0.2: a(0) ~ sigma^4/16 ~ 1e-4).
"Same for all variants" stands; "consistent with 2 - 2/K" does not. Fix: per-sigma exponents,
window starting at >= 10 |a(0)|, smaller sigma.

## LOW

- L1 Lambda-scaling (1.4 item 8): p = -0.960 (binary) is not an "O(1) saturation" effect. Within each
  teacher direction the slope is -1.000 (e1, Lambda s 1-33: -1.0001; e4, 0.008-0.53: -1.0016; MPS
  -1.0001/-1.0007). The pooled deviation is one prefactor jump between the e4 and e1 sub-series
  (local slope -0.60 at the junction: the random init projects differently on e1^4 and e4^4). So the
  prediction holds better than stated, the explanation is wrong, and "3.6 decades" is two series of
  1.6-1.8 decades. The paired (I, Lambda s) runs share the init, so they cannot test the explanation.
- L2 Swap wording: "unlearned and replaced iff the other has the larger Lambda s^2" is necessity only
  (0 swaps after a landscape-correct first in all 24 binary settings); sufficiency holds with
  probability 0.33-0.94. T1 conditionals rest on n = 4, 3, 1 runs; "P -> 1 as sigma -> 0" is from 3
  sigma over 0.7 decade (below the diagnostics.md >= 5 sigma / 1 decade rule); swaps also occur in T1
  where the two orderings agree (summary row 1 ties them to disagreement); "0 of 30" is 0 of 27; "wrong
  plateau not reached (7-11% below)" is T1's 6 runs, T2's 61 swap runs sit a median 2% below.
- L3 "P(first) under (Sigma, s) equals (I, Lambda s) in every setting (identical seeds ...)": binary T2
  sigma = 0.02: 0.800 vs 0.825; same_first 0.975 / 0.95 at sigma = 0.02 / 0.05.
- L4 "Residual shrinks like 1/t_1": true for binary (prop. to sigma); MPS shrinks like sigma^1.1-1.5,
  not sigma^2.
- L5 binary-4 order drift "greedy only logarithmically slowly": theory_saddles 4.5 predicts
  P(smaller first) ~ sigma^((s_r - s_q)/s_q) (0.6 for T1, 0.2 for T2); the data agree with that power
  (T1 0.10 -> 0.025, T2 0.25 -> 0.20) but this is neither stated nor compared.
- L6 4b HSVD "distance to the critical point shrinks with sigma (0.042 / 0.005 / 0.004)": the
  sigma = 0.063 value is one run of teacher 103, which has distances 0.025-0.042 at every sigma.
  Within-teacher shrinkage is x1.2-1.7 for x1.8-3 in sigma (consistent with R11's sigma^0.8, but the
  quoted medians are a composition artefact). Detected plateaus: 20 rank-1 in 150 runs, sigma <= 0.063.
- L7 Float-floor check globbed the pilot sigma-sweep directories (e2_sigma: 20 runs, e4a_sigma pilot),
  not the dense runs behind the claims, and no TTN-teacher runs; min envelope 8.9e-12 vs the stated
  "no signal below 1e-11". Not a real risk at these sigma, but the coverage claim is wrong.
  crosscheck_library.csv records a corrupted e1_order npz (BadZipFile, CRC): verify all raw files.
- L8 "20-40 seeds per setting" (summary line): sigma sweeps use 10 (e2/e4a), 6 (4b), 8 (4c).
- L9 Second-learning exponent for binary (Sigma, s) -1.05 +- 0.01 includes the single swap run at
  sigma = 0.063 and 0.1 (without: -1.010); Dt_2 = t_2 - t_1 (the later-plateau quantity used elsewhere)
  is not reported for step 1 (MPS -2.008 +- 0.004; binary -1.15 +- 0.06 with the swap runs).
- L10 summary_plateau_law.png: right panel has a log y axis versus log(1/sigma), where power laws are
  straight lines and log laws are concave, so it cannot show "logarithmic"; it plots medians while
  the fits use means; step 3 (correlated) is absent although the caption says "every complication".
- L11 Table row 2 answers "Yes" (theory predicts), but the slope is not predicted for misalignment
  (body item 4 says so); row 4a "quantitatively" (see H2).
- L12 e2 item 5 "white_mapped durations within ~20%": up to +36% (MPS alpha 0.5 theta 30) and +94%
  (theta 45), though those are unresolved cases.
- L13 SGD check: binary-4, rho = 0.3, sigma = 0.05, 3 seeds, B >= 256 only; the summary's "online SGD
  with sampled data agrees" should carry that scope (it does not cover MPS-4 or the "collapse").

## Derivations, sections 0-4 (checked by hand; no errors that change a result)

- Fact A (binary-4 = depth-3 DLN, W_(01|23) = G1 R G2^T, loss tr[(W-W*)^T S01 (W-W*) S23]): correct
  (also proven in theory_saddles 4.2 / theory_teachers 6.2).
- Fact B: f* = <S^(1/2) W*, (x) x~_i>, cores G~ = G x S_i^(1/2), GF in G~ preconditioned by S_i: correct.
  Imprecisions: (a) Section 2 "the set of plateau states is exactly that of the whitened problem
  (proven)" - proven for critical points and their Morse indices (linear reparametrisation, Sylvester),
  not for which ones are visited; (b) "per-site overlap u_r^T S_i u_q" must be normalised by the
  S_i-norms; (c) 1.3 cites Eckart-Young for the best rank-1 approximation, which covers binary-4 only;
  for MPS-4 use the odeco best-rank-1 result (largest-weight component).
- 1.1 aligned exact solution: invariance of the diagonal family, da/dt = Lambda (s - Pi) Pi / a,
  dPi/dt = K Lambda (s - Pi) Pi^(2-2/K), clock Lambda_r t, T(s, a0) leading term a0^-(K-2)/((K-2)s),
  widths ~ 1/(K Lambda s^(2-2/K)): all correct. Could be tagged proven.
- 1.2 race substitution (linear term <W, S W*> identical to whitened with weights Lambda s; quadratic
  term O(|W|/s) relative): correct.
- 1.3 swap Hessian: rederived. H11 = s_a^2 lambda_{b,1} lambda_{a,2}, H22 = s_a^2 lambda_{a,1}
  lambda_{b,2}, H12 = -s_a s_b Lambda_b, det = s_a^2 Lambda_b (Lambda_a s_a^2 - Lambda_b s_b^2), Pi
  decouples at second order; equal splits give s_a sqrt(Lambda_b)(s_a sqrt(Lambda_a) - s_b
  sqrt(Lambda_b)) in angle units (/q^2 in parameter units). Correct, and matches the gauge-projected
  library Hessian to ~1e-15 (e1_hessian_check.csv). Terminology clash: the note calls the rotation
  "two cores, k = 2"; in theory_saddles this is m = 2 piggyback terminals with a k = 3 add route
  (k = 2 there means a strict add route with lambda_min = -s_r a^(K-2)). Rename to avoid confusion.
  The binary-4 statement is the DLN result of Achour et al. 2024 Thm 7 plus reduced-rank-regression
  geometry with S^(1/2); the covariance split (race Lambda s vs landscape Lambda s^2) is the content.
- 3.0 correlated inputs: <U_1, M U_2> = (E prod_i x_1^(i))^2 = (3 rho^2)^2 = 9 rho^4 (Isserlis):
  correct; pairing-term bond ranks (1 on binary-4, d on MPS-4): correct; correlated.py moment matrix
  (sum over perfect matchings, site_corr[p mod N, q mod N] delta) is correct.
- 4(a) rank-1 critical points = critical points of g on unit rank-1 tensors, level 1/2(|W*|^2 - g^2):
  correct. c* = (N-2)/N: g'' at psi = beta/2 = 2N cos^(N-2)(beta/2)[(N-1) sin^2 - cos^2], so local max
  iff tan^2(beta/2) < 1/(N-1) iff c > (N-2)/N: correct. Gap: shown only along the symmetric family
  x_i = x; that the transverse directions are negative (needed for "local max", and for "exactly two
  maxima below c*") is in theory_teachers 4.1, which should be cited. Numerics bracket c* between
  0.45 and 0.55 (s2 = 1) and the fold between 0.35 and 0.45 (s2 = 0.7); "fold ~0.4" should quote the
  exact c_fold(4, 0.7) from theory_teachers 7.2 instead of the bracket midpoint.
- Binary-4 non-orthogonal: singular modes of s1 a1 b1^T + s2 a2 b2^T with <a1,a2> = <b1,b2> = c^2,
  plateau sigma_2^2/2: correct.

## Status tags

Mostly appropriate. Misuses: Section 2 "(proven)" (see Fact B a); 4(a).2 verdict "empirical,
confirming the derived R6" (should be "consistent with", see H2); one-line answer states the
correlated-input log law and binary-4 "stays polynomial" without the conjecture / derived tags (M4);
1.4 item 10 tags agreement with 2 - 2/K as empirical though it fails (M6).

## Solid claims (survive the audit)

1. Aligned init: exact per-component clock Lambda_r t (half-times to 6e-10, widths to 4e-8); the
   s -> Lambda s substitution misses widths by up to 47%.
2. Race = whitened problem with weights Lambda s: same first component in 100% (MPS) / 95-100%
   (binary) of 480 pairs; paired t_1 ratio 0.994-1.0006 at 6 sigma.
3. Landscape = whitened with sqrt(Lambda) s: plateau levels 1/2 sum Lambda s^2 (median rel. err.
   1e-8-1e-3 for the regular plateaus); no swap after a landscape-correct first component in any of
   24 binary settings (4 variants x 2 teachers x 3 sigma); exact Hessian = formula (12/12 binary, to
   1e-15), no negative eigenvalue on MPS-4 (12/12); 0 swaps in 960 MPS-4 runs.
4. Race set by the flow metric (mapped-init controls: different first component in 10-53%, 11-23%
   slower): well designed and clean.
5. Covariance does not change the first-escape exponent (t_1: -1.010 +- 0.002 / -1.999 +- 0.002,
   per-seed SEs agree); t_1 proportional to 1/(Lambda s) (exactly -1.000 within each teacher direction).
6. On binary-4, P(swap | landscape-wrong first) rises as sigma decreases (T2: 0.33 / 0.68 / 0.94,
   n ~ 30 each) - trend solid, limit -> 1 is extrapolation.
7. MPS-4 first-component choice independent of sigma (per seed).
8. Resolved rank-1 plateaus sit at rank-1 critical points of the Sigma- or M-loss (levels 1e-6-5e-3,
   |cos| >= 0.998), including the blended saddles of non-orthogonal teachers; rank-1 saddle counts vs
   c (2 -> 1 around c* = 1/2 on MPS-4, single on binary-4).
9. Binary-4 with correlated inputs: first saddle changes content (level 0.33 -> 0.10 L0, predicted
   M-loss critical point reached in 20/20 runs per rho).
10. MPS-4: breaking orthogonality makes the later plateau's sigma-dependence much weaker than
    sigma^-2 (c = 0 / theta = 0: p = 1.97-1.99; c = 0.2, 0.45, theta = 20, random teachers: effective
    p 0.25-1.0) over sigma in [0.02, 0.3]; for theta = 20 a log law is preferred (chi2 1.2 vs 14.3;
    10/10 seeds), A = 15.0 +- 0.4.
11. First escape keeps sigma^-2 on MPS-4 under misalignment, overlap, random MPS teachers (recomputed
    slopes -2.02 to -2.04).
12. Random MPS teachers: rank-1 plateau closer to the bond-rank-1 critical point than to the HSVD
    truncation (19/20 plateaus; clean for 4/5 teachers).
13. Binary-8 student, MPS-8 teacher, sigma = 0.4: root bipartition grows last in 24/24 runs (as an
    observation; discriminating vs weight order only for teacher 203).
14. Integrator cross-check (7e-7) and conservation drift (< 1.1e-9).

## Follow-ups, in priority order

1. Rerun the single-sigma cases of steps 2, 3, 4a on the dense two-stage grid with loss-based or
   f*sigma_2(T) events; for step 3 add a dense pilot grid and a sigma sweep (the log law there is a
   conjecture).
2. Re-analyse e4a/e2/e4b: classify runs by visited saddle and plateau detection, per-seed slopes and
   errors, both laws compared in the same space; extend sigma by 2-3 decades (or eps-seeded phase-2
   runs from the exact rank-1 critical point) for c = 0.2, 0.45, theta = 20 and the 5 MPS teachers.
3. Measure lambda_min of the gauge-projected Hessian at later-plateau midpoints vs sigma (strict
   saddles: sigma-independent) for 4a/4b/2.
4. Binary-4 sigma sweep under complications 2-4 (cheap, K = 3) to back "stays polynomial".
5. 4b/4c event rule with f in {0.03, 0.1}; per-sigma first-edge accuracy; 4c at 2-3 sigma and with a
   teacher whose root cut is not the weakest.
6. One N > 4 MPS student (MPS-6) for the non-orthogonality law, to support "MPS students" in general.
