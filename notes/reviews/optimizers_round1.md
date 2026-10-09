# Review of optimizers round 1 (2026-10-09)

Two independent local reviewers: (A) partial reproduction from the committed code plus an independent reimplementation, (B) adversarial audit of claims vs data. Re-run outputs: `data/review_optimizers/` (local, gitignored).

## A. Reproduction

## Reproduction review: optimizers (B1), round 1

The core claims (1), (2), (4) and (6) and the sharpness part of (5) reproduce. Every reproduced run that uses the agent's code matches its committed row. The finite-sample part of (5) does not hold on new data draws. I also found several analysis and code bugs, one of which explains the defective `adam_escape.png`.

### What I ran
- **README.** `uv run pytest experiments/optimizers` passes (4 tests). The run command `python -m experiments.optimizers.run` fails on Windows: `run.py:124` uses `mp.get_context("fork")`. I wrote a scratch driver that calls the agent's `run.jobs_from_config`, `run_id` and `work` unchanged, in a spawn pool.
- **Agent code, 375 runs.** My reduced YAMLs are strict subsets of the committed ones, so every run has the same run id as a committed row and was compared row by row with the committed `*_summary.csv`.
  - scaling_b4: 273 runs. 13 optimizer settings × 7 σ in [0.002, 0.128] × seeds 0–2.
  - later_b8: 30 runs. GF and Adam lr 1e-3, 5 σ × 3 seeds.
  - eps_mps8: 12 runs.
  - deep: 30 runs. MPS8 GF balanced and fan-in, binary8 fan-in GF, and Riemannian lr 1 / ALS / DMRG at σ 0.3 and 0.8.
  - implicit_bias_b4: 12 runs at n = 45.
  - precond_b4: 18 runs.
- **Independent check, no `experiments/` code.** From the library I used only TTNSpec, the fan-in init, `to_dense` and the teacher factors. I wrote my own:
  - dense loss and component overlaps;
  - Adam and RMSProp updates;
  - GF first-escape time, located exactly with scipy LSODA plus an event function (no output grid);
  - Hessian, balance index and bond ranks;
  - finite-sample data.

**Caveat about the environment (not caused by me).** At about 21:54, mid-run, someone else edited the library working tree without committing: `ttn/dynamics.py`, `diagnostics.py`, `plotting.py`, `theory.py`. The edits add Recorder "fine rows", on by default. Worker processes started after that loaded the new code; `scratchpad/review_optimizers/libversion.py` labels which runs used which version.
- Discrete-method results are identical under both versions.
- GF runs on the new version record fine rows, so their event times are the exact ones. That explains the 1–2% GF timing differences below.

### Numbers
| claim | quantity | committed | reproduced (agent code) | independent |
|---|---|---|---|---|
| 1 | binary4 GF first-escape exponent p | 1.007 ± 0.001 | 1.006 ± 0.001 | 1.006 ± 0.001 |
| 1 | binary4 Adam lr 1e-3 / 1e-2 / 1e-4 | 0.009 / 0.013 / 0.012 | −0.004 / 0.015 / 0.011 | −0.004 ± 0.010 / – / 0.012 ± 0.005 |
| 1 | binary4 RMSProp lr 1e-3 | 0.001 ± 0.019 | −0.005 | −0.005 ± 0.013 |
| 1 | lr·n₁ at lr 1e-3 / 1e-4 | 0.180 / 0.168 | identical runs | 0.177 / 0.169 |
| 1 | binary8 GF p, σ 0.25–0.8 | 4.76 ± 0.06 | 4.68 (3 seeds); 4.96 on σ ≤ 0.5 | 4.74 ± 0.05 (same 42 inits) |
| 1 | binary8 Adam lr 1e-3 / 1e-2 | 0.049 ± 0.022 / 0.136 ± 0.055 | 0.023 (3 seeds) | 0.037 ± 0.019 / 0.145 ± 0.033 (all 42 runs resolved) |
| 2 | eps law: measured/predicted, σ ≤ 0.03 | 0.989–1.053 | n₁ identical | 0.999–1.000 (lr 1e-3), 1.001–1.006 (lr 1e-2, σ ≤ 0.02), 1.01–1.035 (lr 1e-2, σ 0.03); on fan-in inits 0.999 / 1.000 |
| 3 | ALS / DMRG / Riemannian lr 1 event times across σ | identical | identical (spread ≤ 3e-4) | – |
| 4 | first escape relative to GF (median per σ) | GD 1.000–1.010, heavy ball 1.007–1.185 | GD 1.0002–1.0053, heavy ball 1.004–1.105 (max 1.196); same order in 100% | – |
| 5 | λ_max / K·s₁^(2−2/K) at the end of training (binary4) | GF 1.000, Adam 1.09–1.94 | – | GF / GD / heavy ball 1.000; Adam 1.09–1.94; RMSProp 1.11–1.56; balance index 0.23–0.50 vs ≤ 0.04 |
| 5 | runs ending with a rank-4 edge at n = 45, GF vs Adam lr 1e-3 | 1/12 vs 8/12 | identical per run | new data draws: 1/6 vs 1/6; agent's data draws with my code: identical to committed |
| 6 | fan-in GF stall on MPS8 | 0/12 converge | seeds 0 and 1 stall, balanced inits converge; binary8 seed 1 stalls, seed 3 converges | the dormant overlap collapses 1e-4 → 1e-19 identically at rtol 1e-9/atol 1e-14 and rtol 1e-11/atol 1e-30, so it is not a tolerance artifact |

Per-run agreement:
- All discrete methods match the committed values to within 5e-5 relative, and orders, convergence, bond ranks and stop reasons agree 100%.
- GF event times agree within 0.27% on binary4 and 1.7% on deep trees, because of the output grid (see discrepancy 2).
- RMSProp's stopping step differs because the runs hover rather than converge cleanly.
- After overshoots of 1e17–1e29 L0, Gram preconditioning converges up to 15 steps later or earlier.

### Discrepancies and likely causes
1. **Claim 5, finite samples, is not robust to the data draw.** My code on the agent's datasets reproduces the committed ranks and population losses to 4 digits. On 3 new datasets (6 inits), GF and Adam each end with a rank-4 edge in only 1/6 runs, and Adam's population loss is lower than GF's in 4/6. Even in the committed data at n ≤ 45, Adam lr 1e-2 and RMSProp beat GF on median population loss (2.7 and 3.9 vs 8.0). So "worse generalization at n = 45–60" holds only at n = 60.
2. **GF event times are limited by the output grid.** Committed times are interpolated on a fixed grid of 600 log-spaced plus 1201 linear points, giving errors up to ±1.7% on deep trees (e.g. MPS8 balanced seed 0: committed 968.04 vs exact 984.79). This error is the main source of the 0.97–1.05 spread in the eps test. With exact times the law holds to 0.1%.
3. **The eps crossover formula is not general.** On binary4, Adam's GF plateau only appears below σ ≈ 1e-8, where (eps/lr)·T_GF ≈ n_sign. The notes give σ_eps = 2e-4. The real crossover depends on lr; for K = 8 it happens to land near σ_eps.
4. **Heavy ball's first escape is 1–19% slower than GF**, giving p = 0.970 vs 1.007. The agent calls this a "fixed delay" of about 0.18 time units, but the measured delay is 1–2.4 time units. The likely cause is momentum discretization error near the escape.
5. **Adam lr 1e-2 on binary8:** the 10-step fine trace mis-times escapes at 15–25 steps by −7% to +18%. With every step recorded, p = 0.145 ± 0.033: small but not zero.
6. **Riemannian GD at lr 0.1** has event times 10–55% apart between σ 0.3 and 0.8 on deep trees. Only lr 1 is exactly σ-independent.
7. **Gram preconditioning with scale-free damping is not σ-free at lr 0.01:** p = 0.69 ± 0.11, and the agent's own theory predicts ln(1/σ)/lr dependence.
8. **"Order unrelated to teacher weights" is not a real temporal order** for Riemannian lr 1, ALS and DMRG. All components rise within one step, so the "order" just ranks the overlaps after that step.

### Bugs in the code I used
1. **`run.py` uses the fork context**, so the README run command fails on Windows. It also creates the default `--out /root/...` directory even with `--dry`.
2. **`configs/scaling_b4.yaml` precond entries lack `blowup: 1e300`.** All lr 0.1 runs (42/42) and 33/42 lr 0.01 runs stop as "diverged" after 1 step; I reproduced this. Those one-step "orders" feed `order_agreement_with_gf.csv` and the REPORT's "33%" GF-order agreement for Gram-preconditioned GD on binary4.
3. **Float round-trip bug in `analysis.collect`.** It rewrites the cached summary CSV on every call. pandas' default float parser turns 1e-30 into 9.999999999999999e-31 and back again, alternating on each call. `make_figures.eps` and `fig_adam_escape` filter with `eps == 1e-30`.
   - This is why the committed `adam_escape.png` has no eps→0 curves and no dashed prediction: n_sign comes out NaN.
   - On my subset, `make_figures eps` crashes with `KeyError (0.001, 1)`, so README step 3 fails for `eps`.
   - Fix: `read_csv(float_precision="round_trip")` in `collect`, or compare with `np.isclose`.
4. **`plots.per_seed_fits` log-law score is an artifact.** It fits the log law in linear T and then scores it in ln T with clipping at 1e-300. The "rms 370 for a log law" quoted in the notes comes from that clipping.
5. **`fig_implicit_bias` docstring vs code.** The docstring says runs that did not converge are excluded from the rank panel; the code counts them.
6. **README run count:** it says scaling_b4 has 672 runs; the YAML expands to 798.

### Files
- Notes, in parts: `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optimizers/`
  - `notes_1_setup.md`
  - `notes_2_independent.md`
  - `notes_3_code_issues.md`
  - `notes_4_claims.md`
  - `notes_5_reproduction.md`
  - `final_draft.md`
- Scripts, in the same folder: `driver.py`, `rep_*.yaml`, `compare*.py`, `indep_*.py`, `reanalyze_eps.py`, `libversion.py`
- Outputs: `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optimizers/`
  - `raw/` (375 runs)
  - `results/` (`cmp_*.csv`, `indep_*.csv`)
  - `indep/` (b4, b8, b8_balanced, eps, eps_balanced_exact, eps_b4, stall, bias, bias_theirdata, hess)

Nothing tracked under `experiments/` was modified, and nothing was committed.


## B. Claims audit

**Claims audit of `optimizers` round 1: 3 high-severity, 6 medium and several low issues; claims (1), (2) and GD are solid**

Claims (1) and (2) hold. Claim (2) is in fact tighter than reported once the GF reference times are resolved. Claim (3) is not supported for Gram-preconditioned GD or for the "order" part. The headline "about half GF's exponent" and the finite-sample generalization statement are not supported. Supporting numbers are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optimizers/claims_audit_evidence.txt`. Re-runs and CSVs are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optimizers/`, and the scripts are in `scratchpad/review_optimizers/scripts/`.

## Issues, ranked

**1. High: the Gram-preconditioned GD part of claim (3) ("all components at once, sigma-independent time").**
- In `scaling_b4`, 75 of 84 Gram runs stopped after one step. The config has no `blowup` key, so the default 1e6·L0 guard fired. Their "events" fall inside step 1, yet they feed `order_agreement_with_gf.csv` (33%), the cost table and the synthesis "33%".
- On deep trees the events sit at 1e-26 to 1e-12 steps with rho up to 1e27, so 33-67% of runs at sigma = 0.3 are flagged S2S. These are artefacts of the first-step overshoot, which the agent's own §1.7 calls not meaningful.
- Time to converge depends on sigma: p = 0.69 ± 0.11 at lr 0.01, and 0.17 ± 0.03 with absolute damping 1e-2.
- At lr 0.1 the time is flat (about 380 steps for sigma 0.002-0.016). The derived return time (K-1)·ln(1/sigma)/lr predicts a change of about 42 steps over that range, so the law fails in kind, not only by the reported 3x.
- MPS8 odeco converges in 1 of 6 runs.
- **Fix:** remove Gram GD from the claim. Re-run `scaling_b4` with `blowup: 1e300`. Define events only after the overshoot has returned.

**2. High: "order unrelated to teacher weights" for ALS, DMRG and Riemannian GD (claim 3).**
- ALS, DMRG, Riemannian GD and re-gauged GD give one trajectory per seed at every sigma, so "0% of 42 inits" is 0 of 6 independent seeds.
- My micro-step re-run: in 35 of 36 ALS/DMRG runs, at least two components reach 1/2 in the same micro-step. The reported orders come from interpolation inside a micro-step.
- Where the order is resolved, it tracks the weights. Riemannian GD (lr 1) learns the weakest component strictly last in 5 of 6 binary8 runs. That matches the agent's own `deep_table` (102 ×3, 012 ×2).
- Taken at face value, the ALS/DMRG orders put the smallest-weight component first in 12 of 18 seed-problems.
- **Fix:** state "no resolved learning order; Riemannian GD staged over 2-7 steps, weakest component last" and count independent seeds only.

**3. High: "Adam's later plateaus grow at about half GF's rate on binary8" (REPORT summary and practitioner statement).**
- The range is only 0.5 decade, and the local slopes are not constant: about 1.6-3.1 for gap 0 and 1.5-5.5 for gap 1.
- Per seed, the gap-1 slope over the small-sigma half ranges from 1.2 to 6.2. Some seeds grow at GF's rate.
- The agent's own probe (binary8, seed 0, sigma 0.3) contradicts the AEoS attribution. During gap 0 the preconditioned sharpness stays at 0.05-0.4 of the threshold; the AEoS is reached only in the last plateau.
- The agent's memory-phase formula (§1.2 e') gives a similar effective slope of about 2.2 from a logarithmic law. This is suggestive only, since a per-seed fit of that formula is poor.
- The offered conjecture conflicts with §2.7: dormant overlaps stay at init level (1e-22 to 1e-15) under Adam.
- **Fix:** drop "half". Use a saddle-start protocol over at least 1.5 decades, with a probe on every plateau.

**4. Medium: GF event times are not resolved by dense output (analysis standard 1).**
- The GF runs used the library v0 code without fine rows. Re-running with the agent's output grid reproduces their t_first to 7 digits; resolved values differ by -1.1% to +1.7%.
- That error explains the eps ratios' 0.97-1.05 scatter. With resolved GF times the ratios are 0.9989-1.0003 (lr 1e-3, sigma 0.02-0.03) and 1.001-1.006 (lr 1e-2, sigma 0.015-0.02).
- The deviations at sigma 0.03-0.05 (from -5% to +17%) are systematic crossover effects, not noise.
- The P6 check is limited by the same resolution.
- **Fix:** re-run the GF baselines with fine rows and regenerate.

**5. Medium: claim (5), finite samples.**
- At n = 45 there is no consistent "Adam generalizes worse". Median population loss (L0 = 7):

  | sigma | GF | Adam lr 1e-2 | Adam lr 1e-3 |
  |---|---|---|---|
  | 0.01 | 15.2 | 2.8 | 14.4 |
  | 0.1 | 1.6 | 2.2 | 7.2 |

- At n = 60, every GF and Adam failure is a run that never reached the training tolerance (train loss 6e-6 to 1.5e-4 L0). They are not "rank-4 interpolants with population loss ~2": their losses are 2.9-36, and "~2" is the median of a bimodal set.
- The rank-4 counts rest on one threshold (s_4 > 0.1). The entropy rank shows no difference (GF 2.316 vs Adam lr 1e-3 2.315).
- RMSProp never converges.
- "More so at larger lr" is 9 vs 8 of 12 runs.
- The sharper, imbalanced minima part of claim (5) is solid (listed below).

**6. Medium: claim (6) is broader than the evidence.**
- Fan-in GF stalls are specific to MPS8 odeco: 0 of 12 converge, with dead overlaps 1e-35 to 1e-41. Binary8 odeco stalls in only 2 of 12 runs, at overlaps around 1e-13; E8 converges in 22 of 24.
- The R12 trap attribution is unverified: Q_νν was not measured at route ends.
- The balanced init still leaves 2 of 6 MPS8 odeco runs unconverged at sigma 0.3.
- The practitioner statement that a balanced init "avoids it" for Adam is contradicted: at lr 1e-2, 4 of 12 balanced runs fail vs 5 of 12 fan-in; RMSProp balanced fails 6 of 6 at sigma 0.3. R12 does not apply to Adam in any case.
- "chi = 3 stalls too" cannot be checked: the pilot results are not committed.

**7. Medium: heavy ball in claim (4).**
- The first-escape exponent is 0.970 ± 0.002 vs GF's 1.007 ± 0.001.
- The delay is not fixed, as the agent explains it: it grows from 1.00 to 2.61 GF units as sigma falls (about 0.39 per e-fold). This looks like the momentum analogue of P6.
- On deep trees, heavy-ball first events are 0.99-1.11× GF's.
- Only lr ≤ 0.27·η_c was tested, but the synthesis says "lr < η_c".

**8. Medium: "Adam is sequential; later plateaus: yes" is stated for all trees.** On binary4 the rho > 3 rule holds in only 0-21% of runs, so these runs are "rich, non-incremental" by the project's own rule.

**9. Low to medium: re-gauged GD.** The §1.4 claim that order is "set by projections, not by weight" is contradicted on binary4: the largest component comes first in 6 of 6 seeds. Event times are spread 26-361 steps, so "rise together" is overstated. "Stalls on MPS8" is a budget limit (1e3 GF units), not shown to be a stall.

**10. Low.**
- The log-law fit is done in linear T space and clipped, so the reported scores of 370, 175 and 44-52 are artefacts. A fair fit gives 0.62 vs 0.005 for GF on binary4; conclusions mostly unchanged.
- §2.3 says the loss overshoot grows as sigma^-6. It grows as sigma^-12 (×4096 per halving); W grows as sigma^-6.
- §1.2 c bounds |m̂/√v̂| at 3.2. The general bound for default betas is about 7.3; 3.2 is the single-spike case.
- The binary8 GF exponent 4.76 understates the agreement: on sigma ≤ 0.4 it is 5.00 ± 0.02 (5 seeds).
- The committed `adam_escape.png` lacks the eps = 1e-30 curves and the prediction curve. Regenerating from the CSVs draws them, so the figure is stale.
- `component_events` keeps transient crossings that fall back below 1/2 as events (spurious "o201" orders for GD and heavy ball).
- "P2/A2 confirmed" goes beyond A2's scope (χ_e = R; here χ = 4 > 3): it should read "consistent". The P1 correction is not computed.
- sigma_eps "~0.1 for K = 7-8" was tested only at K = 8.
- The binary8 Adam lr 1e-2 first-escape exponent (0.14) is limited by trace resolution and biased by excluding 15 of 42 runs.

## Claims I consider solid
- **Claim (1), Adam/RMSProp first escape is sigma-independent.**
  - Binary4: per-lr exponents 0.009-0.013 (SE 0.003-0.010) over 1.8 decades, against GF 1.007 and GD 1.005. The lr·n_1 sign-flow limit is about 0.165.
  - Binary8: 0.05 ± 0.02.
  - MPS8 with eps → 0: 139 to 121 steps over 1.5 decades.
  - The homogeneity derivation is correct. The small exponent at lr 1e-4 (0.012 ± 0.003) fits a +O(sigma) correction.
- **Claim (2), the eps crossover law.**
  - My independent GF runs give T_GF ∝ sigma^-6.000 from 0.3 to 0.1 (ratios 63.98 and 728.8 vs 64 and 729).
  - With resolved GF times, the per-seed agreement is within 0.3% for sigma ≤ 0.03.
  - Scope: MPS8, odeco, balanced init, first escape, 6 seeds.
- **ALS/DMRG sigma-independence.** Proven, and I checked the proof. Convergence takes under one sweep.
- **Riemannian and re-gauged GD.** Riemannian GD has no plateaus and is sigma-free (2-9 steps). Re-gauged GD's event times are sigma-independent to 4 digits.
- **GD reproduces GF.** Per-seed event times agree within about 2% on deep trees at sigma 0.8; exponents agree within 0.003 on binary4.
- **Sharper, imbalanced minima.** Adam/RMSProp end at imbalanced minima (balance index 0.21-0.96) that are sharper than the balanced one (λ_max ratio 1.01-6.1, never below 1). GF and GD sit at 1.000.
- **Order under Adam.** Adam keeps GF's first component in 92-100% of runs and the full order in 86-98% on binary4. The E8 locality order also survives.
- **Fan-in GF stall on MPS8 odeco.** The stall itself (0 of 12, dead overlaps) is real for that problem, and balanced GF converges in 10 of 12.