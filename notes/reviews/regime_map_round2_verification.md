# Verification of regime-map fix round 1 (c889f2d, 1745550, ac5974c) - 2026-10-10

Independent verifier. Inputs: `notes/reviews/regime_map_round1.md`, the lead's fix list, `notes/regime_map.md`
(section 3), `experiments/regime_map/`, `agent-runs/regime-map/REPORT.md`, raw outputs `data/regime_map/`
(copied remote runs) and `data/regime_map_local/` (read only). Scripts and outputs (not in git):
`C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_rm2/`
(`relabel.py`, `bounds.py`, `lazy_checks.py`, `tensor_check.py`, `ksweep.py`, `bumpcount.py`, `out/`).

**Bottom line.** All eight MUST FIX items and the three SHOULD FIX items were addressed in the notes, and every
key number I recomputed independently reproduces to the quoted digits. Two problems remain:
1. **New problem: K = 10 and 12 in the K sweep.** The floor-stalled runs classified by rho give the medians
   1.16 and 1.15. Most of those brackets end at a run that learned nothing within its budget, not at a
   rho <= 3 crossing:
   - K = 12: 5 of 5 resolved seeds;
   - K = 10: 2 of 4 resolved seeds.

   Two claims depend on these brackets: "sigma_c > 1 at K = 10-12" and "sigma^K = const fails". Drop both
   (errata E1, E2).
2. **Partly fixed: the REPORT body is unchanged.** The REPORT got a correct addendum, but the summary below it
   still states the withdrawn claims.

Minor wording slips are listed in the errata.

## What I ran (about 30 min, at most 5 processes)

- **Event relabel.** My own code applies the last-crossing rule to all 9,165 copied library runs:
  log rows plus fine rows, the stored refined widths, and the label rule of `notes/diagnostics.md`.
- **Per-seed boundaries and fits.** I recomputed every per-seed boundary and every slope fit from my
  relabelled runs.
- **Tensor kernel.**
  - I compared `run_regime.tensor_kernel` with `ttn.diagnostics.tensor_tangent_kernel` and with my own
    implementation (`autograd.functional.jacobian`) on 7 small cases.
  - I recomputed dK_tensor for 104 saved runs from their theta_end and checked them against the agent's
    cache.
  - With the verified cache I recomputed the P5, P6, R4, P16 and P20 checks.
- **K sweep.** My own implementation of the rho walk, followed by a bootstrap comparison of the two
  scaling laws.

## Verdict table

| item | verdict | evidence |
|---|---|---|
| M1 claim 4 (K sweep, sqrt(d)) | **partly fixed + new problem** | Counter fixed: `sigma_c_vs_K.csv` now shows 2/5/5/6/5 unresolved seeds at K = 6, 8, 10, 12, 15. `summary.png` draws brackets and no longer says "6/6 resolved". sqrt(d) is now tagged conjecture. The rho walk is implemented as stated: I get 6/6 seeds and median 0.80 at K = 6, 5/6 and 1.01 at K = 8, 4/6 and 1.16 at K = 10, 5/6 and 1.15 at K = 12, and 13/18 for "first run above the bracket is S2S-like". The law chi^2 values reproduce: 0.8/0.6 for K = 4-7 and 23-25/6.2 for K = 4-12 (bootstrap noise; notes 22.5/6.3). **New problem:** see E1-E2. |
| M2 event rule | **fixed** | `final_rise` takes the last upward crossing of 1/2 in the segment that ends at the first sample >= 0.9, as stated. `relabel_last_crossing` replaces the event time only when n_up > 1; this is equivalent for labels. I confirm 15 label changes, all rich -> S2S: chi_b4 3, sigma_chi_b4 8 (4 centred + 4 plain), sigma_K_d2 1, imbalance_b4 3 (chi = 3, alpha 1e-4 to 1e-3). Log-t and linear-t interpolation give identical labels. chi_b4 chi = 12 seed 4 moves 0.01265 -> 0.03394 [0.0288, 0.040]. The floor explanation is withdrawn, and the 4-10x and 2.7x figures check out. |
| M3 tensor kernel | **fixed** | The kernel matrix matches the library to <= 1.2e-16 relative, and delta_K agrees to <= 1.3e-16. This covers sp, ntk, mup, masked two-copy, MPS5, binary8 and the root-scale case, with no permutation needed. The cache agrees with my recomputation for 104/104 runs (max relative difference 2e-13), and theta0 is rebuilt to 3e-16. P5, P6, R4, P16 and P20 all reproduce; values are in the table below. |
| M4 claim 1 wording | **fixed** (REPORT body: see R) | The note now calls the boundary a calibrated consistency check. The per-seed extrapolation / measured ratio is 0.856-1.231 for 53 seeds and 2.217 for chi = 12 seed 4. The stall onset is 1.6-2.2x above P4. |
| M5 claim 2 | **fixed** | Measured sigma_c sqrt(chi) = 0.267, 0.327, 0.271, 0.332, 0.276 at chi = 32-128. -1/3 lies 3.67 SE away for chi >= 16 and 2.75 SE away for chi >= 32 (pooled fits). The scope statement and the citations of Tu et al. and Geiger et al. are present. See the nuance in E3. |
| M6 claim 6 | **fixed** | P20 values reproduce from the copied runs. Per-seed slopes now carry errors. "Widest" is withdrawn. A heavy root is lazy from alpha = 10 (3/6-4/6 seeds; 6/6 at alpha = 100). Minor slip in E6. |
| M7 section 2.x (two-copy) | **fixed, two slips** | Boundaries 0.817 (two-copy) vs 1.036 (centred) at chi = 36; rho = 5 crossings in 2, 3, 0 seeds. Two slips: "never exceeds 5" and "4-5 of 6" (E4). |
| M8 numbers vs CSVs | **fixed** | Imbalance slope 1.98; S2S slope -0.528 +- 0.053; chi = 9 kernel boundary 3.058 x/ 1.068; threshold ratio 1.972; plain K = 3 boundary 0.170. |
| S1 claim 5 | **fixed** | 4 chi values stated. NTK slope -0.79 +- 0.04 (tensor kernel) and -0.84 (probe kernel). "Approaching -1" is withdrawn. N = 4 only. |
| S2 small items | **fixed** | K = 7 "0.54" is now a bracket. Floor-ending runs count as converged. max_steps cutoffs are listed as a stall cause. Section 1.3 now says "sub-exponentially". |
| S3 code/README | **fixed (not executed end to end)** | Workers use a spawn pool. The raw root is set by $REGIME_MAP_OUT, --out or --raw; the default is still /root/outputs. Step 2 documents its output-directory argument and still refuses to overwrite, by design. `lazy_pred_b8.py` takes an output directory and refuses to overwrite. |
| R REPORT | **partly fixed** | The addendum is correct, but summary items (1), (4), (5) and (6) and line 93 still state the withdrawn claims (E7). |
| check: event rule as stated | **yes** | See M2. |
| check: tensor kernel 2e-16 | **yes** | <= 1.2e-16 (see M3). |

## Key numbers recomputed

All of these come from my own code on `data/regime_map`; "notes" means `notes/regime_map.md`.

**Relabel**

- Labels changed: 15, all rich -> S2S, in 3/8/1/3 runs across the experiments listed in M2. Same as the notes.
- Runs with repeated crossings of 1/2: 130 of 9,094 runs. 9,094 is every library run except
  `large_chi_b8`. Of the 130:
  - 116 have a transient before a component is learned;
  - 14 have repeated crossings only in a component that is never learned (stalled runs). See E5.

**S2S boundaries (binary N = 4)**

- chi_b4, chi = 12, seed 4: 0.0339 [0.029, 0.040]. Same as the notes.
- Table 2.1 at chi = 12: 0.0745 x/ 1.334 over 6 seeds (pre-fix: 0.0884 x/ 1.330 over 5). At chi = 4: 0.0987 x/ 1.514. Same as the notes.
- sigma_chi_b4 rho = 3 crossings: chi = 4 seed 3 moves 0.0290 -> 0.0545; chi = 12 seed 4 moves NaN -> 0.0317.

**Collapse fits** (pooled fit, then the mean of per-seed fits)

| chi range | pooled slope | -1/3 is off by | per-seed slope | -1/3 is off by |
|---|---|---|---|---|
| chi >= 16 | -0.5276 +- 0.0530 | 3.67 SE | -0.528 +- 0.073 | 2.67 SE |
| chi >= 32 | -0.5479 +- 0.0780 | 2.75 SE | -0.548 +- 0.074 | 2.90 SE |
| chi >= 9 | -0.5524 +- 0.049 | — | — | — |

**Lazy boundary with the tensor kernel** (per-seed medians)

| chi | 9 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| boundary | 2.99 | 2.31 | 1.857 | 1.378 | 1.131 | 0.937 | 0.821 | 0.661 | 0.5708 |
| vs P5 bracket | +6.3% | +2.9% | +1.5% | inside | inside | inside | inside | inside | -0.06% (bracket [0.5711, 0.5773]) |

- sigma_lazy sqrt(chi) = 6.40-6.56 for chi >= 32.
- Tensor-kernel slope: -0.504 +- 0.011 for chi >= 32 and -0.553 +- 0.013 for chi >= 16.

**Two-copy lazy boundary (P6) and kernel checks**

- Two-copy boundaries 0.895, 0.651, 0.453 at chi = 32, 64, 128, against P6's 0.912, 0.645, 0.456.
- Two-copy / centred ratio: 0.791, 0.794, 0.794.
- R4 ratio over 43 runs: median 0.957, range 0.69-1.82.
- Probe / tensor kernel change near the lazy boundary: median 0.783, range 0.54-1.11.

**Large chi (P16)**, centred, sigma = 1.5, chi = 16-128 (24 runs per fit)

| parameterization | slope of delta_K vs chi | levels |
|---|---|---|
| SP | -3.018 +- 0.064 | — |
| NTK | -0.791 +- 0.038; local slope -1.011 between chi = 64 and 128 | 0.2505 and 0.1243 at chi = 64, 128 |
| muP-TTN | +0.085 +- 0.024 | 5.13, 5.66, 6.07, 6.23 |

- At sigma = 1.0 the slopes are -2.576 (SP), -0.620 (NTK) and +0.089 (muP-TTN).
- Plain model at sigma = 1.5: -3.06, -0.83 and +0.095.

**Imbalance check point (P20)**

- Tensor-kernel medians: 1.12e-4, 9.93e-4, 0.0096, 0.052, 0.221, 0.515, 0.615, 0.598, 0.595.
- Per-seed local slopes: 1.97 +- 0.01, 1.85 +- 0.06, 1.55 +- 0.11, 1.22 +- 0.09.
- Saturation across seeds: 0.51-0.75.
- Labels: lazy 6/6 at lambda <= 0.3; at lambda = 1, 5 rich and 1 lazy.

**K sweep, floor-stalled runs classified by rho** (medians over resolved seeds)

| K | median | resolved seeds | how the resolved brackets end |
|---|---|---|---|
| 6 | 0.80 | 6/6 | — |
| 8 | 1.01 | 5/6 | label or rho crossing (rho <= 3) |
| 10 | 1.157 | 4/6 | values 0.895, 0.895, 1.418, 1.418; 2 end at a crossing, 2 at no-learning or kernel-like stalls |
| 12 | 1.151 | 5/6 | values 1.151 x4, 1.436; all 5 end at stalls |

## New problems

- **N1 (medium): K = 10 and 12 medians are not S2S -> rich crossings.** See E1-E2.
  - K = 12: all 5 resolved seeds end at a run that learned nothing in its budget (n_events = 0, L/L0 =
    0.92-0.99). This is the "no-learning" or "kernel-like" stall of `fix_round1.RESOLVED`. 0/6 seeds reach
    rho <= 3.
  - K = 10: only 2 seeds end at a crossing (both 0.895). The median 1.157 is the midpoint between two
    clusters, 0.90 and 1.42, and no seed lies there.
  - Laws: if only label- or rho-terminated brackets count (K = 4-10; K = 10 with 2 seeds; K = 12
    dropped), sigma^K = const has chi^2 7.7 and sigma^K/sqrt(D) has 6.9, with 5 dof. The two laws are
    again indistinguishable. "Law A fails (chi^2 22.5)" exists only because of the stall-terminated
    brackets.
- **N2 (low): the REPORT summary still states the withdrawn claims** (E7).

## Errata (exact corrections for anyone citing notes/regime_map.md)

- **E1 (sections 0, 2.3, 3.2 item 4).** Do not cite the medians "K = 10: 1.16; K = 12: 1.15" or the claim
  "the measured sigma_c exceeds 1 at K = 10-12" as S2S boundaries.
  - Replace with: "MPS 10/12, centred, chi = 4 (deficient bonds): S2S-like floor-stalled runs (rho > 3)
    persist up to sigma ~ 1.0 (5/6 seeds) or 1.3 (1 seed) at K = 12. The next runs learn nothing within
    the budget. No S2S -> rich crossing is observed at K = 12 (0/6 seeds) and 2/6 at K = 10 (both 0.90)."
  - The K = 8 value (1.01, 5/6 seeds) does rest on rho crossings. It still refers to deficient bonds and
    floor-stalled runs.
- **E2 (sections 2.3, 3.2 item 4).** Drop "with the rho-classified K = 8-12, sigma^K = const fails
  (chi^2 22.5, spread 98x)".
  - Correct: with only label- or rho-terminated brackets the two laws are indistinguishable on K = 4-10
    (chi^2 7.7 vs 6.9, 5 dof), as on K = 4-7 (0.8 vs 0.6, 3 dof).
- **E3 (sections 0, 2.1b, 3.2 item 2).** "sigma^K chi^(K-1) disfavoured at 3.7 SE (chi >= 16)" uses the
  pooled fit, which treats the 6 seeds at each chi as independent although they share teachers across chi.
  - The per-seed fits give 2.7 SE (chi >= 16) and 2.9 SE (chi >= 32).
  - Cite "2.7-2.9 SE (per-seed fits; 3.7 SE pooled)".
- **E4 (section 2.x, 3.2 item 7).**
  - "min_i rho_i never exceeds 5 there" is false at chi = 18 and 24: the maximum min_rho at sigma <= 0.016
    is 7.1, 14.4 and 4.8 at chi = 18, 24, 36. This is consistent with the 2, 3, 0 seeds that have a rho = 5
    crossing, as the same sentence says.
  - "S2S at 0.010 and 0.016 in 4-5 of 6 seeds" should be: 5, 4, 5 seeds at sigma = 0.010 and 4, 3, 2 seeds
    at 0.016 (chi = 18, 24, 36).
- **E5 (section 3.2 relabel).** "130 of the 9,094 runs have a transient above 1/2 before the component is
  learned" should be: 116 runs have such a transient. 14 more have repeated crossings only in a component
  that is never learned (all stalled). 9,094 excludes `large_chi_b8`. The label counts are unaffected.
- **E6 (minor).**
  - Two-copy / centred lazy-boundary ratio at chi = 64: 0.794, i.e. 0.79 not 0.80.
  - "2-6% above the bracket at chi = 9-16": it is 1.5-6.3% (+6.3, +2.9, +1.5%; chi = 9 has only 3 seeds with
    a tensor boundary).
  - "the probe kernel gives 0.68-0.83 of the tensor value at chi >= 32" (2.5) is the range of per-(lr, chi)
    medians; single runs span 0.50-1.05.
  - P20 saturation, medians 0.595-0.615: say "at the lower edge of P20's 0.6-0.7".
  - Imbalance map at lambda/sqrt(chi) = 0.1, alpha = 1: 3 lazy / 3 rich, a tie, not a lazy majority.
- **E7 (REPORT).** Summary items (1), (4), (5), (6) and line 93 of `agent-runs/regime-map/REPORT.md` still
  state the withdrawn claims:
  - "every boundary is predicted quantitatively ... 0.97-1.08x", "P5 to 4-8%";
  - "sigma_c -> sqrt(d), not 1 (H4 refined)", "derived from theory R9";
  - "chi^-0.84 ... as P16 predicts";
  - "S2S is widest near lambda = sqrt(chi)".

  Cite the notes' section 3, not the REPORT body. Strike-through or annotate those lines.

## Safe to cite (final claims, after this verification)

Scope for all: whitened inputs, exact population loss, odeco teacher, fan-in init, gradient flow, 6 seeds per
point. N = 4 means binary N = 4, d = 3, weights [3, 2, 1]. CSV paths are relative to
`experiments/regime_map/results/`.

| # | claim | tag | known / new | source CSV |
|---|---|---|---|---|
| 1 | N = 4: per-seed rho ~ sigma^-(1.02..1.05) on the S2S side (53/54 chi_b4 seeds: -1.015 to -1.031), predicted -(K-2) = -1. Transient seeds (4) are steeper. | empirical (confirms R9 first gap, derived) | known form (depth-K escape); TTN check new | `fix1/s2s_per_seed_chi_b4.csv`, `fix1/table_2_1_s2s.csv` |
| 2 | Deep-S2S event widths 0.466 / 0.798 / 2.010 for s = 3, 2, 1, matching the aligned-mode theory. | empirical (confirms derived) | known | notes 2.y |
| 3 | N = 4 S2S boundary per seed: 0.237, 0.099, 0.137, 0.136, 0.075, 0.072, 0.067, 0.043 at chi = 3-36. Ratio rho 2/5 is 2.13-2.34. | empirical | new (TTN) | `fix1/table_2_1_s2s.csv` |
| 4 | Lazy boundary (tensor kernel) inside P5's bracket at chi = 24-96 and 0.06% below it at 128. sigma_lazy sqrt(chi) = 6.40-6.56 vs 6.50. Slope -0.504 +- 0.011 (chi 32-128, 0.6 decade). The probe kernel gives ~5% lower boundaries. | empirical, confirms derived P5 | new (TTN) | `fix1/lazy_summary_chi_b4.csv`, `fix1/lazy_slopes_chi_b4.csv` |
| 5 | Two-copy lazy boundary = 0.79 x centred (P6: 2^-1/3), chi = 32-128. The kernel change is second order (1362 vs 1372). | empirical, confirms derived R5/P6 | new (TTN) | `fix1/lazy_summary_chi_b4.csv` |
| 6 | Threshold sensitivity: delta_K 0.05 -> 0.2 moves the lazy boundary by 1.30-1.35 at chi >= 32 (P7) and by 1.97 at chi = 9. | empirical | new (TTN) | `fix1/lazy_summary_chi_b4.csv`, `sensitivity_per_seed_sigma_chi_b4_centred.csv` |
| 7 | Both N = 4 boundaries scale ~ chi^-1/2: S2S slope -0.53 +- 0.05 (chi 16-128) and lazy slope -0.50 (tensor kernel, chi >= 32). sigma^K chi^(K-1) is disfavoured at 2.7-2.9 SE (per-seed fits). Rich window 18-23x in sigma. Under one decade of chi; N = 4 and fan-in only. | empirical | known idea (Tu et al. 2024; Geiger et al. 2020); TTN measurement new | `fix1/slopes_vs_chi_chi_b4.csv`, `fix1/collapse_chi_b4.csv` |
| 8 | Below the cut dimension there is no lazy regime. Centred runs stall at the Eckart-Young floor of W_0 + W* (44/44 runs, N = 4). | derived R7 + empirical | new (TTN) | `boundaries_sigma_chi_b4_centred.csv` |
| 9 | Large chi, N = 4, tensor kernel, chi 16-128: SP delta_K ~ chi^-3.02 +- 0.06. NTK levels 0.250 and 0.124 at chi = 64, 128 (slope -1.01), against R4's 0.240 and 0.120. muP-TTN delta_K -> 6.2 (5.1-6.2). | empirical, confirms P16 | new (TTN muP) | `fix1/large_chi_slopes_tensor.csv`, `fix1/large_chi_levels.csv` |
| 10 | Imbalance at chi = 16, alpha = 3.375: delta_K ~ lambda^2 as lambda -> 0 (local slopes 1.97 +- 0.01, 1.85 +- 0.06), saturating at 0.60 (median, 0.51-0.75). A light root lowers the lazy boundary by up to ~100x in alpha (chi = 24); a heavy root is lazy from alpha = 10, as at balance. | empirical, confirms P20 | new (TTN) | `fix1/imbalance_P20.csv` |
| 11 | K sweep (centred, d = 2, chi = 4): resolved label boundaries 0.17, 0.49, 0.61, 0.68 (4/6 seeds), 0.75 at K = 3-7. sigma^K and sigma^K/sqrt(D) are indistinguishable on K = 4-7 (and on K = 4-10 with crossing-terminated brackets). At K >= 8 there are only brackets or floor-limited values; K = 8: 1.01 (5/6, deficient bonds). | empirical | open question (H4) | `sigma_c_vs_K.csv`, `fix1/k_sweep_per_seed.csv`, `fix1/k_sweep_laws.csv` |
| 12 | sigma_c(K) -> sqrt(d) (or -> 1) as K -> infinity. | conjecture | open | none |
| 13 | At chi = teacher rank and K >= 7, most runs stall (dead routes under fan-in init). Gauge-balanced init removes dead routes but not the stall. | empirical (36 + 18 runs) | known mechanism (theory P14) | notes 2.2 |
| 14 | Two-copy model at large chi: events 1 and 2 merge, so it is not a cross-check of the S2S boundary. | empirical | new | `sensitivity_per_seed_sigma_chi_b4_twocopy.csv` |
