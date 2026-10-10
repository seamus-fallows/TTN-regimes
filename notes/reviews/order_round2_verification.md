# Verification of the `order` fix round (round 2) — 2026-10-10

Verifier: independent agent; no file other than this one edited, nothing committed. Inputs: review
`notes/reviews/order_round1.md`, the lead's fix list (7 MUST, 5 SHOULD), and the merged output (commit
5804535: `notes/order.md` sections 0, 5-11, `agent-runs/order/REPORT.md`, `experiments/order/`). Raw data:
`data/order/` (read only). My scripts are in the session scratchpad (`verify_order/v_multi.py`, `v_h1b.py`,
`v_sigma.py`, `v_sigma3.py`). None of them import code from `experiments/order`. They read the raw JSON lines,
the per-run H1b CSVs and the `.npz` spectra. Compute: at most 5 processes, under 30 CPU-minutes, no new GF runs.

**Overall:** every number I recomputed reproduces, all 7 MUST FIX items are fixed, and the SHOULD FIX items are
fixed. There is one new problem: in R = 3, "GF is less greedy than independent clocks in 6/6 m >= 3 cases
(combined z = -1.9)" is not supported. A paired test shows no effect (E1 below). The n = chi_e wording is now
slightly too pessimistic (E2). The rest are small precision errata. Run classification is rule-based
throughout, with fixed thresholds and the first N wrong-order seeds in seed order. Withdrawn claims are not
restated in the README, and in the REPORT only in dated log lines (E8).

## Verdict table

| # | item | verdict | evidence (my recomputation) |
|---|---|---|---|
| M1 | H1b narrowed to k >= 3, k = 2 branch added | **fixed** | Rule-based classes: binary4 rotated 192/192, MPS4 cores 0,2 rotated 144/144, MPS4 cores 1,2 added 144/144. k = 2: raw lambda_min -0.70704 / -0.89427 / -0.94850 vs -s_1 s_2^((K-2)/K) = -0.70711 / -0.89443 / -0.94868. From the npz spectra at sigma <= 1e-3, exactly 3 eigenvalues lie within 1% of that value in 35-38 of 40 runs (the multiplicity chi_e - 1), and all 40 have 4 negative eigenvalues, the 4th equal to the piggyback value (-0.1789 at rho 0.8). Slope of t1 vs ln(1/sigma) for sigma <= 1e-4: 1.4142 / 1.1180 / 1.0541 = 1/\|lambda\| exactly. a_2 stays 1.00. The wording is consistent in O7, sections 6 and 8, the REPORT, and the report's `rem:A-wrong-order` (which already has the k = 2 branch). |
| M2 | m = 2 law with n; n = chi_e as conjecture | **fixed** (one refinement, E2) | All 17 gamma fits reproduce from raw data (binomial ML, 150-300 seed-bootstrap replicates; e.g. MPS4 adjacent 0.815 +- 0.068 / 0.307 +- 0.016, separated 0.240 +- 0.018 / 0.095 +- 0.006, binary4 within +-0.004 of the CSV). Every fit uses the full measured sigma range, and P = 0 points enter the likelihood. "Still steepening" is withdrawn and n is carried everywhere, with the conjecture tag. Wrong-order seeds are exactly nested across sigma in all 18 m = 2 sets (the 17 fitted plus MPS6 adjacent at rho = 0.7). |
| M3 | R = 3: same subset, exclusions, sigma, floor rule | **fixed, with a new problem (E1)** | Same-subset model, excluded runs' race orders, sigma and d = chi = 4 are all stated, and the counts reproduce. Slice flag: no completed run has a slice ratio below 1e-12 (the minimum before any event in used runs is 8.8e-8), so the 18-19 binary8 runs per rho dropped in round 1 are back (198/200, 196/199 used). But the multi3 "rerun" is outcome-identical to round 1: all 1597 runs have the same status and identical event times as multi2; the slice stop fires only later (median x1.27 in t). The stall labels agree with the reviewer's independent no-stop runs in 9/9 seeds. |
| M4 | R = 4 wording, no-stop check, scope, R10 | **fixed** | multi2: MPS4 289/400, MPS6 396/399, binary8 389/393 incomplete; at the stop MPS4 has 3 of 4 learned in 288/289 runs and MPS6 2 of 4 in 88 + 90 runs. No-stop: all 22 runs reach t = 1e9 with status t_max, 0/11 complete per tree; MPS6 learns 2 of 4 in 7/11 runs, binary8 learns 3 of 4 in 11/11, and the largest component is the unlearned one in 6/11. The R10 cross-reference and scope are stated. |
| M5 | m >= 3 "no strict saddle": scope | **fixed** | Per-seed exponent of \|lambda_min\| vs sigma: MPS4 2.007 +- 0.016 / 1.999 +- 0.014; MPS6 4.218 +- 0.087 (5 seeds; seed 2 stalls at every sigma); binary8 5.25 +- 0.27 (sigma 0.15-0.35). t1 exponents -1.97 / -4.26 / -5.33. No rotation (a_2 >= 0.964 / 0.9996 / 0.995). |
| M6 | d test withdrawn; independent test | **fixed** (precision erratum E3) | The round-1 chi^2 is no longer used. New test: seeds 200000-200999 (d = 2) and 300000-300999 (d = 4), disjoint from every other raw sweep, with equal bond dimensions (binary8 4-4-4-4-4-4, MPS4 2-2-2). P and z reproduce exactly (binary8 0.774/0.774, 0.591/0.572; MPS4 0.788/0.809, 0.606/0.621; \|z\| <= 1.17). |
| M7 | unresolved midpoint Hessians | **fixed** (minor note E6) | binary4 natural: 0/421 unresolved (was 43); MPS4 cores 0,2: 0/240. The new adjacent natural set has 57/150 unresolved. They are near-ties (median gap 0.06), listed in `h1b_fits.csv`, and not used by any claim. |
| S1 | H1b precision; "lambda_min between 0 and the saddle value" | **fixed** | Local slope x \|lambda\| from 1e-2 down to 1e-7: binary4 1.068-1.072, 1.016-1.018, 1.002, 1.0002, 1.0000; MPS4 cores 0,2 0.9995-1.0002; MPS4 cores 1,2 0.9955 -> 1.0000. Per-seed slope at sigma <= 1e-4: 2.5215 +- 0.0017, 3.7568, 5.3901, 10.3654 vs 2.5198, 3.7542, 5.3861, 10.3574. The lambda_min statement holds to within 1.3% (E4). |
| S2 | separated pairs: one fit range, goodness of fit, "consistent with" | **fixed** (E5 adds a better test) | The deviance GOF is uninformative because seeds are shared, as the note says. My seed-bootstrap curvature test (log P quadratic in ln sigma) finds no significant curvature in 12 of 13 non-adjacent sets. The exception is MPS4 0,2 at rho = 0.9 (2.4 SE, local exponent rising toward the prediction). Fits over the three smallest sigma agree with (1-rho)/rho within 1.1 SE in all 13 sets (12 within 1 SE). |
| S3 | common-seed offset; race-curve ranges | **fixed** | MPS offset -0.0184 +- 0.0192 over 102 cells, 76.5% negative. Ranges for m >= 3: GF 0.603-0.882 / 0.505-0.652, race curve 0.658-0.894 / 0.552-0.656. |
| S4 | O8 wording; O5 SE and out-of-sample | **fixed** | K-exponent 0.657, 0.656, 0.594, 0.583, 0.523, 0.530 (chi 1-32). KKT-projection rule: q/(K-2) 0.986-1.247 (above 1 in 22/23), accuracy 0.762-0.920, the same out of sample and at q = K-2 (0.764-0.919). The SE is 1.2-3.6% of q, not "1-9%" (conservative, E7). |
| S5 | code and README (review A bugs 1-4, B12) | **fixed** | `_dir_option` exports ORDER_RAW/ORDER_RES, so spawned workers see the directories. `default_procs` uses the cgroup quota or the CPU count. `run_queue.sh` forwards --raw/--res. The README lists multi2, multi3 and multi_nostop. `multi.yaml` writes `multi2`. tol_check: 500/500 winners equal, MPS8 included, max \|dt1\|/t1 = 1.0e-9. The symmetry check log is complete. |
| new | R = 3 "less greedy than the iid race in 6/6" | **new problem** | See E1. |

## Key recomputations

**m = 2 law** (raw first-event sweeps; binomial ML; SE from a seed bootstrap; all measured sigma used; "tail" =
fit over the three smallest sigma with P > 0; "curv." = coefficient c of (ln sigma)^2 in log P, in SE units,
negative = local exponent rising as sigma falls):

| set (sigma range, seeds) | rho | gamma, full range | tail fit | curv. | n(1-rho)/rho |
|---|---|---|---|---|---|
| binary4 chi 2 / 4 (1e-4..0.1, 1000) | 0.7 | 0.370 +- 0.038 / 0.421 +- 0.044 | 0.39 +- 0.10 / 0.30 +- 0.12 | +0.6 / -0.1 | 0.429 |
| | 0.8 | 0.237 +- 0.020 / 0.248 +- 0.020 | 0.241 +- 0.042 / 0.248 +- 0.045 | +0.6 / +0.5 | 0.25 |
| | 0.9 | 0.103 +- 0.007 / 0.102 +- 0.008 | 0.123 +- 0.016 / 0.106 +- 0.014 | -1.3 / -0.8 | 0.111 |
| | 0.95 | 0.042 +- 0.004 / 0.040 +- 0.004 | 0.047 +- 0.008 / 0.046 +- 0.007 | -0.9 / -0.7 | 0.053 |
| MPS6 ends (0.003..0.1, 400) | 0.7 / 0.8 / 0.9 | 0.457 +- 0.064 / 0.249 +- 0.029 / 0.101 +- 0.015 | 0.36 +- 0.08 / 0.28 +- 0.04 / 0.11 +- 0.02 | +1.7 / -1.0 / -0.7 | 0.429 / 0.25 / 0.111 |
| MPS4 cores 0,2 (1e-5..0.1, 1000) | 0.8 / 0.9 | 0.240 +- 0.018 / 0.095 +- 0.006 | 0.231 +- 0.032 / 0.115 +- 0.011 | +0.3 / **-2.4** | 0.25 / 0.111 |
| MPS4 cores 1,2, adjacent (1e-5..0.1, 1000) | 0.8 / 0.9 | 0.815 +- 0.068 / 0.307 +- 0.016 | 0.79 +- 0.07 / **0.40 +- 0.05** | -0.1 / **-2.4** | 1.0 / 0.444 (n = 4) |
| MPS6 cores 2,3, adjacent (0.003..0.1, 400) | 0.8 / 0.9 | 0.668 +- 0.079 / 0.222 +- 0.028 | 0.78 +- 0.15 / 0.19 +- 0.03 | -1.1 / +1.3 | 1.0 / 0.444 (n = 4) |

The full-range fits of the 13 non-adjacent sets lie below (1-rho)/rho in 12/13, by 0-2 SE except at rho = 0.95
(2.5-3.0 SE) and MPS4 0,2 at rho = 0.9 (2.7 SE). The tail fits agree within 1.1 SE in all 13 (12 within 1 SE). So the full-range
deficit is pre-asymptotic, and the law is better supported at small sigma than the note says.

**R = 3** (multi3; my own code; P(greedy order) on the used runs; model = iid resampling of the same runs' tau;
paired = GF order vs each run's own LO race order on the same runs):

| tree, rho | used / n | P_GF | P_model (z) | own race P | paired: GF greedy & race not / race greedy & GF not |
|---|---|---|---|---|---|
| binary8, 0.8 | 198/200 | 0.227 | 0.277 (-1.56) | 0.237 | 11 / 13 |
| binary8, 0.9 | 196/199 | 0.194 | 0.217 (-0.78) | 0.184 | 10 / 8 |
| MPS4, 0.8 | 195/200 | 0.354 | 0.358 (-0.13) | 0.338 | 10 / 7 |
| MPS4, 0.9 | 195/200 | 0.226 | 0.249 (-0.76) | 0.226 | 4 / 4 |
| MPS6, 0.8 | 91/199 | 0.253 | 0.303 (-1.04) | 0.286 | 2 / 5 |
| MPS6, 0.9 | 97/199 | 0.216 | 0.228 (-0.26) | 0.237 | 4 / 6 |
| total m >= 3 | | | Stouffer -1.85 (indep.), -1.37 (corrected) | | **41 / 43** (McNemar p >= 0.45 in every row) |

binary4 (m = 2): P_GF 0.705 / 0.490 vs model 0.518 / 0.321 (z +5.3 / +5.1), paired 45/13 and 47/13: far more
greedy, as stated. On MPS6, the excluded runs' race-first distribution differs from that of the used runs only
marginally (chi^2 p ~ 0.05 at rho 0.8, 0.2 at rho 0.9). Their race-greedy fraction is the same (0.31 vs 0.29;
0.235 vs 0.237).

**Stall criterion (slice ratio <= 1e-14).** In the 22 no-stop R = 4 runs, the 59 learned components had a
minimum slice ratio of at least 6.7e-6 before their event, so the criterion has no false positive, with a
margin of 8 orders of magnitude. 25 of the 29 unlearned components fell below 1e-14, 17 of them to exactly 0
(underflow with flush-to-zero). The other 4 stayed at 2.9e-7 to 1.2e-2 and would be reported as "unfinished".
No learned component's overlap fell below 1e-25 before its event either, so the round-1 overlap stop did not
create false stalls in these runs. In multi3, the 21 + 15 unfinished MPS6 runs keep the unlearned slice at
1e-11 to 6e-6 to t = 1e9: these are pinned routes, correctly not called dead. The criterion is fixed-threshold
and rule-based. One caveat on its stated rationale ("dead at float64 roundoff"). In the standard teacher
basis, the unlearned slices in the no-stop runs keep falling to 1e-21-1e-35 and then underflow to zero. They
do not stall near 1e-16, so there is no additive roundoff floor: a slice at 1e-14 is a route decaying
exponentially, not one seeded by roundoff. The rule is justified empirically by the gap above. The wording in
the code docstrings, the README and `notes/order.md` section 7.6 is loose but harmless.

## Errata (exact corrections for anyone citing `notes/order.md` or the REPORT)

- **E1 (new problem; O9, section 7.6 "m >= 3" bullet, section 10, REPORT summary and item 5).**
  - Replace "within 1.7 SE of an iid race model per case but less greedy in 6/6 m >= 3 cases (combined
    z = -1.9) ... (conjecture: the coupling after the first escape)" with: "consistent with an iid race model in
    every m >= 3 case (\|z\| <= 1.6). Paired against each run's own leading-order race on the same runs, GF is
    no less greedy: in 41 runs GF is greedy and the race is not, in 43 the reverse. The coupling after the first
    escape changes the full order in 11-30% of runs but not its greediness."
  - Why the original is unsupported:
    - The 6/6 negative sign comes from the realized seed sets: their own race orders are already below the pooled
      iid model in 5/6 cases.
    - The Stouffer z treats rho = 0.8 and 0.9 as independent, but they use the same 200 seeds per tree (the
      GF-greedy indicators correlate 0.73-0.90). Allowing for that gives z = -1.4.
  - Drop the open issue "why GF is slightly less greedy than independent clocks" from section 10.
- **E2 (O6, sections 7.4b and 10, REPORT item 2).** "n = chi_e is not reached by sigma = 1e-5 ... no steady
  approach" is too pessimistic for MPS4 at rho = 0.9.
  - There the local exponent rises as sigma falls: the curvature of log P in ln sigma is -0.013 +- 0.005
    (2.4 SE).
  - The fit over sigma = 1e-3..1e-5 gives 0.40 +- 0.05. That is consistent with n(1-rho)/rho = 0.444 and 5 SE
    above (1-rho)/rho.
  - MPS6 at rho = 0.9 (0.19 +- 0.03 for sigma >= 0.003) and both rho = 0.8 cases (0.79 +- 0.07,
    0.78 +- 0.15) stay below the n = chi_e value.
  - Keep "n = chi_e: conjecture". Cite it as "supported at the small-sigma end on MPS4 (rho = 0.9), not reached
    on MPS6 down to sigma = 0.003".
- **E3 (section 11 item 6, section 7.1b, REPORT result 1).**
  - "No effect at the 0.04 level" should read "differences larger than about 0.06 are excluded at 95%". The CIs
    on the difference are [-0.037, +0.037], [-0.062, +0.024], [-0.014, +0.056] and [-0.028, +0.058].
  - The two-sample SE is 0.018-0.022, not 0.018-0.020.
- **E4 (section 7.3, natural runs).** "lambda_min there lies between 0 and the saddle value for sigma <= 1e-2"
  should read "... at or above the saddle value to within 1.3%". 26 of 296 binary4 runs and 12 of 237 MPS4 0,2
  runs sit 0-1.3% below it.
- **E5 (section 7.2 and O6, optional strengthening).** Add: "fits over the three smallest sigma agree with
  (1-rho)/rho within 1.1 SE in all 13 non-adjacent sets; the full-range fits are low in 12/13 (pre-asymptotic)".
  This replaces the uninformative deviance p-values as the goodness-of-fit statement.
- **E6 (section 7.3).** "All 421 + 240 runs now have a resolved minimum-speed point": add "the adjacent (k = 2)
  natural set has 57/150 without one (near-ties, median gap 0.06; not used)".
- **E7 (O5, section 8).** The bootstrap SE of q is 1-4% (KKT proxy, m >= 3 settings), not 1-9%.
- **E8 (REPORT log).** The 01:13 entry ("steepening toward n = chi_e = 4") and the 21:25 entry ("+-1.5%")
  restate withdrawn round-1 wording. They are dated history; mark them superseded if the log is cited.
- **E9 (section 7.3, wording only).** "A power law fits 10-80x worse" should be 11-88x for the rotation sets
  (104-110x for the adjacent set).
- **E10 (section 7.6, R = 4).**
  - "Not learned by t = 1e9" is observed only for the t_max runs and the 22 no-stop runs. For the other multi2
    runs (round-1 overlap stop) it is inferred from those 22 runs and from the reviewer's MPS4 no-stop check.
  - "Learned at the stop" counts (MPS4 3 of 4 in 288/289; MPS6 88 + 90 runs with 2) are at the stop time.
  - Also state that the multi3 rerun reproduces multi2 exactly, run for run; only the roundoff flag changed.
- **Outside the order files (for the lead).**
  - CLAUDE.md, key findings, "Experiments (interim, pending review): order": write "on non-adjacent matrix-like
    pairs (k >= 3; binary4 = depth-3 DLN)", and add "adjacent pairs (k = 2) add the larger component without
    forgetting".
  - The quoted gamma = 0.42/0.25/0.10/0.04 are the chi = 4 full-range fits.
  - `report/sections/04_saddle_to_saddle.tex` (TODO comment of `sec:s2s-order`) still describes round-1 H1b
    ("rotates the learned component in time O(log 1/sigma)"). Use the narrowed statement when the section is
    written.
  - The report's `rem:A-wrong-order` and `prop:A-matrix-like` are consistent with the fixed note. The GF check
    at line ~365 of `A_derivations.tex` can cite E2 and E5.

## Safe to cite (final claims, after the errata above)

Scope unless stated: odeco or pattern teachers in the standard basis, fan-in Gaussian init, whitened inputs,
full GF in float64. Known/new follows `notes/literature.md` and `notes/reviews/citation_check.md` part b
(rows B2, B4, B9). CSVs are in `experiments/order/results/`.

| # | claim | status | known / new | source |
|---|---|---|---|---|
| C1 | Near the origin the order is set by the leading-order flow of g = <W, W*>, which is exactly scale-invariant. For m >= 3 the first-event winner does not depend on sigma: P is identical for sigma 0.003-0.1 (MPS4) and 0.01-0.1 (binary8), and >= 99% of per-seed winners are unchanged | derived + empirical | mechanism known (Kumar & Haupt 2025a); new for the TTN order | `sigma_table.csv`, `sigma_fits.csv` |
| C2 | m-count rule. m >= 3 cores carrying differing sites: every component is a local max of g on the sphere, so the order is random. m = 2: only the largest is, so the order is greedy as sigma -> 0 | derived | new for trees; DLN case known (Jacot et al. 2021, Thm 2: limit only) | `main_summary.csv` |
| C3 | Odeco teacher on an MPS: exact race, winner = argmin_r tau_r/s_r of independent LO blow-up times (40800/40800 GF runs). Approximate on binary8 (96-99.5%) | derived (exact at LO) + empirical | new (related: Ben Arous et al. 2024, spiked tensors) | `main_summary.csv` (race_agree) |
| C4 | P(larger first) at sigma = 0.01 (400 seeds, SE <= 0.025): 0.60-0.88 at rho = 0.5 and 0.51-0.65 at rho = 0.8 over the m >= 3 settings. Falls toward 1/2 with K, rises with chi. On MPS, GF sits 0.018 +- 0.019 below the pooled race curve (shared-seed fluctuation) | empirical | new | `main_summary.csv`, `main_race_offset.csv` |
| C5 | On an MPS, d enters only through the bond caps. At equal bonds, d = 2 vs 4 differ by <= 0.021 (independent seeds; differences > 0.06 excluded at 95%) on MPS4 and binary8 | derived (MPS); empirical (binary8, one chi) | new | `d_test.csv` |
| C6 | Uncapped bonds: sd(log tau) ~ chi^-(0.48-0.54) per tree, and the K-dependence tends to sqrt(K-2) at large chi (K-exponent 0.66 -> 0.52). The order becomes greedy only slowly (GF checked at chi = 16, 32) | empirical | new; analogue of over-parameterized CP (Ge et al. 2021, modified GF) | `chi_scan.csv`, `chi_scan_K_exponent.csv`, `chi_gf.csv` |
| C7 | KKT-projection selection rule argmax_r s_r X_r^q, q = (0.99-1.25)(K-2): predicts 76-92% of winners on m >= 3 trees (same out of sample and at q = K-2). The LO race is exact | empirical | related: Ben Arous et al. 2024 (weight x overlap^(p-2)) | `main_rules.csv` (proxy kkt_proj) |
| C8 | Non-adjacent m = 2 pairs (binary4, MPS6 ends, MPS4 cores 0,2): P(smaller first) is consistent with C sigma^((1-rho)/rho) for rho = 0.7-0.95. Fits over the three smallest sigma agree within 1.1 SE in 13/13 sets; full-range fits run from 25% low to 7% high (12/13 low; pre-asymptotic) | derived (heuristic exponent) + empirical | new (the power law in sigma); limit P -> 0 known for DLNs (Jacot et al. 2021) | `sigma_fits.csv`, `sigma_table.csv` |
| C9 | Adjacent m = 2 pairs (shared bond chi_e = 4) are steeper: gamma = 0.82 +- 0.07 / 0.31 +- 0.02 (MPS4) and 0.67 +- 0.08 / 0.22 +- 0.03 (MPS6) at rho = 0.8 / 0.9. The asymptotic exponent n(1-rho)/rho with n = chi_e is supported only at the small-sigma end on MPS4, rho = 0.9 (E2) | empirical; n = chi_e **conjecture** | new | `sigma_fits.csv` |
| C10 | Wrong-order seeds are nested across sigma: every seed that is wrong at a smaller sigma is wrong at all larger sigma (18/18 m = 2 sets) | empirical | new | `sigma_fits.csv` (nested_frac_min) |
| C11 | H1b, m = 2 with k >= 3: the wrong saddle is strict with lambda_min = -(s_1-s_2) s_2^((K-2)/K) (midpoint Hessian within 0.05% of it). From it, t1 = ln(1/sigma)/\|lambda\| + const (local slope x \|lambda\| -> 1.0000). The learned index rotates, the smaller component is forgotten (a_2 < 6e-4) and relearned after ~sigma^-(k-2). Checked on binary4 (K = 3) and MPS4 cores 0,2 (K = 4) | derived + empirical | known for binary4 = DLN (strictness: Achour et al. 2024, Thm 7); new for general trees (rate, forget-and-relearn) | `h1b_saddle_b4.csv`, `h1b_saddle_mps4_sep.csv`, `h1b_fits.csv`, `h1b_local_slopes.csv` |
| C12 | H1b, m = 2 with k = 2 (adjacent cores): strict saddle with lambda = -s_1 s_2^((K-2)/K), multiplicity chi_e - 1. The larger component is added on a spare bond index after ln(1/sigma)/\|lambda\|, and nothing is forgotten. Checked on MPS4 cores 1,2 only | derived + empirical | new (citation_check B2: explicit rate and multiplicity) | `h1b_saddle_mps4_adj.csv`, `h1b_fits.csv` |
| C13 | Natural wrong-order runs, k >= 3: rotation in 296/301 (binary4) and 237/240 (MPS4 0,2). The correction takes O(1/\|lambda\|): \|lambda\|(t1-t2) = 0.98-1.38, flat in sigma 1e-4..1e-2 (slope -0.02 +- 0.03 per ln(1/sigma)), and no wrong plateau is resolved. For k = 2 it is a near-tie (\|lambda\|(t1-t2) = 0.19-0.28) | derived + empirical | new | `h1b_natural_b4.csv`, `h1b_natural_mps4_sep.csv`, `h1b_natural_mps4_adj.csv` |
| C14 | m >= 3: no strict saddle after a wrong order. \|lambda_min\| ~ sigma^(k-2) (MPS4 2.00 +- 0.02, MPS6 4.22 +- 0.09, binary8 5.2 +- 0.3 over a factor 2.3 in sigma), no rotation, and the wrong order is kept. Occasionally the route dies (1/40 natural MPS4 runs; 1/6 MPS6 seeds) | derived + empirical | new (citation_check B2) | `h1b_saddle_mps4.csv`, `h1b_saddle_mps6.csv`, `h1b_saddle_b8.csv`, `h1b_natural_mps4.csv` |
| C15 | R = 3 (d = chi = 4, one sigma per tree, 0.05-0.25): for m >= 3, GF orders are consistent with an iid race and no less greedy than each run's own LO race (41 vs 43 discordant runs). The own race predicts the first event in 97-100% of runs and the full order in 70-89%. binary4 (m = 2) is far more greedy (z ~ +5). MPS6 is conditional on completion (91-97 of 199 runs) | empirical | new | `multi3_summary.csv`, `multi3_perms.csv`; paired counts in this note |
| C16 | R = 4 = chi = 4 (one sigma per tree, fan-in init): most runs do not learn all four components. MPS4 289/400 (K = 4, so R10's "K >= 5" is not sharp); MPS6 and binary8 0/11 each without any stall stop to t = 1e9; MPS6 learns only 2 of 4 in 7/11 | empirical | related to `theory_regimes` R10 | `multi2_summary.csv`, `multi_nostop_summary.csv` |
| C17 | Numerics: rtol 1e-9 vs 1e-11 give the same winner in 500/500 inits (5 trees), \|dt1\|/t1 <= 1.0e-9. The slice-ratio stall rule had no false positive in 22 no-stop runs | empirical (method) | - | `tol_check.csv`, `multi_nostop_summary.csv` |

Not safe to cite:
- "GF is less greedy than independent clocks" (E1).
- "n = chi_e is not supported by GF" or "is reached" (E2).
- The round-1 d chi^2 test.
- The R >= 3 statistics outside d = chi = 4 and the single sigma per tree.
