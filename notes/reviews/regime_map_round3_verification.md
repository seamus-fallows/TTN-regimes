# regime-map round 2 (merged as 5fece11): independent verification

Verifier: independent agent, 2026-10-10. Scope: notes/regime_map.md sections 0, 4.1-4.8 and 5;
agent-runs/regime-map/REPORT.md items (j) and (k); experiments/regime_map/results/round2/ and
results/tensor_relabel/; the lead's briefs (round2_regime-map.txt, r2_extra_regime-map.txt).

Raw data, read only: data/regime_map_r2/ (r2_plain 1,070 rows, r2_centred_full 360 rows,
r2_plain_discarded_eventbug 22 rows + 112 trajectories), data/regime_map_r2_pilot/, and data/regime_map/
(round-1 sweeps and tensor_kernel/ cache). Copy check: 13,325 remote files = local, same bytes.

Method. All checks were recomputed with verifier-written code, not the agent's analysis:
- per-run classes and per-seed boundaries re-implemented from notes 4.1 and 4.6;
- the pre-registered WLS fits, with and without Birge scaling;
- the domain limit and its large-K limit, by simulating 300 fan-in TTN draws per point up to K = 127;
- the relabel recount;
- delta_K of the tensor kernel, recomputed with torch.func.jacrev for 89 runs.

The agent's helper functions were reused only for naming (`rid_of`), config rebuilding (`build`) and
the aligned threshold u_0c (`aligned_boundary_a0`). Scripts are in the session scratchpad (`verify/v_*.py`); the lead
may copy them to notes/reviews/scripts/. Four processes or fewer, about 30 min.

## Verdict table

| # | claim | verdict | evidence |
|---|---|---|---|
| 1 | Hypotheses and decision rule committed before any round-2 run | **confirmed** | f57a28c at 12:26:25 UTC. The first round-2 raw file is a pilot at 12:27:57; the first arm-P file is at 12:32:29 and the first arm-C file at 12:50:50 (same remote clock; mtimes preserved by the copy). Sections 4.1-4.5 are byte-identical to f57a28c. The header's "12:50 UTC" is wrong (E7). |
| 2 | Analysis follows the registered rule | **confirmed, with one deviation** | Cells with n_res >= 4, WLS on cell means with bootstrap SEs, joint / per-d / contrast / one-parameter laws, the three-way rule, the no-shrinkage subset and rho* = 2, 5 are all as registered. Deviation: Birge inflation of the SEs was not registered, and it decides rho* = 5 (E6). Post-hoc additions: the Tobit fit (flagged), the median variant (flagged in the code, not in the 4.7 table), the censored-vs-law comparison and the shrinkage diagnostic (both exploratory). |
| 3 | 113 discarded files excluded for a rule-based reason | **confirmed; no effect** | 22 rows + 112 trajectories from the first launch (12:32:29-12:34:06), before the amendment commit (12:34:27). They lack the `a0` field that the domain rule needs. They were excluded wholesale, not by outcome. All 112 configurations were re-run: the re-runs are bit-identical (same grid, same final overlaps and t_end) and the 22 rows have the same rho to every digit. |
| 4 | "At K >= 8, 85 of 96 seeds stay S2S up to the domain limit" | **partly** | 85/96 are censored from above: 76 'domain' and 9 'open' (the budget ended with the run still S2S-like, at lo = 0.79-0.995 of the seed's own domain limit). Statuses reproduced exactly (0 mismatches in 156 seeds; sigma_c to 4e-6). See E2. |
| 5 | "... and at d = 3 every seed" | **refuted** | d = 3, K = 10, seed 0 crosses at sigma_c = 1.08 (rho 35.6 at 0.999, 2.86 at 1.085; domain limit 1.82). d = 3, K >= 8: 40 domain + 7 open + 1 crossing (E1). |
| 6 | The domain scale tends to sqrt(d) | **approximately at K <= 15; wrong as a limit** | log\|z\| falls by about 0.14 per core, so sigma_dom -> ~1.15 sqrt(d): 1.63 (d = 2) and 1.99 (d = 3). Measured medians at K >= 11 are 1.50 and 1.80, i.e. 1.04-1.06 sqrt(d) (E4). |
| 7 | Lower bounds lie above both laws | **confirmed** | 97/97 bounds exceed the H_A and H_B extrapolations (gamma fitted on the resolved cells). The smallest margin is 3% over H_B and 27% over H_A. 95/97 exceed 1. The range is 1.07-2.03 at K >= 8 and 1.18-2.03 at K >= 11 (E12). |
| 8 | sigma_c does not tend to 1; H4 rejected for the plain model | **confirmed within scope** (binary, chi = 4, teacher (3, 2), K <= 15) | All censored bounds at K >= 8 exceed 1. The registered fit also puts H_A 5.5 SE away. Caveats: the rejection uses the domain censoring (amendment 4.6) and a comparison that was not registered. 11/96 seeds at K >= 8 do cross below sqrt(d) (E5). |
| 9 | H_W "confirmed by core shrinkage" | **overstated: consistent, not confirmed** | The shrinkage follows from an exact identity. Its growth with x_W matches H_W, but nothing links it causally to the boundary. H_W's registered signature "arm C stays near beta = 1" failed: beta_C = -0.07 +- 0.51 (E3). |
| 10 | Centred, full bonds: beta = -0.07 +- 0.51, unresolved | **confirmed** | Recomputed -0.070 +- 0.512 (8 cells; Birge = 1): H_A at 0.14 SE, H_B at 2.09 SE, hence unresolved. Medians and CIs per cell reproduced. |
| 11 | Registered arm-P fits | **confirmed** | rho* = 3: 1.211 +- 0.221, H_B favoured (H_A chi^2 36/8, H_B 7.0/8). No-shrinkage subset: 0.613 +- 0.506. rho* = 2: 0.721 +- 0.561. rho* = 5: 1.555 +- 0.294, "H_B favoured" only with Birge (E6). |
| 12 | Censoring logic (bound vs measurement) | **sound, with caveats** | 59 measured values and 97 lower bounds (88 'domain', 9 'open'). Bounds hold at the sampled sigmas only (E13), and 14 rest on the lower-bound rule (E14). |
| 13 | Shrinkage table (0.1% ... 92%) | **confirmed** | Bin medians reproduced. Four runs re-measured from their trajectories match exactly. Arm C has median 0 above x_W = 0.01. |
| 14 | Relabel: 65 of 5,298 labels change | **confirmed** | Independent recount: 9 + 9 + 6 + 15 + 4 + 22. 59 runs go lazy -> non-lazy and 6 plain runs go non-lazy -> lazy. delta_K recomputed for all 65 changed runs and 24 random cached runs agrees to 5e-14, with all 89 on the same side of 0.1. |
| 15 | Lazy boundary moves up by 4-9% ("about 5%") | **confirmed for centred; 1-9% overall** | Per-(model, chi) median tensor/probe ratios reproduced to 5e-6: centred 1.04-1.09 (chi 12-128), plain 1.04-1.07, two-copy 1.01-1.07 (chi >= 9); 1.14-1.15 at chi = 6-9 (E18). "About 5%" is a fair central value. |
| 16 | Tags honest; withdrawn claims kept out | **mostly** | The four withdrawn claims appear only in the history lists (REPORT header, notes 3.2). H_W is tagged three ways: "confirmed" in 4.8 and the REPORT, "heuristic" in 4.7, which is not a project tag (E3). |

## Errata

Major (the claim changes):

- **E1. d = 3 "all" seeds.** Section 0 says "(d = 3: all)". Section 4.8 says "at d = 3 every seed". REPORT (j) says "all at
  d = 3, K >= 8". Section 4.7 says "at d = 3, K >= 8 no seed has an S2S -> rich crossing below the domain limit".
  - All four are false. d = 3, K = 10, seed 0 has a clean crossing at sigma_c = 1.08 (rho 504, 213, 35.6, 2.86, 0.33 at
    sigma = 0.78-1.13), well below its domain limit of 1.82. The 4.7 table itself lists it as "1 / 5 (1.08)".
  - Correct to: "at d = 3, K >= 8, 47 of 48 seeds are censored (40 at the domain limit, 7 open), and one crosses at 1.08".
- **E2. "85 of 96 (97/156) stay S2S up to the domain limit".** This appears in REPORT summary item 4, the REPORT (j) table
  and notes 4.8.
  - Only 76/96 (88/156) reached the domain limit.
  - The other 9 are 'all_s2s' (open): they were still S2S-like at their last run when the budget ended, at 0.79-0.995 of
    their own sigma_dom.
  - Correct to: "censored from above: 85/96 (76 at the domain limit, 9 open)". Section 4.7's "85 of 96 seeds are
    censored" is already right.
- **E3. H_W "confirmed by the core-shrinkage diagnostic".** This appears in notes 4.7 ("This confirms H_W's mechanism and
  direction"), 4.8 and REPORT summary item 4.
  - The shrinkage itself is required by the exact identity d||G_v||^2/dt = -2(||W||^2 - <W, W*>) whenever
    ||W_0||^2 >> <W_0, W*>. Its zero in the centred model also follows from that model's loss.
  - What the data add is the size of the shrinkage, which grows with x_W from about x_W ~ 1 as the heuristic says, and a
    flattening of rho vs sigma near the top of the S2S block.
  - Nothing tests the causal step from shrinkage to longer plateaus to a boundary above both laws. That would need an
    intervention, for example the plain model started from the shrunk state, or a plateau length predicted from the
    shrunk amplitude.
  - The registered H_W signature (4.4 table) also required "arm C stays near beta = 1". Arm C gave -0.07 +- 0.51, 2.1 SE
    from 1, and the notes do not mention this.
  - Correct to: "consistent with H_W (conjecture); shrinkage numbers empirical". "Heuristic" is not a project status tag.
- **E4. "sigma_dom -> sqrt(d)".** This appears in notes 4.6 item 2, 4.7, 4.8 and section 0, and in REPORT summary item 4.
  - z = sqrt(D) <W(xi), u> is a product of K per-core factors with E[log] < 0. The data give a slope of -0.143 (d = 2)
    and -0.105 (d = 3) per core. The verifier's simulation gives -0.14 at K = 31-127 (chi = 4; E log of a 4 x 4 Gaussian
    contraction is about -0.135).
  - Hence sigma_dom -> sqrt(d) e^0.14, about 1.63 (d = 2) and 1.99 (d = 3), not sqrt(d).
  - At the K reached, the medians are 1.50 (d = 2) and 1.80 (d = 3) at K >= 11, i.e. 1.04-1.06 sqrt(d), and still rising.
    They are not "approaching sqrt(d)". The K = 15 values in 4.6 (about 1.3 and 1.6) are 1.48 and 1.86 in the data.
  - Consequence for 4.8 ("sqrt(d) is a lower bound there if a boundary exists at all"): only 38 of the 85 censored
    bounds at K >= 8 exceed sqrt(d) (20/38 at d = 2, 18/47 at d = 3). Say instead: "the bounds sit about 3% (median)
    below each seed's own domain limit, whose median is about sqrt(d) at K = 8-15".
- **E5. "The S2S/rich boundary of the plain model at large K is not a crossing below sqrt(d) at all"** (4.8).
  - 11 of 96 seeds at K >= 8 do cross below sqrt(d): sigma_c = 0.86-1.37 at d = 2, K = 8-12, including 4/6 seeds at
    K = 12; and 1.08 at d = 3, K = 10.
  - Correct to: "for most seeds (85/96) no crossing is found below the domain limit".

Minor (wording, numbers, provenance):

- **E6. Birge scaling decides rho* = 5.** `wls` multiplies the SEs by sqrt(chi^2/dof) when that exceeds 1. The
  pre-registration does not include this.
  - At rho* = 5, plain WLS gives beta = 1.555 +- 0.263, so z_B = 2.11 and the rule says "unresolved". With Birge
    (factor 1.115) z_B = 1.86 and the rule says "H_B favoured".
  - The primary rho* = 3 fit (Birge = 1) and arm C are unaffected.
  - Report rho* = 5 as "H_B favoured with Birge-scaled SEs, unresolved without".
- **E7. Pre-registration header.** It says "pre-registered 2026-10-10 12:50 UTC", but commit f57a28c is 12:26:25 UTC. 12:50
  is the arm-C launch. As written, the header suggests that arm P (first rows at 12:32) started before the
  pre-registration.
- **E8. Amendment 4.6 item 2: the pilot value.** The pilot (binary 16, d = 2, sigma = 2) has a_1(0) = 8.85
  (pilot_plain16_lsoda trajectory), not ~40.
- **E9. Amendment 4.6 item 3: the discarded runs.** "The first 22 rows" are 22 rows plus 112 trajectories; 90 of the
  trajectories have no row because the pool died before they were recorded. All 112 were re-run bit-identically, so the
  results are unaffected.
- **E10. Amendment 4.6 header.** It says "none uses arm results", but amendments 1-3 were committed 21 s after the last
  discarded arm-P trajectory (12:34:06; the first was written at 12:32:29), and two of the discarded rows show the t = 0
  event bug.
  - The domain rule depends only on a(0), so no outcome-based selection is possible.
  - Reword to "made after the first arm-P rows, which were discarded and re-run".
- **E11. Arm-C extension not listed.** Arm C was extended to (K, d) = (9, 2) and (5, 3) at 14:22 (commit 32f24f7), but this
  is not in the amendment list.
  - (5, 3) ran one sigma per seed (rho 2.7-6.6) and enters the arm-C Tobit fit as 5 lower bounds and 1 upper bound.
  - (9, 2) produced no rows.
- **E12. Range of the lower bounds.** "Lower bounds 1.2-2.0" is the K >= 11 range. At K >= 8 (the scope of the "85 of 96"
  sentence) the range is 1.07-2.03, and all 97 bounds span 0.97-2.34.
- **E13. "S2S-like at every sigma up to the domain limit" holds only at the sampled sigmas.**
  - The 85 censored seeds at K >= 8 have a median of 4 in-domain runs (range 1-7), with gaps up to a factor 1.86 in
    sigma (17 seeds have a gap > 1.3).
  - Most top-end rho values are far above 3: the median smallest S2S-like rho is 117 at d = 2 and 2,630 at d = 3.
  - But rho is non-monotone in sigma. For example, K = 11, d = 2, seed 4 has rho = 3.94 at sigma 1.29 and 12.1 at 1.32.
    7 seeds have an S2S-like rho < 6, so a rich window in an unsampled gap cannot be excluded for them.
- **E14. 14 of the 97 lower bounds rest on the lower-bound rule.** All are at d = 3, K = 12-15. In these runs the second
  component was not learned by t_end (up to 1e6), so a stall would be classified the same way.
  - Also, (t_end - t_1)/w_1 is a lower bound on rho = gap / max(w_1, w_2) only if w_2 <= w_1. This caveat applies to
    the registered rule itself.
- **E15. d-contrast and d-ratios.**
  - The d-contrast fit uses K = 3-5 (dof 2; d = 3, K = 6 has only 3/6 resolved seeds), not "K = 3-6".
  - "Each CI contains both 1 and H_B's 1.27-1.31" fails at K = 6: the CI [0.64, 1.22] excludes 1.27. Censoring removes
    3 of the 6 d = 3 seeds there.
  - "Cells K = 3-7 plus d = 2 at K = 12" means d = 2: K = 3-7 and 12, and d = 3: K = 3-5 (rho* = 3).
- **E16. "Open" annotations in the 4.7 table.** The table marks d = 2, K = 14 and d = 3, K = 11 and 12 as "(1 open)".
  per_seed_P.csv has no 'all_s2s' seed in those cells.
  - The CSV has 9 open seeds (d = 2, K = 15: 2; d = 3, K = 13: 3, K = 14: 1, K = 15: 3), as the text says. The table
    shows 12.
  - The extra annotations look like walker states from the run log, not analysis statuses.
- **E17. REPORT (j): "largest K with a resolved cell ... K = 7 at d = 3 (1/6)".** At d = 3 the largest K with a resolved
  seed is K = 10 (1/6), and the largest fit-eligible cell is K = 5 (6/6).
- **E18. Relabel range.** "Moves up by about 4-9%" (notes 5) is the centred range.
  - Plain gives 1.04-1.07. Two-copy gives 1.01-1.07 at chi >= 9, and 1.15 at chi = 6 (2 seeds).
  - Say "1-9% (centred 4-9%); 14-15% at chi = 6-9".
- **E19. Untagged variant.** The "median-based variant" row of the 4.7 fit table should carry "not pre-registered", as the
  Tobit row does.

## Safe to cite

Setting for items 1-6: binary trees, odeco teacher with weights (3, 2), fan-in init, gradient flow on the exact
population loss, 6 seeds per cell, threshold rho* = 3 unless stated. Paths are relative to experiments/regime_map/results/.

| # | statement | status | known / new | source |
|---|---|---|---|---|
| 1 | Plain model, chi = 4 = 2R, d = 2 and 3, K = 8-15: 85 of 96 seeds are censored from above (76 at the event-rule domain limit, 9 open at the end of the budget); 11 cross at sigma_c = 0.86-1.37. | empirical | new | round2/per_seed_P.csv |
| 2 | In the same setting, every censored lower bound at K >= 8 (1.07-2.03; 1.18-2.03 at K >= 11) exceeds 1 and exceeds the sigma^K = const and sigma^K/sqrt(D) = const extrapolations fitted on the resolved cells. So sigma_c -> 1 (H4) is rejected for the plain model at K <= 15. | empirical | new | round2/censored_P.csv, round2/fits.csv |
| 3 | Pre-registered rule on the resolved plain cells (d = 2: K = 3-7 and 12; d = 3: K = 3-5): beta = 1.21 +- 0.22, "H_B favoured". It is unresolved at rho* = 2 (0.72 +- 0.56) and in the no-shrinkage subset (0.61 +- 0.51). At rho* = 5 it is H_B favoured only with Birge-scaled SEs. Cite with "a censored selection; does not answer the large-K question". | empirical | new | round2/fits.csv |
| 4 | Centred model, full bonds (d = 2, K = 3-8; d = 3, K = 3-4): beta = -0.07 +- 0.51, unresolved (H_A at 0.1 SE, H_B at 2.1 SE). sigma_c medians are 0.16, 0.39, 0.50, 0.65, 0.81, 0.83 at K = 3-8 (d = 2). | empirical | new | round2/cells_C_rho3.csv, round2/fits.csv |
| 5 | Per-seed local exponent of rho vs sigma in the centred arm: -1.05, -2.13, -3.21, -4.93, -5.18, -6.93 at K = 3-8, against -(K - 2). | empirical | known scaling (plateau ~ sigma^-(K-2)), new check | round2/per_seed_C.csv |
| 6 | Plain model: the pre-escape loss of squared core norm grows with x_W = sigma^K sqrt(D)/(s_2 d). Medians: 0.1% (x_W < 0.1), 10% (1-10), 34% (10-100), 52% (1e2-1e3), 72% (1e3-1e4), 92% (>= 1e4). It is ~0 in the centred model. That cores shrink while \|\|W\|\|^2 > <W, W*> follows from the exact norm identity. | empirical (numbers); identity proven (standard) | identity known; numbers new | round2/shrink_P.csv, round2/shrink_C.csv |
| 7 | Mechanism H_W (unlearning W_0 lengthens the plateaus and lifts the plain boundary). | conjecture (consistent with items 2 and 6; not tested causally) | new | notes 4.2, 4.7 |
| 8 | Event-rule domain limit sigma_dom = sigma (0.1 / max_r \|a_r(0)\|)^(1/K): medians 1.50 (d = 2) and 1.80 (d = 3) at K >= 11. The large-K limit is about 1.15 sqrt(d) for chi = 4 fan-in init, not sqrt(d). | empirical (medians); limit derived (product of K factors with E log ~ -0.14) and checked by simulation | new | round2/per_seed_P.csv (sigma_dom); verifier simulation `v_domain_asym.py` |
| 9 | Run time under ~330 concurrent runs. Plain arm: median 100-250 s per run at K = 3-5, 300-650 s at K = 6-10, 700-1,650 s at K = 13-15; maximum 1.3 h. Centred full bonds: 500-700 s at P = 832-1,152 and ~1 h at P = 2,349. | empirical | - | round2/timing_P.csv, round2/timing_C.csv |
| 10 | Tensor-kernel relabel of the N = 4 maps: 65 of 5,298 labels change. 59 are lazy -> non-lazy at the lazy boundary; 6 plain runs are non-lazy -> lazy. | empirical | new | tensor_relabel/label_changes.csv |
| 11 | Per-seed lazy boundary, tensor / probe median ratio: 1.04-1.09 (centred, chi = 12-128), 1.04-1.07 (plain), 1.01-1.07 (two-copy, chi >= 9); 1.14-1.15 at chi = 6-9. The S2S side does not depend on the kernel. | empirical | new | tensor_relabel/lazy_boundary_summary_*.csv |

## Not checked

- The imbalance-map statement "7 of 189 majority cells change" (report/imbalance_map_b4.csv) and the regenerated report
  figures. The 22 imbalance label changes behind them were verified.
- The report's LaTeX. Section 1 still has the placeholder "Pending (regime-map round 2, running)".
