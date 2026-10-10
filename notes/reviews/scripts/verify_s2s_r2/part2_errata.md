
## 2. Errata (apply when citing `notes/s2s_core.md`, the REPORT or the README)

**First escape**

- **E1. First component, "100% of runs".** Applies to notes 2.2b and 4.5, REPORT section 4 and the
  fix-round table.
  - Correct figure: GF's first component equals the correlation flow's r1 in **1847/1848** first-escape
    runs, over all sigmas.
  - The exception is binary4, chi = 3, close teacher, balanced init, seed 6 at sigma = 0.1. There GF
    learns the 0.7 component first: a finite-sigma wrong first escape, the type theory_escape E9
    describes.
  - Replace "gradient flow learns the largest component first at every sigma" with: "in 639/640 binary4
    first-escape runs and in all 380 main/balanced runs". What is withdrawn is E9 as the cause of the
    old r1 mismatches, not E9 itself.
  - `first_summary.csv` `r1_agree_corrflow` = 1.0 tests the smallest sigma only.
- **E2. Size of the old bias in T.** Applies to notes 2.2b and 4.5 and the REPORT ("T moved by <=
  1.7e-3").
  - Binary4 close: the old (amplitude 1e6) T was off by up to **2.2e-3 relative** (balanced init, chi 3,
    seed 6). The 1.7e-3 figure is the fan-in maximum (1.75e-3).
  - Binary4 separated: off by up to 3.0e-5 (balanced, chi 6), not 2.4e-5.
  - mps4 and K = 7, 8 changed by <= 2e-7.
- **E3. How the new correlation-flow stop actually ended.** Applies to the notes 2.2b stop description,
  the REPORT "Decisions" entry and the `s2s.correlation_flow_blowup` docstring.
  - The direction criterion (second/first amplitude <= 1e-6) was **never met for the 40 binary4
    close-teacher jobs**. All of them stopped at the amplitude cap 1e30, with the ratio still between
    2e-6 and 7e-3; 2 separated binary4 chi = 6 jobs also reached the cap.
  - T is converged regardless: the time remaining beyond amplitude A is ~A^-(K-2)/K ~ 1e-10, and the
    independent sphere-flow T agrees to <= 1.5e-11.
  - Describe this case as "stopped at amplitude 1e30 (direction converging slowly)", not as direction
    convergence.
- **E16. Residual bound.** Applies to the REPORT summary and notes 3.3. "residual <= 2e-3" should read
  **<= 2.2e-3** (binary4 close, balanced, chi 6: 2.15e-3).
- **E17. Offset-removed exponents.** Applies to notes 2.2b, which states 0.9996 without a teacher.
  - 0.9996 +- 0.0001 is the **separated** teacher.
  - The close teacher gives 0.990-0.996 per cell (0.993 +- 0.003 for fan-in chi 3), which is 2.6-3.8 SE
    below 1. This is consistent with the unconverged O(1) offset; the REPORT table already shows 0.993.

**Saddles and gap exponents**

- **E6. Strict-saddle range.** Applies to notes 2.5, 3.2 and 4.1 and REPORT sections 5 and 2.
  - "-0.118 .. -0.146 over sigma 0.005-0.1, median 0.98" is **seed 8 only**.
  - All 9 wrong-order plateaus at chi = 3, adding seed 5 at sigma = 0.07 and 0.1: projected
    **-0.106 .. -0.146** (raw -0.113 .. -0.149); median ratio 0.98 (seed 8) and 0.93 (seed 5).
  - The exact eigenvalue at W_{1,3} is -0.133186 (this verification).
  - Add: the wrong order, and with it the strict saddle, is **not visited at the smallest sigma**. Seed 8
    is greedy at sigma = 0.003; chi = 6 seed 4 is wrong-order only at sigma >= 0.03. "Flat in sigma"
    therefore covers sigma = 0.005-0.1.
- **E7. Novelty tags.**
  - binary4 is a depth-3 DLN (theory_saddles 4.2). Its S2S sequence, the sigma^-1 plateaus and the
    strict wrong-order saddle are **known DLN results**: [Achour2024] (non-top rank-r critical points
    are strict saddles) and [Jacot2021] Lemma 15, both in `notes/literature.md`.
  - Tag them "replication (known for DLNs; prediction = theory_saddles 4.1)", not as new. New for TTNs:
    the same measurements on mps4, binary8, ht4 and mps8.
- **E8. mps8 gap rows.** Applies to the notes 2.4 table and the REPORT table.
  - The sigma column "0.4-0.03" for mps8 chi = 3 is the t_1 range. The gap-1 fits use **sigma = 0.1 or
    0.15 to 0.4** (4-5 points, <= 0.6 decade).
  - The close-teacher mps8 chi = 3 gap 1 (6.19 +- 0.60, range 5.14-7.21, 3 seeds) is omitted.
- **E9. Binary4 close gap fits.** Applies to notes 2.4 ("within +-0.1 over 1.5 decades").
  - The rho > 3 cut leaves some seeds with only 4-5 sigmas (0.003-0.02, about 0.8 decade). Example:
    chi 3, seed 4, gap 1 = 1.064 from 4 sigmas.
  - The cut is rule-based and applied identically everywhere (my reimplementation reproduces every
    value). Without it, binary4 close gives gap 2 = 0.92 +- 0.11 (chi 3) and **1.12 +- 0.05 [0.99,
    1.45]** (chi 6); every other cell is unchanged to the third digit.
- **E11. R10 magnitude.** Applies to notes 2.5, 3.4 and 4.9 and the REPORT.
  - "0.87-1.15 at K = 3, 4" holds for the cell medians of the main fan-in cells. The balanced binary4
    chi 3 cell median is 1.19.
  - **Per-seed medians reach 20**: binary4 separated chi 3 seed 8, second plateau, where lambda_min is
    about 20x the route value at every sigma. Balanced mps4 chi 3 reaches 7.9.
  - At K = 7, 8 the cell medians are 1.36-3.15 (main, large) and up to 3.9 (k78); per seed up to 20.
  - Cite R10 as a cell-median statement only. The slopes are unaffected.
- **E21 (addition).** binary8 at chi = 3 (main): gap 1 = 4.93 +- 0.05 (separated) and 4.92 +- 0.04
  (close), 10 seeds each, per seed 4.65-5.13, mostly over sigma = 0.03-0.4 (up to 1.1 decades). This is
  the only K = 7 gap fit spanning about 1 decade; it is in `main_exponents.csv` but not in the notes.

**Plateaus and dormant components**

- **E10. Plateau levels.** Applies to notes 2.3 and the REPORT section 2.
  - "binary4 2.0e-2 (both teachers, sigma = 0.1)" should read: separated **8.5e-3** (sigma = 0.1);
    close **2.0e-2 at sigma = 0.003** (chi 6, seed 4, rho_2 = 5.1).
  - "K = 7 <= 2.1e-5" holds at chi = 6 (large). At chi = 3 (main) the level error reaches **1.5e-4**
    (ht4 close, sigma = 0.4).
  - "falls to 6e-7 at sigma = 0.003" holds for the separated teacher and mps4, not for the close
    teacher.
- **E4. Decay-rate levels.** Applies to notes 2.7c and REPORT section 6.
  - The fit starts at 1.5x the last event, inside the post-event transient. Example: seed 3, sigma 0.01,
    where core 0 still grows at first and core 3 decays at 0.7 sigma^2. This biases the per-core values
    by about +-4% (0.251 vs 0.270 for seed 3).
  - The asymptotic local rate is the **same at both end cores**: 0.259 (seed 3), 0.81 (5), 0.276 (6),
    1.00-1.04 (7).
  - "constant to 1-3%" should read "<= 2.4% (seeds 3, 5, 6), <= 5% (seed 7), sigma = 0.003-0.1".
- **E5. Single-end estimate.** "measured rates are 0.18-0.87 of the single-end estimate, i.e. 1.1-5.6x
  below it" appears in notes 2.7c, 3.5 and 4.4 and twice in the REPORT.
  - **Not reproducible:** no script, CSV or log computes Q_nu nu or the estimate. It rests on 2 re-run
    seeds, and the choice of s_q (2 or 3) alone spans a factor 1.5.
  - Do not cite the magnitude. Cite only the sign rule and the sigma^2 scaling.
- **E18. Escape-drop percentages.** Applies to notes 2.7a and 3.5 and the README.
  - ">100x tail 3-8% at K = 7, 8, chi = 3 (later)" should read **2.0-8.3%** (ht4 later escapes 2.0%).
  - "0.6-2% at chi = 6" should read 0-2%.
  - The README "3-11%" should read 2-8%, or 11% for the mps8 separated later escapes only.

**Traps and stalled runs**

- **E15. Trap fraction at K = 4.** Applies to the REPORT summary ("40% (separated) and 18% (close) at
  K = 4, chi = 3, fixed per init direction").
  - For the close teacher the fraction is not fixed per direction: seed 7 traps at 6 of 8 sigmas, and
    seed 5 stalls at all 8.
- **E12. Stalled runs and the "derived bound".** Applies to notes 2.7 and 3.3 and the REPORT.
  - The balanced-route estimate is exceeded in **4 of the 16 measured escapes** (1.08, 1.37, 1.81 and
    2.19x). When quoting data, call it an estimate, not a bound.
  - The "74 k78 stalled runs with estimate <= 100x" are **73 stalled + 1 trapped**. The trapped one,
    mps8 fan-in sigma = 0.3 seed 5, ran to t_max and has a slice of 1.4e-10 sigma; it was not drawn.
  - Overall, 23 distinct stalled or unresolved runs were continued and 16 escaped: 14/15 k78 and 2/8
    large.
  - 91/106 k78 stalled runs (estimate / elapsed: cell medians 17-64x) and 35/43 large mps8
    stalled/unresolved runs (median about 390x) are untested.
  - "Stalls are ever-longer plateaus" is therefore a derived prediction checked on 16 escapes. For the
    untested majority it is a conjecture, supported only by the slice trends: pinned in binary8 and in
    balanced mps8; still decaying in 50% of balanced ht4 runs and 19-32% of fan-in mps8 runs.
- **E13. mps8 chi = 6 spread sample.** Applies to notes 2.7, the REPORT and the README.
  - The run "not escaped at 2x" (s0.2 seed 9) **does not escape by 3x either** (my rerun with
    `--factor 3`).
  - The factor used for the committed CSV is not recorded.
  - The README command (`--spread --n 8 --max_ratio 1000 --factor 3`) picks a different set: the last
    3 of the 8 runs differ. So `continued_large_spread.csv` is a round-1 artefact the README does not
    regenerate.
  - "4 hit the wall limit before reaching the estimate" cannot be checked: the CSV has no reached-time
    column.

**Bookkeeping**

- **E14. Run and job counts.**
  - REPORT "T recomputed for all 238 first-escape jobs" and "first/ (238 jobs)": there are **231**.
  - "Sweep main (1600 runs)": `main/` has **1126** runs (2252 files).
  - The K = 7, 8 first-escape sweep is described as "separated teacher, fan-in only". It also contains
    21 binary8 jobs with the close teacher or balanced init, which are in `first_summary.csv`.
- **E19. `dormant_decay.csv` sweep column.** The table has no sweep column. 13 K = 7, 8 keys appear in
  both main and k78; one pair differs 6x (mps8 sigma 0.15 seed 8). No claim depends on it.
- **E20. ht4 decay example.** The example "ht4 seed 0: 0.0057-0.0060 over sigma = 0.15-0.3" omits
  sigma = 0.4 (main), where the value is 0.0017. This strengthens the "not sigma^2-clean at K = 7, 8"
  statement.
