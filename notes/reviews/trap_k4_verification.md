# Verification: the k = 4 fix of the trap criterion (`notes/theory_trap.md` v1.2, section 2.6, Proposition 3)

Verifier (theory + numerics), 2026-10-10, on the merged commit e1906c6 (`wip/trap-k4`). Scope: section 2.6 and
Proposition 3, `trapcrit.route_tensors` / `reduced_route`, `k4_invariance.py`, `gf_check_k4.py`, the re-tabulated
MPS4 chi = 4 rows and pooled counts, and the report changes listed in section 10 item 7 of the note.

**Bottom line.** Proposition 3 is correct and its proof is complete within the reduced flow. The fix resolves
erratum 4 of `trap_verification.md`. My own code reproduces every verdict: 476/476 dying-end counts and 71/71
k = 4 verdicts. Full gradient flow: the author's 36/36 is reproduced (I repeated 24 of the runs). Out of sample,
running all 71 directions at sigma = 0.03 gives **69/71**. The two misses are near-boundary directions that die
at sigma >= 0.03 and escape at sigma <= 0.01. That is a finite-sigma shift, consistent with a sigma -> 0
verdict, but "the class does not depend on sigma" must be restricted (erratum 1). I recomputed the MPS4 chi = 4
confusion matrix from the raw records and it matches exactly. The listed report numbers are right, but the list
misses several places in the report (erratum 3).

## What was done

- **Own code** (scratchpad, not committed). The library is used only for `spec.init`, `pattern_teacher` and
  `get_tree`. Everything else is mine:
  - MPS4 chain contractions;
  - correlation flow (DOP853 until the overlap reaches 1e10);
  - learned directions from the left environments of the learned component (right environments as a check);
  - complement bases by SVD;
  - block extraction (C_u, route entries x, beta_1, beta_2, x');
  - the vector reduced route ODE and its effective-scalar form;
  - a dense full gradient flow (L = 1/2 |W - W*|^2, hand-written gradient, scipy LSODA with rtol 1e-9 and
    atol 1e-14, events on a_r = 0.5).
- **Draws.** All 476 draws of `k4_invariance.py`: committed seeds 0-7 of {0,3} and {0,1,2,3}, 60 small-init and
  400 saddle-start Monte Carlo draws of {0,3}. Also the 8 committed d = 4 seeds.
- **Their script, re-run.** `k4_invariance.py --out <scratch>` (20 s on 6 processes) gives a CSV identical to
  the committed one in q, n_dying, verdict, v_first, v_tensor, v_eff, first_rot and tensor_rot.
- **Full gradient flow** (own driver). Two sets:
  - the author's 12 keys at sigma = 0.01 and 0.03;
  - all 71 two-dying-end directions at sigma = 0.03, with the two misses repeated at 0.1, 0.01 and 0.003.
- **Confusion matrix.** The 208 MPS4 chi = 4 locality records with k = 4, read from the raw `data/locality`
  parts (twocomp_chi4, robust, stalldiag; last record per key). Observed label: a_final[r] > 0.5. Predicted
  label: my own verdicts.
- **v1 -> v1.2 diffs** of `locality_pilot_predictions.csv.gz` and `montecarlo_runs.csv.gz`.

## Verdict per claim

| claim | verdict | notes |
|---|---|---|
| The inner route entries are vectors beta_1, beta_2 over the whole inner new space N_f | **correct** (note and code) | `route_tensors` keeps every route leg in the full new space of its edge. The drive is the einsum gradient of the full contraction. My own implementation gives the same verdict 71/71. |
| Prop. 3: the flow closes on (x, x', \|Y\|^2, rest, p, S), so the outcome depends on (beta_1, beta_2) only through p and S | **correct, proven** (within the reduced flow) | dp/dt = gS and dS/dt = 4gp, re-derived. The ends see only p, and g depends only on the ends. The effective amplitudes reproduce p and S. Not in the note: S^2 - 4p^2 = \|s\|^2 \|m\|^2 = 4h^2 is conserved, so S >= 2\|p\| holds for all t and the effective amplitudes stay real. The "Reading" bullets are also correct: s and m evolve as exp(+-G); every squared route norm moves at the rate 2P; the loss before p changes sign is S_0/2 - h <= min \|beta_i\|^2, with equality iff beta_1 is parallel to beta_2. |
| Proposition 3, numerically | **checked** | The effective-scalar verdict equals the vector verdict 71/71. Trajectories of x, x' and P agree to 2.4e-8 (own code, rtol 1e-10); the author's 2.4e-7 used rtol 1e-9. |
| Basis invariance | **correct** | 71/71 verdicts are unchanged under 5 own Haar rotations of N_f. The verdict also does not depend on how the learned directions are defined: left vs right environments give 13/13 equal verdicts on the small-init directions (C_u differs by at most 1e-2, median 3e-6). The code's thresholds (scale = max norm, t_end, atol) are themselves invariant. |
| v1 was basis dependent (37/71) and differs from the new verdict in 25/71 (23 dead -> escape, 2 escape -> dead) | **reproduced** | Exactly, with their script. With my own 5 rotations, v1 flips in 31/71. The v1 -> v1.2 prediction diff has exactly 20 changed rows (seed 1 of both teachers, dead -> alive). The Monte Carlo diff changes 23 verdicts, all MPS4 {0,3} chi = 4: saddle start 18 dead -> alive and 1 alive -> dead, small init 3 and 1. (The S column also changed from int to str, which is not a content change.) |
| Verdict derived for sigma -> 0, checked in full GF (36/36) | **correct as a sigma -> 0 statement; checked 36/36 + 69/71** | My GF reproduces their classes 24/24 (12 keys at sigma = 0.01 and 0.03), with escape times equal to 2e-6. Out of sample, at sigma = 0.03, it matches 69/71: dead 26/26, escape 43/45. See erratum 1. |
| Assumption: one new direction on each end edge | **correctly stated and enforced** | `predict` requires bond dimension 2 on both end edges, and `reduced_route` raises otherwise. At d = 4, chi = 4 (end edges uncapped) all 8 seeds have both C > 0, so they are alive without the ODE. |
| MPS4 chi = 4, d = 2 (160 runs): dead 26, all stalled or dead; pinned 68 and alive 66, all escaped | **reproduced exactly** from the raw records with my own predictions | Dead: 24 stalled + 2 stall-diagnosis dead. They come from 3 init directions: {0,3} seed 7, and {0,1,2,3} seeds 0 and 7. The d = 4 runs: 48 alive, all escaped. So the table holds 26/26 dead and 0/182 others. Caveat: these rows rest on 3 dead and 3 escaping init directions (each counted at many sigma). The 71-direction GF test carries the evidential weight. |
| MC P(dead), MPS4 {0,3} chi = 4: 0.0925 / 0.083 -> 0.05 / 0.05 | **reproduced** | Own code: saddle start 20/400, small init 3/60. Of the draws with two dying ends, 38/58 and 4/7 escape (note: "about 65%"). |

## Errata

1. **Section 2.6, "The class does not depend on sigma"** (and T6 "checked in full GF 36/36").
   - This is true for the 12 directions checked. Two of the 71 directions are counterexamples at finite sigma.
     Both are saddle-start Monte Carlo draws with a near-zero end and a small inner overlap:

     | draw | C_u (ends) | p | sigma = 0.1 | sigma = 0.03 | sigma = 0.01 | sigma = 0.003 |
     |---|---|---|---|---|---|---|
     | 100163 | -0.009, -0.030 | -0.022 | dead | dead | escape | escape |
     | 100319 | -0.245, -0.0035 | +0.040 | dead | dead | escape | escape at t = 1.18e8, past `gf_check_k4`'s t_max = 1e8 |

     ("dead" means both route-end entries below 3e-12 sigma and a_r below 1e-30 at t = 1e8.)
   - Suggested wording: "the class is sigma-independent as sigma -> 0. Near the boundary (|C_u| or |p| <~ 0.04)
     it shifts at sigma >~ 0.03 (2 of 71 directions at sigma = 0.03), like the O(sigma) corrections of
     section 5. Slow escapes near the boundary can exceed a t_max of 1e8 at sigma = 0.003."
   - Suggested status: "checked in full GF: 36/36, plus 69/71 at sigma = 0.03 (verification; the two misses are
     near the boundary and agree at sigma <= 0.01)".
2. **`results/stall_diagnosis_confusion.csv` is unfiltered.** It includes 4 runs with q = 1 and reads
   predicted dead 30 / 6 / 4. The note and the report quote 30 / 4 / 2, which is the q = 0, k >= 4 population
   (reproduced). This discrepancy predates v1.2: v1 had 30/8/4 in the CSV and 30/6/2 in the note. Filter
   the CSV or label it. This is not a v1.2 error.
3. **The list of report changes (note section 10, item 7) is incomplete.** All the numbers on the list are
   right. The list misses:
   - D_s2s per-seed counts;
   - D_s2s item (5) (l. 559-560), which says the scalar amplitude "is a choice of basis";
   - the D_s2s Monte Carlo sentence;
   - Sec. 4 l. 835-836;
   - Sec. 7 l. 225-238.

   The full list is below.
4. **Minor, report.** The report never cites the MPS4 chi = 4 Monte Carlo P(dead) (no "0.09/0.08" in the
   report), so "0.09/0.08 -> 0.05/0.05" has nothing to replace. Its range sentence does need the qualifier
   below, because 0.05 falls outside "0.1-0.3" for a two-ended route whose partners have sites.

## Report numbers that are safe to change

Line numbers are as of e1906c6. Every number below was recomputed from `locality_pilot_joined.csv.gz` /
`per_seed.csv` (q = 0, k >= 4), and the MPS4 rows also from the raw records.

**Sec. 4** (`04_saddle_to_saddle.tex`)
- l. 830: "$89\%$ of the predicted-dead runs stall" -> $93\%$. The $1.8\%$ is unchanged.
- l. 835-836: drop "$k=4$ with spare inner channels, where the reduced route model depends on a choice of
  basis" from the known failures.

**Sec. 7** (`07_open_questions.tex`)
- l. 225: $89\%$ -> $93\%$.
- l. 230-231: drop "with spare inner channels at $k=4$ the reduced route model depends on a choice of basis".
- l. 238: drop "a reduced route model with vector amplitudes for spare inner channels".

**`D_s2s.tex`**
- l. 559-560, item (5): replace "(where every route edge has one new direction; with $\chi\ge3$ on an inner
  edge ... its verdicts change with it)" by: one new direction on each end edge; spare inner directions enter
  through vector amplitudes; the verdict depends on them only through $\langle\beta_1,\beta_2\rangle$ and
  $\lvert\beta_1\rvert^2+\lvert\beta_2\rvert^2$ (basis invariant).
- l. 576:
  - dead $408/457$ ($89\%$) -> $408/437$ ($93\%$);
  - alive $38/2{,}101$ -> $38/2{,}121$ ($1.8\%$);
  - pinned $141/934$ is unchanged.
- l. 578: $k=4$: $203/232$ -> $203/212$, and $5/796$ -> $5/816$. The pinned count $2/432$ and $k=5$-$8$ are unchanged.
- l. 580-581, stall diagnosis:
  - predicted dead "30 dead, 6 escaped, 2 pinned" -> 30, 4, 2;
  - predicted alive "2, 56, 0" -> 2, 58, 0;
  - predicted pinned 5, 63, 0 is unchanged.
- l. 585-586, per seed (missing from the note's list):
  - "447 of 462 predicted-alive seeds" -> 449 of 464;
  - "45 of 65 predicted-dead seeds" -> 45 of 63 (16 at some is unchanged; 2 never).
- l. 590-591:
  - "56 false alarms" -> 36 (29 locality + 7 pilot);
  - 40 misses is unchanged;
  - replace the failure "$k=4$ with spare inner channels (MPS4 at $\chi=4$: 20 of 46 ... see (5))" by "MPS4 at
    $\chi=4$: 26 of 26 predicted-dead runs stall or die, none of the other 182".
- l. 621, Monte Carlo sentence: after "$0.1$--$0.3$ per run for routes whose two ends have partners with
  sites", add "($0.05$ on MPS4 at $\chi=4$, where three spare inner directions let about 65% of the draws with
  two dying ends escape)".
- l. 627: "30 of 38 predicted-dead runs dead and 7 of 126 others" -> 30 of 36, 7 of 128.

**App. A** (`A_derivations.tex`, `prop:A-trap`)
- l. 790, statement (iv): replace "(where every route edge has one new direction)" as the note proposes.
- Tags:
  - the reduction to (p, S) is proven within the reduced flow;
  - the k = 4 verdict is derived (sigma -> 0);
  - checked in full gradient flow: 36/36 runs (author) and 69/71 directions at sigma = 0.03 (verification),
    the two misses near the boundary and correct at sigma <= 0.01.

**Unchanged and still correct:** pinned 141/934; R12 417/1,143 and 51/1,233; pilot 56/63, 102/109, 20/182
and 2/146; MPS4 chi = 2 150 vs 0/344; the k = 5-8 rows; "40 misses".

## Safe to cite (k = 4 part)

- With one new direction on each end edge, the k = 4 route decision depends on the inner route entries only
  through p = <beta_1, beta_2> and S = |beta_1|^2 + |beta_2|^2. It is basis invariant (proven within the
  reduced flow; independently reimplemented, 71/71).
- The v1 scalar verdict depended on the basis (37/71 directions).
- Full gradient flow agrees with the invariant verdict in 36/36 author runs and in 69/71 directions at
  sigma = 0.03. The two misses lie at the boundary (|C_u| or |p| <= 0.04) and agree at sigma <= 0.01.
- MPS4 chi = 4, d = 2: 26/26 predicted-dead runs stall or die and 0/134 others do (0/182 including d = 4).
  These are 3 dead and 3 escaping init directions.

**Not safe to cite:** "the class does not depend on sigma" without "as sigma -> 0" or "away from the boundary".
