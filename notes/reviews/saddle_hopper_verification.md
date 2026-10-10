# Verification: saddle hopper (commit 13d2a25, `experiments/saddle_hopper/`, `notes/saddle_hopper.md`)

Verifier: adversarial auditor, 2026-10-10. The lead set five risks to check: (a) rules not fixed before
the comparison, (b) "rare branch" as a catch-all, (c) scoring and trivial baselines, (d) law checks,
(e) reproducibility. Method:
- read all eight scripts;
- re-ran the whole pipeline from empty caches;
- recomputed every headline from the per-run CSVs with independent audit scripts (session scratchpad, not
  committed: `audit_locality.py`, `audit_s2s.py`, `audit_rare.py`, `audit_rare_seeds.py`, `audit_r4c.py`,
  `compare_outputs.py`).

No other repository file was changed.

## Verdict

**Arithmetic and reproducibility: confirmed. Interpretation: overstated. Errata are needed before any
number is cited.**

- **The counts are correct as computed:**
  - locality: 11,082 main, 262 rare, 88 mismatches of 11,432 scored runs;
  - s2s_core: 1,138 main, 13 rare, 0 mismatches of 1,151;
  - data_teacher: 174/174 law checks; locality exponents: 179/217.

  A full re-run gives byte-identical outputs.
- **The sequence match rates do not show predictive power.** The predicted sets are permissive in three ways:
  - a random race allows every order;
  - every non-maximal fixed point becomes a "rare" branch;
  - the k >= 4 trap flag allows both stalling and escaping.

  Where the hopper commits to one order, its main branch is the weight order ("largest component first,
  no rotation") in all but 107 runs. On the informative locality runs it beats that trivial rule by 1.2
  points (97.0% vs 95.8%). On fan-in/balanced two-component runs the two predictions are identical. On
  s2s_core they are identical too (96.6% each).
- **Most of the 174/174 law checks are not independent tests:**
  - 112 recompute static quantities (levels, saddle types, Hessian eigenvalues) at the same saddles
    data_teacher had already computed;
  - 58 reproduce data_teacher's own predictions, digit for digit, against its own measurements;
  - only 4 rows are new, and in them the head start is under 1% of the slope.
- **"No mismatch needs theory wrong" is not supported:**
  - the causes were assigned after seeing the mismatches (the note says so);
  - the "events not separated" excuse is applied to 34 mismatches but not to the 2,711 matched runs with
    the same defect;
  - the 41 balanced-init stalls are failures of the time law;
  - the hopper's core rule "saddle order = k*" is now proven false in general (theory_saddle_order
    Theorem 2, section 5 below).
- **What survives:**
  - a sound, deterministic implementation of the theory notes' rules that reproduces their worked examples;
  - several confirmations of known theory;
  - a few modest new numerical checks (section 8).

  As a whole-sequence validation the hopper is **bookkeeping**, not new evidence.

## 1. Reproducibility (risk e): confirmed

**Setup.**
- Inputs: those of commit 13d2a25, extracted with `git archive` into the scratchpad; raw locality records
  from `data/locality/` (`LOCALITY_OUT`).
- Caches deleted; 8 workers; same order as the README.
- Wall time: examples 14 s, locality 54 s, s2s_core 38 s, data_teacher 160 s, order 55 s, summary 2 s.
  Total 5.4 min (README: about 8 min).

**Outcome.**
- All 25 CSV and PNG outputs are byte-identical to the committed ones once CRLF line endings are
  normalized, both PNGs included.
- `theory_sequences.txt` differs only in its wall-time stamps.
- The 5 prediction caches give identical predicted sequences (class, end, learned sets) in all 418 cases.

**Current main.** Since 13d2a25 only the order inputs changed (order fix round, merge 82da8a5).
`validate_order.py` still runs on them:
- the 32 gamma rows are unchanged;
- the multi-order frequencies agree to 1e-4;
- **but the H1b table comes back empty:** the filter `h1b.set == "saddle_b4"` no longer matches, because the
  set is now `h1b_saddle_b4`. So the "4/4; 4/4" H1b rows (section 5 of the note) are not regenerated
  from current main.

  A one-line fix restores them, and the values are the same: rho = 0.5 gives slope 2.5551 and
  lambda -0.39664.

## 2. Were the rules fixed before the comparison? (risk a)

**No evidence either way, and the note itself admits post-hoc steps.**

- Git cannot show the order of work: the hopper landed as one commit, and there is no brief or
  pre-registration file. `notes/decisions.md` says only "scored ... without tuning".
- The predictions do not use sigma or seed (`case_of` drops both).
- I found no rule or threshold tuned per case. The numerical tolerances (1e-6 route, 1e-7 negativity,
  10% near tie) only affect flags or numerics.

**The data-dependent elements are:**
1. **R4c (rare branches at later escapes).** It was made explicit "after reading the experiment notes",
   and the s2s_core 1-3-2 orders motivated it. **All 13 s2s_core rare runs depend on R4c alone**
   (`audit_r4c.py`: 13/13 match only through a later-escape rare branch). Without R4c they are 13
   mismatches. They cannot also count as support for R4c.

   No locality rare run depends on R4c: all 262 come from first-escape rare branches, i.e. from
   theory_escape E9.
2. **The saddle start from the lowest rank-one attractor.** This copies data_teacher's protocol and is
   harmless.
3. **The mismatch-cause checks** were written after seeing the mismatches (admitted). They have free
   thresholds: the dip test uses 0.5 and 3 final-rise widths; the add-vs-rotation test uses a factor of 3.
   They relabel mismatches but do not change the counts.
4. **The log-slope agreement (34/38) keeps only rows where locality's fit chose a log law.** This is
   selection on the outcome. Ten main-branch k = 2 rows with power-law fits are dropped. Most are plateaus
   locality rarely detected, but one is a clear miss: TC mps8 small init, level 2, 7.74 predicted vs 5.62
   measured.
5. **The plateau-level comparison is computed but not reported in the note.** In `summary_laws.csv` it is
   55/89 agree. Of the 34 misses, 21 are levels detected in fewer than half of the runs and 13 are L_mid
   deviations above 5% on other-branch orders.

## 3. Rare branches (risk b)

**How branches are enumerated.**
- Every fixed point found by HOPM becomes a branch:
  - for the grouped rank-one problem: 32 random starts plus the component starts;
  - for each k*-route problem: 12 random starts plus the component starts.
- Classification by the Riemannian Hessian of g on the unit sphere:
  - no positive eigenvalue: an attractor, i.e. a main branch;
  - otherwise: a rare branch with gamma = sum h_u / (K g);
  - rotations try both signs of the unstable eigenvector.
- So rare branches are fixed by the algorithm, not chosen per run. They carry a heuristic exponent and **no
  prefactor**.
- 112 of 226 locality cases have at least one rare branch; gamma ranges from 0.19 to 8 (median 1.0).
- For two-component teachers the rare branch makes both first-escape orders admissible. What remains
  testable is the follow-up: rotation after a wrong first escape on matrix-like pairs (m = 2, k >= 3),
  direct addition on adjacent pairs (k = 2).

**"Rare" is not rare at the sigma used.**
- In 10 of the 28 locality rows with at least 4 rare runs, the rare branch takes at least 20% of the runs
  (maximum 46%, mps8 S=07 balanced).
- Star teachers: 48/120 runs are rare.
- Order binary4, R = 3 and 4: 30-65% of the gradient-flow runs are rare.

A rare branch therefore cannot fail run by run; it can only be tested through its trend in sigma.

**The sigma trend holds in aggregate.** Locality reuses each seed's init direction at every sigma, so the
note's "18 of 28 rows thin out" compares correlated samples. A seed-level test (`audit_rare_seeds.py`)
uses the 47 rows with a single rare gamma:
- 64 seeds are on the rare branch at the row's largest sigma;
- 7 are still on it at the smallest sigma;
- P ~ sigma^gamma predicts 7.7, a sigma-independent race 64;
- 2 seeds join the rare branch at the smallest sigma.

So the first-escape rare branches behave as sub-dominant branches, as theory_escape E9 says. This supports
the label as a statement about sigma -> 0, consistent with the order agent's gamma fits.

**Strict rates (rare counted as mismatch):**
- locality: 11,082/11,432 = **96.9%**;
- s2s_core: 1,138/1,151 = **98.9%**;
- stars: 72/120 = **60%**;
- order binary4 (R = 3, 4): **30-71%** of runs follow the main branch.

Informative rates are in section 6.

## 4. Scoring and trivial baselines (risk c)

**Matching.**
- Locality: the run's final-1/2-crossing order must be produced by a branch, and locality's rotation class
  (any re-crossing of 1/2) must match the branch's rotation flag exactly.
- s2s_core: order of first full rises, with **the rotation flag ignored** (`validate_s2s_core.score`).
- Within a race or a joint event every permutation counts.
- Locality's final-crossing order turns a corrected wrong order (rotation) back into the weight order.
  So "weight order" is the natural prior for this metric.

**Stalls.**
- 686 scored locality runs stalled. 645 of them (fan-in or saddle start) count as **main** because the next
  route has k >= 4 and carries the trap flag. The flag covers both outcomes, so these are not predictions.
  The 41 balanced-init stalls are mismatches.
- Under unbalanced init the flag is set on every k >= 4 route. "Every stall sits on a flagged route" is
  therefore equivalent to "no stall at k <= 3" (0 of 4,531 scored two-component runs with k = 2, 3).
  That is the citable content.
- s2s_core: all 288 trapped runs count as main through the same flag.

**Unresolved events and dips.**
- 2,745 scored locality runs (24%) have separation ratio rho_min < 3. The excuse is applied to the
  34 mismatches among them, while the other 2,711 count as matches.
- Resolved non-stall runs only (8,001 runs): main 97.6%, main + rare 99.8%.
- The 13 "transient dip" mismatches rest on a dip definition written after seeing them (section 2).
- s2s_core leaves 425 of 1,576 runs (27%) unscored: 215 stalled at the budget, 210 unresolved.

**Baselines** (`audit_locality.py`, `audit_s2s.py`):
- B1: weight order, no rotation. Locality's class "greedy" is exactly B1, and all 9,811 greedy runs count
  as main.
- B1r: weight order, rotation ignored.
- B2: any order, no rotation.
- B3: uniform random order.

| subset | runs | hopper main + rare | hopper main | B1 | B1r | B2 |
|---|---|---|---|---|---|---|
| locality, all scored | 11,432 | 99.2% | 96.9% | 85.8% | 87.8% | 91.6% |
| locality, non-stall | 10,746 | 99.6% | 97.1% | 91.3% | 93.4% | 97.5% |
| one main order, non-stall | 7,965 | 99.7% | 97.0% | 95.8% | 97.9% | 97.5% |
| of which saddle start (first component fixed by protocol) | 2,919 | 99.97% | 99.97% | 99.97% | 99.97% | 100% |
| of which fan-in/balanced, two components | 4,291 | 99.95% | 96.08% | **96.08%** | 99.5% | 96.6% |
| of which fan-in/balanced, multi-component | 755 | 97.4% | 90.6% | 78.0% | 80.5% | 93.5% |
| main order differs from weight order (B4 mps4, E8 mps8) | 107 | 100% | 88.8% | 0% | 0% | 88.8% |
| race / joint runs (several main orders), non-stall | 2,781 | 99.1% | 97.5% | 78.4% | 80.5% | 97.2% |
| s2s_core, all scored | 1,151 | 100% | 98.9% | 52.0% | - | - |
| s2s_core binary4, complete (the only informative cell) | 380 | 100% | 96.6% | **96.6%** | - | - |
| s2s_core other trees, complete (all 6 orders main) | 483 | 100% | 100% | 47.8% | - | - |

B3 (uniform random order) is about 45% overall.

**Reading the table.**
- On two-component teachers the main prediction is B1 itself. Its extra content comes only through the
  branches:
  - rotation after a wrong first escape on matrix-like pairs (all 147 rotation runs on m = 2, k >= 3
    pairs);
  - direct addition on adjacent pairs (19 runs);
  - no rotation on tensor-like pairs: 20 rotation-class runs there, all excused post hoc as dips or
    unresolved events.

  These are theory_saddles 4.4 and H1b, already confirmed by the order agent.
- The only places where the hopper's order beats B1 are 107 multi-component runs (route rule, theory_saddles
  and locality).
- On race runs the hopper's content is "no rotation", the same as B2.

**Smaller labelling errors in the note.**
- The 2,869 "tensor-like races" include 735 k = 1 joint-event runs and 48 R = 4 runs with 4 main orders.
- The 8,563 "single predicted order" runs include 3,176 saddle starts and 598 stalls.

## 5. Law checks (risk d)

**data_teacher, "174/174".** Breakdown:

| group | rows | what it is |
|---|---|---|
| plateau levels | 36 | static recomputation at the saddles data_teacher computed itself: two codes, one object |
| saddle types | 36 | same |
| k = 2 lambda_min | 12 | same |
| star strictness | 20 | same |
| star lambda | 8 | same |
| log slopes | 31 | dynamical |
| exponents | 31 | dynamical |

- The first five groups total **112 static rows**. Agreement there is expected to 6 digits and is not a
  test of the dynamics.
- **27 of the 31 slope rows** have data_teacher's own prediction (`pred_A`), equal to the hopper's value to
  at most 8e-8 relative.
- The 31 exponents are k* - 2, which data_teacher had already compared (its `pred_p` column).
- Saddle-start slopes are 1/|lambda_min| at the point the runs started from, i.e. the linear instability of
  that same point. This is close to a tautology for an S2S test.
- **New rows: 4** (misaligned covariance: `misaligned_20` mps, theta = 20 and 5 deg, `misaligned8_20`).
  Measured within 0.2-0.8%, but the head start is under 1% of the slope, so they test 1/gamma (known).
  The note says so.

**Locality.**
- The hopper's k equals locality's `k_rank` on 197/197 rows.
- The exponent (179/217), lambda-exponent (144/171), slope and Hessian rows compare locality's own fits
  with locality's own predictions (H2): a re-tally.
- Tolerances are loose but declared: 0.2 in the exponent (P14(ii)), 10% on slopes and lambda, 0.3 in the
  lambda exponent.
- The causes given for the 38 exponent misses come from locality's own diagnostics. Five stay
  unexplained (steep MPS routes, locality 7.9).

**s2s_core and order.**
- s2s_core K - 2 exponents (45/47): s2s_core's own comparison.
- The strict saddle lambda -0.133186 equals s2s_core's `lam_theory` (-0.1331856).
- The 32 order gamma rows equal the order agent's `gamma_pred` to its 4-digit rounding. The gamma
  tolerance is max(20%, 2 SE, 0.02).
- The H1b lambda and slopes equal the order agent's `lam_pred` and `slope_pred`.
- All of these are re-tallies.

**Rule R2 is no longer safe.** The hopper sets saddle order = k* (rules R3, R5). theory_saddle_order.md
Theorem 2 shows the order can be lower when the generalized piggyback operator exceeds s_q. Its
verification (`notes/reviews/saddle_order_verification_proof.md`, untracked at the time of writing) calls
Theorem 2 proven. One reproduced example is HT4 1111 (1.5), with order 6 against k* = 7. That is the
wrong-first odeco saddle met in s2s_core's ht4 races.

In the committed data the effect is not clear. The s2s_core per-seed gap-1 exponents are:
- ht4, wrong-first seeds: mean 4.96 (13 seeds; two at 4.03 and 4.38);
- ht4, greedy-first seeds: mean 5.14;
- binary8: 4.95 vs 4.96.

The likely reason is that the order-6 coefficient is tiny (1/432). The main-branch exponents of
two-component teachers are covered by Corollary (ii) of that note. **Hopper exponents on wrong-order
branches or multi-residual saddles with k* >= 5 should not be cited** until the hopper implements the
generalized piggyback test.

## 6. Honest headline numbers

| claim in the note | honest statement |
|---|---|
| locality: 99.2% follow a predicted sequence (96.9% main, 2.3% rare, 0.8% mismatch) | **main branch 96.9%** (11,082/11,432). This includes 645 stalls that count only through the trap flag. |
| (same, stricter) | Non-stall runs: 97.1% (10,437/10,746). Excluding the 2,869 race/joint runs: 96.8% (8,288/8,563). Informative runs (one main order, non-stall): **97.0% vs 95.8% for "weight order, no rotation"**. Fan-in/balanced two-component runs: hopper = baseline = 96.1%. |
| s2s_core: 98.9% main, 13 rare, 0 mismatches | Informative cell (binary4, 380 complete runs): **96.6%, equal to the weight-order baseline**. The 13 rare runs rest on R4c (post hoc). The other 771 scored runs allow every order (483) or match through the trap flag (288). 425 runs are unscored. |
| data_teacher: 174/174 law checks | 112 static recomputations + 58 reproductions of data_teacher's own predictions + 4 new rows (weak). **Cite as implementation agreement.** |
| locality exponents match k - 2 in 179/217 rows | Correct re-tally of locality's own fits; k equals locality's on every row. Not independent evidence. |
| no mismatch needs "theory wrong" | Unsupported: post-hoc causes; 41 balanced stalls (time law); 5 unexplained steep MPS exponents; R2 refuted in general. |
| rare branches thin out at small sigma (18 of 28 rows) | Supported at seed level: 7 of 64 seeds still rare at the smallest sigma, vs 7.7 predicted. But rare branches take up to 46% of a row's runs at the sigma used. |

## 7. Errata (to apply to `notes/saddle_hopper.md` and the README before citing)

- **E1** (sections 0, 4, 7; README "In short"). Replace "99.2% ... follow a predicted sequence" by the
  main-branch rate (96.9%) next to the weight-order baseline. State that rare-branch runs (262) and stalls
  matched by the trap flag (645) are admissible outcomes, not predicted ones.
- **E2** (section 4, "How informative"). The 2,869 runs are not all tensor-like races: they include 735
  k = 1 joint events and 48 runs with 4 main orders. The 8,563 single-order runs include 3,176 saddle
  starts and 598 stalls. Add the baseline table of section 4 above.
- **E3** (section 4, s2s_core). "All 1,151 scored runs follow a predicted sequence" becomes: binary4 is the
  only informative cell, 367/380 main, equal to the weight order. Also state that:
  - the 13 rare runs exist only through R4c, which was added after reading the s2s_core notes on these
    orders;
  - the rotation flag is not scored;
  - 425 runs are unscored.
- **E4** (sections 5, 7). Reword "174/174" as in section 6.
- **E5** (section 5 table). Make three changes:
  - add the locality plateau-level row (55/89);
  - add the s2s_core strict-lambda row (2/3);
  - state that the log-slope check keeps only rows with a log-law fit (10 main-branch k = 2 rows dropped,
    among them TC mps8 small init level 2: 7.74 predicted, 5.62 measured).
- **E6** (sections 6, 7). Drop "none is attributed to the theory being wrong". Say that the causes are
  post hoc, that rho < 3 affects 2,711 matched runs as well, and that the 41 balanced stalls are slow or
  pinned escapes the time law does not predict.
- **E7** (section 4, stalls). Cite "no stall at k <= 3 (0 of 4,531)". "Every stall on a flagged route" is
  the same statement, because the flag is set on every k >= 4 route.
- **E8** (section 4, rare branches). Report the rare fractions at the sigma used (up to 46%; stars 40%;
  order binary4 30-65%). Replace the row-level trend by the seed-level test.
- **E9** (sections 4 and 5, order). On current main `validate_order.py` drops the H1b rows silently
  (`set == "saddle_b4"` should now be `"h1b_saddle_b4"`). Fix it and re-run. Also state that the gamma and
  H1b values are the order agent's own predictions.
- **E10** (section 1, rules R3/R5). Add the caveat of theory_saddle_order Theorem 2 (section 5 above).
  Rule R2 "order = k*" is false in general.
- **E11** (README). The re-run takes 5.4 min on 8 processes here (the README says about 8). Minor, and
  machine-dependent.

## 8. What is safe to cite, and what is new

**Safe to cite:**
1. The hopper is a deterministic, reproducible implementation of the theory notes' rules: byte-identical
   re-run, 5.4 min on 8 processes. It reproduces the 14 worked examples of the theory notes to the stated
   digits, and locality's route degree on every row (197/197).
2. Confirmations of known theory (cite the experiment notes for the claims; the hopper is only a
   consistency check):
   - route-rule orders on B4 mps4 and E8 mps8: 95/107 main, 107/107 with the predicted wrong-first
     branch;
   - rotations after a wrong first escape on matrix-like pairs: all 147 rotation runs on m = 2, k >= 3
     pairs;
   - wrong order kept, with no resolved rotation plateau, on tensor-like and k = 2 pairs;
   - no stall at k <= 3: 0 of 4,531;
   - binary4 greedy first escape: 380/380.
3. Main-branch sequence agreement, always next to the baseline:
   - locality: 96.9% of 11,432 runs;
   - informative locality runs: 97.0% vs 95.8%;
   - s2s_core binary4: 96.6% vs 96.6%.
4. First-escape rare branches thin out as P ~ sigma^gamma in aggregate: 7 of 64 seeds still rare vs 7.7
   predicted. This is new aggregate evidence, consistent with the order agent's gamma fits.
5. Multi-saddle accumulated-utility (AGF) slopes for nested pattern teachers, as a modest new check:
   - TC on mps8 from the saddle: 5 plateaus within 3.5%;
   - N4 on mps4: levels 1-2 within 3.3%;
   - TA and TB: level 1 is 10.5% low (partial carry-over, A-carry, unmodelled);
   - TC from small init: level 2 misses (7.74 vs 5.62).

**Not safe to cite:**
- "99.2%" or "all 1,151 follow a predicted sequence" as evidence of predictive power;
- "174/174" as independent law tests;
- "no mismatch needs theory wrong";
- the 13 s2s_core 1-3-2 runs as support for R4c;
- the H1b rows on current main until E9 is fixed;
- hopper exponents for wrong-order or multi-residual saddles with k* >= 5;
- the misaligned-covariance slopes as a test of the head start.

**New beyond bookkeeping?** Little. The tool itself is new as engineering: it makes theory_saddles rule 5.1
operational. The new numerical content is:
- the multi-saddle AGF slopes (item 5);
- plateau levels on tie branches (B4 ht4, E8 binary8: a small correction to locality's level table);
- the misaligned-covariance random-init slope (weak);
- R4c as a derived rule, untested beyond the 13 runs that motivated it.

Everything else re-derives the experiments' own predictions and comparisons. For the report: at most a
short paragraph or appendix entry tagged **bookkeeping + implementation**. Do not present it as an
independent whole-sequence validation.

## 9. Recommended fixes (if the hopper is kept)

1. Report main-branch rates with the baselines, and report trap-flag stalls separately from matches.
2. Score the s2s_core rotation flag. Treat unresolved events symmetrically: exclude rho_min < 3 runs from
   hits and misses alike, or keep both.
3. Give the rare branches a probability. theory_escape 2.5 has a per-init recipe; the theory gives no
   prefactor.
4. Implement the generalized piggyback test (theory_saddle_order Theorems 1-2), or at least flag k* >= 5
   saddles where a heavier residual splits into two low sets.
5. Fix the H1b filter (E9) and re-run `validate_order.py` and `summarize_results.py` on current main.
