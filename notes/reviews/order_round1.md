# Review of order (round1) — 2026-10-10

Two independent local reviewers: (A) partial reproduction from the committed code plus an independent reimplementation, (B) adversarial audit of claims vs data. Re-run outputs in `data/review_<folder>/` (local, gitignored).

## A. Reproduction

I reran a reduced set of the key `order` results with the committed code, and separately with my own code. All four headline claims reproduce: every number I could match seed for seed is identical. I found one scope error in the H1b claim, one code bug and three README errors.

**Terms used below.** A tree tensor network (TTN) writes a large tensor W, with one index per input, as small "core" tensors placed on a tree. Each tree edge is an index summed over, called a "bond", of dimension chi. An MPS is a chain with one core per input. binary4 is two leaf cores joined by a root core, so W = A R Bᵀ (a 3-layer linear network). The target is W* = s₁·(rank-one term) + s₂·(rank-one term), with ρ = s₂/s₁. Training starts from a small random initialization of scale σ per core and learns the two terms one at a time, with long plateaus in between. m is the number of cores whose inputs differ between the two terms. A strict saddle is a stationary point with a direction of negative curvature (λ_min < 0), which gradient flow leaves in time ~ln(1/σ)/|λ_min|.

**What I reran** (at most 4 busy processes, 34 min wall; outputs in `data/review_order/{raw,res,logs}`)
- Agent code, following the README with my own reduced configs:
  - `sweep.py`: MPS4 (d=2, χ=4), ρ 0.5/0.8, σ 0.01/0.003, plus the leading-order race. Seeds 0–152 only; I stopped it early for time.
  - `sweep.py`: binary4 (χ=4), ρ 0.8, σ 1e-2/1e-4, seeds 0–199.
  - `h1b.py`: `saddle_b4` (24 runs), `natural_b4` (the first 8 wrong-order seeds at each σ), `saddle_mps4` (12 runs).
  - `multi.py`: MPS4 with 3 and 4 components, ρ 0.8, 40 seeds.
  - `analyze_main`, `analyze_sigma` and `analyze_multi` on these outputs.
- Independent check, no code from `experiments/order`:
  - My own numpy contraction with closed-form gradients, the DOP853 integrator (the agent used RK45) and my own leading-order blow-up times. The library's initialization is used only so seeds match.
  - Runs: MPS4, 400 seeds × 6 ρ, plus 3 extra σ; binary4, all 1000 seeds × 7 (ρ, σ) cells; race-only runs for MPS4 uncapped χ=32 (500 seeds) and MPS8 χ=4 (150 seeds).
  - Exact Hessians at the wrong saddle, written in plain torch without the `ttn` library, for 7 tree/teacher cases.

**Numbers: theirs vs mine**

| claim | committed | agent code rerun | independent |
|---|---|---|---|
| (1) MPS4 (2,4), P(larger first) at ρ .5/.7/.8/.9/.95/.99 | .815/.665/.605/.545/.51/.49 | .817/.614 on seeds 0–152 (ρ .5/.8) | **identical**, 400 seeds |
| (1) winner = argmin τ_r/s_r on MPS | 100% | 612/612 | 2400/2400; t₁ and τ match the agent code to ≤6e-10 |
| (1) race curve; sd(log τ) | .81611 …; .54635 | – | identical |
| (1) σ-independence, MPS4: wrong order at σ .1/.03/.01/.003 | 74, 158, 196 of 400 at every σ | 100% of seeds same winner at both σ | identical counts at all 4 σ; 100% of seeds same winner |
| (1) sd(log τ), uncapped χ | MPS4 χ32: .145; MPS8 χ4: .769±.017 | – | .14498 (race P identical at 6 ρ); .728±.030 (150 seeds) |
| (2) binary4, wrong order /1000 at ρ .8, σ 1e-2…1e-4 | 119/91/72/52/41; ρ .9: 293/182 | 400/400 runs agree with mine seed by seed | **identical**; my own fit γ = .232±.026 (theirs .2321±.028; theory .25) |
| (3) binary4 started at the wrong saddle: λ_min | −.39685/−.26637/−.18566/−.09655 | 24/24 rows identical | exact Hessian −.396850/−.266371/−.185664/−.096549 |
| (3) slope of correction time vs ln(1/σ) | 2.555/5.466 vs 1/\|λ\| = 2.520/5.386 | from σ ≤ 1e-4 only: 2.525±.006 / 5.399±.013 | – |
| (3) natural wrong-order runs: \|λ\|(t₁−t₂) | 1.07–1.38, flat in σ | 16/16 rows identical; 1.07 (σ 1e-4) vs 1.08 (σ 1e-2); 16/16 rotate | – |
| (3) m ≥ 3 (MPS4): \|λ_min\| ~ σ², no rotation | exponent 2.00 | 12/12 rows identical; exponents 1.99–2.03; a₂ ≥ .998 | no negative eigenvalue (MPS4 odeco, MPS6 m=3) |
| (4) 3 components, MPS4 ρ .8: P(greedy order) GF vs race model | .354 vs .355; own race = GF order .856 | .35 vs .344; .85 (40 seeds) | – |
| (4) 4 components, χ = 4: runs that stall | 149/200 | 33/40 | – |

**Discrepancies.** None in any number that can be compared exactly. Two statistical gaps come from my smaller samples: MPS8 sd(log τ) is 1.4 standard errors below theirs, and the 4-component stall rate is 82% vs 75%. The H1b slopes they report are 1.5% high only because the fits include σ = 1e-2, which is pre-asymptotic. At small σ the agreement is 0.2%.

**Scope issue in claim (3)**
- The eigenvalue −(s₁−s₂)s₂^((K−2)/K) and the "rotate, forget, relearn" sequence hold only for m = 2 pairs whose two cores do not share a bond. binary4 is such a pair.
- When the two cores are adjacent and share a bond of dimension χ_e > 1, the wrong saddle is still strict, but λ_min = −s₁s₂^((K−2)/K). This eigenvalue appears χ_e−1 times and comes from adding the larger component on a spare bond index. Gradient flow then adds the larger component without forgetting the smaller one.
- Evidence from my exact Hessians: MPS4 cores (1,2) gives −0.8944, three times, at ρ = .8; MPS6 cores (2,3) gives −0.8618.
- Evidence from dynamics: I ran the agent's own `h1b.py` on an MPS4 adjacent pair (new configs in `scratchpad/review_order/configs/`).
  - λ_min = −0.89442, versus −0.894427 predicted.
  - Correction-time slope 1.1175, versus 1/(s₁q^(K−2)) = 1.1180; the rotation law would give 5.59.
  - The smaller component's overlap a₂ stays at 1.0.
  - The separated control (cores 0,2) behaves as the agent says: λ = −0.178885, slope 5.59, a₂ falls to ~1e-16 and is relearned later.
- This agrees with the project's own k* = 2 rule in `theory_saddles.md`. The wording "H1b holds for m = 2" in `notes/order.md` §6/§8 and in the REPORT should be narrowed to non-adjacent pairs. Claim (2), about which component is learned first, is not affected.

**Bugs**
1. **`h1b.py` fails when given `--raw`/`--res` on Windows or macOS**, where worker processes are started fresh rather than copied from the parent.
   - `common.py` removes these options from the command line in the parent process. The workers re-import it without them, fall back to `/root/outputs/order`, and crash at `np.savez` (FileNotFoundError) after all the integration is done, so no CSV is written.
   - All three of my first `h1b.py` reruns were lost this way. Setting the environment variables `ORDER_RAW`/`ORDER_RES` works around it.
   - Fix: pass the directories in each job's arguments, or set them as environment variables inside `_dir_option`.
2. **README:** `analyze_multi.py multi` raises FileNotFoundError, because the config writes `multi2`. It should be `multi2`, or no argument.
3. **README:** it says scripts use all CPUs by default. `sweep.py`, `h1b.py` and `multi.py` actually default to 90 workers, so on this 24-core machine the README commands start 90 processes. (`repair.py` and `tol_check.py` do use the CPU count.)
4. **`run_queue.sh`** passes neither the options nor the environment variables, so a rerun overwrites the committed `results/` files.
5. **Minor analysis gap:** 43 of 421 natural binary4 runs, all at σ = 1e-4, have no Hessian at the plateau midpoint, because fewer than 3 record points fall between the two events. They are silently left out of the λ_min averages in `h1b_fits` and should be reported as unresolved.

**What `order` found**
- Which component a TTN learns first depends on m.
- When the two components differ on 3 or more cores (any MPS with N ≥ 4, binary N = 8), the order is random and set by the initialization, like a race between independent clocks. On an MPS the race is exact. The order does not depend on σ: P(larger first) is 0.60–0.88 at ρ = 0.5. Larger bond dimension makes the order greedier, but slowly.
- When m = 2 (matrix-like pairs, e.g. binary4, a 3-layer linear network), the larger component wins as σ → 0. Mistakes become rare only slowly, ~σ^((1−ρ)/ρ).
- A wrong order is corrected through a strict saddle only for m = 2, and with the adjacency caveat above. For m ≥ 3 a wrong order is kept.
- With 3 components the full order is close to a race of independent clocks.
- With 4 components and bond dimension equal to the teacher rank, training usually stalls before learning the last component.

**Work left on `order`:** all small and cheap.
- Fix bug 1 and the README.
- Narrow the H1b wording, or add the adjacent-pair H1b runs (about 1 CPU-minute).
- Report the unresolved midpoint Hessians.
- The agent's own open items remain: the prefactor of the m = 2 law, how the adjacent-pair exponent approaches its predicted value, and the 4-component stalls with a gauge-balanced initialization.

Files are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_order/`:
- Scripts: `indep_chain.py`, `indep_hessian.py`, `compare_*.py`, `configs/`.
- Notes: `notes_1-5.md`.

Outputs are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_order/`: `raw/`, `res/`, `indep/`, `logs/`, `indep_hessian.txt`. The `claims_audit/` folder there, and the `audit_*`, `h1b_*.py` and `stall_*` files in the scratchpad, belong to another reviewer.


## B. Claims audit

**Claims audit of `order` (Q4): ranked issues and solid claims**

Overall: the two-component results hold up. These are the random, σ-independent order for m ≥ 3, the exact race on MPS, the power law for separated m = 2 pairs, and H1b on binary4. The main problems are about scope. H1b is claimed for every m = 2 pair, but my reproduction shows it is false for adjacent pairs. The m = 2 law in the summary leaves out the adjacent-pair multiplicity n. The three- and four-component (R ≥ 3) results are conditional statistics taken at large σ.

I ran two reproductions with the agent's own functions, using at most 4 processes. Outputs are in `data/review_order/claims_audit/` (`claims_audit_h1b_scope.jsonl`, `stall_mps6_R3*.{log,jsonl}`). Working notes are in `scratchpad/review_order/audit_notes_{1,2,3}.md`. All my processes are stopped.

**Issues, most severe first**

1. **MAJOR: O7, "H1b holds for m = 2" (λ_min = −(s1−s2)s2^((K−2)/K); the learned bond index rotates, the smaller component is forgotten and relearned).**
   - The agent tested this only on binary4. That is a depth-3 deep linear network, where the result is already known.
   - My check used `saddle_init` and `projected_hessian` (ρ = 0.8, d = 2, χ = 4). For adjacent pairs (k = 2), λ_min = −s1·s2^((K−2)/K) with multiplicity χ_e − 1 = 3: −0.89443 on MPS4 (cores 1,2) and −0.86177 on MPS6 (cores 2,3). The rotation eigenvalue is only the 4th (−0.17889 and −0.17235).
   - Gradient flow from that saddle plus 1e-4 noise (3 seeds) learns a1 at t ≈ 10–11 while a2 stays at 1.0000. The larger component is added in a new bond index; nothing rotates and nothing is forgotten. This is case 2 of `theory_saddles` §4.4.
   - Separated pairs (MPS4 cores 0,2; MPS6 ends) match the agent's formula to 1e-15 and do rotate (a2 → 1e-12).
   - Fix: restate H1b as "m = 2 with k ≥ 3", add the k = 2 branch, and run `h1b.py` on `pat_mps4_adj` and `pat_mps4_sep` (~20 min).

2. **MODERATE: O9, R = 3 "random permutation close to an independent race (Thurstone) model".**
   - σ is 0.05–0.25 depending on the tree. σ-independence was only checked for σ ≤ 0.1, and binary8 winners already change at σ = 0.3.
   - The gradient-flow statistics use completed runs only (MPS6: 85/199 and 88/200; binary8: 180/200 and 177/199). The race model is built from τ of all runs, so the comparison is not on the same subset.
   - The "floor" exclusion uses the overlap product |a_r| < 1e-14, not a per-slice roundoff test. It removes 18–19 binary8 runs per ρ whose component was in fact learned later.
   - Where tested, the model is consistent: permutation χ² p = 0.18–0.82 (n = 85–195). But P(greedy order) is below the model in 6/6 m ≥ 3 cases (z from −0.03 to −1.49).
   - My no-stop rerun (MPS6, R = 3, ρ = 0.8, σ = 0.2, 9 seeds, to t = 1e9): 4/9 runs never learn one component (agent: 54%). This agrees with the agent's label every time, and in 3/3 runs where |a| < 1e-25 fired the component never recovered.
   - The stuck component was the largest (seed 3), the middle one (seeds 2, 11) or the smallest (seed 1). So exclusions are not neutral with respect to order.
   - Fix: compare gradient flow and model on the same subset, report the excluded runs' own race orders, state σ and d = χ = 4, and drop or redo the "floor" rule using slice norms.

3. **MODERATE: R = 4 "gradient flow mostly stalls before the last component".**
   - Single σ per tree, d = χ = 4, fan-in init only.
   - "Stalled" means not learned by t = 1e9. That includes slow, pinned routes (seed 2 above never fell below 1e-25).
   - The no-stop cross-check (the abandoned `multi` run) covered binary4 and MPS4 plus 25 fast MPS6 R = 3 runs. It did not cover binary8 or MPS6 R = 4.
   - The number of components learned at the stop is not recorded, so "before the last" is unverified for R = 4.
   - MPS4 (K = 4) stalls in 72% of runs, which conflicts with the "K ≥ 5" framing of `theory_regimes` R10 and needs a cross-reference.
   - Caution for later use: the 1e-25 overlap-product stop also fires during a normal forget phase on a k = 6 rotation (MPS6 ends, t ≈ 200).

4. **MODERATE: the m = 2 law in the REPORT summary and O6 is stated for all matrix-like pairs.**
   - Adjacent pairs measure γ = 0.67±0.09 / 0.22±0.03 (MPS6) and 0.82±0.06 / 0.31±0.02 (MPS4), which is 2–3.4× steeper than (1−ρ)/ρ.
   - The predicted n = χ_e law (exponents 1.0 and 0.444) is not reached by σ = 1e-5.
   - At ρ = 0.9 the local slopes are 0.25, 0.31, 0.42, 0.34 (last one ±~0.08), so "still steepening" is not shown.
   - Fix: carry n in every statement and tag n = χ_e as a conjecture.

5. **MODERATE→MINOR: "m ≥ 3: no strict saddle, wrong order kept"** rests on one tree (MPS4, m = 4): 8 seeds × 5 σ over one decade, plus 40 natural runs at a single σ. State the scope or add binary8 or MPS6.

6. **MINOR: fits for separated pairs.**
   - The fit choices differ: σ ≤ 0.01 for binary4, the full range for MPS6 (where P reaches 0.41), and the restricted fit quoted for MPS4-sep at ρ = 0.9. The full-range MPS4-sep fit there is 0.095±0.006, 2.5 SE low.
   - No goodness-of-fit is reported. At ρ ≥ 0.9, P changes by less than 2× over the binary4 range.
   - `theory_escape` §5.5 reports pre-asymptotic leading-order slopes and a drift between gradient flow and leading order. Use "consistent with", not "confirmed".

7. **MINOR: common seeds bias the P tables.**
   - In all 17 MPS settings, gradient-flow P(larger first) is 0.018 ± 0.02 below the pooled race curve at every ρ, while per-seed agreement with the race is 100%. That is a fluctuation of the shared seed set, rechecked only for one setting.
   - Race-curve ranges are 0.66–0.89 (ρ = 0.5) and 0.55–0.66 (ρ = 0.8), against the quoted 0.60–0.88 and 0.50–0.65.

8. **MINOR: the d-pairs test ("χ² with 6 dof") is invalid.**
   - The 6 ρ rows reuse the same 400 seeds, and d = 2 and d = 4 inits share core 0's random draws (I verified this).
   - χ² = 0.56 and 1.34 are implausibly small for 6 dof (p ≈ 0.003 and 0.03).
   - The MPS conclusion still holds by derivation. The "empirical" binary8 conclusion is untested.

9. **MINOR: sd(log τ) = 0.57 (K−2)^0.61 χ^−0.53 does not separate into a K factor and a χ factor.** On MPS alone, the K-exponent falls from 0.66 (χ ≤ 2) to 0.52 (χ ≥ 16), and binary8 sits +6% above the fit. The ±0.03 comes from structured residuals. Better wording: "≈ √(K−2) at large χ".

10. **MINOR: selection-rule exponent q.** q/(K−2) is 1.0–1.25, above 1 in 22/23 m ≥ 3 settings. No SE is given, the accuracy is in-sample and pooled over ρ, and there is no accuracy reported at fixed q = K−2.

11. **MINOR: H1b precision understated, and one statement too broad.**
    - Local slope × |λ| = 1.0000, 1.0002, 1.002, 1.017, 1.07 going from σ = 1e-7 to 1e-2. The quoted 1.5% excess is curvature from σ ≥ 1e-3; asymptotically the slope is exact to 2e-4.
    - "λ_min lies between 0 and the saddle value" is false at σ = 0.1, where it is ≈ −0.30.

12. **MINOR: reproducibility.**
    - The README still describes `multi.py` as using the library integrator and `analyze_multi.py multi`; the code actually uses `gf_events` and `multi2`.
    - `tol_check` skips MPS8, the tree that needed the clock-restart repair.
    - The `symmetry_check` log shows the MPS8 half unfinished.

Not verifiable without the raw data:
- that wrong-order seeds are nested across σ;
- the claim that later escapes come "10%–2× later";
- the 1679-run match between `multi` and `multi2`.

**Claims I consider solid (within the stated scope)**
- The leading-order scale symmetry holds. First-event P is identical for σ 0.003–0.1 (MPS4), 0.01–0.1 (binary8) and the MPS6 m = 4 pattern, with ≥ 99% of per-seed winners unchanged.
- The m-count classification: I checked the derivation, and it matches `theory_escape` §3.3 and `theory_saddles` §4.5.
- MPS exact race: the gradient-flow winner equals argmin τ_r/s_r in 40800/40800 runs at σ = 0.01. Binary8 agrees in 96–99%.
- P(larger first) tables versus ρ, K, d, χ (400 seeds, SE ≤ 0.025), and the trends: P → 1/2 as K grows and rises with χ. With uncapped χ, sd ∝ χ^−(0.48–0.54) per tree, checked in gradient flow at χ = 16 and 32.
- Separated m = 2 pairs follow a power law with exponent ≈ (1−ρ)/ρ. MPS4-sep at ρ = 0.8 has local slopes 0.244, 0.247, 0.222, 0.245 over σ = 1e-1 to 1e-5 (prediction 0.25); binary4 agrees at ρ = 0.7–0.9.
- H1b for m = 2, k ≥ 3: the eigenvalue formula (now confirmed at K = 3, 4, 6), the rotation, forgetting, the plateau at s2²/2, relearning after ~σ^−(k−2), and a log-law slope of exactly 1/|λ|.
- Natural wrong-order runs on binary4: |λ|(t1−t2) is flat in σ (slope −0.02, where a log law would give +1) and ≈ Exp(1)-distributed. No wrong plateau is resolved.
- The MPS4 contrast: no rotation, |λ_min| ∝ σ^2.00, t1 ∝ σ^−1.97.
- R = 3 on MPS4 (195/200 runs used) is consistent with the independent race; binary4 is far more greedy.
- R = 4 = χ on MPS4 at σ = 0.1: most runs never learn all four components, confirmed without the stall stop.

**Plain-language summary**
- A tree tensor network writes one large weight tensor as a contraction of small "core" tensors linked by internal "bond" indices; an MPS (tensor train) is the chain-shaped tree.
- The target is a sum of orthogonal rank-one tensors with weights s1 > s2 > …. Training from a tiny random start learns them one at a time.
- When two terms differ on m ≥ 3 cores (the generic tensor case), which one comes first is a lottery set by the starting point and independent of the init scale. A 2:1 weight ratio gives the larger term only a 60–89% chance.
- When they differ on m = 2 cores (matrix-like, e.g. a 3-layer linear network), the larger term wins with probability tending to 1 as the init shrinks, but only as a slow power law.
- When the smaller term wins in the non-adjacent m = 2 case, training passes a saddle, swaps the learned term for the larger one, and later relearns the smaller one. For adjacent pairs it simply adds the larger term. For m ≥ 3 the wrong order is kept.

**Work left for `order`:** two small reruns, plus scope and wording edits.
- An H1b run on the adjacent and separated pattern pairs (about 30 CPU-minutes).
- A subset-matched re-analysis of R ≥ 3 from the `multi2` raw outputs, which are not on this machine.

No new sweep is needed for the two-component headline results.