# Review of real-data (round1) — 2026-10-10

Two independent local reviewers: (A) partial reproduction from the committed code plus an independent reimplementation, (B) adversarial audit of claims vs data. Re-run outputs in `data/review_<folder>/` (local, gitignored).

## A. Reproduction

**Bottom line.** Every quantitative result I re-ran reproduces, and so does my independent reimplementation. The first-plateau times match the committed values to about 1e-15 (bit-identical in practice), the gaps between learning events within 1.3%, and the Hessians and route amplitudes within 4%. One full 1e4-horizon run and one lazy run reproduce the committed per-run values to 4–7 digits. No bug I found changes a headline verdict. The problems are in how some results are worded, a measurement artifact in one count, and numbers that cannot be traced to a committed script or config.

Background for a reader new to tensor networks: the model's weight tensor W (16 legs, one per pixel) is stored as a contraction of small "core" tensors placed on a tree. The rank of W unfolded across a tree edge is that edge's "bond rank". Training from a small initialization proceeds in jumps: long flat stretches of the loss (plateaus) broken by sudden drops. In each drop a "route" (a connected set of k cores) adds one more singular value on its edges.

## What I reran (about 30 min wall time, at most 4 processes)
- **The agent's own code**, run through a wrapper because the data path is hardcoded (see the README check). All runs use seed 0 with the main settings and shorter horizons unless stated:
  - quadtree binary tree at σ = 0.05, 0.1, 0.2, 0.3, 0.5;
  - scrambled0 binary tree at σ = 0.1, 0.5;
  - MPS (snake order) at σ = 0.05, 0.1, 0.2, 0.5;
  - two full main-sweep runs to τ = 1e4 (quadtree and scrambled0, σ = 0.1);
  - one centred lazy-regime run (σ = 4);
  - Fashion-MNIST quadtree, σ = 0.14; and χ = 8 quadtree, σ = 0.14.
  - I then ran `summarize.py`, `route_amplitudes.py` and `plateau_hessians.py` on these runs.
- **Independent check**, using none of the agent's experiment code and only `TTNSpec`, `init` and `forward` from the library. I wrote my own:
  - data pipeline;
  - gradient-flow integrator (chunked DOP853 then RK45);
  - loss-drop detection;
  - data-weighted bond spectra (down-messages by autograd);
  - route amplitudes from the dense tensor;
  - closed-form linearized loss L_lin via a full Jacobian.
  
  I applied it to quadtree σ = 0.2 and 0.5, MPS σ = 0.1, scrambled0 σ = 0.5, and the lazy run.

## My numbers vs theirs (seed 0)
| claim | quantity | mine | committed |
|---|---|---|---|
| 1 | t1 exponent, quadtree, σ 0.05–0.5 / MPS | 12.999997 / 13.999999 | 13.000 / 14.000 |
| 1 | t_a, quadtree, σ 0.05–0.3 (agent code) | equal to ≤3e-15 | regime_analysis.json |
| 1 | t_a, independent code (quadtree 0.2, MPS 0.1, scrambled0 0.5) | differs by 2.6e-9, 3e-9, 3.4e-7 | — |
| 2 | k=3 gap, quadtree, σ 0.05/0.1/0.2/0.3/0.5 | 297.80 / 142.96 / 68.45 / 44.77 / 29.2 (independent at 0.2: 68.71) | 297.81 / 142.66 / 68.54 / 44.86 / 29 |
| 2 | k=3 gap, scrambled0, σ 0.1 / 0.5 | 352.51 / 39.08 (independent 38.55) | 352.17 / 38.59 |
| 2 | first edges to gain rank | quadtree [3,4]; scrambled0 [10,11]; MPS [5] (σ ≤ 0.2), [5,6] (σ 0.5); same in independent code | same |
| 2 | MPS edge 4 / edge 3 slope vs ln(1/σ) | 17.0 / 32.8; rms: log 0.15 / 0.07 vs power 0.31 / 0.19 | 15.3±1.5 / 31.4±1.1 |
| 2 | MPS Hessian λ_min, σ 0.05/0.1/0.2 | −0.0567 / −0.0573 / −0.0577 | −0.0563 / −0.0567 / −0.0571 |
| 2 | binary Hessian λ_min, σ 0.05/0.1/0.2/0.3 | −0.0144 / −0.0267 / −0.0502 / −0.0664 | −0.0144 / −0.0260 / −0.0502 / −0.0639 |
| 2, 4 | share of route amplitude on k ≤ 5, quadtree | 0.911–0.938 (independent 0.927) | 0.911–0.938 |
| 2, 4 | same share, scrambled0, σ 0.1 / 0.5 | 0.029 / 0.097 (independent 0.107) | 0.029 / 0.098 |
| 2 | largest-amplitude sibling pair = first pair to grow | yes in all 7 runs | 34/35, 78/89 |
| 4 | L(1e4) / test accuracy, σ 0.1, quadtree | 0.09572 / 0.950; ρ [15.67, 19.83, 290.31, 12.2] | 0.09572 / 0.950; ρ [15.7, 19.99, 290.55, 12.2] |
| 4 | L(1e4) / test accuracy, σ 0.1, scrambled0 | 0.2025 / 0.889 | mean 0.207±0.008 / 0.872 |
| 4, 5 | time to L ≤ 0.2 at σ 0.1: MPS / quadtree / scrambled0 | 52 / 333 / never | 56 / 333 / 6730 |
| 3 | lazy run (σ = 4): L(T) / L_lin(T) / rank J / δK | 0.1919350 / 0.1919350 / 304 / 3.0e-6 | identical |
| 6 | plateau creep, quadtree σ 0.05 | L 0.304 → 0.265 between the first two drops | "creeps" |
| 7 | Fashion σ 0.14 | drops 0.495 → 0.333 → 0.130 with no rank growth; λ_min −0.1246 | 0.50 → 0.33 → 0.12; −0.1266 |
| 7 | χ = 8 first pair to grow | [3,4], the same pair as at χ = 4 | same |

## Discrepancies and issues
1. **"Locality vanishes at σ = 0.5" (claim 4) is only true for the final loss.** In the agent's own table the quadtree still reaches each loss level 3–5× faster than the scrambled trees at σ = 0.5 (L ≤ 0.2 at τ 85 vs 374; L ≤ 0.15 at 134 vs 642). The χ = 8 table shows the same (40 vs 208). In my rerun the quadtree reaches 0.2 at τ = 44.5 while scrambled0 is still at 0.30 at τ = 150. The speed advantage shrinks from about 20× to 3–5×; it does not vanish.
2. **The MPS "single edge (edge 5) first" count (34/35) includes threshold artifacts.** On edge 5 the ratio s2/s1 never falls below the 0.05 threshold at σ ≥ 0.2 (minimum 0.06 at σ 0.2, 0.146 at σ 0.5). `rank_events` then records the event at τ = 0. The agent's `scaling.json` shows the same for seed 4 at all σ and seed 0 at σ ≥ 0.2, so about 10 of the 35 runs have no resolved rank increase. The physical reading (a fast 2-core route during the escape) is plausible, but the count is overstated. `analysis.events` already has a `pre_escape` flag that `summarize` does not use.
3. **Power-law and log-law fits are not weighted the same way** (analysis standard 5, and contrary to the notes' text). The power law is fitted in log space; the log law by unweighted least squares in linear space; both are then scored by relative error. With a fair relative-error fit the log-law error drops 2–4× (quadtree seed 0: 0.28 → 0.16; scrambled0 seed 0: 1.10 → 0.30), but the power law still wins clearly. The verdicts hold; the reported separation is inflated.
4. **The k = 3 gap exponents differ by pixel order.** scrambled1 gives 1.27 ± 0.02, which is 13 standard errors above 1, and the "A/σ + B" model fits it worse than a free power law (error 0.134 vs 0.030). scrambled2 gives 0.74. The headline "σ^-1.0" averages over these. The scrambled1 value also relies on 14 runs that the committed `main.yaml` (220 cases) does not generate; the analysed sweep has 234 runs.
5. **Stale or untraceable numbers:**
   - The notes and REPORT give 0.91 ± 0.23 (2 seeds) for scrambled2; the committed `scaling.json` says 0.74 ± 0.21 (3 seeds).
   - The χ = 8 first-route exponents (1.17 ± 0.06, 0.97 ± 0.11, 0.79 ± 0.28) and the Fashion σ^-0.92…−0.97 are in no committed output, and no committed script produces them. `scaling.py` needs at least 4 σ values; those sweeps have 3.
6. **Fashion:** the notes do not mention that in 12 of 15 binary runs a single edge rises first, so the two siblings do not rise together.
7. **Route amplitudes are pooled over some states that are not near rank 1** (rank-1 error up to 0.995 at σ ≥ 0.3). Excluding states with error above 0.05 leaves the results intact: 27/28 and 64/68 correct predictions, shares 88% vs 20.5%. The amplitude also depends on how the cores are scaled, but the bottom cores are balanced (top singular values 1.09–1.32), and my unit-normalized version gives the same shares.
8. **Minor:**
   - Phase 1 stops at the first integrator step below the 1% loss floor, so the τ origin overshoots by about 0.9 units on MPS σ = 0.1; gaps and t1 are unaffected.
   - `tables.sibling_pair` does not check that the two edges share a parent.
   - The 8×8 table is labelled L(1e4) although that horizon was 3000.
   - The "bottom-up" Kendall tau includes the MPS, where "level" is just position along the chain (Fashion MPS gives −0.24).

## README check
- `download.py` works as written (14 s).
- `data.py` hardcodes `DATA_DIR=/root/outputs/real_data/data` (as the `load_task` default) with no override. `run.py`, `route_amplitudes.py`, `plateau_hessians.py` and `regime_analysis.py` fail unless the data sit at that path. I patched it at runtime without editing repo files.
- `summarize.py` runs as written.
- The README omits `chi8b.yaml`, `tol_check.yaml`, and the commands that produce the `fashion_*`, `chi8_*` and `img8_*` tables.
- The code is deterministic on this machine.
- The repo working tree is clean.

## Files
Notes and scripts are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_real_data/`:
- `findings_part1.md` to `findings_part4.md`
- `indep.py`, `lazy_check.py`, `compare.py`, `rr_run.py`, `rr_script.py`, `configs/`

Run outputs are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_real_data/`, in `repro/`, `long/`, `lazy/`, `robust/`, `indep/` and `analysis/` (summaries, `routes.json`, `hessians.json`).

Another reviewer is also writing to `review_real_data/` (its `code/` and `audit/` subfolders); I did not touch those.


## B. Claims audit

**Claims audit: real-data agent (B5), MNIST 3 vs 5 at 4×4 pixels**

**Terms.** A tree tensor network (TTN) stores the model's weight tensor W, which has one index per pixel, as small "core" tensors placed on a tree. The rank of a tree edge (its "bond rank") is the matrix rank of W when the pixels are split into the two sides of that edge. Training starts from a small random initialisation of scale σ. The loss then sits on plateaus and drops in steps. Each step raises the rank along a "route" of k cores, and theory predicts plateau lengths of about σ^-(k-2), or about log(1/σ) when k = 2. A strict saddle has a negative Hessian eigenvalue that does not depend on σ, so escape is fast. At a higher-order saddle that negative curvature vanishes as σ → 0, so escape is slow.

**What I checked.** All committed JSON files, figures and scripts, plus 18 new gradient-flow runs using the agent's code (σ = 0.1 unless stated, at most 4 processes). The quadtree run at seed 0 reproduces the committed numbers exactly: final loss 0.0957, drops at τ = 15.5, 158.6, 339.6, 2157.8 and 4743.8.

**Bottom line.** The results that follow from the tree structure hold up. The two comparative headlines (locality, and "the MPS is fastest") do not hold as stated. One analysis bug biases the k = 3 exponent low.

### Ranked issues

**1. Major: claim 4 (locality).**
- **An equally local order is among the worst.** I ran the quadtree on the transposed image: same 2×2 blocks, but vertical instead of horizontal pairs. Its loss at τ = 1e4 is 0.226 (seed 0) and 0.194 (seed 1), against 0.096 and 0.099 for the quadtree. At seed 0 it is worse than all 6 new random orders (0.119–0.224). Row-major order gives 0.136 and 0.099.
- **The "21%" baseline is unrepresentative.** It comes from only 3 permutations, whose share of interaction on short routes (k ≤ 5) is in the low tail (0.01–0.12 at seed 0). Six new random orders give 0.02–0.88, mean 0.42, which matches the pair-count chance level of 0.43.
- **The error bars are pseudo-replicated.** The ±0.02–0.03 SEs treat runs at different σ as independent, but the same seed sits at the same rank-1 critical point at every σ (about 13 distinct points in total). Using one run per (order, seed) at σ = 0.05 gives 0.89 ± 0.02 against 0.17 ± 0.07.
- **Early learning is not faster.** Six of the 8 other binary orders reach L ≤ 0.25 before the quadtree.
- **At σ = 0.5 the effect reverses**, it does not vanish: scrambled ends at 0.078, quadtree at 0.090.
- **What survives:** the quadtree is still the best of the 8 other orders run at seed 0, and the k ≤ 5 share does predict the final loss across 9 orders (Spearman −0.75, p = 0.02).
- **Fix:** call it an order/orientation effect, measured by route amplitudes. Test ≥ 10 random orders and several local orders over ≥ 2 seeds, report distributions rather than pooled SEs, and tag the mechanism as a conjecture.

**2. Major: claim 5 (MPS fastest; "route degree beats locality").**
- **It depends on the clock.** It is true only for time measured after the first escape (τ = t − t_a). In absolute gradient-flow time the MPS escapes the origin about 0.76/σ times later than the binary tree; at σ = 0.1 the measured escape times are 8.36e15 against 1.10e15. On that clock the binary tree reaches every loss level first.
- **It fails on Fashion-MNIST.** The MPS is the slowest to reach L ≤ 0.25 / 0.2 / 0.15: 117 / 124 / 374, against 35 / 37 / 44 for the quadtree. This contradicts the report's "fails on every task".
- **Shape and pixel order are confounded**, because only snake-ordered chains were run. New chains at σ = 0.1:
  - row-major: final loss 0.083;
  - scrambled0: final loss 0.084, but τ = 299 to reach L ≤ 0.25 (quadtree: 156), a 3-core first route, and staircase-like plateaus (separation ratio ρ = 8.5–30);
  - scrambled3: final loss 0.128, worse than the quadtree.
- **Per-edge timings mostly don't support it.** Only 2 of the 9 fitted MPS edges (3 and 4) clearly prefer the log law; 6 prefer power laws.

**3. Moderate: claim 2, "k = 3 gap ∝ σ^-1.0" (bug in `scaling.py`).**
- **Bug.** `first_route()` assigns the gap to the next loss drop when the route rose before drop 1, without checking that the rank increase falls inside that drop's window.
- **Confirmed by reruns:**
  - Quadtree seed 3, σ = 0.3: the route rises 12.4 after drop 0, but the code assigns 128.9, which belongs to a different route.
  - Quadtree seed 3, σ = 0.5: the route merges into drop 0, but the code assigns 34.4.
  - Scrambled2 seed 0, σ = 0.5: the route rises at 141.5, but the code assigns 560.
- **Effect:** seed 3's exponent falls from 1.24 to 0.65.
- **Clean series** (same edges on drop 1 at every σ, 11 series) give exponents 1.06–1.36, mean 1.22 ± 0.03. The form A/σ + B with B < 0 fits each one (rms ≤ 0.06), so k = 3 holds, but "1.01 ± 0.14" is an artifact.
- **Fit comparison is lopsided.** The log law is fitted unweighted in linear space but scored by relative residuals. With fair weights its rms is 0.11–0.30, not 0.56–0.91; it is still rejected.
- **Stale and unexplained numbers.** Scrambled2 is quoted as 0.91 ± 0.23 (2 seeds); the committed value is 0.74 ± 0.21 (3 seeds). Scrambled1's 1.27 ± 0.02 is called "yes" without comment.
- **Fix:** time the route by when its edges' rank rises (or require that rise to fall inside the attributed drop), use relative-weight fits for both laws, and report A/σ + B.

**4. Moderate: claim 2, MPS part ("single edge, log gap, strict saddle").**
- **The first route has no gap.** Edge 5's rank rises before the first drop's peak in every run (committed gaps −0.5 to −10.7; my rerun: τ = 9.8 against a drop peak at 11.3). The MPS first route is merged with the escape from the origin.
- **The log law is for later routes** (edges 4 and 3, not the first route).
- **The strict-saddle Hessian is measured after edge 5 has risen.** That state has 2–3 negative directions.
- The verdict should be "partly".

**5. Moderate: claim 3 (regimes).**
- **"Rich" rests on one value of σ (1.4)**, where 2 of 5 runs have no detected drops and are labelled rich by elimination.
- **The S2S/rich boundary is unstable**: it moves across σ = 0.07–1.0 as the ρ threshold changes.
- **The lazy runs are not converged** at the horizon: loss 0.130–0.192 against their own floors of 0.090–0.128. The test-accuracy ranking and the explanation "lazy runs only fit the tangent space" therefore depend on the horizon.
- **"The lazy floor (0.105) lies well above what feature learning reaches" is false per seed.** For seeds 2–4 the floor is 0.090–0.099, while the staircase (S2S) runs end near 0.095.

**6. Moderate: claim 7 (robustness), reported selectively or confounded.**
- **Fashion-MNIST:** under the "rise together" criterion used for MNIST, only 3 of 15 binary runs start with a sibling pair (12 of 15 start with a single edge). Only the weaker "first two edges to rise" count (15/15) is quoted.
- **χ = 8:** all 23 runs stopped by the 2-hour wall limit are quadtree or MPS runs, while every scrambled run reached τ = 1e4. The "L(1e4)" column is mislabelled, and the σ = 0.5 comparison is confounded by unequal training time.
- **χ = 8 exponents:** the quoted 1.17 / 0.97 are two-point slopes that appear in no committed output.
- **8×8 images:** 1 permutation and 2 seeds.

**7. Minor–moderate: pseudo-replicated counts.**
- "34/35, 78/89" counts repeats across σ of about 13 distinct critical points.
- 28 of the 124 states used for route amplitudes are not near rank 1 (relative distance up to 0.995). Two quadtree "hits" were computed after the route had already risen.
- After removing these and de-duplicating, the rule still holds in 28 of 30 (seed, critical point) units.

**8. Minor.**
- **t1 ∝ σ^-(K-2) is a mathematical identity** of small-init flow: t_a·σ^13 is constant per seed to 4 digits. It should be tagged "proven" and not counted as evidence that predictions survive on real data.
- **"Later routes have higher degree" is not established.** It rests on one series grouped by drop index plus selected per-edge exponents. Routes between two bottom cores of a binary tree always have an odd number of cores, so "k = 4–5" is not a route identification.
- **The binary λ_min exponent (0.877) covers less than one decade** and omits seed 4 (1.15).
- **Small factual errors:** whitened "80% S2S" is actually 57%; `scaling_raw.json` does not exist (the file is `scaling.json`); the committed config lists scrambled1 at 4 σ values, but the data have 7.

### Claims I consider solid
- On binary trees, the first route after the escape is a pair of sibling bottom cores, as the rank-1 counting predicts. This holds on MNIST and in the 11 new binary runs.
- Route-amplitude rule: the sibling pair with the largest amplitude goes first in 28 of 30 units and 11 of 11 new runs. The time to that first route scales as 1/amplitude (r = −0.95 across 9 orders).
- New independent check: the growth rate of the quadtree's first route has exponent γ = 1.35, against 1.33 predicted for k = 3.
- k = 3 power law (A/σ + B) beats the log law. The binary rank-1 plateau is a higher-order saddle (λ_min ∝ σ^0.87–0.89); the MPS plateau after the escape is strict (σ-exponent 0.05 ± 0.02).
- MPS edges 3 and 4 rise in log time, with slope ≈ 1/|λ_min|.
- The lazy regime equals the linearised model (to 5–6 digits at σ ≥ 2.8), and the kernel rank of 304 matches the formula.
- Plateaus creep. This is understated: between drops the loss falls by 15–250% of the next drop's size.
- Fashion-MNIST shows no locality advantage, and has a strict saddle between two rank-1 states.
- The integrator and pipeline are reproducible.

### Work left on B5
1. Copy the raw outputs (about 0.9 GB in `/root/outputs/real_data`) before the machine is deleted; every fix above needs them.
2. Fix the gap attribution, switch to relative-weight fits, de-duplicate the counts, then rewrite claims 4, 5 and 7 and the status tags. About half a day of agent work.
3. Optional runs, a few CPU-hours: ≥ 10 random and several local orders over 2 seeds, scrambled MPS chains, and longer lazy-run horizons.

### Files
- Notes: `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_real_data/audit_README.md` and `audit_notes_1.md` … `audit_notes_7.md` in the same folder.
- Runs and JSON: `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_real_data/audit/` (subfolders `runs`, `runs2`, `runs3`; `audit_table.json`, `attrib_summary.json`, `b2/`).