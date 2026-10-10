# Verification of real-data fix round 1 + round 2 (incl. the pre-registered order test) — 2026-10-10

Independent verifier, local, read-only on `data/`. Inputs: `notes/real_data.md` §5, §8, §9; `experiments/real_data/`;
`agent-runs/real-data/REPORT.md`; review `notes/reviews/real_data_round1.md`; the lead's fix list; raw outputs in
`data/real_data/` and `data/real_data_local/`. Own scripts (written from the notes, not from the agent's analysis
code) are in the lead's scratchpad, `verify_rd2/` (`order_recompute.py`, `gaps_edges.py`, `route_check.py`,
`mps_check.py`, `misc_check.py`; outputs `*.out`). About 30 min, at most 3 processes.

**Bottom line.** The pre-registration is intact: the registration and predictor commits come before every new run. The
registered analysis was followed and reproduces to Monte-Carlo error. Every MUST FIX item is fixed in the notes. Two
items are only partly fixed: the REPORT body still carries stale lines, and the locality wording still overstates the
transposed quadtree. The main numbers reproduce from the raw rows with independent code. Small errors:
- 27/29 should be 28/29;
- the "11/11 out of sample" count includes one in-sample case;
- "B < 0 in all series" holds for raw features only;
- the binary lambda_min "consistent with 1" rests on a 3-point seed;
- one interim look at the outcomes is not disclosed.

## 1. Pre-registration integrity

| check | result | verdict |
|---|---|---|
| Registration 2f4d99e (11:51:13 +01:00) before every order-test run | The first new run started at about 11:53:55 (file mtime minus the wall time, which `run.py` starts before data loading). The last finished at 13:04:31 (70.6 min, as stated). §9.1-9.5 at HEAD are byte-identical to 2f4d99e. | pass |
| Predictor 44feb2f (11:52:58) before the outcomes | Committed 57 s before the first run started. `order_predictor.py` and `extra_orders_predictor.json` are unchanged since. Rerunning the predictor for scrambled105, hilbert and scrambled110 gives S and the best rank-1 loss bit-identical. The pixel orders of all 81 runs match the predictor's. | pass |
| Reused runs (seen before registration) | The 7 files (copied 11:53:06) are byte-identical (md5) to the main-sweep quadtree seeds 0-2 and to the review quadtreeT/rowmajor seeds 0-1. Settings equal main.yaml. They are declared in §9.3 and handled by secondary (a). | pass |
| Run config and analysis script | `configs/extra_orders.yaml` and `order_test.py` were first committed in 8f22642 (11:57:38). That is about 4 min after the runs started, but 4.5 min before the first outcome existed (12:02:09). The config matches §9.3. After outcomes, `order_test.py` changed only by a guard for n < 4, which has no effect. | pass (minor: config not committed before launch) |
| Interim look | `data/real_data_local/extra_orders_summary_partial.{csv,json}` was written at 12:58:01, with 72/81 runs. Those include all 60 random-order runs, i.e. the complete within-random primary data. The n < 4 guard suggests `order_test.py` was run on that partial table. Neither is in the §9.6 "Deviations", which list only the process counts. Nothing in the design or analysis changed afterwards. | disclose (no effect) |
| Analysis as registered | Per order: seeds averaged first (O1 = mean L(1e4); O2 = mean log10 tau(L <= 0.25), censored at 1e4, 4 censored seeds). Spearman over 27 orders and within the 20 new random orders. 10,000 bootstrap resamples over orders (percentile CI). 10,000 permutations. Secondary (a)-(c) are as registered. The CSV outcomes match my own computation from the raw npz (6 runs, all digits). | pass |
| Exploratory vs confirmatory | The plateau-state share (rho -0.83 / -0.85) is labelled exploratory. However, the "final form of the claim" in §8.4(5) adds "transposed quadtree among the worst". In the registered data quadtreeT ranks 18/27 by O1 (21, 18 and 15 of 27 per seed), so it sits in the lower half, not among the worst. The quadtree being lowest is descriptive only: its outcome was seen before registration. | wording fix (E2) |

Recomputed primary values (own code, three RNG seeds; the spread shows the Monte-Carlo error):

| scope, outcome | reported | recomputed |
|---|---|---|
| all 27, O1 | -0.60 [-0.82, -0.22], p 0.0013 | -0.600 [-0.82, -0.21...-0.22], p 0.0008-0.0011 |
| random 20, O1 | -0.55 [-0.85, -0.09], p 0.015 | -0.550 [-0.84...-0.85, -0.09...-0.10], p 0.013-0.016 |
| all 27, O2 | -0.30 [-0.59, +0.06], p 0.12 | -0.305 [-0.59, +0.07], p 0.12 |
| random 20, O2 | -0.26 [-0.63, +0.23] | -0.260 [-0.63, +0.21...+0.23] |
| secondary (a), O1 / O2 | -0.51 [-0.80, -0.08] / -0.22 | -0.515 [-0.80, -0.07...-0.08] / -0.223 |

Note (not a deviation): S depends only on which sibling pairs share a half. So quadtree, rowmajor and snake share
S = 0.989, and quadtreeT and colmajor share 0.338. The 27-order analysis therefore has 24 distinct predictor values,
and the cleaner confirmatory result is the one within the random orders.

## 2. Review MUST FIX items

| # | item | status | evidence / residual |
|---|---|---|---|
| 1 | Locality claim | **fixed** (one wording residual) | Renamed to an order effect: route amplitudes are measured, the mechanism is tagged conjecture. One value per (order, seed) and distributions are reported. Early speed is "not faster" (9/25 same-seed comparisons). At sigma = 0.5 the effect is shown as "not gone". The pre-registered test (27 orders x 3 seeds) replaces the optional runs. Residual: "transposed quadtree among the worst" (E2). |
| 2 | "MPS fastest" | **partly fixed** | The notes and README are narrowed correctly (tau clock, MNIST, snake order; absolute clock; Fashion; shape and order confounded). The REPORT body still says "MPS is the fastest tree at every sigma" (Summary) and "fails on every task and size" (Caveats, line 153). Only the top addendum supersedes these (E7). §7 (chi = 8) keeps "MPS is fastest at small sigma" (E8). |
| 3 | k = 3 attribution | **fixed** | Own window-checked attribution on the 124 binary raw runs: 91 kept, 16 between drops, 2 merged, 15 other route, identical to `scaling.json`. scrambled1 on the main.yaml grid: 1.34 / 1.28 / 1.22 / 1.23, as stated. scrambled2 is now one clean series (1.13). New: the "B < 0" statement holds for raw features only (E3). |
| 4 | Fit weighting | **fixed** | `scaling.fit_seed` and `events_table` fit the log law and A/sigma + B with relative residuals. Log-law rms: 0.11-0.30 over the 11 clean series, 0.12-0.30 over the 10 full-sweep series (own fits). |
| 5 | MPS first route -> "partly" | **fixed** | Own count from the round-2 summaries: the first rise is at tau = 0 in 10/35 runs (seed 4 at every sigma; seed 0 at sigma >= 0.2). It is edge 5 alone in 25/25 resolved runs, and before the first drop's peak in 24/25. The Hessian state is re-described. |
| 6 | Regimes | **fixed** (wording) | Rich at one sigma, the boundary depends on the threshold, lazy runs unconverged, and the floor claim is withdrawn. Numbers not recomputed: they come from the unchanged `regime_analysis.json`. |
| 7 | Robustness | **fixed** | Fashion siblings rise together in 3/15 (quadtree) and 2/15 (scrambled0), own count. chi = 8 relabelled L(tau_end) and its sigma = 0.5 comparison withdrawn. The uncommitted exponents are withdrawn. 8x8: 1 permutation, 2 seeds, tau = 3000. |
| 8 | De-duplication | **fixed** | 28/124 states are dropped as not rank 1, and per state 27/28 and 64/68 hit (own count). Whitened S2S is 17/30 (11/15 quadtree, 6/15 scrambled0). The unit count depends on the grouping (E4). |

SHOULD FIX items, all done:
- t1 tagged proven.
- "Later routes higher degree" withdrawn.
- `scaling.json` name fixed.
- `main_extra.yaml` added.
- 8x8 horizon label fixed.
- Kendall tau restricted to binary trees.
- `sibling_pair` parent check added.
- `REAL_DATA_DIR` override added.
- README commands added.

The binary lambda_min item is done but its reading is weak (E5).

## 3. Main numbers recomputed (own code on `data/real_data_local/full/main_summary.json` rows)

| claim | reported | recomputed | verdict |
|---|---|---|---|
| k = 3 gap, 11 clean series (fix 1) | p 1.22 ± 0.03 (1.06-1.36); B < 0 11/11; A/sigma + B rms <= 0.060; log rms 0.11-0.30 | 1.216 ± 0.031 (1.06-1.36); 11/11; 0.060; 0.109-0.301; gaps equal drop1 - drop0 of the round-2 summaries | ok |
| k = 3 gap, full sweep (round 2) | 10 series, 1.20 ± 0.03 (1.02-1.32), B < 0 10/10, log rms 0.12-0.30 | 1.198 ± 0.032 (1.02-1.32), 10/10, 0.122-0.301, A/sigma + B rms <= 0.060 (raw). The one whitened series meeting the same rule has B = +10.8, p = 0.91 | ok for raw (E3) |
| MPS later edges, fair fits | 6 of 9 edges log-time (3, 4, 6-9), 27/29 seed fits; edge 4 14.7 ± 1.5, edge 3 30.3 ± 0.8 | 6 of 9 edges; seed fits 5 + 4 + 4 + 5 + 5 + 5 = **28/29**; 14.7 ± 1.5, 30.3 ± 0.8; edges 1, 2 power-like (0/5, 1/5), edge 0 mixed (2/5) | ok, count E1 |
| Route-amplitude first-route hits | 22/26 units (2 mixed, 2 miss); 11/11 out of sample | Per state 91/96. Units depend on the grouping: 20/24 (2 mixed, 2 miss) when grouped by same predicted pair and plateau loss within 0.01; 19/23 (3 mixed, 1 miss) by loss only; review 28/30; one state per (order, seed) at the smallest sigma 20/20. Review runs 11/11, of which 10 are new (quadtree seed 0 at sigma = 0.1 is a main-sweep case). | ok as "~80-85% of units, chance 25%" (E4) |
| Binary lambda_min exponent | 0.95 ± 0.07, consistent with 1 | 0.945 ± 0.068. Per seed 0.871, 0.888, 0.873 over 5-6 sigma (0.6-0.78 decades), and 1.149 for seed 4 over 3 sigma (0.3 decade). MPS 0.047 ± 0.016 reproduces. | number ok, reading weak (E5) |
| "MPS fastest", narrowed | tau clock first; absolute clock later 4/5 (sigma = 0.1), 5/5 (<= 0.07); Fashion slowest | On the tau clock, the MPS median is first at all 7 sigma x 6 loss levels (sigma = 0.1: 16/56/290/377 vs 122/333/3730/3813). t_a(MPS)/t_a(quadtree) at sigma = 0.1 is 7.6/0.82/131/2.0/6.4, scaling as 1/sigma; the MPS escapes later in 5/5 seeds at sigma <= 0.07, 4/5 at 0.1-0.2 and 3/5 at 0.3-0.5. On Fashion the MPS is slowest to L <= 0.3-0.15 at every sigma but the only tree to reach 0.12/0.1 early. Final loss is MPS < quadtree in 5/5 seeds at sigma <= 0.3 and 3/5 at 0.5. | ok (E9 nuance) |

## 4. Errata (to apply; none changes a verdict)

- **E1.** Change "27 of 29 seed fits" to **28 of 29** (edges 3, 4, 6-9 have 5, 4, 4, 5, 5, 5 log-preferring seeds). Locations: §8.6(3), §5 row "k = 2 gap", README, and the §8.13 table.
- **E2.** "Transposed quadtree among the worst orders" (§8.4(5), §5 locality row, REPORT addendum, README) applies only to seed 0 of round 1. In the registered 27-order test quadtreeT ranks 18/27 by O1 (21, 18 and 15 of 27 per seed) and colmajor 21/27. Proposed wording: "equally local orders (quadtreeT, colmajor, hilbert) rank 13-21 of 27".
- **E3.** The k = 3 full sweep's "B < 0 in 10/10" and "all 10 window-checked seed series" (§8.1, README) hold for **raw features**. The one whitened series that meets the same rule (quadtree seed 3, sigma 0.05-0.2, `scaling.json` first_route binary_quadtree/white) has B = +10.8 and p = 0.91. Say so.
- **E4.** §8.3, §5 and README: "11/11 out of sample" should read "11/11 review runs (10 new orders or seeds; quadtree seed 0 at sigma = 0.1 repeats a main-sweep case)". "22/26 units" holds for the agent's TV grouping. Add that other reasonable groupings give 19-22 hits of 23-26 units (1-2 misses), and 20/20 with one state per (order, seed).
- **E5.** §8.7, §5 and README: replace "0.95 ± 0.07 (4 seeds), consistent with 1" with "0.87-0.89 on the three seeds with 5-6 sigma values (0.6-0.78 decades) and 1.15 on one 3-point seed; mean 0.95 ± 0.07". "Higher order" (far from 0) stands. "Consistent with 1" is not established, so do not withdraw the observation that the exponent is below 1 at these sigma.
- **E6.** §9.6 "Deviations": disclose the interim look (12:58:01, partial summary of 72/81 runs including all 60 random-order runs; then the n < 4 guard in `order_test.py`). Also disclose that the run config and analysis script were committed at 11:57:38, about 4 min after the runs started and before the first outcome. Neither affects the result.
- **E7.** REPORT body: mark the stale lines inline ("[superseded, see addendum]"). These are the Summary sentences ("MPS is the fastest tree at every sigma", "88% ... 21%", "vanishes at sigma = 0.5", "gap ∝ sigma^-1.0") and Caveats line 153 ("fails on every task and size"). The round-0 table rows also need the mark.
- **E8.** §7, chi = 8 paragraph: "The MPS is fastest at small sigma (0.088 / 0.080)" is unannotated. It is L(tau_end), and 9/15 MPS runs are wall-limited. Write "lowest L(tau_end)".
- **E9.** README "on Fashion it is the slowest tree": add "to L <= 0.3-0.15; it is the only tree to reach L <= 0.12 early at sigma <= 0.14".

## 5. Safe to cite

| statement | status | source |
|---|---|---|
| Pre-registered: a data-only route-amplitude share S predicts the final loss across 27 binary16 pixel orders at sigma = 0.1 (Spearman -0.60 [-0.82, -0.22]; within 20 random orders -0.55 [-0.85, -0.09]). It does not predict the time to L <= 0.25 (-0.30 [-0.59, +0.06]). | empirical, pre-registered | `results/extra_orders_summary.csv`, `extra_orders_predictor.json`, `order_test.json` |
| Pixel order changes the final loss at fixed time (sigma = 0.1: O1 0.097-0.244 over 27 orders); the quadtree order is lowest at every seed. Equally local orders are not near the best (ranks 13-21/27) and high-S random orders are (scrambled110 0.102). The mechanism via degree-5 vs degree-7 routes is a conjecture. | empirical (quadtree seen before registration: descriptive) | `extra_orders_summary.csv` |
| Binary first route after the escape = a sibling pair of bottom cores (k = 3), in all 20 (order, seed) pairs. | empirical | `data/real_data_local/full/main_summary.json`, `fix1_analysis.md` |
| k = 3 first-route gap: A/sigma + B fits all 11 clean and all 10 window-checked raw series (relative rms <= 0.06, B < 0). The free exponent is 1.22 ± 0.03 / 1.20 ± 0.03, not robust. The log law is rejected (rms 0.11-0.30). | empirical (form derived) | `scaling.json` first_route, `fix1_analysis.json` first_route |
| MPS: the first rank rise (edge 5 alone, 25/25 resolved) is merged with the escape (24/25 before the first drop's peak; 10/35 at tau = 0). Later edges 3, 4, 6-9 rise in log time (28/29 seed fits; edge 4 slope 14.7 ± 1.5); edges 1, 2 do not. | empirical | `main_summary.json`, `scaling.json` edge_scaling |
| Route-amplitude rule: the largest sibling amplitude picks the first pair in 91/96 rank-1 states, about 80-85% of critical-point units (chance 25%), and 11/11 review runs. | empirical | `results/routes_raw.json`, `fix1_analysis.md`, `data/review_real_data/audit/audit_table.json` |
| Plateau Hessians: the MPS post-escape plateau is strict (exponent 0.047 ± 0.016); the binary rank-1 plateau is higher order (0.87-0.89 on 3 seeds). | empirical | `results/hessians_raw.json` |
| MPS vs binary: on the tau clock (MNIST, snake order) the MPS reaches every loss level first at every sigma. On the absolute clock t_a(MPS)/t_a(binary) ∝ 1/sigma with seed prefactors 0.8-131 at sigma = 0.1. On Fashion it is slowest to L <= 0.15-0.3. Shape and order are confounded. | empirical; 1/sigma scaling proven | `main_summary.json`, `fashion_summary.json`, `tables.md`, `fashion_tables.md` |
| t1 ∝ sigma^-(K-2) per init: an identity of small-init flow. | proven | `scaling.json` |
| Whitened S2S at sigma <= 0.1 is 17/30. Fashion siblings rise together in 3/15 and 2/15 runs. | empirical | `main_summary.json`, `fashion_summary.json` |

Not to cite as stated:
- "transposed quadtree among the worst" (E2);
- binary lambda_min "consistent with 1" (E5);
- "27/29" (E1);
- "11/11 out of sample" (E4);
- the REPORT body's round-0 summary and table (E7).
