
## 3. Safe to cite (final claims after the errata)

**Status tags:**
- **proven, derived, empirical, conjecture** follow CLAUDE.md.
- **known / replication:** the result exists in the literature or in an earlier project theory note,
  and this agent measured it again.
- **new:** not found in `notes/literature.md` for TTNs.

All sources are in `experiments/s2s_core/results/`.

**Aligned init and first escape**

| # | claim | status | known / new | source |
|---|---|---|---|---|
| 1 | Aligned balanced init on odeco teachers follows the exact balanced Saxe modes on binary4, mps4, binary8, ht4 and mps8. Max deviation of the overlaps: 1.97e-10, 3.25e-10, 2.26e-9, 2.59e-9, 2.68e-9 (DP5, rtol 1e-10). | derived + empirical | known (Saxe 2014, DLN); explicit for TTNs | `aligned.csv` |
| 2 | **First-escape law per seed, no free parameter:** t_1 = sigma^-(K-2) T(theta_hat) + c_K. Relative residual at the smallest sigma: <= 8.5e-6 (binary4 separated), <= 2.2e-3 (binary4 close), <= 6e-5 (mps4), <= 4e-8 (binary8, ht4, mps8 at sigma = 0.01). Exponents with c_K removed: 0.9996 (binary4 separated), 0.990-0.996 (binary4 close), 2.0000 (mps4), 4.998-4.999 (K = 7), 5.9995 (mps8). T is confirmed by an independent method to <= 2e-10 (binary4, mps4). | derived (prediction) + empirical | replication of theory_escape E1; the exponent is known (Jacot 2021; Kumar & Haupt 2025) | `first_summary.csv`, `first_fits.csv`, `first_runs.csv` |
| 3 | **Which component goes first:** GF's first component equals the correlation flow's r1 in 1847/1848 runs over all sigmas, and in 100% at the smallest sigma. On tensor-like trees the first component is the same at every sigma for each seed (10/10 seeds in every cell). | empirical | replication (theory_escape E1, E8) | `first_runs.csv`, `main_order.csv` |
| 4 | **Kumar-Haupt time is a lower bound on T.** Median T / t_KH: 3.1-4.8 (binary4), 7.9-13.7 (mps4), 95-210 (ht4), 184-2650 (binary8), 1460 (mps8). | proven (bound) + empirical (ratios) | known (Kumar & Haupt 2025; theory_escape E2) | `first_summary.csv` |
| 5 | **No d-dependence at full bond (chi = d^2).** binary4 median t_hat_1 = 1.544, 1.569, 1.563 for d = 2, 3, 4 (about pi/2), while t_KH falls 1.9x. sd ln t_hat_1 = 0.218 +- 0.010 (d = 2; predicted 0.250 +- 0.02), 0.157, 0.135. mps4: 1.39, 1.41, 1.50. | empirical | replication of theory_escape E5 | `lo_d.csv` |
| 6 | **chi dependence of the median prefactor.** Slope -0.335 +- 0.028 (binary4) and -0.264 +- 0.037 (mps4) over chi = 4-16; -0.76, -0.69, -0.72 (binary8, ht4, mps8) over chi = 2-16 (2-12 for binary8). t_KH slopes are -0.47 to -2.5. | empirical | replication of theory_escape E7 | `lo_chi.csv`, `lo_chi_slopes.csv` |
| 7 | **RMS-overlap heuristic P3** = d^(N(K-2)/(2K)) / ((K-2) s_1) tracks the median t_hat_1 within 0.87-2.13x over K = 3-15 (17 trees). | derived heuristic + empirical | new (project) | `lo_K_table.csv`, `lo_K.csv` |
| 18 | Gauge-balancing the same init changes T per seed by 0.88-1.92x (median 0.99-1.18 per cell) and keeps r1 in 88/94 seeds. | empirical | new (minor) | `first_runs.csv` |

**S2S sequence, gap exponents and saddle type**

| # | claim | status | known / new | source |
|---|---|---|---|---|
| 8 | **Complete S2S sequences from fan-in init** (odeco, d = 4): binary4 320/320; mps4 160/160 at chi 6 and 106/160 at chi 3; binary8 50/50 and ht4 45/50 at chi = 6 (sigma 0.07-0.3). Plateau loss = L_S at bracketed plateaus: <= 8.5e-3 (binary4 separated), 2.8e-3 (mps4), <= 2.1e-5 (K = 7, chi 6). The top singular values on every bond equal the learned weights. One rank step per event on every edge in 497/531 runs with event spectra. mps8: 0/284 runs learn the third component (132 learn two). At chi = R, K = 7 rarely completes: 14/180 (binary8), 12/150 (ht4). | empirical | binary4 known (depth-3 DLN); mps4, binary8, ht4 new | `main_runs.csv`, `main_classes.csv`, `main_plateaus.csv` |
| 9 | **Gap exponents per seed**, on runs that follow a resolved S2S sequence (rho > 3, modal order); predicted K - 2. binary4: cell means 1.003-1.016 (per seed 0.95-1.09). mps4: 1.990-2.004 (per seed 1.97-2.01). binary8 at chi 6: 5.020 +- 0.009 / 4.969 +- 0.022 (per seed 4.88-5.09, 0.6 decade). binary8 at chi 3, gap 1: 4.93 +- 0.05 (up to 1.1 decades). ht4 at chi 6: 5.10 +- 0.08 / 4.75 +- 0.14 (per seed 3.93-5.78; not a clean confirmation). mps8 gap 1: 5.95 +- 0.05 (chi 6, 7 seeds, 0.6 decade). | empirical (prediction derived) | binary4 known (DLN); others new | `main_exponents.csv`, `main_gap_fits.csv` |
| 10 | **Higher-order saddles.** Per-seed slope of log(-lambda_min) vs log sigma: 1.02-1.18 (binary4), 1.99-2.02 (mps4), 5.07-5.10 (binary8, chi 6), 4.88-5.13 (ht4, chi 6), 5.99 (mps8, chi 6, plateau 1); predicted K - 2. The magnitude agrees with R10 only as a cell median (E11). | empirical (prediction derived, theory_saddles R10) | binary4 known (DLN); others new | `main_exponents.csv`, `hess_main.csv`, `hess_other.csv` |
| 11 | **Strict wrong-order saddle on binary4** (close teacher). The exact lambda_min at W_{1,3} is -0.133186 = s_q^((K-2)/K)(s_q - s_r). At 9 plateau midpoints it measures -0.106 .. -0.146 (projected), flat over sigma = 0.005-0.1 (slope 0.03). Its gap does not grow (exponent -0.08). The saddle is not visited at sigma = 0.003. | proven at W_S (theory_saddles 4.1) + empirical | known for DLNs (Achour 2024; Jacot 2021 Lemma 15); replication | `main_strict_saddles.csv`, `main_plateaus.csv` |
| 19 | **Learning order on binary4:** largest component first in all 380 main/balanced runs (20 init directions) and in 639/640 first-escape runs. Full order greedy in 160/160 runs (separated teacher) and 89-95% (close teacher). | empirical | known for DLNs (Jacot 2021 Lemma 15) | `main_order.csv`, `first_runs.csv` |

**Dormant components, traps and stalled runs**

| # | claim | status | known / new | source |
|---|---|---|---|---|
| 12 | **Conservation-law trap at K = 4** (mps4, chi 3, separated teacher). The same 4/10 seeds trap at all 8 sigmas = 0.003-0.1 (32/80). The end-slice decay rate / sigma^2 is constant per seed (<= 2.4%; seed 7 <= 5%). Levels are seed-dependent (0.26-1.04) and equal at both route ends. Close teacher: 14/80 trapped + 8/80 stalled; seed 7 traps at 6/8 sigmas. | empirical (mechanism derived, theory_saddles R12) | new | `main_classes.csv`, `main_runs.csv`, `dormant_decay.csv` (per-core levels carry a +-4% window bias) |
| 13 | **No trap at K = 3** (0/380 runs). | empirical | new | `main_classes.csv` |
| 14 | **Gauge-balanced init.** 0/120 runs trap in the cells where fan-in traps (fan-in: 12/30 and 47/90). But 69/90 balanced K = 7, 8 runs stall. 17 of the 47 fan-in-trapped K = 7, 8 runs complete when balanced; 5 fan-in-complete runs stall. | empirical | new (supports theory_saddles R12) | `main_runs.csv`, `main_classes.csv` |
| 15 | **Trap fractions at K = 7, 8 are lower bounds.** chi = 3 (k78, sigma 0.15-0.3, fan-in): at least 9/30 (binary8), 24/30 (ht4), 14/30 (mps8). chi = 6: 0/50 (binary8), 5/50 (ht4, seed 9 at every sigma), 7/50 (mps8). Classes are stable at 100x tighter atol (9/9). | empirical | new | `main_classes.csv`, `tolcheck_k78.csv`, `slice_trends.csv` |
| 16 | **Escapes shrink dormant components only mildly.** Median factor: x2.1-3.0 (growth) at K = 3; x0.64-0.74 at K = 4; x0.12-0.75 at K = 7, 8. Drops by more than 100x: <= 1.3% (K = 4), 0-2% (K = 7, 8, chi 6), 2-8% (K = 7, 8, chi 3; 11% for mps8 separated later escapes). | empirical | new | `dormant_escape_drops.csv` |
| 17 | **Stalled runs.** Of 23 continued stalled/unresolved runs, 16 escape at 0.24-2.19x the balanced-route estimate: all 8 shortest-estimate k78 runs (0.50-0.87, zoom-resolved), 6 of 7 new runs of a seeded random sample, and 2 of 8 mps8 chi 6 runs. 2 do not escape by 3x, 4 hit the wall limit, 1 failed. The ever-longer-plateau reading of the untested majority is a **conjecture**. | empirical (prediction derived) | new | `continued_k78_resolved.csv`, `continued_k78_random.csv`, `continued_large_spread.csv`, `slice_trends.csv` |

**Not safe to cite as written:**
- "1.1-5.6x below the single-end estimate" (E5).
- "first component 100% / largest first at every sigma" (E1).
- Per-seed R10 magnitudes (E11).
- The balanced-route estimate as an upper bound (E12).
- K = 7, 8 decay rates and their sigma dependence (E19, E20).
- Any K = 7, 8 gap exponent as meeting the project's 1-decade rule, except binary8 chi 3 gap 1 (E21).
- "T moved by <= 1.7e-3" (E2).
- Plateau-level statements in the form given in E10.
