# s2s-core: saddle-to-saddle phenomenology and escape-time scaling in TTNs (Q1, Q2)

Errata from notes/reviews/s2s_core_round2_verification.md applied (2026-10-10).

Status: done (fix round 1 complete) - Branch: agent/s2s-core - Updated: 2026-10-10 10:20 UTC

## Summary
Asked: does gradient flow (GF) on a TTN from small init show saddle-to-saddle (S2S) dynamics, which
saddles it visits, and how plateau durations scale with sigma, the tree (K cores) and the bond
dimension chi, against theory. The lead added a question: are dormant components suppressed during
earlier escapes, and are the runs stuck or on ever-longer plateaus? A fix round after review round 1
(`notes/reviews/s2s_core_round1.md`) re-analysed all raw data; the claims that changed are listed
below and in `notes/s2s_core.md` section 4.

Answer (odeco teachers, d = 4):
- **First escape.** Predicted per seed, with no free parameter, by the correlation-flow blow-up time
  (a replication of theory_escape E1), to <= 2.2e-3. The Kumar-Haupt law is a lower bound, 3-2700x too
  small.
- **Where S2S was measured.** Complete S2S sequences were measured on binary4 and mps4, and on binary8
  and ht4 at chi = 6 (0.6 decade of sigma). On mps8 no run learns the third component. Plateaus sit at
  the bond-rank-constrained partial fits, and every edge gains one rank at each event. binary4 is a
  depth-3 DLN: its S2S sequence, sigma^-1 plateaus and strict wrong-order saddle are known DLN results
  ([Achour2024]; [Jacot2021] Lemma 15), so the binary4 measurements are a replication (known for DLNs;
  prediction = theory_saddles 4.1). New for TTNs: the same measurements on mps4, binary8, ht4, mps8.
- **Gaps.** Fitted on runs classified by order and S2S status, the gaps scale as sigma^-(K-2) per seed:
  - K = 3, 4: within 0.1 (binary4: known for DLNs);
  - binary8: per-seed exponents 4.9-5.1 at chi = 6 (0.6 decade); gap 1 at chi = 3 4.93 +- 0.05 over up
    to 1.1 decades, the only K = 7 fit spanning about 1 decade;
  - ht4: 3.9-5.8;
  - mps8: first gap only.
- **Saddle type.** Plateaus are higher-order saddles except on binary4 wrong-order plateaus. There a
  strict saddle appears, as predicted (known for DLNs): lambda_min = -0.106 .. -0.146 at the 9
  wrong-order plateaus vs the exact -0.133186 (median ratio 0.98 for seed 8, 0.93 for seed 5), flat
  over sigma = 0.005-0.1; the wrong order is not visited at the smallest sigma.
- **Main limit at large K: the conservation-law trap of fan-in init.**
  - Trapped fraction: 0 at K = 3; at K = 4, chi = 3, 40% with the separated teacher (fixed per init
    direction) and 18% with the close teacher (not fixed per direction: seed 7 traps at 6 of 8 sigmas,
    seed 5 stalls at all 8); at least 30-80% at K = 7, 8, chi = 3; 0-14% at chi = 6.
  - Gauge-balanced init: 0/120 trapped where fan-in traps, but most of its K = 7, 8 runs stall.
  - Stalls are predicted to be ever-longer plateaus. 16 of 23 continued stalled runs escape, but the
    balanced-route estimate is exceeded in 4 of those 16 (so it is an estimate, not a bound), and 91 of
    the 106 k78 stalled runs were never tested: for most stalled runs this is a conjecture.

## Fix round (review round 1): claims that changed
| claim before | now | evidence |
|---|---|---|
| no strict saddles with odeco teachers; plateaus higher-order | strict saddle on the binary4 wrong-order plateau (close teacher; seed 8 at 7/8 sigmas, seed 5 at 2): lambda_min -0.106 .. -0.146 over the 9 plateaus (seed 8: -0.118 .. -0.146), flat over sigma = 0.005-0.1 (slope 0.03), median 0.98 (seed 8) / 0.93 (seed 5) of the exact -0.133186; not visited at sigma = 0.003; known for DLNs (replication); higher-order only on greedy binary4 and tensor-like trees | `main_strict_saddles.csv` |
| every gap ~ sigma^-(K-2) (pooled fits) | per-seed fits on runs classified by order and S2S status (rho > 3; for binary4 close the cut leaves some seeds with 0.8 decade); binary4 close gap 2 = 1.016 +- 0.011 (was 0.92 +- 0.11 pooled); ht4 per-seed 3.9-5.8; mps8 gap 1 = 5.95 +- 0.05 (new); binary8 chi = 3 gap 1 = 4.93 +- 0.05 over up to 1.1 decades (added after the round-2 verification) | `main_gap_fits.csv`, `main_exponents.csv` |
| S2S on every tree | complete S2S on binary4, mps4 (chi 3, 6), binary8 / ht4 at chi = 6 only; mps8 0/284 learn a 3rd component; K = 7 at chi = R completes rarely (binary8 14/180, ht4 12/150) | `main_runs.csv` |
| trap decay ~ sigma^2 within 1.8x, two seeds switch levels by 20-100x | rate / sigma^2 constant per seed to <= 2.4% (seeds 3, 5, 6) and <= 5.3% (seed 7) over sigma 0.003-0.1 (4 separated seeds; per-core levels carry a +-4% fit-window bias), 1.25-1.33x (close) once fitted above the float64 floor (1e-9); the switch was an artifact; K = 7, 8 rates not sigma^2-clean | `dormant_decay.csv` |
| measured rates 3-15x below the single-end estimate | withdrawn as unsupported (was "1.1-5.6x below", ratio 0.18-0.87): no script, CSV or log computes Q_nu nu or the estimate, it rests on 2 re-run seeds, and the choice of s_q alone spans 1.5x; only the sign rule and the sigma^2 scaling are cited | notes 2.7c |
| first component = correlation flow's except binary4 close (80-90%, wrong order at finite sigma) | 1847/1848 runs over all sigmas (100% at the smallest sigma): the 80-90% came from stopping the flow at amplitude 1e6. The one exception (binary4 chi 3, close, balanced, seed 6 at sigma = 0.1) is an E9-type wrong first escape: theory_escape E9 is withdrawn only as the cause of the old mismatches. Now stopped on direction convergence or at the amplitude cap 1e30 (all 40 binary4 close jobs end at the cap, direction converging slowly; T converged regardless, independent check to <= 1.5e-11); T moved by <= 2.2e-3 (binary4 close; fan-in max 1.75e-3), <= 3.0e-5 (binary4 separated), <= 2e-7 elsewhere | `first_runs.csv`, `recompute_T.py` |
| stalls are ever-longer plateaus (10 of 11 escape) | derived prediction; 8/8 shortest-estimate k78 runs (3 seeds) escape at 0.50-0.87 of the estimate; mps8 chi = 6 spread: 2 escape at 1.37x and 1.81x, 1 not at 2x (nor by 3x in the verification rerun), 5 unresolved; random sample of 8 of the 74 k78 candidates (73 stalled + 1 trapped) with estimate <= 100x elapsed: 7 escape at 0.24-2.19x, 1 not by 3x. In all 16 of 23 distinct continued runs escape; the estimate is exceeded in 4 of the 16, so it is not a bound; 91/106 k78 and 35/43 large mps8 stalled/unresolved runs untested (conjecture there) | `continued_*.csv` |
| trap fraction 40% at K = 4; 30-80% at K = 7, 8 | 40% separated teacher only (close: 14/80 trapped + 8/80 stalled, not fixed per init direction); K = 7, 8 fractions are lower bounds (stalled runs partly still decaying) | `main_classes.csv`, `slice_trends.csv` |
| balanced init removes the trap (0/210); chi = 2R removes it | 0/120 in the informative cells (fan-in 12/30, 47/90); balanced K = 7, 8 still stall 69/90; chi = 6: ht4 5/50 trapped (seed 9 at every sigma), mps8 7/50; "2R" literal only on binary trees (MPS end / HT leaf bonds capped at d = 4) | `main_classes.csv` |
| edge rank steps 517/551; >100x drops 0% at chi = 2R; later escapes "same picture"; R10 magnitude 0.8-1.7; plateau error <= 2e-2, mps4 2e-3; KH 3-500x | 497/531 (332 untested); 0-2%; 2.0-8.3% >100x at K = 7, 8, chi = 3 (11% for mps8 separated later escapes); cell medians 0.87-1.15 (K = 3, 4) and 1.4-3.2 (K = 7, 8; up to 3.9 in k78), per-seed medians up to 20; binary4 8.5e-3 (separated, sigma = 0.1) and 2.0e-2 (close, sigma = 0.003) (up to 0.50 at unresolved close plateaus), mps4 2.8e-3 (8.1e-3 balanced), K = 7 <= 2.1e-5 at chi = 6 (1.5e-4 at chi = 3); 3-2700x | notes 2.3-2.7 |
| (tags) | first escape, KH bound, d-flatness, chi slopes and race order are replications of theory_escape E1/E2/E5/E7/E8; binary4 "always largest first" rests on 20 init directions; sd ln T (d = 2) 0.218 vs 0.250 +- 0.02; binary4's S2S sequence, sigma^-1 plateaus and strict wrong-order saddle are known DLN results ([Achour2024]; [Jacot2021] Lemma 15), i.e. replications (round-2 verification) | notes 2.2b, 2.2c, 2.3 |

## Results
Gradient flow throughout: LSODA with exact Hessian Jacobian, rtol 1e-8 / atol 1e-11 sigma (main,
first, balanced), rtol 1e-6 / atol 1e-9 sigma (large, k78; event times agree with rtol 1e-8 to 7e-7,
and 9 K = 7, 8 trapped / stalled runs re-run at rtol 1e-8 keep class and order, events to 3.6e-6);
DP5 rtol 1e-10 for the aligned runs. Whitened inputs, d = 4, odeco teachers (seed 1000) separated
[3, 2, 1] and close [1, 0.85, 0.7], fan-in init unless stated, 10 seeds per cell. Event times and
widths are resolved by re-integrating each event window on linear grids (>= 40 samples per rise).
All tables: `experiments/s2s_core/results/`; derivations, status tags and full numbers:
`notes/s2s_core.md`.

**1. Known result** (`aligned.csv`, `fig_a_aligned.png`): aligned balanced init follows the exact
Saxe modes; max |a_GF - a_theory| = 2.0e-10 (binary4), 3.2e-10 (mps4), 2.3e-9 (binary8), 2.6e-9 (ht4),
2.7e-9 (mps8). Derived; known for DLNs (Saxe 2014 eq. 15), explicit for TTNs here.

**2. Phenomenology** (`main_runs.csv`, `main_classes.csv`, `fig_b_run_*.png`; binary4 = depth-3 DLN,
so its S2S sequence and plateaus are a replication of known DLN results; mps4, binary8, ht4, mps8 new):
- complete S2S sequences: binary4 320/320; mps4 160/160 at chi = 6, 48/80 + 58/80 at chi = 3 (rest
  trapped / stalled); binary8 50/50 and ht4 45/50 at chi = 6 (sigma 0.07-0.3); at chi = 3, K = 7 rarely
  (binary8 14/180, ht4 12/150); mps8 never (0/284 learn a third component, 132 learn two);
- plateau loss = partial-fit level L_S at resolved plateaus to 8.5e-3 relative (binary4 separated,
  sigma = 0.1) and 2.0e-2 (binary4 close, at sigma = 0.003: chi 6, seed 4, rho_2 = 5.1), 2.8e-3 (mps4;
  8.1e-3 balanced), <= 2.1e-5 (K = 7 at chi = 6; up to 1.5e-4 at chi = 3, ht4 close, sigma = 0.4),
  -> 6e-7 at sigma = 0.003 (separated teacher and mps4, not binary4 close); up to 0.50 at unresolved
  binary4 close-teacher plateaus. Every edge's top |S| singular values = the learned weights (truncated
  teacher spectrum), next value <= 0.07 s_min: each plateau is the bond-rank-constrained critical point W_S;
- each event raises the rank by one on every edge at the same time in 497/531 runs with event-window
  spectra (332 early complete runs untested; exceptions are overlapping events);
- S2S (every rho > 3) among complete runs: mps4 266/266, binary8 83/83, ht4 74/74; binary4 separated
  158/160, close teacher 5/20 at sigma = 0.1 rising to 20/20 at sigma = 0.003;
- order: binary4 largest first in all 380 main/balanced runs (20 init directions) and in 639/640
  first-escape runs (known for DLNs: [Jacot2021] Lemma 15); on mps, binary8, ht4 the first
  component is random (P(largest) 0.4-0.8) but the same at every sigma for each seed (replication of
  theory_escape E8).

**3. Escape-time exponents** (per-seed fits on runs classified by order and S2S status, mean +- SE
[range of per-seed values]; `main_exponents.csv`, `main_gap_fits.csv`, `first_summary.csv`, figures c, d;
predicted K - 2):

| tree | K | chi | teacher | sigma range | t_1 (first sweep, offset removed) | t_2 - t_1 | t_3 - t_2 | seeds (gap 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3 | sep. | 0.1-0.003 | 0.9996 +- 0.0001 | 1.003 +- 0.001 [1.00, 1.01] | 1.004 +- 0.003 [0.99, 1.02] | 10 / 10 |
| binary4 | 3 | 6 | sep. | 0.1-0.003 | 0.9996 +- 0.0001 | 1.007 +- 0.001 [1.00, 1.01] | 1.011 +- 0.002 [1.00, 1.02] | 10 / 10 |
| binary4 | 3 | 3 | close | 0.1-0.003 | 1.000 +- 0.003 raw (0.993 offset removed) | 1.004 +- 0.010 [0.95, 1.06] | 1.016 +- 0.011 [1.00, 1.09] | 9 / 8 |
| mps4 | 4 | 3 | sep. | 0.1-0.003 | 2.0000 +- 0.0000 | 2.000 +- 0.000 | 1.990 +- 0.005 [1.97, 2.00] | 10 / 6 |
| mps4 | 4 | 6 | sep. | 0.1-0.003 | 2.0000 +- 0.0000 | 2.001 +- 0.001 | 1.998 +- 0.001 | 10 / 10 |
| binary8 | 7 | 6 | sep. | 0.3-0.07 | 4.9990 +- 0.0002 | 5.020 +- 0.009 [4.98, 5.05] | 4.969 +- 0.022 [4.88, 5.09] | 10 / 10 |
| ht4 | 7 | 6 | sep. | 0.3-0.07 | 4.9987 +- 0.0007 | 5.10 +- 0.08 [4.83, 5.78] | 4.75 +- 0.14 [3.94, 5.33] | 10 / 9 |
| binary8 | 7 | 3 | sep. | 0.4-0.03, sweep main (gap 1 mostly 0.03-0.4; t_1: 0.4-0.01) | 4.9983 +- 0.0003 | 4.93 +- 0.05 [4.65, 5.13] | 1 seed only | 10 / 1 |
| binary8 | 7 | 3 | close | 0.4-0.03, sweep main (gap 1 mostly 0.03-0.4; t_1: 0.4-0.01) | 4.9962 +- 0.0008 (6 seeds) | 4.92 +- 0.04 [4.65, 5.10] | 1 seed only | 10 / 1 |
| mps8 | 8 | 6 | sep. | 0.3-0.07 | 6.000 (raw, sweep large) | 5.95 +- 0.05 [5.79, 6.20] | not reached | 7 / 0 |
| mps8 | 8 | 3 | sep. | gap 1: 0.1 or 0.15 to 0.4, 4-5 points (t_1: 0.4-0.01) | 5.9995 +- 0.0002 | 6.0 +- 0.3 [5.7, 6.6] | not reached | 3 / 0 |
| mps8 | 8 | 3 | close | gap 1: 0.1 or 0.15 to 0.4, 4-5 points (t_1: 0.4-0.03, sweep main) | 5.997 +- 0.001 (raw, sweep main) | 6.19 +- 0.60 [5.14, 7.21] | not reached | 3 / 0 |

(Other close-teacher cells within 0.02 of K - 2. K = 7, 8 gaps span <= 0.6 decade, below the
project's 1-decade rule, except binary8 gap 1 at chi = 3 (up to 1.1 decades; added after the round-2
verification, as was the close-teacher mps8 chi = 3 row); ht4 is consistent with 5 in the mean of gap 1
only. binary4 rows: replication of known DLN results. For binary4 close the rho > 3 cut leaves some
seeds with only 4-5 sigmas (0.003-0.02, about 0.8 decade); without the cut binary4 close gives gap 1 =
1.00 +- 0.04 / 1.04 +- 0.04 and gap 2 = 0.92 +- 0.11 / 1.12 +- 0.05 at chi 3 / 6 (notes 2.4).)

**4. First-escape prefactor** (`first_*.csv`, `lo_*.csv`, figures c, e, e2; replications of theory_escape
E1, E2, E5, E7 unless marked):
- per seed, |t_1 / (sigma^-(K-2) T(theta_hat) + c_K) - 1| at the smallest sigma: <= 8.5e-6 (binary4
  separated), <= 2.2e-3 (binary4 close), <= 6e-5 (mps4), <= 4e-8 (binary8, ht4, mps8 at sigma = 0.01);
  first component = the correlation flow's r1 in 1847/1848 runs over all sigmas, 100% at the smallest
  sigma (the exception, binary4 chi 3, close, balanced, seed 6 at sigma = 0.1, is a finite-sigma wrong
  first escape of the theory_escape E9 type). No free parameter. Offset-removed exponents: 0.9996
  (binary4 separated), 0.990-0.996 per cell (binary4 close, offset not converged; 0.993 in the table).
- Kumar-Haupt (rho_0) is a lower bound: median T / t_KH = 3.1-4.8 (binary4), 7.9-13.7 (mps4), 95-210
  (ht4), 184-2650 (binary8), 1460 (mps8); up to 3e7 at K = 15 (leading order).
- d at full bond (200 inits): binary4 median 1.544, 1.569, 1.563 for d = 2, 3, 4 = pi/2, flat, while t_KH
  falls 1.9x; sd ln T 0.218 / 0.157 / 0.135 vs predicted 0.250 / 0.167 / 0.125 (+- 0.02).
- chi (100 inits): slope of the median over chi = 4-16: mps4 -0.26 +- 0.04, binary4 -0.34 +- 0.03 (t_KH
  -0.47); K = 7, 8 trees -0.69 .. -0.76 over chi = 2-16 vs -1.5 .. -2.5 for t_KH.
- K (50 inits, K = 3-15; new, heuristic): the RMS-overlap heuristic P3 = d^{N(K-2)/(2K)}/((K-2) s_1) (no
  fit) matches the median within 0.87-2.1x while the prefactor spans 0.4-814.
- Gauge-balancing the same init changes T per seed by 0.88-1.92x (median 0.99-1.18; usually slower).

**5. Saddle type** (`hess_*.csv`, `main_strict_saddles.csv`, figure f; gauge-projected and raw):
- higher-order where predicted: per-seed slope of log(-lambda_min) vs log sigma 1.06 / 1.02 (binary4
  separated), 1.17 / 1.12 (close) (binary4: known for DLNs), 2.00 / 1.99 (mps4), 5.10 / 5.07
  (binary8), 5.13 / 4.88 (ht4), 5.99 (mps8, first plateau); magnitude vs the route formula
  -(K-1) s_r (s_r |a_r|)^((K-2)/K), as cell medians only: 0.87-1.15 at K = 3, 4 (main fan-in cells;
  balanced binary4 chi 3: 1.19), 1.4-3.2 at K = 7, 8 (up to 3.9 in k78; routes not balanced at these
  sigma). Per-seed medians reach 20 (binary4 separated chi 3 seed 8, second plateau; balanced mps4
  chi 3 up to 7.9), so per-seed R10 magnitudes are not cited; the slopes are unaffected;
- strict where predicted (a known DLN result: [Achour2024]; [Jacot2021] Lemma 15; replication): the
  binary4 wrong-order plateau (learned {1, 0.7}, 0.85 unlearned) has lambda_min -0.118 .. -0.146 for
  seed 8 over sigma 0.005-0.1 and -0.106 .. -0.146 over all 9 wrong-order plateaus (seed 5 adds
  sigma = 0.07, 0.1; raw -0.113 .. -0.149), flat (slope 0.03), median 0.98 (seed 8) and 0.93 (seed 5)
  of the piggyback value s_q^((K-2)/K)(s_q - s_r) = -0.133 (exact Hessian at W_{1,3}: -0.133186); its
  gap does not grow (exponent -0.08). The wrong order is not visited at the smallest sigma (seed 8 is
  greedy at sigma = 0.003; chi = 6 seed 4 is wrong-order only at sigma >= 0.03), so "flat" covers
  sigma = 0.005-0.1.

**6. Dormant-component suppression** (`dormant_*.csv`, `main_classes.csv`, `slice_trends.csv`,
`continued_*.csv`, `tolcheck_k78.csv`, figure g):
- Escapes: median factor x2-3 (K = 3, growth), x0.64-0.74 (K = 4), x0.12-0.75 (K = 7, 8); > 100x drops
  in <= 1.3% (K = 4), 0-2% (K = 7, 8, chi = 6), 2.0-8.3% (K = 7, 8, chi = 3, later escapes; ht4 2.0%;
  11% for mps8 later escapes, separated teacher).
- Plateaus kill them through the conservation-law trap (theory_saddles R12): route end slices decay
  exponentially at a rate ~ sigma^2 per seed (mps4: rate / sigma^2 constant to <= 2.4% (seeds 3, 5, 6)
  and <= 5.3% (seed 7; <= 3% by local-rate median) over sigma 0.003-0.1 for the 4 trapped separated
  seeds; equal at both route ends, 0.26-1.04 per seed; the fitted per-core levels carry a +-4% bias
  from a fit window that starts inside the post-event transient); both ends on the light side of the
  conserved Q (2 runs; no committed script computes Q). The magnitude relative to the single-end
  estimate (quoted before as "1.1-5.6x below") is withdrawn as unsupported: no script, CSV or log
  computes it, it rests on 2 seeds, and the choice of s_q alone spans 1.5x; only the sign rule and the
  sigma^2 scaling are cited. Run classes (separated teacher; trapped = dead route end, slice < 1e-8
  sigma; at K = 7, 8 trapped is a lower bound):

| tree | K | chi | init | sweep, sigma | complete | trapped | stalled | unresolved |
|---|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3, 6 | fan-in, balanced | main, balanced | 220/220 | 0 | 0 | 0 |
| mps4 | 4 | 3 | fan-in | main, 0.003-0.1 | 48/80 | 32 | 0 | 0 |
| mps4 (close) | 4 | 3 | fan-in | main, 0.003-0.1 | 58/80 | 14 | 8 | 0 |
| mps4 | 4 | 3 | balanced | balanced, 0.003-0.03 | 30/30 | 0 | 0 | 0 |
| binary8 | 7 | 3 | fan-in / balanced | k78, 0.15-0.3 | 2 / 3 of 30 | 9 / 0 | 18 / 27 | 1 / 0 |
| ht4 | 7 | 3 | fan-in / balanced | k78, 0.15-0.3 | 3 / 17 of 30 | 24 / 0 | 3 / 12 | 0 / 1 |
| mps8 | 8 | 3 | fan-in / balanced | k78, 0.15-0.3 | 0 / 0 of 30 | 14 / 0 | 16 / 30 | 0 / 0 |
| binary8 | 7 | 6 | fan-in | large, 0.07-0.3 | 50/50 | 0 | 0 | 0 |
| ht4 | 7 | 6 | fan-in | large, 0.07-0.3 | 45/50 | 5 (seed 9) | 0 | 0 |
| mps8 | 8 | 6 | fan-in | large, 0.07-0.3 | 0/50 | 7 | 34 | 9 |

- At K = 4 with the separated teacher the trap is fixed by the init direction: the same 4/10 seeds trap
  at all 8 sigmas (trap time ~ sigma^-2, escape ~ sigma^-2). With the close teacher it is not (seed 1
  traps at 8/8 sigmas, seed 7 at 6/8, seed 5 stalls at all 8). Gauge-balanced init: 0/120 trapped in
  the cells where fan-in traps (12/30, 47/90), but 69/90 balanced K = 7, 8 runs stall.
- Stalled runs with live routes (derived prediction, for routes that obey the balanced reduction:
  remaining plateau ~ the balanced-route estimate b^-(K-2)/((K-2) s_r), b = (s_r |a_r|)^(1/K)):
  continued from their final state, the 8 k78 runs with the shortest estimate (3 seeds) all escape, at
  0.50-0.87 of it (zoom-resolved; per seed constant across sigma); of 8 mps8 chi = 6 runs spread over
  the distribution, 2 escape at 1.37x and 1.81x (the reduction does not hold there; log-grid values,
  round-1 script), 1 not by 2x (nor by 3x in the round-2 verification's rerun), 4 wall-limited
  (whether before the estimate cannot be checked: no reached-time column), 1 integrator failure; a
  seeded random 8 of the 74 k78 candidates (73 stalled + 1 trapped that ran to t_max, not drawn) with
  estimate <= 100x the elapsed time: 7 escape at 0.24-2.19x (median 0.55), 1 not by 3x. In total 16 of
  23 distinct continued stalled runs escape (14/15 k78, 2/8 large), and the estimate is exceeded in 4
  of the 16 (1.08, 1.37, 1.81, 2.19x): it is an estimate, not a bound. Untested: 91 of the 106 k78
  stalled runs (estimate / elapsed: cell medians 17-64x) and 35 of the 43 large mps8 stalled or
  unresolved runs (median about 390x); for them "ever-longer plateaus" is a conjecture, supported only by
  the slice trends: pinned in binary8 and balanced mps8 (0.01-0.04 sigma); still decaying in 50% of the
  balanced ht4 runs and 19-32% of the fan-in mps8 runs, many of the latter near the trap threshold.

## What was done
1. Read brief, notes, library; theory derivation of the first-escape prediction (correlation-flow
   blow-up time; later equal to theory_escape E1), bounds, decoupling, saddle types; scipy pilots
   (stiffness of later plateaus, K sweep) (0c50e36, c53866f, 43b1620).
2. Experiment code on the library (`experiments/s2s_core/`: s2s.py, run.py, aligned.py,
   first_escape.py, lo_sweep.py, analyze_*.py, plot_*.py, recompute_hessians.py,
   continue_stalled.py) (137cc70, 55c4afb, be74002, ab25042).
3. Step 1, aligned init vs exact solution (137cc70).
4. Sweep `main` (1126 runs, K = 3, 4, 7, 8; fan-in), interim reports 1-2 (be74002, 9789170).
5. First-escape sweep (`first`, `first_k78`), leading-order sweeps (`lo_d`, `lo_chi`, `lo_K`)
   (94c4163, 788e235, 117a741, 5df02a7).
6. Dormant suppression: trap decay rate and sign rule (601a1c9, 7b7e7dc), dead-end classification
   (5ade509, 4b5392b), gauge-balanced inits (`balanced`, `k78`), long K = 7, 8 runs (`large`),
   continuation of stalled runs (ce7b78c, e878dc1).
7. Hessians recomputed after the library's projection fix (25fb840); final tables and figures
   (61afd74, 8fa2130); notes `notes/s2s_core.md` (sections 1-3), README.
8. Fix round after review round 1 (merged main 55519f3; WIP 7761da5, final: this commit): correlation
   flow stopped on direction convergence or at the amplitude cap 1e30 and T recomputed for all 231
   first-escape jobs (`recompute_T.py`,
   old values kept as `T_v1`); LO sweeps rerun; aligned runs rerun with one file per integrator;
   `analyze_main.py` refactored (per-plateau table, run-set classification `fit_sets`, strict-saddle
   table, class table); trap decay refitted above 1e-9; escape-drop and slice-trend tables; continuation
   escapes zoom-resolved, random continuation sample, tolerance check (`tolcheck.py`); figures c-g
   regenerated; notes 2.2-2.7, 3, 4 (claims changed), README, this report.

## Decisions
- **d = 4** at every site (odeco needs d >= 3; d = 4 leaves one physical direction outside the
  teacher). **Teacher seed 1000**, init seeds 0-9 (independent streams after the library fix).
- **Integrator: LSODA with the exact Hessian** for random-init runs (later plateaus are stiff: DP5 would
  need ~sigma^-(K-2) steps); DP5 as reference for aligned runs. Radau was tried on a K = 7 run and did
  not finish in 25 min (rejected). Tolerances: rtol 1e-8, atol 1e-11 sigma (main, first, balanced);
  rtol 1e-6, atol 1e-9 sigma for the long K = 7, 8 runs (large, k78) after checking on binary8 chi = 6
  sigma = 0.07 that event times agree with rtol 1e-8 to 7e-7 relative (975 s vs 1499 s).
- **Events = final rise of a_r** (last upward crossing of 1/2), widths 0.1 -> 0.9 of the final rise,
  both refined by re-integrating the window on linear grids until the rise holds >= 40 samples. No
  duration is read off the log grid.
- **Run budgets**: t_max = 1e3 x the RMS-overlap escape time of the weakest component; wall limits
  300-2400 s; a stop when a never-learned component has a dead route end (slice < 1e-8 sigma for a
  whole output chunk: conservation-law trap, nothing after it is physics). Stalled / unresolved runs
  are reported as classes, never dropped.
- **Corrected mid-course**: (1) an amplitude-based roundoff criterion (|<W,U_r>| < 1e-15 -> stop;
  < 1e-14 -> "floor") turned out too strict at K >= 7 (the amplitude is a product of K per-core
  amplitudes): replaced by the per-core dead-end criterion; K = 7, 8 runs affected are re-run (k78).
  (2) The gauge projection of the Hessian was fixed in the library: all lambda_min values recomputed
  from the saved midpoint parameters (raw and projected reported).
- **Cuts for compute** (each documented in the configs): K = 7, 8 full runs at sigma >= 0.07 only (at
  0.05, 0.03 none of ~50 runs completed); chi = 6 at K = 7, 8 only with the separated teacher (large);
  first-escape sweep at K = 7, 8 only with the separated teacher and fan-in init (it also contains 21
  binary8 chi = 3 jobs with the close teacher or balanced init, which are in `first_summary.csv`); k78
  at sigma = 0.3-0.15. Finished runs of the larger grids are kept and reported.
- **sigma grids**: K = 3, 4: 0.1 .. 0.003 (1.5 decades) for full runs, 0.1 .. 0.002 for the first
  escape; K = 7, 8: first escape 0.4 .. 0.01 (1.6 decades), full runs 0.4 .. 0.07 (later gaps at
  smaller sigma are not reachable in float64 within budget; reported as such).
- **Own sweep driver** (`run.py`) and analysis code in the experiment folder (the library runner landed
  after the sweeps had started); the library's integrator, Recorder, diagnostics and theory are used
  throughout.

- **Trapped-class threshold 1e-8 sigma** (7 decades below the init scale, above both atol settings).
  The minimum dormant slice has a large mass below 1e-12 sigma (dead ends) and a continuum 1e-10 ..
  1e-4 sigma (ends still decaying when a run stopped), so the trapped fraction moves by a few runs per
  cell between 1e-10 and 1e-8 (sweep `main`, chi = 3: binary8 / ht4 / mps8 trapped 24 / 77 / 24% at
  1e-10 sigma vs 27 / 82 / 30% at 1e-8 sigma).
- **Pooling**: exponents are fitted per seed on the classified run sets of the fix round (below); K = 7,
  8 cells are taken from one sweep each (large at chi = 6, main at chi = 3, where k78 has only 3
  sigmas), never pooled across tolerance settings.
- **Continuation test** of stalled runs on the 8 with the shortest predicted remaining plateau (to fit
  the time left); a spread sample from `large` was started as a check of the bias (drawn at ~22:43 UTC
  while `large` was still running, from 19 candidates, so `--spread` on the finished sweep picks another
  set; the README regenerates it with `--names`; its continuation factor was not recorded).

- **Fix round: run sets for the fits.** Options: (i) pool complete runs (before; mixes wrong-order and
  overlapping-event runs into the fits); (ii) require every rho > 3 in a run; (iii) classify per step.
  Chose (iii): t_1 on runs with the seed's modal first component, gap 1 on runs whose first two
  components are the modal pair with rho_1 > 3 (so mps8 runs that learn two components count), gap 2
  on complete runs in the modal order with rho_1, rho_2 > 3. The same sets drive the Hessian slopes and
  figure d (`analyze_main.fit_sets`).
- **Fix round: correlation-flow stop.** Options: a larger fixed amplitude stop (the reviewer used 1e10,
  1e14) or a direction criterion. Chose the direction criterion: integrate in segments (target x1e3)
  until the second-largest component amplitude is <= 1e-6 of the largest and still falling, cap 1e30;
  T is extrapolated from the stop as before. The old T, r1, T_single are kept in each JSON (reversible).
  In practice the direction criterion was never met on the 40 binary4 close-teacher jobs: all of them
  stopped at amplitude 1e30 (direction converging slowly; second/first ratio still 2e-6 .. 7e-3), as
  did 2 separated binary4 chi = 6 jobs. T is converged regardless (the time left beyond amplitude A is
  ~A^-(K-2)/K ~ 1e-10; an independent sphere-flow T agrees to <= 1.5e-11, round-2 verification).
- **Fix round: trap-fit floor 1e-9 absolute** (dead slices bottom out at 1e-12 .. 1e-15 absolute; three
  decades of margin; a relative-to-sigma floor would cut long fits at small sigma).
- **Fix round: optional runs done, small**: zoom re-resolution of the 8 continuation escapes (values
  moved by <= 1.8%), a 9-run tolerance check (one per tree / init / class), a seeded random continuation
  sample of 8 (`--seed 1`, among the 74 k78 candidates with estimate <= 100x the elapsed time: 73
  stalled + 1 trapped run that ran to t_max, mps8 fan-in sigma = 0.3 seed 5, which was not drawn; the
  33 with longer estimates would not fit the wall limit).
- **Stale files kept**: `/root/outputs/s2s_core/aligned/{tree}.npz` (pre-fix, one file for both
  integrators) are left in place, not deleted; nothing reads them.

## Files left on the machine
All raw outputs are in `/root/outputs/s2s_core/` (not in git; ~220 MB in total), one `.npz` (log-grid
trajectory, refined event windows with spectra, parameters at plateau midpoints, final state) and one
`.json` summary per run: `main/` (1126 runs = 2252 files, 132 MB), `large/` (150, 36 MB), `k78/` (180,
33 MB), `balanced/` (120, 12 MB), `first/` (231 first-escape jobs, 1 MB; JSONs carry the recomputed T and the
old `T_v1`), `aligned/` (2 MB; `{tree}_{dopri5,lsoda}.npz` current, `{tree}.npz` stale pre-fix copies),
`k78_tight/` (9 tolerance-check reruns, 2 MB). Needed to recompute
the analysis tables (`analyze_*.py`, `recompute_hessians.py`, `continue_stalled.py`); everything is
regenerated by the commands in `experiments/s2s_core/README.md` (~6 h on 60 CPUs).
`/root/outputs/s2s_core/first_test/` is an empty test directory.

## Caveats and next steps
- **K = 7, 8 later gaps** span only sigma = 0.07-0.3 (0.6 decade, below the project's 1-decade rule)
  and come from chi = 6 (binary8, ht4); ht4 per-seed exponents scatter (3.9-5.8); at chi = 3 too few runs
  complete; mps8 never learns its third component, so only its first gap is measured (7 seeds at chi = 6;
  3 seeds per teacher at chi = 3, <= 0.6 decade). The exception is binary8 gap 1 at chi = 3 (sweep main,
  mostly sigma = 0.03-0.4, up to 1.1 decades: 4.93 +- 0.05 / 4.92 +- 0.04), the only K = 7 gap fit
  spanning about 1 decade. Smaller sigma is out of reach: the gaps grow like sigma^-5..-6 and LSODA
  advances slowly on these plateaus.
- **First-escape sweep at K = 7, 8** was cut for compute (separated teacher, fan-in only, plus 21
  binary8 chi = 3 jobs with the close teacher or balanced init); mps8 at chi = 6 is missing (jobs
  stopped at the end: first escapes at sigma = 0.01 are at t ~ 1e13).
- **Trap statistics** are for one teacher basis (seed 1000) and the fan-in init of the model; the trap
  fraction depends on the init's Q_e, i.e. on the init scheme.
- **Stalled vs stuck** is a derived prediction checked on 16 escapes: 8 shortest-estimate k78 runs
  (3 seeds, all escape at 0.50-0.87x), a random 8 of the 74 k78 candidates with estimate <= 100x the
  elapsed time (7 escape at 0.24-2.19x, 1 not by 3x) and 8 mps8 chi = 6 runs (2 escape at 1.37x and
  1.81x, 1 not by 2x nor by 3x, 5 not integrated far enough): 16 of 23 distinct continued runs, with the
  estimate exceeded in 4 of the 16 (1.08, 1.37, 1.81, 2.19x), so it is not a bound. 91 of the 106 k78
  stalled runs (estimate / elapsed: cell medians 17-64x) and 35 of the 43 large mps8 stalled or
  unresolved runs (median about 390x) are untested: for them "ever-longer plateaus" is a conjecture. At
  K = 7, 8 the trapped fractions are lower bounds (stalled fan-in mps8 runs are partly still decaying
  toward the threshold).
- **Trap decay rate ~ sigma^2** is checked at K = 4 only (mps4, 6 seed / teacher pairs); at K = 7, 8 the
  fits are short and the rate / sigma^2 varies with sigma (ht4 seed 0: 0.0057-0.0060 over sigma =
  0.15-0.3 but 0.0017 at 0.4). The K = 7, 8 rows of `dormant_decay.csv` are not citable: the table has no
  sweep column, and 13 K = 7, 8 keys appear in both main and k78 (one pair differs 6x); no claim depends
  on them.
- R10 magnitudes hold as cell medians only: 0.87-1.15 at K = 3, 4 and 1.4-3.2x at K = 7, 8 (up to 3.9 in
  k78; unbalanced routes at moderate sigma); per-seed medians reach 20. The slopes are clean.
- The strict saddle rests on one seed (7 sigmas) plus a second seed at 2 sigmas, close teacher, binary4;
  it is not visited at sigma = 0.003, and it is a known DLN result (replication).
- The "close" teacher [1, 0.85, 0.7] gives overlapping events on binary4 for sigma >= 0.03.
- Next: K = 7, 8 gaps over a wider sigma range would need a cheaper stiff integrator or a reduced
  route ODE (theory_saddles R5) seeded from measured amplitudes; the trap probability vs the init's
  per-bond Q statistics (theory_saddles "all ends unfavourable" rule) could be tested on many seeds at
  the leading order without full runs.

## Log
(Entries are as written at the time; numbers superseded by the fix round or by the round-2 errata are
corrected in the sections above, and in the log only where marked [errata ...].)
- 18:18 Started; read BRIEF.md, CLAUDE.md, notes/model.md, notes/diagnostics.md, PLAN.md, library.
  Machine: cgroup CPU quota 61.4 CPUs (6144000/100000), 251 GB RAM.
- 18:25 Pilot (scipy DOP853, throwaway): binary N=4 GF at sigma 0.3..0.003: t_1 sigma^{K-2} -> 0.7696.
- 18:29 Derived first-escape prediction (P1): t_1 = sigma^{-(K-2)} T(theta_hat) + c_K(s) with T the
  blow-up time of the correlation flow theta' = grad <W, W*> from the sigma=1 init (exact scaling).
  Pilot: matches to 4-5 digits for binary 4/8, MPS 8 (incl. the identity of the first component).
- 18:31 Found stiffness problem: after the first escape, explicit RK steps are capped at
  O(1/s^{2-2/K}), so integrating gaps costs ~ sigma^{-(K-2)}. scipy LSODA/BDF with exact Hessian
  Jacobian (torch.func.hessian, 0.1 s for 850 params): cost independent of sigma (binary 4:
  19 s at sigma 3e-3 and 3e-4 vs 55 s / ~550 s with DOP853), same event times to 1e-6.
- 18:34 Pilot, 40 seeds x 5 trees x chi {3,6} x 2 teachers, correlation flow only: prefactor of t_1 is
  ~ (sigma d^{-N/2K})^{-(K-2)}/((K-2)s) (RMS-overlap heuristic) within factor 1-1.75; the rho_0-based
  hypothesis in PLAN.md (aligned at start) underestimates by 4x-5600x; chi-dependence weak.
- 18:45 Lead messages: merged main (literature review; core fixes 57d5262 incl. teacher/student
  stream bug -> pilots before it marked [old lib], to be redone; draft theory notes). Read
  literature.md 1.3-1.4, 2.1, 2.4, 3, 5 and theory_saddles.md.
- 18:50 K sweep (fixed lib, 17 trees, K=3..15, 30 seeds): P3 within 0.94-2.1x of T; Kumar-Haupt
  formula with delta=||theta_0|| off by 3x..3e7x. Gaps pilot (BDF, 5 trees x 6 sigmas x 5 seeds)
  running to test gap exponents and the conservation-law trap (theory_saddles R12) at K=7,8.
- 19:20 Library v0 landed and merged; experiment code on the library: `experiments/s2s_core/s2s.py`
  (setup, fan-in / gauge-balanced / aligned inits, run with event zoom + plateau Hessians + dormant
  diagnostics, correlation-flow prediction), `run.py` (YAML sweep driver until ttn.runner lands).
- 19:45 Step 1 done: aligned balanced init vs ttn.theory on binary4/8, mps4/8, ht4: max |a_GF - a_th|
  2e-10..3e-9 (dopri5, rtol 1e-10), event times equal to 1e-11 relative (same grid).
- 20:10 Lead: dormant-component suppression is now an explicit question. First observations
  (binary8 / mps8, K = 7, 8, sigma = 0.1-0.2, seed 0, chi = 3): after one or two escapes a dormant
  component's amplitude |<W,U_r>| collapses to 1e-17 .. 1e-41 (below the float64 floor), the loss
  stalls at the partial-fit level, and LSODA crawls. It happens DURING another component's escape
  (e.g. a_2: 2e-12 -> 1e-17 while component 3 escapes), and also slowly on plateaus; the
  gauge-balanced init of the same seed is also suppressed (binary8 chi = 3). chi = 6 keeps the
  dormant amplitude at ~1e-12 in the one test so far. Merged theory_saddles v2.
- 20:45 Interim report 1 (step 1 done, first pass of step 2). Sweep A running.
- 20:58 Sweep A restarted twice: (1) added a roundoff stop (all dormant amplitudes < 1e-15 for a whole
  chunk) and tighter step/wall limits; (2) dropped sigma = 0.05, 0.03 for K = 7, 8 full runs (none of
  ~50 completed; LSODA crawls to the step limit after 1-2 escapes). Per-seed exponents so far: t_1 and
  both gaps 1.00 +- 0.01 (binary4) and 2.000 (mps4) vs K - 2. Trap decay rate ~ sigma^2 per seed.
  Event windows now record bond spectra: every event raises the rank by one on all edges at the same
  sample (mps4 test).
- 21:35 K = 7, 8 later gaps are long (binary8 chi 6 sigma 0.07: events 7.5e5, 5.8e6, 4.0e7) and LSODA
  needs ~15-25 min per run; rtol 1e-6 / atol 1e-9 sigma reproduces rtol 1e-8 event times to 7e-7
  relative, so `large.yaml` (K = 7, 8, chi = 6) uses those. Queue: main -> (large || first) ->
  balanced -> LO sweeps (lo_d, lo_K, lo_chi). Trap sign rule checked (both route ends unfavourable).
- 21:50 Interim report 2. Sweep A done and reclassified (dead-route-end criterion instead of the
  amplitude threshold, which was too strict at K >= 7). Running: first (B), large; queued: k78,
  balanced (K = 3, 4), LO sweeps. Library final merged; projection-bias bug noted.
- 22:44 Mistake, fixed at once: a `mv <file> /dev/null` replaced /dev/null with a regular file for
  under a minute; restored with `mknod -m 666 /dev/null c 1 3` (char device 1,3). No job depended on it.
- 22:44 Continuation test: 8 k78 runs classed "stalled" (route ends alive, budget exhausted), continued
  from their final state, ALL escape, at 0.49-0.85 x the balanced-route estimate of the remaining
  plateau b^-(K-2)/((K-2) s_r), b = (s_r |a_r|)^(1/K) (constant per seed across sigma). Stalled =
  ever-longer plateau, not stuck. A spread sample from `large` is running.
- 23:50 All sweeps done except first_k78 (38/52 jobs) and a continuation batch from `large`. Final
  analysis: Hessians recomputed with the fixed projection; edge check with collision-free thresholds;
  plateau spectra = truncated teacher spectra; K = 7 gap exponents 5.02 / 4.97 (binary8, chi = 6).
- 00:10 Final report written; waiting for the last first-escape jobs (mps8).
- 00:28 Continuation of 8 mps8 chi = 6 stalled runs (`large`): 2 escape at 1.4-1.8x the route estimate,
  1 not by 2x, 4 wall-limited, 1 integrator failure; corrected the claim that the estimate is an upper
  bound (it is only for a balanced-reducible route).
- 00:47 Stopped the last 10 first-escape jobs (mps8 chi = 6; t_1 ~ 1e13 at sigma = 0.01). Final
  first-escape exponents 4.998-4.999 (K = 7), 5.9995 (K = 8, 10 seeds). Figures regenerated. Done.
- 09:46 Machine restarted (replied ok). 09:54 Lead: fix round after review round 1; merged main, read
  `notes/reviews/s2s_core_round1.md` in full.
- 09:56 Correlation flow now stops on direction convergence; T recomputed for all 231 first-escape jobs
  (r1 = GF's first component in 1847/1848 runs; binary4 close T moved <= 2.2e-3) [errata E1, E2, E14:
  was "238", "100%", "1.7e-3"]; lo_d / lo_K / lo_chi rerun
  (unchanged to 3 digits); aligned runs rerun (one file per integrator). Started: zoom-resolved
  continuation of the 8 k78 runs, random continuation sample, 9-run tolerance check.
- 10:05 analyze_main refactored: fits on S2S-classified run sets; strict saddle on binary4 wrong-order
  plateaus confirmed (median 0.98 of the prediction for seed 8, 0.93 for seed 5); class table; mps8 gap
  1 5.95 +- 0.05. Trap fits above 1e-9: rates per seed constant to <= 2.4% (seeds 3, 5, 6) and <= 5.3%
  (seed 7) (the 20-100x switch was a floor artifact) [errata E4, E6: was "0.98", "1-3%"]. Tolerance
  check: 9/9 keep class and order.
- 10:10 WIP commit 7761da5.
- 10:17 Figures c-g regenerated (d and f on the classified sets); slice trends; random continuation
  done (7/8 escape at 0.24-2.19x, 1 not by 3x); notes 2.2-2.7, 3, 4 (claims changed), README, report.
