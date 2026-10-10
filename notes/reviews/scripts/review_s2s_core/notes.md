# review_s2s_core working notes

## Code/README issues
- aligned.py: no --workers; Pool(len(jobs)) = 10 processes; raw output hard-coded to s2s.OUT=/root/outputs (no --raw flag); --out default = committed results/ (overwrites committed CSV if run as README says).
- plot_aligned.py: RAW hard-coded /root/outputs/s2s_core/aligned (no flag).
- continue_stalled.py: Pool(len(pick)) uncapped.
- analyze_main.py: --hess default = results/hess_main.csv (committed); on a re-run with a raw dir named main/large/k78/balanced it silently REPLACES the re-run's lambda_min by the committed values for matching keys. Use --hess with no args.
- s2s.dead_end: RuntimeWarning All-NaN slice (harmless; site-less cores).
- Wall-limit/step-limit stops are machine-speed dependent (classification 'unresolved' can depend on CPU speed).

## Continuation test quantization (continued_k78.csv)
- measured_remaining from crossing_times on the continuation's coarse log grid (n_log=200 over [4e-3 pred, 4 pred] when tau0 > t_max) -> crossing = geometric midpoint of the grid cell. Ratios are grid values 4e-3*1000^((i+0.5)/199): 0.8535 appears for binary8 s0.15 seed2 AND ht4 seed9 at two sigmas (identical to 1e-10); 0.5435 for mps8 seed6 at two sigmas; consecutive values differ by exactly 1000^(1/199)=1.0353. So the ratios are resolved only to +-1.8%; 'same ratio per seed across sigma' is partly the quantization. Conclusion 0.49-0.85x unaffected.

## Rerun results so far (10:07)
- aligned: max dev 1.97e-10 (b4) 3.25e-10 (m4) 2.26e-9 (b8) 2.58e-9 (ht4) 2.68e-9 (m8) vs committed 2.0e-10/3.1e-10/2.2e-9/2.5e-9/2.8e-9; event times identical to all digits.
- main subset (30 runs incl. balanced, chi6): class 30/30, order 30/30, t1 <= 2.1e-9, t2 <= 4.2e-7, t3 <= 2.3e-6 rel; lambda_min ratio 1.00; per-seed exponents on same 4 sigmas identical to 4 decimals.
- binary8 chi6 seed0 s0.2: t1,t2,t3 agree ~1e-7 rel.
- indep T (sphere method): binary4 committed T biased high by up to 2.4e-5 (amp_stop=1e6 truncation in s2s.correlation_flow_blowup; with amp_stop 1e12 agent's own function gives 0.84373211 = mine 0.84373211). mps4 agrees 6.5e-11. KH exact. T/KH medians 4.01 (b4), 10.3 (m4) = committed.
## Bookkeeping discrepancies in committed summaries
- edge-step count: CSV gives 497/531 (main 261/291, large 93/95, k78 23/25, balanced 120/120); notes/REPORT say 517/551 (exceptions 34 consistent).
- mps4 plateau-level max err: CSV 2.8e-3 (main) and 8.1e-3 (balanced, chi6 s0.03 seed4) vs 'max 2e-3 (mps4)'.
- binary4 close teacher: level err up to 0.50 on non-S2S runs (no plateau) - claim restricted to S2S runs implicitly.

## Independent check results (10:15)
- own dense GF (LSODA rtol 1e-10, own events): binary4 s0.01 seed0, mps4 s0.01 seed0, mps4 s0.01 seed3 fan-in (trapped) and own-balanced: event times vs committed t1 <=1.8e-8, t2 <=4.4e-7, t3 <=2.7e-6; plateau levels <=1.4e-7; truncated spectra <=1e-9; one rank step per cut, simultaneous (spread 0 at 0.005 width grid); raw lambda_min within 1-20% of committed.
- trap: own run of mps4 seed3 s0.01 fan-in: comp 3 end slice decays exp from t~1e5, hits 1e-15 at t~1.4e6, stays to t_max=2e7 -> trapped (as committed). Own-balanced init: all 3 learned (t 8170.8, 8778.1, 316608) = committed balanced.
- first escape own GF: binary4 seed0 t1 to 1-2e-8 rel; mps4 seed2 (comp 2 first, race) to 1.3e-8; own ratio with converged T: 2.8e-6 (b4, s0.002), -4e-10 (m4).
- aligned theory escape times reproduced by own quadrature to 8 digits; c_K by quadrature = agent's hyp2f1 to 8 digits.
## BUG: trap decay-rate fit includes float64 floor
- analyze_dormant.py fits log(slice) vs t over all points with slice > 1e-13 after 1.5 t_last; the dead slice sits at ~1e-12..1e-11 (above cutoff) for the rest of the run -> rate ~ ln(drop)/window. Runs at small sigma in main (first ~310 runs, no dead-end stop) ran to t_max -> rates biased low. Committed separated seeds 3 and 7 'switch levels by 20-100x at sigma 0.02-0.03' exactly where stop changes from 'completed' to 'stop_fn' (table). My rerun (dead-end stop) seed 3: 0.075/0.094/0.17/0.16 at s=0.003/0.01/0.03/0.1 (vs committed 0.0095/0.0113/0.247/0.27); own pure-decay fit at s=0.01: 0.26 sigma^2. => sigma^2 scaling holds better; 'switch' is an artifact; fig_g right panel values for * runs are lower bounds.
