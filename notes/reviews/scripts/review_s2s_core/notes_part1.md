# s2s-core claims audit: working notes (part 1)

## Verified numerically from committed CSVs (pandas)
- first_summary: t1 exponents (offset removed) 0.9996/2.0000/4.998-4.999/5.9995; residual at smallest sigma <= 1.75e-3 (binary4 close), <= 6e-5 mps4, <= 2e-7 K=7,8. OK.
- lo_chi: slopes chi=4-16 binary4 -0.335+-0.028, mps4 -0.264+-0.037 (reproduce notes). mps8 chi2->8: 4.02x (tKH 33.4x, notes say 35x).
- lo_d: medians 1.544/1.569/1.563 (binary4), 1.39/1.41/1.50 (mps4). sd lnT binary4 d=2: 0.218 vs pred 0.25 (+-0.02 band of theory_escape 9.3 exceeded).
- aligned.csv: max dev 2e-10..3e-9 (dopri5). OK.
- main_exponents reproduce REPORT table.

## Problems found
1. binary4 close, chi=3, seed 8: order 1-3-2 at 7/8 sigmas; plateau-2 lambda_min (projected) = -0.118..-0.146,
   sigma-independent over 0.005-0.1 = strict saddle; predicted -s3^(1/3)(s2-s3) = -0.133 (notes 1.4 piggyback).
   per-seed lam2 slope 0.025, gap2 exponent -0.08; these runs are s2s=False (rho~1). Included in cell mean
   gap2 0.92+-0.11 and lam2 slope. chi=6 seed 4 similar at sigma>=0.03 (lam2 -0.36..-0.53).
   Restricting fits to S2S runs: gap1/gap2 = 1.004/1.016 (chi3), 1.004/1.014 (chi6) -> clean.
   Contradicts notes sec 3 "no strict saddles seen with odeco teachers" and REPORT "higher-order saddles".
2. r1 mismatch on binary4 close (80-90%) is an artifact of amp_stop=1e6 in correlation_flow_blowup:
   rerun with amp_stop 1e10/1e14 -> r1 = 0 (largest) for all 5 mismatched seed/init cases (data/review_s2s_core/lo_r1.json);
   T changes by up to 1.7e-3 relative. Agent attributes it to E9 finite-sigma wrong order (GF is greedy at every sigma).
3. Continuation ratios quantized by log output grid (n_log=200, factor 4): all five k78 values sit at 0.496 of a
   grid interval (3.5% wide). 8 k78 runs = 3 seeds (binary8 fanin s2, mps8 bal s6, ht4 bal s9); chosen as shortest
   predicted (pred/elapsed 0.2-1.0 vs median 10-60x for stalled runs).
