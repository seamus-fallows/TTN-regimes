# part 7: final saddle-start numbers (all 86 reruns, analyze_r2_saddle.py on r2_repro/raw)

Per-run t_esc (f = 0.05) vs committed: misaligned_20, nonorth_c0.2, nonorth_c0.45 (mps/binary/ht), randbin_302 binary,
randbin8_302 mps8: max rel diff <= 7e-3 (median <= 4e-5; residual = row-grid quantization, t_esc = first row above threshold).
nonorth8_c0.7 mps8: offsets -0.2..-1.0 time units (max 2.7% at eps 5e-2). randbin_302 mps4 / ht4: per-seed offsets up to 30 time
units / factor 2.9 (gauge representative of the saddle chosen by 1-ulp loss ordering; see part 5).
Fits on reruns (per seed, log space): mps4 nonorth_c0.45 A 6.636 +- 0.005 (pred 6.627), randbin_302 12.93 +- 0.07 (13.09; committed
13.04 +- 0.11), mps8 nonorth8_c0.7 seed 0 17.72 (pred 17.41; committed seed 0 17.61), randbin8_302 seed 0 2.715 (2.698; committed 2.710);
log law wins 6/6 seeds. binary4 p 1.0022 +- 0.0003 / 1.0033 +- 0.0004 (eps 1e-5..1e-2), ht4 1.022 +- 0.004 / 1.015 +- 0.007
(eps 1e-4..5e-2); log law 0/8. Committed (wider eps range): 1.005 / 1.009 / 1.011 / 1.008.
Wall time used: 09:54-10:34 (machine at 100% CPU from other reviewers), <= 4 busy processes.
