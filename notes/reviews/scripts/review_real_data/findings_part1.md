# real-data reproduction review: running notes (part 1)

## Setup / README
- download.py works as written (MNIST from ossci S3 mirror, Fashion from zalandoresearch), 14 s.
- data.py hardcodes DATA_DIR = /root/outputs/real_data/data (load_task default arg); run.py and the
  analysis scripts that rebuild problems (route_amplitudes, plateau_hessians, regime_analysis) have no
  data-dir argument. README works only if data sit at that path. Worked around with a wrapper that patches
  load_task.__defaults__ (scratchpad rr_run.py / rr_script.py). No repo files edited.
- summarize.py runs directly. Same code on this machine reproduces t_a to 1e-15 (bitwise-level).

## Reruns (agent code, seed 0, short horizons; main.yaml settings otherwise)
quadtree s0.05/0.1/0.2/0.3/0.5, scrambled0 s0.1/0.5, MPS s0.05/0.1/0.2/0.5.
- t_a identical to regime_analysis.json (rel diff <= 3e-15).
- t1 exponent: quadtree 12.999997 (5 sigma), MPS 13.999999 (4 sigma); t1*sigma^(K-2) const to 6 digits.
- k=3 gaps (drop1->drop2), mine vs events_table.json: 297.80/297.81, 142.96/142.66, 68.45/68.54,
  44.77/44.86, 29.2/29 (notes); scrambled0 352.51/352.17, 39.08/38.59. 2nd route [[3,3],[4,3]]: 181.32/181.61,
  141.79/141.67. rho: 32.19/32.12, 15.75/15.7, 7.27/7.29, 4.25/4.27.
- first rank-2 edges: quadtree [3,4] all sigma; scrambled0 [10,11]; MPS [5] (s<=0.2), [5,6] at s0.5 (the 1/35 exception).
- MPS edges 4/3 (seed 0, 4 sigma): log slopes 17.0/32.8 (ref 15.3+-1.5/31.4+-1.1, 5 seeds); log rms 0.15/0.07 vs power 0.31/0.19.
- locality seed 0 s0.1: quadtree tau(L<=0.25)=156, tau(L<=0.2)=333 (ref medians 122/333); scrambled0 L(500)=0.299.
  MPS tau(L<=0.2)=52 (ref 56). s0.5: quadtree tau(L<=0.2)=44.5, scrambled0 L(150)=0.301, MPS 21.

## Independent check (own pipeline/integrator/spectra/amplitudes; library TTNSpec/forward only)
- quadtree s0.2 seed0: t_a 134309842418.6 vs 134309842769.2 (2.6e-9); drops 16.12, 84.83 -> gap 68.71 (ref 68.54);
  first rank-2 edges [3,4] (76.3, 77.1); plateau L 0.2703; route share k<=5 0.927 (ref 0.928), best k3 pair (3,4);
  bottom-core norms balanced (1.09-1.14) so lambda normalization does not change shares.
- quadtree s0.5 seed0 phase 1: t_a 901330.86 vs agent-code 901331.89 (1.1e-6).
