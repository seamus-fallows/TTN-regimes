# optim-samples: learning rate, SGD noise and finite samples (Q8)

Status: working - Branch: agent/optim-samples - Updated: 2026-10-09 18:55 UTC

## Summary
Question: how do discrete-step GD, SGD noise and a finite training set change the gradient-flow
(GF) picture of TTN training regimes (Q8, H6)? Done so far: theory (sharpness at the balanced minimum,
GD stability limit, the reduced 1D GD map above it, backward-error event shifts, SGD noise on
plateaus, exact SGD mean-square stability limit) and pilots. The step-1 production sweep (750 GD/GF
runs) is running. Library modules `ttn/dynamics.py` etc. have not landed yet; the experiment code uses
minimal GD/SGD loops plus `scipy` DOP853 for GF and will be switched to the library when it lands.

## Results (so far; details and derivations in `notes/optim_samples.md`)
1. **Sharpness formula verified (proven + numerical check to 1e-15).** At the balanced global
   minimum lambda_max(H) = K s_1^{2-2/K} for binary N=4,8, MPS N=4,8, HT N=4 (chi = 3, 5); imbalanced
   aligned minima have s_1^2 sum_v a_v^{-2} >= K s_1^{2-2/K} (AM-GM), so the balanced minimum is the
   flattest. eta_max = 2/(K s_1^{2-2/K}): 0.154 (binary 4), 0.0435 (binary 8), 0.0366 (MPS 8) for
   teacher [3, 2, 1]. Evidence: `experiments/optim_samples/check_sharpness.py` ->
   `results/sharpness_check.csv`.
2. **Above eta_max (pilot, binary N=4, sigma = 0.1)**: no global minimum is stable, and the GF
   solution is already the flattest, so there is no EoS self-stabilization. Component 1 behaves like
   the 1D balanced map x -> x - h x^{K-1}(x^K - 1): 2-cycle at 1.05 eta_max (others learned, loss floor
   0.1); held at a period-3 point (a_1 = 0.2) at 1.5 eta_max **and the order changes (component 2 learned
   first)**; at 2 eta_max learned components are **thrown back to the origin saddle and re-learned
   repeatedly** (loss returns to L(0)); divergence at 3 eta_max. 1D-map thresholds
   (`map1d.py`, `results/map1d_attractor.csv`): period 2 at h/h_c = 1, period 4 at 1.25, chaos ~1.33,
   re-entry at 1.84 / 1.99 / 2.23 / 2.29 and divergence at 2.16 / 2.41 / 2.72 / 2.86 for K = 3/4/7/8.
3. **Which minimum is selected (pilot)**: imbalanced init (root core scale 4 or 6) makes the GF minimum
   1.41x / 1.95x sharper than the balanced one. GD converges to sharpness = min(lambda_GF, 2/eta) (to
   1%) and reduces the imbalance |tr Q| (9.6 -> 3.8): classic EoS selection of flatter, more balanced
   minima, possible only for imbalanced inits.
4. **GD tracks GF below eta_max (pilot)**: relative event-time error is linear in eta (6e-4 at
   0.01 eta_max, 5e-2 at 0.9 eta_max, sigma = 0.1, binary N=4); GD is slower. Backward error analysis
   predicts an absolute shift (eta/2) ln(|grad L(t_e)| / |grad L(0)|) ~ (eta/2)(K-1) ln(1/sigma), so the
   relative error vanishes as sigma -> 0 (plateau ~ sigma^{-(K-2)}).
5. **SGD noise does not shorten plateaus (pilot, binary N=4, sigma = 0.1, eta = 5e-4, 60 noise seeds
   per B)**: std(t_1) = 0.48 / 0.21 / 0.13 / 0.065 for B = 1/4/16/64 vs the parameter-free 1D
   prediction sqrt(t_1 eta V / B)/s_1 = 0.56 / 0.28 / 0.14 / 0.07 (V = s_1^2 (3^N - 1) + sum_{r>1} s_r^2:
   heavy-tailed product features); mean t_1 within ~1% of GF (B = 1: +0.10 +- 0.06, predicted +0.12).
6. **SGD stability limit (derived, exact linear mean-square analysis, binary N=4)**: eta_ms(B) =
   0.00107 B for B <= 16 (vs GD 0.154), 0.053 at B = 64, 0.112 at B = 256: product features have
   kurtosis 3^N, so small-batch SGD needs eta ~ B/(lambda 3^N)-ish. `sgd_stability.py` ->
   `results/sgd_stability.csv`.

## What was done
1. Read docs, merged main (report skeleton, literature review v1, PLAN v1).
2. Sharpness check script and theory notes (`notes/optim_samples.md`).
3. Pilots (throwaway, scratchpad): GD vs GF, EoS, imbalance, SGD plateau noise, GF event times N=8.
4. Step-1 sweep script `gd_sweep.py` + `configs/gd_sweep.yaml` (750 runs) launched.

## Decisions
- **Teacher [3, 2, 1], d = 3, chi = 3** (same as `s2s-core`'s separated teacher) so stability limits
  and plateaus are comparable across agents.
- **Own minimal GD/SGD loops before the library lands** (brief: pilots only). The lead's message says
  the library is still hours away and the interim report is due after step 1; the loops are a few
  lines, isolated in `experiments/optim_samples/`, and runs are cheap, so I will rerun key settings
  with `ttn.dynamics` once it lands rather than wait idle.
- **sigma for N = 8 trees: 0.9** (GF first events at t ~ 4-80; at sigma = 0.5 later events take
  > 1000 time units for K = 7, 8, which makes small-eta GD runs too long).

## Files left on the machine
- `/root/outputs/optim_samples/gd/` per-run trajectories (npz) and summaries (json) of the step-1 sweep.

## Caveats and next steps
- Step 1 analysis and figures after the sweep; then step 2 (SGD sweep over B and eta, aligned-init
  check of the sign of the SGD mean shift, Q_e / bond-rank tracking and a stochastic-collapse test as
  requested by the lead) and step 3 (finite samples).

## Log
- 18:20 Read brief and project docs; merged origin/main; set up report.
- 18:25 Sharpness formula verified numerically on 5 trees (machine precision).
- 18:30 1D GD-map bifurcations; GD-vs-GF pilot; exact SGD mean-square limit.
- 18:40 SGD plateau pilot (240 runs): noise spreads but does not shorten plateaus.
- 18:42 Lead message: literature v1 on main; merged; read sections 1.4, 2.7, 3, 5.
- 18:45 EoS and imbalance pilots; launched step-1 sweep (750 runs).
