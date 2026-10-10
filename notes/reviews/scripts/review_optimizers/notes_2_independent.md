# Part 2: independent checks (no experiments/optimizers code)

Script `indep_check.py`: library TTNSpec/fan-in init/to_dense/odeco factors only; own dense loss, own
overlaps, own Adam/RMSProp update (torch.optim formulas), GF first escape by scipy LSODA (rtol 1e-10,
atol 1e-15) with a terminal event max_r a_r = 1/2 (exact location, no output grid).
Outputs: data/review_optimizers/indep/{b4,b8,eps,cg}.csv; fits in data/review_optimizers/results/indep_*.csv.

## binary4 (fan-in, same inits as scaling_b4), 7 sigma in [0.002, 0.128] x seeds 0-2, per-seed fits
| method | my p (mean +- se) | committed p (6 seeds) |
|---|---|---|
| GF first escape | 1.006 +- 0.001 (1.007, 1.006, 1.004) | 1.007 +- 0.001 |
| Adam lr 1e-3 | -0.004 +- 0.010 | 0.009 +- 0.009 |
| Adam lr 1e-4 | 0.012 +- 0.005 | 0.012 +- 0.003 |
| RMSProp lr 1e-3 | -0.005 +- 0.013 | 0.001 +- 0.019 |
lr*n_1 (median): 0.177 (lr 1e-3), 0.169 (lr 1e-4) vs committed 0.180, 0.168.
Per run: my Adam/RMSProp first-escape steps equal the committed t_first to < 5e-5 relative (63/63 runs);
GF event times within -0.27%..+0.19% (committed = interpolation on the output grid); same first component 84/84.

## binary8 fan-in (indep; committed later_b8 is balanced init), 5 sigma in [0.25, 0.8] x 3 seeds
GF p = 4.66 +- 0.10 (4.72, 4.46, 4.80); Adam lr 1e-3 p = 0.014 +- 0.049 (0.057, -0.083, 0.068).
Committed (balanced, 7 sigma x 6 seeds): GF 4.76 +- 0.06, Adam 0.049 +- 0.022. Per run vs committed deep
fan-in rows (sigma 0.3, 0.8): Adam within 0.05%; GF within -1.3%..+0.1% (grid interpolation, see below).

## c_g = rms grad at init / sigma^(K-1) (fan-in, 6 seeds, median): binary4 0.22, binary8 0.013, mps8 0.010
-> sigma_eps(1e-8) = 2.1e-4, 0.095, 0.14 (committed: 0.22/0.011/0.031 -> 2e-4/0.10/0.12; their mps8
value is for the balanced init). Consistent.

## eps law, MPS8 fan-in (independent; exact GF event times at the same sigma)
| seed | sigma | n_1 (eps 1e-8, lr 1e-3) | n_sign (eps 1e-30) | T_GF(sigma) exact | ratio (direct T) | ratio ((0.3/s)^6 extrap.) |
|---|---|---|---|---|---|---|
| 0 | 0.03 | 8498.1 | 137.1 | 8.3714e8 | 0.9988 | ~0.999 |
| 0 | 0.05 | 500.95 | 137.1 | 3.9058e7 | 0.949 | ~0.949 |
| 1 | 0.03 | 45879.7 | 137.3 | 4.5734e9 | 1.0002 | ~1.000 |
| 1 | 0.05 | 2244.2 | 137.3 | 2.1338e8 | 0.988 | ~0.988 |
GF exponent check: T(0.3)(0.3/s)^6 vs direct T(s): 0.9997-1.0004 (sigma^-6 law exact from 0.3 down).

## fan-in stall on MPS8 (claim 6), sigma 0.8 seed 0, my dense loss + scipy LSODA, t to 2e4
rtol 1e-9/atol 1e-14 and rtol 1e-11/atol 1e-30 give the same trajectory: a_2 ~ 1e-4 before the 2nd escape,
then collapses 1e-4 -> 1.6e-13 (t=190) -> 1.8e-15 (t=197) -> ~1e-19 (float64 floor, random sign) by t=212;
loss stays at 0.500 (s=1 component unlearned) to t=2e4. Committed: same run stalls, a_end_2 ~ 1e-38.
=> the death of the dormant component is dynamical (decay rate ~0.6 per unit time during/after the 2nd
escape), not an integrator-tolerance artifact. Claim 6 supported (for this case).

## eps law re-analysis of the COMMITTED measurements with exact GF times (agent's balanced inits, my integrator)
Exact T_GF(0.3) (scipy event) vs committed deep-sweep grid value: seed0 984.79 vs 968.04 (-1.7%), seed1 4453.56
vs 4503.75 (+1.1%), seed2 7213.72 vs 7168.59 (-0.6%), seed3 5286.66 vs 5196.27 (-1.7%), seed4 1531.92 vs
1540.80 (+0.6%), seed5 3356.50 vs 3388.41 (+0.9%).
Ratios measured/predicted (committed n_1, committed n_sign median):
| subset | committed ratio range | with exact T_GF(0.3) |
|---|---|---|
| lr 1e-3, sigma 0.02-0.03 (8 cases) | 0.989-1.018 | 0.9989-1.0003 |
| lr 1e-2, sigma 0.015-0.02 (8) | 0.990-1.024 | 1.0009-1.0062 |
| lr 1e-2, sigma 0.03 (6) | 1.001-1.053 | 1.010-1.035 (crossover) |
| lr 1e-3, sigma 0.05 (6) | 0.966-1.009 | 0.955-0.994 |
| lr 1e-2, sigma 0.05 (6) | 1.087-1.178 | 1.081-1.172 |
T_GF measured directly at 0.03/0.05 gives the same ratios to < 0.05% (sigma^-6 extrapolation exact).
=> claim 2 holds, more precisely than stated: deep in the eps regime the law is exact to ~1e-3; the committed
0.97-1.05 band is mostly the +-1.7% output-grid error of T_GF(0.3) plus the lr 1e-2 / sigma 0.03 crossover.

## eps law on binary4 (K = 3) far below the stated sigma_eps = 2e-4 (own Adam lr 1e-3, exact GF event)
| seed | sigma | T_GF | n_1 (eps 1e-8) | n_sign (eps 1e-30) | pred (eps/lr)T_GF + n_sign | ratio |
|---|---|---|---|---|---|---|
| 0 | 1e-4 | 6.82e3 | 195.2 | 177.7 | 177.8 | 1.098 |
| 0 | 1e-6 | 6.82e5 | 184.1 | 181.0 | 187.8 | 0.980 |
| 0 | 1e-8 | 6.82e7 | 822.4 | 181.3 | 863.2 | 0.953 |
| 0 | 1e-9 | 6.82e8 | 6988 | 181.3 | 7000 | 0.998 |
| 1 | 1e-4 | 6.23e3 | 177.9 | 176.2 | 176.3 | 1.010 |
| 1 | 1e-6 | 6.23e5 | 181.2 | 176.2 | 182.5 | 0.993 |
| 1 | 1e-8 | 6.23e7 | 762.6 | 176.2 | 799.6 | 0.954 |
| 1 | 1e-9 | 6.23e8 | 6400 | 176.2 | 6410 | 0.998 |
The law n_1 = (eps/lr) T_GF + n_sign holds (0.95-1.10, 0.998 deep in the regime), but on binary4 the GF
plateau only becomes visible below sigma ~ 1e-8, four decades below the notes' sigma_eps = (eps/c_g)^(1/(K-1))
= 2e-4. The visible crossover is where (eps/lr) T_GF(sigma) ~ n_sign, which depends on lr (lr 1e-2 shifts it
down) and on K; for K = 8 it happens to be ~0.08 (close to sigma_eps = 0.12). "Below sigma_eps Adam inherits
GF's plateau" is right for the tested K = 7-8 but the sigma_eps formula is not the general crossover.

## binary8 BALANCED inits (agent's initial_point for the init only; own Adam, exact GF event), full later_b8 grid
7 sigma in [0.25, 0.8] x 6 seeds, every-step resolution (committed: Adam fine trace every 10 steps):
| method | my p (per-seed mean +- se) | committed p |
|---|---|---|
| GF first escape | 4.740 +- 0.046 | 4.764 +- 0.055 |
| Adam lr 1e-3 | 0.037 +- 0.019 (0.085 -0.042 0.067 0.054 0.051 0.008) | 0.049 +- 0.022 |
| Adam lr 1e-2 | 0.145 +- 0.033, all 42 resolved | 0.136 +- 0.055 (5 seeds; 15/42 excluded unresolved) |
Per run vs committed: GF -1.7%..+1.7% (output grid), Adam lr 1e-3 -0.5%..+0.3%, Adam lr 1e-2 -7%..+18% (the
10-step fine trace cannot resolve escapes at 15-25 steps; the agent flagged t < 20 as unresolved, but the
"resolved" ones still carry up to ~5% error). First component identical in 126/126 runs.

## finite samples, binary4 n = 45 (own data draw: numpy seeds 1000-1002; own GF (LSODA to 1e-8 L0 or t = 1e6) and own Adam lr 1e-3)
| seed | sigma | GF ranks / pop loss | Adam ranks / pop loss |
|---|---|---|---|
| 0 | 0.01 | 33 / 37.4 | 33 / 11.9 |
| 0 | 0.1 | 44 / 14.2 | 44 / 7.40 |
| 1 | 0.01 | 33 / 0.050 | 33 / 0.448 |
| 1 | 0.1 | 33 / 0.042 | 33 / 2.23 |
| 2 | 0.01 | 33 / 13.6 (t_max hit, train 4.8e-7 L0) | 33 / 2.99 |
| 2 | 0.1 | 33 / 6.14 (t_max hit, train 3.0e-7 L0) | 33 / 2.98 |
Rank-4 endpoints: GF 1/6, Adam 1/6 (committed, agent data: GF 1/12, Adam lr 1e-3 8/12; seeds 0-2: Adam 5/6).
Adam's population loss lower than GF's in 4/6. => on fresh data draws the "Adam uses the full bond dimension
/ generalizes worse" effect at n = 45 does not appear in 6 inits (P(<= 1/6 | rate 8/12) ~ 0.02). Follow-up with
the agent's exact data draw (torch generator seed + 7e6) and my own optimizer code: indep/bias_theirdata.csv.
Agent's data draw + my code (bias_theirdata): seed 0 sigma 0.01 -> GF 33 / 15.684 (committed 15.684), Adam 44 /
31.247 (committed 44 / 31.240, 30117 vs 30252 steps). So the agent's pipeline is correct; the rank-4 effect
depends on the particular datasets (each seed has its own dataset).
