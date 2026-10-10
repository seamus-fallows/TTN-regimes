# Independent check A (own numpy TTN, own random teacher basis, own init RNG; binary N=4, d=chi=3, [3,2,1])
- Sharpness at balanced minimum: lambda_max(J^T J) = 12.980246132766668, FD Hessian 12.980246132760536,
  random orthogonal gauge 12.980246132766672; formula K s1^{2-2/K} = 12.980246132766679 (rel 8e-16 / 5e-13).
- GD stability (sigma 0.03 & 0.1, 5 own seeds, T = 300 / 100): gf_like 10/10 at 0.90, 0.95, 0.99 eta_max;
  oscillating 10/10 at 1.01, 1.05, 1.1, 1.2, 1.3; 1.5: 8 diverged + 2 reentry; 1.75: 10 diverged;
  2.0: 10 reentry; 2.5: 10 diverged. Boundary bracketed to [0.99, 1.01] eta_max.
- Excess tail loss (mean over last 5% of steps minus GF end loss) vs 9 x 1D-map attractor loss (own map):
  1.01: 0.0273 vs 0.0273; 1.05: 0.1305 vs 0.1306; 1.1: 0.2482 vs 0.2479; 1.2: 0.4515 vs 0.4496;
  1.3: 0.5611 vs 0.5574 (period 4).
