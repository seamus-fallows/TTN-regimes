# part 4: K sweep reruns (committed code)

- mps5 (K = 5), chi = 4, d = 2, centred, seed 0, 14 sigmas: sigma_c 0.861001 [0.7275, 1.019] ok,
  pred 1.878797, rho_slope -2.53186 (commit -2.53202), n_s2s 11: identical.
- mps8 (K = 8) seed 0 at 0.5125 / 0.6729 / 0.8835 / 1.16:
  0.5125 stalled at t = T = 3.25e5, l_end = 0.308 (component 1 learned at t = 32.8, component 2 never:
  dead or very long route); 0.6729 S2S (events 6.5, 5187; rho 4593); 0.8835 S2S (1.4, 61.4; rho 70;
  l_end 2.4e-4, i.e. converged by the 1e-3 rule while on/near the floor); 1.16 stalled with both
  components learned (events 0.5, 28.0), l_end = 0.018 (centred floor, chi = 4 deficient), rho 2.3.
  -> committed bracket [0.8835, 1.16] reproduced; 1.01 is the geometric midpoint between the last S2S
  run and the first floor-stalled run, not an S2S/rich crossing.
- mps8 seed 3 at 0.6729 S2S (rho 27), 0.8835 rich (rho 2.6): resolved boundary reproduced.
- chi = 4 (sigma_chi_b4) seeds 2, 3 (5 sigmas each): seed 2 low boundary is genuine (t1, t2 nearly
  degenerate, no transient); seed 3 has an a_2 transient above 1/2: at sigma = 0.0398 labelled rich
  (rho 0.30) but S2S under the last-crossing rule (rho 4.65) -> per-seed boundary 0.029 -> ~0.05.
