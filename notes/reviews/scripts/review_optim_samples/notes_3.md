# Event shift: semi-independent check B (own numpy GD/GF; init + teacher arrays from the library)
- Own GF (DOP853 rtol 1e-12) and GD reproduce the committed GF/GD event times to >= 7 digits
  (e.g. sigma=0.1 seed0 GF 5.493758 / 8.628071 / 47.644301; shift/eta at 0.1 eta_max 2.67176 / 2.52850 / 5.56089 = committed).
- Richardson c = 2 s(0.025) - s(0.05) (s = shift/eta at r = eta/eta_max) vs predict_shift c_e (committed):
  sigma=0.1 s0: 2.68174/2.53412/5.57029 vs 2.68127/2.53408/5.57048
  sigma=0.1 s1: 3.20134/1.72934/2.76813 vs 3.20185/1.72980/2.76778
  sigma=0.1 s2: 2.96167/3.26027/4.12004 vs 2.96159/3.26029/4.12002
  sigma=0.03 s0: 3.89099/3.70331/6.74887 vs 3.89058/3.70336/6.74886
  sigma=0.03 s1: 4.11072/-0.14713/4.21274 vs 4.11093/-0.14675/4.21287   (measured at 0.1: -0.0501!)
  sigma=0.01 s0: 4.99076/4.76545/7.78307 vs 4.99090/4.76547/7.78310
  -> c_e (variational eq.) is the correct eta->0 limit to 3-5 digits; at 0.1 eta_max second events of
     near-tied inits deviate by up to 66% (O(eta^2) term), so "3 digits per seed at <= 0.1 eta_max, all events" is overstated.
