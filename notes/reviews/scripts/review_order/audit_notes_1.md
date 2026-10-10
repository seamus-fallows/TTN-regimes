# Claims audit: order (Q4) - working notes part 1

## Checks done from committed CSVs
- main_summary: m>=3 P(larger first) rho=0.5 range 0.6025-0.8825 (GF, sigma=0.01); race curves 0.658-0.894.
  rho=0.8: 0.505-0.6525 (race 0.552-0.656). All 17 MPS settings: GF P below pooled race curve by
  0.018 +- 0.02 at every rho (shared seeds 0-399; per-seed race agreement 100%) -> common seed-sample bias.
- d-pairs chi^2: z rows over 6 rho reuse the same 400 seeds (highly correlated), d=2/d=4 share the first
  core's draws (torch.randn sequential; verified: core 0 identical up to 1/sqrt(2)). "6 dof" invalid.
- Selection rule: q/(K-2) = 0.98-1.25, >1.0 in 26/28 settings (MPS4 1.07-1.21, MPS8 1.06-1.22). In-sample.
- sigma fits: binary4 gamma_small_sigma matches notes. pat_mps4_sep rho 0.9 full-range 0.095+-0.006 is
  2.5 SE below 0.111 (restricted fit 0.103+-0.007 OK). Adjacent rho 0.9 local slopes: 0.25, 0.31, 0.42,
  0.34 (1e-1..1e-5): NOT still steepening in the last decade (16 vs 35 events; +-0.08).
- H1b saddle start: local slope x|lam| = 1.0000 (1e-7..1e-6), 1.0002, 1.0022, 1.017, 1.07 (1e-3..1e-2):
  exact asymptotically; reported 1.5% excess and 1.4% SE come from including sigma=1e-2,1e-3.
- H1b natural b4: |lam|(t1-t2) slope vs ln(1/sigma) = -0.02 (log law: +1). Distribution ~Exp(1)
  (mean 1.1-1.2, median 0.69-0.92, sd 0.9). At sigma=0.1 lam_min at min-speed point ~ -0.30, MORE
  negative than saddle values (claim "between 0 and saddle value" only for sigma<=0.01).
- multi2 R=3: perm chi^2 vs iid race (5 dof): m>=3 p=0.18-0.82 (consistent); binary4 p<1e-6.
  P(greedy order) z vs model: -1.49,-0.78,-0.03,-0.71,-0.84,-0.38 (all negative).
  Race model uses pooled tau of ALL runs; GF stats use used runs only (MPS6: 85/199, 88/200).
- Abandoned no-stop `multi` run covered binary4+MPS4 (+25 MPS6 R=3); no binary8, no MPS6 R=4:
  the no-stop cross-check of the stall rule does not cover binary8/MPS6.
- floor-only exclusions (dormant |a|<1e-14, later learned): binary8 R=3 18/19 per rho -> overlaps
  do recover from <1e-14; exclusion criterion uses the overlap PRODUCT, not slice norms.
