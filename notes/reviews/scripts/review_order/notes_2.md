# review_order notes (part 2): MPS4 d=2 chi=4 (claim 1)
- Agent sweep.py rerun (m4_sigma subset + LO race), stopped at 153/400 seeds (time); 612 GF runs.
- Independent code, all 400 seeds, rho .5/.7/.8/.9/.95/.99 at sigma .01 + LO race: EXACT match of committed
  main_summary row mps4 d2 chi4: p_greedy .815/.665/.605/.545/.51/.49; race_agree 1.0 (2400/2400);
  race_curve .81611/.6795/.61472/.55486/.52708/.50531; sd_log_tau .54635.
- Seed by seed indep vs agent code: winners 462/462, |t1 ratio-1| <= 3.3e-10, |tau ratio-1| <= 5.5e-10.
- sigma scan indep (rho .5/.8/.95 x sigma .1/.03/.003): running; first 78 seeds 100% seed-stable.
