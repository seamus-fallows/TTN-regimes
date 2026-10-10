# review_order notes (part 1)
- Machine at 100% CPU (other reviewers). Agent sweep ~10-15 s/job (MPS4 race + 4 GF).
- README: sweep/h1b/multi default --procs 90 (env PROCS), not "all CPUs"; repair/tol_check use cpu_count.
- README: `analyze_multi.py multi` but multi.yaml has name multi2 -> analyzes RAW/multi (abandoned run) or fails.
- README: run_queue.sh writes to default RAW/RES (overwrites committed results); env ORDER_RAW/ORDER_RES work.
- Independent Hessian (own code, no ttn): binary4 lam_min = -0.396850/-0.266371/-0.185664/-0.096549 (rho .5/.7/.8/.9) = committed.
  separated pairs MPS4 (0,2), MPS6 (0,5): -(s1-s2)q^(K-2) exact. odeco MPS4, MPS6 m=3: no negative eig.
  ADJACENT pairs (MPS4 (1,2), MPS6 (2,3), chi_e=4): lam_min = -s1 q^(K-2) (x3 = chi_e-1) plus one -(s1-s2)q^(K-2);
  chi_e=1: -(s1-s2)q^(K-2). => claim-3 formula holds for separated m=2 pairs only.
- h1b saddle slope: committed per-seed t1 diffs per decade -> 12.40 = ln10/|lam| at sigma<=1e-3; the 1.5% excess
  in the reported slope comes from sigma=1e-2 (pre-asymptotic). Not a problem.
- natural b4 at sigma=1e-4: lam_min NaN in many rows (record grid spacing ~2 > gap t1-t2 ~3-10: <3 points); not reported as unresolved count.
- Indep MPS4 code (numpy closed-form grads, DOP853) vs agent sweep: winners identical, t1 and tau agree to ~2e-10/5e-10.
