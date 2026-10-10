# data-teacher round 2 - reproduction review, part 1: setup and independent checks

Setup: experiments/data_teacher/ copied unchanged (diff -rq identical, excluding results/ and __pycache__)
to data/review_data_teacher/r2_repro/code; runs with DATA_TEACHER_OUT=r2_repro/raw,
DATA_TEACHER_RESULTS=r2_repro/results, BLAS threads 1, <= 4 busy processes.

## Independent check 1 (own code, no agent code, no ttn library): saddle-start escape, nonorth N=4 d=2 chi=4
Script r2_repro/indep_saddle.py: dense einsum TTN (mps4/binary4/ht4), HOPM rank-1 point (mps, ht) or top
singular pair of the 01|23 matricization (binary), own Gaussian noise (seed 1000+s), scipy LSODA rtol 1e-10,
event via solve_ivp event function: s_2(cut) >= max(0.05 s_2*(cut), 10 s_2(t=0)).
- c=0.45 mps4: level 0.234463139695 (agent 0.2344631396946), lambda_min_raw -0.1508900825 (agent -0.15089008247),
  9 negative eigenvalues (< -1e-6) = (N-1)(d-1)(chi-1) (agent CSV says n_negative 5 after its threshold rule).
  t_esc seed 0, eps 1e-8..1e-3: 112.827 97.567 82.307 67.047 51.785 36.507 -> 15.260 per decade = A 6.627
  = 1/|lambda_min| 6.627 (to 1e-4). Agent: A = 6.65 +- 0.01 (pred 6.63).
- c=0.2 mps4: level 0.244933777583, lambda_min -0.0284579495 (both = agent); per decade 80.9 -> A = 35.14 = 1/|lambda|.
- c=0.45 binary4: level 0.194510925839 (= agent), grad 8e-16, no negative eigenvalue; t_esc seed 0: 303.0, 3076.1, 30807
  (eps 1e-2, 1e-3, 1e-4), seed 1: 238.8, 2433.2, 24376.8 -> local p 1.007 / 1.001 and 1.008 / 1.001 (agent p 1.005).
- c=0.45 ht4: level 0.234463139695, no negative eigenvalue; eps 1e-2: t_esc 827 (smaller eps not run: time).
- NOTE: first attempt used a rank-1 TENSOR as the binary4 saddle -> gradient 0.215 (not critical). Binary4 bond-rank-1
  = rank-1 matricization (Fact A); the agent's constrained.py gets this right (level 0.19451).

## Independent check 2 (own code): star Hessians (indep_star.py, chi = 3)
Raw Hessian lambda_min at theta_q vs -a^(K-2)(sqrt2 w - 2): equal to 1e-15 on mps4/binary4/ht4 for
w = 1.4146, 1.45, 1.6 (e.g. -0.371572875253811 vs -0.371572875253810); no negative eigenvalue for
w = 1.2, 1.4, 1.4138 -> sign flips exactly between 1.4138 and 1.4146 (w* = sqrt2 = 1.41421).
Blended rank-1 maximum (HOPM, w = 1.6): a_q = 0.6732, a_r = 0.2945, level 2.4595 (agent: ht4 stalls at a_q = 0.67);
Hessian there: ht4 no negative eigenvalue (higher-order saddle), mps4 strict (lambda_min -0.642, 2 modes),
binary4 not a critical point (its bond-rank-1 set is larger). Confirms R2.2 items 1, 2 (Hessian level) and 4.
