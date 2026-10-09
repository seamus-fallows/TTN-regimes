# Input covariance and teacher structure (Q6, Q7, H5) — agent `data-teacher`

Status tags: **proven**, **derived** (under stated assumptions), **empirical**, **conjecture**.
Experiments: `experiments/data_teacher/`. Raw outputs: `/root/outputs/data_teacher/`.

## 0. Notation and two structural facts used throughout

- Teacher components U_r = (x)_i u_r^(i) (unit vectors), W* = sum_r s_r U_r.
- Product covariance Sigma = (x)_i Sigma_i; loss L = 1/2 ||W - W*||_Sigma^2.
- "Aligned": every u_r^(i) is an eigenvector of Sigma_i, Sigma_i u_r^(i) = lambda_{r,i} u_r^(i);
  Lambda_r = prod_i lambda_{r,i} = ||U_r||_Sigma^2.
- g(theta) = <W(theta), Sigma W*> = E[f_theta f*], the correlation that drives the dynamics
  near the origin (the quadratic term 1/2 ||W||_Sigma^2 is smaller by a factor ||W|| / ||W*||).

**Fact A (binary-4 is a depth-3 linear network)** — *proven* (by inspection of the contraction).
`binary_tree(4)` has two bottom cores carrying sites (0,1) and (2,3) and a root with no sites. Both
edges induce the same bipartition (01|23). Writing G_1 as a (d^2 x chi) matrix, the root as a
(chi x chi) matrix R and G_2 as (d^2 x chi):
W_(01|23) = G_1 R G_2^T. So binary-4 is exactly a depth-3 deep linear network acting on the
d^2 x d^2 matricization, and with a product covariance its loss is
1/2 tr[(W - W*)^T Sigma_01 (W - W*) Sigma_23] (DLN with input covariance Sigma_23 and output
metric Sigma_01). Consequences: binary-4 is "matrix-like" (only one cut), all DLN theory applies, and
the rank-constrained critical points are SVD truncations of the (whitened) matricization. Genuinely
tensorial small trees are MPS-3 (K = 3, two distinct cuts) and MPS-4 (K = 4, three cuts).

**Fact B (whitened frame)** — *proven*. Substituting x~_i = Sigma_i^{-1/2} x_i (whitened) gives
f* = <Sigma^{1/2} W*, (x)_i x~_i>, i.e. a whitened problem with teacher W~* = Sigma^{1/2} W*, and the
student's cores carrying site i are multiplied by Sigma_i^{1/2} on that physical leg (still a TTN on
the same tree). Gradient flow in the new cores is preconditioned by Sigma_i on those physical legs
and starts from an anisotropic init. Hence:
- aligned covariance = whitened odeco teacher with weights sqrt(Lambda_r) s_r + preconditioning;
- misaligned covariance = whitened **non-orthogonal** CP teacher with factors Sigma_i^{1/2} u_r^(i)
  (per-site overlap u_r^T Sigma_i u_q) + preconditioning. Step 2 therefore connects to step 4(a).

## 1. Aligned anisotropic product covariance (H5)

### 1.1 Exact solution for aligned (decoupled) init — *derived* (exact, no approximation)

With `aligned_init` (cores diagonal in the teacher basis, amplitudes a_{v,r}) and an aligned
covariance, Sigma (W - W*) = sum_r Lambda_r (Pi_r - s_r) U_r (cross terms vanish because
u_r^(i) T Sigma_i u_q^(i) = lambda_{r,i} delta_rq), so the diagonal family is invariant and
L = 1/2 sum_r Lambda_r (Pi_r - s_r)^2, da_{v,r}/dt = Lambda_r (s_r - Pi_r) Pi_r / a_{v,r}.
For balanced amplitudes (a_{v,r} = a_r, Pi_r = a_r^K):

    dPi_r/dt = K Lambda_r (s_r - Pi_r) Pi_r^{2 - 2/K},   i.e.  Pi_r(t; Sigma) = Pi_r^white(Lambda_r t).

Each component follows the whitened trajectory with its own clock t -> Lambda_r t. Consequences:
- half-time t_r = T(s_r, a_0) / Lambda_r with T(s, a_0) = (1/K) int_{a_0^K}^{s/2} dPi / ((s - Pi) Pi^{2-2/K})
  = a_0^{-(K-2)} / ((K-2) s) + O(s^{-(2-2/K)}) for K >= 3. **To leading order the escape time depends on
  Lambda_r s_r only** (H5's substitution s -> Lambda s);
- widths w_r = w^white(s_r) / Lambda_r ~ 1 / (K Lambda_r s_r^{2-2/K}), **not** the
  1/(K (Lambda_r s_r)^{2-2/K}) that the substitution s -> Lambda s would give;
- final amplitudes are s_r (the network learns W*, not Sigma W*); plateau loss levels are
  L_S = 1/2 sum_{r not in S} Lambda_r s_r^2 = 1/2 sum ||s_r U_r||_Sigma^2 (function-space norms).

So H5 is right for escape times but "behaves as if its weight were Lambda_r s_r" is not literally
true for the whole trajectory; the exact statement is the per-component time rescaling.

### 1.2 Random init: first escape — *derived* (leading order as sigma -> 0)

L = 1/2 ||W||_Sigma^2 - <W, Sigma W*> + const, and Sigma W* = sum_r (Lambda_r s_r) U_r is again
odeco, with weights Lambda_r s_r. Near the origin the quadratic term is O(||W||^2) against
O(||W|| ||W*||) for the linear one, so **up to the first escape the flow is the same as the whitened
flow with teacher weights Lambda_r s_r, started from the same init** (relative error O(||W|| / s)).
Predictions: (i) paired runs (Sigma, s) and (I, Lambda s) from the same init pick the same first
component and have the same first-escape time up to O(1) (vs. t_1 ~ sigma^{-(K-2)});
(ii) the probability that component r is learned first is P_white(r | weights Lambda s): the order
statistics of H1 apply with Lambda s in place of s.

### 1.3 After the first escape: which saddle is stable — *derived* (second-order expansion)

Function-space norms sqrt(Lambda_r) s_r, not Lambda_r s_r, decide the rank-constrained optimum
(Eckart–Young in the whitened frame, Fact B): the best bond-rank-1 approximation in the Sigma-norm is
the component with the largest Lambda_r s_r^2. When the race (1.2) picks a component a with
Lambda_a s_a > Lambda_b s_b but Lambda_a s_a^2 < Lambda_b s_b^2, the plateau after the first escape
sits at a non-optimal rank-1 critical point. Whether the network can leave it quickly depends on
locality (H1b): on binary-4 the "piggyback" move (rotate the learned component's two 2-site blocks
toward component b; two cores, k = 2) has, at the saddle Pi = s_a, the Hessian in the rotation
angles (phi_1, phi_2) of the two bottom cores

    H = [[s_a^2 lambda_{b,1} lambda_{a,2},  -s_a s_b Lambda_b],
         [-s_a s_b Lambda_b,               s_a^2 lambda_{a,1} lambda_{b,2}]],
    det H = s_a^2 Lambda_b (Lambda_a s_a^2 - Lambda_b s_b^2),

(lambda_{r,v} = product of the eigenvalues of component r on the two sites of core v; Pi decouples at
second order). So the saddle with a learned is **strict iff Lambda_b s_b^2 > Lambda_a s_a^2**: the
correction criterion is the loss-greedy one (function-space norm), while the race criterion is
Lambda s. For Sigma = I this reduces to H1b (strict iff s_b > s_a). For equal per-core splits the
negative eigenvalue (in angle units) is s_a sqrt(Lambda_b) (sqrt(Lambda_a) s_a - sqrt(Lambda_b) s_b).
On MPS-4 the same move needs all 4 cores (the cross term is quartic in the angles), so no strict
saddle: a wrong first choice is not corrected quickly.

Three candidate orderings, which the experiments separate: by s_r (teacher weights), by
Lambda_r s_r (race / escape rate, H5), by Lambda_r s_r^2 (loss-greedy / stable saddles).

## 2. Misaligned covariance — theory

Fact B maps a misaligned product covariance to a whitened problem with the **non-orthogonal** CP
teacher W~* = sum_r s_r (x)_i Sigma_i^{1/2} u_r^(i) (per-site overlaps u_r^T Sigma_i u_q), with
the physical legs preconditioned by Sigma_i. Reparametrization maps critical points to critical points,
so **the set of plateau states (saddles) is exactly that of the whitened non-orthogonal problem**
(*proven*); only the timing (preconditioning, anisotropic init) differs. Two different tensors
matter:
- the race near the origin is driven by Sigma W* = sum_r s_r (x)_i Sigma_i u_r^(i) (factors
  Sigma_i u_r, effective weights s_r prod_i ||Sigma_i u_r^(i)||; per-site overlap of the factors
  grows like (lambda_1^2 - lambda_2^2)/(lambda_1^2 + lambda_2^2) at 45 degrees);
- the plateau states are rank-constrained critical points of 1/2 ||W - W*||_Sigma^2, i.e. of the
  whitened frame with factors Sigma_i^{1/2} u_r (overlap (lambda_1 - lambda_2)/(lambda_1 + lambda_2)
  at 45 degrees).
So the escape direction (top of Sigma W*) and the saddle it must reach (critical point in the
Sigma^{1/2} frame) are different tensors: after the escape the state must rotate. *Derived*.

## 4(a). Non-orthogonal CP teacher — theory

W* = s_1 U_1 + s_2 U_2, per-site overlap c = <u_1^(i), u_2^(i)>, whitened.
- Rank-1 critical points: for X = rho (x)_i x_i, L = 1/2 rho^2 - rho <(x)x_i, W*> + 1/2||W*||^2 is
  minimized at rho = <(x)x_i, W*>, so the rank-1 critical points of L are the critical points of
  g(x) = <(x)_i x_i, W*> on unit vectors, with plateau level 1/2 (||W*||^2 - g^2). *Proven*.
- MPS-4 (tensorial): with equal weights, restricted to x_i = x in span(u_1, u_2) at angle psi,
  g = s [cos^N(psi) + cos^N(psi - beta)], beta = arccos c. The symmetric point psi = beta/2 is a
  local max iff tan^2(beta/2) < 1/(N-1), i.e. **c > (N-2)/N = 1/2 for N = 4**. *Derived*. Below it
  there are two rank-1 local maxima (one near each component; the first plateau is at one of them,
  chosen by the init basin); above it a single "blended" maximum, so the first saddle is a blend of
  the two components and the order question disappears.
- Binary-4 (matrix-like, Fact A): the rank-1 critical points are the singular modes of the
  matricization W*_(01|23) = s_1 a_1 b_1^T + s_2 a_2 b_2^T with a_r = u_r (x) u_r,
  <a_1, a_2> = <b_1, b_2> = c^2. The top mode is a blend for every c > 0 (no bifurcation), the plateau
  level is sigma_2^2 / 2 (sigma_k = singular values of the 2x2 problem), and the order is greedy
  as sigma -> 0 (unique local max of g, DLN). *Derived*.

## Results

(filled in as experiments finish)
