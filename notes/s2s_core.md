# s2s-core: saddle-to-saddle phenomenology and escape-time scaling (Q1, Q2)

Agent `s2s-core`. Experiments in `experiments/s2s_core/`. Status tags: **proven**, **derived**
(under stated assumptions), **empirical**, **conjecture**. Setting: `notes/model.md` (whitened
inputs, exact population loss, odeco teacher W* = sum_r s_r U_r, fan-in init with per-core scale
sigma, gradient flow).

## 1. Theory

Notation: K cores, g(theta) = <W(theta), W*> (degree-K homogeneous polynomial in theta),
L = 1/2 ||W||^2 - g + 1/2 ||W*||^2. With the same seed, fan-in init is linear in sigma:
theta_0(sigma) = sigma * theta_hat, theta_hat = the sigma = 1 init.

### 1.1 Correlation flow and exact scaling of the first escape

**Correlation flow.** For ||W|| << s_r the term 1/2 ||W||^2 (degree 2K) is negligible next to g
(degree K): its gradient is smaller by a factor ~||W||/s. To leading order the first plateau is the
gradient flow of g alone, theta' = grad g(theta) ("correlation flow").

**Scaling (proven).** If theta(t) solves theta' = grad g(theta) then lambda theta(lambda^{K-2} t)
solves it too (grad g is homogeneous of degree K-1). For K >= 3 the correlation flow blows up in
finite time T(theta_0) whenever g eventually becomes positive along the flow, and
T(sigma theta_hat) = sigma^{-(K-2)} T(theta_hat).

**First-escape prediction (derived; assumption: the escaping component is balanced and aligned by
the time ||W|| becomes O(s), which the conservation laws guarantee up to O(sigma^2) imbalance).**

    t_1(sigma, seed) = sigma^{-(K-2)} T(theta_hat) + c_K(s_{r1}) + o(1),          (P1)

where r1 = the component whose overlap blows up first in the correlation flow, and
c_K(s) = s^{-(2-2/K)} [ int_0^{2^{-1/K}} x dx / (1 - x^K) - 2^{1-2/K} / (K-2) ] is the difference
between the a = 1/2 crossing time of the balanced Saxe mode q' = q^{K-1}(s - q^K) and the blow-up
time of its correlation part q' = s q^{K-1}, in the limit q_0 -> 0 (derivation: subtract the two
integrals dt = dq / (q^{K-1}(s - q^K)) and dt = dq / (s q^{K-1})). Numerically the bracket is
-0.850, -0.266, -0.047, 0.072, 0.148, 0.201 for K = 3..8.

(P1) has **no free parameters**: T(theta_hat) is computed per seed by integrating the correlation
flow from the sigma = 1 init (cheap: no stiffness, self-similar blow-up), and it predicts the
exponent K - 2 exactly and the prefactor per seed. It also predicts which component is learned
first (r1).

**Spherical decomposition (derived).** Write theta = rho u, |u| = 1, d tau = rho^{K-2} dt. Then
du/dtau = grad_S g(u) (spherical gradient ascent of g, independent of rho), and
d rho / dtau = K g(u) rho. Hence

    T(theta_0) = rho_0^{-(K-2)} int_0^inf exp( -(K-2) K int_0^tau g(u(tau')) dtau' ) dtau.

If u started at a maximizer of g on the sphere, the integral would be 1 / ((K-2) K g_max).

**Lower bound (proven for the trees used here, odeco W*).** For any TTN,
|g(theta)| <= s_1 prod_v ||G_v||_F <= s_1 (||theta|| / sqrt(K))^K. (Write g = sum_r s_r a_r with
a_r the tree contraction evaluated at u_r at every site; bound each internal contraction by the
Frobenius norms; the cores carrying physical legs see disjoint slices G_v(u_r, ...) for different r,
so sum_r prod_{v in P} ||slice_{v,r}|| <= prod_{v in P} (sum_r ||slice_{v,r}||^2)^{1/2} by Hoelder
with |P| >= 2 factors, P = cores with physical legs.) Hence g_max <= s_1 K^{-K/2} on the unit
sphere, with equality at the balanced rank-one configuration aligned with component 1, and

    T(theta_hat) >= ||theta_hat||^{-(K-2)} K^{K/2-1} / ((K-2) s_1).                 (B1)

The PLAN's hypothesis t_1 ~ rho_0^{-(K-2)} [tau_align + K^{K/2-1}/((K-2) s_1)] is (B1) plus an
alignment delay. Only the parameters g depends on matter: at each physical leg, directions
orthogonal to span{u_r} (and for a two-site core, the mixed slices u_r (x) u_r', r != r') have zero
gradient in the correlation flow and stay frozen at O(sigma). Replacing ||theta_hat|| by the norm of
the *active* parameters rho_act (slices along the teacher directions + cores without physical legs)
gives a tighter bound (B1').

### 1.2 Decoupling of components in the correlation flow (derived)

For an odeco teacher, g = sum_r s_r a_r(theta) with a_r = f_theta(u_r^(1), ..., u_r^(N)). On a core
with physical legs, a_r depends only on the slice G_v(u_r, ..., u_r; bonds). These slices are disjoint
for different r. Hence:
- **MPS** (every core carries a site): the correlation flow splits into R independent scalar
  "deep linear networks" (depth K, widths chi, scalar input and output), one per component, each
  with its own blow-up time T_r. Then T = min_r T_r and r1 = argmin: the first component is decided by
  a race between independent DLN escapes (`order` studies the statistics).
- **binary / HT**: the slices at cores carrying sites are disjoint, but the internal cores (no
  physical legs) are shared by all components, so the components are coupled through them already
  in the correlation flow.

### 1.3 Later escapes (derived, heuristic)

After component r1 is learned, its cores are balanced rank-one with amplitude s^{1/K} along a bond
direction b_e on every edge e. Any part of another component's slices along b_e creates mixed terms
(e.g. u_2 (x) u_1 (x) ...) that 1/2 ||W||^2 penalizes with curvature O(s^{2-2/K}); those parts relax in
O(1) time. The rest of component 2's slices keeps following its correlation flow in the
orthogonal complement of b_e. An odeco component differs from the learned one at every site, so
adding it needs all K cores (the k-rule of H2 gives k = K): the next escape is again a degree-K
escape and

    t_{i+1} - t_i ~ sigma^{-(K-2)} * C_i(seed)        (exponent K - 2 for every gap)   (P2)

with C_i from the projected correlation flow (no closed form). Widths of events are O(1) (time from
a = 0.1 to a = 0.9 of a Saxe mode, ~ s^{-(2-2/K)}), so separation ratios grow like sigma^{-(K-2)}:
**S2S for every tree with K >= 3 as sigma -> 0** (derived). Caveat: if two components blow up at
nearly the same rescaled time (close weights), the separation at a given sigma can be small.

### 1.4 Saddle type at partial fits (derived)

At the partial fit W_S (components in S learned, balanced), the second-order change of L along a
direction that adds component r' needs every core to change, so it is of order eps^K: zero Hessian
(higher-order saddle) for K >= 3. Exception ("piggyback", H1b): perturb only the cores carrying
physical legs, G_v += eps u_r'^{(x) sites(v)} (x) (bond directions of the learned component r).
Then W gains eps^m q^K U_r' (m = number of cores with physical legs) plus mixed terms of order eps.
For m = 2 (binary N = 4) the Hessian along this direction is q (s_r - s_r') with q = s_r^{1/K}:
**negative iff a larger component r' is unlearned while r is learned** (a wrong-order saddle is
strict). For m >= 3 (binary N = 8: m = 4; MPS and HT: m = N) the gain is O(eps^m) and the direction
is again higher order. Prediction for the Hessian at plateau midpoints: no eigenvalue below -epsilon
for K >= 3, except binary N = 4 plateaus where a larger component is still unlearned.

### 1.5 Typical prefactor: RMS-overlap heuristic (derived heuristic, P3)

Fan-in init makes W_0 rotation-invariant on every physical leg with E||W_0||^2 = sigma^{2K}, so the
initial overlap with any unit rank-one U_r has RMS a_rms = sigma^K d^{-N/2}, independent of chi.
If component r escaped like an aligned balanced mode started at that amplitude
(q_0 = a_rms^{1/K} = sigma d^{-N/(2K)} per core), then

    T_r ~ T_rms(s_r) = d^{N(K-2)/(2K)} / ((K-2) s_r)     [in units of sigma^{-(K-2)}]      (P3)

and t_1 ~ sigma^{-(K-2)} min_r T_r. This has no free parameters and no chi dependence. It replaces the
PLAN's rho_0-based hypothesis, whose aligned-at-start form is the lower bound (B1) and is off by
factors 4 (K = 3) to ~5000 (K = 8, chi = 6) in pilots (section 2.1).

## 2. Experiments and findings

### 2.1 Pilots (throwaway scipy code, before the library landed; to be redone with the library)

**Library fix 57d5262** (teacher and student shared random streams) changed every teacher draw. My
pilots before it used teacher seed 100 and init seeds 0..39, so they were never correlated with the
teacher, but their per-seed numbers are not comparable with later runs. Pilots marked [old lib] below
will be redone; the K sweep is already on the fixed library.

**t_1 vs K (fixed library; correlation flow, chi = 3, d = 4, weights [3, 2, 1], 30 seeds)**:
geometric-mean T vs P3 (`d^{N(K-2)/(2K)}/((K-2) s_1)`) and vs the Kumar-Haupt formula
`1/(K(K-2) s_1 K^{-K/2} ||theta_hat||^{K-2})`:

| tree | K | T (geo. mean) | T / P3 | T / KH |
|---|---|---|---|---|
| mps 3..10 | 3..10 | 0.67 .. 17.6 | 0.96 .. 1.65 | 3.0 .. 2.5e4 |
| binary 4, 6, 8, 12, 16 | 3, 5, 7, 11, 15 | 0.88 .. 814 | 0.94 .. 2.1 | 4.1 .. 3.1e7 |
| ht 3, 4, 6, 8 | 5, 7, 11, 15 | 0.41 .. 5.4 | 1.06 .. 1.72 | 16 .. 2.1e5 |

So the KH formula (init along the KKT point, delta = ||theta_0||) is a strict lower bound (B1) that
misses by up to 7 orders of magnitude at K = 15, while the RMS-overlap heuristic P3 tracks the
typical prefactor over a factor-1500 range within a factor 0.94-2.1 (the excess grows slowly with K).

All gradient flow with scipy DOP853 / LSODA, rtol 1e-9, atol 1e-14 sigma; correlation flow
DOP853 rtol 1e-10. d = 4, odeco teacher seed 100.

- **(P1) confirmed per seed** (empirical): t_1 sigma^{K-2} at sigma = 0.3..0.003 (binary N=4,
  K=3), 0.3..0.01 (binary N=8, MPS N=8) converges to the correlation-flow blow-up time T(theta_hat)
  to 4-5 digits (binary 4 seed 0: 0.76962 at sigma = 3e-3 vs T = 0.77021; the remaining offset
  t_1 - sigma^{-1} T = -0.198 matches c_3(3) = -0.196). The first component learned is the one whose
  correlation flow blows up first (also when it is the smallest, MPS 8 seed 1).
- **Race model** (MPS: proven by decoupling; others: empirical): the blow-up time with only
  component r in the teacher, T_r, gives T = min_r T_r exactly for MPS (rel. diff 1e-12) and within
  1-3% for binary/HT, with the correct first component in 93-100% of 40 seeds.
- **Typical prefactor (P3)**: median T_r / T_rms(s_r) = 1.0-1.2 (binary 4, MPS 4), 1.3-1.75 (binary 8,
  MPS 8, HT 4), chi = 3, 40 seeds. The aligned-at-start bound (B1, full norm) is 3.5-5600x too small.
- **chi dependence** is weak: median T for chi = 6 vs chi = 3 is lower by factors 1.2 (binary 4,
  MPS 4) to 1.6-1.9 (binary 8, MPS 8, HT 4), much less than the (sum chi)^{-(K-2)/2} ~ 1/8 implied
  by the rho_0 picture for K = 8.
- Seed-to-seed spread of T: coefficient of variation 0.2 (K = 3) to 0.8 (HT 4, MPS 8).

### 2.2 Dormant-component suppression (in progress; sweep `configs/main.yaml`, LSODA rtol 1e-8, atol 1e-11 sigma)

Question (from the lead): do dormant components shrink during earlier escapes and plateaus, by how
much, vs sigma, K, chi, and does a gauge-balanced init avoid it?

- **Two mechanisms are seen.** (i) *Plateau decay (trap, theory_saddles R12)*: on a plateau after an
  escape, the dormant component's slices on the route END cores decay exponentially while the inner
  slices stay put; its amplitude falls to 1e-30 and below. (ii) *Collapse at an escape*: the dormant
  amplitude drops by orders of magnitude while another component escapes, and its slices then stay
  roughly constant (MPS8 floor runs: end-slice decay rate / sigma^2 ~ 1e-9; binary8 chi = 3 seed 0
  with both fan-in and gauge-balanced init).
- **Plateau-decay rate scales as sigma^2 per seed** (empirical, mps4, chi = 3, sigma = 0.003-0.02, 7
  trapped seeds over both teachers): the end-core decay rate divided by sigma^2 is constant per seed
  across 0.8 decade of sigma (e.g. 0.27-0.29, 0.49-0.82, 0.009-0.013, 0.012-0.016); the same seeds trap
  at every sigma. **Derived (reduced model of theory_saddles section 8)**: with the learned components
  balanced at amplitude q = s_q^(1/K) per core, the dead end's new column x and the parent's visible
  slice y_vis obey x' = -q^(2(K-2)) |y_vis|^2 x, y_vis' = -q^(2(K-2)) |x|^2 y_vis, with
  |x|^2 - |y_vis|^2 - |y_route|^2 = Q_nu nu conserved; if Q_nu nu + |y_route|^2 < 0 then
  x ~ exp(-gamma t) with gamma = q^(2(K-2)) |Q_nu nu + |y_route|^2|. Fan-in init makes Q_e = O(sigma^2)
  (sigma^2 times a seed-dependent O(1) matrix) and y_route = O(sigma), so gamma = sigma^2 x (seed
  constant), as measured. Consequence (derived): the trap time 1/gamma ~ sigma^-2 competes with the
  escape time ~ sigma^-(K-2): for K = 4 both scale alike, so whether a seed traps is sigma-independent
  (as measured); for K >= 5 a trapped end dies long before the route can escape.
