# Order of learning (Q4, hypotheses H1, H1b) — agent `order`

Status: **in progress** (theory derived, experiments running). Experiment folder:
`experiments/order/`. Status tags: **proven**, **derived** (assumptions stated), **empirical**,
**conjecture**.

## 1. Setting

Odeco teacher W* = sum_r s_r u_r^(1) (x) ... (x) u_r^(N), whitened inputs, so
L = 1/2 ||W - W*||^2. Two components with weights (s_1, s_2) = (1, rho), rho < 1, unless stated.
"Greedy order" = component 1 (the larger) has the earlier learning event (first time its overlap
a_r = <W, U_r>/s_r crosses 1/2, `notes/diagnostics.md`). P_g = P(greedy) over the random init.

Fan-in Gaussian init is invariant under orthogonal rotations of any physical leg, so the teacher's
site bases can be taken to be the standard basis (u_r^(i) = e_r) without changing the distribution
of anything below. Then the TTN evaluated on component r,
T_r(theta) = <W(theta), e_r (x) ... (x) e_r>, is the contraction of the "slices" G_v[r, ..., r, :] of
the cores that carry physical legs with the full (shared) cores that carry none.

## 2. Leading-order dynamics and its scale invariance  [derived]

L = 1/2 ||W||^2 - g(theta) + const with g = <W, W*> = sum_r s_r T_r(theta), a homogeneous polynomial
of degree K (number of cores). Near the origin the ||W||^2 term is O(|theta|^{2K}) and
dtheta/dt = grad g + O(|theta|^{2K-1}). The leading-order flow F_t of grad g has the exact scaling
symmetry F_t(sigma n) = sigma F_{sigma^{K-2} t}(n). Consequences:

1. Starting from sigma n_0, the trajectory in *direction* space n(t) = theta/|theta| is the same for
   every sigma; only the radius at which a given direction is reached changes.
2. Writing theta = R n and d tau = R^{K-2} dt: dn/dtau = grad_S g(n) (spherical gradient flow) and
   d ln R / d tau = K g(n). The leading-order flow blows up at finite t but at tau = infinity, so
   the direction converges to a critical point of g on the sphere with g > 0, generically a local
   maximum (Kumar & Haupt's directional convergence in this setting).
3. The real dynamics leaves the leading-order flow when |W| ~ s, i.e. at R = O(1), which is reached
   at tau_1 ~ ln(1/sigma)/(K g). So as sigma -> 0 the first component learned is the local maximum of
   g on the sphere whose basin (under the spherical flow) contains n_0 ("basin picture"); at
   finite sigma the direction at R = O(1) may still be in transit.

## 3. Which states are local maxima: the count m  [derived]

Let m = number of cores that carry sites on which the two components differ (for an odeco teacher:
the number of cores with physical legs). At the state B where only component 2 is present
(rank-one, balanced), consider perturbations toward component 1. Component 1's contribution
s_1 T_1 is a product of the m component-1 slices (all zero at B) and of shared cores. So the gain is
O(eps^m) for slices of size eps, while the spherical constraint (and, after escape, the mixed
entries of W) costs O(eps^2).

- **m >= 3** (odeco on `mps` N >= 3, `binary` N >= 8, `ht`): every component is a local maximum of g
  on the sphere. The first component learned is decided by the init's basin: **random order** (H1).
- **m = 2** (odeco on `binary` N = 4, which *is* a 3-layer deep linear network A R B^T; any teacher
  whose components differ on sites carried by exactly two cores): at B the gain is bilinear,
  s_1 x^T S y, with S the shared cores. On the unit sphere, balanced rank-one B has
  |S| = K^{-(K-2)/2}, g(B) = s_2 K^{-K/2}, and the spherical Hessian has one positive eigenvalue
  lambda_u = (s_1 - s_2) K^{-(K-2)/2} > 0. So B is a saddle of the spherical flow and **only the
  largest component is a local maximum**: greedy order as sigma -> 0 (as in deep linear networks).
- **m = 1**: components differing only on sites of one core form a single rank-one TTN component;
  they are learned together (one event), so there is no order.

## 4. m >= 3, MPS: an exact race  [derived]

For an odeco teacher on an MPS every core carries one site, so T_1 and T_2 depend on disjoint
parameter sets (slices r = 1 and r = 2 of every core). The leading-order flow decouples into two
independent flows, d theta^(r)/dt = s_r grad T(theta^(r)) (a scalar-output chain "deep linear
network" with widths = bond dimensions). Each blows up at t_r = sigma^{-(K-2)} tau_r / s_r, with
tau_r the blow-up time of the unit flow from the normalized slice init; tau_1, tau_2 are iid. Hence

    P_g(rho) = P(tau_2 / tau_1 > rho),       independent of sigma at leading order,

with P_g(1) = 1/2 and P_g -> 1 as rho -> 0. The whole curve follows from the distribution of
log tau (spread s_tau): for a lognormal tau, P_g = Phi(ln(1/rho) / (sqrt(2) s_tau)). d enters only
through the capped bond dimensions (and a common scale factor that cancels).

For `binary` N = 8 the four leaf cores carry component-specific slices but the three internal cores
are shared, so the race is coupled; the outcome is still sigma-independent at leading order
(scale invariance) but not given by independent blow-up times. The "independent-race
approximation" (each component's blow-up time computed with the other component's weight set to 0,
from the same init) is tested against the coupled leading-order flow and full gradient flow.

## 5. m = 2 at finite sigma: wrong order with probability ~ sigma^((1-rho)/rho)  [derived]

Near the saddle B of the spherical flow, the distance delta from B's stable manifold grows as
exp(lambda_u tau) while ln R grows at rate K g(B) = s_2 K^{-(K-2)/2}. The ratio is
lambda_u / (K g(B)) = (s_1 - s_2)/s_2 = (1 - rho)/rho. A trajectory entering the neighbourhood of B
at distance delta_0 from the stable manifold at radius R_0 ~ sigma leaves it at radius
R_0 delta_0^{-rho/(1-rho)}. The smaller component is learned first iff R reaches O(1) before
leaving, i.e. delta_0 < R_0^{(1-rho)/rho}. The stable manifold has codimension one, so delta_0 has a
bounded density near 0 and

    P(wrong order) ~ C(rho) sigma^{gamma},   gamma = (1 - rho)/rho      (m = 2, sigma -> 0).

For rho = 0.5, 0.8, 0.9, 0.99: gamma = 1, 0.25, 0.11, 0.01, so near-degenerate weights stay
essentially random over any accessible range of sigma.

## 6. H1b: correction of a wrong order through a strict saddle  [derived]

After the smaller component is learned (state B: W = s_2 U_2, balanced amplitude q = s_2^{1/K} per
core), add x e_1 and y e_1 to the two component-specific slices in the learned bond index
(m = 2). W gains q^{K-1} (x, y) on two mixed entries (teacher 0 there) and q^{K-2} x y on the
component-1 entry, so

    L - L_B = 1/2 q^{2(K-1)} (x^2 + y^2) - s_1 q^{K-2} x y + O(4)
    Hessian = q^{K-2} [[s_2, -s_1], [-s_1, s_2]],   lambda_min = -q^{K-2} (s_1 - s_2)

(for `binary` N = 4, K = 3: lambda_min = -s_2^{1/3}(s_1 - s_2)). B is a **strict saddle**; the
unstable direction rotates the learned component into the larger one while keeping the bond index,
in time t_corr ~ ln(1/delta_1)/|lambda_min|, where delta_1 is the piggyback amplitude on arrival at B.
Afterwards the smaller component has to be learned again in a new bond index (a full K-core escape,
time ~ sigma^{-(K-2)}). Predicted loss levels: 1/2 s_1^2 (wrong plateau) -> 1/2 s_2^2 -> 0.

Two predictions for t_corr:
- **Controlled start** (W at B, all other directions random at scale sigma): delta_1 ~ sigma and
  t_corr = ln(1/sigma)/|lambda_min| + O(1) (slope 1/|lambda_min| vs ln(1/sigma)).
- **Wrong-order runs from random init**: by Section 5, conditioned on a wrong order, delta_1 is
  approximately uniform on (0, O(1)), so |lambda_min| t_corr ~ ln(1/U) + c is exponentially
  distributed with a **sigma-independent** mean; it does *not* grow like log(1/sigma).

For m >= 3 the same computation gives no quadratic gain (gain O(eps^m), cost O(eps^2)): the
wrong-order plateau is not a strict saddle in any direction toward the larger component, the order
is not corrected, and the larger component is learned by a new K-core escape.

## 7. Results

(to be filled)
