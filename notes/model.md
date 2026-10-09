# The model (pinned down in Phase 0)

This is the reference definition used by every experiment. Deviations must be stated in the
experiment's README.

## 1. What the TTN computes

**Inputs.** N sites. Site i carries a local feature vector phi_i(x) in R^{d_i} (default d_i = d).
Default data: phi_i = x^(i) ~ N(0, Sigma_i), independent across sites, with Sigma_i = I_d
("whitened"). Any orthonormal local feature map (Hermite features of scalar inputs, or
(1, x_i) for x_i ~ N(0,1)) gives the same population dynamics, because only E[phi phi^T] enters.

**Model.** f_theta(x) = < W(theta), phi_1(x) (x) ... (x) phi_N(x) >, a scalar. W(theta) in
R^{d_1 x ... x d_N} is a tree tensor network: a rooted tree whose nodes v carry core tensors G_v. Each
core has one leg per physical site attached to v (dimension d_i), one leg per child node (bond
dimension chi_c), and one leg to its parent (bond dimension chi_v; the root's parent leg has
dimension 1, the scalar output). W is the contraction of all cores over all bonds. f is a multilinear
polynomial in the inputs and a multilinear (degree-K homogeneous) polynomial in the K cores.

**Loss.** Population square loss
L(theta) = 1/2 E[(f_theta(x) - f*(x))^2] = 1/2 ||W - W*||_Sigma^2,
with ||A||_Sigma^2 = < A, (Sigma_1 (x) ... (x) Sigma_N) A >. It is computed *exactly* by tree contraction
(cost linear in the number of cores), with no sampling noise. With whitened inputs it is the
Frobenius distance. Finite-sample (empirical) loss and SGD are available for experiments where
sample size matters (implicit bias, generalization, gradient noise).

**Teacher.** Realizable: f*(x) = < W*, Phi(x) >. W* is given either in CP form
(sum_r s_r u_r^(1) (x) ... (x) u_r^(N); "odeco" when the u_r^(i) are orthonormal over r at every site,
so every bond's Schmidt values are exactly the s_r), or as a TTN (on the student's tree or another
tree). CP teachers make the teacher-student inner products tree-agnostic and give exactly known
targets for every bond spectrum.

## 2. Tree topologies

A tree is given by a parent array and the list of sites attached to each node. Built-in shapes
(N sites, K cores):

| name | description | K |
|---|---|---|
| `binary` | balanced binary tree; bottom nodes carry 2 sites; internal nodes have 2 children (for N not a power of 2 a lone site attaches to its parent node) | N - 1 |
| `mps` | chain (caterpillar): node k carries site k; the most unbalanced tree | N |
| `ht` | hierarchical Tucker: balanced binary tree plus one leaf core per site | 2N - 1 |
| custom | any parent array / site assignment | - |

Default for saddle-to-saddle studies: `binary` with N = 4 or 8 and `mps` with the same N, which are
the two extremes of tree shape at nearly equal K. Edge e (between node v and its parent) induces the
bipartition (sites below v | the rest); the bond rank on e is at least the rank of W matricized
across that bipartition, and the singular values of that matricization ("bond spectrum" or Schmidt
values) are the central diagnostics.

**Why this model.** It is the simplest genuine TTN: (i) at least three sites and an internal bond,
so W is a genuine tensor and not a matrix (a deep linear network is the special case of a chain with
physical legs only at the ends); (ii) whitened inputs reduce the population loss to a Frobenius
distance, the exact analogue of Saxe et al.'s setting for deep linear networks, so DLN theory gives
baselines; (iii) it is the regression version of the TN models used in practice (Stoudenmire &
Schwab use the same <W, (x)_i phi_i> form with non-Gaussian local features); (iv) it equals
gradient flow on fully observed hierarchical tensor factorization, the object studied by Razin,
Maman & Cohen (2022), so results connect to that literature.

## 3. Parameterization and initialization

Core v has fan-in F_v = product of the dimensions of its child legs (physical and bond) and fan-out
chi_v (its parent bond). Default ("fan-in") initialization:

G_v = sigma_v * Z_v / sqrt(F_v), Z_v with iid N(0, 1) entries.

Each core is then an approximate isometry times sigma_v, signals keep their size going up the tree,
and E ||W_0||_F^2 = prod_v sigma_v^2 exactly (every leg except the root output is a child leg of
exactly one core). The default is a single per-core scale sigma_v = sigma, so ||W_0|| ~ sigma^K. We
report both sigma (per core) and alpha = sigma^K (function-space scale).

Variants, all supported by the library:
- **Imbalance**: per-core scales sigma_v (e.g. root vs rest, or by depth).
- **Centred model** (main lazy-capable construction): train on the target W* + W_0, where W_0 is
  the student's initial tensor (`ttn.teachers.ShiftedTeacher`); equivalently train f_theta - f_theta0
  on W* (Chizat, Oyallon & Bach 2019). The initial function is removed exactly at any sigma without
  changing the parameterization, so sigma alone moves the TTN from small-init dynamics to the lazy
  limit (time rescaled by the tangent-kernel scale ~ sigma^{2(K-1)}).
- **Two-copy model** (cross-check): W = W(theta+) - W(theta-) with theta+ = theta- at t = 0
  (Woodworth et al. 2020), stored as a TTN with doubled bond dimensions and block-diagonal cores.
  Gradient flow does NOT preserve the block structure on its own (the root's mixed blocks couple the
  "+" subtree of one child to the "-" subtree of another), so the gradient is multiplied by a 0/1
  mask (`ttn.network.antisymmetric_double` returns it).
- **Aligned ("decoupled") init** for CP/odeco teachers: cores diagonal in the teacher's basis, giving
  exactly solvable Saxe-type dynamics (one scalar ODE per component).
- **Per-core learning-rate multipliers** eta_v. A forward multiplier a_v on core v is equivalent to
  init scale and learning rate rescaled by a_v and a_v^2; this is how NTK-type and muP-type
  parameterizations for the large-bond-dimension limit are expressed.

## 4. Training dynamics

- **Gradient flow** (reference dynamics): d theta/dt = -grad L (times eta_v per core), integrated with
  an adaptive Dormand-Prince 5(4) method in float64. Time t is gradient-flow time. Plateaus near the
  origin are long (~ sigma^{-(K-2)}), so adaptive steps are required. Accuracy is checked with the
  bond-wise conservation laws (below) and against closed-form solutions.
- **Gradient descent** with constant learning rate eta (time t = eta * steps) for learning-rate studies.
- **SGD / online SGD** on sampled inputs for finite-sample and noise studies.
- All computation in float64: with sigma = 1e-2 and K = 7, W_0 ~ 1e-14 and gradients ~ 1e-12, which
  float32 cannot resolve over long plateaus.

## 5. Symmetries and conserved quantities

For each bond e = (v, parent p), inserting A A^{-1} on e leaves W unchanged (gauge symmetry). Under
gradient flow with equal per-core learning rates this conserves the chi_v x chi_v matrix
Q_v = G_v^T G_v - G_p,(v) G_p,(v)^T, where G_v is matricized as (fan-in x chi_v) and G_p,(v) is the
parent core matricized with its v-leg as rows. (Derivation: d/dt of the gauge orbit's generator.)
These are the TTN version of DLN balancedness. They are a correctness check for integrators and a
knob (imbalanced inits) for the regime map.

## 6. What later results might depend on this choice

- **Population vs finite samples.** Plateau structure and saddle sequences are properties of the
  population gradient flow. Statements about implicit bias and generalization need finite samples and
  are tested separately.
- **Multilinear (polynomial) features.** No pointwise nonlinearity; any orthonormal local feature map
  gives the same population dynamics. Non-orthonormal features are equivalent to anisotropic Sigma_i.
- **Product input measure.** Makes the population loss a tree contraction. Cross-site correlations
  break this and are studied (if at all) with sampled data.
- **Where the physical legs sit.** `binary` (two sites per bottom core) vs `ht` (one leaf core per
  site) changes K and how many cores must change to learn a component; predicted plateau-time
  exponents depend on this (hypothesis H2 in `notes/prior_knowledge.md`).
- **Scalar output** (regression). Multi-output or classification would add an output leg and change the
  root core.
- **Fan-in init.** Fixes the relative scales of cores. Exponents should not depend on it; prefactors
  and the order of learning might, which is why imbalanced and aligned inits are also supported.
