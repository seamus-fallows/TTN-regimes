# What I knew before the literature review

Written by the lead before any searching (Phase 1, step 1), so the literature review can confirm,
correct, or extend it. Citations here are from memory and are **unverified** unless
`notes/literature.md` confirms them. Status tags: **[known]** I am confident this is established;
**[believed]** I think it is established but am not sure of the details or attribution;
**[hypothesis]** my own conjecture for TTNs.

## 1. Setting in one paragraph

A tree tensor network (TTN) represents a tensor W in R^{d^N} as a contraction of small "core"
tensors placed on the nodes of a tree; tree edges are bonds of dimension chi. A TTN regression model
is f(x) = <W, phi_1(x) (x) ... (x) phi_N(x)>. With product-measure inputs whose local features are
orthonormal (e.g. x^(i) ~ N(0, I_d) and phi_i(x) = x^(i)), E[Phi Phi^T] = I, so the population square
loss is exactly ||W(theta) - W*||_F^2 / 2. Training a TTN on such data is therefore the same as
gradient flow on a fully observed hierarchical tensor factorization. W is multilinear in the cores
(each core appears once), so the model is a homogeneous polynomial of degree K (= number of cores) in
the parameters, the same structural property that drives the dynamics of deep linear networks (DLNs)
of depth K.

## 2. Deep linear networks

- **[known]** Saxe, McClelland & Ganguli (ICLR 2014): exact solutions of gradient flow for DLNs with
  whitened inputs and "decoupled"/aligned initialization. Each singular mode s of the input-output
  correlation is learned independently along a sigmoidal trajectory; learning time is about 1/s
  (times a log of the initial scale for depth 2). Small initialization gives stage-like learning,
  strongest modes first. Saxe et al. (PNAS 2019) used this for semantic development.
- **[known]** Depth L >= 3: for a balanced mode, a' = a^{L-1}(s - a^L) (per-layer amplitude a).
  Starting from a0, the escape time scales as a0^{-(L-2)} / ((L-2) s), polynomial in the inverse init
  scale, versus log(1/a0)/s for L = 2. For L >= 3 the origin is a higher-order (non-strict) saddle
  (Kawaguchi 2016, "Deep learning without poor local minima").
- **[known]** Conservation laws: under gradient flow, W_{l+1}^T W_{l+1} - W_l W_l^T is constant (Arora,
  Cohen & Hazan 2018; Du, Hu & Lee 2018, "layers are automatically balanced"). Small init therefore
  means approximately balanced layers throughout training. More generally, continuous symmetries of
  the parameterization give conserved quantities (Kunin et al. 2021, "Neural mechanics"; Marcotte,
  Gribonval & Peyre 2023, "Abundant conservation laws").
- **[known]** Incremental / greedy low-rank learning: Gidel, Bach & Lacoste-Julien (2019, discrete
  GD in 2-layer linear nets); Gissin, Shalev-Shwartz & Daniely (ICLR 2020, depth sharpens incremental
  learning); Arora, Cohen, Hu & Luo (NeurIPS 2019, deep matrix factorization, implicit bias not
  captured by norms; depth strengthens low-rank bias); Li, Luo & Lyu (ICLR 2021, greedy low-rank
  learning: as init -> 0, GF follows a rank-incremental algorithm); Razin & Cohen (2020, implicit
  regularization may not be explainable by norms).
- **[known]** Jacot, Ged, Simsek, Hongler & Gabriel (2021), "Saddle-to-saddle dynamics in deep linear
  networks": as the init scale -> 0, GF visits a sequence of saddles of increasing rank, each a
  critical point of the rank-constrained problem. They prove the escape from the first saddle and
  conjecture the full sequence. **[believed]** Escape directions follow the "optimal escape path";
  plateau lengths scale with the init in the way implied by depth.
- **[believed]** Advani & Saxe (2017/2020) and Lampinen & Ganguli (ICLR 2019): high-dimensional /
  generalization dynamics of DLNs. Braun, Domine, Fitzgerald & Saxe (NeurIPS 2022): exact dynamics
  for non-aligned inits. Domine et al. (2024/25), "From lazy to rich": exact solutions across the
  lazy-rich spectrum with unbalanced inits. Kunin et al. (NeurIPS 2024), "Get rich quick": the
  relative scale of layers (conserved imbalance), not only the overall scale, decides lazy vs rich.
  Tu, Aranguri & Jacot (2024): mixed lazy/active dynamics in linear networks.

## 3. Diagonal linear networks and the kernel-to-rich transition

- **[known]** Woodworth et al. (COLT 2020), "Kernel and rich regimes in overparametrized models":
  for f = <w_+^2 - w_-^2, x> with init scale alpha, the limit alpha -> inf gives the kernel (min
  l2) solution and alpha -> 0 the rich (min l1) solution; intermediate alpha gives a hypentropy-like
  bias. Depth sharpens the transition. Their "two copies with opposite sign" construction makes the
  initial function exactly zero at any init scale.
- **[known]** Chizat, Oyallon & Bach (NeurIPS 2019), "On lazy training": with model scaling
  alpha * (f(theta) - f(theta_0)) and loss / alpha^2, large alpha gives lazy training (the
  linearization around init is accurate; relative parameter change O(1/alpha)). Laziness is a
  matter of scaling, not width. Jacot, Gabriel & Hongler (2018) NTK; Lee et al. (2019) wide nets
  evolve as linear models.
- **[known]** Pesme & Flammarion (NeurIPS 2023), "Saddle-to-saddle dynamics in diagonal linear
  networks": in the alpha -> 0 limit with time rescaled by log(1/alpha), the trajectory is piecewise
  constant, jumping between saddles; the sequence is given by an explicit algorithm akin to
  LARS/homotopy for the minimum l1 interpolator. Berthier (2023, JMLR), "Incremental learning in
  diagonal linear networks", gives a similar picture.
- **[known]** Geiger, Spigler, Jacot & Wyart (2020): feature vs lazy training phase diagram in
  (width, init scale). Ghorbani et al. (2019): limitations of lazy training.

## 4. Mean-field / muP

- **[known]** Two-layer mean-field limit: Mei, Montanari & Nguyen (2018); Chizat & Bach (2018);
  Rotskoff & Vanden-Eijnden (2018); Sirignano & Spiliopoulos (2020). With 1/width output scaling the
  infinite-width dynamics is a nonlinear PDE over the neuron distribution and features move O(1).
- **[known]** Yang & Hu (ICML 2021), Tensor Programs IV: the abc-parameterization classification;
  every stable parameterization is either kernel-limit or feature-learning, and muP is the maximal
  feature-learning one. Yang, Simon & Bernstein (2023): spectral condition for feature learning
  (weights and their updates should have spectral norm ~ sqrt(fan_out / fan_in)). Bordelon & Pehlevan
  (2022): dynamical mean-field theory of feature learning. Depth-muP for very deep nets (Yang et
  al. 2023; Bordelon et al. 2023).
- **[hypothesis]** Nothing in this line has been worked out specifically for TTNs, whose hidden
  "layers" are combined multiplicatively (bilinear in children's features), not by addition plus a
  pointwise nonlinearity. The spectral condition should still give the right muP-like scalings per
  core when a core is viewed as a map from (children legs) to (parent leg).

## 5. Tensor factorization and tensor networks

- **[known]** Razin, Maman & Cohen (ICML 2021), "Implicit regularization in tensor factorization":
  GF on CP factorization (tensor completion) from small init learns rank-one components
  incrementally (each component's norm evolves with a dynamics that is slow when small and fast when
  large); implicit bias toward low tensor rank, which they propose as a complexity measure.
- **[known]** Razin, Maman & Cohen (ICML 2022), "Implicit regularization in hierarchical tensor
  factorization and deep convolutional neural networks": the same for hierarchical tensor
  factorization (HT format, which is a binary-tree TTN); implicit bias toward low *hierarchical*
  tensor rank via "local components" at each node; link to locality bias of CNNs; an explicit
  regularizer to counteract it. **[believed]** They analyse incremental growth of local component
  norms but do not give plateau-time scaling laws vs tree topology.
- **[believed]** Ge, Ren, Wang & Zhou (NeurIPS 2021), deflation in overparameterized tensor
  decomposition; Wang, Wu, Lee, Ma & Ge (NeurIPS 2020), beyond lazy training for overparameterized
  tensor decomposition. Hariz et al. (ICML 2022), implicit regularization with polynomial growth in
  deep tensor factorization.
- **[known]** Tensor power method for orthogonally decomposable (odeco) tensors (Anandkumar et al.
  JMLR 2014): every component is an attracting fixed point (robust eigenvector), so the component
  found depends on the initialization basin, unlike the matrix case where only the top singular
  vector is stable.
- **[believed]** Ben Arous, Gheissari & Jagannath (2021): online SGD for non-convex losses; escape
  from the uninformative region takes time governed by the "information exponent" (polynomial in
  dimension for exponent >= 3). Relevant analogy: higher-degree saddles mean polynomial escape.
- **[believed]** Kumar & Haupt (2024/25): for homogeneous networks with small init, the weights
  converge in *direction* early in training to a KKT point of the constrained "neural correlation
  function" maximization, before the norm grows. This is the general version of the spherical-flow
  argument below.

## 6. Tensor networks for machine learning

- **[known]** Stoudenmire & Schwab (NeurIPS 2016): MPS classifiers with local feature maps
  phi(x_j) = (cos(pi x_j/2), sin(pi x_j/2)), trained by DMRG-like sweeps. Novikov, Trofimov &
  Oseledets (2016), Exponential Machines (TT-format weights, Riemannian optimization). Stoudenmire
  (2018), multi-scale TTN/MERA. Liu et al. (2019), TTN classifiers; Cheng et al. (2019), TTN
  generative models; Han et al. (2018), MPS Born machines; Glasser et al. (2019), expressive power.
- **[known]** Cohen, Sharir & Shashua (COLT 2016): convolutional arithmetic circuits correspond to
  HT decompositions; depth efficiency. Levine, Yakira, Cohen & Shashua (ICLR 2018): entanglement
  (bond rank across a cut) measures the correlations a network can model; matching tree structure to
  data correlations matters.
- **[believed]** Barren plateaus: Liu, Yu, Duan & Deng (PRL 2022) report that gradient variance
  vanishes exponentially with system size for some TN learning setups (global losses, random unitary
  initialization) but not others; Miao & Barthel (2023/24) show absence of barren plateaus for
  isometric TTN/MERA energy minimization (polynomial decay). Practitioners know that random TN
  initialization makes the norm of W explode or vanish exponentially in N, so init scale needs care.

## 7. Hierarchical tensor formats (numerical analysis)

- **[known]** HT format (Hackbusch & Kuehn 2009), HSVD and quasi-optimal truncation (Grasedyck 2010),
  TT (Oseledets 2011). The bond rank on edge e equals the rank of the matricization of W across the
  bipartition that e induces; fixed-rank HT/TT tensors form smooth manifolds (Uschmajew &
  Vandereycken 2013; Holtz, Rohwedder & Schneider 2012). Dynamical low-rank approximation / time
  dependent variational principle on HT/TT (Lubich, Rohwedder, Schneider & Vandereycken 2013).
- **[known]** Gauge freedom: inserting A A^{-1} on a bond leaves W unchanged. Gradient flow therefore
  conserves, for each bond, the difference of the Gram matrices of the two adjacent cores matricized
  along that bond. This is the TTN version of DLN balancedness.

## 8. My hypotheses for TTNs before reading further

Let g(theta) = <W(theta), W*> and K = number of cores.

- **[hypothesis] H1 (first escape).** Near the origin the dynamics is the gradient flow of g, a
  degree-K homogeneous polynomial. In the time variable tau with d tau = |theta|^{K-2} dt, the
  direction follows the spherical gradient flow of g while the radius grows like
  exp(K * integral of g). So as the init scale sigma -> 0 the escape direction converges to a local
  maximum of g on the sphere. These maxima are rank-one TTN configurations aligned with *local*
  best rank-one approximations of W*. For a matrix-like teacher only the top component is a local
  max (as in DLNs), but for genuinely tensorial teachers (e.g. odeco with N >= 3) every component is a
  local max. So **the first component learned is random, decided by the basin of the initialization,
  not just by the largest teacher weight.** The probability of "greedy" order should depend on the
  ratio of teacher weights, on K (the init-overlap factor enters with power K-2), and maybe on chi.
- **[hypothesis] H2 (escape-time exponent depends on locality).** At a saddle where some components
  are already learned, a new component's amplitude is the product of the "new-direction" entries of
  the cores that must change to add it, times O(1) learned cores. If k cores must change, the escape
  behaves like a depth-k DLN mode: plateau ~ sigma^{-(k-2)} / s_eff for k >= 3, ~ log(1/sigma) / s_eff
  for k = 2, and no plateau (convex, single-core update) for k = 1. k is set by the tree: roughly the
  number of cores in the minimal subtree spanning the sites where the new component differs from what
  is already learned, plus the bonds whose rank must grow. **So the same teacher can be learned with
  different plateau structure by an MPS and a balanced binary tree**, and correlations between sites
  that are far apart in the tree are learned slowly at small init.
- **[hypothesis] H3 (saddle type).** The saddle is strict (Hessian has a negative eigenvalue of order
  -s_eff) exactly when the escape needs k = 2 cores; for k >= 3 it is a higher-order saddle (no
  negative curvature in the escape direction), which is why the escape is polynomially slow.
- **[hypothesis] H4 (regime map).** Overall init scale sigma (per core) is the main knob. The
  function-space init scale is ~sigma^K, so deep or large trees are "small init" already at
  moderate sigma: the saddle-to-saddle boundary sigma_c(K) moves toward 1 as K grows. A lazy regime
  needs a centred or antisymmetric construction (W_0 = 0 with large cores) or alpha-scaling. Large
  bond dimension in NTK parameterization gives lazy dynamics; a muP-type parameterization keeps
  feature learning. Imbalance between cores (e.g. root vs leaves) is a second knob (as in Kunin et
  al. 2024).
- **[hypothesis] H5 (input covariance).** With a product covariance Sigma = (x)_i Sigma_i the loss is a
  Sigma-weighted Frobenius norm. When the teacher is aligned with the Sigma eigenbases this mostly
  rescales effective teacher weights and per-core speeds (Saxe-like reordering), and is not a
  qualitatively new regime. Cross-site correlations break the product structure and might matter
  more.
- **[hypothesis] H6 (learning rate).** GD with small step tracks gradient flow; the stability limit is
  set by the sharpness at the minimum (which grows with core norms and K); near that limit expect
  edge-of-stability oscillation and a bias toward balanced, flatter minima, but no change in the
  saddle sequence at small init.

## 9. Things the literature review must check

1. Is saddle-to-saddle in TTN / MPS / tensor-train / HT factorization already characterized
   (sequence of saddles, plateau-time scaling vs init, vs tree shape)? How far does Razin et al.
   2022 go?
2. Are H1 (init-dependent order for tensorial teachers) and H2 (locality-dependent escape exponents)
   known, even in other models (CP decomposition, deep matrix factorization, polynomial networks)?
3. NTK / lazy regime and infinite-bond-dimension limits for tensor networks (e.g. tensor network
   Gaussian processes, NTK of MPS).
4. Initialization schemes and vanishing/exploding norms for TN learning.
5. Recent (2024-2026) unifying theories of saddle-to-saddle dynamics.
6. Conservation laws and balancedness for TN gradient flow.
7. Landscape results for TT/HT approximation (spurious minima, strict saddles).
