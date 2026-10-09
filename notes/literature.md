# Literature review: training dynamics of tree tensor networks

Status: v1, 2026-10-09, built from seven verified literature sweeps (section 1). Where this file and
`notes/prior_knowledge.md` disagree, this file wins; section 4 lists the corrections. Claim tags follow
CLAUDE.md: **proven** (a theorem in the cited paper), **empirical**, **conjecture** (labelled so by the
authors), **derived** (a calculation). **Derived (review)** marks a calculation made during this review by
a sweep or verification agent; it is in no paper, so re-check it before relying on it. Citation keys
[AuthorYear] are resolved in section 7.

Notation as in `notes/model.md`: N sites, K cores, bond dimension chi, site dimension d, per-core init
scale sigma, alpha = sigma^K, teacher W*, and `g(theta) = <W(theta), W*>`, the "neural correlation
function" (NCF) of the whitened population loss. Papers normalize initializations differently: iid
N(0, sigma^2) core entries give `E||W_0||_F^2 = sigma^(2K) chi^(K-1) d^N` on a tree with uniform chi,
whereas the project's fan-in init gives `prod_v sigma_v^2` (derived (review)). Convert scales before
comparing thresholds across papers.

## 1. Scope and method

### 1.1 Clusters

1. Deep linear networks (DLNs): exact dynamics, incremental learning, saddle-to-saddle, conservation laws.
2. Diagonal linear networks, kernel vs rich regimes, lazy training, unbalanced initializations.
3. Mean-field limits, muP / abc-parameterizations, infinite width and depth, applied to multiplicative and
   tensor-network architectures.
4. Implicit regularization, dynamics and landscapes of tensor factorization (CP, Tucker, hierarchical
   Tucker, tensor train / ring).
5. Tensor networks for machine learning: architectures, training practice, initialization, barren plateaus.
6. Numerical analysis of hierarchical formats: manifolds, gauge, canonical forms, dynamical low-rank
   approximation.
7. Saddle-to-saddle dynamics in nonlinear networks; SGD noise at saddles; learning-rate regimes (edge of
   stability, catapult, large-step balancing).

### 1.2 Method and limits

- Each cluster was first swept (web search, then reading primary sources) and then adversarially
  verified: a second agent re-fetched each paper's arXiv / proceedings / journal page, re-read the cited
  theorem statements, checked venues (PMLR, JMLR and NeurIPS indexes, Crossref), corrected errors and
  added missed papers. All papers in the sweeps exist with the titles and authors given; none was dropped.
- Per-paper confidence (section 7): "read" = theorem statements or results checked in the PDF/HTML;
  "abstract" = checked against the abstract only; "unconfirmed" = listed separately.
- The shared web-search budget (200 queries) ran out during the sweeps. Verifiers then worked only by
  fetching known URLs, so searches for counter-evidence were limited. "Open" below means "not found in
  about 200 searches plus targeted checks"; coverage of 2025-2026 work depends on the sweeps. This
  synthesis step ran no searches; it fetched 24 known arXiv/PMLR pages, only to complete bibliographic
  entries, resolve conflicting venue claims and pin theorem numbers of [KumarHaupt2025b].
- Venues that could not be re-verified are marked in section 7.

### 1.3 Calculations made during the review (not in any paper)

Used throughout and tagged "derived (review)":

1. [KumarHaupt2025a] applies verbatim to TTNs with K >= 3 (section 2.1).
2. For chi >= 2 every KKT point of g on the sphere is degenerate along an orthogonal-gauge orbit, so the
   post-escape theorems of [KumarHaupt2025b] need a quotient extension (section 2.1).
3. For independent zero-mean cores, `E[W (x) W]` and the expected tangent kernel are multiples of the
   identity on any tree; for Gaussian cores the single-input output kurtosis is `3 prod_e (1 + 2/chi_e)`
   (section 2.3).
4. In the project's fan-in init, `E[Q_v] = (sigma_v^2 - sigma_p^2 chi_p/chi_v) I`: internal bonds of
   equal chi are balanced in expectation, bonds into the root are child-heavy (section 2.1).
5. On the unit parameter sphere `g <= K^(-K/2) ||W*||_inj`, attained at balanced bond-rank-one
   configurations (section 3, H1).
6. Single-core Hessian blocks are PSD, so every TTN saddle needs >= 2 cores to descend; a rank-deficient
   core of 2 nodes gives a strict saddle (section 3, H3).
7. The lambda = 0 "spurious local minimum" of [FrandsenGe2022] is a degenerate saddle of order 6
   (section 2.4).
8. The per-core balanced gauge is not Vidal's canonical gauge (section 2.6).
9. Smaller ones are flagged in place (homogeneity rescalings, Gissin's per-layer threshold, covariance
   absorption, the induced Riemannian metric).

### 1.4 Main conclusions

1. No paper characterizes saddle-to-saddle dynamics of gradient flow (GF) on dense-core TTNs: neither the
   sequence of saddles nor how plateaus scale with init or tree shape. The closest works are [Furman2026]
   (the project's exact model and loss; landscape only) and [Razin2022] (CP-structured nodes; per-node
   growth exponent; low-rank bias conditional on an unproven alignment inequality).
2. H1 is partly known. Its near-origin half is a corollary of [KumarHaupt2025a]. Init-dependent learning
   order, with the init overlap entering with power p-2, is proven in analogues (tensor PCA, two-layer nets,
   quadratic units), but over-parameterization tends to restore largest-first order [Ge2021; Ren2025], so
   H1 should be stated as a function of the teacher gap and of chi.
3. H2 and H3 are partly known as landscape statements: descent order <= number of nodes of the
   rank-deficient core [Furman2026], Tucker examples [FrandsenGe2022], the DLN strict/non-strict
   classification [Achour2024]. The dynamical, topology-dependent versions are open.
4. Even for DLNs only the first escape is proven for generic init ([Jacot2021], arXiv only); at depth >= 3
   the full saddle sequence is proven only for decoupled DLNs [Saxe2014] and deep diagonal networks
   [Zhao2026].
5. The lazy limit of a TTN with whitened inputs has no ordering and no dependence on tree shape (isotropic
   expected tangent kernel, derived (review)) and needs chi >> K on every bond. No muP, mean-field or DMFT
   theory exists for TTNs.
6. Learning rate and SGD noise can change the late saddle sequence (edge-of-stability re-balancing,
   stochastic collapse, noise equilibrium along gauge orbits), contrary to the last clause of H6.

## 2. Themes

### 2.1 Deep linear networks

**Known.**

- Exact solutions [Saxe2014] (whitened inputs, square loss, continuous-time GD, layers initialized
  "decoupled" along the SVD of the input-output correlation). Each mode obeys `tau a' = b(s - ab)`,
  `tau b' = a(s - ab)` with `a^2 - b^2` conserved; a balanced mode u = ab follows
  `u(t) = s e^(2st/tau) / (e^(2st/tau) - 1 + s/u0)` (proven). With N_l - 1 weight matrices a balanced mode
  obeys `tau u' = (N_l - 1) u^(2 - 2/(N_l - 1)) (s - u)` (eq. 15). Integrating it gives the escape time
  `~ tau a0^-(D-2) / ((D-2) s)` for D matrices; the paper does not state this. The analytic curves track
  simulations from small random init (empirical). On MNIST, random orthogonal init gives depth-independent
  learning times and scaled Gaussian init does not (empirical). [Saxe2019]: hierarchically structured data
  are learned coarse to fine, in singular-value order.
- Conservation laws. Under GF `W_(j+1)^T W_(j+1) - W_j W_j^T` is constant ([Arora2018]; [DuHuLee2018]
  Thm 2.2, which needs only the two adjacent layers to be linear; their Thm 2.1 gives per-neuron scalar laws
  for homogeneous activations). With balanced init the product obeys
  `dW/dt = -sum_j [W W^T]^((j-1)/N) grad l(W) [W^T W]^((N-j)/N)` [Arora2018] (proven). The known laws are
  proven complete only for two-layer linear nets of full rank and two-layer bias-free ReLU nets; depth > 2
  is open [Marcotte2023].
- Incremental learning at depth 2 [Gidel2019] (aligned init, approximately commuting Sigma_x and
  Sigma_xy; GF and discrete GD): in time rescaled by log(1/init), mode i switches on as a step at
  T_i = 1/sigma_i, and between steps the product is the minimum-norm reduced-rank regression solution
  (proven).
- Depth sharpens incremental learning. [Gissin2020] (toy model sigma_i = w_i^N, the same init for every
  coordinate, depth-normalized loss): `sigma_i' = sigma_i^(2-2/N) (sigma*_i - sigma_i)`; the largest init
  that still separates two modes with ratio r = sigma*_i/sigma*_j is exponentially small in 1/(r-1) at
  N = 2 but only polynomially small (exponent N/(N-2)) at N >= 3 (Thm 2, proven). [Arora2019] (deep matrix
  factorization): signed singular values obey `sigma_r' = -N (sigma_r^2)^(1-1/N) <grad l(W), u_r v_r^T>`
  (proven); depth 3-4 beats the minimum-nuclear-norm solution (empirical); the authors only hypothesise that
  no norm captures the bias. [RazinCohen2020] prove that in some completion problems all norms and
  quasi-norms diverge.
- Greedy low-rank learning [LiLuoLyu2021] (symmetric PSD factorization, C^3 convex loss, infinitesimal
  init): the first phase equals the GLRL trajectory (Thm 5.6); later phases follow GLRL only under an
  alignment assumption (Thm 5.11); for depth L >= 3 from init alpha I only the first phase is proven, with
  time shift `alpha^-(1-1/P) / (2 mu_1 (P-1))`, P = L/2, i.e. `a0^-(L-2) / ((L-2) mu_1)` in per-layer scale
  (Thm 6.2, plus a technical assumption). [Jin2023]: for symmetric low-rank matrix sensing (depth 2), GD from
  small init passes through solutions of increasing rank until recovery; the whole sequence is proven.
- Saddle-to-saddle [Jacot2021] (arXiv only; iid Gaussian init scaled by alpha -> 0; any convex cost).
  Thm 2 proves only the path from the origin to the first rank-one saddle, with probability 1 for L <= 3 and
  >= 1/2 for L > 3 (conjectured 1). Escape time ~ -log alpha (L = 2) or alpha^-(L-2) (L > 2); optimal escape
  speed `s* = L^(-(L-2)/2) s_1`. Lemma 15: every non-optimal positive-speed escape direction has curvature
  < -s* along a perturbation of the first and last layers only; Lemma 16 excludes zero-speed directions only
  for L = 3. Thm 22 (approximately homogeneous costs; covers TTNs with k = K, m = 2K) matches escape paths
  only for speeds above (K-1)/(K+1) of the maximum. Prop 14 assumes that non-optimal escape directions are
  strict saddles. The full rank-increasing sequence is Conjecture 3; Remark 4: the limiting saddles are
  balanced; Remark 6: at L > 2 the rank-one saddle is escaped only through the Hessian null space. Thm 1
  (width scaling sigma^2 = w^-gamma) is about distances at init only; dynamics for 1 < gamma < inf are
  conjectured.
- Hence at depth >= 3 the full saddle sequence is proven only for decoupled DLNs [Saxe2014] and deep
  diagonal networks [Zhao2026] (section 2.2); for generic init, only the first escape.
- Landscape. Every local minimum is global, and for >= 2 hidden layers the set W_1 = ... = W_H = 0 (which
  contains the origin) has identically zero Hessian [Kawaguchi2016] (proven). [Achour2024] Thm 7
  (invertible Sigma_XX, distinct singular values): a critical point of rank r < r_max with index set S is a
  strict saddle if S != [1, r]; if S = [1, r] (it solves the rank-r problem) it is non-strict iff
  "tightened", i.e. every pivot has a complementary block of rank r. For H >= 3 three factors of rank r
  suffice, so 0-balanced such points are non-strict; for H = 2 all saddles are strict. Both kinds exist for
  H >= 3 (Prop 8).
- Homogeneous-model theory (used for TTNs below). [KumarHaupt2025a] Thm 1 (L-homogeneous with L > 2,
  locally Lipschitz gradient, definable; square or logistic loss; init delta w0): `||w|| <= B delta` on
  `[0, T/delta^(L-2)]`; at the end w is either aligned with a non-negative KKT point of
  `max_(||u||=1) NCF(u)` selected by w0, or has shrunk below `2 delta eps` (proven; the theorem does not say
  which KKT point, nor whether it is a maximum). [KumarHaupt2024]: the L = 2 version, also near saddles.
  [KumarHaupt2025b] (JMLR): if w0 lies in the stable set of a Delta-second-order positive KKT point w*
  (Delta > 0), GF escapes after ~delta^-(L-2) (exactly `1 / (L (L-2) N(w*) delta^(L-2))` from delta w*) and
  then tracks the escape path from the origin along w* (Thm 4 for L = 2; Thm 11 for L > 2 in arXiv v3,
  Thm 10 in earlier versions). In feed-forward nets, zero rows/columns of the KKT direction stay small until
  the next saddle (Thm 19); first-order-only KKT points are not handled (Remark 14). [KumarHaupt2025c]
  (Neuron Pursuit, arXiv): started within delta of a saddle where the dormant weights enter the output with
  lowest homogeneity L and the active weights sit at a Lojasiewicz local minimum, the dormant weights stay
  O(delta) for `T/delta^(L-2)` and align with a KKT point of the residual's NCF (Thm 8; a local lower bound
  on the plateau, not an escape time).
- Escape directions [Bantzis2026] (deep ReLU nets, ICLR 2026): in the optimal escape direction the top
  singular value of layer l exceeds the others by a factor ~l^(1/4); for linear nets escape directions are
  rank one in every layer. A sequence of increasing bottleneck rank is conjectured ([Jacot2022]).
- Lazy versus rich in two-layer linear nets. [Tu2024]:
  `dA/dt ~ -sqrt(A A^T + sigma^4 w^2 I) grad C - grad C sqrt(A^T A + sigma^4 w^2 I)`; singular values below
  sigma^2 w move lazily, those above actively; in the wide regime (w = d^gw with gw > 1, sigma^2 = d^gs)
  training is lazy and fails to recover a low-rank target if gs + gw > 1, and active if gs + gw < 1.
  [Domine2025] (whitened, `W_2^T W_2 - W_1 W_1^T = lambda I`, hidden width min(N_i, N_o)): exact solutions;
  lambda -> 0 gives sigmoidal (rich) learning and |lambda| -> inf exponential (lazy) learning; funnel nets
  are lazy as lambda -> +inf and inverted funnels as lambda -> -inf, the opposite limits giving
  lazy-then-delayed-rich behaviour. [Kunin2024]: the conserved `delta = eta_w a^2 - eta_a ||w||^2` (layer
  learning rates included); in linear nets balanced init becomes rich fastest, upstream init (delta >> 0) is
  lazy, downstream is delayed rich. [Braun2022]: exact solutions from zero-balanced inits; large
  task-independent inits make learning exponential instead of sigmoidal. [Min2021]: the convergence rate is
  set by imbalance and margin. [DuHu2019]: widths Omega~(L r d_out kappa^3) give linear convergence from
  Gaussian init. [Shamir2019]: scalar deep chains with standard random init need exponentially many GD steps
  in depth. [ZiyinLiMeng2022]: with weight decay and more than one hidden layer, zero can be a bad minimum.
- Pathways [Saxe2022]: gated DLNs reduce to a race between pathways, so architecture sets what is learned
  and how fast.

**Setting.** Square loss; whitened or commuting covariances for exact solutions; two layers for lazy/rich
theory; at depth >= 3, rigorous results cover the first escape (generic init) or decoupled/diagonal models.

**Carries over to TTNs?**

- Directly. A chain TTN with physical legs only at its two ends is a DLN. The per-bond conservation law is
  proven for TTNs ([Furman2026] Prop F.12; section 2.6). [KumarHaupt2025a] Thm 1 applies verbatim: a K-core
  TTN is a polynomial in its cores (definable, locally Lipschitz gradient), and the whitened population loss
  is a finite-sum square loss over the d^N basis tensors with labels W*_i, whose NCF is g (derived
  (review)). So for K >= 3, within time ~`||theta_0||^-(K-2)` the direction converges to a non-negative KKT
  point of g on the sphere, or the weights shrink; K = 2 is covered by [KumarHaupt2024].
- With a caveat. [KumarHaupt2025b] needs Delta > 0. For chi >= 2 the orthogonal bond gauge
  (`G_v -> G_v O`, `G_p -> O^T G_p`) preserves g and ||theta||, so the Lagrangian Hessian vanishes along
  the gauge orbit of every KKT point and Delta = 0 (derived (review); the same holds for DLNs of width
  >= 2). The post-escape theorems apply directly only to chi = 1, or after a quotient (Morse-Bott)
  extension.
- By analogy. An aligned, balanced TTN component should follow the depth-K mode ODE (the project's aligned
  init is built for this). The rank criterion of [Achour2024] suggests a tree version for H3; [Tu2024],
  [Domine2025] and [Kunin2024] suggest per-bond lazy/rich thresholds set by the bond Gram matrices.
- Not. Jacot's Lemmas 15/16 are DLN-specific (modes are connected by first- and last-layer perturbations,
  which makes the first escape greedy); Arora's closed end-to-end flow has no known TTN analogue.
- Fan-in init imbalance (derived (review)). For the bond between child v and parent p,
  `E[Q_v] = (sigma_v^2 - sigma_p^2 chi_p / chi_v) I` (Q_v as in model.md section 5). With a single sigma,
  internal bonds of equal chi are balanced in expectation (fluctuations only), while each bond into the root
  (chi_p = 1) has `E[Q_v] = sigma^2 (1 - 1/chi_v) I`: the child side is heavier. In [Domine2025] and
  [Kunin2024] the sign of such an imbalance relative to the funnel shape decides between lazy and
  lazy-then-rich behaviour; it matters only when sigma^2 is not small compared with the teacher scale.

### 2.2 Diagonal networks; kernel versus rich regimes

**Known.**

- [Woodworth2020] (`f = <w_+^D - w_-^D, x>` with w_+ = w_- = alpha w0 at init, so f = 0 at any scale;
  underdetermined least squares; GF). D = 2: the limit minimizes `Q_alpha(beta) = alpha^2 sum_i
  q(beta_i/alpha^2)` with `q(z) = 2 - sqrt(4 + z^2) + z arcsinh(z/2)`; alpha -> inf gives minimum l2
  (kernel), alpha -> 0 minimum l1 (rich), with `Q_alpha / log(1/alpha^2) -> ||beta||_1` (Thm 1). Near-l1
  needs exponentially small alpha, near-l2 only polynomially large alpha (Thm 2). D >= 3 (Thm 3): Q depends
  on alpha^D, the limits are still l1 and l2 (not l_(2/D)), and only polynomially small alpha is needed for
  the rich limit. The init shape w0 changes only the kernel limit. For width-k factorizations UV^T the
  control parameter is the lifted scale alpha^2 k (empirical transition at alpha^2 k ~ 1 in matrix
  completion). Starting exactly at w = 0, a saddle that GF never leaves, is avoided by the two-copy
  construction.
- [Chizat2019]: objective `R(alpha h(w)) / alpha^2`. The criterion
  `kappa = ||h(w0) - y*|| ||D^2 h|| / ||Dh||^2 << 1` is sufficient only. For q-homogeneous h, scaling the
  init gives laziness only if the initial output stays bounded. With h(w0) = 0 (symmetrized init):
  `||w - w0|| = O(1/alpha)` (Thms 2.2, 2.4); underparameterized lazy training converges to a local minimum
  (Thm 2.5). Two-layer nets with per-neuron scale alpha(m) are lazy if `m alpha(m) -> inf` (alpha = 1/m is
  mean field). Lazy CNNs generalize worse (empirical).
- NTK: [Jacot2018] the kernel is constant at infinite width; [Lee2019] linearization error O(n^-1/2) for
  `eta < 2 / (lambda_min + lambda_max)`; empirically the relative weight change is ~n^-1/2 in input/output
  layers and ~1/n in intermediate layers.
- [Geiger2020] (centred model `alpha [f(w) - f(w0)]`, loss / alpha^2): crossover alpha* ~ h^-1/2; lazy
  `||dw|| ~ 1/alpha`; in the feature regime the dynamics are linear for t << t_1 ~ sqrt(h) alpha and
  `||Delta Theta|| ~ (sqrt(h) alpha)^-a` with a < 2 (a -> 2 with depth); without centring, training did not
  converge for alpha >> 1.
- [Ghorbani2019] (Gaussian inputs, quadratic target): the NT model fits f* only along a random
  N-dimensional subspace when N < d; the trained network (quadratic activation) fits the top N
  eigendirections and beats NT (proven for N <= d).
- Saddle-to-saddle in diagonal nets. [PesmeFlammarion2023] (`beta = u (.) v`, init (sqrt(2) alpha, 0),
  data in general position): in time `ln(1/alpha) t` the limit is a jump process given by an explicit
  algorithm, identified with the adaptive inverse scale space method: LARS-like but not the Lasso path,
  history-dependent, coordinates can deactivate, at most `min(2^d, sum_(k<=n) C(d,k))` steps, ending at the
  minimum-l1 interpolator (Thm 2, Prop 3). Under 2r-RIP and a gap, jump i happens at
  ~ln(1/alpha)/|beta*_i| (Prop 2). Following the most-negative-curvature direction would give OMP, not this
  path. Weighted init exponents and depth D appear only as remarks. [Berthier2023]: Lotka-Volterra form;
  init exponents k_i weight a lasso path (nonnegative, anti-correlated features, n >= d). [Zhao2026] (ICML
  2026; deep diagonal nets with L >= 3 and general two-layer diagonal models; generic unbalanced init
  direction, scale s -> 0): GF passes near every iterate of a LARS-like algorithm whose growth times
  `t_i^(+-) = +-F_i(+-inf)` depend on the init shape, and the implicit bias is a weighted l1 norm with
  init-shape-dependent weights (Thm 4.5, Cor 4.6). [Kunin2025] (AGF): dormant neurons follow
  `d theta_bar/dt = eta ||theta||^(kappa-2) P_perp grad U` and `d||theta||/dt = eta kappa ||theta||^(kappa-1) U`,
  where U is the utility (correlation with the residual of the active neurons) and kappa its leading degree;
  neuron i activates when `int kappa U_i ds` reaches c_i/eta with `c_i = -log||theta_i(0)||` (kappa = 2) or
  `(||theta_i(0)||^(2-kappa) - 1) / (kappa - 2)` (kappa > 2). Proven only for diagonal linear nets (Thm 3.1:
  converges to Pesme-Flammarion's algorithm as alpha -> 0); conjectured for fully connected linear nets.
- Init shape and imbalance. [Azulay2021] (untied diagonal nets): the regime is set by `alpha / (1 - s^2)`
  with s the init shape; |s| -> 1 gives kernel behaviour even at tiny alpha; balanced s = 0 performed best;
  multi-neuron linear nets with infinitesimal init converge to minimum l2 ("anti-NTK"). [XuZiyin2025]: exact
  one-hidden-layer solutions; three mechanisms (alignment, disalignment, rescaling); diagnostic
  `zeta = u^T w / (||u|| ||w||)`. From [Gissin2020] Thm 2 the per-layer init threshold `(sigma_0^th)^(1/N)`
  tends to 1 (relative to the per-layer target scale) as N grows (derived (review)).
- Silent alignment [Atanasov2022] (small init, whitened data): the tangent kernel aligns while small and
  then only grows in scale; deep linear: `c = u^L`, `dc/dt = c^(2-2/L) (s - c)`,
  `t_(1/2) ~ L sigma^-(L-2) / (s (L-2))`; non-whitened data: early alignment to `Sigma beta`, then
  realignment.
- Steps and noise. [Pesme2021]: SGD noise acts as a smaller effective init
  `alpha_inf = alpha (.) exp(-2 gamma diag(X^T X / n) int L dt)`, i.e. closer to l1. [Even2023]: large steps
  give `alpha_inf^2 = alpha^2 (.) exp(-sum_k q(gamma_k grad L_(B_k)))`; for GD the gain concentrates on the
  support (hinders sparse recovery), for SGD it is homogeneous (helps); the edge-of-stability analysis is
  heuristic. [Nacson2022]: the step size can turn kernel-type bias into sparsity-inducing bias where GF
  would stay kernel. [Lewkowycz2020] (two-layer uv model): lazy for `eta < 2/lambda_0`, catapult (loss
  spike, curvature falls below 2/eta) for `2/lambda_0 < eta < c_act/lambda_0` (c_act = 4 in the model,
  about 12 empirically for ReLU nets), divergent above. [Moroshko2020] (classification, exponential loss):
  kernel vs rich is set jointly by init scale and training accuracy, boundary `eps = exp(-Theta(alpha^D))`.

**Setting.** Diagonal or width-1 models (commuting structure, mirror-descent derivations), sparse
regression, square loss; the imbalance results are two-layer.

**Carries over to TTNs?**

- Directly: Chizat's scaling framework (a per-core scale lambda acts as an output scale alpha = lambda^K).
  With plain iid cores kappa does not decay as lambda grows, because ||W_0|| grows like lambda^K; laziness
  needs centring (the library's `ShiftedTeacher`), the two-copy model, or width-type cancellations. The
  diagnostics (relative parameter change, kernel change, linearization error) transfer per core.
- By analogy: as sigma -> 0 expect a jump process driven by cumulative clocks (Pesme-Flammarion's s_t,
  AGF's S_i) rather than by curvature at the current saddle, so plateau lengths depend on history. For
  kappa > 2 (TTN escapes in which >= 3 cores must grow together) AGF predicts that init-norm ratios enter
  with power kappa - 2 and survive the sigma -> 0 limit (H1); [Zhao2026] is the rigorous depth >= 3
  instance. The function-space scale sigma^K is the right variable for width-1 models ([Woodworth2020]
  Thm 3; [Gissin2020]); once chi enters, lifted Gram scales govern (alpha^2 k; sigma^2 w).
- Not: the mirror-flow / Q-potential derivations need diagonal or commuting structure. With a fully
  observed population loss and representable W* every regime ends at W*; regimes differ in the trajectory,
  and in the final loss only when chi is too small (a lazy finite-chi TTN fits only the projection of W*
  onto the tangent space at W_0: the argument of [Ghorbani2019], [Chizat2019] Thm 2.5, [Wang2020]).

### 2.3 Mean field, muP, infinite width and bond dimension

**Known.**

- Two-layer mean field [MeiMontanariNguyen2018; ChizatBach2018; RotskoffVE2018;
  SirignanoSpiliopoulos2020]: the neuron distribution follows a nonlinear PDE; global convergence holds under
  conditions (Chizat-Bach assume 2-homogeneous or partially 1-homogeneous units, per the verifier's
  recollection, not re-read).
- Depth. [NguyenPham2023]: iid-initialized multilayer mean-field limits degenerate at depth >= 4; global
  convergence for 2-3 layers. [Chizat2024] (three-layer single-output linear net in muP, GD on the expected
  loss): converges to GD on a deterministic infinite-width network; in continuous time the predictor
  converges exponentially to the minimum-l2 risk minimizer. That muP at O(1) init has no plateaus is an
  inference, not a statement of the paper.
- Tensor Programs. [YangHu2021] (TP IV): stability and nontriviality conditions for abc-parameterizations;
  feature learning iff r = 0, kernel regime iff r > 0. This dichotomy needs Assumption 3.1 (tanh or
  sigma-gelu) and the paper says it fails for linear activations; feature-learning parameterizations have
  f_0 -> 0. [YangLittwin2021] (TP IIb): kernel training dynamics for any Tensor Program in NTK
  parameterization. [Yang2021] (TP V): muTransfer. [YangSB2023] (preprint): spectral condition
  `||W||_spec, ||Delta W||_spec = Theta(sqrt(fan_out / fan_in))`; its Claim 1 (a random init scales vectors
  by its spectral norm) holds only when fan-out >= fan-in; init
  `sigma = Theta(n_in^(-1/2) min{1, sqrt(n_out / n_in)})`. [Yang2024] (TP VI) and [Bordelon2024a]:
  depth-muP; both need scaled residual branches. [BordelonPehlevan2022] (DMFT): for a two-layer linear net
  with whitened data, `dDelta/dt = -2 sqrt(1 + gamma_0^2 (y - Delta)^2) Delta` (NTK-like as gamma_0 -> 0,
  logistic as gamma_0 -> inf); the relative NTK change is O(gamma_0^2 L) in deep linear nets
  (perturbative). [Bordelon2024b]: a stable feature-learning limit of attention needs 1/N logits.
  [Atanasov2025]: with muP output divided by gamma, the optimal LR scales as gamma^2 (gamma << 1) and as
  gamma^(2/L) (gamma >> 1); in the "ultra-rich" regime loss curves show a long plateau and then a drop,
  sometimes staircases, with silent alignment and sharpness rising to ~2/eta at the drop (empirical, plus a
  linear toy model).
- Multiplicative and structured models. [Littwin2020] (hypernetworks): the dynamics stay nonlinear unless
  both widths grow; the NNGP kernel is a Hadamard product. [Choraria2022] (polynomial Pi-nets): one
  multiplicative interaction gives a product of kernels. [Vankadara2024] (SSMs incl. Mamba): they are not
  Tensor Programs; standard muP fails; a scaling derived from signal propagation restores feature learning.
  [Qiu2024] (TT and block-TT layers) and [Potapczynski2024] (general Einsum layers): muP-style init and LR
  rules using the min{} init of [YangSB2023]; heuristic, validated empirically. [HaninNica2020]: when depth
  and width grow together the NTK stays random and moves; [HaninNica2020b]: log-norms of products of many
  large random matrices are asymptotically Gaussian; [LiNicaRoy2021]: ResNets are log-Gaussian at a fixed
  depth/width ratio.
- Tensor networks. [GuoDraper2021b] (periodic MPS, entries `N(0, sigma_i^2 / sqrt(|a_i| |a_(i+1)|))`,
  per-element LR `(|a_i| |a_(i+1)|)^(-1/2)`, bond dimensions -> inf one after another; informal proofs): the
  NTK converges to `sum_k phi(x_k).phi(x'_k) prod_(l != k) sigma_l^2 phi(x_l).phi(x'_l)` (K times the
  product kernel when sigma = 1), is positive definite, and stays constant during training under a
  boundedness assumption (lazy). [GuoDraper2021a]: a GP limit for MPS; internally inconsistent, and its
  claim of a GP limit as n -> inf at fixed chi is contradicted by the kurtosis below. [WeselBatselier2025]:
  CPD- and TT-constrained kernel machines converge to the same product-kernel GP (TT ranks -> inf in
  sequence; simultaneous growth via deep linear nets, Cor B.8); stated rates O((MD/P)^(1/2)) for CPD and
  O((MD/P)^((D-1)/4)) for TT. The TT rate treats the R^(D-1) shared-core paths as independent; the exact
  excess kurtosis is `3 [(1 + 2/R)^(D-1) - 1] ~ 6 (D-1) / R` (derived (review)). [WangLiuYu2026]
  (Haar-isometric MPS, normalized expectation values; moment method; no training): GP limits with explicit
  moment bounds; for global observables Gaussianity needs D >> n, and at large d with fixed D global outputs
  stay non-Gaussian. [Yarotsky2026] (ICML 2026; GF on a rank-H CP model of an order-nu tensor with p entries
  per mode, iid N(0, sigma^2) init, fully observed square loss): a diagram expansion of the loss gives
  regimes: free evolution (H ~ p^(nu-1), p^(nu-1) sigma^nu -> inf), NTK (asymmetric model;
  `p^(nu-1) H sigma^(2 nu) ~ 1`, `H sigma^nu -> inf`), mean field (`H sigma^nu ~ 1`, `H / p^(nu-1) -> inf`)
  and under-parameterized rich (`p^(nu-1) sigma^nu ~ 1`, `H / p^(nu-1) -> 0`). Prop 8.1: the initial NTK
  tends to the identity (asymmetric model); Prop 8.2 (symmetric, nu = 2): a stable NTK is equivalent to no
  learning. The authors call the framework not yet mathematically complete.
- Derived (review), for any tree with independent zero-mean cores of entry variance s_v^2:
  `E[W (x) W] = (prod_v s_v^2) (prod_e chi_e) I` and
  `E[J J^T] = (sum_v prod_(u != v) s_u^2) (prod_e chi_e) I`, exactly. For Gaussian cores and one input the
  output kurtosis is exactly `3 prod_e (1 + 2/chi_e)`, independent of tree shape and variances and dominated
  by the narrowest bonds.

**Setting.** Infinite width with additive (neuron) structure; the tensor-network results cover MPS, TT and
CP only and are mostly informal or concern the prior at init.

**Carries over to TTNs?**

- By derivation: the expected tangent kernel is isotropic on any tree, so in the lazy limit with whitened
  inputs `W(t) - W* = e^(-ct) (W_0 - W*)`: every component decays at the same rate, with no ordering and no
  dependence on tree shape (derived (review); consistent with [Yarotsky2026] Prop 8.1 for CP and with
  [GuoDraper2021b] for MPS). The kurtosis formula says a Gaussian/NTK description needs chi >> K on every
  bond; a few narrow bonds should keep the dynamics nonlinear ([Littwin2020], [HaninNica2020], by analogy).
- Not: no muP, mean-field or DMFT limit has been derived for MPS, TT or TTNs used as the model. TP IV's
  dichotomy excludes multilinear models. A TTN core with two or more bond children computes T(u (x) v), a
  chi x chi^2 contraction rather than an n x n matrix multiply, so TTNs are probably not standard Tensor
  Programs (sweep inference, unchecked). The update clause of [YangSB2023] carries over (single-example core
  updates are rank one) but its init clause does not: internal cores have fan-in chi^c > fan-out chi, and
  its init would shrink signals by chi^(-(c-1)/2) per core (derived (review)); the project's fan-in init
  preserves forward norms instead.
- By analogy: the "ultra-rich" gamma of [Atanasov2025] is small init. For an L-homogeneous f, GD with LR
  eta on f/gamma equals GD with LR `eta gamma^(-2/L)` on f from an init scaled by `gamma^(-1/L)` (derived
  (review)); for a K-core TTN gamma corresponds to sigma^-K, and their eta* ~ gamma^(2/K) is a fixed
  learning rate in small-init variables.

### 2.4 Tensor factorization dynamics

**Known.**

- CP. [Razin2021] (tensor completion, order N >= 3, GF from alpha a_r^n): squared-norm differences within a
  component are conserved (Lemma 1); with zero unbalancedness `d sigma_r / dt = N gamma_r sigma_r^(2-2/N)`,
  with gamma_r the alignment of -grad L with the normalized component (Cor 1). Thm 2 proves only the first
  stage: under a Huber loss, zero unbalancedness, a leading-component condition with exponent 1/(N-2) and
  `alpha <= eps D^-1 exp(-O(D^2 T))`, the trajectory tracks a balanced rank-one trajectory. Later stages are
  empirical.
- Hierarchical. [Razin2022] (any mode tree; completion and general losses; GF from small init): each
  internal node is a local CP factorization, a shared (copy) index contracted with a weight matrix W^(nu),
  i.e. CP-structured rather than dense cores; the local component (nu, r) is row r of W^(nu) with column r
  of each child's matrix. Lemma 2: scalar conserved differences within a local component. Thm 1 (zero
  unbalancedness; approximate version in App. B):
  `d sigma^(nu,r) / dt = L_nu (sigma^(nu,r))^(2 - 2/L_nu) <-grad L, C^(nu,r)>`, with L_nu = (number of
  children) + 1, i.e. 3 for binary trees. Eq. 9: trailing local components stay small as alpha -> 0 (needs
  L_nu >= 3), but only assuming an alignment inequality (Eq. 8) that the authors leave to future work.
  Prop 1: small trailing components imply closeness to low hierarchical rank. Experiments: local-component
  norms switch on one after another. No plateau times, ordering, escape directions or dependence on topology
  beyond the per-node exponent.
- Dense cores. [Hariz2024] (deep Tucker: dense core, factor columns premultiplied by matrix chains;
  completion; GF from balanced init): `||w_(r_m)^m||^2 - ||G_(slice r_m)||^2` is conserved (Lemma 4.1, the
  diagonal of the per-bond Gram law); slice-norm dynamics (Thm 4.3); for plain Tucker a slice norm's
  derivative does not depend on its own norm (Cor 4.5), so per-slice ODEs do not close; the experiments use
  Adam. [Hariz2022] (deep CP): `||(x)_n w_r^n||` grows with exponent `2 - 2/N + sum_n k_n / N` (Thm 3.2),
  which is the depth-D law with D = N + sum_n k_n under exact balance (derived (review)).
- Empirical TT/Tucker. [Milanesi2021] (GD with lr 0.1 on completion with full-rank TT and deep Tucker
  factorizations; small Gaussian init): low effective rank, but not minimal nuclear norm; unfolding singular
  values show a long plateau and then an abrupt rise; smaller init recovers better; deeper Tucker gives
  sparser spectra and more abrupt transitions. The random targets have comparable components, so no ordering
  is visible.
- Over-parameterized CP and lazy capacity. [Ge2021] (4th-order odeco target; 2-homogeneous normalized
  model `sum_i w_i^((x)4) / ||w_i||^2` with m = poly(d) components): a modified GF (re-initialization of
  small components, l2 penalty, epochs with decreasing thresholds) provably follows deflation, largest
  components first; for vanilla GF largest-first is empirical (Fig. 1: 5 target components with ratio 1.2,
  m = 50, init norm 1e-15); discovery follows tensor-power dynamics in which winners maximize roughly
  a_i x (initial overlap)^2; non-orthogonal targets can split (App. E). [Wang2020]: lazy training of an
  order-l, rank-r tensor needs Omega(d^(l-1)) components, a GD variant succeeds with `O*(r^(2.5 l) log d)`.
  [Arvanitakis2025]: ALS with over-parameterized rank (content unconfirmed; section 7).
- Odeco tensors. [Anandkumar2014] (symmetric order-3 odeco tensors, lambda_i > 0): the robust
  eigenvectors of the normalized power map are exactly the components (Thm 4.1); isolated local maximizers
  of T(u,u,u) on the ball iff u = v_i (Thm 4.2); the start decides the limit: convergence to
  argmax_i |lambda_i <v_i, theta_0>| (Lemma 5.1); non-robust eigenvectors proportional to
  sum_(i in S) v_i / lambda_i also exist.
- Tensor PCA (learning order). [BenArous2024a] (online SGD on the Stiefel manifold, order p >= 3, r
  orthogonal spikes): sequential elimination; the order is a greedy maximum selection on
  `lambda_i lambda_j m_ij(0)^(p-2)` (Thm 1.5); signal-to-noise order is guaranteed only if
  `lambda_i > C (eta sqrt(log eta))^(p-2) lambda_(i+1)` (Thm 1.3, probability >= 1 - 1/eta); for p = 2,
  separated SNRs are recovered in order and equal SNRs only as a subspace. [BenArous2024b] (gradient flow):
  all spikes are recovered up to a permutation fixed by initial correlations and SNRs, without separation
  assumptions. [BenArous2021] (online SGD, information exponent k): about N, N log^2 N and N^(k-1) log^2 N
  samples suffice for k = 1, 2 and >= 3, and fewer give no weak recovery.
- GD timing. [MaGuoFattahi2022] (preprint; symmetric odeco CP, each student pre-aligned with its teacher):
  within `T <~ alpha^(-(l-2)/l) / (eta l sigma_r)` GD steps the components are learned largest first.
- Landscapes. [FrandsenGe2022] (Tucker, i.e. a star TTN; fully observed): with a balancing regularizer
  lambda l^2 (l = sum of squared per-edge Gram defects) all local minima are global (Thm 1). At balanced
  points the paper exhibits descent of order eps^2, eps^3 and eps^4 when 1, 2 and 3 bonds must grow (2, 3
  and 4 cores move); moving one core gives first-order descent (Lemma 8). These orders come from exhibited
  directions, without lower-bound proofs. The paper's lambda = 0 "spurious local minimum" is a degenerate
  saddle: its overlap with the target is a product of four perturbations, and the best curve gives a
  decrease ~ -eps^6 (derived (review), checked numerically); a valid non-min-norm spurious minimum is
  [Furman2026] Thm G.1. [ChenLiLu2020]: tensor rings have non-strict spurious local minima even at bond
  dimension r^(d-1). [AnandkumarGe2016]: third-order local optima are efficiently findable, fourth-order is
  NP-hard. [Ge2015]: a reformulated orthogonal decomposition has the strict-saddle property. [GeMa2017]: for
  random over-complete tensors, local maxima above (1 + eps) times the mean are approximately global.
- Degree-2 (tubal) models, low relevance: [Karnik2025] (GD from small random init; spectral then
  convergence phase), [MaLavaeiSojoudi2023] (lifted matrix sensing), [AgyeiKodie2026] (strict-saddle
  landscape; nonuniform Fourier ranks give quartically flat directions).

**Setting.** CP, Tucker and CP-node hierarchical factorizations; completion or full observation; mostly GF
from small balanced init; orthogonal (odeco) targets for the ordering results.

**Carries over to TTNs?**

- [Razin2022] is the closest dynamics theorem on trees but not for the project's model: CP-structured nodes
  have per-component scalar invariants, whereas dense cores have a GL(chi) gauge with matrix invariants, and
  [Hariz2024] Cor 4.5 shows that per-slice norm ODEs need not close for dense cores. Its qualitative message
  (the exponent is set by node degree, not depth) supports the locality of H2.
- By analogy: the rule "(teacher weight) x (init overlap)^(p-2)" is established for CP and tensor PCA; in a
  TTN, K per-core overlaps and the bond gauge enter, and the rule must be re-derived. Over-parameterization
  tends to restore largest-first order ([Ge2021], empirical), so chi may play the role of the number of
  students. In the exhibited Tucker examples the descent order equals the number of cores that must move
  (H2/H3), without lower bounds. Lazy training of a low-rank tensor needs far more capacity than rich
  training [Wang2020].

### 2.5 Tensor networks for machine learning

**Known.**

- Training practice. [Stoudenmire2016]: MPS classifier; each sweep step merges two sites, takes a gradient
  step on the merged tensor and truncates by SVD (adaptive bond dimension); not GD on all cores.
  [Novikov2017] (TT weights over all feature interactions): iid Gaussian TT init "sometimes freezes"
  training (completely on one dataset); a TT-rank-2 linear-model init, even with random coefficients,
  works; Riemannian optimization reached a 1e-4 loss about 20x faster than SGD on the cores; SGD failed on
  synthetic high-order data. [Han2018] (MPS Born machines, NLL, two-site updates with SVD, bond dimension
  grown from 2): at a fixed maximal bond dimension the NLL grows linearly in N. [Cheng2019]: a TTN Born
  machine built to respect 2D locality keeps long-range correlations (binary MNIST test NLL 94.25 at
  D_max = 50 vs 101.45 for an MPS at D_max = 100). [Liu2019]: unitary hierarchical TTN classifier trained by
  a MERA-derived algorithm. [Stoudenmire2018]: layered TTN whose layers are mostly fixed without
  supervision. [ChenBarthel2024] (TTN classifiers with CP-rank-constrained cores, Adam, complex
  N(0, 0.4^2) entries): branching ratio 4 beats 2 by about 4% (attributed to graph distance); full TTNs
  overfit; tensor dropout helps. [WallDAguanno2021]: isometric TTN classifiers trained by manifold GD.
  [Klug2026]: a natural Riemannian gradient for functional TTNs converges much better than Riemannian GD.
  [Strashko2022] (teacher-student MPS regression): U-shaped test loss in the student bond dimension, with the
  optimum growing with sample size; training on synthetic-teacher data was "much more challenging" (the
  authors blame barren plateaus).
- Initialization and norms. [Barratt2022]: iid entries of variance sigma^2 on V nodes and E edges of
  dimension D give `Var(W_ij) = D^E sigma^(2V)`; matching a target variance is impractical in floating point
  when V is comparable to the number of features; a "copy-node" init leaves a small dense subnetwork; direct
  GD earlier needed repeated restarts. [MataAli2023]: constant entries a give represented entries
  b^(N-1) a^N; partial-norm renormalization fixes the scale. [Sasaki2026] (random binary TTN, entries
  N(0, sigma_T^2 / (d_alpha d_beta))): output variance `(sigma_T^2)^(N_x - 1) / 2^(N_x)`, critical
  sigma_T = sqrt(2). [Rakhshan2020] (random TT projections):
  `Var(||f(X)||^2) <= (1/k) (3 (1 + 2/R)^(N-1) - 1) ||X||^4`.
- Barren plateaus (normalized, quantum-style outputs). [Liu2022] (MPS embedded in unitaries): global
  losses have `Pr(|dL| > eps) <= eps^-2 O(d^-n)`; local losses have n-independent gradient variance, bounded
  by O(d^-Delta) in the distance Delta between parameter and observable. [MiaoBarthel2024; BarthelMiao2025]
  (energy minimization with isometric MPS, TTNS and MERA): no barren plateaus; the gradient variance decays
  as a power law in bond dimension, with a layer factor (b eta)^tau in hierarchical networks.
  [CerveroMartin2023]: in MPS/TTN/MERA-shaped quantum circuits the variance decays exponentially with
  distance from the canonical centre; for qMPS exponentially in the qubit number, for qTTN/qMERA
  polynomially. [ZhangHK2026] (MPS energy landscapes, imaginary-time TDVP): gauge freedom acts as local
  over-parameterization; above a critical bond dimension local minima concentrate near the global one.
- Training-dynamics observations. [Tang2025] (MPS Born machine for a periodic Ising chain, n = 16): GD
  misses the interaction between sites 1 and n (NLL gap about the mutual information of X_1 and X_n, 0.33,
  persisting at r_max = 20); natural gradient or a sketch-based warm start avoids it. [MasotLlima2024]
  (L-BFGS-B on a fidelity loss): denser geometries train better; MPS training at n = 16 failed in 400
  trials even with chi = 256 >= the target's chi; larger chi lowered success rates. [Pomarico2025]
  (sweep-trained MPS classifiers): grokking coincides with an entanglement transition (volume-law-like to
  sub-volume-law) and "eigenvalue evaporation" in reduced density matrices. [Milanesi2021] (section 2.4) is
  the only plain-GD plateau observation on TT found.
- Expressivity background (no dynamics): [CohenSharirShashua2016], [Levine2018], [Khrulkov2018],
  [Glasser2019], [Alexander2023] (an entanglement-based condition for data suited to locally connected
  networks), [Harada2025] (tree structure learned by minimizing bond mutual information).

**Setting.** Mostly sweeps with SVD truncation, Riemannian or natural gradients, or Adam; NLL or fidelity
losses; isometric or normalized models.

**Carries over to TTNs?** Mostly not directly to plain GD on free cores: sweeps, SVD truncation and
Riemannian or natural-gradient updates change or remove parameter-space plateaus, so every saddle-to-saddle
claim must name the optimizer. By analogy, the locality evidence ([Tang2025], [Cheng2019],
[ChenBarthel2024], [MasotLlima2024]) is consistent with H2. The initialization results ([Barratt2022],
[MataAli2023], [Sasaki2026], [Novikov2017], [Rakhshan2020]) apply directly and justify the fan-in init; the
(1 + 2/R)^N factor means random-init function scales are broad unless chi >> K. Barren-plateau results
concern normalized outputs; for unnormalized regression the analogue is the sigma^K scale. Copy-node and
TT-rank-2 inits start near low-rank saddles (inference) and are candidates for staged dynamics at large
init scale.

### 2.6 Hierarchical tensor formats: geometry, gauge, landscape

**Known.**

- Formats and spectra. HT format [HackbuschKuhn2009]; TT-SVD [Oseledets2011]. [Grasedyck2010] (HSVD):
  root-to-leaves truncation within sqrt(2d-2) of the best (Thm 3.11; sqrt(2d-3) by Remark 3.12),
  leaves-to-root within (2 + sqrt(2)) sqrt(d) (Thm 3.22); all edge singular values from accumulated Gram
  matrices after QR orthogonalization in O(d k^4) (Lemma 4.6, Remark 4.5). [Grasedyck2024]: the Gram route
  squares singular values and loses accuracy; a stable, root-independent variant is accurate to machine
  precision. [ShiDuanVidal2006]: canonical form of tree states (Schmidt values on every edge) in
  O(n chi^4). [TindallFishman2023]: belief propagation is exact on trees (one ordered sweep); bond tensors
  in the Vidal gauge are the edge singular values. [Lubich2015]: TT unfolding singular values equal those of
  the small matrix `S_i = R_(<=i) R_(>=i+1)^T` (a stable route). [Kramer2019]: feasible squared TT spectra
  form polyhedral cones (equivalent to a quantum marginal problem).
- Manifolds. [UschmajewVandereycken2013]: full hierarchical rank iff leaf frames have full column rank and
  transfer tensors have full multilinear rank (Thm 3.7); decompositions are unique up to the GL(k_t) gauge
  (Prop 3.9); the quotient is a smooth manifold of dimension
  `sum_leaves n_t k_t + sum_interior k_t1 k_t2 k_t - sum_(t != root) k_t^2` (Thm 4.3); horizontal lifts
  give an isomorphism onto the tangent space (Cor 4.13); its closure is {rank_T <= k} (Thm 4.14), so best
  approximations exist. TT analogue: [Holtz2012] (content not read). Non-closed sets: bounded CP rank
  [deSilvaLim2008]; TN states on graphs with a cycle that has no subcritical vertex [Landsberg2011].
- Dynamical low-rank approximation. [Lubich2013] (HT/TT, Dirac-Frenkel principle); [Lubich2015] (TT
  projector splitting, robust to small singular values; standard Runge-Kutta on factor ODEs performs
  poorly); [Ceruti2021] (trees); [CerutiLubich2022] (BUG integrator); [Ceruti2023] (rank-adaptive trees;
  dissipates energy in gradient systems up to truncation); [Haegeman2016] (TDVP unifies time evolution and
  optimization; DMRG is imaginary-time evolution with an infinite step).
- Gauge and balance. [Furman2026] App. F: balanced means equal Gram matrices of the two cores matricized
  along every internal edge (Def F.9); on trees, minimum norm in the fibre <=> local minimum norm <=>
  minimum on the gauge orbit (Kempf-Ness) <=> balanced, and the min-norm points of a fibre form one
  prod_e O(r_e) orbit (Prop F.10); `Delta_e = mat_e(W_u)^T mat_e(W_u) - mat_e(W_v)^T mat_e(W_v)` is
  conserved by GF for any loss of T(theta) on any topology (Prop F.12; balanced <=> min-norm only on trees,
  Remark F.14); weight decay gives `Delta_e(t) = e^(-2 lambda t) Delta_e(0)` (Cor F.13); GF started at a
  min-norm point stays min-norm (Prop 4.2). [Acuaviva2023]: minimal canonical forms of uniform MPS/PEPS via
  geometric invariant theory; for open-boundary MPS the minimal canonical form (half-chain norm) coincides
  with Vidal's (Sec. 3.3). The per-core balanced gauge differs from Vidal's: a random 3-site MPS in Vidal's
  symmetric gauge has `||Delta_e||_F ~ 2.8` on both bonds (derived (review), numerical).
  [LindseyMenon2025] (DLN): the l2 regularizer is minimized on the balanced manifold; balancing flows
  converge exponentially. [Bah2022]: from balanced init the DLN product follows a Riemannian gradient flow on
  the fixed-rank manifold (Cor 17) and converges to a global minimizer on some rank-k manifold for almost
  all inits. [Kunin2021], [Zhao2023], [Marcotte2023]: frameworks deriving conservation laws from
  symmetries.
- Landscape of the project's model: [Furman2026] (TTN on any tree with dense cores; inputs uniform on a
  finite product set, i.e. the population square loss `||T(theta) - T*||_F^2 / (2|X|)`; realizable
  targets; no dynamics).
  - Cor 3.4: polynomial-size TTNs with bond dimensions >= 2 are not learnable by GD in polynomially many
    steps (they embed read-once formulas, including parity; statistical-query lower bound).
  - Thm 4.3 (formalized in Lean): every min-norm local minimum is global. Cor 4.4 (not formalized): from
    any non-global point a non-increasing path leads to strictly lower loss.
  - Thm 5.2 (formalized; no min-norm assumption): every non-global critical point lacks full Tucker rank.
    A target is realizable iff rank T*^(e) <= r_e on every internal edge (Prop B.1); at min-norm points
    rank mat_e(W_u) = rank mat_e(W_v) = rank T^(e) (Prop D.2).
  - Lemmas D.7-D.9 and Remark D.10: at a min-norm non-global critical point, take a connected component of
    rank-deficient edges with N nodes; along `T(t) = T + t^N U`,
    `L(t) - L = t^N <R,U> + t^(2N) ||U||^2 / 2` with `<R,U> < 0`. This bounds the saddle order by N from
    above only; the authors note that the more must be learned in one step, the higher the order and the
    harder the escape.
  - Prop 5.3: n-bit parity on a balanced binary tree (n-1 nodes, bond dimension 2), at the uniform
    rank-one point, is a saddle with `L - L_0 >= -C ||theta - theta_0||^(n/2+1)` ("order >= n/2" in their
    convention, where order m means decrease power m+1); Remark H.3 conjectures order n-2, i.e. decrease
    power n-1 = N. In Remark D.10, "order N" is the decrease power itself.
  - Thm G.1: an explicit non-min-norm spurious local minimum (3-leaf Tucker star). App. D.7: min-norm,
    realizability, square loss and tree topology are each needed (the last two only suggested).
  - App. I.3 (heuristic): near a degenerate saddle with L ~ -c s^k the escape time is polynomial in 1/s_0
    for k > 2. Saxe-type exact TTN solutions are listed as future work.
  - Affiliations: Furman and Hodgkinson, University of Melbourne; Wäldchen and Bei, Iliad. Worth checking
    for overlap with this project.
- Other landscape facts. [Levin2024]: multilinear lifts fail "2 => 1" at points with >= 3 zero factors;
  the proof shows such points have zero gradient and zero Hessian for every smooth cost; for LR^T,
  "local => local" holds iff rank L = rank R = rank(LR^T). Best rank-one approximation is NP-hard
  [HillarLim2013] and has non-global local optima (Kofidis and Regalia 2002, Ex. 4, via [Furman2026]).
  [Qin2024]: left-orthogonal TT with Riemannian GD converges linearly near the solution (gauge fixed; no
  small-init structure). [Kressner2014]: Riemannian Tucker completion. [PaezVelasco2026]: Riemannian
  structure of TN families via group actions and submersions (abstract only).

**Setting.** Numerical analysis (fixed-rank manifolds, integrators) and landscape theory; Furman's results
assume realizability, square loss, a tree and, for Thm 4.3, min-norm points.

**Carries over to TTNs?**

- Directly: [Furman2026] studies exactly the project's model and loss. Caveats: landscape only; small random
  init has an O(sigma^2) imbalance that GF conserves exactly, so the min-norm guarantees (Prop 4.2, Thm 4.3)
  hold only approximately along such trajectories, while Thm 5.2 needs no balance. Tooling also transfers:
  bond spectra by QR sweeps or BP/Vidal gauging, avoiding the Gram route for tiny singular values (the
  library's `bond_spectra` already uses QR sweeps). GF preserves neither the Vidal nor the orthogonal
  canonical form, so spectra must be recomputed at each recorded time.
- Derived (review): at full-rank points the horizontal restriction of D phi is an isomorphism onto the
  tangent space ([UschmajewVandereycken2013] Thm 4.3, Cor 4.13), so parameter GF induces
  `dW/dt = -J J^T grad l(W)`, a Riemannian gradient flow on the fixed tree-rank manifold for the
  theta-dependent metric `<xi, (J J^T)^-1 xi>`; dynamical low-rank approximation uses the Euclidean
  projector instead. No TTN paper gives this metric explicitly; the DLN analogue needs balanced init
  [Bah2022]. Saddles lie in rank-deficient strata where J J^T degenerates (a heuristic picture of plateaus,
  not a theorem).

### 2.7 Saddle-to-saddle in nonlinear models; learning-rate and noise regimes

**Known.**

- Unifying views. [ZhangSL2026] (ICLR 2026; models `f = sum_i phi(g_in(x); u_i) v_i`, covering neurons,
  convolution kernels and attention heads; small init, GF): fixed points of narrower networks embed into
  wider ones (Thm 1); GF has invariant manifolds of reduced effective width (Thm 3); saddle-to-saddle means
  moving to a manifold with one more effective unit. Units linear in their own weights give a data-induced
  separation (Thm 4: when the top-r part is O(1) the rest is `O(sigma^(1 - s_(r+1)/s_1))`); units quadratic
  in their own weights give an init-induced separation (Prop 5: the unit with the largest init grows first
  while the others are still O(sigma)); a stronger effect is conjectured for higher degree. Equal data
  singular values remove intermediate plateaus in linear nets but not in quadratic-unit nets. Whether the
  saddle sequence is Markov is left open. [Simsek2021]: embedded critical points form affine subspaces with
  (m - r) zero Hessian eigenvalues; strict saddles stay strict; saddles far outnumber minima when the width is
  just above the minimal width. [Kunin2025] (AGF): section 2.2.
- Escape-time laws. [RawalDeWeese2026] (preprint; scalar output, smooth sigma with sigma'(0) > 0, the first
  r layers at scale eps): an exact imbalance identity driven by `phi_sigma(z) = z sigma'(z) - sigma(z)`
  (Thm 3.1); escape time Theta(1/Gamma), Theta(log(1/eps)/Gamma) and Theta(eps^-(r-2)/Gamma) for r = 1, 2
  and >= 3, with Gamma = ||E[yx]|| (Thm 4.2); a multimode teacher is escaped once, jointly (Sec. 5), because
  the network is linear in x near the origin. [Bantzis2026]: escape directions satisfy
  `grad L_0(rho) = -s rho`, and along them
  `||theta(t)|| = (||theta(t_0)||^(2-L) + (2-L) L s (t - t_0))^(1/(2-L))` (Prop 2.2). [Zhang2025] (ICML
  2025; linear attention trained on in-context regression): with merged key-query weights (degree 2 per
  head) there is one abrupt drop after ~ln(1/w_init); with separate key and query (degree 3 per head) there
  are 2^D fixed points and a staircase through them in eigenvalue order (in-context principal component
  regression), with plateaus polynomial in 1/w_init. The same task under two parameterizations has
  different plateau laws. [BoixAdsera2023] (diagonal attention weights): stages of Theta(log(1/alpha)), the
  rank growing by at most one per stage (Thm 4.1).
- High-dimensional analogues. [Abbe2023] (COLT 2023): leap complexity; conjectured learning time
  Theta~(d^max(Leap, 2)); proven for a class of functions with Gaussian data, two-layer nets and layer-wise
  training; saddle-to-saddle is proven only when leaps increase. [Bietti2023] (multi-index models,
  fast-slow GF): no bad local minima in G = M M^T; the saddles form a flag; escape from saddle k takes
  ~d^(s_k - 1); with a known link, Stiefel GF can be trapped for some targets. [Ren2025] (NeurIPS 2025;
  two-layer, norm-tied student, information exponent 2I > 2): sharp transitions at
  `T_p ~ 1 / (a_pi(p) eta vbar_(p,pi(p))(0)^(2I-2))`, with the student-teacher matching chosen by greedy
  maximum of `a vbar(0)^(2I-2)`; width ~P log P recovers P directions; a power-law teacher gives a power-law
  loss curve. [Zhou2026] (preprint, abstract only): staging also occurs from standard init in high
  dimension. [Boursier2022] (shallow ReLU nets, orthogonal inputs, small balanced init): phases of
  ~log(1/lambda) duration passing near a strict saddle (Fact 1); the limit is a minimum-norm interpolator.
  [BerthierMZ2024] (time-scale separation from layer-wise learning rates): the degree-l Hermite component is
  learned at t ~ c_l eps^(1/(2l)).
- Learning rate. [Cohen2021]: sharpness rises to ~2/eta and hovers there (edge of stability, EoS).
  [Ghosh2025] (deep matrix factorization): the balanced sharpness of mode i is `S_i = L sigma_i^(2-2/L)`;
  when 2/eta lies between max(S_(p+1), S_p / (2 sqrt 2)) and S_p, the top-p singular values oscillate in a
  2-period orbit inside a p-dimensional subspace (Thm 1), with period doubling to chaos beyond; the
  balancing gap that GF conserves breaks and decreases monotonically to 0 (Prop 2). [Hofmann2026]
  (preprint; finite-step maps): balanced two-factor edge at a = 1; with depth scaling a = c/n a Ricker-type
  map with edge c = 1; the imbalance obeys `Delta_+ = (1 - a^2 r^2) Delta`; wide linear nets have a ladder
  of spectral edges. [WangCZT2022]: GD with a large step on XY^T converges beyond 2/L and balances the
  factors. [KimLee2026] (multi-pathway DLNs): GF gives winner-take-all; at EoS, GD re-balances signals
  across pathways. [MarionChizat2024]: the sharpness of DLN minimizers has a lower bound linear in depth, and
  GF reaches it within a constant; small init gives near rank-one weights. [Kreisler2023] (scalar nets): GD
  monotonically decreases the sharpness of the GF solution from the current iterate. [LiangMontufar2025]
  (matrix factorization): near-critical steps are chaotic, with fractal convergence regions.
- Noise. [Corlouer2026] (preprint; DLN SGD as an SDE, aligned and balanced weights): per-mode drift
  `L w^(2(L-1)/L) (s - w)` plus an Ito drift `beta L (L-1) (s - w) w^(2(L-2)/L)`; near the origin the
  outward Ito drift dominates the gradient drift (verifier's reading of their formulas); diffusion along a
  mode peaks before the mode is learned; mode order is kept; "SGD does not fundamentally alter" the
  saddle-to-saddle dynamics. [Ziyin2023]: at Type-II saddles the gradient noise vanishes; probabilistic
  stability gives four phases. [ChenKunin2023]: invariant sets (vanishing or identical neurons) become
  attractive when `L''(0) + D''(0) / 2 > 0`; in linear teacher-student models, modes weaker than
  ~eta zeta^2 / 2 collapse. [Ziyin2024]: along exponential symmetries, SGD noise drives a Noether flow to a
  unique noise equilibrium. [ZhangT2026] (abstract): label-noise SGD moves two-layer linear nets from lazy
  to rich. [Singh2026] (abstract): GD's low-rank bias in factored models comes from gauge equivariance,
  which Adam lacks.

**Setting.** Two-layer or scalar-output nonlinear nets, high-dimensional limits, and DLNs for the
learning-rate and noise results.

**Carries over to TTNs?**

- By analogy, strongly: AGF's dormant dynamics with kappa = K and U = g is H1's spherical-flow picture; at
  later saddles AGF's kappa (or Neuron Pursuit's L), the lowest degree of the dormant block with nonzero
  residual correlation, is a candidate formal definition of H2's k. [Zhang2025] shows that this degree, and
  hence the plateau law, depends on the parameterization: for TTNs, dense vs CP-structured cores and where
  the physical legs sit (`binary` vs `ht`) change it. [RawalDeWeese2026]: with mixed-scale inits the
  first-escape exponent should count only the small cores (#small - 2); their joint escape relies on
  linearity in x near the origin, which TTNs lack.
- Learning rate and noise: with K >= 3 the Hessian vanishes at the origin, so there is no catapult at the
  start (the condition eta lambda_0 > 2 of [Lewkowycz2020] fails); sharpness grows with the learned strength
  (~K s^(2-2/K) per balanced component, by the DLN formula), so EoS is expected only after components are
  learned (inference); EoS and large steps erode the per-bond imbalance; SGD noise moves gauge directions to
  a noise equilibrium; stochastic collapse onto zero bond channels could end the saddle sequence at a lower
  bond rank. The TTN origin and dormant-channel saddles are Type-II saddles (every per-sample gradient has a
  zero factor).

## 3. Status of our hypotheses H1-H6

**H1 (first escape; init-dependent order for tensorial teachers): partly known.**

- Proven for TTNs as a corollary (derived (review)): by [KumarHaupt2025a] Thm 1, the direction converges to
  a non-negative KKT point of g on the sphere within ~`||theta_0||^-(K-2)`, or the weights shrink (K = 2:
  [KumarHaupt2024]). From an init exactly along a nondegenerate KKT point w*, escape takes
  `1 / (K (K-2) g(w*) delta^(K-2))` [KumarHaupt2025b], but nondegeneracy fails for chi >= 2 (gauge orbits).
  PLAN.md Q2's `t_1 ~ rho_0^-(K-2)` has this form.
- Not known: which KKT point random init selects, whether it is a local maximum, and whether local maxima
  are bond-rank-one. Known: the global maximum on the unit sphere is `K^(-K/2) ||W*||_inj`, attained at
  balanced bond-rank-one configurations aligned with a best rank-one approximation of W* (derived (review):
  Cauchy-Schwarz from the leaves up gives `||W(theta)||_nuc <= prod_v ||G_v||_F`, then AM-GM; with
  multi-site cores use the injective norm of the grouping of sites into cores). Best rank-one approximation
  has non-global local optima (Kofidis and Regalia 2002, via [Furman2026]) and is NP-hard [HillarLim2013],
  so several basins are expected for tensorial teachers.
- Mechanism proven in analogues: the order is set by (teacher weight) x (init overlap)^(p-2) for degree
  p >= 3: tensor PCA ([BenArous2024a] Thm 1.5; [BenArous2024b]), two-layer nets [Ren2025], units quadratic
  in their own weights ([ZhangSL2026] Prop 5), AGF thresholds [Kunin2025], the first CP stage ([Razin2021]
  Thm 2, a sufficient condition with exponent 1/(N-2)), deep diagonal nets [Zhao2026], tensor power method
  basins ([Anandkumar2014] Lemma 5.1). Degree-2 (matrix-like) cases are greedy given a gap ([BenArous2024a];
  [Gidel2019]), and DLNs escape greedily first among positive-speed directions ([Jacot2021] Lemma 15).
- Qualifiers (counter-evidence to "random"): largest-first is guaranteed above an SNR gap factor
  `C (eta sqrt(log eta))^(p-2)` ([BenArous2024a] Thm 1.3); over-parameterized CP fits largest first
  empirically [Ge2021]; width ~P log P suffices to learn all directions [Ren2025].
- H1b (PLAN.md Q3, "piggyback" correction): the DLN analogue is known. Rank-r critical points that do not
  solve the rank-r problem (S != [1, r]) are strict saddles [Achour2024], and non-optimal escape directions
  have negative curvature along a two-layer perturbation ([Jacot2021] Lemma 15). In single-unit odeco models
  a wrong component is stable [Anandkumar2014]. Whether a TTN offers a 2-core correction route is open.
- Suggested restatement: "The first escape converges in direction to a KKT point of g selected by the init.
  For tensorial teachers several KKT points attract, so the order is init-dependent unless the teacher gap
  exceeds a K-dependent factor or chi is large."

**H2 (escape exponent set by locality): partly known (landscape bounds, degree counting in related models);
the topology-dependent plateau law is open.**

- Landscape: at min-norm non-global critical points a descent curve of order N exists, with N the number of
  nodes of a connected rank-deficient core ([Furman2026] Lemmas D.8-D.9): an upper bound on the saddle
  order. For parity the order lies between about n/2 (Prop 5.3) and n-1 (D.8 with the whole tree as the
  core; derived (review)). Tucker: when 2, 3, 4 cores must move the exhibited orders are 2, 3, 4
  [FrandsenGe2022].
- Dynamics in related models: per-node exponent `2 - 2/L_nu` with L_nu = children + 1 ([Razin2022] Thm 1,
  CP-structured nodes); the plateau at a saddle is at least ~delta^-(L-2), L the lowest homogeneity degree of
  the dormant block ([KumarHaupt2025c] Thm 8); the escape exponent is (number of small layers) - 2
  ([RawalDeWeese2026] Thm 4.2); degree 2 vs 3 per unit gives logarithmic vs power-law plateaus [Zhang2025];
  leap exponents [Abbe2023; Bietti2023]; the first-escape exponent is K - 2 in every analogue
  [Jacot2021; LiLuoLyu2021; Gissin2020; Atanasov2022].
- Empirical tensor-network evidence: MPS miss long-range correlations [Tang2025]; trees capture them
  [Cheng2019]; higher branching helps [ChenBarthel2024]; denser geometries train better [MasotLlima2024];
  GD on TT shows plateaus and then abrupt rank growth [Milanesi2021].
- Refinements. (i) H2's k (cores in the Steiner subtree plus bonds whose rank must grow) is not Furman's N
  (nodes touching a connected component of rank-deficient edges); compare both with data. (ii) At a later
  saddle the exponent is the lowest degree with nonzero residual correlation, which can be below the subtree
  size. (iii) Dormant entries at later saddles are not O(sigma): earlier phases rescale them, and the
  available bounds (`delta^(Delta/(Delta + 8N))` for L = 2 and `delta^(Delta/(2 L^2 N + Delta))` for L > 2
  in [KumarHaupt2025b]) need Delta > 0, so later plateaus need not scale as sigma^-(k-2). (iv) Imbalance
  can raise the order: the unbalanced 4-core Tucker point of [FrandsenGe2022] escapes only at order 6
  (derived (review)).
- Open: the plateau law versus topology; MPS vs `binary` vs `ht` on the same teacher; tightness of N.

**H3 (strict saddle iff k = 2): DLN analogue known; for TTNs the "if" direction is derived and the
"only if" direction is open.**

- DLN ([Achour2024] Thm 7): rank-r critical points are strict unless they solve the rank-r problem; those
  that do are non-strict iff tightened; 0-balanced ones are tightened for H >= 3; at H = 2 all saddles are
  strict. So the saddles visited from small balanced init are non-strict at depth >= 3.
- TTN facts (derived (review)): the loss is a convex quadratic in each single core, so single-core Hessian
  blocks are PSD and, at a critical point, no single core can descend: every saddle needs >= 2 cores
  (k >= 2). A rank-deficient core with N = 2 nodes gives `L(t) - L = t^2 <R,U> + O(t^4)` along a straight
  line, i.e. negative curvature, i.e. a strict saddle ([Furman2026] Lemma D.9). Points with >= 3 zero cores
  have zero gradient and zero Hessian for every loss ([Levin2024] proof; [Kawaguchi2016] for DLNs). The
  parity saddle is non-strict ([Furman2026] Prop 5.3).
- Correction: "strict exactly when k = 2" is too strong as stated. A saddle is strict whenever some
  direction has a nonzero degree-2 residual term (e.g. points that learned the wrong component, H1b), and
  imbalance adds degeneracy (H2 iv).
- Open: a tree version of "tightened"; lower bounds on the saddle order for general cores.

**H4 (regime map): partly known in analogues; open for TTNs; three corrections.**

- "sigma^K is the main knob": right for width-1 models (Q depends on alpha^D, [Woodworth2020] Thm 3; in
  [Gissin2020] the per-layer threshold tends to 1 with depth, derived (review), a toy instance of
  sigma_c(K) -> 1). With width or bond dimension, lifted Gram scales govern: alpha^2 k ([Woodworth2020]
  Sec. 6), sigma^2 w per singular value [Tu2024], alpha sqrt(h) [Geiger2020]; CP needs two independent
  parameters [Yarotsky2026]. A two-layer ReLU phase diagram (linear, critical, condensed regimes) is in Luo,
  Xu, Ma and Zhang (JMLR 2021; incomplete entry in section 7).
- Correction 1: "a lazy regime needs a centred or antisymmetric construction, or alpha-scaling" is too
  strong. Wide bonds in NTK-type scaling are lazy ([GuoDraper2021b], MPS, informal; [Yarotsky2026]
  Prop 8.1, CP). With the fully observed population loss and an isotropic tangent kernel,
  `W(t) - W* = e^(-ct) (W_0 - W*)` fits W_0 away, so centring matters only at finite chi or with finite
  samples (derived (review)). Laziness needs every bond wide, chi >> K (kurtosis `3 prod_e (1 + 2/chi_e)`,
  derived (review); [Littwin2020]; [HaninNica2020]).
- Correction 2: the init-to-target norm ratio alone does not decide the regime ([Yarotsky2026]; [Tu2024]);
  the TTN analogue of the second parameter is unknown.
- Correction 3: no muP exists for TTNs: the init clause of [YangSB2023] fails for cores with fan-in >
  fan-out, and TP IV's dichotomy excludes linear activations; the "ultra-rich" muP regime is small init by
  homogeneity ([Atanasov2025] plus derived (review)).
- Lazy training at finite chi cannot fit a generic W* ([Ghorbani2019]; [Chizat2019] Thm 2.5; [Wang2020]:
  Omega(d^(l-1)) vs `O*(r^(2.5 l) log d)` components).
- Imbalance is an established second knob in two-layer linear nets [Kunin2024; Domine2025; Tu2024;
  Azulay2021; Min2021; XuZiyin2025]; it is erased by weight decay ([Furman2026] Cor F.13), large steps
  [WangCZT2022; Ghosh2025; Hofmann2026] and SGD noise [Ziyin2024]. Fan-in init has
  `E[Q_v] = sigma^2 (1 - chi_p/chi_v) I` (derived (review)).
- Escape exponents count only small cores [RawalDeWeese2026]; staging occurs without small init in high
  dimension [Zhou2026], so d and chi are knobs too.
- Random iid init makes the function scale exponential in K [Barratt2022; MataAli2023; Sasaki2026;
  Rakhshan2020; Novikov2017; Shamir2019].
- With l2 weight decay on the cores and K >= 3 the origin is a local minimum (the degree-K gain loses to the
  quadratic penalty), as in DLNs [ZiyinLiMeng2022] (derived (review) for TTNs).

**H5 (input covariance): partly known in linear analogues; open for TTNs.**

- Aligned or commuting covariance reorders learning by covariance-weighted strengths [Gidel2019]; for any
  convex loss the first escape follows the covariance-weighted teacher ([Jacot2021] Thm 2; [LiLuoLyu2021];
  [Achour2024] with invertible Sigma_XX); rank-r saddles are reduced-rank regression solutions, not teacher
  directions.
- TTN with product covariance (derived (review)): absorbing Sigma_i^(1/2) into the core carrying site i is
  an invertible reparameterization, so the landscape is that of a white problem with target
  `((x)_i Sigma_i^(1/2)) W*` (critical points, saddle orders and [Furman2026] Thm 5.2 transfer); the GF
  metric and the conservation laws are not preserved, so time scales and order can change. In the lazy
  limit with an isotropic kernel the eigenmodes decay at rates c prod_i lambda_(i,k_i): a pure reordering.
- Misaligned covariance: early alignment to Sigma beta and then realignment, which breaks silent alignment
  [Atanasov2022]; active sets need not grow monotonically [PesmeFlammarion2023] ([Berthier2023] needs
  anti-correlated features for monotone activation); equal singular values remove intermediate plateaus in
  linear nets but not in init-ordered models [ZhangSL2026]; the spectrum sets order and plateau lengths
  [Zhang2025]; the escape rate is ||Sigma w*|| for linear teachers [RawalDeWeese2026].
- Open: cross-site correlations; the exact timing of later plateaus under non-commuting covariance.

**H6 (learning rate): partly supported; "no change in the saddle sequence at small init" is not
established and is contradicted in some settings.**

- Supported: small steps track GF incremental sequences ([Gidel2019], depth 2; in [Even2023] the step-size
  gain vanishes as steps shrink); at small init with K >= 3 the Hessian vanishes, so the catapult condition
  eta lambda_0 > 2 [Lewkowycz2020] cannot hold at the start; sharpness grows with depth and learned strength
  (`S_i = L sigma_i^(2-2/L)` [Ghosh2025]; linear-in-depth lower bound [MarionChizat2024]), so PLAN.md Q8's
  `eta_max ~ 2 / (K s_1^(2-2/K))` matches the DLN formula (to verify for TTNs); EoS oscillations stay in the
  learned top subspace and the balancing gap breaks and decays to 0 [Ghosh2025; Hofmann2026; WangCZT2022];
  SGD keeps the mode order in aligned, balanced DLNs [Corlouer2026].
- Against: large steps re-balance features across pathways after GF-like winner-take-all [KimLee2026];
  the step size changes the implicit bias [Nacson2022; Even2023]; SGD noise lowers the effective init
  [Pesme2021]; stochastic collapse makes zero or duplicate channels attractive [ChenKunin2023]; Type-II
  saddles [Ziyin2023]; noise equilibrium along gauge orbits [Ziyin2024]; near-critical steps are chaotic
  [LiangMontufar2025]; Adam lacks the gauge equivariance behind GD's low-rank bias [Singh2026].

## 4. Corrections to prior_knowledge.md

Numbered by the section of `notes/prior_knowledge.md` they correct.

**Section 2 (deep linear networks)**

1. The escape-time law `a0^-(L-2) / ((L-2) s)` is not stated in [Saxe2014]; it follows from their eq. 15
   (where N_l counts layers of units, i.e. N_l - 1 matrices). Theorem-level sources: [LiLuoLyu2021] Thm 6.2
   (first phase, init alpha I, a technical assumption); [Jacot2021] (first escape); [Gissin2020] Thm 1
   (depth-normalized loss); [KumarHaupt2025b] (`1 / (L (L-2) N(w*) delta^(L-2))` for init along w*);
   [Atanasov2022] (t_1/2).
2. The depth experiment of [Saxe2014] contrasts random orthogonal init (depth-independent learning times)
   with scaled Gaussian init, not with small init.
3. "For L >= 3 the origin is a higher-order saddle (Kawaguchi 2016)": correct; [Kawaguchi2016] exhibits the
   zero-Hessian set (which contains the origin) for >= 2 hidden layers. The full strict/non-strict
   classification is [Achour2024].
4. "Marcotte, Gribonval & Peyre 2023, 'Abundant conservation laws'": the title is "Abide by the Law and
   Follow the Flow: Conservation Laws for Gradient Flows" (NeurIPS 2023). Completeness of the known laws is
   proven only for two-layer linear nets of full rank and two-layer bias-free ReLU nets.
5. [DuHuLee2018] credit the matrix law to [Arora2018]; their version needs only two consecutive linear
   layers. "Kunin et al. 2021, Neural mechanics" is [Kunin2021] (ICLR 2021).
6. [Gidel2019]: depth 2 only, aligned init, approximately commuting Sigma_x and Sigma_xy; the order is set
   by the singular values of Sigma_xy.
7. [Gissin2020]: the same init for every coordinate, depth-normalized loss; random init is not analysed.
8. [Arora2019] show empirically that the nuclear norm fails and only hypothesise that no norm suffices; the
   proof that norms can diverge is [RazinCohen2020] (NeurIPS 2020).
9. [LiLuoLyu2021]: symmetric PSD factorization (the general case reduces to it); later phases need an
   alignment assumption; at depth >= 3 only the first phase from alpha I is proven.
10. [Jacot2021] is an arXiv preprint only (v2, Jan 2022). It proves the first escape (probability 1 for
    L <= 3, >= 1/2 for L > 3), not the sequence (Conjecture 3). "Plateau lengths scale with the init as
    implied by depth" is established for the first escape only. Its width phase diagram (Thm 1) is static.
    PLAN.md Q1 ("full saddle-to-saddle theory in deep linear networks (Jacot et al. 2021)") needs the same
    correction.
11. "Advani & Saxe (2017/2020)": [Advani2017] is arXiv 1710.03667; the journal version is Advani, Saxe and
    Sompolinsky, Neural Networks 132:428-446 (2020). Lampinen and Ganguli is [Lampinen2019] (ICLR 2019).
    Neither was reviewed in depth.
12. [Braun2022]: two-layer, zero-balanced init (not necessarily small or aligned).
13. [Domine2025]: two-layer only (despite the title); whitened inputs; lambda-balanced init
    `W_2^T W_2 - W_1 W_1^T = lambda I` with hidden width min(N_i, N_o), not general unbalanced inits;
    ICLR 2025 (venue not re-verified).
14. [Kunin2024]: delta includes the layer learning rates. In linear nets balanced init becomes rich
    fastest, upstream init is lazy and downstream init is delayed-rich; "upstream accelerates rich
    learning" holds only in piecewise-linear nets.
15. [Tu2024]: NeurIPS 2024; two-layer; threshold sigma^2 w per singular value; narrow widths are outside
    the theory.

**Section 3 (diagonal networks, kernel vs rich)**

16. [Woodworth2020]: COLT 2020, PMLR 125:3635-3673 (arXiv 2002.09277; the older 1906.05827 was withdrawn).
    The rich limit is l1 for every depth D (not l_(2/D)); "hypentropy" refers to D = 2 only; "depth sharpens
    the transition" means that D = 2 needs exponentially small alpha and D >= 3 only polynomially small
    alpha; the init shape changes only the kernel limit; width enters through the lifted scale alpha^2 k.
    The two-copy construction is as stated.
17. [Chizat2019] scale alpha h(w) and arrange h(w0) = 0 with a symmetrized init; the explicitly centred
    form alpha (f(theta) - f(theta_0)) is the variant of [Geiger2020]. Their kappa is a sufficient
    criterion only; homogeneous models become lazy under init scaling only if the initial output stays
    bounded; width helps through cancellations (lazy if m alpha(m) -> inf), so "a matter of scaling, not
    width" is incomplete; underparameterized lazy training converges to a local minimum (Thm 2.5).
18. [PesmeFlammarion2023]: parameterization u (.) v with init (sqrt(2) alpha, 0); the algorithm is the
    adaptive inverse scale space method, LARS-like but not the Lasso/homotopy path, history-dependent, with
    possible deactivation of coordinates; also J. Stat. Mech. (2024) 104016.
19. [Berthier2023]: JMLR 24(171); nonnegative, anti-correlated (n >= d) case only; the path is a weighted
    lasso path whose weights are the init exponents.
20. [Geiger2020]: J. Stat. Mech. (2020) 113301; boundary alpha* ~ h^-1/2; the knob is the output scale,
    which equals an init-scale knob only for homogeneous nets.
21. [Ghorbani2019]: NeurIPS 2019 (the proceedings title ends "...Network"); the NT/NN separation is proven
    for N <= d with quadratic activation.

**Section 4 (mean field / muP)**

22. "Every stable parameterization is either kernel-limit or feature-learning" [YangHu2021] needs
    Assumption 3.1 (tanh or sigma-gelu activations); the paper states that it fails for linear activations,
    hence for multilinear TTNs.
23. [YangSB2023] is an arXiv preprint (no venue found). Its Claim 1 assumes fan-out >= fan-in, which
    internal TTN cores violate; its init would shrink TTN signals by chi^(-(c-1)/2) per core with c bond
    children (derived (review)). Only the update clause carries over, so "the spectral condition gives the
    right per-core scalings" holds for updates, not for the init.
24. Depth-muP: [Yang2024] and [Bordelon2024a] are ICLR 2024 (2023 is the arXiv year); both need scaled
    residual branches, which TTNs lack.
25. "Nothing in this line has been worked out for TTNs" is too strong: GP/NTK limits for MPS and TT exist
    ([GuoDraper2021a], [GuoDraper2021b], [WeselBatselier2025], [WangLiuYu2026]), as do muP-style rules for
    TT and Einsum layers ([Qiu2024], [Potapczynski2024]) and a lazy/mean-field regime diagram for CP
    ([Yarotsky2026]). Still missing: any feature-learning limit, and anything for general trees.
26. Mean-field venues: [MeiMontanariNguyen2018] PNAS 115(33) (journal title "...two-layer neural
    networks"); [ChizatBach2018] NeurIPS 2018; [RotskoffVE2018] NeurIPS 2018, journal version CPAM 75(9)
    (2022) under a different title; [SirignanoSpiliopoulos2020] SIAM J. Appl. Math. 80(2). Multilayer
    mean-field limits degenerate at depth >= 4 with iid init [NguyenPham2023].

**Section 5 (tensor factorization and tensor networks)**

27. [Razin2021] proves only the first stage (Thm 2: Huber loss, zero unbalancedness, a leading-component
    condition with exponent 1/(N-2), exponentially small init); incremental learning of later components is
    empirical.
28. [Razin2022] is not a dense-core binary-tree TTN: any mode tree; each node is a local CP factorization
    (a copy index plus a weight matrix); the invariants are per-component scalars (Lemma 2), not per-bond
    GL(chi) matrices; the Thm 1 exponent 2 - 2/L_nu uses L_nu = children + 1 (node-local, not depth); the
    low-hierarchical-rank bias is conditional on an alignment inequality the authors do not prove. The
    [believed] remark (no plateau-time law versus topology) is confirmed.
29. [Ge2021]: the theorem is for a modified GF (re-initialization, l2 penalty, epochs with decreasing
    thresholds) on a 2-homogeneous normalized model with m = poly(d) components; largest-first for vanilla
    GF is empirical; the escape is logarithmic (matrix-like). [Wang2020]: lazy training needs
    Omega(d^(l-1)) components vs `O*(r^(2.5 l) log d)` for a GD variant.
30. [Hariz2022]: deep CP; the "polynomial growth" exponent equals the depth-D law with D = N + sum k_n
    (derived (review)). The dense-core (deep Tucker) result is the separate [Hariz2024] (AISTATS 2024).
31. [Anandkumar2014]: correct; add the explicit basin rule (Lemma 5.1) and the non-robust eigenvectors;
    setting: symmetric odeco tensors of order 3 with positive weights.
32. [BenArous2021]: JMLR 22(106); [believed] becomes known. Sample thresholds about N, N log^2 N and
    N^(k-1) log^2 N for k = 1, 2 and >= 3. Multi-spike learning order: [BenArous2024a], [BenArous2024b].
33. "Kumar & Haupt (2024/25)" is four papers: [KumarHaupt2024] (TMLR 2024, 2-homogeneous, also near
    saddles), [KumarHaupt2025a] (TMLR, L > 2), [KumarHaupt2025b] (JMLR 26(269), beyond the origin; needs
    Delta > 0, which fails for chi >= 2) and [KumarHaupt2025c] (arXiv, Neuron Pursuit). The limit is a
    non-negative KKT point, not necessarily a maximum.

**Section 6 (tensor networks for ML)**

34. [Liu2022]: unitary-embedded MPS with normalized outputs; global losses are exponentially suppressed
    (d^-n), local losses by O(d^-Delta) in the distance to the observable; this does not transfer to
    unnormalized regression.
35. "Miao & Barthel (2023/24)" is two papers, [MiaoBarthel2024] (PRA 109, L050402) and [BarthelMiao2025]
    (CMP 406, 86); both cover MPS, TTNS and MERA; "polynomial decay" is in bond dimension (and in N only
    through the layer index tau <= log_b N).
36. [Novikov2017]: ICLR 2017 workshop track (arXiv 2016); random TT init "sometimes freezes" training,
    fixed by a TT-rank-2 linear-model init.
37. [Liu2019]: authors Ding Liu, Shi-Ju Ran, Peter Wittek, Cheng Peng, Raul Blázquez García, Gang Su,
    Maciej Lewenstein; the TTN is kept unitary and trained by a MERA-derived algorithm, not GD.
38. [Stoudenmire2016]: the NeurIPS title is "Supervised Learning with Tensor Networks"; the sweeps are
    two-site gradient steps followed by SVD truncation.
39. Exponential growth or decay of ||W|| at random init: confirmed with explicit formulas ([Barratt2022],
    [MataAli2023], [Sasaki2026], [Rakhshan2020]).
40. Bibliographic details: [Glasser2019] full title "Expressive power of tensor-network factorizations for
    probabilistic modeling, with applications from hidden Markov models to quantum machine learning";
    [CohenSharirShashua2016] COLT 2016, pp. 698-728; [Levine2018] ICLR 2018 not re-verified (arXiv lists no
    venue); [Khrulkov2018] ICLR 2018; [Stoudenmire2018] Quantum Sci. Technol. 3, 034003.

**Section 7 (hierarchical formats)**

41. "The bond rank on edge e equals the rank of the matricization": the matricization rank is at most the
    bond dimension r_e; a target is realizable iff rank T*^(e) <= r_e on every internal edge
    ([Furman2026] Prop B.1); the ranks are equal at min-norm points (Prop D.2).
42. Fixed-rank HT/TT sets are smooth embedded submanifolds but are not closed; their closure is
    {rank_T <= k} ([UschmajewVandereycken2013] Thm 4.14). The content of [Holtz2012] was not read (its
    results are known here through [Lubich2015]).
43. The gauge conservation law is correct and proven in [Furman2026] Prop F.12 for any loss that depends on
    theta only through W, on any topology; balanced <=> min-norm holds only on trees (Prop F.10,
    Remark F.14); weight decay drives the defect to zero exponentially (Cor F.13). The per-core balanced
    gauge is not Vidal's canonical gauge (derived (review)), and GF preserves neither.

**Sections 8-9 (hypotheses and checks)**

44. H1: the limit is a KKT point, not necessarily a maximum; "every odeco component is a local max" is
    classical [Anandkumar2014]; in TTN parameter space such maxima are at best non-strict (gauge orbits).
    "For a matrix-like teacher only the top component is a local max (as in DLNs)" holds in DLNs among
    positive-speed escape directions ([Jacot2021] Lemma 15; zero-speed directions are excluded only for
    L <= 3) and fails for diagonal or decoupled models; the governing property is whether components are
    connected through a perturbation of at most 2 factors.
45. H3, H4 and H6: see section 3 (strict saddles beyond k = 2; lazy dynamics without centring at large chi;
    two-parameter regime control; the learning rate can change late saddles).
46. The section 9 checks are answered as follows. (1) No characterization of TTN saddle-to-saddle exists;
    [Razin2022] stops at a conditional bound and [Furman2026] is landscape only. (2) H1's mechanism is known
    in analogues, H2 partly (degree counting) in analogues. (3) GP/NTK limits exist for MPS and TT
    (informal, or about the prior at init), none for general trees. (4) Yes: section 2.5. (5) [Kunin2025],
    [ZhangSL2026], the Kumar-Haupt papers, [RawalDeWeese2026]. (6) [Furman2026] App. F; completeness open
    [Marcotte2023]. (7) [Furman2026], [FrandsenGe2022], [ChenLiLu2020], [Levin2024].

## 5. What is open (ranked)

Ranked by importance to the main question (saddle-to-saddle in TTNs), then tractability. PLAN.md question
numbers in brackets.

1. **Saddle sequence and plateau law of GF on dense-core TTNs from small init** (Q1-Q3).
   Why open: [Furman2026] gives landscape upper bounds only; [Razin2022] has CP-structured nodes and a
   conditional bound; no paper treats dense cores dynamically; even for DLNs the full sequence is unproven.
   Evidence that would settle it: (a) exact reduced ODEs for odeco teachers with aligned init, then
   perturbation theory away from alignment; (b) measured exponents of t_(i+1) - t_i against sigma (at least
   5 scales over at least a decade, several seeds) on MPS, `binary` and `ht` trees with the same teacher,
   compared with the predicted k and with Furman's N; (c) whether each plateau sits at a bond-rank-constrained
   critical point (bond spectra equal to truncated teacher spectra; Tucker-rank deficiency as in Thm 5.2).
2. **First-escape selection rule and P(greedy order)** (Q4).
   Why open: [KumarHaupt2025a] gives convergence to some KKT point; the analogues (tensor PCA, two-layer
   nets) have sphere constraints or independent units; a TTN combines K per-core overlaps and a gauge.
   Evidence: P(greedy) against s_2/s_1, K, chi and d with confidence intervals; a fit of a rule of the form
   "strength x (product of per-core overlaps)^q"; the KKT points of g for small trees (chi = 1 reduces to
   scalar-chain products); whether larger chi restores greedy order as over-parameterization does for CP.
3. **Strict vs higher-order saddles; lower bounds on saddle order** (H3).
   Why open: only the DLN classification [Achour2024] and Tucker examples [FrandsenGe2022] exist; Furman's N
   is an upper bound and parity gives the only lower bound.
   Evidence: a tree version of "tightened" stated through bond-matricization ranks of partial contractions;
   gauge-projected Hessians and higher-order expansions at observed plateaus; analytic N = 4 cases (`binary`
   and `mps`).
4. **Effective degree at later saddles** (H2).
   Why open: [KumarHaupt2025c] gives only a local lower bound; earlier phases rescale the dormant entries;
   no TTN computation exists.
   Evidence: compute, for each tree and teacher, the lowest degree with nonzero residual correlation in the
   dormant blocks; measure dormant-block norms entering each plateau; compare predicted and measured
   exponents.
5. **Escape-path theorem modulo gauge.**
   Why open: [KumarHaupt2025b] needs Delta > 0, which the gauge orbits violate for chi >= 2 (also for DLNs
   of width >= 2).
   Evidence: a quotient (Morse-Bott) version of Thms 4 and 11; numerics showing that trajectories track the
   escape path up to gauge.
6. **Which variables govern the regime map** (Q5).
   Why open: CP needs two control parameters [Yarotsky2026]; a two-core TTN has the per-mode threshold of
   [Tu2024]; general trees are untouched.
   Evidence: phase diagrams with the rules of `notes/diagnostics.md`; collapse tests in candidate variables
   (sigma^K; per-bond Gram scale vs bond singular values; per-path amplitude times number of paths, which is
   sigma^K chi^(K-1) in iid normalization); centred vs plain models at large chi.
7. **Large-chi limits.**
   Why open: no GP/NTK result beyond MPS/TT; no feature-learning limit; [YangSB2023] fails for cores with
   fan-in > fan-out.
   Evidence: NTK fluctuations and drift against chi and K (expected ~sum_e 1/chi_e); a per-core muP derived
   from signal propagation, as [Vankadara2024] did for SSMs; muTransfer tests across chi; whether
   saddle-to-saddle survives chi -> inf at gamma ~ sigma^-K.
8. **Imbalance.**
   Why open: the effect of the per-bond Q_e on order and exponents is unexplored; unbalanced points can
   have higher escape order (H2 iv) and non-min-norm spurious minima exist ([Furman2026] Thm G.1).
   Evidence: runs with controlled per-core sigma_v and with the fan-in root imbalance; check whether
   trajectories stall near non-min-norm critical points.
9. **Learning rate and SGD** (Q8).
   Why open: only DLN, diagonal and two-layer results exist.
   Evidence: GD with eta near 2/sharpness and SGD with varying eta/B, tracking Q_e, bond spectra and the
   effective bond rank; tests of stochastic collapse onto zero channels and of the sharpness formula.
10. **Input covariance** (Q6): misaligned product covariances and cross-site correlations; exact timing of
    later plateaus. Evidence: the data-teacher experiments, compared with the absorbed-covariance white
    problem of section 3 (H5).
11. **Exact theory.** Completeness of the per-bond conservation laws (run the algorithm of [Marcotte2023]
    on small trees); the explicit metric `(J J^T)^-1` for balanced TTNs; Saxe-type exact solutions for
    aligned odeco teachers (listed as future work by [Furman2026]).
12. **Optimizer dependence.** GD vs dynamical low-rank (BUG), Riemannian, natural-gradient, DMRG-style
    sweeps and Adam on the same teacher: are plateaus removed, is the order changed?
13. Lower priority: power-law teacher spectra and power-law loss curves [Ren2025]; grokking as an
    entanglement transition [Pomarico2025] read as a lazy-to-rich effect; a "tree leap complexity" linking
    plateau exponents to target structure and to the hardness result of [Furman2026].

## 6. Regime diagnostics used in the literature

Project equivalents refer to `notes/diagnostics.md`.

1. Relative parameter change `||theta(t) - theta(0)|| / ||theta(0)||`, per layer: [Chizat2019] (O(1/alpha)
   when lazy), [Lee2019] (~n^-1/2 in outer layers, ~1/n in inner layers), [Geiger2020] (~1/alpha).
   Project: delta_theta.
2. Tangent-kernel change `||Theta_t - Theta_0||_F / ||Theta_0||_F`: [Lee2019]; [Geiger2020] (exponent a in
   `(sqrt(h) alpha)^-a`); [BordelonPehlevan2022] (O(gamma_0^2 L)); [GuoDraper2021b] (constant MPS NTK).
   Project: delta_K.
3. Kernel distance `S(t1, t2) = 1 - <K_t1, K_t2> / (||K_t1|| ||K_t2||)`: [Kunin2024]. Not yet in the
   project; cheap to add.
4. Kernel-target alignment and silent alignment (the kernel aligns before it grows; feature change per unit
   loss change ~sigma^-L): [Atanasov2022]. Project: A(t).
5. Linearization error `||f_t - f_t^lin||`: [Lee2019], [Chizat2019]. Not yet in the project.
6. Laziness criterion `kappa = ||h(w0) - y*|| ||D^2 h|| / ||Dh||^2` (sufficient only): [Chizat2019].
7. Scale-collapse tests: test error against sqrt(h) alpha [Geiger2020]; the lifted scale alpha^2 k
   [Woodworth2020]; sigma^2 w against singular values [Tu2024]; Pareto exponents in (p, H, sigma)
   [Yarotsky2026].
8. Conserved imbalance: layer Gram differences [Arora2018; DuHuLee2018]; delta [Kunin2024]; lambda
   [Domine2025]; unbalancedness magnitude [Razin2021; Razin2022]; Delta_e [Furman2026]; the balancing gap at
   EoS [Ghosh2025]; imbalance contraction under GD [Hofmann2026]. Project: Q_e (integrator check and knob).
9. Stage structure: mode or singular-value trajectories [Saxe2014; Gidel2019 (steps in log-rescaled time);
   Arora2019; Jacot2021]; component and local-component norms [Razin2021; Razin2022]; unfolding singular
   values and entropy-based effective ranks [Milanesi2021]; rank of weight updates [BoixAdsera2023]; loss
   plateaus at partial-fit levels [Zhang2025; Kunin2025]; activation order and jump times
   [PesmeFlammarion2023; Kunin2025]. Project: bond spectra s_k^e, overlaps a_r, learning events.
10. (s, f)-incremental learning and the largest init that still gives it [Gissin2020]: the literature
    analogue of the project's separation ratio rho > 3.
11. Plateau-time scaling: log(1/alpha) vs alpha^-(L-2) against the init [Jacot2021; LiLuoLyu2021;
    Gissin2020; Atanasov2022; RawalDeWeese2026; Zhang2025; MaGuoFattahi2022]; d^(Leap-1) against the
    dimension [Abbe2023; Bietti2023]; activation thresholds [Kunin2025].
12. Saddle type: Hessian spectra and "tightened" rank tests [Achour2024]; the order of exhibited descent
    directions [FrandsenGe2022; Furman2026]; escape speeds and escape directions [Jacot2021; Bantzis2026];
    KKT alignment `u*^T w / ||w||` [KumarHaupt2025a]. Project: gauge-projected Hessian at plateau midpoints.
13. Sharpness against 2/eta: [Cohen2021]; [Ghosh2025] (S_i and the oscillation subspace); [Lewkowycz2020]
    (kernel eigenvalue and loss spike); [Kreisler2023] (sharpness of the GF solution);
    [MarionChizat2024]; [Atanasov2025] (sharpness at the loss drop).
14. Gradient-noise diagnostics: per-mode diffusion peaking before a mode completes [Corlouer2026];
    attractivity `L''(0) + D''(0) / 2` of invariant sets [ChenKunin2023]; Lyapunov exponents at Type-II
    saddles [Ziyin2023].
15. Gradient variance at random init against N, chi and distance [Liu2022; MiaoBarthel2024;
    BarthelMiao2025; CerveroMartin2023].
16. Output statistics at init: moments and kurtosis [WeselBatselier2025; WangLiuYu2026; section 2.3];
    Cramer-von Mises tests [WeselBatselier2025]; Var(W) and critical init scale [Barratt2022; Sasaki2026].
17. Entanglement (bond entropy) during training and "eigenvalue evaporation" [Pomarico2025]; captured
    long-range correlations [Cheng2019].
18. Final-solution diagnostics: implicit-bias potentials (Q_alpha; l1 vs l2) [Woodworth2020;
    PesmeFlammarion2023]; nuclear norm vs effective rank [Arora2019; Milanesi2021]; for lazy finite-chi runs,
    final loss against the tangent-space residual `||P_perp W*||^2` (suggested by the argument of
    [Ghorbani2019]).
19. Layer alignment `zeta = u^T w / (||u|| ||w||)` [XuZiyin2025].

Worth adding to `notes/diagnostics.md`: the kernel distance S(0, t), the linearization error, Geiger's
exponent a, the sharpness of the GF solution, the tangent-space residual, and the KKT alignment at the first
escape.

## 7. References

Format: [Key] authors. Title. Venue, year. URL (confidence: read = results checked in the paper; abstract =
abstract only). Venues marked "not re-verified" come from the sweeps' searches and were not re-checked on
a proceedings page. Papers whose confidence is "unconfirmed", and entries with incomplete metadata, are in
section 7.2.

### 7.1 Confirmed

- [Abbe2023] Emmanuel Abbe, Enric Boix-Adserà, Theodor Misiakiewicz. SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics. COLT 2023, PMLR 195:2552-2623. <https://arxiv.org/abs/2302.11055> (read)
- [Achour2024] El Mehdi Achour, François Malgouyres, Sébastien Gerchinovitz. The loss landscape of deep linear neural networks: a second-order analysis. JMLR 25(242):1-76, 2024. <https://arxiv.org/abs/2107.13289> (read)
- [Acuaviva2023] Arturo Acuaviva, Visu Makam, Harold Nieuwboer, David Pérez-García, Friedrich Sittner, Michael Walter, Freek Witteveen. The minimal canonical form of a tensor network. FOCS 2023, pp. 328-362. <https://arxiv.org/abs/2209.14358> (read)
- [Advani2017] Madhu S. Advani, Andrew M. Saxe. High-dimensional dynamics of generalization error in neural networks. arXiv 1710.03667, 2017; journal version with H. Sompolinsky, Neural Networks 132:428-446, 2020. <https://arxiv.org/abs/1710.03667> (metadata only; not reviewed)
- [AgyeiKodie2026] Eugene Agyei-Kodie, Longxiu Huang, Shuang Li, Xiao Liang. Strict-Saddle Landscapes and Multi-Rank Geometry in Low-Tubal-Rank Tensor Sensing. arXiv 2609.37865, 2026. <https://arxiv.org/abs/2609.37865> (abstract)
- [Alexander2023] Yotam Alexander, Nimrod De La Vega, Noam Razin, Nadav Cohen. What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement. NeurIPS 2023. <https://arxiv.org/abs/2303.11249> (metadata only)
- [Anandkumar2014] Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade, Matus Telgarsky. Tensor Decompositions for Learning Latent Variable Models. JMLR 15(80):2773-2832, 2014. <https://jmlr.org/papers/v15/anandkumar14b.html> (read)
- [AnandkumarGe2016] Animashree Anandkumar, Rong Ge. Efficient approaches for escaping higher order saddle points in non-convex optimization. COLT 2016, PMLR 49:81-102. <https://arxiv.org/abs/1602.05908> (abstract)
- [Arora2018] Sanjeev Arora, Nadav Cohen, Elad Hazan. On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization. ICML 2018, PMLR 80:244-253. <https://arxiv.org/abs/1802.06509> (abstract; formula checked via [Arora2019])
- [Arora2019] Sanjeev Arora, Nadav Cohen, Wei Hu, Yuping Luo. Implicit Regularization in Deep Matrix Factorization. NeurIPS 2019. <https://arxiv.org/abs/1905.13655> (read)
- [Atanasov2022] Alexander Atanasov, Blake Bordelon, Cengiz Pehlevan. Neural Networks as Kernel Learners: The Silent Alignment Effect. ICLR 2022. <https://arxiv.org/abs/2111.00034> (read)
- [Atanasov2025] Alexander Atanasov, Alexandru Meterez, James B. Simon, Cengiz Pehlevan. The Optimization Landscape of SGD Across the Feature Learning Strength. ICLR 2025. <https://arxiv.org/abs/2410.04642> (read)
- [Azulay2021] Shahar Azulay, Edward Moroshko, Mor Shpigel Nacson, Blake Woodworth, Nathan Srebro, Amir Globerson, Daniel Soudry. On the Implicit Bias of Initialization Shape: Beyond Infinitesimal Mirror Descent. ICML 2021, PMLR 139:468-477. <https://arxiv.org/abs/2102.09769> (read)
- [Bah2022] Bubacarr Bah, Holger Rauhut, Ulrich Terstiege, Michael Westdickenberg. Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers. Information and Inference 11(1):307-353, 2022. <https://arxiv.org/abs/1910.05505> (read)
- [Bantzis2026] Ioannis Bantzis, James B. Simon, Arthur Jacot. Saddle-To-Saddle Dynamics in Deep ReLU Networks: Low-Rank Bias in the First Saddle Escape. ICLR 2026. <https://arxiv.org/abs/2505.21722> (read)
- [Barratt2022] Fergus Barratt, James Dborin, Lewis Wright. Improvements to Gradient Descent Methods for Quantum Tensor Network Machine Learning. arXiv 2203.03366, 2022 (journal-ref: Second Workshop on Quantum Tensor Networks in Machine Learning, NeurIPS 2021). <https://arxiv.org/abs/2203.03366> (read)
- [BarthelMiao2025] Thomas Barthel, Qiang Miao. Absence of barren plateaus and scaling of gradients in the energy optimization of isometric tensor network states. Commun. Math. Phys. 406, 86, 2025. <https://arxiv.org/abs/2304.00161> (abstract)
- [BenArous2021] Gérard Ben Arous, Reza Gheissari, Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. JMLR 22(106):1-51, 2021. <https://arxiv.org/abs/2003.10409> (read)
- [BenArous2024a] Gérard Ben Arous, Cédric Gerbelot, Vanessa Piccolo. Stochastic gradient descent in high dimensions for multi-spiked tensor PCA. arXiv 2410.18162, 2024 (v2 2025). <https://arxiv.org/abs/2410.18162> (read)
- [BenArous2024b] Gérard Ben Arous, Cédric Gerbelot, Vanessa Piccolo. Permutation recovery of spikes in noisy high-dimensional tensor estimation. arXiv 2412.14650, 2024; to appear in Stochastic Analysis and Applications, Springer Proceedings in Mathematics & Statistics (per arXiv). <https://arxiv.org/abs/2412.14650> (read)
- [Berthier2023] Raphaël Berthier. Incremental Learning in Diagonal Linear Networks. JMLR 24(171):1-26, 2023. <https://arxiv.org/abs/2208.14673> (read)
- [BerthierMZ2024] Raphaël Berthier, Andrea Montanari, Kangjie Zhou. Learning time-scales in two-layers neural networks. Foundations of Computational Mathematics, 2024, DOI 10.1007/s10208-024-09664-9. <https://arxiv.org/abs/2303.00055> (read)
- [Bietti2023] Alberto Bietti, Joan Bruna, Loucas Pillaud-Vivien. On Learning Gaussian Multi-index Models with Gradient Flow. arXiv 2310.19793, 2023. <https://arxiv.org/abs/2310.19793> (read)
- [BoixAdsera2023] Enric Boix-Adserà, Etai Littwin, Emmanuel Abbe, Samy Bengio, Joshua Susskind. Transformers learn through gradual rank increase. NeurIPS 2023. <https://arxiv.org/abs/2306.07042> (read)
- [Bordelon2024a] Blake Bordelon, Lorenzo Noci, Mufan Bill Li, Boris Hanin, Cengiz Pehlevan. Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit. ICLR 2024. <https://arxiv.org/abs/2309.16620> (abstract)
- [Bordelon2024b] Blake Bordelon, Hamza Tahir Chaudhry, Cengiz Pehlevan. Infinite Limits of Multi-head Transformer Dynamics. NeurIPS 2024. <https://arxiv.org/abs/2405.15712> (abstract)
- [BordelonPehlevan2022] Blake Bordelon, Cengiz Pehlevan. Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks. NeurIPS 2022. <https://arxiv.org/abs/2205.09653> (read)
- [Boursier2022] Etienne Boursier, Loucas Pillaud-Vivien, Nicolas Flammarion. Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs. NeurIPS 2022 (venue not re-verified; arXiv v3, 2026, corrects Prop. 1). <https://arxiv.org/abs/2206.00939> (read)
- [Braun2022] Lukas Braun, Clémentine Dominé, James Fitzgerald, Andrew Saxe. Exact learning dynamics of deep linear networks with prior knowledge. NeurIPS 2022. <https://papers.nips.cc/paper_files/paper/2022/hash/2b3bb2c95195130977a51b3bb251c40a-Abstract-Conference.html> (abstract)
- [Ceruti2021] Gianluca Ceruti, Christian Lubich, Hanna Walach. Time integration of tree tensor networks. SIAM J. Numer. Anal. 59(1):289-313, 2021. <https://arxiv.org/abs/2002.11392> (abstract)
- [Ceruti2023] Gianluca Ceruti, Christian Lubich, Dominik Sulz. Rank-adaptive time integration of tree tensor networks. SIAM J. Numer. Anal. 61(1):194-222, 2023. <https://arxiv.org/abs/2201.10291> (abstract)
- [CerutiLubich2022] Gianluca Ceruti, Christian Lubich. An unconventional robust integrator for dynamical low-rank approximation. BIT Numer. Math. 62(1):23-44, 2022. <https://arxiv.org/abs/2010.02022> (abstract)
- [CerveroMartin2023] Enrique Cervero Martín, Kirill Plekhanov, Michael Lubasch. Barren plateaus in quantum tensor network optimization. Quantum 7, 974, 2023. <https://arxiv.org/abs/2209.00292> (abstract)
- [ChenBarthel2024] Hao Chen, Thomas Barthel. Machine learning with tree tensor networks, CP rank constraints, and tensor dropout. IEEE Trans. Pattern Anal. Mach. Intell. 46, 7825, 2024. <https://arxiv.org/abs/2305.19440> (read)
- [ChenKunin2023] Feng Chen, Daniel Kunin, Atsushi Yamamura, Surya Ganguli. Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks. NeurIPS 2023. <https://arxiv.org/abs/2306.04251> (read)
- [ChenLiLu2020] Ziang Chen, Yingzhou Li, Jianfeng Lu. Tensor Ring Decomposition: Optimization Landscape and One-loop Convergence of Alternating Least Squares. SIAM J. Matrix Anal. Appl. 41(3):1416-1442, 2020. <https://arxiv.org/abs/1905.07101> (read)
- [Cheng2019] Song Cheng, Lei Wang, Tao Xiang, Pan Zhang. Tree Tensor Networks for Generative Modeling. Phys. Rev. B 99, 155131, 2019. <https://arxiv.org/abs/1901.02217> (read)
- [Chizat2019] Lénaïc Chizat, Edouard Oyallon, Francis Bach. On Lazy Training in Differentiable Programming. NeurIPS 2019. <https://arxiv.org/abs/1812.07956> (read)
- [Chizat2024] Lénaïc Chizat, Maria Colombo, Xavier Fernández-Real, Alessio Figalli. Infinite-width limit of deep linear neural networks. Comm. Pure Appl. Math. 77(10):3958-4007, 2024. <https://arxiv.org/abs/2211.16980> (read)
- [ChizatBach2018] Lénaïc Chizat, Francis Bach. On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport. NeurIPS 2018. <https://arxiv.org/abs/1805.09545> (abstract)
- [Choraria2022] Moulik Choraria, Leello Tadesse Dadi, Grigorios Chrysos, Julien Mairal, Volkan Cevher. The Spectral Bias of Polynomial Neural Networks. ICLR 2022. <https://arxiv.org/abs/2202.13473> (read)
- [Cohen2021] Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar. Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. ICLR 2021. <https://arxiv.org/abs/2103.00065> (abstract)
- [CohenSharirShashua2016] Nadav Cohen, Or Sharir, Amnon Shashua. On the Expressive Power of Deep Learning: A Tensor Analysis. COLT 2016, pp. 698-728. <https://arxiv.org/abs/1509.05009> (metadata only)
- [Corlouer2026] Guillaume Corlouer, Avi Semler, Alexander Strang, Alexander Gietelink Oldenziel. Stochastic Gradient Descent in the Saddle-to-Saddle Regime of Deep Linear Networks. arXiv 2604.06366, 2026. <https://arxiv.org/abs/2604.06366> (read)
- [deSilvaLim2008] Vin de Silva, Lek-Heng Lim. Tensor rank and the ill-posedness of the best low-rank approximation problem. SIAM J. Matrix Anal. Appl. 30(3):1084-1127, 2008. <https://arxiv.org/abs/math/0607647> (metadata only)
- [Domine2025] Clémentine C. J. Dominé, Nicolas Anguita, Alexandra M. Proca, Lukas Braun, Daniel Kunin, Pedro A. M. Mediano, Andrew M. Saxe. From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks. ICLR 2025 (venue not re-verified). <https://arxiv.org/abs/2409.14623> (read)
- [DuHu2019] Simon S. Du, Wei Hu. Width Provably Matters in Optimization for Deep Linear Neural Networks. ICML 2019, PMLR 97:1655-1664. <https://arxiv.org/abs/1901.08572> (abstract)
- [DuHuLee2018] Simon S. Du, Wei Hu, Jason D. Lee. Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are Automatically Balanced. NeurIPS 2018. <https://arxiv.org/abs/1806.00900> (read)
- [Even2023] Mathieu Even, Scott Pesme, Suriya Gunasekar, Nicolas Flammarion. (S)GD over Diagonal Linear Networks: Implicit Regularisation, Large Stepsizes and Edge of Stability (NeurIPS title: "... Implicit bias, Large Stepsizes and Edge of Stability"). NeurIPS 2023. <https://arxiv.org/abs/2302.08982> (read)
- [FrandsenGe2022] Abraham Frandsen, Rong Ge. Optimization landscape of Tucker decomposition. Math. Program. 193(2):687-712, 2022. <https://arxiv.org/abs/2006.16297> (read)
- [Furman2026] Zach Furman, Stephan Wäldchen, Yangda Bei, Liam Hodgkinson. Benign Loss Landscapes Can Coexist with Worst-Case Hardness. arXiv 2609.13057, 2026. <https://arxiv.org/abs/2609.13057> (read)
- [Ge2015] Rong Ge, Furong Huang, Chi Jin, Yang Yuan. Escaping From Saddle Points --- Online Stochastic Gradient for Tensor Decomposition. COLT 2015, PMLR 40:797-842. <https://arxiv.org/abs/1503.02101> (abstract)
- [Ge2021] Rong Ge, Yunwei Ren, Xiang Wang, Mo Zhou. Understanding Deflation Process in Over-parametrized Tensor Decomposition. NeurIPS 2021. <https://arxiv.org/abs/2106.06573> (read)
- [GeMa2017] Rong Ge, Tengyu Ma. On the Optimization Landscape of Tensor Decompositions. NeurIPS 2017. <https://arxiv.org/abs/1706.05598> (abstract)
- [Geiger2020] Mario Geiger, Stefano Spigler, Arthur Jacot, Matthieu Wyart. Disentangling feature and lazy training in deep neural networks. J. Stat. Mech. (2020) 113301. <https://arxiv.org/abs/1906.08034> (read)
- [Ghorbani2019] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, Andrea Montanari. Limitations of Lazy Training of Two-layers Neural Networks (NeurIPS title ends "...Neural Network"). NeurIPS 2019. <https://arxiv.org/abs/1906.08899> (read)
- [Ghosh2025] Avrajit Ghosh, Soo Min Kwon, Rongrong Wang, Saiprasad Ravishankar, Qing Qu. Learning Dynamics of Deep Linear Networks Beyond the Edge of Stability. ICLR 2025. <https://arxiv.org/abs/2502.20531> (read)
- [Gidel2019] Gauthier Gidel, Francis Bach, Simon Lacoste-Julien. Implicit Regularization of Discrete Gradient Dynamics in Linear Neural Networks. NeurIPS 2019. <https://arxiv.org/abs/1904.13262> (read)
- [Gissin2020] Daniel Gissin, Shai Shalev-Shwartz, Amit Daniely. The Implicit Bias of Depth: How Incremental Learning Drives Generalization. ICLR 2020. <https://arxiv.org/abs/1909.12051> (read)
- [Glasser2019] Ivan Glasser, Ryan Sweke, Nicola Pancotti, Jens Eisert, J. Ignacio Cirac. Expressive power of tensor-network factorizations for probabilistic modeling, with applications from hidden Markov models to quantum machine learning. NeurIPS 2019. <https://arxiv.org/abs/1907.03741> (metadata only)
- [Grasedyck2010] Lars Grasedyck. Hierarchical Singular Value Decomposition of Tensors. SIAM J. Matrix Anal. Appl. 31(4):2029-2054, 2010. <https://doi.org/10.1137/090764189> (read)
- [Grasedyck2024] Lars Grasedyck, Sebastian Krämer, Dieter Moser. Stable Truncation and Root-Independent Normalization of Tree Tensor Networks. In R. DeVore, A. Kunoth (eds.), Multiscale, Nonlinear and Adaptive Approximation II, Springer, 2024, pp. 231-265. <https://doi.org/10.1007/978-3-031-75802-7_12> (abstract)
- [GuoDraper2021a] Erdong Guo, David Draper. Infinitely Wide Tensor Networks as Gaussian Process. arXiv 2101.02333, 2021. <https://arxiv.org/abs/2101.02333> (read)
- [GuoDraper2021b] Erdong Guo, David Draper. Neural Tangent Kernel of Matrix Product States: Convergence and Applications. arXiv 2111.14046, 2021. <https://arxiv.org/abs/2111.14046> (read)
- [HackbuschKuhn2009] W. Hackbusch, S. Kühn. A New Scheme for the Tensor Representation. J. Fourier Anal. Appl. 15(5):706-722, 2009. <https://doi.org/10.1007/s00041-009-9094-9> (abstract)
- [Haegeman2016] Jutho Haegeman, Christian Lubich, Ivan Oseledets, Bart Vandereycken, Frank Verstraete. Unifying time evolution and optimization with matrix product states. Phys. Rev. B 94, 165116, 2016. <https://arxiv.org/abs/1408.5056> (abstract)
- [Han2018] Zhao-Yu Han, Jun Wang, Heng Fan, Lei Wang, Pan Zhang. Unsupervised Generative Modeling Using Matrix Product States. Phys. Rev. X 8, 031012, 2018. <https://arxiv.org/abs/1709.01662> (read)
- [HaninNica2020] Boris Hanin, Mihai Nica. Finite Depth and Width Corrections to the Neural Tangent Kernel. ICLR 2020. <https://arxiv.org/abs/1909.05989> (abstract)
- [HaninNica2020b] Boris Hanin, Mihai Nica. Products of Many Large Random Matrices and Gradients in Deep Neural Networks. Commun. Math. Phys. 376(1):287-322, 2020. <https://arxiv.org/abs/1812.05994> (abstract)
- [Harada2025] Kenji Harada, Tsuyoshi Okubo, Naoki Kawashima. Tensor tree learns hidden relational structures in data to construct generative models. Mach. Learn.: Sci. Technol. 6, 025002, 2025. <https://arxiv.org/abs/2408.10669> (metadata only)
- [Hariz2022] Kais Hariz, Hachem Kadri, Stéphane Ayache, Maher Moakher, Thierry Artières. Implicit Regularization with Polynomial Growth in Deep Tensor Factorization. ICML 2022, PMLR 162:8484-8501. <https://arxiv.org/abs/2207.08942> (read)
- [Hariz2024] Kais Hariz, Hachem Kadri, Stéphane Ayache, Maher Moakher, Thierry Artières. Implicit Regularization in Deep Tucker Factorization: Low-Rankness via Structured Sparsity. AISTATS 2024, PMLR 238:2359-2367. <https://proceedings.mlr.press/v238/hariz24a.html> (read)
- [HillarLim2013] Christopher Hillar, Lek-Heng Lim. Most tensor problems are NP-hard. J. ACM 60(6), 2013. <https://arxiv.org/abs/0911.1393> (metadata only)
- [Hofmann2026] Thomas Hofmann. The Map Behind the Flow: Finite-Step Gradient Descent as a Dynamical System. arXiv 2607.04993, 2026. <https://arxiv.org/abs/2607.04993> (read)
- [Jacot2018] Arthur Jacot, Franck Gabriel, Clément Hongler. Neural Tangent Kernel: Convergence and Generalization in Neural Networks. NeurIPS 2018. <https://arxiv.org/abs/1806.07572> (abstract)
- [Jacot2021] Arthur Jacot, François Ged, Berfin Simsek, Clément Hongler, Franck Gabriel. Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity. arXiv 2106.15933, 2021 (v2 2022; no peer-reviewed venue found). <https://arxiv.org/abs/2106.15933> (read)
- [Jacot2022] Arthur Jacot. Implicit Bias of Large Depth Networks: a Notion of Rank for Nonlinear Functions. arXiv 2209.15055, 2022 (venue not checked). <https://arxiv.org/abs/2209.15055> (metadata only)
- [Jin2023] Jikai Jin, Zhiyuan Li, Kaifeng Lyu, Simon S. Du, Jason D. Lee. Understanding Incremental Learning of Gradient Descent: A Fine-grained Analysis of Matrix Sensing. ICML 2023, PMLR 202:15200-15238. <https://arxiv.org/abs/2301.11500> (abstract)
- [Karnik2025] Santhosh Karnik, Anna Veselovska, Mark Iwen, Felix Krahmer. Implicit Regularization for Tubal Tensor Factorizations via Gradient Descent. ICML 2025, PMLR 267:29148-29204. <https://arxiv.org/abs/2410.16247> (read)
- [Kawaguchi2016] Kenji Kawaguchi. Deep Learning without Poor Local Minima. NeurIPS 2016. <https://arxiv.org/abs/1605.07110> (read)
- [Khrulkov2018] Valentin Khrulkov, Alexander Novikov, Ivan Oseledets. Expressive power of recurrent neural networks. ICLR 2018. <https://arxiv.org/abs/1711.00811> (metadata only)
- [KimLee2026] Hee-Sung Kim, Sungyoon Lee. Gradient Descent with Large Step Size Restores Symmetry in Deep Linear Networks with Multi-Pathway. ICML 2026, PMLR 306:58124-58155. <https://proceedings.mlr.press/v306/kim26aj.html> (abstract)
- [Klug2026] Nikolas Klug, Michael Ulbrich, André Uschmajew, Marius Willner. Natural Riemannian gradient for learning functional tensor networks. arXiv 2604.09263, 2026. <https://arxiv.org/abs/2604.09263> (abstract)
- [Kramer2019] Sebastian Krämer. A Geometric Description of Feasible Singular Values in the Tensor Train Format. SIAM J. Matrix Anal. Appl. 40(3):1153-1178, 2019. <https://arxiv.org/abs/1701.08437> (abstract)
- [Kreisler2023] Itai Kreisler, Mor Shpigel Nacson, Daniel Soudry, Yair Carmon. Gradient Descent Monotonically Decreases the Sharpness of Gradient Flow Solutions in Scalar Networks and Beyond. ICML 2023, PMLR 202:17684-17744. <https://arxiv.org/abs/2305.13064> (abstract)
- [Kressner2014] Daniel Kressner, Michael Steinlechner, Bart Vandereycken. Low-rank tensor completion by Riemannian optimization. BIT Numer. Math. 54(2):447-468, 2014. <https://doi.org/10.1007/s10543-013-0455-z> (abstract, from snippet)
- [KumarHaupt2024] Akshay Kumar, Jarvis Haupt. Directional Convergence Near Small Initializations and Saddles in Two-Homogeneous Neural Networks. TMLR, 06/2024. <https://arxiv.org/abs/2402.09226> (abstract)
- [KumarHaupt2025a] Akshay Kumar, Jarvis Haupt. Early Directional Convergence in Deep Homogeneous Neural Networks for Small Initializations. TMLR, 2025 (month not confirmed). <https://arxiv.org/abs/2403.08121> (read)
- [KumarHaupt2025b] Akshay Kumar, Jarvis Haupt. Towards Understanding Gradient Flow Dynamics of Homogeneous Neural Networks Beyond the Origin. JMLR 26(269):1-83, 2025. <https://arxiv.org/abs/2502.15952> (read)
- [KumarHaupt2025c] Akshay Kumar, Jarvis Haupt. Learning Neural Networks by Neuron Pursuit. arXiv 2509.12154, 2025 (v2 2026). <https://arxiv.org/abs/2509.12154> (read)
- [Kunin2021] Daniel Kunin, Javier Sagastuy-Brena, Surya Ganguli, Daniel L. K. Yamins, Hidenori Tanaka. Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics. ICLR 2021. <https://arxiv.org/abs/2012.04728> (metadata only)
- [Kunin2024] Daniel Kunin, Allan Raventós, Clémentine Dominé, Feng Chen, David Klindt, Andrew Saxe, Surya Ganguli. Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning. NeurIPS 2024. <https://arxiv.org/abs/2406.06158> (read)
- [Kunin2025] Daniel Kunin, Giovanni Luca Marchetti, Feng Chen, Dhruva Karkada, James B. Simon, Michael R. DeWeese, Surya Ganguli, Nina Miolane. Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks. NeurIPS 2025. <https://arxiv.org/abs/2506.06489> (read)

- [Lampinen2019] Andrew K. Lampinen, Surya Ganguli. An analytic theory of generalization dynamics and transfer learning in deep linear networks. ICLR 2019. <https://arxiv.org/abs/1809.10374> (metadata only; not reviewed)
- [Landsberg2011] J. M. Landsberg, Yang Qi, Ke Ye. On the geometry of tensor network states. arXiv 1105.4449, 2011 (v2 2012; "to appear in QIC", journal details not confirmed). <https://arxiv.org/abs/1105.4449> (read)
- [Lee2019] Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, Jeffrey Pennington. Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. NeurIPS 2019. <https://arxiv.org/abs/1902.06720> (read)
- [Levin2024] Eitan Levin, Joe Kileel, Nicolas Boumal. The effect of smooth parametrizations on nonconvex optimization landscapes. Math. Program. 209(1-2):63-111 (online 2024, print 2025). <https://arxiv.org/abs/2207.03512> (read)
- [Levine2018] Yoav Levine, David Yakira, Nadav Cohen, Amnon Shashua. Deep Learning and Quantum Entanglement: Fundamental Connections with Implications to Network Design. ICLR 2018 (venue not re-verified; arXiv lists none). <https://arxiv.org/abs/1704.01552> (metadata only)
- [Lewkowycz2020] Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, Guy Gur-Ari. The large learning rate phase of deep learning: the catapult mechanism. arXiv 2003.02218, 2020. <https://arxiv.org/abs/2003.02218> (read)
- [LiLuoLyu2021] Zhiyuan Li, Yuping Luo, Kaifeng Lyu. Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning. ICLR 2021. <https://arxiv.org/abs/2012.09839> (read)
- [LiNicaRoy2021] Mufan Bill Li, Mihai Nica, Daniel M. Roy. The Future is Log-Gaussian: ResNets and Their Infinite-Depth-and-Width Limit at Initialization. NeurIPS 2021. <https://arxiv.org/abs/2106.04013> (abstract)
- [LiangMontufar2025] Shuang Liang, Guido Montúfar. Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region. arXiv 2509.25351, 2025 (v3 2026; venue not checked). <https://arxiv.org/abs/2509.25351> (abstract)
- [LindseyMenon2025] Kathryn Lindsey, Govind Menon. Regularization Implies balancedness in the deep linear network. arXiv 2511.01137, 2025. <https://arxiv.org/abs/2511.01137> (abstract)
- [Littwin2020] Etai Littwin, Tomer Galanti, Lior Wolf, Greg Yang. On Infinite-Width Hypernetworks. NeurIPS 2020. <https://arxiv.org/abs/2003.12193> (read)
- [Liu2019] Ding Liu, Shi-Ju Ran, Peter Wittek, Cheng Peng, Raul Blázquez García, Gang Su, Maciej Lewenstein. Machine Learning by Unitary Tensor Network of Hierarchical Tree Structure. New J. Phys. 21, 073059, 2019. <https://arxiv.org/abs/1710.04833> (abstract)
- [Liu2022] Zidu Liu, Li-Wei Yu, L.-M. Duan, Dong-Ling Deng. The Presence and Absence of Barren Plateaus in Tensor-network Based Machine Learning. Phys. Rev. Lett. 129, 270501, 2022. <https://arxiv.org/abs/2108.08312> (read)
- [Lubich2013] Christian Lubich, Thorsten Rohwedder, Reinhold Schneider, Bart Vandereycken. Dynamical Approximation by Hierarchical Tucker and Tensor-Train Tensors. SIAM J. Matrix Anal. Appl. 34(2):470-494, 2013. <https://doi.org/10.1137/120885723> (abstract)
- [Lubich2015] Christian Lubich, Ivan V. Oseledets, Bart Vandereycken. Time integration of tensor trains. SIAM J. Numer. Anal. 53(2):917-941, 2015. <https://arxiv.org/abs/1407.2042> (read)
- [MaGuoFattahi2022] Jianhao Ma, Lingjun Guo, Salar Fattahi. Behind the Scenes of Gradient Descent: A Trajectory Analysis via Basis Function Decomposition. arXiv 2210.00346, 2022 (venue not confirmed). <https://arxiv.org/abs/2210.00346> (read)
- [MaLavaeiSojoudi2023] Ziye Ma, Javad Lavaei, Somayeh Sojoudi. Algorithmic Regularization in Tensor Optimization: Towards a Lifted Approach in Matrix Sensing. NeurIPS 2023. <https://arxiv.org/abs/2310.15549> (abstract)
- [Marcotte2023] Sibylle Marcotte, Rémi Gribonval, Gabriel Peyré. Abide by the Law and Follow the Flow: Conservation Laws for Gradient Flows. NeurIPS 2023. <https://arxiv.org/abs/2307.00144> (read)
- [MarionChizat2024] Pierre Marion, Lénaïc Chizat. Deep linear networks for regression are implicitly regularized towards flat minima. NeurIPS 2024. <https://arxiv.org/abs/2405.13456> (abstract)
- [MasotLlima2024] Sergi Masot-Llima, Artur Garcia-Saez. Advantages of density in tensor network geometries for gradient based training. arXiv 2412.17497, 2024. <https://arxiv.org/abs/2412.17497> (read)
- [MataAli2023] Alejandro Mata Ali, Iñigo Perez Delgado, Marina Ristol Roura, Aitor Moreno Fdez. de Leceta. Efficient Finite Initialization with Partial Norms for Tensorized Neural Networks and Tensor Networks Algorithms. arXiv 2309.06577, 2023 (v5 2026). <https://arxiv.org/abs/2309.06577> (read)
- [MeiMontanariNguyen2018] Song Mei, Andrea Montanari, Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. PNAS 115(33), 2018. <https://arxiv.org/abs/1804.06561> (abstract)
- [MiaoBarthel2024] Qiang Miao, Thomas Barthel. Isometric tensor network optimization for extensive Hamiltonians is free of barren plateaus. Phys. Rev. A 109, L050402, 2024. <https://arxiv.org/abs/2304.14320> (abstract)
- [Milanesi2021] Paolo Milanesi, Hachem Kadri, Stéphane Ayache, Thierry Artières. Implicit Regularization in Deep Tensor Factorization. IJCNN 2021. <https://arxiv.org/abs/2105.01346> (read)
- [Min2021] Hancheng Min, Salma Tarmoun, René Vidal, Enrique Mallada. On the Explicit Role of Initialization on the Convergence and Implicit Bias of Overparametrized Linear Networks. ICML 2021, pp. 7760-7768 (arXiv title: "Convergence and Implicit Bias of Gradient Flow on Overparametrized Linear Networks"). <https://arxiv.org/abs/2105.06351> (abstract)
- [Moroshko2020] Edward Moroshko, Blake Woodworth, Suriya Gunasekar, Jason D. Lee, Nathan Srebro, Daniel Soudry. Implicit Bias in Deep Linear Classification: Initialization Scale vs Training Accuracy. NeurIPS 2020. <https://arxiv.org/abs/2007.06738> (read)
- [Nacson2022] Mor Shpigel Nacson, Kavya Ravichandran, Nathan Srebro, Daniel Soudry. Implicit Bias of the Step Size in Linear Diagonal Neural Networks. ICML 2022, PMLR 162:16270-16295. <https://proceedings.mlr.press/v162/nacson22a.html> (abstract)
- [NguyenPham2023] Phan-Minh Nguyen, Huy Tuan Pham. A Rigorous Framework for the Mean Field Limit of Multilayer Neural Networks. Mathematical Statistics and Learning 6(3):201-357, 2023. <https://arxiv.org/abs/2001.11443> (abstract)
- [Novikov2017] Alexander Novikov, Mikhail Trofimov, Ivan Oseledets. Exponential Machines. ICLR 2017 workshop track (arXiv 2016). <https://arxiv.org/abs/1605.03795> (read)
- [Oseledets2011] I. V. Oseledets. Tensor-Train Decomposition. SIAM J. Sci. Comput. 33(5):2295-2317, 2011. <https://doi.org/10.1137/090752286> (abstract)
- [PaezVelasco2026] Pablo Páez-Velasco. Tensor network manifolds and Riemannian fundamental theorem for tensor networks. arXiv 2606.14613, 2026. <https://arxiv.org/abs/2606.14613> (abstract)
- [Pesme2021] Scott Pesme, Loucas Pillaud-Vivien, Nicolas Flammarion. Implicit Bias of SGD for Diagonal Linear Networks: a Provable Benefit of Stochasticity. NeurIPS 2021. <https://arxiv.org/abs/2106.09524> (read)
- [PesmeFlammarion2023] Scott Pesme, Nicolas Flammarion. Saddle-to-Saddle Dynamics in Diagonal Linear Networks. NeurIPS 2023; also J. Stat. Mech. (2024) 104016. <https://arxiv.org/abs/2304.00488> (read)
- [Pomarico2025] Domenico Pomarico, Alfonso Monaco, Giuseppe Magnifico, Antonio Lacalamita, Ester Pantaleo, Loredana Bellantuono, Sabina Tangaro, Tommaso Maggipinto, Marianna La Rocca, Ernesto Picardi, Nicola Amoroso, Graziano Pesole, Sebastiano Stramaglia, Roberto Bellotti. Grokking as an entanglement transition in tensor network machine learning. arXiv 2503.10483, 2025. <https://arxiv.org/abs/2503.10483> (read)
- [Potapczynski2024] Andres Potapczynski, Shikai Qiu, Marc Finzi, Christopher Ferri, Zixi Chen, Micah Goldblum, C. Bayan Bruss, Christopher De Sa, Andrew Gordon Wilson. Searching for Efficient Linear Layers over a Continuous Space of Structured Matrices. NeurIPS 2024. <https://arxiv.org/abs/2410.02117> (abstract)
- [Qin2024] Zhen Qin, Michael B. Wakin, Zhihui Zhu. Guaranteed Nonconvex Factorization Approach for Tensor Train Recovery. JMLR 25(383):1-48, 2024. <https://arxiv.org/abs/2401.02592> (abstract)
- [Qiu2024] Shikai Qiu, Andres Potapczynski, Marc Finzi, Micah Goldblum, Andrew Gordon Wilson. Compute Better Spent: Replacing Dense Layers with Structured Matrices. ICML 2024. <https://arxiv.org/abs/2406.06248> (read)
- [Rakhshan2020] Beheshteh T. Rakhshan, Guillaume Rabusseau. Tensorized Random Projections. AISTATS 2020, PMLR 108:3306-3316. <https://arxiv.org/abs/2003.05101> (read)
- [RawalDeWeese2026] Divit Rawal, Michael R. DeWeese. A Theory of Saddle Escape in Deep Nonlinear Networks. arXiv 2605.01288, 2026. <https://arxiv.org/abs/2605.01288> (read)
- [Razin2021] Noam Razin, Asaf Maman, Nadav Cohen. Implicit Regularization in Tensor Factorization. ICML 2021, PMLR 139:8913-8924. <https://arxiv.org/abs/2102.09972> (read)
- [Razin2022] Noam Razin, Asaf Maman, Nadav Cohen. Implicit Regularization in Hierarchical Tensor Factorization and Deep Convolutional Neural Networks. ICML 2022, PMLR 162:18422-18462. <https://arxiv.org/abs/2201.11729> (read)
- [RazinCohen2020] Noam Razin, Nadav Cohen. Implicit Regularization in Deep Learning May Not Be Explainable by Norms. NeurIPS 2020. <https://arxiv.org/abs/2005.06398> (abstract)
- [Ren2025] Yunwei Ren, Eshaan Nichani, Denny Wu, Jason D. Lee. Emergence and scaling laws in SGD learning of shallow neural networks. NeurIPS 2025. <https://arxiv.org/abs/2504.19983> (read)
- [RotskoffVE2018] Grant M. Rotskoff, Eric Vanden-Eijnden. Parameters as interacting particles: long time convergence and asymptotic error scaling of neural networks. NeurIPS 2018; journal version "Trainability and Accuracy of Artificial Neural Networks: An Interacting Particle System Approach", Comm. Pure Appl. Math. 75(9):1889-1935, 2022. <https://arxiv.org/abs/1805.00915> (abstract)
- [Sasaki2026] Daiki Sasaki, Chih-Chieh Chen, Tomah Sogabe. Tree Tensor Network Reservoir Computing: Hierarchical Ensemble with Invariant Phase Boundaries. arXiv 2607.24127, 2026. <https://arxiv.org/abs/2607.24127> (read)
- [Saxe2014] Andrew M. Saxe, James L. McClelland, Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR 2014. <https://arxiv.org/abs/1312.6120> (read)
- [Saxe2019] Andrew M. Saxe, James L. McClelland, Surya Ganguli. A mathematical theory of semantic development in deep neural networks. PNAS 116(23):11537-11546, 2019. <https://arxiv.org/abs/1810.10531> (abstract)
- [Saxe2022] Andrew Saxe, Shagun Sodhani, Sam Jay Lewallen. The Neural Race Reduction: Dynamics of Abstraction in Gated Networks. ICML 2022, PMLR 162:19287-19309. <https://proceedings.mlr.press/v162/saxe22a.html> (abstract)
- [Shamir2019] Ohad Shamir. Exponential Convergence Time of Gradient Descent for One-Dimensional Deep Linear Neural Networks. COLT 2019, PMLR 99:2691-2713. <https://arxiv.org/abs/1809.08587> (abstract)
- [ShiDuanVidal2006] Yaoyun Shi, Luming Duan, Guifré Vidal. Classical simulation of quantum many-body systems with a tree tensor network. Phys. Rev. A 74, 022320, 2006. <https://arxiv.org/abs/quant-ph/0511070> (read)
- [Simsek2021] Berfin Simsek, François Ged, Arthur Jacot, Francesco Spadaro, Clément Hongler, Wulfram Gerstner, Johanni Brea. Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances. ICML 2021 (PMLR 139). <https://arxiv.org/abs/2105.12221> (read)
- [Singh2026] Devender Singh. The Loss Does Not See the Basis, but Adam Does. arXiv 2608.05136, 2026. <https://arxiv.org/abs/2608.05136> (abstract)
- [SirignanoSpiliopoulos2020] Justin Sirignano, Konstantinos Spiliopoulos. Mean Field Analysis of Neural Networks: A Law of Large Numbers. SIAM J. Appl. Math. 80(2):725-752, 2020. <https://arxiv.org/abs/1805.01053> (abstract)
- [Stoudenmire2016] E. Miles Stoudenmire, David J. Schwab. Supervised Learning with Quantum-Inspired Tensor Networks (NeurIPS title: "Supervised Learning with Tensor Networks"). NeurIPS 2016. <https://arxiv.org/abs/1605.05775> (read)
- [Stoudenmire2018] E. M. Stoudenmire. Learning Relevant Features of Data with Multi-scale Tensor Networks. Quantum Sci. Technol. 3, 034003, 2018. <https://arxiv.org/abs/1801.00315> (metadata only)
- [Strashko2022] Artem Strashko, E. Miles Stoudenmire. Generalization and Overfitting in Matrix Product State Machine Learning Architectures. arXiv 2208.04372, 2022. <https://arxiv.org/abs/2208.04372> (read)
- [Tang2025] Xun Tang, Yuehaw Khoo, Lexing Ying. Initialization and training of matrix product state probabilistic models. arXiv 2505.06419, 2025. <https://arxiv.org/abs/2505.06419> (read)
- [TindallFishman2023] Joseph Tindall, Matthew T. Fishman. Gauging tensor networks with belief propagation. SciPost Phys. 15, 222, 2023. <https://arxiv.org/abs/2306.17837> (read)
- [Tu2024] Zhenfeng Tu, Santiago Aranguri, Arthur Jacot. Mixed Dynamics In Linear Networks: Unifying the Lazy and Active Regimes. NeurIPS 2024. <https://arxiv.org/abs/2405.17580> (read)
- [UschmajewVandereycken2013] André Uschmajew, Bart Vandereycken. The geometry of algorithms using hierarchical tensors. Linear Algebra Appl. 439(1):133-166, 2013. <https://www.epfl.ch/labs/mathicse/wp-content/uploads/2018/10/12.2012_AU-BV.pdf> (read)
- [Vankadara2024] Leena Chennuru Vankadara, Jin Xu, Moritz Haas, Volkan Cevher. On Feature Learning in Structured State Space Models. NeurIPS 2024. <https://proceedings.neurips.cc/paper_files/paper/2024/hash/9c7eeda2dc98e61baa9a5884afd231bc-Abstract-Conference.html> (abstract)
- [WallDAguanno2021] Michael L. Wall, Giuseppe D'Aguanno. Tree tensor network classifiers for machine learning: from quantum-inspired to quantum-assisted. Phys. Rev. A 104, 042408, 2021. <https://arxiv.org/abs/2104.02249> (abstract)
- [Wang2020] Xiang Wang, Chenwei Wu, Jason D. Lee, Tengyu Ma, Rong Ge. Beyond Lazy Training for Over-parameterized Tensor Decomposition. NeurIPS 2020. <https://arxiv.org/abs/2010.11356> (read)
- [WangCZT2022] Yuqing Wang, Minshuo Chen, Tuo Zhao, Molei Tao. Large Learning Rate Tames Homogeneity: Convergence and Balancing Effect. ICLR 2022. <https://arxiv.org/abs/2110.03677> (abstract)
- [WangLiuYu2026] Shi-Tuan Wang, Zidu Liu, Li-Wei Yu. Gaussian Universality and Its Breakdown in Tensor-Network Machine Learning. arXiv 2610.06080, 2026. <https://arxiv.org/abs/2610.06080> (read)
- [WeselBatselier2025] Frederiek Wesel, Kim Batselier. Tensor Network-Constrained Kernel Machines as Gaussian Processes. AISTATS 2025, PMLR 258:2161-2169. <https://proceedings.mlr.press/v258/wesel25a.html> (read)
- [Woodworth2020] Blake Woodworth, Suriya Gunasekar, Jason D. Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, Nathan Srebro. Kernel and Rich Regimes in Overparametrized Models. COLT 2020, PMLR 125:3635-3673. <https://arxiv.org/abs/2002.09277> (read)
- [XuZiyin2025] Yizhou Xu, Liu Ziyin. Three Mechanisms of Feature Learning in a Linear Network. ICLR 2025 (venue not re-verified; arXiv lists none). <https://arxiv.org/abs/2401.07085> (read)
- [Yang2021] Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. NeurIPS 2021 (per arXiv journal-ref; arXiv 2022). <https://arxiv.org/abs/2203.03466> (abstract)
- [Yang2024] Greg Yang, Dingli Yu, Chen Zhu, Soufiane Hayou. Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks. ICLR 2024. <https://arxiv.org/abs/2310.02244> (abstract)
- [YangHu2021] Greg Yang, Edward J. Hu. Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks. ICML 2021. <https://arxiv.org/abs/2011.14522> (read)
- [YangLittwin2021] Greg Yang, Etai Littwin. Tensor Programs IIb: Architectural Universality of Neural Tangent Kernel Training Dynamics. ICML 2021. <https://arxiv.org/abs/2105.03703> (abstract)
- [YangSB2023] Greg Yang, James B. Simon, Jeremy Bernstein. A Spectral Condition for Feature Learning. arXiv 2310.17813, 2023 (v2 2024; no venue found). <https://arxiv.org/abs/2310.17813> (read)
- [Yarotsky2026] Dmitry Yarotsky, Eugene Golikov, Yaroslav Gusev. Gradient Flow Through Diagram Expansions: Learning Regimes and Explicit Solutions. ICML 2026, PMLR 306:147928-147978. <https://arxiv.org/abs/2602.04548> (read)
- [Zhang2025] Yedi Zhang, Aaditya K. Singh, Peter E. Latham, Andrew Saxe. Training Dynamics of In-Context Learning in Linear Attention. ICML 2025. <https://arxiv.org/abs/2501.16265> (read)
- [ZhangHK2026] Hao-Kai Zhang, Chenghong Zhu, Shuo Liu, Shi-Xin Zhang, Tao Xiang. Absence of poor local minima in matrix product states. arXiv 2606.09988, 2026. <https://arxiv.org/abs/2606.09988> (read)
- [ZhangSL2026] Yedi Zhang, Andrew Saxe, Peter E. Latham. Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures. ICLR 2026. <https://arxiv.org/abs/2512.20607> (read)
- [ZhangT2026] Tongcheng Zhang, Zhanpeng Zhou, Mingze Wang, Andi Han, Wei Huang, Taiji Suzuki, Junchi Yan. On the Learning Dynamics of Two-layer Linear Networks with Label Noise SGD. AAAI 2026 (oral, per arXiv comment). <https://arxiv.org/abs/2603.10397> (abstract)
- [Zhao2023] Bo Zhao, Iordan Ganev, Robin Walters, Rose Yu, Nima Dehmamy. Symmetries, flat minima, and the conserved quantities of gradient flow. ICLR 2023. <https://arxiv.org/abs/2210.17216> (metadata only)
- [Zhao2026] Jiajie Zhao, Jianxing Wang, Junjie Yang, Zhiwei Bai, Yaoyu Zhang. Gradient Flow Dynamics and Implicit Bias of Diagonal Linear Networks under Infinitesimal Initialization. ICML 2026, PMLR 306:162217-162242 (arXiv 2607.12332). <https://proceedings.mlr.press/v306/zhao26av.html> (read)
- [Zhou2026] Mo Zhou, Weihang Xu, Simon S. Du, Maryam Fazel. Learning Orthogonal Multi-Index Models Beyond Small Initialization: Incremental Learning, Competitive Dynamics and Symmetry. arXiv 2609.10879, 2026. <https://arxiv.org/abs/2609.10879> (abstract)
- [Ziyin2023] Liu Ziyin, Botao Li, Tomer Galanti, Masahito Ueda. Type-II Saddles and Probabilistic Stability of Stochastic Gradient Descent. arXiv 2303.13093, 2023 (v4 2024). <https://arxiv.org/abs/2303.13093> (abstract)
- [Ziyin2024] Liu Ziyin, Mingze Wang, Hongchao Li, Lei Wu. Parameter Symmetry and Noise Equilibrium of Stochastic Gradient Descent. NeurIPS 2024. <https://arxiv.org/abs/2402.07193> (abstract)
- [ZiyinLiMeng2022] Liu Ziyin, Botao Li, Xiangming Meng. Exact Solutions of a Deep Linear Network. NeurIPS 2022. <https://arxiv.org/abs/2202.04777> (abstract)

### 7.2 Unconfirmed, or metadata incomplete

Unconfirmed content (confidence "unconfirmed" in the sweeps):

- [Holtz2012] Sebastian Holtz, Thorsten Rohwedder, Reinhold Schneider. On manifolds of tensors of fixed TT-rank. Numer. Math. 120(4):701-731, 2012. <https://doi.org/10.1007/s00211-011-0419-7>. Metadata confirmed on Crossref; no abstract or text read; the results cited here come from [Lubich2015].
- [Arvanitakis2025] Dionysis Arvanitakis, Vaidehi Srinivas, Aravindan Vijayaraghavan. Guarantees for Alternating Least Squares in Overparameterized Tensor Decompositions. NeurIPS 2025. <https://openreview.net/forum?id=9FDErIfoVE>. Title, authors and venue confirmed; the claimed result (ALS converges globally from random init with over-parameterized rank O(r^2), improving O(r^7.5 log n)) was not checked.

Mentioned but with incomplete metadata (not reviewed; do not cite without checking):

- Kofidis and Regalia, SIAM J. Matrix Anal. Appl. 23(3):863-884, 2002: best rank-one approximation of supersymmetric tensors; their Ex. 4 has non-global local optima (as used by [Furman2026]). Title not recorded.
- Steinlechner, SIAM J. Sci. Comput. 38(5):S461-S484, 2016: Riemannian tensor-train completion. Title not recorded.
- Luo, Xu, Ma and Zhang, JMLR 22(71):1-47, 2021: phase diagram (linear, critical, condensed regimes) of two-layer ReLU networks at infinite width. Title not recorded.
- Abbe, Boix-Adserà and Misiakiewicz, COLT 2022, PMLR 178:4782-4887, arXiv 2202.08658: the merged-staircase predecessor of [Abbe2023]. Title not recorded.
- Langevin companion of [BenArous2024a]: arXiv 2408.06401 (abstract only). Title not recorded.
- Pomarico, Cilli, Monaco et al., "Transfer entropy and O-information to detect grokking in tensor network multi-class classification problems", Technologies 13(10), 438, 2025, arXiv 2507.23346: follow-up to [Pomarico2025]. Full author list not recorded.
- Simon et al., "On the Stepwise Nature of Self-Supervised Learning", ICML 2023, arXiv 2303.15438: existence verified, not reviewed; a possible gap in cluster 7.

