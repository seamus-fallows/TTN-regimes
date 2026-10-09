# Theory: escape from the origin (Q2, Q4; H1)

Status: v2, 2026-10-09 (theory agent), revised after independent verification (section 8). Every claim
is tagged **proven**, **derived** (assumptions stated), **heuristic**, **empirical** (numerical, this
note), **conjecture**, or **known** (with citation; keys as in `notes/literature.md` section 7). All
numbers come from the core library (`ttn.tree`, `ttn.network`, `ttn.teachers`, `ttn.losses`, after commit
57d5262) and scratch scripts outside the repo (`scratchpad/theory/escape/` for v1,
`scratchpad/theory/escape_v2/` for the v2 checks); the recipe of section 2.5 reproduces every escape time.
Quantitative predictions for the experiment agents are collected in section 9.

## 0. Results at a glance

| # | result | status |
| --- | --- | --- |
| E1 | Near the origin the direction $u=\theta/r$ follows the spherical gradient flow of $g$ in the time $d\tau=r^{K-2}dt$, independently of $r$ and $\sigma$; $d\ln r/d\tau=Kg(u)$; the leading-order blow-up time is $t^*=r_0^{-(K-2)}\int_0^\infty e^{-(K-2)K\int_0^\tau g}\,d\tau$. Exponent $K-2$ is exact; the prefactor is a random $O(1)$ number $\hat t_1(Z)$ fixed by the init, computable per init. | proven (directional convergence itself: known, [KumarHaupt2025a]) |
| E2 | $t^*\ge K^{K/2-1}r_{\rm m}^{-(K-2)}/((K-2)s_1)\ge t_{\rm KH}$, the Kumar-Haupt escape time with $\delta=\rho_0$ that PLAN.md Q2 uses. For fan-in init $t_{\rm KH}$ is low by factors 2.5-5 (binary4), 5-18 (MPS4) and 30-12000 ($K$ = 6-8). | proven (bound), empirical (factors) |
| E3 | $\max_{\lVert\theta\rVert=1}g=K^{-K/2}\lVert W^*\rVert_{\sigma,P}$ (spectral norm for the grouping $P$ of sites into cores), attained at balanced bond-rank-one configurations. | proven |
| E4 | Pattern teachers, distinct weights: the balanced bond-rank-one configuration of component $q$ is not critical if some $r$ differs from $q$ on one core only ($m_{qr}=1$); otherwise each $r$ that differs on exactly two cores ($m_{qr}=2$) gives $n_{qr}$ spherical-Hessian eigenvalues $K^{1-K/2}(s_r-s_q)$, with $n_{qr}=\chi_e$ if the two cores share bond $e$ and 1 if not; everything else is negative or gauge. So $q$ is a strict local max (mod gauge) iff every $m_{qr}=2$ partner is smaller; ties $s_r=s_q$ are decided at higher order; several partners on a shared core: piggyback graph (`notes/theory_saddles.md` 4.1, 4.5). Odeco: every component is a local max on MPS, HT, binary $N\ge6$; only the largest on binary4 ($n=1$). | derived, checked (33 Hessians) |
| E5 | At leading order only the teacher slices of the cores move ("reduced network"); the rest of $\theta$, which carries most of $\rho_0$, is frozen. On MPS (every core carries a site) different odeco components evolve independently. | proven |
| E6 | Under LO every reduced core's squared norm changes by the same convex amount $y(t)=2\int_0^tg$ (cores shrink only while $g<0$), and an end core keeps $(\hat h_0\cdot h)^2\ge\lVert h_0\rVert^2(1-\kappa_e)$, informative only where $\kappa_e\ll1$ (binary4 bottom ends with $\chi_b\gg d$; vacuous for the MPS root end). Empirically the ends keep most of their norm and direction; assuming they do, the escape is an imbalanced product mode with closed forms for binary4 (median $\hat t_1\approx\pi d/(2\sqrt{\chi_b})$, $\to\pi/2$ at full bond) and MPS4. | proven (identities), empirical (ends), derived given frozen ends (closed forms) |
| E7 | $\chi$ enters only through the reduced network. For MPS4 and $K\ge6$ the $\chi$-dependence is much weaker than the $\rho_0$ law (MPS4 log-log slope $-0.20$ vs $-0.43$ to $-0.47$); for binary4 it is only about 20% weaker, but at full bond $\hat t_1$ is independent of $d$ while $t_{\rm KH}\propto1/d$. | empirical, mechanism derived |
| E8 | Order on tensor-like trees: exact race on MPS ($\sigma$-independent $P(\text{larger first})=P(\hat t_{1,2}/\hat t_{1,1}>\rho)$), approximate race with shared internal cores (94-100% per-init agreement); $P\approx\Phi(\ln(1/\rho)/S)$ with $S$ tabulated; $\mathrm{sd}\ln\hat t_1\propto\sqrt{K-2}$ on MPS; $P\to\frac12$ as $K\to\infty$; $P\to1$ needs $d\to\infty$ as well as large $\chi$. | proven (MPS, LO), empirical (shared cores, $\sqrt{K-2}$), conjecture / heuristic (limits) |
| E9 | Matrix-like pairs (binary4; components differing on two cores of any tree): greedy as $\sigma\to0$ (for binary4 known: [Jacot2021] Thm 2); $P(\text{smaller first})\propto\sigma^{n(1-\rho)/\rho}$ with $n$ from E4. LO emulation: slopes ordered by $n$, roughly proportional to it, and 0.67-1.03 (typically 0.8) of the prediction over $\sigma=10^{-1}$-$10^{-8}$; full GF puts the smaller component first on 0.59-0.70 of the emulated cases. | known (binary4 limit), heuristic (exponent), checked (emulation, GF) |

**Consequences for PLAN.md.** Q2: PLAN.md compares $t_1$ with the Kumar-Haupt formula
$1/(K(K-2)g^*\delta^{K-2})$, $g^*=s_1K^{-K/2}$. With $\delta=\rho_0$ this is a strict lower bound on every
init (E2), low by factors 2.5-12000 that grow with $d$, $\chi$ and $K$. Use $t_1=\sigma^{-(K-2)}\hat t_1/s_1$
with $\hat t_1$ computed per init (2.5) or the medians of tables 1-2. The early-draft hypothesis "$t_1$
decreases with bond dimension through $\rho_0^2\sim\sigma^2\sum_v\chi_v$" is refuted on MPS and $K\ge6$
(much weaker dependence) and on binary4 by the $d$-independence at full bond. Q4/H1: confirmed for
tensor-like trees (MPS, HT, binary $N\ge6$ with odeco teachers: $\sigma$-independent random order,
probabilities in table 2); refuted for matrix-like pairs (greedy as $\sigma\to0$, slowly: tables 3-4).

## 1. Setting and notation

- $N$ sites, $K$ cores $G_v$, $\theta=(G_v)_v$, $W(\theta)$ the tree contraction; $r=\lVert\theta\rVert$
  (Euclidean norm of all core entries), $u=\theta/r$; $\rho_0=r(0)$. (A subscript $r$ indexes teacher
  components; the radius $r$ never carries an index.)
- Teacher $W^*=\sum_{r=1}^R s_rU_r$, $U_r=\bigotimes_iu_r^{(i)}$ with unit $u_r^{(i)}$. Pattern teacher:
  component $r$ uses basis vector $p_r(i)$ of an orthonormal basis of site $i$; odeco: $p_r(i)=r$.
  $s_1\ge s_2\ge\dots$; for two components the weights are $(1,\rho)$, $\rho<1$.
- Whitened loss $L=\tfrac12\lVert W^*\rVert^2-g(\theta)+\tfrac12\lVert W(\theta)\rVert^2$ with
  $g(\theta)=\langle W(\theta),W^*\rangle$, homogeneous of degree $K$ and linear in each core.
- Fan-in init $G_v=\sigma Z_v/\sqrt{F_v}$, $Z$ standard Gaussian, $F_v$ = product of $v$'s child-leg
  dimensions (physical and bond). $n_v$ = number of sites on core $v$; $\chi_e$ = bond dimension of edge
  $e$ after the library's cap $\min(\chi,d^{N_e},d^{N-N_e})$, $N_e$ = number of sites below $e$ (listed
  as `bonds`, root's 1 omitted).
- Leading-order (LO) flow: $\dot\theta=\nabla g$ (drops $\tfrac12\lVert W\rVert^2$, of relative size
  $\lVert W\rVert/s\sim r^K/s$). $t^*$ = its blow-up time. $t_1$ = first event of the full gradient flow
  (GF), i.e. first time some $a_r=\langle W,U_r\rangle/s_r$ crosses $\tfrac12$ (`notes/diagnostics.md`).
- Escape-time prefactor $\hat t_1(Z)$: the LO blow-up time from $\theta_0=Z$ ($\sigma=1$) with a
  single component of weight 1, so that $t^*=\sigma^{-(K-2)}\hat t_1/s$.
- "Naive" time: $t_{\rm KH}=K^{K/2-1}\rho_0^{-(K-2)}/((K-2)s_1)$, the Kumar-Haupt escape time
  $1/(K(K-2)g^*\delta^{K-2})$ with $g^*=s_1K^{-K/2}$ and $\delta=\rho_0$ ([KumarHaupt2025b]; PLAN.md Q2).
  $r_{\rm m}$ = norm of the variables that move under LO (4.1); "red" = $t_{\rm KH}$ with $r_{\rm m}$.
- Conserved gauge quantities (any loss that depends on $\theta$ through $W$, hence also LO):
  $Q_e=G_v^\top G_v-G_{p,(v)}G_{p,(v)}^\top$ (`notes/model.md` section 5).
- For components $q,r$ of a pattern teacher: $D_{qr}$ = sites where $p_q(i)\ne p_r(i)$; $m_{qr}$ = number
  of cores carrying sites of $D_{qr}$; $n_{qr}$ = number of unstable directions it creates (3.3).

## 2. Spherical-radial decomposition (task a)

### 2.1 Equations (proven)

Euler's identity $\langle\theta,\nabla g\rangle=Kg$ and $\nabla g(\theta)=r^{K-1}\nabla g(u)$ give
$\dot r=Kr^{K-1}g(u)$ and $\dot u=r^{K-2}[\nabla g(u)-Kg(u)u]$. With $d\tau=r^{K-2}dt$,
$$\frac{du}{d\tau}=\nabla_Sg(u):=\nabla g(u)-Kg(u)\,u,\qquad \frac{d\ln r}{d\tau}=Kg(u).$$
The direction follows the spherical (Riemannian) gradient flow of $g$, which does not involve $r$;
with $\theta_0=\sigma Z$ it does not involve $\sigma$ either. The conservation laws are automatic:
$Q_e(u(\tau))=Q_e(\theta_0)/r(\tau)^2$, so they bind at $r\approx\rho_0$ and fade as $r$ grows.

### 2.2 Blow-up time and lower bound (proven)

With $G(\tau)=\int_0^\tau g(u)\,d\tau'$, $r=\rho_0e^{KG(\tau)}$ and
$$t^*=\int_0^\infty r^{-(K-2)}\,d\tau=\rho_0^{-(K-2)}\int_0^\infty e^{-(K-2)KG(\tau)}\,d\tau .$$
For $\theta_0=\sigma Z$: $t^*=\sigma^{-(K-2)}\hat t_1(Z)/s$, so the escape exponent is exactly $K-2$ (all
cores count, including those without sites) and the prefactor is a random variable of the init.
Since $g\le g_{\max}=s_1K^{-K/2}$ on the sphere (3.1), $\ln r\le\ln\rho_0+Kg_{\max}\tau$ and
$$t^*\ \ge\ \frac{K^{K/2-1}}{(K-2)\,s_1}\,r_{\rm m}^{-(K-2)}\ \ge\ \frac{K^{K/2-1}}{(K-2)\,s_1}\,\rho_0^{-(K-2)}=t_{\rm KH},$$
with equality iff $u_0$ is already a maximizer (frozen variables do not affect the $t$-dynamics, 4.1).
There is no separate additive alignment phase (an early PLAN draft had
$t_1\approx\rho_0^{-(K-2)}[\tau_{\rm align}+K^{K/2-1}/((K-2)s_1)]$): the whole integral is "alignment",
because $g(u)$ starts near 0 ($g(u_0)=\langle W_0,W^*\rangle/\rho_0^K$ is tiny) and grows gradually.

### 2.3 Relation to known results

- **Saxe (known; consistency check).** Aligned balanced init with per-core amplitude $a_0$:
  $\rho_0=\sqrt Ka_0$, $g(u)\equiv sK^{-K/2}$, so $t^*=a_0^{-(K-2)}/((K-2)s)$, the integrated balanced
  mode $\dot a=a^{K-1}(s-a^K)$ of [Saxe2014] eq. 15 (also [LiLuoLyu2021] Thm 6.2). Checked: LO blow-up
  vs formula, relative error $2\cdot10^{-11}$; GF time to $a^K=s/2$ vs the exact integral,
  $10^{-11}$ (binary4, MPS4, binary8; $s=1,2$; verifier: $\le4\cdot10^{-12}$).
- **Kumar & Haupt (known).** [KumarHaupt2025a] Thm 1 (arXiv 2403.08121): for $L$-homogeneous models with
  $L>2$ and small init, the direction approximately converges to a nonnegative KKT point of the NCF ($=g$)
  on the sphere while the norm stays $O(\delta)$; it applies verbatim to TTNs (`notes/literature.md`
  section 2.1). 2.1 is the explicit form: escape needs $r$ to grow by a factor $\sim1/\sigma$, i.e.
  $\tau_{\rm esc}\approx\ln(1/\sigma)/(Kg_*)\to\infty$, so as $\sigma\to0$ the escape direction is
  $\lim_{\tau\to\infty}u(\tau)$, a KKT point of $g$ (generically a local max, section 3). From an init
  exactly along a KKT point $w^*$, [KumarHaupt2025b] give the escape time $1/(L(L-2)N(w^*)\delta^{L-2})$:
  2.2 with $g$ constant.
- **What the conservation laws change (derived).** The direction converges only as $r/\rho_0\to\infty$
  (the constraints $Q_e(u)=Q_e(\theta_0)/r^2$ must fade), but $t^*$ is dominated by $\tau=O(1)$, where
  $r\approx\rho_0$ and the constraints bind. The constraints therefore shape the part of the trajectory
  that decides both the escape time and which KKT point is reached: on MPS the basin of component $r$
  is exactly $\{\hat t_{1,r}/s_r\text{ minimal}\}$ (5.1), a quantity fixed in this phase. What they do not
  change is the form of the limit: KKT points with $g_*>0$ are exactly balanced (`notes/literature.md`
  section 3, critique C5).

### 2.4 From blow-up to the first event (derived, checked)

$t_1=t^*+\delta(u_{\rm esc})\,s^{-2+2/K}+o(1)$: after $r=O(1)$ the remaining time is $\sigma$-independent and
depends on the direction $u_{\rm esc}$ at that point; as $\sigma\to0$, $u_{\rm esc}$ converges to the
balanced KKT point and $\delta\to\delta_K$, the value for the balanced mode (derived),
$$\delta_K=-\frac{2^{(K-2)/K}}{K-2}+\int_0^{2^{-1/K}}\frac{b\,db}{1-b^K}:\quad
\delta_3=-0.850,\ \delta_4=-0.266,\ \delta_5=-0.047,\ \delta_6=+0.072,\ \delta_7=+0.148,\ \delta_8=+0.201$$
(quadrature, checked by direct integration of $\dot b=b^{K-1}(1-b^K)$; v1 had $-0.86$, $-0.27$, $+0.14$). Hence
$t_1\sigma^{K-2}s/\hat t_1-1=\delta(u_{\rm esc})s^{-1+2/K}\sigma^{K-2}/\hat t_1=O(\sigma^{K-2})$. Checked per init
(GF on `population_loss`, DOP853, rtol $10^{-10}$; $Q_e$ drift $\le10^{-11}$):

| tree, d, chi | sigma | inits | max $\lvert t_1\sigma^{K-2}s/\hat t_1-1\rvert$ |
| --- | --- | --- | --- |
| binary4, d = 2-3, chi = 1-9 | 0.01 | 20 per chi | 0.008 |
| mps4, d = 2-3, chi = 1-9 | 0.03 | 20 per chi | 0.001 |
| mps4 d=3 chi=9 / mps6 d=2 chi=4 | 0.03 / 0.15 | 4 / 4 | 0.0002 / 0.0002 |
| binary8 d=2 chi=4 / ht4 d=2 chi=2 | 0.3 (0.5) | 4 / 4 | 0.004 (0.085) / 0.005 (0.15) |

For $K=3,4$ the residual equals the $\delta_K$ term to 2 digits on every init (verifier, 21 inits). For
$K=6$-$7$ at the accessible $\sigma$ it is larger than the $\delta_K$ term, by factors 2-8 in the verifier's
runs (MPS6 at 0.15: $0.9$-$1.2\cdot10^{-4}$ vs $0.3$-$0.5\cdot10^{-4}$; binary8 at 0.3: $5.7$-$9.3\cdot10^{-4}$
vs $0.5$-$2.5\cdot10^{-4}$; HT4 at 0.5: 0.45-1.7% vs 0.2-0.4%) and up to 30 in v1's (HT4 at 0.3: 0.47% vs
0.017%); at $\sigma=0.5$ v1 saw up to 8.5% (binary8) and 15% (HT4). Heuristic reason: $u_{\rm esc}$ has not
converged at these $\sigma$, so $\delta(u_{\rm esc})\ne\delta_K$.

### 2.5 Per-init prediction recipe (for the experiment agents)

With `g = teacher.inner_ttn(spec, spec.unflatten(x))` and torch autograd, integrate in $\tau$ the
state $(u,\ell,T)$, $u_0=\theta_0/\lVert\theta_0\rVert$, $\ell_0=T_0=0$:
$\;du/d\tau=\nabla g(u)-Kg(u)u$ (renormalize $u$ inside the right-hand side),
$\;d\ell/d\tau=Kg(u)$, $\;dT/d\tau=e^{-(K-2)\ell}$; stop at $\ell=40/(K-2)+5$, add the tail
$e^{-(K-2)\ell}/((K-2)Kg(u))$. Then $t^*=\lVert\theta_0\rVert^{-(K-2)}T$ (actual teacher weights,
any number of components) and $t_1\approx t^*$. DOP853, rtol $10^{-10}$: 0.5-3 s per init for the
trees used here. The dominant component of the final $u$ is the first component learned (as
$\sigma\to0$). The prediction is per (init seed, teacher basis) pair. Measure $t_1$ at $\sigma$ small
enough that $\sigma^{K-2}\ll1$: at $\sigma=0.5$ binary8 and HT4 still deviate by up to 9% and 15%.

## 3. Maximum and local maxima of $g$ on the sphere (task b)

### 3.1 Global maximum (proven)

Let $P$ be the partition of the sites into the groups carried by the same core (cores without sites
contribute no group) and $\lVert W^*\rVert_{\sigma,P}=\max\{W^*[Y_1,\dots,Y_m]:\lVert Y_j\rVert_F=1\}$,
the spectral (injective) norm of $W^*$ seen as an order-$m$ tensor whose $j$-th leg is the product of
the legs in group $j$. Then on $\lVert\theta\rVert=1$
$$g(\theta)\le K^{-K/2}\lVert W^*\rVert_{\sigma,P},$$
with equality at balanced bond-rank-one configurations $G_v=K^{-1/2}Y_v\otimes e\otimes\dots\otimes e$
($Y_v$ the maximizing group tensors, $e$ any unit bond vector). This is the statement of
`notes/literature.md` section 3 ("derived (review)"); the proof (re-checked step by step by the verifier):

1. $\langle W(\theta),W^*\rangle\le\lVert W^*\rVert_{\sigma,P}\prod_v\lVert G_v\rVert_F$, by induction on
   the number of cores. Take a leaf $\ell$ with parent $p$ and the SVD
   $G_\ell=\sum_k\sigma_kY_k\otimes z_k$ (group legs | bond leg). Then
   $\langle W,W^*\rangle=\sum_k\sigma_k\langle W'_k,W^*[Y_k]\rangle$, where $W'_k$ is the TTN without
   $\ell$ and with $G_p$ replaced by $G_p\times_\ell z_k$. The induction hypothesis and
   $\lVert W^*[Y_k]\rVert_{\sigma,P'}\le\lVert W^*\rVert_{\sigma,P}$ bound this by
   $\lVert W^*\rVert_{\sigma,P}\prod_{v\ne\ell,p}\lVert G_v\rVert\sum_k\sigma_k\lVert G_p\times_\ell z_k\rVert$,
   and Cauchy-Schwarz with $\sum_k\lVert G_p\times_\ell z_k\rVert^2\le\lVert G_p\rVert^2$ closes it.
2. AM-GM: $\prod_v\lVert G_v\rVert\le(\sum_v\lVert G_v\rVert^2/K)^{K/2}=K^{-K/2}$.

For odeco teachers $\lVert W^*\rVert_{\sigma,P}=s_1$ for every grouping. In general the grouping
matters: for $W^*=(I/\sqrt d)\otimes(I/\sqrt d)$ on sites (01|23), binary4 (2-site cores) reaches
$g=3^{-3/2}\cdot1=0.19245$ and MPS4 only $4^{-2}\cdot(1/d)$ (checked: $d=2$, spherical-flow endpoints
0.19245 and 0.03125 on 6/6 inits; verifier, L-BFGS: 0.19245 for $d=2,3$, 0.03125 and 1/48 for MPS4 at
$d=2,3$).

### 3.2 Critical points (known)

Every KKT point of $g$ on the sphere with $g_*>0$ is exactly balanced on every bond and has
$\lVert G_v\rVert^2=1/K$ (`notes/literature.md` section 3, critique C5).

### 3.3 Which component configurations are local maxima (derived, checked)

Pattern teacher with distinct weights. Let $\theta_q$ be the balanced bond-rank-one configuration of
component $q$ on the unit sphere (per-core amplitude $K^{-1/2}$, channel 0 on every bond; library
`aligned_init`), so $g(\theta_q)=s_qK^{-K/2}$ and the Riemannian Hessian is
$\nabla^2g-s_qK^{1-K/2}I$ on the tangent space. Since $g$ is linear in each core, $\nabla^2g$ only has
cross terms between two cores $v\ne w$, and such a term contracts $W^*$ with $u_q$ on every site
outside $v\cup w$: only components $r$ with $D_{qr}\subseteq\mathrm{sites}(v)\cup\mathrm{sites}(w)$
contribute. Hence:

- **some $r$ has $m_{qr}=1$**: $\nabla g(\theta_q)$ has a component along $r$'s pattern on that core, so
  $\theta_q$ is not critical (the core learns $r$ without a plateau; `notes/theory_saddles.md` 3.1).
- **every $r$ has $m_{qr}\ge3$**: no other component contributes; $q$'s own cross terms (bond-channel and
  amplitude perturbations of two cores) give 0 on the gauge orbit and $-2s_qK^{1-K/2}$ off it, and every
  other tangent direction has $-s_qK^{1-K/2}$. So $\theta_q$ is a strict local max modulo gauge (zero
  modes: $\sum_e(\chi_e-1)$ bond rotations).
- **$r$ with $m_{qr}=2$**, cores $v,w$: perturb $\delta G_v=\varepsilon_v\,u_r|_v\otimes(\text{bond vectors})$
  and $\delta G_w=\varepsilon_w\,u_r|_w\otimes(\dots)$, with channel 0 on every bond except possibly the bond
  $e$ joining $v$ and $w$, if they are adjacent. The cross term is $s_rK^{-(K-2)/2}\varepsilon_v\varepsilon_w$
  when both perturbations use the same channel $c$ of $e$, for any $c=0,\dots,\chi_e-1$. If $v,w$ are not
  adjacent, every bond on the path between them meets the other cores' channel 0, so only $c=0$
  survives. Each surviving channel gives the block
  $K^{1-K/2}\begin{pmatrix}-s_q&s_r\\s_r&-s_q\end{pmatrix}$, with eigenvalues $K^{1-K/2}(s_r-s_q)$ and
  $-K^{1-K/2}(s_r+s_q)$. So $r$ contributes
  $$n_{qr}=\begin{cases}\chi_e&\text{if }v,w\text{ share bond }e,\\1&\text{otherwise}\end{cases}$$
  eigenvalues $K^{1-K/2}(s_r-s_q)$: unstable directions if $s_r>s_q$, stable if $s_r<s_q$; a tie
  $s_r=s_q$ gives zero eigenvalues and is decided at higher order. Channel $c=0$ is the "piggyback"
  direction (the two cores rotate their physical vectors toward $r$ and keep the learned bond index;
  `notes/theory_saddles.md` 4.1); channels $c\ne0$ are the new-index entries of a $k=2$ route
  (`notes/theory_saddles.md` 3.2). v1 counted only $c=0$, which is right only for non-adjacent cores.
- **several $r$ with $m_{qr}=2$ sharing a core**: the blocks combine into the piggyback-graph matrix $S$
  and the adjacent-pair blocks $M_C$ of `notes/theory_saddles.md` 4.1 and 4.5; $\theta_q$ is a local max if
  $\rho(S)<s_q$ and $\sigma_{\max}(M_C)<s_q$ for every adjacent pair $C$, and a saddle if either exceeds $s_q$.

So, for two-component teachers (or components whose $m=2$ partners use different cores), $\theta_q$ is a
strict local max modulo gauge iff no $m_{qr}=1$ and every $r$ with $m_{qr}=2$ has $s_r<s_q$; otherwise it
has $n_q=\sum_{r:\,m_{qr}=2,\,s_r>s_q}n_{qr}$ unstable directions.

For odeco teachers $m_{qr}$ = number of cores carrying sites: $m=N$ for MPS and HT, $m=3,4$ for binary
$N=6,8$, $m=2$ for binary $N=4$, whose two bottom cores are not adjacent ($n=1$). So every odeco
component is a local max on MPS, HT and binary $N\ge6$ ("tensor-like": several basins), while binary4 is
matrix-like: it is exactly a depth-3 deep linear network on the (01|23) matricization
(`notes/theory_saddles.md` 4.2) and only the top component is a local max, as in DLNs ([Jacot2021]
Lemma 15; [Achour2024] Thm 7). Pattern teachers whose components differ on exactly two cores are
matrix-like on any tree, with $n=\chi_e$ when those cores are adjacent.

**Checks.** (i) v1, odeco, weights (1, 0.6): binary4 ($d,\chi$ = 2,2 and 3,3): component 1 has no positive
eigenvalue, component 2 exactly one, $+0.230940=3^{-1/2}\cdot0.4$; MPS4 ($d,\chi$ = 2,2 and 3,3), MPS6,
binary8, HT4 ($d=\chi=2$): none for either component; zero modes $=\sum_e(\chi_e-1)$ in every case
(verifier: also binary6). (ii) v2, pattern teachers, $q$ = all-zeros pattern with weight 0.6, $r$ with
weight 1 (`escape_v2/hess_patterns.py`), 19 cases, all as predicted:

| tree, d, chi | cores carrying $D_{qr}$ | predicted $n_{qr}$ | measured: # positive, value |
| --- | --- | --- | --- |
| mps4, 2, 2 | 0-1, 1-2, 2-3 (adjacent, $\chi_e=2$) | 2 | 2, 0.1 (each) |
| mps4, 2, 2 | 0-2, 0-3, 1-3 | 1 | 1, 0.1 (each) |
| mps4, 2, 4 | 1-2 ($\chi_e=4$) / 0-1 ($\chi_e=2$) | 4 / 2 | 4 / 2, 0.1 |
| mps4, 3, 3 | 0-1 ($\chi_e=3$) / 0-2 | 3 / 1 | 3 / 1, 0.1 |
| mps4, 3, 9 | 1-2 ($\chi_e=9$) | 9 | 9, 0.1 |
| mps6, 2, 2 / 2, 4 | 2-3 ($\chi_e=2$ / 4) | 2 / 4 | 2 / 4, 0.011111 |
| ht4, 2, 2 | sibling leaves 0, 1; leaves 1, 3 | 1, 1 | 1, 1, 0.003085 |
| binary8, 2, 2 | sibling bottom cores (sites 0-3) | 1 | 1, 0.003085 |
| binary4, 2, 2 | odeco (both bottom cores) | 1 | 1, 0.230940 |
| binary4 / binary8, 2, 2 | one core ($D$ = sites 0, 1) | not critical | KKT residual 0.33 / 0.0029 |

Predicted values $K^{1-K/2}\cdot0.4$: 0.1 ($K=4$), 0.011111 ($K=6$), 0.003085 ($K=7$), 0.230940 ($K=3$). The
verifier found the same multiplicities independently. (iii) 200 spherical flows from random init
(MPS4, MPS6, binary8, HT4, binary4; odeco weights (1, 0.7)): every endpoint has $g=s_rK^{-K/2}$ to 6 digits
for the winning $r$ (no other local maxima found; empirical); binary4 ends at component 1 in 40/40.

Not covered: non-orthogonal CP teachers (best rank-one approximation has non-global local optima,
[KofidisRegalia2002]); whether non-odeco teachers have local maxima that are not bond-rank-one
(`notes/literature.md` critique C6).

## 4. Effective initial amplitude; dependence on $\chi$ (task c)

### 4.1 Only the teacher slices move (proven)

For an odeco teacher, $\nabla_{G_v}g=\sum_rs_r\,(\bigotimes_{i\in v}u_r^{(i)})\otimes E_v^{(r)}$ with
$E_v^{(r)}$ the contraction of all other cores with $u_r$ on their sites. Hence under LO:

- the part of $G_v$ orthogonal (on its physical legs) to all $\bigotimes_{i\in v}u_r^{(i)}$ is frozen;
- the moving variables are the reduced cores $H_v^{(r)}=G_v$ contracted with $u_r^{(i)}$ on its physical
  legs (one per component for cores with sites; the whole core, shared, for cores without sites), and
  $g=\sum_rs_r\Phi(H^{(r)})$ with $\Phi$ the scalar contraction of the reduced tree;
- under fan-in init the $H_v^{(r)}$ have iid $N(0,\sigma^2/F_v)$ entries, independent across $r$: a
  fan-in-initialized scalar TTN with per-core scale $\sigma d^{-n_v/2}$.

So $\mathbb E\lVert H_v\rVert^2=\sigma^2\chi_vd^{-n_v}$ while $\mathbb E\lVert G_v\rVert^2=\sigma^2\chi_v$:
binary4 has $r_{\rm m}^2\approx\sigma^2(2\chi_b/d^2+1)$ versus $\rho_0^2\approx\sigma^2(2\chi_b+1)$; MPS has
$r_{\rm m}^2\approx\rho_0^2/d$. $\rho_0$ is dominated by frozen slices that never contribute. For MPS every core
carries a site, so different components have disjoint moving variables and evolve independently at
leading order (used in section 5). The same holds for pattern teachers with one slice per distinct
pattern on each core (components that agree on a core share its slice; `escape_v2/fastpat.py`, which
reproduces the library LO blow-up time to $10^{-11}$).

### 4.2 What the conservation laws do, and what they do not do

The reduced flow conserves $Q_e^{\rm red}=H_v^\top H_v-H_{p,(v)}H_{p,(v)}^\top$ on every bond (gauge
symmetry; the DLN analogue is [Arora2018; DuHuLee2018]). Two consequences are proven; the third
ingredient of the escape mode is empirical.

**(a) Equal norm shift (proven).** $g$ is linear in each core, so
$\frac d{dt}\lVert H_v\rVert^2=2\langle H_v,\nabla_{H_v}g\rangle=2g$ for every core $v$, and under LO
$\dot g=\lVert\nabla g\rVert^2\ge0$. Hence
$$\lVert H_v(t)\rVert^2=\lVert H_v(0)\rVert^2+y(t),\qquad y(t)=2\int_0^tg,\quad y\text{ convex}.$$
All cores (reduced or full; frozen slices are constant) gain or lose the same squared norm; a core
loses norm only while $g<0$, i.e. only if $\langle W_0,W^*\rangle<0$ (half of all inits, by the symmetry
$G_v\to-G_v$), and the loss stops when $g$ changes sign. Under GF the same holds with $2(g-\lVert W\rVert^2)$
in place of $2g$ (the trace of the conservation law). Checked: equal shifts to $10^{-8}$ (binary4) and
$10^{-11}$ (MPS4) relative, 2200 trajectories (`escape_v2/ends.py`).

**(b) End lock along the initial direction (proven; informative only for small $\kappa_e$).** Call a reduced
core an *end* if it is a vector $h\in\mathbb R^{\chi_e}$ (binary bottom cores, HT leaves, both ends of an
MPS), and let $N$ be the neighbouring reduced core matricized with the shared leg as rows (for the MPS
root end, $N=M_2^\top$, from $Q=M_2^\top M_2-yy^\top$). Conservation gives
$hh^\top\succeq h_0h_0^\top-N_0N_0^\top$, hence for all $t$
$$(\hat h_0\cdot h(t))^2\ \ge\ \lVert h_0\rVert^2(1-\kappa_e),\qquad\kappa_e=\frac{\lVert N_0^\top\hat h_0\rVert^2}{\lVert h_0\rVert^2}.$$
Under fan-in init (derived, checked: medians over 200 inits in brackets): binary4 bottom ends
$\kappa\approx d^2/\chi_b^2$ (1.22 and 0.89 at $d=\chi=2$; 0.26-0.29 at $d=2,\chi=4$; 0.10 at $d=3,\chi=9$;
0.06 at $d=4,\chi=16$); MPS leaf end $\kappa\approx\chi_1/\chi_0^2$ (0.51 at $d=\chi=2$, 0.26 at $d=\chi=4$,
1.16 at $d=4,\chi=16$); MPS root end $\kappa\approx1$ (1.0-1.4); HT leaves $\kappa\approx d\chi_b/\chi_l^3$ and
binary8 bottom ends $\kappa\approx d^2\chi_m/\chi_b^3$ (both 0.25-1 in the settings of table 2; derived, not
checked). The lock is
therefore informative only for binary4 with $\chi_b\gg d$ and MPS leaves with $\chi\le d$; it is vacuous
at $\chi_b=d$ on binary4 and always for the MPS root end. It held on every trajectory checked.
v1's statement that the conservation laws lock every end at its initial norm was wrong.

**(c) Ends hardly shrink or rotate (empirical).** Reduced LO from 200 inits per setting
($\sigma=s=1$; binary4: both bottom ends pooled; MPS4: leaf / root):

| tree, d, chi | frac. inits with $\min_t\lVert h\rVert/\lVert h_0\rVert<0.9$ | median $\lVert h(t^*/2)\rVert/\lVert h_0\rVert$ | median (10% quantile) of $\cos(h(t^*/2),h_0)$ | median $\cos(h,h_0)$ after the escape |
| --- | --- | --- | --- | --- |
| binary4, 2, 2 | 0.19-0.21 | 1.61-1.71 | 0.95-0.97 (0.09-0.45) | 0.90-0.92 |
| binary4, 2, 4 | 0.04 | 1.45-1.49 | 0.96-0.97 (0.80-0.85) | 0.93-0.94 |
| binary4, 3, 3 | 0.08-0.13 | 1.64-1.69 | 0.92-0.94 (0.53-0.59) | 0.86-0.87 |
| binary4, 4, 4 | 0.06-0.09 | 1.64-1.67 | 0.92-0.93 (0.62-0.70) | 0.84-0.86 |
| binary4, 3, 9 / 4, 16 | 0.00 | 1.44-1.46 | 0.98-0.99 (0.94-0.98) | 0.96-0.97 |
| mps4, 2, 2 | 0.07 / 0.20 | 1.10 / 1.18 | 0.99 / 0.98 (0.87 / 0.64) | 0.95 / 0.92 |
| mps4, 2, 4 | 0.09 / 0.18 | 1.16 / 1.27 | 1.00 / 0.99 (0.92 / 0.45) | 0.96 / 0.96 |
| mps4, 3, 3 | 0.06 / 0.10 | 1.08 / 1.22 | 0.99 / 0.97 (0.94 / 0.51) | 0.93 / 0.87 |
| mps4, 4, 4 | 0.01 / 0.10 | 1.08 / 1.29 | 0.99 / 0.95 (0.97 / 0.72) | 0.95 / 0.83 |
| mps4, 4, 16 | 0.01 / 0.04 | 1.11 / 1.38 | 1.00 / 0.97 (0.98 / 0.83) | 0.96 / 0.93 |

The median of $\min_t\lVert h\rVert/\lVert h_0\rVert$ is 1.00 in every setting (by (a), no core shrinks when
$g(\theta_0)>0$). The verifier's independent runs agree (0-19% below 0.9). So the ends keep most of their
norm and direction in the median, but at $\chi_b=d$ (binary4) and for the MPS root end a minority of
inits rotate strongly.

**(d) Frozen-end mode (heuristic).** If an end keeps its direction, (b) holds with equality along
$\hat h_0$ and fixes the neighbour's amplitude on the path: $(\text{path amplitude})^2=\gamma_0^2+y$, with
$\gamma_0$ the neighbour's initial component along the path. The escape is then the product mode of 4.3
with end amplitudes $c_e=\lVert h_e(0)\rVert$ and internal amplitudes $\gamma_0$. In the unreduced variables
most of the conserved imbalance sits in the frozen slices.

### 4.3 Imbalanced product mode (derived for the mode; known scaling)

If the escape is a rank-one path with per-core amplitudes $x_v>0$, $\dot x_v=s\prod_{w\ne v}x_w$, then all
$x_v^2-x_w^2$ are conserved; with $x_v^2=y+c_v^2$ ($c_v$ initial amplitudes, $y(0)=0$),
$\dot y=2s\prod_w\sqrt{y+c_w^2}$ and
$$\hat t=\frac1{2s}\int_0^\infty\prod_{v=1}^K\left(y+c_v^2\right)^{-1/2}dy .$$
With $L$ ends at amplitude $\alpha$ (or $c_e$) and $m=K-L$ internal cores at $\gamma\ll\alpha$
(limits checked by quadrature by the verifier):

- $m=0$ (all at $\alpha$): $\hat t=\alpha^{-(K-2)}/((K-2)s)$, Saxe;
- $m=1$: $\hat t\to\tfrac12B(\tfrac12,\tfrac{K-2}2)\,\alpha^{-(K-2)}/s$ as $\gamma\to0$ ($K=3$: $\pi/(2s\alpha)$),
  finite (v1 wrote "$m\le1$"; the formula is for $m=1$);
- $m=2$: logarithmic, $\hat t=\ln(1/\gamma)/(s\prod_ec_e)+O(1)$; $K=4$ with ends $\alpha,\beta$:
  $\hat t\approx\frac1{s\alpha\beta}\ln\frac{2\alpha\beta}{\gamma(\alpha+\beta)}$;
- $m\ge3$: $\hat t\approx\gamma^{-(m-2)}\big/\big((m-2)s\prod_ec_e\big)$: the internal path amplitudes dominate.

This trichotomy is known for deep networks with $r$ layers at small scale $\varepsilon$ and the others at
$O(1)$: [RawalDeWeese2026] Thm 4.2 (escape time $\Theta(\Gamma^{-1})$, $\Theta(\Gamma^{-1}\log(1/\varepsilon))$,
$\Theta(\Gamma^{-1}\varepsilon^{-(r-2)})$ for $r=1,2,\ge3$, $\Gamma=\lVert\mathbb E[yx]\rVert$; scalar output, smooth
activation with $\sigma'(0)>0$, which includes deep linear networks), with the non-small layers moving
only $O(R^2)$ (their eq. (29)). What is new here is the TTN reduction: the "small layers" are the internal
cores' path amplitudes and the "non-small" ones the reduced ends, both set by the fan-in init (4.1, 4.2);
the exact integral with all constants; and the random-init amplitudes below. Binary4 and MPS3 have $m=1$,
MPS4 has $m=2$, binary8 and HT4 have $m=3$, MPS$N$ has $m=N-2$. This is why the naive formula fails most
for large $K$: $\rho_0$ is dominated by internal cores whose norm is spread over $\sim\chi^2$ directions,
while only their path amplitude matters, with power $m-2$. Caveat: in MPS4 the internal amplitude is not
small compared with the root end (4.5), so the $\gamma\ll\alpha$ limits are only qualitative there.

### 4.4 binary $N=4$ ($K=3$): median $\hat t_1\approx\pi d/(2\sqrt{\chi_b})$ (derived given frozen ends; checked)

Reduced: ends $a,b\in\mathbb R^{\chi_b}$ ($\chi_b=\min(\chi,d^2)$, entries $N(0,\sigma^2/d^2)$), root
$C\in\mathbb R^{\chi_b\times\chi_b}$ (entries $N(0,\sigma^2/\chi_b^2)$), $g=s\,a^\top Cb$.

- *Exact LO reduction (proven).* With $Q_1=a_0a_0^\top-C_0C_0^\top$, $Q_2=b_0b_0^\top-C_0^\top C_0$:
  $\ddot a=s^2[(\lVert a\rVert^2+\lVert b\rVert^2)a-Q_1a]$, $\ddot b=s^2[(\lVert a\rVert^2+\lVert b\rVert^2)b-Q_2b]$,
  $\dot a(0)=sC_0b_0$, $\dot b(0)=sC_0^\top a_0$ (a Newtonian particle in $\mathbb R^{2\chi_b}$).
- *Frozen-end mode (heuristic, 4.2(d)).* $\alpha=\lVert a_0\rVert$, $\beta=\lVert b_0\rVert$, signed root path
  amplitude $\gamma_0=\hat a_0^\top C_0\hat b_0\sim N(0,\sigma^2/\chi_b^2)$:
  $$\hat t=\int_{\gamma_0}^\infty\frac{dx}{s\sqrt{(\alpha^2-\gamma_0^2+x^2)(\beta^2-\gamma_0^2+x^2)}}=\hat t\big|_{\gamma_0=0}-\frac{\gamma_0}{s\alpha\beta}+O(\gamma_0^2).$$
  Since $\gamma_0$ is symmetric about 0, the value at $\gamma_0=0$, $\mathbf K(k)/(s\max(\alpha,\beta))$ with
  $k^2=1-\min^2/\max^2$ ($\mathbf K$ the complete elliptic integral of the first kind), predicts the median
  (not the mean). At typical $\alpha=\beta=\sigma\sqrt{\chi_b}/d$:
  $$t_1\approx\frac{\pi d}{2s\sigma\sqrt{\chi_b}}\qquad(d\le\chi_b\le d^2),\qquad t_1\approx\frac{\pi}{2s\sigma}\ \text{at full bond, independent of }d.$$
  In the mode, half of $\hat t$ elapses before the root's path amplitude reaches the ends' *initial*
  amplitude $\alpha$, when the ends have grown to $\sqrt2\alpha$ (measured median
  $\lVert h(t^*/2)\rVert/\lVert h_0\rVert$: 1.44-1.46 at full bond, 1.6-1.7 at $\chi_b=d$; table of 4.2(c)).
- *Accuracy (empirical).* Measured median / $(\pi d/(2\sqrt{\chi_b}))$: 0.84-0.95 for $d\le\chi_b<2d$ (verifier,
  300 fresh inits: 0.81 at $d=\chi=4$, 0.85 at $d=\chi=3$) and 0.94-1.00 for $\chi_b\ge2d$ (verifier
  0.925-0.985). The closed form overestimates, most where the ends rotate most (4.2(c)).
- *Transport correction (heuristic).* The random root adds a second channel (a, b rotate toward the top
  singular vectors of $C_0$ and grow at rate $s\sigma_{\max}(C_0)\approx2s\sigma/\sqrt{\chi_b}$); relative to
  the frozen-end rate $s\sigma\sqrt{\chi_b}/d$ it matters when $\chi_b\lesssim2d$ and makes $t_1$ shorter.
- *Spread (derived from the mode).* From the $\chi^2_{\chi_b}$ fluctuations of $\lVert a_0\rVert,\lVert b_0\rVert$:
  $\mathrm{sd}\ln\hat t_1\approx1/(2\sqrt{\chi_b})$ (measured 0.36, 0.26, 0.18, 0.12 at $\chi_b$ = 2, 4, 8, 16
  vs 0.35, 0.25, 0.18, 0.125; verifier 0.127 at 16).

### 4.5 MPS $N=4$ ($K=4$) (derived given frozen ends; checked)

Reduced: $x\in\mathbb R^{\chi_0}$ ($\lVert x\rVert^2\approx\sigma^2\chi_0/d$), $M_1$, $M_2$, and the root slice
$y\in\mathbb R^{\chi_2}$ with $\lVert y\rVert^2\approx\sigma^2/d$ independent of $\chi$. With the ends frozen in
direction, $p=M_1^\top\hat x$ and $q=M_2\hat y$ obey exactly $\dot p=s\lVert x\rVert\lVert y\rVert q$,
$\dot q=s\lVert x\rVert\lVert y\rVert p$: the internal path amplitude $\gamma=\lVert p+q\rVert/2$ grows
exponentially from $\gamma_0^2\approx\sigma^2(1+\chi_1/\chi_0)/(4d)$. The product mode $(\alpha,\gamma,\gamma,\beta)$:
$$\hat t_P=\frac12\int_0^\infty\frac{dy}{(y+\gamma_0^2)\sqrt{(y+\alpha^2)(y+\beta^2)}},\quad
\alpha^2=\frac{\chi_0}d,\ \beta^2=\frac1d,\ \gamma_0^2=\frac{1+\chi_1/\chi_0}{4d}\ (\sigma=s=1),$$
underestimates the measured median by a factor 1.05-1.39 (table 1). The internal amplitude is *not*
small: $\gamma_0/\beta=\sqrt{(1+\chi_1/\chi_0)/4}=0.71$-$1.12$ (v1's "internal cores near zero" was wrong
for MPS4). $\chi$ enters only through $\chi_0=\min(\chi,d)$ and $\chi_1=\min(\chi,d^2)$ in $\gamma_0$, hence
weakly.

### 4.6 Table 1: $\hat t_1$ for the $N=4$ trees ($\sigma=s=1$; single odeco component; 300 inits each)

"median (95% CI)": component 1, distribution-free CI; "pooled": median of components 1 and 2 together
(same distribution; 600 values, correlated through the shared root on binary4). "naive" =
$t_{\rm KH}$, "red" = the same with $r_{\rm m}$ (both lower bounds, 2.2); closed form =
$\pi d/(2\sqrt{\chi_b})$ (binary4, $\chi_b\ge d$) or $\hat t_P$ (MPS4). Medians carry a relative standard
error $\approx1.25\,\mathrm{sd}\ln\hat t_1/\sqrt n$ (2-7% here).

| tree | d | chi | bonds | median (95% CI) | pooled | 10%-90% | mean (se) | sd ln | median / naive | median / red | closed form |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary4 | 2 | 1 | (1, 1) | 2.75 (2.51-2.99) | 2.75 | 1.40-6.73 | 3.76 (0.26) | 0.63 | 2.5 | 1.52 | - |
| binary4 | 2 | 2 | (2, 2) | 1.95 (1.87-2.15) | 1.96 | 1.30-3.32 | 2.21 (0.05) | 0.36 | 2.5 | 1.50 | 2.22 |
| binary4 | 2 | 3 | (3, 3) | 1.71 (1.65-1.77) | 1.70 | 1.20-2.73 | 1.86 (0.04) | 0.31 | 2.6 | 1.50 | 1.81 |
| binary4 | 2 | 4+ | (4, 4) | 1.56 (1.53-1.62) | 1.55 | 1.17-2.18 | 1.66 (0.03) | 0.26 | 2.7 | 1.54 | 1.57 |
| binary4 | 3 | 1 | (1, 1) | 2.96 (2.69-3.23) | 2.99 | 1.49-8.31 | 4.15 (0.23) | 0.66 | 2.9 | 1.54 | - |
| binary4 | 3 | 2 | (2, 2) | 2.39 (2.29-2.56) | 2.44 | 1.60-4.61 | 2.82 (0.08) | 0.41 | 3.1 | 1.60 | - |
| binary4 | 3 | 3 | (3, 3) | 2.35 (2.25-2.42) | 2.30 | 1.67-3.40 | 2.46 (0.04) | 0.27 | 3.5 | 1.65 | 2.72 |
| binary4 | 3 | 4 | (4, 4) | 2.07 (2.00-2.15) | 2.14 | 1.60-2.93 | 2.19 (0.03) | 0.25 | 3.6 | 1.58 | 2.36 |
| binary4 | 3 | 6 | (6, 6) | 1.83 (1.77-1.88) | 1.84 | 1.44-2.45 | 1.90 (0.02) | 0.21 | 3.8 | 1.58 | 1.92 |
| binary4 | 3 | 9+ | (9, 9) | 1.55 (1.52-1.59) | 1.55 | 1.25-2.01 | 1.60 (0.02) | 0.18 | 3.9 | 1.54 | 1.57 |
| binary4 | 4 | 1 | (1, 1) | 4.09 (3.60-4.41) | 3.79 | 1.84-10.5 | 5.61 (0.36) | 0.67 | 3.7 | 1.70 | - |
| binary4 | 4 | 2 | (2, 2) | 3.09 (2.92-3.34) | 3.05 | 1.89-5.09 | 3.37 (0.08) | 0.38 | 4.0 | 1.78 | - |
| binary4 | 4 | 4 | (4, 4) | 2.65 (2.52-2.72) | 2.61 | 1.91-3.80 | 2.76 (0.04) | 0.27 | 4.5 | 1.78 | 3.14 |
| binary4 | 4 | 8 | (8, 8) | 2.09 (2.06-2.15) | 2.10 | 1.67-2.65 | 2.14 (0.02) | 0.18 | 5.0 | 1.70 | 2.22 |
| binary4 | 4 | 16+ | (16, 16) | 1.57 (1.54-1.61) | 1.56 | 1.34-1.84 | 1.59 (0.01) | 0.12 | 5.2 | 1.55 | 1.57 |
| mps4 | 2 | 1 | (1, 1, 1) | 2.78 (2.51-3.47) | 2.97 | 1.00-12.0 | 5.63 (0.55) | 0.96 | 5.2 | 2.24 | - |
| mps4 | 2 | 2 | (2, 2, 2) | 1.56 (1.45-1.75) | 1.50 | 0.79-3.53 | 2.04 (0.10) | 0.62 | 5.5 | 2.50 | 1.13 |
| mps4 | 2 | 3 | (2, 3, 2) | 1.41 (1.23-1.53) | 1.37 | 0.73-3.04 | 1.68 (0.06) | 0.55 | 5.4 | 2.45 | 1.02 |
| mps4 | 2 | 4+ | (2, 4, 2) | 1.30 (1.21-1.47) | 1.29 | 0.70-2.90 | 1.61 (0.06) | 0.54 | 5.8 | 2.73 | 0.95 |
| mps4 | 3 | 1 | (1, 1, 1) | 4.59 (4.13-4.99) | 4.25 | 1.59-16.7 | 10.30 (1.58) | 1.01 | 8.1 | 2.28 | - |
| mps4 | 3 | 2 | (2, 2, 2) | 2.21 (2.04-2.36) | 2.20 | 1.08-4.87 | 2.70 (0.10) | 0.58 | 7.5 | 2.34 | 1.69 |
| mps4 | 3 | 3 | (3, 3, 3) | 1.69 (1.58-1.81) | 1.71 | 0.92-3.28 | 1.94 (0.06) | 0.47 | 8.3 | 2.69 | 1.48 |
| mps4 | 3 | 4 | (3, 4, 3) | 1.61 (1.51-1.73) | 1.63 | 0.92-3.04 | 1.82 (0.05) | 0.46 | 8.6 | 2.79 | 1.39 |
| mps4 | 3 | 6 | (3, 6, 3) | 1.47 (1.37-1.55) | 1.51 | 0.92-2.86 | 1.72 (0.06) | 0.44 | 9.6 | 3.16 | 1.25 |
| mps4 | 3 | 9+ | (3, 9, 3) | 1.35 (1.25-1.44) | 1.38 | 0.87-2.57 | 1.55 (0.04) | 0.43 | 10.7 | 3.62 | 1.10 |
| mps4 | 4 | 1 | (1, 1, 1) | 5.45 (4.69-6.65) | 5.89 | 1.69-23.5 | 13.86 (2.57) | 1.07 | 11.4 | 2.18 | - |
| mps4 | 4 | 2 | (2, 2, 2) | 2.97 (2.68-3.41) | 2.99 | 1.48-7.60 | 3.93 (0.16) | 0.64 | 10.3 | 2.42 | 2.25 |
| mps4 | 4 | 4 | (4, 4, 4) | 1.92 (1.81-2.05) | 1.90 | 1.22-3.41 | 2.17 (0.06) | 0.41 | 12.8 | 3.13 | 1.78 |
| mps4 | 4 | 8 | (4, 8, 4) | 1.77 (1.68-1.87) | 1.75 | 1.09-2.90 | 1.92 (0.04) | 0.37 | 15.1 | 3.68 | 1.51 |
| mps4 | 4 | 16+ | (4, 16, 4) | 1.46 (1.39-1.53) | 1.50 | 0.95-2.23 | 1.57 (0.03) | 0.34 | 18.4 | 4.50 | 1.21 |

(bonds omit the root's 1; "+" = this and larger $\chi$, capped.) Readings: the lower bound with $r_{\rm m}$
holds on every init (min ratio 1.00; verifier 1.0003 over 5700 inits); for binary4 the median ratio to it
is 1.5-1.8 at every $(d,\chi)$; the naive ratio grows with $d$ and $\chi$.

### 4.7 Larger trees ($m\ge3$)

$\hat t_1$ medians are in table 2 (section 5). The naive formula is off by 30-12000x. A path-amplitude
product mode (ends at full norm; middle cores contracted with their children's path directions; root
projection) predicts binary8 $\hat t_1$ per init with log-correlation 0.77-0.84 and a mean log-bias of
+0.14 to +0.45, i.e. 15-57% too slow (80 inits each for $(d,\chi)$ = (2,2), (2,4), (4,4), (4,8);
heuristic). Use the recipe of 2.5 for quantitative work.

### 4.8 Answer to the brief's question (derived + empirical)

- The effective amplitude is not $\rho_0$: it is set by the reduced end cores,
  $c_e\approx\sigma\sqrt{\chi_e/d^{n_e}}$ for ends whose fan-in is purely physical (saturating at $\sigma$ at
  the cap $\chi_e=d^{n_e}$) and $\sigma d^{-n_e/2}$ for an end whose fan-in includes its bond (the MPS
  root), and by the internal path amplitudes when $m\ge2$.
- $\chi$-dependence at fixed $\sigma$ (empirical): log-log slopes of the median over $\chi\in[d,d^2]$ are
  $-0.20$ for MPS4 ($d=3,4$; verifier $-0.17$, $-0.20$) against $-0.43$ to $-0.47$ for the naive law, and on
  MPS8 ($d=4$) $\chi$ from 2 to 8 speeds the escape up 3.7x against 33x for the naive law. For binary4 the
  slope is $-0.38$ (verifier $-0.36$, $-0.37$) against $-0.45$ to $-0.47$ naive, only about 20% shallower, and the frozen-end value $-0.5$ is
  further from the data than the naive law (the closed form is accurate at the cap and 12-19% high at
  $\chi_b=d$). Saturation at the cap happens for every formula once $\chi$ is capped and is not evidence.
- The clean refutation of the naive law on binary4 is the $d$-dependence at full bond: $\hat t_1$ medians
  1.56, 1.55, 1.57 for $d$ = 2, 3, 4 (verifier 1.525, 1.542, 1.548), whereas $t_{\rm KH}\propto(2d^2+1)^{-1/2}$
  falls by 1.9x from $d=2$ to 4. MPS4: 1.30, 1.35, 1.46 against a naive fall of 2.8x ($\propto(d^2+2d+1)^{-1}$).

## 5. Learning order (task d)

### 5.1 Race model, exact for MPS (proven at leading order)

By 4.1, for trees where every core carries a site (MPS) and an odeco teacher, the components' LO
dynamics are independent; component $r$ blows up at $\sigma^{-(K-2)}\hat t_{1,r}/s_r$ with $\hat t_{1,r}$
iid. Writing $\hat t_{1,r}=c_r^{-(K-2)}/(K-2)$ this is the race "$r$ escapes at $1/((K-2)s_rc_r^{K-2})$",
with $c_r$ the effective amplitude of section 4. For weights $(1,\rho)$, by exchangeability,
$$P(\text{larger first})=P\!\left(\frac{\hat t_{1,2}}{\hat t_{1,1}}>\rho\right)=\frac12+\frac12P\!\left(\left\lvert\ln\frac{\hat t_{1,2}}{\hat t_{1,1}}\right\rvert<\ln\frac1\rho\right),$$
independent of $\sigma$. In the spherical-flow picture, the basin of component $r$ is exactly
$\{u_0:\hat t_{1,r}/s_r<\hat t_{1,q}/s_q\ \forall q\}$. Checked: race = direct two-component LO flow on 100/100
inits (MPS4) and = full GF on 100/100 inits ($\sigma=0.1,0.03$; verifier: 100/100 at $\sigma=0.05$, and
30/30 for three components with $t_1$ within 0.06% of the race time).

**Log-normal approximation (empirical).** $P\approx\Phi(\ln(1/\rho)/S)$ with
$S=\mathrm{sd}\ln(\hat t_{1,2}/\hat t_{1,1})$ ($=\sqrt2\,\mathrm{sd}\ln\hat t_1$ for independent components);
within $\pm0.05$ (mostly $\pm0.02$) of the empirical $P$ in all tensor-like settings of table 2.

**Trends.** (i) $\mathrm{sd}\ln\hat t_1$ grows like $\sqrt{K-2}$ on MPS (empirical): $\mathrm{sd}\ln\hat t_1/\sqrt{K-2}$
= 0.43, 0.49, 0.48 ($d=\chi=2$; $K$ = 4, 6, 8), 0.38, 0.37, 0.37 ($d=2,\chi=4$), 0.30, 0.34, 0.33 ($d=\chi=4$),
0.27, 0.27, 0.27 ($d=4,\chi=8$) (table 2). Heuristic reason: $\ln\hat t_1$ is dominated by a sum of about
$K-2$ independent log path amplitudes (4.3). v1's $S\approx\sqrt2(K-2)s_c$ (linear in $K-2$) contradicts these
numbers and is withdrawn. (ii) Hence $P(\text{larger first})\to\frac12$ as $K\to\infty$ at fixed $\rho,d,\chi$
(conjecture; follows if the $\sqrt{K-2}$ growth persists). (iii) $\mathrm{sd}\ln\hat t_1$ falls as the reduced
cores get more entries, but at fixed $d$ it saturates once $\chi$ reaches the cap (the MPS root end has only
$\min(\chi,d)$ entries): MPS4, $d=4$, $\rho=0.5$: $P$ = 0.86, 0.91, 0.92 at $\chi$ = 4, 8, 16 (verifier: 0.885,
0.920, 0.916). So $P\to1$ requires $d\to\infty$ as well as $\chi\ge d^{n_e}$ (heuristic), consistent with
over-parameterization restoring largest-first order [Ge2021; Ren2025].

### 5.2 Shared internal cores (binary8, HT4): approximate race (empirical)

Cores without sites are shared by all components, so the LO dynamics couple, but weakly: the race
built from single-component $\hat t_{1,r}$ (same init) agrees with the direct two-component LO flow on
94.2-100% of inits (120 inits per setting; binary8 and HT4 with $(d,\chi)$ = (2,2), (2,4), (4,4); $\rho$ =
0.5, 0.9; verifier: binary8 45/45, HT4 44/45 at $\chi=2$ and 38/40 at $\chi=4$, $\rho=0.9$). At $\chi=2$,
$\hat t_{1,1}$ and $\hat t_{1,2}$ are positively correlated (+0.19 to +0.30; near 0 at $\chi\ge4$), which makes
the order more greedy than independent components with the same marginals.

### 5.3 Matrix-like pairs: greedy as $\sigma\to0$ (known for binary4), with a slow power law (heuristic)

If the smaller component $q$ of a pair has $m_{qr}=2$, its configuration is a saddle of the spherical flow
with $n=n_{qr}$ unstable directions (3.3), while the larger one is a local max; so $\lim_\tau u(\tau)$ is the
larger component and $P(\text{smaller first})\to0$ as $\sigma\to0$. For binary4 this is known: it is a depth-3
deep linear network (`notes/theory_saddles.md` 4.2), and [Jacot2021] Thm 2 proves that the first escape of
a depth-$L\le3$ DLN from small Gaussian init goes to the top singular mode with probability 1.

New: the finite-$\sigma$ law (heuristic). A trajectory that reaches $r=O(1)$ near $q$'s configuration must
pass close to it; there the deviations along the $n$ unstable directions grow at rate
$K^{1-K/2}(s_r-s_q)$ in $\tau$ while $d\ln r/d\tau=K^{1-K/2}s_q$, i.e. like $(r/\rho_0)^{(1-\rho)/\rho}$ with
$\rho=s_q/s_r$. Escape before rotation needs all $n$ deviations to start below $\sim\sigma^{(1-\rho)/\rho}$
(since $r/\rho_0\sim1/\sigma$ at escape). With a smooth density of initial deviations,
$$P(\text{smaller first})\propto\sigma^{\,n(1-\rho)/\rho},\qquad n=n_{qr}=\begin{cases}1&\text{binary4, non-adjacent pairs}\\\chi_e&\text{adjacent pair sharing bond }e\end{cases}$$
(v1 had $n=1$ throughout). The same exponent with $n=1$ is in `notes/theory_saddles.md` 4.5. When the
wrong order happens, `notes/theory_saddles.md` 4.4 applies (correction through the strict piggyback
saddle for non-adjacent pairs; direct $k=2$ route, logarithmic, for adjacent pairs).

**Check of the multiplicity (LO emulation, MPS4 pattern teachers, table 4).** The fitted slopes
$d\ln P/d\ln\sigma$ order as predicted, $n=1<2<3<4$, with per-$n$ increments of about $0.09$ at $\rho=0.9$
against the predicted $0.111$ (a linear fit of slope against $n$ has intercept $-0.01$); they are
0.67-1.03 (typically 0.8) of $n(1-\rho)/\rho$ over $\sigma=10^{-1}$-$10^{-8}$ and steepen toward it at small $\sigma$ ($n=4$: local slope 0.24 at $\sigma\sim10^{-1.5}$, 0.43-0.48 at
$\sigma\sim10^{-4}$). The exponent is therefore supported as the $\sigma\to0$ value, with a slow pre-asymptotic
approach; its prefactor is not derived.

### 5.4 Table 2: $\hat t_1$ and race probabilities (leading order, $\sigma\to0$)

$P$ = symmetrized empirical $P(\hat t_{1,2}/\hat t_{1,1}>\rho)$ (bootstrap se $\le0.019$); log-normal column =
$\Phi(\ln(1/\rho)/S)$ for $\rho$ = 0.5/0.7/0.9; median with 95% CI (component 1). For binary4 the race is
shown for reference only; its actual order is greedy as $\sigma\to0$ (table 3).

| tree | K | d | chi | bonds | median $\hat t_1$ (95% CI) | 10%-90% | sd ln | median/naive | P 0.5 | P 0.7 | P 0.8 | P 0.9 | P 0.95 | log-normal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mps4 | 4 | 2 | 2 | (2, 2, 2) | 1.56 (1.45-1.75) | 0.79-3.53 | 0.61 | 5 | 0.79 | 0.68 | 0.62 | 0.56 | 0.53 | 0.79/0.66/0.55 |
| mps4 | 4 | 2 | 4 | (2, 4, 2) | 1.30 (1.21-1.47) | 0.70-2.90 | 0.54 | 6 | 0.82 | 0.68 | 0.61 | 0.55 | 0.52 | 0.82/0.68/0.55 |
| mps4 | 4 | 4 | 2 | (2, 2, 2) | 2.97 (2.68-3.41) | 1.48-7.60 | 0.62 | 10 | 0.78 | 0.65 | 0.59 | 0.54 | 0.52 | 0.78/0.65/0.55 |
| mps4 | 4 | 4 | 4 | (4, 4, 4) | 1.92 (1.81-2.05) | 1.22-3.41 | 0.42 | 13 | 0.86 | 0.70 | 0.62 | 0.57 | 0.54 | 0.87/0.72/0.57 |
| mps4 | 4 | 4 | 8 | (4, 8, 4) | 1.77 (1.68-1.87) | 1.09-2.90 | 0.38 | 15 | 0.91 | 0.75 | 0.67 | 0.57 | 0.54 | 0.90/0.75/0.58 |
| mps4 | 4 | 4 | 16 | (4, 16, 4) | 1.46 (1.39-1.53) | 0.95-2.23 | 0.34 | 18 | 0.92 | 0.79 | 0.69 | 0.60 | 0.55 | 0.92/0.77/0.59 |
| binary4 | 3 | 2 | 2 | (2, 2) | 1.95 (1.87-2.15) | 1.30-3.32 | 0.38 | 2 | (0.92) | (0.74) | (0.66) | (0.58) | (0.54) | - |
| binary4 | 3 | 2 | 4 | (4, 4) | 1.56 (1.53-1.62) | 1.17-2.18 | 0.26 | 3 | (0.97) | (0.82) | (0.72) | (0.63) | (0.56) | - |
| binary4 | 3 | 4 | 2 | (2, 2) | 3.09 (2.92-3.34) | 1.89-5.09 | 0.39 | 4 | (0.95) | (0.78) | (0.69) | (0.60) | (0.56) | - |
| binary4 | 3 | 4 | 4 | (4, 4) | 2.65 (2.52-2.72) | 1.91-3.80 | 0.25 | 5 | (0.98) | (0.87) | (0.77) | (0.65) | (0.59) | - |
| binary4 | 3 | 4 | 8 | (8, 8) | 2.09 (2.06-2.15) | 1.67-2.65 | 0.18 | 5 | (1.00) | (0.93) | (0.82) | (0.65) | (0.58) | - |
| binary4 | 3 | 4 | 16 | (16, 16) | 1.57 (1.54-1.61) | 1.34-1.84 | 0.12 | 5 | (1.00) | (0.98) | (0.90) | (0.72) | (0.61) | - |
| mps6 | 6 | 2 | 2 | (2, 2, 2, 2, 2) | 2.31 (1.95-3.11) | 0.67-10.0 | 0.97 | 29 | 0.69 | 0.61 | 0.57 | 0.53 | 0.52 | 0.70/0.61/0.53 |
| mps6 | 6 | 2 | 4 | (2, 4, 4, 4, 2) | 1.41 (1.24-1.66) | 0.61-3.61 | 0.73 | 44 | 0.74 | 0.65 | 0.59 | 0.54 | 0.52 | 0.74/0.63/0.54 |
| mps6 | 6 | 2 | 8 | (2, 4, 8, 4, 2) | 1.36 (1.18-1.59) | 0.58-3.63 | 0.76 | 63 | 0.77 | 0.65 | 0.60 | 0.54 | 0.51 | 0.75/0.63/0.54 |
| mps6 | 6 | 4 | 2 | (2, 2, 2, 2, 2) | 9.04 (7.61-10.6) | 2.76-27.5 | 0.91 | 115 | 0.69 | 0.60 | 0.57 | 0.54 | 0.51 | 0.70/0.61/0.53 |
| mps6 | 6 | 4 | 4 | (4, 4, 4, 4, 4) | 4.22 (3.69-5.06) | 1.80-11.4 | 0.67 | 201 | 0.76 | 0.63 | 0.58 | 0.54 | 0.52 | 0.75/0.64/0.54 |
| mps6 | 6 | 4 | 8 | (4, 8, 8, 8, 4) | 3.27 (2.88-3.66) | 1.64-6.57 | 0.54 | 377 | 0.78 | 0.65 | 0.62 | 0.55 | 0.52 | 0.80/0.67/0.55 |
| mps8 | 8 | 2 | 2 | (2, 2, 2, 2, 2, 2, 2) | 4.77 (3.74-5.88) | 1.21-26.9 | 1.17 | 182 | 0.67 | 0.58 | 0.55 | 0.53 | 0.52 | 0.66/0.59/0.53 |
| mps8 | 8 | 2 | 4 | (2, 4, 4, 4, 4, 4, 2) | 2.23 (1.92-2.82) | 0.73-6.74 | 0.90 | 372 | 0.68 | 0.62 | 0.57 | 0.53 | 0.52 | 0.71/0.61/0.53 |
| mps8 | 8 | 2 | 8 | (2, 4, 8, 8, 8, 4, 2) | 1.46 (1.27-1.81) | 0.51-4.69 | 0.89 | 852 | 0.73 | 0.62 | 0.57 | 0.54 | 0.52 | 0.72/0.62/0.53 |
| mps8 | 8 | 4 | 2 | (2, 2, 2, 2, 2, 2, 2) | 32.8 (25.7-43.4) | 8.88-199 | 1.16 | 1374 | 0.72 | 0.62 | 0.58 | 0.53 | 0.51 | 0.67/0.59/0.53 |
| mps8 | 8 | 4 | 4 | (4, 4, 4, 4, 4, 4, 4) | 13.5 (11.5-15.5) | 4.70-36.6 | 0.80 | 3724 | 0.75 | 0.64 | 0.60 | 0.57 | 0.54 | 0.73/0.62/0.54 |
| mps8 | 8 | 4 | 8 | (4, 8, 8, 8, 8, 8, 4) | 8.95 (7.58-10.5) | 3.72-20.0 | 0.67 | 12378 | 0.75 | 0.63 | 0.60 | 0.54 | 0.52 | 0.77/0.65/0.54 |
| binary8 | 7 | 2 | 2 | (2, 2, 2, 2, 2, 2) | 4.26 (3.65-5.16) | 1.20-18.1 | 1.13 | 94 | 0.69 | 0.60 | 0.56 | 0.54 | 0.52 | 0.68/0.60/0.53 |
| binary8 | 7 | 2 | 4 | (4, 4, 4, 4, 4, 4) | 1.68 (1.48-2.07) | 0.65-4.37 | 0.74 | 196 | 0.76 | 0.64 | 0.57 | 0.53 | 0.51 | 0.74/0.63/0.54 |
| binary8 | 7 | 2 | 8 | (4, 4, 8, 4, 4, 8) | 1.24 (1.05-1.43) | 0.55-3.60 | 0.66 | 290 | 0.77 | 0.66 | 0.58 | 0.54 | 0.52 | 0.77/0.65/0.54 |
| binary8 | 7 | 4 | 2 | (2, 2, 2, 2, 2, 2) | 22.7 (16.7-27.4) | 6.14-89.2 | 1.04 | 534 | 0.71 | 0.61 | 0.57 | 0.54 | 0.52 | 0.71/0.61/0.53 |
| binary8 | 7 | 4 | 4 | (4, 4, 4, 4, 4, 4) | 11.3 (9.57-12.5) | 4.95-27.8 | 0.68 | 1309 | 0.77 | 0.64 | 0.60 | 0.54 | 0.53 | 0.78/0.66/0.55 |
| binary8 | 7 | 4 | 8 | (8, 8, 8, 8, 8, 8) | 6.87 (6.17-7.48) | 3.76-13.3 | 0.50 | 4463 | 0.85 | 0.73 | 0.64 | 0.56 | 0.53 | 0.86/0.71/0.57 |
| ht4 | 7 | 2 | 2 | (2, 2, 2, 2, 2, 2) | 1.57 (1.20-1.87) | 0.40-7.10 | 1.15 | 35 | 0.69 | 0.60 | 0.57 | 0.54 | 0.52 | 0.69/0.60/0.53 |
| ht4 | 7 | 2 | 4 | (2, 2, 4, 2, 2, 4) | 1.10 (0.94-1.42) | 0.32-4.87 | 1.04 | 53 | 0.68 | 0.60 | 0.57 | 0.53 | 0.52 | 0.69/0.60/0.53 |
| ht4 | 7 | 4 | 2 | (2, 2, 2, 2, 2, 2) | 4.13 (3.43-4.79) | 0.90-22.2 | 1.17 | 93 | 0.68 | 0.60 | 0.55 | 0.53 | 0.52 | 0.68/0.60/0.53 |
| ht4 | 7 | 4 | 4 | (4, 4, 4, 4, 4, 4) | 1.63 (1.38-1.72) | 0.60-4.46 | 0.75 | 181 | 0.76 | 0.64 | 0.58 | 0.54 | 0.52 | 0.75/0.64/0.54 |
| ht4 | 7 | 4 | 8 | (4, 4, 8, 4, 4, 8) | 1.13 (1.01-1.27) | 0.48-2.74 | 0.69 | 270 | 0.76 | 0.63 | 0.58 | 0.53 | 0.52 | 0.75/0.64/0.54 |

(300 inits for $N=4$ rows, 200 otherwise.)

### 5.5 Table 3: binary4, $P(\text{smaller first})$ versus $\sigma$ (Wilson 95% CIs)

Leading-order emulation (200 inits; $a_r=\frac12$ crossing computed from the $\sigma=1$ LO trajectory by
homogeneity, $\theta_\sigma(t)=\sigma\theta_1(\sigma^{K-2}t)$; the verifier's independent emulation matches it
per init on 200/200 inits). The slope column is the fitted $d\ln P/d\ln\sigma$ of the **emulation** (points
with $\ge5$ events), not of GF.

| d, chi | rho | sigma: 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-6 | 1e-8 | emulation slope (pred $(1-\rho)/\rho$) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2, 2 | 0.7 | 0.25 (0.20-0.31) | 0.11 (0.07-0.16) | 0.05 (0.03-0.09) | 0.015 (0.01-0.04) | 0 (0-0.02) | 0 | 0.37 (0.43) |
| 2, 2 | 0.9 | 0.48 (0.41-0.55) | 0.39 (0.32-0.45) | 0.32 (0.26-0.39) | 0.27 (0.21-0.33) | 0.21 (0.15-0.27) | 0.13 (0.09-0.18) | 0.077 (0.111) |
| 2, 4 | 0.7 | 0.20 (0.15-0.26) | 0.07 (0.04-0.11) | 0.03 (0.01-0.06) | 0.005 (0-0.03) | 0 | 0 | 0.43 (0.43) |
| 2, 4 | 0.9 | 0.38 (0.31-0.44) | 0.31 (0.25-0.37) | 0.28 (0.22-0.35) | 0.21 (0.16-0.27) | 0.14 (0.10-0.19) | 0.08 (0.05-0.12) | 0.10 (0.11) |
| 4, 4 | 0.9 | 0.37 (0.30-0.43) | 0.31 (0.25-0.38) | 0.26 (0.20-0.32) | 0.22 (0.17-0.28) | 0.14 (0.10-0.19) | 0.08 (0.05-0.12) | 0.097 (0.111) |
| 4, 8 | 0.9 | 0.37 (0.31-0.44) | 0.30 (0.24-0.36) | 0.24 (0.18-0.30) | 0.19 (0.14-0.24) | 0.11 (0.07-0.16) | 0.06 (0.03-0.10) | 0.12 (0.11) |

Larger samples (d = 2, $\chi$ = 2): v2 emulation, 2000 inits, other teacher basis, $\rho=0.7$: 0.222, 0.096,
0.040, 0.015 at $\sigma=10^{-1}$-$10^{-4}$ (local slopes 0.36, 0.38, 0.43); verifier, 1500 inits: 0.203,
0.078, 0.033, 0.012 (0.42, 0.38, 0.45); verifier, 1100 inits, $\rho=0.9$: slope 0.096.

**Full GF** ($d=\chi=2$; library loss; first crossing of $a_r=\frac12$):

| rho | sigma | v1 (100 inits) | verifier (100 inits, other teacher and seeds) | pooled |
| --- | --- | --- | --- | --- |
| 0.7 | 0.1 | 0.16 (0.10-0.24) | 0.13 (0.08-0.21) | 0.145 (0.10-0.20) |
| 0.7 | 0.01 | 0.06 (0.03-0.12) | 0.02 (0.01-0.07) | 0.04 (0.02-0.08) |
| 0.7 | 0.001 | 0.02 (0.01-0.07) | 0.00 (0-0.04) | 0.01 (0.003-0.04) |
| 0.9 | 0.1 | 0.39 (0.30-0.49) | 0.37 (0.28-0.47) | 0.38 (0.32-0.45) |
| 0.9 | 0.01 | 0.34 (0.25-0.44) | - | 0.34 (0.25-0.44) |
| 0.9 | 0.001 | 0.27 (0.19-0.36) | 0.24 (0.17-0.33) | 0.255 (0.20-0.32) |

**GF versus emulation, per init (v2, `escape_v2/gf_cond.py`).** 2000 new inits (other teacher basis),
$d=\chi=2$, $\rho=0.7$. At each $\sigma$, full GF on 80 inits that the emulation calls "smaller first" (all
of them at $\sigma=10^{-3}$) and on the 40 inits it calls "larger first" with the smallest crossing margin:

| sigma | emulation $P$ (2000 inits) | GF smaller first, among emulation-smaller | among the 40 closest emulation-larger | GF $P$ estimate |
| --- | --- | --- | --- | --- |
| 0.1 | 0.222 (0.204-0.241) | 56/80 = 0.70 (0.59-0.79) | 0/40 | 0.155 (0.130-0.181) |
| 0.01 | 0.096 (0.084-0.110) | 53/80 = 0.66 (0.55-0.76) | 0/40 | 0.064 (0.051-0.077) |
| 0.001 | 0.040 (0.032-0.050) | 47/80 = 0.59 (0.48-0.69) | 0/40 | 0.0235 (0.017-0.030) |

GF agrees with the emulation on all 120 near-boundary "larger first" inits, and puts the smaller component
first on 59-70% of the inits where the emulation does (heuristic reason: GF keeps the term $\tfrac12\lVert W\rVert^2$ of the loss,
which slows whichever component is ahead as it approaches $\tfrac12$ and lets the faster-growing larger
component overtake). The ratio falls slowly (0.70 to 0.59 over two decades), so in this range
the GF local slopes, 0.38 and 0.43 (about $\pm0.06$), are 0.02-0.05 steeper than the emulation's (0.36,
0.38). Whether the ratio tends to a constant as $\sigma\to0$, which would make the GF exponent equal to the
LO one, is open; the pooled GF points above (8 and 2 events) cannot decide it.

At $\rho=0.5$: $P\le0.075$ at $\sigma=0.1$ and $\le0.005$ for $\sigma\le0.01$ in all four emulation settings.

### 5.6 Table 4: matrix-like pattern teachers on MPS4 (LO emulation, 500 inits each)

Component 1 (weight 1) has the pattern shown, component 2 (weight $\rho$) is all zeros; $n$ = predicted
number of unstable directions of component 2's configuration (3.3). $P$(smaller first) with Wilson 95% CIs;
slope = fitted $d\ln P/d\ln\sigma$ over the points with $\ge5$ events (bootstrap 95% CI over inits)
(`escape_v2/emul_patterns.py`, `analyze_emul.py`).

| d, chi | pattern (cores) | $n$ | rho | sigma: 1e-1 | 1e-2 | 1e-4 | 1e-6 | 1e-8 | slope | pred $n(1-\rho)/\rho$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2, 2 | 1010 (0, 2) | 1 | 0.9 | 0.37 (0.33-0.42) | 0.29 (0.25-0.33) | 0.19 (0.16-0.23) | 0.14 (0.11-0.18) | 0.08 (0.06-0.11) | 0.088 (0.073-0.107) | 0.111 |
| 2, 4 | 1010 (0, 2) | 1 | 0.9 | 0.38 (0.34-0.42) | 0.29 (0.26-0.34) | 0.18 (0.15-0.21) | 0.14 (0.11-0.17) | 0.09 (0.06-0.11) | 0.085 (0.071-0.103) | 0.111 |
| 2, 2 | 1100 (0-1, $\chi_e=2$) | 2 | 0.9 | 0.39 (0.34-0.43) | 0.29 (0.25-0.33) | 0.14 (0.11-0.17) | 0.06 (0.04-0.08) | 0.03 (0.02-0.05) | 0.167 (0.142-0.204) | 0.222 |
| 3, 3 | 1100 (0-1, $\chi_e=3$) | 3 | 0.9 | 0.33 (0.29-0.37) | 0.22 (0.18-0.25) | 0.06 (0.04-0.08) | 0.018 (0.009-0.034) | 0.004 (0.001-0.014) | 0.27 (0.22-0.36) | 0.333 |
| 2, 4 | 0110 (1-2, $\chi_e=4$) | 4 | 0.9 | 0.36 (0.32-0.41) | 0.21 (0.18-0.25) | 0.04 (0.03-0.06) | 0.006 (0.002-0.017) | 0 (0-0.008) | 0.36 (0.30-0.48) | 0.444 |
| 2, 2 | 1010 (0, 2) | 1 | 0.8 | 0.27 (0.24-0.32) | 0.18 (0.15-0.21) | 0.06 (0.04-0.08) | 0.014 (0.007-0.029) | 0.006 (0.002-0.017) | 0.26 (0.21-0.34) | 0.25 |
| 2, 2 | 1100 (0-1, $\chi_e=2$) | 2 | 0.8 | 0.28 (0.24-0.32) | 0.12 (0.10-0.15) | 0.020 (0.011-0.036) | 0.008 (0.003-0.020) | 0.002 (0-0.011) | 0.34 (0.27-0.47) | 0.50 |

Fits restricted to $\sigma\le10^{-3}$: $n=4$: 0.43 (0.32-0.67); $n=3$: 0.28 (0.20-0.42); $n=2$, $\rho=0.9$: 0.175
(0.14-0.23); $n=1$: 0.076-0.088. The non-adjacent pair is insensitive to $\chi$ (rows 1-2), the adjacent
pairs are not.

## 6. Numerical checks run (summary)

| check | setting | result |
| --- | --- | --- |
| Saxe / aligned init | binary4, MPS4, binary8; s = 1, 2 | LO blow-up vs $a_0^{-(K-2)}/((K-2)s)$: $2\cdot10^{-11}$; GF vs exact integral: $10^{-11}$ |
| LO vs full GF per init | binary4 and MPS4, d = 2, 3, chi = 1-9, 20 inits each; binary8, HT4, MPS6, MPS4 (d=3, chi=9), 4 inits each | max relative deviation 0.8% (binary4, $\sigma=0.01$), 0.1% (MPS4, $\sigma=0.03$), $\le0.5\%$ ($K$ = 6, 7 at $\sigma=0.15$-0.3) |
| $\delta_K$ (v2) | quadrature and direct ODE, $K$ = 3-8 | agree to 4 digits |
| fast reduced flow vs library LO flow | 6 trees/settings, 2 teachers (v1); pattern-teacher version, 3 settings x 2 inits (v2) | $\hat t_1$ agrees to $10^{-11}$ |
| $\hat t_1$ tables | 30 N = 4 settings x 300 inits; 24 larger-tree settings x 200 inits | tables 1, 2 (v2: 95% CIs) |
| global max of $g$ | grouped vs ungrouped teacher, binary4 vs MPS4 | 0.19245 vs 0.03125 as predicted, 6/6 inits |
| Hessian at component configurations | 7 odeco settings (v1); 19 pattern-teacher cases (v2) | positive eigenvalues: number $n_{qr}$ and value $K^{1-K/2}(s_r-s_q)$ as predicted in every case; zero modes $\sum_e(\chi_e-1)$ |
| endpoints of spherical flow | 200 runs, 5 trees | $g=s_rK^{-K/2}$ to 6 digits |
| end cores during the escape (v2) | binary4, MPS4, 11 settings x 200 inits | equal norm shift to $10^{-8}$; lock never violated; table of 4.2(c) |
| race vs direct LO flow | MPS4 (100), binary8 and HT4 (120 each, 3 settings) | 100% (MPS4), 94.2-100% (shared cores) |
| race vs full GF per init | MPS4 d=2 chi=2, 100 inits, $\rho$ = 0.5, 0.9, $\sigma$ = 0.1, 0.03 | 100% agreement; $P$ 0.26 and 0.49, flat in $\sigma$ |
| binary4 order vs $\sigma$ | emulation 4 settings x 200 inits (v1), 2000 inits (v2); full GF 100 inits x 3 $\sigma$ x 2 $\rho$ (v1), 360 runs selected by the emulation (v2) | table 3; GF/emulation 0.59-0.70 |
| matrix-like pattern teachers (v2) | MPS4, 7 settings x 500 inits, LO emulation | table 4 |

Full-GF runs: `ttn.losses.population_loss` + torch autograd + scipy `solve_ivp` (DOP853, rtol
$10^{-9}$-$10^{-10}$, atol $10^{-14}\sigma$), conservation-law drift $\le10^{-10}$.

## 7. Open issues

- A closed form for internal path amplitudes on general trees ($m\ge3$: binary8, HT, MPS$N\ge6$): the
  path-amplitude mode is only accurate to a factor 1.2-1.6; the $K=3$ transport analysis (4.4) applied
  recursively should improve it. Related: why the ends barely rotate (4.2(c)) is not derived.
- The prefactor of $\mathrm{sd}\ln\hat t_1\approx c(d,\chi)\sqrt{K-2}$ (5.1) and whether the growth persists
  (it decides the conjectured $P\to\frac12$).
- Matrix-like pairs: the exponent $n(1-\rho)/\rho$ is approached slowly (table 4) and its prefactor is not
  derived; whether the GF/emulation ratio (0.70, 0.66, 0.59 at $\sigma=10^{-1},10^{-2},10^{-3}$) tends to a
  constant, i.e. whether GF has the same exponent as LO, is open.
- Large-bond asymptotics with `cap=False` (redundant bonds): the ends' amplitude keeps growing as
  $\sqrt{\chi_e}$, so $t_1\to0$ like $\chi_e^{-1/2}$ (binary4) in that parameterization (derived, not run).
- Non-orthogonal CP and TTN teachers: local maxima of $g$ beyond rank-one are not characterized.
- For $K\ge6$ the approach of $t_1\sigma^{K-2}s/\hat t_1$ to 1 has a coefficient 2-30 times the $\delta_K$ term at
  accessible $\sigma$; its $\sigma$-dependence (convergence of $u_{\rm esc}$) is not quantified.

## 8. Verification (v1 -> v2): issues and resolutions

An independent verifier re-derived E1-E9 with separate code (core library only) and confirmed the
per-init LO prediction, the bounds, the global maximum, tables 1-2 and the MPS race. Issues and what
changed:

| # | issue | resolution |
| --- | --- | --- |
| 1 | Piggyback multiplicity wrong for adjacent cores | Fixed (3.3): $n_{qr}=\chi_e$ for adjacent cores, 1 otherwise; 19 Hessians checked; order exponent becomes $n(1-\rho)/\rho$ and was checked by emulation (5.3) |
| 2 | "Conservation laws lock the ends" not derived | Replaced (4.2): proven equal-norm-shift identity and a lock along $\hat h_0$ valid only for $\kappa_e\ll1$; frozen ends now empirical (measured) and the mode heuristic; "internal cores near zero" corrected for MPS4 |
| 3 | Known results presented as new | Cited: trichotomy of 4.3 = [RawalDeWeese2026] Thm 4.2 and eq. (29) (checked on arXiv 2605.01288); binary4 greedy limit = [Jacot2021] Thm 2; equal norm shift = [Arora2018; DuHuLee2018] |
| 4 | "$m\le1$" formula holds only for $m=1$ | Fixed (4.3); $m=0$ is Saxe |
| 5 | Prediction 3 tolerance too tight | Now $\le20\%$ for $\chi_b<2d$, $\le8\%$ for $\chi_b\ge2d$, closed form overestimates; ratio to the "red" bound 1.5-1.8 |
| 6 | E7 / prediction 5 overstated for binary4 | Rewritten: $\chi$-slope only ~20% shallower than naive for binary4; the clean refutation there is $d$-independence at full bond; MPS4 and $K\ge6$ keep the strong statement; cap-saturation removed as evidence; PLAN reference corrected (the $\tau_{\rm align}$ form was in an early draft) |
| 7 | Table 3 / prediction 10 lack error bars; GF exponent not measured | Wilson 95% CIs added, slopes labelled as emulation slopes, verifier's GF points pooled, conditional GF check run (5.5) |
| 8 | Medians in tables 1-2 lack sampling error | 95% CIs added (order statistics) |
| 9 | $S\approx\sqrt2(K-2)s_c$ contradicts table 2; limits untagged | sd ln $\hat t_1\propto\sqrt{K-2}$ (empirical); $P\to\frac12$ conjecture; $P\to1$ needs $d\to\infty$ too |
| 10 | "Conservation laws change the time, not the direction" misleading | Rewritten (2.3): they shape the phase that selects the basin; they do not change the balanced form of the limit |
| 11 | $\delta_K$ constants and size of the error term | $\delta_K$ recomputed by quadrature; residual 2-30 times the $\delta_K$ term for $K=6,7$ at accessible $\sigma$ stated (2.4, prediction 1) |
| 12 | E4 "iff" omits $m_{qr}=1$ and ties | Fixed (E4, 3.3) |

Rebuttals: none of substance. Two refinements of the verifier's points: (2) the conservation laws do give
a proven lock, but on the component along $\hat h_0$ and only where $\kappa_e\ll1$, and the reason ends
rarely shrink is the proven identity of 4.2(a) (cores shrink only while $g<0$, and all by the same
amount); (1) `notes/theory_saddles.md` already counts the $\chi_e-r_e$ new-index directions of adjacent
pairs for the loss Hessian (its 3.2, $k=2$ routes), so its saddle classification stands, but its 4.5
probability heuristic $\sigma^{(s_r-s_q)/s_q}$ needs the same factor $n_{qr}$ in the exponent for adjacent
pairs.

## 9. Predictions for experiments

Fan-in init at scale $\sigma$, whitened inputs, odeco teachers unless stated; $t_1$ = first time some
$a_r$ crosses $\frac12$; $\hat t_1$ from recipe 2.5; "median" = median over inits at fixed teacher.

1. **Exponent and per-init prefactor** (proven at LO). For every init,
   $t_1\sigma^{K-2}s_1/\hat t_1(Z)\to1$ as $\sigma\to0$, with no free parameter (several components: run 2.5
   with the actual teacher). The exponent counts all cores: 1 (binary4), 2 (MPS4), 4 (MPS6), 5 (binary8,
   HT4), 6 (MPS8). Residual: $\delta_Ks_1^{-1+2/K}\sigma^{K-2}/\hat t_1$ with $\delta_3=-0.850$, $\delta_4=-0.266$
   (to 2 digits for $K=3,4$); for $K=6$-8 ($\delta_6=0.072$, $\delta_7=0.148$, $\delta_8=0.201$) expect 2-30 times
   that at $\sigma=0.15$-0.3. For a 1% test use $\sigma^{K-2}\le10^{-3}$.
2. **The Kumar-Haupt/$\rho_0$ law is a strict lower bound and fails quantitatively** (proven bound, empirical
   size). $t_1/t_{\rm KH}\ge1$ on every init, $t_{\rm KH}=K^{K/2-1}\rho_0^{-(K-2)}/((K-2)s_1)$; medians 2.5-5.2
   (binary4, $d$ = 2-4), 5.2-18 (MPS4), 29-377 (MPS6), 182-12378 (MPS8), 94-4463 (binary8), 35-270 (HT4)
   over the settings of tables 1-2, growing with $d$, $\chi$ and $K$.
3. **binary4 closed form** (derived given frozen ends; checked). Median $t_1\sigma s=c\,\pi d/(2\sqrt{\chi_b})$
   with $c\in[0.80,0.95]$ for $d\le\chi_b<2d$ and $c\in[0.92,1.00]$ for $\chi_b\ge2d$ (the closed form
   overestimates); at full bond the median is $\pi/2=1.57\pm0.04$ for $d$ = 2, 3, 4;
   $\mathrm{sd}\ln\hat t_1=1/(2\sqrt{\chi_b})\pm0.02$; median $\hat t_1$ / (bound with $r_{\rm m}$) = 1.5-1.8.
4. **MPS4** (derived given frozen ends, fitted factor). Median $\hat t_1=(1.05$-$1.39)\,\hat t_P$ (4.5);
   $\mathrm{sd}\ln\hat t_1$ = 0.34-0.64 for $\chi\ge2$, not below about 0.34 at $d=4$.
5. **$d$-dependence at full bond** (empirical; the decisive test of the $\rho_0$ law on binary4). Median
   $t_1\sigma^{K-2}s$: binary4 1.55-1.57 for $d$ = 2, 3, 4 ($t_{\rm KH}$ falls 1.9x); MPS4 1.30, 1.35, 1.46
   ($t_{\rm KH}$ falls 2.8x).
6. **$\chi$-dependence at fixed $\sigma$** (empirical, mechanism derived). MPS4: log-log slope of the
   median over $\chi\in[d,d^2]$ $-0.20\pm0.05$ ($t_{\rm KH}$: $-0.43$ to $-0.47$); MPS8, $d=4$: $\chi$ from 2 to 8
   shortens the median 3.7x ($t_{\rm KH}$: 33x). binary4: $-0.36$ to $-0.38$ against $-0.45$ to $-0.47$, not a
   discriminating test.
7. **Distribution shape** (empirical). $\chi=1$ gives heavy right tails (sd ln 0.6-1.1; mean/median up to
   2.5): compare medians and log-scale statistics; with 200-300 inits median CIs are $\pm2$-10%
   (tables 1-2).
8. **Core norms** (proven; a free consistency check). Along GF all cores' $\lVert G_v\rVert^2$ change by
   the same amount $2\int_0^t(g-\lVert W\rVert^2)$; at small $\sigma$ all cores first shrink iff
   $\langle W_0,W^*\rangle<0$ (half of all inits) and stop shrinking when it turns positive (up to
   $O(\sigma^K)$ corrections from $\lVert W\rVert^2$). In the reduced variables (odeco, LO) the end cores dip
   below 0.9 of their initial norm on 0-21% of inits (4.2(c)).
9. **Hessian of $g$ on the sphere at component configurations** (derived, checked). Number of positive
   eigenvalues at $\theta_q$ = $\sum_{r:\,m_{qr}=2,\,s_r>s_q}n_{qr}$ ($n=\chi_e$ for adjacent cores, else 1), each
   $K^{1-K/2}(s_r-s_q)$: e.g. one, 0.2309, for the smaller odeco component on binary4 at $\rho=0.6$; two,
   0.1 each, for MPS4 pattern 1100 at $d=\chi=2$, $\rho=0.6$; none for odeco components on MPS, HT, binary8;
   zero modes $\sum_e(\chi_e-1)$.
10. **Order on tensor-like trees** (MPS: proven at LO; binary8, HT4: approximate). $P(\text{larger first})$
    does not depend on $\sigma$ and is given by table 2 (bootstrap se $\le0.019$): e.g. $\rho=0.5$: 0.79 (MPS4,
    $d=\chi=2$), 0.69 (MPS6), 0.67 (MPS8), 0.69 (binary8), 0.69 (HT4) at $d=\chi=2$; $\rho=0.9$: 0.53-0.60
    everywhere. Per init on MPS the first component is $\arg\min_r\hat t_{1,r}/s_r$ (also for 3+ components).
    $P$ rises with $\chi$ at fixed $d$ and saturates (MPS4, $d=4$, $\rho=0.5$: 0.86, 0.91, 0.92 at $\chi$ = 4, 8,
    16) and falls toward $\frac12$ with $K$ ($\mathrm{sd}\ln\hat t_1\approx c\sqrt{K-2}$, $c$ = 0.27-0.49). Control:
    on MPS4 ($d=\chi=2$, 100 inits) GF gives $P(\text{smaller first})$ = 0.26 ($\rho=0.5$) and 0.49 ($\rho=0.9$) at
    both $\sigma$ = 0.1 and 0.03, identical per init to the race.
11. **Order on matrix-like pairs** (limit known for binary4; exponent heuristic; GF values empirical).
    $P(\text{smaller first})$ decreases like $\sigma^{n(1-\rho)/\rho}$ as $\sigma\to0$, $n$ as in 9. Full GF,
    binary4, $d=\chi=2$, $\rho=0.7$: 0.155 (0.13-0.18), 0.064 (0.05-0.08), 0.024 (0.017-0.030) at $\sigma$ = 0.1,
    0.01, 0.001 (2000-init estimate; local slopes 0.38, 0.43; direct GF on 200 inits: 0.145, 0.04, 0.01);
    $\rho=0.9$: 0.38 (0.32-0.45), 0.34 (0.25-0.44), 0.255 (0.20-0.32). MPS4 pattern teachers ($\rho=0.9$, LO emulation,
    table 4): fitted slopes 0.085-0.088 ($n=1$, pattern 1010), 0.17 ($n=2$, 1100, $d=\chi=2$), 0.27 ($n=3$,
    1100, $d=\chi=3$), 0.36-0.43 ($n=4$, 0110, $d=2$, $\chi=4$). Test: on MPS4 with $d=\chi=2$, $\rho=0.9$, the slope
    for pattern 1100 should be about twice that for 1010, and only the adjacent pattern should depend on $\chi$.
12. **Escape time with several components on MPS** (proven at LO): the first escape happens at
    $\sigma^{-(K-2)}\min_r\hat t_{1,r}/s_r$ (verifier: within 0.06% on 30/30 inits of a $(3,2,1)$ teacher). Later
    events involve the full loss (`notes/theory_saddles.md`).
