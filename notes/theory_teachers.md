# Theory: which saddles are visited for teachers that are not odeco (Q7)

Theory note for `PLAN.md` Q7 (2026-10-09). Builds on `notes/model.md` (model, fan-in init,
conservation laws) and hypotheses H1–H3 of `notes/prior_knowledge.md`. `notes/theory_escape.md`,
`notes/theory_saddles.md`, `notes/theory_regimes.md` and `notes/literature.md` did not exist yet when
this was written. Section 3 states the locality rule in the general form needed here; it reduces to
H2 for pattern teachers. Notation should be reconciled with `theory_saddles.md` once that note exists.

**Status tags.** *proven*: complete argument given. *derived*: calculation under stated assumptions.
*heuristic*: plausible argument, not a derivation. *conjecture*. "[checked §8.x]" marks a
numerical confirmation, with the numbers in Section 8.

## 0. Results at a glance

| # | result | status |
|---|---|---|
| R1 | A tensor with fixed bond ranks is a critical point of $L$ iff its matricization at every edge is an SVD truncation of the *environment-projected teacher* $T_e$ | proven |
| R2 | An HSVD truncation is critical iff its frames solve R1. It is critical (and a global minimizer) if all single-edge truncations coincide. For generic teachers it is not critical. For near-consistent teachers the loss gap is $O(\varepsilon^4)$ | proven / derived, [checked §8.1] |
| R3 | Small-init gradient flow (GF) visits critical points, not HSVD truncations. The first plateau is a *local* best "bond-rank-one" approximation | derived, [checked §8.2, §8.4] |
| R4 | Escapes from a saddle go along *routes* (connected sets of $k$ cores). $k=2$: strict saddle, time $\propto\log(1/\sigma)$. $k\ge3$: higher-order saddle, time $\propto\sigma^{-(k-2)}$. Generalized locality rule: $k_{\min}$ = smallest admissible route with nonzero overlap with the residual | derived |
| R5 | At the rank-one saddle of an MPS student, bond $e$ escapes at rate $\gamma_e=\lambda^{(N-2)/N}\sigma_2(M_e)$, and $\sigma_2(M_e)\le\sigma_2^e(T)$ | derived (bound proven), [checked §8.2–8.4] |
| R6 | Second escape: $\Delta t_2=A_{e^*}\log(1/\sigma)+O(1)$ with $A_e=(1-\sigma_2(M_e)/\lambda)/\gamma_e$ | derived, [checked §8.3: <1%] |
| R7 | Two non-orthogonal rank-one components: rank-one maxima, threshold $c^*=(N-2)/N$, fold formula. The first plateau is a local best rank-one approximation, never a component | proven / derived, [checked §8.2] |
| R8 | The second phase adds rank bond by bond through strict saddles. Intermediate saddles are critical points of mixed-rank problems. A race model predicts order and timing | derived / heuristic, [checked §8.3] |
| R9 | Generic teachers: an MPS student has one polynomial plateau, then logarithmic ones. Binary/HT students need $k\ge3$ (polynomial) at the rank-one saddle, and always for the root bipartition | derived; MPS and binary4 [checked §8.4–8.5] |
| R10 | Teacher on another tree: required student ranks are cut bounds. A binary4 student is a depth-3 deep linear network (DLN) on $T_{(01\mid23)}$ (nested SVD saddles, $\sigma^{-1}$ plateaus); an MPS student on the same teacher has logarithmic later plateaus | proven / derived, [checked §8.5] |

## 1. Setting and notation

- $N$ sites, physical dimension $d$, tree $\tau$ with $K$ cores. An edge $e=(v,p)$ joins child $v$ to
  parent $p$. $\mathrm{below}(e)$ = sites in the subtree of $v$; $\mathrm{above}(e)$ = the rest.
- Teacher $T=W^*$. Inputs are whitened, so $L(W)=\tfrac12\|W-T\|_F^2$. A product covariance
  $\Sigma=\otimes_i\Sigma_i$ reduces to this case by $T\to\Sigma^{1/2}T$, $W\to\Sigma^{1/2}W$, since the
  bond ranks are unchanged.
- $X_{(e)}$: matricization of a tensor $X$ across $e$ (rows = $\mathrm{below}(e)$). Its singular values
  are the bond (Schmidt) spectrum $\sigma_1^e(X)\ge\sigma_2^e(X)\ge\dots$.
- Bond ranks $r=(r_e)$. $\mathcal M_r$ = tensors with exact ranks $r_e=\mathrm{rank}\,X_{(e)}$; this is a
  smooth manifold (Uschmajew & Vandereycken 2013 for HT; Holtz, Rohwedder & Schneider 2012 for TT;
  the same argument works for any tree). $\mathcal M_{\le r}$ is its closure.
- Frames of $W\in\mathcal M_r$ at $e$: $U_e$ = column space of $W_{(e)}$ (dimension $r_e$, a subspace of
  $\mathbb R^{d^{|\mathrm{below}(e)|}}$) and $V_e$ = row space. $P_S$ is the orthogonal projector onto $S$.
- Local space of node $v$: $\mathcal V_v=\big(\otimes_{c\in\mathrm{ch}(v)}U_c\big)\otimes\mathbb R^{d_v}\otimes V_v$,
  where $d_v=d^{\#\text{sites of }v}$ and the $V_v$ factor is absent at the root.
- Environment spaces of edge $e=(v,p)$:
  $E_e^-=\big(\otimes_{c\in\mathrm{ch}(v)}U_c\big)\otimes\mathbb R^{d_v}$ (on $\mathrm{below}(e)$) and
  $E_e^+=\big(\otimes_{c\in\mathrm{ch}(p)\setminus v}U_c\big)\otimes\mathbb R^{d_p}\otimes V_p$ (on $\mathrm{above}(e)$).
  Then $\mathcal V_v=E_e^-\otimes V_e$ and $\mathcal V_p=U_e\otimes E_e^+$.
- **Environment-projected teacher**: $T_e:=P_{E_e^-}\,T_{(e)}\,P_{E_e^+}$, a matrix.
- HSVD (Grasedyck 2010, root-to-leaves): $\pi_e$ = projector onto the top-$r_e$ left singular
  subspace of $T_{(e)}$, all computed from $T$; $W_H=\prod_e\pi_e\,T$, applied level by level.
- Student init: fan-in scale $\sigma$ per core (`notes/model.md`). GF time is $t$.

## 2. (a) Critical points of the bond-rank-constrained problem; when is HSVD one?

### 2.1 Characterization

**Lemma (matrix).** *proven.* Let $A$ be a matrix, and $U$, $V$ subspaces of dimension $r$ with
$P_UA=AP_V=:W$ of rank $r$. Then $U$ is spanned by $r$ left singular vectors of $A$, and $W$ is the
SVD truncation of $A$ onto them. *Proof:* $P_UAA^\top=AP_VA^\top=WW^\top$ is symmetric, so $P_U$
commutes with $AA^\top$. For $N=2$ this is the classical result that the critical points of
rank-$r$ approximation are truncations onto any $r$ singular pairs (Eckart–Young; Baldi & Hornik 1989).

**Proposition 1 (critical points on $\mathcal M_r$).** *proven.* $W\in\mathcal M_r$ is a critical point
of $L|_{\mathcal M_r}$ iff, for every edge $e$,
$$W_{(e)}=P_{U_e}T_e=T_eP_{V_e},$$
that is, iff $U_e$ is spanned by $r_e$ left singular vectors of $T_e$ and $W_{(e)}$ is the
corresponding SVD truncation of $T_e$. Equivalently, $W$ is a fixed point of one-site ALS/DMRG
($W\leftarrow P_{\mathcal V_v}T$ for every $v$).

*Proof.* At a full-rank parameterization, the differential with respect to core $v$ spans exactly
$\mathcal V_v$, because the other cores' subtree tensors span the frames. So
$T_W\mathcal M_r=\sum_v\mathcal V_v$ (as in Uschmajew & Vandereycken 2013), and $W$ is critical iff
$P_{\mathcal V_v}(T-W)=0$, i.e. $P_{\mathcal V_v}T=W$ for all $v$ (since $W\in\mathcal V_v$). For an edge
$e=(v,p)$, $W\in\mathcal V_v\cap\mathcal V_p$ gives $U_e\subseteq E_e^-$ and $V_e\subseteq E_e^+$. Hence
$P_{\mathcal V_v}T=T_eP_{V_e}$ and $P_{\mathcal V_p}T=P_{U_e}T_e$ (matricized at $e$). Every node is an
endpoint of some edge, so the node conditions are equivalent to the edge conditions, and the Lemma
finishes the proof. ∎

The same holds with "critical point of the parameterized loss $\theta\mapsto L(W(\theta))$" when every
bond is used at full rank. With larger bond dimensions, a critical point of a lower-rank problem,
zero-padded on both sides of each bond, is a critical point of the parameterized loss. This is how
the saddles visited by GF appear in a student with $\chi_e>r_e$.

For MPS3 with ranks $(r_1,r_2)$ (a Tucker problem with the middle mode free), Proposition 1 is the
higher-order orthogonal iteration (HOOI) fixed-point condition of De Lathauwer, De Moor & Vandewalle
(2000): $U_1$ = $r_1$ left singular vectors of $T\times_3P_{V_3}$, and $V_3$ = $r_2$ right singular
vectors of $(P_{U_1}\otimes I)T_{(12)}$. The tree version is new only in its bookkeeping.

### 2.2 When is the HSVD truncation critical?

**Proposition 2.** *proven.*

1. *Criterion.* $W_H$ (assumed to have exact ranks $r$) is critical iff its own frames satisfy
   Proposition 1. For MPS3 this reads: $U_1$ is invariant under $G_1^\perp=T_{(1)}(I\otimes
   P_{V_3}^\perp)T_{(1)}^\top$, the mode-1 Gram matrix of the part of $T$ discarded at the other edge,
   and symmetrically for $V_3$. This "leakage" commutator $[P_{U_1},G_1^\perp]$ is generically nonzero.
   Truncated HOSVD is not a HOOI fixed point.
2. *Sufficient: consistent truncations.* Let $\tau_e(T)$ be the best rank-$r_e$ approximation across
   $e$. If $\tau_e(T)=W$ for every edge, then $W_H=W$, it is critical, and it is the **global
   minimizer** on $\mathcal M_{\le r}$, because $\|T-X\|\ge\|T-\tau_e(T)\|$ for every $X$ with
   $\mathrm{rank}_eX\le r_e$. An equivalent description: $T=W+T_\perp$ with $W$ and $T_\perp$ having
   mutually orthogonal frames on both sides of every edge, and
   $\sigma_{\min}(W_{(e)})>\sigma_{\max}(T_{\perp,(e)})$. Examples:
   - an odeco teacher (or any CP teacher whose factors are orthogonal across components at every
     site) truncated to the same component set on every edge;
   - any tree whose rank-carrying edges all induce the same bipartition, e.g. `binary4`, where
     both edges induce $\{01\}|\{23\}$, so the problem is a matrix problem.
3. *Inconsistent subsets.* An odeco teacher with different component sets $S_e$ per edge gives
   $W_H=\sum_{r\in\cap_eS_e}s_r\otimes_iu_r^{(i)}$, whose ranks are below $r$. In the one case
   checked (MPS4, ranks $(2,1,2)$), ALS on $\mathcal M_{\le r}$ returned this same tensor.
   *Conjecture:* for odeco teachers the minimizer on $\mathcal M_{\le r}$ keeps the top
   $\min_er_e$ components.

**Proposition 3 (near-consistent teachers).** *derived (first-order perturbation), [checked §8.1].*
Let $T=W+\varepsilon E$ with $W$ consistent as in Proposition 2.2 and $E$ generic. The HSVD frames and
the frames of the nearby critical point $W_c$ agree to first order in $\varepsilon$: both are
first-order rotations of $W$'s frames driven by $W_{(e)}E_{(e)}^\top+E_{(e)}W_{(e)}^\top$, and the
projection by the environment does not change these terms because $W$'s rows already lie in it.
Hence
$$\|W_H-W_c\|=O(\varepsilon^2),\qquad L(W_H)-L(W_c)=O(\varepsilon^4).$$
Measured: $\|W_H-W_c\|=0.16\,\varepsilon^2$ and $L_H-L_c=0.013\,\varepsilon^4$ for a rank-2 odeco $W$ plus
a Gaussian $E$, on both `mps4` and `binary8`.

### 2.3 Which critical points does GF visit?

*derived (small-init asymptotics), [checked §8.2–8.4].* Near the origin GF is the gradient flow of
$g(\theta)=\langle W(\theta),T\rangle$, which is homogeneous of degree $K$. Its direction converges to a
local maximum of $g$ on the sphere before the norm grows (H1; Kumar & Haupt 2024/25).

- The **global** maximum of $g$ on the sphere has **bond rank one on every edge** (*proven*). Write
  the bond sums out and apply Hölder edge by edge, as in
  $\sum_{ab}x_ay_{ab}z_b\le\|x\|\|Y\|_F\|z\|$. This gives
  $g\le\|T\|_{\rm blk}\prod_v\|G_v\|_F$, where $\|T\|_{\rm blk}$ is the maximum of $\langle T,\otimes_bM_b\rangle$
  over unit node blocks $M_b$; equality holds at bond rank one.
- That *local* maxima also have bond rank one is a *conjecture*, true for all first plateaus in §8.

Bond rank one is a different object for different trees:

- If every node carries at most one site (MPS, HT), bond rank one means $W=\otimes_iw_i$. The first
  plateau is a **local best rank-one approximation** of $T$ (a critical point of
  $\max\langle T,\otimes_iw_i\rangle$ over unit $w_i$, i.e. a tensor singular vector tuple; Lim 2005).
  Its loss is $L_1=\tfrac12(\|T\|^2-\lambda^2)$ with $\lambda=\langle T,\otimes_iw_i\rangle$.
- For `binary` trees (two sites per bottom node), bond rank one means $W=\otimes_bM_b$ with
  arbitrary $d\times d$ blocks $M_b$. For `binary4` this is the top singular pair of $T_{(01|23)}$:
  a unique maximum, so no H1-type randomness.

Later plateaus are critical points of mixed-rank problems (Proposition 1 with the current ranks). For
generic teachers these are **not** HSVD truncations (§8.1), and their loss is lower than the HSVD
loss: by 6.6% of $L_0$ for MPS4 seed 1 at ranks (1,1,1), and by 6–10% of the plateau value for the
two-component teachers of §8.2. Which critical point is visited is set by the escape routes
(Section 3), not by HSVD.

## 3. Escaping a saddle: routes and the generalized locality rule

### 3.1 Routes

Take a saddle $W\in\mathcal M_r$ with balanced cores $\theta^*$. At small init the conservation laws
$Q_e\approx0$ make the cores balanced (`notes/model.md` §5). Let the student have spare bond dimension
($\chi_e>r_e$), and let $R=T-W$ be the residual.

A **route** is a connected set $S$ of $k\ge2$ cores (a subtree) together with a new bond index on
every edge internal to $S$. The new index is orthogonal to the $r_e$ learned ones. Its variables are
the blocks $Z_u$, $u\in S$: the entries of $G_u$ with new indices on its $S$-internal legs and learned
indices on all other legs. If only these blocks move, then exactly
$$W(\theta^*+Z)=W+W_S(Z),\qquad L=L(W)-\langle R,W_S(Z)\rangle+\tfrac12\|W_S(Z)\|^2,$$
where $W_S$ is $k$-linear in the blocks. *derived:* with balanced blocks of common amplitude $x$, which
the conservation laws enforce, the leading dynamics is that of a depth-$k$ DLN mode,
$$\dot x=\mu_S\,x^{k-1},\qquad \mu_S=\max_{\|Z_u\|=1}\langle R,W_S(Z)\rangle .$$

- $k=2$: the Hessian restricted to the route is $\begin{pmatrix}0&-B\\-B^\top&0\end{pmatrix}$
  with $B=E_v^*\,(T_e-W_{(e)})\,E_p$. Here $E_v$ maps $Z_v$ to $\mathrm{below}(e)$ through $v$'s learned
  child subtrees, and $E_p$ maps $Z_p$ to $\mathrm{above}(e)$. The saddle is **strict**, with eigenvalue
  $-\gamma_e$, $\gamma_e=\sigma_{\max}(B)\le\|E_v\|\|E_p\|\,\sigma_{r_e+1}(T_e)$. The bound holds when
  the kept pairs are the top $r_e$ of $T_e$. The escape is exponential: $x\propto e^{\gamma_et}$.
- $k\ge3$: no negative curvature along the route (a **higher-order saddle**). Starting from $x_0$, the
  blow-up time is $t^*=1/((k-2)\mu_Sx_0^{k-2})$.

**Admissibility ("room").** A route can raise a rank only if each leaf $u$ of $S$ (as a subtree)
offers directions outside the learned frame of its edge into $S$. That requires
$\dim E^-_{(u)}>r_{(u)}$: the leg's environment, made of physical legs and outside bonds, must have
dimension larger than the current rank. A site-less node whose outside legs all have rank 1 has no
room. Example: the internal nodes of `binary` and `ht` trees at the rank-one saddle. A route ending
there only reshuffles the learned component, and its coefficient vanishes at a critical point
(its first-order part is a gradient direction).

**Generalized locality rule.** *derived at leading order; that GF takes this route as $\sigma\to0$ is
a conjecture supported by §8.* At a saddle, the next escape uses the admissible route with the
smallest $k$ and $\mu_S\neq0$; among equal $k$ it uses the largest $\mu_S$. Exponential escape
($k=2$) beats any polynomial one, and $\sigma^{-(k-2)}$ beats $\sigma^{-(k'-2)}$ for $k<k'$.

- **Odeco/pattern teachers** (component 2 equals component 1 except on sites $D$): $\mu_S=0$ unless
  $\mathrm{sites}(S)\supseteq D$, because $\langle u_i,v_i\rangle=0$ on any uncovered site of $D$ kills
  the overlap. So $k=|\mathrm{Steiner}(D)|$, which is H2.
- **Generic teachers**: $\mu_S\neq0$ for every admissible route, so $k_{\min}=2$ whenever some edge has
  room on both sides. In terms of the task's phrasing ("$k$ = cores incident to edges whose rank must
  increase"), the edges that matter are those whose rank must increase *in the next escape*. For
  generic teachers one edge suffices, while for odeco teachers the whole Steiner tree must grow at
  once.

### 3.2 Rate at the rank-one saddle of an MPS student

*derived, [checked §8.2–8.4].* Take the MPS student (core $k$ carries site $k$; $K=N$) at a rank-one
critical point $W_1=\lambda\otimes_iw_i$, a local max of the rank-one problem. Balanced cores have
amplitude $q=\lambda^{1/N}$. The 2-core route on edge $e=(k,k+1)$ has blocks $X\in\mathbb R^d$ (core
$k$, new right index) and $Y\in\mathbb R^d$ (core $k+1$, new left index). Then
$$\langle R,W_S\rangle=q^{N-2}X^\top\big(M_e-\lambda w_kw_{k+1}^\top\big)Y,\qquad
M_e:=T\times_{i\notin\{k,k+1\}}w_i\ \ (d\times d).$$
At a local max, $(w_k,w_{k+1})$ is the top singular pair of $M_e$ with value $\lambda$. Therefore
$$\boxed{\gamma_e=\lambda^{(N-2)/N}\,\sigma_2(M_e)},$$
and each edge contributes one Hessian eigenvalue $-\gamma_e$. Routes on different edges decouple at
second order.

**Link to the teacher's bond spectra** (*proven*): $M_e=P^\top T_{(e)}Q$ with $P=(\otimes_{i<k}w_i)\otimes
I_d$ and $Q=I_d\otimes(\otimes_{i>k+1}w_i)$, both with orthonormal columns. By interlacing for
compressions,
$$\lambda=\sigma_1(M_e)\le\sigma_1^e(T),\qquad \sigma_2(M_e)\le\sigma_2^e(T),\qquad
\gamma_e\le\lambda^{(N-2)/N}\sigma_2^e(T).$$
The teacher's Schmidt values bound the rates. Equality holds when the teacher's second Schmidt
vectors are products with $w$ on the far sites. When the second Schmidt vectors are entangled far
from the edge, $\sigma_2(M_e)\ll\sigma_2^e(T)$ (example in §4.3).

For a general tree at the rank-one saddle, the same holds with $M_e$ = $T$ contracted with the
learned rank-one structure everywhere except cores $v$ and $p$, a $d_v\times d_p$ matrix, and
amplitude factor $\lambda^{(K-2)/K}$. The route is admissible iff both $v$ and $p$ carry sites.

### 3.3 Escape time including growth during the first escape

*derived (assumptions: $\sigma\to0$; balanced cores; the direction converges before the norm grows,
which is asymptotically true since alignment takes $O(1)$ rescaled time while growth takes
$\log(1/\sigma)$).* During the first escape the main amplitude obeys $\dot q=q^{N-1}(\lambda-q^N)$, and
the route blocks obey $\dot x=q^{N-2}\sigma_2(M_e)x$. So
$\int q^{N-2}dt=\lambda^{-1}\log(q/q_0)+O(1)$, and the route grows by a factor
$(1/\sigma)^{\kappa_e}$ with $\kappa_e=\sigma_2(M_e)/\lambda$ before the plateau starts. With
$x(0)\propto q_0\propto\sigma$:
$$\boxed{\Delta t_2=A_{e^*}\log(1/\sigma)+O(1),\qquad A_e=\frac{1-\sigma_2(M_e)/\lambda}{\lambda^{(N-2)/N}\sigma_2(M_e)},\qquad e^*=\arg\max_e\sigma_2(M_e).}$$
Also $s_2^e(t_1)\propto\sigma^{2(1-\kappa_e)}$ at the end of the first escape (on the plateau $s_2^e\propto
q^{N-2}xy$), and $\tfrac{d}{dt}\log s_2^e=2\gamma_e$ on the plateau.

**Measurement caveat.** Time the second escape with a fixed *small* threshold, e.g. the first time
$\max_es_2^e=10^{-2}$, which stays in the linear regime. The `diagnostics.md` event "half of the final
value" can fire only after later escapes, because intermediate saddles have $s_2^e$ well below the
final value (§8.3: 0.148 vs 0.56).

### 3.4 Race model for the whole sequence

*heuristic, [checked §8.3 on one teacher].* Give each pending route $\rho$ a log-amplitude
$\ell_\rho=\log x_\rho/\log(1/\sigma)\in[-1,0]$, starting at $-1+\kappa_\rho$ after the first escape. In
rescaled time $\tau=t/\log(1/\sigma)$:

- while GF sits at saddle $S_n$, $\dot\ell_\rho=\gamma_\rho(S_n)$, the route's rate at that saddle;
- route $\rho$ escapes when $\ell_\rho=0$, and the flow jumps in $O(1)$ time to the next saddle $S_{n+1}$,
  a critical point of the problem with that bond's rank raised by one;
- the other $\ell$'s carry over, because projecting onto the new unstable directions costs only
  $O(1)$ factors.

This is a Pesme–Flammarion-type jump process. It differs from greedy low-rank learning (GLRL; Li, Luo
& Lyu 2021): a route that grew at an earlier saddle keeps its head start. Degree-$k\ge3$ routes enter
with amplitude $\sim\sigma$ and polynomial times, so as $\sigma\to0$ they lose to any $k=2$ route.

## 4. (b) Two non-orthogonal rank-one components

$T=s_1\otimes_iu_i+s_2\otimes_iv_i$, unit factors, $s_1\ge s_2>0$, overlaps
$c_i=\langle u_i,v_i\rangle\in[0,1)$, $\phi_i=\arccos c_i$, $C=\prod_ic_i$, $\rho=s_2/s_1$, and
$\|T\|^2=s_1^2+s_2^2+2s_1s_2C$.

### 4.1 Rank-one critical points

*proven.* Stationarity gives $w_i\propto s_1a_{-i}u_i+s_2b_{-i}v_i$ with
$a_{-i}=\prod_{j\ne i}\langle w_j,u_j\rangle$ and $b_{-i}=\prod_{j\ne i}\langle w_j,v_j\rangle$. So
$w_i\in\mathrm{span}(u_i,v_i)$. Writing $\theta_i$ for the angle of $w_i$ from $u_i$ in that plane,
$$f(\theta)=s_1\prod_i\cos\theta_i+s_2\prod_i\cos(\phi_i-\theta_i),\qquad \lambda=\max_{\rm loc}f .$$

**Number of local maxima.**

- $s_1=s_2$, $c_i=c$ (*proven* for symmetric critical points; *numerically exhaustive* otherwise, §8.2).
  At the merged point $\theta_i=\phi/2$, the Hessian of $f$ has eigenvalue
  $-2\cos^{N-2}\tfrac\phi2[\cos^2\tfrac\phi2-(N-1)\sin^2\tfrac\phi2]$ along $(1,\dots,1)$ and
  $-2\cos^{N-2}\tfrac\phi2<0$ on its complement. Hence
  $$\text{two local maxima (one near each component) iff } c<c^*=\frac{N-2}{N},$$
  and otherwise a single merged maximum at the bisector. For $N=2$ (a matrix) there is always one.
- $s_2<s_1$: the minor maximum (near $v$) disappears at a fold. *derived* for small $\rho$ (expanding
  near $v$ with $\cos(\phi-\psi)\approx c+\psi$):
  $$c_{\rm fold}(N,\rho)\approx\frac{N-2}{N-1}\Big(\frac{\rho}{N-1}\Big)^{1/(N-2)} .$$
  Exact values are in §8.2. Example: $N=4$, $\rho=0.6$ gives $c_{\rm fold}=0.326$. Long tensors keep
  both basins up to larger overlaps ($c_{\rm fold}\to1$ as $N\to\infty$).

### 4.2 First plateau

*derived, [checked §8.2].* The first plateau is the local maximum whose basin contains the init (H1);
for $c>c_{\rm fold}$ it is unique. It is **the best rank-one approximation only when unique, and
never one of the components**. Its loss is $L_1=\tfrac12(\|T\|^2-\lambda^2)$. For small overlaps, near
$u$:
$$\theta_i=\frac{s_2}{s_1}C\tan\phi_i+\dots,\qquad
\lambda=s_1+s_2C+\frac{s_2^2C^2}{2s_1}\sum_i\tan^2\phi_i+\dots$$
The expansion parameter is $s_2C\tan^2\phi_i/s_1$. The plateau is below the "component learned" level
$\tfrac12s_2^2(1-C^2)$ by $\tfrac12(\lambda^2-(s_1+s_2C)^2)$.

### 4.3 Second phase: bond-by-bond strict saddles, not deflation

*derived, [checked §8.3].* For an MPS student the residual after the rank-one fit has a nonzero
transverse part at every edge. By R5, every edge $e=(k,k+1)$ has a strict-saddle route with
$$\sigma_2(M_e)=\frac{s_1s_2\,a_e\,b_e\,\sin\phi_k\sin\phi_{k+1}}{\lambda},\qquad
a_e=\prod_{i\ne k,k+1}\cos\theta_i,\ \ b_e=\prod_{i\ne k,k+1}\cos(\phi_i-\theta_i),$$
which is exact: $\det M_e=s_1s_2a_eb_e\sin\phi_k\sin\phi_{k+1}$ and $\sigma_1(M_e)=\lambda$. For small $C$
it is $\approx s_2\,\big(\prod_{i\ne k,k+1}c_i\big)\sin\phi_k\sin\phi_{k+1}$. **The bond that grows first is
the one whose two sites differ most while the other sites agree most.** This differs from "largest
teacher Schmidt value". For $c=(0.2,0.5,0.7,0.3)$ the teacher's $\sigma_2^e(T)=(0.567,0.566,0.548)$
are nearly tied, so they carry no information. The theory gives $\sigma_2(M_e)=(0.127,0.036,0.059)$,
and edge 0 grew first in 9/9 runs, followed by edge 2, then edge 1. For equal overlaps the teacher's
middle Schmidt value is the largest (0.588 vs 0.555), yet the theory gives equal rates, and the edge
that grows first varies with the seed.

How the second component is learned:

- *Equal overlaps*: all $\gamma_e$ are equal, and all bonds go from rank 1 to 2 in one transition.
- *Unequal overlaps*: rank is added bond by bond. Each intermediate plateau is a critical point of
  a mixed-rank problem, e.g. ranks (2,1,1), then (2,1,2), then $T$; the order and timing follow the
  race model.
- *Not deflation.* The learned rank-one direction $w$ is not a component. It keeps rotating, by
  $\theta_i\to0$, toward $u$ while the rest is learned, and GF converges to $T$ exactly. Greedy
  deflation (fit the best rank-one of $T$, then of the residual) does not recover a rank-2 tensor in
  two steps. The residual can even have higher rank (Stegeman & Comon 2010; Kolda 2001). GF recovers
  $T$ because all cores keep moving. Ge, Ren, Wang & Zhou (2021) prove deflation-like dynamics only
  for orthogonal components.
- *Contrast, `binary4` student*: a depth-3 DLN on $T_{(01|23)}$ (§6.2). Its saddles are the nested SVD
  truncations of that matrix, so it is deflation-like on the matricization. Its first plateau is
  $\tfrac12\sigma_2^2(T_{(01|23)})$, lower than the MPS rank-one plateau since
  $\lambda\le\sigma_1(T_{(01|23)})$, and all its escapes are degree 3 ($\propto\sigma^{-1}$).

### 4.4 The odeco limit $c\to0$

*heuristic, [checked §8.3].* $\sigma_2(M_e)\propto\prod_{i\ne k,k+1}c_i\to0$, so the 2-core rate vanishes
and the $K$-core route (odeco mechanism) takes over. Its time is
$T_{\rm odeco}(\sigma)=a_N\sigma^{-(N-2)}$, where $a_N$ is the odeco second-escape prefactor (it
depends on the init projections). The two routes act in parallel, and the escape follows whichever
is faster. For symmetric small overlaps on an MPS, $A(c)\approx1/(s_2c^{N-2})$, and
$A(c_\times)\log(1/\sigma)=T_{\rm odeco}(\sigma)$ gives
$$c_\times\approx\sigma\Big[\frac{\log(1/\sigma)}{s_2\,a_N}\Big]^{1/(N-2)} .$$
For $c\gg c_\times$ the second plateau is logarithmic with slope $A(c)$; for $c\ll c_\times$ it is
odeco-like, $\propto\sigma^{-(N-2)}$. For MPS3 with $s_2=0.6$ the measured $a_3=18.5$, so
$c_\times\approx\sigma\log(1/\sigma)/11$: 0.014 at $\sigma=0.05$ and 0.0016 at $\sigma=0.003$.

## 5. (c) Random TTN teachers on the student's tree

### 5.1 MPS students

*derived; [checked §8.4] on random MPS4 teachers with $d=2$, $\chi=2$.* Every core carries a site, so
every edge with $r_e<d\min(r_{e-1},r_{e+1})$ has an admissible 2-core route. For generic teachers:

1. **First plateau**: polynomial, duration $\propto\sigma^{-(N-2)}$. Its level is the local best
   rank-one approximation, with $\lambda\le\min_e\sigma_1^e(T)$.
2. **All later plateaus**: strict saddles with duration $\propto\log(1/\sigma)$. The first rank increase
   happens on $e^*=\arg\max_e\sigma_2(M_e)$, with $\Delta t_2$ from §3.3.
3. **The full order** follows the race model. The first-step ranking (sorting edges by
   $\sigma_2(M_e)$) gets the complete order right only for well-separated rates (5/8 in §8.4), because
   rates change at each intermediate saddle.

**What the teacher's bond spectra predict.** The naive rule "next rank increase on the edge with the
largest $\sigma_{r_e+1}^e(T)$" is an upper-bound proxy (§3.2). It agrees with the theory on random
Gaussian MPS teachers, both getting the first edge right in 8/8 cases ($\kappa=0.4$) and 13–14/16
($\kappa=1$). It fails when the second Schmidt vectors are entangled far from the edge, as in the
CP example of §4.3. The quantitative content (rates, slopes, Hessian eigenvalues, intermediate
plateau levels) needs $M_e$ or $T_e$, not the teacher's spectra.

**Odeco/near-odeco teachers** are the exception. Exact orthogonality makes the coefficients of all
routes smaller than the Steiner tree vanish, so later plateaus are polynomial again (H2).

### 5.2 Binary and HT students

*derived; binary4 [checked §8.5].*

- At the rank-one saddle, internal nodes have no sites and rank-one outside legs, so they have no
  room. The minimal admissible routes are sibling bottom nodes through their parent,
  $b_0$–$m_0$–$b_1$ ($k=3$), which raise two bonds at once. For HT the analogue is two sibling leaves
  plus their parent ($k=3$). **Later escapes at the rank-one saddle are polynomial**
  ($\propto\sigma^{-1}$) even for generic teachers.
- The two root edges induce the same bipartition and always grow together, through
  $m_0$–root–$m_1$ ($k=3$). That route is admissible only after both halves have rank $>1$ below
  them. Otherwise it is a $k=5$ route $b$–$m_0$–root–$m_1$–$b'$ ($\propto\sigma^{-3}$).
  **Prediction: bottom-up order.** The root bipartition's rank grows only after bonds on both sides
  have grown, as $\sigma\to0$.
- A 2-core route opens on a bottom edge $(b_0,m_0)$ once $r_{b_1}r_{m_0}>r_{b_0}$. Logarithmic
  plateaus therefore can appear later in binary trees, but never for the root bipartition.

## 6. (d) Teacher on a different tree

### 6.1 Which student bonds must grow

*proven (standard).* Student edge $e$ must reach $\mathrm{rank}\,T_{(e)}$. For a TTN teacher this is at
most $\min$ over cuts $\mathcal C$ of the teacher tree separating $\mathrm{below}(e)$ from
$\mathrm{above}(e)$ of $\prod_{c\in\mathcal C}\dim c$, where cut bonds count their bond dimension and cut
physical legs count $d$. Example: MPS8 teacher with ranks $r$, `binary8` student:

| student bond | bipartition | required rank |
|---|---|---|
| $b_0$–$m_0$ | $\{01\}$ | $r$ |
| $b_1$–$m_0$ | $\{23\}$ | $\min(d^2,r^2)$ |
| $b_2$–$m_1$ | $\{45\}$ | $\min(d^2,r^2)$ |
| $b_3$–$m_1$ | $\{67\}$ | $r$ |
| root bonds | $\{0123\}\mid\{4567\}$ | $r$ |

A blocked student bond in the middle of the chain must carry the *product* of the two teacher ranks
that bound its block, so it needs about $r^2$ rank increments, each a separate saddle.

### 6.2 `binary4` student = depth-3 DLN

*proven.* With root $R$ and bottom cores $A$ (sites 0,1) and $B$ (sites 2,3),
$W_{(01|23)}=ARB^\top$: a 3-layer linear network with whitened input on the matrix $T_{(01|23)}$,
for any teacher tree. Hence (Saxe et al. 2014; Jacot et al. 2021):

- the saddles are nested SVD truncations of $T_{(01|23)}$ (HSVD is exact here, §2.2);
- plateau levels are $L_j=\tfrac12\sum_{i>j}\sigma_i^2(T_{(01|23)})$;
- every escape has degree 3: $\Delta t_j\propto\sigma^{-1}$, and in the decoupled approximation
  $t_j\approx1/(\sigma_ia_i(0))$;
- the number of plateaus equals $\mathrm{rank}\,T_{(01|23)}$. For an MPS teacher this is its middle
  bond rank; the teacher's other bonds are invisible.

### 6.3 The same teacher on MPS4 vs binary4

*derived, [checked §8.5].* An MPS4 student on any generic teacher has a $\sigma^{-2}$ first plateau at
the rank-one level $\tfrac12(\|T\|^2-\lambda^2)$, then $\log(1/\sigma)$ plateaus. A binary4 student has
$\sigma^{-1}$ plateaus at the SVD levels of $T_{(01|23)}$. **The same teacher therefore has a different
plateau structure, different plateau levels and different escape exponents on the two trees.** For
an MPS teacher on a `binary8` student, the correlations crossing the halves (the teacher's middle
bond) are learned last, through a $k\ge3$ route; an MPS student learns them through a 2-core route on
its middle edge.

When the teacher's second singular values are comparable to the first (large $\kappa_e$), the
rank-one plateau of the MPS student is very short at accessible $\sigma$, since it lasts
$\propto(1-\kappa_e)\log(1/\sigma)/\gamma_e$. The first visible plateau is then an intermediate saddle
(§8.5).

## 7. Testable predictions

Symbols: $\lambda$ = rank-one value reached, $w_i$ its site vectors, $M_e=T\times_{i\notin e}w_i$,
$\kappa_e=\sigma_2(M_e)/\lambda$, $\gamma_e=\lambda^{(K-2)/K}\sigma_2(M_e)$, $A_e=(1-\kappa_e)/\gamma_e$.
"GF" = gradient flow from fan-in init with scale $\sigma$.

- **P1 (criticality).** Every GF plateau $W_{\rm pl}$ satisfies Proposition 1 at its current ranks:
  one-site ALS started from it moves it by less than the residual escape amplitude, and the
  criticality defect is below $10^{-3}$ at plateau midpoints. HSVD truncations of generic teachers
  have defect $10^{-2}$–$10^{-1}$, and plateau losses differ from HSVD losses.
- **P2 (near-consistent teachers).** $L(W_H)-L(W_c)\propto\varepsilon^4$ and
  $\|W_H-W_c\|\propto\varepsilon^2$ (prefactors 0.013 and 0.16 for the §8.1 construction).
- **P3 (first plateau, MPS/HT students).** Plateau loss $\to\tfrac12(\|T\|^2-\lambda^2)$ with $\lambda$
  a local max of the rank-one problem, for any teacher. For two components with equal weights and
  overlaps: two possible plateaus iff $c<(N-2)/N$. For $s_2/s_1=\rho$: two iff $c<c_{\rm fold}(N,\rho)$
  (table §8.2).
- **P4 (Hessian at the rank-one plateau; strict vs higher-order, H3).** On an MPS student there is
  one $\sigma$-independent eigenvalue $-\gamma_e$ per bond with an admissible 2-core route. On a
  `binary`/`ht` student, for the same generic teacher, $\lambda_{\min}(H)\to0$ like $\sigma^{\approx1}$
  (a higher-order saddle).
- **P5 (escape rate).** On the rank-one plateau, $\tfrac{d}{dt}\log s_2^{e^*}=2\gamma_{e^*}$.
- **P6 (second escape time, MPS student, generic teacher).** With $t_2$ = first time
  $\max_es_2^e=10^{-2}$: $d(t_2-t_1)/d\log(1/\sigma)\to A_{e^*}$ and $e^*=\arg\max\sigma_2(M_e)$. Also
  $s_2^{e}(t_1)\propto\sigma^{2(1-\kappa_e)}$.
- **P7 (intermediate saddles and order).** With unequal $\gamma_e$, the second component is learned
  bond by bond. Plateaus are critical points of mixed-rank problems, and the order and slopes
  $dt_e/d\log(1/\sigma)$ follow the race model.
- **P8 (odeco crossover).** For two components with symmetric overlap $c$, the second plateau is
  logarithmic with slope $A(c)$ for $c\gg c_\times=\sigma[\log(1/\sigma)/(s_2a_N)]^{1/(N-2)}$, where
  $a_N=\sigma^{N-2}\Delta t_2|_{c=0}$, and $\propto\sigma^{-(N-2)}$ for $c\ll c_\times$.
- **P9 (tree dependence).** For generic teachers, later plateaus are $\propto\log(1/\sigma)$ on an MPS
  student and $\propto\sigma^{-1}$ (or slower) on binary/HT students at the rank-one saddle. The root
  bipartition of a binary/HT student always escapes polynomially.
- **P10 (`binary4`).** For any teacher, plateaus sit at $\tfrac12\sum_{i>j}\sigma_i^2(T_{(01|23)})$
  exactly, with $\Delta t\propto\sigma^{-1}$ and $\mathrm{rank}\,T_{(01|23)}$ events.
- **P11 (bottom-up order, `binary8`).** The root bond's rank rises only after bottom bonds on both
  halves have risen. For an MPS8 teacher, the student bonds $\{23\}$ and $\{45\}$ need
  $\min(d^2,r^2)$ rank increments.

## 8. Numerical checks

All checks use only `ttn.tree`, `ttn.network`, `ttn.teachers` and `ttn.losses`, in float64. GF is
scipy DOP853 with `rtol=1e-10`, `atol=1e-13` and torch autograd of `population_loss`; dense ALS and
the HSVD are written in numpy. Scripts are in
`C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/theory2/teachers/`
(`check_a.py`, `check_a_pert.py`, `check_b_landscape.py`, `mps_saddle.py`, `slope_test.py`,
`check_intermediate.py`, `race_check.py`, `check_b_cross3.py`, `check_c2.py`, `check_d.py`,
`check_d2.py`). Every run took under 5 min of CPU.

### 8.1 HSVD vs critical points (part a)

Criticality defect $\delta=\max_v\|P_{\mathcal V_v}(T-W)\|/\|T-W\|$. $\delta=0$ iff critical.

| teacher, tree, ranks | $\delta$(HSVD) | $L/L_0$ HSVD → ALS fixed point | $\|W_H-W_c\|/\|W_c\|$ |
|---|---|---|---|
| random MPS4 ($d=3$, TT ranks 3) seed 0, $r=(1,1,1)$ | 0.091 | 0.3510 → 0.3432 | 0.15 |
| same, $r=(2,2,2)$ | 0.031 | 0.02728 → 0.02725 | 0.006 |
| seed 1, $r=(1,1,1)$ | 0.180 | 0.6303 → 0.5646 | 0.58 |
| seed 1, $r=(2,2,2)$ | 0.162 | 0.2636 → 0.2371 | 0.41 |
| random MPS3 ($d=3$), $r=(1,1)$, seeds 0/1/2 | 0.098 / 0.057 / 0.056 | gaps 1.4% / 0.3% / 0.6% | 0.05 / 0.02 / 0.04 |
| random binary8 ($d=2$, $\chi=2$), $r=1$, seeds 0/1 | 0.097 / 0.004 | 0.1096 → 0.1085; 0.19112 → 0.19112 | 0.035 / 0.002 |
| odeco MPS4, same set on all edges, $r=(1,1,1)$ / $(2,2,2)$ | $8\times10^{-16}$ / $4\times10^{-15}$ | equal | $10^{-15}$ |
| random binary4 teacher, binary4 tree, $r=1,2$ | $\le5\times10^{-14}$ | equal | $10^{-15}$ |

Proposition 1 was checked at the ALS fixed point (MPS4 seed 0, $r=(2,2,2)$). On every edge,
$\|P_UT_e-W_e\|/\|W_e\|\le2\times10^{-9}$, $\mathrm{span}\,U$ is invariant under $T_eT_e^\top$ to
$10^{-9}$, and the kept values are the top two of $T_e$.

Perturbation (Proposition 3), $W$ = odeco rank 2 (weights 1, 0.7), $E$ Gaussian with unit norm,
ranks 2:

| $\varepsilon$ | 0.2 | 0.1 | 0.05 | 0.025 |
|---|---|---|---|---|
| mps4: $\|W_H-W_c\|/\varepsilon^2$ | 0.169 | 0.163 | 0.161 | 0.160 |
| mps4: $(L_H-L_c)/\varepsilon^4$ | 0.014 | 0.013 | 0.013 | 0.013 |
| binary8: same | 0.161, 0.013 | 0.159, 0.013 | 0.159, 0.013 | 0.158, 0.013 |

### 8.2 Rank-one landscape and first plateau (part b)

$d=2$, multistart ALS (60 starts).

- $s_1=s_2$: number of maxima at $c=c^*\mp0.03$ is 2 → 1 for $N=3$ ($c^*=1/3$), $N=4$ ($1/2$) and
  $N=6$ ($2/3$). No asymmetric maxima were found.
- $c_{\rm fold}(N,\rho)$ from the symmetric reduction. At $\rho=0.2$ the small-$\rho$ formula of §4.1
  gives 0.050, 0.172 and 0.358 for $N=3,4,6$.

| $N$ | $\rho=1$ | 0.8 | 0.6 | 0.4 | 0.2 |
|---|---|---|---|---|---|
| 3 | 0.3333 | 0.2228 | 0.1580 | 0.1021 | 0.0503 |
| 4 | 0.5000 | 0.3953 | 0.3257 | 0.2566 | 0.1763 |
| 6 | 0.6667 | 0.5838 | 0.5241 | 0.4598 | 0.3757 |
| 8 | 0.7500 | 0.6831 | 0.6334 | 0.5782 | 0.5025 |

- GF, MPS4 student ($d=2$, $\chi=2$), $s=(1,0.6)$, $c=0.5$, $\sigma=0.0125$, 3 seeds:
  - plateau loss 0.16630 / 0.16625 / 0.16629 vs predicted 0.16631;
  - $1-\cos$ between the plateau's site vectors and the ALS rank-one critical point:
    $10^{-6}$–$10^{-7}$;
  - Hessian minimum eigenvalue −0.1475 / −0.1473 / −0.1479 vs $-\gamma=-0.1474$.
- $c=0.3$: plateau 0.17942 (all seeds) vs 0.17942; Hessian −0.0549 / −0.0553 / −0.0555 vs −0.0548.
  The Hessian has three negative eigenvalues (−0.0553, −0.0547, −0.0537), each localized on the two
  cores of one bond.
- MPS3, $c=0.3$, $\sigma=0.00625$: plateau 0.17374 vs 0.17381; Hessian −0.1993 vs $-\gamma=-0.2004$.
- The HSVD rank-one truncation (= truncated HOSVD) is not the plateau. Its loss and defect are
  0.19822 (defect 0.20) for $c=0.3$, 0.18100 (0.15) for $c=0.5$, and 0.18646 (0.16) for
  $c=(0.2,0.5,0.7,0.3)$. The GF plateaus sit at the critical points: 0.17942, 0.16631 and 0.17577.

### 8.3 Second escape (parts b, c)

**Slope test** (linear-threshold event $\max_es_2^e=10^{-2}$; MPS4, $s=(1,0.6)$, symmetric $c$;
$\sigma\in\{0.05,0.025,0.0125,0.00625\}$; slope from the last pair):

| $c$ | predicted $A$ | measured, seeds 0/1/2 |
|---|---|---|
| 0.5 | 5.853 | 5.891 / 5.852 / 5.870 |
| 0.3 | 17.255 | 17.174 / 17.275 / 17.184 |

MPS3, $c=0.3$ ($A=4.018$), 50% event, seed 0: 4.00 / 4.04 / 4.04 per halving.

**Caveat on events.** With the `diagnostics.md` 50%-of-final event, two of three $c=0.3$ seeds gave
slopes of about 19.7. That event fired at a later joint escape, after the edge with the largest
initial unstable projection had already grown (its unstable projection was 11× larger than the
middle edge's).

**Phase-1 growth.** $d\log s_2^e(t_1+10)/d\log\sigma$ predicted $2(1-\kappa)$ = 1.891 ($c=0.3$) and
1.726 ($c=0.5$). Measured 1.81–1.85 and 1.61–1.78 on the leading edges. For one trace the growth
over 4.9 decades of $s_1$ was 2.49 and 2.70 decades on edges 0 and 2, vs 2.57 predicted; the middle
edge was off-asymptotic because of a small unstable projection.

**Mixed overlaps, race model.** $c=(0.2,0.5,0.7,0.3)$, rank-one rates
$\gamma=(0.1283,0.0360,0.0592)$, $A=(6.82,26.84,15.93)$.

- First edge: 9/9 runs (3 seeds × 3 $\sigma$) grow edge 0 first. Growth rate of $s_2^{(0)}$ on the
  plateau is 0.259 vs $2\gamma_0=0.2565$.
- Intermediate plateau ($\sigma=0.00625$): $L=0.16640$, $s_2^{(0)}=0.1431$. ALS at ranks (2,1,1)
  started from it gives $L=0.166436$, $s_2^{(0)}=0.1468$, defect $7\times10^{-9}$. The distance
  $\|W_{GF}-W_c\|/\|W_c\|$ is $4.4\times10^{-3}$, down from $7.6\times10^{-3}$ at $\sigma=0.025$.
- A third plateau at ranks (2,1,2) has $L=0.16010$.
- Race model with the Hessian rates measured at each plateau ($(2,1,1)$: edge 2 0.0987, edge 1 0.0760;
  $(2,1,2)$: edge 1 0.1119):

| | edge 0 | edge 2 | edge 1 |
|---|---|---|---|
| predicted order and slope $dt_e/d\log(1/\sigma)$ | 6.821 | 12.283 | 15.005 |
| measured (10% events, $\sigma=0.025\to0.00625$) | 6.81 | 12.28 | 14.63 |

**Odeco limit and crossover** (MPS3, $s=(1,0.6)$, symmetric $c$, linear-threshold event; slope =
$\Delta(\Delta t_2)/\Delta\log(1/\sigma)$ between successive $\sigma$):

| $c$ | predicted $A(c)$ | $\Delta t_2$ at $\sigma$ = 0.05 / 0.0125 / 0.003125 | slopes | regime |
|---|---|---|---|---|
| 0 | $\infty$ | 374.5 / 1483.2 / 5920.2 ($\sigma\Delta t_2$ = 18.72 / 18.54 / 18.50) | — | odeco, $\propto\sigma^{-1}$ |
| 0.01 | 164.7 | 157.7 / 328.8 / 548.4 | 123.5, 158.4 | crossing over ($c_\times$ = 0.014 → 0.0016) |
| 0.03 | 53.6 | 83.1 / 153.0 / 227.9 | 50.4, 54.0 | logarithmic |
| 0.1 | 14.78 | 24.95 / 45.51 / 66.03 | 14.83, 14.81 | logarithmic |
| 0.3 | 4.02 | 5.93 / 11.52 / 17.06 | 4.03, 4.00 | logarithmic |

### 8.4 Random MPS teachers on an MPS4 student (part c)

$d=2$, $\chi=2$, `random_ttn_teacher` (Gaussian cores).

- **Batch A** ($\sigma=0.05$, 16 teachers with a plateau; one more had none, and three runs hit
  the time limit). First edge: theory 13/16, naive (largest teacher $\sigma_2^e$) 14/16. The three
  theory misses:
  - one near-tie: $\sigma_2(M_e)=0.210$ vs 0.205;
  - two teachers with $\kappa_{e^*}\approx0.55$–$0.7$, whose rank-one plateau is too short to resolve
    at $\sigma=0.05$ (plateau/prediction 1.01 and 0.76).
- **Batch B** (second bond slices scaled by $\kappa=0.4$, $\sigma=0.02$, 8 teachers with a plateau).
  - First edge: theory 8/8, naive 8/8. Full order matches the first-step ranking 5/8.
  - Where the plateau is formed (plateau/prediction > 0.98): Hessian min $=-\gamma_{\max}(1\pm0.015)$
    and the growth rate of $s_2^{e^*}$ is $2\gamma_{e^*}(1.00,1.03,1.135)$.
  - Where it is not formed (plateau/prediction 0.77–0.83): deviations of 20–26%.

### 8.5 Same teacher, two student trees (part d)

Teacher tseed 7, $d=2$, $\chi=2$. Students `binary4` and `mps4` ($\chi=2$).

- **MPS4 teacher → binary4.** Plateau 0.13109–0.13112 vs predicted $\tfrac12\sigma_2^2(T_{(01|23)})=0.13112$.
  $\Delta t_2$ = 68.0 / 56.3 at $\sigma=0.1$, 144.8 / 113.9 at 0.05, 298.3 / 229.0 at 0.025 (seeds 0/1):
  ratio per halving 2.01–2.13, i.e. $\propto\sigma^{-1}$.
- **binary4 teacher → binary4.** Plateau 0.03835–0.03836 vs 0.03836. $\Delta t_2$ = 183.8 / 153.2 at
  $\sigma=0.1$, 370.9 / 309.2 at 0.05, 745.3 / 621.7 at 0.025: ratio 2.02 per halving, $\propto\sigma^{-1}$.
- **MPS4 teacher → MPS4.** Theory: $\lambda=0.8416$, $L_1=0.14582$,
  $\sigma_2(M_e)=(0.1665,0.2777,0.045)$ (teacher $\sigma_2^e=(0.185,0.512,0.415)$), $A_e=(5.25,2.63,22.9)$.
  - Seed 0: first edge 1 at all four $\sigma$; plateau 0.14560–0.14572; slopes
    2.71, 2.64, 2.65 vs $A_1=2.63$.
  - Seed 1: slopes 0.89 → 1.21 → 1.70, still converging. Its rank-one plateau is short
    ($\kappa_1=0.33$).
- **binary4 teacher → MPS4.** Theory: $\lambda=0.6993$, $L_1=0.2555$, $\kappa_e=(0.55,0.06,0.81)$,
  $A_e=(1.40,28.1,0.41)$.
  - The rank-one plateau merges with the second escape: the $s_2$ threshold is crossed 2.6–3.3 time
    units before $s_1$ reaches $\lambda/2$. Last-pair slopes are 0.40 / 0.37 vs $A_2=0.41$; the first
    edge is 2, as predicted at small $\sigma$.
  - The first visible plateau is an intermediate saddle at $L\approx0.0859$, with log-type duration:
    17.8, 19.9, 21.9 at $\sigma$ = 0.1, 0.05, 0.025. On the same teacher the binary4 student gives
    184, 371, 745.
- **Hessian at the first plateau, same MPS4 teacher.**
  - binary4: $\lambda_{\min}=-0.0171$ ($\sigma=0.05$) and $-0.0037$ ($\sigma=0.0125$), so it vanishes
    $\propto\sigma^{1.1}$: a higher-order saddle.
  - MPS4: lowest three eigenvalues $(-0.2519,-0.1522,-0.0456)$ at $\sigma=0.0125$ and
    $(-0.2472,-0.1514,-0.0546)$ at 0.05, vs predicted $-\gamma_e=(-0.2548,-0.1527,-0.0413)$: strict.

## 9. Caveats and open issues

- **Asymptotics.** The rate, slope and order predictions are $\sigma\to0$ statements. Finite-$\sigma$
  corrections come from the initial projections onto each route's unstable direction (an $O(1)$
  random factor per route). When rates are comparable they reorder edges at accessible $\sigma$.
  When $\kappa_e$ is close to 1 they merge plateaus; the rank-one plateau lasts only
  $\approx(1-\kappa)\log(1/\sigma)/\gamma$.
- **Event definitions.** Use small fixed thresholds (linear regime) for escape times. The 50% event
  of `diagnostics.md` can skip intermediate saddles (§3.3).
- **Race-model rates at intermediate saddles** were taken from the Hessian of the GF state. Computing
  them from the teacher alone needs the balanced representation of the ALS critical point (not done).
- **Not covered:** large-$d$/large-$\chi$ students (many routes per edge, rates = top singular values
  of $B_e$); anisotropic covariances that are not aligned (the reduction of §1 still holds, since
  only $T\to\Sigma^{1/2}T$ changes); `binary8`/HT predictions (P9, P11), which are untested because
  $K=7$ gives $\sigma^{-5}$ first plateaus.
- **Conjectures to settle:** local maxima of $g$ have bond rank one; the odeco minimizer on
  $\mathcal M_{\le r}$ keeps the top $\min_er_e$ components; GF takes the minimal-$k$ route (supported
  by all runs here).

## References

Baldi & Hornik (1989) *Neural Networks* 2 · De Lathauwer, De Moor & Vandewalle (2000) SIMAX 21
(HOSVD; best rank-$(R_1,\dots,R_N)$/HOOI) · Grasedyck (2010) SIMAX 31(4) (HSVD) · Oseledets (2011)
SISC 33 (TT-SVD) · Holtz, Rohwedder & Schneider (2012) SISC 34 (ALS in TT) · Uschmajew &
Vandereycken (2013) LAA 439 (geometry of hierarchical tensors) · Kolda (2001) SIMAX 23 · Lim (2005)
CAMSAP (tensor singular vectors) · Stegeman & Comon (2010) LAA 433 · Anandkumar et al. (2014) JMLR 15 ·
Ge, Ren, Wang & Zhou (2021) NeurIPS (deflation in over-parameterized tensor decomposition, orthogonal
case) · Razin, Maman & Cohen (2022) ICML (hierarchical tensor factorization; local-component dynamics,
no saddle sequence) · Saxe, McClelland & Ganguli (2014) ICLR · Jacot et al. (2021) arXiv:2106.15933 ·
Pesme & Flammarion (2023) NeurIPS · Li, Luo & Lyu (2021) ICLR (GLRL) · Kumar & Haupt (2024 TMLR;
2025 TMLR).
