# Theory: which saddles are visited for teachers that are not odeco (Q7)

Theory note for `PLAN.md` Q7. **v2, 2026-10-09**, revised after an independent verification. The
verifier's scripts and outputs are in `.../scratchpad/theory2/teachers_verify/`, this revision's in
`.../scratchpad/theory2/teachers_rev/`, the original checks in `.../scratchpad/theory2/teachers/`
(`...` = `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926`).
Builds on `notes/model.md`, hypotheses H1–H3 of `notes/prior_knowledge.md`, `notes/theory_saddles.md`
(routes, local residual $R_C$, the $k^*$ rule, head start), `notes/theory_covariance.md` (product
covariance) and `notes/literature.md`.

**What this note adds to `theory_saddles.md`.** That note treats pattern (odeco-type) teachers and
already contains the route rule (its §2–3), the strictness of later saddles for generic teachers on an
MPS (its §5.6, R13) and the head-start formula (its §3.2). New here: (i) the critical points of the
bond-rank-constrained problem for arbitrary teachers and their relation to the HSVD (§2); (ii) the
2-core rates at the rank-one saddle in terms of the teacher, with interlacing bounds (§3.2); (iii) the
race over a sequence of strict saddles with rates taken at the exact intermediate critical points
(§3.4); (iv) the rank-one landscape of two non-orthogonal components (§4); (v) random TTN teachers and
teachers on another tree (§5–6).

**Status tags.** *proven*: complete argument given. *derived*: calculation under stated assumptions.
*heuristic*: plausible argument, not a derivation. *conjecture*. "[checked §7.x]" marks a numerical
confirmation, with the numbers in Section 7.

**Shared notation** (this note → `theory_saddles.md`):

| this note | `theory_saddles.md` | meaning |
|---|---|---|
| route $S$, blocks $Z_u$ ($u\in S$) | core set $C$, route amplitudes $b_v$ | connected cores whose new entries jointly produce a residual-correlated term |
| $B_e$ (2-core route on edge $e$) | local residual $R_C$, $C=\{v,p\}$ | $\langle R,W_S(Z)\rangle=Z_v^\top B_eZ_p$ |
| $\gamma_e=\sigma_{\max}(B_e)$ | $\lambda=\sigma_{\max}(R_C)$ (its §3.2) | 2-core escape rate; Hessian eigenvalue $-\gamma_e$ |
| $\mu_S$ | coupling $c$ | $k$-linear coupling $\max_{\lVert Z_u\rVert=1}\langle R_C,\otimes_uZ_u\rangle$ |
| $k_{\min}$ | $k^*$ | smallest admissible route degree |
| $\lambda$, $\sigma_2(M_e)$ | $s_q$, $s_r$ | learned amplitude, weight of the new direction |

With $s_q\to\lambda$ and $s_r\to\sigma_2(M_e)$, the head-start slope $(1-s_r/s_q)/(s_rs_q^{(K-2)/K})$ of
`theory_saddles.md` §3.2 becomes $A_e$ of §3.3 below.

## 0. Results at a glance

| # | result | status |
|---|---|---|
| R1 | A tensor with fixed bond ranks is a critical point of $L$ iff its matricization at every edge is an SVD truncation (onto *any* $r_e$ singular pairs) of the *environment-projected teacher* $T_e$, i.e. iff it is a one-site ALS fixed point. Known in substance (HOOI/ALS fixed-point conditions); stated here for any tree, with zero-padding into larger bond dimensions | proven |
| R2 | An HSVD truncation is critical iff its frames solve R1. It is critical (and a global minimizer) if all single-edge truncations coincide; for generic teachers it is not critical. Near-consistent teachers: $\lVert W_H-W_c\rVert=O(\varepsilon^2)$ and $L_H-L_c=\tfrac12\lVert W_H-W_c\rVert^2(1+O(\varepsilon))=O(\varepsilon^4)$ | proven / derived, [checked §7.1] |
| R3 | Small-init gradient flow (GF): the origin plateau ($L\approx L_0$, duration $\propto\sigma^{-(K-2)}$) ends along a KKT direction of $g=\langle W,T\rangle$; the next (rank-one) plateau sits at a critical point of the bond-rank-one problem (MPS, HT: a local best rank-one approximation), not at an HSVD truncation | derived, [checked §7.2] |
| R4 | Escapes go along routes (connected sets of $k$ cores; `theory_saddles.md` R1–R3, Furman et al. 2026 Lemmas D.7–D.9). $k=2$: strict saddle, time $\propto\log(1/\sigma)$; $k\ge3$: higher-order saddle, time $\propto\sigma^{-(k-2)}$. Admissibility ("room") decides which routes exist for generic teachers | derived |
| R5 | MPS student at the rank-one saddle: edge $e$ has Hessian eigenvalues $-\lambda^{(N-2)/N}\sigma_j(M_e)$, $j=2..d$, each with multiplicity $\chi_e-1$; escape rate $\gamma_e=\lambda^{(N-2)/N}\sigma_2(M_e)$; $\sigma_j(M_e)\le\sigma_j^e(T)$ | derived (bound proven), [checked §7.2] |
| R6 | Second escape: $\Delta t_2=A_{e^*}\log(1/\sigma)+O(1)$, $A_e=(1-\sigma_2(M_e)/\lambda)/\gamma_e$ | derived, [checked §7.3: <1%] |
| R7 | Two non-orthogonal rank-one components: equal weights give two rank-one maxima iff $c<c^*=(N-2)/N$; unequal weights: fold $c_{\rm fold}(N,\rho)$ (exact table; small-$\rho$ formula). The rank-one plateau is a local best rank-one approximation, never a component | proven / derived, [checked §7.2] |
| R8 | On an MPS the second component is learned bond by bond through strict saddles at critical points of mixed-rank problems. As $\sigma\to0$ the order and the slopes $dt_e/d\log(1/\sigma)$ follow the accumulated-utility (AGF) race with rates at the exact critical points. At $\sigma\gtrsim10^{-3}$ near-tied later escapes can come in either order | derived ($\sigma\to0$), [checked §7.3: 8 seeds, $\sigma=10^{-2}$ to $10^{-8}$] |
| R9 | Generic teachers: an MPS student has one polynomial plateau (the origin plateau), then only logarithmic ones. Binary/HT students need $k\ge3$ (polynomial) at the rank-one saddle, and always for the root bipartition | derived, [checked §7.4–7.5] |
| R10 | Teacher on another tree: required student ranks are cut bounds. A `binary4` student is a depth-3 deep linear network (DLN) on $T_{(01\mid23)}$ (nested SVD saddles, $\sigma^{-1}$ plateaus) | proven / derived, [checked §7.5] |
| R11 | GF plateaus converge to critical points: at the minimum-speed point of the rank-one plateau the criticality defect falls as $\sigma^{\alpha}$, $\alpha$ set by the slowest stable mode ($\alpha=0.81$ for the MPS4 example), while the HSVD defect does not depend on $\sigma$ | derived (crossing heuristic), [checked §7.6] |

## 1. Setting and notation

- $N$ sites, physical dimension $d$, tree $\tau$ with $K$ cores. An edge $e=(v,p)$ joins child $v$ to
  parent $p$. $\mathrm{below}(e)$ = sites in the subtree of $v$; $\mathrm{above}(e)$ = the rest.
- Teacher $T=W^*$. **Whitened inputs**: $L(W)=\tfrac12\lVert W-T\rVert_F^2$. *Product covariance,
  landscape only:* for $\Sigma=\otimes_i\Sigma_i$ the map $T\to\Sigma^{1/2}T$, $W\to\Sigma^{1/2}W$
  preserves bond ranks, critical points, critical values and saddle types, so §2 and the landscape
  statements of §4–6 transfer with $T$ replaced by $\Sigma^{1/2}T$. It does **not** transfer the
  dynamics: in whitened coordinates GF is preconditioned by $\Sigma_i$ on the physical legs, and the
  near-origin escape is driven by $\Sigma T$ (`theory_covariance.md` C1, C4). All rates, slopes and
  orders in §3–6 are for whitened inputs.
- $X_{(e)}$: matricization of a tensor $X$ across $e$ (rows = $\mathrm{below}(e)$). Its singular values
  are the bond (Schmidt) spectrum $\sigma_1^e(X)\ge\sigma_2^e(X)\ge\dots$; $s_j^e(t)$ denotes the
  spectrum of the student $W(\theta(t))$.
- Bond ranks $r=(r_e)$. $\mathcal M_r$ = tensors with exact ranks $r_e=\mathrm{rank}\,X_{(e)}$, a smooth
  manifold (Holtz, Rohwedder & Schneider 2012, *Numer. Math.* 120, for TT; Uschmajew & Vandereycken
  2013 for hierarchical Tucker; the same argument works for any tree). $\mathcal M_{\le r}$ is its
  closure.
- Frames of $W\in\mathcal M_r$ at $e$: $U_e$ = column space of $W_{(e)}$ (dimension $r_e$) and $V_e$ =
  row space. $P_S$ is the orthogonal projector onto $S$.
- Local space of node $v$: $\mathcal V_v=\big(\otimes_{c\in\mathrm{ch}(v)}U_c\big)\otimes\mathbb R^{d_v}\otimes V_v$,
  where $d_v=d^{\#\text{sites of }v}$ and the $V_v$ factor is absent at the root.
- Environment spaces of $e=(v,p)$: $E_e^-=\big(\otimes_{c\in\mathrm{ch}(v)}U_c\big)\otimes\mathbb R^{d_v}$
  (on $\mathrm{below}(e)$) and $E_e^+=\big(\otimes_{c\in\mathrm{ch}(p)\setminus v}U_c\big)\otimes\mathbb R^{d_p}\otimes V_p$
  (on $\mathrm{above}(e)$), so $\mathcal V_v=E_e^-\otimes V_e$ and $\mathcal V_p=U_e\otimes E_e^+$.
- **Environment-projected teacher**: $T_e:=P_{E_e^-}\,T_{(e)}\,P_{E_e^+}$, a matrix.
- HSVD (Grasedyck 2010, root to leaves): $\pi_e$ = projector onto the top-$r_e$ left singular subspace
  of $T_{(e)}$, all computed from $T$; $W_H=\prod_e\pi_e\,T$, applied level by level. (This is not the
  truncated HOSVD; their rank-one truncations differ, §7.2.)
- Student: fan-in init with scale $\sigma$ per core (`notes/model.md`); GF time $t$; $\kappa$ and $\gamma$
  below are dimensionless rates, not the model's per-core scales.
- **Plateau names.** *Origin plateau*: $L\approx L_0=\tfrac12\lVert T\rVert^2$, duration
  $\propto\sigma^{-(K-2)}$; it ends at $t_1$, the time at which $L$ is halfway between $L_0$ and the next
  plateau. *Rank-one plateau*: the next one, at a bond-rank-one critical point. *Intermediate plateaus*:
  critical points of mixed-rank problems. **Events**: $t_e$ = first time $s_2^e\ge f\,\sigma_2^e(T)$
  (bond $e$ gains rank), with a fraction $f\in[0.03,0.1]$ (§3.3).

## 2. (a) Critical points of the bond-rank-constrained problem; when is HSVD one?

### 2.1 Characterization

**Lemma (matrix).** *proven.* Let $A$ be a matrix, and $U$, $V$ subspaces of dimension $r$ with
$P_UA=AP_V=:W$ of rank $r$. Then $U$ is spanned by $r$ left singular vectors of $A$, and $W$ is the
SVD truncation of $A$ onto them. *Proof:* $P_UAA^\top=AP_VA^\top=WW^\top$ is symmetric, so $P_U$
commutes with $AA^\top$. For $N=2$ this is the classical fact that the critical points of rank-$r$
approximation are truncations onto any $r$ singular pairs (Eckart–Young; Baldi & Hornik 1989).

**Proposition 1 (critical points on $\mathcal M_r$).** *proven.* $W\in\mathcal M_r$ is a critical point
of $L|_{\mathcal M_r}$ iff, for every edge $e$,
$$W_{(e)}=P_{U_e}T_e=T_eP_{V_e},$$
that is, iff $U_e$ is spanned by $r_e$ left singular vectors of $T_e$ and $W_{(e)}$ is the
corresponding SVD truncation of $T_e$. Equivalently, $W$ is a fixed point of one-site ALS/DMRG
($W\leftarrow P_{\mathcal V_v}T$ for every $v$).

*Proof.* At a full-rank parameterization the differential with respect to core $v$ spans exactly
$\mathcal V_v$, because the other cores' subtree tensors span the frames. So
$T_W\mathcal M_r=\sum_v\mathcal V_v$ (as in Uschmajew & Vandereycken 2013), and $W$ is critical iff
$P_{\mathcal V_v}(T-W)=0$, i.e. $P_{\mathcal V_v}T=W$ for all $v$ (since $W\in\mathcal V_v$). For an edge
$e=(v,p)$, $W\in\mathcal V_v\cap\mathcal V_p$ gives $U_e\subseteq E_e^-$ and $V_e\subseteq E_e^+$. Hence
$P_{\mathcal V_v}T=T_eP_{V_e}$ and $P_{\mathcal V_p}T=P_{U_e}T_e$ (matricized at $e$). Every node is an
endpoint of some edge, so the node conditions are equivalent to the edge conditions, and the Lemma
finishes the proof. ∎

With larger bond dimensions $\chi_e>r_e$, a critical point of the rank-$r$ problem, zero-padded on both
sides of each bond, is a critical point of the parameterized loss $\theta\mapsto L(W(\theta))$ (the
padded entries receive zero gradient: their environments contain a zero block). This is how the
saddles visited by GF appear in a student with spare bond dimension.

**Prior art.** For Tucker (MPS3 with ranks $(r_1,r_2)$ and a free middle mode) Proposition 1 is the
HOOI fixed-point condition (De Lathauwer, De Moor & Vandewalle 2000) and the first-order optimality
condition on Grassmannians of Eldén & Savas (2011), who also develop its perturbation theory. For TT
and HT the equivalence "critical point of the restricted functional ⇔ fixed point of one-site ALS" is
standard (Holtz, Rohwedder & Schneider 2012, SISC 34; Uschmajew & Vandereycken 2013 for the tangent
spaces). Furman et al. (2026, Thm 5.2) show that every non-global critical point of exactly this model's
parameterized loss is rank-deficient. What the formulation above adds is bookkeeping: the edge-wise
SVD form makes explicit that the kept pairs need not be the top $r_e$ ones (§2.3, §4.2), and that a
lower-rank critical point lives inside a larger student by zero-padding.

### 2.2 When is the HSVD truncation critical?

**Proposition 2.** *proven.*

1. *Criterion.* $W_H$ (assumed to have exact ranks $r$) is critical iff its own frames satisfy
   Proposition 1. For MPS3 this reads: $U_1$ is invariant under
   $G_1^\perp=T_{(1)}(I\otimes P_{V_3}^\perp)T_{(1)}^\top$, the mode-1 Gram matrix of the part of $T$
   discarded at the other edge,
   and symmetrically for $V_3$. This "leakage" commutator $[P_{U_1},G_1^\perp]$ is generically nonzero,
   so truncated HOSVD/HSVD is not a HOOI fixed point.
2. *Sufficient: consistent truncations.* Let $\tau_e(T)$ be the best rank-$r_e$ approximation across
   $e$. If $\tau_e(T)=W$ for every edge, then $W_H=W$, it is critical, and it is the **global
   minimizer** on $\mathcal M_{\le r}$, because $\lVert T-X\rVert\ge\lVert T-\tau_e(T)\rVert$ for every $X$
   with $\mathrm{rank}_eX\le r_e$. Equivalently: $T=W+T_\perp$ with $W$ and $T_\perp$ having mutually
   orthogonal frames on both sides of every edge, and $\sigma_{\min}(W_{(e)})>\sigma_{\max}(T_{\perp,(e)})$.
   Examples: an odeco teacher truncated to the same component set on every edge; any tree whose
   rank-carrying edges all induce the same bipartition (e.g. `binary4`: both edges induce
   $\{01\}|\{23\}$, so the problem is a matrix problem).
3. *Inconsistent subsets.* An odeco teacher with different component sets $S_e$ per edge gives
   $W_H=\sum_{r\in\cap_eS_e}s_r\otimes_iu_r^{(i)}$, whose ranks are below $r$. In the one case checked
   (MPS4, ranks $(2,1,2)$) ALS on $\mathcal M_{\le r}$ returned this same tensor. *Conjecture:* for odeco
   teachers the minimizer on $\mathcal M_{\le r}$ keeps the top $\min_er_e$ components.

The HSVD is quasi-optimal ($\lVert T-W_H\rVert\le\sqrt{2N-3}\,\min_{X\in\mathcal M_{\le r}}\lVert T-X\rVert$,
Grasedyck 2010), but quasi-optimality says nothing about criticality.

**Proposition 3 (near-consistent teachers).** *derived (first-order perturbation), [checked §7.1].*
Let $T=W+\varepsilon E$ with $W$ consistent as in Proposition 2.2 and $E$ generic. The HSVD frames and
the frames of the nearby critical point $W_c$ agree to first order in $\varepsilon$: both are first-order
rotations of $W$'s frames driven by $W_{(e)}E_{(e)}^\top+E_{(e)}W_{(e)}^\top$, and projecting onto the
environment does not change these terms because $W$'s rows already lie in it. Hence
$\lVert W_H-W_c\rVert=O(\varepsilon^2)$. Moreover $R_c=T-W_c$ is normal to $\mathcal M_r$ at $W_c$ (criticality)
and $O(\varepsilon)$, while $W_H-W_c$ is tangent up to a normal part of order
$\lVert W_H-W_c\rVert^2$ (curvature of $\mathcal M_r$), so
$$L(W_H)-L(W_c)=\tfrac12\lVert W_H-W_c\rVert^2-\langle W_H-W_c,R_c\rangle=\tfrac12\lVert W_H-W_c\rVert^2\,\big(1+O(\varepsilon)\big)=O(\varepsilon^4).$$
The prefactors of $\varepsilon^2$ and $\varepsilon^4$ depend on the draw of $(W,E)$; the ratio
$(L_H-L_c)/(\tfrac12\lVert W_H-W_c\rVert^2)\to1$ does not (§7.1). Eldén & Savas (2011) give the first-order
perturbation theory of the Tucker optimum; we did not find the $\varepsilon^4$ HSVD gap stated.

### 2.3 Which critical points does GF visit?

*derived (small-init asymptotics), [checked §7.2–7.4].* Near the origin GF is the gradient flow of
$g(\theta)=\langle W(\theta),T\rangle$, which is homogeneous of degree $K$. Its direction converges to a
KKT point of $g$ on the unit sphere before the norm grows (H1; Kumar & Haupt 2024, 2025). That the
point reached from a random init is a local maximum is expected but not proven here (gauge orbits make
every KKT point degenerate for $\chi\ge2$).

- The **global** maximum of $g$ on the sphere has **bond rank one on every edge** (*proven*). Write
  the bond sums out and apply Hölder edge by edge, as in
  $\sum_{ab}x_ay_{ab}z_b\le\lVert x\rVert\lVert Y\rVert_F\lVert z\rVert$. This gives $g\le\lVert T\rVert_{\rm blk}\prod_v\lVert G_v\rVert_F$, where
  $\lVert T\rVert_{\rm blk}$ is the maximum of $\langle T,\otimes_bM_b\rangle$ over unit node blocks $M_b$;
  equality holds at bond rank one.
- That *local* maxima also have bond rank one is a *conjecture*, true for every rank-one plateau in §7.

Bond rank one means different things on different trees:

- If every node carries at most one site (MPS, HT), bond rank one means $W=\otimes_iw_i$. The rank-one
  plateau is then a **local best rank-one approximation** of $T$ (a critical point of
  $\max\langle T,\otimes_iw_i\rangle$ over unit $w_i$, i.e. a tensor singular vector tuple; Lim 2005). Its
  loss is $L_1=\tfrac12(\lVert T\rVert^2-\lambda^2)$ with $\lambda=\langle T,\otimes_iw_i\rangle$.
- For `binary` trees (two sites per bottom node) bond rank one means $W=\otimes_bM_b$ with arbitrary
  $d\times d$ blocks $M_b$. For `binary4` this is the top singular pair of $T_{(01|23)}$: a unique
  maximum, so no H1-type randomness.

Later plateaus are critical points of mixed-rank problems (Proposition 1 at the current ranks). For
generic teachers these are **not** HSVD truncations (§7.1). Which critical point is visited is set by
the escape routes (Section 3), not by the HSVD. In the cases of §7 the visited plateau has lower loss
than the HSVD truncation at the same ranks (by 6.6% of $L_0$ for MPS4 seed 1 at ranks (1,1,1), and by
6–10% of the plateau value for the two-component teachers of §7.2), but this is not general: a plateau
in a minor basin is a worse critical point (two components, $c=0.3$: minor rank-one plateau at 0.4975,
HSVD 0.1982).

## 3. Escaping a saddle: routes and the generalized locality rule

### 3.1 Routes

This subsection restates `theory_saddles.md` §2–3 (F1, F2, Proposition 1 there) in the form used
below; the admissibility condition is its §5.6.

Take a saddle $W\in\mathcal M_r$ with balanced cores $\theta^*$ (small init keeps the conservation laws
$Q_e\approx0$, `notes/model.md` §5), spare bond dimension $\chi_e>r_e$, and residual $R=T-W$. A
**route** is a connected set $S$ of $k\ge2$ cores (a subtree) together with a new bond index on every
edge internal to $S$, orthogonal to the $r_e$ learned ones. Its variables are the blocks $Z_u$,
$u\in S$: the entries of $G_u$ with new indices on its $S$-internal legs and learned indices on all
other legs. If only these blocks move, then exactly
$$W(\theta^*+Z)=W+W_S(Z),\qquad L=L(W)-\langle R,W_S(Z)\rangle+\tfrac12\lVert W_S(Z)\rVert^2,$$
with $W_S$ $k$-linear in the blocks. *derived:* with balanced blocks of common amplitude $x$ the leading
dynamics is a depth-$k$ DLN mode, $\dot x=\mu_Sx^{k-1}$, $\mu_S=\max_{\lVert Z_u\rVert=1}\langle R,W_S(Z)\rangle$.

- $k=2$ (edge $e=(v,p)$): the Hessian on the route is $\begin{pmatrix}0&-B_e\\-B_e^\top&0\end{pmatrix}$
  with $B_e=E_v^*(T_e-W_{(e)})E_p$, where $E_v$ maps $Z_v$ to $\mathrm{below}(e)$ through $v$'s learned
  child subtrees and $E_p$ maps $Z_p$ to $\mathrm{above}(e)$. The saddle is **strict**, with eigenvalue
  $-\gamma_e$, $\gamma_e=\sigma_{\max}(B_e)\le\lVert E_v\rVert\lVert E_p\rVert\sigma_{r_e+1}(T_e)$ (the bound
  holds when the kept pairs are the top $r_e$ of $T_e$). Escape is exponential, $x\propto e^{\gamma_et}$.
- $k\ge3$: no negative curvature along the route (a **higher-order saddle**); from $x_0$ the blow-up
  time is $t^*=1/((k-2)\mu_Sx_0^{k-2})$.

This is the dynamical counterpart of Furman et al. (2026, Lemmas D.7–D.9 and Remark D.10), who build
the same multilinear curve along a connected set of rank-deficient edges, bound the saddle order by
its number of nodes, and note that two nodes give a strict saddle. Frandsen & Ge (2022) exhibit the
same orders for Tucker (descent of order $\varepsilon^2,\varepsilon^3,\varepsilon^4$ when 2, 3, 4 cores
must move). The exponent $k-2$ is the tree analogue of the leap of Abbe, Boix-Adserà & Misiakiewicz
(2023).

**Admissibility ("room").** A route can raise a rank only if each leaf $u$ of $S$ (as a subtree) offers
directions outside the learned frame of its edge into $S$: the leg's environment (physical legs and
outside bonds) must have dimension larger than the current rank. A site-less node whose outside legs
all have rank 1 has no room, e.g. the internal nodes of `binary` and `ht` trees at the rank-one
saddle. A route ending there only reshuffles the learned component; its coefficient vanishes at a
critical point.

**Generalized locality rule.** *derived at leading order; that GF takes this route as $\sigma\to0$ is a
conjecture supported by §7.* At a saddle the next escape uses an admissible route of the smallest
degree $k$ with $\mu_S\ne0$ (exponential beats polynomial, $\sigma^{-(k-2)}$ beats $\sigma^{-(k'-2)}$ for
$k<k'$). Among $k=2$ routes the winner is the one whose accumulated log-amplitude reaches zero first
(rate *and* head start, §3.4), not simply the largest $\gamma_e$. Among $k\ge3$ routes ties are decided by
$\mu_Sb_{\rm eff}^{k-2}$, i.e. by the coupling and the random route amplitudes (`theory_saddles.md`
§3.4).

- **Odeco/pattern teachers** (component 2 equals component 1 except on sites $D$): $\mu_S=0$ unless
  $\mathrm{sites}(S)\supseteq D$, so $k=\lvert\mathrm{Steiner}(D)\rvert$: H2, proven in `theory_saddles.md`
  Proposition 2.
- **Generic teachers**: $\mu_S\ne0$ for every admissible route, so $k_{\min}=2$ whenever some edge has
  room on both sides (`theory_saddles.md` §5.6). The edges whose rank must increase *in the next
  escape* are what matters: for generic teachers one edge suffices, for odeco teachers the whole
  Steiner tree must grow at once.

### 3.2 Rates at the rank-one saddle of an MPS student

*derived, [checked §7.2–7.4].* MPS student (core $k$ carries site $k$, $K=N$) at a rank-one critical
point $W_1=\lambda\otimes_iw_i$, a local maximum of the rank-one problem. Balanced cores have amplitude
$q=\lambda^{1/N}$. The 2-core route on edge $e=(k,k+1)$ has blocks $X\in\mathbb R^{d\times(\chi_e-1)}$
(core $k$: learned left index, new right indices) and $Y\in\mathbb R^{(\chi_e-1)\times d}$ (core $k+1$:
new left indices, learned right index), and
$$\langle R,W_S\rangle=q^{N-2}\sum_{a=1}^{\chi_e-1}X_a^\top\big(M_e-\lambda w_kw_{k+1}^\top\big)Y_a,\qquad
M_e:=T\times_{i\notin\{k,k+1\}}w_i\ \ (d\times d).$$
At a local maximum $(w_k,w_{k+1})$ is the top singular pair of $M_e$, with value $\lambda$. So
$B_e=q^{N-2}(M_e-\lambda w_kw_{k+1}^\top)$ has singular values $q^{N-2}\sigma_j(M_e)$, $j=2..d$, and edge
$e$ contributes the Hessian eigenvalues
$$-\lambda^{(N-2)/N}\sigma_j(M_e),\quad j=2,\dots,d,\ \text{each with multiplicity }\chi_e-1;\qquad
\boxed{\gamma_e=\lambda^{(N-2)/N}\,\sigma_2(M_e)}.$$
Routes on different edges decouple at second order. With $d=2$, $\chi=2$ this is one eigenvalue
$-\gamma_e$ per bond. In the language of `theory_saddles.md` §3.2, $\gamma_e=\sigma_{\max}(R_C)$ with
$R_C=B_e$.

**Link to the teacher's bond spectra** (*proven*): $M_e=P^\top T_{(e)}Q$ with
$P=(\otimes_{i<k}w_i)\otimes I_d$ and $Q=I_d\otimes(\otimes_{i>k+1}w_i)$, both with orthonormal columns. By interlacing for
compressions,
$$\lambda=\sigma_1(M_e)\le\sigma_1^e(T),\qquad \sigma_j(M_e)\le\sigma_j^e(T),\qquad
\gamma_e\le\lambda^{(N-2)/N}\sigma_2^e(T).$$
Equality holds when the teacher's Schmidt vectors are products with $w$ on the far sites. When the
second Schmidt vectors are entangled far from the edge, $\sigma_2(M_e)\ll\sigma_2^e(T)$ (§4.3).

For a general tree at the rank-one saddle the same holds with $M_e$ = $T$ contracted with the learned
rank-one structure everywhere except cores $v$ and $p$ (a $d_v\times d_p$ matrix) and amplitude factor
$\lambda^{(K-2)/K}$. The route is admissible iff both $v$ and $p$ carry sites.

### 3.3 Escape time including growth during the first escape

*derived (assumptions: $\sigma\to0$; balanced cores; the route's initial projection is
$\propto\sigma$).* This is the head start of `theory_saddles.md` §3.2 with $s_q\to\lambda$,
$s_r\to\sigma_2(M_e)$, and the 2-homogeneous case of the accumulated-utility rule of Kunin et al. (2025).
During the first escape the main amplitude obeys $\dot q=q^{N-1}(\lambda-q^N)$ and the route blocks
$\dot x=q^{N-2}\sigma_2(M_e)x$, so the route grows by a factor $(1/\sigma)^{\kappa_e}$,
$\kappa_e=\sigma_2(M_e)/\lambda$, before the rank-one plateau starts. With $x(0)\propto\sigma$:
$$\boxed{\Delta t_2=t_{e^*}-t_1=A_{e^*}\log(1/\sigma)+O(1),\qquad A_e=\frac{1-\sigma_2(M_e)/\lambda}{\lambda^{(N-2)/N}\sigma_2(M_e)},\qquad e^*=\arg\min_eA_e.}$$
For an MPS $\arg\min_eA_e=\arg\max_e\sigma_2(M_e)$, since $A_e$ decreases in $\sigma_2(M_e)$. Also
$s_2^e(t_1)\propto\sigma^{2(1-\kappa_e)}$ at the end of the first escape (on the plateau
$s_2^e\propto q^{N-2}xy$), and $\tfrac{d}{dt}\log s_2^e=2\gamma_e$ on the plateau.

**Event thresholds.** Time escapes with a threshold in the linear regime that scales with the target:
$t_e$ = first time $s_2^e\ge f\,\sigma_2^e(T)$ with $f\in[0.03,0.1]$, and check that the slopes do not
depend on $f$. A fixed absolute threshold (the original version used $\max_es_2^e=10^{-2}$) never fires
when $\sigma_2^e(T)<10^{-2}$. The 50%-of-final event of `notes/diagnostics.md` can fire only after later
escapes, because intermediate saddles have $s_2^e$ well below the final value (0.148 vs 0.56 in §7.3).

### 3.4 The whole sequence on an MPS: an accumulated-utility race

*derived as $\sigma\to0$ under assumption (A-carry) below; [checked §7.3, 8 seeds, $\sigma$ down to
$10^{-8}$]; at accessible $\sigma$ only a heuristic.* Give each pending 2-core route $\rho$ the
log-amplitude $\ell_\rho=\log x_\rho/\log(1/\sigma)\in[-1,0]$ and use the rescaled time
$\tau=(t-t_1)/\log(1/\sigma)$:

- $\ell_\rho(0)=-1+\kappa_\rho$ (growth during the first escape, §3.3);
- while GF sits at saddle $S_n$, $d\ell_\rho/d\tau=\gamma_\rho(S_n)$, the top singular value of $B_\rho$ at
  the *exact* critical point $S_n$ (balanced, zero-padded);
- $\rho$ escapes when $\ell_\rho=0$; the flow then moves in $O(1)$ time to $S_{n+1}$, the critical point
  with that edge's rank raised by one;
- (A-carry) the other $\ell$'s carry over: at each transition a pending route's accumulated amplitude
  projects onto its new unstable direction with a $\sigma$-independent factor $p\ne0$, which shifts
  $\ell$ by $\log p/\log(1/\sigma)\to0$.

This is the rule of Alternating Gradient Flows (Kunin et al. 2025): a dormant unit activates when its
utility, integrated over all earlier phases, reaches $\log(1/\lVert\theta_0\rVert)$ (2-homogeneous case).
It differs from greedy low-rank learning (Li, Luo & Lyu 2021), which has no memory between saddles,
and is the same kind of jump process as Pesme & Flammarion (2023). Degree-$k\ge3$ routes enter with
amplitude $\sim\sigma$ and polynomial times, so as $\sigma\to0$ they lose to any $k=2$ route.

*Why (A-carry) holds on an MPS although the route blocks change* (*heuristic*; [checked §7.3]). When
bond $e-1$ gains rank, the edge-$e$ route's left block acquires entries on the new bond-$(e-1)$ index.
At the previous saddle these entries belonged to a $k=3$ route and are still $\sim\sigma$, and the new
unstable direction puts a sizable weight on them (32% for edge 1 of the mixed teacher of §4.3). Its
right block (core $k+1$, 50% of the weight) is unchanged, however, so the head-started right block
carries the accumulated amplitude into the new mode and drives the fresh left entries at the new rate:
in that example the old and new unstable eigenvectors overlap by $p=0.775$. One escape raises one bond,
so the two blocks of a pending route never both change at the same transition. (A-carry) fails only if
the projection vanishes identically (e.g. by orthogonality, as for odeco teachers).

**Finite-$\sigma$ corrections** (*derived from the same picture*). Event times are
$t_e-t_1=\tau_e\log(1/\sigma)+C_e+o(1)$, where the offset $C_e=O(\lvert\log p\rvert/\gamma)$ depends on
the init. Measured offsets reach about 50 time units (§7.3). Two escapes $e,e'$ come in the asymptotic
order only once $\log(1/\sigma)\,\lvert\tau_e-\tau_{e'}\rvert\gg\lvert C_e-C_{e'}\rvert$. At accessible $\sigma$
this often fails for later, near-tied escapes, whose order is then seed-dependent (for the mixed teacher
of §4.3 one seed in eight keeps the non-asymptotic order down to $\sigma=10^{-6}$). The second escape is
robust when the rank-one rates are well separated: its slope $A_{e^*}$ is reached within a few percent at
$\sigma\approx10^{-3}$–$10^{-2}$.

## 4. (b) Two non-orthogonal rank-one components

$T=s_1\otimes_iu_i+s_2\otimes_iv_i$, unit factors, $s_1\ge s_2>0$, overlaps $c_i=\langle u_i,v_i\rangle\in[0,1)$,
$\phi_i=\arccos c_i$, $C=\prod_ic_i$, $\rho=s_2/s_1$, $\lVert T\rVert^2=s_1^2+s_2^2+2s_1s_2C$.

### 4.1 Rank-one critical points

*proven.* Stationarity gives $w_i\propto s_1a_{-i}u_i+s_2b_{-i}v_i$ with
$a_{-i}=\prod_{j\ne i}\langle w_j,u_j\rangle$ and $b_{-i}=\prod_{j\ne i}\langle w_j,v_j\rangle$, so
$w_i\in\mathrm{span}(u_i,v_i)$. With $\theta_i$ the angle of $w_i$ from $u_i$ in that plane,
$$f(\theta)=s_1\prod_i\cos\theta_i+s_2\prod_i\cos(\phi_i-\theta_i),\qquad \lambda=\text{value of the local maximum reached}.$$

**Number of local maxima.**

- $s_1=s_2$, $c_i=c$ (*proven* for symmetric critical points; *numerically exhaustive* otherwise,
  including out-of-plane starts in $d=3$, §7.2). At the merged point $\theta_i=\phi/2$ the Hessian of $f$
  has eigenvalue $-2\cos^{N-2}\tfrac\phi2[\cos^2\tfrac\phi2-(N-1)\sin^2\tfrac\phi2]$ along $(1,\dots,1)$ and
  $-2\cos^{N-2}\tfrac\phi2<0$ on its complement. Hence
  $$\text{two local maxima (one near each component) iff } c<c^*=\frac{N-2}{N},$$
  and otherwise a single merged maximum on the bisector. For $N=2$ (a matrix) there is always one. The
  symmetric-tensor version ($u^{\otimes N}+v^{\otimes N}$) has the same threshold. We did not find this
  threshold in the literature; it is elementary.
- $s_2<s_1$: the minor maximum (near $v$) disappears at a fold $c_{\rm fold}(N,\rho)$ (exact values in
  §7.2, from the symmetric reduction). For **small $\rho$** (*derived*, expanding near $v$ with
  $\cos(\phi-\psi)\approx c+\psi$):
  $$c_{\rm fold}(N,\rho)\approx\frac{N-2}{N-1}\Big(\frac{\rho}{N-1}\Big)^{1/(N-2)}.$$
  It is 1–6% low at $\rho=0.2$ and 5–13% low for $\rho\ge0.6$ (e.g. 0.298 vs the exact 0.3257 at $N=4$,
  $\rho=0.6$). Long tensors keep both basins up to larger overlaps ($c_{\rm fold}\to1$ as
  $N\to\infty$).

### 4.2 The rank-one plateau

*derived, [checked §7.2].* The rank-one plateau is the rank-one critical point selected by the
near-origin flow (H1): the KKT point whose basin contains the init direction. For $c>c_{\rm fold}$ there
is a single local maximum, which is then the best rank-one approximation. For $c<c_{\rm fold}$ there are
two and the init chooses; the plateau is the best rank-one approximation iff the major one is reached
(at $c=0.3$, $s=(1,0.6)$, $N=4$: $\lambda=1.00542$ vs 0.6122 for the minor one; all runs of §7.2 reached
the major one). It is **never one of the components**. Its loss is $L_1=\tfrac12(\lVert T\rVert^2-\lambda^2)$.
For small overlaps, near $u$:
$$\theta_i=\frac{s_2}{s_1}C\tan\phi_i+\dots,\qquad
\lambda=s_1+s_2C+\frac{s_2^2C^2}{2s_1}\sum_i\tan^2\phi_i+\dots$$
The expansion parameter is $s_2C\tan^2\phi_i/s_1$. The plateau lies below the "component learned" level
$\tfrac12s_2^2(1-C^2)$ by $\tfrac12(\lambda^2-(s_1+s_2C)^2)$.

### 4.3 Second phase: bond-by-bond strict saddles, not deflation

*derived, [checked §7.3].* For an MPS student the residual after the rank-one fit has a nonzero
transverse part at every edge, so by §3.2 every edge $e=(k,k+1)$ has a strict-saddle route with
$$\sigma_2(M_e)=\frac{s_1s_2\,a_e\,b_e\,\sin\phi_k\sin\phi_{k+1}}{\lambda},\qquad
a_e=\prod_{i\ne k,k+1}\cos\theta_i,\ \ b_e=\prod_{i\ne k,k+1}\cos(\phi_i-\theta_i),$$
which is exact ($d=2$): $\det M_e=s_1s_2a_eb_e\sin\phi_k\sin\phi_{k+1}$ and $\sigma_1(M_e)=\lambda$. For small
$C$ it is $\approx s_2\big(\prod_{i\ne k,k+1}c_i\big)\sin\phi_k\sin\phi_{k+1}$. **The bond that grows first is
the one whose two sites differ most while the other sites agree most.** This is not "largest teacher
Schmidt value": for $c=(0.2,0.5,0.7,0.3)$, $s=(1,0.6)$, the teacher's $\sigma_2^e(T)=(0.567,0.566,0.548)$
are nearly tied, while $\sigma_2(M_e)=(0.127,0.036,0.059)$, and edge 0 grows first in all runs
(§7.3: 8 seeds at five values of $\sigma$, plus the verifier's 2 seeds). For equal overlaps the teacher's
middle Schmidt value is the largest (0.588 vs 0.555), yet all $\gamma_e$ are equal, and the edge that grows
first varies with the seed.

How the second component is learned:

- *Equal overlaps*: all $\gamma_e$ are equal and all bonds go from rank 1 to 2 in one transition.
- *Unequal overlaps*: rank is added bond by bond. Each intermediate plateau is a critical point of a
  mixed-rank problem. Mixed example: ranks $(1,1,1)$ at $L=0.175767$, then $(2,1,1)$ at 0.166436, then
  $(2,1,2)$ at 0.159878 (or $(2,2,1)$ at 0.150277 if edge 1 goes before edge 2), then $T$. Race
  prediction (§3.4) with the exact rates — rank one $\gamma=(0.1283,0.0360,0.0592)$; at $(2,1,1)$: edge 2
  0.1006, edge 1 0.0770; at $(2,1,2)$: edge 1 0.1341; at $(2,2,1)$: edge 2 0.2105 — gives the
  asymptotic order $0\to2\to1$ with slopes $dt_e/d\log(1/\sigma)=6.823,\ 12.178,\ 14.467$ (confirmed to
  0.5% at $\sigma\le10^{-6}$, §7.3). If edge 1 goes second (a finite-$\sigma$ reordering), its slope is
  $6.823+0.7193/0.0770=16.16$.
- *Not deflation.* The learned rank-one direction $w$ is not a component. It keeps rotating toward $u$
  ($\theta_i\to0$) while the rest is learned, and GF converges to $T$ exactly. Greedy deflation (best
  rank-one of $T$, then of the residual) does not recover a rank-2 tensor in two steps; the residual can
  even have higher rank (Stegeman & Comon 2010; Kolda 2001). GF recovers $T$ because all cores keep
  moving. Ge, Ren, Wang & Zhou (2021) prove deflation-like dynamics only for orthogonal components.
- *Contrast, `binary4` student*: a depth-3 DLN on $T_{(01|23)}$ (§6.2). Its saddles are the nested SVD
  truncations of that matrix, so it is deflation-like on the matricization. Its rank-one plateau is at
  $\tfrac12\sigma_2^2(T_{(01|23)})$, lower than the MPS rank-one plateau since $\lambda\le\sigma_1(T_{(01|23)})$,
  and all its escapes are degree 3 ($\propto\sigma^{-1}$).

### 4.4 The odeco limit $c\to0$

*heuristic, [checked §7.3].* $\sigma_2(M_e)\propto\prod_{i\ne k,k+1}c_i\to0$, so the 2-core rate vanishes and
the $K$-core route (odeco mechanism) takes over, with time $T_{\rm odeco}(\sigma)=a_N\sigma^{-(N-2)}$. The
prefactor $a_N$ is set by the init's projections and is **random**: for MPS3 with $s=(1,0.6)$, $a_3$
ranges over 1.07–7.76 (median 3.35) across 8 uncorrelated seeds (§7.3). The two routes act in parallel and the escape
follows the faster one. For symmetric small overlaps on an MPS, $A(c)\approx1/(s_2c^{N-2})$, and
$A(c_\times)\log(1/\sigma)=T_{\rm odeco}(\sigma)$ gives
$$c_\times\approx\sigma\Big[\frac{\log(1/\sigma)}{s_2\,a_N}\Big]^{1/(N-2)} .$$
Because $a_N$ is random, $c_\times$ is an order-of-magnitude rule: for $c\gg c_\times$ the second plateau
is logarithmic with slope $A(c)$, for $c\ll c_\times$ it is odeco-like ($\propto\sigma^{-(N-2)}$), and in
between the measured slope can overshoot $A(c)$ (the min of a log branch and a faster-growing
power-law branch).

## 5. (c) Random TTN teachers on the student's tree

### 5.1 MPS students

*derived; [checked §7.4] on random MPS4 teachers with $d=2$, $\chi=2$.* Every core carries a site, so
every edge with $r_e<d\min(r_{e-1},r_{e+1})$ has an admissible 2-core route. For generic teachers:

1. **Origin plateau**: $L\approx L_0$, duration $\propto\sigma^{-(N-2)}$ (the only polynomial plateau).
2. **Rank-one plateau**: level $\tfrac12(\lVert T\rVert^2-\lambda^2)$ with $\lambda\le\min_e\sigma_1^e(T)$; a
   strict saddle with duration $A_{e^*}\log(1/\sigma)+O(1)$ (§3.3), ended by a rank increase on
   $e^*=\arg\max_e\sigma_2(M_e)$.
3. **All later plateaus**: strict saddles with duration $\propto\log(1/\sigma)$; as $\sigma\to0$ the order
   and slopes follow the race of §3.4. The first-step ranking (sorting edges by $\sigma_2(M_e)$) gets the
   complete order right only for well-separated rates (5/8 in §7.4), because the rates change at each
   intermediate saddle and the offsets $C_e$ reorder near ties. Ranks thus grow edge by edge in
   logarithmic steps, consistent with the plateaus and abrupt rank growth reported for GD on TT/Tucker
   completion (Milanesi et al. 2021).

**What the teacher's bond spectra predict.** The naive rule "next rank increase on the edge with the
largest $\sigma_{r_e+1}^e(T)$" is an upper-bound proxy (§3.2). On random Gaussian MPS teachers it picks
the first edge about as well as the theory (naive 14/16 and 8/8, theory 13/16 and 8/8 in the two
batches of §7.4; the theory's misses are one near-tie and two teachers whose rank-one plateau is not
resolved at $\sigma=0.05$). It fails when the second Schmidt vectors are entangled far from the edge
(§4.3). Rates, slopes, Hessian eigenvalues and intermediate plateau levels need $M_e$ or $T_e$, not the
teacher's spectra.

**Odeco/near-odeco teachers** are the exception: exact orthogonality makes the coefficients of all routes
smaller than the Steiner tree vanish, so later plateaus are polynomial again (H2).

### 5.2 Binary and HT students

*derived; `binary4` [checked §7.5]; `binary8` near the rank-one saddle [checked by the verifier, §7.5].*

- At the rank-one saddle internal nodes have no sites and rank-one outside legs, so they have no room.
  The minimal admissible routes are sibling bottom nodes through their parent, $b_0$–$m_0$–$b_1$
  ($k=3$), which raise two bonds at once; for HT the analogue is two sibling leaves plus their parent
  ($k=3$). **Later escapes at the rank-one saddle are polynomial** ($\propto\sigma^{-1}$) even for generic
  teachers.
- The two root edges induce the same bipartition and always grow together, through $m_0$–root–$m_1$
  ($k=3$). That route is admissible only after both halves have rank $>1$ below them; otherwise it is a
  $k=5$ route $b$–$m_0$–root–$m_1$–$b'$ ($\propto\sigma^{-3}$). **Prediction: bottom-up order.** The root
  bipartition's rank grows only after bonds on both sides have grown, as $\sigma\to0$.
- A 2-core route opens on a bottom edge $(b_0,m_0)$ once $r_{b_1}r_{m_0}>r_{b_0}$. Logarithmic plateaus
  can therefore appear later in binary trees, but never for the root bipartition.

## 6. (d) Teacher on a different tree

### 6.1 Which student bonds must grow

*proven (standard).* Student edge $e$ must reach $\mathrm{rank}\,T_{(e)}$. For a TTN teacher this is at
most the minimum, over cuts $\mathcal C$ of the teacher tree separating $\mathrm{below}(e)$ from
$\mathrm{above}(e)$, of $\prod_{c\in\mathcal C}\dim c$ (cut bonds count their bond dimension, cut physical
legs count $d$). Example: MPS8 teacher with ranks $r$, `binary8` student:

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
$W_{(01|23)}=ARB^\top$: a 3-layer linear network with whitened input on the matrix $T_{(01|23)}$, for
any teacher tree (also `theory_saddles.md` §4.2). Hence (Saxe et al. 2014; Jacot et al. 2021; Achour et
al. 2024 for the saddle types):

- the saddles are nested SVD truncations of $T_{(01|23)}$ (HSVD is exact here, §2.2);
- plateau levels are $L_j=\tfrac12\sum_{i>j}\sigma_i^2(T_{(01|23)})$;
- every escape has degree 3: $\Delta t_j\propto\sigma^{-1}$;
- the number of plateaus equals $\mathrm{rank}\,T_{(01|23)}$. For an MPS teacher this is its middle bond
  rank; the teacher's other bonds are invisible.

### 6.3 The same teacher on MPS4 vs binary4

*derived, [checked §7.5].* An MPS4 student on a generic teacher has a $\sigma^{-2}$ origin plateau and
then $\log(1/\sigma)$ plateaus, starting with the rank-one plateau at $\tfrac12(\lVert T\rVert^2-\lambda^2)$.
A binary4 student has $\sigma^{-1}$ plateaus at the SVD levels of $T_{(01|23)}$ (after its own
$\sigma^{-1}$ origin plateau, $K=3$). **The same teacher has a different plateau structure, different
plateau levels and different escape exponents on the two trees.** For an MPS teacher on a `binary8`
student, the correlations crossing the halves (the teacher's middle bond) are learned last, through a
$k\ge3$ route; an MPS student learns them through a 2-core route on its middle edge.

When some $\kappa_e$ is close to 1 the rank-one plateau of the MPS student is very short at accessible
$\sigma$ (it lasts $\approx(1-\kappa_e)\log(1/\sigma)/\gamma_e$), and the first visible plateau after the
origin is an intermediate saddle (§7.5).

## 7. Numerical checks

All checks use only `ttn.tree`, `ttn.network`, `ttn.teachers` and `ttn.losses`, in float64. GF is
scipy DOP853 (rtol $10^{-10}$; atol $10^{-13}$–$10^{-14}$, and $10^{-12}\sigma$ during the origin plateau)
on torch-autograd gradients of `population_loss`; dense ALS and HSVD are in numpy. Every run took
under 5 min of CPU.

**Reproducibility.** The original checks (scripts in `.../theory2/teachers/`: `check_a.py`,
`check_a_pert.py`, `check_b_landscape.py`, `mps_saddle.py`, `slope_test.py`, `check_b_cross3.py`,
`check_c.py`, `check_c2.py`, `check_d.py`, `check_d2.py`) ran on the library at commit 929065d, before
the teacher-stream fix 57d5262. The pre-fix `random_ttn_teacher(spec, seed=s)` equals
`spec.init(1.0, seed=s)` rescaled to $\lVert W^*\rVert=1$; that expression reproduces §7.1, §7.4 and §7.5
with the current library (the verifier reproduced §7.1 exactly this way). With the current library the
same seeds give different teachers: e.g. the tseed-7 teacher of §7.5 now has
$\sigma_2(T_{(01|23)})=3.2\times10^{-3}$, so its `binary4` plateau sits at $5.0\times10^{-6}$. In §7.4–7.5
teacher and init seeds were distinct (teacher $1000+ts$ vs init 0; teacher 7 vs init 0, 1), so those GF
runs are unbiased. The original two-component teacher `cp2_teacher(seed=0)` drew its site bases from
the same stream as init seed 0, so for init seed 0 the student's core-0 column 0 was exactly parallel
to $u_0$; such runs are marked *biased* below. The revision's runs (`.../theory2/teachers_rev/`) use the
core library at f2f0d6f and the RNG-free teacher $u_i=e_0$, $v_i=c_ie_0+\sqrt{1-c_i^2}\,e_1$, which is
independent of every init.

### 7.1 HSVD vs critical points (part a)

Criticality defect $\delta=\max_v\lVert P_{\mathcal V_v}(T-W)\rVert/\lVert T-W\rVert$; $\delta=0$ iff critical.
Pre-fix teachers (recipe above).

| teacher, tree, ranks | $\delta$(HSVD) | $L/L_0$: HSVD → ALS fixed point | $\lVert W_H-W_c\rVert/\lVert W_c\rVert$ |
|---|---|---|---|
| random MPS4 ($d=3$, TT ranks 3) seed 0, $r=(1,1,1)$ | 0.091 | 0.3510 → 0.3432 | 0.15 |
| same, $r=(2,2,2)$ | 0.031 | 0.02728 → 0.02725 | 0.006 |
| seed 1, $r=(1,1,1)$ | 0.180 | 0.6303 → 0.5646 | 0.58 |
| seed 1, $r=(2,2,2)$ | 0.162 | 0.2636 → 0.2371 | 0.41 |
| random MPS3 ($d=3$), $r=(1,1)$, seeds 0/1/2 | 0.098 / 0.057 / 0.056 | gaps 1.4% / 0.3% / 0.6% | 0.05 / 0.02 / 0.04 |
| random binary8 ($d=2$, $\chi=2$), $r=1$, seeds 0/1 | 0.097 / 0.004 | 0.1096 → 0.1085; 0.19112 → 0.19112 | 0.035 / 0.002 |
| odeco MPS4, same set on all edges, $r=(1,1,1)$ / $(2,2,2)$ | $8\times10^{-16}$ / $4\times10^{-15}$ | equal | $10^{-15}$ |
| random binary4 teacher, binary4 tree, $r=1,2$ | $\le5\times10^{-14}$ | equal | $10^{-15}$ |

HSVD defects of generic teachers range from 0.004 to 0.18 here; they are teacher-specific and do not
depend on $\sigma$. The verifier reproduced every entry, and found that the ALS fixed points equal the
best of 6 random starts. Proposition 1 at critical points: on every edge
$\lVert P_UT_e-W_e\rVert/\lVert W_e\rVert\le2\times10^{-9}$ (ALS, MPS4 seed 0, $r=(2,2,2)$) and $\le2\times10^{-14}$
(verifier, L-BFGS, ranks (1,1,1) and (2,2,2)); the kept values are the top $r_e$ of $T_e$; at HSVD
truncations the condition fails (relative error 0.005–0.24); a rank-1 critical point zero-padded into a
$\chi=3$ student has gradient $6\times10^{-9}$, as in the rank-1 problem.

Proposition 3 ($W$ = odeco rank 2 with weights $(1,0.7)$, $E$ Gaussian with unit norm, ranks 2). The
exponents hold on every draw; the prefactors do not:

| draws of $(W,E)$ | $\lVert W_H-W_c\rVert/\varepsilon^2$ | $(L_H-L_c)/\varepsilon^4$ | $(L_H-L_c)/(\tfrac12\lVert W_H-W_c\rVert^2)$ |
|---|---|---|---|
| original (1 draw per tree), $\varepsilon=0.2\to0.025$ | mps4 0.169 → 0.160; binary8 0.161 → 0.158 | 0.013–0.014 | — |
| verifier, 8 draws on mps4 and binary8, $\varepsilon=0.1\to0.025$ | 0.131–0.181 | 0.0085–0.0164 | within 3% of 1 |
| revision, 6 new draws (`p2_ratio.py`), $\varepsilon=0.2\to0.05$ | 0.088–0.162 | 0.0038–0.0132 | 0.979–1.007 at $\varepsilon=0.2$; 0.995–1.004 at 0.05 |

The ratio's deviation from 1 halves with $\varepsilon$, as the $O(\varepsilon)$ correction predicts.

### 7.2 Rank-one landscape and the rank-one plateau (part b)

- $s_1=s_2$, $d=2$, 60-start multistart ALS: the number of maxima at $c=c^*\mp0.03$ is 2 → 1 for $N=3$
  ($c^*=1/3$), $N=4$ (1/2) and $N=6$ (2/3); no asymmetric maxima. The verifier repeated this in $d=3$
  (out-of-plane starts allowed) at $c^*\mp0.01$: 2 maxima below, 1 above, all symmetric.
- $c_{\rm fold}(N,\rho)$ from the symmetric reduction (all 16 entries reproduced by the verifier's
  independent fold solve; full-space multistart confirms the fold). In brackets: the small-$\rho$
  formula of §4.1.

| $N$ | $\rho=1$ | 0.8 | 0.6 | 0.4 | 0.2 |
|---|---|---|---|---|---|
| 3 | 0.3333 | 0.2228 (0.200) | 0.1580 (0.150) | 0.1021 (0.100) | 0.0503 (0.050) |
| 4 | 0.5000 | 0.3953 (0.344) | 0.3257 (0.298) | 0.2566 (0.243) | 0.1763 (0.172) |
| 6 | 0.6667 | 0.5838 (0.506) | 0.5241 (0.471) | 0.4598 (0.425) | 0.3757 (0.358) |
| 8 | 0.7500 | 0.6831 (0.597) | 0.6334 (0.569) | 0.5782 (0.532) | 0.5025 (0.474) |

- GF, MPS4 student ($d=2$, $\chi=2$), $s=(1,0.6)$, $c=0.5$, $\sigma=0.0125$, init seeds 0 (biased), 1, 2:
  plateau loss 0.16630 / 0.16625 / 0.16629 vs predicted 0.16631; $1-\cos$ between the plateau's site
  vectors and the ALS rank-one critical point $10^{-6}$–$10^{-7}$; Hessian minimum eigenvalue
  −0.1475 / −0.1473 / −0.1479 vs $-\gamma=-0.1474$. The verifier (3 uncorrelated seeds,
  $\sigma=0.05\to0.00625$) found three eigenvalues converging to −0.1474 and the plateau converging to
  0.16631.
- $c=0.3$: plateau 0.17942 (all seeds) vs 0.17942; three negative Hessian eigenvalues
  (−0.0553, −0.0547, −0.0537), each localized on the two cores of one bond, vs $-\gamma=-0.0548$.
- MPS3, $c=0.3$, $\sigma=0.00625$: plateau 0.17374 vs 0.17381; Hessian −0.1993 vs $-\gamma=-0.2004$.
- Exact-saddle Hessians (verifier): MPS4, $d=3$, generic teacher: $\chi=2$ gives 6 and $\chi=3$ gives 12
  negative eigenvalues, equal to $-\lambda^{1/2}\sigma_j(M_e)$, $j=2,3$, with multiplicity $\chi-1$, to 4
  digits. MPS8, 3 random teachers: 7 negative eigenvalues equal to $-\lambda^{3/4}\sigma_2(M_e)$. `binary8`
  and HT8 at their rank-one saddles: no negative eigenvalue (smallest $\approx-10^{-15}$): higher-order.
- The HSVD rank-one truncation is not the plateau. HSVD loss (defect): 0.19822 (0.20) at $c=0.3$,
  0.18100 (0.15) at $c=0.5$, 0.18646 (0.16) for $c=(0.2,0.5,0.7,0.3)$. Truncated HOSVD (verifier):
  0.21049, 0.18652, 0.20349. GF plateaus: 0.17942, 0.16631, 0.17577.

### 7.3 Second and later escapes (parts b, c)

**Second-escape slope.** Event $\max_es_2^e=10^{-2}$ (all $\sigma_2^e(T)\approx0.5$ here, so this is
$f\approx0.02$); MPS4, $s=(1,0.6)$, symmetric $c$, $\sigma=0.05\to0.00625$, slope from the last pair:

| $c$ | predicted $A$ | measured (init seeds 0 biased / 1 / 2) | verifier (3 uncorrelated seeds) |
|---|---|---|---|
| 0.5 | 5.853 | 5.891 / 5.852 / 5.870 | 5.835–5.879 |
| 0.3 | 17.255 | 17.174 / 17.275 / 17.184 | — |

The verifier also measured the rank-one plateau itself (MPS4, $c=0.5$): it lasts 14.3 / 18.4 / 22.4 /
26.5 at $\sigma$ = 0.05 / 0.025 / 0.0125 / 0.00625 (increments $4.1\approx A\ln2$), while the origin
plateau lasts $t_1$ = 459 / 1837 / 7348 / 29392 ($\propto\sigma^{-2}$). MPS3, $c=0.3$ ($A=4.018$), 50% event,
init seed 0 (biased): slopes 4.00 / 4.04 / 4.04 per unit of $\ln(1/\sigma)$.

**Phase-1 growth.** $d\log s_2^e(t_1+10)/d\log\sigma$: predicted $2(1-\kappa)$ = 1.891 ($c=0.3$) and 1.726
($c=0.5$); measured 1.81–1.85 and 1.61–1.78 on the leading edges (verifier: 1.69–1.76 at $c=0.5$).
Growth rate of $s_2^{e^*}$ on the plateau: 0.292–0.298 vs $2\gamma=0.2949$ ($c=0.5$, verifier).

**Race, mixed overlaps** (`race_tiny.py`, `race_analyze.py`). $c=(0.2,0.5,0.7,0.3)$, $s=(1,0.6)$, MPS4,
$d=\chi=2$, RNG-free teacher, init seeds 0–7, $\sigma\in\{10^{-2},10^{-3},10^{-4},10^{-6},10^{-8}\}$,
events with $f=0.1$ (slopes with $f=0.03$ agree within 0.1). At $\sigma=10^{-8}$ the origin plateau lasts
$t_1\approx1.3\times10^{16}$, so the clock is restarted once the escape starts. Predictions from the race
with the exact rates of §4.3: edge 0 at 6.823, edge 2 at 12.178, edge 1 at 14.467 (order $0\to2\to1$); if
edge 1 goes second, 16.164.

| quantity | prediction | measured, 8 seeds |
|---|---|---|
| order of the bond-rank increases | $0\to2\to1$ as $\sigma\to0$ | $\sigma=10^{-2}$: 6/8; $10^{-3}$ to $10^{-6}$: 7/8; $10^{-8}$: 8/8. The others go $0\to1\to2$: seeds 1 and 3 at $10^{-2}$, seed 3 down to $10^{-6}$ |
| edge 0 slope, $\sigma=10^{-2}\to10^{-3}$ / $10^{-3}\to10^{-4}$ / $10^{-6}\to10^{-8}$ | 6.823 | 6.84–7.81 / 6.69–6.84 / 6.818–6.823 |
| edge 2 slope, $10^{-6}\to10^{-8}$ | 12.178 | 12.178–12.182 (7 seeds); 12.63 (seed 3, order changes in this interval) |
| edge 1 slope, $10^{-6}\to10^{-8}$ | 14.467 | 14.40–14.52, mean 14.469 (7 seeds); 15.02 (seed 3) |
| edge 1 slope when it goes second (seed 3, $10^{-2}\to10^{-3}\to10^{-4}$) | 16.164 | 16.36, 16.30 |
| offset $t_e-t_1-\tau_e\ln(1/\sigma)$ at $\sigma=10^{-8}$, edges 0 / 1 / 2 | — | −24 to 23 / −24 to 33 / −19 to 51 |

With $f=0.03$ the last-pair slopes are 6.818–6.823, 14.40–14.57 and 12.178–12.182. Seed 3 shows the
finite-$\sigma$ mechanism directly: its edge-2 offset is about 50 (an unlucky initial projection), so
edge 1, which escapes at slope 16.16 if it goes second, beats edge 2 (slope 12.18) until
$\ln(1/\sigma)(16.16-12.18)$ exceeds about 56, i.e. for $\sigma\gtrsim10^{-6}$; at $\sigma=10^{-8}$ the
asymptotic order is restored, as observed.

- Plateau levels at $\sigma=10^{-8}$: 0.175767, 0.166436 and 0.15983–0.16004 (exact 0.175767, 0.166436,
  0.159878).
- (A-carry) at the $(1,1,1)\to(2,1,1)$ transition (`carry.py`): the edge-1 unstable eigenvector at the
  exact $(2,1,1)$ saddle puts weight 0.316 on the fresh core-1 entries (new bond-0 index), 0.184 on the
  old core-1 entries and 0.500 on core 2, whose part is identical ($\lvert\cos\rvert=1.000$) to the
  rank-one eigenvector's. Overlap of the two unit eigenvectors: $p=0.775$, i.e. the head start carries
  over at the cost of $\ln(1/p)/0.077=3.3$ time units.
- The verifier's runs ($\sigma=0.05\to0.003125$, init seeds 0 and 1, same teacher) are consistent with
  this: seed 1 has order $0\to1\to2$ down to $\sigma=0.00625$, and seed 0's edge-1 local slopes there
  (20.0, 21.6, 27.2, 19.0) are finite-$\sigma$ wiggles; seed 0 settles to 14.1–14.4 for $\sigma\le10^{-4}$.
  The original race check (one biased init, rates read off GF Hessians: 0.0987, 0.0760, 0.1119, which
  are 2–17% off the exact values) is superseded.

**Odeco limit and crossover** (MPS3, $s=(1,0.6)$, symmetric $c$; slope =
$\Delta(\Delta t_2)/\Delta\ln(1/\sigma)$ between successive $\sigma$; original runs with init seed 0,
biased, except the last two lines):

| $c$ | predicted $A(c)$ | $\Delta t_2$ at $\sigma$ = 0.05 / 0.0125 / 0.003125 | slopes | regime |
|---|---|---|---|---|
| 0 | $\infty$ | 374.5 / 1483.2 / 5920.2 ($\sigma\Delta t_2$ = 18.72 / 18.54 / 18.50) | — | odeco, $\propto\sigma^{-1}$ |
| 0.01 | 164.7 | 157.7 / 328.8 / 548.4 | 123.5, 158.4 | crossing over |
| 0.03 | 53.6 | 83.1 / 153.0 / 227.9 | 50.4, 54.0 | logarithmic |
| 0.1 | 14.78 | 24.95 / 45.51 / 66.03 | 14.83, 14.81 | logarithmic |
| 0.3 | 4.02 | 5.93 / 11.52 / 17.06 | 4.03, 4.00 | logarithmic |
| 0, verifier (uncorrelated seed) | $\infty$ | $\sigma\Delta t_2$ = 4.39 / 4.60 / 4.64 | — | odeco |
| 0.01, verifier | 164.7 | — | 159, 220 | crossing over |

Distribution of $a_3=\sigma\Delta t_2$ at $c=0$ over uncorrelated init seeds (`p8_a3.py`,
$\sigma=0.0125$): 1.07, 1.14, 2.30, 2.98, 3.71, 4.64, 5.65, 7.76 (median 3.35; $f=0.03$ changes them by
less than 0.03). The biased init's 18.5 lies 2.4 times above the largest of these, so the original
estimate of $c_\times$ (0.014 at $\sigma=0.05$, 0.0016 at $\sigma=0.003$) was about 5 times too small for a
typical init ($c_\times\propto1/a_3$ for $N=3$).

### 7.4 Random MPS teachers on an MPS4 student (part c)

$d=2$, $\chi=2$, pre-fix `random_ttn_teacher` (Gaussian cores), init seed 0 (independent).

- **Batch A** ($\sigma=0.05$, 16 teachers with a plateau; one more had none, and three runs hit the time
  limit). First edge: theory 13/16, naive (largest teacher $\sigma_2^e$) 14/16. The three theory misses:
  one near-tie ($\sigma_2(M_e)=0.210$ vs 0.205), and two teachers with $\kappa_{e^*}\approx0.55$–$0.7$ whose
  rank-one plateau is too short to resolve at $\sigma=0.05$ (plateau/prediction 1.01 and 0.76).
- **Batch B** (second bond slices scaled by 0.4, $\sigma=0.02$, 8 teachers with a plateau). First edge:
  theory 8/8, naive 8/8; full order matches the first-step ranking 5/8. Where the plateau is formed
  (plateau/prediction > 0.98): Hessian minimum $=-\gamma_{\max}(1\pm0.015)$ and growth rate of $s_2^{e^*}$
  $=2\gamma_{e^*}\times(1.00,1.03,1.135)$; where it is not (0.77–0.83): deviations of 20–26%.

### 7.5 Same teacher, two student trees (part d)

Pre-fix teacher tseed 7, $d=2$, $\chi=2$; students `binary4` and `mps4` ($\chi=2$), init seeds 0, 1.

- **MPS4 teacher → binary4.** Plateau 0.13109–0.13112 vs $\tfrac12\sigma_2^2(T_{(01|23)})=0.13112$.
  $\Delta t_2$ = 68.0 / 56.3 ($\sigma=0.1$), 144.8 / 113.9 (0.05), 298.3 / 229.0 (0.025): ratio per halving
  2.01–2.13, i.e. $\propto\sigma^{-1}$. Verifier, current-library tseed-7 teacher: plateau
  $5.0132\times10^{-6}$ = $\tfrac12\sigma_2^2$ exactly, $\Delta t_2$ ratios 2.38 and 2.13 per halving,
  $\lambda_{\min}$ = $-3.4\times10^{-4}$, $-1.9\times10^{-4}$, $-9.6\times10^{-5}$ ($\propto\sigma$).
- **binary4 teacher → binary4.** Plateau 0.03835–0.03836 vs 0.03836; $\Delta t_2$ = 183.8 / 153.2,
  370.9 / 309.2, 745.3 / 621.7 at $\sigma$ = 0.1, 0.05, 0.025: ratio 2.02 per halving.
- **MPS4 teacher → MPS4.** Theory: $\lambda=0.8416$, $L_1=0.14582$, $\sigma_2(M_e)=(0.1665,0.2777,0.045)$
  (teacher $\sigma_2^e=(0.185,0.512,0.415)$), $A_e=(5.25,2.63,22.9)$. Seed 0: first edge 1 at all four
  $\sigma$; plateau 0.14560–0.14572; slopes 2.71, 2.64, 2.65 vs $A_1=2.63$. Seed 1: slopes 0.89 → 1.21 →
  1.70, still converging (short rank-one plateau, $\kappa_1=0.33$).
- **binary4 teacher → MPS4.** Theory: $\lambda=0.6993$, $L_1=0.2555$, $\kappa_e=(0.55,0.06,0.81)$,
  $A_e=(1.40,28.1,0.41)$. The rank-one plateau merges with the second escape (the $s_2$ threshold is
  crossed 2.6–3.3 time units before $s_1$ reaches $\lambda/2$); last-pair slopes 0.40 / 0.37 vs
  $A_2=0.41$; first edge 2, as predicted. The first visible plateau after the origin is an intermediate
  saddle at $L\approx0.0859$ with logarithmic duration (17.8, 19.9, 21.9 at $\sigma$ = 0.1, 0.05, 0.025),
  against 184, 371, 745 for the binary4 student on the same teacher.
- **Hessian at the rank-one plateau, same MPS4 teacher.** binary4: $\lambda_{\min}=-0.0171$ ($\sigma=0.05$)
  and −0.0037 ($\sigma=0.0125$), vanishing $\propto\sigma^{1.1}$ (higher-order saddle). MPS4: lowest three
  eigenvalues $(-0.2519,-0.1522,-0.0456)$ at $\sigma=0.0125$ and $(-0.2472,-0.1514,-0.0546)$ at 0.05, vs
  $-\gamma_e=(-0.2548,-0.1527,-0.0413)$ (strict).
- **`binary8` near the rank-one saddle** (verifier; GF started at the exact rank-one saddle plus noise of
  size $\varepsilon$, as a cheap stand-in for small init; 3 teachers): the bottom pairs rise together, the
  root edges rise last, and every event time scales $\propto1/\varepsilon$ (ratio 2.06–2.27 per halving):
  polynomial, bottom-up growth.

### 7.6 Criticality of plateaus (P1, R11)

Verifier's measurement (MPS4, $c=0.5$): criticality defect at the minimum-speed point of the rank-one
plateau, after truncating $W$ to rank one:

| init seed | $\sigma$ = 0.05 | 0.025 | 0.0125 | 0.00625 | local exponent |
|---|---|---|---|---|---|
| 0 | $2.4\times10^{-3}$ | $1.3\times10^{-3}$ | $7.3\times10^{-4}$ | $4.1\times10^{-4}$ | 0.89, 0.83, 0.83 |
| 1 | — | $8.3\times10^{-3}$ | $4.5\times10^{-3}$ | $2.5\times10^{-3}$ | 0.88, 0.85 |
| 2 | $3.4\times10^{-3}$ | — | — | $6.4\times10^{-4}$ | 0.80 (mean) |

Prediction (`p1_alpha.py`, crossing argument of §9 P1): radial mode $\alpha_r=0.834$, slowest angular mode
($h=-0.618$) $\alpha_a=0.811$, so $\alpha=0.81$; measured 0.80–0.89. The HSVD defect at the same ranks is
0.15 for every $\sigma$.

## 8. Caveats and open problems

- **Asymptotics.** Rates, slopes and orders are $\sigma\to0$ statements. At finite $\sigma$ every event
  time carries an init-dependent offset $C_e$ (the logarithm of the route's initial projection, plus the
  $O(1)$ carry-over factors, divided by a rate); measured offsets reach about 50 time units (§7.3).
  Near-tied later escapes therefore come in a seed-dependent order at accessible $\sigma$, and plateaus
  with $\kappa_e$ close to 1 are not resolved (the rank-one plateau lasts $\approx(1-\kappa)\ln(1/\sigma)/\gamma$).
- **Events.** Use thresholds proportional to the target, $s_2^e\ge f\sigma_2^e(T)$ with $f\le0.1$, and
  check $f$-independence; the 50%-of-final event of `notes/diagnostics.md` can skip intermediate saddles.
- **Whitened inputs only.** Under a product covariance the landscape statements transfer with
  $T\to\Sigma^{1/2}T$, the dynamical ones do not (`theory_covariance.md` C1, C4).
- **Not covered.** Large $d$ or $\chi$ (many routes per edge; the rates are the singular values of $B_e$
  with multiplicities $\chi_e-r_e$); GF from small init on `binary8`/HT8 (origin plateau
  $\propto\sigma^{-(K-2)}$ with $K=7$ and 15; checked only from the rank-one saddle plus noise); teachers
  whose rank-one landscape has many local maxima.
- **Conjectures.** (i) Local maxima of $g$ on the sphere have bond rank one. (ii) The KKT point reached
  from a random init is a local maximum. (iii) For odeco teachers the minimizer on $\mathcal M_{\le r}$
  keeps the top $\min_er_e$ components. (iv) GF takes the minimal-$k$ route (supported by every run).
  (v) (A-carry) holds for generic teachers on any tree (checked on one MPS4 teacher: $p=0.775$ at one
  transition, and indirectly through the slopes of §7.3).
- **Open.** The distribution of the offsets $C_e$ over inits, which would give the probability of each
  order at a given $\sigma$; the defect exponent at intermediate plateaus beyond the bound of P1; the
  distribution of the odeco prefactor $a_N$, hence of $c_\times$.

## References

Abbe, Boix-Adserà & Misiakiewicz (2023) COLT, arXiv:2302.11055 (leap complexity) · Achour, Malgouyres
& Gerchinovitz (2024) JMLR, arXiv:2107.13289 · Anandkumar et al. (2014) JMLR 15 · Baldi & Hornik (1989)
*Neural Networks* 2 · De Lathauwer, De Moor & Vandewalle (2000) SIMAX 21 (HOSVD; best
rank-$(R_1,\dots,R_N)$, HOOI) · Eldén & Savas (2011) SIMAX 32(4):1422–1450 (optimality conditions and
perturbation theory for the best multilinear-rank approximation) · Frandsen & Ge (2022) *Math.
Program.* 193:687–712 (Tucker landscape) · Furman, Wäldchen, Bei & Hodgkinson (2026) arXiv:2609.13057
(this model's landscape: Thm 5.2, Lemmas D.7–D.9, Remark D.10) · Ge, Ren, Wang & Zhou (2021) NeurIPS ·
Grasedyck (2010) SIMAX 31(4) (HSVD) · Holtz, Rohwedder & Schneider (2012a) *Numer. Math.*
120(4):701–731 (manifolds of fixed TT rank) · Holtz, Rohwedder & Schneider (2012b) SISC
34(2):A683–A713 (ALS in TT) · Jacot et al. (2021) arXiv:2106.15933 · Kolda (2001) SIMAX 23 · Kumar &
Haupt (2024 TMLR, arXiv:2402.09226; 2025 TMLR) · Kunin et al. (2025) NeurIPS, arXiv:2506.06489
(Alternating Gradient Flows, accumulated utility) · Li, Luo & Lyu (2021) ICLR (greedy low-rank
learning) · Lim (2005) CAMSAP (tensor singular vectors) · Milanesi, Kadri, Ayache & Artières (2021)
IJCNN, arXiv:2105.01346 (plateaus and abrupt rank growth in GD on TT/Tucker) · Oseledets (2011) SISC 33
(TT-SVD) · Pesme & Flammarion (2023) NeurIPS · Razin, Maman & Cohen (2022) ICML · Saxe, McClelland &
Ganguli (2014) ICLR · Stegeman & Comon (2010) LAA 433 · Uschmajew & Vandereycken (2013) LAA 439:133–166
(geometry of hierarchical tensors).

## 9. Predictions for experiments

**Protocol.** Whitened inputs; fan-in init with per-core scale $\sigma$; GF (or GD with small steps);
float64. $t_1$ = time at which $L$ is halfway between $L_0=\tfrac12\lVert T\rVert^2$ and the rank-one level
$L_1$. Events $t_e$ = first time $s_2^e\ge f\,\sigma_2^e(T)$ with $f=0.1$ (repeat with $f=0.03$; slopes
must agree). "Slope" = $d(t_e-t_1)/d\ln(1/\sigma)$ from at least three values of $\sigma$ spaced by factors
$\ge2$. $\lambda$, $w_i$ and $M_e=T\times_{i\notin e}w_i$ from rank-one ALS (HOPM) started at the plateau's
site vectors; intermediate critical points from ALS at the current ranks started at the plateau state;
rates at critical points from the Hessian of the balanced, zero-padded representation (`carry.py` or
the verifier's `c_rates.py`). Reference teacher "mixed": MPS4, $d=\chi=2$, $s=(1,0.6)$,
$u_i=e_0$, $v_i=c_ie_0+\sqrt{1-c_i^2}e_1$, $c=(0.2,0.5,0.7,0.3)$; "sym($c$)": same with all $c_i=c$.

1. **Plateaus are critical points (P1).** At the minimum-speed point of the rank-one plateau, truncate
   $W$ to rank one (HSVD of the GF state); its criticality defect
   $\delta=\max_v\lVert P_{\mathcal V_v}(T-W)\rVert/\lVert T-W\rVert$ falls as $\sigma^\alpha$ with
   $\alpha=\min(\alpha_r,\alpha_a)$, where
   $\alpha_r=(1-\kappa_{e^*})\mu_r/(\mu_r+\gamma_{e^*})$, $\mu_r=K\lambda^{2(K-1)/K}$, and
   $\alpha_a=\lvert h_a\rvert/(\lvert h_a\rvert+\sigma_2(M_{e^*}))$ over the eigenvalues $h_a$ of the Riemannian
   Hessian of the rank-one problem with $\lvert h_a\rvert<\lambda-\sigma_2(M_{e^*})$ (*heuristic*: the
   minimum-speed point is where the slowest decaying stable mode meets the growing route).
   Values: $\alpha=0.81$ for sym(0.5) (measured 0.80–0.89), 0.93 for sym(0.3), 0.84 for mixed. The HSVD
   truncation at the same ranks has a $\sigma$-independent defect (0.15, 0.20, 0.16). At an
   intermediate plateau $\delta$ falls more slowly, with $\alpha$ at most the next route's entry deficit
   $-\ell$ (mixed, $(2,1,1)$ plateau: $\alpha\le0.54$).
2. **Near-consistent teachers (P2; static).** $T=W+\varepsilon E$ with $W$ consistent at ranks $r$:
   $\lVert W_H-W_c\rVert\propto\varepsilon^2$ and $L_H-L_c\propto\varepsilon^4$, with draw-dependent prefactors
   (0.09–0.18 and 0.004–0.017 over 14 draws of odeco $W$ with weights (1, 0.7) plus a unit Gaussian $E$,
   ranks 2, on `mps4` with $d=3$ and `binary8` with $d=2$), and the draw-independent
   $(L_H-L_c)/(\tfrac12\lVert W_H-W_c\rVert^2)=1+O(\varepsilon)$ (within 3% for $\varepsilon\le0.2$, 0.5% at
   $\varepsilon=0.05$).
3. **Rank-one plateau level (P3).** MPS/HT students: $L\to\tfrac12(\lVert T\rVert^2-\lambda^2)$ with
   $\lambda$ a local maximum of the rank-one problem: 0.16631 for sym(0.5), 0.17942 for sym(0.3) (minor
   basin: 0.4975), 0.175767 for mixed, 0.17381 for MPS3 sym(0.3). Equal weights and overlaps: a second
   (minor-basin) level exists iff $c<(N-2)/N$; for $\rho=s_2/s_1<1$ iff $c<c_{\rm fold}(N,\rho)$ (§7.2;
   $N=4$: 0.3953, 0.3257, 0.2566, 0.1763 at $\rho$ = 0.8, 0.6, 0.4, 0.2). The level is not the HSVD loss
   (0.18100 for sym(0.5), 0.19822 for sym(0.3)).
4. **Hessian at the rank-one saddle (P4).** MPS student at the exact balanced, zero-padded rank-one
   critical point: negative eigenvalues $-\lambda^{(N-2)/N}\sigma_j(M_e)$, $j=2..d$, each with
   multiplicity $\chi_e-1$, for every edge: $(N-1)(d-1)(\chi-1)$ of them for a generic teacher. At GF
   plateau points $\lambda_{\min}\to-\gamma_{e^*}$, independent of $\sigma$: −0.1474 (sym 0.5), −0.0548
   (sym 0.3), −0.1283 (mixed). `binary`/HT student on the same teacher: no negative eigenvalue at the
   exact saddle, and $\lambda_{\min}\propto\sigma$ at plateau points (halving $\sigma$ halves it).
5. **Growth on the rank-one plateau (P5).** $d\ln s_2^e/dt\to2\gamma_e$ on every edge whose stable
   component has decayed: 0.2949 (sym 0.5), 0.1096 (sym 0.3), 0.2566 / 0.0720 / 0.1183 (mixed, edges
   0/1/2); and
   $s_2^e(t_1+10)\propto\sigma^{2(1-\kappa_e)}$ (exponent 1.726 for sym(0.5), 1.891 for sym(0.3)).
6. **Second escape (P6).** $t_{e^*}-t_1=A_{e^*}\ln(1/\sigma)+C$ with $A_e=(1-\kappa_e)/\gamma_e$: 5.853
   (sym 0.5), 17.255 (sym 0.3), 6.823 (mixed, $e^*=0$), 4.018 (MPS3 sym 0.3). Measured slopes are within
   1% for $\sigma\le0.0125$ (sym) and within 2% for $\sigma\le10^{-3}$ (mixed; between $10^{-2}$ and
   $10^{-3}$ two of eight seeds are 12–14% high, their rank-one plateau being barely formed).
   $e^*=\arg\max_e\sigma_2(M_e)$, not $\arg\max_e\sigma_2^e(T)$: for mixed,
   $\sigma_2(M_e)=(0.127,0.036,0.059)$ while $\sigma_2^e(T)=(0.567,0.566,0.548)$.
7. **Later escapes on an MPS (P7).** Intermediate plateaus at the critical values of the mixed-rank
   problems: mixed, $(1,1,1)$ 0.175767 → $(2,1,1)$ 0.166436 → $(2,1,2)$ 0.159878 (or $(2,2,1)$ 0.150277
   if edge 1 precedes edge 2) → 0. As $\sigma\to0$: order $0\to2\to1$ with slopes 6.823 / 12.178 /
   14.467 (7 of 8 seeds within 0.5% between $\sigma=10^{-6}$ and $10^{-8}$; the eighth changed order in
   that interval). At accessible $\sigma$ the order of the near-tied edges 1 and 2 is
   seed-dependent (reversed in 2/8 seeds at $\sigma=10^{-2}$ and in 1/8 down to $\sigma=10^{-6}$); given the
   realized order the slopes follow the race for that order (edge 1 second: 16.16). Offsets
   $t_e-t_1-\tau_e\ln(1/\sigma)$ are init-dependent, up to about 50 time units.
8. **Odeco crossover (P8).** MPS3, $s=(1,0.6)$, sym($c$): the second plateau is logarithmic with slope
   $A(c)$ = 164.7, 53.6, 14.78, 4.02 at $c$ = 0.01, 0.03, 0.1, 0.3 when $c\gg c_\times$, and odeco-like,
   $\Delta t_2=a_3/\sigma$, when $c\ll c_\times$, with $a_3$ random over inits (1.07–7.76, median 3.35, over 8
   seeds). Hence $c_\times\approx\sigma\ln(1/\sigma)/(s_2a_3)$ is an order-of-magnitude rule: at
   $\sigma=0.0125$ it ranges over 0.012–0.085 (median 0.028), so $c=0.03$ sits near the crossover for a
   typical init and only $c\gtrsim0.1$ is safely logarithmic. Near $c_\times$ the local slope can overshoot
   $A(c)$.
9. **Tree dependence (P9).** Generic teacher: on an MPS student every plateau after the origin plateau
   lasts $\propto\ln(1/\sigma)$ and has $\sigma$-independent $\lambda_{\min}<0$; on `binary`/HT students
   the escapes from the rank-one saddle last $\propto\sigma^{-1}$ or longer, with $\lambda_{\min}\propto\sigma$,
   and the root bipartition always escapes polynomially.
10. **`binary4` (P10).** For any teacher the plateaus sit exactly at
    $\tfrac12\sum_{i>j}\sigma_i^2(T_{(01|23)})$, gaps scale $\propto\sigma^{-1}$ (ratio 2 per halving), and
    there are $\mathrm{rank}\,T_{(01|23)}$ events (the middle bond rank for an MPS teacher).
11. **Bottom-up order on `binary8` (P11).** The root bond's rank rises only after bottom bonds on both
    halves have risen; started from the rank-one saddle with noise $\varepsilon$, event times scale
    $\propto1/\varepsilon$. For an MPS8 teacher of rank $r$, student bonds $\{23\}$ and $\{45\}$ need
    $\min(d^2,r^2)$ rank increments.
12. **Event definitions (P12).** Slopes from $f=0.03$ and $f=0.1$ agree to about 1%; the 50%-of-final
    event can skip intermediate saddles (sym(0.3): slope about 19.7 instead of 17.3 in 2 of 3 seeds).
