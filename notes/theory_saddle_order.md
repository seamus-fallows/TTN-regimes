# Theory: the saddle order at a learned rank-one saddle (lower bound k*, and when it fails)

Status: **v2, 2026-10-10** (theorist), revised after two independent verifications of v1:
proofs `notes/reviews/saddle_order_verification_proof.md` (Theorems 1 and 2 proven; the exact
orders $2\ell^*$ needed a level version, now Theorem 1', section 3.5; several fixes, all applied) and
numerics `notes/reviews/saddle_order_verification_numerics.md` (orders and constants confirmed; caveats
on the dynamics, now in section 6). Answers the open issue of `theory_saddles.md` section 11
("saddle order = $k^*$, lower bound") for one learned component.
Tags: **proven**, **derived**, **empirical**, **conjecture**, as in `notes/theory_saddles.md`.
Numerics: `experiments/theory_checks/saddle_order/` (README + scripts); the verifier's independent
code: `experiments/theory_checks/saddle_order_verify/`.

## 0. Summary

**Answer in one paragraph.** The claim "saddle order $=k^*$" (R2 of `theory_saddles.md`) is
**false in general** but true under a checkable spectral condition, and **proven** here in that
form. Whether the order equals $k^*$ is decided by a *generalized piggyback operator* $\mathcal S$
that couples sector amplitudes on small core sets (single cores, and, for $k^*\ge5$, sets whose
Steiner tree has fewer than $k^*/2$ cores). If $\lambda_{\max}(\mathcal S)<s_q$ then
$L-L_S\ge -C\lVert\theta-\theta_S\rVert^{k^*}$ (Theorem 1), so with the route curve the order is
exactly $k^*$. Level version (Theorem 1'): if $\mathcal S$ restricted to sets of degree $\le\ell-1$
has $\lambda_{\max}<s_q$, the order is $\ge\min(k^*,2\ell)$. Sharpness (Theorem 2; pattern teacher,
$s_r>0$, a spare bond index per cluster = "index condition"): if $\mathcal S$ restricted to sets of
degree $\le\ell<k^*/2$ has $\lambda_{\max}>s_q$, there is a descent curve of order $2\ell<k^*$.
Together: order $=\min(k^*,2\ell^*)$, $\ell^*$ the first level where $\lambda_{\max}$ exceeds $s_q$
(strict inequalities; pattern teacher and index condition for the upper bound). Consequences:
(i) for $k^*\in\{3,4\}$ the condition is the old-block (classical piggyback) condition
$\lambda_{\max}(S_1)<s_q$ (pattern teacher: $\rho(S)<s_q$), so order $=k^*$ is proven for $k^*=3,4$
whenever the old block is positive definite modulo gauge ($k^*=2$: order 2 always);
(ii) for two-component teachers in the correct order ($s_r<s_q$) the order is $k^*$ on every tree;
(iii) counterexamples exist from $k^*=5$ on (given a PD old block): MPS6, $q=000000$ (1),
$r_1=110001$, $r_2=110010$ (0.8 each) has $k^*=5$, a PSD Hessian, and saddle order 4 (numerics: slope
4.000, constant $-0.0350$ predicted $-0.0350$; reproduced by both verifiers). It has two residuals,
so it refutes the multi-residual use of "order $=k^*$" (Prop. 1, 3.3, 3.4, 5.1 of
`theory_saddles.md`), not R2's literal one-residual statement. R2 itself is refuted by single
wrong-order residuals: $110001$ (1.5) on MPS6 ($k^*=6$, order 4) and $1111$ (1.5) on HT4 ($k^*=7$,
order 6). Numerics on 32 instances (section 5; trees mps4-6, binary4/6, ht4, $k^*=2..7$, pattern and
general residuals) agree with Theorems 1-2 everywhere. Gradient flow on the pair-star escapes with
$t_{\rm esc}\propto\sigma^{-2}$ instead of the route law $\sigma^{-3}$, toward a non-residual pattern,
and then forgets $q$ (section 6; derived, with the verifier's caveats: from small initialization only
some seeds visit $\theta_S$, and the $\sigma^{-2}$ law appears only at small $\sigma$). Among the
project's experiments Theorem 2 applies only to wrong-order two-component runs whose differing sites
cover two leaf pairs: HT4 with $D$ = all sites (locality) and binary8 odeco from the wrong saddle
(`order` agent), order 6, not 7. The HT4 runs were excluded from the locality fits; 6 of the 7 with
$\chi=2$ forget and relearn the smaller component, as the mechanism predicts. The binary8 runs
($\sigma\ge0.15$) follow the route law, as expected at or above the estimated crossover (section 7, item 5).

## 1. Setting and definitions

- Tree with cores $V$ ($\lvert V\rvert=K$), bond dimensions $\chi_e\ge1$ (no other assumption), whitened
  loss $L(\theta)=\tfrac12\lVert W(\theta)-W^*\rVert^2$. Core $v$ carries the physical space
  $\mathbb R^{D_v}$ ($D_v$ = product of its site dimensions).
- Learned component: unit vectors $q_v\in\mathbb R^{D_v}$, $U_q=\bigotimes_vq_v$,
  $s_q=\langle U_q,W^*\rangle>0$, $R=W^*-s_qU_q$. $\theta_S$: $G_v=a_v\,q_v\otimes e_0\otimes\dots\otimes e_0$
  with $\prod_va_v=s_q$ (balanced: $a_v=s_q^{1/K}$; balancedness is **not** needed below).
  Entries with all bond indices $0$ are **old**, the others **new**; $\nu_x$ = vector of new entries of core $x$.
- **Sectors.** $P_v=q_vq_v^\top$, $P_v^\perp=I-P_v$. For $B\subseteq V$,
  $\Pi_B=\bigotimes_{v\in B}P_v^\perp\otimes\bigotimes_{v\notin B}P_v$. The $2^K$ projectors are
  orthogonal and sum to the identity. $T(B)=\mathrm{range}(\Pi_B)\cong\bigotimes_{v\in B}q_v^\perp$,
  and $T(B_1)\otimes T(B_2)\cong T(B_1\sqcup B_2)$ for disjoint sets. $R_B:=\Pi_BR$; $\Pi_\emptyset R=0$ by definition of $s_q$.
- **Criticality** (proven, one line): new entries do not enter $W$ at first order and an old
  entry of core $v$ moves $W$ inside $T(\emptyset)\oplus T(\{v\})$, so $\nabla L(\theta_S)=0$ iff
  $R_{\{v\}}=0$ for every $v$. Assume this.
- $k^*:=\min\{\lvert\mathrm{Steiner}(B)\rvert:\ R_B\ne0\}$ ($\ge2$). Pattern teacher
  $W^*=s_qU_q+\sum_rs_rU_r$: $R_B=\sum_{r:T_r=B}s_rU_r$ with $T_r=N(D_r)$ the terminal cores, so
  $k^*=\min_r\lvert\mathrm{Steiner}(T_r)\rvert$ as in Prop. 2 of `theory_saddles.md`.
- **Degree** of a nonempty set $C$: $\deg C=\lvert\mathrm{Steiner}(C)\rvert$ (so $\deg C=1$ for a
  singleton). **Low sets**: $\mathcal L=\{C\ne\emptyset:\deg C<k^*/2\}$; $\mathcal L_\ell=\{C\ne\emptyset:\deg C\le\ell\}$
  ($\mathcal L_0=\emptyset$; $\mathcal L=\mathcal L_{\lceil k^*/2\rceil-1}$).
  For $k^*\le2$, $\mathcal L=\emptyset$; for $k^*=3,4$, $\mathcal L$ = singletons; for $k^*=5,6$ also
  pairs of adjacent cores; etc. A set containing a core without sites (HT internal nodes, binary root)
  has $T(C)=0$ and contributes nothing.
- **Generalized piggyback operator** $\mathcal S$ on $\mathcal H=\bigoplus_{C\in\mathcal L}T(C)$:
  block $(C_1,C_2)$ = the bilinear form $(x,y)\mapsto\langle R_{C_1\sqcup C_2},x\otimes y\rangle$
  if $C_1\cap C_2=\emptyset$, zero otherwise (zero diagonal blocks, so $\lambda_{\max}(\mathcal S)\ge0$).
  Restricted to singletons it is the classical piggyback operator $S_1$ of `theory_saddles.md` 4.1
  (blocks $R_{\{u,v\}}$). For a pattern teacher, in the pattern basis $\mathcal S$ is the adjacency
  matrix of the **generalized piggyback graph**: nodes $(C,w)$ ($C\in\mathcal L$, $w$ a pattern on
  $C$ differing from $q$ on every core of $C$), one edge of weight $s_r$ between $(C_1,r|_{C_1})$ and
  $(C_2,r|_{C_2})$ for every residual $r$ and every split $T_r=C_1\sqcup C_2$ with $C_1,C_2\in\mathcal L$;
  if every $s_r>0$, $\lambda_{\max}(\mathcal S)=\rho(\text{graph})$ (Perron-Frobenius; with signed
  weights only $\lambda_{\max}\le\rho(\lvert A\rvert)$). Pattern teachers below have $s_r>0$.
  $\mathcal S\vert_{\mathcal L_\ell}$ is the same construction on $\mathcal L_\ell$, a principal
  compression of the next level, so $\lambda_{\max}(\mathcal S\vert_{\mathcal L_\ell})$ is nondecreasing in $\ell$.
- **Saddle order** $p$: the largest $p$ with $L(\theta)-L_S\ge-C\lVert\theta-\theta_S\rVert^p$ near
  $\theta_S$. $L$ is a polynomial, so $\min_{\lVert\delta\rVert\le\varepsilon}L-L_S$ is semialgebraic in
  $\varepsilon$, the supremum is attained, and by the curve-selection lemma (Milnor 1968, Lemma 3.1)
  equivalently no curve with $\lVert\theta(t)-\theta_S\rVert\asymp t$ descends faster than $t^p$.
  **Gauge.** Measuring the distance to the local gauge orbit $\{g\theta_S\}$ ($g$ near the identity)
  instead of to $\theta_S$ gives the same order: upper bounds transfer because
  $\mathrm{dist}(\theta,\text{orbit})\le\lVert\theta-\theta_S\rVert$; lower bounds because the gauge acts
  linearly on $\theta$ and preserves $L$, so for $\theta'=g\theta_S$,
  $L(\theta)-L_S=L(g^{-1}\theta)-L_S\ge-C\lVert g^{-1}\rVert^p\lVert\theta-\theta'\rVert^p$, uniformly for
  $g$ in a neighbourhood of the identity (proof verifier's argument).

## 2. Results

**Theorem 1 (lower bound; proven).** Assume $\theta_S$ critical and $\lambda_{\max}(\mathcal S)<s_q$.
Then there are $C,r_0>0$ with
$$L(\theta)-L_S\ \ge\ -C\,\lVert\theta-\theta_S\rVert^{k^*}\qquad\text{for }\lVert\theta-\theta_S\rVert<r_0 .$$
If moreover $\chi_e\ge2$ on every edge of $\mathrm{Steiner}(B)$ for some $B$ with $R_B\ne0$ and
$\lvert\mathrm{Steiner}(B)\rvert=k^*$, the route curve gives $L-L_S=-ct^{k^*}+O(t^{2k^*})$
(proven, `theory_saddles.md` section 2; any nonzero $R_B$ has a nonzero overlap with some product
tensor), so the saddle order is exactly $k^*$.

**Theorem 1' (level version; proven, section 3.5).** Assume $\theta_S$ critical, let
$2\le\ell\le\lceil k^*/2\rceil$, and assume $\lambda_{\max}(\mathcal S\vert_{\mathcal L_{\ell-1}})<s_q$. Then
$$L(\theta)-L_S\ \ge\ -C\,\lVert\theta-\theta_S\rVert^{\min(k^*,2\ell)}\qquad\text{for }\lVert\theta-\theta_S\rVert<r_0 .$$
$\ell=\lceil k^*/2\rceil$ is Theorem 1 ($\mathcal L_{\lceil k^*/2\rceil-1}=\mathcal L$); $\ell=1$ would be
the trivial bound $-C\varepsilon^2$. The proof verifier proposed this statement and checked it; the
proof is written out in 3.5.

**Theorem 2 (sharpness; proven under an index condition).** Pattern teacher with every $s_r>0$.
Suppose that for some $\ell<k^*/2$ the generalized piggyback graph restricted to $\mathcal L_\ell$
contains a set $X$ of nodes with $\rho(\text{graph}|_X)>s_q$, and (**index condition**) that every
edge $e$ satisfies $\chi_e-1\ge$ (number of multi-core nodes $(C,w)\in X$ such that $e$ is an edge of
$\mathrm{Steiner}(C)$). Then there is a curve with $\lVert\theta(t)-\theta_S\rVert\le t$ and
$L(\theta(t))-L_S\le-ct^{2\ell}$: the order is $\le2\ell<k^*$. ($s_r>0$ makes the Perron vector
nonnegative, so the roots $(\eta x_n)^{1/j}$ of section 4 are real; signed weights would need sign
bookkeeping. Without the index condition: derived, not proven.)

**Corollary 0 (exact order; proven for pattern teachers).** Let $\ell^*<k^*/2$ be the smallest level
with $\lambda_{\max}(\mathcal S\vert_{\mathcal L_{\ell^*}})>s_q$, and assume
$\lambda_{\max}(\mathcal S\vert_{\mathcal L_{\ell^*-1}})<s_q$ (strict; this is what "generic" means
below) and the index condition for a set $X\subseteq\mathcal L_{\ell^*}$ with $\rho>s_q$. Then the order
is exactly $2\ell^*$ (Theorem 1' at level $\ell^*$ gives $\ge\min(k^*,2\ell^*)=2\ell^*$, Theorem 2
gives $\le2\ell^*$). If no level below $k^*/2$ exceeds $s_q$ and $\lambda_{\max}(\mathcal S)<s_q$, the
order is $k^*$ (Theorem 1, with the route). For a general (non-pattern) $R$ the lower bounds hold
unchanged; the upper bound $2\ell^*$ is empirical (three random-$R$ cases, section 5).

**Corollary 1 ($k^*\in\{3,4\}$; proven).** $\mathcal S=S_1$, the classical piggyback operator, i.e.
the old (fixed-rank) block of the Hessian. If $\lambda_{\max}(S_1)<s_q$ (pattern teacher: $\rho(S)<s_q$;
equivalently the old block is positive definite modulo the old gauge: the rotation block is
$s_q(s_q-S_1)$ in relative variables, the radial block has rank one with $K-1$ gauge zeros, and the
cross terms vanish because $R_\emptyset=R_{\{v\}}=0$) the order is exactly $k^*$, given the route
($\chi_e\ge2$ on the edges of one minimal Steiner tree). If $\lambda_{\max}(S_1)>s_q$ the saddle is
strict (order 2). Boundary $\lambda_{\max}(S_1)=s_q$: for $k^*=3$ the Hessian is PSD, so the order is
$\ge3$ by Taylor and $=3$ with the route; for $k^*=4$ it is open (one numerical instance, MPS4 with
$s_r=s_q$, has order 4). For $k^*=2$ the order is 2 whatever $\rho(S)$ is (the two-core route is a
direction of negative curvature).

**Corollary 2 (two-component teachers; proven).** One residual $r$: every node of the generalized
graph has at most one edge (node $(C_1,r|_{C_1})$ is joined only to $(T_r\setminus C_1,\cdot)$), so
$\rho=s_r$ if some split $T_r=C_1\sqcup C_2$ has $\deg C_1,\deg C_2<k^*/2$ and $\rho=0$ otherwise.
(a) Correct order $s_r<s_q$: order $=k^*$ on every tree, for every differing set $D$ and every $k^*$.
(b) Wrong order $s_r>s_q$ with no such split (e.g. $m=\lvert T_r\rvert\ge3$ with all terminals far apart: MPS5,
$r=10101$): order $=k^*$. (c) Wrong order with such a split: order $=2\ell^*$,
$\ell^*=\min_{\text{splits}}\max(\deg C_1,\deg C_2)$, under the index condition (for one residual it
reads $\chi_e\ge2$ on the edges of the two Steiner trees, which are vertex-disjoint): $\le2\ell^*$ by
Theorem 2, $\ge2\ell^*$ by Theorem 1' at level $\ell^*$ (no split has both parts of degree
$\le\ell^*-1$, so $\mathcal S\vert_{\mathcal L_{\ell^*-1}}=0$). Examples: $m=2$: strict, order 2; MPS6,
$r=110001$, split $\{0,1\}|\{5\}$: order $4<k^*=6$; HT4, $r=1111$, split $\{l_0,l_1\}|\{l_2,l_3\}$
(degrees 3, 3): order $6<k^*=7$.

**Corollary 3 (where R2 fails).** Assume a PD old block ($\lambda_{\max}(S_1)<s_q$). Then the first
failures need $k^*\ge5$ and a residual with $m\ge3$ terminals, two of which lie in a set of degree
$<k^*/2$ (adjacent cores, for $k^*=5,6$). ("First failures at $k^*=5$" holds off the boundary: $k^*=4$
with $\lambda_{\max}(S_1)=s_q$ is not excluded.)
Example (correct order, every $s_r<s_q$, Hessian PSD): MPS6, $q=0^6$ ($s_q=1$), $r_1=110001$,
$r_2=110010$ (0.8 each): $k^*=5$, the node $(\{0,1\},11)$ is the centre of a star with leaves
$(\{5\},1)$ and $(\{4\},1)$, $\rho=0.8\sqrt2=1.131>1$, order 4. Mechanism: a $k=2$ "route" toward the
*non-residual* pattern $110000$ (new entries on cores 0, 1) combined with old rotations of cores 4
and 5. At leading order, with $y$ the sector amplitude of $110000$ and $z_4,z_5$ the old rotations,
$L-L_S=\tfrac12(y^2+z_4^2+z_5^2)-y(s_1z_5+s_2z_4)$ (units $s_q=1$), indefinite iff
$s_1^2+s_2^2>s_q^2$, while $y=b_0b_1$ costs only $\lVert\delta\rVert^2\approx2y$. Minimizing on
$\lVert\delta\rVert=t$: $L-L_S=\tfrac18(1-s_1^2-s_2^2)\,t^4$ (checked: $-0.0350\,t^4$).

## 3. Proof of Theorem 1

**3.1 Moments and cumulants.** $N(\theta):=\langle U_q,W(\theta)\rangle$ ($N(\theta_S)=s_q$; $N$ is a
polynomial, so $N>s_q/2$ near $\theta_S$). For $B\ne\emptyset$ let $m_B:=\Pi_BW/N\in T(B)$, $m_\emptyset:=1$.
Let $\mathcal A$ be the space of families $f=(f_B)_{B\subseteq V}$, $f_B\in T(B)$, with product
$(f*g)_B=\sum_{B_1\sqcup B_2=B}f_{B_1}\otimes g_{B_2}$. $\mathcal A$ is a commutative associative
algebra with unit $1=(1,0,0,\dots)$, and any $f$ with $f_\emptyset=0$ is nilpotent ($f^{*(K+1)}=0$),
so $\exp$ and $\log(1+\cdot)$ are finite polynomial series obeying the usual identities
($\log(fg)=\log f+\log g$ for $f,g\in1+\mathcal A_{>0}$). Define the **cumulants**
$\kappa:=\log m$. Explicitly (set-partition moment-cumulant formulas)
$$m_B=\sum_{\pi\in\mathrm{Part}(B)}\ \bigotimes_{C\in\pi}\kappa_C,\qquad
\kappa_B=\sum_{\pi\in\mathrm{Part}(B)}(-1)^{\lvert\pi\rvert-1}(\lvert\pi\rvert-1)!\bigotimes_{C\in\pi}m_C .$$
So $\kappa_B$ is a polynomial in the $m_C$, $C\subseteq B$; all $m_B,\kappa_B$ are real-analytic near
$\theta_S$ (rational in $\theta$ with denominator a power of $N$), and $\kappa_{\{v\}}=m_{\{v\}}$
vanishes at $\theta_S$, so $\lVert\kappa_{\{v\}}\rVert\le C_0\lVert\theta-\theta_S\rVert$.

**3.2 Lemma 1 (exact identity; proven).** With $\langle R_B,\cdot\rangle$ on $T(B)$,
$$L(\theta)-L_S=\tfrac12(N-s_q)^2+\sum_{B\ne\emptyset}\Big[\tfrac{N^2}2\lVert m_B\rVert^2-N\langle R_B,m_B\rangle\Big].$$
*Proof.* Orthogonality of the sectors: $\lVert W-W^*\rVert^2=\sum_B\lVert\Pi_BW-\Pi_BW^*\rVert^2$,
$\Pi_\emptyset(W-W^*)=(N-s_q)U_q$, $\Pi_BW^*=R_B$ for $B\ne\emptyset$, $\Pi_BW=Nm_B$, and
$L_S=\tfrac12\sum_B\lVert R_B\rVert^2$. $\square$

**3.3 Lemma 2 (locality of cumulants; proven).** For $\lvert B\rvert\ge2$ there are $C_0,r_0>0$ with
$$\lVert\kappa_B(\theta)\rVert\le C_0\prod_{x\in\mathrm{Steiner}(B)}\lVert\nu_x\rVert\qquad(\lVert\theta-\theta_S\rVert<r_0).$$
*Proof.* (a) *Vanishing.* Fix $x\in\mathrm{Steiner}(B)$ and a point $\theta$ with $\nu_x=0$. Every
nonzero entry of $G_x$ then has all bond indices $0$, so the contraction factorizes at $x$:
$W=g_x\otimes\bigotimes_bW_b$, where $g_x=G_x[\,\cdot\,,0,\dots,0]$, $b$ runs over the branches at
$x$ (components of the tree minus $x$, vertex sets $V_b$), and $W_b$ is the contraction of branch
$b$ with its bond to $x$ fixed to $0$. The projectors factorize core-wise, so
$\Pi_BW=(\Pi^{(x)}_{B\cap\{x\}}g_x)\otimes\bigotimes_b\Pi^{(b)}_{B\cap V_b}W_b$ and
$N=(q_x^\top g_x)\prod_bN_b$, $N_b=\langle\bigotimes_{v\in V_b}q_v,W_b\rangle$ (all factors are close to
their nonzero values at $\theta_S$). Hence $m=m^{(x)}*\prod_bm^{(b)}$ in $\mathcal A$, where
$m^{(x)}$ is supported on $\{\emptyset,\{x\}\}$ and $m^{(b)}$ on subsets of $V_b$, each with unit
$\emptyset$-component. Taking $\log$: $\kappa=\log m^{(x)}+\sum_b\log m^{(b)}$, and each term is
supported on subsets of $\{x\}$ or of one $V_b$. But $B$ is contained in none of them: if $x\in B$,
$B\not\subseteq\{x\}$ ($\lvert B\rvert\ge2$) and $B\not\subseteq V_b$ ($x\notin V_b$); if $x\notin B$,
then $B$ meets two branches (if $B\subseteq V_b$, the connected set $V_b$ would contain
$\mathrm{Steiner}(B)\ni x$, but $x\notin V_b$). So $\kappa_B=0$
whenever $\nu_x=0$ for some $x\in\mathrm{Steiner}(B)$.
(b) *Bound.* Expand $\kappa_B$ in a power series in $\delta=\theta-\theta_S$, absolutely convergent
on a polydisc of radius $2r_0$. Restricting to $\{\nu_x=0\}$ deletes exactly the monomials that
contain a $\nu_x$ variable; the restriction is $0$ by (a), so by uniqueness of coefficients every
monomial with a nonzero coefficient contains at least one variable of $\nu_x$, for every
$x\in\mathrm{Steiner}(B)$. Factor one such variable per $x$ out of each monomial and bound the rest
on the polydisc of radius $r_0$: $\lVert\kappa_B\rVert\le C_0\prod_x\lVert\nu_x\rVert_\infty$. $\square$

Remark: Lemma 2 is the precise form of "routes": a cumulant on $B$ needs new entries on every core of
$\mathrm{Steiner}(B)$. It replaces F2 (including its disconnected-$J$ correction): disconnected
blocks are products of lower cumulants and are handled by the quadratic form below.

**3.4 Proof of Theorem 1.** Let $\varepsilon=\lVert\theta-\theta_S\rVert\le\min(r_0,1)$ and $y_C=\lVert\kappa_C\rVert$.
By 3.1 and Lemma 2: $y_C\le C_0\varepsilon$ for singletons and $y_C\le C_0\varepsilon^{\lvert\mathrm{Steiner}(C)\rvert}$
otherwise; so $y_C\le C_0\varepsilon$ for all $C$, and $y_C\le C_0\varepsilon^{k^*/2}$ for every
**high** set ($C\notin\mathcal L$, which for $k^*\ge3$ forces $\lvert C\rvert\ge2$, $\lvert\mathrm{Steiner}(C)\rvert\ge k^*/2$).
Norms are multiplicative on tensor products. $M$ denotes a combinatorial constant (number of set
partitions etc., depends on $K$ only); $r_{\max}=\max_B\lVert R_B\rVert$.

*(i) Gauss-Newton part, low sectors.* For $B\in\mathcal L$, $m_B-\kappa_B=\sum_{\lvert\pi\rvert\ge2}\bigotimes\kappa_C$,
and every block of such a $\pi$ is a subset of $B$, hence low ($\mathrm{Steiner}(C)\subseteq\mathrm{Steiner}(B)$).
A product of $\ge2$ factors is at most $C_0\varepsilon\,y_{C}$ for one of its blocks $C$, so
$\lVert m_B-\kappa_B\rVert^2\le MC_0^2\varepsilon^2\sum_{C\in\mathcal L}y_C^2$. With
$\lVert u+v\rVert^2\ge(1-\varepsilon)\lVert u\rVert^2-\varepsilon^{-1}\lVert v\rVert^2$:
$$\sum_{B\in\mathcal L}\lVert m_B\rVert^2\ \ge\ (1-C_2\varepsilon)\,\lVert\kappa_{\mathcal L}\rVert^2,\qquad \lVert\kappa_{\mathcal L}\rVert^2:=\sum_{C\in\mathcal L}y_C^2 .$$
The terms $\lVert m_B\rVert^2$ of high $B$ are $\ge0$ and dropped.

*(ii) Gain.* $\langle R_B,m_B\rangle\ne0$ only if $R_B\ne0$, so $\lvert\mathrm{Steiner}(B)\rvert\ge k^*$. Split
$m_B=\kappa_B+\sum_{\lvert\pi\rvert=2}+\sum_{\lvert\pi\rvert\ge3}$:
- $\lvert\langle R_B,\kappa_B\rangle\rvert\le r_{\max}C_0\varepsilon^{k^*}$ (Lemma 2).
- $\pi=\{C_1,C_2\}$, both low: summed over $B$ and $\pi$ these terms are exactly
  $\tfrac12\langle\kappa_{\mathcal L},\mathcal S\kappa_{\mathcal L}\rangle\le\tfrac12\lambda_{\max}(\mathcal S)\lVert\kappa_{\mathcal L}\rVert^2$
  (each unordered split appears once; $\langle\kappa,\mathcal S\kappa\rangle$ counts ordered pairs).
- one low ($C_1$), one high ($C_2$): $\le r_{\max}y_{C_1}y_{C_2}\le\tfrac\gamma2y_{C_1}^2+\tfrac{r_{\max}^2}{2\gamma}C_0^2\varepsilon^{k^*}$, any $\gamma>0$.
- both high: $\le r_{\max}C_0^2\varepsilon^{k^*}$.
- $\lvert\pi\rvert\ge3$: if two blocks $C_1,C_2$ are low, $\le r_{\max}(C_0\varepsilon)^{\lvert\pi\rvert-2}y_{C_1}y_{C_2}\le r_{\max}C_0\varepsilon\,\tfrac12(y_{C_1}^2+y_{C_2}^2)$;
  otherwise two blocks are high and the term is $\le r_{\max}C_0^{\lvert\pi\rvert}\varepsilon^{k^*}$.

Summing (finitely many terms, each $y_C^2$ used at most $M$ times):
$$\sum_B\langle R_B,m_B\rangle\le\tfrac12\big(\lambda_{\max}(\mathcal S)+M\gamma+Mr_{\max}C_0\varepsilon\big)\lVert\kappa_{\mathcal L}\rVert^2+C_1(\gamma)\,\varepsilon^{k^*}.$$

*(iii) Conclusion.* By Lemma 1, dropping $\tfrac12(N-s_q)^2\ge0$,
$$L-L_S\ge\tfrac N2\Big[N(1-C_2\varepsilon)-\lambda_{\max}(\mathcal S)-M\gamma-Mr_{\max}C_0\varepsilon\Big]\lVert\kappa_{\mathcal L}\rVert^2-NC_1(\gamma)\varepsilon^{k^*}.$$
Let $g=s_q-\lambda_{\max}(\mathcal S)>0$, take $\gamma=g/(4M)$, and use $N\ge s_q-C_3\varepsilon$: for
$\varepsilon$ small the bracket is $\ge g/2>0$, so $L-L_S\ge-2s_qC_1\varepsilon^{k^*}$. $\square$

*What the proof uses.* Only: sector orthogonality (whitened loss, rank-one $W_S$ across the core
partition), criticality, multilinearity of the contraction (Lemma 2), and $N(\theta_S)\ne0$. Not used:
balancedness, pattern structure of $R$, bond dimensions, the gauge. The constant $C$ blows up as
$g\to0$; at $g=0$ the theorem says nothing.

**3.5 Proof of Theorem 1' (level version).** Fix $2\le\ell\le\lceil k^*/2\rceil$ and put
$p:=\min(k^*,2\ell)$. Run 3.4 with the split
$$\text{low}:=\mathcal L_{\ell-1}=\{C\ne\emptyset:\deg C\le\ell-1\},\qquad \text{high}:=\{C\ne\emptyset:\deg C\ge\ell\}.$$
Four facts replace the ones used for $\mathcal L$:
- (F1) *High cumulants are small.* A high set has $\deg C\ge\ell\ge2$, so $\lvert C\rvert\ge2$, and
  Lemma 2 gives $y_C\le C_0\varepsilon^{\deg C}\le C_0\varepsilon^{\ell}$ ($\varepsilon\le1$). Every set
  has $y_C\le C_0\varepsilon$.
- (F2) *Low sets are closed under subsets.* If $C\subseteq B$ then $\deg C\le\deg B$ (equal to 1 for a
  singleton, else $\mathrm{Steiner}(C)\subseteq\mathrm{Steiner}(B)$). So every block of a partition of a
  low set is low.
- (F3) *Low sets carry no residual.* $\deg B\le\ell-1\le\lceil k^*/2\rceil-1<k^*$, so $R_B=0$ for low $B$
  (by the definition of $k^*$).
- (F4) *The quadratic form.* Summing $\langle R_B,\kappa_{C_1}\otimes\kappa_{C_2}\rangle$ over all $B$ and
  all unordered splits $B=C_1\sqcup C_2$ with $C_1,C_2$ low gives exactly
  $\tfrac12\langle\kappa_{\rm low},\mathcal S\vert_{\mathcal L_{\ell-1}}\kappa_{\rm low}\rangle$, because
  each unordered pair of disjoint low sets occurs once (for $B=C_1\sqcup C_2$) and the operator counts
  ordered pairs. Hence it is $\le\tfrac12\lambda_\ell\lVert\kappa_{\rm low}\rVert^2$, $\lambda_\ell:=\lambda_{\max}(\mathcal S\vert_{\mathcal L_{\ell-1}})$.

*(i) Gauss-Newton part.* By (F2), for low $B$, $m_B-\kappa_B$ is a sum of products of $\ge2$ low
cumulants, each product at most $C_0\varepsilon\,y_C$ for one of its blocks. As in 3.4,
$\sum_{B\text{ low}}\lVert m_B\rVert^2\ge(1-C_2\varepsilon)\lVert\kappa_{\rm low}\rVert^2$; the
high-$B$ terms are $\ge0$ and dropped.

*(ii) Gain.* Only $B$ with $R_B\ne0$ contribute, so $\lvert\mathrm{Steiner}(B)\rvert\ge k^*\ge2$. Write
$m_B=\kappa_B+\sum_{\lvert\pi\rvert=2}+\sum_{\lvert\pi\rvert\ge3}$ and bound each type of term:
- single cumulant: $\lvert\langle R_B,\kappa_B\rangle\rvert\le r_{\max}C_0\varepsilon^{k^*}\le r_{\max}C_0\varepsilon^{p}$ (Lemma 2);
- two low blocks: (F4);
- one low ($C_1$), one high ($C_2$): $r_{\max}y_{C_1}y_{C_2}\le\tfrac\gamma2y_{C_1}^2+\tfrac{r_{\max}^2}{2\gamma}C_0^2\varepsilon^{2\ell}$ by Young and (F1);
- two high: $\le r_{\max}C_0^2\varepsilon^{2\ell}$ (F1);
- $\lvert\pi\rvert\ge3$ with two low blocks $C_1,C_2$: $\le r_{\max}(C_0\varepsilon)^{\lvert\pi\rvert-2}y_{C_1}y_{C_2}\le r_{\max}C_0^{\lvert\pi\rvert-2}\varepsilon\cdot\tfrac12(y_{C_1}^2+y_{C_2}^2)$;
- $\lvert\pi\rvert\ge3$ with at most one low block: at least two blocks are high, so the term is
  $\le r_{\max}C_0^{\lvert\pi\rvert}\varepsilon^{2\ell}$.

Every remainder is $O(\varepsilon^{2\ell})$ or $O(\varepsilon^{k^*})$, i.e. $O(\varepsilon^{p})$. Summing
(finitely many terms; each $y_C^2$ used at most $M$ times):
$\sum_B\langle R_B,m_B\rangle\le\tfrac12(\lambda_\ell+M\gamma+Mr_{\max}C_0\varepsilon)\lVert\kappa_{\rm low}\rVert^2+C_1(\gamma)\varepsilon^{p}$.

*(iii) Conclusion.* Exactly as in 3.4(iii) with $g_\ell=s_q-\lambda_\ell>0$, $\gamma=g_\ell/(4M)$:
$L-L_S\ge-2s_qC_1\varepsilon^{p}$. $\square$

What changes compared with Theorem 1 is only the exponent in (F1): with the cut at degree $\ell$
instead of $k^*/2$, two high factors cost $\varepsilon^{2\ell}$ instead of $\varepsilon^{k^*}$. Remarks:
(a) The theorem uses no pattern structure, so it holds for general $R$. (b) It is useful exactly when
the full condition of Theorem 1 fails at a level $\ge\ell$: it then caps how fast that failure can
descend. (c) The verifier's three exact orders follow: MPS6 pair-star and MPS6 $110001$ (1.5) have
$\mathcal S\vert_{\mathcal L_1}=S_1=0$ (no residual with two terminals), so order $\ge4$; HT4 $1111$
(1.5) has $\mathcal S\vert_{\mathcal L_2}=0$ (two leaves are never adjacent, and sets containing an
internal core have $T(C)=0$), so order $\ge6$. With Theorem 2 the orders are exactly 4, 4, 6.

## 4. Proof of Theorem 2 (explicit curve)

Pattern teacher, standard site basis, $q=0\cdots0$, $a=s_q^{1/K}$ (balanced; any $\theta_S$ in the
gauge orbit gives the same order). Let $A_X$ be the adjacency matrix of the generalized graph on $X$,
$x\ge0$ a unit Perron vector, $A_Xx=\rho x$, $\rho>s_q$ (a nonnegative Perron vector exists because
every $s_r>0$). For $0<\eta\le1$ set $\theta(\eta)=\theta_S+\delta$:
- singleton node $(\{v\},w)$: old rotation $G_v[w,0,\dots,0]=a\,\eta\,x_{v,w}$;
- multi-core node $n=(C,w)$, $P=\mathrm{Steiner}(C)$, $j=\lvert P\rvert\le\ell$: pick a new index
  value $\alpha_n(e)\ge1$ on every edge $e$ inside $P$, distinct from the values of the other
  multi-core nodes of $X$ on $e$ (possible by the index condition), and set
  $G_u[\phi_u,\ \alpha_n\text{ on the }P\text{-edges at }u,\ 0\text{ elsewhere}]=a\,(\eta x_n)^{1/j}$ for
  $u\in P$, with $\phi_u=w_u$ ($u\in C$) or $0$ ($u\in P\setminus C$).

Then $\lVert\delta\rVert\le C\eta^{1/\ell}$.

*Polymer representation (exact).* In a nonzero term of the contraction, an edge with a nonzero
index carries some $\alpha_n(e)$, so both its endpoints use entries of node $n$; by connectedness
every connected component $Q$ of the nonzero-index edges uses the entries of a single node $n$, and
since those entries have nonzero indices exactly on the $P$-edges, $Q=\mathrm{Steiner}(C_n)$.
Hence $W=\sum_\Gamma\big(\prod_{n\in\Gamma}\eta x_n\,a^{\lvert P_n\rvert}\,w_n\big)\otimes(\text{old slices elsewhere})$,
summed over sets $\Gamma$ of multi-core nodes with pairwise vertex-disjoint Steiner trees. Old slices
are $a(e_0+\eta\sum_wx_{v,w}e_w)$. So $N=s_q$ exactly, and every $m_B$ is a polynomial in $\eta$
whose $\eta^n$ coefficient sums over compatible configurations of $n$ items (polymers and
rotations) with disjoint decorations covering $B$, the cores of $P_n\setminus C_n$ not in $B$.

*Order $\eta$.* $m^{(1)}_C=\sum_wx_{C,w}e_w$ (one item); $\sum_B\lVert m^{(1)}_B\rVert^2=\lVert x\rVert^2=1$.
No item has $C=T_r$ (that would need $\deg C\ge k^*>\ell$), so $\langle R_B,m^{(1)}_B\rangle=0$.

*Order $\eta^2$ in the gain.* If $(C_1,w_1),(C_2,w_2)$ is an edge (residual $r$, $T_r=C_1\sqcup C_2$),
the two Steiner trees are vertex-disjoint and $\mathrm{Steiner}(C_1)\cap C_2=\emptyset$: otherwise
$\mathrm{Steiner}(C_1)\cup\mathrm{Steiner}(C_2)$ would be a connected set containing $T_r$ with at most
$2\ell-1<k^*$ cores. So every edge is a compatible two-item configuration, and
$\sum_B\langle R_B,m^{(2)}_B\rangle=\sum_{\text{edges}}s_rx_{n_1}x_{n_2}=\tfrac12x^\top A_Xx=\tfrac\rho2$.

*Conclusion.* By Lemma 1 with $N=s_q$:
$L-L_S=s_q^2\big[\tfrac12\eta^2-\tfrac{1}{s_q}\tfrac\rho2\eta^2\big]+O(\eta^3)=-\tfrac{s_q}2(\rho-s_q)\eta^2+O(\eta^3)$.
With $\eta=(t/C)^\ell$: $\lVert\delta\rVert\le t$ and $L-L_S\le-ct^{2\ell}$. $\square$

The same construction with only singleton nodes is the classical piggyback direction; with one
multi-core node and leaves it is the pair-star of Corollary 3. Gradient flow does not need the
index condition (it can use any curve); without it, distinct nodes sharing an edge and a new index
mix at order $\eta$ and the bookkeeping above fails, so that case is only derived.

## 5. Numerical evidence (empirical; `experiments/theory_checks/saddle_order/`)

Method: $m(t)=\min_{\lVert\delta\rVert=t}L(\theta_S+\delta)-L_S$ by L-BFGS on the sphere (12 random
starts per $t$: full, new-only, mostly-old directions; plus continuation from the previous $t$),
$t=0.3\to0.025$ (0.4 $\to$ 0.05 for $k^*=7$), float64, $L-L_S$ computed as
$\tfrac12\lVert\Delta\rVert^2-\langle R,\Delta\rangle$ with $\Delta=W-W_S$ (no cancellation). No gauge
quotient is needed: $m(t)$ measures the order with respect to $\lVert\theta-\theta_S\rVert$, which equals
the order with respect to the distance to the local gauge orbit (section 1, "Gauge": descent curves
transfer because the orbit distance is smaller; lower bounds transfer through the linear,
loss-preserving gauge action with a constant $\lVert g^{-1}\rVert^p$ uniform near the identity).
(v1 argued that the two distances "agree up to constants", which is false: an orbit point has
distance 0 to the orbit.) Order =
local slope of $\log\lvert m\rvert$ vs $\log t$. $d=2$, $\chi=2$, $s_q=1$ unless stated. Predicted
orders from `gp_operator.predicted_order` (builds $\mathcal S$ on $\mathcal L_\ell$ for every $\ell<k^*/2$).
All slopes below are the three smallest-$t$ slopes, equal to the prediction within $0.003$, with one
exception: random $R$ on binary6 at $\lambda_2=1.25$ has $\lambda_1=0.964$ (singleton level close to
$s_q$), so the asymptotics set in late; on a finer grid down to $t=0.003$ its slopes are
3.63, 3.84, 3.95, 3.99, 3.997 (converging to the predicted 4 from below).

| group | instances (tree, residuals) | $k^*$ | $\lambda_{\max}(\mathcal S\vert\mathcal L_\ell)$ | predicted / measured order | constant |
|---|---|---|---|---|---|
| A: one residual, $s_r<s_q$ (Cor. 2a) | mps4 1100, 1010, 1001 (also $\chi=3$, $d=3$, $s_r=0.9$); mps5 10010, 10001, 10101; binary4 1010; binary6 100001, 100100; ht4 1100, 1010; mps6 110001; ht4 1111 | 2-7 | $s_r$ or 0 | $k^*$ / $k^*$ (16 cases) | $m/t^{k^*}$ = single-route value $s_ra^{K-k}k^{-k/2}$ to 4 digits |
| B: several residuals, $\lambda_{\max}<s_q$ (Thm 1) | mps4 star (0.6, 0.6); mps5 $\{10010,10001\}$; mps5 three residuals sharing $\{0,4\}$; mps6 pair-star (0.6, 0.6); general random $R$ (mps5 $d=3$, mps6, binary6) scaled to $\lambda=0.8$ | 3-5 | 0.85, 0.64, 0.72, 0.85, 0.8 | $k^*$ / $k^*$ (7 cases) | shared-$\{0,4\}$: $-0.0129$ vs single route $-0.0089$ (the route fits a superposition) |
| C: $\lambda_{\max}>s_q$ at $\ell<k^*/2$ (Thm 2) | mps5 10001 (1.5): $\ell=1$; mps6 110001 (1.5): $\ell=2$; mps6 pair-star (0.8, 0.8): $\ell=2$; ht4 1111 (1.5): $\ell=3$; random $R$ (3 trees) scaled to $\lambda_2=1.25$ | 5, 6, 5, 7, 5 | 1.5, 1.5, 1.131, 1.5, 1.25 | $2\ell$ / $2\ell$: 2, 4, 4, 6, 4 | leading-order formulas: $-0.15625t^4$ (meas. $-0.1562$), $-0.0350t^4$ ($-0.0350$), $-t^6/432$ ($-0.002315$) |
| D: borderline $\lambda_{\max}=s_q$ (theory silent) | mps4 1001, mps5 10001 with $s_r=s_q$ | 4, 5 | 1.0 | ? / $k^*$ | single-route value |

In groups A and B the minimizer has almost no weight on old entries (fraction $\le10^{-4}$ at
$t=0.025$, shrinking $\propto t^2$): the optimal curve is the route. In group C the old fraction
shrinks $\propto t^2$ as well but is larger ($5\cdot10^{-4}$ at $t=0.025$), as expected from old
rotations of size $t^2$ in the generalized piggyback curve. Numerical caveat: a minimum over a sphere
found by local search is an upper bound on $m(t)$; a missed faster descent would show up as a slope
below the prediction, which never happened, and the measured constants equal the analytic optima.
The group-C constants come from optimizing the reduced quadratic forms, not from the Perron-vector
curve of section 4; that they are the exact leading constants is **empirical**.

Independent reproduction (numerics verifier, own code, `experiments/theory_checks/saddle_order_verify/`;
degree-2 curves $\theta_S+ta_1+t^2a_2$ orthogonal to the gauge tangent, slopes against the actual
distance): 12 instances, $k^*=2..6$ on mps4/5/6, binary4/6, ht4, measured order = predicted order in
every case (slopes within $5\cdot10^{-4}$ of the integer), constants within 0.1-0.4% of the analytic
values; pair-star (0.8): order 3.9997, constant $-0.0350$; control (0.6): order 5.000. Following the
optimal point to smaller scales, the order-4 descent needs the **curved** path (new entries $\propto\tau$,
old rotations $\propto\tau^2$): the straight line through the same point and the new-entries-only line
turn positive at small $\tau$. The proof verifier reproduced the three counterexamples with separate
code (pair-star: Hessian eigenvalues 0/1/6, PSD; $-0.03500\,t^4$; MPS6 $110001$: $-0.1562\,t^4$; HT4
$1111$: $-t^6/432$) and three stress tests of Theorem 1 (MPS5 $10101$ at $s_r=2s_q$: order 5; MPS6
$110001$ at 0.7: order 6; pair-star control: order 5).

## 6. Consequence for the dynamics (derived; checked on one teacher, with caveats)

**Reduced model (derived).** Where Theorem 2 applies at level $\ell$, the old rotations have $O(1)$
curvature and follow the cluster adiabatically ($z\approx s\,y$ in Cor. 3), leaving
$L_{\rm eff}=-\tfrac c2y^2$, $y=\prod_{v\in P}b_v$ ($\ell$ new entries), $-c<0$ the Schur complement of
the section-3 quadratic form on the cluster variable after eliminating the old rotations
($c=s_1^2+s_2^2-1$ for the pair-star, units $s_q=1$). Then $\dot b_v=c\,b_v\prod_{u\ne v}b_u^2$,
homogeneous of degree $2\ell-1$, so from amplitudes $b\sim\sigma$ the escape takes
$t_{\rm esc}\propto\sigma^{-(2\ell-2)}$ ($\log(1/\sigma)$ for $\ell=1$, the classical piggyback). This
replaces the route law $\sigma^{-(k^*-2)}$ when $2\ell<k^*$ at small enough amplitudes; which mechanism
fires first at finite $\sigma$ is a race between the two (leading order: smaller exponent wins as
$\sigma\to0$; prefactors decide at finite $\sigma$). $L_{\rm eff}$ is even in $y$, so unlike a route
there is no sign-correction delay. With two multi-core nodes (HT4 $1111$, binary8 odeco) the reduced
model is a coupled polymer system, $L_{\rm eff}=\tfrac12(y_1^2+y_2^2)-(s_r/s_q)y_1y_2$ (units
$s_q=1$), and the same exponent needs the polymers to align in sign (not checked).
The escape direction is a **non-residual** pattern (pair-star: $110000$) together with old rotations
toward the residuals, so the next plateau is not "$r$ learned".

**What comes next (empirical, numerics verifier).** Within about $1.5$ plateau lengths after the
escape, gradient flow reaches $W\approx0.8(e_{110001}+e_{110010})$ with $W[q]\approx0$ and
$L=0.500=\tfrac12s_q^2$: **$q$ is forgotten** and the residual pair is learned instead. It stays
there and then relearns $q$ ($L\to0$). This is the generalized analogue of the star dynamics of
`theory_saddles.md` 4.6, and it is what rule 5.1-3 gives once the cluster has raised the rank of
edge $(0,1)$: across the edge between cores 1 and 2 the teacher's matricization has singular values
$0.8\sqrt2=1.131$ (the pair, row $11$) and $1$ ($q$), so the best fit with rank one there is the pair.
The $L=0.5$ plateau is much longer than the $q$ plateau (seed 7: $4.7\cdot10^6$, $3.8\cdot10^7$,
$3.0\cdot10^8$ at $\sigma=0.01,0.005,0.0025$, exponent 3.0; seed 2: exponent about 4.2 at
$\sigma=0.035\to0.0175$); not analysed.

**Check 1: start at the saddle** (`gf_escape.py`, gradient flow from $\theta_S+\sigma\xi$, LSODA rtol
$10^{-10}$, escape = first time $L-L_S<-10^{-3}$, $\sigma=0.04\to0.005$, 3 seeds; the proof verifier
added 2 seeds with own code at $\sigma=0.02\to0.005$: exponents 2.035, 2.019 and 1.963, 1.981, same
escape direction). Local exponents $\log_2(t_{\rm esc}(\sigma/2)/t_{\rm esc}(\sigma))$:

| teacher (MPS6, $s_q=1$) | prediction | seed 0 | seed 1 | seed 2 | amplitude of $110000$ at escape |
|---|---|---|---|---|---|
| pair-star (0.8, 0.8), $\rho=1.13$ | 2 | 2.15, 2.07, 2.03 | 2.15, 2.08, 2.04 | 1.98, 1.99, 2.00 | $\pm0.085$ |
| control (0.6, 0.6), $\rho=0.85$ | 3 (route $k^*=5$) | 3.15, 3.33, 3.54 | 2.86, 2.93, 2.97 | no escape by $t=10^7$ ($\sigma=0.04$): $L-L_S\to0^+$, residual amplitudes $\sim10^{-28}$ (trapped) | $\le0.002$ |

The control escapes through the $k^*=5$ route of $r_2=110010$ in both inspected seeds (its amplitude
grows; $r_1$ stays at 0). With the same init directions the pair-star escapes 490 and 93 times earlier
at $\sigma=0.005$ (seeds 0, 1), and also from the seed-2 direction (where the control's routes die),
at $t=5959$ for $\sigma=0.04$ (proof verifier: 178 times earlier, seed 1, $\sigma=0.02$).

**Check 2: start at small fan-in initialization** (numerics verifier, `gf_origin.py`: origin-scale
fan-in init, same direction at every $\sigma$, seeds 0-11, $\sigma=0.1\to0.01$, seed 7 to $0.0025$;
plateau length $\Delta t=t_{\rm esc}-t_q$, $t_q$ = time $W[q]$ crosses $\tfrac12$; only runs that learn
$q$ first). Caveats, all from that review:
1. **Only some seeds visit $\theta_S$.** Of the 5 pair-star seeds finished at review time, 3 (seeds
   0, 1, 8) learn the residual pair first (the $L=0.5$ state) and never visit $\theta_S$; seeds 2 and 7
   learn $q$ first. The control (0.6) learns $q$ first in 12 of 12 seeds.
2. **The $\sigma^{-2}$ law appears only at small $\sigma$ and rested on one seed.** Seed 7: local
   exponents 6.2, 2.91, 2.33, 2.15, 2.08, 2.03, 2.01, **2.005** for $\sigma=0.025\to0.0025$; no $q$
   plateau at all for $\sigma\ge0.035$ (the escape happens during the rise of $q$). So from the origin
   the law sets in for $\sigma\lesssim0.005$, later than from $\theta_S+\sigma\xi$ (where it holds from
   $\sigma=0.04$): the dormant amplitudes at $\theta_S$ are set by the first escape, not by $\sigma$
   alone. Seed 2 followed the same pre-asymptotic curve (3.50, 2.48 at $\sigma=0.025, 0.0175$).
   Escape direction in every $q$-first run with $\Delta t>5$: $W[110000]=\pm0.085$,
   $W[000001]=W[000010]=\pm0.065$, residual entries $0.0056$: the predicted cluster plus rotations.
3. **$q$ is forgotten and then relearned** (previous paragraph).
4. **Control teacher: the route law holds only seed-wise.** 2 of 12 seeds are trapped (dead routes,
   $W[r_1],W[r_2]\sim10^{-38}$ at $t=3\cdot10^{11}$); 10 escape, always through the $k^*=5$ route of
   $r_2$ ($\lvert W[110000]\rvert\le0.003$, $r_1$ at 0), then converge to $L=0$. Their local exponents at
   $\sigma=0.0175\to0.01$ split between about 3 (seeds 0, 3, 8, 11: 3.00, 3.02, 2.90, 3.08) and about 4
   (seeds 2, 5, 7: 3.99, 4.00, 3.96; seeds 4, 9, 10 still rising: 3.40, 3.91, 3.78). The exponent 4
   is **unexplained**: the saddle order (5, route law $\sigma^{-3}$) does not give it, and the only
   $k=6$ route ($r_1$) is unused. The same seed-dependent steepening of $k=5$ routes on MPS from fan-in
   initialization appears in the locality experiments (report Sec. 4, failure modes). No control seed
   has exponent near 2, so in the dynamics the dichotomy $\rho>s_q$ versus $\rho<s_q$ is sharp.

Status of S9: **derived** (reduced model); checked from $\theta_S+\sigma\xi$ on 5 seeds of one teacher,
and from small initialization on the $q$-first seeds only, asymptotically.

## 7. Changes to `notes/theory_saddles.md` (applied there in v3, 2026-10-10)

1. **R2 (results table) and Prop. 2's paragraph "What is and is not proven".** Replace "order $=k^*$
   derived; lower bound open beyond order 3" by: order $\ge k^*$ **proven** whenever
   $\lambda_{\max}(\mathcal S)<s_q$ (Theorem 1), hence order $=k^*$ proven with the route; this covers
   $k^*=2$ (always), $k^*\in\{3,4\}$ with a positive-definite old block (Cor. 1) and all two-component
   teachers in the correct order (Cor. 2a). Add: R2 is **false** in general from $k^*=5$ on (single
   wrong-order residuals, Cor. 2c; with several residuals also in the correct order, Cor. 3); the exact
   orders $2\ell^*$ need Theorem 1' (lower bound) and Theorem 2 (upper bound).
2. **F2.** The disconnected-$J$ discussion ("generalized piggyback blocks ... shown only for this
   case") is superseded by Lemma 2 + the quadratic form $\mathcal S$: disconnected blocks are
   products of cumulants; whether they are GN-dominated is exactly $\lambda_{\max}(\mathcal S)<s_q$,
   and it can fail even when every residual is smaller than $s_q$ and the Hessian is PSD.
3. **Prop. 1 / R1 ("escape = first-escape problem of the route sub-TTN").** Add the hypothesis
   $\lambda_{\max}(\mathcal S)<s_q$ next to the existing caveat (old block a local minimum). When it
   fails at level $\ell<k^*/2$ the escape is a *joint* problem: a degree-$\ell$ new-index cluster
   toward a **non-residual** partial pattern plus old rotations, with $t_{\rm esc}\propto\sigma^{-(2\ell-2)}$
   (section 6; derived for the reduced model, checked for one teacher).
4. **Section 3.3 heading and R4/R5.** "order $k^*$ derived" becomes "proven under
   $\lambda_{\max}(\mathcal S)<s_q$". The plateau law $\sigma^{-(k^*-2)}$ needs the same condition;
   otherwise the exponent is $2\ell^*-2$ with $\ell^*$ the smallest level where $\mathcal S$ exceeds
   $s_q$ (derived; the order $2\ell^*$ itself is proven via Theorem 1' + Theorem 2).
5. **Section 4.1 (piggyback graph) and 4.3 table.** The piggyback graph is the $\ell=1$ level of
   the generalized graph. Row "$s_q<s_r$, $m\ge3$" stays "non-strict" (true), but its order is
   $\min(k^*,2\ell^*)$, $\ell^*$ from the splits of $T_r$ (Cor. 2b-c), e.g. MPS6, $D=\{0,1,5\}$: order 4,
   not 6; and 4.1's "$m\ge3$: piggyback is not an escape route" holds only at level 1.
   **Which experiments are affected (v1 said "$N\le4$: nothing changes", wrong for HT4, $K=7$).**
   For $K\le4$ (mps4, binary4) nothing changes ($k^*\le4$). For two-component teachers in the correct
   order nothing changes on any tree (Cor. 2a), so every fitted exponent of the locality experiments
   stands (they use the runs that learn the larger component first). In the wrong order (smaller
   component learned first; for tensor-like pairs, $m\ge3$, this happens with $\sigma$-independent
   probability) on HT4: two-site $D$ have $m=2$ (classical piggyback, strict); three-site $D$ have
   $k^*=6$ and no low split ($\{l_0,l_1\}$ has degree $3=k^*/2$), so order 6 $=k^*$; $D=\{0,1,2,3\}$ has
   the split $\{l_0,l_1\}\vert\{l_2,l_3\}$ (degrees 3, 3), so order **6, not 7**, a predicted escape
   $\propto\sigma^{-4}$ instead of $\sigma^{-5}$ (derived), along the non-residual patterns that carry the
   larger component on one leaf pair and the smaller on the other, followed by forgetting of the smaller
   component (as in section 6). The same structure (two leaf pairs) holds for binary8 with $D$ = all
   sites (order 6, $k^*=7$).
   **What the data show** (re-read from `data/locality/`, small fan-in initialization, weights
   $(1,0.5)$; these runs were classified "rotation" or "other order" and excluded from the fits).
   HT4, $D$ = all sites, $\chi=2$: 5 runs learn the smaller component first (seeds 1, 2; $\sigma\ge0.40$).
   In 4 of them the smaller component is **forgotten** (overlap below 0.1) within 3 time units of the
   larger one's event, the flow rests at $L\approx0.125=\tfrac12(0.5)^2$ with overlaps $(1,\le0.03)$,
   and the smaller component is relearned; in the fifth ($\sigma=0.6$) it dips only below 0.9. Two
   further wrong-order runs of a re-run sweep (`r2comp`, $\sigma=0.4349$) also forget and relearn. The v1
   rule (`theory_saddles.md` 4.4, case 3: "$m\ge3$: no rotation, $a_q$ stays 1") excluded this; it is
   what the order-6 mechanism predicts. $\chi=4$: one seed learns the smaller component first at all six
   $\sigma$ ($0.6\to0.15$); it keeps it except at $\sigma=0.15$ (a dip below 0.5, not below 0.1), and
   the local exponent of its plateau falls from 6.8 to 5.28, 4.85, 4.70, 4.59, below the route value 5:
   consistent with the predicted 4, not a measurement of it (0.6 decades, one seed). Exponents cannot be extracted
   from the $\chi=2$ runs ($\sigma\in[0.40,0.6]$). The `order` agent's binary8 odeco runs from the
   wrong saddle ($\rho=0.8$, $\chi=4$, $\sigma=0.35\to0.15$, 6 seeds) show the route law ($t_1$ exponent
   $5.25\pm0.27$, no forgetting). That is expected at these $\sigma$: with aligned unit amplitudes $b$
   the polymer escape takes $b^{-4}/(4\kappa)$, $\kappa=(s_r-s_q)s_q^{(K-6)/K}$, and the route
   $b^{-5}/(5s_r)$ (it never sees $q$, $k^*=K$), which cross at $b=4\kappa/(5s_r)\approx0.155$ for binary8
   ($s_q=0.8$, $s_r=1$) and $\approx0.36$ for HT4 ($s_q=0.5$, $s_r=1$) (derived, leading order). So the
   binary8 runs sit at or above the crossover and the HT4 $\chi=2$ runs near it. Prediction: below
   the crossover the wrong-order exponent tends to 4 and the smaller component is forgotten.
6. **Section 11, first open issue.** Close it (Theorems 1-2) and replace by: the borderline
   $\lambda_{\max}(\mathcal S)=s_q$ for $k^*\ge4$; Theorem 2 without the index condition; several
   learned components ($\lvert S\rvert\ge2$, where $W_S$ is not rank one across the core partition and
   the sector decomposition must be replaced by the learned column/row spaces).
7. **Section 9 (literature table), row R2.** Add: the lower bound and the counterexamples are new
   (not found in literature.md); the tools (set-partition moment-cumulant formulas, factorization of
   cumulants of independent blocks) are classical.
8. **Section 5.1 (rule for the whole sequence).** Step "next escape = smallest $k^*$" should compare
   $k^*$ with $2\ell^*$: the next event is the smaller of the two (log for 2, power $p-2$ otherwise).
   This comparison is **derived** (a leading-order race), not proven.
9. **CLAUDE.md / report** (for the lead): Q3 status line "saddle order $=k^*$ derived" becomes (v1
   proposed "proven iff ...", which the proof verifier flagged as overstated: necessity is proven only
   for pattern teachers with $s_r>0$, strict inequality and the index condition): "saddle order $=k^*$
   **proven** when $\lambda_{\max}(\mathcal S)<s_q$ (always for $k^*=2$, for $k^*\in\{3,4\}$ with a PD old
   block, and for two-component teachers in the correct order); otherwise order $2\ell^*<k^*$ (proven
   for pattern teachers with a spare bond index; first case $k^*=5$, MPS6 pair-star)".

## 8. Claims and status

| # | claim | status |
|---|---|---|
| S1 | Exact identity $L-L_S=\tfrac12(N-s_q)^2+\sum_B[\tfrac{N^2}2\lVert m_B\rVert^2-N\langle R_B,m_B\rangle]$ (Lemma 1) | proven |
| S2 | Locality: $\lVert\kappa_B\rVert\le C\prod_{x\in\mathrm{Steiner}(B)}\lVert\nu_x\rVert$ for $\lvert B\rvert\ge2$ (Lemma 2) | proven |
| S3 | Theorem 1: $\lambda_{\max}(\mathcal S)<s_q\Rightarrow L-L_S\ge-C\lVert\delta\rVert^{k^*}$; with the route ($\chi_e\ge2$ on the edges of one minimal Steiner tree), order $=k^*$ (any tree, any $\chi$, any residual with $R_{\{v\}}=0$, balanced or not) | proven |
| S3' | Theorem 1': $\lambda_{\max}(\mathcal S\vert_{\mathcal L_{\ell-1}})<s_q\Rightarrow L-L_S\ge-C\lVert\delta\rVert^{\min(k^*,2\ell)}$ (general $R$) | proven (section 3.5; statement and check by the proof verifier) |
| S4 | Cor. 1, $k^*\in\{3,4\}$: order $=k^*$ if $\lambda_{\max}(S_1)<s_q$ (with the route), strict if $>s_q$; $k^*=3$ also at equality; $k^*=2$: order 2 always | proven ($k^*=4$ at equality: open; one instance has order 4) |
| S5 | Cor. 2a: two-component teachers, correct order: order $=k^*$ on every tree | proven (with the route) |
| S6 | Theorem 2: generalized graph with $\rho>s_q$ at level $\ell<k^*/2$ $\Rightarrow$ order $\le2\ell$ | proven (pattern teacher, $s_r>0$, index condition) |
| S7 | Corollary 0: order $=\min(k^*,2\ell^*)$ ($\lambda_{\max}<s_q$ strictly at level $\ell^*-1$) | proven for pattern teachers with $s_r>0$ under the index condition (Theorems 1', 2); general $R$: lower bound proven, upper bound empirical (3 random-$R$ cases) |
| S8 | "Order $=k^*$" fails from $k^*=5$ on. MPS6 pair-star (order 4, $k^*=5$, correct order, two residuals) refutes the multi-residual use; MPS6 $110001$ wrong order (4 vs 6) and HT4 $1111$ wrong order (6 vs 7) refute R2 itself and 4.1's "$m\ge3$: no escape route" | proven: upper bounds by Theorem 2 alone, exact orders with Theorem 1'; checked numerically to 3 digits by the note and both verifiers |
| S9 | Escape time $\propto\sigma^{-(2\ell-2)}$ when the generalized mechanism wins | derived (reduced model); checked for one teacher: 5 seeds from $\theta_S+\sigma\xi$, and from small init only on $q$-first seeds, for $\sigma\lesssim0.005$ (section 6 caveats) |
| S10 | Numerical saddle orders on 32 instances equal the predictions | empirical (12 instances reproduced independently) |
| S11 | After a generalized-piggyback escape the learned $q$ is forgotten, the competing structure is learned, and $q$ is relearned (pair-star; HT4 wrong order) | empirical (pair-star: numerics verifier; HT4 $D$ = all sites, $\chi=2$: 6 of 7 wrong-order locality runs, all at $\sigma\ge0.4$); consistent with rule 5.1-3 |
| S12 | Wrong-order HT4 and binary8 with $D$ = all sites: order 6 $<k^*=7$, escape exponent 4 below the crossover $b\approx4\kappa/(5s_r)$ | order proven (Cor. 2c); exponent derived; untested below the crossover |

Open: the borderline $\lambda_{\max}(\mathcal S)=s_q$ ($k^*\ge4$); Theorem 2 without the index
condition or with signed weights; several learned components ($W_S$ of higher rank: sectors must be
built from the learned column/row spaces of every edge, and the factorization in Lemma 2 needs the old
part to be a product across a core, which fails); the length of the plateau after the forgetting
(section 6); the exponent about 4 of half the control seeds; the dynamics below the crossover for
wrong-order HT4/binary8 (S12).

