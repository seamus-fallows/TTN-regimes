# Theory: the saddle order at a learned rank-one saddle (lower bound k*, and when it fails)

Status: **v1, 2026-10-10** (theorist), for adversarial verification. Answers the open issue of
`theory_saddles.md` section 11 ("saddle order = $k^*$, lower bound") for one learned component.
Tags: **proven**, **derived**, **empirical**, **conjecture**, as in `notes/theory_saddles.md`.
Numerics: `experiments/theory_checks/saddle_order/` (README + scripts).

## 0. Summary

**Answer in one paragraph.** The claim "saddle order $=k^*$" (R2 of `theory_saddles.md`) is
**false in general** but true under a checkable spectral condition, and **proven** here in that
form. Whether the order equals $k^*$ is decided by a *generalized piggyback operator* $\mathcal S$
that couples sector amplitudes on small core sets (single cores, and, for $k^*\ge5$, sets whose
Steiner tree has fewer than $k^*/2$ cores). If $\lambda_{\max}(\mathcal S)<s_q$ then
$L-L_S\ge -C\lVert\theta-\theta_S\rVert^{k^*}$ (Theorem 1), so with the route curve the order is
exactly $k^*$. If $\mathcal S$ restricted to sets of degree $\le\ell<k^*/2$ has
$\lambda_{\max}>s_q$, there is a descent curve of order $2\ell<k^*$ (Theorem 2). Consequences:
(i) for $k^*\le4$ the condition is the old-block (classical piggyback) condition $\rho(S)<s_q$, so
order $=k^*$ is proven for $k^*=3,4$ whenever the old block is positive definite modulo gauge;
(ii) for two-component teachers in the correct order ($s_r<s_q$) the order is $k^*$ on every tree;
(iii) counterexamples exist from $k^*=5$ on: MPS6, $q=000000$ (1), $r_1=110001$, $r_2=110010$
(0.8 each) has $k^*=5$, a PSD Hessian, and saddle order 4 (numerics: slope 4.000, constant
$-0.0350$ predicted $-0.0350$); a single wrong-order residual $110001$ (1.5) on MPS6 has $k^*=6$
and order 4. Numerics on 32 instances (section 5; trees mps4-6, binary4/6, ht4, $k^*=2..7$, pattern
and general residuals) agree with Theorems 1-2 everywhere, and gradient flow on the pair-star escapes
with $t_{\rm esc}\propto\sigma^{-2}$ instead of the route law $\sigma^{-3}$ (section 6).

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
- **Degree** of a nonempty set $C$: $\deg C=1$ if $\lvert C\rvert=1$, else $\lvert\mathrm{Steiner}(C)\rvert$.
  **Low sets**: $\mathcal L=\{C\ne\emptyset:\deg C<k^*/2\}$; $\mathcal L_\ell=\{C:\deg C\le\ell\}$.
  For $k^*\le2$, $\mathcal L=\emptyset$; for $k^*=3,4$, $\mathcal L$ = singletons; for $k^*=5,6$ also
  sets with $\lvert\mathrm{Steiner}\rvert=2$ (two adjacent cores, or one core plus a neighbour); etc.
- **Generalized piggyback operator** $\mathcal S$ on $\mathcal H=\bigoplus_{C\in\mathcal L}T(C)$:
  block $(C_1,C_2)$ = the bilinear form $(x,y)\mapsto\langle R_{C_1\sqcup C_2},x\otimes y\rangle$
  if $C_1\cap C_2=\emptyset$, zero otherwise (zero diagonal blocks, so $\lambda_{\max}(\mathcal S)\ge0$).
  Restricted to singletons it is the classical piggyback operator $S_1$ of `theory_saddles.md` 4.1
  (blocks $R_{\{u,v\}}$). For a pattern teacher, in the pattern basis $\mathcal S$ is the adjacency
  matrix of the **generalized piggyback graph**: nodes $(C,w)$ ($C\in\mathcal L$, $w$ a pattern on
  $C$ differing from $q$ on every core of $C$), one edge of weight $s_r$ between $(C_1,r|_{C_1})$ and
  $(C_2,r|_{C_2})$ for every residual $r$ and every split $T_r=C_1\sqcup C_2$ with $C_1,C_2\in\mathcal L$;
  $\lambda_{\max}(\mathcal S)=\rho(\text{graph})$.
- **Saddle order** $p$: the largest $p$ with $L(\theta)-L_S\ge-C\lVert\theta-\theta_S\rVert^p$ near
  $\theta_S$ (equivalently, no curve with $\lVert\theta(t)-\theta_S\rVert\asymp t$ descends faster than $t^p$).

## 2. Results

**Theorem 1 (lower bound; proven).** Assume $\theta_S$ critical and $\lambda_{\max}(\mathcal S)<s_q$.
Then there are $C,r_0>0$ with
$$L(\theta)-L_S\ \ge\ -C\,\lVert\theta-\theta_S\rVert^{k^*}\qquad\text{for }\lVert\theta-\theta_S\rVert<r_0 .$$
If moreover $\chi_e\ge2$ on every edge of $\mathrm{Steiner}(B)$ for some $B$ with $R_B\ne0$ and
$\lvert\mathrm{Steiner}(B)\rvert=k^*$, the route curve gives $L-L_S=-ct^{k^*}+O(t^{2k^*})$
(proven, `theory_saddles.md` section 2; any nonzero $R_B$ has a nonzero overlap with some product
tensor), so the saddle order is exactly $k^*$.

**Theorem 2 (sharpness; proven under an index condition).** Pattern teacher. Suppose that for some
$\ell<k^*/2$ the generalized piggyback graph restricted to $\mathcal L_\ell$ contains a set $X$ of nodes with
$\rho(\text{graph}|_X)>s_q$, and that every edge $e$ satisfies $\chi_e-1\ge$ (number of multi-core
nodes $(C,w)\in X$ such that $e$ is an edge of $\mathrm{Steiner}(C)$). Then there is a curve with
$\lVert\theta(t)-\theta_S\rVert\le t$ and $L(\theta(t))-L_S\le-ct^{2\ell}$: the order is $\le2\ell<k^*$.
(Without the index condition: derived, not proven.)

**Corollary 1 ($k^*\le4$; proven).** $\mathcal S=S_1$, the classical piggyback operator, i.e. the
old (fixed-rank) block of the Hessian. If $\lambda_{\max}(S_1)<s_q$ (pattern teacher: $\rho(S)<s_q$;
equivalently the old block is positive definite modulo the old gauge) the order is exactly $k^*$
(given the route). If $\lambda_{\max}(S_1)>s_q$ the saddle is strict (order 2). So for $k^*=3,4$:
order $=k^*$ iff $\rho(S)<s_q$ (boundary case $\rho(S)=s_q$ open; one numerical instance, MPS4 with
$s_r=s_q$, still has order 4).

**Corollary 2 (two-component teachers; proven).** One residual $r$: every node of the generalized
graph has at most one edge (node $(C_1,r|_{C_1})$ is joined only to $(T_r\setminus C_1,\cdot)$), so
$\rho=s_r$ if some split $T_r=C_1\sqcup C_2$ has $\deg C_1,\deg C_2<k^*/2$ and $\rho=0$ otherwise.
(a) Correct order $s_r<s_q$: order $=k^*$ on every tree, for every differing set $D$ and every $k^*$.
(b) Wrong order $s_r>s_q$ with no such split (e.g. $m=\lvert T_r\rvert\ge3$ with all terminals far apart: MPS5,
$r=10101$): order $=k^*$. (c) Wrong order with such a split: order $\le2\ell^*$,
$\ell^*=\min_{\text{splits}}\max(\deg C_1,\deg C_2)$, under the index condition (e.g. $m=2$: strict,
order 2; MPS6, $r=110001$, split $\{0,1\}|\{5\}$: order $4<k^*=6$).

**Corollary 3 (where R2 fails).** The first failures need $k^*\ge5$ and a residual with
$m\ge3$ terminals, two of which lie in a set of degree $<k^*/2$ (adjacent cores, for $k^*=5,6$).
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

## 4. Proof of Theorem 2 (explicit curve)

Pattern teacher, standard site basis, $q=0\cdots0$, $a=s_q^{1/K}$ (balanced; any $\theta_S$ in the
gauge orbit gives the same order). Let $A_X$ be the adjacency matrix of the generalized graph on $X$,
$x\ge0$ a unit Perron vector, $A_Xx=\rho x$, $\rho>s_q$. For $0<\eta\le1$ set $\theta(\eta)=\theta_S+\delta$:
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
$\tfrac12\lVert\Delta\rVert^2-\langle R,\Delta\rangle$ with $\Delta=W-W_S$ (no cancellation). No gauge quotient is needed: $m(t)$ is a minimum over a ball, gauge moves leave $L$ unchanged, and
distances to $\theta_S$ and to its (smooth) gauge orbit agree up to constants, so the order is the same.
Order =
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

## 6. Consequence for the dynamics (derived; checked on one teacher)

Where Theorem 2 applies at level $\ell$, the old rotations have $O(1)$ curvature and follow the
cluster adiabatically ($z\approx s\,y$ in Cor. 3), leaving
$L_{\rm eff}=-\tfrac c2y^2$, $y=\prod_{v\in P}b_v$ ($\ell$ new entries),
$-c<0$ the Schur complement of the section-3 quadratic form on the cluster variable after eliminating the old rotations ($c=s_1^2+s_2^2-1$ for the pair-star, units $s_q=1$). Then
$\dot b_v=c\,b_v\prod_{u\ne v}b_u^2$, homogeneous of degree $2\ell-1$, so from amplitudes
$b\sim\sigma$ the escape takes $t_{\rm esc}\propto\sigma^{-(2\ell-2)}$ ($\log(1/\sigma)$ for
$\ell=1$, the classical piggyback). This replaces the route law $\sigma^{-(k^*-2)}$ when $2\ell<k^*$.
The escape direction is a **non-residual** pattern (pair-star: $110000$) together with old rotations
toward the residuals, so the next plateau is not "$r$ learned" (what it is was not studied).

Check (`gf_escape.py`, gradient flow from $\theta_S+\sigma\xi$, LSODA rtol $10^{-10}$, escape =
first time $L-L_S<-10^{-3}$, $\sigma=0.04\to0.005$, 3 seeds). Local exponents
$\log_2(t_{\rm esc}(\sigma/2)/t_{\rm esc}(\sigma))$:

| teacher (MPS6, $s_q=1$) | prediction | seed 0 | seed 1 | seed 2 | amplitude of $110000$ at escape |
|---|---|---|---|---|---|
| pair-star (0.8, 0.8), $\rho=1.13$ | 2 | 2.15, 2.07, 2.03 | 2.15, 2.08, 2.04 | 1.98, 1.99, 2.00 | $\pm0.085$ |
| control (0.6, 0.6), $\rho=0.85$ | 3 (route $k^*=5$) | 3.15, 3.33, 3.54 | 2.86, 2.93, 2.97 | no escape by $t=10^7$ ($\sigma=0.04$): $L-L_S\to0^+$, residual amplitudes $\sim10^{-28}$ (trapped) | $\le0.002$ |

The control escapes through the $k^*=5$ route of $r_2=110010$ in both inspected seeds (its amplitude
grows; $r_1$ stays at 0). Its seed-0 exponent drifts above 3; this is not explained by the saddle
order and is probably the pinned-route effect of `theory_saddles.md` R12 (random $Q_e\ne0$ from the
perturbation); not investigated. With the same init directions the pair-star escapes 490 and 93 times earlier at $\sigma=0.005$ (seeds 0, 1), and also from the seed-2 direction (where the control's routes die), at $t=5959$ for $\sigma=0.04$.

## 7. What should change in `notes/theory_saddles.md` (not edited here)

1. **R2 (results table) and Prop. 2's paragraph "What is and is not proven".** Replace "order $=k^*$
   derived; lower bound open beyond order 3" by: order $\ge k^*$ **proven** whenever
   $\lambda_{\max}(\mathcal S)<s_q$ (Theorem 1), hence order $=k^*$ proven with the route; this covers
   all $k^*\le4$ with a positive-definite old block (Cor. 1) and all two-component teachers in the
   correct order (Cor. 2a). Add: R2 is **false** in general from $k^*=5$ on (Cor. 3, Theorem 2).
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
   otherwise the exponent is $2\ell^*-2$ with $\ell^*$ the smallest level where $\mathcal S$ exceeds $s_q$.
5. **Section 4.1 (piggyback graph) and 4.3 table.** The piggyback graph is the $\ell=1$ level of
   the generalized graph. Row "$s_q<s_r$, $m\ge3$: PD block, escape by the route" needs: "order
   $\min(k^*,2\ell^*)$, $\ell^*$ from the splits of $T_r$" (Cor. 2b-c); e.g. MPS6, $D=\{0,1,5\}$: order
   4, not 6. For $N\le4$ (all current experiments with two components) nothing changes ($k^*\le4$).
6. **Section 11, first open issue.** Close it (Theorems 1-2) and replace by: the borderline
   $\lambda_{\max}(\mathcal S)=s_q$ for $k^*\ge4$; Theorem 2 without the index condition; several
   learned components ($\lvert S\rvert\ge2$, where $W_S$ is not rank one across the core partition and
   the sector decomposition must be replaced by the learned column/row spaces).
7. **Section 9 (literature table), row R2.** Add: the lower bound and the counterexamples are new
   (not found in literature.md); the tools (set-partition moment-cumulant formulas, factorization of
   cumulants of independent blocks) are classical.
8. **Section 5.1 (rule for the whole sequence).** Step "next escape = smallest $k^*$" should compare
   $k^*$ with $2\ell^*$: the next event is the smaller of the two (log for 2, power $p-2$ otherwise).
9. **CLAUDE.md / report** (for the lead): Q3 status line "saddle order $=k^*$ derived" becomes
   "proven iff the generalized piggyback condition holds (always for $k^*\le4$ and for two-component
   teachers in the correct order); counterexamples with order $4<k^*$ from $k^*=5$".

## 8. Claims and status

| # | claim | status |
|---|---|---|
| S1 | Exact identity $L-L_S=\tfrac12(N-s_q)^2+\sum_B[\tfrac{N^2}2\lVert m_B\rVert^2-N\langle R_B,m_B\rangle]$ (Lemma 1) | proven |
| S2 | Locality: $\lVert\kappa_B\rVert\le C\prod_{x\in\mathrm{Steiner}(B)}\lVert\nu_x\rVert$ for $\lvert B\rvert\ge2$ (Lemma 2) | proven |
| S3 | Theorem 1: $\lambda_{\max}(\mathcal S)<s_q\Rightarrow L-L_S\ge-C\lVert\delta\rVert^{k^*}$; with the route, order $=k^*$ (any tree, any $\chi$, any residual with $R_{\{v\}}=0$, balanced or not) | proven |
| S4 | Cor. 1: $k^*\le4$: order $=k^*$ iff $\rho(S)<s_q$ (strict inequality; $k^*=3$ also at equality) | proven ($k^*=4$, $\rho=s_q$: open; one instance has order 4) |
| S5 | Cor. 2: two-component teachers, correct order: order $=k^*$ on every tree | proven |
| S6 | Theorem 2: generalized graph with $\rho>s_q$ at level $\ell<k^*/2$ $\Rightarrow$ order $\le2\ell$ (pattern teacher, index condition) | proven |
| S7 | Order $=\min(k^*,2\ell^*)$ in general (Theorems 1+2 combined, generic $\lambda\ne s_q$, index condition) | proven for pattern teachers under the index condition; general $R$: lower bound proven, upper bound empirical (3 random-$R$ cases) |
| S8 | R2 fails from $k^*=5$ on: MPS6 pair-star (order 4, $k^*=5$), MPS6 $110001$ wrong order (4 vs 6), HT4 $1111$ wrong order (6 vs 7) | proven (Thm 2), checked numerically to 3 digits |
| S9 | Escape time $\propto\sigma^{-(2\ell-2)}$ when the generalized mechanism wins | derived (reduced model), checked for one teacher (3 seeds) |
| S10 | Numerical saddle orders on 32 instances equal the predictions | empirical |

Open: the borderline $\lambda_{\max}(\mathcal S)=s_q$ ($k^*\ge4$); Theorem 2 without the index
condition; several learned components ($W_S$ of higher rank: sectors must be built from the learned
column/row spaces of every edge, and the factorization in Lemma 2 needs the old part to be a product
across a core, which fails); what the trajectory does after a generalized-piggyback escape.

