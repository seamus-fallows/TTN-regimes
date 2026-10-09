# Theory: saddles after the first escape, and locality (Q3; H2, H3, H1b)

Status: **v1, 2026-10-09** (theorist for Q3). Scratch scripts for every numerical check are in
`C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/theory/saddles/`
(outside the repo; they use only `ttn.tree`, `ttn.network`, `ttn.teachers`, `ttn.losses`). Tags:
**proven** (complete argument given), **derived** (under the stated assumptions, leading order in the
small parameters), **heuristic**, **conjecture**; **checked** marks a numerical confirmation (section 7).

Literature caveat: `notes/literature.md` did not exist while this was written, and the shared web-search
budget ran out early, so the literature check consisted of reading a handful of papers directly
(abstracts or full text; cited below with arXiv ids). Claims marked "not found in the literature" are subject to the literature review.

## 0. Results at a glance

| # | result | status |
|---|---|---|
| R1 | Around a learned saddle, the new parameters enter $L$ through *routes*: connected core sets $C$ whose new entries combine into a term $\propto$ the residual. The lowest route degree $k^*=\min\lvert C\rvert$ sets the saddle type and the escape-time law. | derived |
| R2 | One learned component $q$ (balanced), new component $r$: $k^*=\lvert\mathrm{Steiner}(N(D))\rvert$, where $N(D)$ are the cores carrying the sites where $r$ differs from $q$; the edges whose rank must grow are exactly the Steiner edges. | proven |
| R3 | $k^*=1$: not a critical point (no plateau). $k^*=2$: strict saddle, $\lambda=s_r\,s_q^{(K-2)/K}$ exactly. $k^*\ge3$: no negative curvature along the route (higher-order saddle). | proven (at $\theta_S$), checked |
| R4 | Escape times: $k^*=2$: $\Delta t\simeq(1-s_r/s_q)\log(1/\sigma)/\lambda$; $k^*\ge3$: $\Delta t\simeq\sigma^{-(k-2)}(1-s_r/s_q)/[(k-2)\,s_r\,s_q^{(K-k)/K}]$ (aligned init). The factor $(1-s_r/s_q)$ is a head start accumulated during the previous escape. | derived, checked to <1% (aligned) |
| R5 | For random init the remaining plateau is predicted to 1-3% by the $k$-scalar route ODE started from the route amplitudes measured after the previous event; the exponent $k-2$ holds per seed; the growth exponent $d\log\dot a_r/d\log a_r=2-2/k$ identifies $k$ from one run. | derived, checked |
| R6 | Piggyback block (old bond index, rotated physical vectors), $m$ = number of cores carrying the differing sites: $m=2$: eigenvalues $s_q^{(K-2)/K}(s_q\mp s_r)$; $m\ge3$: positive definite. Binary $N=4$ is exactly a depth-3 deep linear network, so H1b there is a case of Achour et al. (2024, Thm 7). | proven, checked |
| R7 | Wrong order (smaller $q$ learned first) is a strict saddle iff some larger $r$ has $m=2$ or any $r$ has $k^*=2$. Three plateau sequences, depending on $(k,m)$. The H1b "correction" ($L_0\to s_r^2/2\to s_q^2/2\to0$) requires $m=2$, $k\ge3$. | derived, checked |
| R8 | Same teacher, different trees: the next escape is the route with the smallest $k^*$, not the largest weight (locality beats weight); ties in $k^*$ are broken by random route amplitudes. Worked examples for $N=4,8$ (section 5). | derived, checked ($N=4,8$) |
| R9 | With several learned components the escape degree can be smaller than the number of cores needed to fit the component exactly ("partial learning"): an extra plateau at the best fit of a lower rank profile appears. | derived, checked |
| R10 | Hessian: gauge orbit at $\theta_S$ has dimension $\sum_e[\chi_e^2-(\chi_e-r_e)^2]$; null space $\sum_e r_e^2+P_{\rm new}-2\sum_{k=2}c_e$. At a plateau point the gauge-projected Hessian has one negative eigenvalue per active route (others, if any, $O(\sigma)$): $-\lambda$ ($k=2$) or $\approx-(k-1)s_rA\,b^{k-2}=O(\sigma^{k-2})$ ($k\ge3$). | proven (counts at $\theta_S$), derived (plateau), checked |
| R11 | The 1e-3 relative threshold of `notes/diagnostics.md` classifies $k=3$ plateaus as strict for $\sigma\gtrsim0.01$; use the $\sigma$-scaling of $\lambda_{\min}$ instead. | checked |
| R12 | Conservation-law trap: with fan-in init, $Q_e\ne0$ can kill a route during a long plateau (new bond direction of the lighter core decays to 0, the heavier one freezes); seen for every seed on HT4 ($k=5$); gauge-balanced init avoids it. Matters for $k\ge4$. | derived mechanism, checked (3 seeds) |
| R13 | Generic (non-orthogonal) teachers: adjacent pairs with free modes give $k^*=2$; on an MPS every later saddle is strict (log-time, edge-by-edge rank growth), binary/HT trees keep $k^*\ge3$ at rank one. Power-law plateaus need tree-aligned orthogonal structure. | derived, checked (one teacher) |

## 1. Setting and notation

- Tree with cores $v$ ($K$ of them), sites $i=0..N-1$ (physical dimension $d$), edges named by their
  child node $e$; bond dimension $\chi_e$; parameters $\theta=(G_v)$; $P$ = number of parameters.
- Loss (whitened inputs): $L(\theta)=\tfrac12\lVert W(\theta)-W^*\rVert_F^2$.
- **Pattern teacher**: $W^*=\sum_r s_rU_r$, $U_r=\bigotimes_i b_i[:,p_r(i)]$ with orthonormal site
  bases $b_i$ (`ttn.teachers.pattern_teacher`); distinct patterns give orthonormal $U_r$.
- For two components $q,r$: differing set $D=\{i:p_r(i)\ne p_q(i)\}$; **terminals** $N(D)$ = cores
  carrying a site of $D$, $m=\lvert N(D)\rvert$; $\mathrm{Steiner}(N(D))$ = smallest subtree containing
  $N(D)$ (`Tree.steiner_nodes`).
- **Learned saddle**: learned set $S$, $W_S=\sum_{q\in S}s_qU_q$, $r_e$ = rank of $W_S$ matricized
  across $e$, $\theta_S$ = a **balanced** ($Q_e=0$ on every edge) TTN of $W_S$ with bond ranks $r_e$,
  padded with zeros to $\chi_e$. Residual $R=W^*-W_S$.
- Entries of $\theta$ with all bond indices $<r_e$ are **old**, the others (at least one index
  $\ge r_e$) **new**; $c_e=\chi_e-r_e$ new indices per edge. $P_{\rm old}$, $P_{\rm new}$ count them.
- One learned component $q$: balancedness forces equal core norms, so every core is
  $a\cdot(\text{pattern of }q)\otimes e_0\otimes\dots$ with $a=s_q^{1/K}$ (**proven**: $Q_e=0$ with
  bond dimension 1 means $\lVert G_v\rVert=\lVert G_{p}\rVert$ on every edge).

Assumptions used throughout: (A1) whitened inputs; (A2) pattern teacher; (A3) $\theta_S$ balanced
(gradient flow from small init conserves $Q_e=O(\sigma^2)$, so plateaus are balanced to that order);
(A4) $\chi_e\ge r_e+1$ on every edge whose rank must grow (otherwise the route does not exist);
(A5) leading order in the new entries (all of size $\sigma_{\rm eff}\ll1$ on the plateau).

## 2. Expansion around a learned saddle

For a core set $J$ let $W_J[\delta]$ be $W$ with the cores in $J$ replaced by $\delta_v$ and all others
at $\theta_S$. Multilinearity gives exactly

$$L(\theta_S+\delta)=L_S-\sum_{J\neq\emptyset}\langle R,W_J[\delta]\rangle+\tfrac12\Big\lVert\sum_{J\neq\emptyset}W_J[\delta]\Big\rVert^2 .$$

**F1 (new entries are invisible at first order; proven).** If $\delta_v$ has only new entries then
$W_v[\delta_v]=0$: its new index meets a zero block of the neighbouring learned core. New entries
therefore carry no Gauss-Newton (GN) cost at second order.

**F2 (routes; proven).** For new entries, $W_J[\delta]\ne0$ only if $J$ is connected and the new
indices match across the internal edges of $J$; then
$W_J[\delta]=\Phi_J(\delta)\otimes\bigotimes_{f\in\partial J}T_f$, where $\Phi_J$ is the small TTN
formed by the $\delta_v$ (internal bond dimensions $c_e$) and $T_f$ is the contraction of the learned
subtree hanging off the boundary edge $f$ (a map from the $r_f$ old indices of $f$ to tensors on the
hanging sites $H_f$).

**Definition (local residual).** For a connected core set $C$,
$R_C=\langle R,\bigotimes_{f\in\partial C}T_f\rangle$, a tensor with one mode per core of $C$ (its
physical legs and its old boundary legs). Then $\langle R,W_C[\delta]\rangle=\langle R_C,\Phi_C(\delta)\rangle$.

**Proposition 1 (escape degree; derived).** The new entries first lower $L$ at order
$k^*=\min\{\lvert C\rvert: C\text{ connected},\ R_C\ne0\}$. To leading order (old entries frozen at
$\theta_S$), the new entries on the minimizing $C$ follow the gradient flow of a $k^*$-core TTN fitting
the target $R_C$ from small initialization: **the escape from a saddle is the first-escape problem of
the sub-TTN on $C$** (theory_escape applies with $K\to k^*$, target $R_C$). This is the TTN instance of
directional convergence of the small weights near saddles (Kumar & Haupt, TMLR 2024, arXiv:2402.09226,
for 2-homogeneous nets) and of the accumulated-utility rule of Alternating Gradient Flows (Kunin et al.
2025, arXiv:2506.06489): a dormant $\kappa$-homogeneous unit activates when
$\int\kappa\,\mathcal U\,dt$ reaches $(\lVert\theta_0\rVert^{2-\kappa}-1)/(\kappa-2)$ ($\log(1/\lVert\theta_0\rVert)$
for $\kappa=2$; their time normalization), where $\mathcal U$ is the unit's correlation with the
residual, accumulated over *all* earlier phases. What is TTN-specific is the degree: $\kappa=k^*$, set by the tree.

**Proposition 2 (Steiner rule; proven for one learned component).** If $S=\{q\}$, the hanging
contractions are $T_f=a^{n_f}U_q|_{H_f}$ ($n_f$ cores), so
$$R_C=a^{K-\lvert C\rvert}\sum_{r:\,N(D_r)\subseteq C}s_r\,U_r|_C .$$
Hence $k^*=\min_r\lvert\mathrm{Steiner}(N(D_r))\rvert$ and the route of $r$ has coupling
$c_r=s_r\,a^{K-k}=s_r\,s_q^{(K-k)/K}$ ("$s_{\rm new}$ times the learned amplitudes outside $C$").
The rank of $W_S+s_rU_r$ across $e$ exceeds that of $W_S$ iff $D_r$ has sites on both sides of $e$,
i.e. iff $e$ is a Steiner edge; so $E_+$ (edges whose rank must grow) = Steiner edges and
$k=\lvert E_+\rvert+1$; $k=1$ iff $D_r$ sits on one core.

**Lemma 3 (general $S$; proven).** Let $E_+$ = edges where
$\mathrm{rank}_e(W_S+U_r)>\mathrm{rank}_e(W_S)$ (i.e. both restrictions of $U_r$ lie outside the
learned column/row spaces). (i) $E_+$ is connected: "new on $X$" implies "new on any superset of $X$",
so every edge between two $E_+$ edges is in $E_+$. (ii) If $E_+=\emptyset$ there is a core all of
whose branches are "old" (walk towards new branches; the walk cannot return), that core alone can
produce $U_r$, so $\nabla L(\theta_S)\ne0$: no saddle.

**Several learned components (derived).** Define $k_{\rm ex}=\lvert E_+\rvert+1$ (cores needed to
fit $r$ exactly). Always $k^*\le k_{\rm ex}$. Strict inequality happens when a hanging bond space holds
a *superposition* of patterns (two learned components agree on one side of an edge and differ on the
other) that overlaps $U_r|_{H_f}$ without containing it: a cheaper route then learns the projection
first (*partial learning*, section 5.5). With equality, the coupling is
$$c_r=s_r\prod_{f\in\partial C}\big\lVert T_f^{\top}\,U_r|_{H_f}\big\rVert\qquad(\text{balanced gauge}),$$
checked against the exact Hessian for single- and multi-component saddles (section 7, check C1b).

## 3. Saddle type and escape time, by route degree

Notation: $a=s_q^{1/K}$; route amplitudes $b_v$ ($v\in C$) = the new entries along the route
(physical pattern of $r$ on $v$'s sites, new index on the internal edges of $C$, old index on the
boundary edges); coupling $c=s_rA$ with $A=a^{K-k}$ (one learned component) or the general product
formula of section 2. Reduced dynamics of a route (derived from F1, F2 with old entries frozen):
$$\dot b_v=A\prod_{u\in C\setminus v}b_u\,\Big(s_r-A\prod_{u\in C}b_u\Big),\qquad a_r(t)=A\prod_{u\in C}b_u/s_r .$$
It conserves $b_u^2-b_v^2$ (the route's own balancedness).

### 3.1 $k^*=1$: no saddle (proven)
$\nabla L(\theta_S)\ne0$ (its norm is $s_r a^{K-1}$ for one learned component; checked: 0.500 for
$s_r=0.5$, $s_q=1$). The core carrying $D$ learns $r$ in $O(1)$ time, during or right after the rise of
$q$: one event, no plateau (checked: $\Delta t_2=0.02$ on binary4, $D=\{0,1\}$, $\sigma=0.2$).

### 3.2 $k^*=2$: strict saddle (proven at $\theta_S$)
$C=\{u,v\}$ adjacent, shared edge $e$. On the new entries the Hessian is
$\begin{pmatrix}0&-c\,I_{c_e}\\-c\,I_{c_e}&0\end{pmatrix}$, eigenvalues $\pm c$, each with multiplicity
$c_e=\chi_e-r_e$; the GN term vanishes by F1. With one learned component
$$\lambda=s_r\,s_q^{(K-2)/K}\qquad(\text{exact; general: }\lambda=\sigma_{\max}(R_C)).$$
Balanced reduced dynamics: $a_r$ obeys $\dot a_r=2\lambda\,a_r(1-a_r)$, so
$$\Delta t=\frac{1}{\lambda}\Big[\log\frac{1}{b_{\rm eff}}+\tfrac12\log\frac{s_r}{A}\Big]\quad(a_r:\ \text{start}\to\tfrac12),$$
with $b_{\rm eff}$ the projection of the route entries on the unstable eigenvector when the plateau
starts.

**Head start (derived).** The same route existed during the previous escape (with the then-small
learned cores in its hanging part), and grew at the instantaneous rate $c(t)=s_r\,a(t)^{K-2}$ while $q$
rose as $\dot a=s_qa^{K-1}$ (assumes the hanging cores are already aligned with the shared pattern,
i.e. after the alignment phase of the first escape; the alignment phase only adds $O(1)$). Hence $\log b_{\rm eff}=\log b_0+\int c\,dt=\log b_0+(s_r/s_q)\log(a_1/a_0)$
and, for $b_0\sim a_0\sim\sigma$,
$$\boxed{\Delta t_2=\frac{(1-s_r/s_q)\,\log(1/\sigma)+O(1)}{s_r\,s_q^{(K-2)/K}}}$$
The slope in $\log(1/\sigma)$ is $(1-s_r/s_q)/\lambda$, **not** $1/\lambda$ (checked: aligned 1.0067 vs
1.0000 predicted, 2.0 without head start; random init 1.05).

### 3.3 $k^*\ge3$: higher-order saddle (proven at $\theta_S$)
No pair of route entries couples at second order, so the Hessian vanishes on the route (and, with
F1, on all new entries except $k=2$ routes). $L$ decreases only at order $k$:
$L-L_S\approx-c\prod_{v\in C}b_v$. Balanced reduced dynamics $\dot b=c\,b^{k-1}$:
$$\Delta t=\frac{b_{\rm eff}^{-(k-2)}}{(k-2)\,c}+O\!\Big(\frac1c\Big(\frac{s_r}{A}\Big)^{\frac{k-2}{k}}\Big);\qquad\text{unbalanced }(\textstyle\prod b>0):\ \Delta t=\int_0^\infty\frac{d\beta}{2c\prod_{v\in C}(b_v^2+\beta)^{1/2}} .$$
(The integral follows from $b_v^2=b_v(0)^2+\beta(t)$, $\dot\beta=2c\prod b$.) If $\prod b<0$ at the start,
the smallest $\lvert b_v\rvert$ first shrinks through zero (sign correction), which lengthens the
plateau; this is one source of the broad seed-to-seed spread.

**What $\sigma_{\rm eff}$ is after the earlier phases (derived).** With accumulated utility,
$b_{\rm eff}^{-(k-2)}=b_0^{-(k-2)}-(k-2)\int_{\rm earlier}c(t)\,dt$. If the previous phase was the
escape of $q$ from the origin (route coupling $s_r a(t)^{K-k}$, $\dot a=s_qa^{K-1}$):
$$b_{\rm eff}^{-(k-2)}=b_0^{-(k-2)}-\frac{s_r}{s_q}\,a_0^{-(k-2)}\ \Longrightarrow\ \boxed{\Delta t_2=\frac{\sigma^{-(k-2)}\,(1-s_r/s_q)}{(k-2)\,s_r\,s_q^{(K-k)/K}}}\quad(b_0=a_0=\sigma).$$
So $\sigma_{\rm eff}=\sigma(1-s_r/s_q)^{-1/(k-2)}$ in the aligned case. Checked to 0.4% at
$\sigma=2\cdot10^{-3}$ (table in section 7). Special case $k=K$ (route uses every core): the route never
sees $q$, $c=s_r$ throughout, and $r$ escapes at the *absolute* time $\sigma^{-(K-2)}/((K-2)s_r)$
(odeco-like independence).

**Measurable signatures (derived, checked).** (i) Per seed (same init direction, $\sigma$ varied)
$\Delta t\propto\sigma^{-(k-2)}$. (ii) During the plateau, $\dot a_r\propto a_r^{2-2/k}$ once the route is
balanced, so the local exponent $\gamma=d\log\dot a_r/d\log a_r\to2-2/k$ (1, 4/3, 3/2, 8/5 for
$k=2..5$): $k$ can be read off one run. (iii) Random init: measure the $b_v$ on the plateau and
integrate the $k$-scalar route ODE; this predicts the remaining plateau to 1-3%.

### 3.4 Which escape comes next (derived)
At small $\sigma$ the next event is the route with the smallest $k^*$ (log beats power, and smaller
powers beat larger ones). Ties in $k^*$ are decided by $c\,b_{\rm eff}^{k-2}$, i.e. by the coupling
*and* the random route amplitudes; for $k^*\ge3$ the latter usually dominate, so tied components come
in a seed-dependent order (HT4 example B4, balanced init: $q_A$ (0.3) before $q_B$ (0.6), both $k=5$).
Routes of higher
degree accumulate only a vanishing relative head start during a lower-degree plateau. The weight
order is therefore overridden by locality whenever a smaller component has a smaller $k^*$
(section 5). This is the TTN analogue of the "leap" picture of Abbe, Boix-Adserà & Misiakiewicz
(2023, arXiv:2302.11055): there the time to learn a new set of coordinates scales with the leap
(number of new coordinates needed at once); here the exponent is set by the number of *cores* that
must change at once, which depends on the tree. We did not find the tree-dependent exponent
$k^*-2$ in the literature (Razin, Maman & Cohen 2022 give norm dynamics
$\dot\sigma=\sigma^{2-2/L_\nu}L_\nu\langle-\nabla,\mathcal C\rangle$ for local components of their
CP-per-node hierarchical factorization, $L_\nu$ = number of children + 1, but no escape exponents
versus tree position).

## 4. Piggyback directions and the wrong learning order (H1b)

### 4.1 Piggyback block (proven, one learned component)
Piggyback directions keep the learned bond index and rotate the physical vectors of the terminal
cores toward $r$: $\delta G_u=\varepsilon_u\,(\text{pattern of }r\text{ on }u)\otimes e_0\otimes\cdots$,
$u\in N(D)$. Each single $\varepsilon_u$ creates a mixed tensor ($r$ on $u$, $q$ elsewhere) of norm
$a^{K-1}\varepsilon_u$ (GN cost $a^{2(K-1)}\varepsilon_u^2$); $U_r$ itself appears at order $m$:
$a^{K-m}\prod_{u\in N(D)}\varepsilon_u$.

- $m=2$ (terminals $u,v$): the block $\begin{pmatrix}a^{2(K-1)}&-s_ra^{K-2}\\-s_ra^{K-2}&a^{2(K-1)}\end{pmatrix}$ is an invariant subspace of the Hessian, with eigenvalues
$$a^{K-2}(s_q-s_r),\qquad a^{K-2}(s_q+s_r),\qquad a=s_q^{1/K}.$$
For $K=3$ this is the hand derivation in PLAN.md (H1b): $q(s_1-s_2)$, $q=s_q^{1/3}$. Negative iff
$s_r>s_q$.
- $m\ge3$: no second-order coupling; the block is $a^{2(K-1)}I\succ0$; the coupling to $U_r$ is of
order $m$ and is confined by the positive quadratic term, so piggyback is not an escape route.
- $m=1$ is the $k=1$ case (gradient nonzero).

Checked exactly for 18 (tree, $D$) cases in both orders (e.g. binary4, $s_q=0.5$, $s_r=1$:
$-0.39685,+1.19055$ predicted and measured).

### 4.2 Binary $N=4$ is a depth-3 deep linear network (proven)
For the binary tree with $N=4$ (cores $A$: sites 0,1; $B$: sites 2,3; root $R$), with
$\tilde A=[\mathrm{vec}A_{:,:,\alpha}]_\alpha\in\mathbb R^{d^2\times\chi}$ and likewise $\tilde B$,
$$W_{(01)|(23)}=\tilde A\,R\,\tilde B^{\top},\qquad L=\tfrac12\lVert\tilde AR\tilde B^\top-M^*\rVert_F^2,$$
i.e. exactly a depth-3 linear network fitting the $d^2\times d^2$ matrix $M^*=W^*_{(01)|(23)}$. Every
statement about binary4 is therefore a statement about deep linear networks: saddles are the
SVD truncations of $M^*$ (Jacot et al. 2021, arXiv:2106.15933); Achour, Malgouyres & Gerchinovitz
(JMLR 2024, arXiv:2107.13289, Thm 7) prove that critical points selecting non-top singular values are
**strict** saddles, while those selecting the top $r$ values and "tightened" (which includes every
balanced one, hence every saddle visited by gradient flow from small init) are **non-strict** for depth
$\ge3$. (Their Assumption 1 asks for a full-rank target with distinct singular values; our targets are
low-rank, so our direct computation is what covers them.) H1b and H3 for binary4 are thus known
results; the content below is the extension to other trees.

### 4.3 Classification for one learned component $q$ and residual $r$ (derived; exact at $\theta_S$)

| route data | $s_q>s_r$ (correct order) | $s_q<s_r$ (wrong order) |
|---|---|---|
| $k=1$ | not critical | not critical |
| $k=2$ (then $m=2$) | strict, $\lambda_{\min}=-s_ra^{K-2}$ | strict, $\lambda_{\min}=-s_ra^{K-2}$ (piggyback $-(s_r-s_q)a^{K-2}$ is less negative) |
| $k\ge3$, $m=2$ | non-strict (order-$k$ saddle) | **strict via piggyback**, $\lambda_{\min}=-(s_r-s_q)\,s_q^{(K-2)/K}$ |
| $k\ge3$, $m\ge3$ | non-strict | non-strict |

General principle (derived): at a balanced saddle the Hessian splits into a *fixed-rank block* (old
entries: GN plus the piggyback couplings) and a *rank-increasing block* (new entries: only $k^*=2$
routes). The saddle is strict iff $W_S$ is not a local minimum of the fixed-rank approximation problem
(this is the "non-top singular values" case of Achour et al.; for TTNs it happens for coarse-grained
order-2, i.e. $m=2$, pairs) or a $k^*=2$ route exists. For $m\ge3$ the wrong-order state is a local
minimum of the fixed-rank problem, the TTN analogue of every odeco component being a local optimum of
rank-one approximation (Anandkumar et al. 2014).

### 4.4 Plateau sequences after a wrong first escape (derived, checked)
With $\varepsilon$ = piggyback amplitude (relative to $a$) and $b$ = add-route amplitude when the wrong
plateau starts:

1. **$k\ge3$, $m=2$ (e.g. odeco on binary4; $D$ = two cores at distance $\ge2$ on an MPS):**
$L_0\to\frac12s_r^2$ (wrong saddle) $\to$ rotation of the learned bond index from $q$ to $r$ after
$\Delta t\approx\log(1/\varepsilon)/[(s_r-s_q)s_q^{(K-2)/K}]$, $a_q\to0$ $\to\frac12s_q^2$ (correct
saddle) $\to$ $q$ re-added through its $k$-route after $\Delta t\propto\sigma_{\rm eff}^{-(k-2)}$
$\to0$. Requires the rotation to beat the direct add route:
$\log(1/\varepsilon)/\lambda_{\rm pb}<b^{-(k-2)}/((k-2)s_rA)$, which always holds as $\sigma\to0$.
2. **$k=2$:** $r$ is added directly (rate $s_ra^{K-2}>\lambda_{\rm pb}$): $L_0\to\frac12s_r^2\to0$,
log-time, $q$ kept.
3. **$m\ge3$:** no rotation; $r$ added through its order-$k$ route:
$L_0\to\frac12s_r^2\to0$ after $\Delta t\approx b^{-(k-2)}/((k-2)s_rs_q^{(K-k)/K})$; $a_q$ stays 1.

Checks (section 7, C5): binary4 odeco ($s_q=0.5$, $s_r=1$, $K=3$, $\lambda_{\rm pb}=0.3969$): rotation
times 23.17, 34.84, 47.44 for $\varepsilon=10^{-4},10^{-6},10^{-8}$ (spacing 11.67 and 12.60 vs
$\ln100/\lambda_{\rm pb}=11.60$), then a plateau at exactly $L=0.125=\frac12s_q^2$, $a_q$ down to
$10^{-5}$; with a large add-route seed ($b=0.1$) the add route wins (t = 9.1 vs $1/(s_rb)=10$) and
$a_q$ stays 1. The relearning time of $q$ after the rotation was $\approx2400$ for $b=\beta=10^{-3}$
(close to $1/(s_q\beta)=2000$, $\beta$ = seed of $q$ on a new index) but $\approx130$ for $b=10^{-2}$:
the former add-route entries of $r$ also seed $q$'s route after the rotation, so this last plateau is
set by the largest new-index seed (not analysed further). MPS4 odeco ($m=4$): no rotation ($\min a_q=1.000$), $L:0.5\to0$ at $t=4999.97$ vs
$b^{-2}/(2s_r)=5000$.

### 4.5 When does a wrong first escape happen at all? (derived for the local structure; heuristic for probabilities)
At the unit-sphere point of a balanced rank-one $q$ ($a_s=K^{-1/2}$), the Riemannian Hessian of
$g=\langle W,W^*\rangle$ in the $m=2$ piggyback (or $k=2$ route) block is
$a_s^{K-2}\begin{pmatrix}-s_q&s_r\\s_r&-s_q\end{pmatrix}$, so $q$ is **not** a local maximum of the
spherical flow if a larger $r$ has $m=2$ or $k=2$ ("matrix-like pair"); it is one if every larger $r$
has $m\ge3$ and $k\ge3$ (tensor-like pair, H1). Matrix-like pairs: the relative deviation from $q$
grows like $\rho^{(s_r-s_q)/s_q}$ while the radius grows from $\rho_0$ to $O(1)$, so (heuristic)
$P(\text{smaller first})\propto\sigma^{(s_r-s_q)/s_q}$ as $\sigma\to0$ (slow for close weights:
$\sigma^{0.11}$ at $s_q/s_r=0.9$), and when it happens case 1 of 4.4 applies. Tensor-like pairs:
$\sigma$-independent probability (H1), but then the wrong saddle is non-strict (case 3), no correction
happens, and the final state is still $W^*$. **So the H1b correction through a strict saddle exists only
for matrix-like pairs ($m=2$, $k\ge3$), exactly where the wrong order becomes rare as $\sigma\to0$.**

## 5. More components, nested structure, and the same teacher on different trees

### 5.1 Rule for the whole sequence (derived at leading order; heuristic as an algorithm)

1. The first escape (degree $K$ from the origin) ends at a critical point of the bond-rank-one
problem (H1 and theory_escape decide which). For pattern teachers whose components pairwise differ on
at least two cores this is a single component (the largest one for matrix-like pairs); when components
differ on a single core it is their best bond-rank-one combination (e.g. the top singular mode of a
matricization, section 5.5).
2. At every saddle compute, for each candidate route $C$, the local residual $R_C$; escape degree
$k^*=\min\lvert C\rvert$. The next event is the smallest-$k^*$ route (ties: larger coupling, larger
head start). Its plateau lasts $\propto\log(1/\sigma)$ ($k^*=2$) or $\propto\sigma^{-(k^*-2)}$; $k^*=1$
routes fire without a plateau.
3. The escape raises the bond rank by one on the internal edges of $C$; the next saddle is the best
approximation of $W^*$ in the new rank profile (for pattern teachers without superpositions: the
partial sum of the learned components; in general, e.g. nested teachers, a rank-constrained optimum
such as an SVD truncation of one matricization, section 5.5).
4. Plateau levels are the optimal values of those rank-constrained problems.

The same teacher therefore gives different sequences on different trees because $k^*$ is a property
of (teacher, tree, learned set). Below, $d=2$; bond dimension $\chi\ge$ the largest teacher rank
($\chi=3$ for the three-component examples).

### 5.2 Example 1: two components (pilot design)
$q=0^N$ ($s_q=1$), $r=1$ on $D$ ($s_r=0.5$). $k$ (and $m$) per tree:

| $D$ | binary4 ($K$=3) | mps4 ($K$=4) | ht4 ($K$=7) |
|---|---|---|---|
| {0,1} | 1 (no plateau) | 2 | 3 |
| {1,2} | 3 | 2 | 5 |
| {0,2} | 3 | 3 | 5 |
| {0,3} | 3 | 4 | 5 |
| {0,1,2,3} | 3 | 4 ($m$=4) | 7 ($m$=4) |

| $D$ | binary8 ($K$=7) | mps8 ($K$=8) | ht8 ($K$=15) |
|---|---|---|---|
| {0,1} | 1 | 2 | 3 |
| {1,2} | 3 | 2 | 5 |
| {3,4} | 5 | 2 | 7 |
| {2,5} | 5 | 4 | 7 |
| {0,7} | 5 | 8 | 7 |
| {0,1,2,3} | 3 | 4 ($m$=4) | 7 ($m$=4) |
| all | 7 ($m$=4) | 8 ($m$=8) | 15 ($m$=8) |

($m=2$ unless stated.) Predictions: $\Delta t_2=t(a_r{=}\tfrac12)-t(a_q{=}\tfrac12)\propto\sigma^{-(k-2)}$
per seed for $k\ge3$; for $k=2$ slope $d\Delta t_2/d\ln(1/\sigma)=(1-s_r/s_q)/(s_rs_q^{(K-2)/K})=1.00$;
$k=1$: separation ratio $<3$. Aligned prefactor $\Delta t_2\sigma^{k-2}=1/(k-2)$ for these weights.
$D=\{3,4\}$ is the cleanest locality contrast at $N=8$: log-time on the MPS, $\sigma^{-3}$ on the binary
tree, $\sigma^{-5}$ on HT.

### 5.3 Example 2 ("B4", $N=4$): locality reverses the weight order
$q_0=0000$ ($s=1$), $q_A=0110$ ($s=0.3$), $q_B=1001$ ($s=0.6$); $\chi=3$ (the central edge needs rank 3).
All saddles are partial sums (no superpositions; checked $k^*=k_{\rm ex}$ at every step).

| tree | predicted order | $k$ per step | gap exponents | couplings $c=s_rA$ |
|---|---|---|---|---|
| mps4 | $q_0,\,q_A,\,q_B$ | 4 (origin), 2, 4 | $\sigma^{-2}$; $\log$; $\sigma^{-2}$ | $0.3\cdot1$; $0.6\cdot1$ |
| binary4 | $q_0$, then $q_B,q_A$ (tie $k=3$; weight favours $q_B$) | 3, 3, 3 | $\sigma^{-1}$ each | $0.6$; $0.3$ |
| ht4 | $q_0$, then $q_B,q_A$ (tie $k=5$, seed-dependent) | 7, 5, 5 | $\sigma^{-5},\sigma^{-3},\sigma^{-3}$ | $0.6$; $0.3\cdot0.843$ |

MPS4 $q_A$ gap: slope $(1-0.3)/0.3=2.33$ per unit $\ln(1/\sigma)$, i.e. $+1.62$ per halving of $\sigma$.
On MPS4 $q_B$ uses all cores ($k=K$), so (aligned) it escapes at the absolute time
$\sigma^{-2}/(2\cdot0.6)$, independently of $q_0,q_A$. Binary4 is a depth-3 linear network on the
diagonal matrix $\mathrm{diag}(1,0.3,0.6)$: independent Saxe-type modes, aligned escape times
$\sigma^{-1}/s$ (1, 1.67, 3.33 in units of $\sigma^{-1}$); random init makes $q_0$ and $q_B$ nearly
simultaneous for some seeds.

Checked (random fan-in init): MPS4 seed 0, $\sigma=0.2\to0.00625$ (halving): $q_A$ gaps
3.97, 5.77, 7.45, 9.09, 10.71, 12.33, increments $1.80,1.68,1.64,1.62,1.62\to1.62$ predicted, identical
to the two-component teacher (the third component does not affect this escape). MPS4 seed 1,
$\sigma=0.1,0.05,0.025$: $q_B$ gaps 919, 3912, 15958 (local exponents $-2.09,-2.03$). Binary4 seed 1:
$q_B-q_0$ = 14.3, 29.6, 59.9 and $q_A-q_B$ = 102, 212, 430 (exponents $-1.05,-1.02$). HT4 $\sigma=0.2,0.1$:
$q_B-q_0$ = 1007, 7988 (exponent $-2.99$; predicted $-3$); $q_A$ stalled, see section 8.

### 5.4 Example 3 ("E8", $N=8$): MPS8 versus binary8
$q_0=0^8$ ($s=1$), $q_A$ = 1 on sites {3,4} ($s=0.4$), $q_B$ = 1 on sites {1,6} ($s=0.7$); $\chi=3$, $d=2$.
No superpositions ($k^*=k_{\rm ex}$ at every step, checked).

| tree | first escape | next | last |
|---|---|---|---|
| mps8 ($K$=8) | $q_0$, $\sigma^{-6}$ | $q_A$: $k=2$ (cores 3,4), log | $q_B$: $k=6$ (cores 1-6, $A=1.088$), $\sigma^{-4}$ |
| binary8 ($K$=7) | $q_0$, $\sigma^{-5}$ | $q_A$ or $q_B$: both $k=5$ (tie) | the other one, $k=5$ ($A=0.837$ for $q_A$ after $q_B$), $\sigma^{-3}$ |

Checked (random init, $\sigma=0.25\to0.2$, i.e. factor 1.25; seeds 0,1): MPS8 always $q_0,q_A,q_B$ with
the $q_A$ gap below the output-grid resolution (log time) and $q_B$ gaps growing by 2.40 and 2.59
(predicted $1.25^4=2.44$); $t_{q_0}$ grows by 3.81 ($1.25^6=3.81$). Binary8: seed 0 learns
$q_0,q_A,q_B$, seed 1 $q_0,q_B,q_A$ (tie); every gap grows by 1.91-2.09 ($1.25^3=1.95$) and $t_{q_0}$ by
3.05 ($1.25^5=3.05$). The same teacher thus has one long ($\sigma^{-4}$) and one log plateau on the MPS,
two $\sigma^{-3}$ plateaus on the binary tree.

### 5.5 Example 4 ("N4", nested sets): partial learning
$q_0=0000$ ($s=1$), $q_1=1100$ ($s=0.6$), $q_2=1111$ ($s=0.4$): $q_1$ differs from $q_0$ on $\{0,1\}$, $q_2$
on the superset $\{0,1,2,3\}$ and agrees with $q_1$ on $\{0,1\}$. Learned $q_0,q_1$ share sites 2,3, so
on the MPS the bond space of sites $\{0,1\}$ holds the superposition $s_0e_{00}+s_1e_{11}$, which
overlaps $U_{q_2}|_{01}=e_{11}$ without containing it: $k^*(q_2)=2$ < $k_{\rm ex}(q_2)=3$.

- **MPS4** (predicted): $q_0$ ($\sigma^{-2}$) $\to$ $q_1$ ($k=2$) $\to$ partial step ($k^*=2$ route on
cores {2,3}, $\lambda=0.4\cdot0.474=0.190$) to the best approximation with rank 1 on the central edge,
i.e. the rank-1 SVD truncation of $M=W^*_{(01)|(23)}=\begin{pmatrix}1&0\\0.6&0.4\end{pmatrix}$ (pattern
basis): $L=\sigma_2(M)^2/2=0.0569$ ($\sigma_{1,2}=1.186,0.337$) $\to$ remainder ($k=2$ route on cores
{1,2}) $\to0$. **Every gap after the first escape is $\propto\log(1/\sigma)$**, although the exact
rank rule alone would predict $\sigma^{-1}$ for $q_2$. Levels: $0.76\to0.26\to0.08\to0.0569\to0$.
- **binary4** = depth-3 linear network on $M$: SVD modes, $0.76\to0.0569$ (first escape) $\to0$ after
$\propto\sigma^{-1}$ ($k=3$).
- Checked on MPS4 ($\sigma=0.05$, seed 0): plateau points at $L=0.0799$ ($a=(1.00,1.00,0.001)$,
$\lambda_{\min}=-0.159$) and $L=0.0567$ ($a_2=0.30$, $\lambda_{\min}=-0.129$); $q_2$ gaps after $q_1$:
34, 42 (seed 0) and 22, 31 (seed 1) for $\sigma=0.1,0.05$: additive increments of about 8-9 per halving
(log-like), not a doubling.

General lesson: with several learned components the right unit is not "a component" but "a rank
increment on a connected set of edges"; nested teachers are learned through a chain of cheap
($k^*=2$) partial steps on the MPS.

### 5.6 Generic (non-orthogonal) teachers: power-law plateaus need tree-aligned structure (derived, checked)
At any critical point the single-core conditions say that, for an adjacent pair $C=\{u,v\}$, the
matrix $R_C$ (rows: modes of $u$ = its physical legs and old boundary bonds, dimension $D_u$; columns:
modes of $v$, $D_v$) annihilates the $r_e$-dimensional learned subspaces on both sides. So $R_C$ lives on
the complements, of dimensions $D_u-r_e$ and $D_v-r_e$, and for a generic teacher it is nonzero there:
**$k^*=2$ (strict saddle, log-time escape) whenever some adjacent pair has $D_u>r_e$ and $D_v>r_e$.**

- MPS: every core carries a physical leg, so after the first escape ($r_e=1$, $D=d\,r_{\rm side}\ge d$)
every adjacent pair qualifies: all later saddles of a generic teacher are strict, and ranks grow
edge by edge in log-time steps.
- Binary / HT trees at rank one: an internal core has no physical leg and $D=1=r_e$, so pairs through
internal cores have $R_C=0$ and $k^*\ge3$ (binary4: exactly the depth-3 linear network, non-strict).
At higher ranks internal cores can have $D>r_e$ and $k^*=2$ routes appear.
- Pattern (odeco-type) teachers suppress the pair routes by orthogonality; that is what produces the
power-law plateaus of H2. **H2's $\sigma^{-(k-2)}$ laws are a property of teachers whose components are
orthogonal on the relevant site blocks, not of TTNs in general.**

Checked: random CP teacher (`random_cp_teacher(4, 2, [1, 0.5], seed=3)`, non-orthogonal factors),
$d=\chi=2$, seed 0, $\sigma=0.05,0.025,0.0125$. MPS4: second plateau at $L=0.0515$ with gauge-projected
$\lambda_{\min}=-0.129,-0.132,-0.136$ ($\sigma$-independent: strict), gaps 67.0, 73.6, 80.7 (additive,
log-time; the increment $\approx7$ per halving exceeds $\ln2/\lvert\lambda_{\min}\rvert\approx5$, consistent
with more than one rank increment). Binary4: plateau at $L=0.0367$, $\lambda_{\min}=-0.0171,-0.0084,-0.0041$
($\propto\sigma$: non-strict), gaps 108.8, 223.8, 453.8 ($\propto\sigma^{-1}$).

## 6. Hessian at the saddles, gauge zero modes

### 6.1 Exact saddle $\theta_S$ (proven; checked on 21 tree/$D$ cases)
The gauge group is $\prod_e GL(\chi_e)$ ($\sum_e\chi_e^2$ generators $\delta G_v=G_vX$,
$\delta G_p=-XG_p$ on the shared leg). At $\theta_S$ the new columns/rows are zero, so the generators
with $X$ supported on the new-new block act trivially: the stabilizer has dimension
$\sum_e(\chi_e-r_e)^2$ and
$$\dim(\text{gauge orbit at }\theta_S)=\sum_e\big[\chi_e^2-(\chi_e-r_e)^2\big]=\sum_e\big[r_e^2+2r_ec_e\big].$$
The Hessian is block diagonal between old and new entries (F1, F2):

- **old block** = GN + piggyback couplings (the fixed-rank problem): null space = the $X_{\rm old,old}$
gauge directions ($\sum_er_e^2$); the other $P_{\rm old}-\sum_er_e^2$ eigenvalues are nonzero (positive
in the correct order; one negative per $m=2$ pair in the wrong order);
- **new block** = $-2\sum\langle R,W_{uv}\rangle$ restricted to new entries: zero except
$\pm c$ ($c_e$ copies each) for every $k^*=2$ route; it contains the $2\sum_er_ec_e$ gauge directions
$X_{\rm old,new},X_{\rm new,old}$.

Hence
$$N_0=\dim\ker H(\theta_S)=\sum_er_e^2+P_{\rm new}-2\sum_{k=2\text{ routes}}c_e,\qquad N_{\rm flat}=N_0-\dim(\text{orbit})=P_{\rm new}-\sum_e2r_ec_e-2\sum_{k=2}c_e$$
non-gauge flat directions (the degenerate directions of the higher-order saddle).
Example (MPS4, $d=3$, $\chi=2$, one learned component): $P=36$, $P_{\rm old}=12$, orbit 9, $N_0=27$
($k\ge3$) or 25 ($k=2$), all matched exactly.

### 6.2 Plateau point (route amplitudes small but nonzero; derived, checked)
Away from $\theta_S$ the gauge action is free: $\sum_e\chi_e^2$ gauge directions, projected dimension
$P-\sum_e\chi_e^2$. On a route the gauge rescales the route amplitudes while keeping their product, so
the $+c$ partners (and the extra $c_e-1$ copies along an already chosen new direction) become gauge.
**The gauge-projected Hessian has one negative eigenvalue per active route** (any further negative
eigenvalues are $O(\sigma_{\rm eff})$-small; e.g. $-0.016$ next to $-0.464$ for $\chi=3$, $d=3$):
$$\lambda_{\min}\simeq-\lambda=-s_rs_q^{(K-2)/K}\ \ (k=2),\qquad \lambda_{\min}\simeq-(k-1)\,s_rA\,b^{k-2}=O(\sigma_{\rm eff}^{k-2})\ \ (k\ge3,\ \text{balanced route}),$$
with $b=(s_ra_r/A)^{1/k}$ estimated from the measured overlap $a_r$ at that point. The remaining
spectrum: $P_{\rm old}-\sum r_e^2$ "old" eigenvalues ($O(1)$) and $P_{\rm new}-\sum_e(2r_ec_e+c_e^2)-n_{k=2}$
near-zero ones.

Worked spectrum (MPS4, $d=2$, $\chi=2$, $s=(1,0.5)$, $\sigma=0.0125$, plateau midpoint), all 12
projected eigenvalues:
- $k=2$ ($D=\{1,2\}$): $\{-0.4976\ |\ 6\text{ near }0\ (\le2.4\cdot10^{-3})\ |\ 0.498,\,0.994,\,0.995,\,1.494,\,3.969\}$;
predicted $\{-0.5\ |\ 6\times0\ |\ 0.5,1,1,1.5,4\}$ (0.5 and 1.5 are the piggyback pair $s_q\mp s_r$, 4 is
the radial mode $K a^{2(K-1)}$, the two 1's are GN modes $a^{2(K-1)}$).
- $k=3$ ($D=\{0,2\}$): $\{-3.4\cdot10^{-3}\ |\ 6\text{ near }0\ |\ 0.49996,1.0001,1.0002,1.4999,4.0000\}$.
- $k=4$ ($D=\{0,3\}$): $\{-8.8\cdot10^{-6}\ |\ 6\text{ near }0\ |\ 0.49995,1.0001,1.0001,1.4999,4.0000\}$.

### 6.3 Consequence for the saddle-type diagnostic (checked)
$\lvert\lambda_{\min}\rvert/\lambda_{\max}$ at the midpoint scales as $\sigma^0$ ($k=2$), $\sigma^1$
($k=3$), $\sigma^2$ ($k=4$). Measured (MPS4, $\lambda_{\max}=4$): $k=2$: $-0.467,-0.485,-0.493,-0.496,-0.498$
for $\sigma=0.2\dots0.0125$ (halving); $k=3$: $-0.054,-0.028,-0.014,-0.0069,-0.0034$; $k=4$:
$-2.2\cdot10^{-3},-5.6\cdot10^{-4},-1.4\cdot10^{-4},-3.5\cdot10^{-5},-8.8\cdot10^{-6}$. With the rule
"strict if $\lambda_{\min}<-10^{-3}\lambda_{\max}$" the $k=3$ plateaus are called strict for all
$\sigma\ge0.025$ (binary4 $k=3$: for all $\sigma\ge0.0125$). **Recommendation:** classify by the
$\sigma$-dependence of $\lambda_{\min}$ (constant vs $\propto\sigma^{k-2}$), or by the growth exponent
$\gamma$ of section 3.3, not by a fixed threshold.

## 7. Numerical checks (all gradient flow, float64, LSODA with exact Hessian Jacobian, rtol $10^{-10}$-$10^{-11}$, atol $10^{-13}\sigma$-$10^{-15}$)

| id | what | setting | result |
|---|---|---|---|
| C1 | Hessian at exact $\theta_S$ (one learned component) | 21 (tree, $D$) cases: binary4/8, mps4/6, ht4, binary6; $d=3$, $\chi=2,3$; both orders | $\lvert\nabla L\rvert=0.5$ exactly for $k=1$, $<10^{-15}$ otherwise; $k=2$: $\lambda_{\min}=-s_ra^{K-2}$ to 6 digits (e.g. $-0.70711$ for mps4, $s_q=0.5,s_r=1$), multiplicity $\chi-1$; piggyback eigenvalues $a^{K-2}(s_q\mp s_r)$ to 6 digits; no negative eigenvalue for $k\ge3$ (correct order) or $m\ge3$ (any order); $N_0$ and gauge rank equal the formulas of 6.1 in all cases |
| C1b | coupling formula, several learned components | mps4 $\{0000,1100\}$+0110; mps5 three learned + one new; single-learned mps4/5/6, binary4, ht4 | $s_r\prod_f\lVert T_f^\top U_r\rVert$ equals $-\lambda_{\min}$ (e.g. 0.400000 vs $-0.4$, 0.327461 vs $-0.327461$); single learned: $A=s_q^{(K-k)/K}$ exactly |
| C2 | aligned route family (exactly invariant; residual $<10^{-14}$) | $s_q=1,2$, $s_r=0.5$; mps4 $D$=12,02,03; binary4 03; mps6 23; mps5 13; $\sigma=0.1\to0.002$ | $k=2$ slopes 1.0067, 1.0053, 1.0622, 0.9464 vs 1.0000, 1.0000, 1.0607, 0.9449; $k\ge3$: $\Delta t_2/\text{pred}$ = 0.9964, 0.9974, 0.9957, 1.0000 (k=4) at $\sigma=0.002$; exponents $-1.014,-1.010,-1.016,-2.000$ |
| C3 | random fan-in init, two components, $d=\chi=2$, 3 seeds, $\sigma=0.2\to0.0125$ | mps4 $D$=12 ($k$=2), 02 (3), 03 (4); binary4 03 (3), 01 (1); mps5 13 (3) | $k=2$: slope 1.05 (pred 1.00); exponents $-0.969$ (mps4,k=3), $-1.019$ (binary4,k=3), $-1.992$ (mps4,k=4, seed 0); $\gamma$ = 1.33 (k=3), 1.494-1.500 (k=4); $k=1$: $\Delta t_2=0.02$ |
| C3b | reduced route ODE from measured amplitudes | mps4 $D$=02, $\sigma=0.025$, 3 seeds, 3 start times | remaining plateau GF vs ODE: 370.3/366.8, 410.8/401.9, 427.5/416.6 (incl. one sign-correction case) |
| C4 | gauge-projected Hessian at plateau midpoints | as C3 | $k=2$: $\lambda_{\min}\to-0.498$ (pred $-0.5$), one negative eigenvalue (also for $\chi=3$); $k\ge3$: $\lambda_{\min}\propto\sigma^{k-2}$ (factor 2.0/halving for k=3, 4.0 for k=4), within factor 1.2-2 of $-(k-1)s_rAb^{k-2}$ (binary4: $-0.0129$ vs $-0.0129$); counts of near-zero modes as in 6.2 |
| C5 | wrong order (H1b) | binary4 and mps4 odeco, $s_q=0.5$ learned, $s_r=1$ | see 4.4: rotation spacing 11.67/12.60 vs 11.60; $L$ plateau at 0.125; $m=4$: no rotation, escape at 4999.97 vs 5000 |
| C6 | order reversal (B4) | mps4, binary4, ht4; $\chi=3$ | orders as predicted; MPS4 $q_A$ increments $\to1.620$ per halving (pred 1.617), identical with and without $q_B$; exponents $-2.03$ (mps4 $q_B$), $-1.02$ (binary4), $-2.99$ (ht4 $q_B$) |
| C7 | partial learning (N4) | mps4, $\chi=3$ | intermediate plateau $L=0.0567$ (pred 0.0569), log-like gaps |
| C8 | conservation-law trap | ht4, B4, $\sigma=0.2$, seeds 0-2, fan-in vs gauge-balanced init | fan-in: stall at $L=0.045$ in 3/3 seeds up to $t=3\cdot10^6$ (leaf route amplitudes $\sim10^{-14}$, parent new-slice norm$^2=-Q_{\nu\nu}(0)$ to 4 digits); balanced: route alive in 3/3, $q_A$ learned in 2/3, third predicted at $t\approx3.4\cdot10^6$ |
| C9 | E8 on MPS8 vs binary8 | $\sigma=0.25,0.2$, seeds 0,1, $\chi=3$ | MPS8 order $q_0,q_A,q_B$ (2/2 seeds), $q_B$ gap ratio 2.40/2.59 (pred 2.44); binary8 tie order seed-dependent, gap ratios 1.91-2.09 (pred 1.95), $t_1$ ratios 3.81 and 3.05 (pred $1.25^6$, $1.25^5$) |
| C10 | generic (non-orthogonal) CP teacher | `random_cp_teacher(4,2,[1,0.5],seed=3)`, mps4 and binary4, $\sigma=0.05\to0.0125$ | MPS4: $\lambda_{\min}=-0.129,-0.132,-0.136$ (strict), log gaps; binary4: $\lambda_{\min}\propto\sigma$, gaps $\propto\sigma^{-1}$ (section 5.6) |

## 8. Caveat found numerically: the conservation-law trap (derived mechanism, checked on 3 seeds)

On HT4 with example B4 ($\sigma=0.2$, seed 0, $\chi=3$) gradient flow learns $q_0$ and $q_B$ and then
stays at $L=0.045=\tfrac12s_A^2$ with $\lvert\nabla L\rvert\approx10^{-15}$ and a gauge-projected Hessian
$\ge-1.4\cdot10^{-15}$ up to $t=1.7\cdot10^7$, although the $q_A$ route ($k=5$, $A=0.843$) exists.
Diagnosis at $t=2\cdot10^5$: the route amplitudes on the inner cores are $0.037, 0.019, 0.017$, but on
the two leaf cores they are $1.2\cdot10^{-14}$ and $6\cdot10^{-15}$: the leaves' new bond direction has
died. Mechanism:

- On edge $e$ (child $v$, parent $p$) the conserved $Q_e$ restricted to the new bond direction $\nu$
fixes $\lVert G_v\nu\rVert^2-\lVert\nu^\top G_{p,(v)}\rVert^2=Q_{\nu\nu}$.
- On a plateau, the new parts that connect through the other side to *learned* structure
("visible" components) have a GN cost $\propto$ (norm of the other side's new part)$^2\times O(1)$ and
decay at a rate $\sim\lvert Q_{\nu\nu}\rvert\sim\sigma^2$. The lighter side decays to zero, the heavier
side freezes with norm$^2=\lvert Q_{\nu\nu}\rvert$, keeping visible components that confine the lighter
side's route entry. The route then cannot grow (its drive is a product of small amplitudes).
- Fan-in init gives $Q_{\nu\nu}\approx\sigma^2(1-\chi_{p}/\chi_e)$ ($\chi_p$ = bond dimension of the
parent's own parent edge) plus fluctuations: HT leaf edges ($\chi_e=d<\chi_p$) have $Q_{\nu\nu}<0$, so
the leaf side dies. Measured: $Q_{\nu\nu}=-0.0263,-0.0207$ (estimate $-0.02$); final leaf new-column
norm$^2$ $1.5\cdot10^{-28}$, parent new-slice norm$^2$ $0.0263=-Q_{\nu\nu}$ exactly.
- Timescale comparison: decay time $\sim1/\lvert Q_{\nu\nu}\rvert\propto\sigma^{-2}$ versus route
escape $\propto\sigma^{-(k-2)}$: harmless for $k\le3$, seed-dependent for $k=4$ (consistent with the
20-30x spread of $k=4$ plateaus between seeds on MPS4), and fatal as $\sigma\to0$ for $k\ge5$ when the
sign is unfavourable. The point reached is not a local minimum (moving the frozen visible components
is free and then opens a $k=2$ descent), but gradient flow does not move along those flat directions;
SGD noise would.
- **Balanced init removes the trap (checked on 3 seeds; conjecture in general).** Gauge-balancing
the fan-in init ($Q_e=0$ exactly, same $W_0$) keeps the leaf route amplitudes alive: HT4, B4,
$\sigma=0.2$, seeds 0/1/2: fan-in init stalls at $L=0.045$ up to $t=3\cdot10^6$ in all three seeds;
balanced init learns $q_A$ at $t=2850$ (seed 0) and $3.7\cdot10^5$ (seed 1), and for seed 2 the route is
alive at $t=2\cdot10^5$ (leaf amplitudes $-0.0196$, product negative, i.e. in a sign-correction phase)
with escape predicted by the route ODE at $t\approx3.4\cdot10^6$. With $Q_{\nu\nu}=0$ both sides of an
edge lose their visible components together and the leaf column converges to the norm of the parent's
route component instead of to zero.

Practical consequence for Q3 experiments: on HT trees (and for $k\ge4$ routes in general), report
"stalled" runs separately from slow escapes; check the leaf new-column norms (dead route: $\sim10^{-14}$);
use gauge-balanced initialization when the aim is to measure escape exponents. The trap is a property
of the fan-in initialization (its $Q_e\ne0$), not of the TTN landscape.

## 9. Testable predictions for the experiment agents

Conventions: event time $t_r$ = first time $a_r\ge\tfrac12$ (use the integrator's event location;
coarse output grids give errors of several time units on $k=2$ gaps); $\Delta t$ = gap between
consecutive events; "per seed" = same seed (same init direction) at several $\sigma$.

1. **Exponents (H2).** Two-component pattern teachers (section 5.2 tables, $\chi\ge2$):
$\Delta t_2\propto\sigma^{-(k-2)}$ per seed for $k\ge3$ (fitted exponent within $\pm0.1$ of $-(k-2)$
over $\sigma\in[0.0125,0.2]$ for $K\le5$); $k=2$: $\Delta t_2=\text{slope}\cdot\ln(1/\sigma)+\text{const}$
with **slope $=(1-s_r/s_q)/(s_r\,s_q^{(K-2)/K})$** ($=1.00$ for weights $(1,0.5)$, $=2.33$ for $(1,0.3)$;
twice that would refute the head-start picture); $k=1$: no plateau (separation ratio $<3$).
2. **Prefactor, aligned init** (route family of section 3; any tree): $t_1=\sigma^{-(K-2)}/((K-2)s_q)$
(up to ~20% shorter when the second route shares hanging cores with the first: measured ratio 0.79-0.94 for $k=2,3$, 1.000 for $k=K$) and
$\Delta t_2=\sigma^{-(k-2)}(1-s_r/s_q)/((k-2)\,s_r\,s_q^{(K-k)/K})$, to $<1\%$ at $\sigma=2\cdot10^{-3}$.
3. **Prefactor, random init:** measure the route amplitudes $b_v$ when $a_q\ge0.999$; the $k$-scalar
ODE $\dot b_v=A\prod_{u\ne v}b_u(s_r-A\prod b)$, $A=s_q^{(K-k)/K}$, predicts the rest of the plateau
within $\sim3\%$. Seed-to-seed spread of $\Delta t_2$ is broad for $k\ge4$ (factor $>20$ seen at $k=4$).
4. **Growth exponent:** on a $k\ge3$ plateau $d\log\dot a_r/d\log a_r\to2-2/k$ (1.33, 1.50, 1.60 for
$k=3,4,5$); $k=2$: exponential growth at rate $2\lambda$ in $a_r$.
5. **Hessian at plateau midpoints (H3):** gauge-projected spectrum has one negative eigenvalue per
active route (any others are $O(\sigma)$-small). $k=2$: $\lambda_{\min}\to-s_r\,s_q^{(K-2)/K}$ as $\sigma\to0$ (e.g. $-0.5$, measured
$-0.498$). $k\ge3$: $\lambda_{\min}\simeq-(k-1)s_rA\,b^{k-2}\propto\sigma^{k-2}$. Old block: the $m=2$
piggyback pair $s_q^{(K-2)/K}(s_q\mp s_r)$, GN modes $s_q^{2(K-1)/K}$, radial mode $K s_q^{2(K-1)/K}$
(MPS4, $s=(1,0.5)$: $\{0.5,1,1,1.5,4\}$). Near-zero count $P_{\rm new}-\sum_e(2r_ec_e+c_e^2)-n_{k=2}$.
6. **Fixed-threshold misclassification:** with the $10^{-3}\lambda_{\max}$ rule, $k=3$ plateaus read as
strict for $\sigma\gtrsim0.01$-$0.03$; the $\sigma$-scaling of $\lambda_{\min}$ separates them.
7. **Wrong order (H1b).** (a) Matrix-like pair ($m=2$, $k\ge3$; e.g. odeco on binary4, or $D$ = two
non-adjacent single-site cores on an MPS): if the smaller component is learned first, that plateau has
$\lambda_{\min}=-(s_r-s_q)s_q^{(K-2)/K}$, lasts $\approx\ln(1/\varepsilon)/\lvert\lambda_{\min}\rvert$,
and is followed by a plateau at $L=\tfrac12s_q^2$ with $a_q\approx0$ (the smaller component is
forgotten, then relearned after $\propto\sigma^{-(k-2)}$). (b) Tensor-like pair ($m\ge3$; e.g. odeco on
MPS4, $K$=4): no negative eigenvalue, $a_q$ stays at 1, $L:\tfrac12s_r^2\to0$ in one step. (c) $k=2$:
the larger component is added directly. (d) Heuristic: $P(\text{smaller first})$ for matrix-like pairs
decays like $\sigma^{(s_r-s_q)/s_q}$; for tensor-like pairs it does not decay with $\sigma$.
8. **Locality beats weight:** B4 ($N=4$): MPS4 learns $q_0,q_A(0.3),q_B(0.6)$ for every seed ($k$ = 2
vs 4); binary4 and HT4 learn $q_A$ and $q_B$ through tied routes ($k=3$ resp. 5) after $q_0$, in an order
biased toward the weight order but seed-dependent. E8 ($N=8$, $q_A$ = 1 on {3,4}, 0.4; $q_B$ = 1 on
{1,6}, 0.7): MPS8 learns $q_0$, $q_A$ (log gap), $q_B$ ($\sigma^{-4}$); binary8 learns $q_0$ and then $q_A$,
$q_B$ both through $k=5$ routes ($\sigma^{-3}$ each, tied).
9. **Partial learning:** N4 on MPS4 shows an extra plateau at $L=\sigma_2^2/2=0.0569$ between 0.08 and
0, with $a_2\approx0.3$; all its gaps grow like $\log(1/\sigma)$.
10. **Conservation-law trap:** on HT trees with fan-in init, components whose route ends on a leaf with
an already used bond can stall indefinitely (B4 on HT4: $L=0.045$, 3/3 seeds); probability increases
with $k$; leaf new-column norm decays exponentially while the parent's new slice keeps
norm$^2=-Q_{\nu\nu}(0)$. A gauge-balanced version of the same initialization ($Q_e=0$) does not stall
(3/3 seeds).

11. **Generic teachers (for `data-teacher`, Q7):** random CP or random TTN teachers on an MPS: every
plateau after the first is a strict saddle ($\lambda_{\min}$ $\sigma$-independent, gaps $\propto\log(1/\sigma)$);
on binary4 the second plateau is non-strict ($\lambda_{\min}\propto\sigma$, gap $\propto\sigma^{-1}$).
On larger binary/HT trees expect $k^*\ge3$ only while ranks are one.

## 10. Open issues

- **Prefactors for random init** are not predicted from $\sigma$ alone: they are set by the route
amplitudes at the start of each plateau (a first-escape problem of the $k^*$-core sub-TTN with target
$R_C$, to be taken from theory_escape) plus the accumulated head start. Only exponents, slopes of
log-plateaus, couplings and Hessian values are parameter-free here.
- **Whole-sequence prediction for structured multi-component teachers** (section 5.1) is an
algorithm, not a theorem. A numerical "saddle hopper" would make it operational: at a plateau state,
compute $R_C$ for all connected $C$ with $\lvert C\rvert\le4$ (cheap for $N\le8$), seed the best route
of the smallest degree, integrate to the next plateau, repeat; compare with small-$\sigma$ runs.
- **Probability of the wrong order** for matrix-like pairs ($\propto\sigma^{(s_r-s_q)/s_q}$) is a
heuristic from the local spherical-flow linearization; the `order` agent's data should decide.
- **Conservation-law trap:** general conditions on other trees and the claim that balanced
initialization always avoids it are conjectures (3 seeds on one example). It may explain part of the
seed-to-seed spread of $k=4$ plateaus.
- **Anisotropic inputs** (H5): replace the Frobenius inner product by the $\Sigma$-metric; the route
degrees are unchanged, couplings are reweighted by the site eigenvalues on the route's patterns
(not checked).
- **Literature:** the k-rule with tree-dependent exponent, the head-start slope $(1-s_r/s_q)/\lambda$,
the TTN piggyback classification and the conservation-law trap were not found in the papers read;
`notes/literature.md` should confirm or correct this.
