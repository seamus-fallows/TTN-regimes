# Theory: saddles after the first escape, and locality (Q3; H2, H3, H1b)

Status: **v3, 2026-10-10**: R2, F2, Prop. 1, 3.3, 3.4, 4.1, 4.3, 4.4, 5.1, 9, 11 and P9 updated for the
saddle-order results of `notes/theory_saddle_order.md` v2 (lower bound proven under a spectral
condition, counterexamples from $k^*=5$; both verified). **v2, 2026-10-09** (theorist for Q3), revised
after an independent adversarial verification (round 2; issues and resolutions in section 10). Scratch scripts (outside the repo; only `ttn.tree`,
`ttn.network`, `ttn.teachers`, `ttn.losses`, scipy `solve_ivp`, torch autograd, float64):
v1 checks in `.../scratchpad/theory/saddles/`, the verifier's and the v2 re-checks (`r1_*`-`r6_*`) in
`.../scratchpad/theory/saddles_verify2/` (prefix
`C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926`).
Tags: **proven** (complete argument given), **derived** (under the stated assumptions, leading order in
the small parameters), **heuristic**, **empirical**, **conjecture**; **checked** marks a numerical
confirmation (section 7). Which results are known in other models and which are new: section 9.

RNG caveat: the v1 checks ran while teacher and init used a shared random stream (fixed in commit
57d5262). Only runs whose init seed equalled the teacher seed (0) can carry a spurious init-teacher
correlation. Exponents, Hessian values and the head-start slopes were re-checked after the fix by the
verifier (teacher seed 3); the v1 prefactors, learning orders and trap counts are flagged where they
matter.


> **Trap update (2026-10-10):** R12, C8, section 8, section 9 (row R12), section 10 (one bracket), section 11
> ("Trap"), P12 and P14(v) are updated from `notes/theory_trap.md` section 10, as corrected by
> `notes/reviews/trap_verification.md`. Each edit is marked "[trap update 2026-10-10]". The HT4/B4 data of
> section 8 are kept unchanged.

> **Lead's correction (2026-10-09):** the wrong-order exponent carries a factor $n$, the number of unstable
> directions of the matrix-like pair: $n=\chi_e$ when the two differing cores are adjacent and share bond $e$, else
> $n=1$ (`notes/theory_escape.md` v2 §5.3, checked on 19 Hessians). The formulas in this note now include it.

## 0. Results at a glance

| # | result | status |
|---|---|---|
| R1 | Around a learned saddle the new parameters enter $L$ through *routes*: connected core sets $C$ whose new entries combine into a term $\propto$ the residual, with no Gauss-Newton cost at that order. The escape is, to leading order, the first-escape problem of the $k^*$-core sub-TTN with target $R_C$, $k^*=\min\lvert C\rvert$ (the TTN instance of Kumar & Haupt 2025c, Thm 8, and of AGF; section 9), **provided** the old block is a local minimum and the generalized piggyback operator has $\lambda_{\max}(\mathcal S)<s_q$ (`theory_saddle_order.md`); otherwise a cluster of $\ell<k^*/2$ cores plus old rotations can escape first. | derived |
| R2 | One learned component $q$ (balanced), new component $r$: $\min\{\lvert C\rvert: C\text{ connected}, R_C\ne0\}=\lvert\mathrm{Steiner}(N(D))\rvert$, where $N(D)$ are the cores carrying the sites where $r$ differs from $q$; the edges whose rank must grow are exactly the Steiner edges. The route gives a descent curve of order $k^*$ (upper bound on the saddle order). **Lower bound** (`theory_saddle_order.md` v2): order $\ge k^*$ whenever $\lambda_{\max}(\mathcal S)<s_q$, so order $=k^*$ always for $k^*=2$, for $k^*\in\{3,4\}$ with a positive-definite old block, and for two-component teachers in the correct order on every tree. **False in general from $k^*=5$ on:** a wrong-order residual with two terminals in a set of degree $<k^*/2$ has order $2\ell^*<k^*$ (MPS6 $110001$: 4 vs 6; HT4 $1111$: 6 vs 7), and with several residuals also the correct order can fail (MPS6 pair-star: 4 vs 5, PSD Hessian). | proven (identity, upper bound; lower bound under $\lambda_{\max}(\mathcal S)<s_q$; exact orders $2\ell^*$ for pattern teachers with a spare bond index) |
| R3 | At a balanced learned saddle $\theta_S$ the Hessian is block diagonal (old entries / new entries). $\theta_S$ is a **strict** saddle iff the old (fixed-rank) block has a negative eigenvalue or a $k^*=2$ route exists. $k^*=1$: not critical (no plateau). $k^*=2$: $\lambda=s_r\,s_q^{(K-2)/K}$ exactly (one learned component). $k^*\ge3$ and old block PSD: no negative curvature (higher-order saddle). | proven (at $\theta_S$), checked |
| R4 | Escape times (aligned init): $k^*=2$: $\Delta t\simeq(1-s_r/s_q)\log(1/\sigma)/\lambda$; $k^*\ge3$: $\Delta t\simeq\sigma^{-(k-2)}(1-s_r/s_q)/[(k-2)\,s_r\,s_q^{(K-k)/K}]$. The factor $(1-s_r/s_q)$ is a head start accumulated since initialization (for $k\ge3$ mostly during the first plateau). These route laws need $\lambda_{\max}(\mathcal S)<s_q$ (R2); otherwise a generalized piggyback cluster escapes in $\propto\sigma^{-(2\ell^*-2)}$ once it beats the route (`theory_saddle_order.md` section 6). | derived, checked to $10^{-4}$ (aligned) |
| R5 | Exponent $k-2$ per seed. Growth exponent on a balanced route $\gamma=d\log\dot a_r/d\log a_r=2-2/k-a_r/(1-a_r)$: read $k$ from $a_r\in[10^{-4},10^{-2}]$. The $k$-scalar route ODE started from measured route amplitudes predicted the rest of the plateau to 1-3% (v1, not re-checked). | derived, checked |
| R6 | One learned component: the old block is a PSD radial part plus the piggyback block $a^{K-2}(s_qI-S)$, $a=s_q^{1/K}$, where $S$ is the weighted adjacency matrix of the *piggyback graph* (nodes: (terminal core, rotated site pattern); one edge of weight $s_r$ per residual with $m_r=2$ terminals). Strict iff $\rho(S)>s_q$ or a $k^*=2$ route exists. Two-component teachers: $m=2$ gives $a^{K-2}(s_q\mp s_r)$, $m\ge3$ gives a PD block. Binary $N=4$ is exactly a depth-3 deep linear network (Achour et al. 2024, Thm 7 applies). | proven, checked |
| R7 | Wrong order, two-component teachers: strict iff $m=2$ or $k^*=2$; three plateau sequences (the third, "$m\ge3$: $r$ added by its route, $q$ kept", holds only without a low split; with one, e.g. HT4 $D$ = all sites, the order is $2\ell^*<k^*$ and $q$ is forgotten and relearned, 4.4). With $\ge3$ components use R6: a **star** (residuals sharing a rotated site vector at one terminal core) gives $\lambda_{\min}=a^{K-2}(s_q-(\sum s_r^2)^{1/2})$, so even the largest component can be a strict saddle. Gradient flow then passes near $q$, forgets it and relearns it (MPS4: 5/5 seeds). | proven (Hessian), derived (sequences), checked |
| R8 | Same teacher, different trees: the next escape is the route with the smallest $k^*$, not the largest weight (locality beats weight); ties in $k^*$ are broken by random route amplitudes. Worked examples for $N=4,8$ (section 5). | derived, checked ($N=4,8$; v1, pre-fix RNG; tables re-derived by the verifier) |
| R9 | With several learned components the escape degree can be smaller than the number of cores needed to fit the component exactly ("partial learning"): an extra plateau at the best fit of a lower rank profile appears. | derived, checked (v1; levels re-derived by the verifier) |
| R10 | Hessian: gauge orbit at $\theta_S$ has dimension $\sum_e[\chi_e^2-(\chi_e-r_e)^2]$; null space $\sum_e r_e^2+P_{\rm new}-2\sum_{k=2}c_e$. At a plateau point each route contributes one negative direction, $-\lambda$ ($k=2$) or $\approx-(k-1)s_rA\,b^{k-2}=O(\sigma^{k-2})$ ($k\ge3$). Visible new-index pairs add $O(\sigma^2)$ indefinite terms, so "one negative eigenvalue per route" holds for $k\le3$ only (at $k=4$ a second one of the same order appears). | proven (counts at $\theta_S$), derived (plateau), checked |
| R11 | The 1e-3 relative threshold of `notes/diagnostics.md` classifies $k=3$ plateaus as strict for $\sigma\gtrsim0.01$; use the $\sigma$-scaling of $\lambda_{\min}$ instead. | checked (v1 and verifier; boundary seed-dependent, $\sigma\approx0.01$-$0.04$) |
| R12 | Conservation-law trap [trap update 2026-10-10; details in `notes/theory_trap.md`]. On a rank-one plateau the decisive conserved quantity of a route end $u$ (partner $w$ on the route edge) is $C_u=X_u^{\perp\top}X_u^\perp-X_w^{\perp\top}X_w^\perp$, built from the normal-visible blocks (new bond index, physical index normal to the learned pattern); the pattern parts are tangent and do not decay. The end's normal block goes to $[C_u]_+$: with one new direction the end dies iff $C_u<0$. After a small-init escape $C_u=Q_{\nu\nu}+\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+O(\sigma^3)$; the old criterion $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$ kept only the route entry of $Y_{\rm nv}$ and over-predicts death. Routes with $k\le3$ never trap. For $k\le5$, as $\sigma\to0$, a route is dead iff at least two ends die and pinned if one does; from $k=6$ three-core clusters through inner route cores (rate $\propto\sigma^4$) are a further death channel and pre-empt the pinned escape. Per run from the init (fan-in, $k\ge4$, locality): predicted dead 408/457 stall, predicted alive 38/2,101. HT4/B4 at $\sigma=0.2$ (section 8): traps after two learned components. Balanced init: tree-leaf ends cannot die at leading order; internal ends can. | conservation and limit $[C_u]_+$: proven (reduced flow), checked in full GF; identity: proven; death rule: derived for $k\le5$, incomplete for $k\ge6$; pinned escape law $\propto\sigma^{-(2k-6)}$: conjecture; per-run tests: empirical |
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

**F2 (routes; proven; corrected in v2).** For new entries, $W_J[\delta]\ne0$ requires that every
new index of a $\delta_v$ sits on an internal edge of $J$ and is matched there by the neighbour, so
every edge leaving $J$ carries an old index and every connected component of $J$ has $\ge2$ cores.
For **connected** $J$ (a *route*),
$W_J[\delta]=\Phi_J(\delta)\otimes\bigotimes_{f\in\partial J}T_f$, where $\Phi_J$ is the small TTN
formed by the $\delta_v$ (internal bond dimensions $c_e$) and $T_f$ is the contraction of the learned
subtree hanging off the boundary edge $f$ (a map from the $r_f$ old indices of $f$ to tensors on the
hanging sites $H_f$). Every proper part of a route leaves a new index unmatched, so $W_{J'}=0$ for
$J'\subsetneq J$: **a route pays no Gauss-Newton cost at the order of its gain.**
*Disconnected* $J$ also contribute (v1 wrongly excluded them; verifier's example: MPS5, $D=\{0,4\}$,
$J=\{0,1\}\cup\{3,4\}$ reaches $U_r$ at degree 4, below $\lvert\mathrm{Steiner}\rvert=5$). But each
component $J_i$ alone already gives $W_{J_i}\ne0$ (old boundary indices): it acts as a generalized
piggyback rotation of the block $J_i$ and pays the GN cost $\tfrac12\lVert W_{J_i}\rVert^2$ at the same
order as the gain. For one learned component and two blocks with amplitudes $x,y$ this gives
$L-L_S=\tfrac12a^{2(K-\lvert J_1\rvert)}x^2+\tfrac12a^{2(K-\lvert J_2\rvert)}y^2-s_ra^{K-\lvert J\rvert}xy$,
PSD iff $s_q\ge s_r$, the $m=2$ piggyback condition of section 4 (derived; checked on the example:
along $x=y=t^2$, $L-L_S=a(s_q-s_r)t^4$ to 5 digits for $(s_q,s_r)=(1,0.5)$ and $(0.5,1)$, script `r1`).
**v3:** the general case is settled in `theory_saddle_order.md` (one learned component, proven):
in the cumulant expansion of the sector amplitudes, a cumulant on a core set $B$ needs new entries on
every core of $\mathrm{Steiner}(B)$ (its Lemma 2), and disconnected blocks are products of lower
cumulants. Whether they are Gauss-Newton dominated is exactly the condition
$\lambda_{\max}(\mathcal S)<s_q$ on the *generalized piggyback operator* $\mathcal S$, which couples the
sector amplitudes of core sets of degree $<k^*/2$ (single cores and, for $k^*\ge5$, sets such as two
adjacent cores). It can fail although every residual is smaller than $s_q$ and the Hessian is PSD
(MPS6 pair-star, order 4 $<k^*=5$).

**Definition (local residual).** For a connected core set $C$,
$R_C=\langle R,\bigotimes_{f\in\partial C}T_f\rangle$, a tensor with one mode per core of $C$ (its
physical legs and its old boundary legs). Then $\langle R,W_C[\delta]\rangle=\langle R_C,\Phi_C(\delta)\rangle$.

**Proposition 1 (escape degree; derived).** The new entries first lower $L$ at order
$k^*=\min\{\lvert C\rvert: C\text{ connected},\ R_C\ne0\}$. To leading order (old entries frozen at
$\theta_S$), the new entries on the minimizing $C$ follow the gradient flow of a $k^*$-core TTN fitting
the target $R_C$ from small initialization: **the escape from a saddle is the first-escape problem of
the sub-TTN on $C$** (theory_escape applies with $K\to k^*$, target $R_C$). This is not new as a
mechanism. It is Kumar & Haupt 2025c (Neuron Pursuit, arXiv:2509.12154) Thm 8 with lowest dormant
homogeneity $L=k^*$: dormant weights stay $O(\delta)$ for $\sim\delta^{-(L-2)}$ and align with a KKT
point of the residual's correlation function. It is also the accumulated-utility rule of Alternating
Gradient Flows (Kunin et al. 2025, arXiv:2506.06489): a dormant $\kappa$-homogeneous unit activates
when $\int\kappa\,\mathcal U\,dt$ reaches $(\lVert\theta_0\rVert^{2-\kappa}-1)/(\kappa-2)$
($\log(1/\lVert\theta_0\rVert)$ for $\kappa=2$; their time normalization), where $\mathcal U$ is the
unit's correlation with the residual, accumulated over *all* earlier phases. What the TTN adds is
which entries are dormant and the degree $\kappa=k^*$, which is set by the tree. **Caveats:** Thm 8
assumes that the active (old) weights sit at a local minimum of the fixed-rank problem. That fails
exactly at the strict saddles of section 4 (wrong order, stars), where the old block escapes first
and Proposition 1 does not describe the next event. **v3:** Proposition 1 also needs
$\lambda_{\max}(\mathcal S)<s_q$ (`theory_saddle_order.md`, Theorem 1). When this fails at a level
$\ell<k^*/2$, the old block can be positive definite and still the escape is a *joint* problem: a
cluster of $\ell$ cores with new entries, pointing at a **non-residual** partial pattern, together
with old rotations toward the residuals; saddle order $2\ell$ (proven for pattern teachers with a
spare bond index), escape $\propto\sigma^{-(2\ell-2)}$ (derived; checked on one teacher, where gradient
flow then forgets $q$ and relearns it).

**Proposition 2 (Steiner identity; proven for one learned component).** If $S=\{q\}$, the hanging
contractions are $T_f=a^{n_f}U_q|_{H_f}$ ($n_f$ cores), so
$$R_C=a^{K-\lvert C\rvert}\sum_{r:\,N(D_r)\subseteq C}s_r\,U_r|_C .$$
Hence $k^*=\min_r\lvert\mathrm{Steiner}(N(D_r))\rvert$ and the route of $r$ has coupling
$c_r=s_r\,a^{K-k}=s_r\,s_q^{(K-k)/K}$ ("$s_{\rm new}$ times the learned amplitudes outside $C$").
The rank of $W_S+s_rU_r$ across $e$ exceeds that of $W_S$ iff $D_r$ has sites on both sides of $e$,
i.e. iff $e$ is a Steiner edge; so $E_+$ (edges whose rank must grow) = Steiner edges and
$k=\lvert E_+\rvert+1$; $k=1$ iff $D_r$ sits on one core.

*What is and is not proven about the saddle order.* Proven: the identity above and a descent curve
of order $k^*$ (route entries $\propto t$: $L-L_S=-c_rt^{k^*}+O(t^{2k^*})$). This is Furman et al.
2026, Lemma D.8, applied to the Steiner subtree instead of the whole connected component of
rank-deficient edges; at a rank-one saddle with $\chi\ge2$ their component is the whole tree
($N=K$), while $k^*$ can be as small as 2. **Lower bound (v3; `theory_saddle_order.md` v2, both
verifications passed).** Proven: if $\lambda_{\max}(\mathcal S)<s_q$ then
$L-L_S\ge-C\lVert\theta-\theta_S\rVert^{k^*}$ near $\theta_S$ (its Theorem 1; any tree, any $\chi$, any
residual with $R_{\{v\}}=0$, balanced or not), so with the route the order is exactly $k^*$. This
covers $k^*=2$ (always), $k^*\in\{3,4\}$ whenever the old block is positive definite modulo gauge
($\mathcal S$ is then the classical piggyback operator of 4.1; $k^*=3$ also at $\rho(S)=s_q$), and every
two-component teacher in the correct order on every tree. A level version (Theorem 1') gives order
$\ge\min(k^*,2\ell)$ when $\mathcal S$ restricted to sets of degree $\le\ell-1$ stays below $s_q$, and
for pattern teachers with a spare bond index the order is exactly $2\ell^*$ when level $\ell^*<k^*/2$
exceeds $s_q$ (Theorem 2). So "order $=k^*$" is **false in general** from $k^*=5$ on: single
wrong-order residuals with two terminals in a set of degree $<k^*/2$ (MPS6 $110001$, order 4 vs
$k^*=6$; HT4 $1111$, order 6 vs 7), and, with several residuals, even the correct order with a PSD
Hessian (MPS6 $110001$, $110010$ at 0.8 each: order 4 vs $k^*=5$). Before v3 only the upper bound and
order $\ge3$ (PSD Hessian) were known here. Lower bounds on saddle orders were listed as open in
literature.md (section 5, item 3); the only one there is Furman et al. 2026, Prop. 5.3 (one parity
saddle).

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

**Head start (derived).** The same route existed since initialization (with the then-small
learned cores in its hanging part), and grew at the instantaneous rate $c(t)=s_r\,a(t)^{K-2}$ while $q$
rose as $\dot a=s_qa^{K-1}$ (assumes the hanging cores are already aligned with the shared pattern,
i.e. after the alignment phase of the first escape; the alignment phase only adds $O(1)$). Hence $\log b_{\rm eff}=\log b_0+\int c\,dt=\log b_0+(s_r/s_q)\log(a_1/a_0)$
(accrued uniformly in $\log a$ during the rise of $q$) and, for $b_0\sim a_0\sim\sigma$,
$$\boxed{\Delta t_2=\frac{(1-s_r/s_q)\,\log(1/\sigma)+O(1)}{s_r\,s_q^{(K-2)/K}}}$$
The slope in $\log(1/\sigma)$ is $(1-s_r/s_q)/\lambda$, **not** $1/\lambda$ (checked: aligned 1.0067 vs
1.0000 predicted, 2.0 without head start; random init 1.05; verifier, independent code: aligned MPS5
$(1.5,0.6)$ slope 0.78408 vs 0.78405, 1.307 without head start; random init MPS4 $(1.2,0.5)$ 1.066 vs
1.065). The same exponent $\sigma^{1-s_r/s_q}$ is Zhang, Saxe & Latham 2026 (Thm 4: data-induced
separation $O(\sigma^{1-s_{r+1}/s_1})$ for units linear in their own weights); here it is the
$k=2$ case of a TTN route.

### 3.3 $k^*\ge3$: higher-order saddle (Hessian PSD at $\theta_S$ proven when the old block is PSD; order $k^*$ proven when $\lambda_{\max}(\mathcal S)<s_q$, v3)
No pair of route entries couples at second order, so the Hessian vanishes on the route (and, with
F1, on all new entries except $k=2$ routes). If the old block is PSD (always for two-component
teachers in the correct order; in general iff $\rho(S)\le s_q$, section 4.1) the saddle is therefore
non-strict. Along the route $L$ decreases at order $k$, $L-L_S\approx-c\prod_{v\in C}b_v$; that no
other curve decreases faster is **proven** when $\lambda_{\max}(\mathcal S)<s_q$ (section 2, after
Proposition 2; always for $k^*\in\{3,4\}$ with a PD old block and for two-component teachers in the
correct order) and **false** otherwise: then the order is $2\ell^*<k^*$ and the plateau law below is
replaced by $\sigma^{-(2\ell^*-2)}$ at small enough $\sigma$ (derived; `theory_saddle_order.md` sections
6-7). All plateau laws of this section assume $\lambda_{\max}(\mathcal S)<s_q$. Balanced
reduced dynamics $\dot b=c\,b^{k-1}$, saturating at $b_*=(s_r/A)^{1/k}$:
$$\Delta t=\frac{b_{\rm eff}^{-(k-2)}}{(k-2)\,c}+O\!\Big(\frac1c\Big(\frac{A}{s_r}\Big)^{\frac{k-2}{k}}\Big);\qquad\text{unbalanced }(\textstyle\prod b>0):\ \Delta t=\int_0^\infty\frac{d\beta}{2c\prod_{v\in C}(b_v^2+\beta)^{1/2}} .$$
(v1 had the ratio in the correction inverted; $b_*^{-(k-2)}=(A/s_r)^{(k-2)/k}$.)
The no-plateau / $\log(1/\sigma)$ / $\sigma^{-(k-2)}$ trichotomy for $k=1,2,\ge3$ is Rawal & DeWeese
2026, Thm 4.2, with $k$ in the role of their number of small layers; the TTN content is which cores
are "small" at a saddle (the route) and the coupling $c=s_rs_q^{(K-k)/K}$.
(The integral follows from $b_v^2=b_v(0)^2+\beta(t)$, $\dot\beta=2c\prod b$.) If $\prod b<0$ at the start,
the smallest $\lvert b_v\rvert$ first shrinks through zero (sign correction), which lengthens the
plateau; this is one source of the broad seed-to-seed spread.

**What $\sigma_{\rm eff}$ is after the earlier phases (derived).** With accumulated utility,
$b_{\rm eff}^{-(k-2)}=b_0^{-(k-2)}-(k-2)\int_{\rm earlier}c(t)\,dt$. If the previous phase was the
escape of $q$ from the origin (route coupling $s_r a(t)^{K-k}$, $\dot a=s_qa^{K-1}$):
$$b_{\rm eff}^{-(k-2)}=b_0^{-(k-2)}-\frac{s_r}{s_q}\,a_0^{-(k-2)}\ \Longrightarrow\ \boxed{\Delta t_2=\frac{\sigma^{-(k-2)}\,(1-s_r/s_q)}{(k-2)\,s_r\,s_q^{(K-k)/K}}}\quad(b_0=a_0=\sigma).$$
So $\sigma_{\rm eff}=\sigma(1-s_r/s_q)^{-1/(k-2)}$ in the aligned case. The integrand
$a^{K-k}\,dt=a^{1-k}da/s_q$ is largest at $a\approx\sigma$: for $k\ge3$ the head start accrues mostly
during the *first plateau*, while the learned cores are still $O(\sigma)$, not during the escape of
$q$. Checked to 0.4% at $\sigma=2\cdot10^{-3}$ (table in section 7); verifier (MPS5, $(1.5,0.6)$,
$k=3,4,5$, reduced ODE matching full GF to 8 digits): $\Delta t_2/\text{pred}\to1.0000$ down to
$\sigma=3\cdot10^{-5}$, corrections $O(\sigma)$ for $k=3$. Special case $k=K$ (route uses every core): the route never
sees $q$, $c=s_r$ throughout, and $r$ escapes at the *absolute* time $\sigma^{-(K-2)}/((K-2)s_r)$
(odeco-like independence).

**Measurable signatures (derived, checked).** (i) Per seed (same init direction, $\sigma$ varied)
$\Delta t\propto\sigma^{-(k-2)}$. (ii) On a balanced route,
$\dot a_r=kA^{2/k}s_r^{2-2/k}a_r^{2-2/k}(1-a_r)$, so the local exponent is
$$\gamma(a_r)=\frac{d\log\dot a_r}{d\log a_r}=2-\frac2k-\frac{a_r}{1-a_r}\qquad(1,\ 4/3,\ 3/2,\ 8/5\ \text{for }k=2..5\text{ as }a_r\to0).$$
The route is balanced only once $b^2\gg\lvert b_u^2-b_v^2\rvert=O(\sigma^2)$; before that one large
amplitude makes it look like a lower-degree route ($\gamma$ too small). Window: read $k$ from
$a_r\in[10^{-4},10^{-2}]$ (verifier: 0.995-1.000, 1.334-1.344, 1.490-1.500 for $k=2,3,4$ there; at
$a_r=0.1$: 0.93 and 1.39 for $k=2,4$, against 0.89 and 1.39 from the formula with the saturation
term). (iii) Random init: measure the $b_v$ on the plateau and integrate the $k$-scalar route ODE;
this predicted the remaining plateau to 1-3% in the v1 checks (pre-fix RNG; not re-checked).

### 3.4 Which escape comes next (derived)
At small $\sigma$ the next event is the route with the smallest $k^*$ (log beats power, and smaller
powers beat larger ones), provided $\lambda_{\max}(\mathcal S)<s_q$; otherwise compare $k^*$ with
$2\ell^*$ (v3, rule 5.1). Ties in $k^*$ are decided by $c\,b_{\rm eff}^{k-2}$, i.e. by the coupling
*and* the random route amplitudes; for $k^*\ge3$ the latter usually dominate, so tied components come
in a seed-dependent order (HT4 example B4, balanced init: $q_A$ (0.3) before $q_B$ (0.6), both $k=5$).
Routes of higher
degree accumulate only a vanishing relative head start during a lower-degree plateau. The weight
order is therefore overridden by locality whenever a smaller component has a smaller $k^*$
(section 5). This is the TTN analogue of the "leap" picture of Abbe, Boix-Adserà & Misiakiewicz
(2023, arXiv:2302.11055): there the time to learn a new set of coordinates scales with the leap
(number of new coordinates needed at once); here the exponent is set by the number of *cores* that
must change at once, which depends on the tree. The exponent law itself is known in other models
(Rawal & DeWeese 2026 Thm 4.2, Kumar & Haupt 2025c Thm 8; section 9). Razin, Maman & Cohen 2022
give norm dynamics $\dot\sigma=\sigma^{2-2/L_\nu}L_\nu\langle-\nabla,\mathcal C\rangle$ for local
components of their CP-per-node hierarchical factorization ($L_\nu$ = number of children + 1), the
same $2-2/k$ law as any balanced product of $k$ factors. What we did not find (within literature.md)
is the identification of the degree at a TTN saddle with $\lvert\mathrm{Steiner}(N(D))\rvert$, hence the
tree-dependent exponent $k^*-2$ for the same teacher on different trees.

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
order $m$ and is confined by the positive quadratic term, so single-core piggyback is not an escape
route. (v3: a *generalized* piggyback can be. If two terminals lie in a set $C_1$ of degree
$\lvert\mathrm{Steiner}(C_1)\rvert<k^*/2$ and the rest in $C_2$ of degree $<k^*/2$, a new-entry cluster on
$\mathrm{Steiner}(C_1)$ plays the role of a rotation, and for $s_r>s_q$ the order is
$2\max(\deg C_1,\deg C_2)<k^*$; `theory_saddle_order.md` Cor. 2c.)
- $m=1$ is the $k=1$ case (gradient nonzero).

Checked exactly for 18 (tree, $D$) cases in both orders (e.g. binary4, $s_q=0.5$, $s_r=1$:
$-0.39685,+1.19055$ predicted and measured).

**Several residual components: the piggyback graph (proven; v1 was wrong here).** The bullets above
treat one residual $r$. A rotation of core $u$ toward a site pattern $w$ is *one* direction, shared by
every residual whose pattern on $u$'s sites is $w$. With residuals $\{r\}$ the rotation block is
$$H_{\rm rot}=a^{2(K-1)}I-a^{K-2}S=a^{K-2}\,(s_qI-S),$$
where $S$ is the weighted adjacency matrix of the **piggyback graph**: nodes $(u,w)$, one edge
$(u,r|_u)-(v,r|_v)$ of weight $s_r$ for every residual with $N(D_r)=\{u,v\}$ (residuals with
$m_r\ge3$ add no quadratic term, and rotations toward patterns used by no residual have eigenvalue
$a^{K-2}s_q$). Mixed tensors of different nodes are orthogonal, so the GN part is diagonal, and the
radial part is PSD. Hence $\lambda_{\min}(H_{\rm rot})=a^{K-2}(s_q-\rho(S))$ and
$$\theta_q\ \text{is a strict saddle}\iff\rho(S)>s_q\ \text{ or a }k^*=2\text{ route exists.}$$
A **star** (residuals $r_1..r_n$ sharing one node) has $\rho(S)=(\sum_is_{r_i}^2)^{1/2}$; a triangle
with equal weights $s$ has $\rho=2s$. So $\theta_q$ can be strict although every residual is smaller
than $s_q$ and every $k^*\ge3$. With $d=2$ stars are common: any two residuals that differ from $q$
on the same single-site core share a node there, since only one other site vector exists. Checked
(exact $\theta_q$, script `r1`, and verifier V1), with every prediction $a^{K-2}(s_q-\rho(S))$ matched
to 6 digits:

| teacher ($q$ first, weight) | tree | $\rho(S)$ | predicted / measured $\lambda_{\min}$ |
|---|---|---|---|
| $0000$ (2); $1010$, $1001$ (1.6 each) | mps4 | 2.263 | $-0.371573$ / $-0.371573$ |
| same | ht4 | 2.263 | $-0.431073$ / $-0.431073$ |
| same | binary4 | 2.263 | $-0.331034$ / $-0.331034$ |
| $0^8$ (2); $D=\{0,2\}$, $\{0,4\}$ (1.6 each) | binary8 | 2.263 | $-0.431073$ / $-0.431073$ (verifier) |
| $0^6$ (2); $101000$, $100010$, $001010$ (1.1 each), triangle | mps6 | 2.2 | $-0.317480$ / $-0.317480$ |
| controls: star with 1.2, 1.2; triangle with 0.9 each; two disjoint pairs 1.6, 1.6 | mps4, mps6 | 1.70, 1.8, 1.6 | none / none |

On binary4 the star is just the DLN statement that $q$ is not the top singular mode:
$M^*=W^*_{(01)|(23)}$ has singular values $1.6\sqrt2=2.263$ ($r+r'$) and 2 ($q$).

**The generalized piggyback graph (v3; `theory_saddle_order.md`).** The piggyback graph is level
$\ell=1$ of a hierarchy. At level $\ell$ the nodes are pairs $(C,w)$ of a core set $C$ of degree
$\lvert\mathrm{Steiner}(C)\rvert\le\ell$ and a pattern $w$ on $C$ that differs from $q$ on every core of
$C$; a residual $r$ joins $(C_1,r|_{C_1})$ and $(C_2,r|_{C_2})$, with weight $s_r$, for every split
$N(D_r)=C_1\sqcup C_2$ into sets of degree $\le\ell$. A multi-core node is reached by new entries on
$\mathrm{Steiner}(C)$ (a cluster toward the non-residual pattern "$w$ on $C$, $q$ elsewhere"), a
singleton node by an old rotation. Proven for one learned component: the saddle order is $k^*$ if
the graph on the levels below $k^*/2$ has $\rho<s_q$ (with the route), and $2\ell^*<k^*$ if level
$\ell^*<k^*/2$ is the first with $\rho>s_q$ (for the upper bound: pattern teacher, $s_r>0$, a spare
bond index per cluster). Example: MPS6, $q=0^6$ (1), $r_1=110001$, $r_2=110010$ (0.8 each): no
residual has two terminals, so the level-1 graph is empty and the Hessian is PSD, but the level-2 node
$(\{0,1\},11)$ is the centre of a star with leaves $(\{5\},1)$, $(\{4\},1)$, $\rho=0.8\sqrt2>1$: order 4,
$k^*=5$.

### 4.2 Binary $N=4$ is a depth-3 deep linear network (proven)
For the binary tree with $N=4$ (cores $A$: sites 0,1; $B$: sites 2,3; root $R$), with
$\tilde A=[\mathrm{vec}A_{:,:,\alpha}]_\alpha\in\mathbb R^{d^2\times\chi}$ and likewise $\tilde B$,
$$W_{(01)|(23)}=\tilde A\,R\,\tilde B^{\top},\qquad L=\tfrac12\lVert\tilde AR\tilde B^\top-M^*\rVert_F^2,$$
i.e. exactly a depth-3 linear network fitting the $d^2\times d^2$ matrix $M^*=W^*_{(01)|(23)}$. Every
statement about binary4 is therefore a statement about deep linear networks: saddles are the
SVD truncations of $M^*$ (Jacot et al. 2021, arXiv:2106.15933); Achour, Malgouyres & Gerchinovitz
(JMLR 2024, arXiv:2107.13289, Thm 7) prove that critical points selecting non-top singular values are
**strict** saddles, while those selecting the top $r$ values and "tightened" (which includes every
balanced one) are **non-strict** for depth $\ge3$. Gradient flow from small init only visits
*approximately* balanced points ($Q_e=O(\sigma^2)$), and it can pass near non-top critical points
(the star example of 4.6 is one on binary4), so "every saddle visited by gradient flow is
tightened" (v1) was too strong. (Their Assumption 1 asks for a full-rank target with distinct singular values; our targets are
low-rank, so our direct computation is what covers them.) H1b and H3 for binary4 are thus known
results; the content below is the extension to other trees.

### 4.3 Classification for one learned component $q$ and ONE residual $r$ (two-component teachers; exact at $\theta_S$)

Valid only when the piggyback graph has a single edge (or none); with several residuals use
$\rho(S)>s_q$ (4.1), which can make the correct-order rows below strict.

| route data | $s_q>s_r$ (correct order) | $s_q<s_r$ (wrong order) |
|---|---|---|
| $k=1$ | not critical | not critical |
| $k=2$ (then $m=2$) | strict, $\lambda_{\min}=-s_ra^{K-2}$ | strict, $\lambda_{\min}=-s_ra^{K-2}$ (piggyback $-(s_r-s_q)a^{K-2}$ is less negative) |
| $k\ge3$, $m=2$ | non-strict (order-$k$ saddle) | **strict via piggyback**, $\lambda_{\min}=-(s_r-s_q)\,s_q^{(K-2)/K}$ |
| $k\ge3$, $m\ge3$ | non-strict (order $k$, v3: proven) | non-strict; order $k$ without a low split, $2\ell^*<k$ with one (v3) |

(v3) Saddle orders, proven in `theory_saddle_order.md`: correct order: $k$ on every tree (Cor. 2a);
wrong order with $m\ge3$: $\min(k,2\ell^*)$, $\ell^*=\min$ over splits of $N(D)$ into two parts of
$\max(\deg C_1,\deg C_2)$, a low split being one with both degrees $<k/2$ (Cor. 2b-c; index
condition: $\chi_e\ge2$ on both Steiner trees). Low splits need $k\ge5$. Examples: MPS6, $D=\{0,1,5\}$:
order 4, not 6; HT4 and binary8 with $D$ = all sites (two leaf pairs of degree 3): order 6, not 7.
Three-site $D$ on HT4 ($k=6$) have no low split ($\{l_0,l_1\}$ has degree $3=k/2$): order 6.

General principle (proven at balanced $\theta_S$, any learned set): the Hessian is block diagonal
between a *fixed-rank block* (old entries: GN plus the piggyback couplings) and a *rank-increasing
block* (new entries: zero except $\pm$ couplings of $k^*=2$ routes). An old/new cross term would need
an old index to meet a new one, and the GN cross term vanishes by F1. So the saddle is strict iff the
fixed-rank block (the Hessian of the fixed-rank approximation problem at $W_S$) has a negative
eigenvalue, or a $k^*=2$ route exists. This is the TTN version of the "non-top singular values" case
of Achour et al. For one learned component the first condition is $\rho(S)>s_q$: a single $m=2$
coupling in the wrong order, or several $m=2$ couplings combined (stars, cycles) in either order. For
two-component teachers with $m\ge3$ the wrong-order state is a strict local minimum of the fixed-rank
problem (PD block). This is the TTN analogue of every odeco component being a local optimum of rank-one
approximation (Anandkumar et al. 2014).

### 4.4 Plateau sequences after a wrong first escape (two-component teachers; derived, checked in v1, pre-fix RNG)
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
**v3: only without a low split (4.3).** With one, at small enough amplitudes a cluster on each part
escapes first, in $\propto b^{-(2\ell^*-2)}$, toward the non-residual patterns "$r$ on one part, $q$
elsewhere"; then $q$ is forgotten ($L\to\tfrac12s_q^2$) and relearned (derived;
`theory_saddle_order.md` sections 6-7). Locality data, HT4, $D$ = all sites, weights $(1,0.5)$,
fan-in init, $\chi=2$: 6 of the 7 runs that learn the smaller component first ($\sigma\ge0.40$) forget it
when the larger one is learned (overlap $<0.1$, plateau at $L\approx0.125$) and relearn it. The
`order` agent's binary8 odeco runs from the wrong saddle ($\sigma\ge0.15$) keep it and follow the
route law ($t_1\propto\sigma^{-5.25\pm0.27}$), as expected at or above the estimated crossover
$b\approx0.155$ between the two mechanisms.

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
$P(\text{smaller first})\propto\sigma^{n(s_r-s_q)/s_q}$ as $\sigma\to0$ (slow for close weights:
$\sigma^{0.11}$ at $s_q/s_r=0.9$), and when it happens case 1 of 4.4 applies. Tensor-like pairs:
$\sigma$-independent probability (H1), but then the wrong saddle is non-strict (case 3), no correction
happens (v3: unless $N(D)$ has a low split, e.g. HT4 or binary8 with $D$ = all sites, where a
higher-order correction forgets and relearns $q$, 4.4), and the final state is still $W^*$. **So, for two-component teachers, the H1b correction
through a strict saddle exists only for matrix-like pairs ($m=2$, $k\ge3$), exactly where the wrong
order becomes rare as $\sigma\to0$.**

General form (derived; v2): the same computation with several residuals gives the Riemannian block
$a_s^{K-2}(S-s_qI)$ on the rotations and $a_s^{K-2}\big(\begin{smallmatrix}-s_qI&M_C\\M_C^\top&-s_qI\end{smallmatrix}\big)$
on the new entries of each adjacent pair $C$, with $M_C=\sum_{r:\,N(D_r)\subseteq C}s_rU_r|_C$. So
$q$ is a local maximum of the spherical problem if $\rho(S)<s_q$ and $\sigma_{\max}(M_C)<s_q$ for
every adjacent pair, and it is not one if either exceeds $s_q$ (equality: decided at higher order). A star with $\rho(S)>s_q$ therefore stops the
first escape from settling on the *largest* component. The flow then continues to the best rank-one
approximation (a superposition). On an MPS that point is a strict saddle, because its residual is
generic and adjacent pairs give $k^*=2$ routes (5.6). On binary and HT trees it need not be.

### 4.6 Stars: the largest component is passed, forgotten and relearned (v2; Hessian proven, dynamics checked)

Teacher (MPS4, $d=2$): $q=0000$ ($s_q=2$), $r=1010$, $r'=1001$ (1.6 each). Every pair differs on two
cores, $k^*(r)=3$, $k^*(r')=4$, and there is no $k^*=2$ route from $q$. The pairwise rules of v1
predicted a first plateau at $q$ ($L=2.56$) and then $r$, $r'$. Instead $r$ and $r'$ share the node
(core 0, $e_1$), $\rho(S)=2.263>2$, and $\theta_q$ is strict ($-0.3716$, 4.1). Gradient flow (random
fan-in init, $\sigma=0.05$, LSODA rtol $10^{-10}$; verifier seeds 0-3, v2 seed 4):

- In 4 of 5 seeds the first escape passes near $q$ ($a_q$ peaks at 0.95, 0.94, 0.68, 0.99) and leaves
  within $\approx10$ time units. In the fifth ($a_q\le0.43$) it heads straight for the superposition.
  $a_q$ then falls to 0.01-0.10 while $a_r=a_{r'}$ rise, through the best rank-one point ($L=2.4595$,
  HOPM value $2.0496>s_q$), which is strict.
- Next comes the best fit with bond ranks $(1,1,2)$, i.e. "$r$, $r'$ learned, $q$ forgotten"
  ($L=\tfrac12s_q^2=2.0$, $a\approx(0.01,1,1)$). Its plateau length depends on the seed: 98, 47, 4,
  28 and 0 time units (the remnant of $q$ seeds the relearning route).
- Then $q$ is relearned and $L\to0$. No seed has a plateau at $L=2.56$ (at most 2.5 time units
  within 0.01 of it).
- Sub-critical control, same teacher with $1.2,1.2$ ($\rho(S)=1.70<2$), 3/3 seeds (v2): plateau at
  $q$ ($L=1.44$ for 52, 111 and 29 time units, $a_q\ge0.998$), then $r$ ($k=3$), then $r'$.

This is an H1b-type correction caused by *two smaller components together*. With $d=2$, shared
nodes are common in teachers with three or more components. Whether they make $q$ strict depends on
the weights ($\rho(S)>s_q$), so learning-order experiments (Q4) should compute $\rho(S)$ for their
teachers. The $\sigma$-scaling of the effect (depth of forgetting, length of the $L=2.0$ plateau) is
not predicted here.

## 5. More components, nested structure, and the same teacher on different trees

(The "Checked" statements in 5.2-5.6 are v1 runs, made before the RNG fix. The k tables and
plateau levels were re-derived by the verifier and the tables recomputed in v2 (`r5`); the dynamics
of B4, E8, N4 and of the generic teacher were not re-run.)

### 5.1 Rule for the whole sequence (derived at leading order; heuristic as an algorithm)

1. The first escape (degree $K$ from the origin) ends near a critical point of the bond-rank-one
problem (H1 and theory_escape decide which). For pattern teachers it is a single component $q$ only if
$q$ is a local maximum of the spherical problem (4.5: $\rho(S_q)\le s_q$ and $\sigma_{\max}(M_C)\le s_q$
on adjacent pairs; for two-component teachers: no larger component forming a matrix-like pair). When
components differ on a single core it is their best bond-rank-one combination (e.g. the top singular
mode of a matricization, section 5.5). With a star ($\rho(S_q)>s_q$) the flow passes $q$ and reaches
a superposition (4.6).
2. At every saddle first check the fixed-rank block. If it has a negative eigenvalue (wrong order,
stars), the next event is in $\log$-time along the most unstable direction: a rotation of the learned
structure, or a $k^*=2$ route if that is more unstable. Proposition 1 does not apply to the rotation.
Otherwise compute, for each candidate route $C$, the local residual $R_C$; escape
degree $k^*=\min\lvert C\rvert$. The next event is the smallest-$k^*$ route (ties: larger coupling,
larger head start). Its plateau lasts $\propto\log(1/\sigma)$ ($k^*=2$) or $\propto\sigma^{-(k^*-2)}$;
$k^*=1$ routes fire without a plateau. **v3:** with one learned component also compute the
generalized piggyback graph (4.1) on the levels $\ell<k^*/2$; if a level exceeds $s_q$, let $\ell^*$ be
the first. Compare $k^*$ with $2\ell^*$: the next event is the mechanism with the smaller order
(plateau $\propto\sigma^{-(2\ell^*-2)}$ for the cluster), and after a cluster escape the next saddle is
again given by step 3 (pair-star: $q$ is forgotten). This comparison is **derived** (a leading-order
race; at finite $\sigma$ the prefactors decide, e.g. the crossover $b\approx0.155$ in 4.4), not proven.
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
**Each active route contributes exactly one negative direction** (on the route block the Hessian is
$-s_rAb^{k-2}(J-I)$, whose $k-1$ positive directions are the product-preserving rescalings, i.e. gauge):
$$\lambda_{\min}\simeq-\lambda=-s_rs_q^{(K-2)/K}\ \ (k=2),\qquad \lambda_{\min}\simeq-(k-1)\,s_rA\,b^{k-2}=O(\sigma_{\rm eff}^{k-2})\ \ (k\ge3,\ \text{balanced route}),$$
with $b=(s_ra_r/A)^{1/k}$ estimated from the measured overlap $a_r$ at that point (verifier: within a
factor 1.3-1.4 for $k=3$, 3.5 for $k=4$). **But the count "one negative eigenvalue per route" holds only
for $k\le3$ (corrected in v2).** The *visible* new entries (adjacent new-index pairs whose product
connects to learned structure; section 8) decay only on the $\sigma^{-2}$ time scale. Their products
are residual pieces of order $\sigma^2$, which add $O(\sigma^2)$ indefinite terms to the new-index
block. Those are negligible next to the route eigenvalue for $k\le3$ but of the same order at $k=4$.
Verifier: at $k=4$ midpoints a second negative eigenvalue at 10-54% of $\lambda_{\min}$, also
$\propto\sigma^2$, in 3/3 seeds. Diagnosis (v2, `r4`, verifier's seed 1, $\sigma=0.08$, gauge-fixed
old/new bond basis): the 4-entry route block has exactly one negative eigenvalue ($-1.2\cdot10^{-4}$).
The 16-entry new-index block has three ($-4.6,-2.0,-0.9\times10^{-4}$, raw). The extra ones involve
new entries carrying the site vector that neither $q$ nor $r$ uses at sites 1, 2, with amplitudes
$1$-$3\cdot10^{-2}$, as large as the route entries (mechanism heuristic, one point checked). For
$k\ge5$ the midpoint lies beyond the $\sigma^{-2}$ decay time, so whether these terms still dominate
$\lambda_{\min}$ there is open. The remaining spectrum: $P_{\rm old}-\sum r_e^2$ "old" eigenvalues
($O(1)$) and $P_{\rm new}-\sum_e(2r_ec_e+c_e^2)-n_{k=2}$ near-zero ones.

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
| C8 | conservation-law trap | ht4, B4, $\sigma=0.2$, seeds 0-2, fan-in vs gauge-balanced init | fan-in: stall at $L=0.045$ in 3/3 seeds up to $t=3\cdot10^6$ (leaf route amplitudes $\sim10^{-14}$, parent new-slice norm$^2=-Q_{\nu\nu}(0)$ to 4 digits; [trap update 2026-10-10] this is exact at any dead end, the corollary of Prop. 1 in `theory_trap.md`); balanced: route alive in 3/3, $q_A$ learned in 2/3, third predicted at $t\approx3.4\cdot10^6$ |
| C9 | E8 on MPS8 vs binary8 | $\sigma=0.25,0.2$, seeds 0,1, $\chi=3$ | MPS8 order $q_0,q_A,q_B$ (2/2 seeds), $q_B$ gap ratio 2.40/2.59 (pred 2.44); binary8 tie order seed-dependent, gap ratios 1.91-2.09 (pred 1.95), $t_1$ ratios 3.81 and 3.05 (pred $1.25^6$, $1.25^5$) |
| C10 | generic (non-orthogonal) CP teacher | `random_cp_teacher(4,2,[1,0.5],seed=3)`, mps4 and binary4, $\sigma=0.05\to0.0125$ | MPS4: $\lambda_{\min}=-0.129,-0.132,-0.136$ (strict), log gaps; binary4: $\lambda_{\min}\propto\sigma$, gaps $\propto\sigma^{-1}$ (section 5.6) |
| V1-V3 | verifier, post-fix, independent code | exact $\theta_S$ (as C1); aligned MPS5 $(1.5,0.6)$, $k=2..5$, $\sigma\to3\cdot10^{-5}$; random MPS4 $(1.2,0.5)$ | C1 values to 6 digits; $k=2$ slope 0.78408 vs 0.78405; $\Delta t_2$/pred $\to1.0000$; random: slope 1.066 vs 1.065, exponents $-0.986,-0.994$ ($k$=3), $-1.996,-1.999$ ($k$=4); $\gamma$ 1.000, 1.33-1.34, 1.49-1.50 |
| C11 | piggyback-graph criterion at $\theta_q$ (v2 `r1`, verifier V1) | stars on mps4, ht4, binary4, binary8; triangle on mps6; 3 sub-critical controls | $\lambda_{\min}=a^{K-2}(s_q-\rho(S))$ to 6 digits in all 5 strict cases; no negative eigenvalue in the controls (table in 4.1) |
| C12 | star dynamics (verifier V4, v2 `r2`) | mps4, $(2;1.6,1.6)$ and $(2;1.2,1.2)$, $\sigma=0.05$, random fan-in init | super-critical: 5/5 seeds forget $q$ ($a_q\to0.01$-$0.10$) and relearn it; sub-critical: 3/3 plateau at $q$ ($L=1.44$), then $r$, $r'$ (section 4.6) |
| C13 | F2, disconnected $J$ (v2 `r1`) | mps5, $D=\{0,4\}$, $J=\{0,1\}\cup\{3,4\}$, $x=y=t^2$ | $(L-L_S)/t^4=0.50000$ and $-0.43528$ for $(s_q,s_r)=(1,0.5),(0.5,1)$; predicted $a(s_q-s_r)$ exactly |
| C14 | trap, joint condition (verifier V6, v2 `r3`) | ht4, B4, teacher seed 3, $\sigma=0.2$, init seeds 0-6 | dead iff all route ends unfavourable; pinned otherwise (table in section 8; [trap update 2026-10-10] two-ended routes, so "all" = "at least two", the rule of R12) |
| C15 | $k=4$ plateau Hessian (verifier V5, v2 `r4`) | mps4 $(1.2,0.5)$, $D=\{0,3\}$, seeds 0-2 | two negative eigenvalues $\propto\sigma^2$; route block has one, new-index block three (section 6.2) |

RNG note: C3-C10 ran before the teacher/init stream fix (commit 57d5262), and their runs with init
seed 0 used teacher seed 0. C1, C1b and C2 involve no random init and are unaffected. Exponents,
slopes and Hessian values were re-checked after the fix (V1-V3, C11-C15). Prefactors, learning
orders and the v1 trap counts were not.

## 8. Caveat found numerically: the conservation-law trap (HT4/B4 data at $\sigma=0.2$; theory updated 2026-10-10 from `notes/theory_trap.md`)

> **[trap update 2026-10-10]** The theory of this section is superseded by `notes/theory_trap.md` (v1.1,
> after its verification). Replaced below: "Which ends die", "How often", "Timescales" and the balanced-init
> statement; one bracket in the mechanism list. Kept: the HT4/B4 observations, the data table and the
> tally. Two further changes: (i) the HT4/B4 traps happen after two learned components ($q_0$ and $q_B$,
> or $q_0$ and $q_A$). With two learned components the siteless parent's off-diagonal visible entries give
> a second-order normal product ($j=2$), and that kills the leaves. At a rank-one saddle an HT leaf has
> visible degree $j=3$ and dies only through three-core clusters ($k\ge6$). So these runs are not
> instances of the rank-one mechanism. (ii) The section-title tags were: mechanism derived in a reduced
> model, occurrence empirical at $\sigma=0.2$, $\sigma\to0$ conjecture. The current tags are in R12.

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
decay at a rate $\sim\lvert Q_{\nu\nu}\rvert\sim\sigma^2$. [trap update 2026-10-10: only their parts normal to
the learned pattern decay, at rate $A^2\lvert C_u\rvert$; the pattern parts are tangent and free
(`theory_trap.md` T1-T2).] The lighter side decays to zero, the heavier
side freezes with norm$^2=\lvert Q_{\nu\nu}\rvert$, keeping visible components that confine the lighter
side's route entry. The route then cannot grow (its drive is a product of small amplitudes).
- **Which ends die** [trap update 2026-10-10; v1 said "one unfavourable sign", v2 "every end
unfavourable", with the end quantity $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$]. The conserved quantity of an end $u$
with route partner $w$ is $C_u=X_u^{\perp\top}X_u^\perp-X_w^{\perp\top}X_w^\perp$ of the normal-visible blocks. The
end's normal block goes to $[C_u]_+$ (proven in the reduced flow, checked in full gradient flow), so with
one new direction the end dies iff $C_u<0$, at rate $\gamma_u=A^2\lvert C_u\rvert$. Exactly,
$C_u=Q_{\nu\nu}+\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+(y_\parallel^2-x_\parallel^2)$, and the last term is
$O(\sigma^3)$ after a small-init escape. The v2 quantity kept only the route entry of the partner's
non-visible entries $Y_{\rm nv}$, which is why it over-predicts death: per run (fan-in, $k\ge4$) 36% of its
predicted-dead runs stall, against 89% with $C_u$.
  - The route drive on a dying end is proportional to the product of the *other* route amplitudes.
    While another end stays alive, it pins the end at $x_*\approx$ drive/decay rate $\propto\sigma^{k-3}$.
  - Two dying ends: the dead state is linearly stable if $\gamma_1\gamma_2>\kappa^2$ (mutual drive
    $\kappa\sim s_rA\prod_{\rm inner}b\sim\sigma^{k-2}$), i.e. always as $\sigma\to0$ for $k\ge5$.
  - Hence, for $k\le5$ as $\sigma\to0$: **dead iff at least two ends die** (not all ends), pinned if
    one does (derived). From $k=6$ three-core clusters through inner route cores (rate $\propto\sigma^4$) are
    a further death channel: marginal at $k=6$ and dominant for $k\ge7$ against a live route, dominant
    from $k=6$ against the pinned escape. So pinned routes can die there, and the pinned escape law
    $\propto\sigma^{-(2k-6)}$ is a conjecture.
  - For the two-ended routes of the table below, "both", "all" and "at least two" coincide, so the
    table's dead/pinned classification stands (its $Q$ column is the v2 quantity, not $C_u$).
- **How often** [trap update 2026-10-10]. Fan-in init gives $E[Q_e]=\sigma^2(1-\chi_p/\chi_e)I$ (proven;
re-derived by the verifier; $\chi_p$ = bond dimension of the parent's own parent edge). The heuristic
that followed here is **withdrawn**: $P(Q_{\nu\nu}>0)=0.30$ per HT leaf, hence $P\approx0.5$ dead for a
two-ended HT route, and $P\approx0.25$ for MPS with $\chi=d$ ("stalls less often", contradicted by
`notes/locality.md` 7.5 (ii)). The decisive quantity is $C_u$, not $Q_{\nu\nu}$, and an HT leaf has visible
degree $j=3$ at a rank-one saddle. The replacement (`theory_trap.md` section 8):
  - Saddle start, one new direction on the end edge ($c_e=1$):
    $P(C_u<0)=I_{F_u/(F_u+F_w)}(n_u/2,n_w/2)$ ($F$ = fan-ins, $n$ = normal physical rows). This gives
    0.392 at an MPS tree-leaf end and 0.5 at the MPS root end ($d=\chi=2$).
  - A two-ended route with $k=5$ whose ends both have partners with sites is dead with probability
    $p_up_{u'}$ as $\sigma\to0$; for $k\ge6$ the inner-cluster channel adds deaths that the law does not
    count.
  - At a capped end bond with $d\ge3$ there are $c_e=d-1$ new directions, and death needs $C_u\le0$ on
    all of them: about 0.023 ($d=3$) and $1.5\cdot10^{-4}$ ($d=4$), by Monte Carlo.
In trapped runs the parent slice along $\nu$ ends with norm$^2=-Q_{\nu\nu}$ (conservation with $x=0$; exact
at any dead end, `theory_trap.md` Prop. 1; v1: $0.0263$; final leaf new-column norm$^2\sim10^{-28}$).
- **Data** (HT4, B4, $\sigma=0.2$, fan-in init, LSODA rtol $10^{-10}$; $Q$ = $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$ at the
plateau start for v2 seeds 4-6, $Q_{\nu\nu}$ otherwise; dead: leaf $x^2\lesssim10^{-26}$, pinned:
$10^{-11}$-$10^{-7}$):

| run (teacher basis) | stalled route | $Q$ at the two route leaves | leaves at the end | outcome |
|---|---|---|---|---|
| v1 seeds 0-2 (seed 0, pre-fix RNG) | $q_A$ | seed 0: $-0.026$, $-0.021$ | dead | stalled to $3\cdot10^6$ |
| verifier seed 0 (seed 3) | $q_A$ | $+0.011$, $-0.025$ | alive, pinned ($7.7\cdot10^{-8}$) | escaped at $t\approx6\cdot10^5$ |
| verifier seeds 1, 2 | $q_A$ | $-0.034,-0.018$; $-0.028,-0.010$ | dead | stalled to $10^6$ |
| v2 seed 3 | $q_B$ ($q_A$ learned first) | $-0.034$, $-0.050$ | dead | stalled to $10^6$ |
| v2 seed 4 | $q_A$ ($a_{q_A}=0.019$ at the plateau start) | $-0.024$, $-0.011$ | dead | stalled to $2\cdot10^5$ |
| v2 seed 5 | $q_A$ | $+0.052$, $-0.048$ | alive, pinned ($7.8\cdot10^{-11}$) | not escaped by $10^6$ |
| v2 seed 6 | $q_A$ | $-0.011$, $-0.0001$ | dead | stalled to $10^6$ |
| v2 seed 7 | none | ($q_A$ rose together with $q_B$) | alive | all learned by $t\approx3\cdot10^3$ |
| v2 seed 8 | $q_B$ ($q_A$ learned first) | $-0.120$, $-0.042$ | dead | stalled to $2\cdot10^5$ |

Every run with all route ends unfavourable died; every run with a live end had the other end pinned,
not dead. Stalls at a fixed budget therefore include slow pinned escapes. Tally with teacher seed 3
(init seeds 0-8): 6 dead, 2 pinned, 1 no stall; the joint rule classified all 8 stalls correctly. The stalled component can be the
larger one ($q_B$, 2 runs) when the tie of section 3.4 goes to $q_A$.
- **Timescales and $\sigma\to0$** [trap update 2026-10-10; replaces a conjecture, `theory_trap.md` T5,
T8, derived]. An end with visible degree $j$ (cores in the smallest cluster through its new index whose
product is normal to the learned manifold: $j=2$ if the partner carries sites, $j=3$ for a leaf whose
partner is siteless) decays in time $\propto\sigma^{-2(j-1)}$, against the route escape $\propto\sigma^{-(k-2)}$.
So an end can kill its route only if $k\ge2j$:
  - $k\le3$: never (0 stalls in 5,744 committed runs);
  - $k=4$: a $\sigma$-free competition, settled by a reduced route ODE from the plateau-start values
    (derived for one new direction per route edge);
  - $k=5$: the sign test on the $C_u$ (dead iff at least two ends have $C_u<0$);
  - $k\ge6$: the sign test plus the inner three-core channel, which the per-run criterion does not
    include.
The plateau-start state is $\sigma$ times a function of the init direction, so the outcome is a property
of the init direction, not of $\sigma$ (corrections $O(\sigma)$). The dead-state stability condition
$\gamma_1\gamma_2>\kappa^2$ is kept, with $\gamma=A^2\lvert C_u\rvert$. The earlier conjecture
($P(\text{stall})\to P(\text{all ends unfavourable})\approx0.5$ for two-ended HT routes) is withdrawn.
- The point reached is not a local minimum (moving the frozen visible components is free and then
opens a $k=2$ descent), but gradient flow does not move along those flat directions; SGD noise would.
- **Balanced init removes the trap (v1: 3 seeds, pre-fix RNG; conjecture in general; [trap update
2026-10-10] derived for tree-leaf ends only, see the end of this item).** Gauge-balancing
the fan-in init ($Q_e=0$ exactly, same $W_0$) keeps the leaf route amplitudes alive: HT4, B4,
$\sigma=0.2$, seeds 0/1/2: balanced init learns $q_A$ at $t=2850$ (seed 0) and $3.7\cdot10^5$ (seed 1),
and for seed 2 the route is alive at $t=2\cdot10^5$ (leaf amplitudes $-0.0196$, product negative,
i.e. in a sign-correction phase) with escape predicted by the route ODE at $t\approx3.4\cdot10^6$.
With $Q_{\nu\nu}=0$ both sides of an edge lose their visible components together and the leaf column
converges to the norm of the parent's route component instead of to zero.
[trap update 2026-10-10] With $Q_e=0$, $C_u=\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+O(\sigma^3)\ge0$ at an end
that is a tree leaf ($X_{\rm nv}=0$), so tree-leaf ends cannot die at leading order (derived; `theory_trap.md`
T9). Ends with extra legs (pattern teachers, e.g. MPS $\{1,5\}$) still can. In the s2s-core odeco runs
no balanced run traps (0/150), but 69/150 stall (binary8 27/30, HT4 12/30, MPS8 30/30). So balanced
init removes the death of leaf ends, not every stall.

Practical consequence for Q3 experiments: on HT trees (and for $k\ge4$ routes in general), report the
*fraction* of stalled runs. Split stalls into dead routes (leaf new-column norm$^2\lesssim10^{-26}$,
roundoff) and pinned slow escapes (one end at $10^{-11}$-$10^{-7}$, the other alive), and record
$Q_{\nu\nu}$ at every route end ([trap update 2026-10-10] better: $C_u$, computable from the init with
`experiments/theory_checks/trap/`). Use gauge-balanced initialization when the aim is to measure escape
exponents. The trap is a property of unbalanced initialization (any $Q_e\ne0$; [trap update 2026-10-10]
"fan-in init makes it likely on HT leaves" holds only after a second learned component or for $k\ge6$),
not of the TTN landscape. [trap update 2026-10-10] Balanced init still leaves route ends with extra legs
exposed ($C_u$ can be negative there, `theory_trap.md` T9), although none of the balanced locality stalls
was dead in the stall diagnosis.


## 9. Relation to the literature: what is known, what is new (v2)

Based on the summaries in `notes/literature.md` v1.1; the papers were not re-read for this revision,
and "not found" means not found in the papers that file covers.

| result here | overlapping known result | what is new here |
|---|---|---|
| Prop 1 / R1: escape from a saddle = first-escape problem of the route sub-TTN | Kumar & Haupt 2025c (Neuron Pursuit) Thm 8: dormant weights of lowest homogeneity $L$ stay $O(\delta)$ for $\sim\delta^{-(L-2)}$ and align with a KKT point of the residual's correlation, *if the active weights sit at a local minimum*; AGF accumulated utility (Kunin et al. 2025) | $L=k^*=\lvert\mathrm{Steiner}\rvert$ for TTN routes; the cases where the local-minimum hypothesis fails (wrong order, stars) and what happens then (sections 4.4, 4.6) |
| R2: descent of order $k^*$ along a route | Furman et al. 2026, Lemmas D.7-D.8, Remark D.10: descent $t^N\langle R,U\rangle$ along $U$ supported on a connected component of rank-deficient edges with $N$ nodes (an upper bound on the order) | minimizing over connected sub-routes: $k^*=\lvert\mathrm{Steiner}(N(D))\rvert\le N$, often much smaller ($N=K$ at a rank-one saddle with $\chi\ge2$) |
| R2 (v3): lower bound $k^*$ under $\lambda_{\max}(\mathcal S)<s_q$, level version, sharpness ($2\ell^*$), counterexamples from $k^*=5$ (`theory_saddle_order.md`) | Lower bounds on saddle orders: literature.md (section 5, item 3) lists the question as open; known special cases are the DLN strict/non-strict classification (Achour et al. 2024, Thm 7: order 2 versus $>2$), one parity saddle with $L-L_0\ge-C\lVert\theta-\theta_0\rVert^{n/2+1}$ (Furman et al. 2026, Prop. 5.3), and exhibited (not bounded) Tucker descents of order 2-4 (Frandsen & Ge 2022, proof of Lemma 7). Tools: set-partition moment-cumulant formulas and the factorization of cumulants of independent blocks (classical) | the lower bound at a learned rank-one saddle of any TTN, the generalized piggyback operator, the level version, the counterexamples with PSD Hessian (not found in literature.md or the citation audit, part b) |
| R3: $k^*=2$ is strict | Furman 2026, Lemma D.9 ($N=2$ gives $t^2\langle R,U\rangle$) | exact $\lambda=s_rs_q^{(K-2)/K}$ with multiplicity $c_e$; block-diagonal Hessian; strictness criterion including the fixed-rank block |
| R3/R4: no plateau / $\log(1/\sigma)$ / $\sigma^{-(k-2)}$ for $k=1,2,\ge3$ | Rawal & DeWeese 2026, Thm 4.2 ($r$ small layers); leap exponents (Abbe et al. 2023) | which cores are "small" at a TTN saddle (the route); coupling $c=s_rs_q^{(K-k)/K}$; same teacher, different exponents on different trees |
| R4: $k=2$ head start $\sigma^{1-s_r/s_q}$ | Zhang, Saxe & Latham 2026, Thm 4 (separation $O(\sigma^{1-s_{r+1}/s_1})$); AGF; cumulative clocks (Pesme & Flammarion 2023) | the $k\ge3$ version $\sigma_{\rm eff}=\sigma(1-s_r/s_q)^{-1/(k-2)}$ (elementary; not found) |
| R5: growth exponent $2-2/k$ | standard for a balanced product of $k$ factors (Arora et al. 2019; Razin et al. 2022, per node $2-2/L_\nu$) | its use as a $k$-meter on TTN plateaus, with saturation term and window |
| R6/R7: piggyback classification, H1b | Achour et al. 2024, Thm 7 (DLN strict/non-strict; binary4 is a DLN); Anandkumar et al. 2014 (odeco components are local optima of rank-one approximation) | the piggyback graph and $\rho(S)>s_q$ on general trees; stars and cycles make the largest component strict |
| R12: conservation-law trap [trap update 2026-10-10] | imbalance as a knob and stalls near non-min-norm points (Furman 2026, Thm G.1; literature.md H4 and section 5, item 8); a conserved quantity decides which side of a unit can vanish (AGF, Kunin et al. 2025, Lemma B.1; T2 of `theory_trap.md` is that statement for the normal-visible blocks of a route end); with unbalanced init the leading-order flow can converge to a saddle, and balanced init excludes it (Jacot et al. 2021, App. C.1); dormant weights outside the stable set of a KKT point shrink (Kumar & Haupt) | the normal-visible blocks and the identity $C_u=Q_{\nu\nu}+\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2$; the visible degree and the rule $k\ge2j$; two dying ends for death ($k\le5$) and the pinned state; protection by spare directions on the end edge; an init-only per-run predictor of dead routes, validated on committed runs (not found in literature.md or the citation audit; `theory_trap.md` section 9) |
| R13: generic teachers give $k^*=2$ on MPS | - | not found |

## 10. Verification (round 2): issues and resolutions

The verifier (independent derivations and code; `saddles_verify2/verdict.md`) confirmed R2 as an
identity, R3 ($\lambda$ to 6 digits), R4 (aligned and random), R5, R6 for one residual, R10's counts
at $\theta_S$, R11, and the trap mechanism in trapped runs. Issues raised and what v2 does:

| # | issue (verifier) | verdict | resolution in v2 |
|---|---|---|---|
| I1 (major) | With $\ge3$ components the pairwise classification fails: residuals sharing a terminal core and site vector couple through one piggyback direction (star: $\lambda_{\min}=-0.3716$; gradient flow forgets $q$, 4/4 seeds) | agree | Piggyback-graph criterion $\rho(S)>s_q$ (4.1, proven), re-checked on 5 strict cases including a triangle, plus 3 controls (C11). R3, R6, R7, 4.3-4.5 and rule 5.1 restricted or rewritten; new 4.6 with one more super-critical seed and 3 sub-critical control seeds (C12). B4, E8 and N4 have $\rho(S)<s_q$ at their first saddle (disjoint differing sets, or $m\ge3$; checked: the old block at $\theta_{q_0}$ is PSD to $10^{-15}$ on mps4, binary4, ht4, mps8 and binary8, script `r6`) and one residual afterwards, so their predictions stand. |
| I2 (major) | "proven" overstates the saddle order; F2 false for disconnected $J$ | agree | R2 retagged: identity and upper bound proven, order $=k^*$ derived, lower bound only at order 3 (Prop 2 and 3.3). F2 corrected, and the verifier's MPS5 example reproduced exactly (C13). CLAUDE.md's Q3 status line should read "Steiner identity proven; saddle order $=k^*$ derived (upper bound proven, lower bound open beyond order 3)". The theorist does not edit CLAUDE.md; the lead should. (v3: superseded; the lower bound is now proven under $\lambda_{\max}(\mathcal S)<s_q$, see R2, section 11 and `theory_saddle_order.md` section 7, item 9.) |
| I3 (major) | Overlap with Kumar & Haupt 2025c, Rawal & DeWeese 2026, Furman 2026 D.8-D.9 and ZhangSL2026 not acknowledged | agree | Section 9 added; Prop 1, 3.2, 3.3 and 3.4 cite them; novelty narrowed to the Steiner identification, the coupling, the $k\ge3$ head start, the piggyback graph and the trap mechanism. |
| I4 | Trap needs all route ends unfavourable; 2/3 with another basis; $\sigma\to0$ untested; not specific to the fan-in mean | agree | Joint condition stated and tested on 6 more seeds, 3-8 (C14; section 8 table): dead in every run with all ends unfavourable, pinned otherwise. Prediction phrased as a fraction; $\sigma\to0$ and $k$-dependence tagged conjecture; "any $Q_e\ne0$" stated. |
| I5 (minor) | A second negative eigenvalue at $k=4$ | agree | Diagnosed (C15): the route contributes one; visible new-index pairs add $O(\sigma^2)$ terms. The count is a route counter for $k\le3$ only (6.2, prediction P6). |
| I6 (minor) | $\gamma$ needs a window | agree | $\gamma(a_r)=2-2/k-a_r/(1-a_r)$ and window $a_r\in[10^{-4},10^{-2}]$ (3.3, P4). |
| I7 (minor) | Wording; inverted exponent; $t_1$ factor empirical; "tightened" | agree | (i) "since initialization; for $k\ge3$ mostly during the first plateau"; (ii) $(A/s_r)^{(k-2)/k}$; (iii) P3 tags the $t_1$ factor as empirical and quotes the verifier's $\sigma$-independent values; (iv) 4.2 reworded. |

Partial rebuttals (both minor):
- On I1, "rule 5.1-1 refuted": only the first-escape endpoint is wrong. Rule 5.1-3 (next saddle =
  best fit in the new rank profile) correctly gives the $L=2.0$ plateau of the star runs: the
  $(1,1,2)$ optimum is "$r$, $r'$ learned", since $\max_{g_0,g_1}\lVert\langle W^*,g_0g_1\,\cdot\rangle\rVert^2=5.12>4$.
- On I4(d), agreed that only exactly balanced init removes the trap. But the fan-in *mean*
  imbalance matters quantitatively: it raises $P(\text{both HT leaf ends unfavourable})$ from
  $\approx0.25$ (balanced in expectation) to $\approx0.5$ (heuristic, section 8). [trap update 2026-10-10:
  this heuristic is withdrawn, see section 8 "How often".]

## 11. Open issues

- **Saddle order (closed for one learned component in v3; `theory_saddle_order.md`).** Order $\ge k^*$
  is proven under $\lambda_{\max}(\mathcal S)<s_q$, the level version gives $\ge\min(k^*,2\ell)$, and the
  order is exactly $2\ell^*$ for pattern teachers with a spare bond index when the condition fails.
  Still open: the borderline $\lambda_{\max}=s_q$ for $k^*\ge4$; the upper bound without the index
  condition, with signed weights, or for general (non-pattern) residuals (three random cases agree
  numerically); **several learned components** ($W_S$ not rank one across the core partition: the
  sectors must be built from the learned column/row spaces, and the factorization behind the locality
  of cumulants fails), which covers every saddle after the second event of the three- and
  four-component teachers; the dynamics after a cluster escape (forgetting of $q$, length of the next
  plateau) and the exponent about 4 seen in half of the control seeds of the pair-star check.
- **Fixed-rank block with several learned components.** The strictness criterion (old block negative
  or $k^*=2$) is proven, but $\rho(S)$ is explicit only for one learned component. With several learned
  components the analogue of $S$ (couplings through learned superpositions) is not worked out. The
  worked examples avoid the issue (one residual after the first saddle).
- **Star dynamics.** The depth of forgetting, the length of the "$r$, $r'$ learned" plateau and their
  $\sigma$-scaling are not predicted. More generally, the whole-sequence rule (5.1) is an algorithm, not
  a theorem. A numerical "saddle hopper" would make it operational: at a plateau state, compute the old
  block and $R_C$ for all connected $C$ with $\lvert C\rvert\le4$, follow the most unstable direction,
  integrate to the next plateau, repeat.
- **Prefactors for random init** are not predicted from $\sigma$ alone. They are set by the route
  amplitudes at the start of each plateau plus the accumulated head start (verifier: the $k=4$ random
  gap is 46x the aligned value at equal $\sigma$). The $\sigma$-independent shortening of $t_1$ in the
  aligned family follows from a 3-scalar ODE but was not solved in closed form.
- **Trap** [trap update 2026-10-10; the open items of `theory_trap.md` section 11]. The $\sigma\to0$ rules,
  other trees and many-ended routes are now covered there (sections 8 and 12 here). Open:
  - a $\sigma$-free reduced model of the second plateau, needed to predict the odeco (multi-stage) trap
    from the init;
  - $k=4$ routes with spare inner channels: vector route amplitudes; the current code's verdict there
    is basis-dependent;
  - the inner three-core clusters (death channel from $k=6$): derived to act, not in the per-run
    criterion;
  - the marginal $k=6$ case on HT trees;
  - finite $\sigma$: the $k\ge2j$ separation is weak at $\sigma\ge0.2$;
  - the pinned escape law $\propto\sigma^{-(2k-6)}$: a conjecture, testable only at $k=5$;
  - capped end bonds at $d\ge3$: Monte Carlo values only;
  - balanced init with internal route ends: very late escapes instead of deaths, not quantified.
- **Plateau Hessian for $k\ge4$.** At $k=4$ the visible new-index pairs add $O(\sigma^2)$ negative
  directions. For $k\ge5$ they may decay before the midpoint or may dominate $\lambda_{\min}$, which
  would change its $\sigma$-scaling. Untested.
- **Probability of the wrong order** for matrix-like pairs ($\propto\sigma^{n(s_r-s_q)/s_q}$) is a
  heuristic; the `order` agent's data should decide.
- **Anisotropic inputs** (H5): route degrees unchanged, couplings reweighted by the site eigenvalues
  on the route's patterns (not checked).
- **Re-runs after the RNG fix.** The v1 prefactor and order checks (C3b, C5-C10) used init seed =
  teacher seed 0 in some runs. Exponents were re-checked by the verifier; the rest should be re-run by
  the experiment agents with independent streams.

## 12. Predictions for experiments

Conventions: $d=2$; pattern teachers (`pattern_teacher`, independent teacher stream); fan-in init
unless stated; $\chi\ge$ the largest teacher rank needed (2 for two components, 3 for three). Event
time $t_r$ = first time $a_r\ge\tfrac12$, located by the integrator (coarse output grids give errors
of several time units on $k=2$ gaps). $\Delta t$ = gap between consecutive events; "per seed" = same
init direction at several $\sigma$. Status in brackets.

**Table A. Route degree $k$ (in brackets $m=\lvert N(D)\rvert$ where $\ne2$) for two-component teachers
$q=0^N$ ($s_q$), $r=1$ on $D$ ($s_r<s_q$)** [identity proven; v1 table re-derived by hand by the
verifier and recomputed with `Tree.steiner_nodes` in v2, script `r5`]:

| $D$ | binary4 ($K$=3) | mps4 ($K$=4) | ht4 ($K$=7) |
|---|---|---|---|
| {0,1} | 1 [m=1] | 2 | 3 |
| {1,2} | 3 | 2 | 5 |
| {0,2} | 3 | 3 | 5 |
| {0,3} | 3 | 4 | 5 |
| {0,1,2,3} | 3 | 4 [m=4] | 7 [m=4] |

| $D$ | binary8 ($K$=7) | mps8 ($K$=8) | ht8 ($K$=15) |
|---|---|---|---|
| {0,1} | 1 [m=1] | 2 | 3 |
| {1,2} | 3 | 2 | 5 |
| {3,4} | 5 | 2 | 7 |
| {0,4} | 5 | 5 | 7 |
| {2,5} | 5 | 4 | 7 |
| {1,6} | 5 | 6 | 7 |
| {0,7} | 5 | 8 | 7 |
| {0,1,2,3} | 3 | 4 [m=4] | 7 [m=4] |
| all 8 | 7 [m=4] | 8 [m=8] | 15 [m=8] |

Gap law: $k=1$: no plateau (separation ratio $<3$); $k=2$: $\Delta t_2=$ slope$\cdot\ln(1/\sigma)+$const;
$k\ge3$: $\Delta t_2\propto\sigma^{-(k-2)}$ per seed. First escape: $t_1\propto\sigma^{-(K-2)}$, so
large-$K$ trees (ht8, $K=15$) need $\sigma\gtrsim0.3$.

**P1. Exponents (H2)** [derived; checked $k\le4$, also post-fix]. For the teachers of Table A the
per-seed fitted exponent of $\Delta t_2$ against $\sigma$ lies within $\pm0.1$ of $-(k-2)$ over
$\sigma\in[0.0125,0.2]$ for $K\le5$ (measured $-0.97$ to $-1.02$ for $k=3$, $-1.99$ to $-2.00$ for
$k=4$). Exclude stalled runs (P12) and report sign-correction runs separately. Cleanest locality
contrasts at $N=8$: $D=\{3,4\}$ (log on mps8, $\sigma^{-3}$ on binary8, $\sigma^{-5}$ on ht8) and
$D=\{0,7\}$ ($\sigma^{-6}$, $\sigma^{-3}$, $\sigma^{-5}$).

**P2. $k=2$ slope (head start)** [derived; checked to $10^{-4}$ aligned, 1% random]. Slope of
$\Delta t_2$ against $\ln(1/\sigma)$ $=(1-s_r/s_q)/(s_rs_q^{(K-2)/K})$: 1.00 for weights $(1,0.5)$,
2.33 for $(1,0.3)$, 1.065 for $(1.2,0.5)$ on mps4. The no-head-start value $1/\lambda$ (2.0, 3.33,
1.83) would refute the head-start picture.

**P3. Aligned prefactors** [derived; checked]. $\Delta t_2=\sigma^{-(k-2)}(1-s_r/s_q)/((k-2)s_rs_q^{(K-k)/K})$
(ratio to prediction $\to1.0000$ as $\sigma\to0$; $<1\%$ off at $\sigma=2\cdot10^{-3}$). $t_1$ equals
$\sigma^{-(K-2)}/((K-2)s_q)$ times a $\sigma$-independent factor $<1$ when the second route shares
hanging cores with $q$ [empirical: 0.855, 0.913, 0.961 for $k=2,3,4$ on mps5 with $s_r/s_q=0.4$;
1 for $k=K$].

**P4. Growth exponent** [derived; checked]. On a plateau,
$\gamma=d\log\dot a_r/d\log a_r=2-2/k-a_r/(1-a_r)$. Fitted over $a_r\in[10^{-4},10^{-2}]$: 1, 1.33, 1.50,
1.60 for $k=2..5$ (measured 0.995-1.000, 1.334-1.344, 1.490-1.500 for $k=2,3,4$). Lower values at smaller
$a_r$ mean an unbalanced route, not a smaller $k$.

**P5. Random-init prefactors** [empirical]. Not predicted from $\sigma$ alone. The seed-to-seed
spread is broad for $k\ge4$ (factor $>20$; one $k=4$ gap was 46x the aligned value). The $k$-scalar
route ODE, started from route amplitudes measured on the plateau, predicted the remaining plateau to
1-3% (v1).

**P6. Hessian at plateau midpoints (H3)** [derived; checked]. Gauge-projected $\lambda_{\min}$:
$k=2$: $\to-s_rs_q^{(K-2)/K}$, independent of $\sigma$ (mps4 $(1,0.5)$: $-0.5$, measured $-0.498$).
$k=3,4$: $\propto\sigma^{k-2}$ (factor 2.0 and 4.0 per halving of $\sigma$), within a factor 1.2-1.4
($k=3$) or about 3.5 ($k=4$) of $-(k-1)s_rAb^{k-2}$ with $b=(s_ra_r/A)^{1/k}$. The number of negative
eigenvalues equals the number of active routes for $k\le3$. At $k=4$ expect a second negative
eigenvalue at 10-54% of $\lambda_{\min}$ with the same $\sigma^2$ scaling. For $k\ge5$ the scaling
of $\lambda_{\min}$ is untested. Old block: piggyback pair $a^{K-2}(s_q\mp s_r)$, GN modes
$a^{2(K-1)}$, radial mode $Ka^{2(K-1)}$ (mps4 $(1,0.5)$: $\{0.5,1,1,1.5,4\}$).

**P7. Saddle-type diagnostic** [checked]. The fixed rule "strict if $\lambda_{\min}<-10^{-3}\lambda_{\max}$"
calls $k=3$ plateaus strict for $\sigma\gtrsim0.01$-$0.04$ (seed-dependent). Classify instead by the
$\sigma$-scaling of $\lambda_{\min}$ ($\sigma^0$: strict; $\sigma^{k-2}$: higher order) or by $\gamma$ (P4).

**P8. Stars (three or more components)** [Hessian proven; dynamics empirical, mps4 only]. Teacher
$q=0000$ (2), $r=1010$, $r'=1001$ (1.6 each), $\rho(S)=2.263>s_q$. At "$q$ learned":
$\lambda_{\min}=-a^{K-2}(\rho(S)-s_q)$ $=-0.3716$ (mps4), $-0.3310$ (binary4), $-0.4311$ (ht4; binary8
analogue also $-0.4311$). Dynamics on mps4 ($\sigma=0.05$): no plateau at $L=2.56$. $a_q$ peaks at
$\le0.99$ and falls to 0.01-0.10 while $r$, $r'$ are learned. Then a plateau at $L=2.0$ ($a\approx(0,1,1)$)
of seed-dependent length (0-98), and $q$ is relearned. Control with $(1.2,1.2)$
($\rho=1.70<2$): plateau at $q$ ($L=1.44$, $a_q\ge0.998$), then $r$ ($k=3$), then $r'$. On binary4 the
star is a DLN whose top mode is $r+r'$ (2.263) and whose second is $q$ (2), so $r+r'$ and $q$ escape
almost together (aligned times in ratio 1.13). For the `order` agent: with $d=2$ and $\ge3$ components,
expect a non-monotone $a_q$ (rise, fall, rise) whenever $\rho(S_q)>s_q$.

**P9. Wrong order (H1b), two-component teachers** [derived; v1 checks, pre-fix RNG].
(a) Matrix-like pair ($m=2$, $k\ge3$; odeco on binary4, or $D$ = two non-adjacent single-site cores on
an MPS). If the smaller component $q$ is learned first, that plateau has
$\lambda_{\min}=-(s_r-s_q)s_q^{(K-2)/K}$ and lasts $\approx\ln(1/\varepsilon)/\lvert\lambda_{\min}\rvert$.
It is followed by a plateau at $L=\tfrac12s_q^2$ with $a_q\approx0$, and $q$ is relearned after
$\propto\sigma^{-(k-2)}$. (b) Tensor-like pair ($m\ge3$; odeco on mps4): no negative eigenvalue,
$a_q$ stays 1, and $L:\tfrac12s_r^2\to0$ in one step. (v3: only without a low split, 4.3. With one,
e.g. HT4 or binary8 with $D$ = all sites, the order is $2\ell^*=6<k=7$, and below the crossover of 4.4
the plateau scales as $\sigma^{-4}$ and $q$ is forgotten and relearned; derived, untested below the
crossover.) (c) $k=2$: $r$ is added directly. (d) Heuristic:
$P(\text{smaller first})\propto\sigma^{n(s_r-s_q)/s_q}$ for matrix-like pairs; for tensor-like pairs it
does not decay with $\sigma$.

**P10. Locality beats weight** [derived; v1 checks for orders and exponents]. Three-component
teachers, $\chi=3$, $\rho(S)<s_q$ at every saddle (no stars):

| teacher | tree | predicted order | $k$ per event | gap laws |
|---|---|---|---|---|
| B4: $0000$ (1), $q_A=0110$ (0.3), $q_B=1001$ (0.6) | mps4 | $q_0,q_A,q_B$ | 4, 2, 4 | $\sigma^{-2}$; log (slope 2.33, i.e. $+1.62$ per halving); $\sigma^{-2}$ ($q_B$ aligned at absolute time $\sigma^{-2}/1.2$) |
| B4 | binary4 | $q_0$, then $q_B,q_A$ (tie, weight favours $q_B$) | 3, 3, 3 | $\sigma^{-1}$ each (DLN on diag(1, 0.3, 0.6)) |
| B4 | ht4 | $q_0$, then $q_B,q_A$ (tie, seed-dependent) | 7, 5, 5 | $\sigma^{-5}$; $\sigma^{-3}$; $\sigma^{-3}$ (stalls: P12) |
| E8: $0^8$ (1), $q_A$ = 1 on {3,4} (0.4), $q_B$ = 1 on {1,6} (0.7) | mps8 | $q_0,q_A,q_B$ | 8, 2, 6 | $\sigma^{-6}$; log; $\sigma^{-4}$ ($A=1.088$) |
| E8 | binary8 | $q_0$, then $q_A,q_B$ (tie, seed-dependent) | 7, 5, 5 | $\sigma^{-5}$; $\sigma^{-3}$; $\sigma^{-3}$ ($A=0.837$ for $q_A$ after $q_B$) |
| E8 | ht8 | $q_0$, then $q_A,q_B$ (tie) | 15, 7, 7 | $\sigma^{-13}$; $\sigma^{-5}$; $\sigma^{-5}$ (expensive; stall-prone) |

v1 measurements: mps4 $q_A$ increments $\to1.62$ per halving; mps4 $q_B$ exponent $-2.03$; binary4
$-1.02$; ht4 $q_B$ $-2.99$. mps8 $q_B$ gap ratio 2.40-2.59 for $\sigma$ ratio 1.25 (predicted 2.44);
binary8 1.91-2.09 (predicted 1.95).

**P11. Partial learning (N4)** [derived; v1 check]. $q_0=0000$ (1), $q_1=1100$ (0.6), $q_2=1111$
(0.4), mps4, $\chi=3$. Plateau levels $0.76\to0.26\to0.08\to0.0569\to0$. The extra plateau at
$\sigma_2(M)^2/2=0.0569$ has $a_2\approx0.3$, and every gap after the first is $\propto\log(1/\sigma)$
(increments $\approx8$-9 per halving at $\sigma=0.1\to0.05$), although the exact rank rule alone would
give $\sigma^{-1}$ for $q_2$. On binary4: $0.76\to0.0569\to0$, with second gap $\propto\sigma^{-1}$.

**P12. Conservation-law trap** [trap update 2026-10-10: predictions from `theory_trap.md` T5, T6,
Prop. 2 and section 8, as corrected by its verification; tags per item]. For a route of $k$ cores on a
rank-one plateau, with end quantities $C_u$ (R12):
- $k\le3$: never trapped (derived; 0 stalls in 5,744 committed runs).
- $k=4$: a $\sigma$-free competition between end decay and route drive, settled by a reduced route ODE
  from the plateau-start values (derived for one new direction per route edge; at inner edges with
  $\chi\ge3$ the current code's verdict depends on an arbitrary basis choice).
- $k=5$, $\sigma\to0$: dead iff at least two ends have $C_u<0$; with one, that end is pinned at
  $\propto\sigma^{k-3}$ (derived) and the route escapes late (after $\propto\sigma^{-(2k-6)}$: conjecture,
  untested).
- $k\ge6$: in addition, three-core clusters at inner route cores (rate $\propto\sigma^4$) can kill pinned
  routes, and live routes too (marginal at $k=6$, dominant for $k\ge7$) (derived, not in the per-run
  criterion).
- Spare channels: an end with more new directions on its route edge than its partner has
  normal-visible rows ($c_e>n_w$) generically never dies (derived); capped end bonds are not protected.
- Fan-in init, saddle start, $c_e=1$: $P(C_u<0)=0.392$ at an MPS tree-leaf end and 0.5 at the MPS
  root end ($d=\chi=2$; proven for the Gaussian saddle-start model). At a capped end bond with $d\ge3$:
  about 0.023 ($d=3$) and $1.5\cdot10^{-4}$ ($d=4$) (Monte Carlo).
- Per run from the init (fan-in, $k\ge4$, locality): predicted dead 408/457 stall, predicted alive
  38/2,101 (empirical).
- Balanced init: tree-leaf ends cannot die at leading order (derived); in s2s-core 0/150 balanced odeco
  runs trap, 69/150 stall.
Withdrawn: the heuristic dead fraction $\approx0.5$ per two-ended HT route, and "MPS with $\chi=d$ stalls
less often". Original data (unchanged; HT4, B4, fan-in init, $\sigma=0.2$): some runs stall on a $k=5$
route at $L=\tfrac12s^2$ (0.045 for $q_A$, 0.18 for $q_B$). With teacher seed 3 and init seeds 0-8:
6 dead stalls, 2 pinned slow escapes (one escaped at $6\cdot10^5$), 1 run without a stall (section 8
table); every run with both route leaves unfavourable died. These traps act after two learned
components (section 8 note), not through the rank-one mechanism.

**P13. Generic (non-orthogonal) teachers** [derived; v1 check on one teacher]. Random CP or random TTN
teachers on an MPS: every plateau after the first is strict ($\lambda_{\min}$ independent of $\sigma$,
gaps $\propto\log(1/\sigma)$; mps4 example: $\lambda_{\min}=-0.129,-0.132,-0.136$ for
$\sigma=0.05,0.025,0.0125$). On binary4 the second plateau is non-strict ($\lambda_{\min}\propto\sigma$,
gap $\propto\sigma^{-1}$). On larger binary/HT trees expect $k^*\ge3$ only while ranks are one.

**P14. What would refute the picture.** (i) A $k\ge3$ plateau (Table A) whose $\lambda_{\min}$ stays
$\sigma$-independent as $\sigma\to0$ with no star ($\rho(S)\le s_q$): a missed $k=2$ route or a missed
fixed-rank instability. (ii) A per-seed exponent off by more than 0.2 from $-(k-2)$ at
$\sigma\le0.05$ in a run that neither stalled nor went through a sign correction. (iii) For $k=2$, a
slope $1/\lambda$ instead of $(1-s_r/s_q)/\lambda$. (iv) A star teacher with $\rho(S)>s_q$ that
plateaus at $q$ with $a_q\approx1$ for a time growing with $1/\sigma$. (v) [trap update 2026-10-10] A dead
stall at small $\sigma$ on a route with $k\le4$ that has at most one dying end ($C_u>0$ at every other end;
R12). It used to read "a dead stall with a live route end ($Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2>0$ at one
leaf)". Adding only "at small $\sigma$", as `theory_trap.md` v1 proposed, is not enough: the pilot has 20
predicted-pinned runs ($k=5$-7) labelled dead at every $\sigma$ of their grids (`theory_trap.md` section 7,
failure 5). Three-core clusters through inner route cores can kill a pinned route, marginally at $k=5$ and
dominantly from $k=6$; hence the restriction to $k\le4$.
