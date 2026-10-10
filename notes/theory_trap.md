# Theory: the conservation-law trap, predicted from the initial condition

Status: **v1, 2026-10-10** (theorist, branch `wip/trap-theory`). Extends `notes/theory_saddles.md` R12,
section 8, P12, C8, C14. Code and data: `experiments/theory_checks/trap/` (README there). Tags:
**proven** (complete argument), **derived** (stated assumptions, leading order in the small entries),
**empirical**, **conjecture**; **checked** marks a numerical confirmation.

Question: some runs never learn a component because the route that should carry it dies (its end
cores decay to roundoff). R12 explained this by the conserved gauge quantities $Q_e$ in a reduced
model tested on HT4 at $\sigma=0.2$. Since then the trap was measured at scale (s2s-core: 0 at $K=3$,
40% at $K=4$ fixed per init direction, 30-80% at $K=7,8$; locality: 398 of 5,682 runs stall, all with
$k\ge4$; pilot: 95 of 332 runs with $k\ge4$). Here: which quantity decides, evaluated on the init;
why the outcome depends on the init direction only; per-run predictions for the committed runs; the
probability law; failures.

## 0. Results at a glance

| # | result | status |
|---|---|---|
| T1 | On a plateau at a learned rank-one point the entries with all bond indices learned relax fast; the entries with ONE new bond index form two-core products whose Gauss-Newton (GN) cost is only their part **normal** to the learned manifold: for an end $u$ and its route neighbour (partner) $w$ on edge $e$, $\tfrac12A^2\lVert X_u^\perp X_w^{\perp\top}\rVert^2$, where $X^\perp$ = new-index block with learned indices on the other legs and physical index orthogonal to the learned pattern. | normal-part lemma proven; reduced flow derived |
| T2 | In this reduced flow $C_u=X_u^{\perp\top}X_u^\perp-X_w^{\perp\top}X_w^\perp$ is conserved and the flow ends at $X_u^{\perp\top}X_u^\perp=[C_u]_+$. The end's route entry survives iff $C_u$ has a positive eigenvalue (one new direction: iff $C_u>0$). | proven (within the reduced flow) |
| T3 | Identity: $C_u=\nu^\top Q^{(u)}_e\nu+\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+(y_\parallel^2-x_\parallel^2)$: the conserved bond Gram difference (end minus partner) plus the partner's non-visible new entries minus the end's; the last term is $O(\sigma^3)$ after a small-init escape. R12's $c=Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$ kept only the route part of $Y_{\rm nv}$. | exact identity; size of last term derived; checked (3-4 digits) |
| T4 | A partner without sites (HT, binary internal nodes) gives no second-order cost (the product is tangent). The end then decays through a three-core cluster (end, partner, sibling with sites) at rate $\propto\sigma^4$ and dies iff it is the lightest factor. In general an end with visible degree $j$ decays at rate $\propto\sigma^{2(j-1)}$. | derived |
| T5 | Decay time $\sigma^{-2(j-1)}$ against escape time $\sigma^{-(k-2)}$: an end can kill its route only if $k\ge2j$ (marginal at equality). $j=2$ (partner with sites): $k\ge4$; $j=3$ (HT/binary leaves at a rank-one saddle): $k\ge6$; $k\le3$: never. | derived; consistent with every locality case |
| T6 | A route dies iff at least two of its ends die ($k\ge5$: the dead state is linearly stable because the ends' mutual drive is $\propto\sigma^{k-2}\ll$ decay $\propto\sigma^2$). One dying end is pinned at $\propto\sigma^{k-3}$: slow escape after $\propto\sigma^{-(2k-6)}$. At $k=4$ a $\sigma$-free competition decides; a reduced route ODE from the plateau-start values settles it. | derived; checked |
| T7 | Exact: $\sigma_j(G_u)^2\ge\lambda_j(Q^{(u)}_e(0))$ for all $t$; at a dead end the partner's new-slice norm$^2$ is $-Q^{(u)}_{\nu\nu}(0)$ (explains C8). Spare channels: an end with more new directions on its route edge than its partner has normal-visible rows ($c_e>n_w$) generically never dies. | bound proven; protection derived |
| T8 | The plateau-start state is $\sigma\times$ a function of the init direction (exactly for a saddle start; via the $\sigma$-free correlation flow for a small init), so the outcome is a property of the init direction. | derived; checked ($\sigma=0.02$ vs $0.005$: 3 digits) |
| T9 | Balanced init ($Q_e=0$): $C_u=\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+O(\sigma^3)\ge0$ at ends that are tree leaves, so odeco routes cannot die; ends with extra legs (pattern teachers) still can. | derived |
| T10 | Per-run predictions (init + $\sigma$-free correlation flow, no gradient flow) for the committed runs: see section 7. | empirical |
| T11 | Probability law under fan-in init: section 8. | derived (formulas), empirical (Monte Carlo) |

## 1. Setting and notation

Model and notation as in `theory_saddles.md` section 1: tree with cores $v$ ($K$ of them), edges named
by their child, $Q_e=G_c^\top G_c-G_{p,(c)}G_{p,(c)}^\top$ conserved by gradient flow. Learned
rank-one point $\theta_S$ of component $q$: every core is $a\,p_v\otimes\bigotimes_{f\ni v}w_f$ with
$a=s_q^{1/K}$, $p_v=\bigotimes_{i\in v}u_q^{(i)}$ the learned physical pattern and $w_f$ the learned
direction of bond $f$; $N_f$ = orthonormal basis of its complement (the **new** space, dimension
$c_f=\chi_f-1$). The dormant component $r$ needs the route $C=\mathrm{Steiner}(N(D))$, $k=\lvert C\rvert$.

- **End** $u$: a core of $C$ with one neighbour in $C$, its **partner** $w$, joined by the route edge $e$.
  For an end, $Q^{(u)}_e=\pm Q_e$ is oriented end minus partner ($+$ if $u$ is the child of $e$).
- **Visible block** of a core $v$ on edge $e$: its entries with a new index on $e$ and learned
  indices on all its other bond legs; $X^\perp_v$ = that block with the physical index projected
  orthogonally to $p_v$ (empty if $v$ has no sites); $x_\parallel$ = the part along $p_v$.
  $n_v=d^{s_v}-1$ = number of normal physical rows ($s_v$ = sites of $v$).
- **Non-visible block** $X_{\rm nv}$ (end) and $Y_{\rm nv}$ (partner): entries with a new index on $e$
  and a new index on at least one other bond leg. They pair with nothing at second order.
- **Route entry** $b_v$: core $v\in C$ contracted with $u_r$ on its sites, new spaces on its route edges,
  learned directions elsewhere. At an end carrying a differing site, $b_u$ is a row of $X_u^\perp$.
- **Visible degree** $j_u$: number of cores in the smallest connected cluster through $u$'s new index
  on $e$ whose product changes $W$ in a direction normal to the learned manifold. $j_u=2$ if the
  partner carries sites; $j_u=3$ for a leaf whose partner is siteless and has another child with sites.
- Units: the small entries are $O(\sigma)$; everything below is quoted in units of $\sigma$ (saddle
  start) or of the init's own scale (correlation-flow state), so signs and ratios are $\sigma$-free.

## 2. Mechanism (derived)

### 2.1 Which products cost anything: the normal part (lemma proven)
Expand $W(\theta_S+\delta)=W_S+\sum_vW_v[\delta_v]+\sum_{v<v'}W_{vv'}[\delta_v,\delta_{v'}]+\dots$. First-order
changes span the tangent space $T=\mathrm{span}\{T_v\otimes p_{\rm rest}\}$ (anything on the sites of one
core, the learned pattern elsewhere). New-index entries are invisible at first order (`theory_saddles`
F1). At second order, an entry of $v$ with new index $\nu$ on $e$ pairs only with an entry of its
neighbour $w$ with the same index on $e$ and learned indices on all other legs:
$W^{(2)}_e=A\sum_\nu x_\nu\otimes y_\nu\otimes p_{\rm rest}$, $A=a^{K-2}$.

**Lemma 1 (proven).** The component of $W^{(2)}_e$ normal to $T$ is $A\sum_\nu x^\perp_\nu\otimes y^\perp_\nu\otimes
p_{\rm rest}$; it vanishes if either core has no sites; normal parts of different edges are orthogonal.
*Proof.* $x_\parallel\otimes y$ is a single-core change of $w$ and $x\otimes y_\parallel$ one of $v$, so both lie in
$T$. $x^\perp\otimes y^\perp$ deviates from the pattern on the sites of two cores, so it is orthogonal
to every single-core change. For a siteless $w$, $y_\nu$ is a scalar and $W^{(2)}_e=A(\sum_\nu y_\nu x_\nu)
\otimes p_{\rm rest}\in T$. Two edges' normal parts deviate from the pattern on different core pairs. $\square$

**Adiabatic elimination (derived).** Entries with all bond indices learned have $O(1)$ Hessian on
$T$ (GN, rate $a^{2(K-1)}$) and follow the slow entries, cancelling the tangent part of their
$W$-change. The slow entries then follow gradient flow of
$$L_{\rm eff}=L_S-\langle R,W_{\rm route}\rangle+\tfrac12\sum_eA^2\Big\lVert\sum_\nu x^\perp_\nu\otimes y^\perp_\nu\Big\rVert^2+(\text{clusters of }\ge3\text{ cores}),$$
where only the route term (order $k$) overlaps the residual $R=s_rU_r$. The pattern parts $x_\parallel$,
$y_\parallel$ are flat at this order: they do not decay. R12's reduced model let them decay.

### 2.2 The conserved quantity of an end (T2, proven within the reduced flow)
For an end $u$ with $j_u=2$ the GN term is a two-layer zero-target problem,
$\dot X=-A^2XY^\top Y$, $\dot Y=-A^2YX^\top X$ ($X=X^\perp_u$, $Y=X^\perp_w$). Hence
$C_u=X^\top X-Y^\top Y$ is conserved, $\lVert X\rVert$ decreases, every critical point has $XY^\top=0$, so
$X^\top X\to[C_u]_+$, $Y^\top Y\to[C_u]_-$ (PSD with zero product, difference $C_u$). All rows of $X$
evolve as $X(t)=X(0)M(t)$, so the route row survives iff $M(\infty)\ne0$ on it: for $c_e=1$ iff $C_u>0$,
for $c_e>1$ generically iff $C_u$ has a positive eigenvalue. A **dying** end decays at
$\gamma_u=A^2\lvert C_u\rvert$ once the transient is over.

### 2.3 $C_u$ in terms of the conserved $Q_e$ (T3)
Split the new slices along $\nu$: end = $x_\parallel+X^\perp+X_{\rm nv}$, partner = $y_\parallel+X^\perp_w+Y_{\rm nv}$.
Then exactly
$$C_u=Q^{(u)}_{\nu\nu}+\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+(y_\parallel^2-x_\parallel^2).$$
The learned-new element $Q_e[w,\nu]=a(x_\parallel-y_\parallel)+O(\sigma^2)$ is conserved. From a small init it is
$O(\sigma^2)$, so after the escape $x_\parallel-y_\parallel=O(\sigma^2/a)$ and the last term is $O(\sigma^3)$ (derived;
checked at the correlation-flow state: the two sides agree to 3-4 digits on MPS4/6/8 and midroot6
chain ends). From a saddle start it is $O(\sigma)$ and the last term stays $O(\sigma^2)$, so there $C_u$
must be computed from the blocks directly (as the code does). R12's criterion
$Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$ is this identity with only the route entry of $Y_{\rm nv}$ kept; it
under-counts the partner's non-visible entries, which is why it over-predicts death (section 7).

### 2.4 Partners without sites; visible degree (T4)
If $w$ has no sites (HT and binary internal nodes) Lemma 1 gives no second-order cost: the end's
new entries pair with $w$'s visible entry into a tangent change. The lowest normal product through
$u$'s new index is the three-core cluster $(u,w,u')$, $u'$ another child of $w$ with sites:
$x_u\otimes z\otimes x_{u'}$ with $z$ = $w$'s entry new on both legs and learned on its parent leg.
GN cost $\propto\sigma^6$, decay rate $\propto\sigma^4$. For scalars ($c=1$) the three squares decrease at the
same rate, so the smallest goes to zero: with $c_1=\lvert X_u^\perp\rvert^2-z^2$, $c_2=\lvert X_{u'}^\perp\rvert^2-z^2$
(conserved in the cluster), the end dies iff $c_1<\min(0,c_2)$. Usually $z$ (fan-in $\chi^2$) is the
lightest factor and dies; the leaves then lose $z^2$ of their Gram but survive.
**With two or more learned components** a siteless partner's visible entries form a matrix over the
learned indices of its other legs, and only its learned (diagonal) combinations are tangent; the
off-diagonal ones give a second-order normal product again ($j=2$). This is why the HT4 leaves of
`theory_saddles` section 8 (B4 after $q_0,q_B$) and of the odeco runs (last component) do die.

### 2.5 Time scales, death and pinning (T5, T6)
Route entries are $O(\sigma)$; the route drive on one of them is $c\prod_{v\ne u}b_v\propto\sigma^{k-1}$
($c=s_ra^{K-k}$), its relative rate $\propto\sigma^{k-2}$. The GN decay of an end has rate $\propto\sigma^{2(j-1)}$.
- $k<2j$: the route escapes before the end has moved. With $j\ge2$ this covers every $k\le3$: no trap.
- $k>2j$: the GN subsystems finish first. A route with two dying ends $u,u'$: the linearized ends obey
  $\dot x_u=-\gamma_ux_u+\kappa x_{u'}$ (and $u\leftrightarrow u'$), $\kappa=c\prod_{v\ne u,u'}b_v\propto\sigma^{k-2}$, so the
  dead state is stable iff $\gamma_u\gamma_{u'}>\kappa^2$, i.e. always as $\sigma\to0$ ($\sigma^4\gg\sigma^{2k-4}$ for $k\ge5$).
  Three or more dying ends have no linear mutual drive. **Dead iff at least two ends die.**
- Exactly one dying end: its drive from the live route is $\propto\sigma^{k-1}$, so it is **pinned** at
  $x^*=\kappa_u/\gamma_u\propto\sigma^{k-3}$, not dead. The rest of the route then grows at
  $\dot b\approx(c^2/\gamma_u)b^{2k-3}$, so the route escapes after $T_{\rm pin}\approx\gamma_u b^{-(2k-4)}/((2k-4)c^2)\propto
  \sigma^{-(2k-6)}$ instead of $\sigma^{-(k-2)}$ (exponent 4 instead of 3 at $k=5$).
  With a fixed budget such runs are counted as stalls; their end amplitudes scale as $\sigma^{k-3}$ (relative
  $\sigma^{k-4}$), which is the "pinned" class of R12 and plausibly the steep MPS rows of the report
  (section 9).
- $k=2j$ (e.g. $k=4$, $j=2$): decay and drive are of the same order. In rescaled time $\sigma^2t$ the reduced
  model below is $\sigma$-free, so the outcome is still a property of the init direction, but not a pure
  sign test. Also the sign of the route product matters: if $\prod_vb_v<0$ every $b_v^2$ decreases at the same
  rate until the smallest crosses zero (sign correction), which shrinks $\kappa$ while the ends decay.

**Reduced route model** (derived; used for $k=4$ and at finite $\sigma$): for a route whose ends all have
$j=2$ and one new direction on each route edge,
$$\dot b_v=c\prod_{w\ne v}b_w-[v\text{ end}]\,A^2\lvert Y_v\rvert^2b_v,\qquad \frac{d}{dt}\lvert Y_v\rvert^2=-2A^2\lVert X_v^\perp\rVert^2\lvert Y_v\rvert^2,$$
with the end's other normal rows decaying like its route row. Integrated from the plateau-start values
it returns escape / dead / pinned.

## 3. Exact statements (T7)

**Proposition 1 (Gram bound; proven).** For any core $u$ on edge $e$ and all $t$,
$G_u^{(e)}(t)^\top G_u^{(e)}(t)=Q^{(u)}_e(0)+G_w^{(e)}(t)G_w^{(e)}(t)^\top\succeq Q^{(u)}_e(0)$, hence
$\sigma_j(G_u^{(e)}(t))^2\ge\lambda_j(Q^{(u)}_e(0))$ for every $j$ (Weyl). If $\lambda_{r_e+1}(Q^{(u)}_e(0))>0$ the end
keeps $r_e+1$ directions of size $\ge\lambda_{r_e+1}^{1/2}$ forever: it cannot die.
**Corollary (proven).** If an end core is rank-deficient on $e$ at some time, $G_u\nu=0$, then
$\nu^\top Q^{(u)}_e(0)\nu=-\lVert G_w^\top\nu\rVert^2\le0$: the partner's slice along the dead direction has
norm$^2$ exactly $-Q^{(u)}_{\nu\nu}(0)$. This is the C8 observation ("parent new-slice norm$^2=-Q_{\nu\nu}(0)$
to 4 digits"). The bound is rarely active under fan-in init (both sides have equal mean Gram), so
it protects few runs; it is the only fully rigorous init-only statement here.

**Proposition 2 (protection by spare directions; derived).** An end dies only if $C_u\preceq0$ on its
$c_e$ new directions. $\mathrm{rank}(X_w^{\perp\top}X_w^\perp)\le n_w$ (normal-visible rows of the partner),
so if $c_e>n_w$ there is a new direction with $X_w^\perp\nu=0$, and generically $X_u^\perp\nu\ne0$, giving
$\nu^\top C_u\nu>0$: the end survives. At $d=2$ with one-site partners ($n_w=1$) every route edge with
$\chi_e\ge3$ protects its end; edges capped at $\chi_e=d=2$ ($c_e=1$) do not. With two learned components
the count is over the learned-index combinations that are not tangent (section 2.4).

## 4. Operational criterion from the initial condition

Inputs: the init cores (library stream `TTNSpec.init(sigma, seed)`, optionally gauge-balanced), the
teacher, the tree and the differing sites $D$.
1. **Plateau-start state** (T8). Saddle start: the init itself (learned bond index 0, pattern $u_q$).
   Small init: integrate the rescaled correlation flow $\dot\phi=\nabla_\phi\langle W^*,W(\phi)\rangle$ from the
   init direction until one component's overlap reaches $10^{10}$ (`theory_escape`: the $\sigma\to0$ limit
   of the first plateau). It gives the escaping component $q$, the learned directions $w_f$ (top right
   singular vector of each child core) and the small entries, with head start. No gradient flow of the
   loss is integrated. The non-pattern entries are frozen in this flow, so $X^\perp_w$ at an end whose
   partner site is not in $D$ equals its init value (checked: identical at $t=0$ and $t_1$).
2. For each route end: $j_u$, $C_u$ (two-core) or the cluster quantities (three-core), and the
   protection count $c_e$ vs $n_w$.
3. Verdict: an end is **dying** if $C_u\preceq0$ and $k\ge2j_u$. Route **dead** if $\ge2$ ends are dying,
   **pinned** if one is, **alive** otherwise. At $k=4$ with two dying ends the reduced route model decides.
   At finite $\sigma$ the reduced model with the actual $\sigma$ can be integrated instead
   (`reduced_route(..., unit)`).

Three nested levels are compared in section 7: L0 = sign of $Q^{(u)}_{\nu\nu}(0)$ alone; R12 =
$Q^{(u)}_{\nu\nu}+\lvert y_{\rm route}\rvert^2$ at the plateau start; L2 = the full criterion above.

## 5. Why trapping is a property of the init direction (T8, derived)

(i) The first plateau is the correlation flow, homogeneous of degree $K-1$: with $\theta=\sigma\phi$ and
$\tau=\sigma^{K-2}t$ it is $\sigma$-free, so $q$, the $w_f$ and the small entries at the escape are
$\sigma\times F(Z)$ + $O(\sigma^2)$, $Z$ the init direction (the library draws $Z$ from the seed and
multiplies by $\sigma$). (ii) The escape itself takes $O(\log)$ time, during which the GN subsystems
(rates $\propto\sigma^2$) do not move. (iii) On the plateau, with $b=\sigma\hat b$ and $\tau=\sigma^2t$, the GN
subsystems are $\sigma$-free and the route drive carries the factor $\sigma^{k-4}$. So for $k\ge5$ the outcome
tends to the sign test on $F(Z)$, for $k=4$ to a $\sigma$-free ODE on $F(Z)$; corrections are $O(\sigma)$
(the entries) and $O(\sigma^{k-4})$ (the drive). Checked on MPS4 $\{0,3\}$: the plateau-start values in
units of $\sigma$ agree to 3 digits at $\sigma=0.02$ and $0.005$ for all 8 seeds, and the correlation-flow state
reproduces them to 1-5%. Consequences: per seed the class is constant across small $\sigma$; at
moderate $\sigma$ (0.1-0.3, used for $k\ge5$ in the experiments) seeds near the boundary
($C_u\approx0$, or $\gamma\gamma\approx\kappa^2$) flip, which is the "$\sigma$-dependent dropout" of the locality
runs and the close-teacher exceptions of s2s-core (section 7).

## 6. Balanced initialization and spare channels

**Balanced init (T9, derived).** $Q_e(0)=0$ is conserved, so by T3, after a small-init escape,
$C_u=\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+O(\sigma^3)$. An end that is a tree leaf has no other legs,
$X_{\rm nv}=0$, so $C_u\ge0$: it cannot die at leading order. Odeco routes span the tree and end at
tree leaves (also for the later components, where T3 holds with the learned-index combinations), so
balanced init removes the odeco trap (0 of 120 trapped, s2s-core). Pattern-teacher routes that end at
a core with extra legs (MPS $\{1,5\}$, $\{1,6\}$, $\{2,5\}$, $\{1,3,5\}$) can still have $\lVert X_{\rm nv}\rVert>\lVert Y_{\rm nv}\rVert$; all
16 balanced stalls of the locality runs are on such routes or on MPS8 $\{0,7\}$ (section 7). Balanced
init does not make $C_u$ large, only non-negative: ends with small $C_u$ keep small route entries,
so the escapes can be very late (the $10^5\times$ delays of the locality balanced runs).
**Spare channels** (Proposition 2). A route edge with more new directions than its partner's
normal-visible rows protects its end; capped end bonds ($\chi_e=d$ at an MPS chain end or an HT leaf)
never do. So spare bond dimension helps only where it reaches the route's end edges.

## 7. Per-run test on the committed runs (T10, empirical)

Every committed two-component run of the locality sweeps (twocomp, twocomp_balanced, twocomp_chi4,
robust, k5 and khigh extensions, stall diagnosis; 10,107 runs) and of the pilot (1,093 runs) was
rebuilt from its config and seed with the library streams (`TTNSpec.init(sigma, seed)`,
`pattern_teacher(..., seed=1000)`, `s2s.gauge_balance` for balanced runs), and predicted **without
gradient flow** (`predict_locality.py`; 1,532 distinct (case, seed) predictions, about 6 min on 6
processes): plateau-start state from the init (saddle start) or the $\sigma$-free correlation flow (small
init), then section 4. Observed labels: locality "stalled" (gradient below $10^{-12}$ for 5,000
evaluations; includes slow escapes) vs escaped; stall-diagnosis runs dead / pinned / escaped
(`experiments/locality/results/r12.csv`, 60x patience); pilot "collapsed" or unresolved with the second
component at roundoff ($\lvert a_2\rvert<10^{-20}$) = dead, converged = escaped. Runs with $k\le3$ are all
predicted alive and none stalls (5,112 locality runs); the tables use $k\ge4$.

**Pooled, fan-in init** (stalled/runs per predicted class; L2 = this note, R12 = $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$
at the plateau start, L0 = $Q_{\nu\nu}$ alone, both with "dead iff all ends negative"):

| runs | criterion | pred. dead | pred. pinned | pred. alive |
|---|---|---|---|---|
| locality, saddle + small (3,512; 585 stalled) | L2 | **418/469 (89%)** | 141/942 (15%) | **26/2,101 (1.2%)** |
| | R12 | 430/1,169 (37%) | 108/1,092 | 47/1,251 (3.8%) |
| | L0 | 474/1,522 (31%) | 79/1,149 | 32/841 (3.8%) |
| locality, balanced (1,316; 41 stalled, none dead) | L2 | 33/83 | 8/235 | 0/998 |
| pilot, saddle + small (418 resolved; 124 dead) | L2 | **102/109 (94%)** | 20/169 | 2/140 |
| | R12 | 84/161 | 30/168 | 10/89 |

So the criterion separates the classes far better than R12 (the gain comes from the partner's
non-visible entries, T3, and the visible degree, T4), and "alive" is a reliable prediction: 26 stalls
in 2,101 runs, mostly slow escapes at $\sigma\ge0.1$ (below). **Stall diagnosis** (dead / escaped / pinned after
60x patience; R12 had 30 of 55 vs 7 of 67): predicted dead 30 / 8 / 4, pinned 5 / 65 / 0, alive 2 / 56 / 0.

**By $k$** (fan-in, saddle + small; `results/table_by_k.md`): $k=4$: dead 203/232, pinned 2/458, alive
5/814 (at $k=4$ a pinned end is $O(\sigma)$, so "pinned" behaves as alive, as predicted); $k=5$: 73/79, 10/122,
9/1,095; $k=6$: 92/104, 83/248, 8/138; $k=7$: 29/33, 12/72, 0/42; $k=8$: 21/21, 34/42, 4/12. The stall rate of
pinned runs grows with $k$ (the pinned escape time $\propto\sigma^{-(2k-6)}$ exceeds the patience).

**Per tree, bond dimension and protocol** (stalled/runs; pattern teachers with $s_2/s_1=0.5$ except the
robust rows; per (tree, $D$) in `results/table_by_case.md`, per $k$ in `results/table_by_k.md`):

| source | tree | chi | start | pred. dead | pred. pinned | pred. alive |
|---|---|---|---|---|---|---|
| locality | binary8 | 2 | saddle | - | 5/18 | 0/156 |
| locality | binary8 | 2 | small | - | 0/6 | 0/168 |
| locality | binary8 | 2 | small_balanced | - | - | 0/238 |
| locality | binary8 | 4 | small | - | - | 0/174 |
| locality | ht4 | 2 | saddle | 18/24 | 0/24 | 8/172 |
| locality | ht4 | 2 | small | 4/6 | 4/42 | 0/156 |
| locality | ht4 | 2 | small_balanced | - | - | 0/332 |
| locality | ht4 | 4 | small | 7/12 | 6/42 | 0/150 |
| locality | ht8 | 2 | saddle | 7/9 | 2/18 | 1/78 |
| locality | midroot6 | 2 | saddle | 18/18 | 5/24 | 6/30 |
| locality | midroot6 | 2 | small | 12/12 | 12/42 | 0/12 |
| locality | midroot6 | 2 | small_balanced | - | 0/18 | 0/54 |
| locality | midroot6 | 4 | small | 12/12 | 18/42 | 0/24 |
| locality | mps4 | 2 | saddle | 24/24 | 0/36 | 0/36 |
| locality | mps4 | 2 | small | 76/76 | 0/118 | 0/62 |
| locality | mps4 | 2 | small_balanced | - | - | 0/96 |
| locality | mps4 | 4 | small | 26/46 | 0/68 | 0/94 |
| locality | mps4 | 8 | small | - | - | 0/48 |
| locality | mps6 | 2 | saddle | 50/58 | 21/100 | 6/80 |
| locality | mps6 | 2 | small | 47/48 | 0/90 | 1/58 |
| locality | mps6 | 2 | small_balanced | - | 6/92 | 0/140 |
| locality | mps6 | 4 | small | 14/14 | 20/98 | 0/102 |
| locality | mps8 | 2 | saddle | 45/45 | 35/102 | 0/27 |
| locality | mps8 | 2 | small | 43/50 | 13/66 | 2/54 |
| locality | mps8 | 2 | small_balanced | 33/83 | 2/125 | 0/42 |
| locality | mps8 | 4 | small | 15/15 | 0/6 | 2/132 |
| locality | star3 | 2 | saddle | - | - | 0/96 |
| locality | star3 | 2 | small | - | - | 0/96 |
| locality | star3 | 2 | small_balanced | - | - | 0/96 |
| locality | star3 | 4 | small | - | - | 0/96 |
| pilot | binary8 | 2 | saddle | - | 3/6 | 0/22 |
| pilot | binary8 | 2 | small | - | - | 0/55 |
| pilot | mps4 | 2 | saddle | 10/10 | 0/30 | - |
| pilot | mps4 | 2 | small | 40/40 | 0/48 | 0/40 |
| pilot | mps6 | 2 | saddle | 20/25 | 11/38 | 2/5 |
| pilot | mps6 | 2 | small | 32/34 | 6/47 | 0/18 |

Readings: binary8 and star3 (all ends have siteless partners; $j=3$, active only for $k\ge6$) never
stall where predicted alive; MPS4 at $\chi=2$ is predicted exactly (locality and pilot, fan-in: all
150 predicted-dead runs stalled or died, none of the 370 others); HT4 with $k=5$ ($D=\{1,2\}$) is predicted
alive and its 8 saddle-start stalls are slow escapes except two (below); the $\chi=4$ rows reproduce
the measured effect of spare channels: fewer deaths where a route end gains spare directions
(Proposition 2), none fewer where both ends are capped (midroot6 $\{0,5\}$: 30/30 predicted dead or
pinned, 30/30 stalled; MPS6 $\{0,5\}$: 40/40 predicted dead or pinned, 24/40 stalled).

**Per seed** (cases with $\ge3$ values of $\sigma$): locality, predicted alive: 453 of 463 seeds never stall,
1 stalls at every $\sigma$, 9 at some; predicted dead: 46 of 70 stall at every $\sigma$, 4 never, 20 at some
(in 9 of these the escapes are exactly the largest $\sigma$ values, as the $O(\sigma)$ corrections of section 5
predict; 11 interleave); pinned: 126 of 159 never, 13 always. Pilot: alive 23/24 never; dead 15/17
always; pinned 26/31 never. So the class is a property of the init direction wherever the prediction
is alive or dead; the mixed seeds are mostly near the boundary or at $\sigma\ge0.1$.

**Where the criterion fails** (fan-in, 58 false alarms and 28 misses among 3,930 runs incl. pilot):
1. $k=4$ with spare inner channels (MPS4 at $\chi=4$: 20 of 46 predicted dead escape). The reduced route
   model was applied only with one new direction per route edge; with $c=3$ inner directions the
   route's drive is larger than the scalar model assumes. Fix: vector route amplitudes (not done).
2. The marginal $j=3$ case $k=6$ on HT4 ($D=\{0,1,2\}$: 11 of 18 predicted dead escape). At $k=2j$ the
   three-core decay and the escape are of the same order; only the sign test was used.
3. Large $\sigma$ ($\ge0.2$): the time-scale separation behind the $k\ge2j$ rule is weak. HT4 $\{1,2\}$
   ($k=5$, $j=3$): two runs at $\sigma=0.19,0.30$ (seed 1, both cluster signs negative) are dead although
   predicted alive; the long MPS routes ($k=6$-$8$) lose some predicted-alive runs at $\sigma\ge0.08$ (12 runs),
   plausibly through the third-order clusters at the second route core (end, first inner core,
   next core), which act at rate $\propto\sigma^4$ and outrun $k\ge7$ escapes (derived, not implemented).
4. Seeds near the boundary at $k=4$-5 (MPS6 $\{1,4\}$ saddle one seed at all six $\sigma$; MPS8 $\{1,5\}$ small
   one seed; MPS6 $\{0,3\}$, $\{0,4\}$ a few runs): $C_u$ or the reduced competition close to zero.
5. Dead runs predicted pinned (stall diagnosis: 5 of 70): MPS6 $\{0,5\}$ at $\chi=4$, $\sigma=0.35$-$0.5$ (4 runs)
   and MPS6 $\{1,3,5\}$ saddle, $\sigma=0.19$ (1 run, the "both ends live" case of locality 7.5). Large $\sigma$
   again; the second end must have died through a channel not in the model.
Balanced init: 33 of the 83 predicted-dead runs stall (all with internal ends on MPS8, T9), none of
them dead in the diagnosis: with $Q=0$ the predicted-dead ends have small $\lvert C_u\rvert$, and at finite $\sigma$
they escape very late instead (the $10^5\times$ delays). This is a quantitative failure of the dead /
pinned split for balanced runs, not of the sign.

### 7.6 Mechanism check on HT4 and binary8 (odeco, $\chi=3$, $\sigma=0.2$, seeds 0-5)

Gradient flow from the committed init up to the second escape, then the phase-2 sign test at the four
leaf ends (`gf_check_s2s.py tree ...`; $k=7$, so the $\sigma\to0$ rule "dead iff two ends die" is used):
- **HT4: 6/6.** Predicted dead for seeds 0, 1, 2, 3, 5 (two leaves with $C'<0$ each), alive for seed 4;
  observed (k78 and main): trapped, trapped, trapped, trapped, complete, trapped. With two learned
  components the siteless parents' off-diagonal entries are visible ($j=2$, section 2.4), which is what
  kills HT leaves here.
- **binary8: consistent, not decisive.** Seed 5 dead (trapped); seeds 3, 4 pinned (trapped); seed 1 pinned
  and seeds 0, 2 alive (stalled, i.e. not at roundoff). "Trapped" at $K=7$ (a slice below $10^{-8}\sigma$) also
  catches deeply pinned ends, whose value $\kappa_u/\gamma_u$ contains six other route amplitudes; the
  two-leaf-end dead/pinned split is therefore not testable with this label.

## 8. Probability law under fan-in init (T11)

**Saddle start, one new direction, partner with sites (derived, closed form).** With learned
directions fixed by the saddle, $X_u^\perp$ has $n_u=d^{s_u}-1$ i.i.d. $N(0,\sigma^2/F_u)$ entries and $X_w^\perp$ has
$n_w=d^{s_w}-1$ i.i.d. $N(0,\sigma^2/F_w)$ entries ($F$ = fan-in, the product of a core's non-parent leg
dimensions), independent. Hence
$$p_u=P(C_u<0)=I_{F_u/(F_u+F_w)}\big(\tfrac{n_u}2,\tfrac{n_w}2\big)\qquad(d=2,\text{ one site each: }p_u=\tfrac2\pi\arcsin\sqrt{F_u/(F_u+F_w)}).$$
The fan-ins depend on where the root is, which is why rooting matters (midroot6 vs MPS6):
- MPS chain end that is a tree leaf: $F_u=d$, $F_w=d\chi_e$: $p=0.392$ at $d=\chi_e=2$, for every $\chi$.
- MPS chain end that is the root: $F_u=d\chi_e$, $F_w=d\chi_{\rm far}$: $p=\frac2\pi\arcsin\sqrt{\chi_e/(\chi_e+\chi_{\rm far})}$:
  0.5 at $\chi=2$, 0.392 at $\chi=4$ (end bond capped at 2).
- Route end with an extra leg (internal end) on an MPS: $p=0.5$ at $\chi=2$; at $\chi\ge3$ on its route edge it is
  protected (Proposition 2, $c_e\ge2>n_w=1$).
- HT/binary leaves at a rank-one saddle: $j=3$, not active for $k<6$.
For two ends ($k\ge5$, $\sigma\to0$): $P(\text{dead})=p_up_{u'}$, $P(\text{pinned})=p_u+p_{u'}-2p_up_{u'}$; for $m$ ends,
$P(\ge2$ dying$)$; at $k=4$ multiply by the probability that the reduced competition ends dead.
Examples ($d=\chi=2$, saddle start, $k\ge5$): MPS chain with a leaf and a root end $0.196$ dead, $0.50$
pinned; midroot6 $\{0,5\}$ $0.154$ dead, $0.48$ pinned; two internal ends $0.25$ dead.

**Small init.** The correlation flow adds the head start to the route entries (the end's route row and
the partner's route entry in $Y_{\rm nv}$), which raises $C_u$, and aligns the learned directions with
the init, which correlates the ends. No closed form; Monte Carlo below.

**Dependence on tree, $\chi$, $d$, $k$** (summary of T5, Proposition 2 and the formula): $k\le3$: zero.
$k=4$: only ends whose partner carries sites, with probability below $p_up_{u'}$. $k\ge5$: $P(\ge2$ dying
ends$)$, independent of $k$ beyond the activation of channels ($j=3$ ends join at $k\ge6$, inner
three-core clusters at $k\ge7$); pinned routes turn into stalls as $k$ grows ($T_{\rm pin}\propto\sigma^{-(2k-6)}$).
$\chi$: no effect on capped end edges, protection on uncapped ones. $d$: $n_u$, $n_w$ grow as $d^{s}-1$ and
the fan-ins as $d$, so for single-site cores $p_u=I_{1/(1+\chi_e)}(\frac{d-1}2,\frac{d-1}2)$ at a leaf end, which
falls with $d$: 0.392, 0.333, 0.292 for $d=2,3,4$ at $\chi_e=2$, and 0.392, 0.25, 0.142 with the end
bond at its cap $\chi_e=d$.

### 7.5 s2s-core (odeco, three components): order from the init, trap only from the phase-2 state

The odeco runs differ in two ways: every route spans the tree ($k=K$, ends = tree leaves), and the
component that dies is almost always the **last** one (all trapped runs except 10 MPS8 runs end with two
components learned): the trap acts on the third plateau, after two escapes. Per-run results
(`predict_s2s.py`, 1,196 runs of `main`, `k78`, `large`, `balanced` with $K\ge4$; 200 (tree, $\chi$, teacher,
init, seed) predictions):
- **Learning order from the init alone** (correlation flows): the first learned component is
  predicted in **1,196 of 1,196** runs, the first two in order in 872 of 1,029 runs with two learned
  (MPS4 separated: 80/80). The order is a property of the init direction, as `notes/theory_escape.md` says.
- **Trap from the init alone: fails.** The phase-2 criterion evaluated on frozen init values (learned
  directions from the two correlation flows, no phase-1 decay) does not separate the classes: e.g.
  MPS4 $\chi=3$ fan-in: predicted dead 30/64 trapped or stalled, pinned 24/56, alive 0/40; binary8 and HT4
  at $\chi=3$: 103 of 117 and 62 of 74 predicted-alive runs trap or stall. Adding the phase-1 inertia at
  MPS ends (an end that keeps only one new direction after phase 1 cannot carry the last component)
  helps on MPS4 (predicted dead 54/96, pinned 0/64), not on the trees with siteless partners.
- **Mechanism check from the actual phase-2 state** (`gf_check_s2s.py`; gradient flow from the committed
  init only up to the second escape, $a_{q_2}=0.99$, then the criterion; MPS4 $\chi=3$, $\sigma=0.03$):
  separated teacher **10/10** (reduced route model: dead for seeds 3, 5, 6, 7, escape for the others;
  the sign test alone gets 8/10, its two errors are seeds 0, 1 whose late second escape gave the last
  component's route time to grow, so the dead state is unstable); close teacher **10/10** counting
  seed 5, which the model calls an escape with one end at $C'=+0.0097\sigma^2$ (barely alive) and the data call
  stalled at every $\sigma$; seed 7 traps at $\sigma\le0.05$ and completes at $0.07,0.1$, and at $\sigma=0.03$ the model
  says dead. HT4 ($\chi=3$, $\sigma=0.2$): see section 7.6.

So for odeco teachers the criterion is right but its input is not the init: two escapes and a phase of
GN decay with a two-dimensional new space (phase 1, where the second and third components share the
new directions and $j=3$ clusters act on HT and binary leaves) intervene. A $\sigma$-free reduced model of
phase 1 would close the gap; it is not done here.

## 9. Relation to prior work

- **`theory_saddles.md` R12 / section 8.** Same mechanism (conservation laws decide which side of a
  bond can lose its new direction), corrected in four places: (i) only the components orthogonal to
  the learned pattern decay; the pattern parts are tangent and free, so R12's $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$
  is replaced by $C_u$, which adds all non-visible partner entries (T3); per run this raises the
  stall rate of predicted-dead runs from 37% to 89% (section 7); (ii) a partner without sites gives no
  second-order decay at a rank-one saddle, so HT and binary leaves die only through three-core clusters
  ($k\ge6$) or after a second component is learned; the HT4/B4 traps of section 8 are of the second kind
  (they happen after $q_0$ and $q_B$), and the predicted "about half of the two-ended HT routes" is wrong
  (0 of 144 HT4 $k=5$ runs stall, all predicted alive); (iii) death needs two dying ends, not all ends;
  (iv) with one live end the route is pinned, not dead, with escape time $\propto\sigma^{-(2k-6)}$.
- **Furman et al. 2026** (literature.md; citation_check part b). Thm 4.3: runs reach $W^*$ along
  near-min-norm (balanced) trajectories; Thm G.1: an explicit non-min-norm spurious local minimum
  (three-leaf Tucker star). The trapped endpoints here are not local minima (the frozen partner entries
  and the inner route entries are flat directions, and moving them reopens a route), but they are
  reached because the conserved imbalance, although $O(\sigma^2)$ and so negligible against the learned
  cores (near-min-norm in Furman's sense as $\sigma\to0$), is of the same order as the dormant entries. The
  relevant scale for "balanced enough" is the dormant one, $\sigma^2$, not the learned one. Balanced init
  (Furman's imbalance knob set to zero) removes the trap at tree-leaf ends (T9).
- **Two-layer and deep linear networks.** That a conserved quantity decides which side of a unit can
  vanish is AGF's Lemma B.1 (Kunin et al. 2025); T2 is that statement for the normal-visible blocks of a
  route end. Jacot et al. 2021 (App. C.1) note that with unbalanced init the leading-order flow can
  converge to a saddle and that balanced init excludes it; Kumar & Haupt describe the shrinking of
  dormant weights outside the stable set of a KKT point. Counterpoint: Luo et al. 2025 (arXiv 2511.09925,
  Thm 2) find failures of exactly balanced Gaussian init in four-layer matrix factorization with equal
  singular values; here balanced init still leaves internal route ends exposed (T9) and slow escapes.
- **Not found** (in literature.md and the citation audit): the normal-visible blocks and the identity
  $C_u=Q_{\nu\nu}+\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2$; the visible degree and the rule $k\ge2j$; two dying ends
  for death and the pinned escape law; protection by spare directions on the end edge; an init-only
  per-run predictor of dead routes and its validation on committed runs.

## 10. What should change elsewhere (not edited here)

**`notes/theory_saddles.md`.**
1. R12 (results table) and section 8: replace "a route end decays iff $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2<0$; dead iff all
   ends unfavourable" by T2/T3/T6: the conserved quantity is $C_u$ of the normal-visible blocks, a route
   is dead iff two ends die, pinned if one does; tag derived, checked per run (this note, section 7).
2. Section 8, "How often": withdraw $P\approx0.5$ for two-ended HT routes and "MPS with $\chi=d$ stalls less
   often" (contradicted by locality 7.5 (ii)); point to section 8 here (HT leaves have $j=3$ at a rank-one
   saddle; MPS root ends have $p=0.5$, leaf ends 0.392).
3. Section 8, "Timescales and $\sigma\to0$ (conjecture)": replace by T5/T8 (derived): $k\le3$ never, $k=4$ a
   $\sigma$-free competition, $k\ge5$ the sign test; per init direction, not per $\sigma$. The "dead state stable
   iff $\gamma_1\gamma_2>\kappa^2$" heuristic is kept, with $\gamma=A^2\lvert C_u\rvert$.
4. Section 8, the HT4/B4 runs: they are traps after two learned components (off-diagonal visible entries
   of the siteless parent), not of the rank-one mechanism.
5. C8: the "parent new-slice norm$^2=-Q_{\nu\nu}(0)$" observation is exact at any dead end (Prop. 1 here).
6. P12: new predictions (T5, T6, Prop. 2, section 8 formula); P14(v) ("a dead stall with a live route
   end" refutes the picture) should say "at small $\sigma$": the 5 dead runs predicted pinned are all at
   $\sigma\ge0.19$.
7. Section 11, "Trap": replace by the open items of section 11 here; section 9 row R12: add the AGF/Jacot
   relation and the novelty items of section 9 here.

**Report Sec. 4.6 (failure modes).**
1. "Relation to the conservation-law trap": replace the R12 criterion and its weak association (30 of
   55 vs 7 of 67) by the per-run results of section 7 (fan-in: 89% of predicted-dead runs stall vs 1.2%
   of predicted-alive; stall diagnosis: 30 of 42 predicted dead died, 7 of 128 others), and drop
   "counts against the mechanism as stated": the counterexamples were R12's criterion, and the
   remaining ones are at $\sigma\ge0.19$ or in the listed failure classes.
2. Conflict (ii) is resolved (HT leaves: $j=3$; MPS root end unfavourable on average); conflict (iii) is
   explained for the drop (spare directions protect the end that has them, Prop. 2); the rise on capped
   routes from small init is reproduced by the per-run prediction (so it is in the init and the
   correlation flow) but its mechanism is not isolated.
3. Odeco paragraph: add that the dying component is the last one, that its fate is set on the third
   plateau (per seed 20/20 at $K=4$ and 6/6 HT4 runs from the phase-2 state), that the learning order is
   predicted from the init (1,196/1,196 first components), and that an init-only prediction of the odeco
   trap is not available (section 7.5).
4. Balanced paragraph: add T9 (tree-leaf ends cannot die, internal ends can; all 16 balanced stalls are
   on routes with internal ends or on MPS8 $\{0,7\}$).
5. "Steep MPS routes": a pinned end gives exponent $2k-6$ and end amplitudes $\propto\sigma^{k-3}$ (relative
   $\sigma^{k-4}$); a partially decayed end gives values in between, which matches the measured $\alpha\in[1,2]$
   and the exponents 3.3-3.8 at $k=5$ qualitatively (not tested per seed).
6. New statements and tags: T1-T9 as in section 0 here.

## 11. Open issues

- **Multi-stage (odeco) prediction from the init.** Needs a $\sigma$-free reduced model of the second
  plateau (two dormant components sharing a two-dimensional new space, $j=2$ at MPS ends, three-core
  clusters at siteless partners) to carry the init to the third plateau.
- **$k=4$ with spare inner channels** (vector route amplitudes in the reduced model) and the marginal
  three-core case $k=6$ on HT trees.
- **Inner three-core clusters** on long MPS routes ($k\ge7$): derived to act, not implemented.
- **Finite $\sigma$.** The $k\ge2j$ separation is weak at $\sigma\ge0.2$; a finite-$\sigma$ version of the reduced
  model exists for one-new-direction routes (`reduced_route(..., unit)`), not for clusters.
- **Pinned escape law** $T_{\rm pin}\propto\sigma^{-(2k-6)}$: derived, not measured per seed.
- **Balanced init, internal ends**: why predicted-dead balanced runs escape very late instead of dying
  (finite-$\sigma$ size of $C_u$?) is not quantified.

