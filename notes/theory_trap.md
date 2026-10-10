# Theory: the conservation-law trap, predicted from the initial condition

Status: **v1.2, 2026-10-10** (theorist; v1 on branch `wip/trap-theory`, v1.2 on `wip/trap-k4`). Extends `notes/theory_saddles.md` R12,
section 8, P12, C8, C14. Code and data: `experiments/theory_checks/trap/` (README there). Tags:
**proven** (complete argument), **derived** (stated assumptions, leading order in the small entries),
**empirical**, **conjecture**; **checked** marks a numerical confirmation.

> **v1.1, 2026-10-10:** the 12 errata and the status tags of `notes/reviews/trap_verification.md` are applied
> in place, marked "(verification erratum N)". Main changes: "dead iff two ends die" holds for $k\le5$ only
> (inner three-core clusters from $k=6$); the pinned escape law is a conjecture; the $k=4$ route-ODE
> verdicts at inner edges with $c>1$ are basis-dependent (MPS4 $\chi=4$ rows partly arbitrary; resolved in v1.2); the $d\ge3$
> capped-end probabilities are about 0.023 and $1.5\cdot10^{-4}$; failure list and counts corrected.

> **v1.2, 2026-10-10** (branch `wip/trap-k4`): verification erratum 4 is resolved. The $k=4$ route ODE now
> carries the inner route entries as vectors over the whole new space of the inner edge. Its outcome
> depends on them only through $p=\langle\beta_1,\beta_2\rangle$ and $S=\lvert\beta_1\rvert^2+\lvert\beta_2\rvert^2$, so the verdict is basis
> invariant (new section 2.6, Proposition 3). Checked in full gradient flow (36 of 36 runs). The MPS4 $\chi=4$
> rows are re-tabulated (predicted dead 26/26 stall, 0/182 others; v1: 26/46, 0/162), with the pooled
> counts that contain them (sections 7, 8). The "partly arbitrary" caveats are removed.

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
| T2 | In this reduced flow $C_u=X_u^{\perp\top}X_u^\perp-X_w^{\perp\top}X_w^\perp$ is conserved and the flow ends at $X_u^{\perp\top}X_u^\perp=[C_u]_+$. The end's route entry survives iff $C_u$ has a positive eigenvalue (one new direction: iff $C_u>0$). | proven (within the reduced flow); checked in full gradient flow (verification: MPS6 $\{0,5\}$, $\sigma=0.03$) |
| T3 | Identity: $C_u=\nu^\top Q^{(u)}_e\nu+\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+(y_\parallel^2-x_\parallel^2)$: the conserved bond Gram difference (end minus partner) plus the partner's non-visible new entries minus the end's; the last term is $O(\sigma^3)$ after a small-init escape. R12's $c=Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$ kept only the route part of $Y_{\rm nv}$. | proven (exact identity); $O(\sigma^3)$ remainder derived; checked (3-4 digits) |
| T4 | A partner without sites (HT, binary internal nodes) gives no second-order cost (the product is tangent). The end then decays through a three-core cluster (end, partner, sibling with sites) at rate $\propto\sigma^4$ and dies iff it is the lightest factor. In general an end with visible degree $j$ decays at rate $\propto\sigma^{2(j-1)}$. | derived |
| T5 | Decay time $\sigma^{-2(j-1)}$ against escape time $\sigma^{-(k-2)}$: an end can kill its route only if $k\ge2j$ (marginal at equality). $j=2$ (partner with sites): $k\ge4$; $j=3$ (HT/binary leaves at a rank-one saddle): $k\ge6$; $k\le3$: never. | derived; empirical (0 stalls in 5,744 committed $k\le3$ runs, locality + pilot; these verdicts are "alive" by construction of the rule, so this tests the rule, not $C_u$) |
| T6 | For $k\le5$, as $\sigma\to0$, a route dies iff at least two of its ends die. (For every $k\ge5$ two dying ends kill it: the dead state is linearly stable because the ends' mutual drive is $\propto\sigma^{k-2}\ll$ decay $\propto\sigma^2$; the "only if" is what fails from $k=6$.) One dying end is pinned at $\propto\sigma^{k-3}$; a slow escape after $\propto\sigma^{-(2k-6)}$ would follow. For $k\ge6$ three-core clusters through inner route cores (rate $\propto\sigma^4$) are a further death channel: against a live route's escape it is marginal at $k=6$ and dominant for $k\ge7$, and it pre-empts the pinned escape for $k\ge6$ (verification errata 2, 3). At $k=4$ a $\sigma$-free competition decides; a reduced route ODE from the plateau-start values settles it. It needs one new direction on each end edge; spare directions on the inner edge enter through vector route amplitudes, and the outcome depends on them only through $\langle\beta_1,\beta_2\rangle$ and $\lvert\beta_1\rvert^2+\lvert\beta_2\rvert^2$, so it is basis invariant (section 2.6; v1.2, resolves verification erratum 4). | "dead iff $\ge2$ ends die": derived for $k\le5$, incomplete for $k\ge6$; pinned amplitude: derived, checked qualitatively (one GF run); pinned escape law: conjecture (untested; pre-empted for $k\ge6$); $k=4$ competition: derived (ends with $c_e=1$, inner edges any $c$); its reduction to $(p,S)$: proven within the reduced flow; checked in full GF (36/36 runs, section 2.6) |
| T7 | Exact: $\sigma_j(G_u)^2\ge\lambda_j(Q^{(u)}_e(0))$ for all $t$; at a dead end the partner's new-slice norm$^2$ is $-Q^{(u)}_{\nu\nu}(0)$ (explains C8). Spare channels: an end with more new directions on its route edge than its partner has normal-visible rows ($c_e>n_w$) generically never dies. | bound proven; protection derived |
| T8 | The plateau-start state is $\sigma\times$ a function of the init direction (exactly for a saddle start; via the $\sigma$-free correlation flow for a small init), so the outcome is a property of the init direction. | derived ($\sigma\to0$); checked ($\sigma=0.02$ vs $0.005$: 3 digits); empirical per seed, with exceptions (section 7) |
| T9 | Balanced init ($Q_e=0$): $C_u=\lVert Y_{\rm nv}\rVert^2-\lVert X_{\rm nv}\rVert^2+O(\sigma^3)\ge0$ at ends that are tree leaves, so odeco routes cannot die; ends with extra legs (pattern teachers) still can. In s2s-core no balanced odeco run traps (0/150), but 69/150 stall (verification erratum 10). | derived (leading order); empirical (s2s-core counts) |
| T10 | Per-run predictions (init + $\sigma$-free correlation flow, no gradient flow) for the committed runs: see section 7. s2s-core: the first learned component is predicted from the init in 1,196/1,196 runs; the odeco trap is predicted only from the state after the second escape (MPS4 19/20 plus one stall with a barely-alive end; HT4 8/10), not from the init. | empirical; reproduced and independently recomputed by the verification (94/96 verdicts in a stratified sample; the two flips are $K=8$ ends with $\lvert C_u\rvert<0.003$) |
| T11 | Probability law under fan-in init: section 8. The single-end law $p_u=I_{F_u/(F_u+F_w)}(n_u/2,n_w/2)$ holds for one new direction on the end edge ($c_e=1$). At a capped end bond with $d\ge3$, $c_e=d-1>1$ and death needs $C_u\le0$ on every direction: about 0.023 ($d=3$) and $1.5\cdot10^{-4}$ ($d=4$), not 0.25 and 0.142 (verification erratum 1). | single-end law ($c_e=1$): proven (exact for the Gaussian saddle-start model), checked by Monte Carlo; route-level values: derived, empirical (Monte Carlo); $d\ge3$ capped ends: Monte Carlo only |

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
  Three or more dying ends have no linear mutual drive. **Dead iff at least two ends die, for $k\le5$ as
  $\sigma\to0$** (verification erratum 2). A further death channel was left out: three-core clusters
  through inner route cores (an end or inner core, a siteful inner core, the next core) decay at rate
  $\propto\sigma^4$ and can remove an inner route entry (for a non-differing inner site the inner route
  entry *is* the cluster's middle factor). Against a live route's escape ($\propto\sigma^{-(k-2)}$) this
  channel is marginal at $k=6$ and dominant for $k\ge7$; against a pinned route's escape
  ($\propto\sigma^{-(2k-6)}$, next item) it is marginal at $k=5$ and dominant from $k=6$, so it pre-empts
  the pinned escape for $k\ge6$. "At least two ends die" therefore still kills the route for every
  $k\ge5$, but for $k\ge6$ a route can also die without it.
- Exactly one dying end: its drive from the live route is $\propto\sigma^{k-1}$, so it is **pinned** at
  $x^*=\kappa_u/\gamma_u\propto\sigma^{k-3}$, not dead. The rest of the route then grows at
  $\dot b\approx(c^2/\gamma_u)b^{2k-3}$, so the route escapes after $T_{\rm pin}\approx\gamma_u b^{-(2k-4)}/((2k-4)c^2)\propto
  \sigma^{-(2k-6)}$ instead of $\sigma^{-(k-2)}$ (exponent 4 instead of 3 at $k=5$).
  With a fixed budget such runs are counted as stalls; their end amplitudes scale as $\sigma^{k-3}$ (relative
  $\sigma^{k-4}$), which is the "pinned" class of R12 and plausibly the steep MPS rows of report Sec. 4.6
  (section 10, item 5 of the report list).
  Status (verification erratum 3): the pinned amplitude is derived and checked qualitatively in one
  full gradient-flow run; the escape law $T_{\rm pin}\propto\sigma^{-(2k-6)}$ is a **conjecture**: untested at
  $k=5$ (where the inner-cluster channel is marginal) and pre-empted for $k\ge6$. In the verifier's
  gradient flow (MPS6 $\{0,5\}$, $k=6$, seed 3, saddle start, $\sigma=0.03$, one dying end) the end settles
  at $\lvert x\rvert/\sigma\approx2\cdot10^{-7}$ (pinned, as predicted), but the dormant overlap then *decreases*
  ($1.2\cdot10^{-20}$ at $t=2.5\cdot10^5$, $9.1\cdot10^{-21}$ at $10^6$, $1.0\cdot10^{-21}$ at $3\cdot10^8$) instead of growing:
  consistent with the inner-cluster channel, not with a pinned escape.
- $k=2j$ (e.g. $k=4$, $j=2$): decay and drive are of the same order. In rescaled time $\sigma^2t$ the reduced
  model below is $\sigma$-free, so the outcome is still a property of the init direction, but not a pure
  sign test. Also the sign of the route product matters: if $\prod_vb_v<0$ every $b_v^2$ decreases at the same
  rate until the smallest crosses zero (sign correction), which shrinks $\kappa$ while the ends decay.

**Reduced route model** (derived; used for $k=4$ and at finite $\sigma$): for a route whose ends all have
$j=2$ and one new direction on each route edge,
$$\dot b_v=c\prod_{w\ne v}b_w-[v\text{ end}]\,A^2\lvert Y_v\rvert^2b_v,\qquad \frac{d}{dt}\lvert Y_v\rvert^2=-2A^2\lVert X_v^\perp\rVert^2\lvert Y_v\rvert^2,$$
with the end's other normal rows decaying like its route row. Integrated from the plateau-start values
it returns escape / dead / pinned.
(Verification erratum 4, resolved in v1.2.) As first written the model had scalar amplitudes, and the
code applied it to inner route edges with $c>1$ by taking the first basis vector of the new space
(`complement(w)[:, 0]`). That choice is arbitrary, and the verdict could change under a rotation of
the basis (verifier: MPS4 $\{0,3\}$, $\chi=4$, seed 0 "escape" in the code's basis, "dead" in 3 of 3 random
rotations). Section 2.6 replaces the scalars by vector amplitudes and shows that the verdict is basis
invariant.

### 2.6 The $k=4$ competition with spare inner channels (v1.2)

A route with four cores is a path $u-v_1-v_2-u'$ or a star (a centre with three ends). In a star all three
route edges are end edges, so with one new direction per end edge every route entry is a scalar. In a
path the inner edge $f$ can have $c_f>1$ new directions while both end edges are capped ($c_e=c_{e'}=1$).
Example: MPS4 at $d=2$, $\chi=4$, with bond dimensions 2, 4, 2, so $c_f=3$. The inner cores' route entries are then vectors
$\beta_1,\beta_2\in\mathbb R^{c_f}$ (core $v_i$ contracted with $u_r$ on its sites, the new direction on its end
edge, the new space $N_f$ on $f$). The route's overlap with $U_r$ is $c\,x\,\langle\beta_1,\beta_2\rangle\,x'$, with $x$, $x'$ the
ends' route entries. v1 replaced $\beta_i$ by $\nu\cdot\beta_i$ for the first basis vector $\nu$ of $N_f$.

**Assumptions** (those of the reduced route model, section 2.5). Rank-one plateau, leading order in the small
entries ($\sigma\to0$). Both ends dying, with partners carrying sites ($j=2$) and $c_e=1$. The $\beta_i$ have new
indices on both route legs, so they pair with nothing at second order and move only by the route drive.
Three-core clusters (rate $\propto\sigma^4$) are slower than the $k=4$ competition (rate $\propto\sigma^2$) and are
dropped.

**Reduced flow with vector amplitudes** (derived). With $p=\langle\beta_1,\beta_2\rangle$ and $g=c\,x\,x'$:
$$\dot x=c\,p\,x'-A^2\lvert Y_u\rvert^2x,\qquad\dot x'=c\,p\,x-A^2\lvert Y_{u'}\rvert^2x',\qquad\dot\beta_1=g\,\beta_2,\qquad\dot\beta_2=g\,\beta_1,$$
and the partners' blocks evolve as in section 2.5. `trapcrit.reduced_route` now integrates every route entry
as a tensor over the new spaces of its route edges (any route shape). The drive is the gradient of the
full contraction, so the flow is equivariant under rotations of every new space.

**Proposition 3** (exact reduction; proven within the reduced flow). Let $S=\lvert\beta_1\rvert^2+\lvert\beta_2\rvert^2$.
Then $\dot p=gS$ and $\dot S=4gp$. So the reduced flow closes on $(x,x',\lvert Y_u\rvert^2,\lvert Y_{u'}\rvert^2,p,S)$
(plus the ends' other normal rows), and its outcome depends on the inner pair only through $p(0)$ and
$S(0)$. It is the scalar flow of section 2.5 with the effective inner amplitudes
$\tilde b_{1,2}=\tfrac12\big(\sqrt{S+2p}\pm\sqrt{S-2p}\big)$, which are real because $S\ge2\lvert p\rvert$. Both inputs are invariant
under rotations of $N_f$, so the $k=4$ verdict is basis invariant.
*Proof.* $\dot p=\langle\dot\beta_1,\beta_2\rangle+\langle\beta_1,\dot\beta_2\rangle=g(\lvert\beta_2\rvert^2+\lvert\beta_1\rvert^2)$ and $\dot S=2\langle\beta_1,\dot\beta_1\rangle+2\langle\beta_2,\dot\beta_2\rangle=4gp$. The
end equations see the inner pair only through $p$, and $g$ depends only on the ends. Scalars with
$\tilde b_1\tilde b_2=p$ and $\tilde b_1^2+\tilde b_2^2=S$ obey the same two equations. $\square$

**Reading: the flow picks its own direction.** Let $s=\beta_1+\beta_2$ and $m=\beta_1-\beta_2$. Then $\dot s=gs$ and $\dot m=-gm$:
the flow amplifies the fixed vector $s$ (or $m$ while $g<0$), a direction of $N_f$ set by the initialization,
not by a basis. We have $p=(\lvert s\rvert^2-\lvert m\rvert^2)/4$ and $S=(\lvert s\rvert^2+\lvert m\rvert^2)/2$, hence $p(t)=h\sinh(\varphi_0+2G(t))$ and
$S=2h\cosh(\varphi_0+2G)$, with $h=\lvert s\rvert\lvert m\rvert/2$ conserved and $G(t)=\int_0^tg$.
- A projection on a basis vector $\nu$ keeps only $(\nu\cdot s)^2$ and $(\nu\cdot m)^2$. It shrinks the growing mode and
  can flip the sign of $p$. Committed MPS4 $\{0,3\}$, seed 1: $p=+0.204$ (plateau-start units) and
  $\cos\angle(\beta_1,\beta_2)=0.61$, but $(\nu\cdot\beta_1)(\nu\cdot\beta_2)=-0.015$ in the code's basis.
- Sign correction ($P=c\,x\,p\,x'<0$). Every squared route norm ($x^2$, $x'^2$, $\lvert\beta_1\rvert^2$, $\lvert\beta_2\rvert^2$) falls at the
  same rate $2\lvert P\rvert$ (plus the ends' decay), in both models.
  - With $c_f=1$, $p=b_1b_2$ changes sign only when the smaller factor reaches zero, i.e. after a loss
    $\min(b_1^2,b_2^2)$.
  - With non-parallel $\beta_1,\beta_2$ it changes sign (at $S=2h$) after the loss
    $S_0/2-h\le\min(\lvert\beta_1\rvert^2,\lvert\beta_2\rvert^2)$, with equality iff $\beta_1\parallel\beta_2$ (Cauchy-Schwarz).
  - So the ends keep more of their size. This is the plausible reason why more routes with two dying ends
    escape with spare inner channels (section 8).

**Checks.**
- `k4_invariance.py` (`results/k4_invariance.csv`): 71 init directions with two dying ends, from MPS4 $\{0,3\}$ and $\{0,1,2,3\}$ at
  $\chi=4$: the 6 committed (case, seed) keys and 65 of the 460 Monte Carlo draws of $\{0,3\}$.
  - The new verdict is unchanged under 4 Haar-random rotations of $N_f$ in 71/71 cases.
  - The effective-scalar flow of Proposition 3 gives the same verdict in 71/71 cases. Its end and route
    trajectories agree to $\le2.4\cdot10^{-7}$ (relative; integrator rtol $10^{-9}$).
  - The v1 verdict changes under a rotation in 37/71 cases. In the code's basis it differs from the new one in
    25/71 (23 dead $\to$ escape, 2 escape $\to$ dead).
- `gf_check_k4.py` (`results/gf_check_k4.csv`): full gradient flow (library loss, `integrate_gf` with LSODA, rtol $10^{-9}$,
  atol $10^{-14}$, float64, one thread) at $\sigma=0.003$, $0.01$, $0.03$.
  - Directions: 10 of the 25 where v1 and the new verdict disagree (6 small init, incl. committed seed 1 of both
    teachers; 4 saddle start; both directions of disagreement), plus committed $\{0,3\}$ seeds 0 (the verifier's
    case) and 7 as controls.
  - The new verdict matches the gradient flow in 36/36 runs; v1 matches in 6/36, and in 0/30 on the disagreeing
    directions.
  - "Escape": $a_r\to1$, second escape at $t\propto\sigma^{-2}$. "Dead": $\lvert a_r\rvert\le2\cdot10^{-28}$ at the end, both
    route-end entries below $10^{-8}\sigma$.
  - The class does not depend on $\sigma$, as the $\sigma$-free competition predicts.

Status: Proposition 3 **proven** (within the reduced flow). It is elementary two-layer algebra, like T2:
the inner pair is a linear unit driven by the scalar $g$, and $\lvert\beta_1\rvert^2-\lvert\beta_2\rvert^2$ is its usual conserved
balance. It is new only as the repair of the $k=4$ route model. The $k=4$ verdict is **derived** ($\sigma\to0$, the assumptions
above) and basis invariant, **checked** in full gradient flow (36/36), **empirical** per run (section 7). It still
needs $c_e=1$ on both end edges. Ends with $c_e>1$ die only if $C_u\preceq0$ on all their new directions (rare;
section 3, Proposition 2), and no committed $k=4$ run has two such dying ends.

## 3. Exact statements (T7)

**Proposition 1 (Gram bound; proven).** For any core $u$ on edge $e$ and all $t$,
$G_u^{(e)}(t)^\top G_u^{(e)}(t)=Q^{(u)}_e(0)+G_w^{(e)}(t)G_w^{(e)}(t)^\top\succeq Q^{(u)}_e(0)$, hence
$\sigma_j(G_u^{(e)}(t))^2\ge\lambda_j(Q^{(u)}_e(0))$ for every $j$ (Weyl). If $\lambda_{r_e+1}(Q^{(u)}_e(0))>0$ the end
keeps $r_e+1$ directions of size $\ge\lambda_{r_e+1}^{1/2}$ forever; for an end that is a tree leaf (all its
entries on $e$ are visible) it cannot die.
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
   Precision (verification erratum 12): the stop at $M=10^{10}$ leaves the learned/small split
   ambiguous at $O(M^{-1/K})$, about 6% at $K=8$, which flips verdicts of ends with $\lvert C_u\rvert\lesssim0.003$
   (2 of 96 in the verifier's stratified recomputation, both MPS8). $M$ cannot simply be raised: DOP853
   fails at $M=10^{16}$ (finite-time blow-up). A larger $M$ needs the blow-up integrated in a
   log-amplitude variable.
2. For each route end: $j_u$, $C_u$ (two-core) or the cluster quantities (three-core), and the
   protection count $c_e$ vs $n_w$.
3. Verdict: an end is **dying** if $C_u\preceq0$ and $k\ge2j_u$. Route **dead** if $\ge2$ ends are dying,
   **pinned** if one is, **alive** otherwise. At $k=4$ with two dying ends the reduced route model decides
   (ends with $c_e=1$; spare directions on the inner edge enter through vector amplitudes, basis
   invariant; section 2.6). The criterion
   does not include the inner three-core clusters, so for $k\ge6$ "pinned" and "alive" can still die
   (section 2.5, verification erratum 2).
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
no balanced odeco run traps (0 of 150, s2s-core), but 69 of 150 stall (binary8 27/30, HT4 12/30,
MPS8 30/30; verification erratum 10, first written as "removes the odeco trap (0 of 120 trapped)").
Pattern-teacher routes that end at
a core with extra legs (MPS $\{1,5\}$, $\{1,6\}$, $\{2,5\}$, $\{1,3,5\}$) can still have $\lVert X_{\rm nv}\rVert>\lVert Y_{\rm nv}\rVert$; all
16 balanced stalls of the locality runs are on such routes or on MPS8 $\{0,7\}$ (section 7). Balanced
init does not make $C_u$ large, only non-negative: ends with small $C_u$ keep small route entries,
so the escapes can be very late (the $10^5\times$ delays of the locality balanced runs).
**Spare channels** (Proposition 2). A route edge with more new directions than its partner's
normal-visible rows protects its end; capped end bonds ($\chi_e=d$ at an MPS chain end or an HT leaf)
never do. So spare bond dimension helps only where it reaches the route's end edges. (At $d\ge3$ a capped
end bond still has $c_e=d-1\ge2$ new directions, all of which must have $C_u\le0$ for the end to die, so
death there is much rarer; section 8.)

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
(Verification erratum 6.) The pilot rule "unresolved with $\lvert a_2\rvert<10^{-20}$ = dead" is new in this note
(`analyze_locality.py`), not the pilot's own detector: pilot runs that hit $t_{\max}$ or the wall clock
with $\lvert a_2\rvert<10^{-20}$ are relabelled dead. This affects 46 of the 102 "dead" predicted-dead pilot runs,
and 10 of the 102 end with the second singular value at $10^{-9}$-$10^{-4}$, i.e. not at roundoff. With the
pilot's own collapse detector alone, 56 of 63 resolved predicted-dead runs collapse (89%) and no
predicted-alive run does. All other observed classes come from detectors defined before this work.

**Pooled, fan-in init** (stalled/runs per predicted class; runs in which the correlation flow escapes
along the larger component; the 191 wrong-order runs with $k\ge4$ are excluded, their saddle is a
different one; the correlation flow predicts them too: of the 459 escaping locality runs it starts
with the smaller component, 446 are observed out of order or rotating, and of the 8,937 it starts
with the larger, 8,834 are in weight order). L2 = this note, R12 = $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$ at the plateau start, L0 = $Q_{\nu\nu}$
alone (both with "dead iff at least two ends negative", as the code's `classify` does; this was first
written as "all ends", which is the same for two-ended routes; verification erratum 5); "reduced" =
the reduced route model at the run's own
$\sigma$, where it applies (all ends with partners carrying sites, one new direction per route edge):

| runs | criterion | pred. dead | pred. pinned | pred. alive |
|---|---|---|---|---|
| locality, saddle + small (3,492; 587 stalled) | L2 | **408/437 (93%)** | 141/934 (15%) | **38/2,121 (1.8%)** |
| | reduced, at $\sigma$ | 303/319 (95%) | 97/194 (50%) | 25/780 (3.2%) |
| | R12 | 417/1,143 (36%) | 119/1,116 | 51/1,233 (4.1%) |
| | L0 | 462/1,514 (31%) | 89/1,155 | 36/823 (4.4%) |
| locality, balanced (1,275; 28 stalled, none dead) | L2 | 16/50 | 8/235 | 4/990 |
| pilot, saddle + small (437 resolved; 124 dead) | L2 | **102/109 (94%)** die or end with $\lvert a_2\rvert<10^{-20}$; 56/63 by the pilot's own detector | 20/182 | 2/146 |
| | R12 | 84/169 | 30/185 | 10/83 |

So the criterion separates the classes far better than R12 (the gain comes from the partner's
non-visible entries, T3, and the visible degree, T4; figure `results/fig_criteria.png`), and "alive" is
a reliable prediction: 38 stalls in 2,121 runs, mostly slow escapes of barely-alive ends or the inner
three-core channel on long routes (below; "at $\sigma\ge0.08$" was dropped: that is where the grids
start, verification erratum 7). The L2 row is v1.2. In v1 it read 408/457 (89%) and 38/2,101: there,
20 MPS4 $\chi=4$ runs (all escaping) were predicted dead by the basis-dependent $k=4$ ODE (section 2.6).
**Stall diagnosis** (dead / escaped / pinned after 60x patience; R12 had 30 of 55 vs 7 of 67):
predicted dead 30 / 4 / 2, pinned 5 / 63 / 0, alive 2 / 58 / 0. (v1: dead 30 / 6 / 2, alive 2 / 56 / 0;
two MPS4 $\{0,3\}$ $\chi=4$ seed-1 runs moved from dead to alive.)

**By $k$** (fan-in, saddle + small; `results/table_by_k.md`): $k=4$: dead 203/212, pinned 2/432, alive
5/816 (v1: 203/232 and 5/796; at $k=4$ a pinned end is $O(\sigma)$, so "pinned" behaves as alive, as
predicted); $k=5$: 73/79, 10/116,
9/1,089; $k=6$: 86/98, 87/284, 18/150; $k=7$: 25/27, 8/60, 0/42; $k=8$: 21/21, 34/42, 6/24. The stall rate of
pinned runs is not monotone in $k$: 31% at $k=6$, 13% at $k=7$, 81% at $k=8$ (verification erratum 11; first
written as "grows with $k$"). Pinned stalls at $k\ge6$ may be deaths through the inner three-core channel
rather than late pinned escapes (section 2.5).

**Per tree, bond dimension and protocol** (stalled/runs; pattern teachers with $s_2/s_1=0.5$ except the
robust rows; per (tree, $D$) in `results/table_by_case.md`, per $k$ in `results/table_by_k.md`):

| source | tree | chi | start | pred. dead | pred. pinned | pred. alive |
|---|---|---|---|---|---|---|
| locality | binary8 | 2 | saddle | - | 5/18 | 0/156 |
| locality | binary8 | 2 | small | - | 0/6 | 0/168 |
| locality | binary8 | 2 | small_balanced | - | - | 0/238 |
| locality | binary8 | 4 | small | - | - | 0/168 |
| locality | ht4 | 2 | saddle | 18/24 | 0/24 | 8/172 |
| locality | ht4 | 2 | small | - | 0/36 | 0/156 |
| locality | ht4 | 2 | small_balanced | - | - | 0/318 |
| locality | ht4 | 4 | small | 7/12 | 6/36 | 0/150 |
| locality | ht8 | 2 | saddle | 7/9 | 2/18 | 1/78 |
| locality | midroot6 | 2 | saddle | 18/18 | 5/30 | 6/30 |
| locality | midroot6 | 2 | small | 12/12 | 12/42 | 6/24 |
| locality | midroot6 | 2 | small_balanced | - | 0/18 | 0/60 |
| locality | midroot6 | 4 | small | 12/12 | 18/42 | 0/24 |
| locality | mps4 | 2 | saddle | 24/24 | 0/36 | 0/36 |
| locality | mps4 | 2 | small | 76/76 | 0/100 | 0/62 |
| locality | mps4 | 2 | small_balanced | - | - | 0/78 |
| locality | mps4 | 4 (*) | small | 26/26 | 0/68 | 0/114 |
| locality | mps4 | 8 | small | - | - | 0/48 |
| locality | mps6 | 2 | saddle | 50/58 | 21/100 | 10/86 |
| locality | mps6 | 2 | small | 47/48 | 0/102 | 1/58 |
| locality | mps6 | 2 | small_balanced | - | 6/92 | 0/152 |
| locality | mps6 | 4 | small | 14/14 | 20/98 | 0/102 |
| locality | mps8 | 2 | saddle | 45/45 | 35/108 | 0/27 |
| locality | mps8 | 2 | small | 37/44 | 17/64 | 4/66 |
| locality | mps8 | 2 | small_balanced | 16/50 | 2/125 | 4/54 |
| locality | mps8 | 4 | small | 15/15 | 0/6 | 2/126 |
| locality | star3 | 2 | saddle | - | - | 0/96 |
| locality | star3 | 2 | small | - | - | 0/90 |
| locality | star3 | 2 | small_balanced | - | - | 0/90 |
| locality | star3 | 4 | small | - | - | 0/84 |
| pilot | binary8 | 2 | saddle | - | 3/6 | 0/22 |
| pilot | binary8 | 2 | small | - | - | 0/55 |
| pilot | mps4 | 2 | saddle | 10/10 | 0/30 | - |
| pilot | mps4 | 2 | small | 40/40 | 0/40 | 0/40 |
| pilot | mps6 | 2 | saddle | 20/25 | 11/38 | 2/5 |
| pilot | mps6 | 2 | small | 32/34 | 6/68 | 0/24 |

(*) v1.2. The $k=4$ verdicts on this row come from the reduced route ODE with a $c=3$ inner edge. v1
projected that edge on an arbitrary basis vector and gave 26/46, 0/68, 0/94; the invariant version
(section 2.6) gives 26/26, 0/68, 0/114. The 48 $d=4$ robust runs (end bonds uncapped, all predicted
alive, none stalls) are in the alive column both times. Observed class against predicted class for
the 160 $d=2$ runs:

| observed \ predicted | v1: dead | v1: pinned | v1: alive | v1.2: dead | v1.2: pinned | v1.2: alive |
|---|---|---|---|---|---|---|
| stalled (5,000-evaluation detector) | 24 | 0 | 0 | 24 | 0 | 0 |
| dead (stall diagnosis) | 2 | 0 | 0 | 2 | 0 | 0 |
| escaped | 20 | 68 | 46 | 0 | 68 | 66 |

Per teacher: $\{0,3\}$ 14/28, 0/56, 0/76 in v1 and 14/14, 0/56, 0/90 in v1.2; $\{0,1,2,3\}$ 12/18, 0/12, 0/18
and 12/12, 0/12, 0/24. The 20 runs that moved are seed 1 of both teachers, which escapes at every $\sigma$
(0.001-0.2).

Readings: binary8 and star3 (all ends have siteless partners; $j=3$, active only for $k\ge6$) never
stall where predicted alive; MPS4 at $\chi=2$ is predicted exactly (locality and pilot, fan-in: all
150 predicted-dead runs stalled or died, none of the 344 others), and so is MPS4 at $\chi=4$ in v1.2
(26 of 26 and none of the 182 others); HT4 with $k=5$ ($D=\{1,2\}$) is predicted
alive and its 8 saddle-start stalls are slow escapes except two (below); the $\chi=4$ rows reproduce
the measured effect of spare channels: fewer deaths where a route end gains spare directions
(Proposition 2), none fewer where both ends are capped (midroot6 $\{0,5\}$: 30/30 predicted dead or
pinned, 30/30 stalled; MPS6 $\{0,5\}$: 40/40 predicted dead or pinned, 24/40 stalled).

**Per seed** (cases with $\ge3$ values of $\sigma$): locality, predicted alive: 449 of 464 seeds never stall,
2 stall at every $\sigma$, 13 at some; predicted dead: 45 of 63 stall at every $\sigma$, 2 never, 16 at some
(the escapes are mostly at the largest $\sigma$ values, as the $O(\sigma)$ corrections of section 5 predict;
v1: 447 of 462, and 45 of 65 with 4 never);
pinned: 127 of 160 never, 13 always. Pilot: alive 24/25 never; dead 15/17 always; pinned 29/34 never.
So the class is a property of the init direction wherever the prediction is alive or dead; the mixed
seeds are near the boundary or at $\sigma\ge0.1$.

**Where the criterion fails** (fan-in: 36 false alarms and 40 misses among 3,929 runs incl. pilot; v1:
56 false alarms, the 20 extra ones being failure 1):
1. Resolved in v1.2: $k=4$ with spare inner channels (MPS4 at $\chi=4$: in v1, 20 of 46 predicted-dead
   runs escaped). The cause was the projection of the inner route entries on one basis vector of the
   inner new space (verification erratum 4), which is arbitrary and can shrink the growing mode or flip
   the sign of $\langle\beta_1,\beta_2\rangle$. With vector route amplitudes (section 2.6) the 20 runs (seed 1 of both
   teachers) are predicted alive, and all 26 remaining predicted-dead runs stall or die.
2. The marginal $j=3$ case $k=6$ on HT4 ($D=\{0,1,2\}$: 11 of 18 predicted dead escape). At $k=2j$ the
   three-core decay and the escape are of the same order; only the sign test was used.
3. Large $\sigma$, and long routes (rewritten after verification erratum 7):
   - Large $\sigma$, where the time-scale separation behind the $k\ge2j$ rule is weak: HT4 $\{1,2\}$
     ($k=5$, $j=3$), saddle start, seed 1: two runs at $\sigma=0.19,0.30$ are dead although predicted alive
     (both cluster signs negative). The same seed stalls in all 8 of its saddle-start runs, at all six
     $\sigma$ (0.03-0.3); the other six stalls are slow escapes.
   - Not large $\sigma$: long MPS and midroot6 routes stall in 24 predicted-alive runs. The 18 at $k=6$
     are all seed 4: midroot6 $\{0,5\}$ at all six $\sigma$ of its small-init grid and at 0.08-0.167 from a
     saddle; MPS6 $\{0,5\}$ and $\{0,\dots,5\}$ from a saddle at 0.08-0.24 (they escape at 0.347 and 0.5); and
     one small-init MPS6 $\{0,5\}$ run at 0.5. The grids start at 0.08, so "all at $\sigma\ge0.08$" was the
     grid, not a finding, and most of these failures sit at its *small* end. The ends are barely
     alive (MPS6 $\{0,5\}$ seed 4: $C=+0.007$): slow escapes beyond the stall patience, or the inner
     three-core channel, marginal at $k=6$ (section 2.5). Seed-specific, not a finite-$\sigma$ effect. The 6 at $k=8$ (MPS8 $\{0,7\}$, small init, seeds 1-3, $\sigma=0.2$-0.6) fit the inner
     three-core clusters (end or inner core, siteful inner core, next core; rate $\propto\sigma^4$), which
     outrun $k\ge7$ escapes (derived, not implemented).
   - Three predicted-dead runs escape at large $\sigma$: HT8 $\{0,1,2,3\}$ at $\sigma=0.455$ and 0.6, and
     MPS8 $\{1,6\}$ at 0.5 (first written as "two runs").
4. Seeds near the boundary at $k=4$-5: MPS6 $\{1,4\}$ saddle (one seed at all six $\sigma$), MPS8 $\{1,5\}$
   small (one seed, $\sigma\ge0.075$), MPS6 $\{0,3\}$, $\{0,4\}$ (10 runs, locality and pilot) escape although
   predicted dead; 7 $k=4$ saddle-start runs stall although predicted alive.
5. Dead runs predicted pinned (stall diagnosis: 5 of 68): MPS6 $\{0,5\}$ at $\chi=4$, $\sigma=0.35$-$0.5$ (4 runs)
   and MPS6 $\{1,3,5\}$ saddle, $\sigma=0.19$ (1 run, the "both ends live" case of locality 7.5).
   Under-reported (verification erratum 8): the pilot adds 20 predicted-pinned runs labelled dead, at
   every $\sigma$ of their grids (0.02-0.3): MPS6 $\{0,5\}$, $\{0,4\}$, $\{0,\dots,5\}$ and binary8 (all sites)
   from a saddle, seed 3, and MPS6 $\{0,4\}$ small init, seed 4 ($k=5$-7; 16 of the 20 via the relabelling
   rule above). MPS6 $\{0,5\}$ saddle seed 3 is at roundoff at $\sigma=0.2,0.3$, near it at 0.1
   ($s_2=3\cdot10^{-10}$) and still declining at 0.02-0.05 ($s_2\sim5\cdot10^{-8}$ at timeout); the verifier's
   gradient flow at $\sigma=0.03$ shows the same decline after pinning (section 2.5). So pinned $\to$ dead is
   not a large-$\sigma$ artefact: the likely cause is the inner three-core channel (marginal at $k=5$,
   dominant from $k=6$ against the pinned escape), which the criterion does not include.
6. Not listed before (verification erratum 12): at $K=8$ the correlation-flow state at $M=10^{10}$ is
   ambiguous at about 6% (section 4, step 1), which flips verdicts of ends with $\lvert C_u\rvert\lesssim0.003$
   (2 of 96 in the verifier's sample).
Balanced init: 16 of the 50 predicted-dead runs stall (all with internal ends on MPS8, T9), none of
them dead in the diagnosis: with $Q=0$ the predicted-dead ends have small $\lvert C_u\rvert$, and at finite $\sigma$
they escape very late instead (the $10^5\times$ delays). A quantitative failure of the dead / pinned
split for balanced runs, not of the sign.

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
  component's route time to grow, so the dead state is unstable); close teacher **9/10** plus seed 5,
  which the model calls an escape with one end at $C'=+0.0097\sigma^2$ (barely alive) and the data call
  stalled at every $\sigma$: consistent, but not predicted (first written as "10/10 counting seed 5";
  verification erratum 9). Seeds 1 and 7 are dead (observed trapped); seed 7 traps at $\sigma\le0.05$ and
  completes at $0.07,0.1$, and at $\sigma=0.03$ the model says dead. Over both teachers: **19/20 plus one
  stall with a barely-alive end**. HT4 ($\chi=3$, $\sigma=0.2$): see section 7.6.

So for odeco teachers the criterion is right but its input is not the init: two escapes and a phase of
GN decay with a two-dimensional new space (phase 1, where the second and third components share the
new directions and $j=3$ clusters act on HT and binary leaves) intervene. A $\sigma$-free reduced model of
phase 1 would close the gap; it is not done here.

**Init-only phase-2 verdict vs observed class** (t/s/c = trapped / stalled / complete; unresolved
runs omitted; `results/table_s2s.md`). Where it predicts dead or pinned on HT4 the run traps (44 of
44 at $\chi=3$ separated, 23 of 23 close), but most HT4 and all binary8 traps are predicted alive; on
MPS4 half of the predicted-dead runs complete. Balanced init: all 149 resolved runs predicted alive,
none trapped (T9), but 69 stall (binary8 27/30, HT4 12/30, MPS8 30/30; verification erratum 10).

| tree | chi | teacher | init | pred. dead: t/s/c | pred. pinned: t/s/c | pred. alive: t/s/c |
|---|---|---|---|---|---|---|
| binary8 | 3 | close | fanin | - | - | 27/13/8 |
| binary8 | 3 | separated | balanced | - | - | 0/27/3 |
| binary8 | 3 | separated | fanin | - | - | 30/33/6 |
| binary8 | 6 | close | fanin | - | - | 0/0/11 |
| binary8 | 6 | separated | fanin | - | - | 0/0/55 |
| ht4 | 3 | close | fanin | 12/0/0 | 11/0/0 | 22/4/3 |
| ht4 | 3 | separated | balanced | - | - | 0/12/17 |
| ht4 | 3 | separated | fanin | 18/0/0 | 26/0/0 | 29/7/9 |
| ht4 | 6 | separated | fanin | - | - | 5/0/45 |
| mps4 | 3 | close | fanin | 14/0/18 | 0/8/24 | 0/0/16 |
| mps4 | 3 | separated | balanced | - | - | 0/0/30 |
| mps4 | 3 | separated | fanin | 16/0/16 | 16/0/8 | 0/0/24 |
| mps4 | 6 | close | fanin | - | 0/0/32 | 0/0/48 |
| mps4 | 6 | separated | balanced | - | - | 0/0/30 |
| mps4 | 6 | separated | fanin | - | 0/0/40 | 0/0/40 |
| mps8 | 3 | close | fanin | 5/13/0 | 13/6/0 | - |
| mps8 | 3 | separated | balanced | - | - | 0/30/0 |
| mps8 | 3 | separated | fanin | 13/17/0 | 22/11/0 | - |
| mps8 | 6 | close | fanin | - | - | 2/0/0 |
| mps8 | 6 | separated | fanin | 2/2/0 | 1/19/0 | 4/13/0 |

### 7.6 Mechanism check on HT4 and binary8 (odeco, $\chi=3$, $\sigma=0.2$, seeds 0-5; HT4 seeds 6-9 from the verification)

Gradient flow from the committed init up to the second escape, then the phase-2 sign test at the four
leaf ends (`gf_check_s2s.py tree ...`; $k=7$, so the $\sigma\to0$ rule "dead iff two ends die" is used):
- **HT4: 6/6.** Predicted dead for seeds 0, 1, 2, 3, 5 (two leaves with $C'<0$ each), alive for seed 4;
  observed (k78 and main): trapped, trapped, trapped, trapped, complete, trapped. With two learned
  components the siteless parents' off-diagonal entries are visible ($j=2$, section 2.4), which is what
  kills HT leaves here.
  Out of sample (verification erratum 9; the verifier ran `gf_check_s2s.py tree ht 4 0.2 6,7,8,9`):
  seed 7 dead $\to$ trapped (right), seed 9 alive $\to$ stalled, not trapped (right), seeds 6 and 8 pinned
  (one leaf with $C'<0$) $\to$ trapped (wrong under "dead iff two ends die"). So **6/6 on seeds 0-5,
  8/10 on all ten seeds**. At $k=7$ and $\sigma=0.2$ a pinned leaf is still far above the "trapped"
  threshold ($10^{-8}\sigma$; roughly $10^{-5}$ relative), so "trapped = deeply pinned" (the binary8 reading
  below) is unlikely to rescue seeds 6 and 8; a second death channel (section 2.5) is more likely.
- **binary8: consistent, not decisive.** Seed 5 dead (trapped); seeds 3, 4 pinned (trapped); seed 1 pinned
  and seeds 0, 2 alive (stalled, i.e. not at roundoff). "Trapped" at $K=7$ (a slice below $10^{-8}\sigma$) also
  catches deeply pinned ends, whose value $\kappa_u/\gamma_u$ contains six other route amplitudes; the
  two-leaf-end dead/pinned split is therefore not testable with this label.

## 8. Probability law under fan-in init (T11)

**Saddle start, one new direction ($c_e=1$), partner with sites (closed form; proven for this Gaussian
model; checked by the verifier's Monte Carlo, $2\cdot10^5$ draws, in five $c_e=1$ cases to within 0.002).** With learned
directions fixed by the saddle, $X_u^\perp$ has $n_u=d^{s_u}-1$ i.i.d. $N(0,\sigma^2/F_u)$ entries and $X_w^\perp$ has
$n_w=d^{s_w}-1$ i.i.d. $N(0,\sigma^2/F_w)$ entries ($F$ = fan-in, the product of a core's non-parent leg
dimensions), independent. Hence
$$p_u=P(C_u<0)=I_{F_u/(F_u+F_w)}\big(\tfrac{n_u}2,\tfrac{n_w}2\big)\qquad(d=2,\text{ one site each: }p_u=\tfrac2\pi\arcsin\sqrt{F_u/(F_u+F_w)}).$$
This holds for $c_e=1$ only. With $c_e>1$ new directions on the end edge the end dies only if
$C_u\preceq0$ on all of them (T2), which is much rarer (verification erratum 1; below).
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
$k=4$: only ends whose partner carries sites, with probability below $p_up_{u'}$. $k=5$: $P(\ge2$ dying
ends$)$. $k\ge6$: at least that, plus deaths through channels the law does not count ($j=3$ ends join at
$k\ge6$; inner three-core clusters are marginal at $k=6$ and dominant for $k\ge7$ against a live route,
and from $k=6$ they can kill pinned routes; verification erratum 2). The fate of pinned routes is not
predicted (the escape law $T_{\rm pin}\propto\sigma^{-(2k-6)}$ is a conjecture, pre-empted for $k\ge6$).
$\chi$: no effect on capped end edges, protection on uncapped ones. $d$: $n_u$, $n_w$ grow as $d^{s}-1$ and
the fan-ins as $d$, so for single-site cores with $\chi_e=2$ ($c_e=1$), $p_u=I_{1/(1+\chi_e)}(\frac{d-1}2,\frac{d-1}2)$
at a leaf end, which falls with $d$: 0.392, 0.333, 0.292 for $d=2,3,4$. With the end bond at its cap
$\chi_e=d$, $c_e=d-1$ and death needs $C_u\le0$ on all $d-1$ directions: Monte Carlo (verification) about
0.023 at $d=3$ and $1.5\cdot10^{-4}$ at $d=4$ (0.392 at $d=2$). The scalar law first gave 0.25 and 0.142
here, which is wrong (verification erratum 1). No closed form for $c_e>1$ is derived.

**Monte Carlo** (`montecarlo.py`; init seeds 100000+, disjoint from the committed ones; $\sigma\to0$
verdicts; 400 draws per saddle-start case, 60 per small-init case; `results/montecarlo_summary.csv`):

| case ($d=2$) | $k$ | saddle: P(dead) MC / closed form | saddle: P(pinned) | small init: P(dead) | small: P(pinned) |
|---|---|---|---|---|---|
| MPS4 $\{0,3\}$, $\chi=2$ | 4 | 0.11 / (sign only 0.20) | 0.56 | 0.17 | 0.45 |
| MPS4 $\{0,3\}$, $\chi=4$ (*) | 4 | 0.05 / (0.15) | 0.47 | 0.05 | 0.45 |
| MPS6 $\{0,3\}$ | 4 | 0.12 / (0.20) | 0.53 | 0.12 | 0.52 |
| MPS6 $\{1,4\}$ (internal ends) | 4 | 0.15 / (0.25) | 0.50 | 0.10 | 0.57 |
| MPS6 $\{1,3,5\}$ | 5 | 0.22 / 0.25 | 0.50 | 0.20 | 0.43 |
| MPS8 $\{1,5\}$ | 5 | 0.25 / 0.25 | 0.50 | 0.28 | 0.33 |
| MPS6 $\{0,5\}$, $\chi=2$ | 6 | 0.19 / 0.196 | 0.46 | 0.22 | 0.57 |
| MPS6 $\{0,5\}$, $\chi=4$ | 6 | 0.145 / 0.154 | 0.51 | 0.22 | 0.42 |
| midroot6 $\{0,5\}$ | 6 | 0.17 / 0.154 | 0.47 | 0.10 | 0.55 |
| HT4 all sites ($j=3$, 4 ends) | 7 | 0.33 / - | 0.51 | 0.42 | 0.48 |
| MPS8 $\{0,7\}$ | 8 | 0.215 / 0.196 | 0.47 | 0.25 | 0.53 |

(Standard errors: about 0.02 for 400 draws, 0.04-0.06 for 60. (*) v1.2: the $k=4$ verdicts with a
$c=3$ inner edge come from the invariant route ODE of section 2.6; the basis-dependent v1 gave 0.09 and
0.08. Of the 58 saddle-start draws with two dying ends, 38 escape and 20 die; of the 7 small-init ones,
4 escape.) The closed form agrees with the Monte Carlo within two standard errors at every $k\ge5$ case.
At $k=4$ the reduced competition lets about 40% of the draws with two unfavourable ends escape at
$\chi=2$ (MPS4, MPS6), and about 65% with three spare inner directions (MPS4 at $\chi=4$; section 2.6).
The trap probability per run is of order 0.1-0.3 for routes whose two ends have partners with sites,
0.05 on MPS4 at $\chi=4$, and about 0.3-0.4 for the four-ended HT route at $k=7$.

**Comparison with the measured fractions.** The committed cases have 5-10 init seeds each, and the
class is fixed per seed, so a measured fraction has a binomial error of order 0.15-0.2. Example: MPS4
$\{0,3\}$ small init, measured 5 of 8 seeds (30 of 48 runs, locality; 5 of 8 in the pilot); predicted for
those very seeds: the same 5; predicted over 48 seeds 0-47: 0.29; over 60 fresh seeds: 0.17. The
committed seeds are an unusually trap-prone draw (probability about 3% at $p=0.25$), not a failure of
the law. Section 7's per-run tests are the sharper test; the law is for planning (how many seeds,
which trees). Figure `results/fig_probability.png` (`probability_vs_measured.csv`) puts the measured
stall fraction of each committed case next to the predicted range: for $k\ge5$ it lies between P(dead)
and P(dead or pinned) in 8 of 13 cases (one at the edge); above the range on the long chains
(midroot6 $\{0,5\}$ 0.67 and 0.73 vs at most 0.65; MPS8 $\{0,7\}$ saddle 0.93 vs 0.69: the inner three-core
channel of failure 3 is not in the law), below it for HT4 small (0.27 vs 0.42) and MPS6 $\{1,3,5\}$ small
(0.125 vs 0.20; 6-8 seeds each). At $k=4$ the measured fraction should match P(dead): within 0.15 in 7
of 9 cases; the exceptions are the small-init MPS4 $\{0,3\}$ and MPS6 $\{0,3\}$ seeds (0.37 vs 0.17, 0.36 vs
0.12), whose per-run predictions are right (section 7): trap-prone draws.

## 9. Relation to prior work

- **`theory_saddles.md` R12 / section 8.** Same mechanism (conservation laws decide which side of a
  bond can lose its new direction), corrected in four places: (i) only the components orthogonal to
  the learned pattern decay; the pattern parts are tangent and free, so R12's $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2$
  is replaced by $C_u$, which adds all non-visible partner entries (T3); per run this raises the
  stall rate of predicted-dead runs from 36% to 89% (section 7); (ii) a partner without sites gives no
  second-order decay at a rank-one saddle, so HT and binary leaves die only through three-core clusters
  ($k\ge6$) or after a second component is learned; the HT4/B4 traps of section 8 are of the second kind
  (they happen after $q_0$ and $q_B$), and the predicted "about half of the two-ended HT routes" is wrong
  (0 of 144 HT4 $k=5$ runs stall, all predicted alive); (iii) death needs two dying ends, not all ends
  (for $k\le5$ as $\sigma\to0$; from $k=6$ inner three-core clusters are a further channel); (iv) with one
  live end the route is pinned, not dead; its escape time $\propto\sigma^{-(2k-6)}$ is a conjecture, pre-empted
  for $k\ge6$ (verification errata 2, 3).
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
  for death ($k\le5$) and the pinned state; protection by spare directions on the end edge; an init-only
  per-run predictor of dead routes and its validation on committed runs.

## 10. What should change elsewhere (not edited here)

(2026-10-10: items 1-7 below were applied to `notes/theory_saddles.md`, as corrected by the
verification; the corrections are marked in the items. The report items are for the lead.)

**`notes/theory_saddles.md`.**
1. R12 (results table) and section 8: replace "a route end decays iff $Q_{\nu\nu}+\lvert y_{\rm route}\rvert^2<0$; dead iff all
   ends unfavourable" by T2/T3/T6: the conserved quantity is $C_u$ of the normal-visible blocks, a route
   is dead iff two ends die, pinned if one does, for $k\le5$ as $\sigma\to0$ (from $k=6$ inner three-core
   clusters are a further death channel; verification erratum 2); tag derived, checked per run (this
   note, section 7).
2. Section 8, "How often": withdraw $P\approx0.5$ for two-ended HT routes and "MPS with $\chi=d$ stalls less
   often" (contradicted by locality 7.5 (ii)); point to section 8 here (HT leaves have $j=3$ at a rank-one
   saddle; MPS root ends have $p=0.5$, leaf ends 0.392, at $d=\chi=2$ with $c_e=1$; at a capped end bond with
   $d\ge3$ far less, verification erratum 1).
3. Section 8, "Timescales and $\sigma\to0$ (conjecture)": replace by T5/T8 (derived): $k\le3$ never, $k=4$ a
   $\sigma$-free competition (one new direction per route edge), $k=5$ the sign test, $k\ge6$ the sign test
   plus the inner three-core channel (not in the criterion); per init direction, not per $\sigma$. The
   "dead state stable iff $\gamma_1\gamma_2>\kappa^2$" heuristic is kept, with $\gamma=A^2\lvert C_u\rvert$.
4. Section 8, the HT4/B4 runs: they are traps after two learned components (off-diagonal visible entries
   of the siteless parent), not of the rank-one mechanism.
5. C8: the "parent new-slice norm$^2=-Q_{\nu\nu}(0)$" observation is exact at any dead end (Prop. 1 here).
6. P12: new predictions (T5, T6, Prop. 2, section 8 formula). P14(v) ("a dead stall with a live route
   end" refutes the picture): the proposal to add "at small $\sigma$" is withdrawn (verification erratum
   8): the pilot has 20 predicted-pinned runs labelled dead at every $\sigma$ of their grids (section 7,
   failure 5). Instead restrict (v) to routes on which one live end must keep the route alive: $k\le4$
   (at $k=5$ the inner-cluster channel is marginal against the pinned escape; from $k=6$ it dominates).
7. Section 11, "Trap": replace by the open items of section 11 here; section 9 row R12: add the AGF/Jacot
   relation and the novelty items of section 9 here.

**Report Sec. 4.6 (failure modes).**
1. "Relation to the conservation-law trap": replace the R12 criterion and its weak association (30 of
   55 vs 7 of 67) by the per-run results of section 7 (fan-in: 93% of predicted-dead runs stall vs 1.8%
   of predicted-alive; stall diagnosis: 30 of 36 predicted dead died, 7 of 128 others; v1.2 numbers, v1
   had 89% and 30 of 38, 7 of 126), and drop
   "counts against the mechanism as stated": the counterexamples were R12's criterion, and the
   remaining ones are in the listed failure classes (not "at $\sigma\ge0.19$": pinned runs die at every
   $\sigma$ in the pilot, section 7 failure 5, verification erratum 8).
2. Conflict (ii) is resolved (HT leaves: $j=3$; MPS root end unfavourable on average); conflict (iii) is
   explained for the drop (spare directions protect the end that has them, Prop. 2); the rise on capped
   routes from small init is reproduced by the per-run prediction (so it is in the init and the
   correlation flow) but its mechanism is not isolated.
3. Odeco paragraph: add that the dying component is the last one, that its fate is set on the third
   plateau (per seed at $K=4$: 19/20 plus one stall with a barely-alive end; HT4: 6/6 on seeds 0-5,
   8/10 on all ten seeds; both from the phase-2 state, verification erratum 9), that the learning order is
   predicted from the init (1,196/1,196 first components), and that an init-only prediction of the odeco
   trap is not available (section 7.5).
4. Balanced paragraph: add T9 (tree-leaf ends cannot die, internal ends can; all 16 balanced stalls are
   on routes with internal ends or on MPS8 $\{0,7\}$) and that no balanced odeco run traps (0/150) but
   69/150 stall (verification erratum 10).
5. "Steep MPS routes": a pinned end gives exponent $2k-6$ and end amplitudes $\propto\sigma^{k-3}$ (relative
   $\sigma^{k-4}$); a partially decayed end gives values in between, which matches the measured $\alpha\in[1,2]$
   and the exponents 3.3-3.8 at $k=5$ qualitatively (not tested per seed). Tag conjecture: the pinned
   escape law is untested at $k=5$ and pre-empted for $k\ge6$ (verification erratum 3).
6. New statements and tags: T1-T9 as in section 0 here.
7. (v1.2) $k=4$ with spare inner channels. These numbers changed:
   - Fan-in locality: predicted dead $408/457$ (89%) becomes $408/437$ (93%); alive $38/2{,}101$ becomes
     $38/2{,}121$ (1.8%); pinned unchanged.
   - By $k$, $k=4$: $203/232$ becomes $203/212$; $5/796$ becomes $5/816$.
   - Stall diagnosis: predicted dead 30 dead / 6 escaped / 2 pinned becomes 30 / 4 / 2; predicted alive
     2 / 56 / 0 becomes 2 / 58 / 0.
   - Known failures: "56 false alarms" becomes 36. The failure class "$k=4$ with spare inner channels
     (MPS4 at $\chi=4$: 20 of 46 predicted-dead runs escape)" is removed: MPS4 at $\chi=4$ is now 26 of 26
     predicted-dead runs stalled or dead, none of the other 182.
   - Monte Carlo, MPS4 $\{0,3\}$ $\chi=4$: P(dead) 0.05 (saddle and small init; v1 0.09 and 0.08).

   Statement (iv) of `prop:A-trap` should drop "(where every route edge has one new direction)". Replace it
   by: one new direction on each end edge; spare directions on the inner edge enter through vector route
   amplitudes, and the decision depends on them only through $\langle\beta_1,\beta_2\rangle$ and
   $\lvert\beta_1\rvert^2+\lvert\beta_2\rvert^2$, hence is basis invariant (Proposition 3, proven within the
   reduced flow; checked in full gradient flow, 36/36 runs).

## 11. Open issues

- **Multi-stage (odeco) prediction from the init.** Needs a $\sigma$-free reduced model of the second
  plateau (two dormant components sharing a two-dimensional new space, $j=2$ at MPS ends, three-core
  clusters at siteless partners) to carry the init to the third plateau.
- **The marginal three-core case $k=6$ on HT trees** (only the sign test is used there). The $k=4$ case
  with spare inner channels is closed in v1.2 (section 2.6). Two extensions remain open: a finite-$\sigma$
  reduced model for routes with spare inner channels at $k\ge5$, and ends with $c_e>1$. The finite-$\sigma$
  column of section 7 still requires $c=1$ on every route edge, although `reduced_route` now accepts
  tensor amplitudes.
- **Inner three-core clusters** (verification erratum 2): a death channel for live routes from $k=6$
  (marginal) and $k\ge7$ (dominant), and for pinned routes from $k=6$ (marginal at $k=5$). Derived to act,
  not implemented in the criterion; likely behind the pinned $\to$ dead runs (section 7, failure 5) and
  HT4 seeds 6 and 8 (section 7.6).
- **Capped end bonds at $d\ge3$**: $c_e=d-1>1$; only Monte Carlo values (about 0.023 and $1.5\cdot10^{-4}$),
  no closed form (verification erratum 1).
- **Correlation-flow precision at $K\ge8$**: integrate the blow-up in a log-amplitude variable to go
  beyond $M=10^{10}$ (verification erratum 12).
- **Finite $\sigma$.** The $k\ge2j$ separation is weak at $\sigma\ge0.2$; a finite-$\sigma$ version of the reduced
  model exists for one-new-direction routes (`reduced_route(..., unit)`), not for clusters.
- **Pinned escape law** $T_{\rm pin}\propto\sigma^{-(2k-6)}$: a conjecture, not measured per seed; testable
  only at $k=5$ (marginal there), pre-empted for $k\ge6$ (verification erratum 3).
- **Balanced init, internal ends**: why predicted-dead balanced runs escape very late instead of dying
  (finite-$\sigma$ size of $C_u$?) is not quantified.

