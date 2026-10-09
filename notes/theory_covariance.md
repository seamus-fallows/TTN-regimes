# Theory: input covariance (Q6, H5)

Status: **v2, 2026-10-09** (theorist for Q6/H5), revised after an independent verification (verifier scripts
and outputs: `.../scratchpad/theory2/covariance_verify/`). Scratch scripts for every numerical check are in
`C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/theory2/covariance/`
(outside the repo; v1: `check1*`–`check4*`, v2: `check5*`, `diag5c_phases.py`, helpers `common.py`, `common5.py`).
They use only `ttn.tree`, `ttn.network`, `ttn.teachers`, `ttn.losses`; gradient flow (GF) is
`scipy.integrate.solve_ivp` (DOP853, rtol 1e-9 to 1e-11) on torch-autograd gradients of
`ttn.losses.population_loss` in float64 (for cross-site correlations: of the exact non-product quadratic form,
section 7). Tags: **proven** (complete argument given), **derived** (under the stated assumptions, leading order
in the small parameters), **heuristic**, **conjecture**; **checked** marks a numerical confirmation (section 8).

Builds on: `notes/literature.md` (H5 paragraph: [Gidel2019], [Jacot2021], [Achour2024], [Atanasov2022],
[RawalDeWeese2026], [Furman2026]), `notes/theory_saddles.md` (routes, the $k$-rule, the piggyback block, head
starts; notation $q$ = learned component, $r$ = new component, $m$ = number of terminal cores, $k$ = route degree)
and `notes/theory_teachers.md` §1. `notes/theory_escape.md` did not exist when this was written, so section 4 is
phrased as a substitution rule that applies to whatever the whitened escape theory says.

## Changes in v2 (response to the verification)

| # | verifier's point | response |
|---|---|---|
| 1 | MPS route rate under cross-site correlation scales as $c^{N-2}$, not $c^2$ | **Agreed, fixed.** Exact closed form for every even $N$ (§7.4a), checked against exact Hessians for $N=4,6,8$. **Partly rebutted:** "later plateaus collapse is a small-$N$ effect" holds only at fixed small $c$. At $c\gtrsim0.2$ the Gaussian moments compensate ($\lambda_2=0.053,0.031,0.046$ at $c=0.2$ for $N=4,6,8$). More importantly, the plateau that the route competes with grows as $\sigma^{-(N-2)}$, so the crossover is at $c^*=O(\sigma)$ up to logs for every $N$. MPS6 at $\sigma=0.3$ is an example: the whitened second plateau is longer than $3.9\cdot10^4$, but only 150 at $c=0.2$ (§7.4d). |
| 2 | $\rho$ depends on the event definition | **Agreed.** Cross-site runs must use spectral or loss-level events. Overlap events give $\rho=1$ on junk-rich plateaus, and on MPS at $c\gtrsim0.3$ they fire at an intermediate saddle (§7.5, prediction 10). |
| 3 | the coefficient of $\log(1/\sigma)/\lambda_2$ was not confirmed | **Agreed; replaced by a derived and checked law.** The escape from the MPS4 product saddle has two stages and passes through binary4's junk-rich plateau point, which is a strict saddle on the MPS. Plateau increment per halving of $\sigma$: $\ln2\,[(1-\kappa)/\lambda_2+1/\lambda'']$. At $c=0.4$: 2.128 predicted vs 2.13 measured (spectral events), and 1.478 vs 1.48 (overlap events). At $c=0.2$: 14.39 vs 14.47–14.71, still decreasing at $\sigma=1.6\cdot10^{-3}$ (§7.4b–c). |
| 4 | the tolerance in prediction 3 holds only at small $\sigma$ | **Agreed; the offset is now predicted exactly** (§4): $C_Ks_r^{-(2-2/K)}(\Lambda_r^{-(2-2/K)}-\Lambda_r^{-1})$, i.e. $-0.2219$, against $-0.2206$ to $-0.2222$ measured at $\sigma=0.003$ and $0.03$. |
| 5 | C6 and prediction 6 were stated deterministically | **Agreed.** Both are now conditional on which component wins the first escape. Race statistics added: binary4 fails 0/32 at $\sigma\le0.1$ and 2/32 at $\sigma=0.3$; MPS4 fails 5/24 at both $\sigma=0.1$ and 0.05, on the same seeds (§5). |
| 6 | the $m=2$ case also needs $k\ge3$ | **Agreed, fixed**, with a two-line proof that the $k=2$ add route always wins (§5). |
| 7 | wrong example of $P\propto I$ | **Agreed, fixed** (Prop. 1.5). |
| 8 | rank of $\Omega_{ij}\otimes\mathrm{tr}_{ij}W^*$ | **Agreed, fixed** (§7.3). |
| 9 | $O(c)$ corrections to $4c^2$ and $c^2$ | **Agreed.** Replaced by an exact $d\times d$ eigenproblem, plus a first-order formula with the factor $m_{2,2}(c)/\Lambda_{\rm eff}(c,4)$ that is accurate to 0.2% at $c=0.1$ (§7.3). |
| 10 | junk-pairing routes at the MPS saddle were omitted | **Added** (closed form, rate $O(c^N)$ for $R=2$; §7.4a). |
| 11 | the $c\to1$ conditioning remark | **Fixed.** $\lambda_{\min}=(1-c)^{N-1}(1+(N-1)c)$ for $d\ge N$, and numerically $\propto(1-c)^{N-\lceil N/d\rceil}$ otherwise; $\lambda_{\max}$ grows too (§7.4, item 5). |
| 12 | wording | **All fixed.** The CP forms are exact identities. The Hessian congruence holds at every $\theta$. The one-factor Gaussian model is cited as standard. The remark on [Gidel2019] is kept (the verifier confirmed it). |

## 0. Results at a glance

| # | result | status |
|---|---|---|
| C1 | A product covariance is a whitened problem with teacher $SW^*$ ($S=\otimes_i\Sigma_i^{1/2}$), init $A\theta_0$ (anisotropic on physical legs) and gradient flow **preconditioned by $\Sigma_i$ on the physical legs** (identity on bond legs and on cores with no physical legs). Hessians are congruent at every $\theta$. So critical points, plateau levels and Morse indices transfer; times, escape directions and strict-saddle rates do not. The gauge conservation laws hold unchanged in the original coordinates. | proven, checked ($\le2\cdot10^{-8}$) |
| C2 | Aligned covariance with an odeco teacher: on the aligned family $L=\tfrac12\sum_r\Lambda_r(P_r-s_r)^2$, $\dot a_{v,r}=\Lambda_r(s_r-P_r)P_r/a_{v,r}$, and exactly $P_r(t;\Sigma)=P_r^{(I)}(\Lambda_rt)$. Half-time $t_{1/2,r}=a_0^{-(K-2)}/((K-2)\Lambda_rs_r)-C_K/(\Lambda_rs_r^{2-2/K})+O(a_0^2)$; plateaus $L_S=\tfrac12\sum_{r\notin S}\Lambda_rs_r^2$. | proven, checked ($\le1.2\cdot10^{-4}$) |
| C3 | Three orderings: escape speed $\Lambda_rs_r$, loss drop $\Lambda_rs_r^2$, whitened $s_r$. Covariance reverses the order of $s_1>s_2$ iff $\Lambda_2/\Lambda_1>s_1/s_2$. In the **mismatch window** $s_1/s_2<\Lambda_2/\Lambda_1<(s_1/s_2)^2$ the faster component is the smaller loss drop. | derived |
| C4 | Random fan-in init: near the origin $L=L_0-\langle W(\theta),T\rangle+O(\lVert W\rVert^2)$ with $T=\Sigma W^*=\mathbb E[f^*\Phi]$, and the init law does not depend on $\Sigma$. So the whole escape (direction, time, race) is the whitened one with $W^*\to T$; aligned case: $s_r\to\Lambda_rs_r$. The exponent of $t_1$ in $\Lambda$ is exactly $-1$, whatever $K$, the tree, or how $\Lambda_r$ is split over sites and cores (cores without physical legs play no role). Naive whitening (teacher $SW^*$, no preconditioner) is wrong. Problems with equal $T$ differ only by an explicit $O(1)$ saturation offset. | derived, checked (slope $-1.0000$; offset to 1–3%) |
| C5 | The split matters only where curvature enters: the piggyback (old-bond-index) block at a learned saddle with $m=2$ becomes $H_{\rm pb}=\lambda_{\rm sh}s_q^{-2/K}\begin{pmatrix}s_q^2\mu_r\nu_q&-s_rs_q\mu_r\nu_r\\-s_rs_q\mu_r\nu_r&s_q^2\mu_q\nu_r\end{pmatrix}$ (partial products of eigenvalues per terminal core); strict iff $\Lambda_rs_r^2>\Lambda_qs_q^2$. Add-routes (new bond indices) only get $s_r\to\Lambda_rs_r$. | proven at the saddle, checked (rates to 4 digits) |
| C6 | Mismatch window, with 2 the faster component and 1 the larger loss drop, **conditional on 2 winning the first escape** (probability $\to1$ as $\sigma\to0$ on binary $N=4$; $\sigma$-independent on $m\ge3$ trees). Binary $N=4$ ($m=2$, $k=3$) **rotates** from 2 to 1 through a strict saddle (bond rank stays 1), sits at $\tfrac12\Lambda_2s_2^2$, then relearns 2. MPS $N=4$ and every tree with $m\ge3$ keep 2 and plateau at the higher level $\tfrac12\Lambda_1s_1^2$. If 1 wins, no tree rotates (plateau $\tfrac12\Lambda_2s_2^2$). | derived, checked |
| C7 | Misaligned CP teachers: escape weight $\Lambda^{\rm esc}_r=\prod_i\lVert\Sigma_iu_r^{(i)}\rVert\ \ge$ loss weight $\Lambda^{\rm loss}_r=\prod_i\langle u_r^{(i)},\Sigma_iu_r^{(i)}\rangle$ (equality iff aligned). Misalignment alone can reverse the order. The escape direction has factors $\Sigma_iu/\lVert\Sigma_iu\rVert$ and realigns to $u$ afterwards. Binary $N=4$ plateaus are truncated SVDs of $\mathrm{mat}_{01\mid23}(SW^*)$. | proven (inequality, CP forms), derived (leading order in cross overlaps), checked |
| C8 | Shared latent $x^{(i)}=\sqrt{1-c}\,z_i+\sqrt c\,z_0$ (the standard one-factor Gaussian model): $M=\mathbb E[\Phi\Phi^\top]=\mathbb E_{z_0}[\otimes_i((1-c)I+cz_0z_0^\top)]$, so the loss is exactly a Gauss–Hermite mixture of $(N+1)^d$ product-metric tree contractions. $M=I+c^2\sum_{i<j}(\mathrm{SWAP}_{ij}+\lvert\Omega_{ij}\rangle\langle\Omega_{ij}\rvert)+O(c^3)$. Site-aligned weights: $\langle U,MU\rangle=\Lambda_{\rm eff}(c,N)=\sum_k\binom Nk(1-c)^{N-k}c^k(2k-1)!!$. | proven, checked ($10^{-15}$) |
| C9 | Cross-site correlation, first plateau. The escape from the origin keeps its degree $K$ (polynomial plateau). Binary $N=4$ plateaus at the $M$-weighted bond-rank-1 critical point, which is **junk-rich**: pairing tensors $e_a\otimes e_a$ on each leaf, 36–65% of $\lVert W\rVert$ outside the teacher span at $c\ge0.2$, $d=3$. Its escape direction is the top eigenvector of an explicit $d\times d$ pairing matrix. | proven (block structure), derived, checked |
| C10 | Cross-site correlation, later plateaus on MPS. The non-product metric opens strict 2-core routes at the product saddle, with exact rate $\lambda_2=P^{(N-2)/N}[s_2c^{N-2}(N-3)!!\,m_{N-2,2}(c)+\dots]$ ($\approx s_2c^{N-2}((N-3)!!)^2$ at small $c$). On MPS4 the escape has **two stages**, through binary4's plateau point (a strict intermediate saddle with rate $\lambda''$). Second-plateau increment per halving of $\sigma$: $\ln2[(1-\kappa)/\lambda_2+1/\lambda'']$. Whitened power-law plateaus become logarithmic once $c\gtrsim c^*=O(\sigma)$ up to logs. | proven (rates at the saddles), derived (durations), heuristic ($c^*$, $N\ge6$), checked ($N=4$; rates $N=4,6,8$; MPS6 collapse) |
| C11 | Practical: `power_law_cov` is diagonal in the standard basis, so `odeco_teacher(..., basis="random")` is **misaligned** with it; use `basis="standard"` (or rotate $\Sigma_i$ into the teacher bases) for aligned runs. `CPTeacher.component_overlaps` (current version) returns $\langle W,MU_r\rangle/\langle W^*,MU_r\rangle$, which equals $P_r/s_r$ for every $W$ when $\Sigma$ is aligned; with misaligned $\Sigma$ it mixes components, so use `site_metrics=None` for the Frobenius overlap $P_r/s_r$. | checked |

## 1. Setting and notation

- Sites $i=0..N-1$ (physical dimension $d_i=d$), cores $v$ ($K$ of them), $I_v$ = the sites carried by core $v$
  (empty for internal cores of `binary` and `ht` trees). Parameters $\theta=(G_v)_v$; fan-in init
  $G_v=\sigma Z_v/\sqrt{F_v}$ (`notes/model.md` §3).
- Inputs: $x^{(i)}\sim\mathcal N(0,\Sigma_i)$ independent across sites (sections 2–6), $\Sigma_i\succ0$;
  $\Phi(x)=x^{(1)}\otimes\dots\otimes x^{(N)}$; $\Sigma:=\otimes_i\Sigma_i$; $S:=\otimes_i\Sigma_i^{1/2}$.
- Loss: $L(\theta)=\tfrac12\lVert W(\theta)-W^*\rVert_\Sigma^2$ with $\lVert A\rVert_\Sigma^2=\langle A,\Sigma A\rangle$.
- Three tensors built from the teacher: $W^*$ (the truth), $\widetilde W^*:=SW^*$ (the **whitened teacher**: it fixes
  the landscape), $T:=\Sigma W^*=\mathbb E[f^*(x)\Phi(x)]$ (the **escape tensor** or cross-moment: it fixes the
  early dynamics). $g(\theta):=\langle W(\theta),T\rangle$.
- CP teacher $W^*=\sum_rs_rU_r$, $U_r=\otimes_iu_r^{(i)}$, $\lVert u_r^{(i)}\rVert=1$; odeco when the $u_r^{(i)}$ are
  orthonormal over $r$ at every site. **Aligned covariance**: $\Sigma_iu_r^{(i)}=\lambda_{r,i}u_r^{(i)}$. Then
  $\lambda_{v,r}:=\prod_{i\in I_v}\lambda_{r,i}$ (1 for cores without physical legs) and
  $\Lambda_r:=\prod_i\lambda_{r,i}=\prod_v\lambda_{v,r}$.
- $P_r:=\langle W,U_r\rangle$ (Frobenius overlap); $a_r=P_r/s_r$ is the learned fraction of `notes/diagnostics.md`.
- For a learned saddle and a new component: $q$, $r$, $D$, $m$, $k$, route, piggyback as in
  `notes/theory_saddles.md` §1.

## 2. Product covariance = whitened problem with a preconditioned flow (part c, general statement)

**Proposition 1 (proven).** Let $A$ act on each core by $(A\theta)_v=G_v\times_{i\in I_v}\Sigma_i^{1/2}$ (multiply
the physical leg of site $i$ by $\Sigma_i^{1/2}$; bond legs and cores without physical legs untouched). $A$ is
linear, symmetric and invertible. Then:

1. $W(A\theta)=S\,W(\theta)$ (a leg-wise linear map on an open leg commutes with the contraction), hence
   $L(\theta)=\widetilde L(A\theta)$ with $\widetilde L(\tilde\theta)=\tfrac12\lVert W(\tilde\theta)-SW^*\rVert_F^2$.
2. With $\tilde\theta=A\theta$, gradient flow $\dot\theta=-\nabla L(\theta)$ becomes
   $$\dot{\tilde\theta}=-P\,\nabla\widetilde L(\tilde\theta),\qquad P=AA^\top=\text{“}\Sigma_i\text{ on every physical leg”},$$
   from the anisotropic init $\tilde\theta_0=A\theta_0$ (core $v$: Gaussian with covariance
   $\tfrac{\sigma^2}{F_v}\otimes_{i\in I_v}\Sigma_i$ on its physical legs).
3. $\nabla L(\theta)=A\nabla\widetilde L(A\theta)$ and $\nabla^2L(\theta)=A\,\nabla^2\widetilde L(A\theta)\,A$ **at every
   $\theta$** (chain rule through a linear map; checked at a non-critical point to $2\cdot10^{-15}$ by the verifier).
   So critical points correspond, critical values coincide, and at critical points Morse indices and the
   strict/non-strict type of every saddle are those of the whitened problem with teacher $SW^*$ (Sylvester's law
   of inertia). The unstable rates are eigenvalues of $P\nabla^2\widetilde L$, **not** of $\nabla^2\widetilde L$.
4. The gauge group acts on bond legs only, so $L$ is gauge invariant for any input law and the conserved
   quantities $Q_v=G_v^\top G_v-G_{p,(v)}G_{p,(v)}^\top$ of `notes/model.md` §5 hold unchanged in $\theta$. In
   whitened coordinates they read $\tilde G_v^\top(\otimes_{i\in I_v}\Sigma_i^{-1})\tilde G_v-\dots$, i.e.
   small init is balanced in $\theta$, not in $\tilde\theta$. (This refines the remark in `notes/literature.md`, H5
   paragraph, that the conservation laws "are not preserved": they are preserved in the original coordinates.)
5. Plain gradient flow on the whitened problem is recovered iff $P\propto I$. In whitened coordinates $P$ acts on
   core $v$ as $\otimes_{i\in I_v}\Sigma_i$, so $P\propto I$ iff every $\Sigma_i=\kappa_iI$ **and**
   $\prod_{i\in I_v}\kappa_i$ is the same for every core. Cores without physical legs have factor 1, so on
   `binary` and `ht` trees this forces $\prod_{i\in I_v}\kappa_i=1$; on MPS it holds for $\kappa_i\equiv\kappa$.
   (Counterexample, verifier: $\kappa_i=2$ gives per-core factors $(4,4,1)$ on binary4 and $(2,2,1,2,2,1,1)$ on
   HT4.) A different, always-true statement: for $\Sigma_i=\kappa_iI$, $L(\theta)=(\prod_i\kappa_i)\,L_{\Sigma=I}(\theta)$
   with the same teacher, so the $\theta$-flow is the isotropic flow with time multiplied by $\prod_i\kappa_i$.
   Plain whitened GF is also recovered if one trains $\theta$ with the per-core preconditioner $\Sigma_i^{-1}$ on
   physical legs. If some $\Sigma_i$ is singular, the components of the physical legs in $\ker\Sigma_i$ receive no
   gradient and stay at their init values; the rest is the whitened problem on the range.

*So the answer to (c), "is it a whitened problem with a reparameterized student?", is yes, always, but only up to
the metric:* the dynamics is the whitened one with teacher $SW^*$, whitened init, and the metric changed on the
cores with physical legs. The landscape statement of `notes/theory_teachers.md` §1 ("reduces to this case by
$T\to\Sigma^{1/2}T$, $W\to\Sigma^{1/2}W$") is correct for ranks, critical points and saddle types; it is not a
statement about times or about which saddles are visited.

Checked (N5): starting from $\tilde\theta_0=A\theta_0$, the preconditioned whitened flow reproduces $SW(\theta(t))$
to $5.7\cdot10^{-9}$ (binary $N=4$), $2.0\cdot10^{-8}$ (MPS $N=4$) and $2.4\cdot10^{-10}$ (HT $N=4$, $K=7$, internal cores
without physical legs) for $t\le200$; the naive flow (same teacher and init, no $P$) deviates by up to 8%, 4% and
100% (HT: it had not escaped at $t=200$ when the true flow had converged). The verifier reproduced this
independently ($\le3\cdot10^{-9}$; naive flow off by 14%, 63%, 53%; $\lvert\Delta Q\rvert\le1.4\cdot10^{-12}$).

## 3. Aligned covariance and odeco teacher: exact solution (part a)

**Aligned family** $\mathcal F$ (as `ttn.network.aligned_init`): $G_v=\sum_ra_{v,r}E_{v,r}$, where $E_{v,r}$ (unit
Frobenius norm) has $u_r^{(i)}$ on the physical legs of $v$ and the basis vector $e_r$ on every bond leg ($e_1$ on
the root's output leg). Requires bond dimensions $\ge R$. On $\mathcal F$, $W=\sum_rP_rU_r$ with
$P_r=\prod_va_{v,r}$.

**Proposition 2 (proven).**

1. On $\mathcal F$: $L=\tfrac12\sum_r\Lambda_r(P_r-s_r)^2$ (channels with $s_r=0$ included), because
   $\langle U_r,\Sigma U_{r'}\rangle=\delta_{rr'}\Lambda_r$.
2. $\mathcal F$ is invariant: $\Sigma(W-W^*)=\sum_r\Lambda_r(P_r-s_r)U_r$, and contracting $U_r$ with cores that are
   diagonal in the channel basis leaves only channel $r$. The amplitudes obey
   $$\dot a_{v,r}=\Lambda_r\,(s_r-P_r)\,\prod_{w\ne v}a_{w,r}=\Lambda_r(s_r-P_r)P_r/a_{v,r}.$$
   $a_{v,r}^2-a_{w,r}^2$ is conserved; balanced ($a_{v,r}=a_r$): $\dot a_r=\Lambda_ra_r^{K-1}(s_r-a_r^K)$,
   $\dot P_r=K\Lambda_rP_r^{2-2/K}(s_r-P_r)$.
3. **Exact time map**: each component evolves as in the whitened problem with its own clock,
   $$P_r(t;\Sigma)=P_r^{(I)}(\Lambda_rt).$$
   The factor is $\Lambda_r=\prod_i\lambda_{r,i}$, the same for every core, whether or not it has physical legs.
4. **Half-time** (balanced, $a_{v,r}(0)=a_0$, $K\ge3$), from
   $t=\Lambda_r^{-1}s_r^{-(2-2/K)}\int_{b_0}^{2^{-1/K}}\frac{db}{b^{K-1}(1-b^K)}$, $b_0=a_0s_r^{-1/K}$:
   $$t_{1/2,r}=\frac{a_0^{-(K-2)}}{(K-2)\Lambda_rs_r}-\frac{C_K}{\Lambda_rs_r^{2-2/K}}+O(a_0^2),\qquad C_K=\frac{2^{(K-2)/K}}{K-2}-\int_0^{2^{-1/K}}\frac{b\,db}{1-b^K},$$
   $C_3=0.8503$, $C_4=0.2664$, $C_5=0.0471$, $C_6=-0.0720$, $C_7=-0.1478$, $C_8=-0.2006$. For $K=2$:
   $t_{1/2}=\ln(s_r/a_0^2)/(2\Lambda_rs_r)+O(a_0^2)$. This is the Saxe-type solution with $s_r\to\Lambda_rs_r$ as the
   rate and $s_r$ as the target; the depth-2 linear-network version is [Gidel2019] (per-mode
   $\dot w=w(\sigma_i-\lambda_iw^2)$, learning times $1/\sigma_i$, $\sigma_i$ = singular values of the cross-covariance).
5. **Plateau levels**: with the set $S$ learned, $L_S=\tfrac12\sum_{r\notin S}\Lambda_rs_r^2$; each event lowers the
   loss by $\tfrac12\Lambda_rs_r^2$; $L(0)\simeq\tfrac12\sum_r\Lambda_rs_r^2$.
6. **Time between events** $q$ then $r$ (aligned init): $\Delta t=\big(1-\tfrac{\Lambda_rs_r}{\Lambda_qs_q}\big)/\big((K-2)\Lambda_rs_ra_0^{K-2}\big)$ (the
   head start of `notes/theory_saddles.md` R4 with $s\to\Lambda s$).

**When can covariance reverse the order (derived).** Three orderings coexist: by escape speed $\Lambda_rs_r$, by
loss drop $\Lambda_rs_r^2$ (the singular values of the whitened teacher are $\sqrt{\Lambda_r}s_r$), and the whitened
one by $s_r$. With $s_1>s_2$, aligned init learns 2 first iff $\Lambda_2s_2>\Lambda_1s_1$ to leading order in $a_0$
(exactly: iff $t_{1/2,2}<t_{1/2,1}$ in item 4). Cases:

| $\Lambda_2/\Lambda_1$ | first learned | first loss drop vs second |
|---|---|---|
| $<s_1/s_2$ | 1 (whitened order) | larger |
| $(s_1/s_2,\ (s_1/s_2)^2)$ — **mismatch window** | 2 | **smaller** (non-greedy) |
| $>(s_1/s_2)^2$ | 2 | larger |

In the window the visited saddle is not the reduced-rank-regression optimum: the first plateau sits at
$\tfrac12\Lambda_1s_1^2$, higher than $\tfrac12\Lambda_2s_2^2$. [Gidel2019] state that the visited points are the
reduced-rank solutions; in their notation that requires the order of $\sigma_i$ to coincide with the order of
$\sigma_i^2/\lambda_i$, which the window violates. The verifier checked the PDF: their Corollary 2 states the
claim with no ordering condition, and a search found no explicit statement of the window.
With power-law site spectra $\lambda_k\propto k^{-\alpha}$ and component $r$ on eigenvector $k_r$ at every site,
$\Lambda_2/\Lambda_1=(k_1/k_2)^{\alpha N}$: the natural pairing ($k_r=r$) reinforces the whitened order; a reversal
needs the stronger component on lower-variance directions, $(k_1/k_2)^{\alpha N}>s_1/s_2$.

Checked (N1, N1b): loss formula to $7\cdot10^{-16}$, ODE to $3\cdot10^{-15}$, off-family gradient $7\cdot10^{-16}$;
half-times on binary $N=4$ and MPS $N=4$ ($a_0=0.05$, $\Lambda=(0.41,2.07)$, $s=(1,0.6)$) to $\le1.2\cdot10^{-4}$, order
reversed as predicted; time map to $10^{-6}$–$10^{-5}$ (integrator accuracy).

## 4. Escape from the origin with random fan-in init (part b)

**Expansion (derived).** $L(\theta)=L_0-g(\theta)+\tfrac12\lVert W(\theta)\rVert_\Sigma^2$ with $g=\langle W(\theta),T\rangle$,
$T=\Sigma W^*$. Since $g=O(\lvert\theta\rvert^K)$ and the last term is $O(\lvert\theta\rvert^{2K})$, the flow is
$\dot\theta=\nabla g+O(\lvert\theta\rvert^{2K-1})$ until $\lVert W\rVert$ is a fixed small fraction of $\lVert T\rVert$. The
fan-in init does not depend on $\Sigma$ (in $\theta$-coordinates; with `power_law_cov`, normalized to trace $d$,
even $\mathbb E f_0^2=\sigma^{2K}\prod_i\mathrm{tr}\Sigma_i/d=\sigma^{2K}$ is unchanged). Hence:

- **Substitution rule.** Every statement of the whitened escape theory (direction converges to a local maximum
  of $g$ on the sphere, escape time, race between components, the [Jacot2021]/[KumarHaupt2025a] picture) holds with
  $W^*$ replaced by $T$. Two problems with the same $T$ have, for the same seed, the same escape to leading order.
  This is the TTN form of "the first escape follows the covariance-weighted teacher" ([Jacot2021] Thm 2,
  [Atanasov2022], [RawalDeWeese2026]: rate $\lVert E[yx]\rVert$).
- **Same $T$, different saturation (derived).** The quadratic term differs between problems with equal $T$, so
  they separate when $W$ saturates. For aligned components the saturation ODEs are
  $\dot P=K\Lambda P^{2-2/K}(s-P)$ for $(\Sigma,W^*)$ (problem A) and $\dot P=KP^{2-2/K}(\Lambda s-P)$ for $(I,\Sigma W^*)$
  (problem B), started from the same small $P$. Prop. 2.4 applied to both then gives, for the times at which each
  overlap reaches half of its own target ($s_r$ vs $\Lambda_rs_r$),
  $$t^{A}_{1/2}-t^{B}_{1/2}=C_K\,s_r^{-(2-2/K)}\big(\Lambda_r^{-(2-2/K)}-\Lambda_r^{-1}\big)+O(a_0^2),$$
  an $O(1)$ absolute offset, i.e. relative $O(\sigma^{K-2})$. At a fixed small absolute level $P_*$ the offset is
  $t^A(P_*)-t^B(P_*)=(\Lambda_r-1)P_*^{2/K}/(2\Lambda_r^2s_r^2)+O(P_*^{4/K})$ (integrate the difference of $dt/dP$).
- **Maxima of $g$ on the sphere.** On the bond-rank-1 family $W=\bigotimes_vw_v$ ($w_v$ a tensor on the sites of
  core $v$; scalars for cores without physical legs) with $\sum_v\lVert G_v\rVert^2=1$, $g\le K^{-K/2}\tau_1(T)$,
  $\tau_1(T):=\max\{\langle\otimes_vw_v,T\rangle:\lVert w_v\rVert=1\}$ (the injective norm of $T$ for the grouping of
  sites into cores; binary $N=4$: $\sigma_1(\mathrm{mat}_{01\mid23}T)$; MPS and HT: product states). For aligned odeco
  $T=\sum_r\Lambda_rs_rU_r$: critical directions $U_r$ with values $K^{-K/2}\Lambda_rs_r$.
- **Effective init amplitudes.** In $\theta$ they are those of the whitened problem: the projections of
  $\theta_0$ on the escape directions of $g_T$ (fan-in Gaussian cores are invariant under orthogonal maps of every
  leg, so the law of the projection on a fixed direction does not depend on which direction $T$ selects). In
  whitened coordinates the init amplitude of component $r$ in core $v$ is multiplied by $\lambda_{v,r}^{1/2}$ and
  the preconditioner multiplies its rate by $\lambda_{v,r}$; together with $\tilde s_r=\Lambda_r^{1/2}s_r$ these
  reproduce exactly the factor $\Lambda_r$ (Prop. 2.3).
- **Cores without physical legs.** They are special only in the whitened bookkeeping (no preconditioning, isotropic
  init). In $\theta$ nothing distinguishes them: $\Sigma$ enters only through $T$ and the quadratic term, both
  tensors on the physical index space. So binary vs MPS differ under covariance only through $K$ and the tree's
  own (whitened) escape statistics. How the eigenvalues are split over sites and cores is invisible to the
  escape, since $T$ depends only on $\Lambda_r$.
- **Pitfall.** Treating the problem as plain GF on the whitened teacher (weights $\Lambda_r^{1/2}s_r$) predicts the
  wrong $\Lambda$-exponent ($-1/2$ with isotropic init; split-dependent values with whitened init). The correct
  exponent is $-1$.

**Prediction (aligned, a single or dominant component $r^*$):** $t_1\propto1/(\Lambda_{r^*}s_{r^*})$, i.e.
$t_1\Lambda_{r^*}$ is constant per seed. For a race between components, the whitened race probabilities apply at
rate ratio $\Lambda_2s_2/(\Lambda_1s_1)$; the order stays random (H1), covariance only reweights the rates.

Checked (N2, N3, N16): single component, $\Lambda\in\{0.5,1,2,4,8\}$, binary ($\sigma=0.01$) and MPS ($\sigma=0.05$),
3 seeds, three splits (all on site 0, spread evenly, all on site 3, the MPS root core): fitted slope
$d\log t_{1/2}/d\log\Lambda=-1.0000$ in all 18 fits; within a seed $t_{1/2}\Lambda$ varies by $<5\cdot10^{-5}$ over the
16-fold range of $\Lambda$; splits agree to $2\cdot10^{-5}$ (verifier: $-1.0000$ binary/MPS, $-1.0005$ to $-1.0011$ HT at
$\sigma=0.3$; naive whitening $-0.50$). Same-$T$ runs (binary4 and MPS4, window setting $s=(1,0.5)$,
$\Lambda=(1,2^{3/2})$, $d=\chi=3$, seeds 31, 0, 1): same winner in every run; $t^A_{1/2}-t^B_{1/2}=-0.2218,-0.2206,-0.2222,-0.2218$
(binary, $\sigma=0.003,0.03$) vs predicted $-0.2219$, and $-0.1075,-0.1045,-0.1051$ (MPS, $\sigma=0.03$; $-0.104,-0.083,-0.084$
at $\sigma=0.1$, where $O(a_0^2)$ matters) vs $-0.1080$; at $P_*=0.02$: $+0.034$–$0.035$ (binary) and $+0.066$–$0.069$ (MPS) vs
$0.0337$ and $0.0646$. (The v1 numbers "628.7/628.6" etc. were fixed-level times, $P_*=0.02$.)
MPS seed 1 learned component 1 first although $\Lambda_2s_2/(\Lambda_1s_1)=3$: the race is random, as H1 says.

## 5. After the first escape: where the per-site distribution matters

**Routes (derived; `notes/theory_saddles.md` R1–R4 with $R\to\Sigma R$).** At a learned saddle $\theta_S$ the
expansion of `notes/theory_saddles.md` §2 holds with the residual replaced by $R_\Sigma=\Sigma(W^*-W_S)$ and the
quadratic term by $\lVert\cdot\rVert_\Sigma^2$. Routes through **new** bond indices change $W$ only at order $k$ in the
route amplitudes, so they carry no curvature term: for aligned covariance every add-route rate is the whitened one
with $s_r\to\Lambda_rs_r$ (e.g. $k=2$: $\lambda=\Lambda_rs_rs_q^{(K-2)/K}$, confirmed on the exact Hessian by the
verifier). The learned amplitudes are unchanged, $a=s_q^{1/K}$, because the saddle is at $P_q=s_q$ in $\theta$.

**Piggyback block with covariance (proven at the saddle; extends `notes/theory_saddles.md` R6).** Old-index
directions do change $W$ at first order, so the metric enters through partial products. Take one learned
component $q$ (balanced, core norms $a=s_q^{1/K}$), a component $r$ that differs from $q$ on the sites $D$, and
$m=2$ terminal cores $c_1,c_2$. Rotating the physical vectors of $c_1$ (angle $\varphi$) and $c_2$ (angle $\psi$) from
$q$ to $r$ gives, with
$$\mu_x=\prod_{i\in D\cap I_{c_1}}\lambda_{x,i},\qquad\nu_x=\prod_{i\in D\cap I_{c_2}}\lambda_{x,i}\ (x\in\{q,r\}),\qquad\lambda_{\rm sh}=\prod_{i\notin D}\lambda_{q,i},$$
the Hessian per unit parameter displacement
$$H_{\rm pb}=\frac{\lambda_{\rm sh}}{s_q^{2/K}}\begin{pmatrix}s_q^2\mu_r\nu_q&-s_rs_q\mu_r\nu_r\\-s_rs_q\mu_r\nu_r&s_q^2\mu_q\nu_r\end{pmatrix},\qquad\det\propto s_q^2\mu_r\nu_r\,(\Lambda_qs_q^2-\Lambda_rs_r^2)/\lambda_{\rm sh}.$$
(Derivation: $L=\tfrac12P^2\lambda_{\rm sh}(\cos^2\!\varphi\,\mu_q+\sin^2\!\varphi\,\mu_r)(\cos^2\!\psi\,\nu_q+\sin^2\!\psi\,\nu_r)-P(\Lambda_qs_q\cos\varphi\cos\psi+\Lambda_rs_r\sin\varphi\sin\psi)$
at $P=s_q$; $\varphi,\psi$ decouple from $P$ at second order.) At $\Sigma=I$ the eigenvalues are
$s_q^{(K-2)/K}(s_q\mp s_r)$, as in `notes/theory_saddles.md` R6. Consequences:

- The saddle is strict iff $\Lambda_rs_r^2>\Lambda_qs_q^2$, a split-independent criterion (as it must be: Morse
  indices are metric-invariant, Prop. 1.3).
- The rate $\lambda_{\rm pb}=-\lambda_{\min}(H_{\rm pb})$ depends on the split through
  $\mathrm{tr}\propto\mu_r\nu_q+\mu_q\nu_r$; for $\lvert\det\rvert\ll\mathrm{tr}^2$ ($\lambda_{\rm sh}$ cancels),
  $\lambda_{\rm pb}\approx s_q^{-2/K}\mu_r\nu_r(\Lambda_rs_r^2-\Lambda_qs_q^2)/(\mu_r\nu_q+\mu_q\nu_r)$
  (0.0557 vs exact 0.0548 for the lopsided case below; 0.138 vs 0.126 for the even split).
  Putting the variance of the fast component on one terminal core slows the rotation.
- **Adjacent terminal cores ($k=2$): no rotation (proven at the saddle).** Since $r$ and $q$ agree off $D$ and
  $\Sigma$ is aligned, $\lambda_{\rm sh}\mu_r\nu_r=\Lambda_r$, so the off-diagonal entry of $H_{\rm pb}$ equals the
  $k=2$ add-route rate $b:=\Lambda_rs_rs_q^{(K-2)/K}$. A symmetric $2\times2$ matrix with positive diagonal
  entries $h_1,h_2$ has $\lambda_{\min}>-b$, because $(\tfrac{h_1+h_2}2+b)^2-(\tfrac{h_1-h_2}2)^2-b^2=h_1h_2+b(h_1+h_2)>0$.
  So $\lambda_{\rm pb}<b$: with comparable seeds the add route wins, $r$ is added directly and $q$ is kept
  (`notes/theory_saddles.md` R7, case 2). Rotation therefore needs $m=2$ **and** $k\ge3$.

**Tree-dependent consequence of the mismatch window (derived).** Let 2 be the faster component and 1 the larger
loss drop.

*Which component wins the first escape (derived for the local structure, heuristic for probabilities;
`notes/theory_saddles.md` §4.5 with $s\to\Lambda s$).* With aligned init 2 always wins. With random init the race is
random. For a matrix-like pair ($m=2$ or $k=2$, e.g. binary4 odeco) the slower component 1 is not a local
maximum of the spherical escape flow, so 2 wins with probability $\to1$ as $\sigma\to0$ (heuristic failure rate
$\propto\sigma^{(\Lambda_2s_2-\Lambda_1s_1)/(\Lambda_1s_1)}=\sigma^{0.41}$ for the example below). For a tensor-like pair
($m\ge3$ and $k\ge3$: MPS4, binary8, HT with odeco teachers) the probability does not depend on $\sigma$. Checked
(N17; $s=(1,0.5)$, $\Lambda=(1,2^{3/2})$, aligned diagonal $\Sigma_i$, $d=\chi=3$): binary4, component 1 first in
0/32, 0/32, 0/32, 1/32, 2/32 seeds at $\sigma=0.003,0.03,0.1,0.2,0.3$; MPS4 in 5/24 seeds at both $\sigma=0.1$ and 0.05,
the same seeds both times (on a tensor-like pair the init direction alone decides).

*Conditional on 2 winning:*

- $m=2$ and $k\ge3$ for the pair (binary $N=4$ with an odeco teacher; pattern teachers whose components differ on
  two **non-adjacent** terminal cores): the saddle is strict, so after $\Delta t\approx\ln(1/\varepsilon)/\lambda_{\rm pb}$
  (with $\varepsilon$ the piggyback amplitude left over from the escape) the learned rank-one term rotates from $U_2$
  to $U_1$ (bond rank stays 1). The loss plateaus at $\tfrac12\Lambda_2s_2^2$, and component 2 is relearned later
  through its add-route ($\sigma^{-(k-2)}$). This is the H1b correction, here forced by covariance rather than by
  a lost race.
- $m\ge3$ (MPS $N=4$, binary $N=8$, HT with odeco teachers): the piggyback block is positive definite, so there
  is no fast correction. The loss plateaus at the higher level $\tfrac12\Lambda_1s_1^2$ until component 1 is added.

*Conditional on 1 winning* (any tree): the plateau is at $\tfrac12\Lambda_2s_2^2$. The piggyback block at this
saddle is positive definite, because in the window $\Lambda_2s_2^2<\Lambda_1s_1^2$. So there is no rotation, and 2 is
added through its own route ($\sigma^{-(k-2)}$ for $k\ge3$).

Checked (N3, N4), $s=(1,0.5)$, $\Lambda=(1,2\sqrt2)$ (window), binary $N=4$ ($\sigma=0.003$, seeds 0, 1): 2 learned at
630.7/430.4, then 1 up and 2 down at 669.0/671.5 and 449.1/451.8 with bond rank 1 throughout, plateau
$0.3536=\tfrac12\Lambda_2s_2^2$, 2 relearned at 1202/1115 (verifier, seeds 31, 32: same sequence, second singular value
$\le1.5\cdot10^{-8}$ during the rotation). MPS $N=4$ ($\sigma=0.03$, seed 0): 2 at 1179, plateau
$0.5000=\tfrac12\Lambda_1s_1^2$, 1 at 2394, no rotation. Outside the window ($\Lambda=(1,6)$): binary plateau
$0.5000$ with no rotation. Rotation rates measured from the exponential growth of the mixed overlaps
$\langle W,x_1\otimes y_2\rangle$, $\langle W,x_2\otimes y_1\rangle$: uniform split 0.1263 vs predicted 0.1263;
$(\mu_2,\nu_2)=(2.83,1)$: 0.1131 vs 0.1130; $(\mu_2,\nu_2)=(8,0.354)$: 0.0548 vs 0.0548, eigenvector ratio
$\psi/\varphi=0.246$ vs 0.246. Same $\Lambda$ in all three, so the split changes the rate by 2.3x.

## 6. Misaligned covariance (part c)

Prop. 1 holds unchanged. What is new is that $W^*$, $SW^*$ and $\Sigma W^*$ have different components.

**CP teachers.** For every CP teacher, exactly (proven),

$$T=\sum_rs_r\Lambda^{\rm esc}_r\,\widetilde U_r,\quad\widetilde U_r=\bigotimes_i\frac{\Sigma_iu_r^{(i)}}{\lVert\Sigma_iu_r^{(i)}\rVert},\quad\Lambda^{\rm esc}_r=\prod_i\lVert\Sigma_iu_r^{(i)}\rVert;\qquad SW^*=\sum_rs_r\sqrt{\Lambda^{\rm loss}_r}\,\widehat U_r,\quad\Lambda^{\rm loss}_r=\prod_i\langle u_r^{(i)},\Sigma_iu_r^{(i)}\rangle,$$

with $\widehat U_r=\bigotimes_i\Sigma_i^{1/2}u_r^{(i)}/\lVert\Sigma_i^{1/2}u_r^{(i)}\rVert$. What needs the cross overlaps
$\langle u_r^{(i)},\Sigma_iu_{r'}^{(i)}\rangle$ and $\langle\Sigma_iu_r^{(i)},\Sigma_iu_{r'}^{(i)}\rangle$ ($r\ne r'$) to be small (derived to
leading order in them, exact when they vanish) is the orthogonality of the $\widetilde U_r$ (resp. $\widehat U_r$), and
hence the escape ordering and the plateau formulas below.

- **Inequality (proven):** $\Lambda^{\rm esc}_r\ge\Lambda^{\rm loss}_r$ (Cauchy–Schwarz,
  $\langle u,\Sigma u\rangle\le\lVert\Sigma u\rVert$ for $\lVert u\rVert=1$), with equality iff every $u_r^{(i)}$ is an eigenvector
  of $\Sigma_i$. Misalignment always speeds up the escape relative to the loss weight, by the factor
  $\prod_i1/\cos\angle(u_r^{(i)},\Sigma_iu_r^{(i)})$.
- Escape order by $s_r\Lambda^{\rm esc}_r$; loss drop of component $r$ is $\tfrac12\Lambda^{\rm loss}_rs_r^2$. Misalignment
  alone can reverse the order, with equal Rayleigh quotients.
- **Escape direction vs plateau.** The escaping component points along $\widetilde U_r$ (factors $\Sigma_iu$, the
  covariance-weighted teacher), then realigns to the plateau critical point near $U_r$ during saturation, with no
  extra plateau. This is the TTN version of "early alignment to $\Sigma\beta$, then realignment" [Atanasov2022].
  Overlap at small $\lVert W\rVert$: $\langle W,U_r\rangle/\lVert W\rVert=\prod_i\langle u,\Sigma_iu\rangle/\lVert\Sigma_iu\rVert$.
- **Binary $N=4$ is a depth-3 linear network** in the cut $01\mid23$, with input metric $\Sigma_2\otimes\Sigma_3$ and output
  metric $\Sigma_0\otimes\Sigma_1$. Its long plateaus are the truncated SVDs of $\mathrm{mat}_{01\mid23}(SW^*)$:
  $L_k=\tfrac12\sum_{j>k}\sigma_j(\mathrm{mat}_{01\mid23}SW^*)^2$ ([Achour2024] Thm 7 after absorbing the metrics,
  Eckart–Young). The first escape direction is the top singular pair of $\mathrm{mat}_{01\mid23}(\Sigma W^*)$, a different
  matrix. For deeper trees the bond-rank-1 problem has several local minima (one near each component), and the
  basin of the escape direction decides which one is visited.
- **Misalignment can open low-degree routes (derived).** For an add-route on core set $C$, the coupling to the
  residual contains, for every site $i$ outside $C$, the factor $\langle u_q^{(i)},\Sigma_iu_r^{(i)}\rangle$, which
  vanishes when aligned. So under misalignment, routes of degree lower than the whitened $k^*$ exist, with strength
  $O(\varepsilon^{\#\text{sites outside }C})$ at per-site misalignment $\varepsilon$: negligible for mild misalignment
  and $N\ge4$, but they change the $\sigma\to0$ asymptotics of later plateaus (a lower power of $1/\sigma$, or
  $\log(1/\sigma)$ when a 2-core route opens). Section 7.4 works out the same mechanism exactly for cross-site
  correlations, where each outside site contributes a factor $c$.

Checked (N6), "misalignment-only reversal": per site $\Sigma_i=Q_i\,\mathrm{diag}(1.8,0.2,1,1)Q_i^\top$; component 1 ($s=1$)
on the eigenvector with eigenvalue 1, component 2 ($s=0.8$) on $(q_1+q_2)/\sqrt2$, with $q_1,q_2$ the eigenvectors with
eigenvalues 1.8 and 0.2 (Rayleigh quotient 1). So
$\Lambda^{\rm loss}=(1,1)$, $\Lambda^{\rm esc}=(1,2.69)$, escape rates $(1,2.15)$, loss weights $(1,0.64)$ (in the window).
Binary $N=4$ (seeds 0, 1): 2 escapes first (364.9/411.3) with $\cos(W,\widetilde U_2)=1.000$ and $\cos(W,U_2)=0.379/0.380$
(predicted 0.372) at $\lVert W\rVert=0.05$; rotation to 1, plateau $0.3200$ (predicted 0.32), 2 relearned (758.9/670.3).
Same-$T$ escape 356.7/356.6 and 403.1/403.0 (fixed level); whitened control learns 1 first (966.5/830.6). MPS
$N=4$ (seeds 0, 1): 2 first (713.6/2203.3), $\cos(W,\widetilde U_2)=0.997/0.998$, plateau 0.500 with no rotation. The
verifier's independent build ($d=3$): $\cos(W,U_2)=0.3723/0.3714$ at $\lVert W\rVert\approx0.006$ (predicted 0.3718).

## 7. Cross-site correlations (part d)

### 7.1 Model

$x^{(i)}=\sqrt{1-c}\,z_i+\sqrt c\,z_0$ with $z_0,z_1,\dots,z_N\sim\mathcal N(0,I_d)$ iid, $0\le c<1$. Every marginal is
$\mathcal N(0,I_d)$ and $\mathrm{Cov}(x^{(i)},x^{(j)})=cI_d$ for $i\ne j$. This is the standard one-factor
(equicorrelated) Gaussian model, coordinate-wise the one-factor Gaussian copula ([Vasicek2002], [Li2000]). Only its
use to contract the TTN loss along the tree is new here. The loss is $L=\tfrac12\langle\Delta W,M\Delta W\rangle$,
$\Delta W=W-W^*$, $M=\mathbb E[\Phi\Phi^\top]$. Notation for this section: site-aligned components (component $r$
uses the basis vector $e_r$ at every site, `basis="standard"`, $r=1..R$), coordinates $a=1..d$,
$x_a:=e_a\otimes e_a$ (a 2-site pairing vector), $\zeta\sim\mathcal N(0,1)$ a scalar, and
$$m_{p,q}(c):=\mathbb E\big[\zeta^p((1-c)+c\zeta^2)^q\big]=\sum_{j=0}^q\binom qj(1-c)^{q-j}c^j(p+2j-1)!!,\qquad(-1)!!=1.$$

### 7.2 Exact contraction

**Isserlis (proven).** $M_{(a),(b)}=\sum_\pi\prod_{\{p,p'\}\in\pi}\delta_{\alpha_p\alpha_{p'}}C_{i_pi_{p'}}$, summed over the
$(2N-1)!!$ pairings $\pi$ of the $2N$ slots $p$ (slot = (site $i_p$, coordinate $\alpha_p$), $N$ slots from $a$ and $N$
from $b$), with $C_{ii}=1$ and $C_{ij}=c$. Pairs inside the same copy pair two different sites ($a_i$ with $a_j$):
those are the non-product, tree-nonlocal terms.

**Conditional product structure, exact contraction (proven).** Given $z_0$ the sites are independent with second
moments $M_i(z_0)=(1-c)I+cz_0z_0^\top$, so
$$M=\mathbb E_{z_0}\Big[\bigotimes_iM_i(z_0)\Big],\qquad L(\theta)=\mathbb E_{z_0}\Big[\tfrac12\lVert W(\theta)-W^*\rVert^2_{\otimes_iM_i(z_0)}\Big].$$
Each term is one call of `population_loss(..., site_metrics=[M_i(z_0)]*N)`. The integrand is a polynomial of
degree $2N$ in $z_0$, so tensor Gauss–Hermite with $N+1$ nodes per dimension is exact: $Q=(N+1)^d$ product-metric
tree contractions (any cubature of degree $2N$ works; at least $\binom{N+d}{d}$ nodes). The same holds for any
Gaussian with covariance $D+BB^\top$ ($D$ block-diagonal per site, $B\in\mathbb R^{Nd\times k}$): latent
$z\in\mathbb R^k$, $M_i(z)=D_i+B_izz^\top B_i^\top$, $Q=(N+1)^k$. **So it can still be contracted along the tree,
at a cost that is polynomial in $N$ and $K$ but exponential in the latent dimension $k$.** For generic full-rank
cross-site covariances ($k\sim Nd$) we know no tree-local exact contraction: pairings across a bond would have to be
carried as open indices (heuristic). Cheaper unbiased estimators sample $z_0$ only and contract exactly given
$z_0$ (Rao–Blackwellized). Positivity: $M\succeq(1-c)^NI$ (each $M_i(z_0)\succeq(1-c)I$), so for $c<1$ the global
minimum is $W=W^*$, uniquely in function space (proven).

### 7.3 Small $c$, teacher weights, escape direction

**Small $c$ (proven).** The first-order term vanishes because the marginals are unchanged:
$$M=I+c^2\sum_{i<j}\big(\mathrm{SWAP}_{ij}+\lvert\Omega_{ij}\rangle\langle\Omega_{ij}\rvert\big)+O(c^3),\qquad\Omega_{ij}=\sum_ae_a^{(i)}\otimes e_a^{(j)}.$$
$\mathrm{SWAP}_{ij}$ exchanges the legs of sites $i,j$; $\lvert\Omega_{ij}\rangle\langle\Omega_{ij}\rvert$ partially traces sites $i,j$ and
re-inserts the identity pairing. All effects are $O(c^2)$ at small $c$, but the $O(c^3)$ terms are not small
relative to them (e.g. a factor $1+4c$ below).

**Teacher weights (proven).** Site-aligned components:
$$\langle U_r,MU_r\rangle=\Lambda_{\rm eff}(c,N):=m_{0,N}(c)=\sum_{k=0}^N\binom Nk(1-c)^{N-k}c^k(2k-1)!!,\qquad\langle U_r,MU_{r'}\rangle=c^N\big((N-1)!!\big)^2\ (N\text{ even}),\ 0\ (N\text{ odd}).$$
$\Lambda_{\rm eff}(c,4)=1.158,1.832,3.43,6.50,11.75,20.0$ at $c=0.1,\dots,0.6$; $\Lambda_{\rm eff}(0.3,8)=466$: the shared latent
dominates the variance of site-aligned products. With independent random site bases,
$\langle U_r,MU_r\rangle=1+2c^2\sum_{i<j}\langle u_r^{(i)},u_r^{(j)}\rangle^2+O(c^3)\approx1+c^2N(N-1)/d$.

**Escape tensor and junk.** $T=MW^*=W^*+c^2\sum_{i<j}[\mathrm{SWAP}_{ij}W^*+\Omega_{ij}\otimes\mathrm{tr}_{ij}W^*]+O(c^3)$ (derived).
Across a cut, the pairing term $\Omega_{ij}\otimes\mathrm{tr}_{ij}W^*$ has rank $\rho_\Omega\cdot\rho_{\rm tr}$: $\rho_\Omega=d$ if the cut
separates $i$ from $j$, else 1; $\rho_{\rm tr}$ is the rank of $\mathrm{tr}_{ij}W^*$ across the induced cut of the other
sites ($R$ for a site-aligned odeco teacher if that induced cut has sites on both sides, else 1). Example
($d=3$, $R=2$, checked): $\Omega_{01}\otimes\mathrm{tr}_{01}W^*$ has ranks 3, 1, 2 across $0\mid123$, $01\mid23$, $012\mid3$, and
$\Omega_{02}\otimes\mathrm{tr}_{02}W^*$ has ranks 3, 6, 2. Symmetry (proven): flipping coordinate $a$ at every site (and
in $z_0$) preserves the input law, and for even $N$ a site-aligned teacher is invariant. So $T_{a_1\dots a_N}=0$
unless every coordinate value occurs an even number of times (the same parity rule holds for $MX$, for every
site-aligned $X$). Consequences:

- One-site-per-core trees (MPS, HT): $U_r$ is an exact critical point of $g$ over product states; the first escape
  carries no junk.
- **Binary $N=4$: exact escape direction (proven).** By parity, $\mathrm{mat}_{01\mid23}(MW^*)$ is block diagonal, and
  its block on $\mathrm{span}\{x_a\}\otimes\mathrm{span}\{x_b\}$ is the symmetric $d\times d$ matrix
  $$A_{aa}=s_a\Lambda_{\rm eff}(c,4)+9c^4\sum_{r\ne a}s_r,\qquad A_{ab}=c^2m_{2,2}(c)(s_a+s_b)+3c^4\sum_{r\notin\{a,b\}}s_r\quad(a\ne b),$$
  with $s_a:=0$ for $a>R$ and $m_{2,2}(c)=1+4c+10c^2$. The other block is $O(c^2)$ (SWAP terms). So the left (=
  right) factor of the escape direction is the top eigenvector of $A$ whenever that block carries the top singular
  value, which holds for every case computed ($c\le0.3$, $d=3$). First order in the off-diagonal (derived): the
  coefficient of $x_a$ relative to $x_1$ is
  $$\frac{s_1+s_a}{s_1-s_a}\cdot\frac{c^2m_{2,2}(c)}{\Lambda_{\rm eff}(c,4)}+O(c^4)\quad(2\le a\le R),\qquad\frac{c^2m_{2,2}(c)+3c^4\sum_{r\ge2}s_r/s_1}{\Lambda_{\rm eff}(c,4)}\quad(a>R),$$
  with $\Lambda_{\rm eff}(c,4)=1+12c^2+O(c^3)$. So v1's $4c^2$ and $c^2$ (for $s=(1,0.6)$) acquire the factor
  $m_{2,2}/\Lambda_{\rm eff}=1+4c+O(c^2)$, which the verifier measured as $4.08c^2$ at $c=0.005$. The direction mixes in
  the other teacher components (amplified by small gaps) and adds pure pairing junk of norm $\approx c^2\sqrt{d-R}$.

Checked (N15, $d=3$, $s=(1,0.6)$): the $d\times d$ block reproduces the full SVD to all digits (weight outside the
block $\le3\cdot10^{-18}$). Exact $x_2$ coefficient: $4.080c^2$, $4.159c^2$, $4.742c^2$, $5.196c^2$, $4.744c^2$ at
$c=0.005,0.01,0.05,0.1,0.2$. The first-order formula is within 0.2% for $c\le0.1$ and 2% at $c=0.2$ ($x_2$), and
within 0.1%, 3%, 10% at $c=0.02$, 0.1, 0.2 ($x_3$).

### 7.4 Does S2S survive?

1. *First plateau: yes, for every $c<1$ (derived).* The origin is a degree-$K$ critical point for any input law,
   $T\neq0$, and the escape time scales as $\sigma^{-(K-2)}/\tau_1(T)$ up to the seed-dependent init projection.
2. *Saddles exist (proven):* the unused-channel subspaces (bond indices with all entries zero in both adjacent
   cores) are invariant under gradient flow for **any** loss of the form $\ell(W(\theta))$ (the gradient on an unused
   index contracts with the other core's zero entries). So bond-rank-constrained critical points are critical
   points of the full loss, now $M$-weighted.
3. *Later plateaus on MPS: not robust.* A non-product $M$ spreads the residual $M(W^*-W_S)$ over tensors that agree
   with the learned state on most sites, which opens add-routes of lower degree than the whitened $k^*$
   (sub-items a–e).
4. *Junk-rich plateaus.* The plateaus that remain are junk-rich on trees whose cores carry correlated sites
   (binary). On MPS4 the same junk-rich point is the intermediate saddle of item 3b. The junk is unlearned at the
   end ($W\to W^*$ since $M\succ0$), so bond spectra and the teacher overlaps can be non-monotone.
5. *$c\to1$: ill-conditioned from both ends.* The bound $\lambda_{\min}(M)\ge(1-c)^N$ (proven) is loose. For $d\ge N$
   the fully antisymmetric tensors are eigenvectors with eigenvalue $(1-c)^{N-1}(1+(N-1)c)$ (proven: in
   $\otimes_i((1-c)I+cz_0z_0^\top)$ only terms with at most one factor $z_0z_0^\top$ survive). That is $\lambda_{\min}$ in all
   cases computed ($N=d\le4$). For $d<N$, tensors antisymmetric within $\lceil N/d\rceil$ groups of sites suggest
   $\lambda_{\min}\propto(1-c)^{N-\lceil N/d\rceil}$ (conjecture; numerically $N=4,d=3$: 0.625, 0.044, $4.9\cdot10^{-4}$ at
   $c=0.5,0.9,0.99$, i.e. $(2.5$–$4.9)(1-c)^2$; $N=4,d=2$: $(3.3$–$7.9)(1-c)^2$). $\lambda_{\max}$ exceeds $\Lambda_{\rm eff}(c,N)$
   (182 vs 101 at $N=4$, $d=3$, $c=0.99$), so $\kappa(M)=30$, $3.0\cdot10^3$, $3.7\cdot10^5$ at
   $c=0.5,0.9,0.99$. Parts of $W^*$ that are antisymmetric under site exchanges are fitted at rates $\propto\lambda_{\min}$:
   a different, stiff regime that we did not study (heuristic).

**3a. Route rates at the product saddle (proven; checked).** Take $N$ even, MPS$_N$, a site-aligned odeco teacher,
$s_1>s_2\ge\dots$. $W_S=PU_1$ with
$$P=s_1+c^N((N-1)!!)^2\textstyle\sum_{r\ge2}s_r/\Lambda_{\rm eff}(c,N)$$
is a critical point: by parity every gradient entry with a single coordinate $\ne1$ vanishes, and $P$ is the optimal
amplitude. A 2-core route on edge $(j,j+1)$ changes $W$ by $a^{N-2}e_1^{\otimes(N-2)}\otimes(\beta\otimes\gamma)$ ($a=P^{1/N}$,
$\beta,\gamma\in\mathbb R^d$ the new-index entries of the two cores). So the Hessian on $(\beta,\gamma)$ is
$\begin{pmatrix}0&-a^{N-2}B\\-a^{N-2}B^\top&0\end{pmatrix}$ with $B_{ab}=\langle M(W^*-W_S),e_1^{\otimes(N-2)}\otimes e_a\otimes e_b\rangle$.
Conditioning on $z_0$ (each site contributes $(M_i(z_0)e_r)_b=(1-c)\delta_{br}+cz_{0,b}z_{0,r}$) and using parity gives
$B=\mathrm{diag}(0,B_2,\dots,B_d)$ with
$$B_a=s_a\,c^{N-2}(N-3)!!\,m_{N-2,2}(c)+(s_1-P)\,c^2m_{2,N-2}(c)+c^N(N-3)!!(N-1)!!\sum_{r\notin\{1,a\}}s_r\qquad(s_a:=0\text{ for }a>R).$$
The Hessian eigenvalues are $-\lambda_a$, $\lambda_a=P^{(N-2)/N}\lvert B_a\rvert$, on every edge (all edges are equivalent),
each once per new bond index. The three terms come from component $a$ itself (each of the $N-2$ sites outside the
route pays a factor $cz_{0,1}z_{0,a}$), from the learned component's own residual ($O(c^{N+2})$, since
$s_1-P=O(c^N)$), and from the other components. For $N=4$ this is the verifier's
$\lambda_a=P^{1/2}[s_ac^2((1-c)^2+6c(1-c)+15c^2)+\dots]$; v1's prefactor $\lVert W_S\rVert^{1/2}$ is the $N=4$ case of
$P^{(N-2)/N}$.

- Teacher route ($a=2$): $\lambda_2\approx s_2c^{N-2}((N-3)!!)^2$ at small $c$ ($0.6c^2$, $5.4c^4$, $135c^6$ for $N=4,6,8$),
  the $c^{N-2}$ law found by the verifier. v1's "$\approx$ const$\cdot c^2$" was the $N=4$ case only.
- Junk pairings ($a>R$): $\lambda_{\rm junk}=P^{(N-2)/N}c^N(N-1)!!\sum_{r\ge2}s_r\,\lvert(N-3)!!-(N-1)!!\,c^2m_{2,N-2}/\Lambda_{\rm eff}\rvert=O(c^N)$.
  This is the second strict direction per edge that v1 omitted (0.0025 next to 0.0525 at $c=0.2$, $N=4$, $d=3$).

| $c$ | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|
| $\lambda_2$, $N=4$ | $1.84\cdot10^{-3}$ | $9.00\cdot10^{-3}$ | 0.0525 | 0.165 | 0.393 |
| $\lambda_2$, $N=6$ | $4.94\cdot10^{-5}$ | $1.11\cdot10^{-3}$ | 0.0310 | 0.245 | 1.12 |
| $\lambda_2$, $N=8$ | $3.64\cdot10^{-6}$ | $3.63\cdot10^{-4}$ | 0.0459 | 0.872 | 7.38 |
| $\Lambda_{\rm eff}$, $N=4/6/8$ | 1.03/1.10/1.23 | 1.16/1.59/2.98 | 1.83/6.35/45.3 | 3.43/27.6/466 | 6.50/97.2/2970 |

($s=(1,0.6)$; $\lambda_2$ does not depend on $d$.) Checked (N11): the closed form, $B$ computed from the dense $M$, and
the exact Hessian of the dense-$M$ loss at the balanced saddle agree to all printed digits (5 significant) for
$(N,d)=(4,3),(6,2),(6,3),(8,2)$. The multiplicities are $N-1$ for $\chi=2$. Local log-log slopes in $c$ are 2.07–3.02
($N=4$), 4.14–5.28 ($N=6$) and 6.20–7.42 ($N=8$) over $c=0.0125$–0.4 (verifier: 2.2, 4.4, 6.5). Because the moments
$m_{N-2,2}(c)$ grow fast with $N$, the three rates are comparable at $c\approx0.2$, and at $c=0.3$ the $N=8$ rate is the
largest. The $c^{N-2}$ suppression is a small-$c$ statement.

**3b. Two-stage escape on MPS4 (derived; checked).** The 2-core routes do not reach $U_2$ by themselves.

- *Stage 1.* The routes on the two outer edges $(0\mid1)$ and $(2\mid3)$ grow at $\lambda_2$ and saturate at an
  intermediate critical point $W_I$ whose middle bond has rank 1. $W_I$ is a product of two 2-site states, one on
  sites $(0,1)$ and one on $(2,3)$: exactly binary4's $M$-weighted bond-rank-1 critical point, i.e. its junk-rich
  first plateau. $L/L_0=0.24880$ at $c=0.2$ and 0.12544 at $c=0.4$, against binary4's plateaus 0.2488 and 0.1254. At
  $c=0.2$, $W_I$ has amplitude 0.941 on $U_1$, 0.259 on each mixed tensor $x_2\otimes x_1$ and $x_1\otimes x_2$, and 0.072
  on $U_2$.
- *Stage 2.* Unused middle-bond channels are invariant (item 2), so $W_I$ is a critical point of the full MPS
  problem, and there it is a **strict** saddle. The 2-core route through the middle bond (cores 1, 2), which uses
  the second channel already present on the outer bonds, has rate $\lambda''=0.3149$ ($c=0.2$) and 1.0657 ($c=0.4$)
  (exact Hessian; $d=3$, $s=(1,0.6)$; multiplicity 2 for $\chi=3$).

So the sequence is: product saddle ($L/L_0=0.2629$ at $c=0.2$) $\to$ pair-junk saddle (0.2488) $\to W^*$. The second-stage
route entries are not driven during stage 1, because their coupling needs the second outer channel at $O(1)$. So
both stages start from $O(\varepsilon)$ seeds, and an $\varepsilon$-perturbation of the product saddle escapes in
$$T(\varepsilon)=\Big(\frac1{\lambda_2}+\frac1{\lambda''}\Big)\ln\frac1\varepsilon+O(1).$$
Binary4 sits at the same point $W_I$ on its *first* plateau, but there $W_I$ is a higher-order saddle: a rank increase
across binary4's single cut needs all three cores ($k=3$). On MPS4 the same point is strict. So the tree shape decides
the saddle type of one and the same critical point (derived; checked in N18: on binary4 the Hessian at $W_I$ has no
eigenvalue below $-5\cdot10^{-7}$ at $c=0.2$ or 0.4).

Checked (N12, N13, `diag5c_phases.py`). Increments of $T(\varepsilon)$ per decade of $\varepsilon$:

- $c=0.2$: 51.22, against $\ln10\,(1/\lambda_2+1/\lambda'')=51.16$.
- $c=0.4$, spectral event (middle-cut second singular value reaching $s_2/2$): 8.00, against 8.02.
- $c=0.4$, overlap event: 5.86, which is exactly $\ln10/\lambda_2$. $W_I$ already has $a_2=0.53>\tfrac12$ at this $c$, so the
  event fires on arrival at $W_I$ and times stage 1 only.

The trace ($c=0.2$, $\varepsilon=10^{-8}$) shows all three route overlaps growing at $2\lambda_2$. The outer ones then
saturate at 0.25 while the middle one decays. The loss stays briefly at $L/L_0=0.2484$–0.2487 before the jump to $W^*$.

**3c. Plateau length from small init (derived; checked).** From the fan-in init the routes already exist during
the first escape. They grow at the instantaneous rate $a(t)^{N-2}B(t)$, with $B(t)=B_2+(P-P(t))\,c^2m_{2,N-2}(c)$: early
on, the learned component's own residual also feeds the route. Integrating along $\dot a=\Lambda_{\rm eff}a^{N-1}(P-a^N)$
as in `notes/theory_saddles.md` §3.2 (hanging cores aligned) gives a head start. The stage-1 seed at the start of
the plateau scales as $\sigma^{1-\kappa}$, with
$$\kappa=\frac{B_2/P+c^2m_{2,N-2}(c)}{\Lambda_{\rm eff}(c,N)},\qquad B_2=\lambda_2/P^{(N-2)/N},$$
while the stage-2 seed scales as $\sigma$. Hence for MPS4, with spectral events,
$$\Delta t_2(\sigma)=\Big(\frac{1-\kappa}{\lambda_2}+\frac1{\lambda''}\Big)\ln\frac1\sigma+C,\qquad\Delta t_2(\sigma/2)-\Delta t_2(\sigma)\to\ln2\Big(\frac{1-\kappa}{\lambda_2}+\frac1{\lambda''}\Big).$$
At $c=0.2$, $\kappa=0.0765$ and the increment is 14.39. At $c=0.4$, $\kappa=0.162$: the increment is 2.128 with spectral
events and 1.478 with overlap events (stage 1 only). Checked (N14; seeds 0, 1, 2; $\sigma=0.05\cdot2^{-j}$,
$j=0..5$; $d=\chi=3$):

- $c=0.4$: increments 2.12–2.19 converging to 2.13 (spectral) and 1.45–1.59 converging to 1.48 (overlap), in both seeds.
- $c=0.2$: increments 17.43, 15.95, 15.24, 14.88, 14.71 (seed 0); 14.79 → 14.47 (seed 1); 17.29 → 14.67 (seed 2).
  They are still decreasing toward about 14.4; at the larger $\sigma$ the degree-4 route still contributes.
- The stage-1 seed (second singular value across $0\mid123$ at $t_1$) drops by 3.43–3.66 per halving of $\sigma$,
  i.e. $\propto\sigma^{2(1-\kappa_{\rm eff})}$ with $\kappa_{\rm eff}=0.07$–0.11, against $\kappa=0.077$.

This replaces v1's "$\log(1/\sigma)/\lambda_2(c)$", whose coefficient the verifier could not confirm. His 17.5 and 15.9
are the first two increments above.

**3d. When does the correlation route win? (heuristic; checked for $N=4$, one MPS6 example).** Two routes compete:

- the whitened second plateau (degree-$N$ route), $\Delta t_{\rm white}\propto\sigma^{-(N-2)}$, strongly seed dependent and
  accelerated by about $\Lambda_{\rm eff}(c,N)$ like every escape through $U_2$;
- the correlation route, $\Delta t_c\approx A(\sigma)/\lambda_2(c)$ with $A=\lambda_2\Delta t_2=(1-\kappa+\lambda_2/\lambda'')\ln(1/\sigma)+O(1)$.
  For MPS4 at $c=0.2$, $A=2.2$–2.4 at $\sigma=0.05$ and 3.15–3.34 at $\sigma=0.025$.

The plateau shortens markedly once $\Delta t_c<\Delta t_{\rm white}/\Lambda_{\rm eff}$. At small $c$ ($\lambda_2\approx s_2c^{N-2}((N-3)!!)^2$,
$\Lambda_{\rm eff}\approx1$) this gives
$$c^*\approx\Big[\frac{A(\sigma)}{s_2((N-3)!!)^2\,\Delta t_{\rm white}(\sigma)}\Big]^{1/(N-2)}\propto\sigma\,A(\sigma)^{1/(N-2)}:$$
linear in $\sigma$ up to logs, for every $N$, with a seed-dependent prefactor. Using the whitened estimate
$\Delta t_{\rm white}=b_{\rm eff}^{-(N-2)}(1-s_2/s_1)/((N-2)s_2)$ (`notes/theory_saddles.md` §3.3), $c^*\approx b_{\rm eff}\,[(N-2)A/((1-s_2/s_1)((N-3)!!)^2)]^{1/(N-2)}$,
which is $\approx3.4\,b_{\rm eff}$ for MPS4.

Checked on MPS4 ($d=\chi=3$). Let $c_{1/2}$ be the $c$ at which the second plateau is half its $c=0$ value (linear
interpolation; N14):

- $\sigma=0.05$: $c_{1/2}=0.137$, 0.102, 0.059 for seeds 0, 2, 1, whose whitened plateaus are 194, 389, 1208. Seed 0
  has $b_{\rm eff}=0.041$ (from its 194), so the formula gives $c^*\approx0.14$.
- $\sigma=0.025$: $c_{1/2}=0.087$ and 0.062 for seeds 0 and 2 (whitened 780, 1557). The ratio to $\sigma=0.05$ is 0.63
  and 0.61, against $\tfrac12(A(0.025)/A(0.05))^{1/2}=0.59$.
- Full table of gaps vs $c$: prediction 13.

MPS6 example ($d=\chi=2$, $\sigma=0.3$, seed 0; N20): the whitened second plateau is longer than $3.9\cdot10^4$
($t_1=150$; $a_2=1.2\cdot10^{-9}$ at $t=1500$; no second event by $t=4\cdot10^4$). The gap is 3851 at $c=0.1$, 150 at $c=0.2$ and 12.7 at $c=0.3$, so $\lambda_2(c)\cdot$gap
$=4.3,4.6,3.1$ stays nearly constant while the gap changes 300-fold, as $\Delta t_c=A/\lambda_2(c)$ predicts.

*On the verifier's "small-$N$ effect":* at fixed small $c$ the rate does fall like $c^{N-2}$. But the competing
whitened plateau grows like $\sigma^{-(N-2)}$, so the crossover $c^*=O(\sigma)$ does not grow with $N$. At fixed moderate
$c$ ($\approx0.2$–0.3) the rate itself is not smaller for larger $N$ (table in 3a). Larger $N$ is therefore not protected
(heuristic for $N\ge6$; the MPS6 example above). For $N\ge6$ the escape after the 2-core routes is a longer
cascade, so $A(\sigma)$ has a larger, $N$-dependent coefficient (MPS6: $A\approx3$–5 at $\sigma=0.3$). On MPS6 ($c=0.2$,
$d=\chi=2$, product saddle $+\,\varepsilon$; N19), stage 1 takes exactly $\ln(1/\varepsilon)/\lambda_2$: 74.2 per decade of
$\varepsilon$, against $\ln10/\lambda_2=74.19$. It is followed by a slow stage at $L/L_0\approx0.2638$, with second singular values
$\approx0.02$ on every cut, lasting 790, 997 and 1165 time units at $\varepsilon=10^{-4},10^{-6},10^{-8}$. That stage is
$\varepsilon$-dependent but sublinear in $\ln(1/\varepsilon)$, so unlike MPS4 it is not a single strict saddle (open).

**3e. Other trees (heuristic).** Each site outside a route costs a factor $c$. An add-route on a core set $C$ whose
cores carry $n_C$ sites couples to the residual through entries with $e_1$ on the $N-n_C$ outside sites, so its
coupling is $\propto c^{N-n_C}$ times a Gaussian moment (site-aligned components, $N-n_C$ even). This is the exact
version of the rule in section 6.

- MPS: routes of every even degree $k$ exist, with coupling $C_k=s_2c^{N-k}(N-k-1)!!\,m_{N-k,k}(c)$; $k=N$ gives
  $s_2\Lambda_{\rm eff}$. This is a menu between the 2-core route (log time) and the whitened $N$-core route
  ($\sigma^{-(N-2)}$), with $\Delta t_k\approx b^{-(k-2)}/((k-2)C_k)$; the fastest route wins.
- HT (one site per leaf core): the sibling-leaf route (two leaves and their parent, $k=3$, $n_C=2$) couples as
  $c^{N-2}$ ($c^6$ for HT8), against the whitened $k^*=2N-1$.
- Binary8 (two sites per leaf): $k=3$, $n_C=4$, so $c^4$. Its first plateau is already junk-rich, so this is only
  the small-$c$ order. The whitened route has $k^*=7$.

Counting costs: each extra core costs a factor $\varepsilon$ (the seed) and each excluded site a factor $c$. So the
crossover is $c^*\sim\varepsilon^q$, with $q$ = number of cores added per site added to the route: $q\approx1$ for MPS
and binary trees, $q\approx2$ for HT. HT's whitened plateaus should therefore be the most fragile, $c^*\sim\sigma^2$
(conjecture).

### 7.5 Checked numbers, and which events to use

Checked (N7–N10, N14): exact $M$ by quadrature, $d=3$, $s=(1,0.6)$. Events: spectral, i.e. singular values across
$01\mid23$ (binary4) or the middle cut (MPS4) relative to their final values $s_1,s_2$, with widths from 10% to 90%.

- Binary4 keeps separation ratios $\rho=49$–58 (seed 0) and 137–181 (seed 1) for $c=0$–0.6.
- Its first plateau equals the $M$-weighted bond-rank-1 critical value to 4 digits: $L/L_0=0.2647,0.2642,0.2488,0.1831,0.1254,0.0907,0.0708$
  for $c=0,0.1,\dots,0.6$, with junk fraction $0,0.037,0.36,0.59,0.63,0.65,0.65$. The verifier reproduced every printed digit
  (ALS on the dense $M$, and GF).
- MPS4: $\rho=48.5,12.8,2.0,1.3$ (seed 0) and $307,18.3,4.8,3.4$ (seed 1) for $c=0,0.2,0.4,0.6$.
- The second singular value across $0\mid123$ grows at $2\lambda_2$, since it is bilinear in the two route
  amplitudes: 0.112 and 0.839 against 0.105 and 0.786 at $c=0.2$ and 0.4 ($\sigma=0.05$); 0.1045–0.111 against 0.105 at
  $\sigma=1.6\cdot10^{-3}$, $c=0.2$.

**Event definitions matter (verifier, confirmed).** With the overlap events of `notes/diagnostics.md` ($a_r$ crossing
$\tfrac12$, final-rise widths) the same binary4 runs give $\rho=1.0$ at $c=0.2$ and 0.3. On the junk-rich plateau
$a_2=0.12$ ($c=0.2$) or 0.40 ($c=0.3$), and $a_1=0.85$ ($c=0.3$). So $a_2$ never falls back below 0.1, $a_1$ does not
reach 0.9 before the second event, and the final-rise widths span the whole plateau. On MPS4 at $c\gtrsim0.3$ the
overlap event $a_2=\tfrac12$ fires at the intermediate saddle $W_I$ ($a_2=0.53$ there at $c=0.4$), so overlap gaps measure
stage 1 only. **For cross-site runs use spectral events (or loss-level events), with final-rise widths.** First-crossing
widths are not safe either. On MPS4 at $c=0.4$ (seed 0) the middle-cut $s_2$ exceeds $0.1s_2$ during stage 1 (the middle
route grows, then decays to $0.028s_2$), so the first-crossing width grows with $\ln(1/\sigma)$ (3.4 to 6.6), while the
final-rise width is 1.7 (1.6 for seed 1). With final-rise widths, $\rho=10$–11 at $\sigma=1.6\cdot10^{-3}$ against 2–5 at
$\sigma=0.05$: it grows like $\ln(1/\sigma)$, as a log-time plateau should.

## 8. Numerical checks

All with the core library only; GF by `solve_ivp` (DOP853) unless stated. $d=\chi=4$ unless stated. Scripts in the
scratch folder named above (`check*.py`; `out_*.txt` hold the raw output). Cross-site checks use the dense $M$ from
the exact latent quadrature (`common5.py`), cross-validated in N7 against Isserlis and the `population_loss` mixture.

| id | script | what | result |
|---|---|---|---|
| N1 | `check1_aligned.py` | aligned family, aligned init, binary4/MPS4, $s=(1,0.6)$, $\Lambda=(0.41,2.07)$, $a_0=0.05$ | loss formula $7\cdot10^{-16}$; ODE $3\cdot10^{-15}$; off-family gradient $7\cdot10^{-16}$; $t_{1/2}$: 46.749/46.752, 15.263/15.265 (binary), 487.64/487.63, 160.47/160.48 (MPS); order reversed; plateau 0.2048 |
| N1b | `check1b_timemap.py` | $P_r(t;\Sigma)$ vs $P_r^{(I)}(\Lambda_rt)$ | max deviation $1.1\cdot10^{-6}$ (binary), $1.3\cdot10^{-5}$ (MPS) |
| N2 | `check2a_exponent.py` | one component, random init, $\Lambda\in[0.5,8]$, 3 splits, 3 seeds, binary ($\sigma=0.01$), MPS ($\sigma=0.05$) | slope $-1.0000$ (18/18); $t_{1/2}\Lambda$ constant to $<5\cdot10^{-5}$ |
| N3 | `check2b_race.py` | $s=(1,0.5)$, settings A (window), A2 (other split), B ($\Lambda_2=6$), AT (same $T$, whitened) | fixed-level ($P_{\max}=0.02$) escapes equal to $10^{-4}$; event sequences and plateaus as in section 5 |
| N4 | `check2c_rotation.py` | rotation rate at the comp-2 saddle, binary4 | 0.1263/0.1263, 0.1131/0.1130, 0.0548/0.0548 (measured/predicted) |
| N5 | `check3a_equivalence.py` | Prop. 1, random SPD $\Sigma_i$ (eig 2.5, 1, 0.3), $d=\chi=3$, $\sigma=0.3$ | preconditioned flow matches to $5.7\cdot10^{-9}$, $2.0\cdot10^{-8}$, $2.4\cdot10^{-10}$ (binary, MPS, HT); naive flow off by 8%, 4%, 100% |
| N6 | `check3b_misaligned_order.py` | misalignment-only reversal | section 6 |
| N7 | `check4a_crosssite.py` | latent quadrature vs Isserlis vs Monte Carlo; $c^3$ remainder; $\Lambda_{\rm eff}$ | $d=2$, $c=0.4$: 1.27544842 (quadrature) = 1.27544842 (Isserlis), MC $1.2718\pm0.0057$; remainder ratios 8.2, 8.4, 8.9 under $c\to2c$ |
| N8 | `check4b_tilt.py` | small-$c$ escape direction (binary4) | $4.08c^2$, $1.02c^2$ at $c=0.005$; the 2% excess is the factor $1+4c$ (N15), not integrator error |
| N9 | `check4c_crosssite_dyn.py` | GF with exact $M$, site-aligned and random-basis teachers, binary ($\sigma=0.01$), MPS ($\sigma=0.05$), seeds 0, 1 | section 7.5; random-basis binary: $\rho=64$–69 at $c\le0.4$, junk at plateau 0, 0.017, 0.35–0.38, 0.61 for $c=0,0.2,0.4,0.6$ |
| N10 | `check4d_routes.py` | MPS4 2-core route rates at the product critical point | superseded by N11 (same numbers) |
| N11 | `check5a_routes_N.py` | route rates at the MPS product saddle, $N=4,6,8$, $d=2,3$ | closed form = dense $B$ = exact Hessian (5 digits), multiplicity $N-1$; junk routes included |
| N12 | `check5c_eps.py`, `diag5c_phases.py` | escape from the MPS4 product saddle + $\varepsilon$ | per decade: 51.22 (pred. 51.16), $c=0.2$; 8.00 (8.02) spectral and 5.86 (5.86) overlap, $c=0.4$ |
| N13 | `check5d_intermediate.py` | pair-junk saddle $W_I$ (MPS4 with middle bond 1, embedded in $\chi=3$) | $L/L_0=0.24880$, 0.12544 (= binary4 plateaus); $\lambda''=0.3149$, 1.0657 |
| N14 | `check5b_gap.py` | second plateau vs $\sigma$ ($c=0.2,0.4$; $\sigma$ to $1.6\cdot10^{-3}$) and vs $c$ ($\sigma=0.05,0.025$); MPS6 | section 7.4c–d; predictions 12–14 |
| N15 | `check5e_misc.py` | binary4 tilt (exact pairing block), ranks of the pairing term, conditioning of $M$ | sections 7.3, 7.4 item 5 |
| N16 | `check5f_sameT.py` | same-$T$ offsets, binary4 and MPS4 | $-0.2218$ vs $-0.2219$; $-0.105$ vs $-0.108$ (section 4) |
| N17 | `check5g_race.py` | first-escape winner in the window vs $\sigma$ | binary4 0/32 ($\sigma\le0.1$), 1/32, 2/32 ($\sigma=0.2,0.3$); MPS4 5/24 at $\sigma=0.1$ and 0.05 (same seeds) |
| N18 | `check5i_binary_WI.py` | binary4 Hessian at $W_I$ (bond-rank-1 critical point embedded in $\chi=3$) | no eigenvalue below $-5\cdot10^{-7}$ (non-strict); $L/L_0=0.24880$, 0.12542 |
| N19 | `diag5h_mps6.py`, `diag5h2_mps6.py` | escape from the MPS6 product saddle + $\varepsilon$, $c=0.2$, $d=\chi=2$ | stage 1: 74.2 per decade ($=\ln10/\lambda_2$); slow second stage 787, 997, 1165 at $\varepsilon=10^{-4},10^{-6},10^{-8}$ |
| N20 | `check5b_gap.py` (MPS6) | MPS6, $\sigma=0.3$, seed 0, $c=0,0.1,0.2,0.3$ | whitened: $t_1=150.5$, no second event by $t=4\cdot10^4$; gaps 3851, 150, 12.7 |

Caveat: N9's random-basis runs used `site_bases` before the lead's library update of 2026-10-09 (the generator for
random bases changed), so a rerun gives other bases, with the same qualitative picture. Independent verification:
the verifier's scripts (`covariance_verify/v1*`–`v10*`) reproduced C1, C2, C4, C5, C6, C7, C8 and the C9/C10 numbers of
v1, and found the issues addressed in the change log.

## 9. What is new here, and open points

New for TTNs (not found in `notes/literature.md`):

- Covariance reduction: the exact preconditioned-flow form with its consequences (Prop. 1); the exact aligned time
  map for any tree; the exponent $-1$ and split invariance of escapes, plus the exact same-$T$ saturation offset.
- Orderings: the speed-versus-loss mismatch window and its tree-dependent outcome, with the covariance-weighted
  piggyback block; $\Lambda^{\rm esc}\ge\Lambda^{\rm loss}$.
- Cross-site correlations:
  - the latent-mixture contraction (of the standard one-factor model);
  - the exact pairing-block escape direction of binary4;
  - exact 2-core route rates at the MPS product saddle for every even $N$;
  - the two-stage MPS4 escape through binary4's junk-rich plateau point, a critical point that is a higher-order
    saddle on one tree and a strict saddle on the other;
  - the plateau-increment law with head start;
  - the crossover $c^*=O(\sigma)$.

Open:

- MPS$_N$ with $N\ge6$ under correlation. The cascade after stage 1 is not worked out. On MPS6 (N19) the slow
  second stage has small second singular values ($\approx0.02$) on *every* cut rather than a dimerized pattern. Its
  length depends on $\varepsilon$ sublinearly in $\ln(1/\varepsilon)$, so it is not a single strict saddle. Next step:
  locate the critical point(s) of that stage and their Hessians, as N13 did for MPS4.
- Binary8 and HT under correlation (section 7.4e is heuristic); in particular the HT prediction $c^*\sim\sigma^2$.
- A race formula for random init (it comes from the whitened escape theory via $s\to\Lambda s$); the exponent of the
  binary4 failure rate ($\sigma^{0.41}$ heuristic) is untested (0–2 failures in 32 seeds at $\sigma\le0.3$).
- Teachers that are not site-aligned under cross-site correlation (random bases: section 7.3, $O(c^2N^2/d)$).
- Cross-site correlations that decay with distance on the tree (the natural "data matches tree" question).
  Conjecture: trees that put strongly correlated sites on the same core learn pairing structure in the first
  escape; trees that separate them get it through low-degree routes later.
- The $c\to1$ regime ($\lambda_{\min}(M)$ for $d<N$; dynamics of antisymmetric parts).

## 10. Predictions for experiments

Conventions:

- "Aligned" means `odeco_teacher(..., basis="standard")` with diagonal `Sigma_i` (e.g. `power_law_cov`), or $\Sigma_i$
  rotated into the teacher bases.
- Overlaps are Frobenius, $a_r=P_r/s_r$ (`component_overlaps(..., site_metrics=None)`).
- Product-covariance runs use the events of `notes/diagnostics.md`; cross-site runs use the events of item 10.
- "Increment per halving" means $\Delta t(\sigma/2)-\Delta t(\sigma)$.
- Tolerances are what the checks achieved; setting $=$ $d=\chi=3$ unless stated.

1. **Exact aligned dynamics (proven).** Aligned init, aligned $\Sigma$, odeco teacher:
   $t_{1/2,r}=a_0^{-(K-2)}/((K-2)\Lambda_rs_r)-C_K/(\Lambda_rs_r^{2-2/K})+O(a_0^2)$, with $C_3=0.8503$, $C_4=0.2664$, $C_5=0.0471$,
   $C_6=-0.0720$, $C_7=-0.1478$, $C_8=-0.2006$. Also $P_r(t;\Sigma)=P_r^{(I)}(\Lambda_rt)$, event spacing from Prop. 2.6, and
   plateaus at $\tfrac12\sum_{r\notin S}\Lambda_rs_r^2$. Tolerance: $\le1.2\cdot10^{-4}$ relative at $a_0=0.05$.
2. **$\Lambda$-exponent $-1$ (derived, checked).** Random fan-in init, one dominant aligned component:
   $d\log t_1/d\log\Lambda=-1.00\pm0.01$ for binary, MPS and HT trees and for every split of $\Lambda$ over sites. $t_1\Lambda$ is
   constant per seed to $5\cdot10^{-5}$ ($\sigma\le0.05$), or $3\cdot10^{-3}$ (HT, $\sigma=0.3$; verifier). Falsifier: plain GF on the
   whitened teacher with isotropic init gives $-0.50$.
3. **Same-$T$ problems (derived, checked).** Compare $(\Sigma,W^*)$ with $(I,\Sigma W^*)$ from the same seed, aligned.
   The winner is the same. The $a_r=\tfrac12$ times, each relative to its own target, differ by
   $C_Ks_r^{-(2-2/K)}(\Lambda_r^{-(2-2/K)}-\Lambda_r^{-1})$, independent of $\sigma$: $-0.2219$ (binary4) and $-0.1080$ (MPS4) at
   $s_r=0.5$, $\Lambda_r=2^{3/2}$. Tolerance: 1% (binary, $\sigma\le0.03$) and 3% (MPS, $\sigma=0.03$); at $\sigma=0.1$ the MPS offset
   is about 20% smaller ($O(a_0^2)$). Times at a fixed small overlap $P_*$ differ by $(\Lambda_r-1)P_*^{2/K}/(2\Lambda_r^2s_r^2)$
   (0.034 and 0.065 at $P_*=0.02$; measured 0.034–0.035 and 0.066–0.069).
4. **Plateau levels (proven, aligned; derived, misaligned).** Aligned: $L_S=\tfrac12\sum_{r\notin S}\Lambda_rs_r^2$.
   Misaligned binary4: $L_k=\tfrac12\sum_{j>k}\sigma_j(\mathrm{mat}_{01\mid23}SW^*)^2$. Doubly orthogonal CP teachers:
   $\tfrac12\sum_{r\notin S}\Lambda^{\rm loss}_rs_r^2$ (0.3200 in the N6 example).
5. **Learning order (derived; probabilities heuristic, measured).**
   - Aligned init: order by $\Lambda_rs_r$. The whitened order reverses iff $\Lambda_2/\Lambda_1>s_1/s_2$; with power-law
     spectra, $\Lambda_2/\Lambda_1=(k_1/k_2)^{\alpha N}$.
   - Random init: the whitened race at rate ratio $\Lambda_2s_2/(\Lambda_1s_1)$.
   - Matrix-like pair (binary4 odeco), window example $s=(1,0.5)$, $\Lambda=(1,2^{3/2})$: the faster component wins with
     probability $\to1$ as $\sigma\to0$. Failures: 0/32 at $\sigma=0.003$, 0.03, 0.1; 1/32 at 0.2; 2/32 at 0.3.
   - Tensor-like pair (MPS4), same example: the failure probability does not depend on $\sigma$ and is fixed by the init
     direction: 5/24 at both $\sigma=0.1$ and 0.05, with identical seeds.
6. **Mismatch window $s_1/s_2<\Lambda_2/\Lambda_1<(s_1/s_2)^2$ (derived, checked), conditional on the first winner.**
   Example $s=(1,0.5)$, $\Lambda=(1,2\sqrt2)$.
   - (a) 2 (faster) first, binary4 ($m=2$, $k=3$): 1 rises and 2 falls at constant bond rank 1 (second singular value
     of $\mathrm{mat}_{01\mid23}\le10^{-7}$ at $\sigma=0.003$), after $\Delta t\approx\ln(1/\varepsilon)/\lambda_{\rm pb}$. Rates
     $\lambda_{\rm pb}=0.1263$ (even split), 0.1130 ($\mu_2=2.83$, $\nu_2=1$), 0.0548 ($\mu_2=8$, $\nu_2=0.354$). Plateau
     $\tfrac12\Lambda_2s_2^2=0.3536$; 2 is relearned after a time $\propto\sigma^{-1}$.
   - (b) 2 first, MPS4, binary8, HT: no rotation; plateau $\tfrac12\Lambda_1s_1^2=0.5$; then 1 is added.
   - (c) 1 first, any tree: plateau $0.3536$, no rotation, then 2 is added through its route.
   - (d) Components that differ on two adjacent terminal cores ($k=2$): never a rotation; the add route wins.
7. **Misalignment (derived, checked).** Escape order by $s_r\Lambda^{\rm esc}_r$. At the first escape
   $\cos(W,U_r)=\prod_i\langle u,\Sigma_iu\rangle/\lVert\Sigma_iu\rVert$ (0.372 in the N6 example; measured 0.371–0.380) and
   $\cos(W,\widetilde U_r)\ge0.99$. Then $W$ realigns to $U_r$ without an extra plateau.
8. **Cross-site latent model, first plateau (derived, checked).**
   - The plateau is present for every $c<1$, and $t_1\cdot\tau_1(MW^*)$ is nearly constant in $c$ (binary4 seed 0, $c=0\to0.6$:
     $193\to174$ with $\tau_1=\sigma_1(\mathrm{mat}_{01\mid23}MW^*)$).
   - Binary4 ($d=3$, $s=(1,0.6)$), $c=0,0.1,\dots,0.6$: plateau $L/L_0=0.2647$, 0.2642, 0.2488, 0.1831, 0.1254, 0.0907,
     0.0708 and junk fraction 0, 0.037, 0.36, 0.59, 0.63, 0.65, 0.65, to 4 digits.
   - Escape direction: top eigenvector of the pairing matrix $A$ (section 7.3). The $x_2$ coefficient is 0.0520
     ($c=0.1$) and 0.190 ($c=0.2$); the $x_3$ coefficient is 0.0135 and 0.0551.
   - On MPS the first plateau is junk-free: its critical point is exactly $PU_1$ (junk fraction 0.000 at the ALS
     critical point for $c\le0.6$). The junk on binary4 grows with $d$ ($\propto c^2\sqrt{d-R}$ at small $c$).
9. **Teacher weights under the latent model (proven).** $\langle U_r,MU_r\rangle=\Lambda_{\rm eff}(c,N)$ (e.g. 1.832 at
   $c=0.2$, $N=4$; 466 at $c=0.3$, $N=8$) and $L_0=\tfrac12[\Lambda_{\rm eff}\sum_rs_r^2+c^N((N-1)!!)^2\sum_{r\ne r'}s_rs_{r'}]$:
   $L_0=1.2544$ ($c=0.2$), 4.5610 ($c=0.4$) for $N=4$, $s=(1,0.6)$.
10. **Event definitions for cross-site runs (checked).** Use spectral events: singular values across a cut relative to
    their final values ($s_1,s_2$ for site-aligned teachers), the event at $\tfrac12$, final-rise widths (10% to 90%).
    - Expected with these: binary4 $\rho=49$–58 (seed 0) and 137–181 (seed 1) for $c=0$–0.6, $\sigma=0.01$.
    - Overlap events give $\rho=1.0$ for binary4 at $c=0.2$ and 0.3 (on the plateau $a_2=0.12$, 0.40 and $a_1=0.85$), so do
      not use them.
    - On MPS4 at $c\gtrsim0.3$ the overlap event $a_2=\tfrac12$ fires at the intermediate saddle ($a_2=0.53$ at $c=0.4$).
    - First-crossing widths can be inflated by transients (item 7.5).
11. **MPS route rates (proven, checked).** At the MPS$_N$ product saddle (site-aligned odeco teacher, $s=(1,0.6)$) the
    Hessian has $N-1$ copies per new bond index of $-\lambda_2$ and of $-\lambda_{\rm junk}$ (section 7.4a). At
    $c=0.05,0.1,0.2,0.3,0.4$:

    | $N$ | $\lambda_2$ |
    |---|---|
    | 4 | $1.84\cdot10^{-3}$, $9.00\cdot10^{-3}$, 0.0525, 0.165, 0.393 |
    | 6 | $4.94\cdot10^{-5}$, $1.11\cdot10^{-3}$, 0.0310, 0.245, 1.12 |
    | 8 | $3.64\cdot10^{-6}$, $3.63\cdot10^{-4}$, 0.0459, 0.872, 7.38 |

    $\lambda_{\rm junk}=0.00247$ ($N=4$, $c=0.2$) and 0.0321 ($c=0.4$). On a plateau, the second singular value across
    $0\mid$rest grows at $2\lambda_2$. Fitted rates at $c=0.2$ were 0.109–0.130 at $\sigma=0.05$ (the fit window reaches the
    nonlinear stage) and 0.1045–0.111 at $\sigma=1.6\cdot10^{-3}$, against $2\lambda_2=0.105$.
12. **MPS4 second plateau (derived, checked; $s=(1,0.6)$).**
    - Two stages: product saddle ($L/L_0=0.2629$ at $c=0.2$) $\to$ pair-junk saddle $W_I$ ($L/L_0=0.2488$ at $c=0.2$, 0.1254
      at $c=0.4$; $\lambda''=0.315$, 1.066) $\to W^*$. The loss shows a short second step at $L/L_0\approx0.249$ ($c=0.2$).
    - Increment of the second-plateau length per halving of $\sigma$: $\ln2[(1-\kappa)/\lambda_2+1/\lambda'']$.
      At $c=0.2$ this is 14.39 (measured 14.47–14.71 at $\sigma=1.6\cdot10^{-3}$ and still decreasing; 17.4 between
      $\sigma=0.05$ and 0.025). At $c=0.4$ it is 2.128 with spectral events (measured 2.13) and, with overlap events,
      $\ln2(1-\kappa)/\lambda_2=1.478$ (measured 1.48).
    - Lengths at $c=0.2$ (seed 0): 42.5, 60.0, 75.9, 91.1, 106.0, 120.7 for $\sigma=0.05\cdot2^{-j}$, $j=0..5$.
    - From the product saddle plus $\varepsilon$: $(1/\lambda_2+1/\lambda'')\ln(1/\varepsilon)+O(1)$, i.e. 51.2 per decade at $c=0.2$.
    - Saddle types (for the plateau-midpoint Hessians of `notes/diagnostics.md`). MPS4 product plateau: strict,
      $\lambda_{\min}=-\lambda_2$. MPS4 intermediate point: strict, $-\lambda''$. Binary4 first plateau ($c>0$; the same tensor
      $W_I$): no eigenvalue below $-5\cdot10^{-7}$, i.e. a higher-order saddle. Its second plateau should scale as $\sigma^{-1}$
      ($k=3$ is forced by the single cut; derived, not yet checked under correlation).
    - Test: $\sigma$ sweep from 0.05 to $1.6\cdot10^{-3}$ at $c\in\{0.2,0.4\}$, $\ge5$ seeds. The slope of $\Delta t_2$ vs
      $\log_2(1/\sigma)$ should be within 3% of the prediction once $\sigma\le0.0125$.
13. **Crossover in $c$ (heuristic, checked for $N=4$).** At fixed $\sigma$ the second plateau keeps its whitened length
    ($\sigma^{-(N-2)}$ law) for $c\ll c^*$ and follows $A(\sigma)/\lambda_2(c)$ (log law) for $c\gg c^*$, with
    $c^*\propto\sigma A(\sigma)^{1/(N-2)}$. Measured second-plateau lengths:

    | $\sigma$ | seeds | whitened | $c=0.05$ | $c=0.07$ | $c=0.1$ | $c=0.15$ | $c=0.2$ | $c_{1/2}$ |
    |---|---|---|---|---|---|---|---|---|
    | 0.05 | 0/2/1 | 194/389/1208 | 189/347/704 | — | 145/200/258 | 81/95/104 | 42.5/46.2/46.9 | 0.137/0.102/0.059 |
    | 0.025 | 0/2 | 780/1557 | 671/985 | 504/642 | 296/345 | 127/139 | — | 0.087/0.062 |

    Prediction: $c_{1/2}(\sigma/2)/c_{1/2}(\sigma)\approx\tfrac12(A(\sigma/2)/A(\sigma))^{1/2}\approx0.59$ (measured 0.61–0.63). Sweep to
    test: $c\in\{0,0.03,0.05,0.07,0.1,0.15,0.2\}\times\sigma\in\{0.05,0.025,0.0125\}$, $\ge5$ seeds.
14. **Larger $N$ (heuristic; one check).** The collapse is not a small-$N$ effect: on MPS$_N$ the crossover stays at
    $c^*=O(\sigma)$ up to logs, while the whitened plateau grows as $\sigma^{-(N-2)}$. MPS6 ($d=\chi=2$, $\sigma=0.3$, seed 0):
    the whitened second plateau is longer than $3.9\cdot10^4$; at $c=0.1$, 0.2, 0.3 it is 3851, 150 and 12.7. Prediction for a
    sweep: gap$\cdot\lambda_2(c)$ is roughly constant ($A\approx3$–5) wherever the gap is much shorter than the whitened one.
    HT conjecture: $c^*\sim\sigma^2$.
15. **Random-basis teachers under cross-site correlation.** Effects are $O(c^2N^2/d)$ at small $c$. N9 ($d=3$) shows plateau
    junk 0.017 at $c=0.2$ and $\approx0.36$ at $c=0.4$, so use $d\ge8$ for a weakly perturbed baseline.
16. **$c\to1$ (proven / conjecture).** $\kappa(M)=30$, $3.0\cdot10^3$, $3.7\cdot10^5$ at $c=0.5$, 0.9, 0.99 ($N=4$, $d=3$): expect
    stiff GF. $\lambda_{\min}=(1-c)^{N-1}(1+(N-1)c)$ when $d\ge N$.

References added in v2: [Vasicek2002] O. Vasicek, "The distribution of loan portfolio value", Risk 15(12), 2002;
[Li2000] D. X. Li, "On default correlation: a copula function approach", J. Fixed Income 9(4), 2000 (the
one-factor Gaussian copula; cited only for the latent representation of equicorrelated Gaussians).
