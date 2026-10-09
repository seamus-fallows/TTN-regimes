# Theory: regime boundaries and large-bond-dimension limits (Q5, H4)

Theory note for the `regime-map` agent (brief `notes/briefs/regime-map.md`). Model, loss and
diagnostics are those of `notes/model.md` and `notes/diagnostics.md`.

**Status: v2 (2026-10-09), revised after an independent verification of v1.** Every number in §6 comes
from runs on the current library (teacher and init use independent random streams since commit 57d5262;
the verifier's runs are marked), with trees given as explicit parent arrays (§1). Changes from v1:

- **Withdrawn.** The $K=5$ calibration ($\kappa_5=0.50$, $\sigma_c(5)=0.65$) and the $K=7$ range
  $\sigma_c=0.86$–$1.0$ (pre-fix runs, not reproducible); v1's P10 ("a $\sigma$-independent fraction of
  seeds has merged first events", an artifact of first-crossing widths); the escape law
  $\tau_{\rm esc}\propto\sigma^3\ln\sigma$ for the two-copy model with a deficient bond (it is
  $\propto\sigma^{2K}$); the first-order lazy boundary at $\chi\ge32$; "the two-copy boundary is twice as
  sharp".
- **New.** The coherent second-order kernel change (R4), which sets the lazy boundary at large $\chi$
  ($\sigma_{\rm lazy}\propto\chi^{-1/2}$, R6); the adiabatic reduction of the deficient-bond escape and
  the centred-model floor at small $\sigma$ (R7); seed distributions of $\sigma_c$ at $K=3$ and (full
  bonds) $K=7$, and the decay law of transients (R9); stalls at $K\ge5$ when $\chi$ equals the teacher
  rank (R10).
- **Corrected.** Literature attributions (R8, §4.1); the exact $K\to\infty$ limit (R8); the rank
  formula needs consistent bond dimensions (R2); "small root ⇒ lazy" needs full bonds (R13); the
  arithmetic of the $K=7$ estimate.

Status tags: **proven**; **derived** (under the stated assumptions); **heuristic**; **conjecture**;
**empirical** (§6; exact population gradient flow (GF) unless stated otherwise).

## 0. Results at a glance

| # | result | status |
|---|---|---|
| R1 | Per-core tangent kernel $J_vJ_v^\top=(\otimes_c\Phi_c\Phi_c^\top)\otimes I\otimes\Psi_v\Psi_v^\top$; for every $\chi$, $\mathbb E\,\Theta=\frac{\alpha^2}{D}\sum_v\eta_vF_v/s_v^2\,I_D$ and $\mathbb E[W_0\otimes W_0]=\frac{\alpha^2}{D}I_D$ | proven; empirical |
| R2 | $\operatorname{rank}\Theta=\sum_vF_v\chi_v-\sum_e\chi_e^2$ with consistent (effective) bond dimensions; $=D$ iff every bond is full | proven; empirical |
| R3 | $\Theta\approx\bar\lambda I$ needs $\chi_e\gg D_e$ on every bond; at capped full bonds $\lambda_{\min}/\bar\lambda\sim10^{-4}$ | derived; empirical |
| R4 | Centred model: $\delta_K^2\approx(c_1x)^2+(c_2x^2)^2$, a zero-mean first-order term plus a coherent second-order term (bottom features drift toward $T$); closed forms for $N=4$ under SP and NTK | derived ($N=4$, $\chi\gg d^2$); empirical |
| R5 | Two-copy model: no first-order term, $\delta_K\approx\tfrac14c_2x^2$; $\delta_\theta$ is half the centred value | derived; empirical |
| R6 | Lazy boundary ($N=4$, SP): $\sigma_{\rm lazy}=(\Vert T\Vert^2c_2/\delta_*)^{1/6}\propto\chi^{-1/2}$ once $c_2x_c\gg c_1$; two-copy $\times2^{-1/3}$ | derived; empirical |
| R7 | Deficient bond: the centred model has a floor, $(1-\dim\mathcal M_\chi/D)\,\sigma^{2K}/\Vert W^\star\Vert^2$ at small $\sigma$; the two-copy model has a lazy plateau and then escapes after $t_{\rm esc}=2\theta\sigma^2/\Vert T\Vert^2$, i.e. $\tau_{\rm esc}\propto\sigma^{2K}$ | derived; empirical |
| R8 | Aligned odeco dynamics: exact separation ratios for the project's criterion; $a_{0c}(K)\to1$; $u_{0c}(K)\to$ a finite limit ($1.306\times10^{-2}$ for weights $(3,2,1)$) | proven given the aligned reduction; ODE and boundary structure known |
| R9 | Random fan-in init: $t_1\propto\sigma^{-(K-2)}$; gaps and $\rho_i\propto\sigma^{-(K-2)}$ per seed; transients decay like $\sigma^{2(1-q)}$; the boundary is a seed distribution: median $\sigma_c=0.29$ ($K=3$) and $0.71$ ($K=7$, full bonds); effective per-core amplitude $\kappa_K$ grows with $K$ | derived ($t_1$); heuristic; empirical; conjecture ($\kappa_K\to1$) |
| R10 | $K\ge5$ with $\chi$ = teacher rank $R$: the last component needs all $K$ cores and its plateau is very long; fan-in init can freeze it for good; balanced init does not fix $K=7$, spare channels ($\chi\ge2R$) do in the runs so far | empirical; heuristic mechanism |
| R11 | muP for TTNs: the spectral condition must use the injective norm; fan-in init except the root, scaled by $(\sum\text{child dims})^{-1/2}$; $\eta_v=\chi_v/F_v$ | derived; empirical |
| R12 | $\chi\to\infty$ at fixed $x$ ($N=4$): $\delta_K\propto\chi^{-3}$ (SP), $\chi^{-1}$ (NTK), $\chi^0$ (muP-TTN), at first and second order | derived; empirical |
| R13 | Root imbalance at fixed $\alpha$: a small root is lazy ($\delta_K\propto\lambda^2$) only with full bonds; a large root is not lazy | derived (aligned); empirical |

## 1. Notation

- Tree with $K$ cores. Core $v$ has fan-in $F_v$ (product of its child-leg dimensions: physical $d_i$ and
  child bonds $\chi_c$) and fan-out $\chi_v$ (its parent bond; $\chi_{\rm root}=1$). $D=\prod_id_i$.
- Init $G_v=s_vZ_v/\sqrt{F_v}$, $Z_v$ iid $N(0,1)$; default $s_v=\sigma$; $\alpha:=\prod_vs_v$, so
  $\mathbb E\Vert W_0\Vert_F^2=\alpha^2$ exactly.
- GF $\dot G_v=-\eta_v\partial L/\partial G_v$, $L=\tfrac12\Vert W-W_{\rm target}\Vert_F^2$ (whitened
  inputs). $J_v=\partial W/\partial G_v$; $\Theta=\sum_v\eta_vJ_vJ_v^\top$ ($D\times D$).
- Edge between $v$ and its parent: $\Phi_v\in\mathbb R^{D_v\times\chi_v}$ is the subtree tensor below $v$
  ($D_v=\prod_{i\in{\rm sub}(v)}d_i$), $\Psi_v$ the environment; $W_{(e)}=\Phi_v\Psi_v^\top$. The cut
  dimension of the edge is $\min(D_v,D/D_v)$; a bond is **full** if $\chi_e$ reaches it.
- $T$ = what the network must add: $T=W^\star$ for the centred model (target $W^\star+W_0$,
  `ShiftedTeacher` or `centred_population_loss`) and for the two-copy model ($W_0=0$,
  `antisymmetric_double` + mask). $x:=\Vert T\Vert_F/\alpha$.
- $\bar\lambda:=\operatorname{tr}\Theta(0)/D$; kernel time $\tau:=\bar\lambda t$.
- $\delta_K=\Vert\Theta_{\rm end}-\Theta_0\Vert_F/\Vert\Theta_0\Vert_F$ (tensor-space kernel),
  $\delta_\theta=\Vert\theta_{\rm end}-\theta_0\Vert/\Vert\theta_0\Vert$; $\delta_*$ = the laziness threshold
  (0.1 by default).
- Teacher spectrum: odeco weights $s_1>s_2>\dots$, $q_i=s_{i+1}/s_i$,
  $\nu:=\sum_rs_r^4/(\sum_rs_r^2)^2$.
- **Reference case:** binary $N=4$ ($K=3$), $d=3$ ($D=81$), odeco teacher $(3,2,1)$,
  $\Vert T\Vert=\sqrt{14}=3.742$, $\nu=0.5$. Binary $N=4$ is the 3-factor matrix product $W=ABC^\top$ of a
  $d^2\times d^2$ matrix ($A,C\in\mathbb R^{d^2\times\chi}$ bottom cores, $B\in\mathbb R^{\chi\times\chi}$
  root); $n:=d^2$.
- **Trees used in §6** (parent array; sites per node):
  `bin4` $(2,2,-1)$; $((0,1),(2,3),())$ = `binary_tree(4)`, $K=3$.
  `old5` $(2,2,4,4,-1)$; $((0,1),(2),(),(3,4),())$, $K=5$ (the pre-57d5262 `binary_tree(5)`; the current
  `binary_tree(5)` has $K=4$).
  `bin8` $(2,2,6,5,5,6,-1)$; $((0,1),(2,3),(),(4,5),(6,7),(),())$ = current `binary_tree(8)`, $K=7$.

## 2. (a) Lazy regime

### 2.1 Kernel at init: structure, scale, rank, conditioning

**R1a (proven).** $W$ is linear in $G_v$: $W=(\Phi_{c_1}\otimes\Phi_{c_2}\otimes I_{{\rm phys}(v)}\otimes\Psi_v)\,{\rm vec}\,G_v$,
so $J_vJ_v^\top=\Phi_{c_1}\Phi_{c_1}^\top\otimes\Phi_{c_2}\Phi_{c_2}^\top\otimes I_{{\rm phys}(v)}\otimes\Psi_v\Psi_v^\top$
(legs in site order). For $N=4$: $\Theta[X]=\eta_AXM^\top M+\eta_BAA^\top XCC^\top+\eta_CNN^\top X$ with
$M=BC^\top$, $N=AB$.

**R1b (proven).** For any bond dimensions,
$\mathbb E\,\Theta=\frac{\alpha^2}{D}\sum_v\frac{\eta_vF_v}{s_v^2}I_D$ and $\mathbb E[W_0\otimes W_0]=\frac{\alpha^2}{D}I_D$.
(Cores are independent; $\Phi_c$ and $\Psi_v$ use disjoint cores; the init is invariant under $O(d_i)$ on
every physical leg, which acts irreducibly; traces by induction from the leaves and from the root.)

- Kernel scale ($s_v=\sigma$, $\eta=1$): $\bar\lambda=\sigma^{2(K-1)}\sum_vF_v/D$. Rescale time by this,
  not by $\sigma^{2(K-1)}$ alone. Two-copy: $\Theta_2(0)=2\Theta(\theta_0)$.
- Kernel shares: core $v$ carries $\eta_vF_v/s_v^2\big/\sum_u\eta_uF_u/s_u^2$ of $\operatorname{tr}\Theta$;
  under SP the $\chi^2$-fan-in cores (binary roots and internal cores) dominate.
- Prior covariance: $\mathbb E f(x)f(x')=\frac{\alpha^2}{D}\prod_i\langle x_i,x_i'\rangle$ (product linear
  kernel) at every $\chi$. Known context: Gaussian-process limits at infinite bond dimension were stated
  for MPS by Guo & Draper (2021a, arXiv:2101.02333; informal, and internally inconsistent according to
  `notes/literature.md`) and shown for CPD- and TT-constrained kernel machines by Wesel & Batselier
  (AISTATS 2025), both with this product kernel. At finite $\chi$ the output is not Gaussian (kurtosis
  $3\prod_e(1+2/\chi_e)$, `notes/literature.md` §2.3).

**R2 (proven).** $\operatorname{rank}\Theta=\dim T_{W_0}\mathcal M_\chi$, where $\mathcal M_\chi$ is the set
of tensors whose matricization across each edge $e$ has rank $\le\chi_e$. Bond dimensions are
**consistent** when every $\chi_e$ is at most the product of the other leg dimensions at each of its two
cores (true for uniform $\chi$ with capping). For consistent bonds and a generic init,
$$\operatorname{rank}\Theta=\sum_vF_v\chi_v-\sum_e\chi_e^2\quad(\text{capped at }D),$$
the dimension of the fixed-rank TTN manifold (same count as Uschmajew & Vandereycken 2013 for HT). For
inconsistent bonds, first replace each $\chi_e$ by its effective value: iterate
$\chi_e\leftarrow\min(\chi_e,\ \prod\text{other legs of the child core},\ \prod\text{other legs of the parent core})$
to a fixed point (the generic rank of $W_{(e)}$), then apply the formula. Example: `bin8`, $d=3$, bottom
bonds 1, top bonds 3: the nominal formula gives 29, the effective bonds are all 1 and give 33, the
measured rank (§6).

- $\operatorname{rank}\Theta=D$ iff every bond is full. Then the linearized (lazy) fit reaches zero loss;
  otherwise it stops at $\tfrac12\Vert P_\perp T\Vert^2$, $P_\perp$ = projector on the normal space of
  $\mathcal M_\chi$ at $W_0$.
- $N=4$ (derived): the tangent space is $\{X:P_A^\perp XP_C^\perp=0\}$ with Haar-random column spaces, so
  $\mathbb E\Vert P_\perp T\Vert^2/\Vert T\Vert^2=(1-\chi/d^2)^2$. For `bin8` with full bottom bonds and a
  deficient top cut the same holds on the top cut with $d^2\to d^4$; with deficient bottom bonds the
  residual is larger.
- Known analogue: lazy training of over-parameterized CP needs $\Omega(d^{l-1})$ components (Wang, Wu,
  Lee, Ma & Ge 2020), and lazy (NT) models fit only the projection of the target on their tangent
  space (Ghorbani et al. 2019; Chizat et al. 2019 Thm 2.5).

**R3 (derived; empirical).** $\Theta\approx\bar\lambda I$ needs every $\Phi_c\Phi_c^\top$ and
$\Psi_v\Psi_v^\top$ near-isotropic, i.e. $\chi_e\gg D_e$ (Wishart spread $\approx2\sqrt{D_e/\chi_e}$).
At capped full bonds ($\chi=d^2$) the bottom cores are square Gaussian matrices and $\Theta$ is badly
conditioned (§6 item 1). The lazy fit converges in $\tau\approx(\bar\lambda/\lambda_{\min})\ln(1/{\rm tol})$,
so **lazy-regime runs should use $\chi\ge2d^2$ with `cap=False`.**

### 2.2 Laziness constants: first and second order (R4, R5)

The criterion of Chizat, Oyallon & Bach (2019, eq. 1),
$\kappa=\Vert h(w_0)-y^\star\Vert\,\Vert D^2h\Vert/\Vert Dh\Vert^2$, is $\propto x$ for the centred TTN (known
framework). What is new is the TTN constants and the fact that the first-order term is not the one that
sets the boundary.

**R4 (derived; $N=4$; isotropic limit $\chi\gg d^2$; leading order in $d^2/\chi$).** Let
$p:=\chi\sigma^2/d^2$ (the eigenvalue scale of $AA^\top$ and $CC^\top$), so $\bar\lambda\approx p^2$ under SP.
The lazy displacement is $\delta\theta_1=J^+T\approx J^\top T/\bar\lambda$.

- *First order* (v1): $\delta\Theta_1=\frac{2p}{\bar\lambda}(S_L\otimes I+I\otimes S_R)$ with
  $S_L=TW_0^\top+W_0T^\top$, $S_R=T^\top W_0+W_0^\top T$. It is linear in $W_0$, so it has **zero mean**
  over inits; its size is $c_1x$ with $c_1=\frac{4d^3\sqrt{d^2+1}}{\chi^3}$ (SP).
- *Second order, coherent.* Two terms of equal size, both $\propto TT^\top$ and independent of the sign of
  $W_0$:
  (i) curvature: $\tfrac12D^2\Theta[\delta\theta_1,\delta\theta_1]$, dominated by
  $A\,\delta B\,\delta B^\top A^\top\approx p^3TT^\top/\bar\lambda^2$ in the bottom-core kernel $NN^\top$
  (and the mirror term in $M^\top M$);
  (ii) feature drift: during training $\dot A=r(t)M(t)^\top$ with $r(t)\approx e^{-\bar\lambda t}T$ and
  $M(t)-M_0\approx pA^\top T(1-e^{-\bar\lambda t})/\bar\lambda$, which integrates to
  $\delta A_{\rm drift}=pTT^\top A/(2\bar\lambda^2)$: the bottom features rotate toward the singular
  directions of $T$ and change the dominant root kernel $AA^\top\otimes CC^\top$ by
  $p^3TT^\top\otimes I/\bar\lambda^2$ (and $I\otimes T^\top T$ from $C$).
  Total: $\Delta\Theta_2=\frac{2p^3}{\bar\lambda^2}(TT^\top\otimes I+I\otimes T^\top T)$.
- Hence
  $$\delta_K^2\approx(c_1x)^2+2\xi c_1c_2x^3+(c_2x^2)^2,\qquad c_2=\frac{2d^4\sqrt{2d^2\nu+2}}{\chi^3}\ \ (\text{SP}),$$
  with a seed-dependent correlation $\xi\in[-1,1]$ between the two terms (zero mean; measured $-0.7$ and
  $+0.8$ at $\chi=32$, §6 item 2). For $d=3$, weights $(3,2,1)$: $c_1=341.5/\chi^3$, $c_2=537/\chi^3$,
  $c_2/c_1=1.57$, so **the second-order term dominates once $x\gtrsim0.6$**, i.e. at every lazy boundary
  with $\chi\gtrsim25$.
- *NTK parameterization* ($\eta_A=\eta_C=1/d^2$, $\eta_B=1/\chi^2$; all three cores contribute to the
  coherent term): $c_1=\frac23\,d\sqrt{d^2+1}/\chi$ and
  $c_2=\frac{10}{27}\,d^2\sqrt{2d^2\nu+2}/\chi$; for $d=3$, $(3,2,1)$: $c_1=6.32/\chi$, $c_2=11.06/\chi$.
  This explains why the first-order NTK value is about half the measured $\delta_K$ at $x=1.11$:
  $\sqrt{(c_1x)^2+(c_2x^2)^2}=0.240$ ($\chi=64$) and $0.120$ ($\chi=128$) against GF $0.21$–$0.23$ and
  $0.11$–$0.12$ (verifier; first order alone: $0.11$, $0.055$).
- Assumptions: isotropic init Grams ($\chi\gg d^2$), isotropic lazy relaxation, terms of relative order
  $d^2/\chi$ and $d^6x^2/\chi^3$ dropped. At $\chi=16$ the seed spread of $c_1$ is $\pm40\%$ around a mean
  twice the isotropic value; at $\chi\ge32$ the formulas hold to $\sim5$–$30\%$ (§6 item 2).
- Context (known in other models): the kernel change of a wide network has a zero-mean fluctuating part
  and a coherent part that is second order in the feature-learning strength (the $O(\gamma_0^2)$ NTK
  change of Bordelon & Pehlevan 2022); Geiger et al. (2020) measure $\Vert\Delta\Theta\Vert\propto(\sqrt h\alpha)^{-a}$
  with $1<a\le2$, matching the crossover from slope 1 to slope 2 here.

**R5 (derived).** Two-copy model, $\theta_\pm=\theta_0+s\pm\delta$: the antisymmetric part
$\delta=\delta\theta_1/2$ cancels the first-order kernel change; the symmetric part $s$ is the drift,
$s=\tfrac14\delta A_{\rm drift}$ (the residual decays at $2\bar\lambda$). So the second-order change is half
the centred one and, with $\Vert\Theta_2(0)\Vert=2\Vert\Theta_0\Vert$,
$$\delta_K^{(2)}=\tfrac14c_2x^2+O(x^4),\qquad\delta_\theta^{(2)}=\tfrac12\delta_\theta^{(\rm c)}+O(x^2).$$
(The $O(x^3)$ term vanishes by the $\delta\to-\delta$ symmetry.) Pitfall: compute the two-copy kernel on
the trainable (masked) parameters, $J\leftarrow J\cdot{\rm mask}$.

### 2.3 Lazy boundary (R6)

$\sigma_{\rm lazy}=(\Vert T\Vert/x_c)^{1/K}$ with $\delta_K(x_c)=\delta_*$.

- **Centred, $N=4$ (derived).** Solving R4 with $\xi=\pm1$ brackets the boundary:
  $$x_c^{\pm}=\frac{\mp c_1+\sqrt{c_1^2+4c_2\delta_*}}{2c_2},\qquad x_c\in[x_c^+,x_c^-].$$
  When $c_1^2\ll c_2\delta_*$ (large $\chi$), $x_c\to(\delta_*/c_2)^{1/2}$ and
  $$\sigma_{\rm lazy}\to\Big(\frac{\Vert T\Vert^2c_2}{\delta_*}\Big)^{1/6}=\Big(\frac{2d^4\sqrt{2d^2\Sigma_rs_r^4+2(\Sigma_rs_r^2)^2}}{\delta_*}\Big)^{1/6}\chi^{-1/2}=6.50\,\chi^{-1/2}$$
  ($d=3$, $(3,2,1)$, $\delta_*=0.1$). The first-order boundary $x_c=\delta_*/c_1$ is valid only while
  $c_1\gtrsim\delta_*$ (here $\chi\lesssim16$); v1's P4 used it at all $\chi$ and gave
  $\sigma_{\rm lazy}\propto\chi^{-1}$, which is wrong.
- **Two-copy (derived).** $x_c^{(2)}=2(\delta_*/c_2)^{1/2}$, so $\sigma^{(2)}_{\rm lazy}=2^{-1/3}\sigma_{\rm lazy}\to5.16\,\chi^{-1/2}$.
- **Threshold sensitivity.** With local slope $s=d\ln\delta_K/d\ln x\in[1,2]$ at the boundary,
  $\sigma_{\rm lazy}\propto\delta_*^{-1/(Ks)}$: changing $\delta_*$ to $0.05$ or $0.2$ moves the boundary by
  $\times2^{\pm1/(Ks)}$, between $\times1.26^{\pm1}$ ($s=1$) and $\times1.12^{\pm1}$ ($s=2$) for $K=3$.
  Measured local slopes at $\delta_K\approx0.1$ are 1.5–1.8 for both models at $\chi=32$ and 64 (§6 item 2;
  verifier: centred 1.0 at $\chi=16$, two-copy 1.25), so the two-copy boundary is **not** generally sharper
  (v1 claim withdrawn).

GF values: `bin4`, $d=3$, $(3,2,1)$, SP, seeds 0 and 1 (§6 item 2); predictions use the isotropic $c_1,c_2$
(first row: first order with the measured mean $c_1$).

| $\chi$ | model | $x_c$ predicted | $x_c$ GF | $\sigma_{\rm lazy}$ predicted | $\sigma_{\rm lazy}$ GF |
|---|---|---|---|---|---|
| 16 | centred | 0.62 (first order, measured $c_1=0.162$) | 0.44–0.65 (verifier) | 1.82 | 1.80–2.04 |
| 32 | centred | $[2.17,2.81]$ | 2.82, 2.25 (verifier 2.0–2.8) | $[1.10,1.19]$ | 1.10, 1.18 |
| 64 | centred | $[6.67,7.31]$ | 6.73, 7.53 (verifier 6.2–7.5) | $[0.81,0.83]$ | 0.82, 0.79 |
| 128 | centred | $[19.4,20.1]$ | — | 0.57–0.58 | — |
| 32 | two-copy | 4.94 | 5.15, 4.96 (verifier 4.1–5.2) | 0.91 | 0.90, 0.91 |
| 64 | two-copy | 13.97 | 13.6, 14.8 | 0.645 | 0.65, 0.63 |

### 2.4 Deficient bonds (R7)

If some bond is not full, $W^\star+W_0$ generically has bond rank $\operatorname{rank}W^\star_{(e)}+\chi_e>\chi_e$:
**the centred model is unrealizable at every $\sigma$** (derived).

- **Centred floor, small $\sigma$ (derived; first order in $W_0$ around the smooth point $W^\star$ of
  $\mathcal M_\chi$, valid when $\sigma^K\ll$ the smallest teacher Schmidt value).** The best reachable
  point is $W^\star+P_TW_0$ up to $O(\Vert W_0\Vert^2)$ ($P_T$, $P_\perp$: tangent and normal projectors
  at $W^\star$), so $L_{\rm end}\approx\tfrac12\Vert P_\perp W_0\Vert^2$ (for $N=4$ exactly the Eckart–Young
  minimum), with mean, by R1b,
  $$\mathbb E\,\frac{L_{\rm end}}{L_0}\approx\Big(1-\frac{\dim\mathcal M_\chi}{D}\Big)\frac{\sigma^{2K}}{\Vert W^\star\Vert^2}$$
  (seed median $\approx0.7\times$ the mean, §6 item 3). A typical run can then satisfy the convergence
  rule ($L<10^{-3}L_0$) only if $\sigma<\sigma_{\rm floor}=[10^{-3}\Vert W^\star\Vert^2/(1-\dim\mathcal M_\chi/D)]^{1/(2K)}$:
  `bin4`, $d=3$, $(3,2,1)$: $\sigma_{\rm floor}=0.56,0.60,0.71,1.02$ for $\chi=3,4,6,8$
  ($\dim\mathcal M_\chi=18\chi-\chi^2$); `bin8`, $d=3$, $\chi=3$ ($\dim\mathcal M_\chi=117$, $D=6561$):
  $\sigma_{\rm floor}=0.74$.
- **Centred floor, large $\sigma$ (derived).** $L_{\rm end}/L_0\to\Vert P_\perp T\Vert^2/\Vert T\Vert^2$
  (normal component of $T$ at $W_0$), $\approx(1-\chi/d^2)^2$ for $N=4$.
- **Two-copy** (bond $2\chi_e$; realizable iff $2\chi_e\ge\operatorname{rank}W^\star_{(e)}$). At large
  $\sigma$: (i) lazy fit of the tangent part ($\tau\sim1$–$100$), (ii) plateau at
  $L/L_0=\Vert P_\perp T\Vert^2/\Vert T\Vert^2$, (iii) escape by rotating the bond subspaces.

**Escape mechanism (derived by adiabatic elimination; explicit for $N=4$).** Write
$\theta_\pm=\theta_0+s\pm\delta$, so $W=2J(\theta_0+s)\delta+O(\delta^3)$. The antisymmetric part
$\delta$ relaxes on the fast time $1/\bar\lambda$ to the tangent fit, leaving the residual
$P_\perp(\theta_0+s)T$. The symmetric part is slow and follows
$\dot s=-\tfrac12\nabla_sL_{\rm eff}(s)$ with $L_{\rm eff}(s)=\tfrac12\Vert P_\perp(\theta_0+s)T\Vert^2$
(envelope theorem; the $\tfrac12$ because $s$ is the mean of two unit-rate copies). For $N=4$,
$L_{\rm eff}=\tfrac12\Vert P_A^\perp TP_C^\perp\Vert^2$ and
$$\dot A=\tfrac12P_A^\perp TP_C^\perp T^\top(A^+)^\top,\qquad\dot C=\tfrac12P_C^\perp T^\top P_A^\perp T(C^+)^\top .$$
With $A=\sigma U$, $C=\sigma V$ this is a $\sigma$-free flow in $t'=t/(2\sigma^2)$, hence
$$t_{\rm esc}=\frac{2\theta\sigma^2}{\Vert T\Vert^2},\qquad\tau_{\rm esc}=\bar\lambda_2t_{\rm esc}=\frac{4\theta\,\sigma^6(2d^2+\chi^2)}{d^4\Vert T\Vert^2}\quad(N=4),$$
where $\theta$ (dimensionless) is the time the $\sigma$-free reduced flow needs from the init subspaces to
the chosen loss level; it depends on the seed and the level, not on $\sigma$ or $\Vert T\Vert$.

- There is no saddle: $\nabla_sL_{\rm eff}\ne0$ at a generic init, so after the plateau $L$ first falls
  linearly in $t$, not exponentially. v1's exponential-rate heuristic and $\sigma^3\ln\sigma$ law are wrong.
- General trees (heuristic): the same reduction applies with $L_{\rm eff}=\tfrac12\Vert P_\perp(\theta_0+s)T\Vert^2$;
  rotating a core's subspaces by $O(1)$ needs $\Vert s_v\Vert\sim\sigma$ while
  $\Vert\nabla_sL_{\rm eff}\Vert\sim\Vert T\Vert^2/\sigma$, so $t_{\rm esc}\propto\sigma^2/\Vert T\Vert^2$ and
  $\tau_{\rm esc}\propto\sigma^{2K}$. At $K=7$, $\tau_{\rm esc}$ grows by $2^{14}\approx1.6\times10^4$ per
  doubling of $\sigma$: in kernel time the escape is effectively absent.
- Known analogues: mixed lazy/active dynamics in two-layer linear networks (Tu, Aranguri & Jacot 2024),
  lazy-then-rich transitions (Lyu et al. 2024, with weight decay). The reduction to a Grassmannian
  gradient flow of the bond subspaces with a $\sigma^2$ time scale is, as far as we know, new.

"Large $\sigma$ ⇒ lazy" holds only with full bonds. For `bin8` with $d=3$ the top cut has dimension 81:
for $9\le\chi<81$ the centred model at large $\sigma$ ends near $L/L_0=(1-\chi/81)^2$ (0.79 at $\chi=9$, 0.49 at
$\chi=24$), and for $\chi<9$ higher still. With $d=2$, full bonds need $\chi\ge16$.

## 3. (b) Saddle-to-saddle boundary

### 3.1 Aligned (decoupled) odeco dynamics (R8)

**Known.** With `aligned_init` (amplitude $a_0$ on every core and component) each component follows
balanced scalar dynamics; with $u=a^K$,
$$\dot u=K\,u^{2-2/K}(s-u),\qquad u(0)=u_0=a_0^K .$$
This is Saxe, McClelland & Ganguli (2014) eq. (15) with $K$ weight matrices; the same law holds for the
singular values of aligned deep matrix factorization (Arora, Cohen, Hu & Luo 2019) and is eq. (4) of
Gissin, Shalev-Shwartz & Daniely (ICLR 2020), whose depth-normalized loss removes the factor $K$. Its
$K\to\infty$ form $\dot u\propto u^2(s-u)$ is Saxe et al. eqs. (16)–(17) and Gissin et al. Thm 1. Gissin
et al. Thm 2: for two modes with ratio $r=s_i/s_{i+1}>1$ and thresholds $\epsilon\in(0,\frac14)$,
$f\in(\frac34,1)$, the largest common init that still gives "$(\epsilon,f)$-incremental" learning (some
time with $a_{i+1}\le\epsilon$ and $a_i\ge f$) satisfies, for $K\ge3$,
$\epsilon s_{i+1}\big[\tfrac{(1-f)(r-1)}{1+(1-f)(r-1)}\big]^{K/(K-2)}\le u_0^{\rm th}\le\epsilon s_{i+1}\big[\tfrac{r-1}{r-\epsilon}\big]^{K/(K-2)}$
(polynomial in the gap; exponentially small at $K=2$); the per-layer threshold $(u_0^{\rm th})^{1/K}\to1$
as $K\to\infty$ (`notes/literature.md`). So "$a_{0c}\to1$" (H4 for aligned init) is known.

**New here (proven, given the aligned reduction, which is exact for odeco teachers and `aligned_init`):**
the boundary for the project's criterion (separation ratio $\rho_i>3$ with 10–90% widths,
`notes/diagnostics.md`), its constants, and the exact $K\to\infty$ limit. With $v=u/s$ and
$I_K(v_0,v_1)=\frac1K\int_{v_0}^{v_1}\frac{v^{-2+2/K}}{1-v}\,dv$:
$$t_r=s_r^{-(2-2/K)}I_K(u_0/s_r,\tfrac12),\qquad w_r=s_r^{-(2-2/K)}\omega_K,\quad\omega_K=I_K(0.1,0.9),$$
$$\rho_i=\frac{I_K(u_0/s_{i+1},\frac12)-q_i^{\,2-2/K}I_K(u_0/s_i,\frac12)}{\omega_K},\qquad
\rho_i\simeq\Big(\frac{u_0}{s_{i+1}}\Big)^{-(1-2/K)}\frac{1-q_i}{(K-2)\,\omega_K}\ (u_0\to0),$$
so $u_{0c}\simeq s_{i+1}[(1-q_i)/(3(K-2)\omega_K)]^{K/(K-2)}$: Gissin's exponent $K/(K-2)$ with the
constant of our criterion. $\omega_3=2.009$, $\omega_4=1.800$, $\omega_5=1.612$, $\omega_7=1.318$,
$\omega_{10}=1.025$, $\omega_{15}=0.743$.

**$K\to\infty$ (exact).** $\lim_KK\,I_K(v_0,v_1)=I_\infty(v_0,v_1):=\frac1{v_0}-\frac1{v_1}+\ln\frac{v_1}{v_0}+\ln\frac{1-v_0}{1-v_1}$,
$K\omega_K\to I_\infty(0.1,0.9)=\frac{80}9+2\ln9=13.283$, and
$\rho_i\to[I_\infty(u_0/s_{i+1},\frac12)-q_i^2I_\infty(u_0/s_i,\frac12)]/13.283$. The function-space
boundary stays finite: $u_{0c}(\infty)=6.531\times10^{-3}$, $4.122\times10^{-3}$, $2.297\times10^{-3}$ for
weights $(1,q)$ with $q=0.5,0.8,0.9$, and $1.306\times10^{-2}$ for $(3,2,1)$ ($\rho_2$ binds). The
small-$u_0$ asymptote $s_iq(1-q)/39.85$ (v1's "limit") is 2–4% lower. Convergence in $K$ is slow
($q=0.5$: $6.26\times10^{-3}$ at $K=101$, $6.50\times10^{-3}$ at $K=1001$), while $a_{0c}=u_{0c}^{1/K}=1+\ln u_{0c}/K+O(K^{-2})$.

Per-core boundary $a_{0c}$ for two components $(1,q)$ (exact $u_{0c}$ in brackets):

| $K$ | $q=0.5$ | $q=0.8$ | $q=0.9$ |
|---|---|---|---|
| 3 | 0.061 ($2.23\times10^{-4}$) | 0.030 ($2.6\times10^{-5}$) | 0.016 ($3.9\times10^{-6}$) |
| 5 | 0.282 | 0.229 | 0.186 |
| 7 | 0.435 ($2.92\times10^{-3}$) | 0.387 ($1.30\times10^{-3}$) | 0.342 ($5.5\times10^{-4}$) |
| 15 | 0.700 ($4.75\times10^{-3}$) | 0.673 | 0.643 |
| 101 | 0.951 ($6.26\times10^{-3}$) | 0.947 | 0.941 |
| $\infty$ | 1 ($6.531\times10^{-3}$) | 1 ($4.122\times10^{-3}$) | 1 ($2.297\times10^{-3}$) |

Teacher $(3,2,1)$ (binding separation in brackets): $u_{0c}=2.85\times10^{-4}$ ($\rho_1$), $3.56\times10^{-3}$,
$5.85\times10^{-3}$, $8.28\times10^{-3}$, $9.50\times10^{-3}$, $1.131\times10^{-2}$, $1.306\times10^{-2}$ and
$a_{0c}=0.066,0.324,0.480,0.647,0.733,0.865,1$ for $K=3,5,7,11,15,31,\infty$ ($\rho_2$ binds for $K\ge5$).

### 3.2 Random fan-in init (R9)

- **First event (derived).** The near-origin vector field is homogeneous of degree $K-1$, so
  $t_1=\sigma^{-(K-2)}T_1(\hat\theta_0)+O(1)$ (the TTN corollary of Kumar & Haupt 2025a,
  `notes/literature.md`).
- **Later gaps (heuristic).** $t_{i+1}-t_i\propto\sigma^{-(K-2)}$ if the dormant amplitudes entering each
  plateau stay $\propto\sigma$: exact for aligned init, not guaranteed in general (earlier phases can
  rescale them; `notes/literature.md` H2 iii). Widths are $O(1)$, so $\rho_i\approx R_i\sigma^{-(K-2)}$
  with seed-dependent $R_i$. Empirical: at $K=3$ the median of $\rho\sigma$ over 30 seeds is 0.93, 0.90,
  0.84 at $\sigma=0.06,0.12,0.24$, and the verifier found $\rho\propto1/\sigma$ for 30/30 seeds down to
  $\sigma=0.00375$; at $K=7$ the local exponent is 5.0–6.8 (below).
- **Transients (heuristic; replaces v1's P10).** Binary $N=4$ is a depth-3 deep linear network
  (theory_saddles §4.2), so the first escape converges to the top singular pair. In the rank-one
  reduction $W=\lambda uv^\top$ the flow is $\dot u=\lambda^{1/3}(I-uu^\top)Tv$,
  $\dot\lambda=3\lambda^{4/3}u^\top Tv$: the mixing angle with the second pair decays like
  $(\lambda/\lambda_0)^{-(s_1-s_2)/(3s_1)}$ while $\lambda$ grows from $\sim\sigma^3$ to $\sim s_1$, so at the
  end of the escape $\varepsilon\propto\sigma^{1-q}$ and the transient overlap peaks at
  $a_2\approx(s_1/s_2)\varepsilon^2\propto\sigma^{2(1-q)}$ ($\sigma^{2/3}$ for weights $3,2$). The verifier
  measured peaks $0.49,0.27,0.12,<0.09$ at $\sigma=0.06,0.015,0.00375,0.001$ (local exponents
  0.43–0.59). For close weights the decay is slow ($\sigma^{0.2}$ at $q=0.9$), so transients persist
  over any practical range. Consequences: first-crossing widths misclassify (§6 item 4: S2S fraction
  0.80 with first-crossing widths vs 0.97 with final-rise widths at $\sigma=0.06$); with final-rise
  widths every seed has $\rho\propto\sigma^{-(K-2)}$, but seeds with mixed first escapes approach it
  slowly (verifier, one seed: $\rho\sigma=0.25\to0.51$ as $\sigma=0.12\to0.0075$).
- **$K=3$ boundary is a seed distribution (empirical; `bin4`, $d=3$, $\chi=3$, $(3,2,1)$, plain model, 30
  seeds).** S2S fraction 0.97, 0.87, 0.63 at $\sigma=0.06,0.12,0.24$. Per-seed boundary
  $\sigma_{c}=\sigma\rho_{\min}/3$ (using $\rho\propto1/\sigma$): from $\sigma=0.06$, median 0.31, quartiles
  0.22–0.39, range 0.04–1.7; from $\sigma=0.24$, median 0.27 (0.20–0.36). In aligned units
  $\kappa_3:=a_{0c}/\sigma_c=0.21$–$0.24$ (median), and $c_3:=\sqrt D\,u_{0c}/\sigma_c^3=0.09$–$0.13$ (v1:
  $\kappa_3=0.27$ from two pre-fix seeds).
- **$K=7$ with full bonds (empirical; `bin8`, $d=2$, $\chi=16$, weights $(3,2)$, plain, fan-in, 12
  seeds).** S2S fraction $10/12$ at $\sigma=0.5$, $7/12$ at $0.7$, $0/2$ at $0.9$. Between $\sigma=0.5$ and
  $0.7$ the local exponent of $\rho$ is 5.0–6.8 (prediction $K-2=5$). Per-seed $\sigma_c$ (interpolated):
  median 0.71, quartiles 0.62–0.74, range 0.46–1.09. The smaller component is learned first in 5/12
  seeds (tensor-like pair, H1), without a merged event.
- **Effective amplitude (heuristic; conjecture).** Treat random init as aligned init with per-core
  amplitude $\kappa_K\sigma$, so $\sigma_c(K)=a_{0c}(K)/\kappa_K$; equivalently $u_0^{\rm eff}=c_K\sigma^K/\sqrt D$
  with $c_K=\sqrt D\,\kappa_K^K$ (by R1b the RMS overlap of $W_0$ with a unit rank-one tensor is exactly
  $\alpha/\sqrt D$). Measured medians: $\kappa_3=0.21$–$0.24$ (`bin4`, $d=3$, $(3,2,1)$, $a_{0c}=0.066$) and
  $\kappa_7=0.69$ (quartiles 0.66–0.79; `bin8`, $d=2$, $(3,2)$, $a_{0c}=0.488$), i.e. $c_3\approx0.1$,
  $c_7\approx1.2$. **Conjecture (v1, now with support):** $\kappa_K$ grows toward 1 with $K$, so
  $\sigma_c(K)\to a_{0c}(K)\to1$ (H4 for random init). For `bin8` with $d=3$, $\chi=3$ and weights $(3,2,1)$
  ($a_{0c}=0.480$) this would put the boundary at $\sigma_c\approx0.6$–$0.75$, but there it is masked by
  stalls (§3.3). v1's $0.86$–$1.0$ used an arithmetic slip ($\sqrt D\,u_{0c}=81\times5.85\times10^{-3}=0.474$,
  not 0.54) and an untested calibration.

### 3.3 $K\ge5$ with $\chi$ = teacher rank: very long last plateaus and stalls (R10)

Empirical (§6 item 5). `old5` ($K=5$), $\sigma=0.65$, 4 seeds: fan-in init converges in 2/4 seeds (last
event at $t=62$ and $2524$); in the other 2 the last component never starts (to $t=10^5$; smallest
third bond singular value $8\times10^{-18}$ and $9\times10^{-16}$). Gauge-balanced init (same $W_0$,
$Q_e=0$): 4/4 converge, last events at $t=397,172,1151,74$, $\rho_{\min}=30,4.7,21,1.3$. `bin8` ($K=7$),
$\sigma=0.9$, 2 seeds: fan-in init never learns the last component (seed 0 frozen from $t=1500$ to $10^5$,
smallest third bond singular value $5.6\times10^{-9}$; seed 1: $2\times10^{-16}$). Balanced init does not
rescue $K=7$: 2/2 seeds still lack the last component at $t=10^5$, with the third bond
singular value decaying slowly instead of freezing (seed 0: $6.7\times10^{-4}$ at $t=1500$,
$1.3\times10^{-4}$ at $t=10^5$). By contrast, with all bonds full at $d=2$ (`bin8`, $\chi=16$, weights
$(3,2)$) all 26 runs converge by $t=125$ (§3.2), and with spare channels at $d=3$ ($\chi=6=2R$, one
fan-in seed) the last component is learned at $t\approx334$.

Mechanism (heuristic; the conservation-law trap of `notes/theory_saddles.md` §8). With $\chi$ equal to
the teacher rank $R$, the last component must use the last free channel of every bond, through a route
of all $k=K$ cores. Its plateau lasts $\Delta t\approx\bar b^{-(K-2)}/((K-2)s_RA)$, where $\bar b$ is the
geometric mean of the dormant route amplitudes; $\bar b^K$ is a product of $K$ random overlaps, so
$\ln\Delta t$ has a seed spread that grows like $\sqrt K$. With fan-in init the root edges have
$\mathbb EQ_v=\sigma^2(1-1/\chi_v)I\succ0$ (child-heavy), so the root's free channel decays at a rate
$\sim\sigma^2$ while the escape needs $\sim\sigma^{-(K-2)}$: for $K\ge5$ the route can die (measured on
`old5` seed 0: the root-side third singular value falls from 0.035 to 0.006 before the escape at $t=2524$;
balanced init keeps it at $\approx0.27$ and escapes at $t=397$). Balanced init removes the dead routes at
$K=5$ but not the long plateaus, and at $K=7$ the last route still decays slowly; why it decays (a
wrong-sign route that must first pass through zero is one candidate) is open.

Consequences: at $K\ge5$ and $\chi=R$, "stalled" dominates any fixed time budget with fan-in init, so
the S2S boundary cannot be calibrated there; v1's $K=5$ and $K=7$ numbers are withdrawn. Use converged
runs only and report the stalled fraction; to measure the boundary use spare channels ($\chi\ge2R$; at
$K=7$: no stalls in 26 runs with $d=2$, $\chi=16$, and 1/1 with $d=3$, $\chi=6$), balanced init with long
budgets (LSODA makes plateaus cheap), or teachers whose later components need fewer cores.

### 3.4 Plain vs centred model at $N=8$

The centred model needs full bonds (R7): at `bin8`, $d=3$, $\chi<81$ it is unrealizable, and a typical
run cannot reach $10^{-3}L_0$ once $\sigma>\sigma_{\rm floor}=0.74,0.75,0.78$ for $\chi=3,9,24$
($\dim\mathcal M_\chi=117,1377,3312$), exactly where the $K=7$ boundary is expected. So map the
$K=7$ S2S boundary in the plain model (realizable for $\chi\ge R$), or use $d=2$ with $\chi\ge16$ (all
`bin8` cuts full: cut dimensions 4 and 16; an odeco teacher then has at most 2 components). In the plain
model $\Vert W_0\Vert\approx\sigma^7$ reaches $\Vert W^\star\Vert=3.74$ at $\sigma=1.21$, so near the boundary
the run must also remove $W_0$.

## 4. (c) Large bond dimension $\chi\to\infty$

### 4.1 What is known (reconciled with `notes/literature.md` v1.1)

- **MPS NTK.** Guo & Draper (2021b, arXiv:2111.14046): periodic MPS with entry variances
  $\sigma_i^2/\sqrt{|a_i||a_{i+1}|}$ and per-element learning rates $(|a_i||a_{i+1}|)^{-1/2}$; as the bond
  dimensions go to infinity one after another, the NTK converges to a deterministic sum-of-products kernel,
  is positive definite and stays constant during training (lazy); proofs informal. This is the MPS
  version of our NTK row (R12), which adds the $\chi^{-1}$ rate and the second-order term.
- **GP limits.** Guo & Draper (2021a) for MPS (informal; internally inconsistent per
  `notes/literature.md`); Wesel & Batselier (AISTATS 2025) for CPD- and TT-constrained kernel machines
  (product-kernel GP). R1b gives the exact second moment at every $\chi$ and on every tree.
- **muP-style rules for structured layers.** Qiu et al. (ICML 2024; TT and block-TT layers) and
  Potapczynski et al. (NeurIPS 2024; Einsum layers) apply the spectral condition of Yang, Simon &
  Bernstein (2023) to each dense factor, with init std $\propto\sqrt{\min(n_{\rm in},n_{\rm out})}/n_{\rm in}$
  and per-factor learning-rate multipliers; heuristic, validated empirically, for structured matrices
  used as linear layers on arbitrary vectors inside standard networks. In a TTN used as the model, each
  core acts only on product vectors, which changes the relevant norm (4.2): their factor-wise rule is
  the "matrix rule" of 4.3, whose initial fit slows down about linearly in $\chi$ for a TTN (4.4).
- **Spectral condition** (Yang, Simon & Bernstein 2023): $\Vert W_\ell\Vert_*\asymp\Vert\Delta W_\ell\Vert_*\asymp\sqrt{n_\ell/n_{\ell-1}}$;
  its init clause assumes fan-out $\ge$ fan-in, which TTN cores violate. Tensor Programs (Yang & Hu 2021)
  exclude multilinear models. No feature-learning (muP, mean-field, DMFT) limit of TTNs or MPS used as
  the model is known (`notes/literature.md` §2.3).
- **Polynomial networks:** the NTK moves by $O(1/n)$ (Dyer & Gur-Ari 2020; Aitken & Gur-Ari 2020;
  *[not re-verified]*), consistent with R12's $\chi^{-1}$ under NTK scaling.

### 4.2 The injective-norm spectral condition (R11, derived)

A core's input is always a product vector $h_{c_1}\otimes h_{c_2}$, and its gradient
$\mathbb E[r\,b_v\otimes h_{c_1}\otimes h_{c_2}]$ is a sum of rank-one product tensors ($r$ residual,
$b_v=\partial f/\partial h_v$). So the norm that controls $\Vert G_v(\Delta h\otimes h)\Vert$ and
$\Vert\Delta G_v(h\otimes h)\Vert$ is the **injective norm**
$\Vert G\Vert_{\rm inj}=\sup_{\Vert a\Vert=\Vert b\Vert=1}\Vert G(a\otimes b)\Vert$, not the spectral norm of
the $\chi_v\times F_v$ matricization.

- Condition: $\Vert G_v\Vert_{\rm inj}\asymp\Vert\Delta G_v\Vert_{\rm inj}\asymp\sqrt{\chi_v/F_v}$.
- Gaussian core with entry std $t_v$: $\Vert G_v\Vert_{\rm inj}\asymp t_v\sqrt{\chi_v+\sum_cm_c}$ ($m_c$ =
  child-leg dimensions; Tomioka & Suzuki 2014 *[not re-verified]*), so relative to fan-in init the
  multiplier is $m_v=\sqrt{\chi_v/(\chi_v+\sum_cm_c)}$: $\Theta(1)$ for every non-root core (fan-in init
  already satisfies the condition) and $(\sum_cm_c)^{-1/2}$ for the root ($(2\chi)^{-1/2}$ binary,
  $(d+\chi)^{-1/2}$ MPS).
- Learning rates from the update condition: $\eta_v\asymp\chi_v/F_v$.
- Check (§6 item 7): for a fan-in binary core, $\sqrt\chi\Vert G\Vert_{\rm inj}\to2.8$ while the matrix
  spectral norm stays at 1.1–1.25.

**The factor-wise matrix rule is wrong for TTNs used as models.** It gives $m_v=\chi^{-1/2}$ for binary
internal cores and $1/\chi$ for the binary root; by R1b the init kernel then vanishes, e.g.
$\mathbb E\Theta=(4\chi+3\chi^2)/(\chi^4D)\,I\to3/(\chi^2D)\,I$ for `bin8` at $\sigma=1$, versus $4/D\,I$ under
muP-TTN.

### 4.3 Parameterizations (gradient-flow form)

Init $G_v=\sigma m_vZ/\sqrt{F_v}$, rates $\eta_v$; rescale time by $\bar\lambda_\eta=\operatorname{tr}\Theta_\eta(0)/D$.
Each cell gives $(m_v,\ \eta_v)$:

| core type | $F_v$ | $\chi_v$ | SP | NTK | **muP-TTN** | matrix rule (avoid) |
|---|---|---|---|---|---|---|
| bottom (binary, 2 phys) | $d^2$ | $\chi$ | $1,\ 1$ | $1,\ 1/d^2$ | $1,\ \chi/d^2$ | $1,\ \chi/d^2$ |
| internal binary | $\chi^2$ | $\chi$ | $1,\ 1$ | $1,\ 1/\chi^2$ | $1,\ 1/\chi$ | $\chi^{-1/2},\ 1/\chi$ |
| binary root | $\chi^2$ | 1 | $1,\ 1$ | $1,\ 1/\chi^2$ | $(2\chi)^{-1/2},\ 1/\chi^2$ | $1/\chi,\ 1/\chi^2$ |
| MPS first / HT leaf | $d$ | $\chi$ | $1,\ 1$ | $1,\ 1/d$ | $1,\ \chi/d$ | $1,\ \chi/d$ |
| MPS middle | $d\chi$ | $\chi$ | $1,\ 1$ | $1,\ 1/(d\chi)$ | $1,\ 1/d$ | $d^{-1/2},\ 1/d$ |
| MPS root | $d\chi$ | 1 | $1,\ 1$ | $1,\ 1/(d\chi)$ | $(d+\chi)^{-1/2},\ 1/(d\chi)$ | $(d\chi)^{-1/2},\ 1/(d\chi)$ |

Kernel shares at init (R1b; $\propto\eta_vF_v/s_v^2$): SP root-dominated ($\chi^2/(2d^2+\chi^2)$ for $N=4$);
NTK $1/K$ each; muP-TTN $(\tfrac14,\tfrac14,\tfrac12)$ for $N=4$ and bottoms/internals/root
$\tfrac12/\tfrac14/\tfrac14$ in total for $N=8$. Under muP-TTN $f_0\sim\chi^{-1/2}\to0$ and every core's update
is $O(1)$ from the start.

### 4.4 $\chi$-scaling (R12; $N=4$, centred, fixed $\sigma$; derived in the isotropic limit)

Both $c_1$ and $c_2$ of R4 carry the same power of $\chi$, so the exponents hold at fixed $x$ beyond first
order:

| | $\delta_K$ | $\delta_\theta$ | root (Frobenius) | bottom cores |
|---|---|---|---|---|
| SP | $\chi^{-3}$ | $\chi^{-3/2}$ | $\chi^{-1}$ | $\chi^{-5/2}$ |
| NTK | $\chi^{-1}$ | $\chi^{-1/2}$ | $\chi^{-1}$ | $\chi^{-1/2}$ |
| muP-TTN | $\chi^0$ | $\chi^0$ | $\chi^{-1/2}$ (Frobenius), $\chi^0$ (spectral) | $\chi^0$ |

First-order prefactors (v1): SP $\delta_\theta\approx xd^2/(\sqrt2\chi^{3/2})$, root $xd^2/\chi$, bottoms
$xd^3/\chi^{5/2}$; NTK $\delta_\theta\approx xd/(3\sqrt\chi)$, bottoms $xd/(3\sqrt\chi)$, root $xd^2/(3\chi)$.
For $\delta_K$ use the second-order form of R4 (the NTK first-order value is about half of GF at $x=1.1$).
For deeper binary trees the SP and NTK exponents are a conjecture.

- **"Lazy by $\delta_K$" can hide a far-moving root.** Under SP the root's own kernel block
  $\Phi_{c_1}\Phi_{c_1}^\top\otimes\Phi_{c_2}\Phi_{c_2}^\top$ does not depend on the root, so the root (a
  readout) can move far while $\delta_K$ stays small: this is random-feature regression with a trained
  readout. Examples: §6 item 2 ($\chi=64$, $x=4$: $\delta_K=0.04$, root spectral norm changes by 214%);
  verifier: $\chi=64$, $\sigma=0.8$: $\delta_K=0.12$, root spectral change 360%. Report per-core changes
  (Frobenius and spectral) next to $\delta_K$.
- **Matrix rule vs muP-TTN at $N=8$ (resolved; v1 left it open).** Time to halve the loss (`bin8`,
  $d=2$, plain, 2 seeds): matrix rule $5.1/3.3$, $17.1/10.3$, $27.8/14.9$ and muP-TTN $2.0/1.6$, $3.4/2.0$, $2.6/1.6$ at
  $\chi=4,8,16$ (geometric-mean ratio 2.3, 5.1, 10.1; verifier identical). The matrix rule's initial
  slowdown grows about linearly in $\chi$, consistent with its $\chi^{-2}$ init kernel; muP-TTN stays flat. The
  time to reach $10^{-4}L_0$ is not slower under the matrix rule (26–40 vs 35–70): its smaller $W_0$ leaves
  less to remove at the end.
- **Conjecture.** The muP-TTN limit is a two-population mean field with a pairwise (graphon-like) root
  coupling, $W=\iint b(a,c)\,a\,c^\top\,d\mu_A(a)\,d\mu_C(c)$ for $N=4$, nested for deeper trees.

## 5. (d) Imbalance: root scale $\lambda\sigma$, other cores $\sigma$, fixed $\alpha=\lambda\sigma^K$ (R13)

**Conserved quantities (derived, exact in expectation).** $Q_v=G_v^\top G_v/\eta_v-G_{p,(v)}G_{p,(v)}^\top/\eta_p$
has $\mathbb EQ_v=(s_v^2-s_p^2\chi_p/\chi_v)I$ for $\eta=1$: internal edges are balanced in expectation;
root edges have $\mathbb EQ_v=\sigma^2(1-\lambda^2/\chi)I$, balanced at $\lambda=\sqrt\chi$. Default fan-in
init is root-light by $\sqrt\chi$. Kunin et al. (NeurIPS 2024) use $\delta=\eta_wa^2-\eta_a\Vert w\Vert^2$
($a$ downstream), i.e. $-\operatorname{tr}Q$ at root edges: $\lambda>\sqrt\chi$ is their upstream init
($\delta>0$), $\lambda<\sqrt\chi$ downstream.

**Aligned single mode (derived, exact).** Non-root amplitudes $x$, root $b$, $x^2-b^2$ conserved;
$\dot b=(s-bx^{K-1})x^{K-1}$, $\dot x=(s-bx^{K-1})bx^{K-2}$; centred target $s+\alpha$.

- Small root, $\lambda\ll1$: $x$ frozen, $b$ relaxes linearly; lazy iff $s\ll\alpha/\lambda$, with
  $\delta_K\approx2(K-1)\lambda^2s(s+2\alpha)/\alpha^2\propto\lambda^2$. In the plain model at small $\alpha$
  there is no plateau once $\lambda<\alpha/s$.
- $\lambda=1$: $\delta_K=(1+s/\alpha)^{2(K-1)/K}-1$.
- Large root, $\lambda\gg1$: $b$ frozen; the other $K-1$ cores form a degree-$(K-1)$ model,
  $\delta_K\approx(1+s/\alpha)^{2(K-2)/(K-1)}-1$ (not lazy); plain-model plateau
  $t\approx\alpha^{-(1-2/K)}\lambda^{-2/K}/((K-3)s)$ for $K\ge4$, logarithmic for $K=3$.

This differs from Kunin et al.'s two-layer linear result (upstream ⇒ lazy) because a frozen large root
still leaves a degree-$\ge2$ model when $K\ge3$.

**Full bonds needed for "small root ⇒ lazy" (derived; corrected).** With a small root the kernel is the
root's block, whose range is $\{\Phi_{c_1}X\Phi_{c_2}^\top\}$ ($\{AXC^\top\}$ for $N=4$). With full bonds
this is everything and training is lazy. With a deficient bond only $P_ATP_C$ is fitted lazily; the rest
needs the bond subspaces to rotate, at a rate suppressed by the small root (as in R7): lazy, then rich —
Kunin et al.'s "delayed rich" for downstream inits. The first plateau is at
$L/L_0=\Vert T-P_ATP_C\Vert^2/\Vert T\Vert^2$, mean $1-(\chi/d^2)^2$ for $N=4$ (checked: 0.917 measured, 0.918
predicted; §6 item 10).

**Random init, $N=4$, isotropic first order (derived).**
$\delta_K(\lambda)\approx\frac{2\Vert T\Vert\lambda^2\sqrt{d^2+1}\,d^5(\lambda^2/\chi+2\chi/d^2)}{\alpha(2\lambda^2d^2+\chi^2)^2}$:
$\propto\lambda^2$ for $\lambda\ll1$, saturating at $\Vert T\Vert d\sqrt{d^2+1}/(2\chi\alpha)$ for $\lambda\gg\chi/d$;
root-kernel share $\chi^2/(\chi^2+2d^2\lambda^2)$. It captures the trend but underestimates $\delta_K$ at
small $\lambda$ by $\sim10\times$ (the root's relative change is not small there). Heuristic for general
trees: Chizat's $\kappa\approx\Vert T\Vert\min(1,\lambda)/\alpha$.

## 6. Numerical checks (current library)

Scripts: scratchpad `theory/regimes_v2/` (outside the repo). Exact population loss
(`population_loss`; `centred_population_loss` and `sum_difference_cores` at large $\sigma$), float64,
`solve_ivp` (RK45/DOP853 at rtol $10^{-8}$–$10^{-9}$, LSODA for long plateaus), torch autograd. Teacher
$(3,2,1)$ odeco, `odeco_teacher(N, d, w, seed=0)`; init seeds as stated. The verifier's independent
numpy checks of v1 (same library, same seeds) agree wherever both exist and are quoted as "verifier".

1. **Kernel at init** (`bin4`, $d=3$, $\sigma=1$, 10 seeds). Mean eigenvalue vs prediction:
   $0.320\pm0.112$ vs 0.333 ($\chi=3$), $1.28\pm0.36$ vs 1.22 ($\chi=9$), $3.37\pm0.71$ vs 3.38 (16),
   $13.3\pm1.0$ vs 12.9 (32), $52.2\pm3.5$ vs 50.8 (64). Ranks 17/45/72/81 at $\chi=1/3/6/9$ = formula;
   uncapped $\chi\ge16$: rank 81 (effective bottom bonds 9; the nominal formula is wrong there, R2).
   Linearized residual $0.49\pm0.13$ vs 0.444 ($\chi=3$), $0.096\pm0.04$ vs 0.111 ($\chi=6$) (verifier, 400
   seeds: $0.455\pm0.007$, $0.110\pm0.004$). $\lambda_{\min}/\bar\lambda$: $2.3\times10^{-4}$ (9, capped),
   0.016, 0.105, 0.218 (16, 32, 64); $\lambda_{\max}/\bar\lambda$: 9.3, 6.2, 4.1, 3.0. Linear response:
   $c_1=0.162\pm0.057$, $0.0124\pm0.0016$, $0.00133\pm0.0002$ (isotropic 0.083, 0.0104, 0.0013);
   $c_\theta=0.227$, $0.048$, $0.0143$ (isotropic 0.095, 0.035, 0.0124). Rank formula with effective bonds
   = measured rank in 7/7 configurations including inconsistent ones (`bin8` $d=3$ bottom bonds 1, top 3:
   33; nominal 29; `mps5` $d=2$ bonds $(2,8,2,2)$: 20, nominal 4).
2. **Lazy boundary, $N=4$, SP** (`bin4`, $d=3$, seeds 0/1, kernel time, stop at $L=10^{-8}L_0$).
   - Centred, $\chi=32$: $\delta_K=0.0130x^2$ (seed 0, $x=1$–3) and $0.030\to0.017\,x^2$ (seed 1); fitted
     $\xi\approx-0.7$ and $+0.8$. $x_c=2.82$, $2.25$ ($\sigma_{\rm lazy}=1.10$, $1.18$); predicted bracket
     $[2.17,2.81]$, i.e. $[1.10,1.19]$. Local slope at $\delta_K=0.1$: 1.84, 1.48.
   - Centred, $\chi=64$: $\delta_K=0.0032,0.011,0.040,0.070,0.107,0.149,0.194$ at $x=1,2,4,5.5,7,8.5,10$
     (seed 0; the $\xi=0$ formula gives $0.101$ at $x=7$); $x_c=6.73$, $7.53$ ($\sigma_{\rm lazy}=0.82$,
     $0.79$); predicted $[6.67,7.31]$ ($[0.81,0.83]$); slopes 1.74, 1.79.
   - Two-copy: $\delta_K/x^2=0.0046$–$0.0054$ ($\chi=32$, $x\le2$) and $5.3$–$6.4\times10^{-4}$ ($\chi=64$),
     predicted $c_2/4=0.0041$ and $5.1\times10^{-4}$; $x_c=5.15,4.96$ ($\chi=32$; predicted 4.94) and
     $13.6,14.8$ ($\chi=64$; predicted 13.97); slopes 1.69, 1.66 (32) and 1.72 (64).
   - Root motion: centred $\chi=64$, $x=4$: $\delta_K=0.040$ while the root's spectral norm changes by 214%.
   - NTK, $\chi=64$/128, $x=1.11$ (centred, rerun of v1): $\delta_K=0.228,0.212$ / $0.121,0.114$ (R4 formula
     0.240 / 0.120); two-copy NTK at $\chi=64$: $\delta_K/x^2=0.050$–$0.055$ (predicted 0.043).
3. **Deficient bond** (`bin4`, $d=3$, $\chi=3$).
   - Centred, small $\sigma$ (LSODA to $t=3000$, seeds 0/1): final $L/L_0$ equals the Eckart–Young minimum
     to 4 digits in 6/6 runs ($\sigma=0.5$: $2.4,3.9\times10^{-4}$; $0.7$: $1.8,2.8\times10^{-3}$; $1.0$:
     0.015, 0.018); the first-order value $\Vert P_\perp W_0\Vert^2/\Vert W^\star\Vert^2$ is within 10% for
     $\sigma\le0.7$. Over 400 inits, $\Vert P_\perp W_0\Vert^2$ has mean $1.05\times$ the formula, median $0.74\times$
     (`bin8`: 0.67×), 10–90% range 0.28–2.1×.
   - Centred, large $\sigma$ (verifier): stalls at the Eckart–Young minimum, 0.4792 ($\sigma=4$), 0.5519 (8),
     $\to\Vert P_\perp T\Vert^2/\Vert T\Vert^2=0.558$.
   - Two-copy, seed 0 (lazy residual 0.558): $\tau(L/L_0<0.3)=185,\ 1187,\ 8933$ at $\sigma=4,5.66,8$
     (local exponents 5.4, 5.8; reduced flow 138, 1102, 8826, exponent 6.0 by construction);
     $\tau(L/L_0<0.1)=1240$ vs 1220 ($\sigma=4$) and 9809 vs 9765 ($\sigma=5.66$). After the plateau the
     reduced flow tracks GF to 1–6%. $\theta_{0.3}=0.70,0.56,0.53$ from GF, 0.52 from the reduced flow.
     Seed 1 (lazy residual 0.365): $\tau(L/L_0<0.3)=195$ and $11530$ at $\sigma=4$ and 8 (local exponent
     5.88; reduced flow 179 and 11510), $\theta_{0.3}=0.40$, $0.37$ (verifier: 195 and 11480).
4. **S2S, $K=3$** (`bin4`, $d=3$, $\chi=3$, plain, seeds 0–29, RK45 rtol $10^{-8}$). S2S fraction with
   final-rise widths 0.97/0.87/0.63 at $\sigma=0.06/0.12/0.24$ (first-crossing widths: 0.80/0.50/0.30).
   $\sigma\rho_{\min}$ per seed at $\sigma=0.06$: range 0.13–5.2, quartiles 0.67–1.17. Transient peaks
   above 0.2 in 4/30 seeds at $\sigma=0.06$ (max 0.80). Reproduces the verifier's seeds 2 and 9 (e.g.
   seed 9: $\rho_1=2.16$ at $\sigma=0.06$, verifier 2.2).
5. **Stalls at $K\ge5$** (plain, LSODA). `old5`, $\sigma=0.65$, seeds 0–3: fan-in last events at $t=2524$
   (seed 0), $62$ (seed 1), never (seeds 2, 3; third bond singular values $8\times10^{-18}$, $9\times10^{-16}$
   at $t=10^5$); balanced: $t=397,172,1151,74$ with $\rho=[30.1,210]$, $[4.7,88]$, $[21.3,661]$,
   $[1.3,38]$. Per-edge diagnostics (seed 0): fan-in root edges have $Q$ eigenvalues in $[0.09,0.64]$
   (child-heavy); the root-side third singular value falls 0.035 → 0.006; balanced keeps it at 0.27–0.29.
   `bin8`, $\sigma=0.9$: fan-in seed 0 learns $s=3,2$ at $t=2.04,74.1$ (verifier 2.03, 74.1) and nothing more
   to $t=10^5$ (third bond singular value $5.6\times10^{-9}$, frozen); seed 1 learns $s=3$ at 2.41, $s=2$ at
   3834, and nothing more ($2\times10^{-16}$). Balanced: seed 0 learns $s=3,2$ at $t=2.06,102.6$ and nothing
   more to $t=10^5$ (third bond singular value $1.8\times10^{-3}$, $6.7\times10^{-4}$, $2.1\times10^{-4}$,
   $1.3\times10^{-4}$ at $t=200$, 1500, $4\times10^4$, $10^5$); seed 1 learns $s=2$ at $t\approx1270$ and nothing
   more to $t=10^5$ ($3.6\times10^{-5}$ at $2\times10^4$, $2.9\times10^{-5}$ at $10^5$). `bin8`, $d=3$,
   $\chi=6$, fan-in, seed 0: $s=3,2$ learned by $t=50$; the third bond singular value falls from
   $7.4\times10^{-3}$ to $4.4\times10^{-3}$ by $t=200$, then the last component is learned at $t\approx334$.
   `bin8`, $d=2$, $\chi=16$ (all bonds full), weights $(3,2)$, fan-in, RK45: seeds 0–11 at $\sigma=0.5,0.7$
   and seeds 0–1 at 0.9, all converged ($t_{\rm end}\le125$); $\rho$ at $\sigma=0.5$: 2.97, 29.3, 27.3, 45.1,
   207, 16.9, 20.3, 8.6, 15.6, 27.5, 36.1, 2.05; at $\sigma=0.7$: 1.48, 4.01, 3.19, 4.98, 33.6, 3.20, 2.63,
   1.37, 1.57, 3.74, 4.59, 0.03; at $\sigma=0.9$: 1.21, 1.11.
6. **Aligned init** (verifier, current library): `aligned_init` reproduces the exact $\rho$ to 4 decimals at
   $K=3$ ($2.9992,3.5256$ vs $2.9993,3.5256$) and $K=7$ ($a_0=0.48$: $3.2744,2.9911$ vs $3.2745,2.9911$).
   Exact limits recomputed (§3.1).
7. **Injective norm** (fan-in binary core, std $1/\chi$; pure Gaussian tensors, unaffected by the RNG fix):
   $\sqrt\chi\Vert G\Vert_{\rm inj}=2.43,2.81,2.78,2.78$ at $\chi=8,16,32,64$ (v1; verifier 2.46, 2.62, 2.79,
   2.81) while the matricized spectral norm is 1.21, 1.23, 1.16, 1.12.
8. **Parameterizations, $N=4$, $\sigma=1.5$** (seeds 0/1). SP: $\delta_K=0.24/0.18$, $0.016/0.035$,
   $0.0038/0.0024$, $0.00035/0.00033$ at $\chi=16,32,64,128$; $\delta_\theta=0.0051$–$0.0056$ and root change
   0.080–0.088 at 128 (formulas 0.0049, 0.078). NTK: $\delta_K=0.66/0.58$, $0.28/0.49$, $0.23/0.21$,
   $0.12/0.11$; bottom cores 0.093–0.104 and root 0.025 at 128 (formulas 0.098, 0.026). muP-TTN:
   $\delta_K=5.2/4.5$, $5.4/5.7$, $6.0/6.0$, $6.2/6.3$; $\delta_\theta=0.57$–$0.59$ at 128; root Frobenius
   change 0.59–0.67 → 0.24.
9. **$N=8$** (`bin8`, $d=2$, $\sigma=1$, seeds 0/1). Init mean eigenvalue $\operatorname{tr}\Theta_\eta/D$ at
   $\chi=4/8/16$: NTK $0.028/0.025$, $0.026/0.039$, $0.042/0.032$ (pred 0.0273); muP-TTN $0.0150/0.0151$,
   $0.0147/0.0237$, $0.024/0.018$ (pred 0.0156); matrix rule $8.8/9.7\times10^{-4}$, $2.0/3.7\times10^{-4}$,
   $8.0/5.8\times10^{-5}$ (pred $9.8\times10^{-4}$, $2.1\times10^{-4}$, $5.0\times10^{-5}$). muP-TTN shares
   bottom/internal/root 0.36–0.50/0.24–0.27/0.24–0.37 (pred 0.5/0.25/0.25; the top cut is not strongly
   over-parameterized). Plain-model GF, teacher $(3,2)$: time to halve the loss in 4.4; time to $10^{-4}L_0$:
   muP-TTN 36.5/34.9, 43.2/51.3, 61.1/69.7; matrix rule 38.2/39.7, 30.3/25.9, 35.7/34.0.
10. **Imbalance** (`bin4`, $d=3$, $\chi=16$, $\alpha=3.375$, centred, seeds 0/1): $\delta_K=0.0018/0.0007$,
    $0.015/0.007$, $0.071/0.041$, $0.24/0.18$, $0.55/0.45$, $0.69/0.62$, $0.68/0.63$ at
    $\lambda=0.03,0.1,0.3,1,3,10,30$ (local slope 1.78/1.92 between 0.03 and 0.1); at $\lambda=0.03$ the root
    moves by 245%/168% while the kernel moves by 0.18%/0.07%. $\operatorname{tr}Q/\chi=+0.43$ ($\lambda=3$) and
    $-2.70$ ($\lambda=10$), predicted $+0.47$ and $-2.55$. Deficient bond (`bin4`, $\chi=3$, seed 0, same
    $\alpha$): $\lambda=0.01$: plateau at $L/L_0=0.917$ for $\tau\approx1$–$30$ (predicted root-only plateau
    0.918), then a slow fall (0.58 at $\tau=2130$); $\lambda=0.03$: $0.91\to0.13$ by $\tau=2130$; $\lambda=1$: no
    plateau, at the centred floor 0.108 by $\tau\approx12$.

## 7. Notes for the experiment agents

- **Bonds.** Lazy-regime runs need full, preferably over-parameterized bonds ($\chi_e\ge2D_e$, `cap=False`).
  With a deficient bond the centred model has a floor (R7) and never meets $L<10^{-3}L_0$ above
  $\sigma_{\rm floor}$; label such runs "stalled (unrealizable)" and compare $L_{\rm end}$ with the floor.
- **Events.** Use final-rise widths (`event_widths`) or bond-spectrum events; first-crossing widths
  misclassify seeds with transients. Classify per seed and report fractions.
- **Stalls at $K\ge5$.** Record the $R$-th singular value of every bond; a value that falls below
  $\sim10^{-12}$ and freezes marks a dead route (fan-in trap), a value that stays $\gtrsim10^{-4}$ marks a
  long plateau. Prefer $\chi\ge2R$ (spare channels) when mapping the S2S boundary at $K\ge5$; use LSODA
  for long budgets; gauge-balanced init (balance every edge by geometric-mean updates of the two
  adjacent Gram matrices; $W_0$ unchanged) removes frozen routes, but at $K=7$, $\chi=R$ the last
  component still did not escape by $t=10^5$ (2 seeds).
- **Kernel diagnostics.** Report $\delta_K$ for the $\eta$-weighted kernel; mask the two-copy Jacobian.
  Report per-core changes (Frobenius and spectral) next to $\delta_K$: the root can move by more than
  100% at $\delta_K<0.1$. For $N\le8$, $d\le3$, compute the tensor-space $\Theta$ directly (`jacrev` of
  `to_dense`); the 64–128-probe $\delta_K$ is 0.7–1.3 times the tensor-space value.
- **Time.** Use kernel time $\tau=\bar\lambda t$ for the lazy phase, but the deficient-bond escape needs
  $\tau\propto\sigma^{2K}$: in GF time it is $t\approx2\theta\sigma^2/\Vert T\Vert^2$ with $\theta\approx0.5$.

## 8. Predictions for experiments

Setting unless stated: odeco teacher $(3,2,1)$, $\Vert W^\star\Vert=\sqrt{14}$, threshold $\delta_*=0.1$,
convergence $L<10^{-3}L_0$, final-rise widths, exact population GF. "Seed s.e." is binomial over 30 seeds.

**Kernel and lazy regime**

1. **Kernel scale and shares.** $\operatorname{tr}\Theta(0)/D=\sigma^{2(K-1)}\sum_vF_v/D$ in expectation for
   every tree and $\chi$ (seed spread $\pm35\%$ at `bin4` $\chi=3$, $\pm7\%$ at $\chi=64$); core shares
   $\propto\eta_vF_v/s_v^2$ (SP `bin4`: root $\chi^2/(18+\chi^2)$ = 0.934, 0.983, 0.996 at $\chi=16,32,64$).
2. **Rank.** $\operatorname{rank}\Theta(0)=\sum_vF_v\chi_v-\sum_e\chi_e^2$ evaluated with effective bond
   dimensions, capped at $D$; $=D$ iff all bonds are full.
3. **Lazy (linearized) residual with deficient bonds.** $\Vert P_\perp T\Vert^2/\Vert T\Vert^2$, mean
   $(1-\chi/d^2)^2$ for $N=4$: 0.444 ($\chi=3$), 0.111 ($\chi=6$); seed spread $\pm0.13$ at $\chi=3$.
4. **Centred floor (deficient bonds).** $L_{\rm end}/L_0$ = Eckart–Young minimum for $N=4$, $\approx\Vert P_\perp W_0\Vert^2/\Vert W^\star\Vert^2$
   at small $\sigma$, with mean $(1-\dim\mathcal M_\chi/D)\sigma^{2K}/\Vert W^\star\Vert^2$ and seed median
   $0.7\times$ the mean. Runs are "stalled" above $\sigma_{\rm floor}$: `bin4` $d=3$: 0.56, 0.60, 0.71, 1.02 for
   $\chi=3,4,6,8$ (about 5% higher for the median seed); `bin8` $d=3$: 0.74, 0.75, 0.78 for $\chi=3,9,24$. At large $\sigma$,
   $L_{\rm end}/L_0\to(1-\chi/d^2)^2$ for $N=4$.
5. **Centred, full over-parameterized bonds** (`bin4`, $d=3$, SP).
   $\delta_K^2=(c_1x)^2+2\xi c_1c_2x^3+(c_2x^2)^2$, $c_1=341.5/\chi^3$, $c_2=537/\chi^3$, $|\xi|\le1$ per seed.
   Lazy boundary $x_c\in[2.2,2.8]$ ($\sigma_{\rm lazy}\in[1.10,1.19]$) at $\chi=32$; $[6.7,7.3]$ ($[0.81,0.83]$) at
   $\chi=64$; $[19.4,20.1]$ ($0.57$–$0.58$) at $\chi=128$; asymptotically $\sigma_{\rm lazy}=6.50\,\chi^{-1/2}$
   (not $\chi^{-1}$). At $\chi=16$ the isotropic constants fail; measured $x_c=0.44$–$0.65$.
6. **Two-copy** (same setting). $\delta_K=(c_2/4)x^2$ for $x\lesssim2$ (0.0041 at $\chi=32$,
   $5.1\times10^{-4}$ at $\chi=64$; measured 5–30% higher); $x_c^{(2)}=2(\delta_*/c_2)^{1/2}=4.9$, 14.0, 39.5
   and $\sigma^{(2)}_{\rm lazy}=5.16\,\chi^{-1/2}=0.91,0.65,0.46$ at $\chi=32,64,128$; the ratio
   $\sigma^{(2)}_{\rm lazy}/\sigma_{\rm lazy}\to2^{-1/3}=0.79$. $\delta_\theta^{(2)}/\delta_\theta^{(c)}=0.50$ for $x\lesssim1$.
7. **Threshold sensitivity.** Local slope $d\ln\delta_K/d\ln x=1.5$–$1.8$ at $\delta_K=0.1$ for both
   models at $\chi\ge32$, so $\delta_*=0.05/0.2$ moves $\sigma_{\rm lazy}$ by $\times1.13$–$1.17$ / $\times0.86$–$0.89$
   ($N=4$); at $\chi=16$ (centred slope $\approx1$) by $\times1.26/0.79$.
8. **Deficient bond, two-copy, large $\sigma$** (`bin4`, $d=3$, $\chi=3$). Plateau at
   $L/L_0=\Vert P_\perp T\Vert^2/\Vert T\Vert^2$ (computable at init; 0.558 for seed 0), then an escape that
   follows the reduced flow of R7 to within a few percent; $L$ falls linearly in $t$ at first.
   $\tau(L/L_0<0.3)=2\theta\sigma^2\bar\lambda_2/\Vert T\Vert^2=4\theta\sigma^6(2d^2+\chi^2)/(d^4\Vert T\Vert^2)$ with
   $\theta\approx0.5$: 195 ($\sigma=4$), 1.25e4 ($\sigma=8$); $\times64$ per doubling of $\sigma$ (measured local
   exponents 5.4–5.8 approaching 6). For a tree with $K$ cores, $\tau_{\rm esc}\propto\sigma^{2K}$ ($\sigma^{14}$ at $K=7$)
   and $t_{\rm esc}\propto\sigma^2$.
9. **Conditioning.** Capped full bonds (`bin4`, $\chi=9$): $\lambda_{\min}/\bar\lambda\approx2\times10^{-4}$,
   so the lazy fit needs $\tau\sim10^5$; uncapped $\chi=16,32,64$: 0.016, 0.105, 0.22.

**Saddle-to-saddle boundary**

10. **Aligned init.** $\rho_i$ given exactly by §3.1 (to 4 decimals in GF); per-core boundary $a_{0c}$ =
    0.066, 0.324, 0.480 for $K=3,5,7$; function-space $u_{0c}\to1.306\times10^{-2}$ as $K\to\infty$.
11. **Random fan-in init, $K=3$** (`bin4`, $d=3$, $\chi=3$, plain). S2S fraction $0.97\pm0.03$, $0.87\pm0.06$,
    $0.63\pm0.09$ at $\sigma=0.06,0.12,0.24$ (first-crossing widths: 0.80, 0.50, 0.30). Per-seed boundary
    $\sigma_c$: median 0.27–0.31, quartiles 0.20–0.39. Median $\rho\sigma$ over seeds 0.93, 0.90, 0.84 at
    $\sigma=0.06,0.12,0.24$; seeds with a mixed first escape approach $\rho\propto1/\sigma$ slowly.
12. **Scaling of gaps.** On converged runs, the fitted exponent of $\rho_i$ vs $\sigma$ tends to $-(K-2)$
    per seed (first gap derived; later gaps heuristic): $-1$ at $K=3$ (30/30 seeds), $-5$ at $K=7$ (local
    exponents 5.0–6.8 between $\sigma=0.5$ and 0.7 at `bin8`, $d=2$, $\chi=16$). Expect $\pm0.1$ only over a
    decade below the boundary.
13. **Transients.** Peak overlap of a not-yet-learned component during an earlier escape $\propto\sigma^{2(1-q)}$
    for matrix-like pairs (`bin4`): $\sigma^{2/3}$ for 3:2 (0.49 → 0.12 from $\sigma=0.06$ to 0.004); nearly
    $\sigma$-independent for $q\ge0.9$ ($\sigma^{0.2}$).
14. **$K\ge5$ with $\chi=R$ (empirical; mechanism heuristic).** With fan-in init a large fraction of runs
    never learns the last component (`old5`, $\sigma=0.65$: 2/4 dead to $t=10^5$; `bin8`, $d=3$, $\sigma=0.9$:
    2/2), with the $R$-th bond singular value below $10^{-8}$ and frozen. Gauge-balanced init: no frozen
    route; `old5` 4/4 converge (last event $t=74$–$1151$), but `bin8` $d=3$ still 2/2 unconverged at
    $t=10^5$ (route decaying slowly). Expect the stalled fraction at a fixed budget to grow with $K$ and to
    fall as $\sigma$ approaches $\sigma_c$; with spare channels expect few or no stalls (`bin8`, $d=2$,
    $\chi=16$: 26/26 converged; $d=3$, $\chi=6$: 1/1 converged, last event $t\approx334$).
15. **$K=7$ boundary.** `bin8`, $d=2$, $\chi=16$, weights $(3,2)$, fan-in, plain (empirical, 12 seeds):
    S2S fraction 0.83 ($\sigma=0.5$), 0.58 (0.7), 0 (0.9); per-seed $\sigma_c$ median 0.71, quartiles
    0.62–0.74. Conjecture: the effective per-core amplitude $\kappa_K=a_{0c}/\sigma_c$ grows with $K$
    ($\kappa_3\approx0.22$, $\kappa_7\approx0.69$) toward 1, so $\sigma_c(K)\to1$ (H4). Test at $K=15$
    (`binary_tree(16)`, $d=2$, weights $(3,2)$, $\chi\ge16$, plain): the conjecture gives
    $\sigma_c=a_{0c}(15)/\kappa_{15}\in[0.74,0.93]$ for $\kappa_{15}\in[0.8,1]$ ($a_{0c}(15)=0.744$). Map $K=7$ in the
    plain model or with full bonds: the centred model at `bin8`, $d=3$, $\chi=3$ is floored above $\sigma=0.74$.

**Large bond dimension**

16. **Exponents at fixed $\sigma$** (`bin4`): SP $\delta_K\propto\chi^{-3}$ (measured $\times7$–$11$ per doubling
    from $\chi=64$ to 128), $\delta_\theta\propto\chi^{-3/2}$, root change $\propto\chi^{-1}$; NTK $\delta_K\propto\chi^{-1}$ with
    $\delta_K\approx\sqrt{(c_1x)^2+(c_2x^2)^2}$, $c_1=6.32/\chi$, $c_2=11.06/\chi$ (0.240 at $\chi=64$, $x=1.11$), and
    bottom cores $xd/(3\sqrt\chi)$; muP-TTN $\delta_K\to6.2$–$6.3$, $\delta_\theta\to0.57$–$0.59$, root Frobenius change
    $\propto\chi^{-1/2}$ (0.24 at $\chi=128$) with an $O(1)$ spectral change.
17. **Matrix rule vs muP-TTN** (`bin8`, $d=2$): init kernel $\to3/(\chi^2D)$ vs $4/D$; time to halve the loss
    grows about linearly in $\chi$ under the matrix rule (geometric means 4.1, 13.3, 20.4 at $\chi=4,8,16$) and stays
    at 1.8–2.6 under muP-TTN.
18. **Injective norm.** Fan-in binary core: $\sqrt\chi\Vert G\Vert_{\rm inj}\to2.8$ while the matricized
    spectral norm stays at 1.1–1.25.
19. **Root motion under SP.** At $\delta_K<0.1$ the root's spectral norm can change by more than 100%
    (`bin4`, $\chi=64$, $x=4$: $\delta_K=0.04$, $+214\%$).

**Imbalance**

20. **Root scale $\lambda$ at fixed $\alpha$** (`bin4`, $\chi=16$, $\alpha=3.375$): $\delta_K\propto\lambda^2$ as
    $\lambda\to0$ (local slope 1.8–1.9 between $\lambda=0.03$ and 0.1), saturating at $0.6$–$0.7$ for
    $\lambda\ge10$; $\operatorname{tr}Q/\chi$ at the root edges $=\sigma^2(1-\lambda^2/\chi)$, sign change at
    $\lambda=\sqrt\chi=4$. With a deficient bond ($\chi<d^2$) and $\lambda\ll1$: a first (root-only, lazy)
    plateau at $L/L_0=\Vert T-P_ATP_C\Vert^2/\Vert T\Vert^2$, mean $1-(\chi/d^2)^2$ (0.89 at $\chi=3$; 0.918
    measured and predicted for seed 0), then the bottom cores move (rich) on a kernel time $\propto\lambda^{-2}$
    (fall to $L/L_0=0.6$ at $\tau\approx250$ and $2000$ for $\lambda=0.03$ and $0.01$), ending at the centred floor.
